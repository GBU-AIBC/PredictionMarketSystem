from __future__ import annotations

from datetime import UTC, datetime
import os
import uuid
from typing import Any, cast

import asyncpg
import pytest

from pms.research.specs import (
    BacktestDataset,
    BacktestExecutionConfig,
    BacktestSpec,
    ExecutionModel,
    RiskPolicy,
)
from pms.research.execution import BacktestExecutionSimulator
from pms.research.runner import (
    _StrategyAccumulator,
    _fill_from_order,
    _signal_with_replay_resolution,
)
from pms.storage.market_resolution_store import MarketResolutionStore
from pms.storage.market_data_store import PostgresMarketDataStore
from tests.integration.test_market_data_store import (
    _level,
    _market as _md_market,
    _price_change as _md_price_change,
    _snapshot,
    _token as _md_token,
    _trade as _md_trade,
)


PMS_TEST_DATABASE_URL = os.environ.get("PMS_TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("PMS_RUN_INTEGRATION") != "1",
        reason="set PMS_RUN_INTEGRATION=1 to run PostgreSQL integration tests",
    ),
    pytest.mark.skipif(
        PMS_TEST_DATABASE_URL is None,
        reason="set PMS_TEST_DATABASE_URL to the compose-backed PostgreSQL URI",
    ),
]

def _spec(*, market_id: str) -> BacktestSpec:
    dataset = BacktestDataset(
        source="postgresql",
        version="outer-ring-v1",
        coverage_start=datetime(2026, 3, 1, tzinfo=UTC),
        coverage_end=datetime(2026, 3, 31, 23, 59, tzinfo=UTC),
        market_universe_filter={"venue": "polymarket", "market_ids": (market_id,)},
        data_quality_gaps=(),
    )
    return BacktestSpec(
        strategy_versions=(("alpha", "v1"),),
        dataset=dataset,
        execution_model=ExecutionModel.polymarket_paper(),
        risk_policy=RiskPolicy(
            max_position_notional_usdc=100.0,
            max_daily_drawdown_pct=2.5,
            min_order_size_usdc=1.0,
        ),
        date_range_start=datetime(2026, 3, 1, tzinfo=UTC),
        date_range_end=datetime(2026, 3, 31, 23, 59, tzinfo=UTC),
    )


async def _seed_replay_rows(
    store: PostgresMarketDataStore,
    *,
    market_id: str,
    token_id: str,
) -> None:
    await store.write_market(
        _md_market(
            condition_id=market_id,
            slug=market_id,
            question="Will CP02 replay reconstruct outer-ring data?",
            created_at=datetime(2026, 2, 28, tzinfo=UTC),
            last_seen_at=datetime(2026, 3, 31, 23, 59, tzinfo=UTC),
        )
    )
    await store.write_token(_md_token(token_id=token_id, condition_id=market_id))
    await store.write_book_snapshot(
        _snapshot(
            market_id=market_id,
            token_id=token_id,
            ts=datetime(2026, 3, 1, 0, 0, tzinfo=UTC),
            hash_value="snapshot-1",
            source="subscribe",
        ),
        [
            _level(snapshot_id=0, market_id=market_id, side="BUY", price=0.41, size=120.0),
            _level(snapshot_id=0, market_id=market_id, side="SELL", price=0.59, size=95.0),
        ],
    )
    await store.write_price_change(
        _md_price_change(
            market_id=market_id,
            token_id=token_id,
            ts=datetime(2026, 3, 8, 0, 0, tzinfo=UTC),
            side="BUY",
            price=0.43,
            size=140.0,
            best_bid=0.43,
            best_ask=0.57,
            hash_value="delta-1",
        )
    )
    await store.write_trade(
        _md_trade(
            market_id=market_id,
            token_id=token_id,
            ts=datetime(2026, 3, 16, 12, 0, tzinfo=UTC),
            price=0.52,
        )
    )


async def _collect_signals(
    engine: Any,
    spec: BacktestSpec,
    exec_config: BacktestExecutionConfig,
) -> list[Any]:
    signals: list[Any] = []
    async for signal in engine.stream(spec, exec_config):
        signals.append(signal)
    return signals


@pytest.mark.asyncio(loop_scope="session")
async def test_market_universe_replay_engine_streams_ordered_market_signals(
    pg_pool: asyncpg.Pool,
) -> None:
    from pms.research.replay import MarketUniverseReplayEngine

    market_id = "market-replay-cp02"
    token_id = "token-replay-cp02"
    store = PostgresMarketDataStore(pg_pool)
    await _seed_replay_rows(store, market_id=market_id, token_id=token_id)

    engine = MarketUniverseReplayEngine(pool=pg_pool)
    signals = await _collect_signals(
        engine,
        _spec(market_id=market_id),
        BacktestExecutionConfig(chunk_days=7),
    )

    assert [signal.external_signal["raw_event_type"] for signal in signals] == [
        "book",
        "price_change",
        "last_trade_price",
    ]
    assert [signal.fetched_at for signal in signals] == sorted(
        signal.fetched_at for signal in signals
    )
    assert signals[0].orderbook == {
        "bids": [{"price": 0.41, "size": 120.0}],
        "asks": [{"price": 0.59, "size": 95.0}],
    }
    assert signals[1].yes_price == pytest.approx(0.5)
    assert signals[2].yes_price == pytest.approx(0.52)


@pytest.mark.asyncio(loop_scope="session")
async def test_real_replay_ioc_fill_scores_brier_and_pnl_from_market_resolution(
    pg_pool: asyncpg.Pool,
) -> None:
    """Replay labels join only for scoring, including immediate IOC fills."""
    from types import SimpleNamespace
    from pms.core.enums import TimeInForce
    from pms.core.models import TradeDecision
    from pms.research.replay import MarketUniverseReplayEngine

    market_id = "market-replay-resolution-c1"
    token_id = "token-replay-resolution-c1"
    store = PostgresMarketDataStore(pg_pool)
    await _seed_replay_rows(store, market_id=market_id, token_id=token_id)
    await MarketResolutionStore(pg_pool).upsert(
        market_id,
        resolved_outcome=1.0,
        source="test",
    )
    spec = _spec(market_id=market_id)
    engine = MarketUniverseReplayEngine(pool=pg_pool)
    signal = (await _collect_signals(
        engine,
        spec,
        BacktestExecutionConfig(chunk_days=7),
    ))[0]
    scoring_signal = _signal_with_replay_resolution(signal, engine)
    decision = TradeDecision(
        decision_id="decision-replay-resolution-c1",
        market_id=market_id,
        token_id=token_id,
        venue="polymarket",
        side="BUY",
        notional_usdc=10.0,
        order_type="limit",
        max_slippage_bps=50,
        stop_conditions=[],
        prob_estimate=0.7,
        expected_edge=0.2,
        time_in_force=TimeInForce.IOC,
        opportunity_id="opportunity-replay-resolution-c1",
        strategy_id="alpha",
        strategy_version_id="v1",
        limit_price=0.6,
    )
    order_state = await BacktestExecutionSimulator(replay_engine=engine).execute(
        signal=signal,
        decision=decision,
        execution_model=spec.execution_model,
    )
    fill = _fill_from_order(
        order_state,
        decision=decision,
        signal=scoring_signal,
        prior_filled_notional_usdc=0.0,
        prior_filled_quantity=0.0,
        execution_model=spec.execution_model,
    )
    assert fill is not None
    accumulator = _StrategyAccumulator(
        strategy_id="alpha",
        strategy_version_id="v1",
        execution_model=spec.execution_model,
    )
    accumulator.record_decision(
        signal=scoring_signal,
        opportunity=SimpleNamespace(side="yes"),
        decision=decision,
    )
    accumulator.record_fill(signal=scoring_signal, decision=decision, fill=fill)
    args = accumulator.as_insert_args(run_id="run-replay-resolution-c1")
    assert args[8] is not None
    assert args[9] is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_market_universe_replay_engine_chunk_days_changes_chunk_count(
    pg_pool: asyncpg.Pool,
) -> None:
    from pms.research.replay import MarketUniverseReplayEngine

    market_id = "market-replay-chunks"
    token_id = "token-replay-chunks"
    store = PostgresMarketDataStore(pg_pool)
    await _seed_replay_rows(store, market_id=market_id, token_id=token_id)

    chunks_3: list[tuple[datetime, datetime]] = []
    chunks_14: list[tuple[datetime, datetime]] = []

    engine_three = MarketUniverseReplayEngine(
        pool=pg_pool,
        chunk_observer=lambda start, end: chunks_3.append((start, end)),
    )
    engine_fourteen = MarketUniverseReplayEngine(
        pool=pg_pool,
        chunk_observer=lambda start, end: chunks_14.append((start, end)),
    )

    await _collect_signals(engine_three, _spec(market_id=market_id), BacktestExecutionConfig(chunk_days=3))
    await _collect_signals(engine_fourteen, _spec(market_id=market_id), BacktestExecutionConfig(chunk_days=14))

    assert len(chunks_3) == 11
    assert len(chunks_14) == 3


@pytest.mark.asyncio(loop_scope="session")
async def test_market_universe_replay_engine_preserves_price_change_state_across_chunk_boundaries(
    pg_pool: asyncpg.Pool,
) -> None:
    from pms.research.replay import MarketUniverseReplayEngine

    market_id = "market-replay-straddle"
    token_id = "token-replay-straddle"
    store = PostgresMarketDataStore(pg_pool)
    await store.write_market(_md_market(condition_id=market_id, slug=market_id))
    await store.write_token(_md_token(token_id=token_id, condition_id=market_id))
    await store.write_book_snapshot(
        _snapshot(
            market_id=market_id,
            token_id=token_id,
            ts=datetime(2026, 3, 1, tzinfo=UTC),
            hash_value="snapshot-straddle",
            source="subscribe",
        ),
        [
            _level(snapshot_id=0, market_id=market_id, side="BUY", price=0.40, size=100.0),
            _level(snapshot_id=0, market_id=market_id, side="SELL", price=0.60, size=100.0),
        ],
    )
    await store.write_price_change(
        _md_price_change(
            market_id=market_id,
            token_id=token_id,
            ts=datetime(2026, 3, 2, tzinfo=UTC),
            side="BUY",
            price=0.42,
            size=110.0,
            best_bid=0.42,
            best_ask=0.60,
            hash_value="delta-a",
        )
    )
    await store.write_price_change(
        _md_price_change(
            market_id=market_id,
            token_id=token_id,
            ts=datetime(2026, 3, 9, tzinfo=UTC),
            side="SELL",
            price=0.58,
            size=90.0,
            best_bid=0.42,
            best_ask=0.58,
            hash_value="delta-b",
        )
    )

    spec = _spec(market_id=market_id)
    engine = MarketUniverseReplayEngine(pool=pg_pool)
    straddled_signals = await _collect_signals(
        engine,
        spec,
        BacktestExecutionConfig(chunk_days=7),
    )
    single_chunk_signals = await _collect_signals(
        engine,
        spec,
        BacktestExecutionConfig(chunk_days=14),
    )

    assert straddled_signals[-1].orderbook == single_chunk_signals[-1].orderbook
    assert straddled_signals[-1].yes_price == pytest.approx(single_chunk_signals[-1].yes_price)


@pytest.mark.asyncio(loop_scope="session")
async def test_market_universe_replay_engine_raises_invariant_error_for_outer_ring_write_probe(
    pg_pool: asyncpg.Pool,
) -> None:
    from pms.research.replay import MarketUniverseReplayEngine, ReplayEngineInvariantError

    market_id = "market-replay-readonly"
    token_id = "token-replay-readonly"
    store = PostgresMarketDataStore(pg_pool)
    await _seed_replay_rows(store, market_id=market_id, token_id=token_id)

    role_name = f"pms_replay_ro_{uuid.uuid4().hex[:8]}"
    role_password = uuid.uuid4().hex
    async with pg_pool.acquire() as connection:
        database_name = cast(str, await connection.fetchval("SELECT current_database()"))
        await connection.execute(f'CREATE ROLE "{role_name}" LOGIN PASSWORD \'{role_password}\'')
        await connection.execute(f'GRANT CONNECT ON DATABASE "{database_name}" TO "{role_name}"')
        await connection.execute(f'GRANT USAGE ON SCHEMA public TO "{role_name}"')
        await connection.execute(f'GRANT SELECT ON ALL TABLES IN SCHEMA public TO "{role_name}"')

    read_only_pool = await asyncpg.create_pool(
        dsn=cast(str, PMS_TEST_DATABASE_URL),
        user=role_name,
        password=role_password,
        min_size=1,
        max_size=1,
    )
    try:
        async def write_probe(connection: asyncpg.Connection) -> None:
            await connection.execute(
                """
                INSERT INTO markets (
                    condition_id,
                    slug,
                    question,
                    venue,
                    resolves_at,
                    created_at,
                    last_seen_at,
                    volume_24h
                ) VALUES (
                    'readonly-write',
                    'readonly-write',
                    'readonly-write',
                    'polymarket',
                    NULL,
                    now(),
                    now(),
                    0.0
                )
                """
            )

        engine = MarketUniverseReplayEngine(pool=read_only_pool, write_probe=write_probe)

        with pytest.raises(ReplayEngineInvariantError, match="engine attempted WRITE on outer ring"):
            await _collect_signals(
                engine,
                _spec(market_id=market_id),
                BacktestExecutionConfig(chunk_days=7),
            )
    finally:
        await read_only_pool.close()
        async with pg_pool.acquire() as connection:
            await connection.execute(f'DROP OWNED BY "{role_name}"')
            await connection.execute(f'DROP ROLE IF EXISTS "{role_name}"')
