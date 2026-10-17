from __future__ import annotations

from datetime import UTC, datetime
import json
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
    BacktestRunner,
)
from pms.strategies.projections import (
    ActiveStrategy,
    EvalSpec,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)
from pms.core.enums import TimeInForce
from pms.core.models import Opportunity, Portfolio, TradeDecision
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


class _ReplayPipeline:
    def __init__(self, strategy_id: str, strategy_version_id: str) -> None:
        self.strategy_id = strategy_id
        self.strategy_version_id = strategy_version_id
        self._emitted = False

    async def on_signal(
        self,
        signal: Any,
        portfolio: Portfolio | None = None,
    ) -> tuple[Opportunity, TradeDecision] | None:
        del portfolio
        if self._emitted:
            return None
        self._emitted = True
        opportunity = Opportunity(
            opportunity_id="opp-replay-resolution-c1",
            market_id=signal.market_id,
            token_id=signal.token_id,
            side="yes",
            selected_factor_values={},
            expected_edge=0.2,
            rationale="real replay acceptance",
            target_size_usdc=10.0,
            expiry=signal.resolves_at,
            staleness_policy="research",
            strategy_id=self.strategy_id,
            strategy_version_id=self.strategy_version_id,
            created_at=signal.fetched_at,
        )
        return opportunity, TradeDecision(
            decision_id="decision-replay-resolution-c1",
            market_id=signal.market_id,
            token_id=signal.token_id,
            venue=signal.venue,
            side="BUY",
            notional_usdc=10.0,
            order_type="limit",
            max_slippage_bps=50,
            stop_conditions=[],
            prob_estimate=0.7,
            expected_edge=0.2,
            time_in_force=TimeInForce.IOC,
            opportunity_id=opportunity.opportunity_id,
            strategy_id=self.strategy_id,
            strategy_version_id=self.strategy_version_id,
            limit_price=0.6,
        )


class _ReplayPipelineFactory:
    def __init__(self) -> None:
        self.pipeline = _ReplayPipeline("alpha", "v1")

    def build(self, strategy: ActiveStrategy) -> _ReplayPipeline:
        assert strategy.strategy_id == "alpha"
        return self.pipeline


def _active_strategy() -> ActiveStrategy:
    return ActiveStrategy(
        strategy_id="alpha",
        strategy_version_id="v1",
        config=StrategyConfig(
            strategy_id="alpha",
            factor_composition=(),
            metadata=(("owner", "integration"),),
        ),
        risk=RiskParams(
            max_position_notional_usdc=100.0,
            max_daily_drawdown_pct=2.5,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(metrics=("brier", "pnl")),
        forecaster=ForecasterSpec(forecasters=(("rules", ()),)),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=100.0,
        ),
    )


async def _insert_runner_row(
    pool: asyncpg.Pool,
    *,
    run_id: str,
    market_id: str,
) -> None:
    payload = {
        "strategy_versions": [["alpha", "v1"]],
        "dataset": {
            "source": "postgresql",
            "version": "outer-ring-v1",
            "coverage_start": "2026-03-01T00:00:00+00:00",
            "coverage_end": "2026-03-31T23:59:00+00:00",
            "market_universe_filter": {"venue": "polymarket", "market_ids": [market_id]},
            "data_quality_gaps": [],
        },
        "execution_model": {
            "fee_rate": 0.0,
            "slippage_bps": 5.0,
            "latency_ms": 0.0,
            "staleness_ms": 60000.0,
            "fill_policy": "immediate_or_cancel",
        },
        "risk_policy": {
            "max_position_notional_usdc": 100.0,
            "max_daily_drawdown_pct": 2.5,
            "min_order_size_usdc": 1.0,
        },
        "date_range_start": "2026-03-01T00:00:00+00:00",
        "date_range_end": "2026-03-31T23:59:00+00:00",
    }
    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO backtest_runs (
                run_id, spec_hash, status, strategy_ids,
                date_range_start, date_range_end, exec_config_json, spec_json
            ) VALUES ($1::uuid, $2, 'queued', $3::text[], $4, $5, $6::jsonb, $7::jsonb)
            """,
            run_id,
            f"spec-{run_id}",
            ["alpha"],
            datetime(2026, 3, 1, tzinfo=UTC),
            datetime(2026, 3, 31, 23, 59, tzinfo=UTC),
            '{"chunk_days": 7, "time_budget": 1800}',
            json.dumps(payload),
        )


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
    """BacktestRunner + real replay engine persist point-in-time labels."""
    market_id = "market-replay-resolution-c1"
    token_id = "token-replay-resolution-c1"
    store = PostgresMarketDataStore(pg_pool)
    await _seed_replay_rows(store, market_id=market_id, token_id=token_id)
    await MarketResolutionStore(pg_pool).upsert(
        market_id,
        resolved_outcome=1.0,
        source="test",
    )
    run_id = str(uuid.uuid4())
    await _insert_runner_row(pg_pool, run_id=run_id, market_id=market_id)
    from pms.research.replay import MarketUniverseReplayEngine

    engine = MarketUniverseReplayEngine(pool=pg_pool)
    runner = BacktestRunner(
        writable_pool=pg_pool,
        readonly_pool=pg_pool,
        replay_engine=engine,
        execution_simulator=BacktestExecutionSimulator(replay_engine=engine),
        strategy_loader=lambda _keys: _load_one_strategy(),
        controller_factory=_ReplayPipelineFactory(),
    )
    assert await runner.execute(run_id) is True
    async with pg_pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT brier, pnl_cum
            FROM strategy_runs
            WHERE run_id = $1::uuid AND strategy_id = 'alpha'
            """,
            run_id,
        )
    assert row is not None
    assert row["brier"] is not None
    assert row["pnl_cum"] is not None


async def _load_one_strategy() -> list[ActiveStrategy]:
    return [_active_strategy()]


@pytest.mark.asyncio(loop_scope="session")
async def test_real_replay_backtest_fails_when_resolution_label_is_removed(
    pg_pool: asyncpg.Pool,
) -> None:
    """The acceptance path must fail closed when the resolution join is absent."""
    market_id = "market-replay-resolution-c1-mutant"
    await _seed_replay_rows(
        PostgresMarketDataStore(pg_pool),
        market_id=market_id,
        token_id="token-replay-resolution-c1-mutant",
    )
    run_id = str(uuid.uuid4())
    await _insert_runner_row(pg_pool, run_id=run_id, market_id=market_id)
    from pms.research.replay import MarketUniverseReplayEngine

    engine = MarketUniverseReplayEngine(pool=pg_pool)
    runner = BacktestRunner(
        writable_pool=pg_pool,
        readonly_pool=pg_pool,
        replay_engine=engine,
        execution_simulator=BacktestExecutionSimulator(replay_engine=engine),
        strategy_loader=lambda _keys: _load_one_strategy(),
        controller_factory=_ReplayPipelineFactory(),
    )
    assert await runner.execute(run_id) is False
    async with pg_pool.acquire() as connection:
        row = await connection.fetchrow(
            "SELECT status, failure_reason FROM backtest_runs WHERE run_id = $1::uuid",
            run_id,
        )
    assert row is not None
    assert row["status"] == "failed"
    assert "required resolution labels" in row["failure_reason"]


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
