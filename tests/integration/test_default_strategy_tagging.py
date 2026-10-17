from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
import os
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest

from pms.config import ControllerSettings, DatabaseSettings, PMSSettings, RiskSettings
from pms.core.enums import MarketStatus, RunMode
from pms.core.models import Market, MarketSignal, Token
from pms.runner import Runner
from pms.strategies.aggregate import Strategy
from pms.strategies.projections import (
    EvalSpec,
    FactorCompositionStep,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)
from pms.strategies.versioning import serialize_strategy_config_json
from pms.storage.eval_store import EvalStore
from pms.storage.feedback_store import FeedbackStore
from pms.storage.market_data_store import PostgresMarketDataStore
from pms.storage.strategy_registry import PostgresStrategyRegistry
from tests.support.strategy_catalog import seed_factor_catalog


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


class SequenceSensor:
    def __init__(self, signals: list[MarketSignal]) -> None:
        self._signals = list(signals)

    def __aiter__(self) -> AsyncIterator[MarketSignal]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MarketSignal]:
        for signal in self._signals:
            yield signal


def _settings(*, auto_migrate_default_v2: bool = True) -> PMSSettings:
    assert PMS_TEST_DATABASE_URL is not None
    return PMSSettings(
        mode=RunMode.PAPER,
        auto_migrate_default_v2=auto_migrate_default_v2,
        controller=ControllerSettings(strict_factor_gates=False),
        database=DatabaseSettings(
            dsn=PMS_TEST_DATABASE_URL,
            pool_min_size=1,
            pool_max_size=2,
        ),
        risk=RiskSettings(
            max_position_per_market=1000.0,
            max_total_exposure=10_000.0,
        ),
    )


def _signal(
    *,
    market_id: str,
    token_id: str = "yes-token",
    orderbook: dict[str, object],
    external_signal: dict[str, object],
) -> MarketSignal:
    fetched_at = datetime.now(tz=UTC)
    return MarketSignal(
        market_id=market_id,
        token_id=token_id,
        venue="polymarket",
        title="Will CP06 default-tag runtime writes?",
        yes_price=0.4,
        volume_24h=1000.0,
        resolves_at=fetched_at + timedelta(days=7),
        orderbook=orderbook,
        external_signal=external_signal,
        fetched_at=fetched_at,
        market_status=MarketStatus.OPEN.value,
    )


def _default_strategy_config_json() -> str:
    return serialize_strategy_config_json(
        StrategyConfig(
            strategy_id="default",
            factor_composition=(
                FactorCompositionStep(
                    factor_id="factor-a",
                    role="weighted",
                    param="",
                    weight=0.6,
                    threshold=None,
                ),
                FactorCompositionStep(
                    factor_id="factor-b",
                    role="weighted",
                    param="",
                    weight=0.4,
                    threshold=None,
                ),
            ),
            metadata=(("owner", "system"), ("tier", "default")),
        ),
        RiskParams(
            max_position_notional_usdc=100.0,
            max_daily_drawdown_pct=2.5,
            min_order_size_usdc=1.0,
        ),
        EvalSpec(
            metrics=("brier", "pnl", "fill_rate"),
            min_slippage_samples=1,
            min_win_rate_samples=1,
        ),
        ForecasterSpec(
            forecasters=(
                ("rules", (("threshold", "0.55"),)),
                ("stats", (("window", "15m"),)),
            )
        ),
        MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=500.0,
        ),
    )


def _strategy(*, drawdown_pct: float) -> Strategy:
    return Strategy(
        config=StrategyConfig(
            strategy_id="default",
            factor_composition=(
                FactorCompositionStep(
                    factor_id="factor-a",
                    role="weighted",
                    param="",
                    weight=0.6,
                    threshold=None,
                ),
                FactorCompositionStep(
                    factor_id="factor-b",
                    role="weighted",
                    param="",
                    weight=0.4,
                    threshold=None,
                ),
            ),
            metadata=(("owner", "system"), ("tier", "default")),
        ),
        risk=RiskParams(
            max_position_notional_usdc=100.0,
            max_daily_drawdown_pct=drawdown_pct,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(
            metrics=("brier", "pnl", "fill_rate"),
            min_slippage_samples=1,
            min_win_rate_samples=1,
        ),
        forecaster=ForecasterSpec(
            forecasters=(
                ("rules", (("threshold", "0.55"),)),
                ("stats", (("window", "15m"),)),
            )
        ),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=500.0,
        ),
    )


async def _count_null_strategy_tags(connection: asyncpg.Connection, table: str) -> int:
    query = f"""
    SELECT COUNT(*)
    FROM {table}
    WHERE strategy_id IS NULL OR strategy_version_id IS NULL
    """
    count = await connection.fetchval(query)
    assert isinstance(count, int)
    return count


async def _strategy_row_count(
    connection: asyncpg.Connection,
    table: str,
) -> int:
    query = f"""
    SELECT COUNT(*)
    FROM {table}
    WHERE strategy_id = 'default'
    """
    count = await connection.fetchval(query)
    assert isinstance(count, int)
    return count


async def _strategy_pairs(
    connection: asyncpg.Connection,
    table: str,
) -> set[tuple[str, str]]:
    query = f"""
    SELECT DISTINCT strategy_id, strategy_version_id
    FROM {table}
    WHERE strategy_id IS NOT NULL AND strategy_version_id IS NOT NULL
    """
    rows = await connection.fetch(query)
    return {
        (row["strategy_id"], row["strategy_version_id"])
        for row in rows
    }


async def _seed_market_shells(
    pg_pool: asyncpg.Pool,
    *,
    market_tokens: Mapping[str, str],
) -> None:
    store = PostgresMarketDataStore(pg_pool)
    observed_at = datetime.now(tz=UTC)
    for market_id, token_id in market_tokens.items():
        await store.write_market(
            Market(
                condition_id=market_id,
                slug=market_id,
                question=f"Will {market_id} persist runtime rows?",
                venue="polymarket",
                resolves_at=observed_at + timedelta(days=7),
                created_at=observed_at,
                last_seen_at=observed_at,
                volume_24h=1000.0,
            )
        )
        await store.write_token(
            Token(
                token_id=token_id,
                condition_id=market_id,
                outcome="YES",
            )
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_runner_tags_inner_ring_rows_with_default_strategy(
    pg_pool: asyncpg.Pool,
) -> None:
    active_version = await PostgresStrategyRegistry(pg_pool).create_version(
        _strategy(drawdown_pct=3.5)
    )
    async with pg_pool.acquire() as connection:
        await seed_factor_catalog(connection)
    await _seed_market_shells(
        pg_pool,
        market_tokens={
            "paper-empty-book": "empty-token",
            "paper-with-depth": "depth-token",
        },
    )

    runner = Runner(
        config=_settings(auto_migrate_default_v2=False),
        sensors=[
            SequenceSensor(
                [
                    _signal(
                        market_id="paper-empty-book",
                        token_id="empty-token",
                        orderbook={"bids": [], "asks": []},
                        external_signal={"fair_value": 0.7, "resolved_outcome": 1.0},
                    ),
                    _signal(
                        market_id="paper-with-depth",
                        token_id="depth-token",
                        orderbook={
                            "bids": [{"price": 0.399, "size": 250.0}],
                            "asks": [{"price": 0.40, "size": 250.0}],
                        },
                        external_signal={"metaculus_prob": 0.9, "resolved_outcome": 0.0},
                    ),
                ]
            )
        ],
        eval_store=EvalStore(),
        feedback_store=FeedbackStore(),
    )

    try:
        await runner.start()
        await runner.wait_until_idle()
    finally:
        await runner.stop()

    async with pg_pool.acquire() as connection:
        strategies_count = await connection.fetchval(
            "SELECT COUNT(*) FROM strategies WHERE strategy_id = 'default'"
        )
        versions_count = await connection.fetchval(
            """
            SELECT COUNT(*)
            FROM strategy_versions
            WHERE strategy_id = 'default' AND strategy_version_id = $1
            """
            ,
            active_version.strategy_version_id,
        )
        feedback_count = await _strategy_row_count(connection, "feedback")
        eval_count = await _strategy_row_count(connection, "eval_records")
        orders_count = await _strategy_row_count(connection, "orders")
        fills_count = await _strategy_row_count(connection, "fills")

        counts = {
            "feedback": feedback_count,
            "eval_records": eval_count,
            "orders": orders_count,
            "fills": fills_count,
        }
        null_counts = {
            table: await _count_null_strategy_tags(connection, table)
            for table in counts
        }
        strategy_pairs = {
            table: await _strategy_pairs(connection, table)
            for table in counts
        }

    assert strategies_count == 1
    assert versions_count == 1
    assert counts["feedback"] > 0
    assert counts["eval_records"] > 0
    assert null_counts == {
        "feedback": 0,
        "eval_records": 0,
        "orders": 0,
        "fills": 0,
    }
    assert len(strategy_pairs["feedback"]) == 1
    assert strategy_pairs["feedback"] == strategy_pairs["eval_records"]
    tagged_strategy_id, tagged_strategy_version_id = next(iter(strategy_pairs["feedback"]))
    assert tagged_strategy_id == "default"
    assert tagged_strategy_version_id
    assert counts["orders"] > 0
    assert counts["fills"] > 0
    assert strategy_pairs["orders"] == strategy_pairs["feedback"]
    assert strategy_pairs["fills"] == strategy_pairs["feedback"]
