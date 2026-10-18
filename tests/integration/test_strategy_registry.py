from __future__ import annotations

from datetime import UTC, datetime
import importlib
import os
from typing import Any, cast

import asyncpg
import pytest

from pms.strategies.aggregate import Strategy
from pms.strategies.projections import (
    EvalSpec,
    FactorCompositionStep,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)
from pms.strategies.versioning import compute_strategy_version_id, serialize_strategy_config_json


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


class _AcquireConnection:
    def __init__(self, connection: asyncpg.Connection) -> None:
        self._connection = connection

    async def __aenter__(self) -> asyncpg.Connection:
        return self._connection

    async def __aexit__(self, *_: object) -> None:
        return None


class _SingleConnectionPool:
    def __init__(self, connection: asyncpg.Connection) -> None:
        self._connection = connection

    def acquire(self) -> _AcquireConnection:
        return _AcquireConnection(self._connection)


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in red phase
        pytest.fail(f"{module_name} is missing: {exc}")
    return getattr(module, symbol_name)


def _strategy(
    strategy_id: str,
    *,
    owner: str,
    drawdown_pct: float,
) -> Strategy:
    return Strategy(
        config=StrategyConfig(
            strategy_id=strategy_id,
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
            metadata=(("owner", owner), ("tier", "default")),
        ),
        risk=RiskParams(
            max_position_notional_usdc=100.0,
            max_daily_drawdown_pct=drawdown_pct,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(metrics=("brier", "pnl", "fill_rate")),
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


@pytest.mark.asyncio(loop_scope="session")
async def test_create_strategy_and_list_strategies(
    db_conn: asyncpg.Connection,
) -> None:
    registry_cls = _load_symbol("pms.storage.strategy_registry", "PostgresStrategyRegistry")
    strategy_row_cls = _load_symbol("pms.strategies.projections", "StrategyRow")
    registry = registry_cls(pool=cast(Any, _SingleConnectionPool(db_conn)))

    await registry.create_strategy("default", metadata={"owner": "system"})

    rows = await registry.list_strategies()

    assert len(rows) == 1
    assert isinstance(rows[0], strategy_row_cls)
    assert rows[0].strategy_id == "default"
    assert rows[0].active_version_id is None
    assert await db_conn.fetchval(
        "SELECT share_enabled FROM strategies WHERE strategy_id = 'default'"
    ) is False


@pytest.mark.asyncio(loop_scope="session")
async def test_get_by_id_round_trips_seeded_default_strategy(
    db_conn: asyncpg.Connection,
) -> None:
    registry_cls = _load_symbol("pms.storage.strategy_registry", "PostgresStrategyRegistry")
    registry = registry_cls(pool=cast(Any, _SingleConnectionPool(db_conn)))
    strategy = _strategy("default", owner="system", drawdown_pct=2.5)

    await db_conn.execute("SET CONSTRAINTS ALL DEFERRED")
    await db_conn.execute(
        """
        INSERT INTO strategies (strategy_id, active_version_id)
        VALUES ('default', 'default-v1')
        ON CONFLICT (strategy_id) DO NOTHING
        """
    )
    await db_conn.execute(
        """
        INSERT INTO strategy_versions (
            strategy_version_id,
            strategy_id,
            config_json
        ) VALUES ($1, $2, $3::jsonb)
        ON CONFLICT (strategy_version_id) DO NOTHING
        """,
        "default-v1",
        "default",
        serialize_strategy_config_json(*strategy.snapshot()),
    )

    assert await registry.get_by_id("default") == strategy


@pytest.mark.asyncio(loop_scope="session")
async def test_create_version_and_get_by_id_round_trip_strategy(
    db_conn: asyncpg.Connection,
) -> None:
    registry_cls = _load_symbol("pms.storage.strategy_registry", "PostgresStrategyRegistry")
    strategy_version_cls = _load_symbol("pms.strategies.projections", "StrategyVersion")
    registry = registry_cls(pool=cast(Any, _SingleConnectionPool(db_conn)))
    strategy = _strategy("default", owner="system", drawdown_pct=2.5)

    created_version = await registry.create_version(strategy)
    fetched_strategy = await registry.get_by_id("default")

    assert isinstance(created_version, strategy_version_cls)
    assert created_version.strategy_version_id == compute_strategy_version_id(*strategy.snapshot())
    assert fetched_strategy == strategy
    assert await db_conn.fetchval(
        "SELECT share_enabled FROM strategies WHERE strategy_id = 'default'"
    ) is False


@pytest.mark.asyncio(loop_scope="session")
async def test_get_by_id_raises_type_error_for_invalid_config_json(
    db_conn: asyncpg.Connection,
) -> None:
    registry_cls = _load_symbol("pms.storage.strategy_registry", "PostgresStrategyRegistry")
    registry = registry_cls(pool=cast(Any, _SingleConnectionPool(db_conn)))

    await db_conn.execute("SET CONSTRAINTS ALL DEFERRED")
    await db_conn.execute(
        """
        INSERT INTO strategies (strategy_id, active_version_id)
        VALUES ('broken', 'broken-v1')
        """
    )
    await db_conn.execute(
        """
        INSERT INTO strategy_versions (
            strategy_version_id,
            strategy_id,
            config_json
        ) VALUES (
            'broken-v1',
            'broken',
            '{"config":{"strategy_id":7,"factor_composition":[],"metadata":[]},"risk":{"max_position_notional_usdc":100.0,"max_daily_drawdown_pct":2.5,"min_order_size_usdc":1.0},"eval_spec":{"metrics":["brier"]},"forecaster":{"forecasters":[]},"market_selection":{"venue":"polymarket","resolution_time_max_horizon_days":7,"volume_min_usdc":500.0}}'::jsonb
        )
        """
    )

    with pytest.raises(TypeError, match="config.strategy_id"):
        await registry.get_by_id("broken")


@pytest.mark.asyncio(loop_scope="session")
async def test_get_by_id_returns_none_for_unknown_strategy(
    db_conn: asyncpg.Connection,
) -> None:
    registry_cls = _load_symbol("pms.storage.strategy_registry", "PostgresStrategyRegistry")
    registry = registry_cls(pool=cast(Any, _SingleConnectionPool(db_conn)))

    assert await registry.get_by_id("missing-strategy") is None


@pytest.mark.asyncio(loop_scope="session")
async def test_list_versions_returns_rows_in_created_at_order(
    db_conn: asyncpg.Connection,
) -> None:
    registry_cls = _load_symbol("pms.storage.strategy_registry", "PostgresStrategyRegistry")
    strategy_version_cls = _load_symbol("pms.strategies.projections", "StrategyVersion")
    registry = registry_cls(pool=cast(Any, _SingleConnectionPool(db_conn)))

    first_strategy = _strategy("default", owner="system", drawdown_pct=2.5)
    second_strategy = _strategy("default", owner="system", drawdown_pct=3.5)
    expected_ids = [
        compute_strategy_version_id(*first_strategy.snapshot()),
        compute_strategy_version_id(*second_strategy.snapshot()),
    ]

    await registry.create_strategy("default", metadata={"owner": "system"})
    first_version = await registry.create_version(first_strategy)
    await db_conn.execute("SELECT pg_sleep(0.01)")
    second_version = await registry.create_version(second_strategy)

    versions = await registry.list_versions("default")

    assert isinstance(first_version, strategy_version_cls)
    assert isinstance(second_version, strategy_version_cls)
    assert [version.strategy_version_id for version in versions] == expected_ids
    assert all(
        isinstance(version.created_at, datetime) and version.created_at.tzinfo is UTC
        for version in versions
    )
