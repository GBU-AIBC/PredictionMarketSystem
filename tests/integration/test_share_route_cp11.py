from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import cast

import asyncpg
import httpx
import pytest

from pms.api.app import create_app
from pms.config import DatabaseSettings, PMSSettings
from pms.runner import Runner
from pms.storage.strategy_registry import PostgresStrategyRegistry
from pms.strategies.projections import (
    EvalSpec,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)
from pms.strategies.versioning import serialize_strategy_config_json


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


def _settings() -> PMSSettings:
    return PMSSettings(
        auto_migrate_default_v2=False,
        database=DatabaseSettings(dsn=cast(str, PMS_TEST_DATABASE_URL)),
    )


def _config_json(strategy_id: str, *, secret_api_key: str | None = None) -> str:
    metadata = [("owner", "system"), ("tier", "default")]
    if secret_api_key is not None:
        metadata.append(("api_key", secret_api_key))
    return serialize_strategy_config_json(
        StrategyConfig(
            strategy_id=strategy_id,
            factor_composition=(),
            metadata=tuple(metadata),
        ),
        RiskParams(
            max_position_notional_usdc=100.0,
            max_daily_drawdown_pct=2.5,
            min_order_size_usdc=1.0,
        ),
        EvalSpec(metrics=("brier", "pnl", "fill_rate")),
        ForecasterSpec(forecasters=()),
        MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=500.0,
        ),
    )


async def _seed_strategy(
    connection: asyncpg.Connection,
    *,
    strategy_id: str,
    strategy_version_id: str,
    title: str,
    description: str,
    archived: bool = False,
    share_enabled: bool = False,
    metadata_json: dict[str, object] | None = None,
    secret_api_key: str | None = None,
) -> None:
    async with connection.transaction():
        await connection.execute("SET CONSTRAINTS ALL DEFERRED")
        await connection.execute(
            """
            INSERT INTO strategies (
                strategy_id,
                active_version_id,
                metadata_json,
                title,
                description,
                archived,
                share_enabled
            ) VALUES (
                $1, $2, $3::jsonb, $4, $5, $6, $7
            )
            """,
            strategy_id,
            strategy_version_id,
            json.dumps(metadata_json or {}, sort_keys=True, separators=(",", ":")),
            title,
            description,
            archived,
            share_enabled,
        )
        await connection.execute(
            """
            INSERT INTO strategy_versions (
                strategy_version_id,
                strategy_id,
                config_json
            ) VALUES ($1, $2, $3::jsonb)
            """,
            strategy_version_id,
            strategy_id,
            _config_json(strategy_id, secret_api_key=secret_api_key),
        )


async def _seed_eval_record(
    connection: asyncpg.Connection,
    *,
    decision_id: str,
    strategy_id: str,
    strategy_version_id: str,
    brier_score: float,
    pnl: float,
    recorded_at: datetime,
) -> None:
    await connection.execute(
        """
        INSERT INTO eval_records (
            decision_id,
            market_id,
            prob_estimate,
            resolved_outcome,
            brier_score,
            fill_status,
            recorded_at,
            citations,
            category,
            model_id,
            pnl,
            slippage_bps,
            filled,
            strategy_id,
            strategy_version_id
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, $12, $13, $14, $15
        )
        """,
        decision_id,
        "market-cp11",
        0.6,
        1.0,
        brier_score,
        "matched",
        recorded_at,
        '["seed"]',
        "cp11",
        "model-cp11",
        pnl,
        12.0,
        True,
        strategy_id,
        strategy_version_id,
    )


async def _seed_fill(
    connection: asyncpg.Connection,
    *,
    fill_id: str,
    order_id: str,
    strategy_id: str,
    strategy_version_id: str,
    ts: datetime,
) -> None:
    await connection.execute(
        """
        INSERT INTO fills (
            fill_id,
            order_id,
            market_id,
            ts,
            fill_notional_usdc,
            fill_quantity,
            strategy_id,
            strategy_version_id
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8
        )
        """,
        fill_id,
        order_id,
        "market-cp11",
        ts,
        25.0,
        50.0,
        strategy_id,
        strategy_version_id,
    )


def _client(pg_pool: asyncpg.Pool) -> httpx.AsyncClient:
    runner = Runner(config=_settings())
    runner.bind_pg_pool(pg_pool)
    app = create_app(runner, auto_start=False)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_share_route_returns_safe_allowlist_projection_and_hides_secrets(
    pg_pool: asyncpg.Pool,
) -> None:
    now = datetime(2026, 4, 23, 12, 0, tzinfo=UTC)
    async with pg_pool.acquire() as connection:
        await _seed_strategy(
            connection,
            strategy_id="alpha",
            strategy_version_id="alpha-v1234567",
            title="Alpha Theory",
            description="Buy dislocations when liquidity is deep.",
            share_enabled=True,
            metadata_json={
                "api_key": "SECRET_AK",
                "private_key": "SECRET_PK",
                "metadata_json": {"owner": "ops"},
            },
            secret_api_key="SECRET_AK",
        )
        await _seed_eval_record(
            connection,
            decision_id="alpha-decision-1",
            strategy_id="alpha",
            strategy_version_id="alpha-v1234567",
            brier_score=0.09,
            pnl=5.0,
            recorded_at=now,
        )
        await _seed_eval_record(
            connection,
            decision_id="alpha-decision-2",
            strategy_id="alpha",
            strategy_version_id="alpha-v1234567",
            brier_score=0.16,
            pnl=-2.0,
            recorded_at=now,
        )
        await _seed_fill(
            connection,
            fill_id="alpha-fill-1",
            order_id="alpha-order-1",
            strategy_id="alpha",
            strategy_version_id="alpha-v1234567",
            ts=now,
        )
        await _seed_fill(
            connection,
            fill_id="alpha-fill-2",
            order_id="alpha-order-2",
            strategy_id="alpha",
            strategy_version_id="alpha-v1234567",
            ts=now,
        )

    async with _client(pg_pool) as client:
        response = await client.get("/share/alpha")

    assert response.status_code == 200
    assert response.json() == {
        "strategy_id": "alpha",
        "title": "Alpha Theory",
        "description": "Buy dislocations when liquidity is deep.",
        "brier_overall": 0.125,
        "trade_count": 2,
        "version_id_short": "alpha-v1",
    }
    assert b"SECRET_AK" not in response.content
    assert b"SECRET_PK" not in response.content
    assert b"api_key" not in response.content
    assert b"private_key" not in response.content
    assert b"metadata_json" not in response.content


@pytest.mark.asyncio(loop_scope="session")
async def test_share_route_returns_neutral_404_for_unknown_archived_and_unshared_rows(
    pg_pool: asyncpg.Pool,
) -> None:
    async with pg_pool.acquire() as connection:
        await _seed_strategy(
            connection,
            strategy_id="archived",
            strategy_version_id="archived-v1",
            title="Archived Theory",
            description="Old idea.",
            archived=True,
            share_enabled=True,
        )
        await _seed_strategy(
            connection,
            strategy_id="unshared",
            strategy_version_id="unshared-v1",
            title="Hidden Theory",
            description="Not public.",
            share_enabled=False,
        )

    async with _client(pg_pool) as client:
        missing = await client.get("/share/missing")
        archived = await client.get("/share/archived")
        unshared = await client.get("/share/unshared")

    for response in (missing, archived, unshared):
        assert response.status_code == 404
        assert response.json() == {
            "detail": "This strategy doesn't exist or has been unshared"
        }


@pytest.mark.asyncio(loop_scope="session")
async def test_new_strategy_is_private_until_explicitly_shared(
    pg_pool: asyncpg.Pool,
) -> None:
    registry = PostgresStrategyRegistry(pg_pool)
    await registry.create_strategy("private")

    async with _client(pg_pool) as client:
        response = await client.get("/share/private")
        assert response.status_code == 404
        assert response.json() == {
            "detail": "This strategy doesn't exist or has been unshared"
        }

        async with pg_pool.acquire() as connection:
            await connection.execute(
                "UPDATE strategies SET share_enabled = TRUE WHERE strategy_id = 'private'"
            )

        shared = await client.get("/share/private")

    assert shared.status_code == 200
    assert shared.json() == {
        "strategy_id": "private",
        "title": None,
        "description": None,
        "brier_overall": None,
        "trade_count": 0,
        "version_id_short": None,
    }
