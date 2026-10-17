from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import cast

import asyncpg
import httpx
import pytest

from pms.api.app import create_app
from pms.config import DatabaseSettings, PMSSettings
from pms.runner import Runner
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


def _settings() -> PMSSettings:
    return PMSSettings(
        database=DatabaseSettings(dsn=cast(str, PMS_TEST_DATABASE_URL)),
    )


def _app_client(pg_pool: asyncpg.Pool) -> httpx.AsyncClient:
    runner = Runner(config=_settings())
    runner.bind_pg_pool(pg_pool)
    app = create_app(runner)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


async def _seed_catalog_and_series(
    connection: asyncpg.Connection,
) -> tuple[datetime, datetime, datetime]:
    await seed_factor_catalog(
        connection,
        factor_ids=("metaculus_prior", "orderbook_imbalance"),
    )
    ts_1 = datetime(2026, 4, 18, 10, 0, tzinfo=UTC)
    ts_2 = datetime(2026, 4, 18, 10, 5, tzinfo=UTC)
    ts_3 = datetime(2026, 4, 18, 10, 10, tzinfo=UTC)
    await connection.execute(
        """
        INSERT INTO markets (condition_id, slug, question, venue, resolves_at, created_at, last_seen_at)
        VALUES ('factor-route-market', 'factor-route-market', 'Will the factors route return seeded rows?', 'polymarket', NULL, $1, $2)
        ON CONFLICT (condition_id) DO NOTHING
        """,
        ts_1,
        ts_2,
    )
    await connection.execute(
        """
        INSERT INTO tokens (token_id, condition_id, outcome)
        VALUES ('factor-route-market-yes', 'factor-route-market', 'YES')
        ON CONFLICT (token_id) DO NOTHING
        """
    )
    await connection.execute(
        """
        INSERT INTO markets (condition_id, slug, question, venue, resolves_at, created_at, last_seen_at)
        VALUES ('other-market', 'other-market', 'Will the factors route filter by market?', 'polymarket', NULL, $1, $2)
        ON CONFLICT (condition_id) DO NOTHING
        """,
        ts_1,
        ts_2,
    )
    await connection.execute(
        """
        INSERT INTO tokens (token_id, condition_id, outcome)
        VALUES ('other-market-yes', 'other-market', 'YES')
        ON CONFLICT (token_id) DO NOTHING
        """
    )
    await connection.execute(
        """
        INSERT INTO factor_values (factor_id, param, market_id, ts, value)
        VALUES
            ('orderbook_imbalance', '', 'factor-route-market', $1, 0.25),
            ('orderbook_imbalance', '', 'factor-route-market', $2, 0.10),
            ('orderbook_imbalance', '', 'factor-route-market', $3, 0.15),
            ('orderbook_imbalance', '', 'other-market', $2, -0.15),
            ('metaculus_prior', '', 'factor-route-market', $3, 0.72)
        """,
        ts_1,
        ts_2,
        ts_3,
    )
    return ts_1, ts_2, ts_3


@pytest.mark.asyncio(loop_scope="session")
async def test_factors_catalog_and_series_routes_return_seeded_rows(
    pg_pool: asyncpg.Pool,
) -> None:
    async with pg_pool.acquire() as connection:
        ts_1, ts_2, ts_3 = await _seed_catalog_and_series(connection)

    async with _app_client(pg_pool) as client:
        catalog_response = await client.get("/factors/catalog")
        series_response = await client.get(
            "/factors",
            params={
                "factor_id": "orderbook_imbalance",
                "market_id": "factor-route-market",
                "param": "",
                "since": ts_1.isoformat(),
                "limit": 2,
            },
        )

    assert catalog_response.status_code == 200
    assert catalog_response.json() == {
        "catalog": [
            {
                "factor_id": "metaculus_prior",
                "name": "Metaculus Prior",
                "description": "Raw Metaculus probability from the external signal payload.",
                "output_type": "probability",
                "direction": "neutral",
            },
            {
                "factor_id": "orderbook_imbalance",
                "name": "Orderbook Imbalance",
                "description": "Normalized bid-versus-ask depth imbalance from the current orderbook signal.",
                "output_type": "scalar",
                "direction": "neutral",
            },
        ]
    }
    assert series_response.status_code == 200
    assert series_response.json() == {
        "factor_id": "orderbook_imbalance",
        "param": "",
        "market_id": "factor-route-market",
        "points": [
            {
                "ts": ts_2.isoformat(),
                "value": 0.10,
            },
            {
                "ts": ts_3.isoformat(),
                "value": 0.15,
            },
        ],
    }
