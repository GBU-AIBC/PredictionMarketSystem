from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
import pytest

from pms.core.enums import MarketStatus
from pms.core.models import BookLevel, BookSnapshot, Market, MarketSignal, Token
from pms.factors.definitions.orderbook_imbalance import OrderbookImbalance
from pms.factors.service import persist_factor_value
from pms.storage.market_data_store import PostgresMarketDataStore


SCHEMA_PATH = Path("schema.sql")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("PMS_RUN_INTEGRATION") != "1",
        reason="set PMS_RUN_INTEGRATION=1 to run PostgreSQL integration tests",
    ),
    pytest.mark.skipif(
        os.environ.get("PMS_TEST_DATABASE_URL") is None,
        reason="set PMS_TEST_DATABASE_URL to the compose-backed PostgreSQL URI",
    ),
]


def _apply_schema(database_url: str) -> None:
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    env["SCHEMA_PATH"] = str(SCHEMA_PATH)
    subprocess.run(
        ["bash", "-lc", 'psql "$DATABASE_URL" --set ON_ERROR_STOP=1 -f "$SCHEMA_PATH"'],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )


def _signal(*, ts: datetime) -> MarketSignal:
    return MarketSignal(
        market_id="market-factor-e2e",
        token_id="token-factor-e2e",
        venue="polymarket",
        title="Will orderbook imbalance persist?",
        yes_price=0.47,
        volume_24h=1000.0,
        resolves_at=datetime(2026, 4, 30, tzinfo=UTC),
        orderbook={
            "bids": [
                {"price": 0.46, "size": 60.0},
                {"price": 0.45, "size": 40.0},
            ],
            "asks": [
                {"price": 0.48, "size": 30.0},
                {"price": 0.49, "size": 20.0},
            ],
        },
        external_signal={},
        fetched_at=ts,
        market_status=MarketStatus.OPEN.value,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_orderbook_imbalance_compute_and_persist(
    pg_pool: asyncpg.Pool,
) -> None:
    database_url = os.environ.get("PMS_TEST_DATABASE_URL")
    assert database_url is not None
    _apply_schema(database_url)

    store = PostgresMarketDataStore(pg_pool)
    ts = datetime(2026, 4, 18, 1, 10, tzinfo=UTC)

    await store.write_market(
        Market(
            condition_id="market-factor-e2e",
            slug="market-factor-e2e",
            question="Will orderbook imbalance persist?",
            venue="polymarket",
            resolves_at=datetime(2026, 4, 30, tzinfo=UTC),
            created_at=ts,
            last_seen_at=ts,
        )
    )
    await store.write_token(
        Token(
            token_id="token-factor-e2e",
            condition_id="market-factor-e2e",
            outcome="YES",
        )
    )

    snapshot_id = await store.write_book_snapshot(
        BookSnapshot(
            id=0,
            market_id="market-factor-e2e",
            token_id="token-factor-e2e",
            ts=ts,
            hash="factor-hash",
            source="subscribe",
        ),
        [
            BookLevel(
                snapshot_id=0,
                market_id="market-factor-e2e",
                side="BUY",
                price=0.46,
                size=60.0,
            ),
            BookLevel(
                snapshot_id=0,
                market_id="market-factor-e2e",
                side="BUY",
                price=0.45,
                size=40.0,
            ),
            BookLevel(
                snapshot_id=0,
                market_id="market-factor-e2e",
                side="SELL",
                price=0.48,
                size=30.0,
            ),
            BookLevel(
                snapshot_id=0,
                market_id="market-factor-e2e",
                side="SELL",
                price=0.49,
                size=20.0,
            ),
        ],
    )

    latest_snapshot = await store.read_latest_book_snapshot("market-factor-e2e")
    signal = _signal(ts=ts)
    row = OrderbookImbalance().compute(signal, store)

    assert latest_snapshot is not None
    assert latest_snapshot.id == snapshot_id
    assert row is not None

    await persist_factor_value(pg_pool, row)
    await persist_factor_value(pg_pool, row)
    await persist_factor_value(pg_pool, row, definition_version="orderbook-v2")

    async with pg_pool.acquire() as connection:
        factor_name = await connection.fetchval(
            "SELECT name FROM factors WHERE factor_id = $1",
            "orderbook_imbalance",
        )
        stored_row = await connection.fetchrow(
            """
            SELECT factor_id, param, market_id, ts, value
            FROM factor_values
            WHERE factor_id = $1
            """,
            "orderbook_imbalance",
        )
        stored_count = await connection.fetchval(
            """
            SELECT COUNT(*)
            FROM factor_values
            WHERE factor_id = $1
            """,
            "orderbook_imbalance",
        )
        definition_versions = await connection.fetch(
            """
            SELECT definition_version
            FROM factor_values
            WHERE factor_id = $1
            ORDER BY definition_version
            """,
            "orderbook_imbalance",
        )

    assert factor_name == "Orderbook Imbalance"
    assert stored_row is not None
    assert stored_count == 2
    assert [row["definition_version"] for row in definition_versions] == [
        "1",
        "orderbook-v2",
    ]
    assert stored_row["factor_id"] == row.factor_id
    assert stored_row["param"] == row.param
    assert stored_row["market_id"] == row.market_id
    assert stored_row["ts"] == row.ts
    assert stored_row["value"] == pytest.approx(row.value)
