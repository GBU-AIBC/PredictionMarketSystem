from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any, cast

import asyncpg
import pytest

from pms.factors.service import FactorService
from pms.factors.definitions import current_definition_version
from pms.research.cache import FactorPanelCache, FactorPanelKey
from pms.storage.market_data_store import PostgresMarketDataStore
from tests.integration.test_market_data_store import _market as _md_market
from tests.support.seed_factor_panel_fixture import EmptySignalStream
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


class _CountingConnection:
    def __init__(self, connection: asyncpg.Connection, counters: dict[str, int]) -> None:
        self._connection = connection
        self._counters = counters

    async def fetch(self, *args: object) -> list[asyncpg.Record]:
        self._counters["fetch"] += 1
        return cast(list[asyncpg.Record], await self._connection.fetch(*args))


class _CountingAcquireContext:
    def __init__(self, pool: "_CountingPool") -> None:
        self._pool = pool
        self._acquire_context: Any | None = None

    async def __aenter__(self) -> _CountingConnection:
        self._acquire_context = self._pool._pool.acquire()
        connection = await self._acquire_context.__aenter__()
        return _CountingConnection(connection, self._pool.counters)

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> object:
        assert self._acquire_context is not None
        return await self._acquire_context.__aexit__(exc_type, exc, tb)


class _CountingPool:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self.counters = {"fetch": 0}

    def acquire(self) -> _CountingAcquireContext:
        return _CountingAcquireContext(self)


@pytest.mark.asyncio(loop_scope="session")
async def test_factor_panel_cache_wraps_factor_service_with_single_pg_round_trip(
    pg_pool: asyncpg.Pool,
) -> None:
    ts_start = datetime(2026, 4, 1, tzinfo=UTC)
    ts_mid = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
    ts_end = datetime(2026, 4, 30, 23, 59, tzinfo=UTC)
    store = PostgresMarketDataStore(pg_pool)
    await store.write_market(_md_market(condition_id="factor-cache-a", slug="factor-cache-a"))
    await store.write_market(_md_market(condition_id="factor-cache-b", slug="factor-cache-b"))

    async with pg_pool.acquire() as connection:
        await seed_factor_catalog(connection, factor_ids=("orderbook_imbalance",))
        await connection.execute(
            """
            INSERT INTO factor_values (factor_id, param, market_id, ts, value, definition_version)
            VALUES
                ('orderbook_imbalance', '', 'factor-cache-a', $1, 0.11, $3),
                ('orderbook_imbalance', '', 'factor-cache-b', $2, 0.22, $3),
                ('orderbook_imbalance', '', 'factor-cache-a', $1, -0.99, '1')
            """,
            ts_mid,
            ts_mid,
            current_definition_version("orderbook_imbalance"),
        )

    counting_pool = _CountingPool(pg_pool)
    service = FactorService(
        pool=cast(asyncpg.Pool, counting_pool),
        store=store,
        cadence_s=1.0,
        factors=(),
        signal_stream=EmptySignalStream(),
    )
    cache = FactorPanelCache()
    key = FactorPanelKey.from_inputs(
        factor_id="orderbook_imbalance",
        param="",
        market_ids=["factor-cache-b", "factor-cache-a"],
        ts_start=ts_start,
        ts_end=ts_end,
    )

    for _ in range(5):
        panel = cache.get(key)
        if panel is None:
            panel = await service.get_panel(
                "orderbook_imbalance",
                "",
                ["factor-cache-b", "factor-cache-a"],
                ts_start,
                ts_end,
            )
            cache.put(key, panel)

        assert list(panel) == ["factor-cache-b", "factor-cache-a"]
        assert len(panel["factor-cache-a"]) == 1
        assert panel["factor-cache-a"][0].value == 0.11
        assert panel["factor-cache-b"][0].value == 0.22

    assert counting_pool.counters["fetch"] == 1
    assert cache.hits == 4
    assert cache.misses == 1
