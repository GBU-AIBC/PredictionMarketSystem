from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast

import asyncpg
import pytest

from pms.config import DatabaseSettings, PMSSettings
from pms.core.enums import MarketStatus, RunMode
from pms.core.models import Market, MarketSignal, Token
from pms.factors.base import FactorDefinition, FactorValueRow
from pms.factors.service import FactorService
from pms.runner import Runner
from pms.storage.eval_store import EvalStore
from pms.storage.feedback_store import FeedbackStore
from pms.storage.market_data_store import PostgresMarketDataStore
from tests.support.default_strategy_seed import seed_default_v1_strategy
from tests.support.fake_stores import InMemoryEvalStore, InMemoryFeedbackStore
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


class RepeatingSensor:
    def __init__(self, signal: MarketSignal, *, interval_s: float = 0.02) -> None:
        self._signal = signal
        self._interval_s = interval_s

    def __aiter__(self) -> AsyncIterator[MarketSignal]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MarketSignal]:
        tick = 0
        while True:
            yield replace(
                self._signal,
                fetched_at=self._signal.fetched_at + timedelta(milliseconds=20 * tick),
            )
            tick += 1
            await asyncio.sleep(self._interval_s)


class AlwaysFactor(FactorDefinition):
    factor_id = "orderbook_imbalance"
    required_inputs = ("orderbook",)

    def compute(
        self,
        signal: MarketSignal,
        outer_ring: object,
    ) -> FactorValueRow | None:
        del outer_ring
        return FactorValueRow(
            factor_id=self.factor_id,
            param="",
            market_id=signal.market_id,
            ts=signal.timestamp,
            value=0.25,
        )


class DeleteMarketAfterReadStore(PostgresMarketDataStore):
    def __init__(self, pool: asyncpg.Pool, *, market_id: str) -> None:
        super().__init__(pool)
        self._market_id = market_id
        self._deleted = False

    async def read_market(self, market_id: str) -> Market | None:
        market = await super().read_market(market_id)
        if market is not None and market_id == self._market_id and not self._deleted:
            async with self.pool.acquire() as connection:
                await connection.execute(
                    "DELETE FROM markets WHERE condition_id = $1",
                    market_id,
                )
            self._deleted = True
        return market


def _settings(*, factor_cadence_s: float = 0.05) -> PMSSettings:
    assert PMS_TEST_DATABASE_URL is not None
    return PMSSettings(
        mode=RunMode.PAPER,
        auto_migrate_default_v2=True,
        factor_cadence_s=factor_cadence_s,
        database=DatabaseSettings(
            dsn=PMS_TEST_DATABASE_URL,
            pool_min_size=1,
            pool_max_size=2,
        ),
    )


def _signal() -> MarketSignal:
    return MarketSignal(
        market_id="factor-lifecycle",
        token_id="factor-lifecycle-token",
        venue="polymarket",
        title="Will FactorService clean up on stop?",
        yes_price=0.4,
        volume_24h=1000.0,
        resolves_at=datetime(2026, 4, 30, tzinfo=UTC),
        orderbook={
            "bids": [{"price": 0.39, "size": 100.0}],
            "asks": [{"price": 0.41, "size": 100.0}],
        },
        external_signal={},
        fetched_at=datetime(2026, 4, 18, 9, 30, tzinfo=UTC),
        market_status=MarketStatus.OPEN.value,
    )


async def _seed_boot_prereqs(pg_pool: asyncpg.Pool) -> None:
    async with pg_pool.acquire() as connection:
        async with connection.transaction():
            await seed_factor_catalog(connection)
            await seed_default_v1_strategy(connection)
            store = PostgresMarketDataStore(pg_pool)
            ts = datetime(2026, 4, 18, 9, 30, tzinfo=UTC)
            await store.write_market(
                Market(
                    condition_id="factor-lifecycle",
                    slug="factor-lifecycle",
                    question="Will FactorService clean up on stop?",
                    venue="polymarket",
                    resolves_at=datetime(2026, 4, 30, tzinfo=UTC),
                    created_at=ts,
                    last_seen_at=ts,
                )
            )
            await store.write_token(
                Token(
                    token_id="factor-lifecycle-token",
                    condition_id="factor-lifecycle",
                    outcome="YES",
                )
            )


async def _assert_pool_still_works(pg_pool: asyncpg.Pool) -> None:
    async with pg_pool.acquire() as connection:
        value = await connection.fetchval("SELECT 1")
    assert value == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_factor_service_reseeds_source_factor_after_catalog_truncation(
    pg_pool: asyncpg.Pool,
) -> None:
    async with pg_pool.acquire() as connection:
        await connection.execute("DELETE FROM factor_values")
        await connection.execute(
            "DELETE FROM factors WHERE factor_id = 'orderbook_imbalance'"
        )

    service = FactorService(
        pool=pg_pool,
        store=PostgresMarketDataStore(pg_pool),
        cadence_s=0.05,
        factors=(AlwaysFactor,),
        signal_stream=RepeatingSensor(_signal()),
    )

    persisted = await service.compute_once([_signal()])

    async with pg_pool.acquire() as connection:
        factor_exists = await connection.fetchval(
            "SELECT EXISTS(SELECT 1 FROM factors WHERE factor_id = 'orderbook_imbalance')"
        )
        factor_values = await connection.fetchval(
            """
            SELECT COUNT(*)
            FROM factor_values
            WHERE factor_id = 'orderbook_imbalance'
            """
        )

    assert persisted == 1
    assert factor_exists is True
    assert factor_values == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_factor_service_recreates_market_shell_after_runtime_delete(
    pg_pool: asyncpg.Pool,
) -> None:
    await _seed_boot_prereqs(pg_pool)
    service = FactorService(
        pool=pg_pool,
        store=DeleteMarketAfterReadStore(pg_pool, market_id="factor-lifecycle"),
        cadence_s=0.05,
        factors=(AlwaysFactor,),
        signal_stream=RepeatingSensor(_signal()),
    )

    persisted = await service.compute_once([_signal()])

    async with pg_pool.acquire() as connection:
        market_exists = await connection.fetchval(
            "SELECT EXISTS(SELECT 1 FROM markets WHERE condition_id = 'factor-lifecycle')"
        )
        factor_values = await connection.fetchval(
            """
            SELECT COUNT(*)
            FROM factor_values
            WHERE factor_id = 'orderbook_imbalance'
              AND market_id = 'factor-lifecycle'
            """
        )

    assert persisted == 1
    assert market_exists is True
    assert factor_values == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_factor_service_task_cancels_on_normal_stop(
    pg_pool: asyncpg.Pool,
) -> None:
    await _seed_boot_prereqs(pg_pool)
    runner = Runner(
        config=_settings(),
        sensors=[RepeatingSensor(_signal())],
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )

    try:
        await runner.start()
        task = runner.factor_service_task
        assert task is not None
        await asyncio.sleep(0.15)
        await runner.stop()

        assert task.done()
    finally:
        if runner.pg_pool is not None or runner.tasks:
            await runner.stop()
    await _assert_pool_still_works(pg_pool)


@pytest.mark.asyncio(loop_scope="session")
async def test_factor_service_task_cancels_during_compute(
    pg_pool: asyncpg.Pool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_boot_prereqs(pg_pool)
    entered = asyncio.Event()

    async def blocking_persist(pool: object, row: object, *, definition_version: str = "1") -> None:
        del pool, row, definition_version
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("pms.factors.service.persist_factor_value", blocking_persist)
    monkeypatch.setattr("pms.runner.REGISTERED", (AlwaysFactor,))
    runner = Runner(
        config=_settings(),
        sensors=[RepeatingSensor(_signal())],
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )

    try:
        await runner.start()
        task = runner.factor_service_task
        assert task is not None
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        await runner.stop()

        assert task.done()
    finally:
        if runner.pg_pool is not None or runner.tasks:
            await runner.stop()
    await _assert_pool_still_works(pg_pool)


@pytest.mark.asyncio(loop_scope="session")
async def test_factor_service_stop_succeeds_after_compute_exception(
    pg_pool: asyncpg.Pool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_boot_prereqs(pg_pool)

    async def exploding_persist(pool: object, row: object, *, definition_version: str = "1") -> None:
        del pool, row, definition_version
        raise RuntimeError("factor persist boom")

    monkeypatch.setattr("pms.factors.service.persist_factor_value", exploding_persist)
    monkeypatch.setattr("pms.runner.REGISTERED", (AlwaysFactor,))
    runner = Runner(
        config=_settings(),
        sensors=[RepeatingSensor(_signal())],
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )

    try:
        await runner.start()
        task = runner.factor_service_task
        assert task is not None

        # Persistence failures are retried on the dirty market at the next
        # cadence; stopping the runner must still cancel the worker cleanly.
        await asyncio.sleep(0.12)
        assert not task.done()
        await runner.stop()
        assert task.done()
    finally:
        if runner.pg_pool is not None or runner.tasks:
            await runner.stop()
    await _assert_pool_still_works(pg_pool)
