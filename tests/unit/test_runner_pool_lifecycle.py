from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from pms.config import DatabaseSettings, PMSSettings, RiskSettings
from pms.core.enums import MarketStatus, RunMode
from pms.core.models import MarketSignal
from pms.runner import Runner
from tests.support.fake_stores import LegacyPathEvalStore


FIXTURE_PATH = Path("tests/fixtures/polymarket_7day_synthetic.jsonl")


@dataclass
class FakePool:
    close_calls: int = 0
    closed: bool = False

    def __post_init__(self) -> None:
        self._release_acquires = asyncio.Event()
        self._release_acquires.set()

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        self._release_acquires.set()

    def acquire(self) -> "_FakeAcquireContext":
        return _FakeAcquireContext(self)


class _FakeAcquireContext:
    def __init__(self, pool: FakePool) -> None:
        self._pool = pool
        self._connection = object()

    async def _wait(self) -> object:
        await self._pool._release_acquires.wait()
        return self._connection

    def __await__(self) -> Any:
        return self._wait().__await__()

    async def __aenter__(self) -> object:
        return await self._wait()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> bool:
        del exc_type, exc, tb
        return False


class HoldingSensor:
    def __aiter__(self) -> AsyncIterator[MarketSignal]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MarketSignal]:
        while True:
            await asyncio.sleep(60.0)
            yield _signal()


class OneShotSensor:
    def __init__(self, signal: MarketSignal) -> None:
        self.signal = signal

    def __aiter__(self) -> AsyncIterator[MarketSignal]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MarketSignal]:
        yield self.signal


class ExplodingController:
    async def decide(self, signal: MarketSignal, portfolio: Any) -> Any:
        raise RuntimeError("steady-state boom")


def _settings() -> PMSSettings:
    return PMSSettings(
        mode=RunMode.BACKTEST,
        auto_migrate_default_v2=False,
        database=DatabaseSettings(
            dsn="postgresql://localhost/pms_test_runner",
            pool_min_size=2,
            pool_max_size=10,
        ),
        risk=RiskSettings(
            max_position_per_market=1000.0,
            max_total_exposure=10_000.0,
        ),
    )


def _runner(tmp_path: Path, **kwargs: Any) -> Runner:
    return Runner(
        config=_settings(),
        historical_data_path=FIXTURE_PATH,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _stub_factor_catalog_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop_ensure_factor_catalog(pool: object, *, factor_ids: object = None) -> None:
        del pool, factor_ids

    monkeypatch.setattr("pms.runner.ensure_factor_catalog", _noop_ensure_factor_catalog)


@pytest.fixture(autouse=True)
def _stub_factor_service(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoopFactorService:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        async def run(self) -> None:
            return None

    monkeypatch.setattr("pms.runner.FactorService", _NoopFactorService)


def _signal() -> MarketSignal:
    return MarketSignal(
        market_id="runner-pool-market",
        token_id="yes-token",
        venue="polymarket",
        title="Will runner pool lifecycle tests pass?",
        yes_price=0.42,
        volume_24h=1_000.0,
        resolves_at=datetime(2026, 4, 30, tzinfo=UTC),
        orderbook={
            "bids": [{"price": 0.41, "size": 100.0}],
            "asks": [{"price": 0.43, "size": 100.0}],
        },
        external_signal={"fair_value": 0.55, "resolved_outcome": 1.0},
        fetched_at=datetime(2026, 4, 16, tzinfo=UTC),
        market_status=MarketStatus.OPEN.value,
    )


async def _wait_for(predicate: Any, *, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_runner_start_creates_pool_with_configured_args_and_stop_closes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool = FakePool()
    call_args: dict[str, Any] = {}

    async def fake_create_pool(*, dsn: str, min_size: int, max_size: int) -> FakePool:
        call_args["dsn"] = dsn
        call_args["min_size"] = min_size
        call_args["max_size"] = max_size
        return fake_pool

    monkeypatch.setattr("pms.runner.asyncpg.create_pool", fake_create_pool)
    runner = _runner(tmp_path, sensors=[HoldingSensor()])

    await runner.start()
    await asyncio.wait_for(runner.stop(), timeout=5.0)

    assert call_args == {
        "dsn": "postgresql://localhost/pms_test_runner",
        "min_size": 2,
        "max_size": 10,
    }
    assert fake_pool.closed is True
    assert fake_pool.close_calls == 1
    assert runner.pg_pool is None


@pytest.mark.asyncio
async def test_runner_start_re_raises_when_create_pool_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_create_pool(*, dsn: str, min_size: int, max_size: int) -> FakePool:
        raise RuntimeError("pool create failed")

    monkeypatch.setattr("pms.runner.asyncpg.create_pool", fake_create_pool)
    runner = _runner(tmp_path, sensors=[HoldingSensor()])

    with pytest.raises(RuntimeError, match="pool create failed"):
        await runner.start()

    assert runner.pg_pool is None


@pytest.mark.asyncio
async def test_runner_start_closes_pool_when_startup_fails_after_acquire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool = FakePool()

    async def fake_create_pool(*, dsn: str, min_size: int, max_size: int) -> FakePool:
        return fake_pool

    async def fake_sensor_start(sensors: Any, **kwargs: Any) -> None:
        del sensors, kwargs
        raise RuntimeError("sensor start failed")

    monkeypatch.setattr("pms.runner.asyncpg.create_pool", fake_create_pool)
    runner = _runner(tmp_path, sensors=[HoldingSensor()])
    monkeypatch.setattr(runner.sensor_stream, "start", fake_sensor_start)

    with pytest.raises(RuntimeError, match="sensor start failed"):
        await runner.start()

    assert fake_pool.closed is True
    assert runner.pg_pool is None


@pytest.mark.asyncio
async def test_runner_run_closes_pool_on_steady_state_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool = FakePool()

    async def fake_create_pool(*, dsn: str, min_size: int, max_size: int) -> FakePool:
        return fake_pool

    monkeypatch.setattr("pms.runner.asyncpg.create_pool", fake_create_pool)
    runner = _runner(
        tmp_path,
        sensors=[OneShotSensor(_signal())],
        controller=ExplodingController(),
    )

    with pytest.raises(RuntimeError, match="steady-state boom"):
        await runner.run()

    assert fake_pool.closed is True
    assert runner.pg_pool is None
    assert runner.task is None


@pytest.mark.asyncio
async def test_runner_task_cancellation_closes_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool = FakePool()

    async def fake_create_pool(*, dsn: str, min_size: int, max_size: int) -> FakePool:
        return fake_pool

    monkeypatch.setattr("pms.runner.asyncpg.create_pool", fake_create_pool)
    runner = _runner(tmp_path, sensors=[HoldingSensor()])

    task = asyncio.create_task(runner.run())
    await _wait_for(lambda: runner.task is not None and runner.pg_pool is not None)
    assert runner.task is not None
    runner.task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_pool.closed is True
    assert runner.pg_pool is None
    assert runner.task is None


@pytest.mark.asyncio
async def test_runner_stop_completes_with_outstanding_pool_acquires(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool = FakePool()

    async def fake_create_pool(*, dsn: str, min_size: int, max_size: int) -> FakePool:
        return fake_pool

    monkeypatch.setattr("pms.runner.asyncpg.create_pool", fake_create_pool)
    runner = _runner(tmp_path, sensors=[HoldingSensor()])

    await runner.start()
    fake_pool._release_acquires.clear()
    async def _acquire() -> object:
        return await fake_pool.acquire()

    acquires = [asyncio.create_task(_acquire()) for _ in range(3)]
    await asyncio.wait_for(runner.stop(), timeout=5.0)
    await asyncio.wait_for(asyncio.gather(*acquires), timeout=5.0)

    assert fake_pool.closed is True
    assert runner.pg_pool is None


@pytest.mark.asyncio
async def test_runner_bind_pg_pool_publicly_without_taking_ownership(
    tmp_path: Path,
) -> None:
    fake_pool = FakePool()
    runner = _runner(tmp_path, sensors=[HoldingSensor()])

    runner.bind_pg_pool(cast(Any, fake_pool))

    assert runner.pg_pool is fake_pool
    await runner.close_pg_pool()

    assert runner.pg_pool is None
    assert fake_pool.close_calls == 0  # type: ignore[unreachable]


@pytest.mark.asyncio
async def test_runner_close_pg_pool_unbinds_eval_and_feedback_stores(
    tmp_path: Path,
) -> None:
    from pms.storage.decision_store import DecisionStore
    from pms.storage.eval_store import EvalStore
    from pms.storage.feedback_store import FeedbackStore
    from pms.storage.fill_store import FillStore
    from pms.storage.order_store import OrderStore

    fake_pool = FakePool()
    runner = _runner(
        tmp_path,
        sensors=[HoldingSensor()],
        eval_store=EvalStore(),
        feedback_store=FeedbackStore(),
        decision_store=DecisionStore(),
        order_store=OrderStore(),
        fill_store=FillStore(),
    )
    eval_store = runner.eval_store
    feedback_store = runner.feedback_store
    decision_store = runner.decision_store
    order_store = runner.order_store
    fill_store = runner.fill_store

    runner.bind_pg_pool(cast(Any, fake_pool))

    assert eval_store.pool is fake_pool
    assert feedback_store.pool is fake_pool
    assert decision_store.pool is fake_pool
    assert order_store.pool is fake_pool
    assert fill_store.pool is fake_pool

    await runner.close_pg_pool()

    assert getattr(eval_store, "pool", None) is None
    assert getattr(feedback_store, "pool", None) is None
    assert getattr(decision_store, "pool", None) is None
    assert getattr(order_store, "pool", None) is None
    assert getattr(fill_store, "pool", None) is None


@pytest.mark.asyncio
async def test_runner_start_rejects_legacy_jsonl_store_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_pool_called = False

    async def fake_create_pool(*, dsn: str, min_size: int, max_size: int) -> FakePool:
        nonlocal create_pool_called
        del dsn, min_size, max_size
        create_pool_called = True
        raise AssertionError("create_pool should not be called")

    monkeypatch.setattr("pms.runner.asyncpg.create_pool", fake_create_pool)
    runner = _runner(
        tmp_path,
        sensors=[HoldingSensor()],
        eval_store=cast(Any, LegacyPathEvalStore(tmp_path / "eval_records.jsonl")),
    )

    with pytest.raises(RuntimeError, match="legacy JSONL path referenced"):
        await runner.start()

    assert create_pool_called is False
    assert runner.pg_pool is None


def test_backtest_runtime_ignores_auto_migrate_without_explicit_database() -> None:
    runner = Runner(config=PMSSettings(mode=RunMode.BACKTEST, auto_migrate_default_v2=True))

    assert runner._should_boot_postgres_runtime() is False
