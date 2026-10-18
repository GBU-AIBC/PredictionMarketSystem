from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pms.config import PMSSettings, SensorSettings
from pms.core.enums import RunMode
from pms.core.models import MarketSignal, Portfolio, TradeDecision
from pms.runner import Runner, StrategyControllerRuntime


class RecordingController:
    def __init__(self) -> None:
        self.signals: list[MarketSignal] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def decide(
        self, signal: MarketSignal, portfolio: Portfolio | None = None,
    ) -> TradeDecision | None:
        self.signals.append(signal)
        self.entered.set()
        await self.release.wait()
        return None


class BurstSensor:
    def __init__(self, count: int) -> None:
        self.count = count
        self.emitted = 0
        self.closed = asyncio.Event()

    def __aiter__(self) -> AsyncIterator[MarketSignal]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MarketSignal]:
        try:
            for tick in range(self.count):
                self.emitted += 1
                yield _signal(("a", "b", "c")[tick % 3], tick)
                await asyncio.sleep(0)
        finally:
            self.closed.set()


def _runner(mode: RunMode, *, limit: int | None = 3) -> Runner:
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.PAPER,
            auto_migrate_default_v2=False,
            sensor=SensorSettings(max_subscription_asset_ids=limit),
        ),
        historical_data_path=Path("tests/fixtures/polymarket_7day_synthetic.jsonl"),
    )
    # Exercise LIVE dispatch without starting sensors, adapters or order submission.
    runner.config.mode = mode
    return runner


def _attach(
    runner: Runner, controller: RecordingController, *, strategy_id: str = "test",
    assets: frozenset[str] | None = None,
) -> asyncio.Queue[MarketSignal]:
    runner._attach_controller_runtime(StrategyControllerRuntime(
        strategy_id=strategy_id, strategy_version_id="v1",
        controller=controller, asset_ids=assets,
    ))
    return runner._controller_signal_queues[strategy_id]


def _signal(token: str, tick: int = 0) -> MarketSignal:
    return MarketSignal(
        market_id=f"market-{token}", token_id=token, venue="polymarket",
        title=token, yes_price=0.5, volume_24h=1000.0,
        resolves_at=datetime(2030, 1, 1, tzinfo=UTC),
        orderbook={"bids": [], "asks": []}, external_signal={"tick": tick},
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=tick),
        market_status="open",
    )


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
async def test_sustained_overload_keeps_latest_per_token_and_fifo_fairness(
    mode: RunMode,
) -> None:
    runner = _runner(mode)
    controller = RecordingController()
    queue = _attach(runner, controller, assets=frozenset({"a", "b", "c"}))
    runner._controller_task = asyncio.create_task(runner._controller_loop())
    try:
        await runner.sensor_stream.queue.put(_signal("a"))
        await asyncio.wait_for(controller.entered.wait(), 1)
        for tick in range(1, 2001):
            for token in ("b", "a", "c"):
                await runner.sensor_stream.queue.put(_signal(token, tick))
        await asyncio.wait_for(runner.sensor_stream.queue.join(), 5)
        assert queue.maxsize == 3
        assert queue.qsize() == 3
        controller.release.set()
        await asyncio.wait_for(queue.join(), 2)
        assert [(s.token_id, s.external_signal["tick"]) for s in controller.signals] == [
            ("a", 0), ("b", 2000), ("a", 2000), ("c", 2000),
        ]
    finally:
        await runner.stop()


async def test_pending_replacement_uses_event_time_and_preserves_other_tokens() -> None:
    runner = _runner(RunMode.PAPER)
    controller = RecordingController()
    queue = _attach(runner, controller)
    try:
        # Queue before the consumer starts; the FIFO position must survive replacement.
        await queue.put(_signal("a", 1))
        await queue.put(_signal("b", 2))
        await queue.put(_signal("a", 3))
        await queue.put(_signal("a", 0))
        assert queue.qsize() == 2
        controller.release.set()
        await asyncio.wait_for(queue.join(), 1)
        assert controller.signals == [_signal("a", 3), _signal("b", 2)]
    finally:
        await runner.stop()


async def test_pending_replacement_prefers_book_time_over_fetch_time() -> None:
    runner = _runner(RunMode.PAPER)
    controller = RecordingController()
    queue = _attach(runner, controller)
    newest_book = replace(_signal("a", 1), external_signal={
        "book_received_at": "2026-01-01T00:00:10Z",
    })
    older_book = replace(_signal("a", 20), external_signal={
        "book_received_at": "2026-01-01T00:00:09Z",
    })
    try:
        await queue.put(newest_book)
        await queue.put(older_book)
        assert queue.qsize() == 1
        assert queue.get_nowait() == newest_book
        queue.task_done()
        await asyncio.wait_for(queue.join(), 1)
    finally:
        await runner.stop()


async def test_backtest_bounds_backlog_and_preserves_every_ordered_tick() -> None:
    runner = _runner(RunMode.BACKTEST, limit=2)
    controller = RecordingController()
    queue = _attach(runner, controller)
    runner._controller_task = asyncio.create_task(runner._controller_loop())
    signals = [_signal("a" if tick % 2 else "b", tick) for tick in range(60)]
    producer: asyncio.Task[None] | None = None

    async def produce() -> None:
        for signal in signals[1:]:
            await runner.sensor_stream.queue.put(signal)

    try:
        await runner.sensor_stream.queue.put(signals[0])
        await asyncio.wait_for(controller.entered.wait(), 1)
        producer = asyncio.create_task(produce())
        await _wait_until(lambda: queue.qsize() >= 2)
        await asyncio.sleep(0.01)
        assert queue.maxsize == 2
        assert queue.qsize() == 2
        assert len(runner.state.signals) == 4  # in flight, two queued, one blocked put
        controller.release.set()
        await asyncio.wait_for(producer, 2)
        await asyncio.wait_for(runner.sensor_stream.queue.join(), 2)
        await asyncio.wait_for(queue.join(), 2)
        assert controller.signals == signals
    finally:
        if producer is not None and not producer.done():
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
        await runner.stop()


async def test_backtest_does_not_coalesce_prefilled_ticks() -> None:
    runner = _runner(RunMode.BACKTEST)
    controller = RecordingController()
    queue = _attach(runner, controller)
    signals = [_signal("a", tick) for tick in range(3)]
    try:
        for signal in signals:
            await queue.put(signal)
        controller.release.set()
        await asyncio.wait_for(queue.join(), 1)
        assert controller.signals == signals
    finally:
        await runner.stop()


@pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.BACKTEST])
async def test_detach_wakes_blocked_fanout_and_clears_pending_state(mode: RunMode) -> None:
    runner = _runner(mode, limit=1)
    slow = RecordingController()
    healthy = RecordingController()
    healthy.release.set()
    queue = _attach(runner, slow)
    _attach(runner, healthy, strategy_id="healthy")
    runner._controller_task = asyncio.create_task(runner._controller_loop())
    try:
        await runner.sensor_stream.queue.put(_signal("a"))
        await asyncio.wait_for(slow.entered.wait(), 1)
        await runner.sensor_stream.queue.put(_signal("b"))
        await runner.sensor_stream.queue.put(_signal("c"))
        await _wait_until(lambda: len(runner.state.signals) == 3)
        await asyncio.sleep(0.01)
        assert queue.full()
        await asyncio.wait_for(runner._release_controller_runtime("test"), 1)
        await asyncio.wait_for(runner.sensor_stream.queue.join(), 1)
        await asyncio.wait_for(queue.join(), 1)
        assert queue.empty()
        assert "test" not in runner._controller_signal_queues
        await _wait_until(lambda: len(healthy.signals) == 3)
        assert [s.token_id for s in healthy.signals] == ["a", "b", "c"]
        with pytest.raises(asyncio.QueueShutDown):
            await queue.put(_signal("b", 99))
        replacement = RecordingController()
        replacement.release.set()
        new_queue = _attach(runner, replacement)
        await runner.sensor_stream.queue.put(_signal("b", 100))
        await asyncio.wait_for(runner.sensor_stream.queue.join(), 1)
        await asyncio.wait_for(new_queue.join(), 1)
        assert replacement.signals == [_signal("b", 100)]
    finally:
        await runner.stop()


@pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.BACKTEST])
async def test_stop_cancels_blocked_put_and_releases_queue_join(mode: RunMode) -> None:
    runner = _runner(mode, limit=1)
    controller = RecordingController()
    queue = _attach(runner, controller)
    runner._controller_task = asyncio.create_task(runner._controller_loop())
    await runner.sensor_stream.queue.put(_signal("a"))
    await asyncio.wait_for(controller.entered.wait(), 1)
    await runner.sensor_stream.queue.put(_signal("b"))
    await runner.sensor_stream.queue.put(_signal("c"))
    await _wait_until(lambda: len(runner.state.signals) == 3)
    await asyncio.wait_for(runner.stop(), 1)
    await asyncio.wait_for(queue.join(), 1)
    assert queue.empty()
    assert runner._controller_signal_queues == {}


async def test_unconstrained_scope_still_has_finite_queue_capacity() -> None:
    runner = _runner(RunMode.PAPER, limit=None)
    queue = _attach(runner, RecordingController())
    try:
        assert 0 < queue.maxsize <= 1000
    finally:
        await runner.stop()


@pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.BACKTEST])
async def test_restart_uses_fresh_queue_without_old_pending_signals(mode: RunMode) -> None:
    runner = _runner(mode)
    controller = RecordingController()
    runner.controller = controller
    runner.sensors = []
    try:
        await runner.start()
        old_queue = runner._controller_signal_queues["default"]
        await runner.sensor_stream.queue.put(_signal("a"))
        await asyncio.wait_for(controller.entered.wait(), 1)
        await old_queue.put(_signal("b"))
        await runner.stop()
        await asyncio.wait_for(old_queue.join(), 1)
        assert old_queue.empty()
        controller.release.set()
        await runner.start()
        new_queue = runner._controller_signal_queues["default"]
        assert new_queue is not old_queue
        await runner.sensor_stream.queue.put(_signal("c"))
        await asyncio.wait_for(runner.sensor_stream.queue.join(), 1)
        await asyncio.wait_for(new_queue.join(), 1)
        assert [s.token_id for s in controller.signals] == ["a", "c"]
    finally:
        await runner.stop()


async def test_start_failure_closes_queues_and_releases_pending_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(RunMode.BACKTEST)
    queues: list[asyncio.Queue[MarketSignal]] = []

    async def configure_then_fail() -> None:
        queue = _attach(runner, RecordingController())
        queues.append(queue)
        await queue.put(_signal("a"))
        raise RuntimeError("startup failed after attaching strategy")

    monkeypatch.setattr(runner, "_configure_controllers", configure_then_fail)
    with pytest.raises(RuntimeError, match="startup failed"):
        await runner.start()
    assert runner._controller_signal_queues == {}
    assert queues[0].empty()
    await asyncio.wait_for(queues[0].join(), 1)


async def test_actual_backtest_producer_is_bounded_and_delivers_every_tick() -> None:
    runner = _runner(RunMode.BACKTEST, limit=2)
    controller = RecordingController()
    strategy_queue = _attach(runner, controller)
    runner._controller_task = asyncio.create_task(runner._controller_loop())
    sensor = BurstSensor(10_000)
    try:
        await runner.sensor_stream.start([sensor])
        await asyncio.wait_for(controller.entered.wait(), 1)
        await _wait_until(lambda: strategy_queue.full())
        await asyncio.sleep(0.02)
        shared = runner.sensor_stream.queue
        assert shared.maxsize == 2
        assert shared.qsize() == 2
        # Two queues, one controller, one dispatcher, one blocked sensor put.
        assert sensor.emitted == 7
        assert not sensor.closed.is_set()
        controller.release.set()
        await asyncio.wait_for(asyncio.gather(*runner.sensor_stream.tasks), 10)
        await asyncio.wait_for(shared.join(), 2)
        await asyncio.wait_for(strategy_queue.join(), 2)
        assert [s.external_signal["tick"] for s in controller.signals] == list(range(10_000))
    finally:
        await runner.stop()


@pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
async def test_actual_producer_stalled_dispatcher_has_bounded_pending_work(
    mode: RunMode, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(mode)
    controller = RecordingController()
    queue = _attach(runner, controller, assets=frozenset({"a", "b", "c"}))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def stalled_exit(_signal: MarketSignal) -> None:
        entered.set()
        await release.wait()

    monkeypatch.setattr(runner, "_emit_position_exit_decisions", stalled_exit)
    runner._controller_task = asyncio.create_task(runner._controller_loop())
    sensor = BurstSensor(10_000)
    try:
        await runner.sensor_stream.start([sensor])
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.sleep(0.02)
        assert runner.sensor_stream.queue.maxsize == 3
        assert runner.sensor_stream.queue.qsize() <= 3
        assert queue.qsize() <= 3
        # All retained signals are in bounded queues or one in-flight slot per task.
        assert sensor.emitted <= 5
        release.set()
        await asyncio.wait_for(asyncio.gather(*runner.sensor_stream.tasks), 10)
        await asyncio.wait_for(runner.sensor_stream.queue.join(), 2)
        controller.release.set()
        await asyncio.wait_for(queue.join(), 2)
        latest = {s.token_id: s.external_signal["tick"] for s in controller.signals}
        assert latest == {"a": 9999, "b": 9997, "c": 9998}
    finally:
        await runner.stop()


@pytest.mark.parametrize("exit_path", ["stop", "failure", "cancellation"])
async def test_blocked_sensor_put_cleanup_and_restart(
    exit_path: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(RunMode.BACKTEST, limit=1)
    sensor = BurstSensor(10_000)
    runner.sensors = [sensor]
    runner.controller = RecordingController()
    entered = asyncio.Event()
    release = asyncio.Event()
    original_start = runner._evaluator_spool.start

    async def stalled_start() -> None:
        entered.set()
        await release.wait()
        if exit_path == "failure":
            raise RuntimeError("startup after sensors")
        await original_start()

    monkeypatch.setattr(runner._evaluator_spool, "start", stalled_start)
    start = asyncio.create_task(runner.start())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await _wait_until(lambda: sensor.emitted == 2)
        old_queue = runner.sensor_stream.queue
        if exit_path == "cancellation":
            start.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(start, 1)
        elif exit_path == "failure":
            release.set()
            with pytest.raises(RuntimeError, match="startup after sensors"):
                await asyncio.wait_for(start, 1)
        else:
            release.set()
            await asyncio.wait_for(start, 1)
            await asyncio.wait_for(runner.stop(), 1)
        assert sensor.closed.is_set()
        assert runner.sensor_stream.tasks == ()
        assert old_queue.empty()
        await asyncio.wait_for(old_queue.join(), 1)
        with pytest.raises(asyncio.QueueShutDown):
            await old_queue.put(_signal("stale"))
        monkeypatch.setattr(runner._evaluator_spool, "start", original_start)
        runner.sensors = []
        await runner.start()
        assert runner.sensor_stream.queue is not old_queue
        assert runner.sensor_stream.queue.empty()
    finally:
        if not start.done():
            start.cancel()
            await asyncio.gather(start, return_exceptions=True)
        await runner.stop()


@pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
async def test_ingress_backpressure_preserves_existing_stale_book_gate(
    mode: RunMode, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(mode, limit=1)
    controller = RecordingController()
    controller.release.set()
    queue = _attach(runner, controller)
    entered = asyncio.Event()
    release = asyncio.Event()
    now = datetime.now(tz=UTC)
    stale = replace(_signal("a"), external_signal={
        "book_received_at": (now - timedelta(minutes=1)).isoformat(),
    })
    fresh = replace(_signal("b"), external_signal={"book_received_at": now.isoformat()})

    class Sensor:
        async def __aiter__(self) -> AsyncIterator[MarketSignal]:
            for signal in (stale, fresh):
                yield signal

    async def stalled_exit(_signal: MarketSignal) -> None:
        entered.set()
        await release.wait()

    monkeypatch.setattr(runner, "_emit_position_exit_decisions", stalled_exit)
    runner._controller_task = asyncio.create_task(runner._controller_loop())
    try:
        await runner.sensor_stream.start([Sensor()])
        await asyncio.wait_for(entered.wait(), 1)
        assert runner.sensor_stream.queue.maxsize == 1
        release.set()
        await asyncio.wait_for(asyncio.gather(*runner.sensor_stream.tasks), 1)
        await asyncio.wait_for(runner.sensor_stream.queue.join(), 1)
        await asyncio.wait_for(queue.join(), 1)
        assert controller.signals == [fresh]
        assert any(d.code == "router_gate:book_too_stale" for d in runner.state.controller_diagnostics)
    finally:
        await runner.stop()
