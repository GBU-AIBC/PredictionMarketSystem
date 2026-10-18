from __future__ import annotations

import asyncio
from collections.abc import Callable
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
    try:
        await runner.sensor_stream.queue.put(signals[0])
        await asyncio.wait_for(controller.entered.wait(), 1)
        for signal in signals[1:]:
            await runner.sensor_stream.queue.put(signal)
        await _wait_until(lambda: queue.qsize() >= 2)
        await asyncio.sleep(0.01)
        assert queue.maxsize == 2
        assert queue.qsize() == 2
        assert len(runner.state.signals) == 4  # in flight, two queued, one blocked put
        controller.release.set()
        await asyncio.wait_for(runner.sensor_stream.queue.join(), 2)
        await asyncio.wait_for(queue.join(), 2)
        assert controller.signals == signals
    finally:
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
