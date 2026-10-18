from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, cast

import pytest
import asyncpg

from pms.core.models import Market, MarketSignal
from pms.core.interfaces import ISensor
from pms.factors.service import FactorService
from pms.sensor.stream import SensorStream, SignalSubscription
from tests.unit.test_factor_service import FakeStore, SequenceSignalStream
from tests.unit.test_runner_signal_queues import BurstSensor, _signal, _wait_until


async def test_full_subscription_close_wakes_put_and_drains_in_order() -> None:
    subscription = SignalSubscription(capacity=2)
    await subscription.push(_signal("a", 1))
    await subscription.push(_signal("b", 2))
    blocked = asyncio.create_task(subscription.push(_signal("c", 3)))
    await asyncio.sleep(0)
    assert not blocked.done()
    subscription.close()
    subscription.close()
    await asyncio.wait_for(blocked, 1)
    assert [signal async for signal in subscription] == [_signal("a", 1), _signal("b", 2)]
    await asyncio.wait_for(subscription._queue.join(), 1)


async def test_subscriber_backpressure_preserves_all_historical_events() -> None:
    stream = SensorStream(capacity=2)
    subscription = stream.subscribe()
    sensor = BurstSensor(1000)

    async def drain_main() -> list[MarketSignal]:
        signals = []
        for _ in range(sensor.count):
            signals.append(await stream.queue.get())
            stream.queue.task_done()
        return signals

    await stream.start([sensor])
    main_task = asyncio.create_task(drain_main())
    try:
        await _wait_until(lambda: subscription._queue.full())
        await asyncio.sleep(0.02)
        assert sensor.emitted == 3  # two subscribed signals and one blocked tee put
        subscribed = [signal async for signal in subscription]
        main = await asyncio.wait_for(main_task, 2)
        assert subscribed == main
        assert [s.external_signal["tick"] for s in main] == list(range(sensor.count))
    finally:
        main_task.cancel()
        await asyncio.gather(main_task, return_exceptions=True)
        await stream.stop()


async def test_stopped_full_tee_releases_both_queues_and_sensor_iterator() -> None:
    stream = SensorStream(capacity=1)
    subscription = stream.subscribe()
    sensor = BurstSensor(1000)
    await stream.start([sensor])
    first = await stream.queue.get()
    stream.queue.task_done()
    await _wait_until(lambda: subscription._queue.full() and stream.queue.full())
    await asyncio.wait_for(stream.stop(), 1)
    assert sensor.closed.is_set()
    assert stream.queue.empty()
    await asyncio.wait_for(stream.queue.join(), 1)
    assert [signal async for signal in subscription] == [first]


async def test_failed_sensor_closes_full_subscription_after_pending_drain() -> None:
    class FailingSensor(BurstSensor):
        async def _iterate(self) -> AsyncIterator[MarketSignal]:
            yield _signal("a")
            raise RuntimeError("sensor failed")

    stream = SensorStream(capacity=1)
    subscription = stream.subscribe()
    await stream.start([FailingSensor(1)])
    results = await asyncio.gather(*stream.tasks, return_exceptions=True)
    assert isinstance(results[0], RuntimeError)
    assert [signal async for signal in subscription] == [_signal("a")]
    await stream.stop()


async def test_partial_sensor_start_failure_cancels_created_tasks() -> None:
    stream = SensorStream(capacity=1)
    created: list[asyncio.Task[None]] = []

    def factory(index: int, sensor: ISensor) -> asyncio.Task[None]:
        if index == 1:
            raise RuntimeError("sensor factory failed")
        task = asyncio.create_task(stream.consume(sensor))
        created.append(task)
        return task

    with pytest.raises(RuntimeError, match="sensor factory failed"):
        await stream.start([BurstSensor(10), BurstSensor(10)], task_factory=factory)
    assert all(task.cancelled() for task in created)
    assert stream.tasks == ()
    assert stream.queue.empty()
    await stream.start([])
    await stream.stop()


async def test_stalled_factor_compute_cannot_displace_backlog_into_forward_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = SensorStream(capacity=2)
    subscription = stream.subscribe()
    service = FactorService(
        pool=cast(Any, object()), store=cast(Any, FakeStore()),
        cadence_s=0.001, factors=(), signal_stream=subscription,
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    computing: list[MarketSignal] = []

    async def stalled_compute(signals: list[MarketSignal]) -> int:
        computing[:] = signals
        entered.set()
        await release.wait()
        return 0

    async def drain_main() -> None:
        while True:
            await stream.queue.get()
            stream.queue.task_done()

    monkeypatch.setattr(service, "compute_once", stalled_compute)
    sensor = BurstSensor(10_000)
    factor_task = asyncio.create_task(service.run())
    main_task = asyncio.create_task(drain_main())
    try:
        await stream.start([sensor])
        await asyncio.wait_for(entered.wait(), 1)
        await _wait_until(lambda: service._signal_queue.full() and subscription._queue.full())
        await asyncio.sleep(0.02)
        retained = (
            stream.queue.qsize() + subscription._queue.qsize()
            + service._signal_queue.qsize() + len(computing)
        )
        assert service._signal_queue.maxsize == subscription.capacity == 2
        assert retained <= 8
        assert sensor.emitted <= retained + 2  # one forward put and one sensor tee put
        assert not sensor.closed.is_set()
        # Cancellation must not wait on a terminal sentinel put into a full queue.
        factor_task.cancel()
        await asyncio.wait_for(asyncio.gather(factor_task, return_exceptions=True), 1)
        assert service._signal_queue.empty()
        await asyncio.wait_for(service._signal_queue.join(), 1)
    finally:
        factor_task.cancel()
        main_task.cancel()
        await asyncio.gather(factor_task, main_task, return_exceptions=True)
        await asyncio.wait_for(stream.stop(), 1)


async def test_factor_forwarder_drains_last_batch_after_source_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = [_signal(str(tick), tick) for tick in range(201)]
    service = FactorService(
        pool=cast(Any, object()), store=cast(Any, FakeStore()),
        cadence_s=0.001, factors=(), signal_stream=SequenceSignalStream(signals),
    )
    seen: list[MarketSignal] = []

    async def compute(batch: list[MarketSignal]) -> int:
        seen.extend(batch)
        await asyncio.sleep(0.002)  # source ends while the preceding batch computes
        return 0

    monkeypatch.setattr(service, "compute_once", compute)
    await asyncio.wait_for(service.run(), 1)
    assert seen == signals
    assert service._signal_queue.empty()


async def test_factor_retry_batch_backpressures_new_markets_until_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = SensorStream(capacity=2)
    subscription = stream.subscribe()
    service = FactorService(
        pool=cast(Any, object()), store=cast(Any, FakeStore()),
        cadence_s=0.001, factors=(), signal_stream=subscription,
    )
    signals = [_signal(str(tick), tick) for tick in range(100)]
    produced = 0
    attempts = 0
    largest_batch = 0
    recover = False
    seen: list[MarketSignal] = []

    async def compute(batch: list[MarketSignal]) -> int:
        nonlocal attempts, largest_batch
        attempts += 1
        largest_batch = max(largest_batch, len(batch))
        service._last_failed_market_ids = {s.market_id for s in batch} if not recover else set()
        if recover:
            seen.extend(batch)
        return 0

    class Sensor:
        async def __aiter__(self) -> AsyncIterator[MarketSignal]:
            nonlocal produced
            for signal in signals:
                produced += 1
                yield signal

    async def drain_main() -> None:
        for _ in signals:
            await stream.queue.get()
            stream.queue.task_done()

    monkeypatch.setattr(service, "compute_once", compute)
    factor_task = asyncio.create_task(service.run())
    main_task = asyncio.create_task(drain_main())
    try:
        await stream.start([Sensor()])
        await _wait_until(lambda: attempts >= 10)
        assert largest_batch <= 2
        assert produced <= 10  # three queues, one retry batch, two blocked puts
        recover = True
        await asyncio.wait_for(asyncio.gather(*stream.tasks), 2)
        await asyncio.wait_for(factor_task, 2)
        await asyncio.wait_for(main_task, 2)
        assert seen == signals
    finally:
        factor_task.cancel()
        main_task.cancel()
        await asyncio.gather(factor_task, main_task, return_exceptions=True)
        await stream.stop()


@pytest.mark.parametrize("transient", [False, True])
async def test_factor_shell_retry_is_bounded_for_bad_data_and_lossless_for_outages(
    transient: bool,
) -> None:
    attempted = asyncio.Event()
    bad = replace(_signal("a"), title="malformed")
    corrected = replace(_signal("a", 1), title="corrected")

    class Store(FakeStore):
        attempts = 0

        async def write_market(self, market: Market) -> None:
            self.attempts += 1
            attempted.set()
            if transient and self.attempts <= 6:
                raise asyncpg.CannotConnectNowError("temporary outage")
            if not transient and market.question == "malformed":
                raise ValueError("invalid shell observation")
            await super().write_market(market)

    store = Store()
    subscription = SignalSubscription(capacity=1)
    service = FactorService(
        pool=cast(Any, object()), store=cast(Any, store),
        cadence_s=0.001, factors=(), signal_stream=subscription,
    )
    await subscription.push(bad)
    task = asyncio.create_task(service.run())
    try:
        await asyncio.wait_for(attempted.wait(), 1)
        await subscription.push(corrected)
        subscription.close()
        await asyncio.wait_for(task, 1)
        if transient:
            assert store.attempts == 7  # outage exceeds the deterministic retry budget
            assert store.markets[bad.market_id].question == "malformed"
        else:
            assert store.attempts == 5  # initial try, three retries, corrected shell
            assert store.markets[bad.market_id].question == "corrected"
        assert service._shell_failures == {}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
