from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable

from pms.core.interfaces import ISensor
from pms.core.models import MarketSignal


DEFAULT_SIGNAL_BUFFER_CAPACITY = 100


class SignalSubscription:
    def __init__(self, *, capacity: int = DEFAULT_SIGNAL_BUFFER_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("signal buffer capacity must be positive")
        self._queue: asyncio.Queue[MarketSignal] = asyncio.Queue(maxsize=capacity)
        self._closed = False
        self._close_requested = False

    @property
    def capacity(self) -> int:
        return self._queue.maxsize

    def close(self) -> None:
        if self._close_requested:
            return
        self._close_requested = True
        # Shutdown drains queued signals and wakes blocked puts without a sentinel slot.
        self._queue.shutdown()

    async def push(self, signal: MarketSignal) -> None:
        if self._close_requested:
            return
        try:
            await self._queue.put(signal)
        except asyncio.QueueShutDown:
            return

    def __aiter__(self) -> "SignalSubscription":
        return self

    async def __anext__(self) -> MarketSignal:
        if self._closed:
            raise StopAsyncIteration
        try:
            item = await self._queue.get()
        except asyncio.QueueShutDown:
            self._closed = True
            raise StopAsyncIteration from None
        self._queue.task_done()
        return item


class SensorStream:
    def __init__(self, *, capacity: int = DEFAULT_SIGNAL_BUFFER_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("signal buffer capacity must be positive")
        self.queue: asyncio.Queue[MarketSignal] = asyncio.Queue(maxsize=capacity)
        self._tasks: tuple[asyncio.Task[None], ...] = ()
        self._subscriptions: list[SignalSubscription] = []
        self._active_consumers = 0
        self._started = False
        self._stopped = False

    def configure_capacity(self, capacity: int) -> None:
        if self._started or self._subscriptions or not self.queue.empty():
            raise RuntimeError("cannot resize an active sensor stream")
        if capacity < 1:
            raise ValueError("signal buffer capacity must be positive")
        self.queue = asyncio.Queue(maxsize=capacity)

    @property
    def tasks(self) -> tuple[asyncio.Task[None], ...]:
        return self._tasks

    def subscribe(self) -> SignalSubscription:
        self._subscriptions = [s for s in self._subscriptions if not s._close_requested]
        subscription = SignalSubscription(capacity=self.queue.maxsize)
        if self._is_exhausted():
            subscription.close()
        self._subscriptions.append(subscription)
        return subscription

    async def start(
        self,
        sensors: Iterable[ISensor],
        *,
        task_factory: Callable[[int, ISensor], asyncio.Task[None]] | None = None,
    ) -> None:
        if self._tasks:
            msg = "SensorStream is already started"
            raise RuntimeError(msg)
        if self._stopped:
            self.queue = asyncio.Queue(maxsize=self.queue.maxsize)
            self._stopped = False
        sensor_list = tuple(sensors)
        self._started = True
        self._active_consumers = len(sensor_list)
        tasks: list[asyncio.Task[None]] = []
        try:
            for index, sensor in enumerate(sensor_list):
                task = (
                    asyncio.create_task(self._consume(sensor))
                    if task_factory is None
                    else task_factory(index, sensor)
                )
                task.add_done_callback(self._consumer_done)
                tasks.append(task)
        except BaseException:
            self._tasks = tuple(tasks)
            await self.stop()
            raise
        self._tasks = tuple(tasks)
        if self._active_consumers == 0:
            self._close_subscriptions()

    async def stop(self, *, timeout: float = 5.0) -> None:
        for task in self._tasks:
            task.cancel()
        try:
            if self._tasks:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=timeout,
                )
        finally:
            self._tasks = ()
            self._active_consumers = 0
            self._started = False
            self._stopped = True
            self.queue.shutdown(immediate=True)
            self._close_subscriptions()
            self._subscriptions = []

    async def _consume(self, sensor: ISensor) -> None:
        iterator = sensor.__aiter__()
        try:
            async for signal in iterator:
                # Backpressure is lossless, including historical replay and the factor tee.
                await self.queue.put(signal)
                for subscription in tuple(self._subscriptions):
                    await subscription.push(signal)
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    async def consume(self, sensor: ISensor) -> None:
        """Consume one sensor; callers may supervise this coroutine."""
        await self._consume(sensor)

    def _consumer_done(self, _task: asyncio.Task[None]) -> None:
        if self._active_consumers > 0:
            self._active_consumers -= 1
        if self._active_consumers == 0:
            self._close_subscriptions()

    def _close_subscriptions(self) -> None:
        for subscription in self._subscriptions:
            subscription.close()

    def _is_exhausted(self) -> bool:
        return (
            self._started
            and self._active_consumers == 0
            and all(task.done() for task in self._tasks)
        )
