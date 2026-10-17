from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from typing import cast

from pms.core.interfaces import ISensor
from pms.core.models import MarketSignal


_CLOSE_SENTINEL = object()


class SignalSubscription:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[MarketSignal | object] = asyncio.Queue()
        self._closed = False
        self._close_requested = False

    def close(self) -> None:
        if self._close_requested:
            return
        self._close_requested = True
        self._queue.put_nowait(_CLOSE_SENTINEL)

    async def push(self, signal: MarketSignal) -> None:
        if self._close_requested:
            return
        await self._queue.put(signal)

    def __aiter__(self) -> "SignalSubscription":
        return self

    async def __anext__(self) -> MarketSignal:
        if self._closed:
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is _CLOSE_SENTINEL:
            self._closed = True
            raise StopAsyncIteration
        return cast(MarketSignal, item)


class SensorStream:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[MarketSignal] = asyncio.Queue()
        self._tasks: tuple[asyncio.Task[None], ...] = ()
        self._subscriptions: list[SignalSubscription] = []
        self._active_consumers = 0
        self._started = False

    @property
    def tasks(self) -> tuple[asyncio.Task[None], ...]:
        return self._tasks

    def subscribe(self) -> SignalSubscription:
        subscription = SignalSubscription()
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
        sensor_list = tuple(sensors)
        self._started = True
        self._active_consumers = len(sensor_list)
        tasks: list[asyncio.Task[None]] = []
        for index, sensor in enumerate(sensor_list):
            task = (
                asyncio.create_task(self._consume(sensor))
                if task_factory is None
                else task_factory(index, sensor)
            )
            task.add_done_callback(self._consumer_done)
            tasks.append(task)
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
            self._close_subscriptions()
            self._subscriptions = []

    async def _consume(self, sensor: ISensor) -> None:
        async for signal in sensor:
            await self.queue.put(signal)
            for subscription in tuple(self._subscriptions):
                await subscription.push(signal)

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
