from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from pms.config import PMSSettings, SensorSettings
from pms.core.enums import RunMode
from pms.core.exceptions import SensorDataQualityError
from pms.sensor.adapters.market_data import MarketDataSensor, _message_timestamp
from pms.storage.market_data_store import PostgresMarketDataStore


REPLAY_FIXTURE = Path("tests/fixtures/polymarket_ws_replay.jsonl")
_REAL_SLEEP = asyncio.sleep


async def _wait_for(
    predicate: Callable[[], bool],
    *,
    timeout_s: float = 0.5,
    interval_s: float = 0.005,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            msg = "timed out waiting for predicate"
            raise AssertionError(msg)
        await asyncio.sleep(interval_s)


class _AcquireContext:
    def __init__(self, pool: "ProbePool") -> None:
        self._pool = pool

    async def __aenter__(self) -> object:
        delay = self._pool.delays[min(self._pool.attempts, len(self._pool.delays) - 1)]
        self._pool.attempts += 1
        await _REAL_SLEEP(delay)
        return object()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


@dataclass
class ProbePool:
    delays: list[float] = field(default_factory=lambda: [0.0])
    attempts: int = 0

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self)


@dataclass
class StoreMock:
    pool: ProbePool = field(default_factory=ProbePool)
    write_book_snapshot_mock: AsyncMock = field(default_factory=AsyncMock)
    write_price_change_mock: AsyncMock = field(default_factory=AsyncMock)
    write_trade_mock: AsyncMock = field(default_factory=AsyncMock)
    read_market_signal_metadata_mock: AsyncMock = field(
        default_factory=lambda: AsyncMock(return_value={})
    )

    async def write_book_snapshot(self, snapshot: Any, levels: list[Any]) -> int:
        result = await self.write_book_snapshot_mock(snapshot, levels)
        return cast(int, result if result is not None else 1)

    async def write_price_change(self, price_change: Any) -> None:
        await self.write_price_change_mock(price_change)

    async def write_trade(self, trade: Any) -> None:
        await self.write_trade_mock(trade)

    async def read_market_signal_metadata(self, market_id: str) -> dict[str, str]:
        return cast(
            dict[str, str],
            await self.read_market_signal_metadata_mock(market_id),
        )


class FakeWebSocket:
    def __init__(self, messages: Sequence[str | bytes]) -> None:
        self._messages = list(messages)
        self.sent_messages: list[dict[str, Any]] = []
        self.closed = False

    async def send(self, payload: str) -> None:
        self.sent_messages.append(cast(dict[str, Any], json.loads(payload)))

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        return self

    async def __anext__(self) -> str | bytes:
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


class ClosingSendWebSocket(FakeWebSocket):
    async def send(self, payload: str) -> None:
        del payload
        raise ConnectionClosedError(None, Close(1000, ""))


class FakeConnect:
    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> Any:
        return self.websocket

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.websocket.close()


class HeartbeatWebSocket:
    def __init__(
        self,
        messages: Sequence[str | bytes],
        *,
        ping_outcomes: Sequence[float | None | BaseException] | None = None,
        message_delays_s: Sequence[float] | None = None,
    ) -> None:
        self._messages = list(messages)
        self._ping_outcomes = list(ping_outcomes or [0.0])
        self._message_delays_s = list(message_delays_s or [])
        self._message_index = 0
        self.sent_messages: list[dict[str, Any]] = []
        self.closed = False
        self.ping_count = 0
        self._closed_event = asyncio.Event()

    async def send(self, payload: str) -> None:
        self.sent_messages.append(cast(dict[str, Any], json.loads(payload)))

    async def ping(self) -> asyncio.Future[float]:
        outcome = self._ping_outcomes[
            min(self.ping_count, len(self._ping_outcomes) - 1)
        ]
        self.ping_count += 1
        if isinstance(outcome, BaseException):
            raise outcome
        future: asyncio.Future[float] = asyncio.get_running_loop().create_future()
        if outcome is not None:
            future.set_result(outcome)
        return future

    async def close(self) -> None:
        self.closed = True
        self._closed_event.set()

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        return self

    async def __anext__(self) -> str | bytes:
        if self._messages:
            if self._message_delays_s:
                delay = self._message_delays_s[
                    min(self._message_index, len(self._message_delays_s) - 1)
                ]
                if delay > 0:
                    await _REAL_SLEEP(delay)
            self._message_index += 1
            return self._messages.pop(0)
        await self._closed_event.wait()
        raise StopAsyncIteration


class ConnectSequence:
    def __init__(self, responses: Sequence[Any]) -> None:
        self._responses = list(responses)

    def __call__(self, url: str) -> FakeConnect:
        del url
        if not self._responses:
            msg = "connect called more times than planned"
            raise AssertionError(msg)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeConnect(cast(Any, response))


@dataclass
class WatchdogProbe:
    notify_calls: int = 0

    def notify_message(self) -> None:
        self.notify_calls += 1


def _store_mock() -> PostgresMarketDataStore:
    return cast(PostgresMarketDataStore, StoreMock())


def _replay_messages() -> list[str]:
    return [line for line in REPLAY_FIXTURE.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize(
    ("raw_timestamp", "expected"),
    [
        (1_757_908_892_351, datetime.fromtimestamp(1_757_908_892_351 / 1000.0, tz=UTC)),
        (1_757_908_892, datetime.fromtimestamp(1_757_908_892, tz=UTC)),
    ],
)
def test_message_timestamp_supports_millisecond_and_second_epoch_values(
    raw_timestamp: int,
    expected: datetime,
) -> None:
    assert _message_timestamp(raw_timestamp) == expected


@pytest.mark.asyncio
async def test_market_data_sensor_default_keeps_price_change_in_memory_without_raw_db_write() -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    sensor = MarketDataSensor(store=store)

    signals = await sensor._handle_message(
        {
            "event_type": "price_change",
            "market": "m-scan-only",
            "timestamp": "1757908892352",
            "price_changes": [
                {
                    "asset_id": "asset-scan-only",
                    "price": "0.49",
                    "size": "12",
                    "side": "BUY",
                    "hash": "delta-hash",
                    "best_bid": "0.49",
                    "best_ask": "0.52",
                }
            ],
        }
    )
    await sensor.aclose()

    assert len(signals) == 1
    assert signals[0].orderbook["bids"] == [{"price": 0.49, "size": 12.0}]
    assert store_mock.write_price_change_mock.await_count == 0


@pytest.mark.asyncio
async def test_market_data_sensor_enriches_signals_with_market_risk_metadata() -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.read_market_signal_metadata_mock.return_value = {
        "risk_group_id": "event:2028-us-presidential-election",
        "event_id": "2028-us-presidential-election",
        "category": "Politics",
        "yes_token_id": "asset-election-risk-yes",
        "no_token_id": "asset-election-risk",
    }
    sensor = MarketDataSensor(store=store)

    signal = await sensor._handle_book(
        {
            "event_type": "book",
            "market": "m-election-risk",
            "asset_id": "asset-election-risk",
            "timestamp": "1757908892352",
            "bids": [{"price": "0.48", "size": "12"}],
            "asks": [{"price": "0.51", "size": "10"}],
        }
    )
    await sensor.aclose()

    assert signal.external_signal["risk_group_id"] == (
        "event:2028-us-presidential-election"
    )
    assert signal.external_signal["event_id"] == "2028-us-presidential-election"
    assert signal.external_signal["category"] == "Politics"
    assert signal.external_signal["yes_token_id"] == "asset-election-risk-yes"
    assert signal.external_signal["no_token_id"] == "asset-election-risk"
    assert signal.external_signal["signal_token_outcome"] == "NO"
    assert isinstance(signal.external_signal["book_received_at"], str)
    store_mock.read_market_signal_metadata_mock.assert_awaited_once_with(
        "m-election-risk"
    )


@pytest.mark.asyncio
async def test_market_data_sensor_retries_empty_market_risk_metadata_lookup() -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.read_market_signal_metadata_mock.side_effect = [
        {},
        {"risk_group_id": "category:Politics", "category": "Politics"},
    ]
    sensor = MarketDataSensor(store=store)
    message = {
        "event_type": "book",
        "market": "m-late-metadata",
        "asset_id": "asset-late-metadata",
        "timestamp": "1757908892352",
        "bids": [{"price": "0.48", "size": "12"}],
        "asks": [{"price": "0.51", "size": "10"}],
    }

    first = await sensor._handle_book(message)
    second = await sensor._handle_book(message)
    await sensor.aclose()

    assert "risk_group_id" not in first.external_signal
    assert second.external_signal["risk_group_id"] == "category:Politics"
    assert store_mock.read_market_signal_metadata_mock.await_count == 2


@pytest.mark.asyncio
async def test_market_data_sensor_retries_incomplete_outcome_token_metadata_lookup() -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.read_market_signal_metadata_mock.side_effect = [
        {"category": "Politics"},
        {
            "category": "Politics",
            "yes_token_id": "asset-late-metadata-yes",
            "no_token_id": "asset-late-metadata-no",
        },
    ]
    sensor = MarketDataSensor(store=store)
    message = {
        "event_type": "book",
        "market": "m-late-token-metadata",
        "asset_id": "asset-late-metadata-no",
        "timestamp": "1757908892352",
        "bids": [{"price": "0.48", "size": "12"}],
        "asks": [{"price": "0.51", "size": "10"}],
    }

    first = await sensor._handle_book(message)
    second = await sensor._handle_book(message)
    await sensor.aclose()

    assert first.external_signal["category"] == "Politics"
    assert "signal_token_outcome" not in first.external_signal
    assert second.external_signal["signal_token_outcome"] == "NO"
    assert store_mock.read_market_signal_metadata_mock.await_count == 2


@pytest.mark.asyncio
async def test_market_data_sensor_replays_fixture_losslessly_and_persists_piecewise_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.side_effect = [1, 2]
    fake_websocket = HeartbeatWebSocket(_replay_messages(), ping_outcomes=[0.0])
    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        lambda url: FakeConnect(fake_websocket),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-yes"],
        persist_price_changes=True,
    )
    sensor._heartbeat_interval_s = 1.0

    signals: list[Any] = []
    iterator = cast(Any, sensor.__aiter__())
    while len(signals) < 7:
        signals.append(await asyncio.wait_for(anext(iterator), timeout=0.5))
    await asyncio.wait_for(iterator.aclose(), timeout=0.5)
    await sensor.aclose()

    assert fake_websocket.sent_messages == [
        {
            "assets_ids": ["asset-yes"],
            "type": "market",
            "initial_dump": True,
            "level": 2,
        }
    ]
    assert store_mock.write_book_snapshot_mock.await_count == 2
    assert store_mock.write_price_change_mock.await_count == 5
    delta_sizes = [
        call.args[0].size for call in store_mock.write_price_change_mock.await_args_list
    ]
    assert delta_sizes[:3] == [5.0, 0.0, 3.0]
    final_orderbook = signals[-1].orderbook
    assert [level["price"] for level in final_orderbook["bids"]] == [0.47, 0.45, 0.44]
    assert [level["size"] for level in final_orderbook["bids"]] == [3.0, 7.0, 2.0]
    assert [level["price"] for level in final_orderbook["asks"]] == [0.54, 0.55, 0.56]
    assert [level["size"] for level in final_orderbook["asks"]] == [6.0, 4.0, 1.0]


@pytest.mark.asyncio
async def test_market_data_sensor_bounds_websocket_close_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store_mock()
    captured_kwargs: dict[str, Any] = {}
    fake_websocket = HeartbeatWebSocket(
        [
            json.dumps(
                {
                    "event_type": "book",
                    "market": "m-close-timeout",
                    "asset_id": "asset-close-timeout",
                    "timestamp": "1757908892351",
                    "hash": "book-close-timeout",
                    "bids": [{"price": "0.40", "size": "5"}],
                    "asks": [{"price": "0.60", "size": "5"}],
                    "last_trade_price": "0.50",
                }
            )
        ],
        ping_outcomes=[0.0],
    )

    def fake_connect(url: str, **kwargs: Any) -> FakeConnect:
        del url
        captured_kwargs.update(kwargs)
        return FakeConnect(fake_websocket)

    monkeypatch.setattr("pms.sensor.adapters.market_data.connect", fake_connect)
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-close-timeout"],
    )
    iterator = cast(Any, sensor.__aiter__())

    signal = await asyncio.wait_for(anext(iterator), timeout=0.5)
    await asyncio.wait_for(iterator.aclose(), timeout=0.5)
    await sensor.aclose()

    assert signal.market_id == "m-close-timeout"
    assert captured_kwargs["close_timeout"] == 1.0
    assert captured_kwargs["max_size"] == 8 * 1024 * 1024
    assert captured_kwargs["proxy"] is None


@pytest.mark.asyncio
async def test_market_data_sensor_persists_last_trade_price_and_emits_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.return_value = 1
    messages = [
        json.dumps(
            {
                "event_type": "book",
                "market": "m-trade",
                "asset_id": "asset-trade",
                "timestamp": "1757908892351",
                "hash": "book-hash",
                "bids": [{"price": "0.48", "size": "30"}],
                "asks": [{"price": "0.52", "size": "25"}],
                "last_trade_price": "0.50",
            }
        ),
        json.dumps(
            {
                "event_type": "last_trade_price",
                "market": "m-trade",
                "asset_id": "asset-trade",
                "price": "0.456",
                "side": "BUY",
                "size": "219.217767",
                "fee_rate_bps": "0",
                "timestamp": "1750428146322",
            }
        ),
    ]
    fake_websocket = HeartbeatWebSocket(messages, ping_outcomes=[0.0])
    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        lambda url: FakeConnect(fake_websocket),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-trade"],
    )
    sensor._heartbeat_interval_s = 1.0

    iterator = cast(Any, sensor.__aiter__())
    await asyncio.wait_for(anext(iterator), timeout=0.5)
    trade_signal = await asyncio.wait_for(anext(iterator), timeout=0.5)
    await asyncio.wait_for(iterator.aclose(), timeout=0.5)
    await sensor.aclose()

    assert store_mock.write_trade_mock.await_count == 1
    assert store_mock.write_trade_mock.await_args is not None
    trade = store_mock.write_trade_mock.await_args.args[0]
    assert trade.market_id == "m-trade"
    assert trade.token_id == "asset-trade"
    assert trade.price == 0.456
    assert trade_signal.market_id == "m-trade"
    assert trade_signal.token_id == "asset-trade"
    assert trade_signal.yes_price == 0.456
    assert trade_signal.external_signal["last_trade_price"] == 0.456
    assert trade_signal.external_signal["fee_rate_bps"] == 0.0


@pytest.mark.asyncio
async def test_market_data_sensor_carries_last_trade_fee_rate_into_book_signal() -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.return_value = 1
    sensor = MarketDataSensor(store=store)

    trade_signals = await sensor._handle_message(
        {
            "event_type": "last_trade_price",
            "market": "m-fee",
            "asset_id": "asset-fee",
            "price": "0.456",
            "side": "BUY",
            "size": "219.217767",
            "fee_rate_bps": "300",
            "timestamp": "1750428146322",
        }
    )
    book_signals = await sensor._handle_message(
        {
            "event_type": "book",
            "market": "m-fee",
            "asset_id": "asset-fee",
            "timestamp": "1757908892351",
            "hash": "book-hash-fee",
            "bids": [{"price": "0.48", "size": "30"}],
            "asks": [{"price": "0.52", "size": "25"}],
            "last_trade_price": "0.50",
        }
    )

    assert trade_signals[0].external_signal["fee_rate_bps"] == 300.0
    assert book_signals[0].external_signal["fee_rate_bps"] == 300.0


@pytest.mark.asyncio
async def test_market_data_sensor_preserves_cached_fee_rate_on_trade_without_fee() -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.return_value = 1
    sensor = MarketDataSensor(store=store)

    await sensor._handle_message(
        {
            "event_type": "book",
            "market": "m-trade-fee-cache",
            "asset_id": "asset-trade-fee-cache",
            "timestamp": "1757908892351",
            "hash": "book-hash-fee-cache",
            "bids": [{"price": "0.48", "size": "30"}],
            "asks": [{"price": "0.52", "size": "25"}],
            "fee_rate_bps": "300",
        }
    )
    trade_signals = await sensor._handle_message(
        {
            "event_type": "last_trade_price",
            "market": "m-trade-fee-cache",
            "asset_id": "asset-trade-fee-cache",
            "price": "0.456",
            "side": "BUY",
            "size": "219.217767",
            "timestamp": "1750428146322",
        }
    )
    await sensor.aclose()

    assert trade_signals[0].external_signal["fee_rate_bps"] == 300.0


@pytest.mark.asyncio
async def test_market_data_sensor_carries_last_trade_price_into_book_signal() -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.return_value = 1
    sensor = MarketDataSensor(store=store)

    trade_signals = await sensor._handle_message(
        {
            "event_type": "last_trade_price",
            "market": "m-last-trade",
            "asset_id": "asset-last-trade",
            "price": "0.456",
            "side": "BUY",
            "size": "219.217767",
            "timestamp": "1750428146322",
        }
    )
    book_signals = await sensor._handle_message(
        {
            "event_type": "book",
            "market": "m-last-trade",
            "asset_id": "asset-last-trade",
            "timestamp": "1757908892351",
            "hash": "book-hash-last-trade",
            "bids": [{"price": "0.48", "size": "30"}],
            "asks": [{"price": "0.52", "size": "25"}],
        }
    )

    assert trade_signals[0].external_signal["last_trade_price"] == 0.456
    assert book_signals[0].external_signal["last_trade_price"] == 0.456


@pytest.mark.asyncio
async def test_market_data_sensor_ignores_book_payload_last_trade_without_token_event() -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.return_value = 1
    sensor = MarketDataSensor(store=store)

    book_signals = await sensor._handle_message(
        {
            "event_type": "book",
            "market": "m-book-last-trade",
            "asset_id": "yes-token",
            "timestamp": "1757908892351",
            "hash": "book-hash-last-trade",
            "bids": [{"price": "0.09", "size": "150"}],
            "asks": [{"price": "0.10", "size": "150"}],
            "last_trade_price": "0.904",
        }
    )

    assert book_signals[0].yes_price == pytest.approx(0.095)
    assert "last_trade_price" not in book_signals[0].external_signal


@pytest.mark.asyncio
async def test_market_data_sensor_update_subscription_does_not_probe_pool(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.pool = ProbePool(delays=[0.12, 0.0])
    sensor = MarketDataSensor(store=store)
    fake_websocket = FakeWebSocket([])
    sensor._websocket = cast(Any, fake_websocket)

    with caplog.at_level(logging.WARN):
        await sensor.update_subscription(["asset-a", "asset-b"])

    assert store_mock.pool.attempts == 0
    assert "pool saturated" not in caplog.text
    assert fake_websocket.sent_messages == [
        {
            "assets_ids": ["asset-a", "asset-b"],
            "operation": "subscribe",
        }
    ]


@pytest.mark.asyncio
async def test_market_data_sensor_dynamic_update_sends_subscribe_and_unsubscribe_deltas(
) -> None:
    sensor = MarketDataSensor(store=_store_mock(), asset_ids=["asset-a", "asset-b"])
    fake_websocket = FakeWebSocket([])
    sensor._websocket = cast(Any, fake_websocket)
    sensor._subscribed_asset_ids = frozenset({"asset-a", "asset-b"})

    await sensor.update_subscription(["asset-b", "asset-c"])

    assert fake_websocket.sent_messages == [
        {
            "assets_ids": ["asset-a"],
            "operation": "unsubscribe",
        },
        {
            "assets_ids": ["asset-c"],
            "operation": "subscribe",
        },
    ]
    assert sensor._subscribed_asset_ids == frozenset({"asset-b", "asset-c"})


@pytest.mark.asyncio
async def test_market_data_sensor_serializes_initial_and_dynamic_subscription_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingInitialWebSocket(FakeWebSocket):
        def __init__(self) -> None:
            super().__init__([])
            self.initial_send_started = asyncio.Event()
            self.release_initial_send = asyncio.Event()
            self.wait_for_close = asyncio.Event()

        async def send(self, payload: str) -> None:
            decoded = cast(dict[str, Any], json.loads(payload))
            if not self.sent_messages:
                self.initial_send_started.set()
                await self.release_initial_send.wait()
            self.sent_messages.append(decoded)

        async def __anext__(self) -> str | bytes:
            await self.wait_for_close.wait()
            raise StopAsyncIteration

    websocket = BlockingInitialWebSocket()
    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        lambda url: FakeConnect(websocket),
    )
    sensor = MarketDataSensor(
        store=_store_mock(),
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-a"],
    )
    sensor._INITIAL_BACKOFF_S = 0.0
    iterator = cast(Any, sensor.__aiter__())
    consume_task = asyncio.create_task(anext(iterator))

    await asyncio.wait_for(websocket.initial_send_started.wait(), timeout=0.5)
    update_task = asyncio.create_task(sensor.update_subscription(["asset-b"]))
    await asyncio.sleep(0)
    assert not update_task.done()

    websocket.release_initial_send.set()
    await asyncio.wait_for(update_task, timeout=0.5)
    assert websocket.sent_messages == [
        {
            "assets_ids": ["asset-a"],
            "type": "market",
            "initial_dump": True,
            "level": 2,
        },
        {"assets_ids": ["asset-a"], "operation": "unsubscribe"},
        {"assets_ids": ["asset-b"], "operation": "subscribe"},
    ]
    assert sensor._subscribed_asset_ids == frozenset({"asset-b"})

    consume_task.cancel()
    await asyncio.gather(consume_task, return_exceptions=True)
    await sensor.aclose()


@pytest.mark.asyncio
async def test_market_data_sensor_accepts_subscription_update_when_socket_closes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensor = MarketDataSensor(store=_store_mock(), asset_ids=["asset-a"])
    fake_websocket = ClosingSendWebSocket([])
    sensor._websocket = cast(Any, fake_websocket)
    sensor._subscribed_asset_ids = frozenset({"asset-a"})

    with caplog.at_level(logging.WARNING):
        await sensor.update_subscription(["asset-b"])

    assert sensor._asset_ids == ["asset-b"]
    assert fake_websocket.closed is True
    assert sensor._websocket is None
    assert sensor._subscribed_asset_ids == frozenset()
    assert "market data sensor websocket closed; reconnecting" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


@pytest.mark.asyncio
async def test_market_data_sensor_survives_persistence_type_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.side_effect = [TypeError("boom"), 1]
    payload = json.dumps(
        {
            "event_type": "book",
            "market": "m-broken",
            "asset_id": "asset-broken",
            "timestamp": "1757908892351",
            "hash": "book-broken",
            "bids": [{"price": "0.48", "size": "30"}],
            "asks": [{"price": "0.52", "size": "25"}],
            "last_trade_price": "0.50",
        }
    )
    fake_websocket = FakeWebSocket([payload])
    second_websocket = FakeWebSocket([payload.replace("book-broken", "book-recovered")])
    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        ConnectSequence([fake_websocket, second_websocket]),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-broken"],
    )
    sensor._heartbeat_interval_s = 1.0
    sensor._INITIAL_BACKOFF_S = 0.0
    iterator = cast(Any, sensor.__aiter__())

    signal = await asyncio.wait_for(anext(iterator), timeout=0.5)

    await sensor.aclose()
    assert signal.market_id == "m-broken"
    assert store_mock.write_book_snapshot_mock.await_count == 2


@pytest.mark.asyncio
async def test_market_data_sensor_fails_after_repeated_persistence_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.side_effect = TypeError("poison frame")
    payload = json.dumps(
        {
            "event_type": "book",
            "market": "m-poison",
            "asset_id": "asset-poison",
            "timestamp": "1757908892351",
            "hash": "book-poison",
            "bids": [{"price": "0.48", "size": "30"}],
            "asks": [{"price": "0.52", "size": "25"}],
        }
    )
    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        ConnectSequence([FakeWebSocket([payload]) for _ in range(5)]),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-poison"],
    )
    sensor._INITIAL_BACKOFF_S = 0.0
    iterator = cast(Any, sensor.__aiter__())

    with pytest.raises(SensorDataQualityError, match="consecutive"):
        await asyncio.wait_for(anext(iterator), timeout=0.5)

    assert sensor.consecutive_data_error_reconnects == 5
    await sensor.aclose()


@pytest.mark.asyncio
async def test_market_data_sensor_poison_after_good_frame_keeps_backoff_and_escalates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.side_effect = [
        value
        for _ in range(5)
        for value in (1, TypeError("poison frame"))
    ]

    def _book_payload(hash_value: str) -> str:
        return json.dumps(
            {
                "event_type": "book",
                "market": "m-poison-after-good",
                "asset_id": "asset-poison-after-good",
                "timestamp": "1757908892351",
                "hash": hash_value,
                "bids": [{"price": "0.48", "size": "30"}],
                "asks": [{"price": "0.52", "size": "25"}],
            }
        )

    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        ConnectSequence(
            [
                FakeWebSocket([_book_payload(f"good-{index}"), _book_payload(f"poison-{index}")])
                for index in range(5)
            ]
        ),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-poison-after-good"],
    )
    sensor._INITIAL_BACKOFF_S = 0.01
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await asyncio.sleep(0)

    monkeypatch.setattr(sensor, "_sleep", fake_sleep)
    iterator = cast(Any, sensor.__aiter__())

    for _ in range(5):
        await asyncio.wait_for(anext(iterator), timeout=0.5)
    with pytest.raises(SensorDataQualityError, match="consecutive"):
        await asyncio.wait_for(anext(iterator), timeout=0.5)

    assert sleeps == [0.01, 0.02, 0.04, 0.08]
    assert sensor.consecutive_data_error_reconnects == 5
    await sensor.aclose()


@pytest.mark.asyncio
async def test_market_data_sensor_skips_malformed_json_and_continues_loop(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.return_value = 1
    fake_websocket = FakeWebSocket(
        [
            "not-valid-json",
            json.dumps(
                {
                    "event_type": "book",
                    "market": "m-malformed",
                    "asset_id": "asset-malformed",
                    "timestamp": "1757908892351",
                    "hash": "book-malformed",
                    "bids": [{"price": "0.48", "size": "30"}],
                    "asks": [{"price": "0.52", "size": "25"}],
                    "last_trade_price": "0.50",
                }
            ),
        ]
    )
    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        lambda url: FakeConnect(fake_websocket),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-malformed"],
    )
    sensor._heartbeat_interval_s = 1.0
    iterator = cast(Any, sensor.__aiter__())

    with caplog.at_level(logging.WARN):
        signal = await asyncio.wait_for(anext(iterator), timeout=0.5)

    await asyncio.wait_for(iterator.aclose(), timeout=0.5)
    await sensor.aclose()

    assert sensor.malformed_messages_total == 1
    assert signal.market_id == "m-malformed"
    assert "malformed market-data payload: not-valid-json" in caplog.text


@pytest.mark.asyncio
async def test_market_data_sensor_raises_quality_error_after_one_hundred_malformed_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store_mock()
    sensor = MarketDataSensor(store=store)

    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.time.monotonic",
        lambda: 0.0,
    )

    with pytest.raises(SensorDataQualityError):
        for _ in range(100):
            await sensor._handle_raw_message("not-valid-json")

    assert sensor.malformed_messages_total == 100


@pytest.mark.asyncio
async def test_market_data_sensor_allows_malformed_rate_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.return_value = 1
    sensor = MarketDataSensor(store=store)
    valid_message = json.dumps(
        {
            "event_type": "book",
            "market": "m-threshold",
            "asset_id": "asset-threshold",
            "timestamp": "1757908892351",
            "hash": "book-threshold",
            "bids": [{"price": "0.48", "size": "30"}],
            "asks": [{"price": "0.52", "size": "25"}],
            "last_trade_price": "0.50",
        }
    )

    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.time.monotonic",
        lambda: 0.0,
    )

    for index in range(100):
        if index % 5 < 2:
            await sensor._handle_raw_message("not-valid-json")
            continue
        await sensor._handle_raw_message(valid_message)

    assert sensor.malformed_messages_total == 40
    assert sensor.malformed_rate_threshold == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_market_data_sensor_rate_limits_malformed_payload_warnings(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _store_mock()
    sensor = MarketDataSensor(store=store)
    sensor.malformed_rate_threshold = 1.1

    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.time.monotonic",
        lambda: 0.0,
    )

    with caplog.at_level(logging.WARN):
        for _ in range(100):
            await sensor._handle_raw_message("not-valid-json")

    warnings = [
        record for record in caplog.records if "malformed market-data payload" in record.message
    ]
    assert len(warnings) <= 2
    assert sensor.malformed_messages_total == 100


@pytest.mark.asyncio
async def test_market_data_sensor_logs_unrecognized_payload_at_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _store_mock()
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-yes"],
    )

    with caplog.at_level(logging.WARN):
        signals = await sensor._handle_raw_message(
            json.dumps({"event_type": "mystery", "foo": "bar"})
        )
    await sensor.aclose()

    assert signals == []
    assert "mystery" in caplog.text
    assert '"foo": "bar"' in caplog.text


@pytest.mark.asyncio
async def test_market_data_sensor_logs_scalar_payload_at_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _store_mock()
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-yes"],
    )

    with caplog.at_level(logging.WARN):
        signals = await sensor._handle_raw_message("123")
    await sensor.aclose()

    assert signals == []
    assert "unrecognised market-data payload: 123" in caplog.text


@pytest.mark.asyncio
async def test_market_data_sensor_reconnects_and_marks_next_snapshot_as_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.side_effect = [1, 2]
    first_websocket = FakeWebSocket(
        [
            json.dumps(
                {
                    "event_type": "book",
                    "market": "m-reconnect",
                    "asset_id": "asset-reconnect",
                    "timestamp": "1757908892351",
                    "hash": "book-1",
                    "bids": [{"price": "0.48", "size": "10"}],
                    "asks": [{"price": "0.52", "size": "8"}],
                    "last_trade_price": "0.50",
                }
            ),
            json.dumps(
                {
                    "event_type": "price_change",
                    "market": "m-reconnect",
                    "timestamp": "1757908892352",
                    "price_changes": [
                        {
                            "asset_id": "asset-reconnect",
                            "price": "0.49",
                            "size": "12",
                            "side": "BUY",
                            "hash": "delta-1",
                            "best_bid": "0.49",
                            "best_ask": "0.52",
                        }
                    ],
                }
            ),
        ]
    )
    second_websocket = FakeWebSocket(
        [
            json.dumps(
                {
                    "event_type": "book",
                    "market": "m-reconnect",
                    "asset_id": "asset-reconnect",
                    "timestamp": "1757908892451",
                    "hash": "book-2",
                    "bids": [{"price": "0.31", "size": "9"}],
                    "asks": [{"price": "0.69", "size": "6"}],
                    "last_trade_price": "0.32",
                }
            )
        ]
    )
    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        ConnectSequence([first_websocket, second_websocket]),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-reconnect"],
    )
    sensor._INITIAL_BACKOFF_S = 0.0
    iterator = cast(Any, sensor.__aiter__())

    await anext(iterator)
    await anext(iterator)
    reconnect_signal = await asyncio.wait_for(anext(iterator), timeout=0.2)
    await iterator.aclose()
    await sensor.aclose()

    snapshot_sources = [
        call.args[0].source for call in store_mock.write_book_snapshot_mock.await_args_list
    ]
    assert snapshot_sources == ["subscribe", "reconnect"]
    assert reconnect_signal.orderbook["bids"] == [{"price": 0.31, "size": 9.0}]
    assert reconnect_signal.orderbook["asks"] == [{"price": 0.69, "size": 6.0}]


@pytest.mark.asyncio
async def test_market_data_sensor_tags_every_asset_initial_dump_after_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.side_effect = [1, 2, 3, 4, 5, 6]
    asset_ids = ["asset-a", "asset-b", "asset-c"]

    def _book_event(asset_id: str, ts: str, hash_value: str) -> str:
        return json.dumps(
            {
                "event_type": "book",
                "market": f"m-{asset_id}",
                "asset_id": asset_id,
                "timestamp": ts,
                "hash": hash_value,
                "bids": [{"price": "0.40", "size": "5"}],
                "asks": [{"price": "0.60", "size": "5"}],
                "last_trade_price": "0.50",
            }
        )

    first_websocket = FakeWebSocket(
        [_book_event(asset, "1757908892351", f"first-{asset}") for asset in asset_ids]
    )
    second_websocket = FakeWebSocket(
        [_book_event(asset, "1757908892451", f"second-{asset}") for asset in asset_ids]
    )
    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        ConnectSequence([first_websocket, second_websocket]),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=asset_ids,
    )
    sensor._INITIAL_BACKOFF_S = 0.0
    iterator = cast(Any, sensor.__aiter__())

    # Drain the first connection's three initial dumps (subscribe).
    for _ in range(3):
        await asyncio.wait_for(anext(iterator), timeout=0.5)

    # The first websocket exhausts its messages -> StopAsyncIteration ->
    # _iterate falls into the reconnect path. Drain the three reconnect dumps.
    for _ in range(3):
        await asyncio.wait_for(anext(iterator), timeout=0.5)

    await iterator.aclose()
    await sensor.aclose()

    snapshot_sources = [
        call.args[0].source for call in store_mock.write_book_snapshot_mock.await_args_list
    ]
    assert snapshot_sources == [
        "subscribe",
        "subscribe",
        "subscribe",
        "reconnect",
        "reconnect",
        "reconnect",
    ]


@pytest.mark.asyncio
async def test_market_data_sensor_retries_after_websocket_handshake_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from websockets.exceptions import InvalidHandshake

    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.return_value = 1

    handshake_error = InvalidHandshake("server returned 503")
    second_websocket = FakeWebSocket(
        [
            json.dumps(
                {
                    "event_type": "book",
                    "market": "m-handshake",
                    "asset_id": "asset-handshake",
                    "timestamp": "1757908892351",
                    "hash": "book-after-handshake",
                    "bids": [{"price": "0.40", "size": "5"}],
                    "asks": [{"price": "0.60", "size": "5"}],
                    "last_trade_price": "0.50",
                }
            )
        ]
    )
    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        ConnectSequence([handshake_error, second_websocket]),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-handshake"],
    )
    sensor._INITIAL_BACKOFF_S = 0.0
    iterator = cast(Any, sensor.__aiter__())

    signal = await asyncio.wait_for(anext(iterator), timeout=2.0)
    await iterator.aclose()
    await sensor.aclose()

    assert signal.market_id == "m-handshake"
    assert store_mock.write_book_snapshot_mock.await_count == 1


@pytest.mark.asyncio
async def test_market_data_sensor_reconnects_after_transient_storage_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import asyncpg

    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.side_effect = [
        asyncpg.PostgresError("deadlock detected"),
        1,
    ]

    def _book_message(book_hash: str) -> str:
        return json.dumps(
            {
                "event_type": "book",
                "market": "m-storage",
                "asset_id": "asset-storage",
                "timestamp": "1757908892351",
                "hash": book_hash,
                "bids": [{"price": "0.40", "size": "5"}],
                "asks": [{"price": "0.60", "size": "5"}],
                "last_trade_price": "0.50",
            }
        )

    first_websocket = FakeWebSocket([_book_message("book-before-db-error")])
    second_websocket = FakeWebSocket([_book_message("book-after-db-error")])
    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        ConnectSequence([first_websocket, second_websocket]),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-storage"],
    )
    sensor._INITIAL_BACKOFF_S = 0.0
    iterator = cast(Any, sensor.__aiter__())

    with caplog.at_level(logging.WARNING):
        signal = await asyncio.wait_for(anext(iterator), timeout=2.0)

    await iterator.aclose()
    await sensor.aclose()

    assert signal.market_id == "m-storage"
    assert store_mock.write_book_snapshot_mock.await_count == 2
    assert "market data sensor storage error; reconnecting" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


@pytest.mark.asyncio
async def test_market_data_sensor_logs_expected_websocket_close_as_reconnect(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.return_value = 1
    active_close = ConnectionClosedError(None, Close(1000, ""))
    second_websocket = HeartbeatWebSocket(
        [
            json.dumps(
                {
                    "event_type": "book",
                    "market": "m-active-close",
                    "asset_id": "asset-active-close",
                    "timestamp": "1757908892351",
                    "hash": "book-reconnect",
                    "bids": [{"price": "0.44", "size": "11"}],
                    "asks": [{"price": "0.56", "size": "7"}],
                    "last_trade_price": "0.45",
                }
            )
        ],
        ping_outcomes=[0.0],
    )
    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        ConnectSequence([active_close, second_websocket]),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-active-close"],
    )
    sensor._INITIAL_BACKOFF_S = 0.0
    sensor._heartbeat_interval_s = 1.0
    iterator = cast(Any, sensor.__aiter__())

    with caplog.at_level(logging.WARNING):
        signal = await asyncio.wait_for(anext(iterator), timeout=0.5)

    await asyncio.wait_for(iterator.aclose(), timeout=0.5)
    await sensor.aclose()

    assert signal.market_id == "m-active-close"
    assert store_mock.write_book_snapshot_mock.await_count == 1
    assert "market data sensor websocket closed; reconnecting" in caplog.text
    assert "market data sensor receive loop failed" not in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


@pytest.mark.asyncio
async def test_market_data_sensor_reconnects_when_pong_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.return_value = 1
    first_websocket = HeartbeatWebSocket([], ping_outcomes=[None])
    second_websocket = HeartbeatWebSocket(
        [
            json.dumps(
                {
                    "event_type": "book",
                    "market": "m-pong",
                    "asset_id": "asset-pong",
                    "timestamp": "1757908892351",
                    "hash": "book-reconnect",
                    "bids": [{"price": "0.44", "size": "11"}],
                    "asks": [{"price": "0.56", "size": "7"}],
                    "last_trade_price": "0.45",
                }
            )
        ],
        ping_outcomes=[0.0],
    )
    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        ConnectSequence([first_websocket, second_websocket]),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-pong"],
    )
    sensor._INITIAL_BACKOFF_S = 0.0
    sensor._heartbeat_interval_s = 0.01
    sensor._pong_timeout_s = 0.02
    sensor._watchdog.timeout_s = 1.0
    iterator = cast(Any, sensor.__aiter__())

    reconnect_signal = await asyncio.wait_for(anext(iterator), timeout=0.5)
    await iterator.aclose()
    await sensor.aclose()

    assert first_websocket.ping_count >= 1
    assert first_websocket.closed is True
    assert store_mock.write_book_snapshot_mock.await_args is not None
    snapshot = store_mock.write_book_snapshot_mock.await_args.args[0]
    assert snapshot.source == "reconnect"
    assert reconnect_signal.market_id == "m-pong"


@pytest.mark.asyncio
async def test_market_data_sensor_logs_heartbeat_connection_close_as_reconnect(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.return_value = 1
    first_websocket = HeartbeatWebSocket(
        [],
        ping_outcomes=[ConnectionClosedError(None, Close(1000, ""))],
    )
    second_websocket = HeartbeatWebSocket(
        [
            json.dumps(
                {
                    "event_type": "book",
                    "market": "m-heartbeat-close",
                    "asset_id": "asset-heartbeat-close",
                    "timestamp": "1757908892351",
                    "hash": "book-reconnect",
                    "bids": [{"price": "0.44", "size": "11"}],
                    "asks": [{"price": "0.56", "size": "7"}],
                    "last_trade_price": "0.45",
                }
            )
        ],
        ping_outcomes=[0.0],
    )
    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        ConnectSequence([first_websocket, second_websocket]),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-heartbeat-close"],
    )
    sensor._INITIAL_BACKOFF_S = 0.0
    sensor._heartbeat_interval_s = 0.01
    sensor._pong_timeout_s = 0.02
    sensor._watchdog.timeout_s = 1.0
    iterator = cast(Any, sensor.__aiter__())

    with caplog.at_level(logging.WARNING):
        reconnect_signal = await asyncio.wait_for(anext(iterator), timeout=0.5)

    await iterator.aclose()
    await sensor.aclose()

    assert first_websocket.closed is True
    assert store_mock.write_book_snapshot_mock.await_args is not None
    assert reconnect_signal.market_id == "m-heartbeat-close"
    assert "market data sensor websocket closed; reconnecting" in caplog.text
    assert "market data sensor heartbeat failed" not in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


@pytest.mark.asyncio
async def test_market_data_sensor_book_resets_watchdog() -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.return_value = 1
    sensor = MarketDataSensor(store=store)
    probe = WatchdogProbe()
    sensor._watchdog = cast(Any, probe)

    await sensor._handle_message(
        {
            "event_type": "book",
            "market": "m-book",
            "asset_id": "asset-book",
            "timestamp": "1757908892351",
            "hash": "book-hash",
            "bids": [{"price": "0.48", "size": "30"}],
            "asks": [{"price": "0.52", "size": "25"}],
            "last_trade_price": "0.50",
        }
    )

    assert probe.notify_calls == 1


@pytest.mark.asyncio
async def test_market_data_sensor_price_change_resets_watchdog() -> None:
    store = _store_mock()
    sensor = MarketDataSensor(store=store)
    probe = WatchdogProbe()
    sensor._watchdog = cast(Any, probe)

    await sensor._handle_message(
        {
            "event_type": "price_change",
            "market": "m-delta",
            "timestamp": "1757908892352",
            "price_changes": [
                {
                    "asset_id": "asset-delta",
                    "price": "0.49",
                    "size": "12",
                    "side": "BUY",
                    "hash": "delta-hash",
                    "best_bid": "0.49",
                    "best_ask": "0.52",
                }
            ],
        }
    )

    assert probe.notify_calls == 1


@pytest.mark.asyncio
async def test_market_data_sensor_last_trade_price_resets_watchdog() -> None:
    store = _store_mock()
    sensor = MarketDataSensor(store=store)
    probe = WatchdogProbe()
    sensor._watchdog = cast(Any, probe)

    await sensor._handle_message(
        {
            "event_type": "last_trade_price",
            "market": "m-trade-watchdog",
            "asset_id": "asset-trade-watchdog",
            "price": "0.456",
            "side": "BUY",
            "size": "219.217767",
            "fee_rate_bps": "0",
            "timestamp": "1750428146322",
        }
    )

    assert probe.notify_calls == 1


@pytest.mark.asyncio
async def test_market_data_sensor_watchdog_warns_after_silence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensor = MarketDataSensor(store=_store_mock())
    sensor._watchdog.timeout_s = 0.01

    with caplog.at_level(logging.WARN):
        await sensor._watchdog.start()
        try:
            await _wait_for(lambda: sensor.watchdog_timeout_count == 1)
        finally:
            await sensor._watchdog.stop()

    assert sensor.watchdog_timeout_count == 1
    assert "market data sensor silent for" in caplog.text


@pytest.mark.asyncio
async def test_market_data_sensor_watchdog_timeout_closes_websocket() -> None:
    sensor = MarketDataSensor(store=_store_mock())

    closed_event = asyncio.Event()

    class TrackingWebSocket:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True
            closed_event.set()

    websocket = TrackingWebSocket()
    sensor._websocket = websocket

    await sensor._handle_watchdog_timeout()

    assert websocket.closed is True
    assert closed_event.is_set()
    assert sensor.watchdog_timeout_count == 1


@pytest.mark.asyncio
async def test_market_data_sensor_fresh_connection_does_not_trigger_watchdog(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.return_value = 1
    websocket = HeartbeatWebSocket(
        [
            json.dumps({"event_type": "keepalive"}),
            json.dumps(
                {
                    "event_type": "book",
                    "market": "m-fresh",
                    "asset_id": "asset-fresh",
                    "timestamp": "1757908892351",
                    "hash": "book-fresh",
                    "bids": [{"price": "0.40", "size": "15"}],
                    "asks": [{"price": "0.60", "size": "9"}],
                    "last_trade_price": "0.41",
                }
            ),
        ],
        message_delays_s=[0.0, 0.01],
    )
    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        lambda url: FakeConnect(websocket),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-fresh"],
    )
    sensor._watchdog.timeout_s = 0.05
    sensor._heartbeat_interval_s = 1.0
    iterator = cast(Any, sensor.__aiter__())

    with caplog.at_level(logging.WARN):
        signal = await asyncio.wait_for(anext(iterator), timeout=0.2)
    await iterator.aclose()
    await sensor.aclose()

    assert signal.market_id == "m-fresh"
    assert sensor.watchdog_timeout_count == 0
    assert "market data sensor silent for" not in caplog.text


@pytest.mark.asyncio
async def test_market_data_sensor_reconnect_backoff_caps_and_resets_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store_mock()
    store_mock = cast(StoreMock, store)
    store_mock.write_book_snapshot_mock.side_effect = [1, 2]
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        if seconds >= 1.0:
            await real_sleep(0.05)
            return
        sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(
        "pms.sensor.adapters.market_data.connect",
        ConnectSequence(
            [
                OSError("boom-1"),
                OSError("boom-2"),
                OSError("boom-3"),
                FakeWebSocket(
                    [
                        json.dumps(
                            {
                                "event_type": "book",
                                "market": "m-reset-1",
                                "asset_id": "asset-reset",
                                "timestamp": "1757908892351",
                                "hash": "book-reset-1",
                                "bids": [{"price": "0.48", "size": "10"}],
                                "asks": [{"price": "0.52", "size": "8"}],
                                "last_trade_price": "0.50",
                            }
                        )
                    ]
                ),
                OSError("boom-4"),
                FakeWebSocket(
                    [
                        json.dumps(
                            {
                                "event_type": "book",
                                "market": "m-reset-2",
                                "asset_id": "asset-reset",
                                "timestamp": "1757908892451",
                                "hash": "book-reset-2",
                                "bids": [{"price": "0.44", "size": "11"}],
                                "asks": [{"price": "0.56", "size": "7"}],
                                "last_trade_price": "0.45",
                            }
                        )
                    ]
                ),
            ]
        ),
    )
    sensor = MarketDataSensor(
        store=store,
        ws_url="ws://market-data.example.test",
        asset_ids=["asset-reset"],
    )
    monkeypatch.setattr(sensor, "_sleep", fake_sleep)
    sensor._INITIAL_BACKOFF_S = 0.01
    sensor.max_reconnect_interval_s = 0.02
    sensor._watchdog.timeout_s = 1.0
    iterator = cast(Any, sensor.__aiter__())

    await asyncio.wait_for(anext(iterator), timeout=0.2)
    await asyncio.wait_for(anext(iterator), timeout=0.2)
    await iterator.aclose()
    await sensor.aclose()

    # The first successful connection closes cleanly; the fourth sleep is the
    # bounded delay after that clean close, before the next transient error.
    assert sleeps[:5] == [0.01, 0.02, 0.02, 0.01, 0.01]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
async def test_runner_builds_market_discovery_and_market_data_sensors_for_non_backtest_modes(
    mode: RunMode,
) -> None:
    from pms.runner import Runner
    from pms.sensor.adapters.market_discovery import MarketDiscoverySensor

    runner = Runner(
        config=PMSSettings(
            mode=mode,
            sensor=SensorSettings(
                poll_interval_s=7.5,
                max_reconnect_interval_s=9.0,
            ),
        )
    )
    runner.bind_pg_pool(cast(Any, object()))

    sensors = runner._build_sensors()

    assert len(sensors) == 2
    assert isinstance(sensors[0], MarketDiscoverySensor)
    assert isinstance(sensors[1], MarketDataSensor)
    assert sensors[1].max_reconnect_interval_s == 9.0
    assert sensors[1].max_message_size_bytes == 8 * 1024 * 1024
    await sensors[0].aclose()
    await sensors[1].aclose()
