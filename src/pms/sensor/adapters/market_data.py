from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from collections import deque
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

import asyncpg
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from pms.core.enums import MarketStatus
from pms.core.exceptions import SensorDataQualityError
from pms.core.models import (
    BookLevel,
    BookSide,
    BookSnapshot,
    BookSource,
    MarketSignal,
    PriceChange,
    Trade,
)
from pms.sensor.watchdog import SensorWatchdog
from pms.storage.market_data_store import PostgresMarketDataStore


logger = logging.getLogger(__name__)
_POLYMARKET_MILLISECONDS_EPOCH_THRESHOLD = 10_000_000_000
_MALFORMED_RATE_WINDOW_SIZE = 100
_MALFORMED_RATE_THRESHOLD_DEFAULT = 0.5
_MALFORMED_WARNING_RATE_LIMIT_S = 10.0
_MALFORMED_PREVIEW_LIMIT = 200


@dataclass
class _BookState:
    market_id: str
    asset_id: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_trade_price: float | None = None
    fee_rate_bps: float | None = None
    last_hash: str | None = None


class MarketDataSensor:
    # Polymarket's observed market-channel idle timeout is ~60 s. Keep heartbeat
    # comfortably below that so missed pongs trigger reconnect before the server
    # silently drops the subscription.
    _INITIAL_BACKOFF_S = 1.0
    _DEFAULT_MAX_RECONNECT_INTERVAL_S = 60.0
    _HEARTBEAT_INTERVAL_S = 10.0
    _PONG_TIMEOUT_S = 15.0
    _WATCHDOG_TIMEOUT_S = 120.0
    _CLOSE_TIMEOUT_S = 1.0
    _DEFAULT_MAX_MESSAGE_SIZE_BYTES = 8 * 1024 * 1024
    _MAX_CONSECUTIVE_DATA_ERROR_RECONNECTS = 5

    def __init__(
        self,
        store: PostgresMarketDataStore,
        ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        asset_ids: list[str] | None = None,
        persist_price_changes: bool = False,
    ) -> None:
        self.store = store
        self.ws_url = ws_url
        self._asset_ids: list[str] = list(asset_ids) if asset_ids else []
        self.persist_price_changes = persist_price_changes
        self._subscribed_asset_ids: frozenset[str] = frozenset()
        self._books: dict[str, _BookState] = {}
        self._websocket: Any = None
        self._send_lock = asyncio.Lock()
        self._max_reconnect_interval_s = self._DEFAULT_MAX_RECONNECT_INTERVAL_S
        self._max_message_size_bytes = self._DEFAULT_MAX_MESSAGE_SIZE_BYTES
        self._heartbeat_interval_s = self._HEARTBEAT_INTERVAL_S
        self._pong_timeout_s = self._PONG_TIMEOUT_S
        self._connection_book_source: BookSource | None = None
        self._pending_reconnect_assets: set[str] = set()
        self._market_signal_metadata: dict[str, dict[str, str]] = {}
        self._connected_once = False
        self._watchdog_timeout_count = 0
        self._malformed_messages_total = 0
        self._malformed_message_window: deque[bool] = deque(
            maxlen=_MALFORMED_RATE_WINDOW_SIZE
        )
        self._malformed_rate_threshold = _MALFORMED_RATE_THRESHOLD_DEFAULT
        self._consecutive_data_error_reconnects = 0
        self._last_malformed_warning_at_s = -_MALFORMED_WARNING_RATE_LIMIT_S
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._watchdog = SensorWatchdog(
            timeout_s=self._WATCHDOG_TIMEOUT_S,
            fallback=self._handle_watchdog_timeout,
        )

    @property
    def max_reconnect_interval_s(self) -> float:
        return self._max_reconnect_interval_s

    @max_reconnect_interval_s.setter
    def max_reconnect_interval_s(self, value: float) -> None:
        self._max_reconnect_interval_s = value

    @property
    def max_message_size_bytes(self) -> int:
        return self._max_message_size_bytes

    @max_message_size_bytes.setter
    def max_message_size_bytes(self, value: int) -> None:
        if value < 1:
            msg = "max_message_size_bytes must be positive"
            raise ValueError(msg)
        self._max_message_size_bytes = value

    @property
    def watchdog_timeout_count(self) -> int:
        return self._watchdog_timeout_count

    @property
    def malformed_messages_total(self) -> int:
        return self._malformed_messages_total

    @property
    def malformed_rate_threshold(self) -> float:
        return self._malformed_rate_threshold

    @malformed_rate_threshold.setter
    def malformed_rate_threshold(self, value: float) -> None:
        self._malformed_rate_threshold = value

    @property
    def consecutive_data_error_reconnects(self) -> int:
        return self._consecutive_data_error_reconnects

    def __aiter__(self) -> AsyncIterator[MarketSignal]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MarketSignal]:
        backoff = self._INITIAL_BACKOFF_S
        await self._watchdog.start()
        try:
            while True:
                connection_book_source: BookSource = (
                    "reconnect" if self._connected_once else "subscribe"
                )
                connection_had_data = False
                try:
                    async with self._connect() as websocket:
                        self._reset_book_state()
                        self._connection_book_source = connection_book_source
                        heartbeat_task = asyncio.create_task(
                            self._heartbeat_loop(websocket)
                        )
                        self._heartbeat_task = heartbeat_task
                        try:
                            # Initial subscribe and dynamic updates share this
                            # lock. Assigning the socket and subscribed state
                            # under the same lock prevents an update from
                            # observing an empty state between the handshake
                            # and the initial subscribe.
                            async with self._send_lock:
                                self._websocket = websocket
                                self._pending_reconnect_assets = set(self._asset_ids)
                                await self._send_subscription_locked(websocket)
                                self._subscribed_asset_ids = frozenset(
                                    self._asset_ids
                                )
                            self._connected_once = True
                            async for raw_message in websocket:
                                signals = await self._handle_raw_message(raw_message)
                                connection_had_data = True
                                self._consecutive_data_error_reconnects = 0
                                for signal in signals:
                                    yield signal
                        finally:
                            heartbeat_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await heartbeat_task
                            self._heartbeat_task = None
                            async with self._send_lock:
                                if self._websocket is websocket:
                                    self._websocket = None
                                self._subscribed_asset_ids = frozenset()
                            self._connection_book_source = None
                            self._pending_reconnect_assets = set()
                            self._reset_book_state()
                except asyncio.CancelledError:
                    raise
                except (
                    ConnectionClosed,
                    ConnectionError,
                    OSError,
                    TimeoutError,
                    InvalidHandshake,
                    asyncpg.PostgresError,
                    SensorDataQualityError,
                    ValueError,
                    TypeError,
                ) as error:
                    is_data_error = isinstance(
                        error,
                        (SensorDataQualityError, ValueError, TypeError),
                    )
                    if is_data_error:
                        self._consecutive_data_error_reconnects += 1
                        if (
                            self._consecutive_data_error_reconnects
                            >= self._MAX_CONSECUTIVE_DATA_ERROR_RECONNECTS
                        ):
                            raise SensorDataQualityError(
                                "market data sensor exceeded consecutive "
                                "data-error reconnect limit"
                            ) from error
                    if isinstance(error, SensorDataQualityError):
                        self._malformed_message_window.clear()
                        logger.warning(
                            "market data sensor quality threshold exceeded; reconnecting: %s",
                            error,
                        )
                    elif isinstance(error, (ValueError, TypeError)):
                        logger.warning(
                            "market data sensor data/persistence error; reconnecting: %s",
                            error,
                        )
                    else:
                        _log_reconnectable_error(error)
                    if connection_had_data:
                        backoff = self._INITIAL_BACKOFF_S
                    await self._sleep(backoff)
                    if not connection_had_data:
                        backoff = min(backoff * 2.0, self._max_reconnect_interval_s)
                else:
                    # A normal websocket iterator completion is a clean
                    # server close. It still needs bounded backoff to avoid a
                    # tight reconnect loop. Successful data resets backoff;
                    # a close before data keeps the current backoff.
                    if connection_had_data:
                        backoff = self._INITIAL_BACKOFF_S
                    await self._sleep(backoff)
                    if not connection_had_data:
                        backoff = min(backoff * 2.0, self._max_reconnect_interval_s)
        finally:
            self._heartbeat_task = None
            await self._watchdog.stop()
            async with self._send_lock:
                self._websocket = None
                self._subscribed_asset_ids = frozenset()
            self._connection_book_source = None
            self._pending_reconnect_assets = set()
            self._reset_book_state()

    async def update_subscription(self, asset_ids: list[str]) -> None:
        next_asset_ids = list(dict.fromkeys(asset_ids))
        websocket: Any = None
        try:
            async with self._send_lock:
                self._asset_ids = next_asset_ids
                websocket = self._websocket
                if websocket is None:
                    return
                await self._send_subscription_update_locked(
                    websocket=websocket,
                    previous_asset_ids=self._subscribed_asset_ids,
                    next_asset_ids=next_asset_ids,
                )
                self._subscribed_asset_ids = frozenset(next_asset_ids)
        except (ConnectionClosed, ConnectionError, OSError, TimeoutError) as error:
            _log_reconnectable_error(error)
            if websocket is not None:
                with suppress(Exception):
                    await websocket.close()
                async with self._send_lock:
                    if self._websocket is websocket:
                        self._websocket = None
                        self._connection_book_source = None
                        self._subscribed_asset_ids = frozenset()
                        self._pending_reconnect_assets = set()
            return

    async def aclose(self) -> None:
        heartbeat_task = self._heartbeat_task
        self._heartbeat_task = None
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task

        async with self._send_lock:
            websocket = self._websocket
            self._websocket = None
            self._subscribed_asset_ids = frozenset()
        if websocket is not None:
            await websocket.close()
        await self._watchdog.stop()

    async def _handle_raw_message(self, raw_message: str | bytes) -> list[MarketSignal]:
        try:
            loaded = json.loads(raw_message)
        except (json.JSONDecodeError, ValueError, TypeError):
            self._record_malformed_message(raw_message)
            return []

        if isinstance(loaded, list):
            signals: list[MarketSignal] = []
            for item in loaded:
                try:
                    if not isinstance(item, dict):
                        raise TypeError("market-data payload item must be a dict")
                    signals.extend(await self._handle_message(item))
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                    if not _is_malformed_payload_error(exc):
                        raise
                    self._record_malformed_message(raw_message)
                    continue
                self._record_message_quality(ok=True)
            return signals

        if isinstance(loaded, dict):
            try:
                signals = await self._handle_message(loaded)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                if not _is_malformed_payload_error(exc):
                    raise
                self._record_malformed_message(raw_message)
                return []
            self._record_message_quality(ok=True)
            return signals

        logger.warning("unrecognised market-data payload: %s", loaded)
        return []

    async def _handle_message(
        self,
        message: dict[str, Any],
    ) -> list[MarketSignal]:
        event_type = str(message.get("event_type", ""))
        if event_type in {"keepalive", "pong", "tick_size_change"}:
            return []
        if event_type == "book":
            self._watchdog.notify_message()
            return [await self._handle_book(message)]
        if event_type == "price_change":
            self._watchdog.notify_message()
            return await self._handle_price_change(message)
        if event_type == "last_trade_price":
            self._watchdog.notify_message()
            return [await self._handle_last_trade_price(message)]
        logger.warning(
            "unrecognised market-data payload: %s",
            json.dumps(message, sort_keys=True),
        )
        return []

    async def _handle_book(
        self,
        message: dict[str, Any],
    ) -> MarketSignal:
        market_id = _required_str(message, "market")
        asset_id = _required_str(message, "asset_id")
        timestamp = _message_timestamp(message.get("timestamp"))
        state = self._book_state(market_id, asset_id)
        state.bids = _levels_to_map(message.get("bids"))
        state.asks = _levels_to_map(message.get("asks"))
        state.last_hash = _optional_str(message.get("hash"))
        fee_rate_bps = _optional_float(message.get("fee_rate_bps"))
        if fee_rate_bps is not None:
            state.fee_rate_bps = fee_rate_bps

        source: BookSource = "subscribe"
        if asset_id in self._pending_reconnect_assets:
            self._pending_reconnect_assets.discard(asset_id)
            source = self._connection_book_source or "subscribe"

        snapshot = BookSnapshot(
            id=0,
            market_id=market_id,
            token_id=asset_id,
            ts=timestamp,
            hash=state.last_hash,
            source=source,
        )
        await self.store.write_book_snapshot(snapshot, _book_levels_from_state(state))
        metadata = await self._market_metadata_for_signal(market_id)
        return _signal_from_state(
            state=state,
            timestamp=timestamp,
            price=_book_signal_price(state),
            event_type="book",
            extra=metadata,
        )

    async def _handle_price_change(self, message: dict[str, Any]) -> list[MarketSignal]:
        market_id = _required_str(message, "market")
        timestamp = _message_timestamp(message.get("timestamp"))
        changes = message.get("price_changes")
        if not isinstance(changes, list):
            msg = "price_changes payload must be a list"
            raise ValueError(msg)

        signals: list[MarketSignal] = []
        for change in changes:
            if not isinstance(change, dict):
                continue
            asset_id = _required_str(change, "asset_id")
            state = self._book_state(market_id, asset_id)
            price = _required_float(change, "price")
            size = _required_float(change, "size")
            side = _required_side(change.get("side"))
            _apply_price_change(state, side=side, price=price, size=size)
            fee_rate_bps = _optional_float(change.get("fee_rate_bps"))
            if fee_rate_bps is not None:
                state.fee_rate_bps = fee_rate_bps
            best_bid = _optional_float(change.get("best_bid"))
            best_ask = _optional_float(change.get("best_ask"))
            state.last_hash = _optional_str(change.get("hash"))

            if self.persist_price_changes:
                await self.store.write_price_change(
                    PriceChange(
                        id=0,
                        market_id=market_id,
                        token_id=asset_id,
                        ts=timestamp,
                        side=side,
                        price=price,
                        size=size,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        hash=state.last_hash,
                    )
                )
            metadata = await self._market_metadata_for_signal(market_id)
            signals.append(
                _signal_from_state(
                    state=state,
                    timestamp=timestamp,
                    price=_signal_price(price=price, best_bid=best_bid, best_ask=best_ask),
                    event_type="price_change",
                    extra={
                        **metadata,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "side": side,
                    },
                )
            )
        return signals

    async def _handle_last_trade_price(self, message: dict[str, Any]) -> MarketSignal:
        market_id = _required_str(message, "market")
        asset_id = _required_str(message, "asset_id")
        price = _required_float(message, "price")
        timestamp = _message_timestamp(message.get("timestamp"))
        state = self._book_state(market_id, asset_id)
        state.last_trade_price = price
        fee_rate_bps = _optional_float(message.get("fee_rate_bps"))
        if fee_rate_bps is not None:
            state.fee_rate_bps = fee_rate_bps
        await self.store.write_trade(
            Trade(
                id=0,
                market_id=market_id,
                token_id=asset_id,
                ts=timestamp,
                price=price,
            )
        )
        metadata = await self._market_metadata_for_signal(market_id)
        extra: dict[str, Any] = {
            **metadata,
            "side": _optional_str(message.get("side")),
            "size": _optional_float(message.get("size")),
        }
        if fee_rate_bps is not None:
            extra["fee_rate_bps"] = fee_rate_bps
        return _signal_from_state(
            state=state,
            timestamp=timestamp,
            price=price,
            event_type="last_trade_price",
            extra=extra,
        )

    def _book_state(self, market_id: str, asset_id: str) -> _BookState:
        state = self._books.get(asset_id)
        if state is None:
            state = _BookState(market_id=market_id, asset_id=asset_id)
            self._books[asset_id] = state
        else:
            state.market_id = market_id
        return state

    async def _market_metadata_for_signal(self, market_id: str) -> dict[str, str]:
        cached = self._market_signal_metadata.get(market_id)
        if cached is not None:
            return cached
        metadata = await self.store.read_market_signal_metadata(market_id)
        normalized = {
            key: value
            for key, value in metadata.items()
            if key
            in {
                "risk_group_id",
                "category",
                "event_id",
                "yes_token_id",
                "no_token_id",
            }
            and isinstance(value, str)
            and value.strip() != ""
        }
        if _has_complete_outcome_token_metadata(normalized):
            self._market_signal_metadata[market_id] = normalized
        return normalized

    async def _send_subscription_locked(self, websocket: Any) -> None:
        await websocket.send(json.dumps(_subscription_payload(self._asset_ids)))

    def _connect(self) -> Any:
        connect_kwargs: tuple[dict[str, Any], ...] = (
            {
                "close_timeout": self._CLOSE_TIMEOUT_S,
                "max_size": self._max_message_size_bytes,
                "proxy": None,
            },
            {
                "close_timeout": self._CLOSE_TIMEOUT_S,
                "max_size": self._max_message_size_bytes,
            },
            {},
        )
        for kwargs in connect_kwargs:
            try:
                return connect(self.ws_url, **kwargs)
            except TypeError as exc:
                if not _is_unsupported_connect_kwarg_error(exc):
                    raise
        return connect(self.ws_url)

    async def _send_subscription_update_locked(
        self,
        *,
        websocket: Any,
        previous_asset_ids: frozenset[str],
        next_asset_ids: list[str],
    ) -> None:
        next_asset_set = frozenset(next_asset_ids)
        removed_asset_ids = sorted(previous_asset_ids - next_asset_set)
        added_asset_ids = [
            asset_id
            for asset_id in next_asset_ids
            if asset_id not in previous_asset_ids
        ]
        if removed_asset_ids:
            await websocket.send(
                json.dumps(
                    _subscription_update_payload(
                        removed_asset_ids,
                        operation="unsubscribe",
                    )
                )
            )
            for asset_id in removed_asset_ids:
                self._books.pop(asset_id, None)
        if added_asset_ids:
            await websocket.send(
                json.dumps(
                    _subscription_update_payload(
                        added_asset_ids,
                        operation="subscribe",
                    )
                )
            )

    async def _sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    async def _heartbeat_loop(self, websocket: Any) -> None:
        while True:
            await self._sleep(self._heartbeat_interval_s)
            try:
                pong_waiter = await websocket.ping()
                await asyncio.wait_for(pong_waiter, timeout=self._pong_timeout_s)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                logger.warning("market data sensor missed pong; reconnecting")
                await websocket.close()
                return
            except Exception as error:
                if isinstance(error, ConnectionClosed):
                    _log_reconnectable_error(error)
                else:
                    logger.error("market data sensor heartbeat failed: %s", error)
                await websocket.close()
                return

    async def _handle_watchdog_timeout(self) -> None:
        self._watchdog_timeout_count += 1
        logger.warning(
            "market data sensor silent for %.1fs; forcing reconnect",
            self._watchdog.timeout_s,
        )
        websocket = self._websocket
        if websocket is not None:
            with suppress(Exception):
                await websocket.close()

    def _reset_book_state(self) -> None:
        self._books.clear()

    def _record_malformed_message(self, raw_message: str | bytes) -> None:
        self._malformed_messages_total += 1
        self._log_malformed_message(raw_message)
        self._record_message_quality(ok=False)

    def _record_message_quality(self, *, ok: bool) -> None:
        self._malformed_message_window.append(not ok)
        if len(self._malformed_message_window) < _MALFORMED_RATE_WINDOW_SIZE:
            return

        malformed_rate = sum(self._malformed_message_window) / _MALFORMED_RATE_WINDOW_SIZE
        if malformed_rate > self._malformed_rate_threshold:
            raise SensorDataQualityError(
                "malformed market-data rate exceeded quality threshold"
            )

    def _log_malformed_message(self, raw_message: str | bytes) -> None:
        now_s = time.monotonic()
        if now_s - self._last_malformed_warning_at_s < _MALFORMED_WARNING_RATE_LIMIT_S:
            return
        self._last_malformed_warning_at_s = now_s
        logger.warning(
            "malformed market-data payload: %s",
            _sanitize_payload_preview(raw_message),
        )


def _subscription_payload(asset_ids: list[str]) -> dict[str, Any]:
    return {
        "assets_ids": asset_ids,
        "type": "market",
        "initial_dump": True,
        "level": 2,
    }


def _is_unsupported_connect_kwarg_error(exc: TypeError) -> bool:
    message = str(exc)
    return (
        "unexpected keyword argument" in message
        or "got an unexpected keyword" in message
        or "takes no keyword arguments" in message
    )


def _log_reconnectable_error(error: BaseException) -> None:
    if isinstance(error, ConnectionClosed):
        logger.warning(
            "market data sensor websocket closed; reconnecting: %s",
            error,
        )
        return
    if isinstance(error, asyncpg.PostgresError):
        logger.warning(
            "market data sensor storage error; reconnecting: %s",
            error,
        )
        return
    logger.warning(
        "market data sensor transport interrupted; reconnecting: %s",
        error,
    )


def _subscription_update_payload(
    asset_ids: list[str],
    *,
    operation: str,
) -> dict[str, Any]:
    return {
        "assets_ids": asset_ids,
        "operation": operation,
    }


def _levels_to_map(raw_levels: object) -> dict[float, float]:
    if not isinstance(raw_levels, list):
        msg = "book levels must be a list"
        raise ValueError(msg)
    levels: dict[float, float] = {}
    for level in raw_levels:
        if not isinstance(level, dict):
            continue
        price = _required_float(level, "price")
        size = _required_float(level, "size")
        levels[price] = size
    return levels


def _book_levels_from_state(state: _BookState) -> list[BookLevel]:
    return [
        *[
            BookLevel(
                snapshot_id=0,
                market_id=state.market_id,
                side="BUY",
                price=price,
                size=size,
            )
            for price, size in sorted(state.bids.items(), key=lambda item: item[0], reverse=True)
        ],
        *[
            BookLevel(
                snapshot_id=0,
                market_id=state.market_id,
                side="SELL",
                price=price,
                size=size,
            )
            for price, size in sorted(state.asks.items(), key=lambda item: item[0])
        ],
    ]


def _signal_from_state(
    *,
    state: _BookState,
    timestamp: datetime,
    price: float | None,
    event_type: str,
    extra: dict[str, Any] | None = None,
) -> MarketSignal:
    orderbook = {
        "bids": [
            {"price": level_price, "size": level_size}
            for level_price, level_size in sorted(
                state.bids.items(), key=lambda item: item[0], reverse=True
            )
        ],
        "asks": [
            {"price": level_price, "size": level_size}
            for level_price, level_size in sorted(state.asks.items(), key=lambda item: item[0])
        ],
    }
    external_signal: dict[str, Any] = {
        "raw_event_type": event_type,
        "book_received_at": datetime.now(tz=UTC).isoformat(),
    }
    if extra is not None:
        external_signal.update(extra)
    if state.fee_rate_bps is not None and "fee_rate_bps" not in external_signal:
        external_signal["fee_rate_bps"] = state.fee_rate_bps
    if state.last_trade_price is not None and "last_trade_price" not in external_signal:
        external_signal["last_trade_price"] = state.last_trade_price
    token_outcome = _signal_token_outcome(
        asset_id=state.asset_id,
        yes_token_id=external_signal.get("yes_token_id"),
        no_token_id=external_signal.get("no_token_id"),
    )
    if token_outcome is not None:
        external_signal["signal_token_outcome"] = token_outcome
    return MarketSignal(
        market_id=state.market_id,
        token_id=state.asset_id,
        venue="polymarket",
        title="",
        yes_price=price if price is not None else 0.0,
        volume_24h=None,
        resolves_at=None,
        orderbook=orderbook,
        external_signal=external_signal,
        fetched_at=timestamp,
        market_status=MarketStatus.OPEN.value,
    )


def _signal_token_outcome(
    *,
    asset_id: str,
    yes_token_id: object,
    no_token_id: object,
) -> str | None:
    if isinstance(yes_token_id, str) and asset_id == yes_token_id:
        return "YES"
    if isinstance(no_token_id, str) and asset_id == no_token_id:
        return "NO"
    return None


def _has_complete_outcome_token_metadata(metadata: dict[str, str]) -> bool:
    return (
        metadata.get("yes_token_id", "").strip() != ""
        and metadata.get("no_token_id", "").strip() != ""
    )


def _signal_price(
    *,
    price: float,
    best_bid: float | None,
    best_ask: float | None,
) -> float:
    if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
        return (best_bid + best_ask) / 2.0
    return price


def _book_signal_price(state: _BookState) -> float | None:
    if state.bids and state.asks:
        best_bid = max(state.bids)
        best_ask = min(state.asks)
        if best_bid > 0.0 and best_ask > 0.0:
            return (best_bid + best_ask) / 2.0
    return state.last_trade_price


def _required_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if value is None or value == "":
        msg = f"{key} is required"
        raise KeyError(msg)
    return str(value)


def _required_float(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if value is None or value == "":
        msg = f"{key} is required"
        raise KeyError(msg)
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str | bytes | bytearray | int | float):
        msg = f"unsupported float payload: {value!r}"
        raise TypeError(msg)
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _required_side(value: object) -> BookSide:
    side = _optional_str(value)
    if side not in {"BUY", "SELL"}:
        msg = f"unsupported side: {value!r}"
        raise ValueError(msg)
    return cast(BookSide, side)


def _message_timestamp(value: object) -> datetime:
    if value is None or value == "":
        return datetime.now(tz=UTC)
    if isinstance(value, int | float):
        return _timestamp_number(float(value))
    text = str(value)
    if text.isdigit():
        return _timestamp_number(float(text))
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _timestamp_number(value: float) -> datetime:
    if value > _POLYMARKET_MILLISECONDS_EPOCH_THRESHOLD:
        return datetime.fromtimestamp(value / 1000.0, tz=UTC)
    return datetime.fromtimestamp(value, tz=UTC)


def _apply_price_change(
    state: _BookState,
    *,
    side: str,
    price: float,
    size: float,
) -> None:
    """Apply a price-level delta to the in-memory book state.

    Break points: size > 0 (level add/update), size == 0 (level removal).
    """

    side_levels = state.bids if side == "BUY" else state.asks
    if size == 0:
        side_levels.pop(price, None)
        return
    side_levels[price] = size


def _sanitize_payload_preview(raw_message: str | bytes) -> str:
    if isinstance(raw_message, bytes):
        text = raw_message.decode("utf-8", errors="replace")
    else:
        text = raw_message

    first_line = text.splitlines()[0] if text.splitlines() else text
    sanitized = "".join(ch for ch in first_line if ord(ch) >= 32 and ord(ch) != 127)
    return sanitized[:_MALFORMED_PREVIEW_LIMIT]


def _is_malformed_payload_error(exc: BaseException) -> bool:
    if isinstance(exc, json.JSONDecodeError):
        return True
    if isinstance(exc, KeyError):
        return True
    if not isinstance(exc, (ValueError, TypeError)):
        return False

    malformed_frame_names = {
        "_apply_price_change",
        "_handle_book",
        "_handle_last_trade_price",
        "_handle_message",
        "_handle_price_change",
        "_handle_raw_message",
        "_levels_to_map",
        "_message_timestamp",
        "_optional_float",
        "_required_float",
        "_required_side",
        "_required_str",
        "_timestamp_number",
    }
    persistence_frame_names = {
        "write_book_snapshot",
        "write_book_snapshot_mock",
        "write_price_change",
        "write_price_change_mock",
        "write_trade",
        "write_trade_mock",
    }
    frame_names = {frame.name for frame in traceback.extract_tb(exc.__traceback__)}
    if frame_names & persistence_frame_names:
        return False
    if frame_names & malformed_frame_names:
        return True
    return False
