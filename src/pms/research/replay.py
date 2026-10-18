"""Market-universe replay engine for research backtests."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
import json
from math import nan
import os
from pathlib import Path
import stat
from typing import Any, TypeAlias, cast

import asyncpg

from pms.core.enums import MarketStatus
from pms.core.models import (
    BookLevel,
    BookSide,
    BookSnapshot,
    BookSource,
    MarketSignal,
    PriceChange,
    Trade,
    Venue,
)
from pms.core.validation import normalize_resolved_outcome
from pms.research.specs import BacktestExecutionConfig, BacktestSpec


ChunkObserver: TypeAlias = Callable[[datetime, datetime], None]
WriteProbe: TypeAlias = Callable[[asyncpg.Connection], Awaitable[None]]

_SNAPSHOT_PRIORITY = 0
_PRICE_CHANGE_PRIORITY = 1
_TRADE_PRIORITY = 2


class ReplayEngineInvariantError(RuntimeError):
    """Raised when the replay engine violates the outer-ring read-only contract."""


@dataclass(frozen=True, slots=True)
class _ChunkWindow:
    start: datetime
    end: datetime
    is_final: bool


@dataclass(frozen=True, slots=True)
class _MarketMetadata:
    title: str
    venue: Venue
    resolves_at: datetime | None
    resolved_outcome: float | None = None
    resolved_at: datetime | None = None
    observed_at: datetime | None = None
    yes_token_id: str | None = None
    no_token_id: str | None = None


@dataclass(slots=True)
class _LastBookState:
    market_id: str
    token_id: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_trade_price: float | None = None
    last_hash: str | None = None
    book_ts: datetime | None = None


@dataclass(frozen=True, slots=True)
class _SnapshotPayload:
    snapshot: BookSnapshot
    levels: tuple[BookLevel, ...]


@dataclass(frozen=True, slots=True)
class _ReplayEvent:
    ts: datetime
    priority: int
    sequence: int
    payload: _SnapshotPayload | PriceChange | Trade


@dataclass(frozen=True, slots=True)
class _SignalCheckpoint:
    ts: datetime
    sequence: int
    signal: MarketSignal


@dataclass(slots=True)
class MarketUniverseReplayEngine:
    pool: asyncpg.Pool | None = None
    jsonl_path: Path | None = None
    write_probe: WriteProbe | None = None
    chunk_observer: ChunkObserver | None = None
    _spec: BacktestSpec | None = field(init=False, default=None)
    _exec_config: BacktestExecutionConfig | None = field(init=False, default=None)
    _lookup_chunks: AsyncGenerator[tuple[_ChunkWindow, list[MarketSignal]], None] | None = field(init=False, default=None)
    _lookup_window: _ChunkWindow | None = field(init=False, default=None)
    _signal_history: dict[tuple[str, str], tuple[_SignalCheckpoint, ...]] = field(
        init=False,
        default_factory=dict,
    )
    _market_metadata_cache: dict[str, _MarketMetadata] = field(
        init=False,
        default_factory=dict,
    )

    @classmethod
    def from_jsonl(cls, path: str | Path) -> MarketUniverseReplayEngine:
        return cls(jsonl_path=Path(path))

    async def stream(
        self,
        spec: BacktestSpec,
        exec_config: BacktestExecutionConfig,
    ) -> AsyncGenerator[MarketSignal, None]:
        if exec_config.chunk_days <= 0:
            raise ValueError("BacktestExecutionConfig.chunk_days must be positive")
        await self.aclose()
        self._spec, self._exec_config = spec, exec_config
        if self.jsonl_path is not None:
            metadata: dict[str, _MarketMetadata] = {}
            for _event in _iter_jsonl_events(self.jsonl_path, metadata):
                pass
            self._market_metadata_cache = _filter_market_metadata(
                metadata, market_universe_filter=spec.dataset.market_universe_filter,
            )
        else:
            self._market_metadata_cache = await self._load_market_metadata(
                spec.dataset.market_universe_filter
            )
        if not self._market_metadata_cache:
            raise ValueError("BacktestSpec.dataset.market_universe_filter matched zero markets")
        chunks = self._signal_chunks(spec, exec_config)
        try:
            async for _window, signals in chunks:
                for signal in signals:
                    yield _clone_signal(signal)
        finally:
            await chunks.aclose()
            await self.aclose()

    async def book_state_at(
        self,
        ts: datetime,
        *,
        market_id: str,
        token_id: str | None,
    ) -> dict[str, list[dict[str, float]]]:
        signal = await self.signal_at(ts, market_id=market_id, token_id=token_id)
        return _clone_orderbook(signal.orderbook)

    async def signal_at(
        self,
        ts: datetime,
        *,
        market_id: str,
        token_id: str | None,
    ) -> MarketSignal:
        if self._spec is None or self._exec_config is None:
            raise LookupError("replay stream has not been initialized")
        if self._lookup_window is not None and ts < self._lookup_window.start:
            await self.aclose()
        if self._lookup_chunks is None:
            self._lookup_chunks = self._signal_chunks(self._spec, self._exec_config)
        while self._lookup_window is None or (
            not self._lookup_window.is_final and ts >= self._lookup_window.end
        ):
            try:
                window, signals = await anext(self._lookup_chunks)
            except StopAsyncIteration:
                break
            # Keep one prior book per token, including tokens absent in this chunk.
            history = {key: [items[-1]] for key, items in self._signal_history.items()}
            for sequence, signal in enumerate(signals):
                key = (signal.market_id, signal.token_id or "")
                history.setdefault(key, []).append(
                    _SignalCheckpoint(signal.fetched_at, sequence, signal)
                )
            self._signal_history = {key: tuple(items) for key, items in history.items()}
            self._lookup_window = window
        checkpoints = self._signal_history.get((market_id, token_id or ""), ())
        for checkpoint in reversed(checkpoints):
            if checkpoint.ts <= ts:
                return _clone_signal(checkpoint.signal)
        raise LookupError(f"no replay history as of {ts} for {market_id!r}/{token_id!r}")

    async def latest_signal(
        self,
        *,
        market_id: str,
        token_id: str | None,
    ) -> MarketSignal:
        if self._spec is None:
            raise LookupError("replay stream has not been initialized")
        return await self.signal_at(
            self._spec.date_range_end, market_id=market_id, token_id=token_id,
        )

    async def aclose(self) -> None:
        """Release the lookup cursor, including after queries outside stream()."""
        if self._lookup_chunks is not None:
            await self._lookup_chunks.aclose()
        self._lookup_chunks = None
        self._lookup_window = None
        self._signal_history.clear()

    def resolution_for(
        self,
        market_id: str,
        *,
        as_of: datetime | None = None,
    ) -> float | None:
        metadata = self._market_metadata_cache.get(market_id)
        if metadata is None:
            return None
        if as_of is not None:
            known_at = metadata.resolved_at or metadata.observed_at
            if known_at is None or known_at > as_of:
                return None
        return metadata.resolved_outcome

    async def _signal_chunks(
        self,
        spec: BacktestSpec,
        exec_config: BacktestExecutionConfig,
    ) -> AsyncGenerator[tuple[_ChunkWindow, list[MarketSignal]], None]:
        states: dict[tuple[str, str], _LastBookState] = {}
        market_ids = tuple(sorted(self._market_metadata_cache))
        source = (
            _iter_jsonl_events(self.jsonl_path, {})
            if self.jsonl_path is not None else None
        )
        pending = next(source, None) if source is not None else None
        try:
            for window in _chunk_windows(
                start=spec.date_range_start, end=spec.date_range_end,
                chunk_days=exec_config.chunk_days,
            ):
                if self.chunk_observer is not None:
                    self.chunk_observer(window.start, window.end)
                if source is None:
                    events = await self._load_chunk_events(window=window, market_ids=market_ids)
                else:
                    events = []
                    while pending is not None and (
                        pending.ts < window.end or (window.is_final and pending.ts == window.end)
                    ):
                        if pending.ts >= window.start and _event_market_id(pending) in self._market_metadata_cache:
                            events.append(pending)
                        pending = next(source, None)
                signals = [
                    _event_to_signal(event=event, states=states, market_metadata=self._market_metadata_cache)
                    for event in events
                ]
                del events
                yield window, signals
                del signals
        finally:
            if source is not None:
                source.close()

    async def _load_market_metadata(
        self,
        market_universe_filter: Mapping[str, Any],
    ) -> dict[str, _MarketMetadata]:
        if self.pool is None:
            msg = "MarketUniverseReplayEngine.pool is required for PostgreSQL replay"
            raise RuntimeError(msg)

        connection = await self.pool.acquire()
        try:
            explicit_market_ids = _market_ids_from_filter(
                market_universe_filter.get("market_ids")
            ) or _market_ids_from_filter(market_universe_filter.get("condition_ids"))
            venue = _optional_str(market_universe_filter.get("venue"))
            if explicit_market_ids:
                rows = await connection.fetch(
                    """
                    SELECT markets.condition_id, markets.question, markets.venue,
                           markets.resolves_at, markets.closed,
                           markets.yes_price, markets.no_price,
                           resolutions.resolved_outcome,
                           resolutions.resolved_at,
                           resolutions.observed_at,
                           markets.last_seen_at,
                           (SELECT MAX(token_id) FROM tokens WHERE condition_id = markets.condition_id
                            AND outcome = 'YES') AS yes_token_id,
                           (SELECT MAX(token_id) FROM tokens WHERE condition_id = markets.condition_id
                            AND outcome = 'NO') AS no_token_id
                    FROM markets
                    LEFT JOIN market_resolutions AS resolutions
                      ON resolutions.market_id = markets.condition_id
                    WHERE markets.condition_id = ANY($1::text[])
                      AND ($2::text IS NULL OR markets.venue = $2)
                    ORDER BY markets.condition_id ASC
                    """,
                    list(explicit_market_ids),
                    venue,
                )
            else:
                rows = await connection.fetch(
                    """
                    SELECT markets.condition_id, markets.question, markets.venue,
                           markets.resolves_at, markets.closed,
                           markets.yes_price, markets.no_price,
                           resolutions.resolved_outcome,
                           resolutions.resolved_at,
                           resolutions.observed_at,
                           markets.last_seen_at,
                           (SELECT MAX(token_id) FROM tokens WHERE condition_id = markets.condition_id
                            AND outcome = 'YES') AS yes_token_id,
                           (SELECT MAX(token_id) FROM tokens WHERE condition_id = markets.condition_id
                            AND outcome = 'NO') AS no_token_id
                    FROM markets
                    LEFT JOIN market_resolutions AS resolutions
                      ON resolutions.market_id = markets.condition_id
                    WHERE ($1::text IS NULL OR markets.venue = $1)
                    ORDER BY markets.condition_id ASC
                    """,
                    venue,
                )
        finally:
            await self.pool.release(connection)

        return {
            cast(str, row["condition_id"]): _MarketMetadata(
                title=cast(str, row["question"]),
                venue=cast(Venue, row["venue"]),
                resolves_at=cast(datetime | None, row["resolves_at"]),
                resolved_outcome=_market_resolution_label(row),
                resolved_at=cast(datetime | None, row.get("resolved_at")),
                observed_at=cast(
                    datetime | None,
                    row.get("observed_at") or row.get("last_seen_at"),
                ),
                yes_token_id=cast(str | None, row.get("yes_token_id")),
                no_token_id=cast(str | None, row.get("no_token_id")),
            )
            for row in rows
        }

    async def _load_chunk_events(
        self,
        *,
        window: _ChunkWindow,
        market_ids: tuple[str, ...],
    ) -> list[_ReplayEvent]:
        if self.pool is None:
            msg = "MarketUniverseReplayEngine.pool is required for PostgreSQL replay"
            raise RuntimeError(msg)

        connection = await self.pool.acquire()
        try:
            if self.write_probe is not None:
                await self._run_write_probe(connection)
            snapshots = await _fetch_snapshots(
                connection=connection,
                window=window,
                market_ids=market_ids,
            )
            levels_by_snapshot = await _fetch_levels(
                connection=connection,
                snapshot_ids=tuple(snapshot.id for snapshot in snapshots),
            )
            price_changes = await _fetch_price_changes(
                connection=connection,
                window=window,
                market_ids=market_ids,
            )
            trades = await _fetch_trades(
                connection=connection,
                window=window,
                market_ids=market_ids,
            )
        finally:
            await self.pool.release(connection)

        events: list[_ReplayEvent] = []
        for snapshot in snapshots:
            events.append(
                _ReplayEvent(
                    ts=snapshot.ts,
                    priority=_SNAPSHOT_PRIORITY,
                    sequence=snapshot.id,
                    payload=_SnapshotPayload(
                        snapshot=snapshot,
                        levels=levels_by_snapshot.get(snapshot.id, ()),
                    ),
                )
            )
        for price_change in price_changes:
            events.append(
                _ReplayEvent(
                    ts=price_change.ts,
                    priority=_PRICE_CHANGE_PRIORITY,
                    sequence=price_change.id,
                    payload=price_change,
                )
            )
        for trade in trades:
            events.append(
                _ReplayEvent(
                    ts=trade.ts,
                    priority=_TRADE_PRIORITY,
                    sequence=trade.id,
                    payload=trade,
                )
            )
        return sorted(events, key=lambda event: (event.ts, event.priority, event.sequence))

    async def _run_write_probe(self, connection: asyncpg.Connection) -> None:
        assert self.write_probe is not None
        try:
            await self.write_probe(connection)
        except asyncpg.PostgresError as exc:
            sqlstate = getattr(exc, "sqlstate", "")
            lowered = str(exc).lower()
            if sqlstate == "42501" or "permission denied" in lowered:
                probe_name = getattr(self.write_probe, "__qualname__", repr(self.write_probe))
                msg = f"engine attempted WRITE on outer ring: {probe_name}: {exc}"
                raise ReplayEngineInvariantError(msg) from exc
            raise


def _market_resolution_label(row: Any) -> float | None:
    raw_label = row.get("resolved_outcome") if hasattr(row, "get") else None
    label = normalize_resolved_outcome(raw_label)
    if label is not None:
        return label
    closed = row.get("closed") if hasattr(row, "get") else None
    if closed is not True:
        return None
    try:
        yes_price = float(row.get("yes_price"))
        no_price = float(row.get("no_price"))
    except (TypeError, ValueError):
        return None
    if yes_price >= 0.999 and no_price <= 0.001:
        return 1.0
    if no_price >= 0.999 and yes_price <= 0.001:
        return 0.0
    return None


def _iter_jsonl_events(
    path: Path, market_metadata: dict[str, _MarketMetadata],
) -> Generator[_ReplayEvent, None, None]:
    last_ts: datetime | None = None
    last_sequence_by_market: dict[str, int] = {}

    for line_number, line in enumerate(
        _read_jsonl_lines_no_follow(path),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            msg = f"{path}:{line_number} must contain a JSON object per line"
            raise ValueError(msg)
        ts = _parse_datetime(cast(object, row.get("ts")), path=path, line_number=line_number)
        if last_ts is not None and ts < last_ts:
            msg = f"{path}:{line_number} violates monotonic ts ordering"
            raise ValueError(msg)
        last_ts = ts

        market_id = _required_str(row, "market_id", path=path, line_number=line_number)
        token_id = str(row.get("token_id") or f"{market_id}-token")
        sequence = _required_int(row, "sequence", path=path, line_number=line_number)
        previous_sequence = last_sequence_by_market.get(market_id)
        if previous_sequence is not None and sequence <= previous_sequence:
            msg = f"{path}:{line_number} violates monotonic sequence ordering for {market_id}"
            raise ValueError(msg)
        last_sequence_by_market[market_id] = sequence

        market_metadata.setdefault(
            market_id,
            _MarketMetadata(
                title=str(row.get("title") or market_id),
                venue=cast(Venue, str(row.get("venue") or "polymarket")),
                resolves_at=_optional_datetime(row.get("resolves_at")),
                resolved_outcome=normalize_resolved_outcome(row.get("resolved_outcome")),
                resolved_at=_optional_datetime(row.get("resolved_at")),
                observed_at=_optional_datetime(row.get("observed_at")) or ts,
                yes_token_id=_optional_str(row.get("yes_token_id")),
                no_token_id=_optional_str(row.get("no_token_id")),
            ),
        )

        # Token identity is immutable metadata, even when supplied on a later row.
        metadata = market_metadata[market_id]
        market_metadata[market_id] = replace(
            metadata,
            yes_token_id=metadata.yes_token_id or _optional_str(row.get("yes_token_id")),
            no_token_id=metadata.no_token_id or _optional_str(row.get("no_token_id")),
        )

        event_type = _required_str(row, "event_type", path=path, line_number=line_number)
        if event_type == "book_snapshot":
            levels = _jsonl_snapshot_levels(
                row,
                market_id=market_id,
                snapshot_id=sequence,
                path=path,
                line_number=line_number,
            )
            payload: _SnapshotPayload | PriceChange | Trade = _SnapshotPayload(
                snapshot=BookSnapshot(
                    id=sequence,
                    market_id=market_id,
                    token_id=token_id,
                    ts=ts,
                    hash=cast(str | None, row.get("hash")),
                    source="checkpoint",
                ),
                levels=levels,
            )
            priority = _SNAPSHOT_PRIORITY
        elif event_type == "level_update":
            payload = PriceChange(
                id=sequence,
                market_id=market_id,
                token_id=token_id,
                ts=ts,
                side=_jsonl_side(row, path=path, line_number=line_number),
                price=_required_float(row, "price", path=path, line_number=line_number),
                size=_required_float(row, "size", path=path, line_number=line_number),
                best_bid=_optional_float(row.get("best_bid")),
                best_ask=_optional_float(row.get("best_ask")),
                hash=cast(str | None, row.get("hash")),
            )
            priority = _PRICE_CHANGE_PRIORITY
        elif event_type == "trade":
            payload = Trade(
                id=sequence,
                market_id=market_id,
                token_id=token_id,
                ts=ts,
                price=_required_float(row, "price", path=path, line_number=line_number),
            )
            priority = _TRADE_PRIORITY
        else:
            msg = f"{path}:{line_number} unsupported event_type={event_type!r}"
            raise ValueError(msg)

        yield _ReplayEvent(ts=ts, priority=priority, sequence=sequence, payload=payload)


def _read_jsonl_lines_no_follow(path: Path) -> Iterator[str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        fd = os.open(path, flags)
        path_stat = os.fstat(fd)
        if not stat.S_ISREG(path_stat.st_mode):
            raise OSError("not a regular file")
        if path_stat.st_nlink != 1:
            raise OSError("multiple hardlinks")
        with os.fdopen(fd, "r", encoding="utf-8") as file:
            fd = -1
            yield from file
    except OSError as exc:
        msg = f"replay JSONL cannot be read safely: {path}"
        raise ValueError(msg) from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _filter_market_metadata(
    market_metadata: Mapping[str, _MarketMetadata],
    *,
    market_universe_filter: Mapping[str, Any],
) -> dict[str, _MarketMetadata]:
    explicit_market_ids = set(
        _market_ids_from_filter(market_universe_filter.get("market_ids"))
        or _market_ids_from_filter(market_universe_filter.get("condition_ids"))
    )
    venue = _optional_str(market_universe_filter.get("venue"))
    filtered: dict[str, _MarketMetadata] = {}
    for market_id, metadata in market_metadata.items():
        if explicit_market_ids and market_id not in explicit_market_ids:
            continue
        if venue is not None and metadata.venue != venue:
            continue
        filtered[market_id] = metadata
    return filtered


def _jsonl_snapshot_levels(
    row: dict[str, object],
    *,
    market_id: str,
    snapshot_id: int,
    path: Path,
    line_number: int,
) -> tuple[BookLevel, ...]:
    raw_orderbook = row.get("orderbook")
    bids: object
    asks: object
    if isinstance(raw_orderbook, dict):
        bids = raw_orderbook.get("bids", [])
        asks = raw_orderbook.get("asks", [])
    else:
        bids = row.get("bids", [])
        asks = row.get("asks", [])

    levels: list[BookLevel] = []
    for side_name, entries in (("BUY", bids), ("SELL", asks)):
        if not isinstance(entries, list):
            msg = f"{path}:{line_number} snapshot {side_name.lower()} levels must be arrays"
            raise ValueError(msg)
        for entry in entries:
            if not isinstance(entry, dict):
                msg = f"{path}:{line_number} snapshot levels must be objects"
                raise ValueError(msg)
            levels.append(
                BookLevel(
                    snapshot_id=snapshot_id,
                    market_id=market_id,
                    side=cast(BookSide, side_name),
                    price=_required_float(
                        cast(dict[str, object], entry),
                        "price",
                        path=path,
                        line_number=line_number,
                    ),
                    size=_required_float(
                        cast(dict[str, object], entry),
                        "size",
                        path=path,
                        line_number=line_number,
                    ),
                )
            )
    return tuple(levels)


def _event_to_signal(
    *,
    event: _ReplayEvent,
    states: dict[tuple[str, str], _LastBookState],
    market_metadata: Mapping[str, _MarketMetadata],
) -> MarketSignal:
    payload = event.payload
    if isinstance(payload, _SnapshotPayload):
        snapshot = payload.snapshot
        state = _state_for(states, market_id=snapshot.market_id, token_id=snapshot.token_id)
        state.bids = {
            level.price: level.size
            for level in payload.levels
            if level.side == "BUY"
        }
        state.asks = {
            level.price: level.size
            for level in payload.levels
            if level.side == "SELL"
        }
        state.last_hash = snapshot.hash
        state.book_ts = snapshot.ts
        return _market_signal(
            state=state,
            metadata=market_metadata[snapshot.market_id],
            timestamp=snapshot.ts,
            price=_signal_price(
                price=state.last_trade_price if state.last_trade_price is not None else nan,
                best_bid=max(state.bids, default=None),
                best_ask=min(state.asks, default=None),
            ),
            event_type="book",
        )
    if isinstance(payload, PriceChange):
        state = _state_for(states, market_id=payload.market_id, token_id=payload.token_id)
        _apply_price_change(state, change=payload)
        return _market_signal(
            state=state,
            metadata=market_metadata[payload.market_id],
            timestamp=payload.ts,
            price=_signal_price(
                price=payload.price,
                best_bid=payload.best_bid,
                best_ask=payload.best_ask,
            ),
            event_type="price_change",
            extra={
                "best_bid": payload.best_bid,
                "best_ask": payload.best_ask,
                "side": payload.side,
            },
        )

    state = _state_for(states, market_id=payload.market_id, token_id=payload.token_id)
    state.last_trade_price = payload.price
    return _market_signal(
        state=state,
        metadata=market_metadata[payload.market_id],
        timestamp=payload.ts,
        price=payload.price,
        event_type="last_trade_price",
    )


def _state_for(
    states: dict[tuple[str, str], _LastBookState],
    *,
    market_id: str,
    token_id: str,
) -> _LastBookState:
    key = (market_id, token_id)
    state = states.get(key)
    if state is None:
        state = _LastBookState(market_id=market_id, token_id=token_id)
        states[key] = state
    return state


def _apply_price_change(state: _LastBookState, *, change: PriceChange) -> None:
    levels = state.bids if change.side == "BUY" else state.asks
    if change.size <= 0:
        levels.pop(change.price, None)
    else:
        levels[change.price] = change.size
    state.last_hash = change.hash
    state.book_ts = change.ts


def _market_signal(
    *,
    state: _LastBookState,
    metadata: _MarketMetadata,
    timestamp: datetime,
    price: float | None,
    event_type: str,
    extra: dict[str, Any] | None = None,
) -> MarketSignal:
    external_signal: dict[str, Any] = {"raw_event_type": event_type}
    if state.book_ts is not None:
        external_signal["book_received_at"] = state.book_ts.isoformat()
    if metadata.yes_token_id is not None:
        external_signal["yes_token_id"] = metadata.yes_token_id
    if metadata.no_token_id is not None:
        external_signal["no_token_id"] = metadata.no_token_id
    if extra is not None:
        external_signal.update(extra)
    return MarketSignal(
        market_id=state.market_id,
        token_id=state.token_id,
        venue=metadata.venue,
        title=metadata.title,
        yes_price=price if price is not None else nan,
        volume_24h=None,
        resolves_at=metadata.resolves_at,
        orderbook=_orderbook_from_state(state),
        external_signal=external_signal,
        fetched_at=timestamp,
        market_status=_market_status(timestamp=timestamp, resolves_at=metadata.resolves_at),
    )


def _clone_signal(signal: MarketSignal) -> MarketSignal:
    return replace(
        signal,
        orderbook=_clone_orderbook(signal.orderbook),
        external_signal=dict(signal.external_signal),
    )


def _clone_orderbook(orderbook: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    return {
        "bids": [
            {"price": _coerce_level_float(level, "price"), "size": _coerce_level_float(level, "size")}
            for level in cast(list[dict[str, object]], orderbook.get("bids", []))
        ],
        "asks": [
            {"price": _coerce_level_float(level, "price"), "size": _coerce_level_float(level, "size")}
            for level in cast(list[dict[str, object]], orderbook.get("asks", []))
        ],
    }


def _orderbook_from_state(state: _LastBookState) -> dict[str, list[dict[str, float]]]:
    return {
        "bids": [
            {"price": price, "size": size}
            for price, size in sorted(state.bids.items(), key=lambda item: item[0], reverse=True)
        ],
        "asks": [
            {"price": price, "size": size}
            for price, size in sorted(state.asks.items(), key=lambda item: item[0])
        ],
    }


def _signal_price(
    *,
    price: float,
    best_bid: float | None,
    best_ask: float | None,
) -> float:
    if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
        return (best_bid + best_ask) / 2.0
    return price


def _market_status(*, timestamp: datetime, resolves_at: datetime | None) -> str:
    if resolves_at is not None and timestamp >= resolves_at:
        return MarketStatus.CLOSED.value
    return MarketStatus.OPEN.value


def _chunk_windows(
    *,
    start: datetime,
    end: datetime,
    chunk_days: int,
) -> Iterator[_ChunkWindow]:
    if start > end:
        msg = "BacktestSpec.date_range_start must not exceed date_range_end"
        raise ValueError(msg)

    if start == end:
        yield _ChunkWindow(start=start, end=end, is_final=True)
        return

    cursor = start
    while cursor < end:
        next_cursor = min(cursor + timedelta(days=chunk_days), end)
        yield _ChunkWindow(
            start=cursor, end=next_cursor, is_final=next_cursor == end,
        )
        cursor = next_cursor


async def _fetch_snapshots(
    *,
    connection: asyncpg.Connection,
    window: _ChunkWindow,
    market_ids: Sequence[str],
) -> list[BookSnapshot]:
    if not market_ids:
        return []
    comparator = "<=" if window.is_final else "<"
    rows = await connection.fetch(
        f"""
        SELECT id, market_id, token_id, ts, hash, source
        FROM book_snapshots
        WHERE ts >= $1
          AND ts {comparator} $2
          AND market_id = ANY($3::text[])
        ORDER BY ts ASC, id ASC
        """,
        window.start,
        window.end,
        list(market_ids),
    )
    return [
        BookSnapshot(
            id=cast(int, row["id"]),
            market_id=cast(str, row["market_id"]),
            token_id=cast(str, row["token_id"]),
            ts=cast(datetime, row["ts"]),
            hash=cast(str | None, row["hash"]),
            source=cast(BookSource, row["source"]),
        )
        for row in rows
    ]


async def _fetch_levels(
    *,
    connection: asyncpg.Connection,
    snapshot_ids: tuple[int, ...],
) -> dict[int, tuple[BookLevel, ...]]:
    if not snapshot_ids:
        return {}
    rows = await connection.fetch(
        """
        SELECT snapshot_id, market_id, side, price, size
        FROM book_levels
        WHERE snapshot_id = ANY($1::bigint[])
        ORDER BY
            snapshot_id ASC,
            CASE side WHEN 'BUY' THEN 0 ELSE 1 END,
            CASE side WHEN 'BUY' THEN price END DESC,
            CASE side WHEN 'SELL' THEN price END ASC
        """,
        list(snapshot_ids),
    )
    grouped: dict[int, list[BookLevel]] = {}
    for row in rows:
        snapshot_id = cast(int, row["snapshot_id"])
        grouped.setdefault(snapshot_id, []).append(
            BookLevel(
                snapshot_id=snapshot_id,
                market_id=cast(str, row["market_id"]),
                side=cast(BookSide, row["side"]),
                price=cast(float, row["price"]),
                size=cast(float, row["size"]),
            )
        )
    return {snapshot_id: tuple(levels) for snapshot_id, levels in grouped.items()}


async def _fetch_price_changes(
    *,
    connection: asyncpg.Connection,
    window: _ChunkWindow,
    market_ids: Sequence[str],
) -> list[PriceChange]:
    if not market_ids:
        return []
    comparator = "<=" if window.is_final else "<"
    rows = await connection.fetch(
        f"""
        SELECT id, market_id, token_id, ts, side, price, size, best_bid, best_ask, hash
        FROM price_changes
        WHERE ts >= $1
          AND ts {comparator} $2
          AND market_id = ANY($3::text[])
        ORDER BY ts ASC, id ASC
        """,
        window.start,
        window.end,
        list(market_ids),
    )
    return [
        PriceChange(
            id=cast(int, row["id"]),
            market_id=cast(str, row["market_id"]),
            token_id=cast(str, row["token_id"]),
            ts=cast(datetime, row["ts"]),
            side=cast(BookSide, row["side"]),
            price=cast(float, row["price"]),
            size=cast(float, row["size"]),
            best_bid=cast(float | None, row["best_bid"]),
            best_ask=cast(float | None, row["best_ask"]),
            hash=cast(str | None, row["hash"]),
        )
        for row in rows
    ]


async def _fetch_trades(
    *,
    connection: asyncpg.Connection,
    window: _ChunkWindow,
    market_ids: Sequence[str],
) -> list[Trade]:
    if not market_ids:
        return []
    comparator = "<=" if window.is_final else "<"
    rows = await connection.fetch(
        f"""
        SELECT id, market_id, token_id, ts, price
        FROM trades
        WHERE ts >= $1
          AND ts {comparator} $2
          AND market_id = ANY($3::text[])
        ORDER BY ts ASC, id ASC
        """,
        window.start,
        window.end,
        list(market_ids),
    )
    return [
        Trade(
            id=cast(int, row["id"]),
            market_id=cast(str, row["market_id"]),
            token_id=cast(str, row["token_id"]),
            ts=cast(datetime, row["ts"]),
            price=cast(float, row["price"]),
        )
        for row in rows
    ]


def _event_market_id(event: _ReplayEvent) -> str:
    payload = event.payload
    if isinstance(payload, _SnapshotPayload):
        return payload.snapshot.market_id
    return payload.market_id


def _market_ids_from_filter(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(str(item) for item in value))
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(sorted(str(item) for item in value))
    return ()


def _optional_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return float(value)


def _optional_datetime(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        msg = "expected ISO-8601 string for datetime field"
        raise ValueError(msg)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        msg = "expected timezone-aware ISO-8601 datetime"
        raise ValueError(msg)
    return parsed


def _parse_datetime(raw_value: object, *, path: Path, line_number: int) -> datetime:
    if not isinstance(raw_value, str):
        msg = f"{path}:{line_number} ts must be an ISO-8601 string"
        raise ValueError(msg)
    value = datetime.fromisoformat(raw_value)
    if value.tzinfo is None or value.utcoffset() is None:
        msg = f"{path}:{line_number} ts must be timezone-aware"
        raise ValueError(msg)
    return value


def _required_str(
    row: Mapping[str, object],
    key: str,
    *,
    path: Path,
    line_number: int,
) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{path}:{line_number} {key} must be a non-empty string"
        raise ValueError(msg)
    return value


def _required_int(
    row: Mapping[str, object],
    key: str,
    *,
    path: Path,
    line_number: int,
) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{path}:{line_number} {key} must be an integer"
        raise ValueError(msg)
    return value


def _required_float(
    row: Mapping[str, object],
    key: str,
    *,
    path: Path,
    line_number: int,
) -> float:
    value = row.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        msg = f"{path}:{line_number} {key} must be numeric"
        raise ValueError(msg)
    return float(value)


def _jsonl_side(
    row: Mapping[str, object],
    *,
    path: Path,
    line_number: int,
) -> BookSide:
    side = _required_str(row, "side", path=path, line_number=line_number).lower()
    if side == "bid":
        return "BUY"
    if side == "ask":
        return "SELL"
    msg = f"{path}:{line_number} side must be 'bid' or 'ask'"
    raise ValueError(msg)


def _coerce_level_float(level: dict[str, object], key: str) -> float:
    value = level.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        msg = f"orderbook level {key} must be numeric"
        raise ValueError(msg)
    return float(value)
