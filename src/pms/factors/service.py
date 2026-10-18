from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
import logging
from typing import Any

import asyncpg

from pms.core.models import Market, MarketSignal
from pms.core.validation import validate_finite, validate_probability, validate_quantity
from pms.factors.base import FactorDefinition, FactorValueRow, is_no_token_signal
from pms.factors.catalog import ensure_factor_catalog, validate_registered_factor_catalog
from pms.research.cache import FactorPanel, load_factor_panel
from pms.storage.market_data_store import PostgresMarketDataStore


logger = logging.getLogger(__name__)
_MAX_PERSIST_RETRIES = 3
_PersistenceKey = tuple[str, str, str, datetime]


async def persist_factor_value(
    pool: asyncpg.Pool,
    row: FactorValueRow,
    *,
    definition_version: str = "1",
) -> None:
    _validate_factor_value(row)
    query = """
    INSERT INTO factor_values (
        factor_id,
        param,
        market_id,
        ts,
        value,
        definition_version
    ) VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (
        factor_id,
        param,
        market_id,
        ts,
        definition_version
    ) DO UPDATE
    SET value = EXCLUDED.value
    """
    async with pool.acquire() as connection:
        await connection.execute(
            query,
            row.factor_id,
            row.param,
            row.market_id,
            row.ts,
            row.value,
            definition_version,
        )


@dataclass
class FactorService:
    pool: asyncpg.Pool
    store: PostgresMarketDataStore
    cadence_s: float
    factors: tuple[type[FactorDefinition], ...]
    signal_stream: AsyncIterable[MarketSignal]
    _signal_queue: asyncio.Queue[MarketSignal | None] = field(
        init=False,
        default_factory=asyncio.Queue,
    )
    _last_persisted_ts: dict[tuple[str, str, str], datetime] = field(
        init=False,
        default_factory=dict,
    )
    _stream_exhausted: bool = field(init=False, default=False)
    _known_market_ids: set[str] = field(init=False, default_factory=set)
    _last_failed_market_ids: set[str] = field(init=False, default_factory=set)
    _quarantined_compute: dict[tuple[str, str], datetime] = field(
        init=False,
        default_factory=dict,
    )
    _persistence_failures: dict[_PersistenceKey, int] = field(
        init=False,
        default_factory=dict,
    )
    _quarantined_persistence: set[_PersistenceKey] = field(
        init=False,
        default_factory=set,
    )

    def __post_init__(self) -> None:
        if self.cadence_s <= 0.0:
            msg = "cadence_s must be positive"
            raise ValueError(msg)
        validate_registered_factor_catalog()

    async def run(self) -> None:
        stream_task = asyncio.create_task(self._forward_signals())
        dirty_signals: dict[str, MarketSignal] = {}
        try:
            while True:
                signals = self._drain_signal_buffer()
                for signal in signals:
                    # Coalesce bursts for the same market and only recompute
                    # markets that received a new observation this tick.
                    dirty_signals[signal.market_id] = signal
                if dirty_signals:
                    await self.compute_once(list(dirty_signals.values()))
                    failed_market_ids = self._last_failed_market_ids
                    dirty_signals = {
                        market_id: signal
                        for market_id, signal in dirty_signals.items()
                        if market_id in failed_market_ids
                    }
                if self._stream_exhausted:
                    self._raise_stream_error(stream_task)
                    if not dirty_signals:
                        logger.warning(
                            "factor signal stream exhausted; factor service exiting"
                        )
                        return
                    # Keep retryable shell and transient persistence failures
                    # queued even when the finite source has ended.  A later
                    # database recovery must not require a process restart.
                    await asyncio.sleep(self.cadence_s)
                    continue
                await asyncio.sleep(self.cadence_s)
        except asyncio.CancelledError:
            logger.info("factor service cancelled")
            raise
        except Exception:
            logger.exception("factor service failed")
            raise
        finally:
            if not stream_task.done():
                stream_task.cancel()
            await asyncio.gather(stream_task, return_exceptions=True)

    async def compute_once(self, signals: list[MarketSignal]) -> int:
        persisted = 0
        failed_market_ids: set[str] = set()
        for signal in signals:
            # Prices are YES probabilities; orderbooks retain their token identity.
            if is_no_token_signal(signal):
                signal = replace(signal, yes_price=1.0 - signal.yes_price)
            try:
                await self._ensure_market_shell(signal)
            except Exception:
                logger.exception(
                    "market shell upsert failed market_id=%s; retrying next cadence",
                    signal.market_id,
                )
                failed_market_ids.add(signal.market_id)
                continue
            for factor_cls in self.factors:
                if _requires_open_yes_price(factor_cls) and not _has_open_yes_price(
                    signal
                ):
                    continue
                quarantine_key = (signal.market_id, factor_cls.factor_id)
                if self._quarantined_compute.get(quarantine_key) == signal.timestamp:
                    continue
                try:
                    row = factor_cls().compute(signal, self.store)
                except Exception:
                    logger.exception(
                        "factor compute failed factor_id=%s market_id=%s ts=%s",
                        factor_cls.factor_id,
                        signal.market_id,
                        signal.timestamp.isoformat(),
                    )
                    # A deterministic error for this observation must not be
                    # retried on every cadence.  Quarantine the factor/market
                    # pair until a newer observation arrives.
                    self._quarantined_compute[quarantine_key] = signal.timestamp
                    continue
                self._quarantined_compute.pop(quarantine_key, None)
                if row is None:
                    continue
                dedupe_key = (row.factor_id, row.market_id, row.param)
                persistence_key = (*dedupe_key, row.ts)
                if persistence_key in self._quarantined_persistence:
                    continue
                if self._last_persisted_ts.get(dedupe_key) == row.ts:
                    continue
                try:
                    if factor_cls.definition_version == "1":
                        await self._persist_factor_value(row, signal)
                    else:
                        await self._persist_factor_value(
                            row,
                            signal,
                            definition_version=factor_cls.definition_version,
                        )
                except Exception as error:
                    # Dedupe key intentionally not advanced: the same row
                    # retries on the next cadence instead of being lost.
                    logger.exception(
                        "factor persist failed factor_id=%s market_id=%s ts=%s",
                        row.factor_id,
                        row.market_id,
                        row.ts.isoformat(),
                    )
                    if _is_transient_persistence_error(error):
                        failed_market_ids.add(signal.market_id)
                        logger.warning(
                            "transient factor persist failure; retrying "
                            "factor_id=%s market_id=%s",
                            row.factor_id,
                            signal.market_id,
                        )
                        continue
                    attempts = self._persistence_failures.get(persistence_key, 0) + 1
                    self._persistence_failures[persistence_key] = attempts
                    if attempts <= _MAX_PERSIST_RETRIES:
                        failed_market_ids.add(signal.market_id)
                    else:
                        self._quarantined_persistence.add(persistence_key)
                        logger.error(
                            "factor persist quarantined after %d attempts "
                            "factor_id=%s market_id=%s",
                            attempts,
                            row.factor_id,
                            signal.market_id,
                        )
                    continue
                self._persistence_failures.pop(persistence_key, None)
                for failed_key in tuple(self._persistence_failures):
                    if failed_key[:3] == dedupe_key:
                        self._persistence_failures.pop(failed_key, None)
                for failed_key in tuple(self._quarantined_persistence):
                    if failed_key[:3] == dedupe_key:
                        self._quarantined_persistence.discard(failed_key)
                self._last_persisted_ts[dedupe_key] = row.ts
                persisted += 1
        self._last_failed_market_ids = failed_market_ids
        return persisted

    async def _persist_factor_value(
        self,
        row: FactorValueRow,
        signal: MarketSignal,
        *,
        definition_version: str = "1",
    ) -> None:
        try:
            await persist_factor_value(
                self.pool,
                row,
                definition_version=definition_version,
            )
        except asyncpg.ForeignKeyViolationError as exc:
            if _is_constraint_violation(exc, "factor_values_factor_id_fkey"):
                await ensure_factor_catalog(self.pool, factor_ids=(row.factor_id,))
            elif _is_constraint_violation(exc, "factor_values_market_id_fkey"):
                self._known_market_ids.discard(signal.market_id)
                await self._ensure_market_shell(signal)
            else:
                raise
            await persist_factor_value(
                self.pool,
                row,
                definition_version=definition_version,
            )

    async def get_panel(
        self,
        factor_id: str,
        param: str | Mapping[str, Any] | None,
        market_ids: Sequence[str],
        ts_start: datetime,
        ts_end: datetime,
    ) -> FactorPanel:
        return await load_factor_panel(
            self.pool,
            factor_id=factor_id,
            param=param,
            market_ids=market_ids,
            ts_start=ts_start,
            ts_end=ts_end,
        )

    async def _forward_signals(self) -> None:
        try:
            async for signal in self.signal_stream:
                await self._signal_queue.put(signal)
        finally:
            self._stream_exhausted = True
            await self._signal_queue.put(None)

    def _drain_signal_buffer(self) -> list[MarketSignal]:
        signals: list[MarketSignal] = []
        while True:
            try:
                item = self._signal_queue.get_nowait()
            except asyncio.QueueEmpty:
                return signals
            # None is the terminal FIFO sentinel from _forward_signals().
            # Producers must never enqueue more items after it, or they will
            # be orphaned once the service marks the stream exhausted here.
            if item is None:
                self._stream_exhausted = True
                return signals
            signals.append(item)

    def _raise_stream_error(self, stream_task: asyncio.Task[None]) -> None:
        if not stream_task.done():
            return
        exception = stream_task.exception()
        if exception is not None:
            raise exception

    async def _ensure_market_shell(self, signal: MarketSignal) -> None:
        if signal.market_id in self._known_market_ids:
            return
        if await self.store.read_market(signal.market_id) is not None:
            self._known_market_ids.add(signal.market_id)
            return
        await self.store.write_market(
            Market(
                condition_id=signal.market_id,
                slug=signal.market_id,
                question=signal.title,
                venue=signal.venue,
                resolves_at=signal.resolves_at,
                created_at=signal.timestamp,
                last_seen_at=signal.timestamp,
            )
        )
        self._known_market_ids.add(signal.market_id)


def _validate_factor_value(row: FactorValueRow) -> None:
    value = validate_finite(row.value, field_name=f"factor {row.factor_id} value")
    if row.factor_id == "metaculus_prior":
        validate_probability(value, field_name="metaculus_prior")
    elif row.factor_id in {"yes_count", "no_count"}:
        validate_quantity(value, field_name=row.factor_id)


def _is_constraint_violation(
    exc: asyncpg.ForeignKeyViolationError,
    constraint_name: str,
) -> bool:
    actual_constraint = getattr(exc, "constraint_name", None)
    return actual_constraint == constraint_name or constraint_name in str(exc)


def _is_transient_persistence_error(error: Exception) -> bool:
    transient_types = (
        asyncpg.CannotConnectNowError,
        asyncpg.ConnectionDoesNotExistError,
        asyncpg.InterfaceError,
        asyncpg.TooManyConnectionsError,
        asyncpg.PostgresConnectionError,
        asyncio.TimeoutError,
        TimeoutError,
        OSError,
    )
    return isinstance(error, transient_types)


def _requires_open_yes_price(factor_cls: type[FactorDefinition]) -> bool:
    return "yes_price" in factor_cls.required_inputs


def _has_open_yes_price(signal: MarketSignal) -> bool:
    yes_price = signal.yes_price
    return 0.0 < yes_price < 1.0 and yes_price == yes_price
