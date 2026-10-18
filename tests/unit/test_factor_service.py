from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
import logging
from typing import Any, cast

import pytest

from pms.core.enums import MarketStatus
from pms.core.models import Market, MarketSignal
from pms.factors.base import FactorDefinition, FactorValueRow
from pms.factors.definitions import FavoriteLongshotBias, OrderbookImbalance
from pms.factors.service import FactorService, persist_factor_value
from pms.sensor.stream import SensorStream


def _signal(
    *,
    market_id: str = "factor-service-market",
    yes_price: float = 0.4,
) -> MarketSignal:
    return MarketSignal(
        market_id=market_id,
        token_id="yes-token",
        venue="polymarket",
        title="Will FactorService persist factors?",
        yes_price=yes_price,
        volume_24h=1000.0,
        resolves_at=datetime(2026, 4, 30, tzinfo=UTC),
        orderbook={
            "bids": [{"price": 0.39, "size": 100.0}],
            "asks": [{"price": 0.41, "size": 50.0}],
        },
        external_signal={"metaculus_prob": 0.7},
        fetched_at=datetime(2026, 4, 18, tzinfo=UTC),
        market_status=MarketStatus.OPEN.value,
    )


class SequenceSignalStream:
    def __init__(self, signals: list[MarketSignal]) -> None:
        self._signals = list(signals)

    def __aiter__(self) -> AsyncIterator[MarketSignal]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MarketSignal]:
        for signal in self._signals:
            yield signal


class RepeatingSignalStream:
    def __init__(self, signal: MarketSignal) -> None:
        self.signal = signal

    def __aiter__(self) -> AsyncIterator[MarketSignal]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MarketSignal]:
        while True:
            yield self.signal
            await asyncio.sleep(0)


class FakeStore:
    def __init__(self) -> None:
        self.markets: dict[str, Market] = {}

    async def read_market(self, market_id: str) -> Market | None:
        return self.markets.get(market_id)

    async def write_market(self, market: Market) -> None:
        self.markets[market.condition_id] = market


class PersistedFactor(FactorDefinition):
    factor_id = "persisted_factor"
    required_inputs = ("yes_price",)

    def compute(
        self,
        signal: MarketSignal,
        outer_ring: object,
    ) -> FactorValueRow | None:
        del outer_ring
        return FactorValueRow(
            factor_id=self.factor_id,
            param="",
            market_id=signal.market_id,
            ts=signal.timestamp,
            value=signal.yes_price,
        )


class MissingFactor(FactorDefinition):
    factor_id = "missing_factor"
    required_inputs = ()

    def compute(
        self,
        signal: MarketSignal,
        outer_ring: object,
    ) -> FactorValueRow | None:
        del signal, outer_ring
        return None


class OrderbookOnlyFactor(FactorDefinition):
    factor_id = "orderbook_only_factor"
    required_inputs = ("orderbook",)

    def compute(
        self,
        signal: MarketSignal,
        outer_ring: object,
    ) -> FactorValueRow | None:
        del outer_ring
        return FactorValueRow(
            factor_id=self.factor_id,
            param="",
            market_id=signal.market_id,
            ts=signal.timestamp,
            value=float(len(signal.orderbook.get("bids", []))),
        )


class ExplodingFactor(FactorDefinition):
    factor_id = "exploding_factor"
    required_inputs = ("yes_price",)

    def compute(
        self,
        signal: MarketSignal,
        outer_ring: object,
    ) -> FactorValueRow | None:
        del signal, outer_ring
        raise ValueError("factor exploded")


@pytest.mark.asyncio
async def test_persist_factor_value_rejects_non_finite_before_database_access() -> None:
    row = FactorValueRow(
        factor_id="persisted_factor",
        param="",
        market_id="factor-service-market",
        ts=datetime(2026, 4, 18, tzinfo=UTC),
        value=float("nan"),
    )

    with pytest.raises(ValueError, match="finite"):
        await persist_factor_value(cast(Any, object()), row)


@pytest.mark.asyncio
async def test_factor_service_compute_once_persists_non_none_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted: list[FactorValueRow] = []

    async def fake_persist(pool: object, row: FactorValueRow, *, definition_version: str = "1") -> None:
        del pool, definition_version
        persisted.append(row)

    monkeypatch.setattr("pms.factors.service.persist_factor_value", fake_persist)
    store = FakeStore()
    service = FactorService(
        pool=cast(Any, object()),
        store=cast(Any, store),
        cadence_s=0.1,
        factors=(PersistedFactor, MissingFactor),
        signal_stream=SequenceSignalStream([]),
    )

    count = await service.compute_once([_signal()])

    assert count == 1
    assert persisted == [
        FactorValueRow(
            factor_id="persisted_factor",
            param="",
            market_id="factor-service-market",
            ts=datetime(2026, 4, 18, tzinfo=UTC),
            value=0.4,
        )
    ]
    assert "factor-service-market" in store.markets


@pytest.mark.asyncio
async def test_factor_service_persists_neutral_row_after_tail_signal() -> None:
    persisted: list[FactorValueRow] = []

    async def fake_persist(
        pool: object,
        row: FactorValueRow,
        *,
        definition_version: str = "1",
    ) -> None:
        del pool, definition_version
        persisted.append(row)

    # The service writes rows rather than relying on a missing-row sentinel;
    # the second signal must supersede the earlier tail value.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("pms.factors.service.persist_factor_value", fake_persist)
    try:
        service = FactorService(
            pool=cast(Any, object()),
            store=cast(Any, FakeStore()),
            cadence_s=0.1,
            factors=(FavoriteLongshotBias,),
            signal_stream=SequenceSignalStream([]),
        )
        tail = _signal(market_id="neutral-market", yes_price=0.05)
        neutral = replace(
            _signal(market_id="neutral-market", yes_price=0.50),
            fetched_at=tail.fetched_at.replace(second=1),
        )
        await service.compute_once([tail])
        await service.compute_once([neutral])
    finally:
        monkeypatch.undo()

    assert [row.value for row in persisted] == [pytest.approx(-0.05), pytest.approx(0.0)]


@pytest.mark.asyncio
async def test_factor_service_compute_once_continues_after_factor_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    persisted: list[FactorValueRow] = []

    async def fake_persist(pool: object, row: FactorValueRow, *, definition_version: str = "1") -> None:
        del pool, definition_version
        persisted.append(row)

    monkeypatch.setattr("pms.factors.service.persist_factor_value", fake_persist)
    service = FactorService(
        pool=cast(Any, object()),
        store=cast(Any, FakeStore()),
        cadence_s=0.1,
        factors=(ExplodingFactor, PersistedFactor),
        signal_stream=SequenceSignalStream([]),
    )

    with caplog.at_level(logging.ERROR, logger="pms.factors.service"):
        count = await service.compute_once([_signal(market_id="survives-factor-error")])

    assert count == 1
    assert [row.market_id for row in persisted] == ["survives-factor-error"]
    assert "factor compute failed" in caplog.text
    assert "exploding_factor" in caplog.text


@pytest.mark.asyncio
async def test_factor_service_quarantines_deterministic_compute_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    original_compute = ExplodingFactor.compute

    def counting_compute(
        factor: ExplodingFactor,
        signal: MarketSignal,
        outer_ring: object,
    ) -> FactorValueRow | None:
        nonlocal calls
        calls += 1
        return original_compute(factor, signal, outer_ring)

    monkeypatch.setattr(ExplodingFactor, "compute", counting_compute)
    service = FactorService(
        pool=cast(Any, object()),
        store=cast(Any, FakeStore()),
        cadence_s=0.1,
        factors=(ExplodingFactor,),
        signal_stream=SequenceSignalStream([]),
    )
    signal = _signal(market_id="quarantine-market")

    await service.compute_once([signal])
    await service.compute_once([signal])

    assert calls == 1


@pytest.mark.asyncio
async def test_factor_service_skips_yes_price_factors_for_sentinel_book_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted: list[FactorValueRow] = []

    async def fake_persist(pool: object, row: FactorValueRow, *, definition_version: str = "1") -> None:
        del pool, definition_version
        persisted.append(row)

    monkeypatch.setattr("pms.factors.service.persist_factor_value", fake_persist)
    service = FactorService(
        pool=cast(Any, object()),
        store=cast(Any, FakeStore()),
        cadence_s=0.1,
        factors=(PersistedFactor, OrderbookOnlyFactor),
        signal_stream=SequenceSignalStream([]),
    )

    count = await service.compute_once(
        [_signal(market_id="sentinel-book-signal", yes_price=0.0)]
    )

    assert count == 1
    assert [(row.factor_id, row.market_id) for row in persisted] == [
        ("orderbook_only_factor", "sentinel-book-signal")
    ]


@pytest.mark.asyncio
async def test_factor_service_run_exits_when_signal_stream_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted: list[FactorValueRow] = []

    async def fake_persist(pool: object, row: FactorValueRow, *, definition_version: str = "1") -> None:
        del pool, definition_version
        persisted.append(row)

    monkeypatch.setattr("pms.factors.service.persist_factor_value", fake_persist)
    service = FactorService(
        pool=cast(Any, object()),
        store=cast(Any, FakeStore()),
        cadence_s=0.01,
        factors=(PersistedFactor,),
        signal_stream=SequenceSignalStream([_signal(market_id="run-exit-market")]),
    )

    await asyncio.wait_for(service.run(), timeout=1.0)

    assert [row.market_id for row in persisted] == ["run-exit-market"]


@pytest.mark.asyncio
async def test_factor_service_skips_repersisting_the_same_factor_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted: list[FactorValueRow] = []

    async def fake_persist(pool: object, row: FactorValueRow, *, definition_version: str = "1") -> None:
        del pool, definition_version
        persisted.append(row)

    monkeypatch.setattr("pms.factors.service.persist_factor_value", fake_persist)
    service = FactorService(
        pool=cast(Any, object()),
        store=cast(Any, FakeStore()),
        cadence_s=0.1,
        factors=(PersistedFactor,),
        signal_stream=SequenceSignalStream([]),
    )
    signal = _signal()

    first = await service.compute_once([signal])
    second = await service.compute_once([signal])

    assert first == 1
    assert second == 0
    assert persisted == [
        FactorValueRow(
            factor_id="persisted_factor",
            param="",
            market_id="factor-service-market",
            ts=datetime(2026, 4, 18, tzinfo=UTC),
            value=0.4,
        )
    ]


@pytest.mark.asyncio
async def test_sensor_stream_subscription_receives_signals_without_consuming_main_queue() -> None:
    class OneShotSensor:
        def __aiter__(self) -> AsyncIterator[MarketSignal]:
            return self._iterate()

        async def _iterate(self) -> AsyncIterator[MarketSignal]:
            yield _signal(market_id="tee-market")

    stream = SensorStream()
    subscription = stream.subscribe()

    await stream.start([OneShotSensor()])

    main_signal = await asyncio.wait_for(stream.queue.get(), timeout=1.0)
    tee_signal = await asyncio.wait_for(anext(subscription), timeout=1.0)

    stream.queue.task_done()
    await asyncio.wait_for(stream.stop(), timeout=5.0)

    assert main_signal == tee_signal
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(subscription), timeout=1.0)


@pytest.mark.asyncio
async def test_sensor_stream_closes_late_subscriptions_after_consumers_finish() -> None:
    class OneShotSensor:
        def __aiter__(self) -> AsyncIterator[MarketSignal]:
            return self._iterate()

        async def _iterate(self) -> AsyncIterator[MarketSignal]:
            yield _signal(market_id="late-subscription-market")

    stream = SensorStream()
    await stream.start([OneShotSensor()])
    signal = await asyncio.wait_for(stream.queue.get(), timeout=1.0)
    stream.queue.task_done()
    await asyncio.gather(*stream.tasks, return_exceptions=True)

    late_subscription = stream.subscribe()

    assert signal.market_id == "late-subscription-market"
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(late_subscription), timeout=1.0)
    await asyncio.wait_for(stream.stop(), timeout=5.0)


@pytest.mark.asyncio
async def test_compute_once_survives_transient_persist_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One transient Postgres error must not kill the factor task.

    The dedupe key must not advance either, so the same row retries on
    the next cadence instead of being silently lost.
    """
    import asyncpg

    persisted: list[FactorValueRow] = []
    calls = 0

    async def flaky_persist(pool: Any, row: FactorValueRow, *, definition_version: str = "1") -> None:
        nonlocal calls
        del pool, definition_version
        calls += 1
        if calls == 1:
            raise asyncpg.PostgresError("transient db error")
        persisted.append(row)

    monkeypatch.setattr("pms.factors.service.persist_factor_value", flaky_persist)
    service = FactorService(
        pool=cast(Any, object()),
        store=cast(Any, FakeStore()),
        cadence_s=0.1,
        factors=(PersistedFactor,),
        signal_stream=SequenceSignalStream([]),
    )
    signal = _signal(market_id="persist-retry-market")

    with caplog.at_level(logging.ERROR, logger="pms.factors.service"):
        first = await service.compute_once([signal])
    second = await service.compute_once([signal])

    assert first == 0
    assert second == 1
    assert [row.market_id for row in persisted] == ["persist-retry-market"]


@pytest.mark.asyncio
async def test_factor_service_retries_failed_dirty_market_on_next_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    persisted = asyncio.Event()

    async def flaky_persist(
        pool: Any,
        row: FactorValueRow,
        *,
        definition_version: str = "1",
    ) -> None:
        nonlocal calls
        del pool, row, definition_version
        calls += 1
        if calls > 1:
            persisted.set()
        if calls == 1:
            raise RuntimeError("transient factor persistence failure")

    monkeypatch.setattr("pms.factors.service.persist_factor_value", flaky_persist)
    service = FactorService(
        pool=cast(Any, object()),
        store=cast(Any, FakeStore()),
        cadence_s=0.001,
        factors=(PersistedFactor,),
        signal_stream=RepeatingSignalStream(_signal(market_id="retry-market")),
    )
    task = asyncio.create_task(service.run())
    try:
        await asyncio.wait_for(persisted.wait(), timeout=1.0)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert calls >= 2


@pytest.mark.asyncio
async def test_factor_service_quarantines_persistence_after_bounded_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def always_failing_persist(
        pool: Any,
        row: FactorValueRow,
        *,
        definition_version: str = "1",
    ) -> None:
        nonlocal calls
        del pool, row, definition_version
        calls += 1
        raise RuntimeError("persistent database failure")

    monkeypatch.setattr(
        "pms.factors.service.persist_factor_value",
        always_failing_persist,
    )
    service = FactorService(
        pool=cast(Any, object()),
        store=cast(Any, FakeStore()),
        cadence_s=0.1,
        factors=(PersistedFactor,),
        signal_stream=SequenceSignalStream([]),
    )
    signal = _signal(market_id="persistence-quarantine-market")

    for _ in range(6):
        await service.compute_once([signal])

    assert calls == 4  # initial attempt plus three bounded retries


@pytest.mark.asyncio
async def test_factor_service_retries_connection_outage_then_persists_new_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncpg

    calls = 0
    persisted: list[FactorValueRow] = []

    async def outage_then_recovery(
        pool: Any,
        row: FactorValueRow,
        *,
        definition_version: str = "1",
    ) -> None:
        nonlocal calls
        del pool, definition_version
        calls += 1
        if calls <= 5:
            raise asyncpg.CannotConnectNowError("database outage")
        persisted.append(row)

    monkeypatch.setattr(
        "pms.factors.service.persist_factor_value",
        outage_then_recovery,
    )
    service = FactorService(
        pool=cast(Any, object()),
        store=cast(Any, FakeStore()),
        cadence_s=0.1,
        factors=(PersistedFactor,),
        signal_stream=SequenceSignalStream([]),
    )
    first_signal = _signal(market_id="outage-recovery-market")
    healthy_signal = replace(
        first_signal,
        fetched_at=first_signal.fetched_at.replace(second=1),
    )

    for _ in range(5):
        await service.compute_once([first_signal])
    assert await service.compute_once([healthy_signal]) == 1
    assert persisted == [
        FactorValueRow(
            factor_id="persisted_factor",
            param="",
            market_id=healthy_signal.market_id,
            ts=healthy_signal.timestamp,
            value=healthy_signal.yes_price,
        )
    ]


@pytest.mark.asyncio
async def test_factor_service_finite_stream_keeps_transient_failure_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncpg

    calls = 0
    persisted = asyncio.Event()

    async def outage_then_recovery(
        pool: Any,
        row: FactorValueRow,
        *,
        definition_version: str = "1",
    ) -> None:
        nonlocal calls
        del pool, definition_version
        calls += 1
        if calls <= 2:
            raise asyncpg.ConnectionDoesNotExistError("database outage")
        del row
        persisted.set()

    monkeypatch.setattr(
        "pms.factors.service.persist_factor_value",
        outage_then_recovery,
    )
    service = FactorService(
        pool=cast(Any, object()),
        store=cast(Any, FakeStore()),
        cadence_s=0.001,
        factors=(PersistedFactor,),
        signal_stream=SequenceSignalStream([_signal(market_id="finite-outage")]),
    )
    task = asyncio.create_task(service.run())
    await asyncio.wait_for(persisted.wait(), timeout=1.0)
    await asyncio.wait_for(task, timeout=1.0)
    assert calls == 3


@pytest.mark.asyncio
async def test_factor_service_quarantine_is_scoped_to_failed_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted: list[FactorValueRow] = []

    async def reject_first_observation(
        pool: Any,
        row: FactorValueRow,
        *,
        definition_version: str = "1",
    ) -> None:
        del pool, definition_version
        if row.ts == first_signal.timestamp:
            raise ValueError("deterministic integrity failure")
        persisted.append(row)

    monkeypatch.setattr(
        "pms.factors.service.persist_factor_value",
        reject_first_observation,
    )
    service = FactorService(
        pool=cast(Any, object()),
        store=cast(Any, FakeStore()),
        cadence_s=0.1,
        factors=(PersistedFactor,),
        signal_stream=SequenceSignalStream([]),
    )
    first_signal = _signal(market_id="row-scoped-quarantine-market")
    healthy_signal = replace(
        first_signal,
        fetched_at=first_signal.fetched_at.replace(second=1),
    )

    for _ in range(5):
        await service.compute_once([first_signal])
    assert await service.compute_once([healthy_signal]) == 1
    assert persisted[0].ts == healthy_signal.timestamp


@pytest.mark.asyncio
async def test_factor_service_recovers_market_shell_after_fk_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncpg

    calls = 0
    persisted: list[FactorValueRow] = []

    async def flaky_persist(
        pool: Any,
        row: FactorValueRow,
        *,
        definition_version: str = "1",
    ) -> None:
        nonlocal calls
        del pool, definition_version
        calls += 1
        if calls == 1:
            raise asyncpg.ForeignKeyViolationError("factor_values_market_id_fkey")
        persisted.append(row)

    monkeypatch.setattr("pms.factors.service.persist_factor_value", flaky_persist)
    store = FakeStore()
    service = FactorService(
        pool=cast(Any, object()),
        store=cast(Any, store),
        cadence_s=0.1,
        factors=(PersistedFactor,),
        signal_stream=SequenceSignalStream([]),
    )
    signal = _signal(market_id="shell-retry-market")
    service._known_market_ids.add(signal.market_id)  # noqa: SLF001
    row = FactorValueRow(
        factor_id="persisted_factor",
        param="",
        market_id=signal.market_id,
        ts=signal.timestamp,
        value=signal.yes_price,
    )

    await service._persist_factor_value(row, signal)  # noqa: SLF001

    assert calls == 2
    assert persisted == [row]
    assert signal.market_id in store.markets


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", ["tokens", "outcome"])
async def test_factor_service_no_quote_persists_yes_coordinate(
    monkeypatch: pytest.MonkeyPatch, identity: str,
) -> None:
    persisted: dict[str, tuple[float, str]] = {}

    async def capture(pool: object, row: FactorValueRow, *, definition_version: str = "1") -> None:
        persisted[row.factor_id] = (row.value, definition_version)

    monkeypatch.setattr("pms.factors.service.persist_factor_value", capture)
    signal = replace(
        _signal(yes_price=0.001), token_id="no-token",
        external_signal=({"yes_token_id": "yes-token", "no_token_id": "no-token"}
                         if identity == "tokens" else {"signal_token_outcome": "NO"}),
        orderbook={"bids": [], "asks": [{"price": 0.001, "size": 100.0}]},
    )
    service = FactorService(
        pool=cast(Any, object()), store=cast(Any, FakeStore()), cadence_s=0.1,
        factors=(FavoriteLongshotBias, OrderbookImbalance),
        signal_stream=SequenceSignalStream([]),
    )
    assert await service.compute_once([signal]) == 2
    assert persisted["favorite_longshot_bias"][0] == pytest.approx(0.099)
    assert persisted["orderbook_imbalance"][0] == pytest.approx(1.0)
    assert {version for _, version in persisted.values()} == {"2"}
    assert signal.yes_price == 0.001
    assert signal.orderbook["bids"] == []
