from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from pms.config import ControllerSettings, PMSSettings
from pms.core.models import BookLevel, BookSnapshot
from pms.runner import _controller_direct_book_reader
from pms.core.enums import RunMode
from pms.sensor.adapters.direct_book import (
    PolymarketPublicBookClient,
    RefreshingDirectBookSnapshotReader,
    VenueBook,
    fee_rate_bps_from_clob_payload,
    venue_book_from_clob_payload,
)
from pms.storage.market_data_store import PostgresMarketDataStore


@dataclass
class FakePrimaryBookReader:
    snapshot: BookSnapshot | None
    levels: list[BookLevel] = field(default_factory=list)
    snapshot_calls: list[tuple[str, str]] = field(default_factory=list)
    level_calls: list[int] = field(default_factory=list)

    async def read_latest_snapshot(
        self,
        market_id: str,
        token_id: str,
    ) -> BookSnapshot | None:
        self.snapshot_calls.append((market_id, token_id))
        return self.snapshot

    async def read_levels_for_snapshot(self, snapshot_id: int) -> list[BookLevel]:
        self.level_calls.append(snapshot_id)
        return list(self.levels)


@dataclass
class FakeVenueBookClient:
    book: VenueBook
    calls: list[tuple[str, str]] = field(default_factory=list)
    fee_rate_bps: float | None = None
    fee_calls: list[tuple[str, str]] = field(default_factory=list)

    async def read_order_book(self, market_id: str, token_id: str) -> VenueBook:
        self.calls.append((market_id, token_id))
        return self.book

    async def read_fee_rate_bps(self, market_id: str, token_id: str) -> float | None:
        self.fee_calls.append((market_id, token_id))
        return self.fee_rate_bps


@pytest.mark.asyncio
async def test_refreshing_direct_book_reader_uses_fresh_primary_snapshot() -> None:
    now = datetime(2026, 6, 1, 22, 0, tzinfo=UTC)
    primary = FakePrimaryBookReader(
        snapshot=BookSnapshot(
            id=42,
            market_id="market-1",
            token_id="token-yes",
            ts=now - timedelta(milliseconds=250),
            hash="pg-hash",
            source="subscribe",
        ),
        levels=[
            BookLevel(42, "market-1", "BUY", 0.49, 10.0),
            BookLevel(42, "market-1", "SELL", 0.51, 10.0),
        ],
    )
    venue = FakeVenueBookClient(
        VenueBook(
            market_id="market-1",
            token_id="token-yes",
            ts=now,
            hash="venue-hash",
            bids=(BookLevel(0, "market-1", "BUY", 0.50, 10.0),),
            asks=(BookLevel(0, "market-1", "SELL", 0.52, 10.0),),
        )
    )
    reader = RefreshingDirectBookSnapshotReader(
        primary=primary,
        venue_client=venue,
        max_snapshot_age_ms=1_000.0,
        clock=lambda: now,
    )

    snapshot = await reader.read_latest_snapshot("market-1", "token-yes")
    levels = await reader.read_levels_for_snapshot(42)

    assert snapshot == primary.snapshot
    assert levels == primary.levels
    assert venue.calls == []
    assert primary.level_calls == [42]


@pytest.mark.asyncio
async def test_refreshing_direct_book_reader_refreshes_stale_primary_snapshot() -> None:
    now = datetime(2026, 6, 1, 22, 0, tzinfo=UTC)
    primary = FakePrimaryBookReader(
        snapshot=BookSnapshot(
            id=42,
            market_id="market-1",
            token_id="token-no",
            ts=now - timedelta(seconds=30),
            hash="stale-pg-hash",
            source="subscribe",
        ),
        levels=[
            BookLevel(42, "market-1", "BUY", 0.40, 10.0),
            BookLevel(42, "market-1", "SELL", 0.42, 10.0),
        ],
    )
    venue = FakeVenueBookClient(
        VenueBook(
            market_id="market-1",
            token_id="token-no",
            ts=now,
            hash="venue-hash",
            bids=(BookLevel(0, "market-1", "BUY", 0.43, 20.0),),
            asks=(BookLevel(0, "market-1", "SELL", 0.44, 30.0),),
        )
    )
    reader = RefreshingDirectBookSnapshotReader(
        primary=primary,
        venue_client=venue,
        max_snapshot_age_ms=1_000.0,
        clock=lambda: now,
    )

    snapshot = await reader.read_latest_snapshot("market-1", "token-no")
    assert snapshot is not None
    levels = await reader.read_levels_for_snapshot(snapshot.id)

    assert snapshot.id < 0
    assert snapshot.source == "venue_direct"
    assert snapshot.ts == now
    assert snapshot.hash == "venue-hash"
    assert levels == [
        BookLevel(snapshot.id, "market-1", "BUY", 0.43, 20.0),
        BookLevel(snapshot.id, "market-1", "SELL", 0.44, 30.0),
    ]
    assert venue.calls == [("market-1", "token-no")]
    assert primary.level_calls == []


@pytest.mark.asyncio
async def test_refreshing_direct_book_reader_evicts_synthetic_levels_after_read() -> None:
    now = datetime(2026, 6, 1, 22, 0, tzinfo=UTC)
    primary = FakePrimaryBookReader(
        snapshot=None,
        levels=[BookLevel(-1, "market-1", "BUY", 0.41, 5.0)],
    )
    venue = FakeVenueBookClient(
        VenueBook(
            market_id="market-1",
            token_id="token-no",
            ts=now,
            hash="venue-hash",
            bids=(BookLevel(0, "market-1", "BUY", 0.43, 20.0),),
            asks=(BookLevel(0, "market-1", "SELL", 0.44, 30.0),),
        )
    )
    reader = RefreshingDirectBookSnapshotReader(
        primary=primary,
        venue_client=venue,
        max_snapshot_age_ms=1_000.0,
        clock=lambda: now,
    )

    snapshot = await reader.read_latest_snapshot("market-1", "token-no")
    assert snapshot is not None
    first_levels = await reader.read_levels_for_snapshot(snapshot.id)
    second_levels = await reader.read_levels_for_snapshot(snapshot.id)

    assert first_levels == [
        BookLevel(snapshot.id, "market-1", "BUY", 0.43, 20.0),
        BookLevel(snapshot.id, "market-1", "SELL", 0.44, 30.0),
    ]
    assert second_levels == [BookLevel(-1, "market-1", "BUY", 0.41, 5.0)]
    assert primary.level_calls == [snapshot.id]


@pytest.mark.asyncio
async def test_refreshing_direct_book_reader_reads_venue_fee_rate_bps() -> None:
    now = datetime(2026, 6, 1, 22, 0, tzinfo=UTC)
    primary = FakePrimaryBookReader(snapshot=None)
    venue = FakeVenueBookClient(
        VenueBook(
            market_id="market-1",
            token_id="token-no",
            ts=now,
            hash="venue-hash",
            bids=(BookLevel(0, "market-1", "BUY", 0.43, 20.0),),
            asks=(BookLevel(0, "market-1", "SELL", 0.44, 30.0),),
        ),
        fee_rate_bps=300.0,
    )
    reader = RefreshingDirectBookSnapshotReader(
        primary=primary,
        venue_client=venue,
        max_snapshot_age_ms=1_000.0,
        clock=lambda: now,
    )

    fee_rate_bps = await reader.read_fee_rate_bps("market-1", "token-no")

    assert fee_rate_bps == pytest.approx(300.0)
    assert venue.fee_calls == [("market-1", "token-no")]


def test_venue_book_from_clob_payload_parses_public_book_response() -> None:
    observed_at = datetime(2026, 6, 1, 22, 0, tzinfo=UTC)

    book = venue_book_from_clob_payload(
        {
            "asset_id": "token-yes",
            "hash": "book-hash",
            "bids": [
                {"price": "0.01", "size": "1.0"},
                {"price": "0.49", "size": "10.5"},
                {"price": "0.25", "size": "2.0"},
            ],
            "asks": [
                {"price": "0.99", "size": "1.0"},
                {"price": "0.51", "size": "9.5"},
                {"price": "0.75", "size": "2.0"},
            ],
        },
        market_id="market-1",
        token_id="token-yes",
        observed_at=observed_at,
    )

    assert book.market_id == "market-1"
    assert book.token_id == "token-yes"
    assert book.ts == observed_at
    assert book.hash == "book-hash"
    assert book.bids == (
        BookLevel(0, "market-1", "BUY", 0.49, 10.5),
        BookLevel(0, "market-1", "BUY", 0.25, 2.0),
        BookLevel(0, "market-1", "BUY", 0.01, 1.0),
    )
    assert book.asks == (
        BookLevel(0, "market-1", "SELL", 0.51, 9.5),
        BookLevel(0, "market-1", "SELL", 0.75, 2.0),
        BookLevel(0, "market-1", "SELL", 0.99, 1.0),
    )


def test_fee_rate_bps_from_clob_payload_accepts_base_fee_response() -> None:
    assert fee_rate_bps_from_clob_payload({"base_fee": 1000}) == pytest.approx(
        1000.0
    )


def test_venue_book_from_clob_payload_rejects_token_mismatch() -> None:
    with pytest.raises(ValueError, match="asset_id"):
        venue_book_from_clob_payload(
            {"asset_id": "other-token", "bids": [], "asks": []},
            market_id="market-1",
            token_id="token-yes",
            observed_at=datetime(2026, 6, 1, 22, 0, tzinfo=UTC),
        )


def test_runner_direct_book_reader_wraps_store_when_refresh_enabled() -> None:
    store = cast(PostgresMarketDataStore, object())
    settings = PMSSettings(
        mode=RunMode.PAPER,
        controller=ControllerSettings(
            max_book_age_ms=15_000.0,
            venue_book_refresh_enabled=True,
            venue_book_refresh_timeout_s=3.0,
        )
    )

    reader = _controller_direct_book_reader(settings, store)

    assert isinstance(reader, RefreshingDirectBookSnapshotReader)
    assert reader.primary is store
    assert reader.max_snapshot_age_ms == pytest.approx(15_000.0)
    assert isinstance(reader.venue_client, PolymarketPublicBookClient)
    assert reader.venue_client.timeout_s == pytest.approx(3.0)


def test_runner_direct_book_reader_keeps_plain_store_by_default() -> None:
    store = cast(PostgresMarketDataStore, object())

    reader = _controller_direct_book_reader(PMSSettings(mode=RunMode.PAPER), store)

    assert reader is store
