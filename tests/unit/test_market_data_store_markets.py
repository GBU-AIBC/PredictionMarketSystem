from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest

from pms.core.models import Market
from pms.storage.market_data_store import MarketFilters, PostgresMarketDataStore


@dataclass
class FakeConnection:
    fetch_results: list[list[dict[str, object]]] = field(default_factory=list)
    fetchrow_results: list[dict[str, object] | None] = field(default_factory=list)
    fetch_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    fetchrow_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    execute_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        self.fetch_calls.append((query, args))
        if not self.fetch_results:
            return []
        return self.fetch_results.pop(0)

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        self.fetchrow_calls.append((query, args))
        if not self.fetchrow_results:
            return None
        return self.fetchrow_results.pop(0)

    async def execute(self, query: str, *args: object) -> str:
        self.execute_calls.append((query, args))
        return "INSERT 0 1"


class FakeAcquire:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> FakeConnection:
        return self._connection

    async def __aexit__(self, *_: object) -> None:
        return None


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self._connection)


REFERENCE_NOW = datetime(2026, 4, 24, 12, 0, tzinfo=UTC)


def _row(
    *,
    market_id: str,
    question: str,
    venue: str = "polymarket",
    volume_24h: float | None = 1500.0,
    updated_at: datetime,
    resolves_at: datetime | None = None,
    yes_token_id: str | None,
    no_token_id: str | None,
    total_count: int,
    yes_price: float | None = None,
    no_price: float | None = None,
    best_bid: float | None = None,
    best_ask: float | None = None,
    last_trade_price: float | None = None,
    liquidity: float | None = None,
    spread_bps: int | None = None,
    price_updated_at: datetime | None = None,
    subscription_source: str | None = None,
) -> dict[str, object]:
    return {
        "market_id": market_id,
        "question": question,
        "venue": venue,
        "volume_24h": volume_24h,
        "updated_at": updated_at,
        "resolves_at": resolves_at,
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "yes_price": yes_price,
        "no_price": no_price,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "last_trade_price": last_trade_price,
        "liquidity": liquidity,
        "spread_bps": spread_bps,
        "price_updated_at": price_updated_at,
        "subscription_source": subscription_source,
        "total_count": total_count,
    }


async def _read_markets_query(
    *,
    filters: MarketFilters | None = None,
    current_asset_ids: frozenset[str] = frozenset(),
) -> tuple[str, tuple[object, ...]]:
    connection = FakeConnection(fetch_results=[[]])
    store = PostgresMarketDataStore(FakePool(connection))

    await store.read_markets(
        limit=20,
        offset=0,
        filters=filters,
        current_asset_ids=current_asset_ids,
        now=REFERENCE_NOW,
    )

    return connection.fetch_calls[0]


@pytest.mark.asyncio
async def test_read_eligible_markets_prioritizes_volume_and_liquidity_in_sql() -> None:
    connection = FakeConnection(fetch_results=[[]])
    store = PostgresMarketDataStore(FakePool(connection))

    await store.read_eligible_markets(
        venue="polymarket",
        max_horizon_days=31,
        min_volume_usdc=100.0,
    )

    query, _ = connection.fetch_calls[0]
    assert "COALESCE(markets.volume_24h, 0) DESC" in query
    assert "COALESCE(markets.liquidity, 0) DESC" in query
    assert query.index("COALESCE(markets.volume_24h, 0) DESC") < query.index(
        "markets.condition_id ASC"
    )


@pytest.mark.asyncio
async def test_read_eligible_markets_excludes_closed_inactive_and_stale_rows() -> None:
    connection = FakeConnection(fetch_results=[[]])
    store = PostgresMarketDataStore(FakePool(connection))

    await store.read_eligible_markets(
        venue="polymarket",
        max_horizon_days=31,
        min_volume_usdc=100.0,
    )

    query, args = connection.fetch_calls[0]
    assert "COALESCE(markets.closed, false) = false" in query
    assert "COALESCE(markets.active, true) = true" in query
    assert "markets.last_seen_at >=" in query
    assert args[4] == pytest.approx(900.0)


@pytest.mark.asyncio
async def test_latest_book_summary_is_scoped_to_yes_token() -> None:
    timestamp = datetime(2026, 4, 24, 12, 0, tzinfo=UTC)
    connection = FakeConnection(
        fetch_results=[
            [
                {"ts": timestamp, "side": "BUY", "price": 0.49, "size": 10.0},
                {"ts": timestamp, "side": "SELL", "price": 0.51, "size": 10.0},
            ]
        ]
    )
    store = PostgresMarketDataStore(FakePool(connection))

    summary = await store.get_latest_book_summary("market-yes")

    assert summary is not None
    assert summary.best_bid == pytest.approx(0.49)
    query, args = connection.fetch_calls[0]
    assert "tokens.outcome = 'YES'" in query
    assert args == ("market-yes",)


@pytest.mark.asyncio
async def test_read_markets_returns_rows_total_and_filters_to_active_markets_in_sql() -> None:
    updated_at = datetime(2026, 4, 23, 9, 30, tzinfo=UTC)
    connection = FakeConnection(
        fetch_results=[
            [
                _row(
                    market_id="market-1",
                    question="Will market 1 resolve?",
                    updated_at=updated_at,
                    yes_token_id="market-1-yes",
                    no_token_id="market-1-no",
                    total_count=2,
                ),
                _row(
                    market_id="market-2",
                    question="Will market 2 resolve?",
                    updated_at=updated_at,
                    yes_token_id="market-2-yes",
                    no_token_id="market-2-no",
                    total_count=2,
                ),
            ]
        ]
    )
    store = PostgresMarketDataStore(FakePool(connection))

    rows, total = await store.read_markets(limit=20, offset=5)

    assert total == 2
    assert [row.market_id for row in rows] == ["market-1", "market-2"]
    assert rows[0].yes_token_id == "market-1-yes"
    assert rows[0].no_token_id == "market-1-no"
    assert rows[0].resolves_at is None
    assert len(connection.fetch_calls) == 1

    query, args = connection.fetch_calls[0]
    assert "markets.resolves_at" in query
    assert "resolves_at IS NULL OR markets.resolves_at > $1" in query
    assert "COUNT(*) OVER()" in query
    assert "market_subscriptions" in query
    assert args[10:12] == (20, 5)
    assert isinstance(args[0], datetime)


@pytest.mark.asyncio
async def test_read_markets_filter_volume_min() -> None:
    updated_at = datetime(2026, 4, 23, 9, 30, tzinfo=UTC)
    connection = FakeConnection(
        fetch_results=[
            [
                _row(
                    market_id="volume-match",
                    question="Will volume filter match?",
                    volume_24h=2500.0,
                    updated_at=updated_at,
                    yes_token_id="volume-match-yes",
                    no_token_id="volume-match-no",
                    total_count=1,
                )
            ]
        ]
    )
    store = PostgresMarketDataStore(FakePool(connection))

    rows, total = await store.read_markets(
        limit=20,
        offset=0,
        now=REFERENCE_NOW,
        filters=MarketFilters(volume_min=2000.0),
    )

    query, args = connection.fetch_calls[0]
    assert total == 1
    assert rows[0].market_id == "volume-match"
    assert "markets.volume_24h IS NOT NULL" in query
    assert "markets.volume_24h >= $3" in query
    assert args[2] == 2000.0


@pytest.mark.asyncio
async def test_read_markets_filter_liquidity_min() -> None:
    query, args = await _read_markets_query(filters=MarketFilters(liquidity_min=5000.0))

    assert "markets.liquidity IS NOT NULL" in query
    assert "markets.liquidity >= $4" in query
    assert args[3] == 5000.0


@pytest.mark.asyncio
async def test_read_markets_filter_spread_max_bps() -> None:
    query, args = await _read_markets_query(filters=MarketFilters(spread_max_bps=250))

    assert "markets.spread_bps IS NOT NULL" in query
    assert "markets.spread_bps <= $5" in query
    assert args[4] == 250


@pytest.mark.asyncio
async def test_read_markets_filter_yes_price_band() -> None:
    query, args = await _read_markets_query(filters=MarketFilters(yes_min=0.2, yes_max=0.8))

    assert "markets.yes_price IS NOT NULL" in query
    assert "markets.yes_price >= $6" in query
    assert "markets.yes_price <= $7" in query
    assert args[5:7] == (0.2, 0.8)


@pytest.mark.asyncio
async def test_read_markets_filter_resolves_within_days() -> None:
    query, args = await _read_markets_query(
        filters=MarketFilters(resolves_within_days=7)
    )

    assert "markets.resolves_at IS NOT NULL" in query
    assert "markets.resolves_at <= $8" in query
    assert args[7] == REFERENCE_NOW + timedelta(days=7)


@pytest.mark.asyncio
async def test_read_markets_filter_subscribed_only() -> None:
    query, args = await _read_markets_query(
        filters=MarketFilters(subscribed="only"),
        current_asset_ids=frozenset({"market-2-no", "market-1-yes"}),
    )

    assert "$9 = 'only'" in query
    assert "EXISTS" in query
    assert "subscribed_tokens.token_id = ANY($10::text[])" in query
    assert "market_subscriptions AS user_subscriptions" in query
    assert args[8] == "only"
    assert args[9] == ["market-1-yes", "market-2-no"]


@pytest.mark.asyncio
async def test_read_markets_filter_subscribed_idle() -> None:
    query, args = await _read_markets_query(
        filters=MarketFilters(subscribed="idle"),
        current_asset_ids=frozenset({"market-1-yes"}),
    )

    assert "$9 = 'idle'" in query
    assert "NOT EXISTS" in query
    assert "subscribed_tokens.token_id = ANY($10::text[])" in query
    assert "market_subscriptions AS user_subscriptions" in query
    assert args[8] == "idle"
    assert args[9] == ["market-1-yes"]


@pytest.mark.asyncio
async def test_read_markets_filter_q_substring() -> None:
    query, args = await _read_markets_query(filters=MarketFilters(q="Consensus"))

    assert "markets.question ILIKE '%' || $2 || '%'" in query
    assert args[1] == "Consensus"


@pytest.mark.asyncio
async def test_read_markets_null_price_excluded_from_band() -> None:
    query, args = await _read_markets_query(filters=MarketFilters(yes_min=0.2))

    assert "$6 = 0" in query
    assert "markets.yes_price IS NOT NULL AND markets.yes_price >= $6" in query
    assert args[5] == 0.2


@pytest.mark.asyncio
async def test_read_markets_combined_filters() -> None:
    query, args = await _read_markets_query(
        filters=MarketFilters(
            q="election",
            volume_min=1000.0,
            liquidity_min=2500.0,
            spread_max_bps=300,
            yes_min=0.25,
            yes_max=0.75,
            resolves_within_days=14,
            subscribed="only",
        ),
        current_asset_ids=frozenset({"token-a", "token-b"}),
    )

    assert "markets.question ILIKE '%' || $2 || '%'" in query
    assert args == (
        REFERENCE_NOW,
        "election",
        1000.0,
        2500.0,
        300,
        0.25,
        0.75,
        REFERENCE_NOW + timedelta(days=14),
        "only",
        ["token-a", "token-b"],
        20,
        0,
        None,
    )


@pytest.mark.asyncio
async def test_read_markets_returns_price_fields_and_subscription_source() -> None:
    price_updated_at = datetime(2026, 4, 23, 9, 31, tzinfo=UTC)
    resolves_at = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    connection = FakeConnection(
        fetch_results=[
            [
                _row(
                    market_id="market-priced",
                    question="Will read_markets include prices?",
                    updated_at=datetime(2026, 4, 23, 9, 32, tzinfo=UTC),
                    resolves_at=resolves_at,
                    yes_token_id="market-priced-yes",
                    no_token_id="market-priced-no",
                    total_count=1,
                    yes_price=0.62,
                    no_price=0.38,
                    best_bid=0.61,
                    best_ask=0.63,
                    last_trade_price=0.62,
                    liquidity=2500.25,
                    spread_bps=200,
                    price_updated_at=price_updated_at,
                    subscription_source="user",
                )
            ]
        ]
    )
    store = PostgresMarketDataStore(FakePool(connection))

    rows, total = await store.read_markets(limit=20, offset=0)

    assert total == 1
    assert rows[0].yes_price == 0.62
    assert rows[0].resolves_at == resolves_at
    assert rows[0].no_price == 0.38
    assert rows[0].best_bid == 0.61
    assert rows[0].best_ask == 0.63
    assert rows[0].last_trade_price == 0.62
    assert rows[0].liquidity == 2500.25
    assert rows[0].spread_bps == 200
    assert rows[0].price_updated_at == price_updated_at
    assert rows[0].subscription_source == "user"


@pytest.mark.asyncio
async def test_read_markets_returns_zero_total_when_query_returns_no_rows() -> None:
    connection = FakeConnection(fetch_results=[[]])
    store = PostgresMarketDataStore(FakePool(connection))

    rows, total = await store.read_markets(limit=20, offset=0)

    assert rows == []
    assert total == 0


@pytest.mark.asyncio
async def test_write_market_upserts_current_price_fields() -> None:
    connection = FakeConnection()
    store = PostgresMarketDataStore(FakePool(connection))
    created_at = datetime(2026, 4, 24, 8, 0, tzinfo=UTC)
    price_updated_at = datetime(2026, 4, 24, 8, 1, tzinfo=UTC)

    await store.write_market(
        Market(
            condition_id="market-priced",
            slug="market-priced",
            question="Will write_market persist prices?",
            venue="polymarket",
            resolves_at=None,
            created_at=created_at,
            last_seen_at=price_updated_at,
            volume_24h=500.0,
            risk_group_id="event:market-priced",
            category="Politics",
            event_id="market-priced",
            yes_price=0.52,
            no_price=0.48,
            best_bid=0.51,
            best_ask=0.53,
            last_trade_price=0.52,
            liquidity=1500.0,
            spread_bps=200,
            price_updated_at=price_updated_at,
        )
    )

    query, args = connection.execute_calls[0]
    assert "yes_price" in query
    assert "no_price" in query
    assert "price_updated_at = EXCLUDED.price_updated_at" in query
    assert args[8:] == (
        "event:market-priced",
        "Politics",
        "market-priced",
        0.52,
        0.48,
        0.51,
        0.53,
        0.52,
        1500.0,
        200,
        price_updated_at,
        None,
        None,
        None,
        None,
    )


@pytest.mark.asyncio
async def test_read_market_signal_metadata_returns_nonblank_fields() -> None:
    connection = FakeConnection(
        fetchrow_results=[
            {
                "risk_group_id": "event:2028-us-presidential-election",
                "category": "Politics",
                "event_id": "",
                "yes_token_id": "yes-token",
                "no_token_id": "no-token",
            }
        ]
    )
    store = PostgresMarketDataStore(FakePool(connection))

    metadata = await store.read_market_signal_metadata("m-election-risk")

    assert metadata == {
        "risk_group_id": "event:2028-us-presidential-election",
        "category": "Politics",
        "yes_token_id": "yes-token",
        "no_token_id": "no-token",
    }
    assert connection.fetchrow_calls[0][1] == ("m-election-risk",)


@pytest.mark.asyncio
async def test_write_price_snapshot_inserts_row() -> None:
    connection = FakeConnection()
    store = PostgresMarketDataStore(FakePool(connection))
    snapshot_at = datetime(2026, 4, 24, 9, 0, tzinfo=UTC)

    await store.write_price_snapshot(
        condition_id="market-snapshot",
        snapshot_at=snapshot_at,
        yes_price=0.61,
        no_price=0.39,
        best_bid=0.60,
        best_ask=0.62,
        last_trade_price=0.61,
        liquidity=2500.0,
        volume_24h=1250.0,
    )

    query, args = connection.execute_calls[0]
    assert "INSERT INTO market_price_snapshots" in query
    assert "condition_id" in query
    assert "volume_24h" in query
    assert args == (
        "market-snapshot",
        snapshot_at,
        0.61,
        0.39,
        0.60,
        0.62,
        0.61,
        2500.0,
        1250.0,
    )
