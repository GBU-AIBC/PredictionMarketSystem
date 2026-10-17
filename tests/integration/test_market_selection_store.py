from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest

from pms.core.models import Market, Token
from pms.storage.market_data_store import PostgresMarketDataStore


PMS_TEST_DATABASE_URL = os.environ.get("PMS_TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("PMS_RUN_INTEGRATION") != "1",
        reason="set PMS_RUN_INTEGRATION=1 to run PostgreSQL integration tests",
    ),
    pytest.mark.skipif(
        PMS_TEST_DATABASE_URL is None,
        reason="set PMS_TEST_DATABASE_URL to the compose-backed PostgreSQL URI",
    ),
]


def _market(
    *,
    condition_id: str,
    venue: str,
    resolves_at: datetime | None,
    created_at: datetime,
    volume_24h: float,
    yes_price: float | None = None,
    no_price: float | None = None,
    best_bid: float | None = None,
    best_ask: float | None = None,
    spread_bps: int | None = None,
    last_seen_at: datetime | None = None,
    active: bool | None = None,
    closed: bool | None = None,
) -> Market:
    return Market(
        condition_id=condition_id,
        slug=condition_id,
        question=f"Will {condition_id} resolve?",
        venue=venue,  # type: ignore[arg-type]
        resolves_at=resolves_at,
        created_at=created_at,
        last_seen_at=last_seen_at or created_at,
        volume_24h=volume_24h,
        yes_price=yes_price,
        no_price=no_price,
        best_bid=best_bid,
        best_ask=best_ask,
        spread_bps=spread_bps,
        active=active,
        closed=closed,
    )


def _token(
    *,
    token_id: str,
    condition_id: str,
    outcome: str,
) -> Token:
    return Token(
        token_id=token_id,
        condition_id=condition_id,
        outcome=outcome,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_read_eligible_markets_filters_by_venue_and_horizon_and_keeps_null_resolves_at_for_open_horizon(
    pg_pool: asyncpg.Pool,
) -> None:
    store = PostgresMarketDataStore(pg_pool)
    now = datetime.now(tz=UTC)

    future_in_horizon = _market(
        condition_id="market-in-horizon",
        venue="polymarket",
        resolves_at=now + timedelta(days=10),
        created_at=now,
        volume_24h=1_000.0,
        yes_price=0.62,
        no_price=0.38,
        best_bid=0.61,
        best_ask=0.63,
        spread_bps=200,
    )
    future_out_of_horizon = _market(
        condition_id="market-out-of-horizon",
        venue="polymarket",
        resolves_at=now + timedelta(days=60),
        created_at=now,
        volume_24h=1_000.0,
    )
    already_resolved = _market(
        condition_id="market-past",
        venue="polymarket",
        resolves_at=now - timedelta(days=1),
        created_at=now,
        volume_24h=1_000.0,
    )
    no_resolves_at = _market(
        condition_id="market-open-ended",
        venue="polymarket",
        resolves_at=None,
        created_at=now,
        volume_24h=1_000.0,
    )
    other_venue = _market(
        condition_id="market-kalshi",
        venue="kalshi",
        resolves_at=now + timedelta(days=5),
        created_at=now,
        volume_24h=1_000.0,
    )
    low_volume = _market(
        condition_id="market-low-volume",
        venue="polymarket",
        resolves_at=now + timedelta(days=3),
        created_at=now,
        volume_24h=100.0,
    )
    stale_closed = _market(
        condition_id="market-stale-closed",
        venue="polymarket",
        resolves_at=now + timedelta(days=3),
        created_at=now,
        last_seen_at=now - timedelta(hours=1),
        volume_24h=2_000.0,
        closed=True,
    )

    for market in (
        future_in_horizon,
        future_out_of_horizon,
        already_resolved,
        no_resolves_at,
        other_venue,
        low_volume,
        stale_closed,
    ):
        await store.write_market(market)

    await store.write_token(
        _token(token_id="in-yes", condition_id=future_in_horizon.condition_id, outcome="YES")
    )
    await store.write_token(
        _token(token_id="in-no", condition_id=future_in_horizon.condition_id, outcome="NO")
    )
    await store.write_token(
        _token(token_id="out-yes", condition_id=future_out_of_horizon.condition_id, outcome="YES")
    )
    await store.write_token(
        _token(token_id="past-yes", condition_id=already_resolved.condition_id, outcome="YES")
    )
    await store.write_token(
        _token(token_id="other-yes", condition_id=other_venue.condition_id, outcome="YES")
    )
    await store.write_token(
        _token(token_id="low-yes", condition_id=low_volume.condition_id, outcome="YES")
    )

    bounded = await store.read_eligible_markets("polymarket", 30, 500.0)
    open_horizon = await store.read_eligible_markets("polymarket", None, 500.0)

    assert [market.condition_id for market, _ in bounded] == ["market-in-horizon"]
    assert bounded[0][0].yes_price == pytest.approx(0.62)
    assert bounded[0][0].no_price == pytest.approx(0.38)
    assert bounded[0][0].best_bid == pytest.approx(0.61)
    assert bounded[0][0].best_ask == pytest.approx(0.63)
    assert bounded[0][0].spread_bps == 200
    assert [token.token_id for token in bounded[0][1]] == ["in-yes", "in-no"]

    assert [market.condition_id for market, _ in open_horizon] == [
        "market-in-horizon",
        "market-open-ended",
        "market-out-of-horizon",
    ]
    assert [token.token_id for token in open_horizon[0][1]] == ["in-yes", "in-no"]
    assert open_horizon[1][1] == []


@pytest.mark.asyncio(loop_scope="session")
async def test_read_eligible_markets_returns_empty_list_when_no_rows_match(
    pg_pool: asyncpg.Pool,
) -> None:
    store = PostgresMarketDataStore(pg_pool)

    assert await store.read_eligible_markets("polymarket", 7, 500.0) == []
