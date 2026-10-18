from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from pms.core.models import BookSummary, Market, Token
from pms.market_selection.merge import UnionMergePolicy
from pms.market_selection.selector import MarketSelector, SelectionExplanation
from pms.strategies.projections import MarketSelectionSpec


@dataclass
class _MarketStoreDouble:
    token_ids: list[str]

    async def read_eligible_markets(
        self,
        venue: str,
        max_horizon_days: int | None,
        min_volume_usdc: float,
    ) -> list[tuple[Market, list[Token]]]:
        assert venue == "polymarket"
        assert max_horizon_days == 30
        assert min_volume_usdc == 500.0
        now = datetime(2026, 4, 24, 9, 0, tzinfo=UTC)
        market = Market(
            condition_id="strategy-market",
            slug="strategy-market",
            question="Will strategy market resolve?",
            venue="polymarket",
            resolves_at=now + timedelta(days=7),
            created_at=now,
            last_seen_at=now,
            volume_24h=1_000.0,
        )
        tokens = [
            Token(token_id=token_id, condition_id=market.condition_id, outcome="YES")
            for token_id in self.token_ids
        ]
        return [(market, tokens)]

    async def get_latest_book_summary(self, market_id: str) -> BookSummary | None:
        return None

    async def read_market(self, market_id: str) -> Market | None:
        rows = await self.read_eligible_markets("polymarket", 30, 500.0)
        return next((market for market, _tokens in rows if market.condition_id == market_id), None)

    async def read_tokens_for_market(self, market_id: str) -> list[Token]:
        rows = await self.read_eligible_markets("polymarket", 30, 500.0)
        return next((tokens for market, tokens in rows if market.condition_id == market_id), [])


@dataclass
class _RegistryDouble:
    async def list_market_selections(
        self,
    ) -> list[tuple[str, str, MarketSelectionSpec]]:
        return [
            (
                "strategy-a",
                "strategy-a-v1",
                MarketSelectionSpec(
                    venue="polymarket",
                    resolution_time_max_horizon_days=30,
                    volume_min_usdc=500.0,
                ),
            )
        ]


@dataclass
class _UserSubscriptionStoreDouble:
    results: list[set[str]]
    calls: int = 0
    seen_results: list[set[str]] = field(default_factory=list)

    async def read_user_subscriptions(self) -> set[str]:
        self.calls += 1
        result = self.results.pop(0)
        self.seen_results.append(set(result))
        return result


def _selector(
    *,
    strategy_token_ids: list[str],
    user_results: list[set[str]],
) -> tuple[MarketSelector, _UserSubscriptionStoreDouble]:
    user_store = _UserSubscriptionStoreDouble(user_results)
    return (
        MarketSelector(
            store=_MarketStoreDouble(strategy_token_ids),
            registry=_RegistryDouble(),
            merge_policy=UnionMergePolicy(),
            market_subscription_store=user_store,
        ),
        user_store,
    )


@pytest.mark.asyncio
async def test_select_prioritizes_user_subscriptions_before_cap() -> None:
    selector, user_store = _selector(
        strategy_token_ids=["strategy-token"],
        user_results=[{"user-token"}],
    )

    result = await selector.select()

    assert result.asset_ids == ["user-token", "strategy-token"]
    assert user_store.calls == 1


@pytest.mark.asyncio
async def test_select_deduplicates_overlapping_user_and_strategy_token() -> None:
    selector, _ = _selector(
        strategy_token_ids=["shared-token"],
        user_results=[{"shared-token"}],
    )

    result = await selector.select()

    assert result.asset_ids == ["shared-token"]


@pytest.mark.asyncio
async def test_select_reads_user_subscriptions_on_every_call() -> None:
    selector, user_store = _selector(
        strategy_token_ids=["strategy-token"],
        user_results=[{"first-user-token"}, {"second-user-token"}],
    )

    first = await selector.select()
    second = await selector.select()

    assert first.asset_ids == ["first-user-token", "strategy-token"]
    assert second.asset_ids == ["second-user-token", "strategy-token"]
    assert user_store.calls == 2
    assert user_store.seen_results == [{"first-user-token"}, {"second-user-token"}]


@pytest.mark.asyncio
async def test_select_no_user_rows_preserves_existing_merge_result() -> None:
    selector, user_store = _selector(
        strategy_token_ids=["strategy-token"],
        user_results=[set()],
    )

    result = await selector.select()

    assert result.asset_ids == ["strategy-token"]
    assert result.conflicts == []
    assert user_store.calls == 1


@pytest.mark.asyncio
async def test_explain_reports_pins_and_active_subscription_cap() -> None:
    selector, _ = _selector(
        strategy_token_ids=["strategy-token", "second-token"],
        user_results=[{"second-token"}],
    )

    explanation = await selector.explain(
        "strategy-market",
        max_subscription_asset_ids=2,
        current_asset_ids=("existing-token", "another-token"),
    )

    assert explanation.available is True
    assert explanation.pinned_token_ids == ("second-token",)
    assert "user_pin" in explanation.reasons
    assert "cap" in explanation.reasons
    assert explanation.capped is True


@pytest.mark.asyncio
async def test_explain_reports_per_strategy_selection_reasons() -> None:
    now = datetime(2026, 4, 24, 9, 0, tzinfo=UTC)
    market = Market(
        condition_id="disagreement-market",
        slug="disagreement-market",
        question="Will disagreement market resolve?",
        venue="polymarket",
        resolves_at=now + timedelta(days=7),
        created_at=now,
        last_seen_at=now,
        volume_24h=1_000.0,
    )
    token = Token(token_id="disagreement-token", condition_id=market.condition_id, outcome="YES")

    class Store:
        async def read_market(self, market_id: str) -> Market | None:
            return market if market_id == market.condition_id else None

        async def read_tokens_for_market(self, market_id: str) -> list[Token]:
            return [token] if market_id == market.condition_id else []

        async def read_eligible_markets(self, venue: str, max_horizon_days: int | None, min_volume_usdc: float) -> list[tuple[Market, list[Token]]]:
            del venue, max_horizon_days
            return [(market, [token])] if min_volume_usdc <= (market.volume_24h or 0.0) else []

        async def get_latest_book_summary(self, market_id: str) -> BookSummary | None:
            del market_id
            return None

    class Registry:
        async def list_market_selections(self) -> list[tuple[str, str, MarketSelectionSpec]]:
            return [
                ("strategy-a", "a-v1", MarketSelectionSpec(venue="polymarket", resolution_time_max_horizon_days=30, volume_min_usdc=500.0)),
                ("strategy-b", "b-v1", MarketSelectionSpec(venue="polymarket", resolution_time_max_horizon_days=30, volume_min_usdc=5_000.0)),
            ]

    selector = MarketSelector(
        store=Store(),
        registry=Registry(),
        merge_policy=UnionMergePolicy(),
    )
    explanation = await selector.explain(market.condition_id)

    assert explanation.strategy_reasons == {"strategy-a": "selected", "strategy-b": "volume"}
    assert "selected" in explanation.reasons
    assert "volume" in explanation.reasons


async def _explain_single_market(
    market: Market | None,
    spec: MarketSelectionSpec,
    summary: BookSummary | None = None,
) -> SelectionExplanation:
    token = Token(token_id="single-token", condition_id="single-market", outcome="YES")

    class Store:
        async def read_market(self, market_id: str) -> Market | None:
            return market if market_id == "single-market" else None

        async def read_tokens_for_market(self, market_id: str) -> list[Token]:
            return [token] if market_id == "single-market" else []

        async def read_eligible_markets(self, venue: str, max_horizon_days: int | None, min_volume_usdc: float) -> list[tuple[Market, list[Token]]]:
            del venue, max_horizon_days, min_volume_usdc
            return []

        async def get_latest_book_summary(self, market_id: str) -> BookSummary | None:
            del market_id
            return summary

    class Registry:
        async def list_market_selections(self) -> list[tuple[str, str, MarketSelectionSpec]]:
            return [("single", "single-v1", spec)]

    selector = MarketSelector(Store(), Registry(), UnionMergePolicy())
    return await selector.explain("single-market")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("accepting_orders", False, "accepting_orders"),
        ("liquidity", 1.0, "liquidity"),
        ("yes_price", 0.1, "yes_price_min"),
    ],
)
async def test_explain_reports_market_filter_reasons(
    field: str,
    value: object,
    reason: str,
) -> None:
    now = datetime.now(tz=UTC)
    market = Market(
        condition_id="single-market", slug="single", question="q", venue="polymarket",
        resolves_at=now + timedelta(days=1), created_at=now, last_seen_at=now,
        volume_24h=1_000.0, accepting_orders=True, liquidity=1_000.0, yes_price=0.5,
    )
    market = replace(market, **cast(Any, {field: value}))
    spec = MarketSelectionSpec(
        venue="polymarket", resolution_time_max_horizon_days=30, volume_min_usdc=500.0,
        accepting_orders=True, liquidity_min_usdc=500.0, yes_price_min=0.4,
    )
    explanation = await _explain_single_market(market, spec)
    assert reason in explanation.reasons
    assert "selected" not in explanation.reasons
    assert explanation.strategy_reasons == {"single": reason}


@pytest.mark.asyncio
async def test_explain_reports_spread_depth_bootstrap_and_unresolved_market() -> None:
    now = datetime.now(tz=UTC)
    market = Market(
        condition_id="single-market", slug="single", question="q", venue="polymarket",
        resolves_at=now + timedelta(days=1), created_at=now, last_seen_at=now,
        volume_24h=1_000.0, spread_bps=100, liquidity=1_000.0,
    )
    spec = MarketSelectionSpec(
        venue="polymarket", resolution_time_max_horizon_days=30, volume_min_usdc=500.0,
        spread_max_bps=20, depth_min_usdc=1_000.0,
    )
    explanation = await _explain_single_market(market, spec)
    assert "spread" in explanation.reasons
    unresolved = await _explain_single_market(None, spec)
    assert unresolved.available is False
    assert unresolved.reasons == ("unresolved_market",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("closed", True, "closed"),
        ("active", False, "inactive"),
        ("last_seen_at", datetime.now(tz=UTC) - timedelta(days=2), "stale"),
        ("resolves_at", datetime.now(tz=UTC) - timedelta(days=1), "resolved"),
    ],
)
async def test_explain_reports_sql_discovery_regimes(
    field: str,
    value: object,
    reason: str,
) -> None:
    now = datetime.now(tz=UTC)
    market = Market(
        condition_id="single-market", slug="single", question="q", venue="polymarket",
        resolves_at=now + timedelta(days=1), created_at=now, last_seen_at=now,
        volume_24h=1_000.0,
    )
    market = replace(market, **cast(Any, {field: value}))
    spec = MarketSelectionSpec(
        venue="polymarket", resolution_time_max_horizon_days=30, volume_min_usdc=500.0,
    )
    explanation = await _explain_single_market(market, spec)
    assert reason in explanation.reasons


@pytest.mark.asyncio
async def test_explain_depth_reason_comes_from_book_summary() -> None:
    now = datetime.now(tz=UTC)
    market = Market(
        condition_id="single-market", slug="single", question="q", venue="polymarket",
        resolves_at=now + timedelta(days=1), created_at=now, last_seen_at=now,
        volume_24h=1_000.0,
    )
    summary = BookSummary(
        best_bid=0.4, best_ask=0.6, spread_bps=10.0, depth_usdc=1.0, timestamp=now,
    )
    spec = MarketSelectionSpec(
        venue="polymarket", resolution_time_max_horizon_days=30, volume_min_usdc=500.0,
        depth_min_usdc=100.0,
    )
    explanation = await _explain_single_market(market, spec, summary)
    assert "depth" in explanation.reasons
