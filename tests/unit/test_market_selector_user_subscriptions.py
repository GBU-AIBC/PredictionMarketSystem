from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest

from pms.core.models import BookSummary, Market, Token
from pms.market_selection.merge import UnionMergePolicy
from pms.market_selection.selector import MarketSelector
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
