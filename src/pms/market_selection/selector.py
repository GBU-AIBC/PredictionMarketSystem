from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Protocol

from pms.core.enums import Venue
from pms.core.exceptions import KalshiStubError  # noqa: F401
from pms.core.interfaces import MarketDataStore, StrategySelectionRegistry
from pms.core.models import Market, Token
from pms.core.venue_support import kalshi_stub_error, normalize_venue
from pms.market_selection.merge import MergePolicy, MergeResult, StrategyMarketSet
from pms.metrics import (
    SELECTION_FUNNEL_DISCOVERED_TOTAL_METRIC,
    SELECTION_FUNNEL_SELECTED_TOTAL_METRIC,
    increment_metric,
)
from pms.strategies.projections import MarketSelectionSpec


logger = logging.getLogger(__name__)


class UserSubscriptionStore(Protocol):
    async def read_user_subscriptions(self) -> set[str]: ...


class MarketSelector:
    def __init__(
        self,
        store: MarketDataStore,
        registry: StrategySelectionRegistry,
        merge_policy: MergePolicy,
        market_subscription_store: UserSubscriptionStore | None = None,
    ) -> None:
        self._store = store
        self._registry = registry
        self._merge_policy = merge_policy
        self._market_subscription_store = market_subscription_store
        self._last_discovered_count = 0
        self._last_user_subscription_asset_ids: frozenset[str] = frozenset()

    @property
    def last_user_subscription_asset_ids(self) -> frozenset[str]:
        return self._last_user_subscription_asset_ids

    async def select(self) -> MergeResult:
        selections = await self.select_per_strategy()
        logger.info(
            "market selector processed %d active strategies",
            len(selections),
        )
        if not selections:
            logger.warning(
                "no active_version_id rows found; data sensor will idle until "
                "a strategy is activated",
            )
        merged = self._merge_policy.merge(selections)
        user_asset_ids = await self._read_user_subscriptions()
        self._last_user_subscription_asset_ids = user_asset_ids
        if not user_asset_ids:
            _log_selector_funnel(self._last_discovered_count, len(merged.asset_ids))
            return merged
        # User pins are ordered first so the runner's hard cap preserves them
        # ahead of strategy-selected assets. The runner also passes these ids
        # as protected ids when it applies the cap during runtime refreshes.
        asset_ids = sorted(user_asset_ids)
        seen_asset_ids = set(asset_ids)
        asset_ids.extend(
            asset_id
            for asset_id in merged.asset_ids
            if asset_id not in seen_asset_ids
        )
        result = MergeResult(
            asset_ids=asset_ids,
            conflicts=merged.conflicts,
        )
        _log_selector_funnel(self._last_discovered_count, len(result.asset_ids))
        return result

    async def select_per_strategy(self) -> list[StrategyMarketSet]:
        strategy_specs = await self._registry.list_market_selections()
        selections: list[StrategyMarketSet] = []
        self._last_discovered_count = 0
        for strategy_id, strategy_version_id, spec in strategy_specs:
            venue = normalize_venue(
                spec.venue,
                context="MarketSelector.select_per_strategy",
            )
            if venue == Venue.KALSHI.value:
                raise kalshi_stub_error("MarketSelector.select_per_strategy")
            eligible_markets = await self._store.read_eligible_markets(
                venue,
                spec.resolution_time_max_horizon_days,
                spec.volume_min_usdc,
            )
            self._last_discovered_count += len(
                _asset_ids_from_eligible_markets(eligible_markets)
            )
            filtered_markets = await self._filter_markets(eligible_markets, spec)
            ranked_asset_ids = _asset_ids_from_eligible_markets(filtered_markets)
            selections.append(
                StrategyMarketSet(
                    strategy_id=strategy_id,
                    strategy_version_id=strategy_version_id,
                    asset_ids=frozenset(ranked_asset_ids),
                    ranked_asset_ids=ranked_asset_ids,
                )
            )
        return selections

    async def _filter_markets(
        self,
        eligible_markets: Sequence[tuple[Market, Sequence[Token]]],
        spec: MarketSelectionSpec,
    ) -> list[tuple[Market, Sequence[Token]]]:
        filtered_markets: list[tuple[Market, Sequence[Token]]] = []
        for market, tokens in eligible_markets:
            if not await self._market_passes_filters(market, spec):
                continue
            filtered_markets.append((market, tokens))
        return filtered_markets

    async def _market_passes_filters(
        self,
        market: Market,
        spec: MarketSelectionSpec,
    ) -> bool:
        if spec.accepting_orders and market.accepting_orders is False:
            return False

        if spec.liquidity_min_usdc is not None and (
            market.liquidity is None or market.liquidity < spec.liquidity_min_usdc
        ):
            return False

        if spec.yes_price_min is not None and (
            market.yes_price is None or market.yes_price < spec.yes_price_min
        ):
            return False

        if spec.yes_price_max is not None and (
            market.yes_price is None or market.yes_price > spec.yes_price_max
        ):
            return False

        if spec.spread_max_bps is None and spec.depth_min_usdc is None:
            return True

        summary = await self._store.get_latest_book_summary(market.condition_id)
        if summary is None:
            return _market_passes_discovery_bootstrap_filters(market, spec)
        if spec.spread_max_bps is not None and summary.spread_bps > spec.spread_max_bps:
            return False
        if spec.depth_min_usdc is not None and summary.depth_usdc < spec.depth_min_usdc:
            return False
        return True

    async def _read_user_subscriptions(self) -> frozenset[str]:
        if self._market_subscription_store is None:
            return frozenset()
        return frozenset(await self._market_subscription_store.read_user_subscriptions())


def _log_selector_funnel(discovered_count: int, selected_count: int) -> None:
    increment_metric(SELECTION_FUNNEL_DISCOVERED_TOTAL_METRIC, discovered_count)
    increment_metric(SELECTION_FUNNEL_SELECTED_TOTAL_METRIC, selected_count)
    logger.info(
        "market selector funnel discovered=%d selected=%d",
        discovered_count,
        selected_count,
        extra={
            "event": "funnel_selector",
            "discovered_count": discovered_count,
            "selected_count": selected_count,
        },
    )


def _market_passes_discovery_bootstrap_filters(
    market: Market,
    spec: MarketSelectionSpec,
) -> bool:
    if spec.spread_max_bps is not None:
        if market.spread_bps is None or market.spread_bps > spec.spread_max_bps:
            return False

    if spec.depth_min_usdc is not None:
        if market.liquidity is None or market.liquidity < spec.depth_min_usdc:
            return False

    return True


def _asset_ids_from_eligible_markets(
    eligible_markets: Sequence[tuple[Market, Sequence[Token]]],
) -> tuple[str, ...]:
    ordered_asset_ids: list[str] = []
    seen: set[str] = set()
    for _, tokens in eligible_markets:
        market_asset_ids = sorted(
            token.token_id for token in tokens if token.outcome in {"YES", "NO"}
        )
        for asset_id in market_asset_ids:
            if asset_id in seen:
                continue
            seen.add(asset_id)
            ordered_asset_ids.append(asset_id)
    return tuple(ordered_asset_ids)
