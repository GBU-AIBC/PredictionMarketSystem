from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
from typing import Literal, Protocol

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


SelectionReasonCode = Literal[
    "selected",
    "user_pin",
    "cap",
    "unresolved_market",
    "venue",
    "resolution_horizon",
    "resolved",
    "closed",
    "inactive",
    "stale",
    "volume",
    "accepting_orders",
    "liquidity",
    "yes_price_min",
    "yes_price_max",
    "spread",
    "depth",
    "not_selected",
]


@dataclass(frozen=True, slots=True)
class SelectionExplanation:
    market_id: str
    available: bool
    reasons: tuple[SelectionReasonCode, ...]
    strategy_ids: tuple[str, ...] = ()
    token_ids: tuple[str, ...] = ()
    pinned_token_ids: tuple[str, ...] = ()
    capped: bool = False
    strategy_reasons: dict[str, SelectionReasonCode] | None = None


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

    async def explain(
        self,
        market_id: str,
        *,
        max_subscription_asset_ids: int | None = None,
        current_asset_ids: Sequence[str] = (),
    ) -> SelectionExplanation:
        read_market = getattr(self._store, "read_market", None)
        if not callable(read_market):
            return SelectionExplanation(market_id, False, ("unresolved_market",))
        market = await read_market(market_id)
        if market is None:
            return SelectionExplanation(market_id, False, ("unresolved_market",))
        strategy_ids: list[str] = []
        token_ids: list[str] = []
        reasons: list[SelectionReasonCode] = []
        strategy_reasons: dict[str, SelectionReasonCode] = {}
        specs = await self._registry.list_market_selections()
        for strategy_id, _version_id, spec in specs:
            strategy_ids.append(strategy_id)
            venue = normalize_venue(spec.venue, context="MarketSelector.explain")
            eligible = await self._store.read_eligible_markets(
                venue,
                spec.resolution_time_max_horizon_days,
                spec.volume_min_usdc,
            )
            candidate = next(
                (
                    (candidate_market, candidate_tokens)
                    for candidate_market, candidate_tokens in eligible
                    if candidate_market.condition_id == market_id
                ),
                None,
            )
            if candidate is None:
                reason = await self._discovery_reason(
                    market,
                    spec,
                    eligible_markets=eligible,
                )
                if reason is None:
                    reason = await self._explain_for_spec(market, spec)
            else:
                candidate_market, candidate_tokens = candidate
                reason = await self._filter_reason(candidate_market, spec)
            strategy_reasons[strategy_id] = "selected" if reason is None else reason
            if reason is not None:
                reasons.append(reason)
            if candidate is not None:
                token_ids.extend(
                    token.token_id
                    for token in candidate_tokens
                    if token.outcome in {"YES", "NO"}
                )
            else:
                read_tokens = getattr(self._store, "read_tokens_for_market", None)
                if callable(read_tokens):
                    tokens = await read_tokens(market_id)
                    token_ids.extend(
                        token.token_id
                        for token in tokens
                        if token.outcome in {"YES", "NO"}
                    )
        pinned = tuple(sorted(await self._read_user_subscriptions()))
        token_ids = list(dict.fromkeys(token_ids))
        if pinned and any(token_id in pinned for token_id in token_ids):
            reasons.append("user_pin")
        if max_subscription_asset_ids is not None and token_ids:
            active_assets = set(current_asset_ids)
            capped = (
                bool(active_assets)
                and len(active_assets) >= max_subscription_asset_ids
                and any(token_id not in active_assets for token_id in token_ids)
            ) or (
                not active_assets and len(token_ids) > max_subscription_asset_ids
            )
            if capped:
                reasons.append("cap")
        else:
            capped = False
        selected_by_strategy = any(reason == "selected" for reason in strategy_reasons.values())
        pinned_for_market = bool(pinned and any(token_id in pinned for token_id in token_ids))
        if selected_by_strategy or pinned_for_market:
            reasons.append("selected")
        elif not reasons:
            reasons.append("not_selected")
        return SelectionExplanation(
            market_id=market_id,
            available=True,
            reasons=tuple(dict.fromkeys(reasons)),
            strategy_ids=tuple(strategy_ids),
            token_ids=tuple(token_ids),
            pinned_token_ids=tuple(token_id for token_id in pinned if token_id in token_ids),
            capped=capped,
            strategy_reasons=strategy_reasons,
        )

    async def _discovery_reason(
        self,
        market: Market,
        spec: MarketSelectionSpec,
        *,
        eligible_markets: Sequence[tuple[Market, Sequence[Token]]] | None = None,
    ) -> SelectionReasonCode | None:
        """Use the store's eligibility query as the discovery source of truth."""
        venue = normalize_venue(spec.venue, context="MarketSelector.explain")
        eligible = (
            eligible_markets
            if eligible_markets is not None
            else await self._store.read_eligible_markets(
                venue,
                spec.resolution_time_max_horizon_days,
                spec.volume_min_usdc,
            )
        )
        if any(candidate.condition_id == market.condition_id for candidate, _ in eligible):
            return None
        if getattr(market.venue, "value", market.venue) != venue:
            return "venue"
        if spec.volume_min_usdc > 0 and (
            market.volume_24h is None or market.volume_24h < spec.volume_min_usdc
        ):
            return "volume"
        if market.closed is True:
            return "closed"
        if market.active is False:
            return "inactive"
        freshness_s = float(getattr(self._store, "_market_freshness_s", 900.0))
        if market.last_seen_at < datetime.now(tz=UTC) - timedelta(seconds=freshness_s):
            return "stale"
        if market.resolves_at is not None and market.resolves_at <= datetime.now(tz=UTC):
            return "resolved"
        if spec.resolution_time_max_horizon_days is not None:
            upper_bound = datetime.now(tz=UTC) + timedelta(
                days=spec.resolution_time_max_horizon_days
            )
            if market.resolves_at is None or market.resolves_at > upper_bound:
                return "resolution_horizon"
        return None

    async def _explain_for_spec(
        self,
        market: Market,
        spec: MarketSelectionSpec,
    ) -> SelectionReasonCode | None:
        market_venue = getattr(market.venue, "value", market.venue)
        if normalize_venue(spec.venue, context="MarketSelector.explain") != market_venue:
            return "venue"
        if spec.resolution_time_max_horizon_days is not None:
            if market.resolves_at is None:
                return "resolution_horizon"
            horizon = market.resolves_at - datetime.now(tz=UTC)
            if horizon.total_seconds() > spec.resolution_time_max_horizon_days * 86400:
                return "resolution_horizon"
        if spec.volume_min_usdc > 0 and (
            market.volume_24h is None or market.volume_24h < spec.volume_min_usdc
        ):
            return "volume"
        return await self._filter_reason(market, spec)

    async def _filter_reason(
        self,
        market: Market,
        spec: MarketSelectionSpec,
    ) -> SelectionReasonCode | None:
        """Apply the same discovery predicates used by selection and explain."""
        if spec.accepting_orders and market.accepting_orders is False:
            return "accepting_orders"
        if spec.liquidity_min_usdc is not None and (
            market.liquidity is None or market.liquidity < spec.liquidity_min_usdc
        ):
            return "liquidity"
        if spec.yes_price_min is not None and (
            market.yes_price is None or market.yes_price < spec.yes_price_min
        ):
            return "yes_price_min"
        if spec.yes_price_max is not None and (
            market.yes_price is None or market.yes_price > spec.yes_price_max
        ):
            return "yes_price_max"
        if spec.spread_max_bps is None and spec.depth_min_usdc is None:
            return None
        summary = await self._store.get_latest_book_summary(market.condition_id)
        if summary is None:
            if not _market_passes_discovery_bootstrap_filters(market, spec):
                if spec.spread_max_bps is not None:
                    return "spread"
                return "depth"
            return None
        if spec.spread_max_bps is not None and summary.spread_bps > spec.spread_max_bps:
            return "spread"
        if spec.depth_min_usdc is not None and summary.depth_usdc < spec.depth_min_usdc:
            return "depth"
        return None

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
        return await self._filter_reason(market, spec) is None

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
