"""Observation source for the H2 anchoring-lag strategy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from pms.core.enums import TimeInForce
from pms.core.models import BookSide, Outcome, Portfolio, Venue
from pms.factors.definitions.anchoring_lag_divergence import DEFAULT_DECAY_WINDOW_HOURS
from pms.strategies.aggregate import Strategy
from pms.strategies.anchoring.evaluator import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_DIVERGENCE,
    DEFAULT_MIN_EXPECTED_EDGE,
)
from pms.strategies.intents import StrategyContext, StrategyObservation
from pms.strategies.projections import (
    EvalSpec,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)


ANCHORING_RESEARCH_REF = "research:h2-anchoring-lag-spec#h2"
LIVE_ANCHORING_SOURCE = "live_anchoring_lag_source"
ANCHORING_ECONOMICS_VERSION = "anchoring-economics-v1"
ANCHORING_LEGACY_VERSION_IDS = frozenset({"h2-anchoring-lag-v1"})
ANCHORING_MAX_POSITION_NOTIONAL_USDC = 1.0
ANCHORING_DEFAULT_MIN_DIVERGENCE = DEFAULT_MIN_DIVERGENCE
ANCHORING_DEFAULT_MIN_CONFIDENCE = DEFAULT_MIN_CONFIDENCE
ANCHORING_DEFAULT_MIN_EXPECTED_EDGE = DEFAULT_MIN_EXPECTED_EDGE
ANCHORING_DEFAULT_MAX_SLIPPAGE_BPS = 50
ANCHORING_DEFAULT_TIME_IN_FORCE = TimeInForce.GTC


def build_anchoring_strategy(
    strategy_id: str,
    *,
    time_in_force: str = "GTC",
) -> Strategy:
    """Build the immutable projection that owns H2 runtime economics."""
    return Strategy(
        config=StrategyConfig(
            strategy_id=strategy_id,
            factor_composition=(),
            metadata=(
                ("economics_version", ANCHORING_ECONOMICS_VERSION),
                ("min_divergence", str(ANCHORING_DEFAULT_MIN_DIVERGENCE)),
                ("min_confidence", str(ANCHORING_DEFAULT_MIN_CONFIDENCE)),
                ("min_expected_edge", str(ANCHORING_DEFAULT_MIN_EXPECTED_EDGE)),
                ("decay_window_hours", str(DEFAULT_DECAY_WINDOW_HOURS)),
                ("max_slippage_bps", str(ANCHORING_DEFAULT_MAX_SLIPPAGE_BPS)),
                ("time_in_force", time_in_force.upper()),
            ),
        ),
        risk=RiskParams(
            max_position_notional_usdc=ANCHORING_MAX_POSITION_NOTIONAL_USDC,
            max_daily_drawdown_pct=20.0,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(metrics=("pnl", "fill_rate")),
        forecaster=ForecasterSpec(forecasters=(("anchoring", ()),)),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=0.0,
        ),
    )


class AnchoringMarketSnapshotReader(Protocol):
    async def latest(
        self,
        market_id: str,
        *,
        as_of: datetime,
    ) -> AnchoringMarketSnapshot | None: ...


class AnchoringPositionSizer(Protocol):
    def size(
        self,
        *,
        prob: float,
        market_price: float,
        portfolio: Portfolio,
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class AnchoringMarketSnapshot:
    market_id: str
    title: str
    yes_token_id: str
    no_token_id: str
    yes_price: float
    llm_posterior: float
    llm_confidence: float
    news_timestamp: datetime
    observed_at: datetime
    yes_best_ask: float | None = None
    no_best_ask: float | None = None
    resolves_at: datetime | None = None
    news_ref: str | None = None
    venue: Venue = "polymarket"

    def __post_init__(self) -> None:
        _require_non_empty(self.market_id, "market_id")
        _require_non_empty(self.title, "title")
        _require_non_empty(self.yes_token_id, "yes_token_id")
        _require_non_empty(self.no_token_id, "no_token_id")
        _require_open_probability(self.yes_price, "yes_price")
        _require_open_probability(self.llm_posterior, "llm_posterior")
        _require_probability(self.llm_confidence, "llm_confidence")
        if self.yes_best_ask is not None:
            _require_open_probability(self.yes_best_ask, "yes_best_ask")
        if self.no_best_ask is not None:
            _require_open_probability(self.no_best_ask, "no_best_ask")


@dataclass(frozen=True, slots=True)
class LiveAnchoringSource:
    """Paper-safe source for H2 anchoring-lag signals.

    This source consumes already prepared LLM posterior/news observations. It
    does not call the LLM provider or fetch news itself, keeping the first H2
    slice deterministic and suitable for paper-soak validation.
    """

    market_ids: Sequence[str]
    market_reader: AnchoringMarketSnapshotReader
    position_sizer: AnchoringPositionSizer
    portfolio: Portfolio
    min_divergence: float = DEFAULT_MIN_DIVERGENCE
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_expected_edge: float = DEFAULT_MIN_EXPECTED_EDGE
    decay_window_hours: float = DEFAULT_DECAY_WINDOW_HOURS
    max_slippage_bps: int = 50
    time_in_force: TimeInForce = TimeInForce.GTC

    def __post_init__(self) -> None:
        if not self.market_ids:
            msg = "market_ids must not be empty"
            raise ValueError(msg)
        for market_id in self.market_ids:
            _require_non_empty(market_id, "market_id")
        if self.min_divergence <= 0.0:
            msg = "min_divergence must be > 0.0"
            raise ValueError(msg)
        if self.min_expected_edge <= 0.0:
            msg = "min_expected_edge must be > 0.0"
            raise ValueError(msg)
        _require_probability(self.min_confidence, "min_confidence")
        if self.decay_window_hours <= 0.0:
            msg = "decay_window_hours must be > 0.0"
            raise ValueError(msg)
        if self.max_slippage_bps < 0:
            msg = "max_slippage_bps must be >= 0"
            raise ValueError(msg)

    async def observe(self, context: StrategyContext) -> Sequence[StrategyObservation]:
        observations: list[StrategyObservation] = []
        for market_id in self.market_ids:
            market = await self.market_reader.latest(market_id, as_of=context.as_of)
            if market is None or _is_resolved(context.as_of, market.resolves_at):
                continue
            observation = _observation_from_market(
                context=context,
                market=market,
                position_sizer=self.position_sizer,
                portfolio=self.portfolio,
                min_divergence=self.min_divergence,
                min_confidence=self.min_confidence,
                min_expected_edge=self.min_expected_edge,
                decay_window_hours=self.decay_window_hours,
                max_slippage_bps=self.max_slippage_bps,
                time_in_force=self.time_in_force,
            )
            if observation is not None:
                observations.append(observation)
        return tuple(observations)


@dataclass(frozen=True, slots=True)
class _AnchoringSignal:
    signal_name: str
    thesis: str
    token_id: str
    outcome: Outcome
    side: BookSide
    limit_price: float
    probability_estimate: float
    delta_effective: float


def _observation_from_market(
    *,
    context: StrategyContext,
    market: AnchoringMarketSnapshot,
    position_sizer: AnchoringPositionSizer,
    portfolio: Portfolio,
    min_divergence: float,
    min_confidence: float,
    min_expected_edge: float,
    decay_window_hours: float,
    max_slippage_bps: int,
    time_in_force: TimeInForce,
) -> StrategyObservation | None:
    signal = _classify_market(
        context=context,
        market=market,
        min_divergence=min_divergence,
        min_confidence=min_confidence,
        decay_window_hours=decay_window_hours,
    )
    if signal is None:
        return None

    expected_edge = signal.probability_estimate - signal.limit_price
    if expected_edge < min_expected_edge:
        return None
    notional_usdc = position_sizer.size(
        prob=signal.probability_estimate,
        market_price=signal.limit_price,
        portfolio=portfolio,
    )
    if notional_usdc <= 0.0:
        return None

    evidence_refs = _evidence_refs(market)
    metadata = {
        "source": LIVE_ANCHORING_SOURCE,
        "economics_version": ANCHORING_ECONOMICS_VERSION,
        "min_divergence": min_divergence,
        "min_confidence": min_confidence,
        "min_expected_edge": min_expected_edge,
        "decay_window_hours": decay_window_hours,
        "max_slippage_bps": max_slippage_bps,
        "h2_signal": signal.signal_name,
        "yes_price": market.yes_price,
        "llm_posterior": market.llm_posterior,
        "llm_confidence": market.llm_confidence,
        "news_timestamp": market.news_timestamp.isoformat(),
        "yes_best_ask": market.yes_best_ask,
        "no_best_ask": market.no_best_ask,
        "observed_at": market.observed_at.isoformat(),
        "resolves_at": market.resolves_at.isoformat() if market.resolves_at else None,
        "delta_effective": signal.delta_effective,
    }
    payload = {
        "market_id": market.market_id,
        "title": market.title,
        "thesis": signal.thesis,
        "probability_estimate": signal.probability_estimate,
        "expected_edge": expected_edge,
        "confidence": market.llm_confidence,
        "token_id": signal.token_id,
        "venue": market.venue,
        "side": signal.side,
        "outcome": signal.outcome,
        "limit_price": signal.limit_price,
        "notional_usdc": notional_usdc,
        "expected_price": signal.probability_estimate,
        "max_slippage_bps": max_slippage_bps,
        "time_in_force": time_in_force,
        "contradiction_refs": (),
        "metadata": metadata,
    }
    return StrategyObservation(
        observation_id=f"live-anchoring-{market.market_id}-{context.as_of.isoformat()}",
        strategy_id=context.strategy_id,
        strategy_version_id=context.strategy_version_id,
        source=LIVE_ANCHORING_SOURCE,
        observed_at=context.as_of,
        summary=signal.thesis,
        payload=payload,
        evidence_refs=evidence_refs,
    )


def _classify_market(
    *,
    context: StrategyContext,
    market: AnchoringMarketSnapshot,
    min_divergence: float,
    min_confidence: float,
    decay_window_hours: float,
) -> _AnchoringSignal | None:
    if market.llm_confidence < min_confidence:
        return None
    decay = _linear_decay(
        now=context.as_of,
        news_timestamp=market.news_timestamp,
        decay_window_hours=decay_window_hours,
    )
    delta_effective = (market.llm_posterior - market.yes_price) * decay
    if abs(delta_effective) <= min_divergence:
        return None

    effective_yes_probability = _bounded_probability(
        market.yes_price + delta_effective,
        "effective_yes_probability",
    )
    if delta_effective > 0.0:
        limit_price = _bounded_probability(
            market.yes_best_ask if market.yes_best_ask is not None else market.yes_price,
            "yes_limit_price",
        )
        return _AnchoringSignal(
            signal_name="positive_news_underreaction_buy_yes",
            thesis=(
                "H2 anchoring lag: LLM posterior is above market price after "
                "fresh news; buy YES exposure."
            ),
            token_id=market.yes_token_id,
            outcome="YES",
            side="BUY",
            limit_price=limit_price,
            probability_estimate=effective_yes_probability,
            delta_effective=delta_effective,
        )

    limit_price = _bounded_probability(
        market.no_best_ask if market.no_best_ask is not None else 1.0 - market.yes_price,
        "no_limit_price",
    )
    return _AnchoringSignal(
        signal_name="negative_news_underreaction_buy_no",
        thesis=(
            "H2 anchoring lag: LLM posterior is below market price after "
            "fresh news; buy NO exposure."
        ),
        token_id=market.no_token_id,
        outcome="NO",
        side="BUY",
        limit_price=limit_price,
        probability_estimate=_bounded_probability(
            1.0 - effective_yes_probability,
            "no_probability_estimate",
        ),
        delta_effective=delta_effective,
    )


def _is_resolved(as_of: datetime, resolves_at: datetime | None) -> bool:
    return resolves_at is not None and resolves_at <= as_of


def _evidence_refs(market: AnchoringMarketSnapshot) -> tuple[str, ...]:
    market_ref = f"market_snapshot:{market.market_id}:{market.observed_at.isoformat()}"
    news_ref = market.news_ref or f"news_timestamp:{market.news_timestamp.isoformat()}"
    return (ANCHORING_RESEARCH_REF, market_ref, news_ref)


def _linear_decay(
    *,
    now: datetime,
    news_timestamp: datetime,
    decay_window_hours: float,
) -> float:
    elapsed_hours = (now - news_timestamp).total_seconds() / 3600.0
    if elapsed_hours < 0.0:
        return 0.0
    return max(0.0, 1.0 - elapsed_hours / decay_window_hours)


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        msg = f"{field_name} must be non-empty"
        raise ValueError(msg)


def _require_probability(value: float, field_name: str) -> None:
    if value < 0.0 or value > 1.0 or value != value:
        msg = f"{field_name} must satisfy 0.0 <= value <= 1.0"
        raise ValueError(msg)


def _require_open_probability(value: float, field_name: str) -> None:
    if value <= 0.0 or value >= 1.0 or value != value:
        msg = f"{field_name} must satisfy 0.0 < value < 1.0"
        raise ValueError(msg)


def _bounded_probability(value: float, field_name: str) -> float:
    if value != value:
        msg = f"{field_name} must not be NaN"
        raise ValueError(msg)
    return min(max(float(value), 0.0001), 0.9999)
