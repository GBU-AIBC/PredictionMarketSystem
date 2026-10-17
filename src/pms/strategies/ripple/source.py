"""Observation sources for the ripple strategy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from pms.core.enums import TimeInForce
from pms.core.models import BookSide, Outcome, Portfolio, Venue
from pms.strategies.intents import StrategyContext, StrategyObservation
from pms.strategies.aggregate import Strategy
from pms.strategies.projections import (
    EvalSpec,
    FactorCompositionStep,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)
from pms.strategies.ripple.evaluator import (
    DEFAULT_MIN_EXPECTED_EDGE,
    DEFAULT_PRIOR_STRENGTH,
    beta_binomial_posterior_probability,
    entry_edge_threshold,
    posterior_confidence,
)


LIVE_RIPPLE_SOURCE = "live_factor_service"
RIPPLE_ECONOMICS_VERSION = "ripple-economics-v1"
RIPPLE_MAX_POSITION_NOTIONAL_USDC = 1.0
RIPPLE_DEFAULT_PRIOR_STRENGTH = DEFAULT_PRIOR_STRENGTH
RIPPLE_DEFAULT_MIN_EXPECTED_EDGE = DEFAULT_MIN_EXPECTED_EDGE
RIPPLE_DEFAULT_MAX_SLIPPAGE_BPS = 50
RIPPLE_DEFAULT_TIME_IN_FORCE = TimeInForce.GTC
RIPPLE_FACTOR_REQUIREMENTS: tuple[FactorCompositionStep, ...] = (
    FactorCompositionStep(
        factor_id="metaculus_prior",
        role="posterior_prior",
        param="",
        weight=2.0,
        threshold=None,
        required=True,
        freshness_sla_s=3600.0,
    ),
    FactorCompositionStep(
        factor_id="orderbook_imbalance",
        role="weighted",
        param="",
        weight=1.0,
        threshold=None,
        required=True,
        freshness_sla_s=30.0,
    ),
    FactorCompositionStep(
        factor_id="fair_value_spread",
        role="threshold_edge",
        param="",
        weight=1.0,
        threshold=0.0,
        required=True,
        freshness_sla_s=3600.0,
    ),
    FactorCompositionStep(
        factor_id="yes_count",
        role="posterior_success",
        param="",
        weight=1.0,
        threshold=None,
        required=True,
        freshness_sla_s=3600.0,
    ),
    FactorCompositionStep(
        factor_id="no_count",
        role="posterior_failure",
        param="",
        weight=1.0,
        threshold=None,
        required=True,
        freshness_sla_s=3600.0,
    ),
)


def build_ripple_strategy(strategy_id: str, *, time_in_force: str = "GTC") -> Strategy:
    """Build the immutable projection that owns Ripple's runtime economics."""
    return Strategy(
        config=StrategyConfig(
            strategy_id=strategy_id,
            factor_composition=RIPPLE_FACTOR_REQUIREMENTS,
            metadata=(
                ("economics_version", RIPPLE_ECONOMICS_VERSION),
                ("prior_strength", str(RIPPLE_DEFAULT_PRIOR_STRENGTH)),
                ("min_expected_edge", str(RIPPLE_DEFAULT_MIN_EXPECTED_EDGE)),
                ("max_slippage_bps", str(RIPPLE_DEFAULT_MAX_SLIPPAGE_BPS)),
                ("time_in_force", time_in_force.upper()),
            ),
        ),
        risk=RiskParams(
            max_position_notional_usdc=RIPPLE_MAX_POSITION_NOTIONAL_USDC,
            max_daily_drawdown_pct=20.0,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(metrics=("pnl", "fill_rate")),
        forecaster=ForecasterSpec(forecasters=(("ripple", ()),)),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=0.0,
        ),
    )


FactorKey = tuple[str, str]


class RippleFactorSnapshot(Protocol):
    values: Mapping[FactorKey, float]
    missing_factors: tuple[FactorKey, ...]
    stale_factors: tuple[FactorKey, ...]
    snapshot_hash: str | None


class RippleFactorSnapshotReader(Protocol):
    async def snapshot(
        self,
        *,
        market_id: str,
        as_of: datetime,
        required: Sequence[FactorCompositionStep],
        strategy_id: str,
        strategy_version_id: str,
    ) -> RippleFactorSnapshot: ...


class RippleMarketSnapshotReader(Protocol):
    async def latest(
        self,
        market_id: str,
        *,
        as_of: datetime,
    ) -> RippleMarketSnapshot | None: ...


@runtime_checkable
class RipplePositionSizer(Protocol):
    def size(
        self,
        *,
        prob: float,
        market_price: float,
        portfolio: Portfolio,
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class RippleMarketSnapshot:
    market_id: str
    title: str
    token_id: str
    yes_price: float
    observed_at: datetime
    best_bid: float | None = None
    best_ask: float | None = None
    resolves_at: datetime | None = None
    venue: Venue = "polymarket"

    def __post_init__(self) -> None:
        _require_non_empty(self.market_id, "market_id")
        _require_non_empty(self.title, "title")
        _require_non_empty(self.token_id, "token_id")
        _require_open_probability(self.yes_price, "yes_price")
        if self.best_bid is not None:
            _require_open_probability(self.best_bid, "best_bid")
        if self.best_ask is not None:
            _require_open_probability(self.best_ask, "best_ask")


@dataclass(frozen=True, slots=True)
class RippleObservationFixture:
    observation_id: str
    market_id: str
    title: str
    thesis: str
    probability_estimate: float
    expected_edge: float
    confidence: float
    token_id: str
    limit_price: float
    notional_usdc: float
    expected_price: float
    max_slippage_bps: int
    evidence_refs: tuple[str, ...]
    contradiction_refs: tuple[str, ...] = ()
    venue: Venue = "polymarket"
    side: BookSide = "BUY"
    outcome: Outcome = "YES"
    time_in_force: TimeInForce = TimeInForce.GTC
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.observation_id, "observation_id")
        _require_non_empty(self.market_id, "market_id")
        _require_non_empty(self.title, "title")
        _require_non_empty(self.thesis, "thesis")
        _require_non_empty(self.token_id, "token_id")
        if not 0.0 <= self.probability_estimate <= 1.0:
            msg = "probability_estimate must satisfy 0.0 <= value <= 1.0"
            raise ValueError(msg)
        if not 0.0 <= self.confidence <= 1.0:
            msg = "confidence must satisfy 0.0 <= value <= 1.0"
            raise ValueError(msg)
        if self.notional_usdc <= 0.0:
            msg = "notional_usdc must be > 0.0"
            raise ValueError(msg)
        if self.max_slippage_bps < 0:
            msg = "max_slippage_bps must be >= 0"
            raise ValueError(msg)

    def payload(self) -> Mapping[str, Any]:
        return {
            "market_id": self.market_id,
            "title": self.title,
            "thesis": self.thesis,
            "probability_estimate": self.probability_estimate,
            "expected_edge": self.expected_edge,
            "confidence": self.confidence,
            "token_id": self.token_id,
            "venue": self.venue,
            "side": self.side,
            "outcome": self.outcome,
            "limit_price": self.limit_price,
            "notional_usdc": self.notional_usdc,
            "expected_price": self.expected_price,
            "max_slippage_bps": self.max_slippage_bps,
            "time_in_force": self.time_in_force,
            "contradiction_refs": self.contradiction_refs,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RippleObservationSource:
    fixtures: Sequence[RippleObservationFixture]

    async def observe(self, context: StrategyContext) -> Sequence[StrategyObservation]:
        return tuple(
            StrategyObservation(
                observation_id=fixture.observation_id,
                strategy_id=context.strategy_id,
                strategy_version_id=context.strategy_version_id,
                source="ripple-fixture",
                observed_at=context.as_of,
                summary=fixture.thesis,
                payload=fixture.payload(),
                evidence_refs=fixture.evidence_refs,
            )
            for fixture in self.fixtures
        )


@dataclass(frozen=True, slots=True)
class LiveRippleSource:
    """Live factor-backed source.

    The concrete production factor reader should be
    ``PostgresFactorSnapshotReader`` from the controller layer. This strategy
    plugin keeps that dependency injected to preserve the strategy boundary
    enforced by the import-linter contract.
    """

    market_ids: Sequence[str]
    factor_reader: RippleFactorSnapshotReader
    market_reader: RippleMarketSnapshotReader
    position_sizer: RipplePositionSizer
    portfolio: Portfolio
    prior_strength: float = DEFAULT_PRIOR_STRENGTH
    min_expected_edge: float = DEFAULT_MIN_EXPECTED_EDGE
    max_slippage_bps: int = 50
    time_in_force: TimeInForce = TimeInForce.GTC

    def __post_init__(self) -> None:
        if not self.market_ids:
            msg = "market_ids must not be empty"
            raise ValueError(msg)
        for market_id in self.market_ids:
            _require_non_empty(market_id, "market_id")
        if self.prior_strength <= 0.0:
            msg = "prior_strength must be > 0.0"
            raise ValueError(msg)
        if self.min_expected_edge <= 0.0:
            msg = "min_expected_edge must be > 0.0"
            raise ValueError(msg)
        if self.max_slippage_bps < 0:
            msg = "max_slippage_bps must be >= 0"
            raise ValueError(msg)

    async def observe(self, context: StrategyContext) -> Sequence[StrategyObservation]:
        observations: list[StrategyObservation] = []
        for market_id in self.market_ids:
            market = await self.market_reader.latest(
                market_id,
                as_of=context.as_of,
            )
            if market is None:
                continue
            snapshot = await self.factor_reader.snapshot(
                market_id=market_id,
                as_of=context.as_of,
                required=RIPPLE_FACTOR_REQUIREMENTS,
                strategy_id=context.strategy_id,
                strategy_version_id=context.strategy_version_id,
            )
            observation = _live_observation(
                context=context,
                market=market,
                snapshot=snapshot,
                position_sizer=self.position_sizer,
                portfolio=self.portfolio,
                prior_strength=self.prior_strength,
                min_expected_edge=self.min_expected_edge,
                max_slippage_bps=self.max_slippage_bps,
                time_in_force=self.time_in_force,
            )
            if observation is not None:
                observations.append(observation)
        return tuple(observations)


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        msg = f"{field_name} must be non-empty"
        raise ValueError(msg)


def _require_open_probability(value: float, field_name: str) -> None:
    if value <= 0.0 or value >= 1.0:
        msg = f"{field_name} must satisfy 0.0 < value < 1.0"
        raise ValueError(msg)


def _live_observation(
    *,
    context: StrategyContext,
    market: RippleMarketSnapshot,
    snapshot: RippleFactorSnapshot,
    position_sizer: RipplePositionSizer,
    portfolio: Portfolio,
    prior_strength: float,
    min_expected_edge: float,
    max_slippage_bps: int,
    time_in_force: TimeInForce,
) -> StrategyObservation | None:
    prior_key = ("metaculus_prior", "")
    if (
        prior_key in snapshot.missing_factors
        or prior_key in snapshot.stale_factors
        or prior_key not in snapshot.values
        or snapshot.snapshot_hash is None
    ):
        # Ripple's posterior is only meaningful when its external prior was
        # actually observed in the point-in-time snapshot.  Falling back to
        # the market price turns missing evidence into false provenance.
        return None
    prior_probability = _bounded_probability(
        _factor_value(snapshot, "metaculus_prior", fallback=market.yes_price),
        "metaculus_prior",
    )
    yes_count = _non_negative_count(snapshot, "yes_count")
    no_count = _non_negative_count(snapshot, "no_count")
    probability_estimate = beta_binomial_posterior_probability(
        prior_probability=prior_probability,
        prior_strength=prior_strength,
        yes_count=yes_count,
        no_count=no_count,
    )
    orderbook_imbalance = _factor_value(snapshot, "orderbook_imbalance", fallback=0.0)
    limit_price = _bounded_probability(
        market.best_ask if market.best_ask is not None else market.yes_price,
        "limit_price",
    )
    expected_edge = probability_estimate - limit_price
    dynamic_entry_threshold = entry_edge_threshold(
        as_of=context.as_of,
        resolves_at=market.resolves_at,
        min_expected_edge=min_expected_edge,
    )
    confidence = _live_confidence(
        snapshot,
        prior_strength=prior_strength,
        yes_count=yes_count,
        no_count=no_count,
    )
    notional_usdc = position_sizer.size(
        prob=probability_estimate,
        market_price=limit_price,
        portfolio=portfolio,
    )
    if notional_usdc <= 0.0:
        return None
    evidence_refs = _live_evidence_refs(market, snapshot)
    payload_metadata = {
        "source": LIVE_RIPPLE_SOURCE,
        "economics_version": RIPPLE_ECONOMICS_VERSION,
        "prior_strength": prior_strength,
        "min_expected_edge": min_expected_edge,
        "max_slippage_bps": max_slippage_bps,
        "factor_snapshot_hash": snapshot.snapshot_hash,
        "factor_values": _string_factor_values(snapshot.values),
        "missing_factors": _string_factor_keys(snapshot.missing_factors),
        "stale_factors": _string_factor_keys(snapshot.stale_factors),
        "yes_price": market.yes_price,
        "best_bid": market.best_bid,
        "best_ask": market.best_ask,
        "observed_at": market.observed_at.isoformat(),
        "resolves_at": market.resolves_at.isoformat() if market.resolves_at else None,
        "orderbook_imbalance": orderbook_imbalance,
        "metaculus_prior": prior_probability,
        "yes_count": yes_count,
        "no_count": no_count,
        "posterior_probability": probability_estimate,
        "entry_edge_threshold": dynamic_entry_threshold,
    }
    payload = {
        "market_id": market.market_id,
        "title": market.title,
        "thesis": (
            "Live factor snapshot supports a YES edge of "
            f"{expected_edge:.4f} against market price {market.yes_price:.4f}."
        ),
        "probability_estimate": probability_estimate,
        "expected_edge": expected_edge,
        "confidence": confidence,
        "token_id": market.token_id,
        "venue": market.venue,
        "side": "BUY",
        "outcome": "YES",
        "limit_price": limit_price,
        "notional_usdc": notional_usdc,
        "expected_price": probability_estimate,
        "max_slippage_bps": max_slippage_bps,
        "time_in_force": time_in_force,
        "contradiction_refs": (),
        "metadata": payload_metadata,
    }
    return StrategyObservation(
        observation_id=(
            f"live-ripple-{market.market_id}-{context.as_of.isoformat()}"
        ),
        strategy_id=context.strategy_id,
        strategy_version_id=context.strategy_version_id,
        source=LIVE_RIPPLE_SOURCE,
        observed_at=context.as_of,
        summary=str(payload["thesis"]),
        payload=payload,
        evidence_refs=evidence_refs,
    )


def _factor_value(
    snapshot: RippleFactorSnapshot,
    factor_id: str,
    *,
    fallback: float,
) -> float:
    value = snapshot.values.get((factor_id, ""))
    if value is None:
        return fallback
    return float(value)


def _non_negative_count(snapshot: RippleFactorSnapshot, factor_id: str) -> float:
    value = _factor_value(snapshot, factor_id, fallback=0.0)
    if value < 0.0:
        msg = f"{factor_id} must be >= 0.0"
        raise ValueError(msg)
    return value


def _bounded_probability(value: float, field_name: str) -> float:
    if value <= 0.0:
        return 0.0001
    if value >= 1.0:
        return 0.9999
    if value != value:
        msg = f"{field_name} must not be NaN"
        raise ValueError(msg)
    return float(value)


def _live_confidence(
    snapshot: RippleFactorSnapshot,
    *,
    prior_strength: float,
    yes_count: float,
    no_count: float,
) -> float:
    return posterior_confidence(
        prior_strength=prior_strength,
        yes_count=yes_count,
        no_count=no_count,
        degraded=bool(snapshot.missing_factors or snapshot.stale_factors),
    )


def _live_evidence_refs(
    market: RippleMarketSnapshot,
    snapshot: RippleFactorSnapshot,
) -> tuple[str, ...]:
    factor_ref = (
        f"factor_snapshot:{snapshot.snapshot_hash}"
        if snapshot.snapshot_hash
        else "factor_snapshot:unhashed"
    )
    market_ref = (
        f"market_snapshot:{market.market_id}:{market.observed_at.isoformat()}"
    )
    return (factor_ref, market_ref)


def _string_factor_values(values: Mapping[FactorKey, float]) -> dict[str, float]:
    return {
        _factor_key_label(key): float(values[key])
        for key in sorted(values)
    }


def _string_factor_keys(keys: Sequence[FactorKey]) -> tuple[str, ...]:
    return tuple(_factor_key_label(key) for key in keys)


def _factor_key_label(key: FactorKey) -> str:
    factor_id, param = key
    return factor_id if not param else f"{factor_id}:{param}"
