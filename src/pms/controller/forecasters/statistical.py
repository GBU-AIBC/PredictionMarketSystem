from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from pms.controller.factor_snapshot import (
    FactorSnapshotReader,
    NullFactorSnapshotReader,
)
from pms.core.models import MarketSignal
from pms.factors.catalog import catalog_entry
from pms.strategies.projections import FactorCompositionStep

ForecastResult = tuple[float, float, str]
MIN_STATISTICAL_PROBABILITY = 0.01
MAX_STATISTICAL_PROBABILITY = 0.99
STATISTICAL_MODEL_ID = "statistical-v1"
STATISTICAL_ROLES = frozenset(
    {
        "posterior_prior",
        "posterior_success",
        "posterior_failure",
        "weighted",
    }
)
EDGE_FACTOR_IDS = frozenset(
    {
        "anchoring_lag_divergence",
        "fair_value_spread",
        "favorite_longshot_bias",
    }
)
PROBABILITY_FACTOR_IDS = frozenset(
    {
        "metaculus_prior",
        "rules",
        "statistical",
        "llm",
    }
)


@dataclass(frozen=True)
class StatisticalForecaster:
    factor_reader: FactorSnapshotReader = field(
        default_factory=NullFactorSnapshotReader
    )
    composition: Sequence[FactorCompositionStep] = ()
    strategy_id: str = "default"
    strategy_version_id: str = "default-v1"
    prior_strength: float = 2.0

    def __post_init__(self) -> None:
        if self.prior_strength <= 0.0:
            msg = "prior_strength must be positive"
            raise ValueError(msg)

    def predict(self, signal: MarketSignal) -> ForecastResult | None:
        if not _enabled_statistical_steps(self.composition):
            return None
        return asyncio.run(self.apredict(signal))

    async def apredict(self, signal: MarketSignal) -> ForecastResult | None:
        statistical_steps = _enabled_statistical_steps(self.composition)
        if not statistical_steps:
            return None
        snapshot = await self.factor_reader.snapshot(
            market_id=signal.market_id,
            as_of=signal.timestamp,
            required=statistical_steps,
            strategy_id=self.strategy_id,
            strategy_version_id=self.strategy_version_id,
        )
        if _missing_required_keys(
            statistical_steps,
            snapshot.values,
            snapshot.missing_factors,
        ):
            return None
        return _forecast_from_factors(
            signal=signal,
            statistical_steps=statistical_steps,
            factor_values=snapshot.values,
            prior_strength=self.prior_strength,
        )

    async def forecast(self, signal: MarketSignal) -> float:
        result = await self.apredict(signal)
        return signal.yes_price if result is None else result[0]


def _enabled_statistical_steps(
    composition: Sequence[FactorCompositionStep],
) -> tuple[FactorCompositionStep, ...]:
    return tuple(
        step for step in composition if step.enabled and step.role in STATISTICAL_ROLES
    )


def _missing_required_keys(
    steps: Sequence[FactorCompositionStep],
    factor_values: Mapping[tuple[str, str], float],
    reported_missing: Sequence[tuple[str, str]],
) -> bool:
    reported = set(reported_missing)
    return any(
        step.required
        and (
            (step.factor_id, step.param) in reported
            or (step.factor_id, step.param) not in factor_values
        )
        for step in steps
    )


def _forecast_from_factors(
    *,
    signal: MarketSignal,
    statistical_steps: Sequence[FactorCompositionStep],
    factor_values: Mapping[tuple[str, str], float],
    prior_strength: float,
) -> ForecastResult | None:
    candidates: list[tuple[float, float]] = []
    posterior = _posterior_candidate(
        statistical_steps,
        factor_values,
        prior_strength=prior_strength,
    )
    if posterior is not None:
        candidates.append(posterior)
    for step in statistical_steps:
        if step.role not in {"posterior_prior", "weighted"}:
            continue
        if posterior is not None and step.role == "posterior_prior":
            continue
        factor_value = factor_values.get((step.factor_id, step.param))
        if factor_value is None:
            continue
        probability = _factor_probability(
            factor_id=step.factor_id,
            value=factor_value,
            market_price=signal.yes_price,
        )
        if probability is None:
            continue
        candidates.append((_bounded_probability(probability), step.weight))
    if not candidates:
        return None
    probability = _weighted_average(candidates)
    probabilities = tuple(candidate[0] for candidate in candidates)
    weights = tuple(candidate[1] for candidate in candidates)
    return probability, _confidence(probabilities, weights), STATISTICAL_MODEL_ID


def _posterior_candidate(
    steps: Sequence[FactorCompositionStep],
    factor_values: Mapping[tuple[str, str], float],
    *,
    prior_strength: float,
) -> tuple[float, float] | None:
    success_steps = tuple(step for step in steps if step.role == "posterior_success")
    failure_steps = tuple(step for step in steps if step.role == "posterior_failure")
    if not success_steps and not failure_steps:
        return None
    successes = sum(
        (
            Decimal(str(step.weight))
            * Decimal(str(factor_values.get((step.factor_id, step.param), 0.0)))
            for step in success_steps
        ),
        Decimal("0"),
    )
    failures = sum(
        (
            Decimal(str(step.weight))
            * Decimal(str(factor_values.get((step.factor_id, step.param), 0.0)))
            for step in failure_steps
        ),
        Decimal("0"),
    )
    if successes == 0 and failures == 0:
        return None
    prior_steps = tuple(step for step in steps if step.role == "posterior_prior")
    prior_probability = Decimal("0.5")
    prior_weight = Decimal(str(prior_strength))
    if prior_steps:
        prior_step = prior_steps[0]
        prior_probability = Decimal(
            str(
                factor_values.get(
                    (prior_step.factor_id, prior_step.param),
                    0.5,
                )
            )
        )
        prior_weight = Decimal(str(prior_step.weight))
    alpha = prior_probability * prior_weight + successes
    beta = (Decimal("1.0") - prior_probability) * prior_weight + failures
    total = alpha + beta
    if total == 0:
        return None
    return (
        _bounded_probability(float(alpha / total)),
        float(max(Decimal("1.0"), successes + failures)),
    )


def _factor_probability(
    *,
    factor_id: str,
    value: float,
    market_price: float,
) -> float | None:
    if factor_id in PROBABILITY_FACTOR_IDS or _catalog_output_type(factor_id) == "probability":
        return value
    if factor_id in EDGE_FACTOR_IDS:
        return market_price + value
    return None


def _catalog_output_type(factor_id: str) -> str | None:
    entry = catalog_entry(factor_id)
    return None if entry is None else entry.output_type


def _weighted_average(candidates: Sequence[tuple[float, float]]) -> float:
    total_weight = sum((Decimal(str(weight)) for _, weight in candidates), Decimal("0"))
    if total_weight == 0:
        total_probability = sum(
            (Decimal(str(probability)) for probability, _ in candidates),
            Decimal("0"),
        )
        return float(total_probability / Decimal(len(candidates)))
    weighted_sum = sum(
        (
            Decimal(str(probability)) * Decimal(str(weight))
            for probability, weight in candidates
        ),
        Decimal("0"),
    )
    return float(weighted_sum / total_weight)


def _confidence(probabilities: Sequence[float], weights: Sequence[float]) -> float:
    if len(probabilities) == 1:
        return 0.5
    probability_decimals = tuple(
        Decimal(str(probability)) for probability in probabilities
    )
    weight_decimals = tuple(Decimal(str(weight)) for weight in weights)
    total_weight = sum(weight_decimals, Decimal("0"))
    if total_weight == 0:
        weight_decimals = tuple(Decimal("1.0") for _ in probabilities)
        total_weight = Decimal(len(probabilities))
    mean = (
        sum(
            probability * weight
            for probability, weight in zip(
                probability_decimals,
                weight_decimals,
                strict=True,
            )
        )
        / total_weight
    )
    mean_abs_deviation = (
        sum(
            abs(probability - mean) * weight
            for probability, weight in zip(
                probability_decimals,
                weight_decimals,
                strict=True,
            )
        )
        / total_weight
    )
    agreement = max(
        Decimal("0"),
        Decimal("1.0") - (mean_abs_deviation / Decimal("0.5")),
    )
    return float(min(Decimal("0.95"), Decimal("0.35") + Decimal("0.60") * agreement))


def _bounded_probability(value: float) -> float:
    return max(MIN_STATISTICAL_PROBABILITY, min(MAX_STATISTICAL_PROBABILITY, value))
