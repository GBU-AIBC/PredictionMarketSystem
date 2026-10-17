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
from pms.factors.catalog import FACTOR_CATALOG_ROWS
from pms.strategies.projections import FactorCompositionStep

ForecastResult = tuple[float, float, str]
MIN_RULE_PROBABILITY = 0.01
MAX_RULE_PROBABILITY = 0.99
RULES_MODEL_ID = "rules-v1"


@dataclass(frozen=True)
class RulesForecaster:
    factor_reader: FactorSnapshotReader = field(
        default_factory=NullFactorSnapshotReader
    )
    composition: Sequence[FactorCompositionStep] = ()
    strategy_id: str = "default"
    strategy_version_id: str = "default-v1"
    min_edge: float = 0.02

    def predict(self, signal: MarketSignal) -> ForecastResult | None:
        if not _enabled_rule_steps(self.composition):
            return None
        return asyncio.run(self.apredict(signal))

    async def apredict(self, signal: MarketSignal) -> ForecastResult | None:
        rule_steps = _enabled_rule_steps(self.composition)
        if not rule_steps:
            return None
        snapshot = await self.factor_reader.snapshot(
            market_id=signal.market_id,
            as_of=signal.timestamp,
            required=rule_steps,
            strategy_id=self.strategy_id,
            strategy_version_id=self.strategy_version_id,
        )
        return _forecast_from_rules(
            signal=signal,
            rule_steps=rule_steps,
            factor_values=snapshot.values,
        )

    async def forecast(self, signal: MarketSignal) -> float:
        result = await self.apredict(signal)
        return signal.yes_price if result is None else result[0]


def _enabled_rule_steps(
    composition: Sequence[FactorCompositionStep],
) -> tuple[FactorCompositionStep, ...]:
    return tuple(
        step for step in composition if step.role == "rule_delta" and step.enabled
    )


def _forecast_from_rules(
    *,
    signal: MarketSignal,
    rule_steps: Sequence[FactorCompositionStep],
    factor_values: Mapping[tuple[str, str], float],
) -> ForecastResult | None:
    probability = Decimal(str(signal.yes_price))
    max_abs_contribution = Decimal("0")
    for step in rule_steps:
        factor_value = factor_values.get((step.factor_id, step.param))
        if factor_value is None:
            if step.required:
                return None
            continue
        delta = Decimal(
            str(_rule_delta(step, factor_value, market_price=signal.yes_price))
        )
        if step.threshold is not None and abs(delta) < Decimal(str(step.threshold)):
            continue
        contribution = delta * Decimal(str(step.weight))
        probability += contribution
        max_abs_contribution = max(max_abs_contribution, abs(contribution))
    return (
        _bounded_probability(float(probability)),
        float(min(max_abs_contribution * Decimal("5.0"), Decimal("0.95"))),
        RULES_MODEL_ID,
    )


def _rule_delta(
    step: FactorCompositionStep,
    factor_value: float,
    *,
    market_price: float,
) -> float:
    if _catalog_output_type(step.factor_id) == "probability":
        return factor_value - market_price
    if step.factor_id == "subset_pricing_violation":
        return -factor_value
    return factor_value


def _catalog_output_type(factor_id: str) -> str | None:
    for entry in FACTOR_CATALOG_ROWS:
        if entry.factor_id == factor_id:
            return entry.output_type
    return None


def _bounded_probability(value: float) -> float:
    return max(MIN_RULE_PROBABILITY, min(MAX_RULE_PROBABILITY, value))
