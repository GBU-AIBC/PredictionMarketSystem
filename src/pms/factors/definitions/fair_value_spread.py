from __future__ import annotations

from decimal import Decimal

from pms.core.models import MarketSignal
from pms.factors.base import FactorDefinition, FactorValueRow, OuterRingReader


class FairValueSpread(FactorDefinition):
    factor_id = "fair_value_spread"
    definition_version = "2"
    required_inputs = ("external_signal.fair_value", "yes_price")

    def compute(
        self,
        signal: MarketSignal,
        outer_ring: OuterRingReader,
    ) -> FactorValueRow | None:
        del outer_ring

        raw_fair_value = signal.external_signal.get("fair_value")
        if raw_fair_value is None:
            return None

        fair_value = Decimal(str(raw_fair_value))
        yes_price = Decimal(str(signal.yes_price))
        return FactorValueRow(
            factor_id=self.factor_id,
            param="",
            market_id=signal.market_id,
            ts=signal.timestamp,
            value=float(fair_value - yes_price),
        )
