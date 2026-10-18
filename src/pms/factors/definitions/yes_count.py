from __future__ import annotations

from pms.core.models import MarketSignal
from pms.core.validation import validate_quantity
from pms.factors.base import FactorDefinition, FactorValueRow, OuterRingReader


class YesCount(FactorDefinition):
    factor_id = "yes_count"
    required_inputs = ("external_signal.yes_count",)

    def compute(
        self,
        signal: MarketSignal,
        outer_ring: OuterRingReader,
    ) -> FactorValueRow | None:
        del outer_ring

        raw_yes_count = signal.external_signal.get("yes_count")
        if raw_yes_count is None:
            return None

        return FactorValueRow(
            factor_id=self.factor_id,
            param="",
            market_id=signal.market_id,
            ts=signal.timestamp,
            value=validate_quantity(raw_yes_count, field_name="external_signal.yes_count"),
        )
