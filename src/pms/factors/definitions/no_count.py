from __future__ import annotations

from pms.core.models import MarketSignal
from pms.core.validation import validate_quantity
from pms.factors.base import FactorDefinition, FactorValueRow, OuterRingReader


class NoCount(FactorDefinition):
    factor_id = "no_count"
    required_inputs = ("external_signal.no_count",)

    def compute(
        self,
        signal: MarketSignal,
        outer_ring: OuterRingReader,
    ) -> FactorValueRow | None:
        del outer_ring

        raw_no_count = signal.external_signal.get("no_count")
        if raw_no_count is None:
            return None

        return FactorValueRow(
            factor_id=self.factor_id,
            param="",
            market_id=signal.market_id,
            ts=signal.timestamp,
            value=validate_quantity(raw_no_count, field_name="external_signal.no_count"),
        )
