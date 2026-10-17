from __future__ import annotations

from pms.core.models import MarketSignal
from pms.core.validation import validate_probability
from pms.factors.base import FactorDefinition, FactorValueRow, OuterRingReader


class MetaculusPrior(FactorDefinition):
    factor_id = "metaculus_prior"
    required_inputs = ("external_signal.metaculus_prob",)

    def compute(
        self,
        signal: MarketSignal,
        outer_ring: OuterRingReader,
    ) -> FactorValueRow | None:
        del outer_ring

        raw_metaculus_prob = signal.external_signal.get("metaculus_prob")
        if raw_metaculus_prob is None:
            return None

        return FactorValueRow(
            factor_id=self.factor_id,
            param="",
            market_id=signal.market_id,
            ts=signal.timestamp,
            value=validate_probability(
                raw_metaculus_prob,
                field_name="external_signal.metaculus_prob",
            ),
        )
