from __future__ import annotations

from decimal import Decimal

from pms.core.models import MarketSignal
from pms.factors.base import FactorDefinition, FactorValueRow, OuterRingReader


LONGSHOT_YES_THRESHOLD = Decimal("0.10")
FAVORITE_YES_THRESHOLD = Decimal("0.90")


class FavoriteLongshotBias(FactorDefinition):
    """Signed H1 contrarian signal from the market YES price.

    Negative values mean the YES longshot bucket is overpriced, so the
    actionable contrarian side is buy NO. Positive values mean the YES favorite
    bucket is underpriced, so the actionable side is buy YES.
    """

    factor_id = "favorite_longshot_bias"
    definition_version = "2"
    required_inputs = ("yes_price",)

    def compute(
        self,
        signal: MarketSignal,
        outer_ring: OuterRingReader,
    ) -> FactorValueRow | None:
        del outer_ring

        yes_price = Decimal(str(signal.yes_price))
        if yes_price < LONGSHOT_YES_THRESHOLD:
            value = yes_price - LONGSHOT_YES_THRESHOLD
        elif yes_price > FAVORITE_YES_THRESHOLD:
            value = yes_price - FAVORITE_YES_THRESHOLD
        else:
            # A raw factor row is a time series.  Persisting an explicit
            # neutral value clears any previously active tail signal from the
            # snapshot instead of leaving the last tail value live.
            value = Decimal("0")

        return FactorValueRow(
            factor_id=self.factor_id,
            param="",
            market_id=signal.market_id,
            ts=signal.timestamp,
            value=float(value),
        )
