from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from pms.config import PMSSettings, RiskSettings
from pms.core.models import Portfolio
from pms.core.validation import validate_probability, validate_price, validate_finite

# Kelly sizing is implemented directly because the checkpoint's fee-aware
# binary formula is authoritative; third-party packages may use incompatible
# odds conventions.


@dataclass(frozen=True)
class KellySizer:
    risk: RiskSettings | None = None
    fraction: Decimal = Decimal("0.25")

    def __post_init__(self) -> None:
        if self.risk is None:
            object.__setattr__(self, "risk", PMSSettings().risk)

    def size(
        self,
        *,
        prob: float,
        market_price: float,
        portfolio: Portfolio,
        cost_edge: float = 0.0,
    ) -> float:
        try:
            probability = validate_probability(prob, field_name="probability")
            effective_price = validate_price(
                validate_finite(market_price, field_name="market_price")
                + validate_finite(cost_edge, field_name="cost_edge"),
                field_name="effective market price",
            )
        except ValueError:
            return 0.0
        p = min(Decimal("0.99"), max(Decimal("0.01"), Decimal(str(probability))))
        q = Decimal("1") - p
        price = Decimal(str(effective_price))
        b = (Decimal("1") - price) / price
        kelly_fraction = (p * b - q) / b
        if kelly_fraction <= 0:
            return 0.0
        scaled_fraction = kelly_fraction * self.fraction
        raw_size = Decimal(str(portfolio.free_usdc)) * scaled_fraction
        if self.risk is None:
            msg = "KellySizer risk is not initialized"
            raise RuntimeError(msg)
        capped = min(raw_size, Decimal(str(self.risk.max_position_per_market)))
        return float(capped)
