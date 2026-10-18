from __future__ import annotations

from dataclasses import dataclass, field
import logging
from math import inf, isfinite
from typing import Any

from pms.config import ControllerSettings
from pms.core.models import MarketSignal


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Router:
    controller: ControllerSettings = field(default_factory=ControllerSettings)
    allow_extreme_prices: bool = False

    def gate(self, signal: MarketSignal) -> bool:
        reason = self.gate_reason(signal)
        passed = reason is None
        logger.info(
            "router funnel market_id=%s routed=%d",
            signal.market_id,
            int(passed),
            extra={
                "event": "funnel_router",
                "market_id": signal.market_id,
                "routed_count": int(passed),
            },
        )
        return passed

    def gate_reason(self, signal: MarketSignal) -> str | None:
        if (
            signal.volume_24h is not None
            and signal.volume_24h < self.controller.min_volume
        ):
            return "min_volume_too_low"
        if not isfinite(signal.yes_price):
            return "yes_price_out_of_band"
        if signal.yes_price < 0.0 or signal.yes_price > 1.0:
            return "yes_price_out_of_band"
        if not self.allow_extreme_prices and (
            signal.yes_price < 0.02 or signal.yes_price > 0.98
        ):
            return "yes_price_out_of_band"
        if signal.resolves_at is not None and signal.resolves_at <= signal.timestamp:
            return "resolves_at_in_past"
        market_status = str(
            signal.external_signal.get("market_status", signal.market_status)
        ).lower()
        if market_status not in {"open", "active"}:
            return "market_status_not_open"
        spread_bps = _optional_float(signal.external_signal.get("spread_bps"))
        if spread_bps is not None and spread_bps > self.controller.max_spread_bps:
            return "spread_too_wide"
        book_age_ms = _optional_float(signal.external_signal.get("book_age_ms"))
        if book_age_ms is not None and book_age_ms > self.controller.max_book_age_ms:
            return "book_too_stale"
        return None

    def stop_conditions(self, signal: MarketSignal) -> list[str]:
        price_band = "0.00-1.00" if self.allow_extreme_prices else "0.02-0.98"
        conditions = [
            f"min_volume:{self.controller.min_volume:.2f}",
            f"near_resolution_price_band:{price_band}",
        ]
        if signal.resolves_at is not None:
            conditions.append(f"resolves_at:{signal.resolves_at.isoformat()}")
        return conditions


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return inf
    return parsed if isfinite(parsed) else inf
