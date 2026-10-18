from __future__ import annotations

from decimal import Decimal

from pms.core.models import MarketSignal
from pms.factors.base import FactorDefinition, FactorValueRow, OuterRingReader, is_no_token_signal


def _depth(levels: object) -> Decimal:
    if not isinstance(levels, list):
        return Decimal("0")

    total = Decimal("0")
    for level in levels:
        if not isinstance(level, dict):
            continue
        size = level.get("size")
        if size is None:
            continue
        total += Decimal(str(size))
    return total


class OrderbookImbalance(FactorDefinition):
    factor_id = "orderbook_imbalance"
    definition_version = "2"
    required_inputs = ("orderbook",)

    def compute(
        self,
        signal: MarketSignal,
        outer_ring: OuterRingReader,
    ) -> FactorValueRow | None:
        # Break points: total_depth == 0 returns None; one-sided books resolve to
        # +/- 1.0 because the observed side supplies the full signed depth.
        del outer_ring

        bid_depth = _depth(signal.orderbook.get("bids", []))
        ask_depth = _depth(signal.orderbook.get("asks", []))
        total_depth = bid_depth + ask_depth
        if total_depth == Decimal("0"):
            return None

        return FactorValueRow(
            factor_id=self.factor_id,
            param="",
            market_id=signal.market_id,
            ts=signal.timestamp,
            value=float((bid_depth - ask_depth) / total_depth) * (-1 if is_no_token_signal(signal) else 1),
        )
