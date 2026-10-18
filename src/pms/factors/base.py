from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Protocol

from pms.core.models import BookSnapshot, MarketSignal

__all__ = (
    "EMPTY_OUTER_RING",
    "FactorDefinition",
    "FactorValueRow",
    "OuterRingReader",
    "is_no_token_signal",
)


@dataclass(frozen=True, slots=True)
class FactorValueRow:
    factor_id: str
    param: str
    market_id: str
    ts: datetime
    value: float


class OuterRingReader(Protocol):
    async def read_latest_book_snapshot(self, market_id: str) -> BookSnapshot | None: ...


class _EmptyOuterRing:
    async def read_latest_book_snapshot(self, market_id: str) -> BookSnapshot | None:
        return None


EMPTY_OUTER_RING: OuterRingReader = _EmptyOuterRing()


class FactorDefinition(ABC):
    """Signal-first factor contract with an outer-ring lookback seam."""

    factor_id: ClassVar[str]
    required_inputs: ClassVar[tuple[str, ...]]
    definition_version: ClassVar[str] = "1"

    @abstractmethod
    def compute(
        self,
        signal: MarketSignal,
        outer_ring: OuterRingReader,
    ) -> FactorValueRow | None:
        """Read current inputs from signal and optional history from outer_ring."""


def is_no_token_signal(signal: MarketSignal) -> bool:
    no_token_id = signal.external_signal.get("no_token_id")
    yes_token_id = signal.external_signal.get("yes_token_id")
    if no_token_id and signal.token_id == no_token_id:
        return True
    if yes_token_id and signal.token_id == yes_token_id:
        return False
    return str(signal.external_signal.get("signal_token_outcome", "YES")).upper() == "NO"
