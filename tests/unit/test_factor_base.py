from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any, cast, get_type_hints

import pytest

from pms.core.enums import MarketStatus
from pms.core.models import MarketSignal
from pms.factors import FactorDefinition, FactorValueRow, base
from pms.factors.base import EMPTY_OUTER_RING, OuterRingReader


def _signal() -> MarketSignal:
    return MarketSignal(
        market_id="factor-market",
        token_id="factor-token",
        venue="polymarket",
        title="Will the factor base compile?",
        yes_price=0.5,
        volume_24h=1000.0,
        resolves_at=datetime(2026, 4, 30, tzinfo=UTC),
        orderbook={"bids": [], "asks": []},
        external_signal={},
        fetched_at=datetime(2026, 4, 18, 1, 0, tzinfo=UTC),
        market_status=MarketStatus.OPEN.value,
    )


def test_factor_base_exports_expected_symbols() -> None:
    assert base.__all__ == (
        "EMPTY_OUTER_RING",
        "FactorDefinition",
        "FactorValueRow",
        "OuterRingReader",
        "is_no_token_signal",
    )
    assert getattr(FactorDefinition.compute, "__isabstractmethod__", False)
    assert inspect.iscoroutinefunction(OuterRingReader.read_latest_book_snapshot)


def test_factor_value_row_is_frozen() -> None:
    row = FactorValueRow(
        factor_id="orderbook_imbalance",
        param="",
        market_id="factor-market",
        ts=datetime(2026, 4, 18, 1, 0, tzinfo=UTC),
        value=0.25,
    )
    mutable_row = row

    with pytest.raises(FrozenInstanceError):
        cast_any = mutable_row  # keep the failing mutation runtime-visible for the test
        cast(Any, cast_any).value = 0.5


@pytest.mark.asyncio
async def test_empty_outer_ring_returns_none() -> None:
    assert await EMPTY_OUTER_RING.read_latest_book_snapshot("factor-market") is None


def test_market_signal_timestamp_alias_matches_fetched_at() -> None:
    signal = _signal()

    assert signal.timestamp == signal.fetched_at


def test_factor_value_row_annotations_match_contract() -> None:
    assert get_type_hints(FactorValueRow) == {
        "factor_id": str,
        "param": str,
        "market_id": str,
        "ts": datetime,
        "value": float,
    }
