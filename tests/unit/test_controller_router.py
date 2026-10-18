from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pms.config import ControllerSettings
from pms.controller.router import Router
from pms.core.models import MarketSignal


def _signal(**overrides: Any) -> MarketSignal:
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    data: dict[str, Any] = {
        "market_id": "router-market",
        "token_id": "router-token",
        "venue": "polymarket",
        "title": "Router market",
        "yes_price": 0.5,
        "volume_24h": 100.0,
        "resolves_at": now + timedelta(days=1),
        "orderbook": {"bids": [], "asks": []},
        "external_signal": {},
        "fetched_at": now,
        "market_status": "open",
    }
    data.update(overrides)
    return MarketSignal(**data)


def test_router_rejects_resolved_market() -> None:
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    assert not Router().gate(_signal(fetched_at=now, resolves_at=now))


def test_router_rejects_wide_spread_when_available() -> None:
    router = Router(ControllerSettings(max_spread_bps=100.0))
    assert not router.gate(_signal(external_signal={"spread_bps": 101.0}))


def test_router_rejects_stale_book_when_available() -> None:
    router = Router(ControllerSettings(max_book_age_ms=1_000.0))
    assert not router.gate(_signal(external_signal={"book_age_ms": 1_001.0}))


def test_router_rejects_non_open_status_from_signal_or_external_signal() -> None:
    router = Router()
    assert not router.gate(_signal(market_status="closed"))
    assert not router.gate(_signal(external_signal={"market_status": "halted"}))


def test_router_allows_signal_when_optional_quote_fields_are_absent() -> None:
    assert Router().gate(_signal())


def test_router_gate_reason_returns_none_when_signal_passes_all_gates() -> None:
    assert Router().gate_reason(_signal()) is None


def test_backtest_router_allows_resolved_price_extremes() -> None:
    router = Router(allow_extreme_prices=True)
    assert router.gate_reason(_signal(yes_price=0.0)) is None
    assert router.gate_reason(_signal(yes_price=0.999)) is None


def test_router_gate_reason_names_each_specific_gate_failure() -> None:
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    router_default = Router()
    router_capped = Router(
        ControllerSettings(
            min_volume=100.0,
            max_spread_bps=100.0,
            max_book_age_ms=1_000.0,
        )
    )

    assert (
        router_capped.gate_reason(_signal(volume_24h=50.0))
        == "min_volume_too_low"
    )
    assert (
        router_default.gate_reason(_signal(yes_price=0.01))
        == "yes_price_out_of_band"
    )
    assert (
        router_default.gate_reason(
            _signal(fetched_at=now, resolves_at=now)
        )
        == "resolves_at_in_past"
    )
    assert (
        router_default.gate_reason(_signal(market_status="halted"))
        == "market_status_not_open"
    )
    assert (
        router_capped.gate_reason(
            _signal(external_signal={"spread_bps": 101.0})
        )
        == "spread_too_wide"
    )
    assert (
        router_capped.gate_reason(
            _signal(external_signal={"book_age_ms": 1_001.0})
        )
        == "book_too_stale"
    )
