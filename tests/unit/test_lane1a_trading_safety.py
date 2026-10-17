from __future__ import annotations

from datetime import UTC, datetime
from math import nan
from pathlib import Path

import pytest

from pms.actuator.adapters.polymarket import (
    LiveOrderPreview,
    _approval_payload_matches,
    _compare_positions,
)
from pms.actuator.risk import RiskManager
from pms.config import RiskSettings
from pms.core.enums import TimeInForce
from pms.core.models import FillRecord, MarketSignal, Position, Portfolio, TradeDecision
from pms.runner import _portfolio_with_fill, _resolved_outcome, _risk_trade_result_from_fill
from pms.storage.live_emergency_audit import LiveEmergencyAuditWriter, read_active_halt


def _decision(**changes: object) -> TradeDecision:
    values: dict[str, object] = {
        "decision_id": "lane-1a-decision",
        "market_id": "lane-1a-market",
        "token_id": "lane-1a-token",
        "venue": "polymarket",
        "side": "BUY",
        "notional_usdc": 10.0,
        "order_type": "limit",
        "max_slippage_bps": 25,
        "stop_conditions": [],
        "prob_estimate": 0.6,
        "expected_edge": 0.1,
        "time_in_force": TimeInForce.IOC,
        "opportunity_id": "lane-1a-opportunity",
        "strategy_id": "lane-1a-strategy",
        "strategy_version_id": "lane-1a-v1",
        "limit_price": 0.5,
    }
    values.update(changes)
    return TradeDecision(**values)  # type: ignore[arg-type]


def _fill(*, side: str, price: float, notional: float, quantity: float) -> FillRecord:
    now = datetime(2026, 10, 18, tzinfo=UTC)
    return FillRecord(
        trade_id=f"trade-{side}-{price}",
        order_id=f"order-{side}-{price}",
        decision_id=f"decision-{side}-{price}",
        market_id="lane-1a-market",
        token_id="lane-1a-token",
        venue="polymarket",
        side=side,
        fill_price=price,
        fill_notional_usdc=notional,
        fill_quantity=quantity,
        executed_at=now,
        filled_at=now,
        status="filled",
        anomaly_flags=[],
        strategy_id="lane-1a-strategy",
        strategy_version_id="lane-1a-v1",
    )


def test_trade_decision_and_risk_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="notional_usdc"):
        _decision(notional_usdc=nan)

    decision = _decision()
    object.__setattr__(decision, "notional_usdc", nan)
    result = RiskManager().check(
        decision,
        Portfolio(100.0, 100.0, 0.0, []),
    )
    assert result.approved is False
    assert result.reason == "non_finite"


def test_live_loss_fill_updates_daily_loss_and_drawdown_inputs() -> None:
    opened = _fill(side="BUY", price=0.5, notional=50.0, quantity=100.0)
    closed = _fill(side="SELL", price=0.3, notional=30.0, quantity=100.0)
    portfolio = Portfolio(100.0, 100.0, 0.0, [])
    portfolio = _portfolio_with_fill(portfolio, opened)
    result = _risk_trade_result_from_fill(portfolio, closed)
    assert result is not None
    assert result.pnl == pytest.approx(-20.0)

    risk = RiskManager(RiskSettings(max_daily_loss_usdc=20.0))
    risk.record_trade_result(result)
    halt = risk.check_auto_halt(portfolio, now=result.filled_at)
    assert halt.halted is True
    assert halt.trigger_kind == "daily_loss_limit"


def test_malformed_resolution_metadata_cannot_abort_fill_accounting() -> None:
    signal = MarketSignal(
        market_id="lane-1a-market",
        token_id="lane-1a-token",
        venue="polymarket",
        title="lane-1a",
        yes_price=0.5,
        volume_24h=1.0,
        resolves_at=None,
        orderbook={},
        external_signal={"resolved_outcome": "not-a-probability"},
        fetched_at=datetime(2026, 10, 18, tzinfo=UTC),
        market_status="open",
    )
    assert _resolved_outcome(signal) is None


@pytest.mark.asyncio
async def test_halt_persistence_requires_explicit_clear(tmp_path: Path) -> None:
    path = tmp_path / "halt.jsonl"
    writer = LiveEmergencyAuditWriter(path)
    risk = RiskManager()
    state = risk._halt(  # noqa: SLF001
        trigger_kind="daily_loss_limit",
        reason="daily_loss_limit",
        triggered_at=datetime(2026, 10, 18, tzinfo=UTC),
        trace_id=None,
    )
    await writer.append_halt(state)
    assert read_active_halt(path) == state
    await writer.append_halt_clear(cleared_by="operator")
    assert read_active_halt(path) is None


def test_exit_approval_accepts_only_bounded_mark_drift() -> None:
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="lane-1a-market",
        token_id="lane-1a-token",
        side="SELL",
        limit_price=0.5,
        max_slippage_bps=25,
    )
    payload = {
        "approved": True,
        "venue": "polymarket",
        "market_id": "lane-1a-market",
        "token_id": "lane-1a-token",
        "side": "SELL",
        "outcome": "YES",
        "max_slippage_bps": 25,
        "max_notional_usdc": 10.01,
        "limit_price": 0.501,
    }
    assert _approval_payload_matches(payload, preview)
    assert not _approval_payload_matches({**payload, "limit_price": 0.53}, preview)


def test_venue_reconciliation_aggregates_strategy_rows_by_contract() -> None:
    def position(strategy: str, shares: float, locked: float) -> Position:
        return Position(
            market_id="lane-1a-market",
            token_id="lane-1a-token",
            venue="polymarket",
            side="BUY",
            shares_held=shares,
            avg_entry_price=locked / shares,
            unrealized_pnl=0.0,
            locked_usdc=locked,
            strategy_id=strategy,
            strategy_version_id=f"{strategy}-v1",
        )

    assert _compare_positions(
        [position("a", 10.0, 5.0), position("b", 20.0, 10.0)],
        [position("venue", 30.0, 15.0)],
        share_tolerance=1e-6,
        notional_tolerance=1e-6,
    ) == []
