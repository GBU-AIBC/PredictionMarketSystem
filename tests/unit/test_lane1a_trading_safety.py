from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import asyncpg
from datetime import UTC, datetime
from math import nan
from pathlib import Path
from typing import Any, cast

import pytest

from pms import live_cli
from pms.actuator.adapters.polymarket import (
    LiveOrderPreview,
    LivePreSubmitQuote,
    OperatorApprovalRequiredError,
    PolymarketActuator,
    PolymarketOrderResult,
    PolymarketSubmissionUnknownError,
    _order_request,
    _approval_payload_matches,
    _compare_positions,
)
from pms.actuator.executor import ActuatorExecutor
from pms.actuator.feedback import ActuatorFeedback
from pms.actuator.risk import RiskManager, RiskTradeResult
from pms.config import ControllerSettings, PMSSettings, PolymarketSettings, RiskSettings
from pms.core.enums import OrderStatus, RunMode, TimeInForce
from pms.core.models import (
    FillRecord,
    LiveTradingDisabledError,
    MarketSignal,
    OrderState,
    Position,
    Portfolio,
    TradeDecision,
)
from pms.runner import (
    _fill_from_order,
    _portfolio_with_fill,
    _resolved_outcome,
    _risk_trade_result_from_fill,
    Runner,
    ActuatorWorkItem,
)
from pms.storage.fill_store import StoredTradeRow
from pms.storage.live_emergency_audit import LiveEmergencyAuditWriter, read_active_halt
from pms.storage.feedback_store import FeedbackStore
from tests.support.fake_stores import InMemoryFeedbackStore


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


def _order(*, fees: float | None = None) -> OrderState:
    now = datetime(2026, 10, 18, tzinfo=UTC)
    quote = {} if fees is None else {"fees": fees, "fee_bps": 10}
    return OrderState(
        order_id="lane-1a-order",
        decision_id="lane-1a-decision",
        status="filled",
        market_id="lane-1a-market",
        token_id="lane-1a-token",
        venue="polymarket",
        requested_notional_usdc=10.0,
        filled_notional_usdc=10.0,
        remaining_notional_usdc=0.0,
        fill_price=0.5,
        submitted_at=now,
        last_updated_at=now,
        raw_status="filled",
        strategy_id="lane-1a-strategy",
        strategy_version_id="lane-1a-v1",
        filled_quantity=20.0,
        pre_submit_quote=quote,
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
    for field_name in ("limit_price", "max_slippage_bps"):
        corrupted = _decision()
        object.__setattr__(corrupted, field_name, nan)
        assert RiskManager().check(
            corrupted,
            Portfolio(100.0, 100.0, 0.0, []),
        ).reason == "non_finite"


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
    restored = Runner(
        config=PMSSettings(
            mode=RunMode.LIVE,
            live_emergency_audit_path=str(path),
        )
    )
    assert restored.actuator_executor.risk.active_halt() == state
    await writer.append_halt_clear(cleared_by="operator")
    assert read_active_halt(path) is None


@pytest.mark.asyncio
async def test_live_fee_reconcile_cli_updates_store_and_audits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Pool:
        async def close(self) -> None:
            pass

    calls: dict[str, object] = {}

    class Store:
        def __init__(self, pool: object) -> None:
            calls["pool"] = pool

        async def reconcile_fee(
            self,
            fill_id: str,
            *,
            fees: float,
            fee_bps: int | None,
            slippage_bps: float | None,
            reconciled_by: str | None = None,
            evidence: str | None = None,
            override: bool = False,
            audit_pending: bool = False,
        ) -> bool:
            calls["fee"] = (
                fill_id,
                fees,
                fee_bps,
                slippage_bps,
                reconciled_by,
                evidence,
                override,
            )
            return True

        async def mark_fee_reconcile_audited(self, fill_id: str) -> bool:
            calls["audit_marked"] = fill_id
            return True

    class Audit:
        def __init__(self, path: Path) -> None:
            calls["audit_path"] = path

        async def append_fee_reconcile(self, **kwargs: object) -> None:
            calls["audit"] = kwargs

    settings = PMSSettings(
        live_emergency_audit_path=str(tmp_path / "audit.jsonl"),
    )

    async def create_pool(**_: object) -> Pool:
        return Pool()

    async def ensure_schema(_: object) -> None:
        calls["schema"] = True

    monkeypatch.setattr(live_cli, "_load_cli_settings", lambda _: settings)
    monkeypatch.setattr(asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(live_cli, "ensure_schema_current", ensure_schema)
    monkeypatch.setattr(live_cli, "FillStore", Store)
    monkeypatch.setattr(live_cli, "LiveEmergencyAuditWriter", Audit)
    args = live_cli.build_parser().parse_args(
        [
            "reconcile-live-fee",
            "--fill-id",
            "fill-live-1",
            "--fees",
            "0.12",
            "--fee-bps",
            "15",
            "--reconciled-by",
            "operator",
            "--evidence",
            "venue-trade-1",
            "--database-url",
            "postgres://test",
        ]
    )

    assert await live_cli._main_async(args) == 0
    assert calls["fee"] == (
        "fill-live-1",
        0.12,
        15,
        None,
        "operator",
        "venue-trade-1",
        False,
    )
    assert calls["audit"] == {
        "fill_id": "fill-live-1",
        "fees": 0.12,
        "fee_bps": 15,
        "slippage_bps": None,
        "reconciled_by": "operator",
        "evidence": "venue-trade-1",
        "override": False,
    }
    assert json.loads(capsys.readouterr().out)["updated"] is True
    bad_args = live_cli.build_parser().parse_args(
        [
            "reconcile-live-fee",
            "--fill-id",
            "fill-live-1",
            "--fees",
            "0.13",
            "--reconciled-by",
            "operator",
            "--evidence",
            "TODO paste venue trade id",
            "--database-url",
            "postgres://test",
        ]
    )
    assert await live_cli._main_async(bad_args) == 1
    assert calls["fee"] == (
        "fill-live-1",
        0.12,
        15,
        None,
        "operator",
        "venue-trade-1",
        False,
    )
    assert "placeholder" in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.asyncio
async def test_same_day_daily_loss_clear_rewrites_durable_halt(tmp_path: Path) -> None:
    path = tmp_path / "daily-halt.jsonl"
    settings = PMSSettings(
        mode=RunMode.LIVE,
        live_emergency_audit_path=str(path),
        risk=RiskSettings(max_daily_loss_usdc=5.0),
    )
    runner = Runner(config=settings)
    now = datetime.now(tz=UTC)
    risk = runner.actuator_executor.risk
    risk.record_trade_result(
        RiskTradeResult(pnl=-10.0, slippage_bps=0.0, filled_at=now)
    )
    state = risk.check_auto_halt(runner.portfolio, now=now)
    assert state.trigger_kind == "daily_loss_limit"
    writer = LiveEmergencyAuditWriter(path)
    await writer.append_halt(state)

    remaining = await runner.clear_live_halt(cleared_by="operator")

    assert remaining is not None
    assert remaining.trigger_kind == "daily_loss_limit"
    persisted = read_active_halt(path)
    assert persisted is not None
    assert persisted.trigger_kind == "daily_loss_limit"


def test_unreadable_halt_audit_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "halt.jsonl"

    def deny_read(*_: object, **__: object) -> str:
        raise PermissionError("audit path unreadable")

    monkeypatch.setattr(Path, "read_text", deny_read)
    with pytest.raises(PermissionError, match="audit path unreadable"):
        read_active_halt(path)


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
    residuals = _compare_positions(
        [position("a", 10.0, 5.0), position("b", 20.0, 10.0)],
        [position("venue", 10.0, 5.0)],
        share_tolerance=1e-6,
        notional_tolerance=1e-6,
    )
    assert any("strategy position residual" in detail for detail in residuals)


@pytest.mark.asyncio
async def test_cancelled_live_submit_consumes_approval_and_marks_unknown() -> None:
    @dataclass
    class Gate:
        consumed: int = 0

        async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
            del preview
            return True

        async def consume(self, preview: LiveOrderPreview) -> None:
            del preview
            self.consumed += 1

    @dataclass
    class Client:
        entered: asyncio.Event
        requires_live_mode: bool = False

        async def submit_order(
            self,
            order: object,
            credentials: object,
        ) -> PolymarketOrderResult:
            del order, credentials
            self.entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    @dataclass(frozen=True)
    class Quote:
        async def quote(
            self,
            order: object,
            credentials: object,
        ) -> LivePreSubmitQuote:
            del order, credentials
            return LivePreSubmitQuote(
                market_status="open",
                book_age_ms=1.0,
                executable_notional_usdc=10.0,
                best_executable_price=0.5,
                spread_bps=1.0,
                quote_hash="lane-1a-quote",
                book_ts=datetime.now(tz=UTC),
            )

    gate = Gate()
    client = Client(asyncio.Event())
    settings = PMSSettings(
        live_trading_enabled=True,
        controller=ControllerSettings(time_in_force="IOC"),
        polymarket=PolymarketSettings(
            private_key="private",
            api_key="key",
            api_secret="secret",
            api_passphrase="pass",
            signature_type=1,
            funder_address="0x1111111111111111111111111111111111111111",
        ),
    )
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=Quote(),
    )
    task = asyncio.create_task(actuator.execute(_decision(), Portfolio(100, 100, 0, [])))
    await asyncio.wait_for(client.entered.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(PolymarketSubmissionUnknownError) as error:
        await task
    assert error.value.order_state is not None
    assert error.value.order_state.raw_status == "submission_unknown"
    assert gate.consumed == 1


@pytest.mark.asyncio
async def test_cancelled_post_submit_consume_becomes_unknown() -> None:
    @dataclass
    class Gate:
        entered: asyncio.Event
        release: asyncio.Event

        async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
            del preview
            return True

        async def consume(self, preview: LiveOrderPreview) -> None:
            del preview
            self.entered.set()
            await self.release.wait()

    @dataclass(frozen=True)
    class Client:
        requires_live_mode: bool = False

        async def submit_order(
            self,
            order: object,
            credentials: object,
        ) -> PolymarketOrderResult:
            del order, credentials
            return PolymarketOrderResult(
                order_id="lane-1a-post-submit",
                status="matched",
                raw_status="matched",
                filled_notional_usdc=10.0,
                remaining_notional_usdc=0.0,
                fill_price=0.5,
                filled_quantity=20.0,
                fees=0.1,
            )

    @dataclass(frozen=True)
    class Quote:
        async def quote(
            self,
            order: object,
            credentials: object,
        ) -> LivePreSubmitQuote:
            del order, credentials
            return LivePreSubmitQuote(
                market_status="open",
                book_age_ms=1.0,
                executable_notional_usdc=10.0,
                best_executable_price=0.5,
                spread_bps=1.0,
                quote_hash="lane-1a-post-quote",
                book_ts=datetime.now(tz=UTC),
            )

    gate = Gate(asyncio.Event(), asyncio.Event())
    settings = PMSSettings(
        live_trading_enabled=True,
        controller=ControllerSettings(time_in_force="IOC"),
        polymarket=PolymarketSettings(
            private_key="private",
            api_key="key",
            api_secret="secret",
            api_passphrase="pass",
            signature_type=1,
            funder_address="0x1111111111111111111111111111111111111111",
        ),
    )
    actuator = PolymarketActuator(
        settings,
        client=Client(),
        operator_gate=gate,
        quote_provider=Quote(),
    )
    task = asyncio.create_task(actuator.execute(_decision(), Portfolio(100, 100, 0, [])))
    await asyncio.wait_for(gate.entered.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(PolymarketSubmissionUnknownError) as error:
        await task
    assert error.value.order_state is not None
    assert error.value.order_state.order_id == "lane-1a-post-submit"
    gate.release.set()


def test_reduction_passes_both_halt_and_risk_drawdown_checks() -> None:
    position = Position(
        market_id="lane-1a-market",
        token_id="lane-1a-token",
        venue="polymarket",
        side="BUY",
        shares_held=100.0,
        avg_entry_price=0.5,
        unrealized_pnl=0.0,
        locked_usdc=50.0,
        strategy_id="lane-1a-strategy",
        strategy_version_id="lane-1a-v1",
    )
    portfolio = Portfolio(100.0, 50.0, 50.0, [position], max_drawdown_pct=15.0)
    decision = _decision(
        side="SELL",
        action="SELL",
        notional_usdc=30.0,
        limit_price=0.3,
    )
    risk = RiskManager(RiskSettings(max_drawdown_pct=10.0))
    assert risk.check_auto_halt(portfolio, decision=decision).halted is False
    assert risk.check(decision, portfolio).approved is True
    assert RiskManager(RiskSettings(max_drawdown_pct=15.0)).check_auto_halt(
        portfolio,
    ).trigger_kind == "drawdown_circuit_breaker"


@pytest.mark.asyncio
async def test_halt_clear_persists_before_memory_and_rebases_drawdown(tmp_path: Path) -> None:
    settings = PMSSettings(
        mode=RunMode.LIVE,
        live_emergency_audit_path=str(tmp_path / "halt.jsonl"),
        live_trading_enabled=True,
    )
    runner = Runner(config=settings)
    runner.actuator_executor.risk.restore_halt(
        runner.actuator_executor.risk._halt(  # noqa: SLF001
            trigger_kind="daily_loss_limit",
            reason="daily_loss_limit",
            triggered_at=datetime.now(tz=UTC),
            trace_id=None,
        )
    )
    with pytest.raises(ValueError, match="cleared_by"):
        await runner.clear_live_halt(cleared_by="")
    assert runner.actuator_executor.risk.active_halt() is not None
    await runner.clear_live_halt(cleared_by="operator")
    assert runner.actuator_executor.risk.active_halt() is None


@pytest.mark.asyncio
async def test_gate_denial_is_terminal_and_live_venue_rejection_is_distinct() -> None:
    class DeniedAdapter:
        async def execute(
            self,
            decision: TradeDecision,
            portfolio: Portfolio | None = None,
        ) -> OrderState:
            del decision, portfolio
            raise OperatorApprovalRequiredError("approval required")

    feedback_store = InMemoryFeedbackStore()
    executor = ActuatorExecutor(
        adapter=DeniedAdapter(),
        risk=RiskManager(),
        feedback=ActuatorFeedback(cast(FeedbackStore, feedback_store)),
    )
    with pytest.raises(OperatorApprovalRequiredError):
        await executor.execute(_decision(), Portfolio(100, 100, 0, []))
    assert feedback_store._items[-1].category == "operator_gate_denied"  # noqa: SLF001


def test_live_fill_uses_venue_fee_and_malformed_outcome_does_not_abort() -> None:
    signal = MarketSignal(
        market_id="lane-1a-market",
        token_id="lane-1a-token",
        venue="polymarket",
        title="lane-1a",
        yes_price=0.5,
        volume_24h=1.0,
        resolves_at=None,
        orderbook={},
        external_signal={"resolved_outcome": "bad"},
        fetched_at=datetime(2026, 10, 18, tzinfo=UTC),
        market_status="open",
    )
    fill = _fill_from_order(_order(fees=0.12), _decision(), signal)
    assert fill is not None
    assert fill.fees == pytest.approx(0.12)
    assert fill.resolved_outcome is None
    assert _portfolio_with_fill(Portfolio(100, 100, 0, []), fill).locked_usdc == 10.0


def test_execution_slippage_ignores_profitable_price_moves_but_halts_adverse_fill() -> None:
    risk = RiskManager(RiskSettings(max_daily_loss_usdc=1_000.0))
    now = datetime(2026, 10, 18, tzinfo=UTC)
    position = Position(
        market_id="lane-1a-market",
        token_id="lane-1a-token",
        venue="polymarket",
        side="BUY",
        shares_held=10.0,
        avg_entry_price=0.50,
        unrealized_pnl=0.0,
        locked_usdc=5.0,
        strategy_id="lane-1a-strategy",
        strategy_version_id="lane-1a-v1",
    )
    portfolio = Portfolio(100.0, 95.0, 5.0, [position])
    expected_exit = _decision(
        side="SELL", action="SELL", notional_usdc=5.6, limit_price=0.56
    )
    for index in range(10):
        profitable_exit = _fill(
            side="SELL", price=0.56, notional=5.6, quantity=10.0
        )
        result = _risk_trade_result_from_fill(
            portfolio, profitable_exit, decision=expected_exit
        )
        assert result is not None
        assert result.pnl > 0.0
        assert result.slippage_bps == 0.0
        risk.record_trade_result(result)
    assert risk.check_auto_halt(portfolio, now=now).halted is False

    adverse_exit = _fill(side="SELL", price=0.55, notional=5.5, quantity=10.0)
    result = _risk_trade_result_from_fill(portfolio, adverse_exit, decision=expected_exit)
    assert result is not None
    assert result.slippage_bps > 100.0
    risk.record_trade_result(result)
    halt = risk.check_auto_halt(portfolio, now=now)
    assert halt.halted is True
    assert halt.trigger_kind == "slippage_spike"


def test_operator_clear_keeps_daily_loss_until_next_utc_day() -> None:
    now = datetime(2026, 10, 18, tzinfo=UTC)
    risk = RiskManager(RiskSettings(max_daily_loss_usdc=10.0))
    risk.record_trade_result(
        RiskTradeResult(pnl=-10.0, slippage_bps=0.0, filled_at=now)
    )
    assert risk.check_auto_halt(Portfolio(100, 100, 0, []), now=now).halted is True
    risk.clear_halt()
    assert risk.check_auto_halt(Portfolio(100, 100, 0, []), now=now).halted is True
    risk.clear_halt()
    assert risk.check_auto_halt(
        Portfolio(100, 100, 0, []),
        now=now.replace(day=19),
    ).halted is False


def test_live_order_request_rejects_non_finite_after_entity_boundary() -> None:
    decision = _decision()
    object.__setattr__(decision, "limit_price", nan)
    with pytest.raises(LiveTradingDisabledError, match="invalid or non-finite"):
        _order_request(decision)


def test_matched_fill_path_survives_malformed_resolution_metadata() -> None:
    signal = MarketSignal(
        market_id="lane-1a-market",
        token_id="lane-1a-token",
        venue="polymarket",
        title="lane-1a",
        yes_price=0.5,
        volume_24h=1.0,
        resolves_at=None,
        orderbook={},
        external_signal={"resolved_outcome": {"bad": True}},
        fetched_at=datetime(2026, 10, 18, tzinfo=UTC),
        market_status="open",
    )
    fill = _fill_from_order(_order(), _decision(), signal)
    assert fill is not None
    assert fill.resolved_outcome is None


@pytest.mark.asyncio
async def test_live_runner_losing_sequence_trips_daily_loss_halt() -> None:
    @dataclass
    class SequenceAdapter:
        calls: int = 0

        async def execute(
            self,
            decision: TradeDecision,
            portfolio: Portfolio | None = None,
        ) -> OrderState:
            del portfolio
            self.calls += 1
            now = datetime.now(tz=UTC)
            return OrderState(
                order_id=f"lane-1a-sequence-{self.calls}",
                decision_id=decision.decision_id,
                status=OrderStatus.MATCHED.value,
                market_id=decision.market_id,
                token_id=decision.token_id,
                venue=decision.venue,
                requested_notional_usdc=decision.notional_usdc,
                filled_notional_usdc=decision.notional_usdc,
                remaining_notional_usdc=0.0,
                fill_price=decision.limit_price,
                submitted_at=now,
                last_updated_at=now,
                raw_status="matched",
                strategy_id=decision.strategy_id,
                strategy_version_id=decision.strategy_version_id,
                filled_quantity=decision.notional_usdc / decision.limit_price,
                action=decision.action,
                outcome=decision.outcome,
                time_in_force=decision.time_in_force.value,
                pre_submit_quote={"fees": 0.0, "fee_bps": 0},
            )

    class Store:
        def __init__(self) -> None:
            self.items: list[object] = []

        async def insert(self, item: object) -> None:
            self.items.append(item)

    settings = PMSSettings(
        mode=RunMode.LIVE,
        live_trading_enabled=True,
        risk=RiskSettings(
            max_position_per_market=100.0,
            max_total_exposure=100.0,
            max_drawdown_pct=50.0,
            max_daily_loss_usdc=10.0,
            min_order_usdc=1.0,
        ),
        controller=ControllerSettings(time_in_force="IOC"),
    )
    runner = Runner(config=settings)
    adapter = SequenceAdapter()
    runner.actuator_executor = ActuatorExecutor(
        adapter=adapter,
        risk=RiskManager(settings.risk),
        feedback=ActuatorFeedback(cast(FeedbackStore, InMemoryFeedbackStore())),
    )
    runner.order_store = cast(Any, Store())
    runner.fill_store = cast(Any, Store())
    runner._controller_task = asyncio.create_task(asyncio.sleep(0))  # noqa: SLF001
    buy = _decision(decision_id="lane-1a-live-buy", notional_usdc=20.0)
    sell = _decision(
        decision_id="lane-1a-live-sell",
        side="SELL",
        action="SELL",
        notional_usdc=8.0,
        limit_price=0.2,
    )
    next_entry = _decision(decision_id="lane-1a-live-next", notional_usdc=5.0)
    await runner._decision_queue.put(ActuatorWorkItem(buy, None))  # noqa: SLF001
    await runner._decision_queue.put(ActuatorWorkItem(sell, None))  # noqa: SLF001
    await runner._decision_queue.put(ActuatorWorkItem(next_entry, None))  # noqa: SLF001
    await runner._actuator_loop()  # noqa: SLF001
    assert adapter.calls == 2
    halt = runner.actuator_executor.risk.active_halt()
    assert halt is not None
    assert halt.trigger_kind == "daily_loss_limit"
    assert len(runner.state.fills) == 2


@pytest.mark.asyncio
async def test_risk_replay_keeps_pre_clear_inventory_basis_for_post_clear_loss(
    tmp_path: Path,
) -> None:
    opened_at = datetime(2026, 10, 18, 1, 0, tzinfo=UTC)
    cleared_at = datetime(2026, 10, 18, 2, 0, tzinfo=UTC)
    closed_at = datetime(2026, 10, 18, 3, 0, tzinfo=UTC)
    trades = [
        StoredTradeRow(
            trade_id="trade-open",
            fill_id="fill-open",
            order_id="order-open",
            decision_id="decision-open",
            market_id="lane-1a-market",
            question="lane-1a",
            token_id="lane-1a-token",
            venue="polymarket",
            side="BUY",
            fill_price=0.50,
            fill_notional_usdc=50.0,
            fill_quantity=100.0,
            executed_at=opened_at,
            filled_at=opened_at,
            status="filled",
            strategy_id="lane-1a-strategy",
            strategy_version_id="lane-1a-v1",
            fees=0.0,
            slippage_bps=0.0,
        ),
        StoredTradeRow(
            trade_id="trade-close",
            fill_id="fill-close",
            order_id="order-close",
            decision_id="decision-close",
            market_id="lane-1a-market",
            question="lane-1a",
            token_id="lane-1a-token",
            venue="polymarket",
            side="SELL",
            fill_price=0.40,
            fill_notional_usdc=40.0,
            fill_quantity=100.0,
            executed_at=closed_at,
            filled_at=closed_at,
            status="filled",
            strategy_id="lane-1a-strategy",
            strategy_version_id="lane-1a-v1",
            fees=0.0,
            slippage_bps=0.0,
        ),
    ]
    audit_path = tmp_path / "live-emergency-audit.jsonl"
    audit_path.write_text(
        json.dumps(
            {
                "timestamp": cleared_at.isoformat(),
                "phase": "risk_halt_clear",
                "event": "risk_halt_clear",
                "cleared_by": "operator",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class Store:
        pool = object.__new__(asyncpg.Pool)

        async def count_trades(self) -> int:
            return len(trades)

        async def read_trades(
            self,
            *,
            limit: int,
            offset: int = 0,
        ) -> list[StoredTradeRow]:
            assert limit == 10_000
            return trades[offset : offset + limit]

    settings = PMSSettings(
        mode=RunMode.LIVE,
        live_emergency_audit_path=str(audit_path),
        risk=RiskSettings(max_daily_loss_usdc=5.0),
    )
    runner = Runner(config=settings)
    risk = RiskManager(settings.risk)
    runner.actuator_executor = cast(Any, type("Executor", (), {"risk": risk})())
    runner.fill_store = cast(Any, Store())
    runner._pg_pool = cast(Any, object())  # noqa: SLF001

    await runner._reconcile_risk_from_db()  # noqa: SLF001

    assert len(risk._daily_trades) == 1  # noqa: SLF001
    assert risk._daily_trades[0].pnl == pytest.approx(-10.0)  # noqa: SLF001
    halt = risk.check_auto_halt(
        Portfolio(100.0, 100.0, 0.0, []),
        now=closed_at,
    )
    assert halt.halted is True
    assert halt.trigger_kind == "daily_loss_limit"


@pytest.mark.asyncio
async def test_risk_replay_paginates_history_before_recording_loss(tmp_path: Path) -> None:
    timestamp = datetime(2026, 10, 18, 1, 0, tzinfo=UTC)

    def trade(
        *,
        trade_id: str,
        market_id: str,
        side: str,
        price: float,
        filled_at: datetime,
    ) -> StoredTradeRow:
        return StoredTradeRow(
            trade_id=trade_id,
            fill_id=trade_id,
            order_id=f"order-{trade_id}",
            decision_id=f"decision-{trade_id}",
            market_id=market_id,
            question=market_id,
            token_id=f"token-{market_id}",
            venue="polymarket",
            side=side,
            fill_price=price,
            fill_notional_usdc=price * 100.0,
            fill_quantity=100.0,
            executed_at=filled_at,
            filled_at=filled_at,
            status="filled",
            strategy_id="lane-1a-strategy",
            strategy_version_id="lane-1a-v1",
            fees=0.0,
            slippage_bps=0.0,
        )

    trades = [
        trade(
            trade_id="old-open",
            market_id="history-market",
            side="BUY",
            price=0.50,
            filled_at=timestamp,
        )
    ]
    trades.extend(
        trade(
            trade_id=f"filler-{index}",
            market_id=f"filler-market-{index}",
            side="BUY",
            price=0.50,
            filled_at=timestamp,
        )
        for index in range(9_999)
    )
    trades.append(
        trade(
            trade_id="new-close",
            market_id="history-market",
            side="SELL",
            price=0.40,
            filled_at=timestamp.replace(hour=2),
        )
    )

    class Store:
        pool = object.__new__(asyncpg.Pool)

        async def count_trades(self) -> int:
            return len(trades)

        async def read_trades(
            self,
            *,
            limit: int,
            offset: int = 0,
        ) -> list[StoredTradeRow]:
            # Model the old bounded query: the 100,000-row request silently
            # omitted the final close, while page-sized requests are complete.
            if limit > 10_000:
                return trades[:10_000]
            return trades[offset : offset + limit]

    settings = PMSSettings(
        mode=RunMode.LIVE,
        live_emergency_audit_path=str(tmp_path / "replay-audit.jsonl"),
        risk=RiskSettings(max_daily_loss_usdc=5.0),
    )
    runner = Runner(config=settings)
    risk = RiskManager(settings.risk)
    runner.actuator_executor = cast(Any, type("Executor", (), {"risk": risk})())
    runner.fill_store = cast(Any, Store())
    runner._pg_pool = cast(Any, object())  # noqa: SLF001

    await runner._reconcile_risk_from_db()  # noqa: SLF001
    assert risk._daily_trades[-1].trace_id == "new-close"  # noqa: SLF001
    assert risk._daily_trades[-1].pnl == pytest.approx(-10.0)  # noqa: SLF001
    assert risk.check_auto_halt(
        Portfolio(100.0, 100.0, 0.0, []), now=timestamp.replace(hour=3)
    ).trigger_kind == "daily_loss_limit"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_client", [True, False], ids=["client", "duck-typed"])
async def test_live_missing_fee_persists_fill_before_fail_closed(
    with_client: bool,
    tmp_path: Path,
) -> None:
    @dataclass
    class LiveAdapter:
        client: object | None = object()

        async def execute(
            self,
            decision: TradeDecision,
            portfolio: Portfolio | None = None,
        ) -> OrderState:
            del portfolio
            now = datetime.now(tz=UTC)
            return OrderState(
                order_id="lane-1a-fee-unknown",
                decision_id=decision.decision_id,
                status=OrderStatus.MATCHED.value,
                market_id=decision.market_id,
                token_id=decision.token_id,
                venue=decision.venue,
                requested_notional_usdc=decision.notional_usdc,
                filled_notional_usdc=decision.notional_usdc,
                remaining_notional_usdc=0.0,
                fill_price=decision.limit_price,
                submitted_at=now,
                last_updated_at=now,
                raw_status="matched",
                strategy_id=decision.strategy_id,
                strategy_version_id=decision.strategy_version_id,
                filled_quantity=decision.notional_usdc / decision.limit_price,
                action=decision.action,
                outcome=decision.outcome,
                time_in_force=decision.time_in_force.value,
            )

    class Store:
        def __init__(self) -> None:
            self.items: list[object] = []

        async def insert(self, item: object) -> None:
            self.items.append(item)

    settings = PMSSettings(
        mode=RunMode.LIVE,
        live_trading_enabled=True,
        live_emergency_audit_path=str(tmp_path / "lane-1a-fee-audit.jsonl"),
        risk=RiskSettings(
            max_position_per_market=100.0,
            max_total_exposure=100.0,
            max_drawdown_pct=50.0,
            max_daily_loss_usdc=50.0,
        ),
    )
    runner = Runner(config=settings)
    runner.actuator_executor = ActuatorExecutor(
        adapter=LiveAdapter(client=object() if with_client else None),
        risk=RiskManager(settings.risk),
        feedback=ActuatorFeedback(cast(FeedbackStore, InMemoryFeedbackStore())),
    )
    order_store = Store()
    fill_store = Store()
    runner.order_store = cast(Any, order_store)
    runner.fill_store = cast(Any, fill_store)
    runner._controller_task = asyncio.create_task(asyncio.sleep(0))  # noqa: SLF001
    await runner._decision_queue.put(  # noqa: SLF001
        ActuatorWorkItem(_decision(decision_id="lane-1a-fee-unknown"), None)
    )
    with pytest.raises(RuntimeError, match="fee_unreconciled|fee is unavailable"):
        await runner._actuator_loop()  # noqa: SLF001
    assert len(order_store.items) == 1
    assert len(fill_store.items) == 1
    assert len(runner.state.fills) == 1
    assert runner.portfolio.locked_usdc > 0.0
    assert runner.live_trading_suspended is True
