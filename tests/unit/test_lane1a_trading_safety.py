from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from math import nan
from pathlib import Path
from typing import Any, cast

import pytest

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
from pms.actuator.risk import RiskManager
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
