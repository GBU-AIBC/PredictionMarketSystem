from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from uuid import uuid4

from pms.actuator.adapters.polymarket import (
    PolymarketSubmissionUnknownError,
    PolymarketVenueRejectionError,
)
from pms.actuator.feedback import ActuatorFeedback
from pms.actuator.risk import InsufficientLiquidityError, RiskManager
from pms.alerting.events import HaltEvent as AlertHaltEvent
from pms.core.enums import OrderStatus, TimeInForce, Venue
from pms.core.exceptions import KalshiStubError
from pms.core.interfaces import DedupStore
from pms.core.models import LiveTradingDisabledError, OrderState, Portfolio, TradeDecision
from pms.core.venue_support import kalshi_stub_error
from pms.event_stream import RuntimeEventBus
from pms.storage.dedup_store import InMemoryDedupStore


logger = logging.getLogger(__name__)


@runtime_checkable
class ActuatorAdapter(Protocol):
    async def execute(
        self,
        decision: TradeDecision,
        portfolio: Portfolio | None = None,
    ) -> OrderState: ...


@dataclass
class ActuatorExecutor:
    adapter: ActuatorAdapter
    risk: RiskManager
    feedback: ActuatorFeedback
    dedup_store: DedupStore = field(default_factory=InMemoryDedupStore)
    event_bus: RuntimeEventBus | None = None
    _missing_event_bus_warned: bool = field(default=False, init=False)

    async def execute(
        self,
        decision: TradeDecision,
        portfolio: Portfolio,
        *,
        dedup_acquired: bool = False,
    ) -> OrderState:
        final_state: OrderState | None = None
        release_outcome: str | None = None
        acquired = dedup_acquired
        if not acquired:
            acquired = await self.dedup_store.acquire(decision)
        if not acquired:
            final_state = _rejected_order_state(decision, "duplicate_decision")
            await self.feedback.generate(final_state, reason="duplicate_decision")
            return final_state

        try:
            halt_state = self.risk.check_auto_halt(portfolio, decision=decision)
            if halt_state.halted:
                await self._publish_halt_event(decision, halt_state)
                final_state = _rejected_order_state(decision, halt_state.trigger_kind)
                await self.feedback.generate(
                    final_state,
                    reason=halt_state.trigger_kind,
                )
                return final_state

            risk_decision = self.risk.check(decision, portfolio)
            if not risk_decision.approved:
                final_state = _rejected_order_state(decision, risk_decision.reason)
                await self.feedback.generate(final_state, reason=risk_decision.reason)
                return final_state

            if decision.venue == Venue.KALSHI.value:
                final_state = _rejected_order_state(decision, "venue_rejection")
                release_outcome = "venue_rejection"
                await self.feedback.generate(final_state, reason="venue_rejection")
                error: KalshiStubError = kalshi_stub_error("ActuatorExecutor.execute")
                raise error

            try:
                final_state = await self.adapter.execute(decision, portfolio)
                _record_order_lifecycle(self.risk, final_state)
                return final_state
            except InsufficientLiquidityError:
                final_state = _rejected_order_state(decision, "insufficient_liquidity")
                await self.feedback.generate(
                    final_state,
                    reason="insufficient_liquidity",
                )
                return final_state
            except PolymarketSubmissionUnknownError as error:
                # Order may have reached the venue. Categorize distinctly
                # from venue_rejection so the operator knows to reconcile,
                # and so dedup release does not green-light a retry that
                # could double-spend.
                final_state = error.order_state or _rejected_order_state(
                    decision,
                    "submission_unknown",
                )
                error.order_state = final_state
                release_outcome = "submission_unknown"
                await self.feedback.generate(
                    final_state,
                    reason="submission_unknown",
                )
                raise
            except PolymarketVenueRejectionError:
                final_state = _rejected_order_state(decision, "venue_rejection")
                release_outcome = "venue_rejection"
                await self.feedback.generate(final_state, reason="venue_rejection")
                raise
            except LiveTradingDisabledError as error:
                # Gate failures are terminal decision outcomes, not venue
                # failures. Preserve the exception for the runner's incident
                # logging while returning a durable rejected state to callers
                # that use the executor directly.
                reason = _live_gate_reason(error)
                final_state = _rejected_order_state(decision, reason)
                release_outcome = "rejected"
                await self.feedback.generate(final_state, reason=reason)
                raise
            except Exception:
                final_state = _rejected_order_state(decision, "venue_rejection")
                release_outcome = "venue_rejection"
                await self.feedback.generate(final_state, reason="venue_rejection")
                raise
        except BaseException:
            if release_outcome is None and final_state is None:
                release_outcome = "venue_rejection"
            raise
        finally:
            if acquired:
                await self._release_dedup_state(
                    decision_id=decision.decision_id,
                    order_state=final_state,
                    fallback_outcome=release_outcome,
                )

    async def _release_dedup_state(
        self,
        *,
        decision_id: str,
        order_state: OrderState | None,
        fallback_outcome: str | None,
    ) -> None:
        try:
            outcome = fallback_outcome
            if outcome is None and order_state is not None:
                outcome = _dedup_release_outcome(order_state)
            if outcome is None:
                return
            await self.dedup_store.release(decision_id, outcome)
        except Exception:
            logger.exception(
                "Failed to release dedup state for decision_id=%s",
                decision_id,
            )

    async def _publish_halt_event(
        self,
        decision: TradeDecision,
        halt_state: object,
    ) -> None:
        if self.event_bus is None:
            if not self._missing_event_bus_warned:
                logger.warning("Auto-halt detected but no RuntimeEventBus is attached")
                self._missing_event_bus_warned = True
            return
        trigger_kind = getattr(halt_state, "trigger_kind")
        reason = getattr(halt_state, "reason")
        triggered_at = getattr(halt_state, "triggered_at")
        halt_event = AlertHaltEvent(
            reason=reason,
            trigger_kind=trigger_kind,
            triggered_at=triggered_at,
            market_id=decision.market_id,
            decision_id=decision.decision_id,
        )
        await self.event_bus.publish(
            halt_event.event_type,
            halt_event.summary,
            created_at=halt_event.triggered_at,
            market_id=decision.market_id,
            decision_id=decision.decision_id,
        )


def _rejected_order_state(decision: TradeDecision, reason: str) -> OrderState:
    now = datetime.now(tz=UTC)
    return OrderState(
        order_id=f"rejected-{uuid4().hex}",
        decision_id=decision.decision_id,
        status=OrderStatus.INVALID.value,
        market_id=decision.market_id,
        token_id=decision.token_id,
        venue=decision.venue,
        requested_notional_usdc=decision.notional_usdc,
        filled_notional_usdc=0.0,
        remaining_notional_usdc=decision.notional_usdc,
        fill_price=None,
        submitted_at=now,
        last_updated_at=now,
        raw_status=reason,
        strategy_id=decision.strategy_id,
        strategy_version_id=decision.strategy_version_id,
        filled_quantity=0.0,
        risk_group_id=decision.risk_group_id,
    )


def _live_gate_reason(error: LiveTradingDisabledError) -> str:
    if type(error).__name__ == "OperatorApprovalRequiredError":
        return "operator_gate_denied"
    return "live_disabled"


def _dedup_release_outcome(order_state: OrderState) -> str | None:
    status = _normalize_order_status(order_state.status)
    raw_status = order_state.raw_status.lower()

    if raw_status == "venue_rejection":
        return "venue_rejection"
    if status in {
        OrderStatus.LIVE.value,
        OrderStatus.UNMATCHED.value,
    }:
        return None
    if status in {OrderStatus.MATCHED.value, "partial"}:
        return "matched"
    if status == "rejected":
        return "rejected"
    if status == OrderStatus.INVALID.value:
        if raw_status == "insufficient_liquidity":
            return "rejected"
        return "invalid"
    if status == OrderStatus.CANCELLED.value:
        cancelled_outcomes = {
            "ttl": "cancelled_ttl",
            "cancelled_ttl": "cancelled_ttl",
            "limit_invalidated": "cancelled_limit_invalidated",
            "cancelled_limit_invalidated": "cancelled_limit_invalidated",
            "session_end": "cancelled_session_end",
            "cancelled_session_end": "cancelled_session_end",
        }
        cancelled_outcome = cancelled_outcomes.get(raw_status)
        if cancelled_outcome is not None:
            return cancelled_outcome
    if status == OrderStatus.CANCELED_MARKET_RESOLVED.value:
        return "cancelled_market_resolved"

    raise ValueError(
        f"unsupported dedup release mapping for status={order_state.status!r} "
        f"raw_status={order_state.raw_status!r}"
    )


def _normalize_order_status(status: str) -> str:
    normalized = status.lower()
    if normalized == "open":
        return OrderStatus.LIVE.value
    if normalized == "canceled":
        return OrderStatus.CANCELLED.value
    return normalized


def _record_order_lifecycle(risk: RiskManager, order_state: OrderState) -> None:
    status = _normalize_order_status(order_state.status)
    if (
        status in {
            OrderStatus.LIVE.value,
            OrderStatus.UNMATCHED.value,
            OrderStatus.PARTIAL.value,
        }
        and order_state.remaining_notional_usdc > 1e-9
        and not is_terminal_partial_fill(order_state)
    ):
        risk.record_order_placed(order_state.order_id, at=order_state.submitted_at)
        return

    risk.record_order_filled(order_state.order_id)


def is_terminal_partial_fill(order_state: OrderState) -> bool:
    """True when a PARTIAL order state is already terminal at the venue.

    Non-GTC (IOC/FOK) orders never rest: the venue cancels the unfilled
    remainder on arrival, so a PARTIAL response is the final state for that
    order_id and no further fill will ever arrive. Treating it as an open
    order would arm a stale-order timer and exposure reservation that
    nothing can clear.
    """
    if _normalize_order_status(order_state.status) != OrderStatus.PARTIAL.value:
        return False
    if order_state.time_in_force is None:
        return False
    return order_state.time_in_force.strip().upper() != TimeInForce.GTC.value
