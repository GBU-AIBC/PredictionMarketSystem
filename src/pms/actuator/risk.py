from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

from pms.config import RiskSettings
from pms.core.enums import ReasonCode
from pms.core.exceptions import DomainValidationError
from pms.core.models import OrderState, Portfolio, Position, TradeDecision
from pms.core.validation import validate_finite, validate_positive


class InsufficientLiquidityError(RuntimeError):
    """Raised when an actuator cannot fill a decision from available depth."""


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str


@dataclass(frozen=True)
class ReductionSplit:
    reducing_shares: float
    residual_shares: float


HaltTriggerKind = Literal[
    "none",
    "consecutive_losses",
    "slippage_spike",
    "credential_failure",
    "order_without_fill",
    "rate_limit_exceeded",
    "drawdown_circuit_breaker",
    "daily_loss_limit",
]


@dataclass(frozen=True)
class HaltState:
    halted: bool
    reason: str
    triggered_at: datetime
    trigger_kind: HaltTriggerKind


@dataclass(frozen=True)
class HaltEvent:
    state: HaltState
    trace_id: str | None = None


@dataclass(frozen=True)
class HaltRecoveryCycle:
    halt_event: HaltEvent
    cleared_at: datetime


@dataclass(frozen=True)
class RiskTradeResult:
    pnl: float
    slippage_bps: float
    filled_at: datetime
    trace_id: str | None = None


@dataclass(frozen=True)
class _OpenOrderRiskReservation:
    market_id: str
    risk_group_id: str | None
    remaining_notional_usdc: float


@dataclass
class RiskManager:
    risk: RiskSettings = field(default_factory=RiskSettings)
    _halt_state: HaltState | None = field(default=None, init=False)
    _active_halt_event: HaltEvent | None = field(default=None, init=False)
    _halt_events: list[HaltEvent] = field(default_factory=list, init=False)
    _halt_recovery_cycles: list[HaltRecoveryCycle] = field(
        default_factory=list,
        init=False,
    )
    _recent_trades: list[RiskTradeResult] = field(default_factory=list, init=False)
    _daily_trades: list[RiskTradeResult] = field(default_factory=list, init=False)
    _recent_rate_limits: list[tuple[datetime, str | None]] = field(
        default_factory=list,
        init=False,
    )
    _credential_failure: tuple[datetime, str | None] | None = field(
        default=None,
        init=False,
    )
    _open_order_submitted_at: dict[str, datetime] = field(
        default_factory=dict,
        init=False,
    )
    _open_order_risk_reservations: dict[str, _OpenOrderRiskReservation] = field(
        default_factory=dict,
        init=False,
    )

    @property
    def halt_events(self) -> tuple[HaltEvent, ...]:
        return tuple(self._halt_events)

    @property
    def halt_recovery_cycles(self) -> tuple[HaltRecoveryCycle, ...]:
        return tuple(self._halt_recovery_cycles)

    def halt_recovery_cycles_since(self, since: datetime) -> tuple[HaltRecoveryCycle, ...]:
        cutoff = _coerce_aware(since)
        return tuple(
            cycle
            for cycle in self._halt_recovery_cycles
            if cycle.cleared_at >= cutoff
        )

    def check(self, decision: TradeDecision, portfolio: Portfolio) -> RiskDecision:
        try:
            notional = validate_finite(
                decision.notional_usdc,
                field_name="TradeDecision.notional_usdc",
            )
            validate_finite(portfolio.free_usdc, field_name="Portfolio.free_usdc")
            validate_finite(
                portfolio.locked_usdc,
                field_name="Portfolio.locked_usdc",
            )
            if portfolio.max_drawdown_pct is not None:
                validate_finite(
                    portfolio.max_drawdown_pct,
                    field_name="Portfolio.max_drawdown_pct",
                )
        except DomainValidationError:
            return RiskDecision(False, ReasonCode.NON_FINITE.value)
        if notional <= 0.0:
            return RiskDecision(False, "non_positive_size")

        reduction = _split_reduction_shares(portfolio, decision)
        if reduction.reducing_shares > 0.0 and reduction.residual_shares > 1e-9:
            return RiskDecision(False, "partial_reduction_unsupported")
        reduces_position = reduction.reducing_shares > 0.0
        residual_notional = 0.0 if reduces_position else notional
        open_order_exposure = self._open_order_total_exposure()
        if residual_notional > 0.0 and notional < self.risk.min_order_usdc:
            return RiskDecision(False, "min_order_usdc")

        if residual_notional > 0.0:
            market_exposure = (
                _market_exposure(portfolio, decision.market_id)
                + self._open_order_market_exposure(decision.market_id)
                + residual_notional
            )
            if market_exposure > self.risk.max_position_per_market:
                return RiskDecision(False, "max_position_per_market")

            total_exposure = (
                portfolio.locked_usdc + open_order_exposure + residual_notional
            )
            if total_exposure > self.risk.max_total_exposure:
                return RiskDecision(False, "max_total_exposure")

        if (
            residual_notional > 0.0
            and self.risk.max_exposure_per_risk_group is not None
        ):
            if decision.risk_group_id is None:
                return RiskDecision(False, "missing_risk_group_id")
            if (
                _risk_group_exposure(portfolio, decision.risk_group_id)
                + self._open_order_risk_group_exposure(decision.risk_group_id)
                + residual_notional
                > self.risk.max_exposure_per_risk_group
            ):
                return RiskDecision(False, "max_exposure_per_risk_group")

        if (
            self.risk.max_drawdown_pct is not None
            and portfolio.max_drawdown_pct is not None
            and portfolio.max_drawdown_pct > self.risk.max_drawdown_pct
        ):
            return RiskDecision(False, "drawdown_circuit_breaker")

        if (
            self.risk.max_open_positions is not None
            and not _has_open_position(portfolio, decision)
            and not reduces_position
            and len(portfolio.open_positions) >= self.risk.max_open_positions
        ):
            return RiskDecision(False, "max_open_positions")

        if decision.max_slippage_bps > self.risk.slippage_threshold_bps:
            return RiskDecision(False, "slippage_threshold_bps")

        if residual_notional > portfolio.free_usdc - open_order_exposure:
            return RiskDecision(False, "insufficient_free_usdc")

        if (
            residual_notional > 0.0
            and self.risk.max_quantity_shares is not None
            and decision.limit_price > 0.0
        ):
            estimated_quantity = residual_notional / decision.limit_price
            if estimated_quantity > self.risk.max_quantity_shares:
                return RiskDecision(False, "max_quantity_shares")

        return RiskDecision(True, "approved")

    def check_auto_halt(
        self,
        portfolio: Portfolio,
        *,
        decision: TradeDecision | None = None,
        now: datetime | None = None,
        trace_id: str | None = None,
    ) -> HaltState:
        checked_at = _coerce_aware(now)
        # A verified reduction is allowed to unwind risk even while an entry
        # breaker is latched. A mixed order (close plus residual entry) still
        # goes through the normal halt path and is therefore fail-closed.
        if decision is not None:
            reduction = _split_reduction_shares(portfolio, decision)
            if reduction.reducing_shares > 0.0 and reduction.residual_shares <= 1e-9:
                return HaltState(
                    halted=False,
                    reason="position_reduction_allowed",
                    triggered_at=checked_at,
                    trigger_kind="none",
                )
        if self._halt_state is not None and self._halt_state.halted:
            return self._halt_state

        if self._credential_failure is not None:
            _, credential_trace_id = self._credential_failure
            return self._halt(
                trigger_kind="credential_failure",
                reason="api_credential_failure",
                triggered_at=checked_at,
                trace_id=credential_trace_id or trace_id,
            )

        if (
            self.risk.max_drawdown_pct is not None
            and portfolio.max_drawdown_pct is not None
            and portfolio.max_drawdown_pct > self.risk.max_drawdown_pct
        ):
            return self._halt(
                trigger_kind="drawdown_circuit_breaker",
                reason="drawdown_circuit_breaker",
                triggered_at=checked_at,
                trace_id=trace_id,
            )

        daily_loss_trade = self._daily_loss_trigger_trade(checked_at)
        if daily_loss_trade is not None:
            return self._halt(
                trigger_kind="daily_loss_limit",
                reason="daily_loss_limit",
                triggered_at=checked_at,
                trace_id=daily_loss_trade.trace_id or trace_id,
            )

        if len(self._recent_trades) >= 5 and all(
            trade.pnl < 0.0 for trade in self._recent_trades[-5:]
        ):
            return self._halt(
                trigger_kind="consecutive_losses",
                reason="five_consecutive_losses",
                triggered_at=checked_at,
                trace_id=self._recent_trades[-1].trace_id or trace_id,
            )

        if len(self._recent_trades) >= 10:
            last_ten = self._recent_trades[-10:]
            avg_slippage = sum(trade.slippage_bps for trade in last_ten) / len(last_ten)
            if avg_slippage > 100.0:
                return self._halt(
                    trigger_kind="slippage_spike",
                    reason="avg_slippage_above_100bps",
                    triggered_at=checked_at,
                    trace_id=last_ten[-1].trace_id or trace_id,
                )

        self._prune_rate_limits(checked_at)
        if len(self._recent_rate_limits) >= 3:
            _, rate_limit_trace_id = self._recent_rate_limits[-1]
            return self._halt(
                trigger_kind="rate_limit_exceeded",
                reason="three_rate_limits_10m",
                triggered_at=checked_at,
                trace_id=rate_limit_trace_id or trace_id,
            )

        stale_order_id = self._stale_order_id(checked_at)
        if stale_order_id is not None:
            return self._halt(
                trigger_kind="order_without_fill",
                reason="order_without_fill_30m",
                triggered_at=checked_at,
                trace_id=trace_id,
            )

        return HaltState(
            halted=False,
            reason="ok",
            triggered_at=checked_at,
            trigger_kind="none",
        )

    def active_halt(self) -> HaltState | None:
        if self._halt_state is not None and self._halt_state.halted:
            return self._halt_state
        return None

    def restore_halt(self, state: HaltState) -> None:
        """Restore a durable breaker during runner construction."""
        if not state.halted:
            return
        event = HaltEvent(state=state)
        self._halt_state = state
        self._active_halt_event = event
        self._halt_events.append(event)

    def record_trade_result(self, result: RiskTradeResult) -> None:
        try:
            pnl = validate_finite(result.pnl, field_name="RiskTradeResult.pnl")
            slippage_bps = validate_finite(
                result.slippage_bps,
                field_name="RiskTradeResult.slippage_bps",
            )
        except DomainValidationError as exc:
            raise ValueError("RiskTradeResult contains a non-finite value") from exc
        normalized = replace(
            result,
            pnl=pnl,
            slippage_bps=slippage_bps,
            filled_at=_coerce_aware(result.filled_at),
        )
        self._recent_trades.append(normalized)
        if len(self._recent_trades) > 100:
            del self._recent_trades[:-100]
        self._daily_trades.append(normalized)

    def record_api_error(
        self,
        status_code: int,
        *,
        at: datetime | None = None,
        trace_id: str | None = None,
    ) -> None:
        occurred_at = _coerce_aware(at)
        if status_code in {401, 403}:
            self._credential_failure = (occurred_at, trace_id)
        if status_code == 429:
            self._recent_rate_limits.append((occurred_at, trace_id))
            self._prune_rate_limits(occurred_at)

    def record_order_placed(self, order_id: str, *, at: datetime | None = None) -> None:
        self._open_order_submitted_at[order_id] = _coerce_aware(at)

    def record_open_order_state(self, order_state: OrderState) -> None:
        if (
            order_state.remaining_notional_usdc <= 1e-9
            or _is_exposure_reducing_open_order(order_state)
        ):
            self._open_order_risk_reservations.pop(order_state.order_id, None)
            return
        reservation = _OpenOrderRiskReservation(
            market_id=order_state.market_id,
            risk_group_id=order_state.risk_group_id,
            remaining_notional_usdc=order_state.remaining_notional_usdc,
        )
        self._open_order_risk_reservations[order_state.order_id] = reservation

    def record_order_filled(self, order_id: str) -> None:
        self._open_order_submitted_at.pop(order_id, None)
        self._open_order_risk_reservations.pop(order_id, None)

    def clear_halt(self, *, at: datetime | None = None) -> None:
        if self._halt_state is not None and self._active_halt_event is not None:
            self._halt_recovery_cycles.append(
                HaltRecoveryCycle(
                    halt_event=self._active_halt_event,
                    cleared_at=_coerce_aware(at),
                )
            )
        self._halt_state = None
        self._active_halt_event = None
        self._credential_failure = None
        self._recent_trades.clear()
        self._recent_rate_limits.clear()
        self._open_order_submitted_at.clear()
        self._open_order_risk_reservations.clear()

    def _open_order_risk_group_exposure(self, risk_group_id: str) -> float:
        return sum(
            reservation.remaining_notional_usdc
            for reservation in self._open_order_risk_reservations.values()
            if reservation.risk_group_id == risk_group_id
        )

    def _open_order_market_exposure(self, market_id: str) -> float:
        return sum(
            reservation.remaining_notional_usdc
            for reservation in self._open_order_risk_reservations.values()
            if reservation.market_id == market_id
        )

    def _open_order_total_exposure(self) -> float:
        return sum(
            reservation.remaining_notional_usdc
            for reservation in self._open_order_risk_reservations.values()
        )

    def _halt(
        self,
        *,
        trigger_kind: HaltTriggerKind,
        reason: str,
        triggered_at: datetime,
        trace_id: str | None,
    ) -> HaltState:
        state = HaltState(
            halted=True,
            reason=reason,
            triggered_at=triggered_at,
            trigger_kind=trigger_kind,
        )
        event = HaltEvent(state=state, trace_id=trace_id)
        self._halt_state = state
        self._active_halt_event = event
        self._halt_events.append(event)
        return state

    def _prune_rate_limits(self, now: datetime) -> None:
        cutoff = now - timedelta(minutes=10)
        self._recent_rate_limits = [
            event for event in self._recent_rate_limits if event[0] >= cutoff
        ]

    def _stale_order_id(self, now: datetime) -> str | None:
        cutoff = now - timedelta(minutes=30)
        for order_id, submitted_at in self._open_order_submitted_at.items():
            if submitted_at <= cutoff:
                return order_id
        return None

    def _daily_loss_trigger_trade(self, now: datetime) -> RiskTradeResult | None:
        if self.risk.max_daily_loss_usdc is None:
            return None

        start_of_day = _start_of_utc_day(now)
        checked_at_utc = now.astimezone(UTC)
        self._daily_trades = [
            trade
            for trade in self._daily_trades
            if trade.filled_at.astimezone(UTC) >= start_of_day
        ]
        current_day_trades = [
            trade
            for trade in self._daily_trades
            if trade.filled_at.astimezone(UTC) <= checked_at_utc
        ]
        daily_pnl = sum(trade.pnl for trade in current_day_trades)
        if daily_pnl > -self.risk.max_daily_loss_usdc:
            return None
        return max(current_day_trades, key=lambda trade: trade.filled_at)


def _market_exposure(portfolio: Portfolio, market_id: str) -> float:
    return sum(
        position.locked_usdc
        for position in portfolio.open_positions
        if position.market_id == market_id
    )


def _risk_group_exposure(portfolio: Portfolio, risk_group_id: str) -> float:
    return sum(
        position.locked_usdc
        for position in portfolio.open_positions
        if position.risk_group_id == risk_group_id
    )


def _is_exposure_reducing_open_order(order_state: OrderState) -> bool:
    return str(order_state.action or "").upper() == "SELL"


def _has_open_position(portfolio: Portfolio, decision: TradeDecision) -> bool:
    return any(
        _same_contract(position, decision)
        and _same_strategy_version(position, decision)
        and position.side == decision.side
        for position in portfolio.open_positions
    )


def _split_reduction_shares(
    portfolio: Portfolio,
    decision: TradeDecision,
) -> ReductionSplit:
    decision_shares = _decision_contracts(decision)
    if decision_shares is None:
        return ReductionSplit(0.0, 0.0)
    closable_shares = sum(
        position.shares_held
        for position in portfolio.open_positions
        if _same_contract(position, decision)
        and _same_strategy_version(position, decision)
        and position.side != decision.side
    )
    reducing_shares = min(decision_shares, closable_shares)
    residual_shares = max(0.0, decision_shares - reducing_shares)
    return ReductionSplit(reducing_shares, residual_shares)


def _same_contract(position: Position, decision: TradeDecision) -> bool:
    return (
        position.market_id == decision.market_id
        and position.token_id == decision.token_id
        and position.venue == decision.venue
    )


def _same_strategy_version(position: Position, decision: TradeDecision) -> bool:
    return (
        position.strategy_id == decision.strategy_id
        and position.strategy_version_id == decision.strategy_version_id
    )


def _decision_contracts(decision: TradeDecision) -> float | None:
    if decision.limit_price <= 0.0:
        return None
    return float(
        Decimal(str(decision.notional_usdc)) / Decimal(str(decision.limit_price))
    )


def _coerce_aware(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(tz=UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _start_of_utc_day(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
