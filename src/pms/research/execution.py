from __future__ import annotations

# Venue.KALSHI dispatch must still raise KalshiStubError via the shared stub.

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from typing import Any, Protocol, cast
from uuid import uuid4

from pms.actuator.risk import InsufficientLiquidityError
from pms.core.enums import OrderStatus, Side, TimeInForce, Venue
from pms.core.models import MarketSignal, OrderState, Portfolio, TradeDecision
from pms.core.validation import validate_finite, validate_positive, validate_price
from pms.core.venue_support import kalshi_stub_error
from pms.research.specs import ExecutionModel


class ReplayLookup(Protocol):
    async def book_state_at(
        self,
        ts: datetime,
        *,
        market_id: str,
        token_id: str | None,
    ) -> dict[str, list[dict[str, float]]]: ...


@dataclass(frozen=True, slots=True)
class _FillComputation:
    filled_notional_usdc: float
    filled_quantity: float
    remaining_notional_usdc: float
    fill_price: float | None
    eligible_notional_usdc: float


class MissingReplayOrderbookError(InsufficientLiquidityError):
    """Raised when replay cannot provide the decision token's own book."""


@dataclass(slots=True)
class _OpenOrderState:
    order_id: str
    decision: TradeDecision
    submitted_at: datetime
    last_updated_at: datetime
    requested_notional_usdc: float
    filled_notional_usdc: float = 0.0
    filled_quantity: float = 0.0
    remaining_notional_usdc: float = 0.0
    price_invalidation_streak: int = 0
    last_fill_price: float | None = None


@dataclass(slots=True)
class BacktestExecutionSimulator:
    replay_engine: ReplayLookup | None = None
    open_orders_ledger: dict[str, _OpenOrderState] = field(default_factory=dict)

    async def execute(
        self,
        *,
        signal: MarketSignal,
        decision: TradeDecision,
        portfolio: Portfolio | None = None,
        execution_model: ExecutionModel,
    ) -> OrderState:
        del portfolio
        if signal.venue == Venue.KALSHI.value:
            raise kalshi_stub_error("BacktestExecutionSimulator.execute(signal)")
        if decision.venue == Venue.KALSHI.value:
            raise kalshi_stub_error("BacktestExecutionSimulator.execute(decision)")
        _validate_replay_window(execution_model)
        submitted_at = _fill_timestamp(signal, execution_model)
        if signal.resolves_at is not None and submitted_at >= signal.resolves_at:
            return _order_state(
                order_id=f"backtest-sim-{uuid4().hex}",
                decision=decision,
                status=OrderStatus.CANCELED_MARKET_RESOLVED.value,
                submitted_at=submitted_at,
                last_updated_at=submitted_at,
                requested_notional_usdc=decision.notional_usdc,
                filled_notional_usdc=0.0,
                remaining_notional_usdc=decision.notional_usdc,
                filled_quantity=0.0,
                fill_price=None,
                raw_status="market_resolved_before_execution",
            )
        if (
            not math.isinf(execution_model.staleness_ms)
            and _latency_ms(signal, execution_model) > execution_model.staleness_ms
        ):
            return _order_state(
                order_id=f"backtest-sim-{uuid4().hex}",
                decision=decision,
                status=OrderStatus.CANCELLED.value,
                submitted_at=submitted_at,
                last_updated_at=submitted_at,
                requested_notional_usdc=decision.notional_usdc,
                filled_notional_usdc=0.0,
                remaining_notional_usdc=decision.notional_usdc,
                filled_quantity=0.0,
                fill_price=None,
                raw_status="stale_signal",
            )

        try:
            orderbook = await self._orderbook_at_fill_time(
                signal=signal,
                decision=decision,
                submitted_at=submitted_at,
            )
        except MissingReplayOrderbookError:
            return _order_state(
                order_id=f"backtest-sim-{uuid4().hex}",
                decision=decision,
                status="rejected",
                submitted_at=submitted_at,
                last_updated_at=submitted_at,
                requested_notional_usdc=decision.notional_usdc,
                filled_notional_usdc=0.0,
                remaining_notional_usdc=decision.notional_usdc,
                filled_quantity=0.0,
                fill_price=None,
                raw_status="missing_token_orderbook",
            )
        fill = _walk_book(
            orderbook=orderbook,
            decision=decision,
            target_notional_usdc=decision.notional_usdc,
            slippage_bps=execution_model.slippage_bps,
            displayed_depth_fill_ratio=execution_model.displayed_depth_fill_ratio,
            adverse_selection_bps=execution_model.adverse_selection_bps,
        )
        tif = _effective_time_in_force(decision, execution_model)
        order_id = f"backtest-sim-{uuid4().hex}"

        if tif == TimeInForce.FOK and fill.eligible_notional_usdc + 1e-9 < decision.notional_usdc:
            return _order_state(
                order_id=order_id,
                decision=decision,
                status="rejected",
                submitted_at=submitted_at,
                last_updated_at=submitted_at,
                requested_notional_usdc=decision.notional_usdc,
                filled_notional_usdc=0.0,
                remaining_notional_usdc=decision.notional_usdc,
                filled_quantity=0.0,
                fill_price=None,
                raw_status="fok_unfillable",
            )

        if fill.filled_notional_usdc <= 0.0:
            if tif == TimeInForce.GTC:
                open_order = _OpenOrderState(
                    order_id=order_id,
                    decision=decision,
                    submitted_at=submitted_at,
                    last_updated_at=submitted_at,
                    requested_notional_usdc=decision.notional_usdc,
                    remaining_notional_usdc=decision.notional_usdc,
                )
                self.open_orders_ledger[order_id] = open_order
                return _order_state(
                    order_id=order_id,
                    decision=decision,
                    status=OrderStatus.LIVE.value,
                    submitted_at=submitted_at,
                    last_updated_at=submitted_at,
                    requested_notional_usdc=decision.notional_usdc,
                    filled_notional_usdc=0.0,
                    remaining_notional_usdc=decision.notional_usdc,
                    filled_quantity=0.0,
                    fill_price=None,
                    raw_status="open",
                )
            if execution_model.fill_policy == "limit_if_touched":
                return _order_state(
                    order_id=order_id,
                    decision=decision,
                    status=OrderStatus.UNMATCHED.value,
                    submitted_at=submitted_at,
                    last_updated_at=submitted_at,
                    requested_notional_usdc=decision.notional_usdc,
                    filled_notional_usdc=0.0,
                    remaining_notional_usdc=decision.notional_usdc,
                    filled_quantity=0.0,
                    fill_price=None,
                    raw_status="limit_not_touched",
                )
            return _order_state(
                order_id=order_id,
                decision=decision,
                status="rejected",
                submitted_at=submitted_at,
                last_updated_at=submitted_at,
                requested_notional_usdc=decision.notional_usdc,
                filled_notional_usdc=0.0,
                remaining_notional_usdc=decision.notional_usdc,
                filled_quantity=0.0,
                fill_price=None,
                raw_status="ioc_unfilled",
            )

        if fill.remaining_notional_usdc <= 1e-9:
            return _order_state(
                order_id=order_id,
                decision=decision,
                status=OrderStatus.MATCHED.value,
                submitted_at=submitted_at,
                last_updated_at=submitted_at,
                requested_notional_usdc=decision.notional_usdc,
                filled_notional_usdc=fill.filled_notional_usdc,
                remaining_notional_usdc=0.0,
                filled_quantity=fill.filled_quantity,
                fill_price=fill.fill_price,
                raw_status="matched",
            )

        if tif == TimeInForce.GTC:
            open_order = _OpenOrderState(
                order_id=order_id,
                decision=decision,
                submitted_at=submitted_at,
                last_updated_at=submitted_at,
                requested_notional_usdc=decision.notional_usdc,
                filled_notional_usdc=fill.filled_notional_usdc,
                filled_quantity=fill.filled_quantity,
                remaining_notional_usdc=fill.remaining_notional_usdc,
                last_fill_price=fill.fill_price,
            )
            self.open_orders_ledger[order_id] = open_order
            return _order_state(
                order_id=order_id,
                decision=decision,
                status=OrderStatus.PARTIAL.value,
                submitted_at=submitted_at,
                last_updated_at=submitted_at,
                requested_notional_usdc=decision.notional_usdc,
                filled_notional_usdc=fill.filled_notional_usdc,
                remaining_notional_usdc=fill.remaining_notional_usdc,
                filled_quantity=fill.filled_quantity,
                fill_price=fill.fill_price,
                raw_status="partially_filled",
            )

        return _order_state(
            order_id=order_id,
            decision=decision,
            status=OrderStatus.CANCELLED.value,
            submitted_at=submitted_at,
            last_updated_at=submitted_at,
            requested_notional_usdc=decision.notional_usdc,
            filled_notional_usdc=fill.filled_notional_usdc,
            remaining_notional_usdc=fill.remaining_notional_usdc,
            filled_quantity=fill.filled_quantity,
            fill_price=fill.fill_price,
            raw_status="ioc_partial_remainder_cancelled",
        )

    async def advance(
        self,
        *,
        signal: MarketSignal,
        execution_model: ExecutionModel,
    ) -> list[OrderState]:
        _validate_replay_window(execution_model)
        results: list[OrderState] = []
        for order_id, open_order in list(self.open_orders_ledger.items()):
            evaluated_at = _fill_timestamp(signal, execution_model)
            if evaluated_at - open_order.submitted_at >= timedelta(
                milliseconds=execution_model.order_ttl_ms
            ):
                results.append(
                    _order_state(
                        order_id=order_id,
                        decision=open_order.decision,
                        status=OrderStatus.CANCELLED.value,
                        submitted_at=open_order.submitted_at,
                        last_updated_at=evaluated_at,
                        requested_notional_usdc=open_order.requested_notional_usdc,
                        filled_notional_usdc=open_order.filled_notional_usdc,
                        remaining_notional_usdc=open_order.remaining_notional_usdc,
                        filled_quantity=open_order.filled_quantity,
                        fill_price=open_order.last_fill_price,
                        raw_status="cancelled_ttl",
                    )
                )
                self.open_orders_ledger.pop(order_id, None)
                continue

            if not _same_market(open_order.decision, signal):
                continue

            try:
                orderbook = await self._orderbook_at_fill_time(
                    signal=signal,
                    decision=open_order.decision,
                    submitted_at=evaluated_at,
                )
            except MissingReplayOrderbookError:
                results.append(
                    _order_state(
                        order_id=order_id,
                        decision=open_order.decision,
                        status=OrderStatus.CANCELLED.value,
                        submitted_at=open_order.submitted_at,
                        last_updated_at=evaluated_at,
                        requested_notional_usdc=open_order.requested_notional_usdc,
                        filled_notional_usdc=open_order.filled_notional_usdc,
                        remaining_notional_usdc=open_order.remaining_notional_usdc,
                        filled_quantity=open_order.filled_quantity,
                        fill_price=open_order.last_fill_price,
                        raw_status="missing_token_orderbook",
                    )
                )
                self.open_orders_ledger.pop(order_id, None)
                continue
            best_price = _best_available_price(
                orderbook,
                open_order.decision,
                adverse_selection_bps=execution_model.adverse_selection_bps,
            )
            if best_price is None:
                continue
            if _is_limit_eligible(
                action=_action(open_order.decision),
                fill_price=best_price,
                limit_price=open_order.decision.limit_price,
            ):
                open_order.price_invalidation_streak = 0
            else:
                open_order.price_invalidation_streak += 1
                if (
                    open_order.price_invalidation_streak
                    >= execution_model.price_invalidation_streak
                ):
                    results.append(
                        _order_state(
                            order_id=order_id,
                            decision=open_order.decision,
                            status=OrderStatus.CANCELLED.value,
                            submitted_at=open_order.submitted_at,
                            last_updated_at=evaluated_at,
                            requested_notional_usdc=open_order.requested_notional_usdc,
                            filled_notional_usdc=open_order.filled_notional_usdc,
                            remaining_notional_usdc=open_order.remaining_notional_usdc,
                            filled_quantity=open_order.filled_quantity,
                            fill_price=open_order.last_fill_price,
                            raw_status="cancelled_limit_invalidated",
                        )
                    )
                    self.open_orders_ledger.pop(order_id, None)
                continue

            fill = _walk_book(
                orderbook=orderbook,
                decision=open_order.decision,
                target_notional_usdc=open_order.remaining_notional_usdc,
                slippage_bps=execution_model.slippage_bps,
                displayed_depth_fill_ratio=execution_model.displayed_depth_fill_ratio,
                adverse_selection_bps=execution_model.adverse_selection_bps,
            )
            if fill.filled_notional_usdc <= 0.0:
                continue

            open_order.filled_notional_usdc += fill.filled_notional_usdc
            open_order.filled_quantity += fill.filled_quantity
            open_order.remaining_notional_usdc = max(
                0.0,
                open_order.remaining_notional_usdc - fill.filled_notional_usdc,
            )
            open_order.last_updated_at = evaluated_at
            open_order.last_fill_price = fill.fill_price
            if open_order.remaining_notional_usdc <= 1e-9:
                results.append(
                    _order_state(
                        order_id=order_id,
                        decision=open_order.decision,
                        status=OrderStatus.MATCHED.value,
                        submitted_at=open_order.submitted_at,
                        last_updated_at=evaluated_at,
                        requested_notional_usdc=open_order.requested_notional_usdc,
                        filled_notional_usdc=open_order.filled_notional_usdc,
                        remaining_notional_usdc=0.0,
                        filled_quantity=open_order.filled_quantity,
                        fill_price=fill.fill_price,
                        raw_status="matched",
                    )
                )
                self.open_orders_ledger.pop(order_id, None)
            else:
                results.append(
                    _order_state(
                        order_id=order_id,
                        decision=open_order.decision,
                        status=OrderStatus.PARTIAL.value,
                        submitted_at=open_order.submitted_at,
                        last_updated_at=evaluated_at,
                        requested_notional_usdc=open_order.requested_notional_usdc,
                        filled_notional_usdc=open_order.filled_notional_usdc,
                        remaining_notional_usdc=open_order.remaining_notional_usdc,
                        filled_quantity=open_order.filled_quantity,
                        fill_price=fill.fill_price,
                        raw_status="partially_filled",
                    )
                )
        return results

    async def cancel_open_orders(self, *, session_end: datetime) -> list[OrderState]:
        results: list[OrderState] = []
        for order_id, open_order in list(self.open_orders_ledger.items()):
            results.append(
                _order_state(
                    order_id=order_id,
                    decision=open_order.decision,
                    status=OrderStatus.CANCELLED.value,
                    submitted_at=open_order.submitted_at,
                    last_updated_at=session_end,
                    requested_notional_usdc=open_order.requested_notional_usdc,
                    filled_notional_usdc=open_order.filled_notional_usdc,
                    remaining_notional_usdc=open_order.remaining_notional_usdc,
                    filled_quantity=open_order.filled_quantity,
                    fill_price=open_order.last_fill_price,
                    raw_status="cancelled_session_end",
                )
            )
            self.open_orders_ledger.pop(order_id, None)
        return results

    async def _orderbook_at_fill_time(
        self,
        *,
        signal: MarketSignal,
        decision: TradeDecision,
        submitted_at: datetime,
    ) -> dict[str, list[dict[str, float]]]:
        if self.replay_engine is None:
            return _fallback_signal_orderbook(signal, decision)
        lookup = getattr(self.replay_engine, "book_state_at", None)
        if lookup is None:
            return _fallback_signal_orderbook(signal, decision)
        try:
            return cast(
                dict[str, list[dict[str, float]]],
                await lookup(
                    submitted_at,
                    market_id=decision.market_id,
                    token_id=decision.token_id,
                ),
            )
        except LookupError:
            return _fallback_signal_orderbook(signal, decision)


def _walk_book(
    *,
    orderbook: dict[str, Any],
    decision: TradeDecision,
    target_notional_usdc: float,
    slippage_bps: float,
    displayed_depth_fill_ratio: float,
    adverse_selection_bps: float,
) -> _FillComputation:
    requested_notional_usdc = _decision_notional_usdc(target_notional_usdc)
    try:
        displayed_depth_fill_ratio = validate_finite(
            displayed_depth_fill_ratio, field_name="displayed_depth_fill_ratio"
        )
        slippage_bps = validate_finite(slippage_bps, field_name="slippage_bps")
        adverse_selection_bps = validate_finite(
            adverse_selection_bps, field_name="adverse_selection_bps"
        )
    except ValueError as exc:
        raise InsufficientLiquidityError("execution parameters must be finite") from exc
    side_key = _side_key(decision)
    levels = orderbook.get(side_key)
    if not isinstance(levels, list) or not levels:
        return _FillComputation(0.0, 0.0, requested_notional_usdc, None, 0.0)

    filled_notional_usdc = 0.0
    filled_quantity = 0.0
    eligible_notional_usdc = 0.0
    remaining_notional_usdc = requested_notional_usdc

    for raw_level in levels:
        if not isinstance(raw_level, dict):
            continue
        try:
            level_size = validate_finite(
                raw_level.get("size", 0.0), field_name="orderbook level size"
            )
        except (TypeError, ValueError):
            continue
        level_size *= displayed_depth_fill_ratio
        if level_size <= 0.0:
            continue
        action = _action(decision)
        level_price = _apply_slippage(
            _apply_adverse_selection(
                _effective_level_price(decision, raw_level),
                action=action,
                adverse_selection_bps=adverse_selection_bps,
            ),
            action=action,
            slippage_bps=slippage_bps,
        )
        try:
            level_price = validate_price(level_price)
        except (TypeError, ValueError):
            raise InsufficientLiquidityError("fill_price must be finite and positive")
        if not _is_limit_eligible(
            action=action,
            fill_price=level_price,
            limit_price=decision.limit_price,
        ):
            break
        level_notional = level_size * level_price
        eligible_notional_usdc += level_notional
        consumed_notional = min(remaining_notional_usdc, level_notional)
        consumed_quantity = consumed_notional / level_price
        filled_notional_usdc += consumed_notional
        filled_quantity += consumed_quantity
        remaining_notional_usdc -= consumed_notional
        if remaining_notional_usdc <= 1e-9:
            remaining_notional_usdc = 0.0
            break

    fill_price = None
    if filled_quantity > 0.0:
        fill_price = filled_notional_usdc / filled_quantity
    return _FillComputation(
        filled_notional_usdc=filled_notional_usdc,
        filled_quantity=filled_quantity,
        remaining_notional_usdc=remaining_notional_usdc,
        fill_price=fill_price,
        eligible_notional_usdc=eligible_notional_usdc,
    )


def _effective_time_in_force(
    decision: TradeDecision,
    execution_model: ExecutionModel,
) -> TimeInForce:
    if execution_model.fill_policy == "good_til_cancelled":
        return TimeInForce.GTC
    if execution_model.fill_policy == "fill_or_kill":
        return TimeInForce.FOK
    if execution_model.fill_policy in {"immediate_or_cancel", "limit_if_touched"}:
        return TimeInForce.IOC
    return decision.time_in_force


def _side_key(decision: TradeDecision) -> str:
    action = _action(decision)
    return "asks" if action == Side.BUY.value else "bids"


def _best_available_price(
    orderbook: dict[str, Any],
    decision: TradeDecision,
    *,
    adverse_selection_bps: float,
) -> float | None:
    side_key = _side_key(decision)
    levels = orderbook.get(side_key)
    if not isinstance(levels, list) or not levels:
        return None
    for raw_level in levels:
        if not isinstance(raw_level, dict):
            continue
        try:
            raw_size = validate_finite(
                raw_level.get("size", 0.0), field_name="orderbook level size"
            )
        except (TypeError, ValueError):
            continue
        if raw_size <= 0.0:
            continue
        return _apply_adverse_selection(
            _effective_level_price(decision, raw_level),
            action=_action(decision),
            adverse_selection_bps=adverse_selection_bps,
        )
    return None


def _effective_level_price(decision: TradeDecision, raw_level: dict[str, Any]) -> float:
    del decision
    try:
        return validate_price(raw_level["price"], field_name="orderbook level price")
    except (TypeError, ValueError) as exc:
        raise InsufficientLiquidityError(
            "orderbook level price must be finite and in (0, 1)"
        ) from exc


def _apply_slippage(price: float, *, action: str, slippage_bps: float) -> float:
    multiplier = slippage_bps / 10_000.0
    if action == Side.SELL.value:
        return max(0.0, price * (1.0 - multiplier))
    return min(1.0, price * (1.0 + multiplier))


def _apply_adverse_selection(
    price: float,
    *,
    action: str,
    adverse_selection_bps: float,
) -> float:
    multiplier = adverse_selection_bps / 10_000.0
    if action == Side.SELL.value:
        return max(0.0, price * (1.0 - multiplier))
    return min(1.0, price * (1.0 + multiplier))


def _is_limit_eligible(*, action: str, fill_price: float, limit_price: float) -> bool:
    if action == Side.SELL.value:
        return fill_price >= limit_price
    return fill_price <= limit_price


def _fill_timestamp(signal: MarketSignal, execution_model: ExecutionModel) -> datetime:
    return signal.fetched_at + timedelta(milliseconds=_latency_ms(signal, execution_model))


def _latency_ms(signal: MarketSignal, execution_model: ExecutionModel) -> float:
    if execution_model.latency_model is not None:
        return validate_finite(
            execution_model.latency_model(signal.fetched_at.timestamp()),
            field_name="latency_ms",
        )
    return validate_finite(execution_model.latency_ms, field_name="latency_ms")


def _decision_notional_usdc(raw_notional_usdc: float) -> float:
    try:
        return validate_positive(raw_notional_usdc, field_name="decision notional")
    except (TypeError, ValueError) as exc:
        raise InsufficientLiquidityError(
            "decision notional must be finite and positive"
        ) from exc


def _action(decision: TradeDecision) -> str:
    return decision.action if decision.action is not None else decision.side


def _same_market(decision: TradeDecision, signal: MarketSignal) -> bool:
    return (
        decision.market_id == signal.market_id
        and (decision.token_id or "") == (signal.token_id or "")
    )


def _fallback_signal_orderbook(
    signal: MarketSignal,
    decision: TradeDecision,
) -> dict[str, list[dict[str, float]]]:
    if (decision.token_id or "") != (signal.token_id or ""):
        raise MissingReplayOrderbookError(
            f"missing replay orderbook for token={decision.token_id}"
        )
    return _clone_orderbook(signal.orderbook)


def _clone_orderbook(orderbook: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    return {
        "bids": [
            {"price": _level_float(level, "price"), "size": _level_float(level, "size")}
            for level in cast(list[dict[str, object]], orderbook.get("bids", []))
        ],
        "asks": [
            {"price": _level_float(level, "price"), "size": _level_float(level, "size")}
            for level in cast(list[dict[str, object]], orderbook.get("asks", []))
        ],
    }


def _validate_replay_window(execution_model: ExecutionModel) -> None:
    if execution_model.order_ttl_ms > execution_model.replay_window_ms:
        msg = "ExecutionModel.order_ttl_ms must not exceed replay_window_ms"
        raise ValueError(msg)


def _order_state(
    *,
    order_id: str,
    decision: TradeDecision,
    status: str,
    submitted_at: datetime,
    last_updated_at: datetime,
    requested_notional_usdc: float,
    filled_notional_usdc: float,
    remaining_notional_usdc: float,
    filled_quantity: float,
    fill_price: float | None,
    raw_status: str,
) -> OrderState:
    return OrderState(
        order_id=order_id,
        decision_id=decision.decision_id,
        status=status,
        market_id=decision.market_id,
        token_id=decision.token_id,
        venue=decision.venue,
        requested_notional_usdc=requested_notional_usdc,
        filled_notional_usdc=filled_notional_usdc,
        remaining_notional_usdc=remaining_notional_usdc,
        fill_price=fill_price,
        submitted_at=submitted_at,
        last_updated_at=last_updated_at,
        raw_status=raw_status,
        strategy_id=decision.strategy_id,
        strategy_version_id=decision.strategy_version_id,
        filled_quantity=filled_quantity,
    )


def _level_float(level: dict[str, object], key: str) -> float:
    value = level.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        msg = f"orderbook level {key} must be numeric"
        raise ValueError(msg)
    return float(value)
