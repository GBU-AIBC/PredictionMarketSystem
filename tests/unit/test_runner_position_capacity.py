from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, cast

import pytest

from pms.config import PMSSettings, RiskSettings
from pms.core.enums import OrderStatus, RunMode, TimeInForce
from pms.core.models import MarketSignal, OrderState, Position, TradeDecision, Venue
from pms.runner import (
    ActuatorWorkItem,
    Runner,
    StrategyControllerRuntime,
    _estimated_decision_quantity,
)


FIXTURE_PATH = Path("tests/fixtures/polymarket_7day_synthetic.jsonl")


def _position(
    *,
    market_id: str = "market-a",
    token_id: str = "token-a",
    venue: Venue = "polymarket",
) -> Position:
    return Position(
        market_id=market_id,
        token_id=token_id,
        venue=venue,
        side="BUY",
        shares_held=10.0,
        avg_entry_price=0.5,
        unrealized_pnl=0.0,
        locked_usdc=5.0,
    )


def _decision(
    *,
    market_id: str = "market-b",
    token_id: str = "token-b",
    venue: Venue = "polymarket",
) -> TradeDecision:
    return TradeDecision(
        decision_id="decision-cap",
        market_id=market_id,
        token_id=token_id,
        venue=venue,
        side="BUY",
        action="BUY",
        notional_usdc=5.0,
        order_type="limit",
        max_slippage_bps=25,
        stop_conditions=[],
        prob_estimate=0.6,
        expected_edge=0.1,
        time_in_force=TimeInForce.IOC,
        opportunity_id="opp-cap",
        strategy_id="default",
        strategy_version_id="default-v1",
        limit_price=0.5,
        outcome="YES",
    )


def _signal(
    *,
    market_id: str = "market-b",
    token_id: str = "token-b",
) -> MarketSignal:
    return MarketSignal(
        market_id=market_id,
        token_id=token_id,
        venue="polymarket",
        title="Will runner reject oversize share quantity?",
        yes_price=0.001,
        volume_24h=1_000.0,
        resolves_at=datetime(2026, 6, 30, tzinfo=UTC),
        orderbook={"bids": [], "asks": []},
        external_signal={},
        fetched_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
        market_status="open",
    )


def _runner(*, max_open_positions: int | None = 2) -> Runner:
    settings = PMSSettings(
        risk=RiskSettings(
            max_open_positions=max_open_positions,
            max_quantity_shares=500.0,
        ),
    )
    return Runner(config=settings, historical_data_path=FIXTURE_PATH)


def _live_runner(*, max_open_positions: int | None = 2) -> Runner:
    settings = PMSSettings(
        mode=RunMode.LIVE,
        risk=RiskSettings(
            max_open_positions=max_open_positions,
            max_quantity_shares=500.0,
        ),
    )
    return Runner(config=settings, historical_data_path=FIXTURE_PATH)


def _open_order_state(decision: TradeDecision) -> OrderState:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    return OrderState(
        order_id=f"open-{decision.decision_id}",
        decision_id=decision.decision_id,
        status=OrderStatus.LIVE.value,
        market_id=decision.market_id,
        token_id=decision.token_id,
        venue=decision.venue,
        requested_notional_usdc=decision.notional_usdc,
        filled_notional_usdc=0.0,
        remaining_notional_usdc=decision.notional_usdc,
        fill_price=None,
        submitted_at=now,
        last_updated_at=now,
        raw_status="open",
        strategy_id=decision.strategy_id,
        strategy_version_id=decision.strategy_version_id,
        action=decision.action,
        outcome=decision.outcome,
        time_in_force=decision.time_in_force.value,
        intent_key=decision.intent_key,
        risk_group_id=decision.risk_group_id,
        pre_submit_quote={"fees": 0.0, "fee_bps": 0},
    )


class _RecordingDecisionStore:
    def __init__(self) -> None:
        self.transitions: list[tuple[str, str, str, datetime]] = []

    async def update_status(
        self,
        decision_id: str,
        *,
        current_status: str,
        next_status: str,
        updated_at: datetime,
    ) -> bool:
        self.transitions.append(
            (decision_id, current_status, next_status, updated_at)
        )
        return True


class _RecordingDedupStore:
    def __init__(self) -> None:
        self.release_calls: list[tuple[str, str]] = []

    async def release(self, decision_id: str, outcome: str) -> None:
        self.release_calls.append((decision_id, outcome))


class _RecordingOrderStore:
    def __init__(self) -> None:
        self.orders: list[OrderState] = []

    async def insert(self, order_state: OrderState) -> None:
        self.orders.append(order_state)


class _RecordingRisk:
    def __init__(self) -> None:
        self.open_orders: list[OrderState] = []
        self.filled_order_ids: list[str] = []

    def record_open_order_state(self, order_state: OrderState) -> None:
        self.open_orders.append(order_state)

    def record_order_filled(self, order_id: str) -> None:
        self.filled_order_ids.append(order_id)


class _RecordingFillStore:
    def __init__(self) -> None:
        self.fills: list[Any] = []

    async def insert(self, fill: Any) -> None:
        self.fills.append(fill)


class _EvalSpoolDouble:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    def enqueue(
        self,
        fill: Any,
        decision: TradeDecision,
        *,
        decision_evidence: dict[str, object] | None = None,
    ) -> None:
        del fill, decision_evidence
        self.enqueued.append(decision.decision_id)


class _StaticOpenOrderExecutor:
    def __init__(self, order_state: OrderState) -> None:
        self.order_state = order_state
        self.risk = _RecordingRisk()

    async def execute(
        self,
        decision: TradeDecision,
        portfolio: object,
        *,
        dedup_acquired: bool = False,
    ) -> OrderState:
        del decision, portfolio, dedup_acquired
        return self.order_state


class _HeartbeatPool:
    def __init__(self) -> None:
        self.component_status: dict[str, object] | None = None

    def acquire(self) -> "_HeartbeatAcquireContext":
        return _HeartbeatAcquireContext(_HeartbeatConnection(self))


class _HeartbeatAcquireContext:
    def __init__(self, connection: "_HeartbeatConnection") -> None:
        self._connection = connection

    async def __aenter__(self) -> "_HeartbeatConnection":
        return self._connection

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        return None


class _HeartbeatConnection:
    def __init__(self, pool: _HeartbeatPool) -> None:
        self._pool = pool

    async def execute(self, query: str, *args: object) -> None:
        del query
        self._pool.component_status = cast(dict[str, object], json.loads(str(args[-1])))


class TestWouldExceedPositionCapacity:
    def test_no_limit_set(self) -> None:
        runner = _runner(max_open_positions=None)
        runner.portfolio = replace(
            runner.portfolio,
            open_positions=[_position(), _position(market_id="market-c")],
        )
        assert not runner._would_exceed_position_capacity(_decision())

    def test_under_limit(self) -> None:
        runner = _runner(max_open_positions=3)
        runner.portfolio = replace(
            runner.portfolio,
            open_positions=[_position()],
        )
        assert not runner._would_exceed_position_capacity(_decision())

    def test_at_limit_new_market(self) -> None:
        runner = _runner(max_open_positions=2)
        runner.portfolio = replace(
            runner.portfolio,
            open_positions=[
                _position(),
                _position(market_id="market-c", token_id="token-c"),
            ],
        )
        assert runner._would_exceed_position_capacity(_decision())

    def test_at_limit_existing_market(self) -> None:
        runner = _runner(max_open_positions=2)
        runner.portfolio = replace(
            runner.portfolio,
            open_positions=[
                _position(),
                _position(market_id="market-b", token_id="token-b"),
            ],
        )
        assert not runner._would_exceed_position_capacity(_decision())

    def test_at_limit_same_market_different_token(self) -> None:
        runner = _runner(max_open_positions=2)
        runner.portfolio = replace(
            runner.portfolio,
            open_positions=[
                _position(),
                _position(market_id="market-b", token_id="token-OTHER"),
            ],
        )
        assert runner._would_exceed_position_capacity(_decision())

    def test_pending_new_position_counts_against_limit(self) -> None:
        runner = _runner(max_open_positions=2)
        runner.portfolio = replace(
            runner.portfolio,
            open_positions=[_position()],
        )
        first = _decision(market_id="market-b", token_id="token-b")
        second = replace(
            _decision(market_id="market-c", token_id="token-c"),
            decision_id="decision-cap-2",
        )

        assert not runner._would_exceed_position_capacity(first)
        assert runner._reserve_position_capacity(first) is not None

        assert runner._would_exceed_position_capacity(second)

    def test_released_pending_position_no_longer_counts_against_limit(self) -> None:
        runner = _runner(max_open_positions=2)
        runner.portfolio = replace(
            runner.portfolio,
            open_positions=[_position()],
        )
        first = _decision(market_id="market-b", token_id="token-b")
        second = replace(
            _decision(market_id="market-c", token_id="token-c"),
            decision_id="decision-cap-2",
        )

        assert runner._reserve_position_capacity(first) is not None
        runner._release_position_capacity_reservation(first.decision_id)

        assert not runner._would_exceed_position_capacity(second)

    def test_reserve_position_capacity_is_atomic_with_capacity_check(self) -> None:
        runner = _runner(max_open_positions=2)
        runner.portfolio = replace(
            runner.portfolio,
            open_positions=[_position()],
        )
        first = _decision(market_id="market-b", token_id="token-b")
        second = replace(
            _decision(market_id="market-c", token_id="token-c"),
            decision_id="decision-cap-2",
        )

        assert runner._reserve_position_capacity(first) is not None

        assert runner._reserve_position_capacity(second) is None

    def test_pre_actuator_diagnostic_rejects_max_quantity_before_queue(self) -> None:
        runner = _runner(max_open_positions=50)
        signal = _signal()
        runner._remember_paper_orderbook(signal)
        decision = replace(
            _decision(),
            notional_usdc=1.0,
            limit_price=0.001,
        )

        diagnostic = runner._pre_actuator_diagnostic(decision, signal)

        assert diagnostic is not None
        assert diagnostic.code == "max_quantity_shares"
        assert diagnostic.metadata["estimated_quantity_shares"] == pytest.approx(
            1000.0
        )
        assert diagnostic.metadata["max_quantity_shares"] == pytest.approx(500.0)

    def test_pre_actuator_diagnostic_returns_missing_paper_orderbook(self) -> None:
        runner = Runner(
            config=PMSSettings(
                mode=RunMode.PAPER,
                risk=RiskSettings(
                    max_open_positions=50,
                    max_quantity_shares=500.0,
                ),
            ),
            historical_data_path=FIXTURE_PATH,
        )
        signal = _signal(token_id="signal-token")
        decision = replace(
            _decision(),
            token_id="target-token",
            outcome="NO",
        )

        diagnostic = runner._pre_actuator_diagnostic(decision, signal)

        assert diagnostic is not None
        assert diagnostic.code == "paper_orderbook_missing_for_decision_token"
        assert diagnostic.metadata["decision_token_id"] == "target-token"
        assert diagnostic.metadata["signal_token_id"] == "signal-token"

    def test_estimated_quantity_uses_decimal_internal_arithmetic(self) -> None:
        decision = replace(
            _decision(),
            notional_usdc=0.3,
            limit_price=0.1,
        )

        assert _estimated_decision_quantity(decision) == 3.0

    @pytest.mark.asyncio
    async def test_runtime_heartbeat_reports_crashed_sensor_tasks_unhealthy(
        self,
    ) -> None:
        async def crash() -> None:
            raise RuntimeError("sensor failed")

        runner = _runner(max_open_positions=50)
        pool = _HeartbeatPool()
        runner._pg_pool = cast(Any, pool)  # noqa: SLF001
        runner.state.runtime_run_id = "run-sensor-crash"
        runner.state.runner_started_at = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
        runner._controller_runtimes["default"] = StrategyControllerRuntime(  # noqa: SLF001
            strategy_id="default",
            strategy_version_id="default-v1",
            controller=cast(Any, object()),
            asset_ids=None,
        )
        sensor_task = asyncio.create_task(crash())
        await asyncio.sleep(0)
        assert sensor_task.done()
        runner.sensor_stream._tasks = (sensor_task,)  # noqa: SLF001

        await runner._write_runtime_heartbeat()  # noqa: SLF001

        assert pool.component_status is not None
        assert pool.component_status["running"] is False
        assert pool.component_status["sensor_running"] is False
        assert pool.component_status["sensor_tasks"] == 0
        assert pool.component_status["sensor_tasks_total"] == 1
        assert pool.component_status["sensor_task_failures"] == 1
        sensor_task.exception()

    @pytest.mark.asyncio
    async def test_enqueue_rechecks_capacity_before_queuing(self) -> None:
        runner = _runner(max_open_positions=2)
        runner.portfolio = replace(
            runner.portfolio,
            open_positions=[_position()],
        )
        first = _decision(market_id="market-b", token_id="token-b")
        second = replace(
            _decision(market_id="market-c", token_id="token-c"),
            decision_id="decision-cap-2",
        )
        assert runner._reserve_position_capacity(first) is not None

        enqueued = await runner._enqueue_decision(second, signal=None)

        assert enqueued is False
        assert runner._decision_queue.empty()

    @pytest.mark.asyncio
    async def test_actuator_loop_keeps_live_open_order_capacity_reserved(self) -> None:
        runner = _live_runner(max_open_positions=2)
        runner.portfolio = replace(
            runner.portfolio,
            open_positions=[_position()],
        )
        first = _decision(market_id="market-b", token_id="token-b")
        second = replace(
            _decision(market_id="market-c", token_id="token-c"),
            decision_id="decision-cap-2",
        )
        order_state = _open_order_state(first)
        executor = _StaticOpenOrderExecutor(order_state)
        runner.actuator_executor = cast(Any, executor)
        runner.order_store = cast(Any, _RecordingOrderStore())
        runner.decision_store = cast(Any, _RecordingDecisionStore())
        assert runner._reserve_position_capacity(first) is not None
        await runner._decision_queue.put(  # noqa: SLF001
            ActuatorWorkItem(decision=first, signal=None)
        )

        actuator_task = asyncio.create_task(runner._actuator_loop())  # noqa: SLF001
        await asyncio.wait_for(runner._decision_queue.join(), timeout=1.0)  # noqa: SLF001
        runner._stop_event.set()  # noqa: SLF001
        await asyncio.wait_for(actuator_task, timeout=1.0)

        assert runner._would_exceed_position_capacity(second)  # noqa: SLF001
        assert executor.risk.open_orders == [order_state]
        assert executor.risk.filled_order_ids == []

    @pytest.mark.asyncio
    async def test_actuator_loop_releases_reservations_for_terminal_ioc_partial_fill(
        self,
    ) -> None:
        """A PARTIAL fill on an IOC order is terminal — the venue cancelled
        the unfilled remainder — so the runner must record the order as
        filled (no permanent open-order risk reservation) and release the
        pending position-capacity slot instead of leaking both forever."""
        runner = _live_runner(max_open_positions=2)
        runner.portfolio = replace(
            runner.portfolio,
            open_positions=[_position()],
        )
        first = _decision(market_id="market-b", token_id="token-b")
        order_state = replace(
            _open_order_state(first),
            status=OrderStatus.PARTIAL.value,
            raw_status="partial",
            filled_notional_usdc=first.notional_usdc / 2,
            remaining_notional_usdc=first.notional_usdc / 2,
            fill_price=first.limit_price,
            filled_quantity=(first.notional_usdc / 2) / first.limit_price,
        )
        assert order_state.time_in_force == TimeInForce.IOC.value
        executor = _StaticOpenOrderExecutor(order_state)
        runner.actuator_executor = cast(Any, executor)
        runner.order_store = cast(Any, _RecordingOrderStore())
        runner.decision_store = cast(Any, _RecordingDecisionStore())
        runner.fill_store = cast(Any, _RecordingFillStore())
        runner._evaluator_spool = cast(Any, _EvalSpoolDouble())  # noqa: SLF001
        assert runner._reserve_position_capacity(first) is not None
        await runner._decision_queue.put(  # noqa: SLF001
            ActuatorWorkItem(decision=first, signal=None)
        )

        actuator_task = asyncio.create_task(runner._actuator_loop())  # noqa: SLF001
        await asyncio.wait_for(runner._decision_queue.join(), timeout=1.0)  # noqa: SLF001
        runner._stop_event.set()  # noqa: SLF001
        await asyncio.wait_for(actuator_task, timeout=1.0)

        assert executor.risk.open_orders == []
        assert executor.risk.filled_order_ids == [order_state.order_id]
        assert (
            first.decision_id
            not in runner._pending_open_position_keys_by_decision  # noqa: SLF001
        )

    @pytest.mark.asyncio
    async def test_actuator_loop_keeps_live_open_exit_key_reserved(self) -> None:
        runner = _live_runner(max_open_positions=50)
        decision = replace(
            _decision(market_id="market-exit", token_id="token-exit"),
            decision_id="decision-exit",
            side="SELL",
            action="SELL",
            stop_conditions=["position_exit:stop_loss"],
        )
        exit_key_value = (
            decision.strategy_id,
            decision.strategy_version_id,
            decision.market_id,
            decision.token_id,
            "stop_loss",
        )
        runner._emitted_position_exit_keys.add(exit_key_value)  # noqa: SLF001
        runner._position_exit_keys_by_decision[decision.decision_id] = (  # noqa: SLF001
            exit_key_value
        )
        order_state = _open_order_state(decision)
        executor = _StaticOpenOrderExecutor(order_state)
        runner.actuator_executor = cast(Any, executor)
        runner.order_store = cast(Any, _RecordingOrderStore())
        runner.decision_store = cast(Any, _RecordingDecisionStore())
        await runner._decision_queue.put(  # noqa: SLF001
            ActuatorWorkItem(decision=decision, signal=None)
        )

        actuator_task = asyncio.create_task(runner._actuator_loop())  # noqa: SLF001
        await asyncio.wait_for(runner._decision_queue.join(), timeout=1.0)  # noqa: SLF001
        runner._stop_event.set()  # noqa: SLF001
        await asyncio.wait_for(actuator_task, timeout=1.0)

        assert exit_key_value in runner._emitted_position_exit_keys  # noqa: SLF001
        assert (  # noqa: SLF001
            runner._position_exit_keys_by_decision[decision.decision_id]
            == exit_key_value
        )
        assert executor.risk.open_orders == [order_state]

    @pytest.mark.asyncio
    async def test_enqueue_rejects_and_releases_accepted_decision_when_capacity_fills(
        self,
    ) -> None:
        runner = _runner(max_open_positions=2)
        runner.portfolio = replace(
            runner.portfolio,
            open_positions=[_position()],
        )
        first = _decision(market_id="market-b", token_id="token-b")
        second = replace(
            _decision(market_id="market-c", token_id="token-c"),
            decision_id="decision-cap-2",
        )
        queued_at = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
        decision_store = _RecordingDecisionStore()
        dedup_store = _RecordingDedupStore()
        runner.decision_store = decision_store  # type: ignore[assignment]
        runner.actuator_executor.dedup_store = dedup_store  # type: ignore[assignment]
        assert runner._reserve_position_capacity(first) is not None

        enqueued = await runner._enqueue_decision(
            second,
            signal=None,
            dedup_acquired=True,
            queued_at=queued_at,
        )

        assert enqueued is False
        assert runner._decision_queue.empty()
        assert decision_store.transitions == [
            ("decision-cap-2", "accepted", "rejected", queued_at)
        ]
        assert dedup_store.release_calls == [("decision-cap-2", "rejected")]

    @pytest.mark.asyncio
    async def test_enqueue_rejects_and_releases_risk_rejected_decision(self) -> None:
        runner = _runner(max_open_positions=50)
        queued_at = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
        decision_store = _RecordingDecisionStore()
        dedup_store = _RecordingDedupStore()
        runner.decision_store = decision_store  # type: ignore[assignment]
        runner.actuator_executor.dedup_store = dedup_store  # type: ignore[assignment]
        decision = replace(
            _decision(),
            notional_usdc=1.0,
            limit_price=0.001,
        )

        enqueued = await runner._enqueue_decision(
            decision,
            signal=None,
            dedup_acquired=True,
            queued_at=queued_at,
        )

        assert enqueued is False
        assert runner._decision_queue.empty()
        assert decision_store.transitions == [
            ("decision-cap", "accepted", "rejected", queued_at)
        ]
        assert dedup_store.release_calls == [("decision-cap", "rejected")]
