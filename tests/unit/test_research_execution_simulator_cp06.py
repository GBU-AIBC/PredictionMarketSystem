from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from dataclasses import replace
from typing import Literal

import pytest

from pms.core.enums import OrderStatus, TimeInForce
from pms.core.models import MarketSignal, Portfolio, TradeDecision
from pms.research.execution import BacktestExecutionSimulator
from pms.research.specs import ExecutionModel


class _ReplayLookup:
    def __init__(self, book_by_ts: dict[datetime, dict[str, list[dict[str, float]]]]) -> None:
        self._book_by_ts = dict(sorted(book_by_ts.items()))

    async def book_state_at(
        self,
        ts: datetime,
        *,
        market_id: str,
        token_id: str | None,
    ) -> dict[str, list[dict[str, float]]]:
        del market_id, token_id
        selected_ts = max(key for key in self._book_by_ts if key <= ts)
        return deepcopy(self._book_by_ts[selected_ts])


class _MissingTokenReplayLookup:
    async def book_state_at(
        self,
        ts: datetime,
        *,
        market_id: str,
        token_id: str | None,
    ) -> dict[str, list[dict[str, float]]]:
        del ts, market_id, token_id
        raise LookupError("missing token book")


def _signal(
    *,
    fetched_at: datetime | None = None,
    asks: list[dict[str, float]] | None = None,
    bids: list[dict[str, float]] | None = None,
    resolves_at: datetime | None = None,
) -> MarketSignal:
    return MarketSignal(
        market_id="sim-market",
        token_id="yes-token",
        venue="polymarket",
        title="Will the simulator work?",
        yes_price=0.5,
        volume_24h=1_000.0,
        resolves_at=resolves_at or datetime(2026, 4, 30, tzinfo=UTC),
        orderbook={
            "bids": bids or [{"price": 0.24, "size": 200.0}],
            "asks": asks or [{"price": 0.25, "size": 200.0}],
        },
        external_signal={},
        fetched_at=fetched_at or datetime(2026, 4, 20, 12, 0, tzinfo=UTC),
        market_status="open",
    )


def _portfolio() -> Portfolio:
    return Portfolio(
        total_usdc=1_000.0,
        free_usdc=1_000.0,
        locked_usdc=0.0,
        open_positions=[],
    )


def _decision(
    *,
    time_in_force: TimeInForce = TimeInForce.IOC,
    limit_price: float = 0.30,
    notional_usdc: float = 100.0,
    outcome: Literal["YES", "NO"] = "YES",
) -> TradeDecision:
    return TradeDecision(
        decision_id="decision-sim",
        market_id="sim-market",
        token_id="yes-token" if outcome == "YES" else "no-token",
        venue="polymarket",
        side="BUY",
        notional_usdc=notional_usdc,
        order_type="limit",
        max_slippage_bps=100,
        stop_conditions=[],
        prob_estimate=0.6,
        expected_edge=0.2,
        time_in_force=time_in_force,
        opportunity_id="opp-sim",
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        action="BUY",
        limit_price=limit_price,
        outcome=outcome,
        model_id="rules",
    )


def _execution_model(
    *,
    fill_policy: Literal[
        "immediate_or_cancel",
        "limit_if_touched",
        "good_til_cancelled",
        "fill_or_kill",
    ] = "immediate_or_cancel",
    latency_ms: float = 0.0,
    slippage_bps: float = 0.0,
    displayed_depth_fill_ratio: float = 1.0,
    adverse_selection_bps: float = 0.0,
    order_ttl_ms: int = 60_000,
    price_invalidation_streak: int = 3,
) -> ExecutionModel:
    return ExecutionModel(
        fee_rate=0.0,
        slippage_bps=slippage_bps,
        latency_ms=latency_ms,
        staleness_ms=1_000.0,
        fill_policy=fill_policy,
        displayed_depth_fill_ratio=displayed_depth_fill_ratio,
        adverse_selection_bps=adverse_selection_bps,
        order_ttl_ms=order_ttl_ms,
        price_invalidation_streak=price_invalidation_streak,
        replay_window_ms=86_400_000,
    )


@pytest.mark.asyncio
async def test_simulator_walks_book_and_preserves_notional_quantity_invariant() -> None:
    simulator = BacktestExecutionSimulator()
    signal = _signal(
        asks=[
            {"price": 0.25, "size": 200.0},
            {"price": 0.28, "size": 400.0},
            {"price": 0.32, "size": 600.0},
        ]
    )

    state = await simulator.execute(
        signal=signal,
        decision=_decision(limit_price=0.30, notional_usdc=100.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(),
    )

    assert state.status == OrderStatus.MATCHED.value
    assert state.filled_notional_usdc == pytest.approx(100.0)
    assert state.filled_quantity == pytest.approx(378.57142857)
    assert state.fill_price == pytest.approx(
        state.filled_notional_usdc / state.filled_quantity,
        rel=1e-6,
    )


@pytest.mark.asyncio
async def test_simulator_limit_blocks_and_returns_partial_for_ioc() -> None:
    simulator = BacktestExecutionSimulator()
    signal = _signal(
        asks=[
            {"price": 0.25, "size": 200.0},
            {"price": 0.28, "size": 400.0},
            {"price": 0.32, "size": 600.0},
        ]
    )

    state = await simulator.execute(
        signal=signal,
        decision=_decision(limit_price=0.27, notional_usdc=200.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(),
    )

    assert state.status == OrderStatus.CANCELLED.value
    assert state.filled_notional_usdc == pytest.approx(50.0)
    assert state.remaining_notional_usdc == pytest.approx(150.0)
    assert state.raw_status == "ioc_partial_remainder_cancelled"


@pytest.mark.asyncio
async def test_simulator_haircuts_displayed_depth_for_queue_position() -> None:
    simulator = BacktestExecutionSimulator()
    signal = _signal(asks=[{"price": 0.25, "size": 400.0}])

    state = await simulator.execute(
        signal=signal,
        decision=_decision(limit_price=0.25, notional_usdc=100.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(displayed_depth_fill_ratio=0.5),
    )

    assert state.status == OrderStatus.CANCELLED.value
    assert state.filled_notional_usdc == pytest.approx(50.0)
    assert state.filled_quantity == pytest.approx(200.0)
    assert state.remaining_notional_usdc == pytest.approx(50.0)
    assert state.raw_status == "ioc_partial_remainder_cancelled"


@pytest.mark.asyncio
async def test_simulator_applies_adverse_selection_before_limit_eligibility() -> None:
    simulator = BacktestExecutionSimulator()

    state = await simulator.execute(
        signal=_signal(asks=[{"price": 0.25, "size": 400.0}]),
        decision=_decision(limit_price=0.251, notional_usdc=50.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(adverse_selection_bps=100.0),
    )

    assert state.status == "rejected"
    assert state.raw_status == "ioc_unfilled"
    assert state.filled_notional_usdc == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_simulator_handles_extreme_longshot_quantity() -> None:
    simulator = BacktestExecutionSimulator()
    signal = _signal(asks=[{"price": 0.001, "size": 1_000_000.0}])

    state = await simulator.execute(
        signal=signal,
        decision=_decision(limit_price=0.001, notional_usdc=1.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(),
    )

    assert state.status == OrderStatus.MATCHED.value
    assert state.filled_notional_usdc == pytest.approx(1.0)
    assert state.filled_quantity == pytest.approx(1_000.0)


@pytest.mark.asyncio
async def test_simulator_uses_replay_lookup_after_latency() -> None:
    base_ts = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    simulator = BacktestExecutionSimulator(
        replay_engine=_ReplayLookup(
            {
                base_ts: {
                    "bids": [{"price": 0.20, "size": 300.0}],
                    "asks": [{"price": 0.25, "size": 200.0}],
                },
                base_ts + timedelta(milliseconds=200): {
                    "bids": [{"price": 0.20, "size": 300.0}],
                    "asks": [{"price": 0.28, "size": 400.0}],
                },
            }
        )
    )

    state = await simulator.execute(
        signal=_signal(fetched_at=base_ts),
        decision=_decision(limit_price=0.30, notional_usdc=100.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(latency_ms=200.0),
    )

    assert state.fill_price == pytest.approx(0.28)
    assert state.submitted_at == base_ts + timedelta(milliseconds=200)


@pytest.mark.asyncio
async def test_simulator_uses_custom_latency_model_when_present() -> None:
    base_ts = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    simulator = BacktestExecutionSimulator(
        replay_engine=_ReplayLookup(
            {
                base_ts: {
                    "bids": [{"price": 0.20, "size": 300.0}],
                    "asks": [{"price": 0.25, "size": 200.0}],
                },
                base_ts + timedelta(milliseconds=500): {
                    "bids": [{"price": 0.20, "size": 300.0}],
                    "asks": [{"price": 0.27, "size": 400.0}],
                },
            }
        )
    )
    model = _execution_model(latency_ms=0.0)
    model = ExecutionModel(
        fee_rate=model.fee_rate,
        slippage_bps=model.slippage_bps,
        latency_ms=model.latency_ms,
        staleness_ms=model.staleness_ms,
        fill_policy=model.fill_policy,
        order_ttl_ms=model.order_ttl_ms,
        price_invalidation_streak=model.price_invalidation_streak,
        replay_window_ms=model.replay_window_ms,
        latency_model=lambda _ts: 500.0,
    )

    state = await simulator.execute(
        signal=_signal(fetched_at=base_ts),
        decision=_decision(limit_price=0.30, notional_usdc=100.0),
        portfolio=_portfolio(),
        execution_model=model,
    )

    assert state.fill_price == pytest.approx(0.27)
    assert state.submitted_at == base_ts + timedelta(milliseconds=500)


@pytest.mark.asyncio
async def test_simulator_ioc_matches_using_book_price_with_latency_applied() -> None:
    simulator = BacktestExecutionSimulator()

    order_state = await simulator.execute(
        signal=_signal(
            bids=[{"price": 0.39, "size": 100.0}],
            asks=[{"price": 0.41, "size": 100.0}],
        ),
        decision=_decision(limit_price=0.50, notional_usdc=10.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(latency_ms=250.0),
    )

    assert order_state.status == OrderStatus.MATCHED.value
    assert order_state.fill_price == pytest.approx(0.41)
    assert order_state.submitted_at == datetime(2026, 4, 20, 12, 0, tzinfo=UTC) + timedelta(
        milliseconds=250
    )
    assert order_state.last_updated_at == order_state.submitted_at


@pytest.mark.asyncio
async def test_simulator_applies_slippage_bps_to_fill_price() -> None:
    simulator = BacktestExecutionSimulator()

    order_state = await simulator.execute(
        signal=_signal(
            bids=[{"price": 0.39, "size": 100.0}],
            asks=[{"price": 0.41, "size": 100.0}],
        ),
        decision=_decision(limit_price=0.50, notional_usdc=10.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(slippage_bps=100.0),
    )

    assert order_state.status == OrderStatus.MATCHED.value
    assert order_state.fill_price == pytest.approx(0.41 * 1.01)


@pytest.mark.asyncio
async def test_simulator_combines_adverse_selection_and_slippage() -> None:
    simulator = BacktestExecutionSimulator()

    order_state = await simulator.execute(
        signal=_signal(
            bids=[{"price": 0.39, "size": 100.0}],
            asks=[{"price": 0.40, "size": 100.0}],
        ),
        decision=_decision(limit_price=0.50, notional_usdc=10.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(
            adverse_selection_bps=100.0,
            slippage_bps=50.0,
        ),
    )

    assert order_state.status == OrderStatus.MATCHED.value
    assert order_state.fill_price == pytest.approx(0.40 * 1.01 * 1.005)


@pytest.mark.asyncio
async def test_gtc_price_invalidation_uses_adverse_selection_price() -> None:
    simulator = BacktestExecutionSimulator()
    opened = await simulator.execute(
        signal=_signal(asks=[{"price": 0.25, "size": 0.0}]),
        decision=_decision(
            time_in_force=TimeInForce.GTC,
            limit_price=0.251,
            notional_usdc=50.0,
        ),
        portfolio=_portfolio(),
        execution_model=_execution_model(fill_policy="good_til_cancelled"),
    )

    assert opened.status == OrderStatus.LIVE.value

    results = await simulator.advance(
        signal=_signal(asks=[{"price": 0.25, "size": 400.0}]),
        execution_model=_execution_model(
            fill_policy="good_til_cancelled",
            adverse_selection_bps=100.0,
            price_invalidation_streak=1,
        ),
    )

    assert len(results) == 1
    assert results[0].status == OrderStatus.CANCELLED.value
    assert results[0].raw_status == "cancelled_limit_invalidated"


@pytest.mark.asyncio
async def test_simulator_rejects_when_latency_exceeds_staleness_budget() -> None:
    simulator = BacktestExecutionSimulator()

    stale_state = await simulator.execute(
        signal=_signal(
            bids=[{"price": 0.39, "size": 100.0}],
            asks=[{"price": 0.41, "size": 100.0}],
        ),
        decision=_decision(limit_price=0.50, notional_usdc=10.0),
        portfolio=_portfolio(),
        execution_model=ExecutionModel(
            fee_rate=0.0,
            slippage_bps=0.0,
            latency_ms=250.0,
            staleness_ms=100.0,
            fill_policy="immediate_or_cancel",
            order_ttl_ms=60_000,
            price_invalidation_streak=3,
            replay_window_ms=86_400_000,
        ),
    )

    assert stale_state.status == OrderStatus.CANCELLED.value
    assert stale_state.fill_price is None
    assert stale_state.raw_status == "stale_signal"


@pytest.mark.asyncio
async def test_simulator_limit_if_touched_leaves_order_unmatched_when_book_never_touches_limit(
) -> None:
    simulator = BacktestExecutionSimulator()

    order_state = await simulator.execute(
        signal=_signal(
            bids=[{"price": 0.39, "size": 100.0}],
            asks=[{"price": 0.41, "size": 100.0}],
        ),
        decision=_decision(limit_price=0.40, notional_usdc=10.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(fill_policy="limit_if_touched"),
    )

    assert order_state.status == OrderStatus.UNMATCHED.value
    assert order_state.fill_price is None
    assert order_state.raw_status == "limit_not_touched"


@pytest.mark.asyncio
async def test_simulator_cancels_when_latency_pushes_execution_past_resolution() -> None:
    simulator = BacktestExecutionSimulator()

    order_state = await simulator.execute(
        signal=_signal(
            bids=[{"price": 0.39, "size": 100.0}],
            asks=[{"price": 0.41, "size": 100.0}],
            resolves_at=datetime(2026, 4, 20, 12, 0, 0, 100_000, tzinfo=UTC),
        ),
        decision=_decision(limit_price=0.50, notional_usdc=10.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(latency_ms=250.0),
    )

    assert order_state.status == OrderStatus.CANCELED_MARKET_RESOLVED.value
    assert order_state.fill_price is None
    assert order_state.raw_status == "market_resolved_before_execution"


@pytest.mark.asyncio
async def test_simulator_gtc_fills_remaining_notional_on_later_signal() -> None:
    simulator = BacktestExecutionSimulator()
    initial = await simulator.execute(
        signal=_signal(asks=[{"price": 0.25, "size": 200.0}]),
        decision=_decision(time_in_force=TimeInForce.GTC, limit_price=0.28, notional_usdc=100.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(fill_policy="good_til_cancelled", order_ttl_ms=7_200_000),
    )

    assert initial.status == OrderStatus.PARTIAL.value
    assert initial.remaining_notional_usdc == pytest.approx(50.0)

    later_signal = _signal(
        fetched_at=datetime(2026, 4, 20, 13, 0, tzinfo=UTC),
        asks=[{"price": 0.28, "size": 400.0}],
    )
    advanced = await simulator.advance(
        signal=later_signal,
        execution_model=_execution_model(fill_policy="good_til_cancelled", order_ttl_ms=7_200_000),
    )

    assert len(advanced) == 1
    assert advanced[0].status == OrderStatus.MATCHED.value
    assert advanced[0].filled_notional_usdc == pytest.approx(100.0)
    assert advanced[0].remaining_notional_usdc == pytest.approx(0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("after_ms", [0, 1])
@pytest.mark.parametrize("other_market", [False, True])
async def test_gtc_cancels_at_resolution_without_filling_remainder(
    after_ms: int, other_market: bool,
) -> None:
    simulator = BacktestExecutionSimulator()
    resolved_at = datetime(2026, 4, 20, 12, 0, 1, tzinfo=UTC)
    model = _execution_model(fill_policy="good_til_cancelled")
    initial = await simulator.execute(
        signal=_signal(resolves_at=resolved_at),
        decision=_decision(time_in_force=TimeInForce.GTC),
        execution_model=model,
    )
    assert initial.status == OrderStatus.PARTIAL.value
    signal = replace(
        _signal(fetched_at=resolved_at + timedelta(milliseconds=after_ms)),
        resolves_at=None,
        market_id="other-market" if other_market else "sim-market",
    )
    results = await simulator.advance(signal=signal, execution_model=model)
    assert len(results) == 1
    assert results[0].status == OrderStatus.CANCELED_MARKET_RESOLVED.value
    assert results[0].filled_notional_usdc == initial.filled_notional_usdc
    assert results[0].filled_quantity == initial.filled_quantity
    assert results[0].remaining_notional_usdc == initial.remaining_notional_usdc
    assert simulator.open_orders_ledger == {}


@pytest.mark.asyncio
async def test_simulator_gtc_cancels_on_ttl_and_session_end() -> None:
    simulator = BacktestExecutionSimulator()
    await simulator.execute(
        signal=_signal(asks=[{"price": 0.25, "size": 200.0}]),
        decision=_decision(time_in_force=TimeInForce.GTC, limit_price=0.28, notional_usdc=100.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(fill_policy="good_til_cancelled", order_ttl_ms=100),
    )

    ttl_results = await simulator.advance(
        signal=_signal(
            fetched_at=datetime(2026, 4, 20, 12, 0, 0, 200_000, tzinfo=UTC),
            asks=[{"price": 0.29, "size": 100.0}],
        ),
        execution_model=_execution_model(fill_policy="good_til_cancelled", order_ttl_ms=100),
    )
    assert ttl_results[0].status == OrderStatus.CANCELLED.value
    assert ttl_results[0].raw_status == "cancelled_ttl"

    await simulator.execute(
        signal=_signal(asks=[{"price": 0.25, "size": 200.0}]),
        decision=_decision(time_in_force=TimeInForce.GTC, limit_price=0.28, notional_usdc=100.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(fill_policy="good_til_cancelled"),
    )
    session_end_results = await simulator.cancel_open_orders(
        session_end=datetime(2026, 4, 20, 14, 0, tzinfo=UTC)
    )
    assert session_end_results[0].status == OrderStatus.CANCELLED.value
    assert session_end_results[0].raw_status == "cancelled_session_end"


@pytest.mark.asyncio
async def test_simulator_price_invalidation_streak_and_recovery() -> None:
    simulator = BacktestExecutionSimulator()
    await simulator.execute(
        signal=_signal(asks=[{"price": 0.25, "size": 200.0}]),
        decision=_decision(time_in_force=TimeInForce.GTC, limit_price=0.28, notional_usdc=100.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(
            fill_policy="good_til_cancelled",
            order_ttl_ms=7_200_000,
            price_invalidation_streak=3,
        ),
    )

    for minutes in (1, 2):
        results = await simulator.advance(
            signal=_signal(
                fetched_at=datetime(2026, 4, 20, 12, minutes, tzinfo=UTC),
                asks=[{"price": 0.35, "size": 100.0}],
            ),
            execution_model=_execution_model(
                fill_policy="good_til_cancelled",
                order_ttl_ms=7_200_000,
                price_invalidation_streak=3,
            ),
        )
        assert results == []

    recovery = await simulator.advance(
        signal=_signal(
            fetched_at=datetime(2026, 4, 20, 12, 3, tzinfo=UTC),
            asks=[{"price": 0.27, "size": 300.0}],
        ),
        execution_model=_execution_model(
            fill_policy="good_til_cancelled",
            order_ttl_ms=7_200_000,
            price_invalidation_streak=3,
        ),
    )
    assert recovery[0].status == OrderStatus.MATCHED.value


@pytest.mark.asyncio
async def test_simulator_fok_rejects_when_depth_is_insufficient() -> None:
    simulator = BacktestExecutionSimulator()

    state = await simulator.execute(
        signal=_signal(asks=[{"price": 0.25, "size": 200.0}]),
        decision=_decision(time_in_force=TimeInForce.FOK, limit_price=0.28, notional_usdc=200.0),
        portfolio=_portfolio(),
        execution_model=_execution_model(fill_policy="fill_or_kill"),
    )

    assert state.status == "rejected"
    assert state.raw_status == "fok_unfillable"


@pytest.mark.asyncio
async def test_simulator_rejects_no_order_when_token_book_is_missing() -> None:
    simulator = BacktestExecutionSimulator(replay_engine=_MissingTokenReplayLookup())

    state = await simulator.execute(
        signal=_signal(),
        decision=_decision(
            outcome="NO",
            limit_price=0.30,
            notional_usdc=50.0,
        ),
        portfolio=_portfolio(),
        execution_model=_execution_model(),
    )

    assert state.status == "rejected"
    assert state.raw_status == "missing_token_orderbook"
    assert state.filled_notional_usdc == 0.0


@pytest.mark.asyncio
async def test_simulator_does_not_mutate_source_orderbook() -> None:
    simulator = BacktestExecutionSimulator()
    raw_orderbook = {
        "bids": [{"price": 0.24, "size": 200.0}],
        "asks": [{"price": 0.25, "size": 200.0}, {"price": 0.28, "size": 400.0}],
    }
    signal = _signal(asks=deepcopy(raw_orderbook["asks"]), bids=deepcopy(raw_orderbook["bids"]))

    for _ in range(3):
        await simulator.execute(
            signal=signal,
            decision=_decision(limit_price=0.30, notional_usdc=50.0),
            portfolio=_portfolio(),
            execution_model=_execution_model(),
        )

    assert signal.orderbook == raw_orderbook
