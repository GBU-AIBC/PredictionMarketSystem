from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import inf, nan
from pathlib import Path
from typing import Any, cast

import pytest

from pms.config import ControllerSettings, PMSSettings, RiskSettings
from pms.core.enums import MarketStatus, RunMode, Side, TimeInForce
from pms.core.models import (
    BookSide,
    MarketSignal,
    Opportunity,
    Portfolio,
    Position,
    TradeDecision,
)
from pms.runner import (
    Runner,
    StrategyControllerRuntime,
    _controller_execution_signal_for_decision,
    _decision_evidence_from_signal,
    _decision_evidence_signal_for_decision,
    _decision_expires_at,
)
from pms.strategies.projections import (
    ActiveStrategy,
    EvalSpec,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)


FIXTURE_PATH = Path("tests/fixtures/polymarket_7day_synthetic.jsonl")


def _settings() -> PMSSettings:
    return PMSSettings(
        mode=RunMode.BACKTEST,
        auto_migrate_default_v2=False,
        risk=RiskSettings(
            max_position_per_market=1000.0,
            max_total_exposure=10_000.0,
        ),
    )


def _runner() -> Runner:
    return Runner(
        config=_settings(),
        historical_data_path=FIXTURE_PATH,
    )


def _paper_runner_with_strict_book_age() -> Runner:
    return Runner(
        config=PMSSettings(
            mode=RunMode.PAPER,
            auto_migrate_default_v2=False,
            controller=ControllerSettings(max_book_age_ms=1_000.0),
            risk=RiskSettings(
                max_position_per_market=1000.0,
                max_total_exposure=10_000.0,
            ),
        ),
        historical_data_path=FIXTURE_PATH,
    )


def _signal() -> MarketSignal:
    return MarketSignal(
        market_id="market-cp07",
        token_id="token-cp07-yes",
        venue="polymarket",
        title="Will CP07 persist decisions?",
        yes_price=0.41,
        volume_24h=1200.0,
        resolves_at=datetime(2026, 4, 30, tzinfo=UTC),
        orderbook={
            "bids": [{"price": 0.40, "size": 20.0}],
            "asks": [{"price": 0.42, "size": 25.0}],
        },
        external_signal={"resolved_outcome": 1.0, "book_age_ms": 75.0},
        fetched_at=datetime(2026, 4, 23, 10, 0, tzinfo=UTC),
        market_status=MarketStatus.OPEN.value,
    )


def _opportunity() -> Opportunity:
    return Opportunity(
        opportunity_id="opportunity-cp07",
        market_id="market-cp07",
        token_id="token-cp07-yes",
        side="yes",
        selected_factor_values={"edge": 0.21},
        expected_edge=0.21,
        rationale="persist the decision",
        target_size_usdc=25.0,
        expiry=datetime(2026, 4, 23, 10, 15, tzinfo=UTC),
        staleness_policy="cp07",
        strategy_id="default",
        strategy_version_id="default-v1",
        created_at=datetime(2026, 4, 23, 10, 0, tzinfo=UTC),
        factor_snapshot_hash="snapshot-cp07",
    )


def _decision() -> TradeDecision:
    return TradeDecision(
        decision_id="decision-cp07",
        market_id="market-cp07",
        token_id="token-cp07-yes",
        venue="polymarket",
        side=Side.BUY.value,
        notional_usdc=25.0,
        order_type="limit",
        max_slippage_bps=50,
        stop_conditions=["cp07"],
        prob_estimate=0.68,
        expected_edge=0.21,
        time_in_force=TimeInForce.GTC,
        opportunity_id="opportunity-cp07",
        strategy_id="default",
        strategy_version_id="default-v1",
        limit_price=0.41,
        action=Side.BUY.value,
        model_id="model-cp07",
    )


class _OpportunityAwareControllerDouble:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.last_forecaster_results = [
            {
                "name": "rules",
                "model_id": "rules-v1",
                "status": "accepted",
                "probability": 0.31,
                "confidence": 0.8,
                "rationale": "rules evidence",
            },
            {
                "name": "llm",
                "model_id": "llm-v2",
                "status": "accepted",
                "probability": 0.77,
                "confidence": 0.6,
                "rationale": "llm evidence",
            },
        ]
        self.last_gate_outcomes = [
            {"name": "router", "passed": True, "reason": None},
            {"name": "factor_gates", "passed": True, "reason": None},
        ]

    async def on_signal(
        self,
        signal: MarketSignal,
        portfolio: Portfolio | None = None,
    ) -> tuple[Opportunity, TradeDecision] | None:
        del portfolio
        self.calls.append(signal.market_id)
        return (
            replace(
                _opportunity(),
                created_at=datetime(2026, 4, 23, 10, 0, 1, 250000, tzinfo=UTC),
            ),
            _decision(),
        )


class _RecordingOpportunityStore:
    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self.calls: list[str] = []

    async def insert(self, opportunity: Opportunity) -> None:
        assert self.runner.state.opportunities == []
        self.calls.append(opportunity.opportunity_id)


class _CapacityFillingOpportunityStore(_RecordingOpportunityStore):
    async def insert(self, opportunity: Opportunity) -> None:
        await super().insert(opportunity)
        competing = replace(
            _decision(),
            decision_id="decision-competing-capacity",
            market_id="market-competing-capacity",
            token_id="token-competing-capacity",
        )
        assert self.runner._reserve_position_capacity(competing) is not None  # noqa: SLF001


class _RecordingDecisionStore:
    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self.calls: list[tuple[str, str | None, str]] = []
        self.decision_evidence: dict[str, object] | None = None
        self.transitions: list[tuple[str, str, str, datetime]] = []
        self.created_at: datetime | None = None
        self.expires_at: datetime | None = None

    async def insert(
        self,
        decision: TradeDecision,
        *,
        factor_snapshot_hash: str | None,
        created_at: datetime,
        expires_at: datetime,
        status: str = "pending",
        decision_evidence: dict[str, object] | None = None,
    ) -> None:
        assert self.runner.state.decisions == []
        assert self.runner._decision_queue.empty()  # noqa: SLF001
        self.calls.append((decision.decision_id, factor_snapshot_hash, status))
        self.decision_evidence = decision_evidence
        self.created_at = created_at
        self.expires_at = expires_at

    async def update_status(
        self,
        decision_id: str,
        *,
        current_status: str,
        next_status: str,
        updated_at: datetime,
    ) -> bool:
        if current_status == "accepted" and next_status == "queued":
            assert self.runner._decision_queue.empty()  # noqa: SLF001
        self.transitions.append(
            (decision_id, current_status, next_status, updated_at)
        )
        return True


class _RecordingSweepStore:
    def __init__(self) -> None:
        self.calls: list[datetime] = []
        self.prune_calls: list[datetime] = []

    async def expire_pending(self, *, before: datetime) -> int:
        self.calls.append(before)
        return 2

    async def prune_expired(self, *, before: datetime) -> object:
        self.prune_calls.append(before)
        return object()


@pytest.mark.asyncio
async def test_controller_pipeline_persists_and_enqueues_decision() -> None:
    runner = _runner()
    runner.actuator_executor.risk.record_api_error(401, trace_id="trace-cp07")
    controller = _OpportunityAwareControllerDouble()
    queue: asyncio.Queue[MarketSignal] = asyncio.Queue()
    runner._controller_runtimes["default"] = StrategyControllerRuntime(
        strategy_id="default",
        strategy_version_id="default-v1",
        controller=cast(Any, controller),
        asset_ids=None,
    )
    runner._controller_signal_queues["default"] = queue
    runner.opportunity_store = cast(Any, _RecordingOpportunityStore(runner))
    runner.decision_store = cast(Any, _RecordingDecisionStore(runner))
    runner._controller_task = asyncio.create_task(asyncio.sleep(0))  # noqa: SLF001
    await runner._controller_task
    await queue.put(_signal())

    await asyncio.wait_for(runner._controller_pipeline_loop("default"), timeout=1.0)  # noqa: SLF001

    assert controller.calls == ["market-cp07"]
    assert [item.opportunity_id for item in runner.state.opportunities] == [
        "opportunity-cp07"
    ]
    assert [item.decision_id for item in runner.state.decisions] == ["decision-cp07"]
    work_item = runner._decision_queue.get_nowait()  # noqa: SLF001
    assert work_item.decision.decision_id == "decision-cp07"
    assert work_item.signal is not None
    assert work_item.signal.market_id == "market-cp07"

    decision_store = cast(_RecordingDecisionStore, runner.decision_store)
    assert decision_store.calls == [("decision-cp07", "snapshot-cp07", "pending")]
    assert decision_store.decision_evidence is not None
    assert decision_store.decision_evidence["factor_snapshot_hash"] == "snapshot-cp07"
    assert decision_store.decision_evidence["book_age_ms"] == 75.0
    assert decision_store.decision_evidence["decision_latency_ms"] == 1250.0
    assert decision_store.decision_evidence["quote_source"] == "postgres_snapshot"
    assert decision_store.decision_evidence["book_hash"]
    assert decision_store.decision_evidence["book_top_levels"] == {
        "bids": [{"price": 0.4, "size": 20.0}],
        "asks": [{"price": 0.42, "size": 25.0}],
    }
    trace = cast(dict[str, Any], decision_store.decision_evidence["decision_trace"])
    assert trace["strategy_id"] == "default"
    assert trace["strategy_version_id"] == "default-v1"
    assert trace["forecasts"]["results"] == controller.last_forecaster_results
    assert trace["gates"]["outcomes"][:2] == controller.last_gate_outcomes
    assert trace["risk"]["halted"] is True
    assert trace["risk"]["halt_reason"] == "api_credential_failure"
    assert decision_store.transitions == [
        (
            "decision-cp07",
            "pending",
            "accepted",
            datetime(2026, 4, 23, 10, 0, 1, 250000, tzinfo=UTC),
        ),
        (
            "decision-cp07",
            "accepted",
            "queued",
            datetime(2026, 4, 23, 10, 0, 1, 250000, tzinfo=UTC),
        ),
    ]
    assert decision_store.created_at == datetime(
        2026, 4, 23, 10, 0, 1, 250000, tzinfo=UTC
    )
    assert decision_store.expires_at == datetime(2026, 4, 23, 10, 15, tzinfo=UTC)


@pytest.mark.asyncio
async def test_controller_pipeline_coalesces_superseded_token_signals_before_forecast() -> None:
    runner = _paper_runner_with_strict_book_age()
    controller = _OpportunityAwareControllerDouble()
    queue: asyncio.Queue[MarketSignal] = asyncio.Queue()
    runner._controller_runtimes["default"] = StrategyControllerRuntime(  # noqa: SLF001
        strategy_id="default",
        strategy_version_id="default-v1",
        controller=cast(Any, controller),
        asset_ids=None,
    )
    runner._controller_signal_queues["default"] = queue  # noqa: SLF001
    runner.opportunity_store = cast(Any, _RecordingOpportunityStore(runner))
    runner.decision_store = cast(Any, _RecordingDecisionStore(runner))
    runner._controller_task = asyncio.create_task(asyncio.sleep(0))  # noqa: SLF001
    await runner._controller_task
    now = datetime.now(tz=UTC)
    older = replace(
        _signal(),
        market_id="market-superseded-cp07",
        token_id="token-cp07-yes",
        fetched_at=now,
        orderbook={
            "bids": [{"price": 0.40, "size": 20.0}],
            "asks": [{"price": 0.41, "size": 100.0}],
        },
        external_signal={
            **_signal().external_signal,
            "book_received_at": now.isoformat(),
        },
    )
    newer = replace(
        _signal(),
        market_id="market-latest-cp07",
        token_id="token-cp07-yes",
        fetched_at=now + timedelta(milliseconds=10),
        orderbook={
            "bids": [{"price": 0.40, "size": 20.0}],
            "asks": [{"price": 0.41, "size": 100.0}],
        },
        external_signal={
            **_signal().external_signal,
            "book_received_at": (now + timedelta(milliseconds=10)).isoformat(),
        },
    )
    runner._remember_paper_orderbook(newer)  # noqa: SLF001
    await queue.put(older)
    await queue.put(newer)

    await asyncio.wait_for(
        runner._controller_pipeline_loop("default"),  # noqa: SLF001
        timeout=1.0,
    )

    assert controller.calls == ["market-latest-cp07"]
    assert runner.state.controller_diagnostics == []


@pytest.mark.asyncio
async def test_controller_pipeline_drops_stale_queued_book_signal_before_controller() -> None:
    runner = _paper_runner_with_strict_book_age()
    controller = _OpportunityAwareControllerDouble()
    queue: asyncio.Queue[MarketSignal] = asyncio.Queue()
    runner._controller_runtimes["default"] = StrategyControllerRuntime(  # noqa: SLF001
        strategy_id="default",
        strategy_version_id="default-v1",
        controller=cast(Any, controller),
        asset_ids=None,
    )
    runner._controller_signal_queues["default"] = queue  # noqa: SLF001
    runner.opportunity_store = cast(Any, _RecordingOpportunityStore(runner))
    runner.decision_store = cast(Any, _RecordingDecisionStore(runner))
    runner._controller_task = asyncio.create_task(asyncio.sleep(0))  # noqa: SLF001
    await runner._controller_task
    now = datetime.now(tz=UTC)
    stale = replace(
        _signal(),
        market_id="market-stale-cp07",
        token_id="token-stale-cp07",
        fetched_at=now,
        external_signal={
            **_signal().external_signal,
            "book_received_at": (now - timedelta(seconds=5)).isoformat(),
        },
    )
    fresh = replace(
        _signal(),
        market_id="market-fresh-cp07",
        token_id="token-fresh-cp07",
        fetched_at=now,
        external_signal={
            **_signal().external_signal,
            "book_received_at": (now + timedelta(milliseconds=100)).isoformat(),
        },
    )
    await queue.put(stale)
    await queue.put(fresh)

    await asyncio.wait_for(
        runner._controller_pipeline_loop("default"),  # noqa: SLF001
        timeout=1.0,
    )

    assert controller.calls == ["market-fresh-cp07"]
    stale_diagnostic = runner.state.controller_diagnostics[0]
    assert stale_diagnostic.code == "router_gate:book_too_stale"
    assert stale_diagnostic.market_id == "market-stale-cp07"
    assert stale_diagnostic.metadata["phase"] == "queue_dequeue"
    assert stale_diagnostic.metadata["max_book_age_ms"] == 1_000.0


@pytest.mark.asyncio
async def test_controller_pipeline_allows_stale_queued_signal_when_controller_can_refresh_book() -> None:
    runner = _paper_runner_with_strict_book_age()
    controller = _OpportunityAwareControllerDouble()
    setattr(controller, "direct_book_reader", object())
    queue: asyncio.Queue[MarketSignal] = asyncio.Queue()
    runner._controller_runtimes["default"] = StrategyControllerRuntime(  # noqa: SLF001
        strategy_id="default",
        strategy_version_id="default-v1",
        controller=cast(Any, controller),
        asset_ids=None,
    )
    runner._controller_signal_queues["default"] = queue  # noqa: SLF001
    runner.opportunity_store = cast(Any, _RecordingOpportunityStore(runner))
    runner.decision_store = cast(Any, _RecordingDecisionStore(runner))
    runner._controller_task = asyncio.create_task(asyncio.sleep(0))  # noqa: SLF001
    await runner._controller_task
    now = datetime.now(tz=UTC)
    stale_refreshable = replace(
        _signal(),
        market_id="market-refreshable-cp07",
        token_id="token-cp07-yes",
        fetched_at=now,
        orderbook={
            "bids": [{"price": 0.40, "size": 20.0}],
            "asks": [{"price": 0.41, "size": 100.0}],
        },
        external_signal={
            **_signal().external_signal,
            "book_received_at": (now - timedelta(seconds=5)).isoformat(),
        },
    )
    runner._remember_paper_orderbook(stale_refreshable)  # noqa: SLF001
    await queue.put(stale_refreshable)

    await asyncio.wait_for(
        runner._controller_pipeline_loop("default"),  # noqa: SLF001
        timeout=1.0,
    )

    assert controller.calls == ["market-refreshable-cp07"]
    assert runner.state.controller_diagnostics == []


@pytest.mark.asyncio
async def test_controller_pipeline_skips_runtime_state_when_enqueue_rejects() -> None:
    settings = PMSSettings(
        mode=RunMode.BACKTEST,
        auto_migrate_default_v2=False,
        risk=RiskSettings(
            max_open_positions=2,
            max_position_per_market=1000.0,
            max_total_exposure=10_000.0,
        ),
    )
    runner = Runner(config=settings, historical_data_path=FIXTURE_PATH)
    runner.portfolio = replace(
        runner.portfolio,
        open_positions=[
            Position(
                market_id="market-existing-capacity",
                token_id="token-existing-capacity",
                venue="polymarket",
                side=Side.BUY.value,
                shares_held=10.0,
                avg_entry_price=0.5,
                unrealized_pnl=0.0,
                locked_usdc=5.0,
            )
        ],
    )
    controller = _OpportunityAwareControllerDouble()
    queue: asyncio.Queue[MarketSignal] = asyncio.Queue()
    runner._controller_runtimes["default"] = StrategyControllerRuntime(  # noqa: SLF001
        strategy_id="default",
        strategy_version_id="default-v1",
        controller=cast(Any, controller),
        asset_ids=None,
    )
    runner._controller_signal_queues["default"] = queue  # noqa: SLF001
    runner.opportunity_store = cast(Any, _CapacityFillingOpportunityStore(runner))
    runner.decision_store = cast(Any, _RecordingDecisionStore(runner))
    runner._controller_task = asyncio.create_task(asyncio.sleep(0))  # noqa: SLF001
    await runner._controller_task
    await queue.put(_signal())

    await asyncio.wait_for(
        runner._controller_pipeline_loop("default"),  # noqa: SLF001
        timeout=1.0,
    )

    decision_store = cast(_RecordingDecisionStore, runner.decision_store)
    assert runner._decision_queue.empty()  # noqa: SLF001
    assert runner.state.decisions == []
    assert decision_store.transitions == [
        (
            "decision-cp07",
            "pending",
            "accepted",
            datetime(2026, 4, 23, 10, 0, 1, 250000, tzinfo=UTC),
        ),
        (
            "decision-cp07",
            "accepted",
            "rejected",
            datetime(2026, 4, 23, 10, 0, 1, 250000, tzinfo=UTC),
        ),
    ]


@pytest.mark.asyncio
async def test_runner_sweep_expired_decisions_once_delegates_to_store() -> None:
    runner = _runner()
    runner.decision_store = cast(Any, _RecordingSweepStore())
    now = datetime(2026, 4, 23, 11, 0, tzinfo=UTC)

    expired = await runner._sweep_expired_decisions_once(now=now)  # noqa: SLF001

    assert expired == 2
    store = cast(_RecordingSweepStore, runner.decision_store)
    assert store.calls == [now]
    assert store.prune_calls == [now - timedelta(hours=24)]


def test_decision_expiry_uses_signal_resolution_when_opportunity_has_no_expiry() -> None:
    created_at = datetime(2026, 4, 23, 10, 0, tzinfo=UTC)
    opportunity = _opportunity()
    signal = _signal()
    object.__setattr__(opportunity, "expiry", None)
    object.__setattr__(signal, "resolves_at", datetime(2026, 4, 23, 10, 5, tzinfo=UTC))

    expires_at = _decision_expires_at(signal, opportunity, created_at=created_at)

    assert expires_at == datetime(2026, 4, 23, 10, 5, tzinfo=UTC)


def test_decision_evidence_drops_non_finite_orderbook_values() -> None:
    signal = replace(
        _signal(),
        orderbook={
            "bids": [
                {"price": nan, "size": 10.0},
                {"price": 0.39, "size": inf},
                {"price": 0.38, "size": 7.0},
            ],
            "asks": [
                {"price": "Inf", "size": 8.0},
                {"price": 0.42, "size": "-Infinity"},
                {"price": 0.43, "size": 9.0},
            ],
        },
        external_signal={"book_age_ms": "NaN"},
    )

    evidence = _decision_evidence_from_signal(
        signal,
        settings=_settings(),
        decision=_decision(),
        factor_snapshot_hash="snapshot-cp07",
        quote_source="postgres_snapshot",
        decision_created_at=signal.fetched_at,
    )

    assert evidence["book_age_ms"] is None
    assert evidence["book_top_levels"] == {
        "bids": [{"price": 0.38, "size": 7.0}],
        "asks": [{"price": 0.43, "size": 9.0}],
    }
    json.dumps(evidence, allow_nan=False)


def test_decision_evidence_computes_book_age_for_live_market_data_signal() -> None:
    book_received_at = _signal().fetched_at + timedelta(milliseconds=400)
    signal = replace(
        _signal(),
        external_signal={
            "raw_event_type": "book",
            "book_received_at": book_received_at.isoformat(),
        },
    )
    evidence = _decision_evidence_from_signal(
        signal,
        settings=_settings(),
        decision=_decision(),
        factor_snapshot_hash="snapshot-cp07",
        quote_source="postgres_snapshot",
        decision_created_at=signal.fetched_at + timedelta(milliseconds=1250),
    )

    assert evidence["book_age_ms"] == pytest.approx(850.0)
    external_signal_keys = evidence["external_signal_keys"]
    assert isinstance(external_signal_keys, list)
    assert "book_received_at" in external_signal_keys


def test_decision_evidence_records_submitted_token_when_buy_no_uses_yes_book() -> None:
    signal = replace(_signal(), token_id="token-cp07-yes")
    decision = replace(
        _decision(),
        token_id="token-cp07-no",
        outcome="NO",
        spread_bps_at_decision=500,
    )

    evidence = _decision_evidence_from_signal(
        signal,
        settings=_settings(),
        decision=decision,
        factor_snapshot_hash="snapshot-cp07",
        quote_source="postgres_snapshot",
        decision_created_at=signal.fetched_at,
    )

    assert evidence["book_token_id"] == "token-cp07-yes"
    assert evidence["decision_token_id"] == "token-cp07-no"
    assert evidence["decision_outcome"] == "NO"
    assert evidence["decision_side"] == "BUY"
    assert evidence["spread_bps_at_decision"] == 500
    assert evidence["book_hash"]


def test_decision_evidence_prefers_recent_target_token_signal_for_flipped_side() -> None:
    yes_signal = replace(
        _signal(),
        token_id="token-cp07-yes",
        orderbook={
            "bids": [{"price": 0.40, "size": 20.0}],
            "asks": [{"price": 0.42, "size": 25.0}],
        },
        external_signal={
            "book_age_ms": 70.0,
            "yes_token_id": "token-cp07-yes",
            "no_token_id": "token-cp07-no",
            "signal_token_outcome": "YES",
        },
    )
    no_signal = replace(
        _signal(),
        token_id="token-cp07-no",
        yes_price=0.59,
        orderbook={
            "bids": [{"price": 0.58, "size": 30.0}],
            "asks": [{"price": 0.60, "size": 35.0}],
        },
        external_signal={
            "book_age_ms": 80.0,
            "yes_token_id": "token-cp07-yes",
            "no_token_id": "token-cp07-no",
            "signal_token_outcome": "NO",
        },
    )
    decision = replace(
        _decision(),
        token_id="token-cp07-no",
        outcome="NO",
        limit_price=0.60,
    )

    evidence_signal = _decision_evidence_signal_for_decision(
        yes_signal,
        decision,
        latest_signals_by_token={"token-cp07-no": no_signal},
    )
    evidence = _decision_evidence_from_signal(
        evidence_signal,
        settings=_settings(),
        decision=decision,
        factor_snapshot_hash="snapshot-cp07",
        quote_source="postgres_snapshot",
        decision_created_at=yes_signal.fetched_at,
    )

    assert evidence["book_token_id"] == "token-cp07-no"
    assert evidence["book_token_outcome"] == "NO"
    assert evidence["book_top_levels"] == {
        "bids": [{"price": 0.58, "size": 30.0}],
        "asks": [{"price": 0.60, "size": 35.0}],
    }
    assert evidence["book_age_ms"] == 80.0
    assert evidence["mid_quote_baseline_prob_estimate"] == pytest.approx(0.41)


def test_decision_evidence_prefers_controller_execution_signal() -> None:
    class ControllerWithExecutionSignal:
        last_execution_signal: MarketSignal | None

        def __init__(self, execution_signal: MarketSignal) -> None:
            self.last_execution_signal = execution_signal

    raw_signal = replace(
        _signal(),
        token_id="token-cp07-yes",
        orderbook={
            "bids": [{"price": 0.40, "size": 20.0}],
            "asks": [{"price": 0.42, "size": 25.0}],
        },
        external_signal={
            "book_age_ms": 30_000.0,
            "yes_token_id": "token-cp07-yes",
            "no_token_id": "token-cp07-no",
            "signal_token_outcome": "YES",
        },
    )
    execution_signal = replace(
        raw_signal,
        token_id="token-cp07-no",
        yes_price=0.59,
        orderbook={
            "bids": [{"price": 0.58, "size": 30.0}],
            "asks": [{"price": 0.60, "size": 35.0}],
        },
        external_signal={
            "book_age_ms": 12.0,
            "yes_token_id": "token-cp07-yes",
            "no_token_id": "token-cp07-no",
            "signal_token_outcome": "NO",
            "direct_outcome_book_source": "venue_direct",
        },
    )
    decision = replace(
        _decision(),
        token_id="token-cp07-no",
        outcome="NO",
        limit_price=0.60,
    )

    evidence_signal = _controller_execution_signal_for_decision(
        ControllerWithExecutionSignal(execution_signal),
        fallback_signal=raw_signal,
        decision=decision,
        latest_signals_by_token={},
    )
    evidence = _decision_evidence_from_signal(
        evidence_signal,
        settings=_settings(),
        decision=decision,
        factor_snapshot_hash="snapshot-cp07",
        quote_source="postgres_snapshot",
        decision_created_at=raw_signal.fetched_at,
    )

    assert evidence_signal is execution_signal
    assert evidence["book_token_id"] == "token-cp07-no"
    assert evidence["book_age_ms"] == 12.0
    assert evidence["direct_outcome_book_source"] == "venue_direct"
    assert evidence["book_top_levels"] == {
        "bids": [{"price": 0.58, "size": 30.0}],
        "asks": [{"price": 0.60, "size": 35.0}],
    }


def test_decision_evidence_records_secondary_baseline_probabilities() -> None:
    signal = replace(
        _signal(),
        external_signal={
            "resolved_outcome": 1.0,
            "book_age_ms": 75.0,
            "last_trade_price": 0.43,
            "category_prior_baseline_prob_estimate": 0.52,
        },
    )
    decision = replace(
        _decision(),
        token_id="token-cp07-no",
        outcome="NO",
        limit_price=0.39,
    )

    evidence = _decision_evidence_from_signal(
        signal,
        settings=_settings(),
        decision=decision,
        factor_snapshot_hash="snapshot-cp07",
        quote_source="postgres_snapshot",
        decision_created_at=signal.fetched_at,
    )

    assert evidence["market_implied_baseline_prob_estimate"] == pytest.approx(0.61)
    assert evidence["baseline_probability_coordinate"] == "YES"
    assert evidence["decision_outcome_market_implied_prob_estimate"] == (
        pytest.approx(0.39)
    )
    assert evidence["mid_quote_baseline_prob_estimate"] == pytest.approx(0.41)
    assert evidence["last_trade_baseline_prob_estimate"] == pytest.approx(0.43)
    assert evidence["category_prior_baseline_prob_estimate"] == pytest.approx(0.52)


def test_decision_evidence_records_cost_basis_at_decision_time() -> None:
    signal = replace(
        _signal(),
        external_signal={
            "resolved_outcome": 1.0,
            "book_age_ms": 75.0,
            "fee_rate_bps": 300.0,
        },
    )
    decision = replace(
        _decision(),
        expected_edge=0.21,
        limit_price=0.41,
        max_slippage_bps=50,
        spread_bps_at_decision=80,
    )

    evidence = _decision_evidence_from_signal(
        signal,
        settings=_settings(),
        decision=decision,
        factor_snapshot_hash="snapshot-cp07",
        quote_source="postgres_snapshot",
        decision_created_at=signal.fetched_at,
    )

    assert evidence["fee_rate_bps"] == pytest.approx(300.0)
    assert evidence["fee_rate"] == pytest.approx(0.03)
    assert evidence["fee_edge_at_decision"] == pytest.approx(0.0177)
    assert evidence["spread_edge_at_decision"] == pytest.approx(0.00328)
    assert evidence["slippage_edge_at_decision"] == pytest.approx(0.00205)
    assert evidence["total_cost_edge_at_decision"] == pytest.approx(0.02303)
    assert evidence["net_edge_after_costs"] == pytest.approx(0.18697)


def test_decision_evidence_excludes_spread_when_already_in_price() -> None:
    """Ask-frame sibling of the cost-basis test: when the decision price is
    the executable ask, the spread is already inside expected_edge and must
    not be charged again — but the spread_edge_at_decision key stays present
    (value 0.0) for the h1_flb smoke check."""
    signal = replace(
        _signal(),
        external_signal={
            "resolved_outcome": 1.0,
            "book_age_ms": 75.0,
            "fee_rate_bps": 300.0,
        },
    )
    decision = replace(
        _decision(),
        expected_edge=0.21,
        limit_price=0.41,
        max_slippage_bps=50,
        spread_bps_at_decision=80,
    )

    evidence = _decision_evidence_from_signal(
        signal,
        settings=_settings(),
        decision=decision,
        factor_snapshot_hash="snapshot-cp07",
        quote_source="postgres_snapshot",
        decision_created_at=signal.fetched_at,
        spread_already_in_price=True,
    )

    assert evidence["spread_edge_at_decision"] == 0.0
    assert evidence["spread_already_in_price"] is True
    assert evidence["fee_edge_at_decision"] == pytest.approx(0.0177)
    assert evidence["slippage_edge_at_decision"] == pytest.approx(0.00205)
    assert evidence["total_cost_edge_at_decision"] == pytest.approx(0.01975)
    assert evidence["net_edge_after_costs"] == pytest.approx(0.19025)


def test_decision_evidence_records_ask_frame_without_spread_bps() -> None:
    signal = replace(
        _signal(),
        external_signal={
            "resolved_outcome": 1.0,
            "book_age_ms": 75.0,
            "fee_rate_bps": 300.0,
        },
    )
    decision = replace(
        _decision(),
        expected_edge=0.21,
        limit_price=0.41,
        max_slippage_bps=50,
        spread_bps_at_decision=None,
    )

    evidence = _decision_evidence_from_signal(
        signal,
        settings=_settings(),
        decision=decision,
        factor_snapshot_hash="snapshot-cp07",
        quote_source="postgres_snapshot",
        decision_created_at=signal.fetched_at,
        spread_already_in_price=True,
    )

    assert evidence["spread_edge_at_decision"] == 0.0
    assert evidence["spread_already_in_price"] is True
    assert evidence["fee_edge_at_decision"] == pytest.approx(0.0177)
    assert evidence["slippage_edge_at_decision"] == pytest.approx(0.00205)
    assert evidence["total_cost_edge_at_decision"] == pytest.approx(0.01975)
    assert evidence["net_edge_after_costs"] == pytest.approx(0.19025)


def test_decision_evidence_projects_direct_no_book_baselines_to_yes_probability() -> None:
    signal = replace(
        _signal(),
        token_id="token-cp07-no",
        yes_price=0.585,
        orderbook={
            "bids": [{"price": 0.58, "size": 20.0}],
            "asks": [{"price": 0.60, "size": 25.0}],
        },
        external_signal={
            "resolved_outcome": 1.0,
            "book_age_ms": 75.0,
            "last_trade_price": 0.57,
        },
    )
    decision = replace(
        _decision(),
        token_id="token-cp07-no",
        outcome="NO",
        limit_price=0.60,
    )

    evidence = _decision_evidence_from_signal(
        signal,
        settings=_settings(),
        decision=decision,
        factor_snapshot_hash="snapshot-cp07",
        quote_source="postgres_snapshot",
        decision_created_at=signal.fetched_at,
    )

    assert evidence["book_token_id"] == "token-cp07-no"
    assert evidence["book_token_outcome"] == "NO"
    assert evidence["yes_price"] == pytest.approx(0.415)
    assert evidence["direct_token_price"] == pytest.approx(0.585)
    assert evidence["market_implied_baseline_prob_estimate"] == pytest.approx(0.40)
    assert evidence["baseline_probability_coordinate"] == "YES"
    assert evidence["decision_outcome_market_implied_prob_estimate"] == (
        pytest.approx(0.60)
    )
    assert evidence["mid_quote_baseline_prob_estimate"] == pytest.approx(0.41)
    assert evidence["last_trade_baseline_prob_estimate"] == pytest.approx(0.43)


def _active_strategy_with_metadata(
    metadata: tuple[tuple[str, str], ...],
) -> ActiveStrategy:
    return ActiveStrategy(
        strategy_id="default",
        strategy_version_id="default-v1",
        config=StrategyConfig(
            strategy_id="default",
            factor_composition=(),
            metadata=metadata,
        ),
        risk=RiskParams(
            max_position_notional_usdc=500.0,
            max_daily_drawdown_pct=2.5,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(metrics=("brier", "pnl")),
        forecaster=ForecasterSpec(forecasters=(("rules", ()),)),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=100.0,
        ),
    )


class _ControllerWithStrategy:
    def __init__(self, strategy: ActiveStrategy) -> None:
        self.strategy = strategy

    async def decide(
        self,
        signal: MarketSignal,
        portfolio: Portfolio | None = None,
    ) -> TradeDecision | None:
        del signal, portfolio
        return None


def _runner_with_runtime_strategy(
    metadata: tuple[tuple[str, str], ...],
) -> Runner:
    runner = _runner()
    runner._controller_runtimes["default"] = StrategyControllerRuntime(  # noqa: SLF001
        strategy_id="default",
        strategy_version_id="default-v1",
        controller=cast(Any, _ControllerWithStrategy(
            _active_strategy_with_metadata(metadata)
        )),
        asset_ids=None,
    )
    return runner


def test_decision_uses_ask_pricing_requires_ask_frame_and_matching_price() -> None:
    runner = _runner_with_runtime_strategy((("price_reference", "best_ask"),))
    signal = _signal()  # best executable ask = 0.42
    ask_priced = replace(_decision(), limit_price=0.42)
    mid_priced = replace(_decision(), limit_price=0.41)

    assert runner._decision_uses_ask_pricing(ask_priced, signal) is True  # noqa: SLF001
    assert runner._decision_uses_ask_pricing(mid_priced, signal) is False  # noqa: SLF001


def test_decision_uses_ask_pricing_is_false_without_best_ask_metadata() -> None:
    runner = _runner_with_runtime_strategy((("owner", "test"),))
    ask_priced = replace(_decision(), limit_price=0.42)

    assert runner._decision_uses_ask_pricing(ask_priced, _signal()) is False  # noqa: SLF001


def test_decision_uses_ask_pricing_is_false_for_unknown_strategy() -> None:
    runner = _runner()
    ask_priced = replace(_decision(), limit_price=0.42)

    assert runner._decision_uses_ask_pricing(ask_priced, _signal()) is False  # noqa: SLF001


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [Side.BUY.value, Side.SELL.value])
@pytest.mark.parametrize(
    ("venue_fee_bps", "strategy_fee", "expected_rate"),
    [(None, None, 0.07), (None, 0.02, 0.02), (1000.0, 0.02, 0.10), (0.0, 0.02, 0.0)],
)
async def test_controller_persistence_keeps_fee_inclusive_costs(
    action: BookSide,
    venue_fee_bps: float | None,
    strategy_fee: float | None,
    expected_rate: float,
) -> None:
    # Price/size/edge reproduce the public PAPER fill from 2026-10-18.
    metadata: tuple[tuple[str, str], ...] = (("price_reference", "best_ask"),)
    if strategy_fee is not None:
        metadata += (("fee_rate", str(strategy_fee)),)
    strategy = _active_strategy_with_metadata(metadata)

    class CostedController(_OpportunityAwareControllerDouble):
        strategy: ActiveStrategy

        async def on_signal(
            self, signal: MarketSignal, portfolio: Portfolio | None = None,
        ) -> tuple[Opportunity, TradeDecision] | None:
            return (
                replace(_opportunity(), target_size_usdc=1.0),
                replace(
                    _decision(), side=action, action=action, notional_usdc=1.0,
                    prob_estimate=0.99, limit_price=0.978,
                    expected_edge=0.012, spread_bps_at_decision=20,
                ),
            )

    controller = CostedController()
    controller.strategy = strategy
    runner = _runner()
    runner._active_strategies_by_version[("default", "default-v1")] = strategy
    queue: asyncio.Queue[MarketSignal] = asyncio.Queue()
    runner._controller_runtimes["default"] = StrategyControllerRuntime(
        strategy_id="default", strategy_version_id="default-v1",
        controller=cast(Any, controller), asset_ids=None,
    )
    runner._controller_signal_queues["default"] = queue
    runner.opportunity_store = cast(Any, _RecordingOpportunityStore(runner))
    store = _RecordingDecisionStore(runner)
    runner.decision_store = cast(Any, store)
    runner._controller_task = asyncio.create_task(asyncio.sleep(0))
    await runner._controller_task
    external: dict[str, Any] = {}
    if venue_fee_bps is not None:
        external["fee_rate_bps"] = venue_fee_bps
    await queue.put(replace(
        _signal(), yes_price=0.977, external_signal=external,
        orderbook={"bids": [{"price": 0.976, "size": 979.0}],
                   "asks": [{"price": 0.978, "size": 55.0}]},
    ))

    await asyncio.wait_for(runner._controller_pipeline_loop("default"), timeout=1.0)

    assert store.decision_evidence is not None
    evidence = store.decision_evidence
    fee_edge = expected_rate * (1.0 - 0.978)
    assert evidence["fee_rate"] == pytest.approx(expected_rate)
    assert evidence["fee_edge_at_decision"] == pytest.approx(fee_edge)
    assert evidence["spread_edge_at_decision"] == 0.0
    assert evidence["slippage_edge_at_decision"] == pytest.approx(0.00489)
    assert evidence["total_cost_edge_at_decision"] == pytest.approx(fee_edge + 0.00489)
    assert evidence["net_edge_after_costs"] == pytest.approx(0.012 - fee_edge - 0.00489)
