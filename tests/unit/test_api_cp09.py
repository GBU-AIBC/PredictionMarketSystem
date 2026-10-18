from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import SecretStr

from pms.api.app import create_app
from pms.actuator.risk import RiskTradeResult
from pms.config import (
    ControllerSettings,
    DiscordSettings,
    PMSSettings,
    PolymarketSettings,
    RiskSettings,
)
from pms.core.enums import (
    FeedbackSource,
    FeedbackTarget,
    OrderStatus,
    RunMode,
    Side,
    TimeInForce,
)
from pms.core.models import (
    EvalRecord,
    Feedback,
    FillRecord,
    LiveTradingDisabledError,
    MarketSignal,
    Position,
    QuoteEvalRecord,
    TradeDecision,
)
from pms.metrics import (
    MARKETS_SNAPSHOT_LAG_SECONDS_MAX_METRIC,
    SENSOR_DISCOVERY_PRICE_FIELDS_POPULATED_RATIO_METRIC,
    SENSOR_DISCOVERY_SNAPSHOTS_WRITTEN_TOTAL_METRIC,
    set_metric,
)
from pms.runner import Runner
from pms.storage.eval_store import EvalStore
from pms.storage.feedback_store import FeedbackStore
from pms.storage.fill_store import FillStore
from pms.storage.quote_eval_store import QuoteEvalStore
from pms.storage.schema_check import SchemaVersionMismatchError
from tests.support.fake_stores import InMemoryEvalStore, InMemoryFeedbackStore
from tests.support.live_paths import make_live_report_paths, make_private_live_paths


FIXTURE_PATH = Path("tests/fixtures/polymarket_7day_synthetic.jsonl")
AUTH_HEADERS = {"Authorization": "Bearer live-api-token"}


class _AuditConnection:
    async def execute(self, *_args: object) -> None:
        return


class _AuditAcquire:
    async def __aenter__(self) -> _AuditConnection:
        return _AuditConnection()

    async def __aexit__(self, *_args: object) -> None:
        return


class _AvailableAuditPool:
    def acquire(self) -> _AuditAcquire:
        return _AuditAcquire()

    async def fetch(self, *_args: object) -> list[object]:
        return []

    async def fetchval(self, *_args: object) -> int:
        return 1


class ReportingFillStoreDouble:
    async def read_trades(
        self,
        *,
        limit: int,
        offset: int = 0,
        until: datetime | None = None,
    ) -> list[Any]:
        del limit, offset, until
        return []

    async def read_positions(self) -> list[Position]:
        return []


def _settings() -> PMSSettings:
    return PMSSettings(
        mode=RunMode.BACKTEST,
        auto_migrate_default_v2=False,
        risk=RiskSettings(
            max_position_per_market=1000.0,
            max_total_exposure=10_000.0,
        ),
    )


def _stub_schema_check(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop_schema_check(_pool: object) -> None:
        return

    monkeypatch.setattr("pms.api.app.ensure_schema_current", _noop_schema_check)


def _stub_orphan_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop_scan(_pool: object) -> None:
        return

    monkeypatch.setattr("pms.api.app.scan_orphaned_backtest_runs", _noop_scan)


def _runner_with_state() -> Runner:
    pending = _feedback("fb-pending")
    runner = Runner(
        config=_settings(),
        eval_store=cast(EvalStore, InMemoryEvalStore([_eval_record()])),
        feedback_store=cast(
            FeedbackStore,
            InMemoryFeedbackStore(
                [
                    pending,
                    replace(_feedback("fb-resolved"), resolved=True),
                ]
            ),
        ),
    )
    runner.state.runner_started_at = datetime(2026, 4, 14, tzinfo=UTC)
    runner.state.signals.append(_signal())
    runner.state.decisions.append(_decision())
    runner.state.fills.append(_fill())
    return runner


def _live_transition_settings(
    *,
    include_operator_readiness: bool,
) -> PMSSettings:
    approval_path, first_order_audit_path = make_private_live_paths(
        prefix="pms-api-live-transition-"
    )
    paper_report_path, rehearsal_report_path = make_live_report_paths(
        prefix="pms-api-live-transition-reports-"
    )
    attested_at = datetime.now(tz=UTC)
    return PMSSettings(
        mode=RunMode.PAPER,
        live_trading_enabled=True,
        secret_source="fly",
        api_token="live-api-token",
        auto_migrate_default_v2=False,
        live_emergency_audit_path=str(
            Path(approval_path).parent / "live-emergency-audit.jsonl"
        ),
        live_first_order_audit_path=first_order_audit_path,
        live_paper_soak_report_path=(
            paper_report_path if include_operator_readiness else None
        ),
        live_operator_rehearsal_report_path=(
            rehearsal_report_path if include_operator_readiness else None
        ),
        live_preflight_artifact_path=str(
            Path(approval_path).parent / "credentialed-preflight.json"
        ),
        live_exit_criteria_ratified_by=(
            "operator" if include_operator_readiness else None
        ),
        live_exit_criteria_ratified_at=(
            attested_at if include_operator_readiness else None
        ),
        live_compliance_reviewed_by=(
            "counsel" if include_operator_readiness else None
        ),
        live_compliance_reviewed_at=(
            attested_at if include_operator_readiness else None
        ),
        live_compliance_jurisdiction=(
            "US-operator-approved" if include_operator_readiness else None
        ),
        risk=RiskSettings(
            max_position_per_market=50.0,
            max_total_exposure=50.0,
            max_drawdown_pct=20.0,
            max_daily_loss_usdc=20.0,
            max_open_positions=5,
            max_exposure_per_risk_group=15.0,
            max_quantity_shares=500.0,
            min_order_usdc=1.0,
        ),
        controller=ControllerSettings(time_in_force="IOC", quote_source="dual"),
        discord=DiscordSettings(
            webhook_url=SecretStr("https://discord.example/webhooks/api/unit"),
            alert_dir=str(Path(approval_path).parent / "discord-alerts"),
        ),
        polymarket=PolymarketSettings(
            private_key="private-key",
            api_key="api-key",
            api_secret="api-secret",
            api_passphrase="passphrase",
            signature_type=1,
            funder_address="0x1111111111111111111111111111111111111111",
            operator_approval_mode="every_order",
            first_live_order_approval_path=approval_path,
        ),
    )


def _signal() -> MarketSignal:
    return MarketSignal(
        market_id="api-market",
        token_id="yes-token",
        venue="polymarket",
        title="Will the API expose state?",
        yes_price=0.4,
        volume_24h=1000.0,
        resolves_at=datetime(2026, 4, 20, tzinfo=UTC),
        orderbook={"bids": [], "asks": []},
        external_signal={"metaculus_prob": 0.7},
        fetched_at=datetime(2026, 4, 14, tzinfo=UTC),
        market_status="open",
    )


async def _empty_signal_stream() -> AsyncIterator[MarketSignal]:
    signals: tuple[MarketSignal, ...] = ()
    for signal in signals:
        yield signal


def _decision() -> TradeDecision:
    return TradeDecision(
        decision_id="decision-api",
        market_id="api-market",
        token_id="yes-token",
        venue="polymarket",
        side=Side.BUY.value,
        limit_price=0.4,
        notional_usdc=12.5,
        order_type="limit",
        max_slippage_bps=100,
        stop_conditions=["min_volume:100.00"],
        prob_estimate=0.7,
        expected_edge=0.3,
        time_in_force=TimeInForce.GTC,
        opportunity_id="opportunity-api",
        strategy_id="default",
        strategy_version_id="default-v1",
        action=Side.BUY.value,
        model_id="model-a",
    )


def _fill() -> FillRecord:
    now = datetime(2026, 4, 14, tzinfo=UTC)
    return FillRecord(
        trade_id="trade-api",
        order_id="order-api",
        decision_id="decision-api",
        market_id="api-market",
        token_id="yes-token",
        venue="polymarket",
        side=Side.BUY.value,
        fill_price=0.41,
        fill_notional_usdc=12.5,
        fill_quantity=12.5 / 0.41,
        executed_at=now,
        filled_at=now,
        status=OrderStatus.MATCHED.value,
        anomaly_flags=[],
        strategy_id="default",
        strategy_version_id="default-v1",
        resolved_outcome=1.0,
    )


def _eval_record() -> EvalRecord:
    return EvalRecord(
        market_id="api-market",
        decision_id="decision-api",
        strategy_id="default",
        strategy_version_id="default-v1",
        prob_estimate=0.7,
        baseline_prob_estimate=0.4,
        resolved_outcome=1.0,
        brier_score=0.09,
        baseline_brier_score=0.36,
        fill_status=OrderStatus.MATCHED.value,
        recorded_at=datetime(2026, 4, 14, tzinfo=UTC),
        citations=["trade-api"],
        category="model-a",
        model_id="model-a",
        pnl=1.0,
        slippage_bps=10.0,
        filled=True,
    )


def _feedback(feedback_id: str) -> Feedback:
    return Feedback(
        feedback_id=feedback_id,
        target=FeedbackTarget.CONTROLLER.value,
        source=FeedbackSource.EVALUATOR.value,
        message="threshold crossed",
        severity="warning",
        created_at=datetime(2026, 4, 14, tzinfo=UTC),
        category="brier:model-a",
    )


def _quote_eval_record() -> QuoteEvalRecord:
    return QuoteEvalRecord(
        fill_id="trade-api",
        decision_id="decision-api",
        market_id="api-market",
        token_id="yes-token",
        strategy_id="default",
        strategy_version_id="default-v1",
        prob_estimate=0.7,
        quote_price=0.64,
        quote_source="postgres_snapshot",
        quote_lag_seconds=3600,
        quote_score=0.0036,
        mtm_pnl=1.2,
        book_ts=datetime(2026, 4, 14, 1, tzinfo=UTC),
        recorded_at=datetime(2026, 4, 14, 1, tzinfo=UTC),
        citations=["trade-api"],
        category="model-a",
        model_id="model-a",
    )


class _PositionStore:
    async def read_positions(self) -> list[Position]:
        return [
            Position(
                market_id="api-market",
                token_id="yes-token",
                venue="polymarket",
                side=Side.BUY.value,
                shares_held=10.0,
                avg_entry_price=0.50,
                unrealized_pnl=1.2,
                locked_usdc=5.0,
            )
        ]


class _QuoteEvalStore:
    def __init__(self, records: list[QuoteEvalRecord] | None = None) -> None:
        self._records: list[QuoteEvalRecord] = (
            [_quote_eval_record()] if records is None else list(records)
        )

    async def all(self) -> list[QuoteEvalRecord]:
        return list(self._records)


@pytest.mark.asyncio
async def test_api_quality_payload_separates_brier_mtm_and_quote_calibration() -> None:
    runner = _runner_with_state()
    runner.fill_store = cast(FillStore, _PositionStore())
    runner.quote_eval_store = cast(QuoteEvalStore, _QuoteEvalStore())
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        status = (await client.get("/status")).json()
        metrics = (await client.get("/metrics")).json()

    assert status["quality"]["final_brier"] == {
        "record_count": 1,
        "brier_overall": 0.09,
    }
    assert status["quality"]["mark_to_market"] == {
        "open_positions": 1,
        "locked_usdc": 5.0,
        "unrealized_pnl": 1.2,
    }
    assert status["quality"]["quote_calibration"] == {
        "record_count": 1,
        "quote_score_overall": 0.0036,
        "mtm_pnl": 1.2,
    }
    assert metrics["mark_to_market"]["unrealized_pnl"] == 1.2
    assert metrics["quote_calibration"]["quote_score_overall"] == 0.0036


@pytest.mark.asyncio
async def test_api_routes_expose_mock_runner_state() -> None:
    set_metric(SENSOR_DISCOVERY_PRICE_FIELDS_POPULATED_RATIO_METRIC, 0.625)
    set_metric(SENSOR_DISCOVERY_SNAPSHOTS_WRITTEN_TOTAL_METRIC, 3.0)
    set_metric(MARKETS_SNAPSHOT_LAG_SECONDS_MAX_METRIC, 12.5)
    app = create_app(_runner_with_state())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        status = (await client.get("/status")).json()
        signals = (await client.get("/signals?limit=1")).json()
        decisions = (await client.get("/decisions?limit=1")).json()
        metrics = (await client.get("/metrics")).json()
        feedback = (await client.get("/feedback?resolved=false")).json()

    assert status["mode"] == "backtest"
    assert status["runner_started_at"] == "2026-04-14T00:00:00+00:00"
    assert status["sensors"][0].keys() == {
        "name",
        "status",
        "last_signal_at",
        "last_signal_age_seconds",
        "stale_after_seconds",
        "task_done",
    }
    assert status["controller"] == {
        "current_runtimes_total": 0,
        "decisions_total": 1,
        "diagnostics_total": 0,
        "diagnostic_counts": {},
    }
    assert status["supervision"] == {"unresolved_feedback_total": 1}
    assert status["actuator"] == {
        "fills_total": 1,
        "mode": "backtest",
        "halt_recovery_cycles_7d": 0,
        "halted": False,
        "halt_reason": None,
        "halt_trigger_kind": None,
        "halt_triggered_at": None,
    }
    assert status["evaluator"] == {
        "eval_records_total": 1,
        "brier_overall": 0.09,
        "baseline_brier_overall": 0.36,
        "brier_improvement_overall": 0.27,
        "brier_14d": None,
        "baseline_brier_14d": None,
        "brier_improvement_14d": None,
        "resolution_sweeps_total": 0,
        "resolution_fills_resolved_total": 0,
        "resolution_sweep_failures_total": 0,
    }
    assert signals[0]["market_id"] == "api-market"
    assert decisions[0]["forecaster"] == "model-a"
    assert decisions[0]["prob_estimate"] == 0.7
    assert decisions[0]["expected_edge"] == 0.3
    assert decisions[0]["action"] == "BUY"
    assert decisions[0]["limit_price"] == 0.4
    assert decisions[0]["kelly_size"] == 12.5
    assert metrics["brier_overall"] == 0.09
    assert metrics["baseline_brier_overall"] == 0.36
    assert metrics["brier_improvement_overall"] == 0.27
    assert metrics[SENSOR_DISCOVERY_PRICE_FIELDS_POPULATED_RATIO_METRIC] == 0.625
    assert metrics[SENSOR_DISCOVERY_SNAPSHOTS_WRITTEN_TOTAL_METRIC] == 3.0
    assert metrics[MARKETS_SNAPSHOT_LAG_SECONDS_MAX_METRIC] == 12.5
    assert metrics["ops_view"]["brier_overall"] == 0.09
    assert metrics["ops_view"]["pnl"] == 1.0
    assert metrics["per_strategy"] == [
        {
            "strategy_id": "default",
            "strategy_version_id": "default-v1",
            "record_count": 1,
            "insufficient_samples": False,
            "brier_overall": 0.09,
            "baseline_brier_overall": 0.36,
            "brier_improvement_overall": 0.27,
            "pnl": 1.0,
            "fill_rate": 1.0,
            "slippage_bps": 10.0,
            "drawdown": 0.0,
        }
    ]
    assert [item["feedback_id"] for item in feedback] == ["fb-pending"]


@pytest.mark.asyncio
async def test_status_exposes_live_exit_health_rolling_windows() -> None:
    now = datetime.now(tz=UTC)
    recent_eval = replace(
        _eval_record(),
        decision_id="decision-recent",
        recorded_at=now - timedelta(days=1),
        brier_score=0.09,
        baseline_brier_score=0.36,
    )
    old_eval = replace(
        _eval_record(),
        decision_id="decision-old",
        recorded_at=now - timedelta(days=30),
        brier_score=0.81,
        baseline_brier_score=0.04,
    )
    runner = Runner(
        config=_settings(),
        eval_store=cast(EvalStore, InMemoryEvalStore([old_eval, recent_eval])),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    risk = runner.actuator_executor.risk
    risk.record_api_error(401, at=now - timedelta(days=8), trace_id="trace-old")
    risk.check_auto_halt(runner.portfolio, now=now - timedelta(days=8))
    risk.clear_halt(at=now - timedelta(days=8) + timedelta(minutes=1))
    risk.record_api_error(401, at=now - timedelta(days=1), trace_id="trace-recent")
    risk.check_auto_halt(runner.portfolio, now=now - timedelta(days=1))
    risk.clear_halt(at=now - timedelta(days=1) + timedelta(minutes=1))
    risk.record_trade_result(
        RiskTradeResult(
            pnl=1.0,
            slippage_bps=5.0,
            filled_at=now,
        )
    )
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        status = (await client.get("/status")).json()

    assert status["actuator"]["halt_recovery_cycles_7d"] == 1
    assert status["evaluator"]["brier_improvement_14d"] == 0.27
    assert status["evaluator"]["baseline_brier_14d"] == 0.36


@pytest.mark.asyncio
async def test_metrics_route_filters_eval_records_by_recorded_at_window() -> None:
    old_eval = replace(
        _eval_record(),
        decision_id="decision-before-window",
        recorded_at=datetime(2026, 4, 29, 23, 59, 59, tzinfo=UTC),
        brier_score=0.81,
        baseline_brier_score=0.04,
        pnl=-25.0,
        slippage_bps=200.0,
        filled=False,
    )
    window_eval = replace(
        _eval_record(),
        decision_id="decision-inside-window",
        recorded_at=datetime(2026, 5, 30, tzinfo=UTC),
        brier_score=0.09,
        baseline_brier_score=0.36,
        pnl=2.0,
        slippage_bps=10.0,
        filled=True,
    )
    runner = Runner(
        config=_settings(),
        eval_store=cast(EvalStore, InMemoryEvalStore([old_eval, window_eval])),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/metrics",
            params={
                "since": "2026-04-30T00:00:00+00:00",
                "until": "2026-05-31T00:00:00+00:00",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["record_count"] == 1
    assert payload["brier_overall"] == 0.09
    assert payload["baseline_brier_overall"] == 0.36
    assert payload["brier_improvement_overall"] == 0.27
    assert payload["pnl"] == 2.0
    assert payload["fill_rate"] == 1.0
    assert payload["win_rate"] == 1.0
    assert payload["slippage_bps"] == 10.0
    assert payload["window_started_at"] == "2026-04-30T00:00:00+00:00"
    assert payload["window_ended_at"] == "2026-05-31T00:00:00+00:00"


@pytest.mark.asyncio
async def test_metrics_route_filters_quote_records_by_recorded_at_window() -> None:
    old_quote = replace(
        _quote_eval_record(),
        fill_id="quote-before-window",
        recorded_at=datetime(2026, 4, 29, 23, 59, 59, tzinfo=UTC),
        quote_score=0.9,
        mtm_pnl=-50.0,
    )
    window_quote = replace(
        _quote_eval_record(),
        fill_id="quote-inside-window",
        recorded_at=datetime(2026, 5, 30, tzinfo=UTC),
        quote_score=0.01,
        mtm_pnl=2.5,
    )
    runner = _runner_with_state()
    runner.quote_eval_store = cast(
        QuoteEvalStore, _QuoteEvalStore([old_quote, window_quote])
    )
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/metrics",
            params={
                "since": "2026-04-30T00:00:00+00:00",
                "until": "2026-05-31T00:00:00+00:00",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    quote_calibration = payload["quote_calibration"]
    # Only the in-window record contributes. Out-of-window record's score
    # (0.9) and PnL (-50.0) must NOT contaminate these aggregates.
    assert quote_calibration["record_count"] == 1
    assert quote_calibration["quote_score_overall"] == 0.01
    assert quote_calibration["mtm_pnl"] == 2.5


@pytest.mark.asyncio
async def test_metrics_route_exposes_quote_mtm_pnl_series() -> None:
    later_quote = replace(
        _quote_eval_record(),
        fill_id="quote-later",
        recorded_at=datetime(2026, 5, 30, 2, tzinfo=UTC),
        mtm_pnl=-0.25,
    )
    earlier_quote = replace(
        _quote_eval_record(),
        fill_id="quote-earlier",
        recorded_at=datetime(2026, 5, 30, 1, tzinfo=UTC),
        mtm_pnl=1.5,
    )
    runner = _runner_with_state()
    runner.quote_eval_store = cast(
        QuoteEvalStore,
        _QuoteEvalStore([later_quote, earlier_quote]),
    )
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")

    payload = response.json()
    assert response.status_code == 200
    assert payload["quote_calibration"]["pnl_series"] == [
        {
            "recorded_at": "2026-05-30T01:00:00+00:00",
            "pnl": 1.5,
            "source": "quote_mtm",
        },
        {
            "recorded_at": "2026-05-30T02:00:00+00:00",
            "pnl": 1.25,
            "source": "quote_mtm",
        },
    ]


@pytest.mark.asyncio
async def test_metrics_route_deduplicates_repeated_quote_mtm_marks_by_fill() -> None:
    first_mark = replace(
        _quote_eval_record(),
        fill_id="same-fill",
        quote_lag_seconds=60,
        recorded_at=datetime(2026, 5, 30, 1, tzinfo=UTC),
        mtm_pnl=1.0,
    )
    latest_mark = replace(
        _quote_eval_record(),
        fill_id="same-fill",
        quote_lag_seconds=120,
        recorded_at=datetime(2026, 5, 30, 2, tzinfo=UTC),
        mtm_pnl=2.0,
    )
    other_fill_mark = replace(
        _quote_eval_record(),
        fill_id="other-fill",
        quote_lag_seconds=60,
        recorded_at=datetime(2026, 5, 30, 3, tzinfo=UTC),
        mtm_pnl=-0.5,
    )
    runner = _runner_with_state()
    runner.quote_eval_store = cast(
        QuoteEvalStore,
        _QuoteEvalStore([first_mark, latest_mark, other_fill_mark]),
    )
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")

    payload = response.json()
    assert response.status_code == 200
    assert payload["quote_calibration"]["mtm_pnl"] == 1.5
    assert payload["quote_calibration"]["pnl_series"] == [
        {
            "recorded_at": "2026-05-30T01:00:00+00:00",
            "pnl": 1.0,
            "source": "quote_mtm",
        },
        {
            "recorded_at": "2026-05-30T02:00:00+00:00",
            "pnl": 2.0,
            "source": "quote_mtm",
        },
        {
            "recorded_at": "2026-05-30T03:00:00+00:00",
            "pnl": 1.5,
            "source": "quote_mtm",
        },
    ]


@pytest.mark.asyncio
async def test_metrics_route_exposes_quote_mtm_max_drawdown_pct() -> None:
    quote_records = [
        replace(
            _quote_eval_record(),
            fill_id="quote-peak",
            recorded_at=datetime(2026, 5, 30, 1, tzinfo=UTC),
            mtm_pnl=5.0,
        ),
        replace(
            _quote_eval_record(),
            fill_id="quote-trough",
            recorded_at=datetime(2026, 5, 30, 2, tzinfo=UTC),
            mtm_pnl=-15.0,
        ),
        replace(
            _quote_eval_record(),
            fill_id="quote-recovery",
            recorded_at=datetime(2026, 5, 30, 3, tzinfo=UTC),
            mtm_pnl=3.0,
        ),
    ]
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.BACKTEST,
            auto_migrate_default_v2=False,
            risk=RiskSettings(max_total_exposure=50.0),
        ),
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    runner.quote_eval_store = cast(QuoteEvalStore, _QuoteEvalStore(quote_records))
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")

    payload = response.json()
    assert response.status_code == 200
    assert payload["quote_calibration"]["max_drawdown_pct"] == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_metrics_route_exposes_max_drawdown_pct_from_windowed_pnl() -> None:
    records = [
        replace(
            _eval_record(),
            decision_id="decision-peak",
            recorded_at=datetime(2026, 5, 30, 0, 0, tzinfo=UTC),
            pnl=5.0,
        ),
        replace(
            _eval_record(),
            decision_id="decision-trough",
            recorded_at=datetime(2026, 5, 30, 1, 0, tzinfo=UTC),
            pnl=-15.0,
        ),
        replace(
            _eval_record(),
            decision_id="decision-recovery",
            recorded_at=datetime(2026, 5, 30, 2, 0, tzinfo=UTC),
            pnl=10.0,
        ),
    ]
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.BACKTEST,
            auto_migrate_default_v2=False,
            risk=RiskSettings(max_total_exposure=50.0),
        ),
        eval_store=cast(EvalStore, InMemoryEvalStore(records)),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/metrics",
            params={
                "since": "2026-05-30T00:00:00+00:00",
                "until": "2026-05-31T00:00:00+00:00",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["max_drawdown_pct"] == pytest.approx(30.0)
    assert payload["ops_view"]["max_drawdown_pct"] == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_metrics_route_exposes_sharpe_ratio_from_windowed_daily_pnl() -> None:
    records = [
        replace(
            _eval_record(),
            decision_id="decision-day-1",
            recorded_at=datetime(2026, 5, 28, 12, 0, tzinfo=UTC),
            pnl=1.0,
        ),
        replace(
            _eval_record(),
            decision_id="decision-day-2",
            recorded_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
            pnl=3.0,
        ),
        replace(
            _eval_record(),
            decision_id="decision-day-3",
            recorded_at=datetime(2026, 5, 30, 12, 0, tzinfo=UTC),
            pnl=5.0,
        ),
    ]
    runner = Runner(
        config=_settings(),
        eval_store=cast(EvalStore, InMemoryEvalStore(records)),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/metrics",
            params={
                "since": "2026-05-28T00:00:00+00:00",
                "until": "2026-05-31T00:00:00+00:00",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["sharpe_ratio"] == pytest.approx(1.5)
    assert payload["ops_view"]["sharpe_ratio"] == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_status_marks_running_sensor_stale_when_last_signal_is_old() -> None:
    class MarketDataSensorStub:
        def __aiter__(self) -> AsyncIterator[MarketSignal]:
            return _empty_signal_stream()

    async def _pending() -> None:
        await asyncio.Event().wait()

    runner = _runner_with_state()
    runner.config.dashboard.stale_snapshot_threshold_s = 120.0
    runner._active_sensors = (MarketDataSensorStub(),)  # noqa: SLF001
    task = asyncio.create_task(_pending())
    runner.sensor_stream._tasks = (task,)  # noqa: SLF001
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            status = (await client.get("/status")).json()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        runner.sensor_stream._tasks = ()  # noqa: SLF001

    assert status["running"] is True
    assert status["sensors"] == [
        {
            "name": "MarketDataSensorStub",
            "status": "stale",
            "last_signal_at": "2026-04-14T00:00:00+00:00",
            "last_signal_age_seconds": status["sensors"][0]["last_signal_age_seconds"],
            "stale_after_seconds": 120.0,
            "task_done": False,
        }
    ]
    assert status["sensors"][0]["last_signal_age_seconds"] > 120.0


@pytest.mark.asyncio
async def test_status_marks_market_data_stale_when_no_signal_arrives_after_start() -> None:
    class MarketDataSensorStub:
        def __aiter__(self) -> AsyncIterator[MarketSignal]:
            return _empty_signal_stream()

    async def _pending() -> None:
        await asyncio.Event().wait()

    runner = Runner(config=_settings(), eval_store=cast(EvalStore, InMemoryEvalStore([])))
    runner.state.runner_started_at = datetime(2026, 4, 14, tzinfo=UTC)
    runner.config.dashboard.stale_snapshot_threshold_s = 120.0
    runner._active_sensors = (MarketDataSensorStub(),)  # noqa: SLF001
    task = asyncio.create_task(_pending())
    runner.sensor_stream._tasks = (task,)  # noqa: SLF001
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            status = (await client.get("/status")).json()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        runner.sensor_stream._tasks = ()  # noqa: SLF001

    assert status["sensors"][0]["status"] == "stale"
    assert status["sensors"][0]["last_signal_at"] is None
    assert status["sensors"][0]["last_signal_age_seconds"] > 120.0


@pytest.mark.asyncio
async def test_api_feedback_resolve_and_config_errors() -> None:
    runner = _runner_with_state()
    runner.config.api_token = "live-api-token"
    runner.bind_pg_pool(cast(Any, _AvailableAuditPool()))
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resolved = await client.post("/feedback/fb-pending/resolve", headers=AUTH_HEADERS)
        missing = await client.post("/feedback/not-found/resolve", headers=AUTH_HEADERS)
        blocked_live = await client.post(
            "/config",
            json={"mode": "live"},
            headers=AUTH_HEADERS,
        )

    assert resolved.status_code == 200
    assert resolved.json()["resolved"] is True
    assert missing.status_code == 404
    assert blocked_live.status_code == 400
    assert blocked_live.json() == {
        "detail": "Live trading is disabled. Set live_trading_enabled=true in config."
    }


@pytest.mark.asyncio
async def test_api_config_rejects_live_mode_when_credentials_are_missing() -> None:
    runner = _runner_with_state()
    runner.config.live_trading_enabled = True
    runner.config.api_token = "live-api-token"
    runner.bind_pg_pool(cast(Any, _AvailableAuditPool()))
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/config",
            json={"mode": "live"},
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Missing Polymarket credential fields:")


@pytest.mark.asyncio
async def test_api_config_validates_candidate_live_mode_readiness() -> None:
    runner = Runner(
        config=_live_transition_settings(include_operator_readiness=False),
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    runner.bind_pg_pool(cast(Any, _AvailableAuditPool()))
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/config",
            json={"mode": "live"},
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 400
    assert "operator readiness" in response.json()["detail"]
    assert runner.config.mode == RunMode.PAPER
    assert runner.state.mode == RunMode.PAPER


@pytest.mark.asyncio
async def test_api_config_rejects_live_mode_without_credentialed_preflight_artifact() -> None:
    runner = Runner(
        config=_live_transition_settings(include_operator_readiness=True),
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    runner.bind_pg_pool(cast(Any, _AvailableAuditPool()))
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/config",
            json={"mode": "live"},
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 400
    assert "preflight artifact" in response.json()["detail"]
    assert runner.config.mode == RunMode.PAPER
    assert runner.state.mode == RunMode.PAPER


@pytest.mark.asyncio
async def test_api_config_redacts_live_transition_refusal_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_values = (
        "private-key-secret",
        "api-key-secret",
        "api-secret-secret",
        "passphrase-secret",
        "0xfundersecret",
    )
    secret_dsn = "postgresql://admin:supersecret@db.internal.example.com:5432/pms_live"
    base_settings = _live_transition_settings(include_operator_readiness=True)
    runner = Runner(
        config=base_settings.model_copy(
            update={
                "polymarket": PolymarketSettings(
                    private_key=credential_values[0],
                    api_key=credential_values[1],
                    api_secret=credential_values[2],
                    api_passphrase=credential_values[3],
                    signature_type=1,
                    funder_address=credential_values[4],
                    operator_approval_mode="every_order",
                    first_live_order_approval_path=(
                        base_settings.polymarket.first_live_order_approval_path
                    ),
                )
            }
        ),
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    runner.bind_pg_pool(cast(Any, _AvailableAuditPool()))

    def _refuse_live_transition(settings: PMSSettings) -> None:
        assert settings.mode == RunMode.LIVE
        raise LiveTradingDisabledError(
            "LIVE transition refused "
            f"{credential_values[0]} {credential_values[1]} "
            f"{credential_values[2]} {credential_values[3]} {credential_values[4]} "
            f"{secret_dsn} password=keyword-secret"
        )

    monkeypatch.setattr("pms.api.app.validate_live_mode_ready", _refuse_live_transition)
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/config",
            json={"mode": "live"},
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail.startswith("LIVE transition refused")
    assert "<redacted-polymarket-credential>" in detail
    assert "<redacted-database-url>" in detail
    assert "password=<redacted>" in detail
    for credential in credential_values:
        assert credential not in detail
    assert "supersecret" not in detail
    assert "keyword-secret" not in detail
    assert "admin" not in detail


@pytest.mark.asyncio
async def test_api_config_rejects_live_mode_switch_while_runner_is_active() -> None:
    runner = Runner(
        config=_live_transition_settings(include_operator_readiness=False),
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    stop_pending = asyncio.Event()

    async def _pending() -> None:
        await stop_pending.wait()

    task = asyncio.create_task(_pending())
    runner.sensor_stream._tasks = (task,)  # noqa: SLF001
    runner.bind_pg_pool(cast(Any, _AvailableAuditPool()))
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/config",
                json={"mode": "live"},
                headers=AUTH_HEADERS,
            )

        assert response.status_code == 409
        assert response.json()["detail"] == "Stop the runner before changing mode."
        assert runner.config.mode == RunMode.PAPER
        assert runner.state.mode == RunMode.PAPER
    finally:
        stop_pending.set()
        await task
        runner.sensor_stream._tasks = ()  # noqa: SLF001


@pytest.mark.asyncio
async def test_live_config_validation_runs_after_available_audit_sink() -> None:
    runner = Runner(
        config=_live_transition_settings(include_operator_readiness=False),
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    runner.bind_pg_pool(cast(Any, _AvailableAuditPool()))
    app = create_app(runner, auto_start=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post("/config", json={"mode": "live"}, headers=AUTH_HEADERS)
    assert response.status_code == 400
    assert "operator readiness" in response.json()["detail"]
    assert runner.config.mode == RunMode.PAPER


@pytest.mark.asyncio
async def test_live_start_validation_runs_after_available_audit_sink() -> None:
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.LIVE,
            live_trading_enabled=True,
            api_token="live-api-token",
            auto_migrate_default_v2=False,
        ),
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    runner.bind_pg_pool(cast(Any, _AvailableAuditPool()))
    app = create_app(runner, auto_start=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post("/run/start", headers=AUTH_HEADERS)
    assert response.status_code == 400
    assert response.json()["detail"].startswith("Missing Polymarket credential fields:")


@pytest.mark.asyncio
async def test_api_run_start_rejects_live_mode_when_credentials_are_missing() -> None:
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.LIVE,
            live_trading_enabled=True,
            api_token="live-api-token",
            auto_migrate_default_v2=False,
        ),
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    runner.bind_pg_pool(cast(Any, _AvailableAuditPool()))
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/run/start", headers=AUTH_HEADERS)

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Missing Polymarket credential fields:")


@pytest.mark.asyncio
@pytest.mark.parametrize("exception_type", [RuntimeError, ConnectionError])
async def test_api_run_start_redacts_live_runtime_startup_refusal(
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[Exception],
) -> None:
    secret_dsn = "postgresql://admin:supersecret@db.internal.example.com:5432/pms_live"
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.LIVE,
            live_trading_enabled=True,
            api_token="live-api-token",
            auto_migrate_default_v2=False,
        ),
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    runner.bind_pg_pool(cast(Any, _AvailableAuditPool()))

    async def _failing_start() -> None:
        raise exception_type(
            "LIVE start refused: database "
            f"{secret_dsn} is unavailable password=keyword-secret"
        )

    monkeypatch.setattr(runner, "start", _failing_start)
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/run/start", headers=AUTH_HEADERS)

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail.startswith("LIVE start refused")
    assert "<redacted-database-url>" in detail
    assert "password=<redacted>" in detail
    assert "supersecret" not in detail
    assert "keyword-secret" not in detail
    assert "admin" not in detail


@pytest.mark.asyncio
async def test_api_run_start_redacts_polymarket_credentials_from_live_startup_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_values = (
        "private-key-secret",
        "api-key-secret",
        "api-secret-secret",
        "passphrase-secret",
        "0xfundersecret",
    )
    secret_dsn = "postgresql://admin:supersecret@db.internal.example.com:5432/pms_live"
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.LIVE,
            live_trading_enabled=True,
            api_token="live-api-token",
            auto_migrate_default_v2=False,
            polymarket=PolymarketSettings(
                private_key=credential_values[0],
                api_key=credential_values[1],
                api_secret=credential_values[2],
                api_passphrase=credential_values[3],
                signature_type=1,
                funder_address=credential_values[4],
            ),
        ),
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    runner.bind_pg_pool(cast(Any, _AvailableAuditPool()))

    async def _failing_start() -> None:
        raise ConnectionError(
            "venue auth failed "
            f"{credential_values[0]} {credential_values[1]} "
            f"{credential_values[2]} {credential_values[3]} {credential_values[4]} "
            f"{secret_dsn} password=keyword-secret"
        )

    monkeypatch.setattr(runner, "start", _failing_start)
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/run/start", headers=AUTH_HEADERS)

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail.startswith("venue auth failed")
    assert "<redacted-polymarket-credential>" in detail
    assert "<redacted-database-url>" in detail
    assert "password=<redacted>" in detail
    for credential in credential_values:
        assert credential not in detail
    assert "supersecret" not in detail
    assert "keyword-secret" not in detail
    assert "admin" not in detail


@pytest.mark.asyncio
async def test_api_run_start_redacts_live_trading_disabled_error_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_values = (
        "private-key-secret",
        "api-key-secret",
        "api-secret-secret",
        "passphrase-secret",
        "0xfundersecret",
    )
    secret_dsn = "postgresql://admin:supersecret@db.internal.example.com:5432/pms_live"
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.LIVE,
            live_trading_enabled=True,
            api_token="live-api-token",
            auto_migrate_default_v2=False,
            polymarket=PolymarketSettings(
                private_key=credential_values[0],
                api_key=credential_values[1],
                api_secret=credential_values[2],
                api_passphrase=credential_values[3],
                signature_type=1,
                funder_address=credential_values[4],
            ),
        ),
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    runner.bind_pg_pool(cast(Any, _AvailableAuditPool()))

    async def _failing_start() -> None:
        raise LiveTradingDisabledError(
            "LIVE guard refused "
            f"{credential_values[0]} {credential_values[1]} "
            f"{credential_values[2]} {credential_values[3]} {credential_values[4]} "
            f"{secret_dsn} password=keyword-secret"
        )

    monkeypatch.setattr(runner, "start", _failing_start)
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/run/start", headers=AUTH_HEADERS)

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail.startswith("LIVE guard refused")
    assert "<redacted-polymarket-credential>" in detail
    assert "<redacted-database-url>" in detail
    assert "password=<redacted>" in detail
    for credential in credential_values:
        assert credential not in detail
    assert "supersecret" not in detail
    assert "keyword-secret" not in detail
    assert "admin" not in detail


@pytest.mark.asyncio
async def test_api_run_start_stop_cycle(tmp_path: Path) -> None:
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.BACKTEST,
            auto_migrate_default_v2=False,
            risk=RiskSettings(
                max_position_per_market=1000.0,
                max_total_exposure=10_000.0,
            ),
            controller=ControllerSettings(strict_factor_gates=False),
            live_request_audit_fallback_path=str(tmp_path / "api-request-audit-fallback.jsonl"),
        ),
        historical_data_path=FIXTURE_PATH,
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            initial_status = (await client.get("/status")).json()
            start_resp = await client.post("/run/start", headers=AUTH_HEADERS)
            conflict_resp = await client.post("/run/start", headers=AUTH_HEADERS)
            await runner.wait_until_idle()
            running_status = (await client.get("/status")).json()
            stop_resp = await client.post("/run/stop", headers=AUTH_HEADERS)
            stopped_status = (await client.get("/status")).json()
    finally:
        await runner.stop()

    assert initial_status["running"] is False
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] == "started"
    assert start_resp.json()["mode"] == "backtest"
    assert conflict_resp.status_code == 409
    assert running_status["controller"]["decisions_total"] > 0
    assert stop_resp.status_code == 200
    assert stop_resp.json() == {"status": "stopped"}
    assert stopped_status["running"] is False


@pytest.mark.asyncio
async def test_api_run_stop_keeps_reporting_pool_bound(tmp_path: Path) -> None:
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.PAPER,
            auto_migrate_default_v2=False,
            api_token="live-api-token",
            live_request_audit_fallback_path=str(tmp_path / "api-request-audit-fallback.jsonl"),
        ),
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
        fill_store=cast(FillStore, ReportingFillStoreDouble()),
    )
    fake_pool = object()
    runner.bind_pg_pool(cast(Any, fake_pool))
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        stop_resp = await client.post("/run/stop", headers=AUTH_HEADERS)
        trades_resp = await client.get("/trades", headers=AUTH_HEADERS)
        positions_resp = await client.get("/positions", headers=AUTH_HEADERS)

    assert stop_resp.status_code == 200
    assert runner.pg_pool is fake_pool
    assert trades_resp.status_code == 200
    assert trades_resp.json() == {"trades": [], "limit": 50, "offset": 0}
    assert positions_resp.status_code == 200
    assert positions_resp.json() == {"positions": []}


@pytest.mark.asyncio
async def test_api_auto_start_lifespan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_schema_check(monkeypatch)
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.BACKTEST,
            auto_migrate_default_v2=False,
            risk=RiskSettings(
                max_position_per_market=1000.0,
                max_total_exposure=10_000.0,
            ),
        ),
        historical_data_path=FIXTURE_PATH,
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    app = create_app(runner, auto_start=True)

    async with app.router.lifespan_context(app):
        assert any(not task.done() for task in runner.tasks)
        await runner.wait_until_idle()

    assert all(task.done() for task in runner.tasks)


@pytest.mark.asyncio
async def test_api_auto_start_disabled_keeps_runner_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_schema_check(monkeypatch)
    runner = Runner(
        config=PMSSettings(mode=RunMode.BACKTEST, auto_migrate_default_v2=False),
        historical_data_path=FIXTURE_PATH,
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    app = create_app(runner, auto_start=False)

    async with app.router.lifespan_context(app):
        assert runner.tasks == ()


@pytest.mark.asyncio
async def test_api_backtest_lifespan_skips_schema_check_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_check_calls: list[object] = []

    async def _record_schema_check(pool: object) -> None:
        schema_check_calls.append(pool)

    monkeypatch.setattr("pms.api.app.ensure_schema_current", _record_schema_check)
    _stub_orphan_scan(monkeypatch)
    runner = Runner(
        config=PMSSettings(mode=RunMode.BACKTEST, auto_migrate_default_v2=False),
        historical_data_path=FIXTURE_PATH,
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    app = create_app(runner, auto_start=False)

    async with app.router.lifespan_context(app):
        assert runner.pg_pool is not None

    assert schema_check_calls == []


@pytest.mark.asyncio
async def test_api_backtest_lifespan_can_opt_into_schema_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_check_calls: list[object] = []

    async def _record_schema_check(pool: object) -> None:
        schema_check_calls.append(pool)

    monkeypatch.setattr("pms.api.app.ensure_schema_current", _record_schema_check)
    _stub_orphan_scan(monkeypatch)
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.BACKTEST,
            auto_migrate_default_v2=False,
            enforce_schema_check=True,
        ),
        historical_data_path=FIXTURE_PATH,
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    app = create_app(runner, auto_start=False)

    async with app.router.lifespan_context(app):
        assert runner.pg_pool is not None

    assert len(schema_check_calls) == 1


@pytest.mark.asyncio
async def test_api_lifespan_closes_owned_pool_when_schema_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TrackingPool:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    pool = _TrackingPool()

    async def _fake_create_pool(*, dsn: str, min_size: int, max_size: int) -> _TrackingPool:
        del dsn, min_size, max_size
        return pool

    async def _raise_schema_check(_pool: object) -> None:
        raise SchemaVersionMismatchError("schema behind")

    monkeypatch.setattr("pms.runner.asyncpg.create_pool", _fake_create_pool)
    monkeypatch.setattr("pms.api.app.ensure_schema_current", _raise_schema_check)
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.BACKTEST,
            auto_migrate_default_v2=False,
            enforce_schema_check=True,
        ),
        historical_data_path=FIXTURE_PATH,
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    app = create_app(runner, auto_start=False)

    with pytest.raises(SchemaVersionMismatchError, match="schema behind"):
        async with app.router.lifespan_context(app):
            pass

    assert pool.close_calls == 1
    assert runner.pg_pool is None


@pytest.mark.asyncio
async def test_api_lifespan_stops_runner_started_via_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for codex-bot C1: /run/start-triggered runner must also be stopped on shutdown."""
    _stub_schema_check(monkeypatch)
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.BACKTEST,
            auto_migrate_default_v2=False,
            risk=RiskSettings(
                max_position_per_market=1000.0,
                max_total_exposure=10_000.0,
            ),
        ),
        historical_data_path=FIXTURE_PATH,
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            started = await client.post("/run/start", headers=AUTH_HEADERS)
        assert started.status_code == 200
        assert any(not task.done() for task in runner.tasks)

    assert all(task.done() for task in runner.tasks)


@pytest.mark.asyncio
async def test_api_status_redacts_autostart_error_to_class_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for CodeRabbit Major: /status is unauthenticated and
    must NOT echo the raw exception text from a failed `runner.start()`.
    Connection errors / OSError / schema failures can otherwise leak
    DSNs, hostnames, file paths, and user info to anyone who can hit
    the endpoint.

    The fix exposes only `type(exc).__name__`; server logs keep redacted
    diagnostic detail so an operator can still diagnose without leaking secrets.
    """
    _stub_schema_check(monkeypatch)
    _stub_orphan_scan(monkeypatch)

    secret_dsn = (
        "postgresql://admin:supersecret@db.internal.example.com:5432/prod_db"
    )

    class _LeakyConnectionError(ConnectionError):
        pass

    runner = Runner(
        config=PMSSettings(mode=RunMode.BACKTEST, auto_migrate_default_v2=False),
        historical_data_path=FIXTURE_PATH,
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )

    async def _failing_start() -> None:
        # Sensitive DSN intentionally embedded in the message to simulate
        # the kind of leak this redaction prevents.
        raise _LeakyConnectionError(
            f"could not connect to {secret_dsn} (host=/var/secret/path)"
        )

    monkeypatch.setattr(runner, "start", _failing_start)
    app = create_app(runner, auto_start=True)
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["autostart_attempted"] is True
    # Only the class name leaks; full message stays server-side.
    assert payload["autostart_error"] == "_LeakyConnectionError"
    # Defence in depth: no token from the secret DSN appears anywhere
    # in the JSON body, even if a future change re-introduced a leak via
    # a different field.
    body_text = response.text
    for forbidden in ("supersecret", "db.internal.example.com", "prod_db", "/var/secret"):
        assert forbidden not in body_text, (
            f"sensitive token {forbidden!r} must not appear in /status payload"
        )


@pytest.mark.asyncio
async def test_api_autostart_failure_log_redacts_live_credentials(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _stub_schema_check(monkeypatch)
    _stub_orphan_scan(monkeypatch)
    credential_values = (
        "private-key-secret",
        "api-key-secret",
        "api-secret-secret",
        "passphrase-secret",
        "0x2222222222222222222222222222222222222222",
    )
    secret_dsn = "postgresql://admin:supersecret@db.internal.example.com:5432/pms_live"

    class _BoundPool:
        async def close(self) -> None:
            return None

    runner = Runner(
        config=PMSSettings(
            mode=RunMode.LIVE,
            live_trading_enabled=True,
            auto_migrate_default_v2=False,
            polymarket=PolymarketSettings(
                private_key=credential_values[0],
                api_key=credential_values[1],
                api_secret=credential_values[2],
                api_passphrase=credential_values[3],
                signature_type=1,
                funder_address=credential_values[4],
            ),
        ),
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, InMemoryFeedbackStore()),
    )
    runner.bind_pg_pool(cast(Any, _BoundPool()))

    async def _failing_start() -> None:
        raise ConnectionError(
            "LIVE autostart failed "
            f"{credential_values[0]} {credential_values[1]} "
            f"{credential_values[2]} {credential_values[3]} {credential_values[4]} "
            f"{secret_dsn} password=keyword-secret"
        )

    monkeypatch.setattr(runner, "start", _failing_start)
    app = create_app(runner, auto_start=True)

    with caplog.at_level(logging.CRITICAL, logger="pms.api.app"):
        async with app.router.lifespan_context(app):
            pass

    log_text = caplog.text
    assert "PMS_AUTO_START failed" in log_text
    assert "ConnectionError" in log_text
    assert "<redacted-polymarket-credential>" in log_text
    assert "<redacted-database-url>" in log_text
    assert "password=<redacted>" in log_text
    for credential in credential_values:
        assert credential not in log_text
    assert "supersecret" not in log_text
    assert "keyword-secret" not in log_text
    assert "admin" not in log_text


@pytest.mark.asyncio
async def test_run_start_serializes_concurrent_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent /run/start requests must not interleave through
    Runner.start: exactly one wins (200) and the loser gets 409 instead
    of starting a second runtime whose cleanup destroys the winner's."""
    runner = _runner_with_state()
    start_calls = 0
    first_start_entered = asyncio.Event()
    release_first_start = asyncio.Event()
    keepalive: list[asyncio.Task[None]] = []

    async def slow_start() -> None:
        nonlocal start_calls
        start_calls += 1
        first_start_entered.set()
        await release_first_start.wait()
        task = asyncio.create_task(asyncio.sleep(30.0))
        keepalive.append(task)
        runner._actuator_task = task  # noqa: SLF001

    monkeypatch.setattr(runner, "start", slow_start)
    app = create_app(runner)
    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            first = asyncio.create_task(
                client.post("/run/start", headers=AUTH_HEADERS)
            )
            await asyncio.wait_for(first_start_entered.wait(), timeout=2.0)
            second = asyncio.create_task(
                client.post("/run/start", headers=AUTH_HEADERS)
            )
            await asyncio.sleep(0.05)
            release_first_start.set()
            first_response, second_response = await asyncio.gather(first, second)
    finally:
        for task in keepalive:
            task.cancel()

    statuses = sorted(
        (first_response.status_code, second_response.status_code)
    )
    assert statuses == [200, 409]
    assert start_calls == 1
