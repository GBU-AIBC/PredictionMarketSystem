from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import os
from collections import defaultdict
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from math import sqrt
from pathlib import Path
from typing import Any, Literal, TypeVar, cast
from uuid import UUID

import asyncpg
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.routing import Match

from pms.api.auth import authenticate, principal, require_any_scope, require_scope
from pms.api.health import health_payload, readiness_payload
from pms.api.routes.decisions import (
    AcceptDecisionRequest,
    AcceptDecisionResponse,
    DecisionEnqueueRejectedError,
    DecisionMarketChangedError,
    DecisionNotFoundError,
    DecisionRiskRejectedError,
    accept_decision as accept_decision_item,
    get_decision as get_decision_item,
    list_decisions as list_decisions_items,
)
from pms.api.routes.agent_evidence import router as agent_evidence_router
from pms.api.routes.agent_factors import router as agent_factors_router
from pms.api.routes.decay import get_strategy_decay_status as get_strategy_decay_status_item
from pms.api.routes.events import encode_sse_event
from pms.api.research_routes import (
    compute_backtest_live_comparison,
    enqueue_backtest_runs,
    fetch_backtest_run,
    list_backtest_runs,
    list_backtest_strategy_runs,
    scan_orphaned_backtest_runs,
)
from pms.api.routes.factors import list_factor_catalog, list_factor_series
from pms.api.routes.factors import FactorCatalogResponse, FactorSeriesResponse
from pms.api.routes.feedback import list_feedback as list_feedback_items
from pms.api.routes.feedback import resolve_feedback as resolve_feedback_item
from pms.api.routes.markets import (
    MarketNotFoundError,
    MarketPriceHistoryNotFoundError,
    MarketsFilterParams,
    SubscribedFilter,
    get_market as get_market_item,
    get_price_history as get_price_history_item,
    list_markets as list_markets_items,
    MarketRow,
    MarketsListResponse,
    PriceHistoryResponse,
)
from pms.api.routes.market_subscriptions import (
    UnknownSubscriptionTokenError,
    subscribe_market as subscribe_market_item,
    unsubscribe_market as unsubscribe_market_item,
    SubscribeMarketResponse,
    UnsubscribeMarketResponse,
)
from pms.api.routes.positions import PositionsResponse, list_positions as list_positions_items
from pms.api.routes.share import SHARE_NOT_FOUND_DETAIL, ShareResponse, get_shared_strategy
from pms.api.routes.signals import (
    SignalDepthNotFoundError,
    get_signal_depth,
)
from pms.api.routes.strategies import (
    StrategiesResponse,
    StrategyMetricsResponse,
    list_strategy_metrics as list_strategy_metrics_items,
    list_strategies as list_strategies_items,
)
from pms.api.routes.trades import TradesResponse, list_trades as list_trades_items
from pms.config import (
    PMSSettings,
    load_settings,
    normalize_webhook_url,
    validate_live_mode_ready,
)
from pms.live_preflight import redact_live_error, require_live_preflight_artifact
from pms.alerting.discord import DiscordWebhookClient
from pms.alerting.scheduler import EODScheduler
from pms.alerting.subscriber import run_alerting_subscription
from pms.core.enums import RunMode
from pms.core.models import (
    EvalRecord,
    LiveTradingDisabledError,
    MarketSignal,
    Position,
    QuoteEvalRecord,
    TradeDecision,
)
from pms.evaluation.metrics import MetricsCollector, MetricsSnapshot
from pms.evaluation.quote_metrics import QuoteMetricsCollector, QuoteMetricsSnapshot
from pms.metrics import metrics_snapshot
from pms.runner import READINESS_REQUIRED_WORKERS, Runner
from pms.supervision import ALIVE_WORKER_STATES
from pms.storage.schema_check import ensure_schema_current
from pms.storage.decision_store import DecisionStore
from pms.storage.market_data_store import PostgresMarketDataStore
from pms.storage.market_subscription_store import PostgresMarketSubscriptionStore
from pms.storage.runtime_heartbeat_store import RuntimeHeartbeatStore
from pms.storage.live_reconciliation import (
    SubmissionUnknownReconciliationStore,
    normalize_submission_unknown_decision_id,
    normalize_submission_unknown_reconciled_by,
    normalize_submission_unknown_venue_order_id,
)
from pms.storage.api_request_audit import append_request_audit


T = TypeVar("T")
LIVE_DISABLED_DETAIL = (
    "Live trading is disabled. Set live_trading_enabled=true in config."
)
RUNNER_ALREADY_RUNNING_DETAIL = "Runner is already running."
MODE_CHANGE_WHILE_RUNNING_DETAIL = "Stop the runner before changing mode."
FIRST_TRADE_TIME_SECONDS_METRIC = "pms.ui.first_trade_time_seconds"
logger = logging.getLogger(__name__)


class ConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: RunMode


class ResearchRunRequest(BaseModel):
    """Typed JSON wrapper; text/YAML clients remain accepted by the decoder."""

    model_config = ConfigDict(extra="forbid")
    sweep_yaml: str


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    live_window_start: str
    live_window_end: str
    denominator: Literal["backtest_set", "live_set", "union"]
    strategy_id: str | None = None
    strategy_version_id: str | None = None
    time_alignment_policy: dict[str, Any] | None = None
    symbol_normalization_policy: dict[str, Any] | None = None

    @field_validator("time_alignment_policy", "symbol_normalization_policy")
    @classmethod
    def _finite_policy_values(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        for key, item in value.items():
            if isinstance(item, (int, float)) and not math.isfinite(float(item)):
                raise ValueError(f"{key} must be finite")
        return value


class ConfigResponse(BaseModel):
    mode: RunMode


class StatusResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    mode: RunMode
    runner_started_at: str | None = None
    running: bool
    healthy: bool
    workers: dict[str, Any]
    autostart_attempted: bool
    autostart_error: str | None = None
    runtime_continuity: Any = None
    sensors: Any = Field(default_factory=dict)
    controller: dict[str, Any]
    actuator: dict[str, Any]
    evaluator: dict[str, Any]
    supervision: dict[str, Any]
    quality: dict[str, Any]


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    checks: dict[str, Any]
    workers: dict[str, Any]


class RunStartResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    mode: RunMode
    runner_started_at: str | None = None


class RunStopResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str


class DecisionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    decision_id: str
    market_id: str | None = None
    token_id: str | None = None
    side: str | None = None
    notional_usdc: float | None = None
    status: str | None = None
    factor_snapshot_hash: str | None = None
    decision_evidence: dict[str, Any] = Field(default_factory=dict)


class SignalResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    market_id: str
    token_id: str | None
    venue: str
    title: str
    yes_price: float
    volume_24h: float | None
    resolves_at: datetime | None
    orderbook: dict[str, Any]
    external_signal: dict[str, Any]
    fetched_at: datetime
    market_status: str


class SignalDepthLevelResponse(BaseModel):
    price: float
    size: float


class SignalDepthResponse(BaseModel):
    best_bid: float | None
    best_ask: float | None
    bids: list[SignalDepthLevelResponse]
    asks: list[SignalDepthLevelResponse]
    last_update_ts: str | None
    stale: bool


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str | None = None


class DecisionExplainResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    decision_id: str
    available: bool
    reason: str | None = None
    decision_trace: dict[str, Any] | None = None
    forecasts: dict[str, Any] | None = None
    gates: dict[str, Any] | None = None
    sizing: dict[str, Any] | None = None
    risk: dict[str, Any] | None = None


class RiskCheckResponse(BaseModel):
    decision_id: str
    approved: bool
    reason: str


class HaltResponse(BaseModel):
    status: str
    halted: bool | None = None
    reason: str | None = None
    cleared_by: str | None = None
    active_trigger_kind: str | None = None


class SelectionExplainResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    market_id: str
    available: bool
    reason: str | None = None
    reasons: list[str] = Field(default_factory=list)
    strategy_ids: list[str] = Field(default_factory=list)
    token_ids: list[str] = Field(default_factory=list)
    pinned_token_ids: list[str] = Field(default_factory=list)
    capped: bool = False
    strategy_reasons: dict[str, str] = Field(default_factory=dict)


class EvaluationReportsResponse(BaseModel):
    reports: list["EvaluationReportResponse"]


class EvaluationReportResponse(BaseModel):
    report_id: UUID
    run_id: UUID
    ranking_metric: Literal["brier", "sharpe", "pnl_cum"]
    ranked_strategies: list[dict[str, Any]]
    benchmark_rows: list[dict[str, Any]]
    attribution_commentary: str | None = None
    warnings: list[Any]
    next_action: str | None = None
    generated_at: datetime


EvaluationReportsResponse.model_rebuild()


class DecayStatusResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    strategy_id: str
    strategy_version_id: str
    decay_status: Literal[
        "healthy",
        "degraded",
        "negative",
        "insufficient_peak_data",
        "insufficient_resolved_outcomes",
        "producer_not_wired",
    ]
    reason: str | None = None


class ApprovalPendingResponse(BaseModel):
    approvals: list[dict[str, Any]]


class MetricsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    ops_view: dict[str, Any]
    per_strategy: list[dict[str, Any]]
    mark_to_market: dict[str, Any]
    quality: dict[str, Any]


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    feedback_id: str
    target: str
    source: str
    message: str
    severity: str
    created_at: datetime
    resolved: bool = False
    resolved_at: datetime | None = None
    category: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    strategy_id: str = "default"
    strategy_version_id: str = "default-v1"


class StreamEventResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    event: dict[str, Any] | None = None


class ResearchRunResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    run_ids: list[str] = Field(default_factory=list)
    unique_run_count: int | None = None
    runs: list[dict[str, Any]] = Field(default_factory=list)
    run_id: str | None = None
    status: str | None = None


class ResearchStrategyRunResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    strategy_run_id: str | None = None
    run_id: str | None = None
    strategy_id: str | None = None
    strategy_version_id: str | None = None


class ComparisonResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    run_id: str | None = None
    strategy_id: str | None = None
    strategy_version_id: str | None = None


class ReconcileResponse(BaseModel):
    status: str
    decision_id: str
    resolution: str


def _require_mode_scope(request: Request, mode: RunMode) -> None:
    auth = principal(request)
    if auth is None:
        if mode == RunMode.LIVE:
            raise HTTPException(status_code=401, detail="LIVE control requires an API token.")
        return
    required = "live_admin" if mode == RunMode.LIVE else "paper_operate"
    if required not in auth.scopes:
        raise HTTPException(status_code=403, detail="API token does not have the required scope.")
    request.state.auth_scope = required


def _is_live_control_request(
    request: Request,
    current_mode: RunMode,
    *,
    request_body: bytes | None = None,
) -> bool:
    if request.method.upper() not in {"POST", "PUT", "PATCH", "DELETE"}:
        return False
    path = request.url.path
    if path.startswith("/live/"):
        return True
    if path.startswith("/run/"):
        return current_mode == RunMode.LIVE and path not in {"/run/stop", "/run/kill"}
    if path.startswith("/decisions/") and path.endswith("/accept"):
        return current_mode == RunMode.LIVE
    if path.startswith("/feedback/"):
        return current_mode == RunMode.LIVE
    if path == "/config":
        if current_mode == RunMode.LIVE:
            return True
        if request_body:
            try:
                payload = json.loads(request_body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return False
            return isinstance(payload, dict) and payload.get("mode") == RunMode.LIVE.value
    return False


def _require_stop_scope(request: Request, mode: RunMode) -> None:
    auth = principal(request)
    if auth is None:
        if mode == RunMode.LIVE:
            raise HTTPException(status_code=401, detail="LIVE control requires an API token.")
        return
    if "kill" in auth.scopes:
        request.state.auth_scope = "kill"
        return
    required = "live_admin" if mode == RunMode.LIVE else "paper_operate"
    if required not in auth.scopes:
        raise HTTPException(status_code=403, detail="API token does not have the required scope.")
    request.state.auth_scope = required


def _live_audit_actor_allowed(path: str, auth: Any) -> bool:
    if auth is None:
        return False
    scopes: frozenset[str] = getattr(auth, "scopes", frozenset())
    if path in {"/run/stop", "/run/kill"}:
        return bool({"kill", "live_admin"}.intersection(scopes))
    return "live_admin" in scopes


def _scope_for_request(app: FastAPI, request: Request) -> str:
    """Resolve the declared route scope before LIVE audit preflight."""
    for route in app.routes:
        matcher = getattr(route, "matches", None)
        if not callable(matcher):
            continue
        match, _child_scope = matcher(request.scope)
        if match is not Match.FULL:
            continue
        names = {
            getattr(dependency.call, "__name__", "")
            for dependency in getattr(getattr(route, "dependant", None), "dependencies", ())
        }
        declared = sorted(
            name.removeprefix("require_scope_")
            for name in names
            if name.startswith("require_scope_")
        )
        if declared:
            return declared[0]
        any_declared = sorted(
            {
                "require_any_scope_paper_operate_live_admin": "paper_operate,live_admin",
                "require_any_scope_kill_paper_operate_live_admin": "kill,paper_operate,live_admin",
            }.get(name, name.removeprefix("require_any_scope_"))
            for name in names
            if name.startswith("require_any_scope_")
        )
        if any_declared:
            return any_declared[0]
    return "unknown"


class SubscriptionStateResponse(BaseModel):
    asset_ids: list[str]
    count: int
    last_updated_at: str | None


class SubmissionUnknownReconcileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_id: str
    venue_order_id: str | None = None
    status: Literal["filled", "not_found", "open"]
    reconciled_by: str
    note: str | None = None
    fill_price: float | None = None
    fill_notional_usdc: float | None = None
    fill_quantity: float | None = None
    filled_at: datetime | None = None
    fees: float | None = None
    fee_bps: int | None = None


class HaltClearRequest(BaseModel):
    cleared_by: str


def create_app(
    runner: Runner | None = None,
    *,
    auto_start: bool | None = None,
    config_path: str | None = None,
) -> FastAPI:
    active_runner = runner or Runner(config=load_settings(config_path))
    if auto_start is None:
        auto_start = os.environ.get("PMS_AUTO_START", "").lower() in {"1", "true", "yes"}

    @asynccontextmanager
    async def lifespan(app_inst: FastAPI) -> AsyncIterator[None]:
        pool_was_bound = active_runner.pg_pool is not None
        runner_pool_initialized = False
        startup_complete = False
        app_inst.state.autostart_attempted = False
        app_inst.state.autostart_error = None
        app_inst.state.shutting_down = False
        app_inst.state.alerting_task = None
        app_inst.state.alerting_stop_event = None
        app_inst.state.discord_client = None
        app_inst.state.eod_scheduler_task = None
        app_inst.state.eod_stop_event = None
        try:
            await _ensure_runner_pool(active_runner)
            runner_pool_initialized = active_runner.pg_pool is not None and not pool_was_bound
            if (
                active_runner.pg_pool is not None
                and _should_enforce_schema_check(active_runner.config)
            ):
                await ensure_schema_current(active_runner.pg_pool)
            if auto_start and not _is_runner_running(active_runner):
                app_inst.state.autostart_attempted = True
                logger.info(
                    "PMS_AUTO_START enabled — starting runner in %s mode",
                    active_runner.state.mode.value,
                )
                try:
                    await active_runner.start()
                except Exception as exc:  # noqa: BLE001
                    # Capture the failure so /status can surface it. We
                    # *intentionally* keep the API process alive so an
                    # operator can hit /status, see the failure mode, and
                    # remediate (e.g. create the missing DB) without losing
                    # the API control plane. The previous behaviour
                    # (silent escape, 32-hour zombie) is the load-bearing
                    # incident this guard exists for.
                    #
                    # `/status` is unauthenticated, so the exposed value is
                    # restricted to the exception class name. Connection
                    # errors / schema failures / OSError can otherwise leak
                    # DSNs, hostnames, paths, and user info via the raw
                    # str(exc). Server-side logs keep the exception class and
                    # a redacted diagnostic string so operators can still
                    # diagnose without leaking credentials into production
                    # logs.
                    app_inst.state.autostart_error = type(exc).__name__
                    logger.critical(
                        "PMS_AUTO_START failed: %s: %s",
                        type(exc).__name__,
                        redact_live_error(str(exc), active_runner.config),
                    )
            if active_runner.pg_pool is not None:
                await scan_orphaned_backtest_runs(active_runner.pg_pool)
            _start_alerting_if_configured(app_inst, active_runner)
            startup_complete = True
            yield
        finally:
            app_inst.state.shutting_down = True
            await _stop_alerting_if_started(app_inst)
            # Always stop a running runner on shutdown — covers both auto_start
            # and runners launched by callers via POST /run/start, so sensor
            # resources (for example venue HTTP clients) close cleanly.
            if startup_complete:
                if _is_runner_running(active_runner):
                    await active_runner.stop()
                else:
                    await _close_runner_pool(active_runner)
            elif runner_pool_initialized and not _is_runner_running(active_runner):
                await _close_runner_pool(active_runner)

    app = FastAPI(title="PMS API", lifespan=lifespan)
    app.state.runner = active_runner
    app.state.settings = active_runner.config
    app.include_router(agent_evidence_router)
    app.include_router(agent_factors_router)

    @app.middleware("http")
    async def request_audit(request: Request, call_next: Any) -> Any:
        if "token_id" in request.query_params:
            # Reject client-supplied identity before dispatching the handler,
            # but still record the mutating request outcome.
            try:
                authenticate(request)
            except HTTPException:
                pass
            response: Any = JSONResponse(
                status_code=422,
                content={"detail": "token_id is server-assigned."},
            )
            if request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
                pool = getattr(active_runner, "pg_pool", None)
                if pool is not None and hasattr(pool, "acquire"):
                    try:
                        await append_request_audit(
                            pool,
                            token_id=str(
                                getattr(request.state, "token_id", None)
                                or "anonymous"
                            ),
                            scope="unknown",
                            method=request.method.upper(),
                            path=request.url.path,
                            outcome=422,
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception("API request audit append failed for spoofed identity")
                        _write_audit_fallback(request, 422, active_runner.config)
                elif request.url.path in {"/run/stop", "/run/kill"}:
                    _write_audit_fallback(request, 422, active_runner.config)
            return response
        pool = getattr(active_runner, "pg_pool", None)
        request_body: bytes | None = None
        if request.url.path == "/config":
            request_body = await request.body()
        live_control = _is_live_control_request(
            request,
            active_runner.config.mode,
            request_body=request_body,
        )
        preflight_principal = None
        if live_control and (
            active_runner.config.api_token or active_runner.config.api_tokens
        ):
            try:
                preflight_principal = authenticate(request)
            except HTTPException:
                preflight_principal = None
        if preflight_principal is not None:
            request.state.auth_scope = _scope_for_request(app, request)
        live_actor_allowed = _live_audit_actor_allowed(
            request.url.path,
            preflight_principal,
        )
        if (
            live_control
            and live_actor_allowed
            and pool is None
        ):
            return JSONResponse(status_code=503, content={"detail": "Request audit unavailable."})
        if live_control and live_actor_allowed and pool is not None and hasattr(pool, "fetchval"):
            try:
                await append_request_audit(
                    pool,
                    token_id=str(getattr(request.state, "token_id", None) or "anonymous"),
                    scope=str(getattr(request.state, "auth_scope", "unknown")),
                    method=request.method.upper(),
                    path=request.url.path,
                    outcome=102,
                )
            except Exception:  # noqa: BLE001
                logger.exception("API request audit sink is unavailable")
                return JSONResponse(status_code=503, content={"detail": "Request audit unavailable."})
        response = None
        try:
            response = await call_next(request)
        finally:
            if request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
                pool = getattr(active_runner, "pg_pool", None)
                audit_available = (
                    pool is not None
                    and hasattr(pool, "acquire")
                    and hasattr(pool, "fetch")
                )
                if audit_available:
                    try:
                        await append_request_audit(
                            pool,
                            token_id=str(getattr(request.state, "token_id", None) or "anonymous"),
                            scope=str(getattr(request.state, "auth_scope", "unknown")),
                            method=request.method.upper(),
                            path=request.url.path,
                            outcome=500 if response is None else response.status_code,
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception("API request audit append failed")
                        if live_control:
                            # The preflight reservation prevents an unavailable
                            # sink before a risk-increasing operation. If the
                            # sink fails after the handler has run, preserve the
                            # successful side effect and make degradation
                            # explicit while recording a local fallback.
                            _write_audit_fallback(
                                request,
                                500 if response is None else response.status_code,
                                active_runner.config,
                            )
                            if response is not None:
                                response.headers["X-Audit-Degraded"] = "true"
                                response = await _audit_degraded_response(response)
                        elif request.url.path in {"/run/stop", "/run/kill"}:
                            _write_audit_fallback(
                                request,
                                500 if response is None else response.status_code,
                                active_runner.config,
                            )
                            if response is not None and active_runner.config.mode == RunMode.LIVE:
                                response.headers["X-Audit-Degraded"] = "true"
                                response = await _audit_degraded_response(response)
                elif request.url.path in {"/run/stop", "/run/kill"}:
                    _write_audit_fallback(
                        request,
                        500 if response is None else response.status_code,
                        active_runner.config,
                    )
                    if response is not None and active_runner.config.mode == RunMode.LIVE:
                        response.headers["X-Audit-Degraded"] = "true"
                        response = await _audit_degraded_response(response)
        return response

    @app.get("/status", response_model=StatusResponse, tags=["info-collection"], dependencies=[Depends(require_scope("read"))])
    async def status(request: Request) -> dict[str, Any]:
        records = await active_runner.eval_store.all()
        unresolved_feedback = await list_feedback_items(
            active_runner.feedback_store,
            resolved=False,
        )
        status_now = datetime.now(tz=UTC)
        metrics_snapshot = MetricsCollector(records).global_ops_snapshot()
        quote_records = await _quote_eval_records(active_runner)
        quote_snapshot = QuoteMetricsCollector(quote_records).global_ops_snapshot()
        mark_to_market = await _mark_to_market_payload(active_runner)
        metrics_14d = MetricsCollector(
            _eval_records_since(records, since=status_now - timedelta(days=14))
        ).global_ops_snapshot()
        return {
            "mode": active_runner.state.mode.value,
            "runner_started_at": _jsonable(active_runner.state.runner_started_at),
            # `running` keeps its historical any-task-alive semantics for
            # smoke-script back-compat; `healthy`/`workers` are additive and
            # carry per-worker supervision truth.
            "running": _is_runner_running(active_runner),
            "healthy": _runner_workers_healthy(active_runner),
            "workers": active_runner.worker_component_payload(),
            "autostart_attempted": getattr(
                request.app.state, "autostart_attempted", False
            ),
            "autostart_error": getattr(request.app.state, "autostart_error", None),
            "runtime_continuity": await _runtime_continuity_status(active_runner),
            "sensors": _sensor_statuses(active_runner),
            "controller": {
                "current_runtimes_total": active_runner.controller_runtime_count,
                "decisions_total": len(active_runner.state.decisions),
                "diagnostics_total": len(active_runner.state.controller_diagnostics),
                "diagnostic_counts": _controller_diagnostic_counts(
                    active_runner.state.controller_diagnostics,
                ),
            },
            "actuator": _actuator_status(active_runner, now=status_now),
            "evaluator": {
                "eval_records_total": len(records),
                "brier_overall": metrics_snapshot.brier_overall,
                "baseline_brier_overall": metrics_snapshot.baseline_brier_overall,
                "brier_improvement_overall": metrics_snapshot.brier_improvement_overall,
                "brier_14d": metrics_14d.brier_overall,
                "baseline_brier_14d": metrics_14d.baseline_brier_overall,
                "brier_improvement_14d": metrics_14d.brier_improvement_overall,
                "resolution_sweeps_total": active_runner.resolution_sweeps_total,
                "resolution_fills_resolved_total": (
                    active_runner.resolution_fills_resolved_total
                ),
                "resolution_sweep_failures_total": (
                    active_runner.resolution_sweep_failures_total
                ),
            },
            "supervision": {
                "unresolved_feedback_total": len(unresolved_feedback),
            },
            "quality": _quality_payload(
                records=records,
                metrics_snapshot=metrics_snapshot,
                mark_to_market=mark_to_market,
                quote_snapshot=quote_snapshot,
            ),
        }

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(request: Request) -> JSONResponse:
        code, payload = health_payload(
            shutting_down=bool(getattr(request.app.state, "shutting_down", False))
        )
        return JSONResponse(status_code=code, content=payload)

    @app.get("/readiness", response_model=ReadinessResponse, tags=["ops"])
    async def readiness(request: Request) -> JSONResponse:
        code, payload = readiness_payload(
            active_runner,
            halt_subscriber_task=getattr(request.app.state, "alerting_task", None),
            eod_scheduler_task=getattr(request.app.state, "eod_scheduler_task", None),
            shutting_down=bool(getattr(request.app.state, "shutting_down", False)),
            forced_running=bool(getattr(request.app.state, "runner_readiness_forced", False)),
        )
        return JSONResponse(status_code=code, content=payload)

    @app.get("/signals", response_model=list[SignalResponse], tags=["info-collection"], dependencies=[Depends(require_scope("read"))])
    async def signals(limit: int = 50) -> list[dict[str, Any]]:
        return [
            cast(dict[str, Any], _jsonable(signal))
            for signal in _latest(active_runner.state.signals, limit)
        ]

    @app.get("/markets", response_model=MarketsListResponse, tags=["info-collection"], dependencies=[Depends(require_scope("read"))])
    async def markets(
        limit: int = Query(default=20, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        q: str = Query(default=""),
        volume_min: float = Query(default=0.0, ge=0.0),
        liquidity_min: float = Query(default=0.0, ge=0.0),
        spread_max_bps: int | None = Query(default=None, ge=0),
        yes_min: float = Query(default=0.0, ge=0.0, le=1.0),
        yes_max: float = Query(default=1.0, ge=0.0, le=1.0),
        resolves_within_days: int | None = Query(default=None, ge=0),
        subscribed: SubscribedFilter = Query(default="all"),
    ) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        if yes_min > yes_max:
            raise HTTPException(
                status_code=422,
                detail="yes_min must be less than or equal to yes_max",
            )
        payload = await list_markets_items(
            PostgresMarketDataStore(active_runner.pg_pool),
            current_asset_ids=_current_subscription_asset_ids(active_runner),
            limit=limit,
            offset=offset,
            filters=MarketsFilterParams(
                q=q,
                volume_min=volume_min,
                liquidity_min=liquidity_min,
                spread_max_bps=spread_max_bps,
                yes_min=yes_min,
                yes_max=yes_max,
                resolves_within_days=resolves_within_days,
                subscribed=subscribed,
            ),
        )
        return payload.model_dump(mode="json")

    @app.get("/markets/{condition_id}", response_model=MarketRow, tags=["info-collection"], dependencies=[Depends(require_scope("read"))])
    async def market_detail(condition_id: str) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        try:
            payload = await get_market_item(
                PostgresMarketDataStore(active_runner.pg_pool),
                current_asset_ids=_current_subscription_asset_ids(active_runner),
                market_id=condition_id,
            )
        except MarketNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Market not found") from exc
        return payload.model_dump(mode="json")

    @app.get(
        "/markets/{condition_id}/price-history",
        response_model=PriceHistoryResponse,
        tags=["info-collection"],
        dependencies=[Depends(require_scope("read"))],
    )
    async def market_price_history(
        condition_id: str,
        since: datetime | None = None,
        limit: int = Query(default=1440, ge=1),
    ) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        try:
            payload = await get_price_history_item(
                PostgresMarketDataStore(active_runner.pg_pool),
                condition_id=condition_id,
                since=since,
                limit=limit,
            )
        except MarketPriceHistoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Market not found") from exc
        return payload.model_dump(mode="json")

    @app.post(
        "/markets/{token_id}/subscribe",
        response_model=SubscribeMarketResponse,
        tags=["info-collection"],
        dependencies=[Depends(require_scope("paper_operate"))],
    )
    async def subscribe_market(token_id: str, request: Request) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        try:
            payload = await subscribe_market_item(
                PostgresMarketSubscriptionStore(active_runner.pg_pool),
                token_id=token_id,
                request_metadata=_request_metadata(request),
                max_subscription_asset_ids=(
                    active_runner.config.sensor.max_subscription_asset_ids
                ),
                protected_asset_ids=tuple(
                    position.token_id
                    for position in active_runner.portfolio.open_positions
                    if position.token_id is not None
                ),
            )
        except UnknownSubscriptionTokenError as exc:
            raise HTTPException(status_code=404, detail="Token not found") from exc
        request_reselection = getattr(active_runner, "_request_reselection", None)
        if callable(request_reselection):
            await request_reselection()
        return payload.model_dump(mode="json")

    @app.delete(
        "/markets/{token_id}/subscribe",
        response_model=UnsubscribeMarketResponse,
        tags=["info-collection"],
        dependencies=[Depends(require_scope("paper_operate"))],
    )
    async def unsubscribe_market(token_id: str, request: Request) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        payload = await unsubscribe_market_item(
            PostgresMarketSubscriptionStore(active_runner.pg_pool),
            token_id=token_id,
            request_metadata=_request_metadata(request),
        )
        await active_runner._request_reselection()
        return payload.model_dump(mode="json")

    @app.get("/positions", response_model=PositionsResponse, tags=["trading"], dependencies=[Depends(require_scope("read"))])
    async def positions() -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        payload = await list_positions_items(
            active_runner.fill_store,
            run_mode=active_runner.config.mode.value,
        )
        return payload.model_dump(mode="json")

    @app.get("/trades", response_model=TradesResponse, tags=["trading"], dependencies=[Depends(require_scope("read"))])
    async def trades(
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        until: datetime | None = Query(default=None),
    ) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        payload = await list_trades_items(
            active_runner.fill_store,
            limit=limit,
            offset=offset,
            until=until,
            run_mode=active_runner.config.mode.value,
        )
        return payload.model_dump(mode="json")

    @app.get("/signals/{market_id}/depth", response_model=SignalDepthResponse, tags=["info-collection"], dependencies=[Depends(require_scope("read"))])
    async def signal_depth(
        market_id: str,
        limit: int = Query(default=20, ge=1, le=200),
    ) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        try:
            payload = await get_signal_depth(
                PostgresMarketDataStore(active_runner.pg_pool),
                market_id=market_id,
                limit=limit,
                stale_snapshot_threshold_s=active_runner.config.dashboard.stale_snapshot_threshold_s,
            )
        except SignalDepthNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return cast(dict[str, Any], _jsonable(payload))

    @app.get("/decisions", response_model=list[DecisionResponse], tags=["decision"], dependencies=[Depends(require_scope("read"))])
    async def decisions(
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        status: str | None = None,
        include: str | None = None,
        until: datetime | None = Query(default=None),
    ) -> list[dict[str, Any]]:
        include_opportunity = include == "opportunity"
        if _should_use_durable_decisions(active_runner):
            rows = await list_decisions_items(
                cast(Any, active_runner.decision_store),
                limit=limit,
                offset=offset,
                status=status,
                include_opportunity=include_opportunity,
                until=until,
            )
            return [item.model_dump(mode="json") for item in rows]

        payloads: list[dict[str, Any]] = []
        for decision in _latest(active_runner.state.decisions, limit, offset=offset):
            payload = cast(dict[str, Any], _jsonable(decision))
            payload["forecaster"] = _forecaster(decision)
            payload["kelly_size"] = decision.notional_usdc
            payloads.append(payload)
        return payloads

    @app.get("/decisions/{decision_id}", response_model=DecisionResponse, tags=["decision"], dependencies=[Depends(require_scope("read"))])
    async def decision_detail(
        decision_id: str,
        include: str | None = None,
    ) -> dict[str, Any]:
        if _should_use_durable_decisions(active_runner):
            row = await get_decision_item(
                cast(Any, active_runner.decision_store),
                decision_id=decision_id,
                include_opportunity=include == "opportunity",
            )
            if row is None:
                raise HTTPException(status_code=404, detail="Decision not found")
            return row.model_dump(mode="json")

        for decision in active_runner.state.decisions:
            if decision.decision_id != decision_id:
                continue
            payload = cast(dict[str, Any], _jsonable(decision))
            payload["forecaster"] = _forecaster(decision)
            payload["kelly_size"] = decision.notional_usdc
            return payload
        raise HTTPException(status_code=404, detail="Decision not found")

    @app.post(
        "/decisions/{decision_id}/accept",
        response_model=AcceptDecisionResponse,
        tags=["decision"],
        dependencies=[Depends(require_any_scope("paper_operate", "live_admin"))],
    )
    async def accept_decision(
        decision_id: str,
        body: AcceptDecisionRequest,
        request: Request,
    ) -> Any:
        _require_mode_scope(request, active_runner.config.mode)
        logger.info(
            "decision.accept.request",
            extra={"token_id": getattr(request.state, "token_id", None), "decision_id": decision_id},
        )
        try:
            payload = await accept_decision_item(
                cast(Any, active_runner.decision_store),
                decision_id=decision_id,
                factor_snapshot_hash=body.factor_snapshot_hash,
                dedup_store=cast(Any, active_runner.actuator_executor.dedup_store),
                risk=cast(Any, active_runner.actuator_executor.risk),
                portfolio=active_runner.portfolio,
                enqueue=active_runner.enqueue_accepted_decision,
            )
        except DecisionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DecisionMarketChangedError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": str(exc),
                    "current_factor_snapshot_hash": exc.current_factor_snapshot_hash,
                },
            )
        except DecisionEnqueueRejectedError as exc:
            raise HTTPException(status_code=409, detail=exc.reason) from exc
        except DecisionRiskRejectedError as exc:
            raise HTTPException(status_code=422, detail=exc.reason) from exc
        return payload.model_dump(mode="json")

    @app.get("/metrics", response_model=MetricsResponse, tags=["ops"], dependencies=[Depends(require_scope("read"))])
    async def metrics(
        since: datetime | None = Query(default=None),
        until: datetime | None = Query(default=None),
    ) -> dict[str, Any]:
        records = sorted(
            await active_runner.eval_store.all(),
            key=lambda record: (record.recorded_at, record.decision_id),
        )
        quote_records = sorted(
            await _quote_eval_records(active_runner),
            key=lambda record: (record.recorded_at, record.fill_id),
        )
        mark_to_market = await _mark_to_market_payload(active_runner)
        records = _eval_records_in_window(records, since=since, until=until)
        quote_records = _quote_records_in_window(quote_records, since=since, until=until)
        first_trade_time_seconds = await _first_trade_time_seconds(active_runner.pg_pool)
        return _metrics_payload(
            records,
            quote_records=quote_records,
            mark_to_market=mark_to_market,
            first_trade_time_seconds=first_trade_time_seconds,
            window_started_at=since,
            window_ended_at=until,
            capital_base_usdc=active_runner.config.risk.max_total_exposure,
        )

    @app.get("/feedback", response_model=list[FeedbackResponse], tags=["ops"], dependencies=[Depends(require_scope("read"))])
    async def feedback(resolved: bool | None = None) -> list[dict[str, Any]]:
        return [
            cast(dict[str, Any], _jsonable(item))
            for item in await list_feedback_items(active_runner.feedback_store, resolved=resolved)
        ]

    @app.get("/stream/events", response_model=StreamEventResponse, tags=["ops"], dependencies=[Depends(require_scope("read"))])
    async def stream_events(
        request: Request,
        last_event_id: int | None = Query(default=None, ge=0),
    ) -> StreamingResponse:
        async def event_generator() -> AsyncIterator[str]:
            replay, subscriber = await active_runner.event_bus.subscribe(
                last_event_id=last_event_id
            )
            try:
                for event in replay:
                    yield encode_sse_event(event)

                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        event = await asyncio.wait_for(subscriber.get(), timeout=10.0)
                    except TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield encode_sse_event(event)
            finally:
                await active_runner.event_bus.unsubscribe(subscriber)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "connection": "keep-alive",
            },
        )

    @app.get("/subscriptions", response_model=SubscriptionStateResponse, tags=["info-collection"], dependencies=[Depends(require_scope("read"))])
    async def subscriptions() -> dict[str, Any]:
        subscription_controller = active_runner.subscription_controller
        asset_ids = sorted(_current_subscription_asset_ids(active_runner))
        response = SubscriptionStateResponse(
            asset_ids=asset_ids,
            count=len(asset_ids),
            last_updated_at=(
                _jsonable(subscription_controller.last_updated_at)
                if subscription_controller is not None and asset_ids
                else None
            ),
        )
        return response.model_dump(mode="json")

    @app.get("/strategies", response_model=StrategiesResponse, tags=["decision"], dependencies=[Depends(require_scope("read"))])
    async def strategies() -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        return await list_strategies_items(
            active_runner.pg_pool,
            settings=active_runner.config,
        )

    @app.get("/strategies/metrics", response_model=StrategyMetricsResponse, tags=["decision"], dependencies=[Depends(require_scope("read"))])
    async def strategy_metrics(
        since: datetime | None = Query(default=None),
        until: datetime | None = Query(default=None),
    ) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        return await list_strategy_metrics_items(
            active_runner.pg_pool,
            since=since,
            until=until,
            settings=active_runner.config,
        )

    @app.get(
        "/strategies/{strategy_id}/decay-status",
        response_model=DecayStatusResponse,
        tags=["decision"],
        dependencies=[Depends(require_scope("read"))],
    )
    async def strategy_decay_status(
        strategy_id: str,
        strategy_version_id: str | None = None,
    ) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        try:
            payload = await get_strategy_decay_status_item(
                active_runner.pg_pool,
                strategy_id=strategy_id,
                strategy_version_id=strategy_version_id,
                min_resolved_samples=active_runner.config.decay_min_resolved_samples,
            )
            if payload.get("decay_status") == "producer_not_wired":
                payload["reason"] = "producer_not_wired"
            return payload
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/share/{strategy_id}", response_model=ShareResponse, tags=["info-collection"])
    async def share_strategy(strategy_id: str) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        payload = await get_shared_strategy(active_runner.pg_pool, strategy_id)
        if payload is None:
            raise HTTPException(status_code=404, detail=SHARE_NOT_FOUND_DETAIL)
        return payload.model_dump(mode="json")

    @app.get("/factors/catalog", response_model=FactorCatalogResponse, tags=["factors"], dependencies=[Depends(require_scope("read"))])
    async def factors_catalog() -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        return await list_factor_catalog(active_runner.pg_pool)

    @app.get("/factors", response_model=FactorSeriesResponse, tags=["factors"], dependencies=[Depends(require_scope("read"))])
    async def factors(
        factor_id: str,
        market_id: str,
        param: str = "",
        since: datetime | None = None,
        limit: int = Query(default=500, ge=1, le=2000),
    ) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        return await list_factor_series(
            active_runner.pg_pool,
            factor_id=factor_id,
            market_id=market_id,
            param=param,
            since=since,
            limit=limit,
        )

    @app.post(
        "/research/backtest",
        response_model=ResearchRunResponse,
        response_model_exclude_unset=True,
        tags=["research"],
        dependencies=[Depends(require_scope("research"))],
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {"schema": ResearchRunRequest.model_json_schema()},
                    "text/plain": {"schema": {"type": "string"}},
                },
            },
        },
    )
    async def create_backtest_run(request: Request) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        try:
            raw = await request.body()
            try:
                parsed = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, dict) and "sweep_yaml" in parsed:
                sweep_yaml = ResearchRunRequest.model_validate(parsed).sweep_yaml
            else:
                sweep_yaml = raw.decode("utf-8")
            return await enqueue_backtest_runs(active_runner.pg_pool, sweep_yaml)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/research/backtest", response_model=list[ResearchRunResponse], response_model_exclude_unset=True, tags=["research"], dependencies=[Depends(require_scope("research"))])
    async def get_backtest_runs(limit: int = Query(default=25, ge=1, le=100)) -> list[dict[str, Any]]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        return await list_backtest_runs(active_runner.pg_pool, limit=limit)

    @app.get("/research/backtest/{run_id}", response_model=ResearchRunResponse, response_model_exclude_unset=True, tags=["research"], dependencies=[Depends(require_scope("research"))])
    async def get_backtest_run(run_id: UUID) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        payload = await fetch_backtest_run(active_runner.pg_pool, str(run_id))
        if payload is None:
            raise HTTPException(status_code=404, detail="Backtest run not found")
        return payload

    @app.get(
        "/research/backtest/{run_id}/strategies",
        response_model=list[ResearchStrategyRunResponse],
        response_model_exclude_unset=True,
        tags=["research"],
        dependencies=[Depends(require_scope("research"))],
    )
    async def get_backtest_strategy_runs(run_id: UUID) -> list[dict[str, Any]]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        canonical_run_id = str(run_id)
        if await fetch_backtest_run(active_runner.pg_pool, canonical_run_id) is None:
            raise HTTPException(status_code=404, detail="Backtest run not found")
        return await list_backtest_strategy_runs(active_runner.pg_pool, canonical_run_id)

    @app.post(
        "/research/backtest/{run_id}/compare",
        response_model=ComparisonResponse,
        response_model_exclude_unset=True,
        tags=["research"],
        dependencies=[Depends(require_scope("research"))],
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {"schema": ComparisonRequest.model_json_schema()},
                },
            },
        },
    )
    async def compare_backtest_run(run_id: UUID, request: Request) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        canonical_run_id = str(run_id)
        if await fetch_backtest_run(active_runner.pg_pool, canonical_run_id) is None:
            raise HTTPException(status_code=404, detail="Backtest run not found")
        try:
            raw_payload = await request.json()
            payload = ComparisonRequest.model_validate(raw_payload).model_dump(mode="json")
            return await compute_backtest_live_comparison(
                active_runner.pg_pool,
                canonical_run_id,
                payload,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post(
        "/feedback/{feedback_id}/resolve",
        response_model=FeedbackResponse,
        tags=["ops"],
        dependencies=[Depends(require_any_scope("paper_operate", "live_admin"))],
    )
    async def resolve_feedback(feedback_id: str, request: Request) -> dict[str, Any]:
        _require_mode_scope(request, active_runner.config.mode)
        resolved = await resolve_feedback_item(active_runner.feedback_store, feedback_id)
        if resolved is None:
            raise HTTPException(status_code=404, detail="Feedback not found")
        return cast(dict[str, Any], _jsonable(resolved))

    @app.post(
        "/live/reconcile-submission-unknown",
        response_model=ReconcileResponse,
        tags=["ops"],
        dependencies=[Depends(require_scope("live_admin"))],
    )
    async def reconcile_submission_unknown(
        payload: SubmissionUnknownReconcileRequest,
        request: Request,
    ) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=503, detail="Runner PostgreSQL pool is not initialized")
        try:
            venue_order_id = normalize_submission_unknown_venue_order_id(
                status=payload.status,
                venue_order_id=payload.venue_order_id,
            )
            decision_id = normalize_submission_unknown_decision_id(
                payload.decision_id
            )
            reconciled_by = normalize_submission_unknown_reconciled_by(
                payload.reconciled_by
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            await ensure_schema_current(active_runner.pg_pool)
            updated = await SubmissionUnknownReconciliationStore(
                active_runner.pg_pool
            ).reconcile_submission_unknown(
                decision_id=decision_id,
                venue_order_id=venue_order_id,
                status=payload.status,
                reconciled_by=reconciled_by,
                note=payload.note,
                fill_price=payload.fill_price,
                fill_notional_usdc=payload.fill_notional_usdc,
                fill_quantity=payload.fill_quantity,
                filled_at=payload.filled_at,
                fees=payload.fees,
                fee_bps=payload.fee_bps,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=redact_live_error(str(exc), active_runner.config),
            ) from exc
        if not updated:
            raise HTTPException(
                status_code=404,
                detail="submission_unknown incident not found",
            )
        logger.info(
            "live.reconcile_submission_unknown",
            extra={"token_id": getattr(request.state, "token_id", None), "decision_id": decision_id},
        )
        return {
            "status": "reconciled",
            "decision_id": decision_id,
            "resolution": payload.status,
        }

    @app.post(
        "/live/halt/clear",
        response_model=HaltResponse,
        tags=["ops"],
        dependencies=[Depends(require_scope("live_admin"))],
    )
    async def clear_live_halt(payload: HaltClearRequest, request: Request) -> dict[str, str]:
        try:
            remaining_halt = await active_runner.clear_live_halt(
                cleared_by=payload.cleared_by,
                token_id=getattr(request.state, "token_id", None),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        response = {
            "status": "cleared" if remaining_halt is None else "clear_recorded",
            "cleared_by": payload.cleared_by,
        }
        if remaining_halt is not None:
            response["active_trigger_kind"] = remaining_halt.trigger_kind
        return response

    @app.post("/config", response_model=ConfigResponse, tags=["ops"], dependencies=[Depends(require_any_scope("paper_operate", "live_admin"))])
    async def update_config(update: ConfigUpdate, request: Request) -> dict[str, str]:
        current_mode = active_runner.config.mode
        if current_mode == RunMode.BACKTEST and update.mode != RunMode.LIVE:
            return {"mode": active_runner.state.mode.value}
        if current_mode == RunMode.LIVE or update.mode == RunMode.LIVE:
            auth = principal(request)
            if auth is None:
                raise HTTPException(status_code=401, detail="LIVE control requires an API token.")
            if "live_admin" not in auth.scopes:
                raise HTTPException(status_code=403, detail="API token does not have the required scope.")
            request.state.auth_scope = "live_admin"
        else:
            _require_mode_scope(request, update.mode)
        if (
            update.mode != active_runner.config.mode
            and _is_runner_running(active_runner)
        ):
            raise HTTPException(
                status_code=409,
                detail=MODE_CHANGE_WHILE_RUNNING_DETAIL,
            )
        if update.mode == RunMode.LIVE and not active_runner.config.live_trading_enabled:
            raise HTTPException(status_code=400, detail=LIVE_DISABLED_DETAIL)
        if update.mode == RunMode.LIVE:
            live_candidate = active_runner.config.model_copy(
                update={"mode": RunMode.LIVE}
            )
            try:
                validate_live_mode_ready(live_candidate)
                require_live_preflight_artifact(live_candidate)
            except LiveTradingDisabledError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=redact_live_error(str(exc), live_candidate),
                ) from exc
        active_runner.switch_mode(update.mode)
        return {"mode": active_runner.state.mode.value}

    # Serializes /run/start and /run/stop: Runner.start's await-laden
    # critical section is not safe to interleave — a losing concurrent
    # start's cleanup destroys the winner's runtime and closes the pool.
    run_lifecycle_lock = asyncio.Lock()

    @app.post("/run/start", response_model=RunStartResponse, tags=["ops"], dependencies=[Depends(require_any_scope("paper_operate", "live_admin"))])
    async def run_start(request: Request) -> dict[str, Any]:
        _require_mode_scope(request, active_runner.config.mode)
        async with run_lifecycle_lock:
            return await _run_start_locked()

    async def _run_start_locked() -> dict[str, Any]:
        if _is_runner_running(active_runner):
            raise HTTPException(status_code=409, detail=RUNNER_ALREADY_RUNNING_DETAIL)
        try:
            await active_runner.start()
        except LiveTradingDisabledError as exc:
            raise HTTPException(
                status_code=400,
                detail=redact_live_error(str(exc), active_runner.config),
            ) from exc
        except Exception as exc:
            if active_runner.config.mode == RunMode.LIVE:
                raise HTTPException(
                    status_code=400,
                    detail=redact_live_error(str(exc), active_runner.config),
                ) from exc
            raise
        return {
            "status": "started",
            "mode": active_runner.state.mode.value,
            "runner_started_at": _jsonable(active_runner.state.runner_started_at),
        }

    @app.post("/run/stop", response_model=RunStopResponse, tags=["ops"], dependencies=[Depends(require_any_scope("kill", "paper_operate", "live_admin"))])
    async def run_stop(request: Request) -> dict[str, Any]:
        _require_stop_scope(request, active_runner.config.mode)
        async with run_lifecycle_lock:
            await active_runner.stop(close_pool=False)
        return {"status": "stopped"}

    @app.post("/run/kill", response_model=RunStopResponse, tags=["ops"], dependencies=[Depends(require_scope("kill"))])
    async def run_kill() -> dict[str, Any]:
        # Kill is deliberately idempotent and does not clear risk halts or
        # switch mode; stopping the worker graph prevents new orders.
        async with run_lifecycle_lock:
            await active_runner.stop(close_pool=False)
        return {"status": "killed"}

    @app.get("/selection/explain", response_model=SelectionExplainResponse, tags=["info-collection"], dependencies=[Depends(require_scope("read"))])
    async def selection_explain(market_id: str) -> dict[str, Any]:
        selector = getattr(active_runner, "_market_selector", None)
        explain = getattr(selector, "explain", None)
        if not callable(explain):
            raise HTTPException(status_code=503, detail="selector_not_running")
        payload = explain(
            market_id,
            max_subscription_asset_ids=active_runner.config.sensor.max_subscription_asset_ids,
            current_asset_ids=_current_subscription_asset_ids(active_runner),
        )
        if asyncio.iscoroutine(payload):
            payload = await payload
        if isinstance(payload, dict):
            return {"market_id": market_id, "available": True, **payload}
        payload_dict = cast(dict[str, Any], _jsonable(payload))
        return payload_dict

    @app.get("/decisions/{decision_id}/explain", response_model=DecisionExplainResponse, tags=["decision"], dependencies=[Depends(require_scope("read"))])
    async def decision_explain(decision_id: str) -> dict[str, Any]:
        if not _should_use_durable_decisions(active_runner):
            return {"decision_id": decision_id, "available": False, "reason": "evidence_unavailable"}
        row = await get_decision_item(cast(Any, active_runner.decision_store), decision_id=decision_id, include_opportunity=True)
        if row is None:
            raise HTTPException(status_code=404, detail="Decision not found")
        evidence = dict(row.decision_evidence)
        trace = evidence.get("decision_trace")
        if not isinstance(trace, dict):
            return {
                "decision_id": decision_id,
                "available": False,
                "reason": "decision_trace_unavailable",
            }
        return {
            **evidence,
            "decision_id": decision_id,
            "available": True,
            "forecasts": trace.get("forecasts"),
            "gates": trace.get("gates"),
            "sizing": trace.get("sizing"),
            "risk": trace.get("risk"),
        }

    @app.post("/decisions/{decision_id}/risk-check", response_model=RiskCheckResponse, tags=["decision"], dependencies=[Depends(require_scope("read"))])
    async def decision_risk_check(decision_id: str) -> dict[str, Any]:
        row = await active_runner.decision_store.get_decision(decision_id, include_opportunity=False)
        if row is None:
            raise HTTPException(status_code=404, detail="Decision not found")
        risk = copy.deepcopy(active_runner.actuator_executor.risk)
        portfolio = copy.deepcopy(active_runner.portfolio)
        if row.status != "pending":
            return {
                "decision_id": decision_id,
                "approved": False,
                "reason": f"decision_{row.status}",
            }
        halt = risk.check_auto_halt(portfolio)
        if halt.halted:
            return {
                "decision_id": decision_id,
                "approved": False,
                "reason": halt.reason,
            }
        result = risk.check(copy.deepcopy(row.decision), portfolio)
        return {"decision_id": decision_id, "approved": result.approved, "reason": result.reason}

    @app.get("/approvals/pending", response_model=ApprovalPendingResponse, tags=["ops"], dependencies=[Depends(require_scope("read"))])
    async def approvals_pending() -> dict[str, Any]:
        path = active_runner.config.live_first_order_audit_path
        if not path:
            return {"approvals": []}
        return {"approvals": _pending_approval_events(Path(path))}

    @app.get("/evaluation/reports", response_model=EvaluationReportsResponse, tags=["research"], dependencies=[Depends(require_scope("read"))])
    async def evaluation_reports(
        limit: int = Query(default=25, ge=1, le=100),
    ) -> dict[str, Any]:
        if active_runner.pg_pool is None:
            return {"reports": []}
        async with active_runner.pg_pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT report_id, run_id, ranking_metric, ranked_strategies, benchmark_rows, attribution_commentary, warnings, next_action, generated_at FROM evaluation_reports ORDER BY generated_at DESC LIMIT $1",
                limit,
            )
        return {"reports": [_evaluation_report_payload(row) for row in rows]}

    @app.get("/evaluation/reports/{report_id}", response_model=EvaluationReportResponse, tags=["research"], dependencies=[Depends(require_scope("read"))])
    async def evaluation_report(report_id: UUID) -> EvaluationReportResponse:
        if active_runner.pg_pool is None:
            raise HTTPException(status_code=404, detail="Evaluation report not found")
        async with active_runner.pg_pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT report_id, run_id, ranking_metric, ranked_strategies, benchmark_rows, attribution_commentary, warnings, next_action, generated_at FROM evaluation_reports WHERE report_id = $1",
                report_id,
            )
        if row is None:
            raise HTTPException(status_code=404, detail="Evaluation report not found")
        return _evaluation_report_payload(row)

    return app


def _latest(items: Sequence[T], limit: int, *, offset: int = 0) -> list[T]:
    bounded_limit = max(limit, 0)
    if bounded_limit == 0:
        return []
    bounded_offset = max(offset, 0)
    if bounded_offset == 0:
        return list(items[-bounded_limit:])
    end = len(items) - bounded_offset
    if end <= 0:
        return []
    start = max(0, end - bounded_limit)
    return list(items[start:end])


async def _quote_eval_records(runner: Runner) -> list[QuoteEvalRecord]:
    store = getattr(runner, "quote_eval_store", None)
    all_records = getattr(store, "all", None)
    if not callable(all_records):
        return []
    try:
        return list(await all_records())
    except Exception:  # noqa: BLE001
        logger.exception("quote evaluation metrics unavailable")
        return []


async def _mark_to_market_payload(runner: Runner) -> dict[str, float | int]:
    mode_reader = getattr(runner.fill_store, "read_positions_for_mode", None)
    if callable(mode_reader):
        read_positions = mode_reader
        has_mode_reader = True
    else:
        read_positions = getattr(runner.fill_store, "read_positions", None)
        has_mode_reader = False
    if not callable(read_positions):
        return _mark_to_market_from_positions([])
    try:
        if has_mode_reader:
            positions = list(await read_positions(runner.config.mode.value))
        else:
            positions = list(await read_positions())
    except Exception:  # noqa: BLE001
        logger.exception("mark-to-market metrics unavailable")
        return _mark_to_market_from_positions([])
    return _mark_to_market_from_positions(positions)


def _mark_to_market_from_positions(
    positions: Sequence[Position],
) -> dict[str, float | int]:
    return {
        "open_positions": len(positions),
        "locked_usdc": float(sum(Decimal(str(item.locked_usdc)) for item in positions)),
        "unrealized_pnl": float(
            sum(Decimal(str(item.unrealized_pnl)) for item in positions)
        ),
    }


def _quality_payload(
    *,
    records: Sequence[EvalRecord],
    metrics_snapshot: MetricsSnapshot,
    mark_to_market: dict[str, float | int],
    quote_snapshot: QuoteMetricsSnapshot,
) -> dict[str, Any]:
    return {
        "final_brier": {
            "record_count": len(records),
            "brier_overall": metrics_snapshot.brier_overall,
        },
        "mark_to_market": mark_to_market,
        "quote_calibration": _quote_calibration_payload(quote_snapshot),
    }


def _quote_calibration_payload(snapshot: QuoteMetricsSnapshot) -> dict[str, Any]:
    return {
        "record_count": snapshot.record_count,
        "quote_score_overall": snapshot.quote_score_overall,
        "mtm_pnl": snapshot.mtm_pnl,
    }


def _quote_mtm_pnl_series(
    quote_records: Sequence[QuoteEvalRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record, cumulative_pnl in _quote_mtm_portfolio_snapshots(quote_records):
        rows.append(
            {
                "recorded_at": _coerce_aware_datetime(record.recorded_at).isoformat(),
                "pnl": float(cumulative_pnl),
                "source": "quote_mtm",
            }
        )
    return rows


def _quote_mtm_max_drawdown_pct(
    quote_records: Sequence[QuoteEvalRecord],
    *,
    capital_base_usdc: float | None,
) -> float | None:
    if capital_base_usdc is None or capital_base_usdc <= 0.0:
        return None

    peak_equity = Decimal("0")
    max_drawdown = Decimal("0")
    for _record, cumulative_pnl in _quote_mtm_portfolio_snapshots(quote_records):
        peak_equity = max(peak_equity, cumulative_pnl)
        max_drawdown = max(max_drawdown, peak_equity - cumulative_pnl)
    return float(max_drawdown / Decimal(str(capital_base_usdc)) * Decimal("100"))


def _quote_mtm_portfolio_snapshots(
    quote_records: Sequence[QuoteEvalRecord],
) -> list[tuple[QuoteEvalRecord, Decimal]]:
    latest_mtm_by_fill: dict[str, Decimal] = {}
    rows: list[tuple[QuoteEvalRecord, Decimal]] = []
    ordered_records = sorted(
        quote_records,
        key=lambda record: (
            _coerce_aware_datetime(record.recorded_at),
            record.fill_id,
            record.quote_lag_seconds,
        ),
    )
    for record in ordered_records:
        latest_mtm_by_fill[record.fill_id] = Decimal(str(record.mtm_pnl))
        rows.append(
            (
                record,
                sum(latest_mtm_by_fill.values(), Decimal("0")),
            )
        )
    return rows


def _eval_records_since(
    records: Sequence[EvalRecord],
    *,
    since: datetime,
) -> list[EvalRecord]:
    cutoff = _coerce_aware_datetime(since)
    return [
        record
        for record in records
        if _coerce_aware_datetime(record.recorded_at) >= cutoff
    ]


def _eval_records_in_window(
    records: Sequence[EvalRecord],
    *,
    since: datetime | None,
    until: datetime | None,
) -> list[EvalRecord]:
    lower = _coerce_aware_datetime(since) if since is not None else None
    upper = _coerce_aware_datetime(until) if until is not None else None
    filtered: list[EvalRecord] = []
    for record in records:
        recorded_at = _coerce_aware_datetime(record.recorded_at)
        if lower is not None and recorded_at < lower:
            continue
        if upper is not None and recorded_at >= upper:
            continue
        filtered.append(record)
    return filtered


def _quote_records_in_window(
    records: Sequence[QuoteEvalRecord],
    *,
    since: datetime | None,
    until: datetime | None,
) -> list[QuoteEvalRecord]:
    lower = _coerce_aware_datetime(since) if since is not None else None
    upper = _coerce_aware_datetime(until) if until is not None else None
    filtered: list[QuoteEvalRecord] = []
    for record in records:
        recorded_at = _coerce_aware_datetime(record.recorded_at)
        if lower is not None and recorded_at < lower:
            continue
        if upper is not None and recorded_at >= upper:
            continue
        filtered.append(record)
    return filtered


def _coerce_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _metrics_payload(
    records: list[EvalRecord],
    *,
    quote_records: list[QuoteEvalRecord] | None = None,
    mark_to_market: dict[str, float | int] | None = None,
    first_trade_time_seconds: float | None = None,
    window_started_at: datetime | None = None,
    window_ended_at: datetime | None = None,
    capital_base_usdc: float | None = None,
) -> dict[str, Any]:
    collector = MetricsCollector(records)
    quote_records = [] if quote_records is None else quote_records
    quote_snapshot = QuoteMetricsCollector(quote_records).global_ops_snapshot()
    ops_view = _metrics_aggregate_payload(
        records,
        collector.global_ops_snapshot(),
        capital_base_usdc=capital_base_usdc,
    )
    strategy_snapshots = collector.snapshot_by_strategy()
    grouped_records: dict[tuple[str, str], list[EvalRecord]] = defaultdict(list)
    for record in records:
        grouped_records[(record.strategy_id, record.strategy_version_id)].append(record)

    per_strategy = []
    for key in sorted(strategy_snapshots):
        snapshot = strategy_snapshots[key]
        strategy_records = grouped_records[key]
        per_strategy.append(
            {
                "strategy_id": key[0],
                "strategy_version_id": key[1],
                "record_count": len(strategy_records),
                "insufficient_samples": len(strategy_records) == 0,
                "brier_overall": snapshot.brier_overall,
                "baseline_brier_overall": snapshot.baseline_brier_overall,
                "brier_improvement_overall": snapshot.brier_improvement_overall,
                "pnl": snapshot.pnl,
                "fill_rate": snapshot.fill_rate,
                "slippage_bps": snapshot.slippage_bps,
                "drawdown": _max_drawdown(strategy_records),
            }
        )

    payload = dict(ops_view)
    payload["window_started_at"] = (
        None
        if window_started_at is None
        else _coerce_aware_datetime(window_started_at).isoformat()
    )
    payload["window_ended_at"] = (
        None
        if window_ended_at is None
        else _coerce_aware_datetime(window_ended_at).isoformat()
    )
    payload[FIRST_TRADE_TIME_SECONDS_METRIC] = first_trade_time_seconds
    payload.update(metrics_snapshot())
    payload["per_strategy"] = per_strategy
    payload["ops_view"] = ops_view
    payload["mark_to_market"] = (
        _mark_to_market_from_positions([])
        if mark_to_market is None
        else mark_to_market
    )
    payload["quote_calibration"] = _quote_calibration_payload(quote_snapshot)
    payload["quote_calibration"]["pnl_series"] = _quote_mtm_pnl_series(quote_records)
    payload["quote_calibration"]["max_drawdown_pct"] = _quote_mtm_max_drawdown_pct(
        quote_records,
        capital_base_usdc=capital_base_usdc,
    )
    payload["quality"] = _quality_payload(
        records=records,
        metrics_snapshot=collector.global_ops_snapshot(),
        mark_to_market=payload["mark_to_market"],
        quote_snapshot=quote_snapshot,
    )
    return payload


async def _first_trade_time_seconds(pg_pool: asyncpg.Pool | None) -> float | None:
    if pg_pool is None:
        return None

    async with pg_pool.acquire() as connection:
        fill_payloads_table_exists = await connection.fetchval(
            "SELECT to_regclass('public.fill_payloads') IS NOT NULL"
        )
        if not fill_payloads_table_exists:
            return None

        seconds = await connection.fetchval(
            """
            WITH first_fills AS (
                SELECT
                    decisions.decision_id,
                    EXTRACT(EPOCH FROM MIN(fills.ts) - decisions.created_at) AS elapsed_seconds
                FROM decisions
                INNER JOIN fill_payloads
                    ON fill_payloads.payload->>'decision_id' = decisions.decision_id
                INNER JOIN fills
                    ON fills.fill_id = fill_payloads.fill_id
                WHERE fills.ts >= decisions.created_at
                GROUP BY decisions.decision_id, decisions.created_at
            )
            SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY elapsed_seconds)
            FROM first_fills
            """
        )

    if seconds is None:
        return None
    return float(seconds)


def _metrics_aggregate_payload(
    records: list[EvalRecord],
    snapshot: MetricsSnapshot,
    *,
    capital_base_usdc: float | None = None,
) -> dict[str, Any]:
    payload = cast(dict[str, Any], _jsonable(snapshot))
    payload["brier_series"] = [
        {
            "recorded_at": record.recorded_at.isoformat(),
            "brier_score": record.brier_score,
        }
        for record in records
    ]
    payload["calibration_curve"] = [
        {
            "prob_estimate": record.prob_estimate,
            "resolved_outcome": record.resolved_outcome,
        }
        for record in records
    ]
    cumulative_pnl = Decimal("0")
    pnl_series: list[dict[str, Any]] = []
    for record in records:
        cumulative_pnl += Decimal(str(record.pnl))
        pnl_series.append(
            {
                "recorded_at": record.recorded_at.isoformat(),
                "pnl": float(cumulative_pnl),
            }
        )
    payload["pnl_series"] = pnl_series
    payload["max_drawdown_pct"] = _max_drawdown_pct(
        records,
        capital_base_usdc=capital_base_usdc,
    )
    payload["sharpe_ratio"] = _sharpe_ratio(_daily_pnl_values(records))
    return payload


def _max_drawdown(records: list[EvalRecord]) -> float:
    cumulative_pnl = Decimal("0")
    peak_equity = Decimal("0")
    max_drawdown = Decimal("0")
    for record in records:
        cumulative_pnl += Decimal(str(record.pnl))
        peak_equity = max(peak_equity, cumulative_pnl)
        max_drawdown = max(max_drawdown, peak_equity - cumulative_pnl)
    return float(max_drawdown)


def _max_drawdown_pct(
    records: list[EvalRecord],
    *,
    capital_base_usdc: float | None,
) -> float | None:
    if capital_base_usdc is None or capital_base_usdc <= 0.0:
        return None
    return _max_drawdown(records) / capital_base_usdc * 100.0


def _daily_pnl_values(records: list[EvalRecord]) -> list[float]:
    daily_pnl: defaultdict[date, Decimal] = defaultdict(lambda: Decimal("0"))
    for record in records:
        recorded_day = _coerce_aware_datetime(record.recorded_at).astimezone(UTC).date()
        daily_pnl[recorded_day] += Decimal(str(record.pnl))
    return [float(daily_pnl[day]) for day in sorted(daily_pnl)]


def _sharpe_ratio(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    if variance == 0.0:
        return mean if mean >= 0.0 else -abs(mean)
    return mean / sqrt(variance)


def _should_use_durable_decisions(runner: Runner) -> bool:
    store = runner.decision_store
    if not hasattr(store, "read_decisions") or not hasattr(store, "get_decision"):
        return False
    if isinstance(store, DecisionStore) and store.pool is None:
        return False
    return True


def _is_runner_running(runner: Runner) -> bool:
    return any(not task.done() for task in runner.tasks)


def _runner_workers_healthy(runner: Runner) -> bool:
    """True iff the runner is running and every spawned required worker is
    alive. Stays additive: `/status.running` semantics are untouched."""
    if not _is_runner_running(runner):
        return False
    snapshot = runner.worker_health_snapshot()
    return all(
        snapshot[name].state in ALIVE_WORKER_STATES
        for name in READINESS_REQUIRED_WORKERS
        if name in snapshot
    )


async def _ensure_runner_pool(runner: Runner) -> None:
    await runner.ensure_pg_pool()


async def _close_runner_pool(runner: Runner) -> None:
    await runner.close_pg_pool()


def _should_enforce_schema_check(settings: PMSSettings) -> bool:
    if settings.enforce_schema_check is not None:
        return settings.enforce_schema_check
    return settings.mode in {RunMode.PAPER, RunMode.LIVE}


def _current_subscription_asset_ids(runner: Runner) -> frozenset[str]:
    subscription_controller = runner.subscription_controller
    if (
        runner.state.runner_started_at is None
        or runner.state.mode == RunMode.BACKTEST
        or subscription_controller is None
        or not _is_runner_running(runner)
    ):
        return frozenset()
    return subscription_controller.current_asset_ids


def _request_metadata(request: Request) -> dict[str, object]:
    return {
        "request_method": request.method,
        "request_path": request.url.path,
        "request_token_id": getattr(request.state, "token_id", None),
    }


def _evaluation_report_payload(row: Any) -> EvaluationReportResponse:
    payload = dict(row)
    for key in ("ranked_strategies", "benchmark_rows", "warnings"):
        value = payload.get(key)
        if isinstance(value, str):
            try:
                payload[key] = json.loads(value)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=503, detail="Evaluation report JSON is invalid.") from exc
    return EvaluationReportResponse.model_validate(payload)


_APPROVAL_PREVIEW_KEYS = (
    "venue",
    "market_id",
    "token_id",
    "side",
    "outcome",
    "max_notional_usdc",
    "limit_price",
    "max_slippage_bps",
    "market_slug",
    "question",
)
_APPROVAL_MATCH_KEYS = tuple(name for name in _APPROVAL_PREVIEW_KEYS if name != "question")


def _pending_approval_events(path: Path) -> list[dict[str, Any]]:
    pending: dict[tuple[object, ...], dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return []
    for line in lines:
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        key = tuple(event.get(name) for name in _APPROVAL_MATCH_KEYS)
        event_name = event.get("event")
        if event_name == "approval_denied":
            pending[key] = {
                name: event.get(name)
                for name in _APPROVAL_PREVIEW_KEYS
                if name in event
            }
            pending[key]["status"] = "pending"
        elif event_name in {"approval_matched", "approval_consumed"}:
            pending.pop(key, None)
    return list(pending.values())


def _write_audit_fallback(request: Request, outcome: int, settings: PMSSettings) -> None:
    record = {
        "token_id": getattr(request.state, "token_id", None),
        "scope": getattr(request.state, "auth_scope", "unknown"),
        "method": request.method,
        "path": request.url.path,
        "outcome": outcome,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }
    logger.critical("request audit fallback: %s", record)
    try:
        emergency_path = Path(settings.live_emergency_audit_path).expanduser()
        fallback = Path(
            settings.live_request_audit_fallback_path
            or emergency_path.with_name("api-request-audit-fallback.jsonl")
        ).expanduser()
        fallback.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with fallback.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        logger.critical("request audit fallback file unavailable", exc_info=True)


async def _audit_degraded_response(response: Any) -> Any:
    """Add an explicit degradation marker without changing the status code."""
    body = getattr(response, "body", None)
    if body is None:
        iterator = getattr(response, "body_iterator", None)
        if iterator is not None:
            body = b"".join([chunk async for chunk in iterator])
    if not isinstance(body, (bytes, bytearray)):
        return response
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return response
    if not isinstance(payload, dict):
        return response
    payload["audit_degraded"] = True
    headers = dict(getattr(response, "headers", {}))
    headers.pop("content-length", None)
    return JSONResponse(status_code=response.status_code, content=payload, headers=headers)


def _actuator_status(runner: Runner, *, now: datetime) -> dict[str, Any]:
    active_halt = runner.actuator_executor.risk.active_halt()
    status: dict[str, Any] = {
        "fills_total": len(runner.state.fills),
        "mode": runner.state.mode.value,
        "halt_recovery_cycles_7d": len(
            runner.actuator_executor.risk.halt_recovery_cycles_since(
                now - timedelta(days=7)
            )
        ),
        "halted": active_halt is not None,
        "halt_reason": None if active_halt is None else active_halt.reason,
        "halt_trigger_kind": None if active_halt is None else active_halt.trigger_kind,
        "halt_triggered_at": (
            None if active_halt is None else _jsonable(active_halt.triggered_at)
        ),
    }
    return status


async def _runtime_continuity_status(runner: Runner) -> dict[str, Any] | None:
    if runner.pg_pool is None or runner.state.runtime_run_id is None:
        return None
    continuity = await RuntimeHeartbeatStore(runner.pg_pool).continuity(
        run_id=runner.state.runtime_run_id
    )
    if continuity is None:
        return {
            "run_id": runner.state.runtime_run_id,
            "source": "postgres_runtime_heartbeats",
            "first_observed_at": None,
            "last_observed_at": None,
            "heartbeat_count": 0,
            "healthy_days": 0,
            "max_gap_seconds": None,
            "unhealthy_heartbeat_count": 0,
            "min_controller_runtimes": 0,
        }
    return continuity.to_payload()


def _sensor_statuses(
    runner: Runner,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    current_time = now or datetime.now(tz=UTC)
    last_signal_at = _last_signal_at(runner.state.signals)
    last_signal_age_seconds = _signal_freshness_age_seconds(
        runner.state.signals,
        runner_started_at=runner.state.runner_started_at,
        now=current_time,
    )
    stale_after_seconds = runner.config.dashboard.stale_snapshot_threshold_s
    if not runner.active_sensors:
        return [
            {
                "name": "unknown",
                "status": "stopped"
                if runner.state.runner_started_at is None
                else "idle",
                "last_signal_at": last_signal_at,
                "last_signal_age_seconds": last_signal_age_seconds,
                "stale_after_seconds": stale_after_seconds,
                "task_done": True,
            }
        ]

    tasks = runner.sensor_stream.tasks
    return [
        {
            "name": sensor.__class__.__name__,
            "status": _sensor_status(
                sensor=sensor,
                task=tasks[index] if index < len(tasks) else None,
                last_signal_age_seconds=last_signal_age_seconds,
                stale_after_seconds=stale_after_seconds,
            ),
            "last_signal_at": last_signal_at,
            "last_signal_age_seconds": last_signal_age_seconds,
            "stale_after_seconds": stale_after_seconds,
            "task_done": tasks[index].done() if index < len(tasks) else True,
        }
        for index, sensor in enumerate(runner.active_sensors)
    ]


def _controller_diagnostic_counts(
    diagnostics: Sequence[Any],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for diagnostic in diagnostics:
        code = getattr(diagnostic, "code", None)
        if not isinstance(code, str) or not code:
            continue
        counts[code] += 1
    return dict(sorted(counts.items()))


def _last_signal_at(signals: Sequence[MarketSignal]) -> str | None:
    if not signals:
        return None
    return signals[-1].fetched_at.isoformat()


def _signal_freshness_age_seconds(
    signals: Sequence[MarketSignal],
    *,
    runner_started_at: datetime | None,
    now: datetime,
) -> float | None:
    if signals:
        signal_time = signals[-1].fetched_at
    elif runner_started_at is not None:
        signal_time = runner_started_at
    else:
        return None
    if signal_time.tzinfo is None:
        signal_time = signal_time.replace(tzinfo=UTC)
    return max(0.0, (now - signal_time).total_seconds())


def _sensor_status(
    *,
    sensor: object,
    task: asyncio.Task[None] | None,
    last_signal_age_seconds: float | None,
    stale_after_seconds: float,
) -> str:
    if task is None:
        return "idle"
    if task.cancelled():
        return "idle"
    if task.done():
        return "failed" if task.exception() is not None else "idle"
    if (
        _sensor_depends_on_signal_freshness(sensor)
        and last_signal_age_seconds is not None
        and last_signal_age_seconds > stale_after_seconds
    ):
        return "stale"
    return "running"


def _sensor_depends_on_signal_freshness(sensor: object) -> bool:
    name = sensor.__class__.__name__
    return name.startswith("MarketDataSensor") or callable(
        getattr(sensor, "update_subscription", None)
    )


def _start_alerting_if_configured(app_inst: FastAPI, runner: Runner) -> None:
    webhook = normalize_webhook_url(runner.config.discord.webhook_url)
    if webhook is None:
        return
    client = DiscordWebhookClient(webhook, alert_dir=runner.config.discord.alert_dir)
    stop_event = asyncio.Event()
    task = asyncio.create_task(
        run_alerting_subscription(runner.event_bus, client, stop_event=stop_event)
    )
    app_inst.state.discord_client = client
    app_inst.state.alerting_stop_event = stop_event
    app_inst.state.alerting_task = task
    eod_stop_event = asyncio.Event()
    eod_task = asyncio.create_task(EODScheduler(client).run(eod_stop_event))
    app_inst.state.eod_stop_event = eod_stop_event
    app_inst.state.eod_scheduler_task = eod_task


async def _stop_alerting_if_started(app_inst: FastAPI) -> None:
    first_exc: BaseException | None = None
    eod_stop_event = getattr(app_inst.state, "eod_stop_event", None)
    if isinstance(eod_stop_event, asyncio.Event):
        eod_stop_event.set()
    eod_task = getattr(app_inst.state, "eod_scheduler_task", None)
    if isinstance(eod_task, asyncio.Task):
        try:
            await asyncio.wait_for(eod_task, timeout=2.0)
        except TimeoutError:
            eod_task.cancel()
            with suppress(asyncio.CancelledError):
                await eod_task
        except Exception as exc:
            logger.exception("EOD scheduler shutdown failed")
            first_exc = first_exc or exc
    stop_event = getattr(app_inst.state, "alerting_stop_event", None)
    if isinstance(stop_event, asyncio.Event):
        stop_event.set()
    task = getattr(app_inst.state, "alerting_task", None)
    if isinstance(task, asyncio.Task):
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        except Exception as exc:
            logger.exception("Alert subscriber shutdown failed")
            first_exc = first_exc or exc
    client = getattr(app_inst.state, "discord_client", None)
    close = getattr(client, "aclose", None)
    if callable(close):
        try:
            await close()
        except Exception as exc:
            logger.exception("Discord client close failed")
            first_exc = first_exc or exc
    if first_exc is not None:
        raise first_exc


def _forecaster(decision: TradeDecision) -> str:
    return "unknown" if decision.model_id is None else decision.model_id


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
