from __future__ import annotations

import asyncio
import json
import inspect
import logging
from copy import deepcopy
from math import isfinite
from hashlib import sha256
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, cast, runtime_checkable

import asyncpg
import httpx
from pms.actuator.adapters.backtest import BacktestActuator
from pms.actuator.adapters.paper import PaperActuator, paper_orderbook_fill_failure
from pms.actuator.adapters.polymarket import (
    DenyFirstLiveOrderGate,
    FileFirstLiveOrderGate,
    LiveQuoteProvider,
    MissingLiveQuoteProvider,
    PolymarketActuator,
    PolymarketBookQuoteProvider,
    PolymarketDirectQuoteProvider,
    PolymarketRoutingQuoteProvider,
    PolymarketSDKClient,
    PolymarketSubmissionUnknownError,
    PolymarketVenueAccountReconciler,
)
from pms.actuator.executor import (
    ActuatorAdapter,
    ActuatorExecutor,
    is_terminal_partial_fill,
)
from pms.actuator.exit_monitor import (
    PositionExitMonitor,
    build_exit_decision,
    exit_key,
    mark_position_from_signal,
)
from pms.actuator.feedback import ActuatorFeedback
from pms.actuator.risk import RiskManager
from pms.config import PMSSettings, validate_live_mode_ready
from pms.controller._price_utils import best_ask
from pms.controller.baselines import (
    CategoryPriorBaselineEstimator,
    enrich_signal_with_category_prior,
    load_category_prior_observations_csv,
)
from pms.controller.diagnostics import ControllerDiagnostic
from pms.controller.factory import ControllerPipelineFactory
from pms.controller.factor_snapshot import (
    RAW_FACTOR_ROLES,
    PostgresFactorSnapshotReader,
)
from pms.controller.outcome_tokens import MarketDataOutcomeTokenResolver
from pms.controller.sizers.kelly import KellySizer
from pms.core.enums import OrderStatus, RunMode, Side, TimeInForce
from pms.core.exceptions import SensorDataQualityError
from pms.core.interfaces import (
    DiscoveryPollCompleteSensor,
    IController,
    ISensor,
    MarketSelectorLike,
    SubscriptionControllerLike,
    SubscriptionManagedSensor,
)
from pms.core.models import (
    EvalRecord,
    FillRecord,
    LiveTradingDisabledError,
    MarketSignal,
    Opportunity,
    OrderState,
    Portfolio,
    Position,
    ReconciliationReport as ReconciliationReport,
    TradeDecision,
    Venue,
    VenueAccountSnapshot as VenueAccountSnapshot,
    VenueCredentials,
)
from pms.artifact_path_safety import (
    require_path_outside_working_tree,
    require_private_parent,
)
from pms.evaluation.adapters.scoring import Scorer
from pms.evaluation.feedback import EvaluatorFeedback
from pms.evaluation.metrics import (
    MetricsCollector,
    StrategyMetricsSnapshot,
    StrategyVersionKey,
)
from pms.evaluation.quote_reader import TokenBookQuoteReader
from pms.evaluation.resolution import (
    GammaResolutionSource,
    ResolutionSweeper,
    ResolutionSweepResult,
)
from pms.evaluation.spool import EvalSpool
from pms.event_stream import RuntimeEventBus
from pms.execution.fees import market_fee_rate_from_metadata
from pms.factors.catalog import ensure_factor_catalog
from pms.factors.definitions import REGISTERED
from pms.factors.service import FactorService
from pms.live_preflight import (
    ensure_live_market_data_freshness,
    live_preflight_active_strategies_fingerprint,
    require_live_preflight_artifact,
    require_live_preflight_active_strategies_artifact,
)
from pms.market_selection import (
    MarketSelector,
    SensorSubscriptionController,
    UnionMergePolicy,
)
from pms.metrics import SELECTION_FUNNEL_TRADED_TOTAL_METRIC, increment_metric
from pms.redaction import redact_live_error_values
from pms.sensor.adapters.direct_book import (
    PolymarketPublicBookClient,
    RefreshingDirectBookSnapshotReader,
)
from pms.sensor.adapters.historical import HistoricalSensor
from pms.sensor.adapters.market_data import MarketDataSensor
from pms.sensor.adapters.market_discovery import MarketDiscoverySensor
from pms.sensor.stream import SensorStream, SignalSubscription
from pms.storage.dedup_store import InMemoryDedupStore, PgDedupStore
from pms.storage.decision_store import DecisionStore
from pms.storage.eval_store import EvalStore
from pms.storage.feedback_store import FeedbackStore
from pms.storage.fill_store import FillStore
from pms.storage.first_order_audit import JsonlFirstOrderAuditWriter
from pms.storage.live_emergency_audit import LiveEmergencyAuditWriter
from pms.storage.market_data_store import PostgresMarketDataStore
from pms.storage.market_resolution_store import MarketResolutionStore
from pms.storage.market_subscription_store import PostgresMarketSubscriptionStore
from pms.storage.opportunity_store import OpportunityStore
from pms.storage.order_store import OrderStore
from pms.storage.quote_eval_store import QuoteEvalStore
from pms.storage.runtime_heartbeat_store import RuntimeHeartbeatStore
from pms.storage.strategy_registry import PostgresStrategyRegistry
from pms.strategies.aggregate import Strategy
from pms.strategies.defaults import DEFAULT_STRATEGY_COMPOSITION
from pms.strategies.anchoring import (
    AnchoringAgent,
    AnchoringController,
    AnchoringLagStrategyModule,
    LiveAnchoringSource,
)
from pms.strategies.anchoring.source import (
    ANCHORING_LEGACY_VERSION_IDS,
    AnchoringMarketSnapshotReader,
    build_anchoring_strategy,
)
from pms.strategies.flb import FlbAgent, FlbController, FlbStrategyModule
from pms.strategies.flb.artifacts import (
    flb_calibration_provenance_path,
    require_flb_calibration_provenance_for_model,
)
from pms.strategies.flb.source import (
    FlbCalibrationModel,
    FlbMarketSnapshot,
    FlbMarketSnapshotReader,
    LiveFlbSource,
    load_flb_calibration_csv,
)
from pms.strategies.flb.projection import (
    H1_FLB_LEGACY_VERSION_IDS,
    H1_FLB_STRATEGY_ID,
    build_h1_flb_strategy,
)
from pms.strategies.intents import StrategyContext
from pms.strategies.paper_multifactor import (
    PAPER_MULTI_FACTOR_STRATEGY_ID,
    build_paper_multi_factor_strategy,
)
from pms.strategies.projections import (
    ActiveStrategy,
    CalibrationSpec,
    EvalSpec,
    FactorCompositionStep,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)
from pms.strategies.ripple import (
    LiveRippleSource,
    RippleAgent,
    RippleController,
    RippleMarketSnapshot,
    RippleStrategyModule,
)
from pms.strategies.ripple.source import (
    RIPPLE_LEGACY_VERSION_IDS,
    RippleFactorSnapshotReader,
    RippleMarketSnapshotReader,
    build_ripple_strategy,
)
from pms.strategies.runtime_bridge import StrategyRunResult
from pms.strategies.economics import strategy_economic_float
from pms.strategies.versioning import compute_strategy_version_id
from pms.supervision import (
    ALIVE_WORKER_STATES,
    WorkerFactory,
    WorkerHealth,
    WorkerSpec,
    WorkerSupervisor,
)


logger = logging.getLogger(__name__)

DEFAULT_BACKTEST_FIXTURE = Path("tests/fixtures/polymarket_7day_synthetic.jsonl")
RUNNER_STATE_LIMIT = 1000
CONTROLLER_SIGNAL_COOLESCE_LIMIT = 1_000
RESELECTION_INTERVAL_S = 300.0
DECISION_PENDING_TTL = timedelta(minutes=15)
DECISION_SWEEP_INTERVAL_S = 5.0
RUNTIME_HEARTBEAT_INTERVAL_S = 60.0
DEFAULT_OPERATIONAL_MARKET_SELECTION_HORIZON_DAYS = 90

# --- Worker supervision (see pms.supervision) -------------------------------
# Restart only on this transient set, within budget, with exponential
# backoff. Inner per-cycle catches stay the first line of defence; the
# supervisor is the last resort.
WORKER_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    asyncpg.PostgresError,
    OSError,
    TimeoutError,
    SensorDataQualityError,
)
WORKER_MAX_RESTARTS = 5
WORKER_RESTART_WINDOW_S = 600.0
WORKER_RESTART_BACKOFF_INITIAL_S = 1.0
WORKER_RESTART_BACKOFF_MAX_S = 60.0
WORKER_FACTOR_SERVICE = "factor_service"
WORKER_CONTROLLER_DISPATCHER = "controller_dispatcher"
WORKER_ACTUATOR = "actuator"
WORKER_DECISION_EXPIRY = "decision_expiry"
WORKER_RUNTIME_HEARTBEAT = "runtime_heartbeat"
# Workers the trading path cannot live without: used to tighten the
# heartbeat `running` flag. The factor service only counts when it was
# actually spawned (PostgreSQL runtime).
TRADING_REQUIRED_WORKERS: tuple[str, ...] = (
    WORKER_CONTROLLER_DISPATCHER,
    WORKER_ACTUATOR,
    WORKER_FACTOR_SERVICE,
)
# Readiness additionally requires the heartbeat writer when it was spawned
# (PAPER/LIVE with PostgreSQL).
READINESS_REQUIRED_WORKERS: tuple[str, ...] = (
    *TRADING_REQUIRED_WORKERS,
    WORKER_RUNTIME_HEARTBEAT,
)
REGISTERED_FACTOR_IDS = frozenset(factor_cls.factor_id for factor_cls in REGISTERED)
DEFAULT_V2_FACTOR_COMPOSITION = DEFAULT_STRATEGY_COMPOSITION
T = TypeVar("T")
PositionSlotKey = tuple[str, str | None, str, str, str, str]
PositionExitReentryKey = tuple[str, str, str, str | None, str | None]


class LivePersistenceFailureError(RuntimeError):
    """Raised after a venue submit when LIVE persistence safety is broken."""


ControllerReleaseCancelPoint = Literal[
    "before_first_cleanup_await",
    "between_cleanup_awaits",
    "after_last_cleanup_await",
]


@runtime_checkable
class AsyncCloseable(Protocol):
    async def aclose(self) -> None: ...


@runtime_checkable
class OpportunityAwareController(Protocol):
    async def on_signal(
        self,
        signal: MarketSignal,
        portfolio: Portfolio | None = None,
    ) -> tuple[Opportunity, TradeDecision] | None: ...


@dataclass(frozen=True)
class StrategyControllerRuntime:
    strategy_id: str
    strategy_version_id: str
    controller: IController
    asset_ids: frozenset[str] | None


@dataclass
class DetachedControllerRuntime:
    strategy_id: str
    runtime: StrategyControllerRuntime | None
    queue: asyncio.Queue[MarketSignal] | None
    task: asyncio.Task[None] | None


def _runtime_calibration_add_samples(
    runtime: StrategyControllerRuntime,
) -> Callable[[str, list[EvalRecord]], None] | None:
    # Duck-typed like pipeline._resolved_sample_count: extending the
    # ICalibrator Protocol would break test fakes that satisfy only
    # `calibrate`/`sample_count`.
    calibrator = getattr(runtime.controller, "calibrator", None)
    add_samples = getattr(calibrator, "add_samples", None)
    if not callable(add_samples):
        return None
    return cast(Callable[[str, list[EvalRecord]], None], add_samples)


@dataclass(frozen=True)
class ActuatorWorkItem:
    decision: TradeDecision
    signal: MarketSignal | None
    dedup_acquired: bool = False
    decision_evidence: Mapping[str, object] = field(default_factory=dict)
    paper_orderbook: Mapping[str, Any] | None = None

    def __iter__(self) -> Iterator[TradeDecision | MarketSignal | None]:
        yield self.decision
        yield self.signal


class VenueAccountReconciler(Protocol):
    async def snapshot(self, credentials: VenueCredentials) -> VenueAccountSnapshot: ...

    async def compare(
        self,
        db_portfolio: Portfolio,
        venue_snapshot: VenueAccountSnapshot,
    ) -> ReconciliationReport: ...


@dataclass
class RunnerState:
    mode: RunMode
    runner_started_at: datetime | None = None
    runtime_run_id: str | None = None
    signals: list[MarketSignal] = field(default_factory=list)
    opportunities: list[Opportunity] = field(default_factory=list)
    decisions: list[TradeDecision] = field(default_factory=list)
    orders: list[OrderState] = field(default_factory=list)
    fills: list[FillRecord] = field(default_factory=list)
    controller_diagnostics: list[ControllerDiagnostic] = field(default_factory=list)


@dataclass
class Runner:
    config: PMSSettings = field(default_factory=PMSSettings)
    historical_data_path: Path = DEFAULT_BACKTEST_FIXTURE
    sensors: Sequence[ISensor] | None = None
    eval_store: EvalStore = field(default_factory=EvalStore)
    feedback_store: FeedbackStore = field(default_factory=FeedbackStore)
    decision_store: DecisionStore = field(default_factory=DecisionStore)
    order_store: OrderStore = field(default_factory=OrderStore)
    fill_store: FillStore = field(default_factory=FillStore)
    quote_eval_store: QuoteEvalStore = field(default_factory=QuoteEvalStore)
    opportunity_store: OpportunityStore = field(default_factory=OpportunityStore)
    venue_account_reconciler: VenueAccountReconciler | None = None
    portfolio: Portfolio = field(default_factory=lambda: Portfolio(
        total_usdc=1000.0,
        free_usdc=1000.0,
        locked_usdc=0.0,
        open_positions=[],
    ))
    sensor_stream: SensorStream = field(default_factory=SensorStream)
    event_bus: RuntimeEventBus = field(default_factory=RuntimeEventBus)
    controller: IController | None = None
    state: RunnerState = field(init=False)
    actuator_executor: ActuatorExecutor = field(init=False)
    _evaluator_spool: EvalSpool = field(init=False)
    _decision_queue: asyncio.Queue[ActuatorWorkItem] = field(
        init=False,
    )
    _stop_event: asyncio.Event = field(init=False)
    _controller_task: asyncio.Task[None] | None = field(init=False, default=None)
    _actuator_task: asyncio.Task[None] | None = field(init=False, default=None)
    _factor_service: FactorService | None = field(init=False, default=None)
    _factor_service_fresh: bool = field(init=False, default=False)
    _factor_service_task: asyncio.Task[None] | None = field(init=False, default=None)
    _supervisor: WorkerSupervisor = field(init=False)
    _task: asyncio.Task[None] | None = field(init=False, default=None)
    _pg_pool: asyncpg.Pool | None = field(init=False, default=None)
    _owns_pg_pool: bool = field(init=False, default=False)
    _decision_expiry_task: asyncio.Task[None] | None = field(init=False, default=None)
    _runtime_heartbeat_task: asyncio.Task[None] | None = field(
        init=False,
        default=None,
    )
    _resolution_sweep_task: asyncio.Task[None] | None = field(
        init=False,
        default=None,
    )
    _resolution_sweeps_total: int = field(init=False, default=0)
    _resolution_fills_resolved_total: int = field(init=False, default=0)
    _resolution_sweep_failures_total: int = field(init=False, default=0)
    _market_selector: MarketSelectorLike | None = field(init=False, default=None)
    _subscription_controller: SubscriptionControllerLike | None = field(
        init=False,
        default=None,
    )
    _strategy_registry: PostgresStrategyRegistry | None = field(
        init=False,
        default=None,
    )
    _controller_factory: ControllerPipelineFactory = field(init=False)
    _controller_runtimes: dict[str, StrategyControllerRuntime] = field(
        init=False,
        default_factory=dict,
    )
    _active_strategies_by_version: dict[tuple[str, str], ActiveStrategy] = field(
        init=False,
        default_factory=dict,
    )
    _controller_signal_queues: dict[str, asyncio.Queue[MarketSignal]] = field(
        init=False,
        default_factory=dict,
    )
    _controller_pipeline_tasks: dict[str, asyncio.Task[None]] = field(
        init=False,
        default_factory=dict,
    )
    _controller_pipeline_error: BaseException | None = field(
        init=False,
        default=None,
    )
    _controller_lifecycle_lock: asyncio.Lock = field(init=False)
    _controller_release_cancel_point: ControllerReleaseCancelPoint | None = field(
        init=False,
        default=None,
    )
    _reselection_task: asyncio.Task[None] | None = field(init=False, default=None)
    _reselection_lock: asyncio.Lock = field(init=False)
    _market_selection_lock: asyncio.Lock = field(init=False)
    _reselection_generation: int = field(init=False, default=0)
    _reselection_requested: asyncio.Event = field(init=False)
    _strategy_change_callbacks_registered: bool = field(init=False, default=False)
    _active_sensors: tuple[ISensor, ...] = field(init=False, default=())
    _paper_orderbooks: dict[str, dict[str, Any]] = field(
        init=False, default_factory=dict
    )
    _latest_signal_by_token: dict[str, MarketSignal] = field(
        init=False, default_factory=dict
    )
    _position_exit_monitor: PositionExitMonitor = field(init=False)
    _emitted_position_exit_keys: set[tuple[str, str, str, str | None, str]] = field(
        init=False,
        default_factory=set,
    )
    _position_exit_keys_by_decision: dict[
        str, tuple[str, str, str, str | None, str]
    ] = field(init=False, default_factory=dict)
    _pending_open_position_keys_by_decision: dict[str, PositionSlotKey] = field(
        init=False,
        default_factory=dict,
    )
    _position_exit_reentry_quarantine_until: dict[
        PositionExitReentryKey, datetime
    ] = field(init=False, default_factory=dict)
    _category_prior_estimator: CategoryPriorBaselineEstimator | None = field(
        init=False,
        default=None,
    )
    _flb_calibration_model: FlbCalibrationModel | None = field(
        init=False,
        default=None,
    )
    _live_trading_suspended_reason: str | None = field(init=False, default=None)
    _live_preflight_artifact_validated: bool = field(init=False, default=False)
    _live_preflight_active_strategies_fingerprint: str | None = field(
        init=False,
        default=None,
    )

    def __post_init__(self) -> None:
        if self.config.mode == RunMode.LIVE:
            self.portfolio = _portfolio_with_configured_budget(
                self.portfolio,
                total_budget_usdc=self.config.risk.max_total_exposure,
            )
        self.state = RunnerState(mode=self.config.mode)
        self._controller_factory = ControllerPipelineFactory(settings=self.config)
        self._evaluator_spool = EvalSpool(
            store=self.eval_store,
            scorer=Scorer(),
            feedback_generator=EvaluatorFeedback(self.feedback_store),
            metrics_provider=self._metrics_by_strategy,
            quote_store=self.quote_eval_store,
            calibration_sink=self._on_eval_record_for_calibration,
        )
        self._decision_queue = asyncio.Queue()
        self._stop_event = asyncio.Event()
        self._controller_lifecycle_lock = asyncio.Lock()
        self._reselection_lock = asyncio.Lock()
        self._market_selection_lock = asyncio.Lock()
        self._reselection_requested = asyncio.Event()
        self._position_exit_monitor = PositionExitMonitor(self.config.position_exit)
        self._category_prior_estimator = _category_prior_estimator_from_settings(
            self.config
        )
        self._flb_calibration_model = _flb_calibration_model_from_settings(self.config)
        self._supervisor = WorkerSupervisor(
            event_publisher=self._publish_worker_event,
        )
        self.actuator_executor = self._build_executor(self.config.mode)

    @property
    def controller_task(self) -> asyncio.Task[None] | None:
        return self._controller_task

    @property
    def actuator_task(self) -> asyncio.Task[None] | None:
        return self._actuator_task

    @property
    def controller_pipeline_tasks(self) -> tuple[asyncio.Task[None], ...]:
        return tuple(self._controller_pipeline_tasks.values())

    @property
    def factor_service_task(self) -> asyncio.Task[None] | None:
        return self._factor_service_task

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    @property
    def pg_pool(self) -> asyncpg.Pool | None:
        return self._pg_pool

    async def enqueue_accepted_decision(self, decision: TradeDecision) -> bool:
        return await self._enqueue_decision(
            decision,
            signal=None,
            dedup_acquired=True,
        )

    async def ensure_pg_pool(self) -> None:
        if self._pg_pool is not None:
            return
        self._pg_pool = await asyncpg.create_pool(
            dsn=self.config.database.dsn,
            min_size=self.config.database.pool_min_size,
            max_size=self.config.database.pool_max_size,
        )
        self._owns_pg_pool = True
        self._bind_runtime_stores()

    def bind_pg_pool(self, pool: asyncpg.Pool) -> None:
        self._pg_pool = pool
        self._owns_pg_pool = False
        self._bind_runtime_stores()

    async def close_pg_pool(self) -> None:
        if self._pg_pool is None:
            return
        pool = self._pg_pool
        owns_pool = self._owns_pg_pool
        self._pg_pool = None
        self._owns_pg_pool = False
        self._unbind_runtime_stores()
        if owns_pool:
            await pool.close()

    @property
    def evaluator_task(self) -> asyncio.Task[None] | None:
        return self._evaluator_spool._task

    @property
    def resolution_sweeps_total(self) -> int:
        return self._resolution_sweeps_total

    @property
    def resolution_fills_resolved_total(self) -> int:
        return self._resolution_fills_resolved_total

    @property
    def resolution_sweep_failures_total(self) -> int:
        return self._resolution_sweep_failures_total

    @property
    def active_sensors(self) -> tuple[ISensor, ...]:
        return self._active_sensors

    @property
    def subscription_controller(self) -> SubscriptionControllerLike | None:
        return self._subscription_controller

    @property
    def controller_runtime_count(self) -> int:
        return len(self._controller_runtimes)

    @property
    def strategy_registry(self) -> PostgresStrategyRegistry | None:
        return self._strategy_registry

    @property
    def tasks(self) -> tuple[asyncio.Task[None], ...]:
        tasks: list[asyncio.Task[None]] = []
        tasks.extend(self.sensor_stream.tasks)
        if self._factor_service_task is not None:
            tasks.append(self._factor_service_task)
        if self._controller_task is not None:
            tasks.append(self._controller_task)
        tasks.extend(self._controller_pipeline_tasks.values())
        if self._actuator_task is not None:
            tasks.append(self._actuator_task)
        if self._decision_expiry_task is not None:
            tasks.append(self._decision_expiry_task)
        if self._runtime_heartbeat_task is not None:
            tasks.append(self._runtime_heartbeat_task)
        if self._resolution_sweep_task is not None:
            tasks.append(self._resolution_sweep_task)
        if self._reselection_task is not None:
            tasks.append(self._reselection_task)
        if self.evaluator_task is not None:
            tasks.append(self.evaluator_task)
        return tuple(tasks)

    @property
    def live_trading_suspended(self) -> bool:
        return self._live_trading_suspended_reason is not None

    async def start(self) -> None:
        if any(not task.done() for task in self.tasks):
            msg = "Runner is already started"
            raise RuntimeError(msg)

        if self.config.mode == RunMode.LIVE:
            validate_live_mode_ready(self.config)
            require_live_preflight_artifact(self.config)
            self._live_preflight_artifact_validated = True

        self._stop_event.clear()
        self._controller_pipeline_error = None
        self._live_trading_suspended_reason = None
        self._position_exit_monitor = PositionExitMonitor(self.config.position_exit)
        self._emitted_position_exit_keys.clear()
        self._position_exit_keys_by_decision.clear()
        self._pending_open_position_keys_by_decision.clear()
        self._position_exit_reentry_quarantine_until.clear()
        started_at = datetime.now(tz=UTC)
        self.state = RunnerState(
            mode=self.config.mode,
            runner_started_at=started_at,
            runtime_run_id=_runtime_run_id(mode=self.config.mode, started_at=started_at),
        )
        self._paper_orderbooks.clear()
        self._latest_signal_by_token.clear()
        self._resolution_sweeps_total = 0
        self._resolution_fills_resolved_total = 0
        self._resolution_sweep_failures_total = 0
        # Fresh supervisor per run: stop() latches its no-respawn flag, so a
        # restarted runner needs a new instance with clean health state.
        self._supervisor = WorkerSupervisor(
            event_publisher=self._publish_worker_event,
        )

        try:
            self._assert_no_legacy_jsonl_paths()
            if self._should_boot_postgres_runtime():
                await self.ensure_pg_pool()
                if self._pg_pool is None:
                    msg = "Runner PostgreSQL pool is not initialized"
                    raise RuntimeError(msg)
                self._strategy_registry = PostgresStrategyRegistry(self._pg_pool)
                # Test/runtime registry doubles retain their legacy constructor;
                # the concrete registry exposes this optional versioning hook.
                setattr(
                    self._strategy_registry,
                    "_llm_identity",
                    (
                        self.config.llm.provider,
                        self.config.llm.model,
                        self.config.llm.prompt_version,
                    ),
                )
                if self.config.mode == RunMode.LIVE:
                    self._live_preflight_active_strategies_fingerprint = (
                        await require_live_preflight_active_strategies_artifact(
                            self.config,
                            self._strategy_registry,
                        )
                    )
                await ensure_factor_catalog(self._pg_pool)
                if self.config.mode == RunMode.LIVE:
                    await self._assert_no_unresolved_submission_unknown_incidents()
                    await self._assert_live_market_data_freshness()
                await self._ensure_default_v2_version()
                await self._ensure_paper_soak_strategy()
                if self._pg_pool is None:
                    msg = "Runner PostgreSQL pool is not initialized"
                    raise RuntimeError(msg)
                self._factor_service = self._build_factor_service()
                self._factor_service_fresh = True
                self._factor_service_task = self._supervisor.spawn(
                    self._worker_spec(
                        WORKER_FACTOR_SERVICE,
                        self._run_factor_service_worker,
                    )
                )
                # Rebuild the portfolio from persisted fills BEFORE sensors
                # start producing signals. Without this, a restart in LIVE
                # mode would forget open Polymarket exposure: every new BUY
                # would size against the hardcoded `Portfolio(free_usdc=…)`
                # default while the venue actually holds real exposure. See
                # also `_reconcile_portfolio_from_db`.
                await self._reconcile_portfolio_from_db()
                if self.config.mode == RunMode.LIVE:
                    await self._reconcile_venue_account()
            self._active_sensors = self._build_sensors()
            self._wire_active_perception(self._active_sensors)
            await self._configure_controllers()
            await self.sensor_stream.start(
                self._active_sensors,
                task_factory=self._supervised_sensor_task,
            )
            await self._evaluator_spool.start()
            self._controller_task = self._supervisor.spawn(
                self._worker_spec(
                    WORKER_CONTROLLER_DISPATCHER,
                    self._controller_loop,
                )
            )
            self._actuator_task = self._supervisor.spawn(
                self._worker_spec(WORKER_ACTUATOR, self._actuator_loop)
            )
            if self._pg_pool is not None:
                self._decision_expiry_task = self._supervisor.spawn(
                    self._worker_spec(
                        WORKER_DECISION_EXPIRY,
                        self._decision_expiry_loop,
                    )
                )
            if self._pg_pool is not None and self.config.mode in {RunMode.PAPER, RunMode.LIVE}:
                self._runtime_heartbeat_task = self._supervisor.spawn(
                    self._worker_spec(
                        WORKER_RUNTIME_HEARTBEAT,
                        self._runtime_heartbeat_loop,
                    )
                )
                self._resolution_sweep_task = asyncio.create_task(
                    self._resolution_sweep_loop()
                )
        except Exception:
            self._live_preflight_artifact_validated = False
            self._live_preflight_active_strategies_fingerprint = None
            await self._cleanup_after_start_failure()
            raise

    async def stop(self, *, close_pool: bool = True) -> None:
        # Block respawns before any cancellation so a supervised restart
        # cannot race the teardown ordering below.
        self._supervisor.stop()
        self._stop_event.set()
        error: BaseException | None = None

        factor_task = self._factor_service_task
        if factor_task is not None and not factor_task.done():
            factor_task.cancel()
        decision_expiry_task = self._decision_expiry_task
        if decision_expiry_task is not None and not decision_expiry_task.done():
            decision_expiry_task.cancel()
        runtime_heartbeat_task = self._runtime_heartbeat_task
        if runtime_heartbeat_task is not None and not runtime_heartbeat_task.done():
            runtime_heartbeat_task.cancel()
        resolution_sweep_task = self._resolution_sweep_task
        if resolution_sweep_task is not None and not resolution_sweep_task.done():
            resolution_sweep_task.cancel()
        reselection_task = self._reselection_task
        self._reselection_requested.set()
        self._clear_discovery_poll_complete_hook()
        self._unregister_strategy_change_callbacks()
        if reselection_task is not None and not reselection_task.done():
            reselection_task.cancel()

        try:
            await self.sensor_stream.stop()
        except BaseException as exc:  # pragma: no cover - exercised via unit tests
            error = exc

        for task in (self._controller_task, *self._controller_pipeline_tasks.values(), self._actuator_task):
            if task is not None and not task.done():
                task.cancel()

        try:
            await asyncio.gather(
                *(
                    task
                    for task in (
                        factor_task,
                        self._controller_task,
                        *self._controller_pipeline_tasks.values(),
                        self._actuator_task,
                        decision_expiry_task,
                        runtime_heartbeat_task,
                        resolution_sweep_task,
                    )
                    if task
                ),
                return_exceptions=True,
            )
        except BaseException as exc:  # pragma: no cover - defensive
            if error is None:
                error = exc

        try:
            await self._evaluator_spool.stop()
        except BaseException as exc:  # pragma: no cover - defensive
            if error is None:
                error = exc

        try:
            await self._close_active_sensors()
        except BaseException as exc:  # pragma: no cover - defensive
            if error is None:
                error = exc

        try:
            await self._await_cancelled_task(reselection_task)
        except BaseException as exc:  # pragma: no cover - defensive
            if error is None:
                error = exc

        self._factor_service = None
        self._factor_service_task = None
        self._controller_task = None
        self._actuator_task = None
        self._decision_expiry_task = None
        self._runtime_heartbeat_task = None
        self._resolution_sweep_task = None
        self._market_selector = None
        self._subscription_controller = None
        self._strategy_registry = None
        self._controller_runtimes = {}
        self._active_strategies_by_version = {}
        self._controller_signal_queues = {}
        self._controller_pipeline_tasks = {}
        self._controller_release_cancel_point = None
        self._reselection_task = None
        self._live_preflight_artifact_validated = False
        self._live_preflight_active_strategies_fingerprint = None

        if close_pool:
            try:
                await self._close_pg_pool()
            except BaseException as exc:  # pragma: no cover - defensive
                if error is None:
                    error = exc

        if error is not None:
            raise error

    async def run(self) -> None:
        self._task = asyncio.current_task()
        try:
            await self.start()
            await self.wait_until_idle()
        finally:
            try:
                await self.stop()
            finally:
                self._task = None

    def switch_mode(self, new_mode: RunMode) -> None:
        self._live_preflight_artifact_validated = False
        self._live_preflight_active_strategies_fingerprint = None
        self.config.mode = new_mode
        self.state.mode = new_mode
        if new_mode == RunMode.LIVE:
            self.portfolio = _portfolio_with_configured_budget(
                self.portfolio,
                total_budget_usdc=self.config.risk.max_total_exposure,
            )
        self.actuator_executor = self._build_executor(new_mode)

    async def wait_until_idle(self) -> None:
        if self.sensor_stream.tasks:
            await asyncio.gather(*self.sensor_stream.tasks, return_exceptions=True)
        await self.sensor_stream.queue.join()
        if self._factor_service_task is not None:
            await self._factor_service_task
        if self._controller_task is not None:
            await self._controller_task
        controller_queues = tuple(self._controller_signal_queues.values())
        for queue in controller_queues:
            await queue.join()
        controller_tasks = tuple(self._controller_pipeline_tasks.values())
        if controller_tasks:
            await asyncio.gather(*controller_tasks, return_exceptions=True)
        if self._controller_pipeline_error is not None:
            raise self._controller_pipeline_error
        await self._decision_queue.join()
        if self._actuator_task is not None:
            await self._actuator_task
        if (
            self._decision_expiry_task is not None
            and self.config.mode == RunMode.BACKTEST
        ):
            await self._decision_expiry_task
        await self._evaluator_spool.join()

    async def wait_for_signals(self, count: int) -> None:
        while len(self.state.signals) < count:
            await asyncio.sleep(0.1)

    def build_flb_strategy_module(
        self,
        *,
        strategy_id: str,
        strategy_version_id: str,
        market_ids: Sequence[str],
        market_reader: FlbMarketSnapshotReader | None = None,
    ) -> FlbStrategyModule:
        if not market_ids:
            msg = "market_ids must not be empty"
            raise ValueError(msg)
        reader = market_reader
        if reader is None:
            if self._pg_pool is None:
                msg = (
                    "Runner PostgreSQL pool is required to build the production "
                    "FLB market reader"
                )
                raise RuntimeError(msg)
            reader = _PostgresFlbMarketSnapshotReader(
                PostgresMarketDataStore(self._pg_pool)
            )
        h1_strategy: Strategy | None = None
        if strategy_id == H1_FLB_STRATEGY_ID:
            calibration_model = self._flb_calibration_model
            h1_strategy = (
                _h1_flb_strategy_from_settings(
                    self.config,
                    calibration_model=calibration_model,
                )
                if calibration_model is not None
                else build_h1_flb_strategy(
                    time_in_force=self.config.controller.time_in_force,
                )
            )
        if h1_strategy is not None:
            _verify_runtime_strategy_version(
                h1_strategy,
                strategy_version_id,
                legacy_ids=H1_FLB_LEGACY_VERSION_IDS,
            )
        position_sizing_risk = self.config.risk
        if h1_strategy is not None:
            position_sizing_risk = self.config.risk.model_copy(
                update={
                    "max_position_per_market": (
                        h1_strategy.risk.max_position_notional_usdc
                    )
                }
            )
        h1_metadata = dict(h1_strategy.config.metadata) if h1_strategy else {}

        return FlbStrategyModule(
            source=LiveFlbSource(
                market_ids=tuple(market_ids),
                market_reader=reader,
                position_sizer=KellySizer(risk=position_sizing_risk),
                portfolio=self.portfolio,
                max_slippage_bps=(
                    int(h1_metadata["max_slippage_bps"])
                    if h1_strategy is not None
                    else self.config.controller.max_slippage_bps
                ),
                entry_execution_cost_bps=(
                    float(h1_metadata["entry_execution_cost_bps"])
                    if h1_strategy is not None
                    else self.config.strategies.flb_entry_execution_cost_bps
                ),
                fee_rate=(
                    float(h1_metadata["fee_rate"])
                    if h1_strategy is not None
                    else self.config.strategies.flb_fee_rate
                ),
                longshot_yes_threshold=(
                    float(h1_metadata["longshot_yes_threshold"])
                    if h1_strategy is not None
                    else 0.10
                ),
                favorite_yes_threshold=(
                    float(h1_metadata["favorite_yes_threshold"])
                    if h1_strategy is not None
                    else 0.90
                ),
                min_expected_edge=(
                    float(h1_metadata["min_expected_edge"])
                    if h1_strategy is not None
                    else 0.02
                ),
                confidence=(
                    float(h1_metadata["source_confidence"])
                    if h1_strategy is not None
                    else 0.65
                ),
                time_in_force=(
                    TimeInForce(h1_metadata["time_in_force"])
                    if h1_strategy is not None
                    else TimeInForce(self.config.controller.time_in_force.upper())
                ),
                calibration_model=self._flb_calibration_model,
            ),
            controller=FlbController(),
            agent=FlbAgent(
                min_confidence=(
                    float(h1_metadata["min_confidence"])
                    if h1_strategy is not None
                    else 0.6
                ),
                min_expected_edge=(
                    float(h1_metadata["min_expected_edge"])
                    if h1_strategy is not None
                    else 0.02
                ),
            ),
            strategy_id=strategy_id,
            strategy_version_id=strategy_version_id,
        )

    async def run_flb_strategy_once(
        self,
        *,
        strategy_id: str,
        strategy_version_id: str,
        market_ids: Sequence[str],
        as_of: datetime | None = None,
        market_reader: FlbMarketSnapshotReader | None = None,
    ) -> Sequence[StrategyRunResult]:
        module = self.build_flb_strategy_module(
            strategy_id=strategy_id,
            strategy_version_id=strategy_version_id,
            market_ids=market_ids,
            market_reader=market_reader,
        )
        return tuple(
            await module.run_with_artifacts(
                StrategyContext(
                    strategy_id=strategy_id,
                    strategy_version_id=strategy_version_id,
                    as_of=as_of or datetime.now(tz=UTC),
                )
            )
        )

    def build_ripple_strategy_module(
        self,
        *,
        strategy_id: str,
        strategy_version_id: str,
        market_ids: Sequence[str],
        factor_reader: RippleFactorSnapshotReader | None = None,
        market_reader: RippleMarketSnapshotReader | None = None,
    ) -> RippleStrategyModule:
        if not market_ids:
            msg = "market_ids must not be empty"
            raise ValueError(msg)
        resolved_factor_reader: RippleFactorSnapshotReader
        if factor_reader is None:
            if self._pg_pool is None:
                msg = (
                    "Runner PostgreSQL pool is required to build the production "
                    "Ripple factor reader"
                )
                raise RuntimeError(msg)
            resolved_factor_reader = cast(
                RippleFactorSnapshotReader,
                PostgresFactorSnapshotReader(self._pg_pool),
            )
        else:
            resolved_factor_reader = factor_reader
        resolved_market_reader: RippleMarketSnapshotReader
        if market_reader is None:
            if self._pg_pool is None:
                msg = (
                    "Runner PostgreSQL pool is required to build the production "
                    "Ripple market reader"
                )
                raise RuntimeError(msg)
            resolved_market_reader = _PostgresRippleMarketSnapshotReader(
                PostgresMarketDataStore(self._pg_pool)
            )
        else:
            resolved_market_reader = market_reader

        ripple_strategy = build_ripple_strategy(
            strategy_id,
            time_in_force=self.config.controller.time_in_force,
        )
        _verify_runtime_strategy_version(
            ripple_strategy,
            strategy_version_id,
            legacy_ids=RIPPLE_LEGACY_VERSION_IDS,
        )
        ripple_metadata = dict(ripple_strategy.config.metadata)

        return RippleStrategyModule(
            source=LiveRippleSource(
                market_ids=tuple(market_ids),
                factor_reader=resolved_factor_reader,
                market_reader=resolved_market_reader,
                position_sizer=KellySizer(
                    risk=self.config.risk.model_copy(
                        update={
                            "max_position_per_market": (
                                ripple_strategy.risk.max_position_notional_usdc
                            )
                        }
                    ),
                    fraction=Decimal("0.25"),
                ),
                portfolio=self.portfolio,
                prior_strength=float(ripple_metadata["prior_strength"]),
                min_expected_edge=float(ripple_metadata["min_expected_edge"]),
                max_slippage_bps=int(ripple_metadata["max_slippage_bps"]),
                time_in_force=TimeInForce(ripple_metadata["time_in_force"]),
            ),
            controller=RippleController(),
            agent=RippleAgent(min_confidence=float(ripple_metadata["min_confidence"])),
            strategy_id=strategy_id,
            strategy_version_id=strategy_version_id,
        )

    async def run_ripple_strategy_once(
        self,
        *,
        strategy_id: str,
        strategy_version_id: str,
        market_ids: Sequence[str],
        as_of: datetime | None = None,
        factor_reader: RippleFactorSnapshotReader | None = None,
        market_reader: RippleMarketSnapshotReader | None = None,
    ) -> Sequence[StrategyRunResult]:
        module = self.build_ripple_strategy_module(
            strategy_id=strategy_id,
            strategy_version_id=strategy_version_id,
            market_ids=market_ids,
            factor_reader=factor_reader,
            market_reader=market_reader,
        )
        return tuple(
            await module.run_with_artifacts(
                StrategyContext(
                    strategy_id=strategy_id,
                    strategy_version_id=strategy_version_id,
                    as_of=as_of or datetime.now(tz=UTC),
                )
            )
        )

    def build_anchoring_strategy_module(
        self,
        *,
        strategy_id: str,
        strategy_version_id: str,
        market_ids: Sequence[str],
        market_reader: AnchoringMarketSnapshotReader | None = None,
    ) -> AnchoringLagStrategyModule:
        if not market_ids:
            msg = "market_ids must not be empty"
            raise ValueError(msg)
        if market_reader is None:
            msg = (
                "explicit H2 anchoring market reader is required until the "
                "LLM/news observation source is wired"
            )
            raise RuntimeError(msg)

        anchoring_strategy = build_anchoring_strategy(
            strategy_id,
            time_in_force=self.config.controller.time_in_force,
        )
        _verify_runtime_strategy_version(
            anchoring_strategy,
            strategy_version_id,
            legacy_ids=ANCHORING_LEGACY_VERSION_IDS,
        )
        anchoring_metadata = dict(anchoring_strategy.config.metadata)

        return AnchoringLagStrategyModule(
            source=LiveAnchoringSource(
                market_ids=tuple(market_ids),
                market_reader=market_reader,
                position_sizer=KellySizer(
                    self.config.risk.model_copy(
                        update={
                            "max_position_per_market": (
                                anchoring_strategy.risk.max_position_notional_usdc
                            )
                        }
                    )
                ),
                portfolio=self.portfolio,
                min_divergence=float(anchoring_metadata["min_divergence"]),
                min_confidence=float(anchoring_metadata["min_confidence"]),
                min_expected_edge=float(anchoring_metadata["min_expected_edge"]),
                max_slippage_bps=int(anchoring_metadata["max_slippage_bps"]),
                time_in_force=TimeInForce(anchoring_metadata["time_in_force"]),
            ),
            controller=AnchoringController(),
            agent=AnchoringAgent(
                min_confidence=float(anchoring_metadata["min_confidence"]),
                min_expected_edge=float(anchoring_metadata["min_expected_edge"]),
                min_divergence=float(anchoring_metadata["min_divergence"]),
            ),
            strategy_id=strategy_id,
            strategy_version_id=strategy_version_id,
        )

    async def run_anchoring_strategy_once(
        self,
        *,
        strategy_id: str,
        strategy_version_id: str,
        market_ids: Sequence[str],
        as_of: datetime | None = None,
        market_reader: AnchoringMarketSnapshotReader | None = None,
    ) -> Sequence[StrategyRunResult]:
        module = self.build_anchoring_strategy_module(
            strategy_id=strategy_id,
            strategy_version_id=strategy_version_id,
            market_ids=market_ids,
            market_reader=market_reader,
        )
        return tuple(
            await module.run_with_artifacts(
                StrategyContext(
                    strategy_id=strategy_id,
                    strategy_version_id=strategy_version_id,
                    as_of=as_of or datetime.now(tz=UTC),
                )
            )
        )

    def _build_sensors(self) -> tuple[ISensor, ...]:
        if self.sensors is not None:
            return tuple(self.sensors)
        if self.config.mode == RunMode.BACKTEST:
            return (HistoricalSensor(self.historical_data_path),)
        if self._pg_pool is None:
            msg = "Runner PostgreSQL pool is not initialized"
            raise RuntimeError(msg)
        market_data_sensor = MarketDataSensor(
            store=PostgresMarketDataStore(self._pg_pool),
            asset_ids=[],
            persist_price_changes=self.config.sensor.persist_price_changes,
        )
        market_data_sensor.max_reconnect_interval_s = (
            self.config.sensor.max_reconnect_interval_s
        )
        market_data_sensor.max_message_size_bytes = (
            self.config.sensor.market_data_ws_max_size_bytes
        )
        return (
            MarketDiscoverySensor(
                store=PostgresMarketDataStore(self._pg_pool),
                http_client=_build_discovery_http_client(self.config),
                poll_interval_s=self.config.sensor.poll_interval_s,
                page_limit=self.config.sensor.discovery_page_limit,
                max_pages=self.config.sensor.discovery_max_pages,
                pagination_mode=self.config.sensor.discovery_pagination_mode,
                order=self.config.sensor.discovery_order,
                ascending=self.config.sensor.discovery_ascending,
                persist_price_snapshots=(
                    self.config.sensor.persist_discovery_price_snapshots
                ),
            ),
            market_data_sensor,
        )

    def _wire_active_perception(self, sensors: tuple[ISensor, ...]) -> None:
        self._clear_discovery_poll_complete_hook()
        self._market_selector = None
        self._subscription_controller = None
        self._reselection_task = None
        self._reselection_requested.clear()
        self._reselection_generation = 0
        if self.config.mode == RunMode.BACKTEST:
            return
        if self._pg_pool is None:
            msg = "Runner PostgreSQL pool is not initialized"
            raise RuntimeError(msg)

        if self._strategy_registry is not None:
            self._register_strategy_change_callbacks()

        discovery_sensor = _find_discovery_sensor(sensors)
        subscription_sink = _find_subscription_sink(sensors)
        if discovery_sensor is None or subscription_sink is None:
            return

        if self._strategy_registry is None:
            msg = "Runner strategy registry is not initialized"
            raise RuntimeError(msg)
        discovery_cycle_budget_s = self.config.sensor.discovery_max_pages * (
            self.config.sensor.discovery_http_timeout_s
            + self.config.sensor.discovery_http_pool_timeout_s
            + 30.0
        )
        self._market_selector = MarketSelector(
            PostgresMarketDataStore(
                self._pg_pool,
                market_freshness_s=max(
                    3.0 * self.config.sensor.poll_interval_s,
                    300.0,
                    discovery_cycle_budget_s,
                ),
            ),
            self._strategy_registry,
            UnionMergePolicy(),
            PostgresMarketSubscriptionStore(self._pg_pool),
        )
        self._subscription_controller = SensorSubscriptionController(subscription_sink)
        self._register_strategy_change_callbacks()
        discovery_sensor.on_poll_complete = self._handle_discovery_poll_complete
        self._reselection_task = asyncio.create_task(self._periodic_reselection_loop())

    def _supervised_sensor_task(
        self,
        index: int,
        sensor: ISensor,
    ) -> asyncio.Task[None]:
        name = f"sensor:{index}:{type(sensor).__name__}"

        async def consume_sensor() -> None:
            await self.sensor_stream.consume(sensor)

        return self._supervisor.spawn(
            self._worker_spec(
                name,
                consume_sensor,
            )
        )

    def _register_strategy_change_callbacks(self) -> None:
        if self._strategy_change_callbacks_registered:
            return
        registry = self._strategy_registry
        if registry is None:
            msg = "Runner strategy registry is not initialized"
            raise RuntimeError(msg)
        registry.register_change_callback(self._request_reselection)
        registry.register_change_callback(self._sync_controller_runtimes)
        self._strategy_change_callbacks_registered = True

    def _unregister_strategy_change_callbacks(self) -> None:
        registry = self._strategy_registry
        if not self._strategy_change_callbacks_registered:
            return
        if registry is None:
            self._strategy_change_callbacks_registered = False
            return
        registry.unregister_change_callback(self._request_reselection)
        registry.unregister_change_callback(self._sync_controller_runtimes)
        self._strategy_change_callbacks_registered = False

    def _assert_no_legacy_jsonl_paths(self) -> None:
        for store in (self.eval_store, self.feedback_store):
            if isinstance(getattr(store, "path", None), Path):
                msg = "legacy JSONL path referenced"
                raise RuntimeError(msg)

    def _should_boot_postgres_runtime(self) -> bool:
        if self._pg_pool is not None:
            return True
        if self.config.mode != RunMode.BACKTEST:
            return True
        return "database" in self.config.model_fields_set

    def worker_health_snapshot(self) -> dict[str, WorkerHealth]:
        return self._supervisor.snapshot()

    def worker_component_payload(self) -> dict[str, object]:
        return self._supervisor.component_payload()

    def _worker_spec(self, name: str, factory: WorkerFactory) -> WorkerSpec:
        return WorkerSpec(
            name=name,
            factory=factory,
            transient=WORKER_TRANSIENT_ERRORS,
            max_restarts=WORKER_MAX_RESTARTS,
            restart_window_s=WORKER_RESTART_WINDOW_S,
            backoff_initial_s=WORKER_RESTART_BACKOFF_INITIAL_S,
            backoff_max_s=WORKER_RESTART_BACKOFF_MAX_S,
        )

    async def _publish_worker_event(self, event_type: str, summary: str) -> None:
        await self.event_bus.publish(event_type, summary)

    def _trading_workers_ok(self) -> bool:
        snapshot = self._supervisor.snapshot()
        return all(
            health.state in ALIVE_WORKER_STATES
            for name, health in snapshot.items()
            if name in TRADING_REQUIRED_WORKERS or name.startswith("sensor:")
        )

    def _build_factor_service(self) -> FactorService:
        pool = self._pg_pool
        if pool is None:
            msg = "Runner PostgreSQL pool is not initialized"
            raise RuntimeError(msg)
        return FactorService(
            pool=pool,
            store=PostgresMarketDataStore(pool),
            cadence_s=self.config.factor_cadence_s,
            factors=REGISTERED,
            signal_stream=self.sensor_stream.subscribe(),
        )

    async def _run_factor_service_worker(self) -> None:
        service = self._factor_service
        if service is None or not self._factor_service_fresh:
            # FactorService instances are single-use: _stream_exhausted
            # latches once the subscription ends, so a supervised restart
            # must rebuild the instance with a fresh subscription. Close the
            # dead subscription so SensorStream stops buffering into it.
            self._close_factor_signal_subscription(service)
            service = self._build_factor_service()
            self._factor_service = service
        self._factor_service_fresh = False
        await service.run()

    def _close_factor_signal_subscription(
        self,
        service: FactorService | None,
    ) -> None:
        if service is None:
            return
        stream = service.signal_stream
        if isinstance(stream, SignalSubscription):
            stream.close()

    async def _ensure_default_v2_version(self) -> None:
        if not self.config.auto_migrate_default_v2:
            return
        if self._pg_pool is None:
            msg = "Runner PostgreSQL pool is not initialized"
            raise RuntimeError(msg)

        async with self._pg_pool.acquire() as connection:
            active_version_id = await connection.fetchval(
                """
                SELECT active_version_id
                FROM strategies
                WHERE strategy_id = 'default'
                """
            )
        if active_version_id != "default-v1":
            return

        registry = self._strategy_registry
        if registry is None:
            msg = "Runner strategy registry is not initialized"
            raise RuntimeError(msg)
        strategy = await registry.get_by_id("default")
        if strategy is None:
            return
        migrated_market_selection = _operational_default_market_selection(
            strategy.market_selection
        )
        if (
            strategy.config.factor_composition == DEFAULT_V2_FACTOR_COMPOSITION
            and strategy.market_selection == migrated_market_selection
        ):
            return

        migrated_strategy = Strategy(
            config=replace(
                strategy.config,
                factor_composition=DEFAULT_V2_FACTOR_COMPOSITION,
            ),
            risk=strategy.risk,
            eval_spec=strategy.eval_spec,
            forecaster=strategy.forecaster,
            market_selection=migrated_market_selection,
        )
        version = await registry.create_version(migrated_strategy)
        await registry.set_active("default", version.strategy_version_id)
        await registry.populate_strategy_factors(
            "default",
            version.strategy_version_id,
            _raw_factor_steps(migrated_strategy.config.factor_composition),
        )

    async def _ensure_paper_soak_strategy(self) -> None:
        if self.config.mode != RunMode.PAPER:
            return
        if self.config.paper_soak_strategy_id is None:
            return
        registry = self._strategy_registry
        if registry is None:
            msg = "Runner strategy registry is not initialized"
            raise RuntimeError(msg)
        if self.config.paper_soak_strategy_id == PAPER_MULTI_FACTOR_STRATEGY_ID:
            strategy = build_paper_multi_factor_strategy()
        elif self.config.paper_soak_strategy_id == H1_FLB_STRATEGY_ID:
            if self._flb_calibration_model is None:
                msg = "h1_flb paper soak requires strategies.flb_calibration_path"
                raise ValueError(msg)
            strategy = _h1_flb_strategy_from_settings(
                self.config,
                calibration_model=self._flb_calibration_model,
            )
        else:
            msg = (
                "Unsupported paper_soak_strategy_id: "
                f"{self.config.paper_soak_strategy_id!r}"
            )
            raise ValueError(msg)
        version = await registry.create_version(strategy, activate=False)
        await registry.populate_strategy_factors(
            strategy.config.strategy_id,
            version.strategy_version_id,
            _raw_factor_steps(strategy.config.factor_composition),
        )
        if self.config.paper_soak_archive_default:
            try:
                await registry.archive_strategy("default")
            except LookupError:
                logger.info(
                    "default strategy missing while activating paper soak strategy"
                )
        await registry.set_active(
            strategy.config.strategy_id,
            version.strategy_version_id,
        )

    def _bind_runtime_stores(self) -> None:
        if self._pg_pool is None:
            msg = "Runner PostgreSQL pool is not initialized"
            raise RuntimeError(msg)
        if isinstance(self._controller_factory, ControllerPipelineFactory):
            market_data_store = PostgresMarketDataStore(self._pg_pool)
            direct_book_reader = _controller_direct_book_reader(
                self.config,
                market_data_store,
            )
            self._controller_factory = ControllerPipelineFactory(
                settings=self.config,
                factor_reader=PostgresFactorSnapshotReader(self._pg_pool),
                outcome_token_resolver=MarketDataOutcomeTokenResolver(market_data_store),
                direct_book_reader=direct_book_reader,
            )
        if isinstance(self.eval_store, EvalStore):
            self.eval_store.bind_pool(self._pg_pool)
        if isinstance(self.feedback_store, FeedbackStore):
            self.feedback_store.bind_pool(self._pg_pool)
        if isinstance(self.decision_store, DecisionStore):
            self.decision_store.bind_pool(self._pg_pool)
        if isinstance(self.order_store, OrderStore):
            self.order_store.bind_pool(self._pg_pool)
        if isinstance(self.fill_store, FillStore):
            self.fill_store.bind_pool(self._pg_pool)
        if isinstance(self.quote_eval_store, QuoteEvalStore):
            self.quote_eval_store.bind_pool(self._pg_pool)
            self._evaluator_spool.quote_store = self.quote_eval_store
            self._evaluator_spool.quote_reader = TokenBookQuoteReader(
                PostgresMarketDataStore(self._pg_pool)
            )
        if isinstance(self.opportunity_store, OpportunityStore):
            self.opportunity_store.bind_pool(self._pg_pool)
        self.actuator_executor = self._build_executor(self.config.mode)

    def _unbind_runtime_stores(self) -> None:
        if isinstance(self._controller_factory, ControllerPipelineFactory):
            self._controller_factory = ControllerPipelineFactory(settings=self.config)
        if isinstance(self.eval_store, EvalStore):
            self.eval_store.pool = None
        if isinstance(self.feedback_store, FeedbackStore):
            self.feedback_store.pool = None
        if isinstance(self.decision_store, DecisionStore):
            self.decision_store.pool = None
        if isinstance(self.order_store, OrderStore):
            self.order_store.pool = None
        if isinstance(self.fill_store, FillStore):
            self.fill_store.pool = None
        if isinstance(self.quote_eval_store, QuoteEvalStore):
            self.quote_eval_store.pool = None
            self._evaluator_spool.quote_reader = None
        if isinstance(self.opportunity_store, OpportunityStore):
            self.opportunity_store.pool = None
        self.actuator_executor = self._build_executor(self.config.mode)

    def _build_executor(self, mode: RunMode) -> ActuatorExecutor:
        return ActuatorExecutor(
            adapter=self._build_adapter(mode),
            risk=RiskManager(self.config.risk),
            feedback=ActuatorFeedback(self.feedback_store),
            dedup_store=self._build_dedup_store(),
            event_bus=self.event_bus,
        )

    def _build_dedup_store(self) -> PgDedupStore | InMemoryDedupStore:
        if self._pg_pool is not None:
            return PgDedupStore(self._pg_pool)
        return InMemoryDedupStore()

    def _build_adapter(self, mode: RunMode) -> ActuatorAdapter:
        if mode == RunMode.BACKTEST:
            return BacktestActuator(self.historical_data_path)
        if mode == RunMode.PAPER:
            return PaperActuator(orderbooks=self._paper_orderbooks)
        quote_provider: LiveQuoteProvider = MissingLiveQuoteProvider()
        if self._pg_pool is not None:
            quote_provider = PolymarketRoutingQuoteProvider(
                snapshot_provider=PolymarketBookQuoteProvider(
                    PostgresMarketDataStore(self._pg_pool),
                    allowed_clock_skew_ms=(
                        self.config.controller.allowed_book_clock_skew_ms
                    ),
                ),
                direct_provider=PolymarketDirectQuoteProvider(
                    book_client=PolymarketSDKClient(),
                    allowed_clock_skew_ms=(
                        self.config.controller.allowed_book_clock_skew_ms
                    ),
                ),
                settings=self.config,
            )
        return PolymarketActuator(
            self.config,
            client=PolymarketSDKClient(),
            operator_gate=_first_live_order_gate(self.config),
            quote_provider=quote_provider,
            audit_writer=JsonlFirstOrderAuditWriter(
                Path(self.config.live_first_order_audit_path)
            ),
            live_preflight_validated=(
                mode == RunMode.LIVE and self._live_preflight_artifact_validated
            ),
            live_preflight_active_strategies_fingerprint=(
                self._live_preflight_active_strategies_fingerprint
            ),
        )

    def _remember_paper_orderbook(self, signal: MarketSignal) -> None:
        if signal.token_id is not None:
            self._paper_orderbooks[signal.token_id] = signal.orderbook
            return
        self._paper_orderbooks[signal.market_id] = signal.orderbook

    def _remember_signal_for_decision_evidence(self, signal: MarketSignal) -> None:
        if signal.token_id is not None:
            self._latest_signal_by_token[signal.token_id] = signal

    def _coalesce_controller_signal_batch(
        self,
        signal: MarketSignal,
        *,
        queue: asyncio.Queue[MarketSignal],
    ) -> MarketSignal:
        selected_key = _controller_signal_coalesce_key(signal)
        selected_signal = signal
        deferred_signals: list[MarketSignal] = []
        drained = 0
        while drained < CONTROLLER_SIGNAL_COOLESCE_LIMIT:
            try:
                queued_signal = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            drained += 1
            key = _controller_signal_coalesce_key(queued_signal)
            if key == selected_key:
                if _is_newer_controller_signal(queued_signal, selected_signal):
                    selected_signal = queued_signal
            else:
                deferred_signals.append(queued_signal)
            queue.task_done()

        for deferred_signal in deferred_signals:
            queue.put_nowait(deferred_signal)
        return selected_signal

    def _enrich_signal_with_controller_baselines(
        self,
        signal: MarketSignal,
    ) -> MarketSignal:
        if self._category_prior_estimator is None:
            return signal
        return enrich_signal_with_category_prior(
            signal,
            self._category_prior_estimator,
        )

    async def _controller_loop(self) -> None:
        while True:
            if self._should_stop_controller():
                return
            try:
                signal = await asyncio.wait_for(self.sensor_stream.queue.get(), 0.05)
            except TimeoutError:
                continue

            try:
                _append_bounded(self.state.signals, signal)
                self._remember_signal_for_decision_evidence(signal)
                if self.config.mode == RunMode.PAPER:
                    self._remember_paper_orderbook(signal)
                await self.event_bus.publish(
                    "sensor.signal",
                    _signal_event_summary(signal),
                    created_at=signal.fetched_at,
                    market_id=signal.market_id,
                )
                try:
                    await self._emit_position_exit_decisions(signal)
                except (asyncpg.PostgresError, OSError, TimeoutError) as error:
                    # A transient storage error must not kill the dispatcher:
                    # every pipeline loop treats a finished dispatcher task as
                    # clean shutdown and the whole trading path dismantles
                    # silently. The exit key is only recorded after a
                    # successful insert, so the exit retries on the next
                    # signal for the same market.
                    detail = _runtime_error_detail(error, self.config)
                    await self.event_bus.publish(
                        "error",
                        f"position exit emission failed: {detail}",
                        market_id=signal.market_id,
                    )
                    logger.warning("position exit emission failed: %s", detail)
                for strategy_id, runtime in tuple(self._controller_runtimes.items()):
                    if not _matches_strategy_scope(runtime.asset_ids, signal):
                        continue
                    queue = self._controller_signal_queues.get(strategy_id)
                    if queue is None:
                        continue
                    await queue.put(signal)
            finally:
                self.sensor_stream.queue.task_done()

    def _decision_uses_ask_pricing(
        self,
        decision: TradeDecision,
        evidence_signal: MarketSignal,
    ) -> bool:
        """True when the decision's limit price is the executable best ask of
        a price_reference=best_ask strategy — the frame where the spread is
        already inside expected_edge and must not be charged again in the
        persisted cost evidence (mirrors the pipeline's net-edge gate)."""
        runtime = self._controller_runtimes.get(decision.strategy_id)
        if runtime is None:
            return False
        strategy = getattr(runtime.controller, "strategy", None)
        config = getattr(strategy, "config", None)
        metadata = getattr(config, "metadata", None)
        if metadata is None:
            return False
        try:
            price_reference = dict(metadata).get("price_reference")
        except (TypeError, ValueError):
            return False
        if price_reference != "best_ask":
            return False
        executable_ask = best_ask(evidence_signal)
        return executable_ask is not None and (
            decision.limit_price == executable_ask
        )

    async def _controller_pipeline_loop(self, strategy_id: str) -> None:
        runtime = self._controller_runtimes[strategy_id]
        queue = self._controller_signal_queues[strategy_id]
        try:
            while True:
                if self._should_stop_controller_pipeline(strategy_id):
                    return
                try:
                    signal = await asyncio.wait_for(queue.get(), 0.05)
                except TimeoutError:
                    continue

                try:
                    signal = self._coalesce_controller_signal_batch(
                        signal,
                        queue=queue,
                    )
                    stale_signal_diagnostic = self._stale_queued_signal_diagnostic(
                        signal,
                        runtime,
                    )
                    if stale_signal_diagnostic is not None:
                        _append_bounded(
                            self.state.controller_diagnostics,
                            stale_signal_diagnostic,
                        )
                        continue
                    signal = self._enrich_signal_with_controller_baselines(signal)
                    opportunity: Opportunity | None = None
                    decision: TradeDecision | None = None
                    if isinstance(runtime.controller, OpportunityAwareController):
                        emission = await runtime.controller.on_signal(
                            signal,
                            portfolio=self.portfolio,
                        )
                        if emission is None:
                            diagnostic = _controller_diagnostic(runtime.controller)
                            if diagnostic is not None:
                                _append_bounded(self.state.controller_diagnostics, diagnostic)
                            continue
                        opportunity, decision = emission
                    else:
                        decision = await runtime.controller.decide(
                            signal,
                            portfolio=self.portfolio,
                        )
                    if decision is not None:
                        decision_signal = _controller_execution_signal_for_decision(
                            runtime.controller,
                            fallback_signal=signal,
                            decision=decision,
                            latest_signals_by_token=self._latest_signal_by_token,
                        )
                        if self.config.mode == RunMode.PAPER:
                            self._remember_signal_for_decision_evidence(
                                decision_signal
                            )
                            self._remember_paper_orderbook(decision_signal)
                        pre_actuator_diagnostic = self._pre_actuator_diagnostic(
                            decision,
                            decision_signal,
                        )
                        if pre_actuator_diagnostic is not None:
                            _append_bounded(
                                self.state.controller_diagnostics,
                                pre_actuator_diagnostic,
                            )
                            continue
                    if opportunity is not None:
                        await self.opportunity_store.insert(opportunity)
                        _append_bounded(self.state.opportunities, opportunity)
                    if decision is not None:
                        created_at = _decision_created_at(signal, opportunity)
                        expires_at = _decision_expires_at(
                            signal,
                            opportunity,
                            created_at=created_at,
                        )
                        factor_snapshot_hash = (
                            opportunity.factor_snapshot_hash
                            if opportunity is not None
                            else None
                        )
                        decision_evidence = _decision_evidence_from_signal(
                            decision_signal,
                            decision=decision,
                            factor_snapshot_hash=factor_snapshot_hash,
                            quote_source=self.config.controller.quote_source,
                            decision_created_at=created_at,
                            spread_already_in_price=(
                                self._decision_uses_ask_pricing(
                                    decision,
                                    decision_signal,
                                )
                            ),
                        )
                        await self.decision_store.insert(
                            decision,
                            factor_snapshot_hash=factor_snapshot_hash,
                            created_at=created_at,
                            expires_at=expires_at,
                            decision_evidence=decision_evidence,
                        )
                        update_status = getattr(
                            self.decision_store,
                            "update_status",
                            None,
                        )
                        if callable(update_status):
                            await update_status(
                                decision.decision_id,
                                current_status="pending",
                                next_status="accepted",
                                updated_at=created_at,
                        )
                        enqueued = await self._enqueue_decision(
                            decision,
                            signal=decision_signal,
                            queued_at=created_at,
                            decision_evidence=decision_evidence,
                        )
                        if not enqueued:
                            continue
                        _append_bounded(self.state.decisions, decision)
                        await self.event_bus.publish(
                            "controller.decision",
                            _decision_event_summary(decision),
                            created_at=created_at,
                            market_id=decision.market_id,
                            decision_id=decision.decision_id,
                        )
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "controller pipeline failed for %s: %s",
                strategy_id,
                error,
            )
            await self.event_bus.publish(
                "error",
                f"controller pipeline failed for {strategy_id}: {error}",
            )
            if self._controller_pipeline_error is None:
                self._controller_pipeline_error = error
            raise
        finally:
            current_task = asyncio.current_task()
            if (
                not self._stop_event.is_set()
                and self._controller_pipeline_tasks.get(strategy_id) is current_task
            ):
                await self._release_controller_runtime(strategy_id)

    async def _actuator_loop(self) -> None:
        while True:
            if self._should_stop_actuator():
                return
            try:
                raw_work_item = await asyncio.wait_for(self._decision_queue.get(), 0.05)
            except TimeoutError:
                continue

            try:
                work_item = _coerce_actuator_work_item(raw_work_item)
            except Exception as error:
                await self.event_bus.publish(
                    "error",
                    f"malformed actuator work item: {error}",
                )
                logger.warning("malformed actuator work item: %s", error)
                self._decision_queue.task_done()
                continue

            decision = work_item.decision
            signal = work_item.signal
            order_state: OrderState | None = None
            try:
                if (
                    self.config.mode == RunMode.LIVE
                    and self._live_trading_suspended_reason is not None
                ):
                    await self.event_bus.publish(
                        "error",
                        (
                            "live trading suspended: "
                            f"{self._live_trading_suspended_reason}"
                        ),
                        market_id=decision.market_id,
                        decision_id=decision.decision_id,
                    )
                    continue
                if self.config.mode == RunMode.PAPER and signal is not None:
                    self._remember_paper_orderbook(signal)
                await self._update_decision_status_if_supported(
                    decision.decision_id,
                    current_status="queued",
                    next_status="submitted",
                )
                order_state = await _execute_actuator_work_item(
                    self.actuator_executor,
                    decision,
                    self.portfolio,
                    dedup_acquired=work_item.dedup_acquired,
                    paper_orderbook=work_item.paper_orderbook,
                )
                _append_bounded(self.state.orders, order_state)
                try:
                    await self.order_store.insert(order_state)
                except Exception as error:  # noqa: BLE001
                    if self.config.mode == RunMode.LIVE:
                        await self._hard_halt_live_persistence_failure(
                            phase="order_state",
                            decision=decision,
                            order_state=order_state,
                            error=error,
                        )
                    logger.warning("order persistence failed: %s", error)
                self._record_live_order_risk_reservation(order_state)
                # LIVE fills get their real fee from venue reconciliation; for
                # paper/backtest, model it from the configured rate so net-edge
                # metrics are not inflated.
                fill_fee_rate = (
                    None
                    if self.config.mode == RunMode.LIVE
                    else self._paper_fill_fee_rate(decision)
                )
                fill = _fill_from_order(
                    order_state, decision, signal, fee_rate=fill_fee_rate
                )
                if fill is not None:
                    _append_bounded(self.state.fills, fill)
                    if _is_selection_funnel_entry_fill(fill):
                        increment_metric(SELECTION_FUNNEL_TRADED_TOTAL_METRIC)
                    try:
                        await self.fill_store.insert(fill)
                    except Exception as error:  # noqa: BLE001
                        if self.config.mode == RunMode.LIVE:
                            await self._hard_halt_live_persistence_failure(
                                phase="fill",
                                decision=decision,
                                order_state=order_state,
                                error=error,
                            )
                        logger.warning("fill persistence failed: %s", error)
                    self.portfolio = _portfolio_with_fill(self.portfolio, fill)
                    self._record_position_exit_reentry_quarantine(fill, decision)
                    decision_evidence = work_item.decision_evidence
                    if not decision_evidence and signal is not None:
                        evidence_signal = _decision_evidence_signal_for_decision(
                            signal,
                            decision,
                            latest_signals_by_token=self._latest_signal_by_token,
                        )
                        decision_evidence = _decision_evidence_from_signal(
                            evidence_signal,
                            decision=decision,
                            factor_snapshot_hash=None,
                            quote_source=self.config.controller.quote_source,
                            decision_created_at=signal.fetched_at,
                            spread_already_in_price=(
                                self._decision_uses_ask_pricing(
                                    decision,
                                    evidence_signal,
                                )
                            ),
                        )
                    self._evaluator_spool.enqueue(
                        fill,
                        decision,
                        decision_evidence=decision_evidence,
                    )
                    await self.event_bus.publish(
                        "actuator.fill",
                        _fill_event_summary(fill),
                        created_at=fill.filled_at,
                        market_id=fill.market_id,
                        decision_id=fill.decision_id,
                        fill_id=fill.fill_id or fill.order_id,
                    )
                await self._update_decision_status_if_supported(
                    decision.decision_id,
                    current_status="submitted",
                    next_status=_decision_status_from_order(order_state),
                )
            except PolymarketSubmissionUnknownError as error:
                detail = _runtime_error_detail(error, self.config)
                self._live_trading_suspended_reason = "submission_unknown"
                self._stop_event.set()
                unknown_order_state = error.order_state
                if unknown_order_state is not None:
                    _append_bounded(self.state.orders, unknown_order_state)
                    try:
                        await self.order_store.insert(unknown_order_state)
                    except Exception as store_error:  # noqa: BLE001
                        if self.config.mode == RunMode.LIVE:
                            await self._hard_halt_live_persistence_failure(
                                phase="submission_unknown_order_state",
                                decision=decision,
                                order_state=unknown_order_state,
                                error=store_error,
                            )
                        logger.warning(
                            "submission_unknown order persistence failed: %s",
                            store_error,
                        )
                await self._update_decision_status_if_supported(
                    decision.decision_id,
                    current_status="submitted",
                    next_status="submission_unknown",
                )
                await self.event_bus.publish(
                    "error",
                    f"live trading suspended for submission_unknown: {detail}",
                    market_id=decision.market_id,
                    decision_id=decision.decision_id,
                )
                logger.warning(
                    "live trading suspended for submission_unknown: %s",
                    detail,
                )
            except LivePersistenceFailureError:
                raise
            except Exception as error:
                detail = _runtime_error_detail(error, self.config)
                await self.event_bus.publish(
                    "error",
                    f"actuator execution failed: {detail}",
                    market_id=decision.market_id,
                    decision_id=decision.decision_id,
                )
                logger.warning("actuator execution failed: %s", detail)
            finally:
                if not self._keeps_live_open_order_reservations(order_state):
                    self._release_position_exit_key(decision.decision_id)
                    self._release_position_capacity_reservation(decision.decision_id)
                self._decision_queue.task_done()

    def _paper_fill_fee_rate(self, decision: TradeDecision) -> float:
        strategy = self._active_strategies_by_version.get(
            (decision.strategy_id, decision.strategy_version_id)
        )
        if strategy is None:
            return self.config.strategies.flb_fee_rate
        return strategy_economic_float(
            strategy,
            "fee_rate",
            self.config.strategies.flb_fee_rate,
        )

    async def _decision_expiry_loop(self) -> None:
        while True:
            if self._should_stop_decision_expiry():
                return
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=DECISION_SWEEP_INTERVAL_S,
                )
            except TimeoutError:
                pass
            if self._should_stop_decision_expiry():
                return
            try:
                await self._sweep_expired_decisions_once()
            except Exception as error:  # noqa: BLE001
                logger.warning("decision expiry sweep failed: %s", error)

    async def _runtime_heartbeat_loop(self) -> None:
        while True:
            if self._stop_event.is_set():
                return
            try:
                await self._write_runtime_heartbeat()
            except Exception as error:  # noqa: BLE001
                logger.warning("runtime heartbeat write failed: %s", error)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=RUNTIME_HEARTBEAT_INTERVAL_S,
                )
            except TimeoutError:
                continue

    async def _write_runtime_heartbeat(self) -> None:
        if self._pg_pool is None:
            return
        run_id = self.state.runtime_run_id
        started_at = self.state.runner_started_at
        if run_id is None or started_at is None:
            return
        await RuntimeHeartbeatStore(self._pg_pool).append(
            run_id=run_id,
            mode=self.config.mode.value,
            started_at=started_at,
            observed_at=datetime.now(tz=UTC),
            strategy_fingerprint=_runtime_strategy_fingerprint(
                tuple(self._controller_runtimes.values())
            ),
            component_status={
                **self._runtime_sensor_component_status(),
                "controller_runtimes": len(self._controller_runtimes),
            },
        )

    async def _resolution_sweep_loop(self) -> None:
        while True:
            if self._stop_event.is_set():
                return
            try:
                result = await self._resolution_sweep_once()
                self._resolution_sweeps_total += 1
                self._resolution_fills_resolved_total += result.fills_resolved
                # Per-fill processing failures (eval append or scoring) leave
                # the fill unresolved for retry; they count as sweep failures
                # so /status can distinguish them from healthy idle.
                self._resolution_sweep_failures_total += result.fills_failed
                if result.fills_resolved > 0:
                    logger.info(
                        "resolution sweep resolved %d fill(s), appended %d "
                        "eval record(s) (%d still unresolved)",
                        result.fills_resolved,
                        result.eval_records_appended,
                        result.unresolved_fills - result.fills_resolved,
                    )
            except Exception:  # noqa: BLE001
                self._resolution_sweep_failures_total += 1
                logger.exception("resolution sweep failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.resolution_poll_interval_s,
                )
            except TimeoutError:
                continue

    async def _resolution_sweep_once(self) -> ResolutionSweepResult:
        # The HTTP client is scoped to a single sweep so every exit path
        # (success, transient failure, cancellation) releases it; at the
        # default 300s cadence per-cycle construction is negligible, and no
        # network request is made when there are no unresolved fills.
        async with _build_discovery_http_client(self.config) as http_client:
            sweeper = ResolutionSweeper(
                fill_store=self.fill_store,
                decision_reader=self.decision_store,
                eval_spool=self._evaluator_spool,
                resolution_source=GammaResolutionSource(http_client=http_client),
                resolution_store=(
                    MarketResolutionStore(self._pg_pool)
                    if self._pg_pool is not None
                    else None
                ),
            )
            return await sweeper.sweep_once()

    async def _enqueue_decision(
        self,
        decision: TradeDecision,
        *,
        signal: MarketSignal | None,
        dedup_acquired: bool = False,
        queued_at: datetime | None = None,
        decision_evidence: Mapping[str, object] | None = None,
    ) -> bool:
        if self._would_exceed_position_capacity(decision):
            if signal is not None:
                _append_bounded(
                    self.state.controller_diagnostics,
                    self._position_capacity_diagnostic(decision, signal),
                )
            await self._reject_unqueued_decision(
                decision,
                dedup_acquired=dedup_acquired,
                updated_at=(
                    queued_at or (signal.fetched_at if signal is not None else None)
                ),
                reason="max_open_positions_capacity",
            )
            return False
        risk_rejection_reason = self._risk_rejection_reason(decision)
        if risk_rejection_reason is not None:
            if signal is not None:
                _append_bounded(
                    self.state.controller_diagnostics,
                    self._risk_rejection_diagnostic(
                        decision,
                        signal,
                        reason=risk_rejection_reason,
                    ),
                )
            await self._reject_unqueued_decision(
                decision,
                dedup_acquired=dedup_acquired,
                updated_at=(
                    queued_at or (signal.fetched_at if signal is not None else None)
                ),
                reason=risk_rejection_reason,
            )
            return False
        reserved_key = self._reserve_position_capacity(decision)
        try:
            await self._update_decision_status_if_supported(
                decision.decision_id,
                current_status="accepted",
                next_status="queued",
                updated_at=(
                    queued_at or (signal.fetched_at if signal is not None else None)
                ),
            )
            await self._decision_queue.put(
                ActuatorWorkItem(
                    decision=decision,
                    signal=signal,
                    dedup_acquired=dedup_acquired,
                    decision_evidence=decision_evidence or {},
                    paper_orderbook=(
                        _paper_orderbook_snapshot_for_decision(
                            decision,
                            signal,
                            latest_signals_by_token=self._latest_signal_by_token,
                        )
                        if self.config.mode == RunMode.PAPER
                        else None
                    ),
                )
            )
            return True
        except BaseException:
            if reserved_key is not None:
                self._release_position_capacity_reservation(decision.decision_id)
            raise

    async def _reject_unqueued_decision(
        self,
        decision: TradeDecision,
        *,
        dedup_acquired: bool,
        updated_at: datetime | None,
        reason: str,
    ) -> None:
        await self._update_decision_status_if_supported(
            decision.decision_id,
            current_status="accepted",
            next_status="rejected",
            updated_at=updated_at,
        )
        if dedup_acquired:
            await self.actuator_executor.dedup_store.release(
                decision.decision_id,
                "rejected",
            )
        logger.warning(
            "decision rejected before actuator queue: decision_id=%s reason=%s",
            decision.decision_id,
            reason,
        )

    async def _emit_position_exit_decisions(self, signal: MarketSignal) -> int:
        emitted = 0
        if not self.config.position_exit.enabled:
            return emitted
        signal = await self._signal_with_exit_outcome_tokens(signal)
        for position in tuple(self.portfolio.open_positions):
            marked_position = mark_position_from_signal(position, signal)
            if marked_position is None:
                continue
            exit_signal = self._position_exit_monitor.evaluate(
                marked_position,
                now=signal.timestamp,
            )
            if exit_signal is None:
                continue
            key = exit_key(exit_signal)
            if key in self._emitted_position_exit_keys:
                continue
            decision = build_exit_decision(
                signal,
                exit_signal,
                max_slippage_bps=self.config.controller.max_slippage_bps,
                time_in_force=TimeInForce(
                    str(self.config.controller.time_in_force).upper()
                ),
            )
            created_at = signal.timestamp
            expires_at = _decision_expires_at(signal, None, created_at=created_at)
            await self.decision_store.insert(
                decision,
                factor_snapshot_hash=None,
                created_at=created_at,
                expires_at=expires_at,
            )
            update_status = getattr(self.decision_store, "update_status", None)
            if callable(update_status):
                await update_status(
                    decision.decision_id,
                    current_status="pending",
                    next_status="accepted",
                    updated_at=created_at,
                )
            self._emitted_position_exit_keys.add(key)
            self._position_exit_keys_by_decision[decision.decision_id] = key
            try:
                enqueued = await self._enqueue_decision(
                    decision,
                    signal=signal,
                    queued_at=created_at,
                )
            except Exception:
                self._release_position_exit_key(decision.decision_id)
                raise
            if not enqueued:
                self._release_position_exit_key(decision.decision_id)
                continue
            _append_bounded(self.state.decisions, decision)
            await self.event_bus.publish(
                "controller.decision",
                _decision_event_summary(decision),
                created_at=created_at,
                market_id=decision.market_id,
                decision_id=decision.decision_id,
            )
            emitted += 1
        return emitted

    async def _signal_with_exit_outcome_tokens(
        self,
        signal: MarketSignal,
    ) -> MarketSignal:
        if (
            "yes_token_id" in signal.external_signal
            and "no_token_id" in signal.external_signal
        ):
            return signal
        if self._pg_pool is None:
            return signal
        tokens = await MarketDataOutcomeTokenResolver(
            PostgresMarketDataStore(self._pg_pool)
        ).resolve(market_id=signal.market_id, signal_token_id=signal.token_id)
        token_metadata: dict[str, str] = {}
        if tokens.yes_token_id is not None:
            token_metadata["yes_token_id"] = tokens.yes_token_id
        if tokens.no_token_id is not None:
            token_metadata["no_token_id"] = tokens.no_token_id
        if not token_metadata:
            return signal
        return replace(
            signal,
            external_signal={**signal.external_signal, **token_metadata},
        )

    def _would_exceed_position_capacity(self, decision: TradeDecision) -> bool:
        max_positions = self.config.risk.max_open_positions
        if max_positions is None:
            return False
        if not self._decision_opens_new_position_slot(decision):
            return False
        open_keys = _open_position_keys(self.portfolio)
        pending_new_keys = set(
            self._pending_open_position_keys_by_decision.values()
        ) - open_keys
        return len(self.portfolio.open_positions) + len(pending_new_keys) >= max_positions

    def _reserve_position_capacity(
        self,
        decision: TradeDecision,
    ) -> PositionSlotKey | None:
        if self._would_exceed_position_capacity(decision):
            return None
        if not self._decision_opens_new_position_slot(decision):
            return None
        key = _position_key_from_decision(decision)
        self._pending_open_position_keys_by_decision[decision.decision_id] = key
        return key

    def _release_position_capacity_reservation(self, decision_id: str) -> None:
        self._pending_open_position_keys_by_decision.pop(decision_id, None)

    def _decision_opens_new_position_slot(self, decision: TradeDecision) -> bool:
        key = _position_key_from_decision(decision)
        if key in _open_position_keys(self.portfolio):
            return False
        if key in set(self._pending_open_position_keys_by_decision.values()):
            return False
        return not _decision_reduces_existing_position(self.portfolio, decision)

    def _pre_actuator_diagnostic(
        self,
        decision: TradeDecision,
        signal: MarketSignal,
    ) -> ControllerDiagnostic | None:
        reentry_diagnostic = self._position_exit_reentry_diagnostic(
            decision,
            signal,
        )
        if reentry_diagnostic is not None:
            return reentry_diagnostic
        if (
            self.config.mode == RunMode.PAPER
            and not self._paper_orderbook_ready_for_decision(decision)
        ):
            return ControllerDiagnostic(
                code="paper_orderbook_missing_for_decision_token",
                message=(
                    "Skipping paper decision because the target token orderbook "
                    "has not been observed yet."
                ),
                market_id=decision.market_id,
                strategy_id=decision.strategy_id,
                strategy_version_id=decision.strategy_version_id,
                token_id=decision.token_id,
                severity="warning",
                metadata={
                    "decision_outcome": decision.outcome,
                    "decision_token_id": decision.token_id,
                    "signal_token_id": signal.token_id,
                    "known_orderbook_ids": sorted(self._paper_orderbooks.keys()),
                },
            )
        if self.config.mode == RunMode.PAPER:
            paper_liquidity_diagnostic = self._paper_liquidity_diagnostic(
                decision,
                signal,
            )
            if paper_liquidity_diagnostic is not None:
                return paper_liquidity_diagnostic
        if self._would_exceed_position_capacity(decision):
            return self._position_capacity_diagnostic(decision, signal)
        risk_rejection_reason = self._risk_rejection_reason(decision)
        if risk_rejection_reason is not None:
            return self._risk_rejection_diagnostic(
                decision,
                signal,
                reason=risk_rejection_reason,
            )
        return None

    def _stale_queued_signal_diagnostic(
        self,
        signal: MarketSignal,
        runtime: StrategyControllerRuntime,
    ) -> ControllerDiagnostic | None:
        if self.config.mode == RunMode.BACKTEST:
            return None
        book_received_at = _external_signal_datetime(
            signal.external_signal.get("book_received_at")
        )
        if book_received_at is None:
            return None
        if _controller_can_refresh_queued_book(runtime.controller):
            return None
        max_book_age_ms = self.config.controller.max_book_age_ms
        raw_age_ms = (
            _aware_utc(datetime.now(tz=UTC)) - _aware_utc(book_received_at)
        ).total_seconds() * 1000.0
        if raw_age_ms < -self.config.controller.allowed_book_clock_skew_ms:
            book_age_ms = float("inf")
        else:
            book_age_ms = max(0.0, raw_age_ms)
        if book_age_ms <= max_book_age_ms:
            return None
        return ControllerDiagnostic(
            code="router_gate:book_too_stale",
            message=(
                "Skipping queued signal because the orderbook was already stale "
                "before controller processing."
            ),
            market_id=signal.market_id,
            strategy_id=runtime.strategy_id,
            strategy_version_id=runtime.strategy_version_id,
            token_id=signal.token_id,
            severity="info",
            metadata={
                "gate_reason": "book_too_stale",
                "phase": "queue_dequeue",
                "book_age_ms": book_age_ms,
                "max_book_age_ms": max_book_age_ms,
            },
        )

    def _risk_rejection_reason(self, decision: TradeDecision) -> str | None:
        risk = getattr(self.actuator_executor, "risk", None)
        check = getattr(risk, "check", None)
        if not callable(check):
            return None
        risk_decision = check(decision, self.portfolio)
        if risk_decision.approved:
            return None
        return str(risk_decision.reason)

    def _risk_rejection_diagnostic(
        self,
        decision: TradeDecision,
        signal: MarketSignal,
        *,
        reason: str,
    ) -> ControllerDiagnostic:
        estimated_quantity = _estimated_decision_quantity(decision)
        metadata: dict[str, object] = {
            "risk_reason": reason,
            "decision_notional_usdc": decision.notional_usdc,
            "decision_limit_price": decision.limit_price,
            "decision_token_id": decision.token_id,
            "decision_outcome": decision.outcome,
            "signal_token_id": signal.token_id,
        }
        if estimated_quantity is not None:
            metadata["estimated_quantity_shares"] = estimated_quantity
        if self.config.risk.max_quantity_shares is not None:
            metadata["max_quantity_shares"] = self.config.risk.max_quantity_shares
        return ControllerDiagnostic(
            code=reason,
            message=(
                "Skipping decision because the risk envelope would reject it "
                "before actuator execution."
            ),
            market_id=decision.market_id,
            strategy_id=decision.strategy_id,
            strategy_version_id=decision.strategy_version_id,
            token_id=decision.token_id,
            severity="warning",
            metadata=metadata,
        )

    def _position_capacity_diagnostic(
        self,
        decision: TradeDecision,
        signal: MarketSignal,
    ) -> ControllerDiagnostic:
        max_positions = self.config.risk.max_open_positions
        return ControllerDiagnostic(
            code="max_open_positions_capacity",
            message=(
                "Skipping decision because open plus queued new positions "
                "would exceed the configured max_open_positions cap."
            ),
            market_id=decision.market_id,
            strategy_id=decision.strategy_id,
            strategy_version_id=decision.strategy_version_id,
            token_id=decision.token_id,
            severity="warning",
            metadata={
                "max_open_positions": max_positions,
                "open_positions": len(self.portfolio.open_positions),
                "pending_new_positions": len(
                    set(self._pending_open_position_keys_by_decision.values())
                    - _open_position_keys(self.portfolio)
                ),
                "decision_token_id": decision.token_id,
                "decision_outcome": decision.outcome,
                "signal_token_id": signal.token_id,
            },
        )

    def _paper_orderbook_ready_for_decision(self, decision: TradeDecision) -> bool:
        if decision.token_id is not None and decision.token_id in self._paper_orderbooks:
            return True
        return (
            (decision.outcome == "YES" or decision.token_id is None)
            and decision.market_id in self._paper_orderbooks
        )

    def _paper_liquidity_diagnostic(
        self,
        decision: TradeDecision,
        signal: MarketSignal,
    ) -> ControllerDiagnostic | None:
        action = decision.action or decision.side
        if action not in {Side.BUY.value, Side.SELL.value}:
            return None
        orderbook = _paper_orderbook_for_decision(self._paper_orderbooks, decision)
        if orderbook is None:
            return None
        failure = paper_orderbook_fill_failure(orderbook, decision)
        if failure is None:
            return None
        return ControllerDiagnostic(
            code="paper_orderbook_insufficient_liquidity",
            message=(
                "Skipping paper decision because the observed paper orderbook "
                "cannot fill the target notional within the configured limit "
                "and slippage."
            ),
            market_id=decision.market_id,
            strategy_id=decision.strategy_id,
            strategy_version_id=decision.strategy_version_id,
            token_id=decision.token_id,
            severity="warning",
            metadata={
                "decision_outcome": decision.outcome,
                "decision_token_id": decision.token_id,
                "signal_token_id": signal.token_id,
                "decision_notional_usdc": decision.notional_usdc,
                "decision_limit_price": decision.limit_price,
                "max_slippage_bps": decision.max_slippage_bps,
                "failure": failure,
            },
        )

    def _release_position_exit_key(self, decision_id: str) -> None:
        key = self._position_exit_keys_by_decision.pop(decision_id, None)
        if key is not None:
            self._emitted_position_exit_keys.discard(key)

    def _keeps_live_open_order_reservations(
        self,
        order_state: OrderState | None,
    ) -> bool:
        return (
            self.config.mode == RunMode.LIVE
            and order_state is not None
            and _is_open_order_state(order_state)
        )

    def _record_live_order_risk_reservation(
        self,
        order_state: OrderState | None,
    ) -> None:
        if self.config.mode != RunMode.LIVE or order_state is None:
            return
        risk = getattr(self.actuator_executor, "risk", None)
        if risk is None:
            return
        if _is_open_order_state(order_state):
            record_open_order = getattr(risk, "record_open_order_state", None)
            if callable(record_open_order):
                record_open_order(order_state)
            return
        record_order_filled = getattr(risk, "record_order_filled", None)
        if callable(record_order_filled):
            record_order_filled(order_state.order_id)

    def _runtime_sensor_component_status(self) -> dict[str, object]:
        tasks = self.sensor_stream.tasks
        running_tasks = 0
        failed_tasks = 0
        for task in tasks:
            if not task.done():
                running_tasks += 1
                continue
            if task.cancelled():
                continue
            if task.exception() is not None:
                failed_tasks += 1
        sensor_running = bool(tasks) and running_tasks == len(tasks)
        return {
            "running": (
                sensor_running
                and len(self._controller_runtimes) > 0
                and not self._stop_event.is_set()
                # Dead trading workers (dispatcher/actuator/factor) must
                # count as unhealthy heartbeat periods: this is the
                # correction of the false-green the 2026-06-10 audit found.
                and self._trading_workers_ok()
            ),
            "sensor_running": sensor_running,
            "sensor_tasks": running_tasks,
            "sensor_tasks_total": len(tasks),
            "sensor_task_failures": failed_tasks,
            "workers": self._supervisor.component_payload(),
        }

    def _record_position_exit_reentry_quarantine(
        self,
        fill: FillRecord,
        decision: TradeDecision,
    ) -> None:
        cooldown_s = self.config.position_exit.reentry_cooldown_s
        if cooldown_s <= 0.0 or not _is_position_exit_decision(decision):
            return
        key = _position_exit_reentry_key_from_decision(decision)
        self._position_exit_reentry_quarantine_until[key] = (
            fill.filled_at + timedelta(seconds=cooldown_s)
        )

    def _position_exit_reentry_diagnostic(
        self,
        decision: TradeDecision,
        signal: MarketSignal,
    ) -> ControllerDiagnostic | None:
        cooldown_s = self.config.position_exit.reentry_cooldown_s
        if cooldown_s <= 0.0 or _is_position_exit_decision(decision):
            return None
        if _decision_reduces_existing_position(self.portfolio, decision):
            return None
        now = signal.timestamp
        self._drop_expired_position_exit_reentry_quarantines(now)
        key = _position_exit_reentry_key_from_decision(decision)
        quarantined_until = self._position_exit_reentry_quarantine_until.get(key)
        if quarantined_until is None or quarantined_until <= now:
            return None
        seconds_remaining = (quarantined_until - now).total_seconds()
        return ControllerDiagnostic(
            code="position_exit_reentry_cooldown",
            message=(
                "Skipping decision because the same market/token is cooling down "
                "after a position-exit fill."
            ),
            market_id=decision.market_id,
            strategy_id=decision.strategy_id,
            strategy_version_id=decision.strategy_version_id,
            token_id=decision.token_id,
            severity="warning",
            metadata={
                "cooldown_s": cooldown_s,
                "seconds_remaining": seconds_remaining,
                "risk_group_id": decision.risk_group_id,
            },
        )

    def _drop_expired_position_exit_reentry_quarantines(
        self,
        now: datetime,
    ) -> None:
        expired = [
            key
            for key, quarantined_until in (
                self._position_exit_reentry_quarantine_until.items()
            )
            if quarantined_until <= now
        ]
        for key in expired:
            del self._position_exit_reentry_quarantine_until[key]

    async def _update_decision_status_if_supported(
        self,
        decision_id: str,
        *,
        current_status: str,
        next_status: str,
        updated_at: datetime | None = None,
    ) -> bool:
        update_status = getattr(self.decision_store, "update_status", None)
        if not callable(update_status):
            return False
        try:
            return bool(
                await update_status(
                    decision_id,
                    current_status=current_status,
                    next_status=next_status,
                    updated_at=updated_at or datetime.now(tz=UTC),
                )
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "decision status update failed for decision_id=%s %s->%s: %s",
                decision_id,
                current_status,
                next_status,
                error,
            )
            return False

    async def _hard_halt_live_persistence_failure(
        self,
        *,
        phase: str,
        decision: TradeDecision,
        order_state: OrderState | None,
        error: BaseException,
    ) -> None:
        self._live_trading_suspended_reason = "persistence_failure"
        self._stop_event.set()
        detail = _runtime_error_detail(error, self.config)
        audit_writer = LiveEmergencyAuditWriter(
            Path(self.config.live_emergency_audit_path)
        )
        try:
            await audit_writer.append(
                phase=phase,
                decision=decision,
                order_state=order_state,
                error=error,
                error_detail=detail,
            )
        except Exception as audit_error:  # noqa: BLE001
            logger.warning("live emergency audit append failed: %s", audit_error)
        await self.event_bus.publish(
            "error",
            f"live trading suspended: persistence failure during {phase}: {detail}",
            market_id=decision.market_id,
            decision_id=decision.decision_id,
        )
        msg = f"LIVE persistence failure during {phase}: {detail}"
        raise LivePersistenceFailureError(msg) from error

    async def _sweep_expired_decisions_once(
        self,
        *,
        now: datetime | None = None,
    ) -> int:
        sweep_time = now or datetime.now(tz=UTC)
        expired = await self.decision_store.expire_pending(before=sweep_time)
        await self.decision_store.prune_expired(
            before=sweep_time
            - timedelta(seconds=self.config.database.expired_decision_retention_s)
        )
        return expired

    async def _assert_no_unresolved_submission_unknown_incidents(self) -> None:
        pool = self._pg_pool
        if pool is None:
            return
        acquire = getattr(pool, "acquire", None)
        if acquire is None:
            return
        try:
            async with acquire() as connection:
                count = await connection.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM order_intents
                    WHERE outcome = 'submission_unknown'
                      AND reconciled_at IS NULL
                    """
                )
        except Exception as error:  # noqa: BLE001
            detail = _runtime_error_detail(error, self.config)
            msg = (
                "LIVE submission_unknown guard failed "
                f"({type(error).__name__}: {detail}); refusing to start "
                "without verified incident state"
            )
            raise RuntimeError(msg) from error
        if int(count or 0) > 0:
            msg = (
                "LIVE start refused: unresolved submission_unknown incident "
                "requires venue reconciliation before resume"
            )
            raise RuntimeError(msg)

    async def _assert_live_market_data_freshness(self) -> None:
        if self.config.mode != RunMode.LIVE:
            return
        pool = self._pg_pool
        if pool is None:
            return
        acquire = getattr(pool, "acquire", None)
        if acquire is None:
            return
        await ensure_live_market_data_freshness(self.config, pool)

    async def _metrics_by_strategy(
        self,
    ) -> dict[StrategyVersionKey, tuple[StrategyMetricsSnapshot, EvalSpec]]:
        records = await self.eval_store.all()
        snapshots = MetricsCollector(records).snapshot_by_strategy()
        eval_specs = await self._active_eval_specs()
        return {
            key: (snapshot, eval_specs[key])
            for key, snapshot in snapshots.items()
            if key in eval_specs
        }

    async def _active_eval_specs(self) -> dict[StrategyVersionKey, EvalSpec]:
        if self._strategy_registry is None or not hasattr(
            self._strategy_registry,
            "list_active_strategies",
        ):
            if self._controller_runtimes:
                default_eval_spec = _default_active_strategy(self.config).eval_spec
                return {
                    (runtime.strategy_id, runtime.strategy_version_id): default_eval_spec
                    for runtime in self._controller_runtimes.values()
                }
            default_strategy = _default_active_strategy(self.config)
            return {
                (
                    default_strategy.strategy_id,
                    default_strategy.strategy_version_id,
                ): default_strategy.eval_spec
            }

        active_strategies = await self._strategy_registry.list_active_strategies()
        eval_specs = {
            (strategy.strategy_id, strategy.strategy_version_id): strategy.eval_spec
            for strategy in active_strategies
        }
        if self._controller_runtimes:
            default_eval_spec = _default_active_strategy(self.config).eval_spec
            for runtime in self._controller_runtimes.values():
                eval_specs.setdefault(
                    (runtime.strategy_id, runtime.strategy_version_id),
                    default_eval_spec,
                )
        if not self._should_build_strategy_registry_controllers():
            default_strategy = _default_active_strategy(self.config)
            eval_specs.setdefault(
                (default_strategy.strategy_id, default_strategy.strategy_version_id),
                default_strategy.eval_spec,
            )
        return eval_specs

    def _should_stop_controller(self) -> bool:
        if self._stop_event.is_set() and self.sensor_stream.queue.empty():
            return True
        return self._sensors_finished() and self.sensor_stream.queue.empty()

    def _should_stop_actuator(self) -> bool:
        if self._stop_event.is_set() and self._decision_queue.empty():
            return True
        controller_done = (
            self._controller_task is not None
            and self._controller_task.done()
            and all(task.done() for task in self._controller_pipeline_tasks.values())
        )
        return controller_done and self._decision_queue.empty()

    def _should_stop_decision_expiry(self) -> bool:
        if self._stop_event.is_set():
            return True
        return self.config.mode == RunMode.BACKTEST and self._should_stop_actuator()

    def _should_stop_controller_pipeline(self, strategy_id: str) -> bool:
        queue = self._controller_signal_queues.get(strategy_id)
        if queue is None:
            return True
        if self._stop_event.is_set() and queue.empty():
            return True
        dispatcher_done = self._controller_task is not None and self._controller_task.done()
        return dispatcher_done and queue.empty()

    def _sensors_finished(self) -> bool:
        return bool(self.sensor_stream.tasks) and all(
            task.done() for task in self.sensor_stream.tasks
        )

    async def _close_active_sensors(self) -> None:
        self._clear_discovery_poll_complete_hook()
        for sensor in self._active_sensors:
            if isinstance(sensor, AsyncCloseable):
                await sensor.aclose()
        self._active_sensors = ()

    async def _close_pg_pool(self) -> None:
        await self.close_pg_pool()

    async def _cleanup_after_start_failure(self) -> None:
        self._supervisor.stop()
        stop_error: BaseException | None = None
        if self._factor_service_task is not None and not self._factor_service_task.done():
            self._factor_service_task.cancel()
        decision_expiry_task = self._decision_expiry_task
        if decision_expiry_task is not None and not decision_expiry_task.done():
            decision_expiry_task.cancel()
        runtime_heartbeat_task = self._runtime_heartbeat_task
        if runtime_heartbeat_task is not None and not runtime_heartbeat_task.done():
            runtime_heartbeat_task.cancel()
        resolution_sweep_task = self._resolution_sweep_task
        if resolution_sweep_task is not None and not resolution_sweep_task.done():
            resolution_sweep_task.cancel()
        for task in self._controller_pipeline_tasks.values():
            if not task.done():
                task.cancel()
        reselection_task = self._reselection_task
        self._reselection_requested.set()
        self._clear_discovery_poll_complete_hook()
        self._unregister_strategy_change_callbacks()
        if reselection_task is not None and not reselection_task.done():
            reselection_task.cancel()
        try:
            await asyncio.gather(
                *(
                    task
                    for task in (
                        self._factor_service_task,
                        *self._controller_pipeline_tasks.values(),
                        decision_expiry_task,
                        runtime_heartbeat_task,
                        resolution_sweep_task,
                    )
                    if task is not None
                ),
                return_exceptions=True,
            )
        except BaseException as exc:  # pragma: no cover - defensive
            stop_error = exc
        try:
            await self._await_cancelled_task(reselection_task)
        except BaseException as exc:  # pragma: no cover - defensive
            if stop_error is None:
                stop_error = exc
        try:
            await self.sensor_stream.stop()
        except BaseException as exc:  # pragma: no cover - defensive
            if stop_error is None:
                stop_error = exc

        try:
            await self._evaluator_spool.stop()
        except BaseException as exc:  # pragma: no cover - defensive
            if stop_error is None:
                stop_error = exc

        try:
            await self._close_active_sensors()
        except BaseException as exc:  # pragma: no cover - defensive
            if stop_error is None:
                stop_error = exc

        self._factor_service = None
        self._factor_service_task = None
        self._controller_task = None
        self._actuator_task = None
        self._decision_expiry_task = None
        self._runtime_heartbeat_task = None
        self._resolution_sweep_task = None
        self._market_selector = None
        self._subscription_controller = None
        self._strategy_registry = None
        self._controller_runtimes = {}
        self._active_strategies_by_version = {}
        self._controller_signal_queues = {}
        self._controller_pipeline_tasks = {}
        self._controller_release_cancel_point = None
        self._reselection_task = None
        self._live_preflight_artifact_validated = False
        await self._close_pg_pool()

        if stop_error is not None:
            raise stop_error

    async def _reconcile_portfolio_from_db(self) -> None:
        """Rebuild the in-memory portfolio from persisted fills.

        On a fresh process start the default Runner portfolio is seeded from
        `risk.max_total_exposure`. Without this step, restarting in LIVE mode
        forgets all open Polymarket exposure: every new BUY decision sizes
        against the configured free budget while the venue actually holds real
        positions. Empirically observed during the 2026-04-25 PAPER soak — DB
        held 44 positions / $1984 locked while a fresh runner instance reported
        `locked_usdc=0`.

        Reconciliation aggregates fills via `FillStore.read_positions`,
        sums `locked_usdc`, and rebuilds the in-memory `Portfolio`. We
        clamp `free_usdc` to 0 if reconciled exposure exceeds
        `total_usdc` and surface a warning so operators tighten risk
        caps before resuming.
        """
        pool = self._pg_pool
        if pool is None:
            return
        try:
            persisted_positions = await self.fill_store.read_positions()
        except Exception as error:  # noqa: BLE001
            if self.config.mode == RunMode.LIVE:
                # Fail closed in LIVE: silent degradation here recreates
                # the exact restart-exposure bug this method exists to
                # prevent. Operator gets a clear signal via the
                # autostart_error path on /status (see api/app.py
                # lifespan). Do not start trading without a verified
                # baseline.
                detail = _runtime_error_detail(error, self.config)
                msg = (
                    "LIVE portfolio reconciliation failed "
                    f"({type(error).__name__}: {detail}); refusing to start "
                    "without a verified portfolio"
                )
                raise RuntimeError(msg) from error
            logger.warning("portfolio reconciliation failed: %s", error)
            return
        open_positions = [p for p in persisted_positions if p.shares_held > 0.0]
        if not open_positions:
            # Reset to baseline so a stop/start cycle on a runner that
            # previously held positions does not leave stale `locked_usdc`,
            # `free_usdc`, or `open_positions` in `self.portfolio`. Without
            # this, a second `start()` on the same Runner instance after
            # all positions closed would still report the old locked
            # exposure and undercount free budget, even though the DB
            # is empty.
            total_budget = self.portfolio.total_usdc
            self.portfolio = replace(
                self.portfolio,
                locked_usdc=0.0,
                free_usdc=total_budget,
                open_positions=[],
            )
            logger.info(
                "Reconciled portfolio from DB: 0 positions, $0.00 locked of $%.2f total",
                total_budget,
            )
            return
        total_locked = sum(position.locked_usdc for position in open_positions)
        total_budget = self.portfolio.total_usdc
        if total_locked > total_budget:
            logger.warning(
                "Reconciled locked exposure $%.2f exceeds total budget $%.2f"
                " — clamping free_usdc=0; review risk caps before resuming",
                total_locked,
                total_budget,
            )
            free_usdc = 0.0
        else:
            free_usdc = total_budget - total_locked
        self.portfolio = replace(
            self.portfolio,
            locked_usdc=total_locked,
            free_usdc=free_usdc,
            open_positions=open_positions,
        )
        logger.info(
            "Reconciled portfolio from DB: %d positions, $%.2f locked of $%.2f total",
            len(open_positions),
            total_locked,
            total_budget,
        )

    async def _reconcile_venue_account(self) -> None:
        if self.config.mode != RunMode.LIVE:
            return
        reconciler = self.venue_account_reconciler
        if reconciler is None:
            if not self.config.live_account_reconciliation_required:
                logger.warning("LIVE venue account reconciliation is not configured")
                return
            reconciler = PolymarketVenueAccountReconciler()
        credentials = validate_live_mode_ready(self.config)
        try:
            snapshot = await reconciler.snapshot(credentials)
            report = await reconciler.compare(self.portfolio, snapshot)
        except Exception as error:  # noqa: BLE001
            detail = _runtime_error_detail(error, self.config)
            msg = (
                "LIVE venue account reconciliation failed "
                f"({type(error).__name__}: {detail}); refusing to start "
                "without a verified venue account"
            )
            raise RuntimeError(msg) from error
        if not report.ok:
            raw_details = "; ".join(report.mismatches) or "unknown mismatch"
            details = _runtime_error_detail(RuntimeError(raw_details), self.config)
            msg = f"LIVE venue account reconciliation mismatch: {details}"
            raise RuntimeError(msg)

    async def _reselect(self) -> None:
        selector = self._market_selector
        subscription_controller = self._subscription_controller
        if selector is None or subscription_controller is None:
            return
        async with self._reselection_lock:
            self._reselection_generation += 1
            generation = self._reselection_generation
        # Serialize selector reads without holding the subscription lock; the
        # DB pool may wait here, while the bounded websocket update below does
        # not hold a pool-backed lock.
        async with self._market_selection_lock:
            result = await selector.select()
        protected_asset_ids = [
            *_open_position_token_ids(self.portfolio),
            *sorted(_selector_user_subscription_asset_ids(selector)),
        ]
        async with self._reselection_lock:
            if generation != self._reselection_generation:
                self._reselection_requested.set()
                return
            await subscription_controller.update(
                _cap_subscription_asset_ids(
                    list(result.asset_ids),
                    self.config,
                    protected_asset_ids=protected_asset_ids,
                )
            )

    async def _periodic_reselection_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._reselection_requested.wait(),
                    timeout=RESELECTION_INTERVAL_S,
                )
                self._reselection_requested.clear()
                if self._stop_event.is_set():
                    return
                try:
                    await self._reselect()
                except Exception as error:  # noqa: BLE001
                    logger.warning("periodic reselection failed: %s", error)
            except TimeoutError:
                if self._stop_event.is_set():
                    return
                try:
                    await self._reselect()
                except Exception as error:  # noqa: BLE001
                    logger.warning("periodic reselection failed: %s", error)

    def _clear_discovery_poll_complete_hook(self) -> None:
        discovery_sensor = _find_discovery_sensor(self._active_sensors)
        if discovery_sensor is not None:
            discovery_sensor.on_poll_complete = None

    async def _request_reselection(self) -> None:
        self._reselection_requested.set()

    async def _handle_discovery_poll_complete(self) -> None:
        try:
            await self._sync_controller_runtimes()
        except Exception as error:  # noqa: BLE001
            logger.warning("discovery-driven controller sync failed: %s", error)
        await self._request_reselection()

    async def _await_cancelled_task(
        self,
        task: asyncio.Task[None] | None,
    ) -> None:
        if task is None:
            return
        with suppress(asyncio.CancelledError):
            await task

    def _on_eval_record_for_calibration(self, record: EvalRecord) -> None:
        runtime = self._controller_runtimes.get(record.strategy_id)
        if (
            runtime is None
            or runtime.strategy_version_id != record.strategy_version_id
        ):
            # Stale-version records must not feed a new version's calibrator:
            # those outcomes never tested the new forecaster spec.
            logger.debug(
                "dropping eval record for calibration without a matching "
                "runtime: (%s, %s)",
                record.strategy_id,
                record.strategy_version_id,
            )
            return
        add_samples = _runtime_calibration_add_samples(runtime)
        if add_samples is None:
            return
        add_samples(record.model_id or "unknown", [record])

    async def _hydrate_runtime_calibration(
        self,
        runtime: StrategyControllerRuntime,
    ) -> None:
        add_samples = _runtime_calibration_add_samples(runtime)
        if add_samples is None:
            return
        records = await self.eval_store.all_for_strategy(
            runtime.strategy_id,
            runtime.strategy_version_id,
        )
        grouped: dict[str, list[EvalRecord]] = {}
        for record in records:
            grouped.setdefault(record.model_id or "unknown", []).append(record)
        for model_id, model_records in grouped.items():
            add_samples(model_id, model_records)
        logger.info(
            "calibration hydration loaded %d resolved eval records for %s@%s",
            len(records),
            runtime.strategy_id,
            runtime.strategy_version_id,
            extra={
                "event": "calibration_hydration",
                "strategy_id": runtime.strategy_id,
                "strategy_version_id": runtime.strategy_version_id,
                "resolved_records_by_model": {
                    model_id: len(model_records)
                    for model_id, model_records in grouped.items()
                },
            },
        )

    def _attach_controller_runtime(self, runtime: StrategyControllerRuntime) -> None:
        signal_queue: asyncio.Queue[MarketSignal] = asyncio.Queue()
        self._controller_runtimes[runtime.strategy_id] = runtime
        self._controller_signal_queues[runtime.strategy_id] = signal_queue
        task = asyncio.create_task(
            self._controller_pipeline_loop(runtime.strategy_id),
            name=f"controller-pipeline:{runtime.strategy_id}",
        )
        task.add_done_callback(self._capture_controller_pipeline_exception)
        # Watch-only: pipeline tasks already have capture/release/re-attach
        # machinery; the supervisor only tracks their health.
        self._supervisor.observe(
            task,
            name=f"controller-pipeline:{runtime.strategy_id}",
        )
        self._controller_pipeline_tasks[runtime.strategy_id] = task

    def _capture_controller_pipeline_exception(
        self,
        task: asyncio.Task[None],
    ) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and self._controller_pipeline_error is None:
            self._controller_pipeline_error = error

    async def _sync_controller_runtimes(self) -> None:
        if self.controller is not None or self._strategy_registry is None:
            return
        async with self._controller_lifecycle_lock:
            desired_runtimes = await self._build_controller_runtimes()
            desired_by_strategy = {
                runtime.strategy_id: runtime
                for runtime in desired_runtimes
            }
            current_strategy_ids = set(self._controller_runtimes)
            desired_strategy_ids = set(desired_by_strategy)
            if not current_strategy_ids and not desired_strategy_ids:
                return

            for strategy_id in sorted(current_strategy_ids - desired_strategy_ids):
                await self._release_controller_runtime_locked(strategy_id)

            for strategy_id in sorted(current_strategy_ids & desired_strategy_ids):
                current_runtime = self._controller_runtimes.get(strategy_id)
                desired_runtime = desired_by_strategy[strategy_id]
                if current_runtime is None:
                    continue
                if (
                    current_runtime.strategy_version_id != desired_runtime.strategy_version_id
                    or current_runtime.asset_ids != desired_runtime.asset_ids
                ):
                    await self._release_controller_runtime_locked(strategy_id)
                    # Attach before the hydration DB await: a record pushed
                    # by the calibration sink during that await would
                    # otherwise be dropped (no runtime registered) AND can
                    # miss the in-flight all_for_strategy query — a lost
                    # sample. Duplicate delivery is safe: the calibrator
                    # dedups per (model_id, decision_id). Attach-first also
                    # keeps the strategy attached if hydration fails.
                    self._attach_controller_runtime(desired_runtime)
                    await self._hydrate_runtime_calibration(desired_runtime)

            for strategy_id in sorted(desired_strategy_ids - current_strategy_ids):
                desired_runtime = desired_by_strategy[strategy_id]
                # Same attach-then-hydrate ordering as the replace branch:
                # the sink drops records for unattached runtimes, so the
                # push window exists here too (e.g. a sweep resolving fills
                # of a re-added strategy).
                self._attach_controller_runtime(desired_runtime)
                await self._hydrate_runtime_calibration(desired_runtime)

            await self._refresh_subscription_assets_locked()

    async def _release_controller_runtime(self, strategy_id: str) -> None:
        async with self._controller_lifecycle_lock:
            await self._release_controller_runtime_locked(strategy_id)

    async def _release_controller_runtime_locked(self, strategy_id: str) -> None:
        detached = self._detach_controller_runtime(strategy_id)
        if detached is None:
            return
        try:
            await self._finalize_controller_runtime_release_locked(
                detached,
                allow_cancel_injection=True,
            )
        except asyncio.CancelledError:
            await asyncio.shield(
                self._finalize_controller_runtime_release_locked(
                    detached,
                    allow_cancel_injection=False,
                )
            )
            raise
        except Exception:
            try:
                await asyncio.shield(
                    self._finalize_controller_runtime_release_locked(
                        detached,
                        allow_cancel_injection=False,
                    )
                )
            except Exception as cleanup_error:  # noqa: BLE001
                logger.warning(
                    "controller runtime cleanup retry failed for %s: %s",
                    detached.strategy_id,
                    cleanup_error,
                )
            raise

    def _detach_controller_runtime(
        self,
        strategy_id: str,
    ) -> DetachedControllerRuntime | None:
        runtime = self._controller_runtimes.pop(strategy_id, None)
        queue = self._controller_signal_queues.pop(strategy_id, None)
        task = self._controller_pipeline_tasks.pop(strategy_id, None)
        if runtime is None and queue is None and task is None:
            return None
        current_task = asyncio.current_task()
        if task is not None and task is not current_task and not task.done():
            task.cancel()
        return DetachedControllerRuntime(
            strategy_id=strategy_id,
            runtime=runtime,
            queue=queue,
            task=task,
        )

    async def _finalize_controller_runtime_release_locked(
        self,
        detached: DetachedControllerRuntime,
        *,
        allow_cancel_injection: bool,
    ) -> None:
        if allow_cancel_injection:
            await self._maybe_inject_controller_release_cancel(
                "before_first_cleanup_await"
            )
        current_task = asyncio.current_task()
        if detached.task is not None and detached.task is not current_task:
            await self._await_cancelled_task(detached.task)
        if allow_cancel_injection:
            await self._maybe_inject_controller_release_cancel(
                "between_cleanup_awaits"
            )
        self._drain_controller_signal_queue(detached.queue)
        await self._refresh_subscription_assets_locked()
        if allow_cancel_injection:
            await self._maybe_inject_controller_release_cancel(
                "after_last_cleanup_await"
            )

    async def _refresh_subscription_assets_locked(self) -> None:
        subscription_controller = self._subscription_controller
        if subscription_controller is None:
            return
        scoped_asset_ids: list[frozenset[str]] = []
        for runtime in self._controller_runtimes.values():
            if runtime.asset_ids is None:
                return
            scoped_asset_ids.append(runtime.asset_ids)
        merged_asset_ids = sorted(
            {
                asset_id
                for asset_ids in scoped_asset_ids
                for asset_id in asset_ids
            }
        )
        selector = self._market_selector
        async with self._reselection_lock:
            self._reselection_generation += 1
            await subscription_controller.update(
                _cap_subscription_asset_ids(
                    merged_asset_ids,
                    self.config,
                    protected_asset_ids=[
                        *_open_position_token_ids(self.portfolio),
                        *(
                            sorted(_selector_user_subscription_asset_ids(selector))
                            if selector is not None
                            else []
                        ),
                    ],
                )
            )

    async def _maybe_inject_controller_release_cancel(
        self,
        point: ControllerReleaseCancelPoint,
    ) -> None:
        if self._controller_release_cancel_point == point:
            self._controller_release_cancel_point = None
            raise asyncio.CancelledError

    def _drain_controller_signal_queue(
        self,
        queue: asyncio.Queue[MarketSignal] | None,
    ) -> None:
        if queue is None:
            return
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            queue.task_done()

    async def _configure_controllers(self) -> None:
        if self._controller_task is not None or self._controller_pipeline_tasks:
            msg = "Runner controllers already configured"
            raise RuntimeError(msg)
        runtimes = await self._build_controller_runtimes()
        for runtime in runtimes:
            await self._hydrate_runtime_calibration(runtime)
            self._attach_controller_runtime(runtime)

    async def _build_controller_runtimes(self) -> list[StrategyControllerRuntime]:
        self._active_strategies_by_version = {}
        if self.controller is not None:
            strategy = getattr(self.controller, "strategy", None)
            if isinstance(strategy, ActiveStrategy):
                self._active_strategies_by_version = {
                    (strategy.strategy_id, strategy.strategy_version_id): strategy
                }
            return [
                StrategyControllerRuntime(
                    strategy_id="default",
                    strategy_version_id="default-v1",
                    controller=self.controller,
                    asset_ids=None,
                )
            ]

        if self._should_build_strategy_registry_controllers():
            registry = self._strategy_registry
            if registry is None:
                msg = "Runner strategy registry is not initialized"
                raise RuntimeError(msg)
            active_strategies = await registry.list_active_strategies()
            if not active_strategies:
                return []
            self._active_strategies_by_version = {
                (strategy.strategy_id, strategy.strategy_version_id): strategy
                for strategy in active_strategies
            }
            self._require_live_active_strategies_match_preflight(active_strategies)
            scopes: dict[str, frozenset[str]] = {}
            if (
                self._market_selector is not None
                and hasattr(self._market_selector, "select_per_strategy")
            ):
                selections = await self._market_selector.select_per_strategy()
                scopes = {
                    selection.strategy_id: selection.asset_ids
                    for selection in selections
                }
            pipelines = self._controller_factory.build_many(active_strategies)
            return [
                StrategyControllerRuntime(
                    strategy_id=strategy.strategy_id,
                    strategy_version_id=strategy.strategy_version_id,
                    controller=pipelines[strategy.strategy_id],
                    asset_ids=scopes.get(strategy.strategy_id),
                )
                for strategy in active_strategies
            ]

        default_strategy = _default_active_strategy(self.config)
        self._active_strategies_by_version = {
            (default_strategy.strategy_id, default_strategy.strategy_version_id): (
                default_strategy
            )
        }
        pipeline = self._controller_factory.build(default_strategy)
        return [
            StrategyControllerRuntime(
                strategy_id=default_strategy.strategy_id,
                strategy_version_id=default_strategy.strategy_version_id,
                controller=pipeline,
                asset_ids=None,
            )
        ]

    def _should_build_strategy_registry_controllers(self) -> bool:
        registry = self._strategy_registry
        if registry is None or not hasattr(registry, "list_active_strategies"):
            return False
        if self.config.mode == RunMode.LIVE:
            return True
        if (
            self._market_selector is not None
            and hasattr(self._market_selector, "select_per_strategy")
        ):
            return True
        return not self._owns_pg_pool

    def _require_live_active_strategies_match_preflight(
        self,
        active_strategies: Sequence[ActiveStrategy],
    ) -> None:
        if self.config.mode != RunMode.LIVE:
            return
        expected = self._live_preflight_active_strategies_fingerprint
        if expected is None:
            return
        observed = live_preflight_active_strategies_fingerprint(active_strategies)
        if observed == expected:
            return
        self._live_preflight_artifact_validated = False
        self._live_preflight_active_strategies_fingerprint = None
        self._live_trading_suspended_reason = "active_strategy_preflight_mismatch"
        self.actuator_executor = self._build_executor(self.config.mode)
        msg = (
            "LIVE active strategies changed after credentialed preflight; "
            "rerun pms-live preflight before live trading"
        )
        raise LiveTradingDisabledError(msg)


@dataclass(frozen=True)
class _PostgresFlbMarketSnapshotReader:
    store: PostgresMarketDataStore

    async def latest(
        self,
        market_id: str,
        *,
        as_of: datetime,
    ) -> FlbMarketSnapshot | None:
        async with self.store.pool.acquire() as connection:
            resolved_market_id = await _resolve_flb_market_id(connection, market_id)
            if resolved_market_id is None:
                return None
            market_row = await _read_flb_market_row(connection, resolved_market_id)
            if market_row is None:
                return None
            price_row = await _read_flb_price_row(
                connection,
                resolved_market_id,
                as_of=as_of,
            )

        if _future_market_status_change(market_row, as_of=as_of):
            return None
        if market_row["closed"] is True or market_row["accepting_orders"] is False:
            return None

        yes_price_source = price_row if price_row is not None else market_row
        observed_at = _observed_at(yes_price_source)
        if observed_at is None or observed_at > as_of:
            return None

        yes_price = _open_probability_or_none(yes_price_source["yes_price"])
        yes_token_id = market_row["yes_token_id"]
        no_token_id = market_row["no_token_id"]
        if (
            yes_price is None
            or not isinstance(yes_token_id, str)
            or not isinstance(no_token_id, str)
        ):
            return None

        return FlbMarketSnapshot(
            market_id=cast(str, market_row["market_id"]),
            title=cast(str, market_row["question"]),
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            yes_price=yes_price,
            yes_best_ask=_open_probability_or_none(yes_price_source["best_ask"]),
            no_best_ask=_no_best_ask(
                no_price=yes_price_source["no_price"],
                yes_best_bid=yes_price_source["best_bid"],
            ),
            observed_at=observed_at,
            resolves_at=cast(datetime | None, market_row["resolves_at"]),
            venue=cast(Venue, market_row["venue"]),
        )


@dataclass(frozen=True)
class _PostgresRippleMarketSnapshotReader:
    store: PostgresMarketDataStore

    async def latest(
        self,
        market_id: str,
        *,
        as_of: datetime,
    ) -> RippleMarketSnapshot | None:
        async with self.store.pool.acquire() as connection:
            resolved_market_id = await _resolve_flb_market_id(connection, market_id)
            if resolved_market_id is None:
                return None
            market_row = await _read_flb_market_row(connection, resolved_market_id)
            if market_row is None:
                return None
            price_row = await _read_flb_price_row(
                connection,
                resolved_market_id,
                as_of=as_of,
            )

        if _future_market_status_change(market_row, as_of=as_of):
            return None
        if market_row["closed"] is True or market_row["accepting_orders"] is False:
            return None

        yes_price_source = price_row if price_row is not None else market_row
        observed_at = _observed_at(yes_price_source)
        if observed_at is None or observed_at > as_of:
            return None

        yes_price = _open_probability_or_none(yes_price_source["yes_price"])
        yes_token_id = market_row["yes_token_id"]
        if yes_price is None or not isinstance(yes_token_id, str):
            return None

        return RippleMarketSnapshot(
            market_id=cast(str, market_row["market_id"]),
            title=cast(str, market_row["question"]),
            token_id=yes_token_id,
            yes_price=yes_price,
            best_bid=_open_probability_or_none(yes_price_source["best_bid"]),
            best_ask=_open_probability_or_none(yes_price_source["best_ask"]),
            observed_at=observed_at,
            resolves_at=cast(datetime | None, market_row["resolves_at"]),
            venue=cast(Venue, market_row["venue"]),
        )


async def _resolve_flb_market_id(connection: Any, lookup_id: str) -> str | None:
    query = """
    WITH matches AS (
        SELECT markets.condition_id AS market_id, 0 AS match_rank
        FROM markets
        WHERE markets.condition_id = $1

        UNION ALL

        SELECT tokens.condition_id AS market_id, 1 AS match_rank
        FROM tokens
        WHERE tokens.token_id = $1
    )
    SELECT market_id
    FROM matches
    ORDER BY match_rank ASC, market_id ASC
    LIMIT 1
    """
    value = await connection.fetchval(query, lookup_id)
    if not isinstance(value, str):
        return None
    return value


async def _read_flb_market_row(connection: Any, market_id: str) -> Any | None:
    query = """
        SELECT
            markets.condition_id AS market_id,
            markets.question,
            markets.venue,
            markets.resolves_at,
            markets.last_seen_at,
            markets.yes_price,
            markets.no_price,
            markets.best_bid,
            markets.best_ask,
            markets.price_updated_at,
            markets.closed,
            markets.accepting_orders,
            markets.status_updated_at,
            MAX(CASE WHEN tokens.outcome = 'YES' THEN tokens.token_id END) AS yes_token_id,
            MAX(CASE WHEN tokens.outcome = 'NO' THEN tokens.token_id END) AS no_token_id
        FROM markets
        LEFT JOIN tokens
            ON tokens.condition_id = markets.condition_id
        WHERE markets.condition_id = $1
        GROUP BY
            markets.condition_id,
            markets.question,
            markets.venue,
            markets.resolves_at,
            markets.last_seen_at,
            markets.yes_price,
            markets.no_price,
            markets.best_bid,
            markets.best_ask,
            markets.price_updated_at,
            markets.closed,
            markets.accepting_orders,
            markets.status_updated_at
        """
    return await connection.fetchrow(query, market_id)


async def _read_flb_price_row(
    connection: Any,
    market_id: str,
    *,
    as_of: datetime,
) -> Any | None:
    query = """
    SELECT
        snapshot_at,
        yes_price,
        no_price,
        best_bid,
        best_ask
    FROM market_price_snapshots
    WHERE condition_id = $1
      AND snapshot_at <= $2
    ORDER BY snapshot_at DESC
    LIMIT 1
    """
    return await connection.fetchrow(query, market_id, as_of)


def _observed_at(row: Any) -> datetime | None:
    observed_at = row["snapshot_at"] if "snapshot_at" in row else row["price_updated_at"]
    if observed_at is None and "last_seen_at" in row:
        observed_at = row["last_seen_at"]
    if isinstance(observed_at, datetime):
        return observed_at
    return None


def _future_market_status_change(row: Any, *, as_of: datetime) -> bool:
    status_updated_at = row["status_updated_at"]
    return isinstance(status_updated_at, datetime) and status_updated_at > as_of


def _open_probability_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0 or parsed >= 1.0:
        return None
    return parsed


def _no_best_ask(
    *,
    no_price: object,
    yes_best_bid: object,
) -> float | None:
    parsed_no_price = _open_probability_or_none(no_price)
    if parsed_no_price is not None:
        return parsed_no_price
    parsed_yes_best_bid = _open_probability_or_none(yes_best_bid)
    if parsed_yes_best_bid is None:
        return None
    return _open_probability_or_none(1.0 - parsed_yes_best_bid)


def _append_bounded(items: list[T], item: T) -> None:
    items.append(item)
    overflow = len(items) - RUNNER_STATE_LIMIT
    if overflow > 0:
        del items[:overflow]


def _first_live_order_gate(
    settings: PMSSettings,
) -> DenyFirstLiveOrderGate | FileFirstLiveOrderGate:
    approval_path = settings.polymarket.first_live_order_approval_path
    if approval_path is None or approval_path.strip() == "":
        return DenyFirstLiveOrderGate()
    return FileFirstLiveOrderGate(
        Path(approval_path),
        require_approver_sidecar=settings.mode == RunMode.LIVE,
        approval_max_age_s=settings.polymarket.operator_approval_max_age_s,
    )


def _controller_diagnostic(controller: object) -> ControllerDiagnostic | None:
    diagnostic = getattr(controller, "last_diagnostic", None)
    if isinstance(diagnostic, ControllerDiagnostic):
        return diagnostic
    return None


def _find_discovery_sensor(
    sensors: Sequence[ISensor],
) -> DiscoveryPollCompleteSensor | None:
    for sensor in sensors:
        if hasattr(sensor, "on_poll_complete"):
            return cast(DiscoveryPollCompleteSensor, sensor)
    return None


def _find_subscription_sink(
    sensors: Sequence[ISensor],
) -> SubscriptionManagedSensor | None:
    for sensor in sensors:
        update_subscription = getattr(sensor, "update_subscription", None)
        if callable(update_subscription):
            return cast(SubscriptionManagedSensor, sensor)
    return None


def _selector_user_subscription_asset_ids(
    selector: MarketSelectorLike,
) -> frozenset[str]:
    value: object = getattr(
        selector,
        "last_user_subscription_asset_ids",
        frozenset(),
    )
    if isinstance(value, frozenset):
        return cast(frozenset[str], value)
    return frozenset(cast(Sequence[str], value))


def _matches_strategy_scope(
    asset_ids: frozenset[str] | None,
    signal: MarketSignal,
) -> bool:
    if asset_ids is None:
        return True
    candidate_id = signal.token_id or signal.market_id
    return candidate_id in asset_ids


def _default_active_strategy(settings: PMSSettings) -> ActiveStrategy:
    config = StrategyConfig(
        strategy_id="default",
        factor_composition=DEFAULT_V2_FACTOR_COMPOSITION,
        metadata=(
            ("owner", "system"),
            ("tier", "default"),
            ("live_allowed", "true"),
            ("alpha_source", "warehouse_flb_decile_model_v1"),
            ("edge_model_source", "paper_soak_net_edge_model_v1"),
            ("calibration_source", "paper_soak_eval_records_v1"),
            ("evidence_source", "paper_soak_go_report_v1"),
        ),
    )
    risk = RiskParams(
        max_position_notional_usdc=settings.risk.max_position_per_market,
        max_daily_drawdown_pct=(
            0.0 if settings.risk.max_drawdown_pct is None else settings.risk.max_drawdown_pct
        ),
        min_order_size_usdc=settings.risk.min_order_usdc,
    )
    eval_spec = EvalSpec(metrics=("brier", "pnl", "fill_rate"))
    forecaster = ForecasterSpec(
        forecasters=(
            ("rules", (("threshold", "0.55"),)),
            ("stats", (("window", "15m"),)),
            ("llm", ()),
        )
    )
    market_selection = MarketSelectionSpec(
        venue="polymarket",
        resolution_time_max_horizon_days=(
            DEFAULT_OPERATIONAL_MARKET_SELECTION_HORIZON_DAYS
        ),
        volume_min_usdc=500.0,
    )
    strategy_version_id = compute_strategy_version_id(
        config,
        risk,
        eval_spec,
        forecaster,
        market_selection,
        llm_identity=(
            settings.llm.provider,
            settings.llm.model,
            settings.llm.prompt_version,
        ),
    )
    return ActiveStrategy(
        strategy_id="default",
        strategy_version_id=strategy_version_id,
        config=config,
        risk=risk,
        eval_spec=eval_spec,
        forecaster=forecaster,
        market_selection=market_selection,
        calibration=CalibrationSpec(enabled=True),
    )


def _operational_default_market_selection(
    market_selection: MarketSelectionSpec,
) -> MarketSelectionSpec:
    if market_selection.resolution_time_max_horizon_days != 7:
        return market_selection
    return replace(
        market_selection,
        resolution_time_max_horizon_days=(
            DEFAULT_OPERATIONAL_MARKET_SELECTION_HORIZON_DAYS
        ),
    )


def _cap_subscription_asset_ids(
    asset_ids: list[str],
    settings: PMSSettings,
    *,
    protected_asset_ids: Sequence[str] = (),
) -> list[str]:
    protected = list(dict.fromkeys(protected_asset_ids))
    protected_set = set(protected)
    selected = [
        asset_id
        for asset_id in dict.fromkeys(asset_ids)
        if asset_id not in protected_set
    ]
    merged_asset_ids = [*protected, *selected]
    limit = settings.sensor.max_subscription_asset_ids
    if limit is None or len(merged_asset_ids) <= limit:
        return merged_asset_ids
    dropped_asset_ids = merged_asset_ids[limit:]
    logger.warning(
        "subscription asset set capped at %d of %d selected assets; "
        "%d protected assets; dropped=%s",
        limit,
        len(merged_asset_ids),
        len(protected),
        dropped_asset_ids,
    )
    return merged_asset_ids[:limit]


def _open_position_token_ids(portfolio: Portfolio) -> list[str]:
    return list(
        dict.fromkeys(
            position.token_id
            for position in portfolio.open_positions
            if position.token_id is not None
        )
    )


def _open_position_keys(portfolio: Portfolio) -> set[PositionSlotKey]:
    return {
        _position_key_from_position(position)
        for position in portfolio.open_positions
    }


def _estimated_decision_quantity(decision: TradeDecision) -> float | None:
    limit_price = Decimal(str(decision.limit_price))
    if limit_price <= Decimal("0"):
        return None
    notional_usdc = Decimal(str(decision.notional_usdc))
    return float(notional_usdc / limit_price)


def _position_key_from_position(position: Position) -> PositionSlotKey:
    return (
        position.market_id,
        position.token_id,
        position.venue,
        position.side,
        position.strategy_id,
        position.strategy_version_id,
    )


def _position_key_from_decision(decision: TradeDecision) -> PositionSlotKey:
    return (
        decision.market_id,
        decision.token_id,
        decision.venue,
        decision.side,
        decision.strategy_id,
        decision.strategy_version_id,
    )


def _position_exit_reentry_key_from_decision(
    decision: TradeDecision,
) -> PositionExitReentryKey:
    return (
        decision.strategy_id,
        decision.strategy_version_id,
        decision.market_id,
        decision.token_id,
        decision.risk_group_id,
    )


def _is_position_exit_decision(decision: TradeDecision) -> bool:
    return any(
        condition.startswith("position_exit:")
        for condition in decision.stop_conditions
    )


def _decision_reduces_existing_position(
    portfolio: Portfolio,
    decision: TradeDecision,
) -> bool:
    return any(
        position.market_id == decision.market_id
        and position.token_id == decision.token_id
        and position.venue == decision.venue
        and position.strategy_id == decision.strategy_id
        and position.strategy_version_id == decision.strategy_version_id
        and position.side != decision.side
        for position in portfolio.open_positions
    )


def _build_discovery_http_client(settings: PMSSettings) -> httpx.AsyncClient:
    sensor = settings.sensor
    return httpx.AsyncClient(
        base_url="https://gamma-api.polymarket.com",
        timeout=httpx.Timeout(
            timeout=sensor.discovery_http_timeout_s,
            pool=sensor.discovery_http_pool_timeout_s,
        ),
        limits=httpx.Limits(
            max_connections=sensor.discovery_http_max_connections,
            max_keepalive_connections=sensor.discovery_http_max_keepalive_connections,
            keepalive_expiry=sensor.discovery_http_keepalive_expiry_s,
        ),
        trust_env=False,
    )


def _portfolio_with_configured_budget(
    portfolio: Portfolio,
    *,
    total_budget_usdc: float,
) -> Portfolio:
    if not _is_default_runner_portfolio(portfolio):
        return portfolio
    return replace(
        portfolio,
        total_usdc=total_budget_usdc,
        free_usdc=total_budget_usdc,
    )


def _is_default_runner_portfolio(portfolio: Portfolio) -> bool:
    return (
        portfolio.total_usdc == 1000.0
        and portfolio.free_usdc == 1000.0
        and portfolio.locked_usdc == 0.0
        and portfolio.open_positions == []
        and portfolio.max_drawdown_pct is None
        and portfolio.max_open_positions is None
    )


def _fill_from_order(
    order_state: OrderState,
    decision: TradeDecision,
    signal: MarketSignal | None,
    *,
    fee_rate: float | None = None,
) -> FillRecord | None:
    # Emit a FillRecord whenever the venue reports a non-zero positive
    # fill, regardless of the order's terminal status. Polymarket can
    # return PARTIAL/LIVE statuses with positive `filled_notional_usdc`
    # (e.g. an IOC limit that filled half its size and cancelled the
    # rest). Pre-fix, only `MATCHED` produced a FillRecord — every
    # partial fill was silently dropped from the inner ring, breaking
    # both portfolio accounting and evaluator metrics.
    if order_state.fill_price is None or order_state.fill_price <= 0.0:
        return None
    if order_state.filled_notional_usdc <= 0.0:
        return None

    # Simulated (paper/backtest) fills carry the modelled Polymarket fee so the
    # evaluator's net-edge gate can compute instead of rendering N/A, and so
    # paper P&L is not inflated relative to live. When signal metadata carries
    # market fee evidence, prefer it over the global fallback. fee_rate is None
    # for LIVE fills — venue reconciliation owns real fees there, and we must not
    # synthesize over it. Formula matches ExecutionModel.compute_fee.
    fees: float | None = None
    fee_bps: int | None = None
    if fee_rate is not None:
        effective_fee_rate = (
            market_fee_rate_from_metadata(
                signal.external_signal,
                fallback_rate=fee_rate,
            )
            if signal is not None
            else fee_rate
        )
        fees = (
            order_state.filled_notional_usdc
            * effective_fee_rate
            * (1.0 - order_state.fill_price)
        )
        fee_bps = int(effective_fee_rate * 10_000)

    return FillRecord(
        trade_id=order_state.order_id,
        order_id=order_state.order_id,
        decision_id=decision.decision_id,
        market_id=order_state.market_id,
        token_id=order_state.token_id,
        venue=order_state.venue,
        side=decision.action if decision.action is not None else decision.side,
        fill_price=order_state.fill_price,
        fill_notional_usdc=order_state.filled_notional_usdc,
        fill_quantity=order_state.filled_quantity,
        executed_at=order_state.submitted_at,
        filled_at=order_state.last_updated_at,
        status=order_state.status,
        anomaly_flags=[],
        strategy_id=decision.strategy_id,
        strategy_version_id=decision.strategy_version_id,
        resolved_outcome=_resolved_outcome(signal),
        risk_group_id=decision.risk_group_id,
        fees=fees,
        fee_bps=fee_bps,
    )


def _is_selection_funnel_entry_fill(fill: FillRecord) -> bool:
    return fill.side == Side.BUY.value


def _decision_status_from_order(order_state: OrderState) -> str:
    normalized_status = order_state.status.lower()
    raw_status = order_state.raw_status.lower()
    if (
        order_state.filled_notional_usdc > 0.0
        and order_state.remaining_notional_usdc > 1e-9
    ):
        return "partially_filled"
    if order_state.filled_notional_usdc > 0.0:
        return "filled"
    if normalized_status == OrderStatus.MATCHED.value:
        return "filled"
    if normalized_status in {
        OrderStatus.CANCELLED.value,
        OrderStatus.CANCELED_MARKET_RESOLVED.value,
        "canceled",
    }:
        return "cancelled"
    if raw_status == "venue_rejection" or normalized_status == "rejected":
        return "venue_rejected"
    if normalized_status == OrderStatus.INVALID.value:
        return "rejected"
    return "submitted"


def _is_open_order_state(order_state: OrderState) -> bool:
    if order_state.remaining_notional_usdc <= 1e-9:
        return False
    if is_terminal_partial_fill(order_state):
        return False
    normalized_status = order_state.status.lower()
    raw_status = order_state.raw_status.lower()
    return normalized_status in {
        OrderStatus.LIVE.value,
        OrderStatus.UNMATCHED.value,
        OrderStatus.PARTIAL.value,
        "open",
    } or raw_status in {
        OrderStatus.LIVE.value,
        OrderStatus.UNMATCHED.value,
        OrderStatus.PARTIAL.value,
        "open",
    }


def _portfolio_with_fill(portfolio: Portfolio, fill: FillRecord) -> Portfolio:
    positions = list(portfolio.open_positions)
    fill_size = fill.fill_notional_usdc
    contracts = _filled_contracts(fill)
    for index, position in enumerate(positions):
        if _same_position(position, fill):
            new_shares = position.shares_held + contracts
            avg_entry_price = (
                position.avg_entry_price * position.shares_held
                + fill.fill_price * contracts
            ) / new_shares
            positions[index] = replace(
                position,
                shares_held=new_shares,
                avg_entry_price=avg_entry_price,
                locked_usdc=position.locked_usdc + fill_size,
                current_price=fill.fill_price,
                risk_group_id=position.risk_group_id or fill.risk_group_id,
            )
            break
    else:
        for index, position in enumerate(positions):
            if _opposes_position(position, fill):
                closing_contracts = min(position.shares_held, contracts)
                released_locked = position.avg_entry_price * closing_contracts
                remaining_shares = position.shares_held - closing_contracts
                if remaining_shares <= 1e-9:
                    del positions[index]
                else:
                    positions[index] = replace(
                        position,
                        shares_held=remaining_shares,
                        locked_usdc=max(0.0, position.locked_usdc - released_locked),
                        current_price=fill.fill_price,
                    )
                return replace(
                    portfolio,
                    free_usdc=portfolio.free_usdc + fill_size,
                    locked_usdc=max(0.0, portfolio.locked_usdc - released_locked),
                    open_positions=positions,
                )
        positions.append(
            Position(
                market_id=fill.market_id,
                token_id=fill.token_id,
                venue=fill.venue,
                side=fill.side,
                shares_held=contracts,
                avg_entry_price=fill.fill_price,
                unrealized_pnl=0.0,
                locked_usdc=fill_size,
                current_price=fill.fill_price,
                opened_at=fill.filled_at,
                strategy_id=fill.strategy_id,
                strategy_version_id=fill.strategy_version_id,
                risk_group_id=fill.risk_group_id,
            )
        )
    return replace(
        portfolio,
        free_usdc=portfolio.free_usdc - fill_size,
        locked_usdc=portfolio.locked_usdc + fill_size,
        open_positions=positions,
    )


def _filled_contracts(fill: FillRecord) -> float:
    return fill.fill_quantity


def _signal_event_summary(signal: MarketSignal) -> str:
    return f"Signal {signal.market_id} @ {(signal.yes_price * 100):.1f}¢"


def _decision_event_summary(decision: TradeDecision) -> str:
    return f"Accepted {decision.side} ${decision.notional_usdc:.2f} on {decision.market_id}"


def _fill_event_summary(fill: FillRecord) -> str:
    return f"Filled {fill.side} ${fill.fill_notional_usdc:.2f} on {fill.market_id}"


def _same_position(position: Position, fill: FillRecord) -> bool:
    return (
        position.market_id == fill.market_id
        and position.token_id == fill.token_id
        and position.venue == fill.venue
        and position.side == fill.side
        and position.strategy_id == fill.strategy_id
        and position.strategy_version_id == fill.strategy_version_id
    )


def _opposes_position(position: Position, fill: FillRecord) -> bool:
    return (
        position.market_id == fill.market_id
        and position.token_id == fill.token_id
        and position.venue == fill.venue
        and position.side != fill.side
        and position.strategy_id == fill.strategy_id
        and position.strategy_version_id == fill.strategy_version_id
    )


def _resolved_outcome(signal: MarketSignal | None) -> float | None:
    if signal is None:
        return None
    raw_outcome = signal.external_signal.get("resolved_outcome")
    if raw_outcome is not None:
        return min(max(float(raw_outcome), 0.0), 1.0)
    return None


def _coerce_actuator_work_item(
    work_item: object,
) -> ActuatorWorkItem:
    if isinstance(work_item, ActuatorWorkItem):
        return work_item
    if isinstance(work_item, tuple) and len(work_item) == 2:
        decision, signal = work_item
        return ActuatorWorkItem(
            decision=cast(TradeDecision, decision),
            signal=cast(MarketSignal | None, signal),
            dedup_acquired=False,
        )
    if isinstance(work_item, tuple) and len(work_item) == 3:
        decision, signal, dedup_acquired = work_item
        return ActuatorWorkItem(
            decision=cast(TradeDecision, decision),
            signal=cast(MarketSignal | None, signal),
            dedup_acquired=bool(dedup_acquired),
        )
    msg = f"unsupported actuator work item: {type(work_item)!r}"
    raise TypeError(msg)


def _runtime_error_detail(error: BaseException, settings: PMSSettings) -> str:
    message = str(error)
    if settings.mode != RunMode.LIVE:
        return message
    return redact_live_error_values(
        message,
        (
            settings.polymarket.private_key,
            settings.polymarket.api_key,
            settings.polymarket.api_secret,
            settings.polymarket.api_passphrase,
            settings.polymarket.funder_address,
        ),
    )


async def _execute_actuator_work_item(
    executor: Any,
    decision: TradeDecision,
    portfolio: Portfolio,
    *,
    dedup_acquired: bool,
    paper_orderbook: Mapping[str, Any] | None = None,
) -> OrderState:
    executor = _executor_with_paper_orderbook_snapshot(
        executor,
        decision,
        paper_orderbook,
    )
    if dedup_acquired and _executor_accepts_dedup_acquired(executor):
        return cast(
            OrderState,
            await executor.execute(
                decision,
                portfolio,
                dedup_acquired=True,
            ),
        )
    return cast(OrderState, await executor.execute(decision, portfolio))


def _executor_with_paper_orderbook_snapshot(
    executor: Any,
    decision: TradeDecision,
    paper_orderbook: Mapping[str, Any] | None,
) -> Any:
    if (
        paper_orderbook is None
        or not isinstance(executor, ActuatorExecutor)
        or not isinstance(executor.adapter, PaperActuator)
    ):
        return executor
    return replace(
        executor,
        adapter=PaperActuator(
            orderbooks={
                _paper_orderbook_key(decision): cast(
                    dict[str, Any],
                    deepcopy(paper_orderbook),
                )
            }
        ),
    )


def _paper_orderbook_key(decision: TradeDecision) -> str:
    return decision.token_id or decision.market_id


def _paper_orderbook_for_decision(
    orderbooks: Mapping[str, dict[str, Any]],
    decision: TradeDecision,
) -> dict[str, Any] | None:
    if decision.token_id is not None and decision.token_id in orderbooks:
        return orderbooks[decision.token_id]
    if (
        (decision.outcome == "YES" or decision.token_id is None)
        and decision.market_id in orderbooks
    ):
        return orderbooks[decision.market_id]
    return None


def _executor_accepts_dedup_acquired(executor: Any) -> bool:
    execute = getattr(executor, "execute", None)
    if execute is None:
        return False
    try:
        parameters = inspect.signature(execute).parameters.values()
    except (TypeError, ValueError):
        return True

    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "dedup_acquired":
            return True
    return False


def _raw_factor_steps(
    steps: Sequence[FactorCompositionStep],
) -> tuple[FactorCompositionStep, ...]:
    raw_steps: list[FactorCompositionStep] = []
    for step in steps:
        role = getattr(step, "role", None)
        factor_id = getattr(step, "factor_id", None)
        if role not in RAW_FACTOR_ROLES:
            continue
        if factor_id not in REGISTERED_FACTOR_IDS:
            continue
        raw_steps.append(step)
    return tuple(raw_steps)


def _decision_evidence_from_signal(
    signal: MarketSignal,
    *,
    decision: TradeDecision,
    factor_snapshot_hash: str | None,
    quote_source: str,
    decision_created_at: datetime,
    spread_already_in_price: bool = False,
) -> dict[str, object]:
    book_top_levels = _top_orderbook_levels(signal.orderbook)
    book_token_outcome = _book_token_outcome(signal, decision)
    canonical_yes_price = _canonical_yes_probability(
        signal.yes_price,
        book_token_outcome=book_token_outcome,
    )
    return {
        "market_id": signal.market_id,
        "token_id": signal.token_id,
        "book_token_id": signal.token_id,
        "book_token_outcome": book_token_outcome,
        "decision_token_id": decision.token_id,
        "decision_outcome": decision.outcome,
        "decision_side": decision.side,
        "venue": signal.venue,
        "market_status": signal.market_status,
        "category": _optional_text(
            signal.external_signal.get("category")
        ) or _optional_text(signal.external_signal.get("market_category")),
        "signal_fetched_at": signal.fetched_at.isoformat(),
        "yes_price": canonical_yes_price,
        "direct_token_price": signal.yes_price,
        "book_top_levels": book_top_levels,
        "book_hash": _decision_book_hash(
            market_id=signal.market_id,
            token_id=signal.token_id,
            book_top_levels=book_top_levels,
        ),
        "book_depth": 5,
        "baseline_probability_coordinate": "YES",
        "market_implied_baseline_prob_estimate": (
            _market_implied_baseline_prob_estimate(decision)
        ),
        "decision_outcome_market_implied_prob_estimate": decision.limit_price,
        "mid_quote_baseline_prob_estimate": (
            _mid_quote_baseline_prob_estimate(
                book_top_levels,
                book_token_outcome=book_token_outcome,
            )
        ),
        "last_trade_baseline_prob_estimate": (
            _last_trade_baseline_prob_estimate(
                signal,
                book_token_outcome=book_token_outcome,
            )
        ),
        "category_prior_baseline_prob_estimate": (
            _category_prior_baseline_prob_estimate(signal)
        ),
        "book_age_ms": _decision_evidence_book_age_ms(
            signal,
            decision_created_at=decision_created_at,
        ),
        "decision_latency_ms": _signal_to_decision_latency_ms(
            signal,
            decision_created_at=decision_created_at,
        ),
        "quote_source": quote_source,
        "direct_outcome_book_source": _optional_text(
            signal.external_signal.get("direct_outcome_book_source")
        ),
        "factor_snapshot_hash": factor_snapshot_hash,
        "spread_bps_at_decision": decision.spread_bps_at_decision,
        **_decision_cost_evidence(
            signal,
            decision,
            spread_already_in_price=spread_already_in_price,
        ),
        "external_signal_keys": sorted(str(key) for key in signal.external_signal),
    }


def _decision_cost_evidence(
    signal: MarketSignal,
    decision: TradeDecision,
    *,
    spread_already_in_price: bool = False,
) -> dict[str, float | bool]:
    price = Decimal(str(decision.limit_price))
    if not price.is_finite() or price < 0:
        return {}

    evidence: dict[str, float | bool] = {}
    total_cost_edge = Decimal("0")
    has_cost_evidence = False

    fee_rate_bps = _optional_float(signal.external_signal.get("fee_rate_bps"))
    if fee_rate_bps is not None and 0.0 <= fee_rate_bps <= 10_000.0:
        fee_rate = Decimal(str(fee_rate_bps)) / Decimal("10000")
        fee_edge = fee_rate * (Decimal("1.0") - price)
        evidence["fee_rate_bps"] = fee_rate_bps
        evidence["fee_rate"] = float(fee_rate)
        evidence["fee_edge_at_decision"] = float(fee_edge)
        total_cost_edge += fee_edge
        has_cost_evidence = True

    spread_bps = _optional_float(decision.spread_bps_at_decision)
    if spread_already_in_price:
        # Ask-frame decisions already paid the spread inside expected_edge;
        # keep the key present (the h1_flb smoke check requires it) but
        # charge zero so net_edge_after_costs matches the gate arithmetic.
        spread_edge = Decimal("0")
        evidence["spread_edge_at_decision"] = float(spread_edge)
        evidence["spread_already_in_price"] = True
        total_cost_edge += spread_edge
        has_cost_evidence = True
    elif spread_bps is not None and spread_bps >= 0.0:
        spread_edge = (Decimal(str(spread_bps)) / Decimal("10000")) * price
        evidence["spread_edge_at_decision"] = float(spread_edge)
        total_cost_edge += spread_edge
        has_cost_evidence = True

    slippage_bps = _optional_float(decision.max_slippage_bps)
    if slippage_bps is not None and slippage_bps >= 0.0:
        slippage_edge = (Decimal(str(slippage_bps)) / Decimal("10000")) * price
        evidence["slippage_edge_at_decision"] = float(slippage_edge)
        total_cost_edge += slippage_edge
        has_cost_evidence = True

    if not has_cost_evidence:
        return evidence

    evidence["total_cost_edge_at_decision"] = float(total_cost_edge)
    expected_edge = Decimal(str(decision.expected_edge))
    if expected_edge.is_finite():
        evidence["net_edge_after_costs"] = float(expected_edge - total_cost_edge)
    return evidence


def _controller_execution_signal_for_decision(
    controller: object,
    *,
    fallback_signal: MarketSignal,
    decision: TradeDecision,
    latest_signals_by_token: Mapping[str, MarketSignal],
) -> MarketSignal:
    execution_signal = getattr(controller, "last_execution_signal", None)
    if (
        isinstance(execution_signal, MarketSignal)
        and execution_signal.market_id == decision.market_id
        and (
            decision.token_id is None
            or execution_signal.token_id is None
            or execution_signal.token_id == decision.token_id
        )
    ):
        return execution_signal
    return _decision_evidence_signal_for_decision(
        fallback_signal,
        decision,
        latest_signals_by_token=latest_signals_by_token,
    )


def _decision_evidence_signal_for_decision(
    signal: MarketSignal,
    decision: TradeDecision,
    *,
    latest_signals_by_token: Mapping[str, MarketSignal],
) -> MarketSignal:
    target_token_id = decision.token_id
    if target_token_id is None or target_token_id == signal.token_id:
        return signal
    candidate = latest_signals_by_token.get(target_token_id)
    if candidate is None or candidate.market_id != signal.market_id:
        return signal
    return candidate


def _paper_orderbook_snapshot_for_decision(
    decision: TradeDecision,
    signal: MarketSignal | None,
    *,
    latest_signals_by_token: Mapping[str, MarketSignal],
) -> dict[str, Any] | None:
    if signal is None:
        return None
    source_signal = signal
    if decision.token_id is not None and decision.token_id != signal.token_id:
        candidate = latest_signals_by_token.get(decision.token_id)
        if candidate is None or candidate.market_id != decision.market_id:
            return None
        source_signal = candidate
    return deepcopy(source_signal.orderbook)


def _decision_evidence_book_age_ms(
    signal: MarketSignal,
    *,
    decision_created_at: datetime,
) -> float | None:
    explicit_age = _optional_float(signal.external_signal.get("book_age_ms"))
    if explicit_age is not None:
        return explicit_age
    book_received_at = _external_signal_datetime(
        signal.external_signal.get("book_received_at")
    )
    if book_received_at is None:
        return None
    age_ms = (
        _aware_utc(decision_created_at) - _aware_utc(book_received_at)
    ).total_seconds() * 1000.0
    return max(0.0, age_ms)


def _controller_signal_coalesce_key(signal: MarketSignal) -> tuple[str, str]:
    signal_id = signal.token_id if signal.token_id is not None else signal.market_id
    return signal.venue, signal_id


def _is_newer_controller_signal(
    candidate: MarketSignal,
    current: MarketSignal,
) -> bool:
    return _controller_signal_event_time(candidate) >= _controller_signal_event_time(
        current
    )


def _controller_signal_event_time(signal: MarketSignal) -> datetime:
    book_received_at = _external_signal_datetime(
        signal.external_signal.get("book_received_at")
    )
    return _aware_utc(book_received_at or signal.fetched_at)


def _controller_can_refresh_queued_book(controller: object) -> bool:
    return getattr(controller, "direct_book_reader", None) is not None


def _controller_direct_book_reader(
    settings: PMSSettings,
    market_data_store: PostgresMarketDataStore,
) -> PostgresMarketDataStore | RefreshingDirectBookSnapshotReader:
    if not settings.controller.venue_book_refresh_enabled:
        return market_data_store
    return RefreshingDirectBookSnapshotReader(
        primary=market_data_store,
        venue_client=PolymarketPublicBookClient(
            host=settings.polymarket.host,
            timeout_s=settings.controller.venue_book_refresh_timeout_s,
        ),
        max_snapshot_age_ms=settings.controller.max_book_age_ms,
        allowed_clock_skew_ms=settings.controller.allowed_book_clock_skew_ms,
    )


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _external_signal_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    raw_value = value.strip()
    if not raw_value:
        return None
    if raw_value.endswith("Z"):
        raw_value = f"{raw_value[:-1]}+00:00"
    try:
        return datetime.fromisoformat(raw_value)
    except ValueError:
        return None


def _category_prior_estimator_from_settings(
    settings: PMSSettings,
) -> CategoryPriorBaselineEstimator | None:
    raw_path = settings.controller.category_prior_observations_path
    if raw_path is None or raw_path.strip() == "":
        return None

    path = Path(raw_path).expanduser()
    require_path_outside_working_tree(path, label="category-prior artifact")
    require_private_parent(path, label="category-prior artifact")
    loaded = load_category_prior_observations_csv(path)
    logger.info(
        "loaded category prior observations",
        extra={
            "event": "category_prior_observations_loaded",
            "path": raw_path,
            "observation_count": len(loaded.observations),
            "skipped_ambiguous_count": loaded.skipped_ambiguous_count,
        },
    )
    return CategoryPriorBaselineEstimator(
        observations=loaded.observations,
        min_category_samples=settings.controller.category_prior_min_category_samples,
        min_global_samples=settings.controller.category_prior_min_global_samples,
        smoothing_alpha=settings.controller.category_prior_smoothing_alpha,
        smoothing_beta=settings.controller.category_prior_smoothing_beta,
    )


def _flb_calibration_model_from_settings(
    settings: PMSSettings,
) -> FlbCalibrationModel | None:
    raw_path = settings.strategies.flb_calibration_path
    if raw_path is None or raw_path.strip() == "":
        return None

    path = Path(raw_path).expanduser()
    require_path_outside_working_tree(path, label="FLB calibration artifact")
    require_private_parent(path, label="FLB calibration artifact")
    require_path_outside_working_tree(
        flb_calibration_provenance_path(path),
        label="FLB calibration provenance JSON",
    )
    require_private_parent(
        flb_calibration_provenance_path(path),
        label="FLB calibration provenance JSON",
    )
    model = load_flb_calibration_csv(
        path,
        min_sample_count=settings.strategies.flb_min_calibration_samples,
    )
    require_flb_calibration_provenance_for_model(path, model=model)
    logger.info(
        "loaded FLB calibration model",
        extra={
            "event": "flb_calibration_model_loaded",
            "path": raw_path,
            "signal_count": len(model.calibrations),
            "min_sample_count": model.min_sample_count,
        },
    )
    return model


def _h1_flb_strategy_from_settings(
    settings: PMSSettings,
    *,
    calibration_model: FlbCalibrationModel,
) -> Strategy:
    raw_path = settings.strategies.flb_calibration_path
    if raw_path is None or raw_path.strip() == "":
        msg = "h1_flb strategy requires strategies.flb_calibration_path"
        raise ValueError(msg)
    provenance = require_flb_calibration_provenance_for_model(
        Path(raw_path).expanduser(),
        model=calibration_model,
    )
    return build_h1_flb_strategy(
        calibration_csv_sha256=provenance.calibration_csv_sha256,
        calibration_source_label=provenance.calibration_source_label,
        time_in_force=settings.controller.time_in_force,
    )


def _verify_runtime_strategy_version(
    strategy: Strategy,
    strategy_version_id: str,
    *,
    legacy_ids: Collection[str],
) -> str:
    expected_version_id = compute_strategy_version_id(*strategy.snapshot())
    if strategy_version_id == expected_version_id:
        return expected_version_id
    if strategy_version_id in legacy_ids:
        logger.warning(
            "legacy runtime strategy version accepted with economics drift",
            extra={
                "event": "strategy_economics_version_drift",
                "strategy_id": strategy.config.strategy_id,
                "strategy_version_id": strategy_version_id,
                "expected_version_id": expected_version_id,
            },
        )
        return strategy_version_id
    msg = (
        "strategy_version_id does not match runtime strategy projection: "
        f"row={strategy_version_id!r}, expected={expected_version_id!r}"
    )
    raise ValueError(msg)


def _market_implied_baseline_prob_estimate(decision: TradeDecision) -> float:
    if decision.outcome == "NO":
        return 1.0 - decision.limit_price
    return decision.limit_price


def _book_token_outcome(signal: MarketSignal, decision: TradeDecision) -> str:
    raw_outcome = str(signal.external_signal.get("signal_token_outcome", "")).upper()
    if raw_outcome in {"YES", "NO"}:
        return raw_outcome
    if decision.outcome == "NO" and decision.token_id == signal.token_id:
        return "NO"
    return "YES"


def _mid_quote_baseline_prob_estimate(
    book_top_levels: dict[str, list[dict[str, float]]],
    *,
    book_token_outcome: str = "YES",
) -> float | None:
    bids = book_top_levels["bids"]
    asks = book_top_levels["asks"]
    if not bids or not asks:
        return None
    return _canonical_yes_probability(
        (bids[0]["price"] + asks[0]["price"]) / 2.0,
        book_token_outcome=book_token_outcome,
    )


def _last_trade_baseline_prob_estimate(
    signal: MarketSignal,
    *,
    book_token_outcome: str = "YES",
) -> float | None:
    for value in (
        signal.external_signal.get("last_trade_price"),
        signal.orderbook.get("last_trade_price"),
    ):
        probability = _canonical_yes_probability(
            value,
            book_token_outcome=book_token_outcome,
        )
        if probability is not None:
            return probability
    return None


def _canonical_yes_probability(
    value: object,
    *,
    book_token_outcome: str,
) -> float | None:
    probability = _probability_or_none(value)
    if probability is None:
        return None
    if book_token_outcome == "NO":
        return 1.0 - probability
    return probability


def _category_prior_baseline_prob_estimate(signal: MarketSignal) -> float | None:
    return _probability_or_none(
        signal.external_signal.get("category_prior_baseline_prob_estimate")
    )


def _top_orderbook_levels(
    orderbook: dict[str, Any],
    *,
    depth: int = 5,
) -> dict[str, list[dict[str, float]]]:
    return {
        "bids": _side_levels(orderbook.get("bids"), depth=depth),
        "asks": _side_levels(orderbook.get("asks"), depth=depth),
    }


def _side_levels(value: object, *, depth: int) -> list[dict[str, float]]:
    if not isinstance(value, list):
        return []

    levels: list[dict[str, float]] = []
    for raw_level in value[:depth]:
        if not isinstance(raw_level, dict):
            continue
        price = _optional_float(raw_level.get("price"))
        size = _optional_float(raw_level.get("size"))
        if price is None or size is None:
            continue
        levels.append({"price": price, "size": size})
    return levels


def _decision_book_hash(
    *,
    market_id: str,
    token_id: str | None,
    book_top_levels: dict[str, list[dict[str, float]]],
) -> str:
    payload = {
        "market_id": market_id,
        "token_id": token_id,
        "book_top_levels": book_top_levels,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _runtime_run_id(*, mode: RunMode, started_at: datetime) -> str:
    payload = {
        "mode": mode.value,
        "started_at": started_at.isoformat(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _runtime_strategy_fingerprint(
    runtimes: Sequence[StrategyControllerRuntime],
) -> str | None:
    labels = sorted(
        f"{runtime.strategy_id}@{runtime.strategy_version_id}"
        for runtime in runtimes
    )
    if not labels:
        return None
    canonical = json.dumps(labels, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _signal_to_decision_latency_ms(
    signal: MarketSignal,
    *,
    decision_created_at: datetime,
) -> float:
    return max(0.0, (decision_created_at - signal.fetched_at).total_seconds() * 1000.0)


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(cast(int | float | str, value))
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return parsed


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _probability_or_none(value: object) -> float | None:
    parsed = _optional_float(value)
    if parsed is None or parsed < 0.0 or parsed > 1.0:
        return None
    return parsed


def _decision_created_at(
    signal: MarketSignal,
    opportunity: Opportunity | None,
) -> datetime:
    del signal
    if opportunity is not None:
        return opportunity.created_at
    return datetime.now(tz=UTC)


def _decision_expires_at(
    signal: MarketSignal,
    opportunity: Opportunity | None,
    *,
    created_at: datetime,
) -> datetime:
    candidates = [created_at + DECISION_PENDING_TTL]
    if opportunity is not None and opportunity.expiry is not None:
        candidates.append(opportunity.expiry)
    if signal.resolves_at is not None:
        candidates.append(signal.resolves_at)
    return min(candidates)
