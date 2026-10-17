"""Research backtest runner for queued backtest runs."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import os
import socket
from typing import Any, Literal, Protocol, TypeAlias, cast
from uuid import uuid4

import asyncpg

from pms.actuator.risk import InsufficientLiquidityError
from pms.controller.factory import ControllerPipelineFactory
from pms.controller.factor_snapshot import PostgresFactorSnapshotReader
from pms.controller.outcome_tokens import MarketDataOutcomeTokenResolver
from pms.core.enums import OrderStatus
from pms.core.models import (
    FillRecord,
    MarketSignal,
    OrderState,
    Portfolio,
    Position,
    TradeDecision,
)
from pms.core.validation import normalize_resolved_outcome
from pms.evaluation.metrics import StrategyVersionKey
from pms.research.entities import PortfolioTarget, serialize_portfolio_target_json
from pms.research.execution import BacktestExecutionSimulator
from pms.research.replay import MarketUniverseReplayEngine
from pms.research.spec_codec import (
    deserialize_backtest_spec as _codec_deserialize_backtest_spec,
    deserialize_execution_config as _codec_deserialize_execution_config,
)
from pms.research.specs import (
    BacktestExecutionConfig,
    BacktestSpec,
    ExecutionModel,
)
from pms.storage.market_data_store import PostgresMarketDataStore
from pms.storage.strategy_registry import _strategy_from_config_json
from pms.strategies.projections import ActiveStrategy


CancelPoint = Literal["before_first_strategy", "between_strategies", "after_last_strategy"]
CancelProbe: TypeAlias = Callable[[CancelPoint], Awaitable[None] | None]
StrategyLoader: TypeAlias = Callable[
    [tuple[StrategyVersionKey, ...]],
    Awaitable[list[ActiveStrategy]],
]
HostProvider: TypeAlias = Callable[[], str]
PidProvider: TypeAlias = Callable[[], int]


class ReplayEngineLike(Protocol):
    def stream(
        self,
        spec: BacktestSpec,
        exec_config: BacktestExecutionConfig,
    ) -> AsyncIterator[MarketSignal]: ...


class ControllerPipelineLike(Protocol):
    async def on_signal(
        self,
        signal: MarketSignal,
        portfolio: Portfolio | None = None,
    ) -> tuple[object, object] | None: ...


class ControllerPipelineFactoryLike(Protocol):
    def build(self, strategy: ActiveStrategy) -> ControllerPipelineLike: ...


class BacktestExecutionSimulatorLike(Protocol):
    async def execute(
        self,
        *,
        signal: MarketSignal,
        decision: TradeDecision,
        portfolio: Portfolio | None = None,
        execution_model: ExecutionModel,
    ) -> OrderState: ...


@dataclass(frozen=True, slots=True)
class _ClaimedRun:
    run_id: str
    spec: BacktestSpec
    exec_config: BacktestExecutionConfig


@dataclass(frozen=True, slots=True)
class _SliceDescriptor:
    label: str
    start: datetime
    end: datetime
    kind: Literal["walk_forward", "category", "liquidity"]


@dataclass(slots=True)
class _StrategySliceAccumulator:
    strategy_id: str
    strategy_version_id: str
    execution_model: ExecutionModel
    slice_label: str
    slice_start: datetime
    slice_end: datetime
    slice_kind: str = "walk_forward"
    opportunity_count: int = 0
    decision_count: int = 0
    fill_count: int = 0
    fills_with_resolution: int = 0
    decisions_with_resolution: int = 0
    brier_scores: list[Decimal] = field(default_factory=list)
    slippage_bps_values: list[Decimal] = field(default_factory=list)
    cumulative_pnl: Decimal = Decimal("0")
    peak_equity: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")

    def record_decision(
        self,
        *,
        signal: MarketSignal,
        decision: object,
    ) -> None:
        self.opportunity_count += 1
        self.decision_count += 1
        resolved_outcome = _resolved_outcome(signal)
        if resolved_outcome is not None:
            self.decisions_with_resolution += 1
            prob_estimate = _yes_probability(decision)
            resolved = Decimal(str(resolved_outcome))
            self.brier_scores.append((prob_estimate - resolved) ** 2)

    def record_fill(
        self,
        *,
        signal: MarketSignal,
        decision: object,
        fill: FillRecord,
    ) -> None:
        self.fill_count += 1
        self.slippage_bps_values.append(
            Decimal(
                str(
                    _decision_slippage_bps(
                        decision=decision,
                        fill_price=fill.fill_price,
                    )
                )
            )
        )
        if _resolved_outcome(signal) is not None:
            self.fills_with_resolution += 1
        pnl_delta = _pnl_delta(
            signal=signal,
            decision_outcome=cast(str, getattr(decision, "outcome", "YES")),
            fill=fill,
            execution_model=self.execution_model,
        )
        self.cumulative_pnl += pnl_delta
        if self.cumulative_pnl > self.peak_equity:
            self.peak_equity = self.cumulative_pnl
        drawdown = self.peak_equity - self.cumulative_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def as_insert_args(self, *, run_id: str) -> tuple[object, ...]:
        brier = (
            float(sum(self.brier_scores) / Decimal(len(self.brier_scores)))
            if self.brier_scores
            else None
        )
        slippage_bps = (
            float(sum(self.slippage_bps_values) / Decimal(len(self.slippage_bps_values)))
            if self.slippage_bps_values
            else None
        )
        fill_rate = (
            self.fill_count / self.decision_count if self.decision_count > 0 else 0.0
        )
        pnl_cum: float | None
        drawdown_max: float | None
        if self.fills_with_resolution > 0:
            pnl_cum = float(self.cumulative_pnl)
            drawdown_max = float(self.max_drawdown)
        else:
            pnl_cum = None
            drawdown_max = None
        return (
            str(uuid4()),
            run_id,
            self.strategy_id,
            self.strategy_version_id,
            self.slice_label,
            self.slice_start,
            self.slice_end,
            self.slice_kind,
            brier,
            pnl_cum,
            drawdown_max,
            fill_rate,
            slippage_bps,
            self.opportunity_count,
            self.decision_count,
            self.fill_count,
        )


@dataclass(slots=True)
class _BacktestExecutionRow:
    decision_id: str
    strategy_id: str
    strategy_version_id: str
    market_id: str
    status: Literal["filled", "rejected"]
    slippage_bps: float | None
    pnl: float | None
    rejection_reason: str | None
    created_at: datetime

    def as_insert_args(self, *, run_id: str) -> tuple[object, ...]:
        return (
            str(uuid4()),
            run_id,
            self.strategy_id,
            self.strategy_version_id,
            self.decision_id,
            self.market_id,
            self.status,
            self.slippage_bps,
            self.pnl,
            self.rejection_reason,
            self.created_at,
        )


@dataclass(slots=True)
class _StrategyAccumulator:
    strategy_id: str
    strategy_version_id: str
    execution_model: ExecutionModel
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    finished_at: datetime | None = None
    opportunity_count: int = 0
    decision_count: int = 0
    fill_count: int = 0
    # Fills whose signal carried a `resolved_outcome`. Only those can
    # contribute real P&L; when this stays zero we must not publish a
    # spurious 0.0 pnl_cum (see `as_insert_args`).
    fills_with_resolution: int = 0
    decisions_with_resolution: int = 0
    brier_scores: list[Decimal] = field(default_factory=list)
    slippage_bps_values: list[Decimal] = field(default_factory=list)
    cumulative_pnl: Decimal = Decimal("0")
    peak_equity: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    targets: dict[tuple[str, str, Literal["buy_yes", "buy_no"], datetime], float] = field(
        default_factory=dict
    )
    slice_accumulators: dict[
        tuple[str, str, datetime, datetime],
        _StrategySliceAccumulator,
    ] = field(default_factory=dict)
    _execution_rows_by_decision: dict[str, _BacktestExecutionRow] = field(
        default_factory=dict
    )

    @property
    def execution_rows(self) -> tuple[_BacktestExecutionRow, ...]:
        return tuple(self._execution_rows_by_decision.values())

    def record_decision(
        self,
        *,
        signal: MarketSignal,
        opportunity: object,
        decision: object,
    ) -> None:
        self.opportunity_count += 1
        self.decision_count += 1
        opportunity_side = cast(str, getattr(opportunity, "side"))
        token_id = cast(str | None, getattr(decision, "token_id"))
        if token_id is not None:
            target_side: Literal["buy_yes", "buy_no"]
            target_side = "buy_yes" if opportunity_side == "yes" else "buy_no"
            self.targets[(signal.market_id, token_id, target_side, signal.fetched_at)] = float(
                cast(float, getattr(decision, "notional_usdc"))
            )
        resolved_outcome = _resolved_outcome(signal)
        if resolved_outcome is not None:
            self.decisions_with_resolution += 1
            prob_estimate = _yes_probability(decision)
            resolved = Decimal(str(resolved_outcome))
            self.brier_scores.append((prob_estimate - resolved) ** 2)

    def record_fill(
        self,
        *,
        signal: MarketSignal,
        decision: object,
        fill: FillRecord,
    ) -> None:
        self.fill_count += 1
        self.slippage_bps_values.append(
            Decimal(
                str(
                    _decision_slippage_bps(
                        decision=decision,
                        fill_price=fill.fill_price,
                    )
                )
            )
        )
        if _resolved_outcome(signal) is not None:
            self.fills_with_resolution += 1
        pnl_delta = _pnl_delta(
            signal=signal,
            decision_outcome=cast(str, getattr(decision, "outcome", "YES")),
            fill=fill,
            execution_model=self.execution_model,
        )
        self.cumulative_pnl += pnl_delta
        if self.cumulative_pnl > self.peak_equity:
            self.peak_equity = self.cumulative_pnl
        drawdown = self.peak_equity - self.cumulative_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        previous_execution = self._execution_rows_by_decision.get(fill.decision_id)
        previous_pnl = (
            None
            if previous_execution is None
            else previous_execution.pnl
        )
        execution_pnl = (
            None
            if _resolved_outcome(signal) is None
            else float(pnl_delta + Decimal(str(previous_pnl or 0.0)))
        )
        self._execution_rows_by_decision[fill.decision_id] = _BacktestExecutionRow(
            decision_id=fill.decision_id,
            strategy_id=self.strategy_id,
            strategy_version_id=self.strategy_version_id,
            market_id=fill.market_id,
            status="filled",
            slippage_bps=_decision_slippage_bps(
                decision=decision,
                fill_price=fill.fill_price,
            ),
            pnl=execution_pnl,
            rejection_reason=None,
            created_at=signal.fetched_at,
        )

    def record_execution_rejection(
        self,
        *,
        signal: MarketSignal,
        decision: TradeDecision,
        rejection_reason: str,
    ) -> None:
        self.record_execution_terminal_rejection(
            decision=decision,
            rejection_reason=rejection_reason,
            created_at=signal.fetched_at,
        )

    def record_execution_terminal_rejection(
        self,
        *,
        decision: TradeDecision,
        rejection_reason: str,
        created_at: datetime,
    ) -> None:
        existing = self._execution_rows_by_decision.get(decision.decision_id)
        if existing is not None and existing.status == "filled":
            return
        self._execution_rows_by_decision[decision.decision_id] = _BacktestExecutionRow(
            decision_id=decision.decision_id,
            strategy_id=decision.strategy_id,
            strategy_version_id=decision.strategy_version_id,
            market_id=decision.market_id,
            status="rejected",
            slippage_bps=None,
            pnl=0.0,
            rejection_reason=rejection_reason,
            created_at=existing.created_at if existing is not None else created_at,
        )

    def record_slice_decision(
        self,
        *,
        signal: MarketSignal,
        opportunity: object,
        decision: object,
        slice_descriptor: _SliceDescriptor,
    ) -> None:
        del opportunity
        self._slice_accumulator(
            slice_descriptor=slice_descriptor,
        ).record_decision(signal=signal, decision=decision)

    def record_slice_fill(
        self,
        *,
        signal: MarketSignal,
        decision: object,
        fill: FillRecord,
        slice_descriptor: _SliceDescriptor,
    ) -> None:
        self._slice_accumulator(
            slice_descriptor=slice_descriptor,
        ).record_fill(signal=signal, decision=decision, fill=fill)

    def _slice_accumulator(
        self,
        *,
        slice_descriptor: _SliceDescriptor,
    ) -> _StrategySliceAccumulator:
        key = (
            slice_descriptor.kind,
            slice_descriptor.label,
            slice_descriptor.start,
            slice_descriptor.end,
        )
        accumulator = self.slice_accumulators.get(key)
        if accumulator is None:
            accumulator = _StrategySliceAccumulator(
                strategy_id=self.strategy_id,
                strategy_version_id=self.strategy_version_id,
                execution_model=self.execution_model,
                slice_label=slice_descriptor.label,
                slice_start=slice_descriptor.start,
                slice_end=slice_descriptor.end,
                slice_kind=slice_descriptor.kind,
            )
            self.slice_accumulators[key] = accumulator
        return accumulator

    def slice_insert_args(self, *, run_id: str) -> tuple[tuple[object, ...], ...]:
        return tuple(
            accumulator.as_insert_args(run_id=run_id)
            for _key, accumulator in sorted(
                self.slice_accumulators.items(),
                key=lambda item: (
                    item[0][2],
                    item[0][0],
                    item[0][1],
                    item[0][3],
                ),
            )
        )

    def portfolio_target(self) -> PortfolioTarget:
        return PortfolioTarget(
            strategy_id=self.strategy_id,
            strategy_version_id=self.strategy_version_id,
            targets=dict(self.targets),
        )

    def as_insert_args(self, *, run_id: str) -> tuple[object, ...]:
        self.finished_at = self.finished_at or datetime.now(tz=UTC)
        brier = (
            float(sum(self.brier_scores) / Decimal(len(self.brier_scores)))
            if self.brier_scores
            else None
        )
        slippage_bps = (
            float(sum(self.slippage_bps_values) / Decimal(len(self.slippage_bps_values)))
            if self.slippage_bps_values
            else None
        )
        fill_rate = (
            self.fill_count / self.decision_count if self.decision_count > 0 else 0.0
        )
        # When zero fills had resolution data, cumulative_pnl is structurally
        # 0 from summing no-op `_pnl_delta` contributions — not a genuine
        # zero outcome. Emit NULL so downstream reports can distinguish
        # "no P&L to compute" from "computed P&L that happens to be zero".
        pnl_cum: float | None
        drawdown_max: float | None
        if self.fills_with_resolution > 0:
            pnl_cum = float(self.cumulative_pnl)
            drawdown_max = float(self.max_drawdown)
        else:
            pnl_cum = None
            drawdown_max = None
        return (
            str(uuid4()),
            run_id,
            self.strategy_id,
            self.strategy_version_id,
            brier,
            pnl_cum,
            drawdown_max,
            fill_rate,
            slippage_bps,
            self.opportunity_count,
            self.decision_count,
            self.fill_count,
            serialize_portfolio_target_json(self.portfolio_target()),
            self.started_at,
            self.finished_at,
        )


@dataclass(slots=True)
class BacktestRunner:
    writable_pool: asyncpg.Pool
    readonly_pool: asyncpg.Pool
    controller_factory: ControllerPipelineFactoryLike = field(
        default_factory=ControllerPipelineFactory
    )
    replay_engine: ReplayEngineLike | None = None
    execution_simulator: BacktestExecutionSimulatorLike = field(
        default_factory=BacktestExecutionSimulator
    )
    strategy_loader: StrategyLoader | None = None
    evaluation_as_of: datetime | None = None
    cancel_probe: CancelProbe | None = None
    host_provider: HostProvider = socket.gethostname
    pid_provider: PidProvider = os.getpid

    def __post_init__(self) -> None:
        if self.replay_engine is None:
            self.replay_engine = MarketUniverseReplayEngine(pool=self.readonly_pool)
        replay_target = getattr(self.execution_simulator, "replay_engine", None)
        if replay_target is None and hasattr(self.execution_simulator, "replay_engine"):
            setattr(self.execution_simulator, "replay_engine", self.replay_engine)
        if isinstance(self.controller_factory, ControllerPipelineFactory):
            market_data_store = PostgresMarketDataStore(self.readonly_pool)
            self.controller_factory.factor_reader = PostgresFactorSnapshotReader(
                self.readonly_pool
            )
            self.controller_factory.outcome_token_resolver = (
                MarketDataOutcomeTokenResolver(market_data_store)
            )

    async def execute(self, run_id: str) -> bool:
        try:
            # Legacy or manually inserted rows can still fail spec deserialization
            # after the run is claimed; surface that as a failed run instead of
            # leaving the claimed row stuck in status='running'.
            claimed_run = await self._claim_run(run_id)
            if claimed_run is None:
                return False
            await asyncio.wait_for(
                self._execute_claimed(claimed_run),
                timeout=claimed_run.exec_config.time_budget,
            )
        except TimeoutError:
            await self._mark_failed(run_id, "time_budget_exceeded")
            return False
        except asyncio.CancelledError:
            await self._mark_failed(run_id, "cancelled")
            raise
        except Exception as exc:
            await self._mark_failed(run_id, _failure_reason(exc))
            return False

        await self._mark_completed(run_id)
        return True

    async def _execute_claimed(self, claimed_run: _ClaimedRun) -> None:
        if not claimed_run.spec.strategy_versions:
            msg = "no_strategy_versions"
            raise ValueError(msg)
        strategies = await self._load_strategies(claimed_run.spec.strategy_versions)
        evaluation_as_of = self.evaluation_as_of or datetime.now(tz=UTC)
        accumulators: list[_StrategyAccumulator] = []
        for index, strategy in enumerate(strategies):
            if index == 0:
                await _maybe_call_cancel_probe(self.cancel_probe, "before_first_strategy")
            else:
                await _maybe_call_cancel_probe(self.cancel_probe, "between_strategies")
            accumulator = await self._run_strategy(
                strategy=strategy,
                spec=claimed_run.spec,
                exec_config=claimed_run.exec_config,
                evaluation_as_of=evaluation_as_of,
            )
            accumulators.append(accumulator)
        missing_labels = [
            accumulator.strategy_id
            for accumulator in accumulators
            if accumulator.decision_count > 0
            and accumulator.decisions_with_resolution == 0
        ]
        if missing_labels:
            raise ValueError(
                "required resolution labels missing for strategy runs: "
                + ", ".join(missing_labels)
            )
        # atomic persist: either every strategy_runs row lands or none — prevents
        # partial success leaking into reports when a later strategy raises.
        await self._insert_strategy_runs_atomically(
            run_id=claimed_run.run_id,
            accumulators=accumulators,
        )
        await _maybe_call_cancel_probe(self.cancel_probe, "after_last_strategy")

    async def _run_strategy(
        self,
        *,
        strategy: ActiveStrategy,
        spec: BacktestSpec,
        exec_config: BacktestExecutionConfig,
        evaluation_as_of: datetime | None = None,
    ) -> _StrategyAccumulator:
        replay_engine = self._required_replay_engine()
        pipeline = self.controller_factory.build(strategy)
        portfolio = _default_portfolio(strategy)
        accumulator = _StrategyAccumulator(
            strategy_id=strategy.strategy_id,
            strategy_version_id=strategy.strategy_version_id,
            execution_model=spec.execution_model,
        )
        observed_order_totals: dict[str, tuple[float, float]] = {}
        order_decisions: dict[str, TradeDecision] = {}

        evaluation_cutoff = evaluation_as_of or datetime.now(tz=UTC)
        async for signal in replay_engine.stream(spec, exec_config):
            scoring_signal = _signal_with_replay_resolution(
                signal,
                replay_engine,
                evaluation_as_of=evaluation_cutoff,
            )
            slice_descriptors = _slice_descriptors(
                signal=signal,
                spec=spec,
                exec_config=exec_config,
            )
            for advanced_state in await _advance_open_orders(
                self.execution_simulator,
                signal=signal,
                execution_model=spec.execution_model,
            ):
                decision_for_fill = order_decisions.get(advanced_state.order_id)
                if decision_for_fill is None:
                    msg = f"missing cached decision for order_id={advanced_state.order_id}"
                    raise AssertionError(msg)
                prior_notional, prior_quantity = observed_order_totals.get(
                    advanced_state.order_id,
                    (0.0, 0.0),
                )
                observed_order_totals[advanced_state.order_id] = (
                    advanced_state.filled_notional_usdc,
                    advanced_state.filled_quantity,
                )
                fill = _fill_from_order(
                    advanced_state,
                    decision=decision_for_fill,
                    signal=scoring_signal,
                    prior_filled_notional_usdc=prior_notional,
                    prior_filled_quantity=prior_quantity,
                    execution_model=spec.execution_model,
                )
                if fill is None:
                    continue
                portfolio = _portfolio_with_fill(portfolio, fill)
                accumulator.record_fill(
                    signal=scoring_signal,
                    decision=decision_for_fill,
                    fill=fill,
                )
                for slice_descriptor in slice_descriptors:
                    accumulator.record_slice_fill(
                        signal=scoring_signal,
                        decision=decision_for_fill,
                        fill=fill,
                        slice_descriptor=slice_descriptor,
                    )
                if advanced_state.status in {
                    OrderStatus.MATCHED.value,
                    OrderStatus.CANCELLED.value,
                    "rejected",
                }:
                    order_decisions.pop(advanced_state.order_id, None)
            emission = await pipeline.on_signal(signal, portfolio=portfolio)
            if emission is None:
                continue
            opportunity, decision = emission
            accumulator.record_decision(
                signal=scoring_signal,
                opportunity=opportunity,
                decision=decision,
            )
            for slice_descriptor in slice_descriptors:
                accumulator.record_slice_decision(
                    signal=scoring_signal,
                    opportunity=opportunity,
                    decision=decision,
                    slice_descriptor=slice_descriptor,
                )
            try:
                order_state = await self.execution_simulator.execute(
                    signal=signal,
                    decision=cast(Any, decision),
                    portfolio=portfolio,
                    execution_model=spec.execution_model,
                )
            except InsufficientLiquidityError:
                accumulator.record_execution_rejection(
                    signal=signal,
                    decision=cast(TradeDecision, decision),
                    rejection_reason="insufficient_liquidity",
                )
                continue
            order_decisions[order_state.order_id] = cast(TradeDecision, decision)
            prior_notional, prior_quantity = observed_order_totals.get(
                order_state.order_id,
                (0.0, 0.0),
            )
            observed_order_totals[order_state.order_id] = (
                order_state.filled_notional_usdc,
                order_state.filled_quantity,
            )
            fill = _fill_from_order(
                order_state,
                decision=cast(TradeDecision, decision),
                signal=scoring_signal,
                prior_filled_notional_usdc=prior_notional,
                prior_filled_quantity=prior_quantity,
                execution_model=spec.execution_model,
            )
            if fill is None:
                accumulator.record_execution_rejection(
                    signal=signal,
                    decision=cast(TradeDecision, decision),
                    rejection_reason=order_state.raw_status,
                )
                continue
            portfolio = _portfolio_with_fill(portfolio, fill)
            accumulator.record_fill(
                signal=scoring_signal,
                decision=decision,
                fill=fill,
            )
            for slice_descriptor in slice_descriptors:
                accumulator.record_slice_fill(
                    signal=scoring_signal,
                    decision=decision,
                    fill=fill,
                    slice_descriptor=slice_descriptor,
                )
            if order_state.status in {
                OrderStatus.MATCHED.value,
                OrderStatus.CANCELLED.value,
                "rejected",
            }:
                order_decisions.pop(order_state.order_id, None)

        for final_state in await _cancel_open_orders(
            self.execution_simulator,
            session_end=spec.date_range_end,
        ):
            decision_for_final = order_decisions.get(final_state.order_id)
            if decision_for_final is not None:
                accumulator.record_execution_terminal_rejection(
                    decision=decision_for_final,
                    rejection_reason=final_state.raw_status,
                    created_at=final_state.submitted_at,
                )
            observed_order_totals[final_state.order_id] = (
                final_state.filled_notional_usdc,
                final_state.filled_quantity,
            )
            order_decisions.pop(final_state.order_id, None)

        accumulator.finished_at = datetime.now(tz=UTC)
        return accumulator

    async def _claim_run(self, run_id: str) -> _ClaimedRun | None:
        connection = await self.writable_pool.acquire()
        try:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    WITH candidate AS (
                        SELECT run_id
                        FROM backtest_runs
                        WHERE run_id = $1::uuid
                          AND status = 'queued'
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE backtest_runs AS runs
                    SET
                        status = 'running',
                        started_at = clock_timestamp(),
                        worker_pid = $2,
                        worker_host = $3
                    FROM candidate
                    WHERE runs.run_id = candidate.run_id
                    RETURNING runs.run_id, runs.spec_json, runs.exec_config_json
                    """,
                    run_id,
                    self.pid_provider(),
                    self.host_provider(),
                )
        finally:
            await self.writable_pool.release(connection)

        if row is None:
            return None
        return _ClaimedRun(
            run_id=str(cast(object, row["run_id"])),
            spec=_codec_deserialize_backtest_spec(row["spec_json"]),
            exec_config=_codec_deserialize_execution_config(row["exec_config_json"]),
        )

    async def _load_strategies(
        self,
        strategy_versions: tuple[StrategyVersionKey, ...],
    ) -> list[ActiveStrategy]:
        if self.strategy_loader is not None:
            return await self.strategy_loader(strategy_versions)

        connection = await self.readonly_pool.acquire()
        try:
            strategies: list[ActiveStrategy] = []
            for strategy_id, strategy_version_id in strategy_versions:
                row = await connection.fetchrow(
                    """
                    SELECT strategy_id, strategy_version_id, config_json
                    FROM strategy_versions
                    WHERE strategy_id = $1 AND strategy_version_id = $2
                    """,
                    strategy_id,
                    strategy_version_id,
                )
                if row is None:
                    msg = (
                        "BacktestRunner could not load strategy version "
                        f"{strategy_id}:{strategy_version_id}"
                    )
                    raise LookupError(msg)
                strategy = _strategy_from_config_json(row["config_json"])
                strategies.append(
                    ActiveStrategy(
                        strategy_id=cast(str, row["strategy_id"]),
                        strategy_version_id=cast(str, row["strategy_version_id"]),
                        config=strategy.config,
                        risk=strategy.risk,
                        eval_spec=strategy.eval_spec,
                        forecaster=strategy.forecaster,
                        market_selection=strategy.market_selection,
                    )
                )
        finally:
            await self.readonly_pool.release(connection)
        return strategies

    async def _insert_strategy_runs_atomically(
        self,
        *,
        run_id: str,
        accumulators: Sequence[_StrategyAccumulator],
    ) -> None:
        if not accumulators:
            return
        connection = await self.writable_pool.acquire()
        try:
            async with connection.transaction():
                for accumulator in accumulators:
                    await connection.execute(
                        """
                        INSERT INTO strategy_runs (
                            strategy_run_id,
                            run_id,
                            strategy_id,
                            strategy_version_id,
                            brier,
                            pnl_cum,
                            drawdown_max,
                            fill_rate,
                            slippage_bps,
                            opportunity_count,
                            decision_count,
                            fill_count,
                            portfolio_target_json,
                            started_at,
                            finished_at
                        ) VALUES (
                            $1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9,
                            $10, $11, $12, $13::jsonb, $14, $15
                        )
                        """,
                        *accumulator.as_insert_args(run_id=run_id),
                    )
                    for execution_row in accumulator.execution_rows:
                        await connection.execute(
                            """
                            INSERT INTO backtest_execution_rows (
                                backtest_execution_row_id,
                                run_id,
                                strategy_id,
                                strategy_version_id,
                                decision_id,
                                market_id,
                                status,
                                slippage_bps,
                                pnl,
                                rejection_reason,
                                created_at
                            ) VALUES (
                                $1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8,
                                $9, $10, $11
                            )
                            ON CONFLICT (
                                run_id,
                                strategy_id,
                                strategy_version_id,
                                decision_id
                            ) DO UPDATE
                            SET
                                market_id = EXCLUDED.market_id,
                                status = EXCLUDED.status,
                                slippage_bps = EXCLUDED.slippage_bps,
                                pnl = EXCLUDED.pnl,
                                rejection_reason = EXCLUDED.rejection_reason,
                                created_at = EXCLUDED.created_at
                            """,
                            *execution_row.as_insert_args(run_id=run_id),
                        )
                    for slice_args in accumulator.slice_insert_args(run_id=run_id):
                        await connection.execute(
                            """
                            INSERT INTO strategy_run_slices (
                                strategy_run_slice_id,
                                run_id,
                                strategy_id,
                                strategy_version_id,
                                slice_label,
                                slice_start,
                                slice_end,
                                slice_kind,
                                brier,
                                pnl_cum,
                                drawdown_max,
                                fill_rate,
                                slippage_bps,
                                opportunity_count,
                                decision_count,
                                fill_count
                            ) VALUES (
                                $1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8,
                                $9, $10, $11, $12, $13, $14, $15, $16
                            )
                            ON CONFLICT (
                                run_id,
                                strategy_id,
                                strategy_version_id,
                                slice_label
                            ) DO UPDATE
                            SET
                                slice_start = EXCLUDED.slice_start,
                                slice_end = EXCLUDED.slice_end,
                                slice_kind = EXCLUDED.slice_kind,
                                brier = EXCLUDED.brier,
                                pnl_cum = EXCLUDED.pnl_cum,
                                drawdown_max = EXCLUDED.drawdown_max,
                                fill_rate = EXCLUDED.fill_rate,
                                slippage_bps = EXCLUDED.slippage_bps,
                                opportunity_count = EXCLUDED.opportunity_count,
                                decision_count = EXCLUDED.decision_count,
                                fill_count = EXCLUDED.fill_count
                            """,
                            *slice_args,
                        )
        finally:
            await self.writable_pool.release(connection)

    async def _mark_completed(self, run_id: str) -> None:
        connection = await self.writable_pool.acquire()
        try:
            await connection.execute(
                """
                UPDATE backtest_runs
                SET status = 'completed',
                    finished_at = clock_timestamp(),
                    failure_reason = NULL
                WHERE run_id = $1::uuid
                """,
                run_id,
            )
        finally:
            await self.writable_pool.release(connection)

    async def _mark_failed(self, run_id: str, reason: str) -> None:
        connection = await self.writable_pool.acquire()
        try:
            await connection.execute(
                """
                UPDATE backtest_runs
                SET status = 'failed',
                    finished_at = clock_timestamp(),
                    failure_reason = $2
                WHERE run_id = $1::uuid
                """,
                run_id,
                reason,
            )
        finally:
            await self.writable_pool.release(connection)

    def _required_replay_engine(self) -> ReplayEngineLike:
        if self.replay_engine is None:
            msg = "BacktestRunner replay_engine is not configured"
            raise RuntimeError(msg)
        return self.replay_engine


def _default_portfolio(strategy: ActiveStrategy) -> Portfolio:
    starting_cash = max(1_000.0, strategy.risk.max_position_notional_usdc * 10.0)
    return Portfolio(
        total_usdc=starting_cash,
        free_usdc=starting_cash,
        locked_usdc=0.0,
        open_positions=[],
        max_drawdown_pct=0.0,
    )


def _fill_from_order(
    order_state: OrderState,
    decision: TradeDecision | None,
    signal: MarketSignal,
    *,
    prior_filled_notional_usdc: float,
    prior_filled_quantity: float,
    execution_model: ExecutionModel,
) -> FillRecord | None:
    if decision is None:
        return None
    if order_state.status not in {OrderStatus.MATCHED.value, OrderStatus.PARTIAL.value}:
        return None
    if order_state.fill_price is None or order_state.fill_price <= 0.0:
        return None
    delta_notional = order_state.filled_notional_usdc - prior_filled_notional_usdc
    delta_quantity = order_state.filled_quantity - prior_filled_quantity
    if delta_notional <= 0.0 or delta_quantity <= 0.0:
        return None

    return FillRecord(
        trade_id=order_state.order_id,
        order_id=order_state.order_id,
        decision_id=order_state.decision_id,
        market_id=order_state.market_id,
        token_id=order_state.token_id,
        venue=order_state.venue,
        side=decision.action if decision.action is not None else decision.side,
        fill_price=order_state.fill_price,
        fill_notional_usdc=delta_notional,
        fill_quantity=delta_quantity,
        executed_at=order_state.submitted_at,
        filled_at=order_state.last_updated_at,
        status=order_state.status,
        anomaly_flags=[],
        strategy_id=order_state.strategy_id,
        strategy_version_id=order_state.strategy_version_id,
        fees=execution_model.compute_fee(
            notional_usdc=delta_notional,
            fill_price=order_state.fill_price,
        ),
        fee_bps=int(execution_model.fee_rate * 10_000),
        resolved_outcome=_resolved_outcome(signal),
    )


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
            )
            break
    else:
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


def _decision_slippage_bps(*, decision: object, fill_price: float) -> float:
    raw_limit_price = getattr(decision, "limit_price", getattr(decision, "price", 0.0))
    limit_price = float(cast(float, raw_limit_price))
    if limit_price <= 0.0:
        return 0.0
    action = cast(str, getattr(decision, "action", getattr(decision, "side", "BUY")))
    if action == "SELL":
        slippage = limit_price - fill_price
    else:
        slippage = fill_price - limit_price
    return max(0.0, slippage / limit_price * 10_000.0)


def _yes_probability(decision: object) -> Decimal:
    prob_estimate = Decimal(str(cast(float, getattr(decision, "prob_estimate"))))
    decision_outcome = cast(str, getattr(decision, "outcome", "YES"))
    if decision_outcome == "NO":
        return Decimal("1") - prob_estimate
    return prob_estimate


def _same_position(position: Position, fill: FillRecord) -> bool:
    return (
        position.market_id == fill.market_id
        and position.token_id == fill.token_id
        and position.venue == fill.venue
        and position.side == fill.side
    )


async def _maybe_call_cancel_probe(
    probe: CancelProbe | None,
    point: CancelPoint,
) -> None:
    if probe is None:
        return
    outcome = probe(point)
    if asyncio.iscoroutine(outcome):
        await cast(Awaitable[None], outcome)


def _slice_bounds(
    signal_timestamp: datetime,
    *,
    spec: BacktestSpec,
    exec_config: BacktestExecutionConfig,
) -> tuple[datetime, datetime]:
    chunk_delta = timedelta(days=exec_config.chunk_days)
    chunk_seconds = chunk_delta.total_seconds()
    effective_timestamp = signal_timestamp
    if effective_timestamp < spec.date_range_start:
        effective_timestamp = spec.date_range_start
    if effective_timestamp >= spec.date_range_end:
        effective_timestamp = spec.date_range_end - timedelta(microseconds=1)
    elapsed_seconds = max(
        0.0,
        (effective_timestamp - spec.date_range_start).total_seconds(),
    )
    slice_index = int(elapsed_seconds // chunk_seconds)
    slice_start = spec.date_range_start + (chunk_delta * slice_index)
    slice_end = min(slice_start + chunk_delta, spec.date_range_end)
    return slice_start, slice_end


def _slice_descriptors(
    *,
    signal: MarketSignal,
    spec: BacktestSpec,
    exec_config: BacktestExecutionConfig,
) -> tuple[_SliceDescriptor, ...]:
    slice_start, slice_end = _slice_bounds(
        signal.fetched_at,
        spec=spec,
        exec_config=exec_config,
    )
    descriptors = [
        _SliceDescriptor(
            label=_slice_label(slice_start=slice_start, slice_end=slice_end),
            start=slice_start,
            end=slice_end,
            kind="walk_forward",
        )
    ]
    category_label = _category_slice_label(signal)
    if category_label is not None:
        descriptors.append(
            _SliceDescriptor(
                label=category_label,
                start=spec.date_range_start,
                end=spec.date_range_end,
                kind="category",
            )
        )
    liquidity_label = _liquidity_slice_label(signal)
    if liquidity_label is not None:
        descriptors.append(
            _SliceDescriptor(
                label=liquidity_label,
                start=spec.date_range_start,
                end=spec.date_range_end,
                kind="liquidity",
            )
        )
    return tuple(descriptors)


def _slice_label(*, slice_start: datetime, slice_end: datetime) -> str:
    return f"{slice_start.date().isoformat()}/{slice_end.date().isoformat()}"


def _category_slice_label(signal: MarketSignal) -> str | None:
    raw_category = signal.external_signal.get("category")
    if not isinstance(raw_category, str):
        raw_category = signal.external_signal.get("market_category")
    if not isinstance(raw_category, str):
        return None
    normalized = _slug_fragment(raw_category)
    if not normalized:
        return None
    return f"category:{normalized}"


def _liquidity_slice_label(signal: MarketSignal) -> str | None:
    volume_24h = signal.volume_24h
    if volume_24h is None or volume_24h < 0.0:
        return None
    if volume_24h < 1_000.0:
        bucket = "lt1000"
    elif volume_24h < 10_000.0:
        bucket = "1000-10000"
    else:
        bucket = "gte10000"
    return f"liquidity:volume_24h:{bucket}"


def _slug_fragment(value: str) -> str:
    normalized = value.strip().lower()
    fragments: list[str] = []
    previous_dash = False
    for char in normalized:
        if char.isalnum():
            fragments.append(char)
            previous_dash = False
            continue
        if not previous_dash:
            fragments.append("-")
            previous_dash = True
    return "".join(fragments).strip("-")


def _resolved_outcome(signal: MarketSignal) -> float | None:
    return normalize_resolved_outcome(signal.external_signal.get("resolved_outcome"))


def _signal_with_replay_resolution(
    signal: MarketSignal,
    replay_engine: object,
    *,
    evaluation_as_of: datetime | None = None,
) -> MarketSignal:
    resolver = getattr(replay_engine, "resolution_for", None)
    if not callable(resolver):
        return signal
    cutoff = evaluation_as_of or datetime.now(tz=UTC)
    parameters = inspect.signature(resolver).parameters
    accepts_cutoff = "as_of" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_cutoff:
        resolved_value = resolver(signal.market_id, as_of=cutoff)
    else:
        # Keep compatibility with test/research replay adapters that expose
        # only the historical one-argument resolver contract.  Do not retry
        # after a resolver that accepted ``as_of`` raised internally.
        resolved_value = resolver(signal.market_id)
    resolved = normalize_resolved_outcome(resolved_value)
    if resolved is None:
        return signal
    return replace(
        signal,
        external_signal={**signal.external_signal, "resolved_outcome": resolved},
    )


def _pnl_delta(
    *,
    signal: MarketSignal,
    decision_outcome: str,
    fill: FillRecord,
    execution_model: ExecutionModel,
) -> Decimal:
    resolved_outcome = _resolved_outcome(signal)
    if resolved_outcome is None:
        return Decimal("0")

    notional = Decimal(str(fill.fill_notional_usdc))
    fill_price_decimal = Decimal(str(fill.fill_price))
    resolved = Decimal(str(resolved_outcome))
    if decision_outcome == "YES":
        shares = Decimal(str(fill.fill_quantity))
        payout = shares * resolved
    else:
        shares = Decimal(str(fill.fill_quantity))
        payout = shares * (Decimal("1") - resolved)
    fee = Decimal(
        str(
            fill.fees
            if fill.fees is not None
            else execution_model.compute_fee(
                notional_usdc=float(notional),
                fill_price=float(fill_price_decimal),
            )
        )
    )
    return payout - notional - fee


async def _advance_open_orders(
    execution_simulator: BacktestExecutionSimulatorLike,
    *,
    signal: MarketSignal,
    execution_model: ExecutionModel,
) -> list[OrderState]:
    method = getattr(execution_simulator, "advance", None)
    if method is None:
        return []
    result = method(signal=signal, execution_model=execution_model)
    if asyncio.iscoroutine(result):
        return cast(list[OrderState], await result)
    return cast(list[OrderState], result)


async def _cancel_open_orders(
    execution_simulator: BacktestExecutionSimulatorLike,
    *,
    session_end: datetime,
) -> list[OrderState]:
    method = getattr(execution_simulator, "cancel_open_orders", None)
    if method is None:
        return []
    result = method(session_end=session_end)
    if asyncio.iscoroutine(result):
        return cast(list[OrderState], await result)
    return cast(list[OrderState], result)


def _failure_reason(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


__all__ = ["BacktestRunner"]
