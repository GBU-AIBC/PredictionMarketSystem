from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import re
from typing import Any, cast

import pytest

from pms.core.enums import TimeInForce
from pms.core.models import MarketSignal, Opportunity, Portfolio, TradeDecision
from pms.research.entities import (
    PortfolioTarget,
    deserialize_portfolio_target_json,
    serialize_portfolio_target_json,
)
from pms.research.runner import BacktestRunner
from pms.research.specs import (
    BacktestDataset,
    BacktestExecutionConfig,
    BacktestSpec,
    ExecutionModel,
)
from pms.strategies.projections import (
    ActiveStrategy,
    EvalSpec,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)


def test_portfolio_target_round_trips_through_json() -> None:
    target = PortfolioTarget(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        targets={
            ("market-a", "token-a", "buy_yes", datetime(2026, 4, 20, tzinfo=UTC)): 12.5,
            ("market-b", "token-b", "buy_no", datetime(2026, 4, 21, tzinfo=UTC)): 7.0,
        },
    )

    encoded = serialize_portfolio_target_json(target)
    decoded = deserialize_portfolio_target_json(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        raw_value=encoded,
    )

    assert decoded == target


def test_portfolio_target_round_trips_empty_targets() -> None:
    target = PortfolioTarget(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        targets={},
    )

    encoded = serialize_portfolio_target_json(target)
    decoded = deserialize_portfolio_target_json(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        raw_value=encoded,
    )

    assert decoded == target


def test_portfolio_target_remains_research_only() -> None:
    forbidden_pattern = re.compile(
        r"class PortfolioTarget\b|from pms\.research.*import.*PortfolioTarget"
    )

    for root in (
        Path("src/pms/controller"),
        Path("src/pms/actuator"),
        Path("src/pms/sensor"),
    ):
        for path in root.rglob("*.py"):
            assert forbidden_pattern.search(path.read_text(encoding="utf-8")) is None


class _ReplayEngine:
    def __init__(self, signals: list[MarketSignal]) -> None:
        self._signals = signals

    async def stream(
        self,
        spec: BacktestSpec,
        exec_config: BacktestExecutionConfig,
    ) -> AsyncIterator[MarketSignal]:
        del spec, exec_config
        for signal in self._signals:
            yield signal


class _PortfolioRecordingPipeline:
    def __init__(self) -> None:
        self.portfolios: list[Portfolio] = []

    async def on_signal(
        self,
        signal: MarketSignal,
        portfolio: Portfolio | None = None,
    ) -> tuple[Opportunity, TradeDecision] | None:
        assert portfolio is not None
        self.portfolios.append(portfolio)
        emission_index = len(self.portfolios)
        return (
            Opportunity(
                opportunity_id=f"opp-{emission_index}",
                market_id=signal.market_id,
                token_id=cast(str, signal.token_id),
                side="yes",
                selected_factor_values={},
                expected_edge=0.1,
                rationale="portfolio-state-check",
                target_size_usdc=10.0,
                expiry=signal.resolves_at,
                staleness_policy="research",
                strategy_id="alpha",
                strategy_version_id="alpha-v1",
                created_at=signal.fetched_at,
            ),
            TradeDecision(
                decision_id=f"decision-{emission_index}",
                market_id=signal.market_id,
                token_id=signal.token_id,
                venue=signal.venue,
                side="BUY",
                limit_price=signal.yes_price,
                notional_usdc=10.0,
                order_type="limit",
                max_slippage_bps=50,
                stop_conditions=[],
                prob_estimate=0.7,
                expected_edge=0.2,
                time_in_force=TimeInForce.IOC,
                opportunity_id=f"opp-{emission_index}",
                strategy_id="alpha",
                strategy_version_id="alpha-v1",
                model_id="rules",
            ),
        )


class _PipelineFactory:
    def __init__(self, pipeline: _PortfolioRecordingPipeline) -> None:
        self._pipeline = pipeline

    def build(self, strategy: ActiveStrategy) -> _PortfolioRecordingPipeline:
        del strategy
        return self._pipeline


class _GtcUnfilledPipeline(_PortfolioRecordingPipeline):
    async def on_signal(
        self,
        signal: MarketSignal,
        portfolio: Portfolio | None = None,
    ) -> tuple[Opportunity, TradeDecision] | None:
        emission = await super().on_signal(signal, portfolio=portfolio)
        assert emission is not None
        opportunity, decision = emission
        return (
            opportunity,
            replace(
                decision,
                time_in_force=TimeInForce.GTC,
                limit_price=0.4,
            ),
        )


def _active_strategy() -> ActiveStrategy:
    return ActiveStrategy(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        config=StrategyConfig(
            strategy_id="alpha",
            factor_composition=(),
            metadata=(("owner", "test"),),
        ),
        risk=RiskParams(
            max_position_notional_usdc=100.0,
            max_daily_drawdown_pct=2.5,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(metrics=("brier", "pnl")),
        forecaster=ForecasterSpec(forecasters=(("rules", ()),)),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=500.0,
        ),
    )


def _signal(ts: datetime) -> MarketSignal:
    return MarketSignal(
        market_id="research-runner-market",
        token_id="yes-token",
        venue="polymarket",
        title="Will CP04b runner tests pass?",
        yes_price=0.41,
        volume_24h=1000.0,
        resolves_at=datetime(2026, 4, 30, tzinfo=UTC),
        orderbook={
            "bids": [{"price": 0.39, "size": 100.0}],
            "asks": [{"price": 0.41, "size": 100.0}],
        },
        external_signal={"resolved_outcome": 1.0},
        fetched_at=ts,
        market_status="open",
    )


def _spec() -> BacktestSpec:
    return BacktestSpec(
        strategy_versions=(("alpha", "alpha-v1"),),
        dataset=BacktestDataset(
            source="fixture",
            version="v1",
            coverage_start=datetime(2026, 4, 1, tzinfo=UTC),
            coverage_end=datetime(2026, 4, 30, tzinfo=UTC),
            market_universe_filter={"market_ids": ["research-runner-market"]},
            data_quality_gaps=(),
        ),
        execution_model=ExecutionModel(
            fee_rate=0.0,
            slippage_bps=0.0,
            latency_ms=0.0,
            staleness_ms=60000.0,
            fill_policy="immediate_or_cancel",
        ),
        risk_policy=RiskParams(
            max_position_notional_usdc=100.0,
            max_daily_drawdown_pct=2.5,
            min_order_size_usdc=1.0,
        ),
        date_range_start=datetime(2026, 4, 1, tzinfo=UTC),
        date_range_end=datetime(2026, 4, 30, tzinfo=UTC),
    )


def test_strategy_accumulator_scores_buy_no_brier_on_yes_probability_basis() -> None:
    from pms.research.runner import _StrategyAccumulator

    signal = replace(
        _signal(datetime(2026, 4, 20, 0, 0, tzinfo=UTC)),
        token_id="no-token",
        external_signal={"resolved_outcome": 0.0},
    )
    opportunity = Opportunity(
        opportunity_id="opp-buy-no",
        market_id=signal.market_id,
        token_id="no-token",
        side="no",
        selected_factor_values={},
        expected_edge=0.1,
        rationale="buy-no-brier-check",
        target_size_usdc=10.0,
        expiry=signal.resolves_at,
        staleness_policy="research",
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        created_at=signal.fetched_at,
    )
    decision = TradeDecision(
        decision_id="decision-buy-no",
        market_id=signal.market_id,
        token_id="no-token",
        venue=signal.venue,
        side="BUY",
        limit_price=0.3,
        notional_usdc=10.0,
        order_type="limit",
        max_slippage_bps=50,
        stop_conditions=[],
        prob_estimate=0.7,
        expected_edge=0.2,
        time_in_force=TimeInForce.IOC,
        opportunity_id="opp-buy-no",
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        model_id="rules",
        outcome="NO",
    )
    accumulator = _StrategyAccumulator(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        execution_model=ExecutionModel.polymarket_paper(),
    )

    accumulator.record_decision(
        signal=signal,
        opportunity=opportunity,
        decision=decision,
    )

    args = accumulator.as_insert_args(
        run_id="11111111-1111-1111-1111-111111111111"
    )
    assert args[4] == pytest.approx(0.09)


@pytest.mark.asyncio
async def test_backtest_runner_updates_portfolio_between_replay_signals() -> None:
    pipeline = _PortfolioRecordingPipeline()
    runner = BacktestRunner(
        writable_pool=cast(Any, object()),
        readonly_pool=cast(Any, object()),
        replay_engine=_ReplayEngine(
            [
                _signal(datetime(2026, 4, 20, 0, 0, tzinfo=UTC)),
                _signal(datetime(2026, 4, 20, 0, 1, tzinfo=UTC)),
            ]
        ),
        controller_factory=_PipelineFactory(pipeline),
    )

    accumulator = await runner._run_strategy(
        strategy=_active_strategy(),
        spec=_spec(),
        exec_config=BacktestExecutionConfig(),
    )

    assert accumulator.fill_count == 2
    assert len(pipeline.portfolios) == 2
    assert pipeline.portfolios[0].free_usdc == pytest.approx(1000.0)
    assert pipeline.portfolios[0].open_positions == []
    assert pipeline.portfolios[1].free_usdc == pytest.approx(990.0)
    assert pipeline.portfolios[1].locked_usdc == pytest.approx(10.0)
    assert len(pipeline.portfolios[1].open_positions) == 1
    expected_fill_price = 0.41 * (1.0 + _spec().execution_model.slippage_bps / 10_000.0)
    assert pipeline.portfolios[1].open_positions[0].shares_held == pytest.approx(
        10.0 / expected_fill_price
    )
    assert accumulator.execution_rows[0].decision_id == "decision-1"
    assert accumulator.execution_rows[0].status == "filled"
    assert accumulator.execution_rows[0].market_id == "research-runner-market"
    assert accumulator.execution_rows[0].slippage_bps == pytest.approx(0.0)
    assert accumulator.execution_rows[0].pnl is not None


@pytest.mark.asyncio
async def test_backtest_runner_propagates_unexpected_actuator_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _PortfolioRecordingPipeline()
    runner = BacktestRunner(
        writable_pool=cast(Any, object()),
        readonly_pool=cast(Any, object()),
        replay_engine=_ReplayEngine([_signal(datetime(2026, 4, 20, 0, 0, tzinfo=UTC))]),
        controller_factory=_PipelineFactory(pipeline),
    )

    async def _boom(
        self: object,
        *,
        signal: object,
        decision: object,
        portfolio: object | None = None,
        execution_model: object,
    ) -> object:
        del self, signal, decision, portfolio, execution_model
        raise RuntimeError("execution simulator exploded")

    monkeypatch.setattr("pms.research.execution.BacktestExecutionSimulator.execute", _boom)

    with pytest.raises(RuntimeError, match="execution simulator exploded"):
        await runner._run_strategy(
            strategy=_active_strategy(),
            spec=_spec(),
            exec_config=BacktestExecutionConfig(),
        )


class _InsertTrackingTransaction:
    def __init__(self, journal: list[str]) -> None:
        self._journal = journal

    async def __aenter__(self) -> None:
        self._journal.append("BEGIN")

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc_type is not None:
            self._journal.append("ROLLBACK")
        else:
            self._journal.append("COMMIT")
        return False  # propagate exceptions


class _InsertTrackingConnection:
    def __init__(self, journal: list[str]) -> None:
        self._journal = journal

    def transaction(self) -> _InsertTrackingTransaction:
        return _InsertTrackingTransaction(self._journal)

    async def execute(self, query: str, *args: object) -> str:
        del args
        if "INSERT INTO strategy_runs" in query:
            self._journal.append("INSERT strategy_runs")
        if "INSERT INTO backtest_execution_rows" in query:
            self._journal.append("INSERT backtest_execution_rows")
        if "INSERT INTO strategy_run_slices" in query:
            self._journal.append("INSERT strategy_run_slices")
        return "INSERT 1"


class _InsertTrackingPool:
    def __init__(self, journal: list[str]) -> None:
        self._journal = journal
        self._connection = _InsertTrackingConnection(journal)

    async def acquire(self) -> _InsertTrackingConnection:
        return self._connection

    async def release(self, connection: _InsertTrackingConnection) -> None:
        del connection


@pytest.mark.asyncio
async def test_insert_strategy_runs_atomically_rolls_back_on_mid_batch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strategy_runs insert batch must be all-or-nothing: if a later row
    raises, none of the earlier rows stay committed."""
    journal: list[str] = []
    pool = _InsertTrackingPool(journal)
    runner = BacktestRunner(
        writable_pool=cast(Any, pool),
        readonly_pool=cast(Any, object()),
    )

    # Build two accumulators; make the second one's insert raise.
    from pms.research.runner import _StrategyAccumulator

    acc_one = _StrategyAccumulator(
        strategy_id="alpha",
        strategy_version_id="v1",
        execution_model=ExecutionModel.polymarket_paper(),
    )
    acc_two = _StrategyAccumulator(
        strategy_id="beta",
        strategy_version_id="v1",
        execution_model=ExecutionModel.polymarket_paper(),
    )

    call_count = {"execute": 0}
    original_execute = pool._connection.execute

    async def _maybe_boom(query: str, *args: object) -> str:
        call_count["execute"] += 1
        if call_count["execute"] == 2:
            raise RuntimeError("simulated DB failure")
        return await original_execute(query, *args)

    monkeypatch.setattr(pool._connection, "execute", _maybe_boom)

    with pytest.raises(RuntimeError, match="simulated DB failure"):
        await runner._insert_strategy_runs_atomically(
            run_id="11111111-1111-1111-1111-111111111111",
            accumulators=[acc_one, acc_two],
        )

    assert journal[0] == "BEGIN"
    assert journal[-1] == "ROLLBACK"
    assert "COMMIT" not in journal


@pytest.mark.asyncio
async def test_backtest_runner_records_chunk_day_slice_metrics() -> None:
    pipeline = _PortfolioRecordingPipeline()
    runner = BacktestRunner(
        writable_pool=cast(Any, object()),
        readonly_pool=cast(Any, object()),
        replay_engine=_ReplayEngine(
            [
                _signal(datetime(2026, 4, 1, 12, 0, tzinfo=UTC)),
                _signal(datetime(2026, 4, 8, 12, 0, tzinfo=UTC)),
            ]
        ),
        controller_factory=_PipelineFactory(pipeline),
    )

    accumulator = await runner._run_strategy(
        strategy=_active_strategy(),
        spec=_spec(),
        exec_config=BacktestExecutionConfig(chunk_days=7),
    )

    slice_args = getattr(accumulator, "slice_insert_args", None)
    assert callable(slice_args)
    rows = slice_args(run_id="11111111-1111-1111-1111-111111111111")
    walk_forward_rows = [row for row in rows if row[7] == "walk_forward"]

    assert len(walk_forward_rows) == 2
    assert walk_forward_rows[0][4] == "2026-04-01/2026-04-08"
    assert walk_forward_rows[0][5] == datetime(2026, 4, 1, tzinfo=UTC)
    assert walk_forward_rows[0][6] == datetime(2026, 4, 8, tzinfo=UTC)
    assert walk_forward_rows[0][8] == pytest.approx(0.09)
    assert walk_forward_rows[0][9] is not None
    assert walk_forward_rows[0][10] is not None
    assert walk_forward_rows[0][11] == pytest.approx(1.0)
    assert walk_forward_rows[0][12] == pytest.approx(0.0)
    assert walk_forward_rows[0][13] == 1
    assert walk_forward_rows[0][14] == 1
    assert walk_forward_rows[0][15] == 1
    assert walk_forward_rows[1][4] == "2026-04-08/2026-04-15"
    assert walk_forward_rows[1][5] == datetime(2026, 4, 8, tzinfo=UTC)
    assert walk_forward_rows[1][6] == datetime(2026, 4, 15, tzinfo=UTC)


@pytest.mark.asyncio
async def test_backtest_runner_records_category_and_liquidity_slice_metrics() -> None:
    pipeline = _PortfolioRecordingPipeline()
    categorized_signal = replace(
        _signal(datetime(2026, 4, 1, 12, 0, tzinfo=UTC)),
        volume_24h=2_500.0,
        external_signal={"resolved_outcome": 1.0, "category": "Politics"},
    )
    runner = BacktestRunner(
        writable_pool=cast(Any, object()),
        readonly_pool=cast(Any, object()),
        replay_engine=_ReplayEngine([categorized_signal]),
        controller_factory=_PipelineFactory(pipeline),
    )

    accumulator = await runner._run_strategy(
        strategy=_active_strategy(),
        spec=_spec(),
        exec_config=BacktestExecutionConfig(chunk_days=7),
    )

    rows = accumulator.slice_insert_args(
        run_id="11111111-1111-1111-1111-111111111111"
    )
    rows_by_label = {cast(str, row[4]): row for row in rows}

    assert "category:politics" in rows_by_label
    assert rows_by_label["category:politics"][5] == _spec().date_range_start
    assert rows_by_label["category:politics"][6] == _spec().date_range_end
    assert rows_by_label["category:politics"][7] == "category"
    assert rows_by_label["category:politics"][8] == pytest.approx(0.09)
    assert rows_by_label["category:politics"][11] == pytest.approx(1.0)
    assert rows_by_label["category:politics"][13] == 1
    assert rows_by_label["category:politics"][14] == 1
    assert rows_by_label["category:politics"][15] == 1

    assert "liquidity:volume_24h:1000-10000" in rows_by_label
    assert rows_by_label["liquidity:volume_24h:1000-10000"][5] == _spec().date_range_start
    assert rows_by_label["liquidity:volume_24h:1000-10000"][6] == _spec().date_range_end
    assert rows_by_label["liquidity:volume_24h:1000-10000"][7] == "liquidity"
    assert rows_by_label["liquidity:volume_24h:1000-10000"][8] == pytest.approx(0.09)
    assert rows_by_label["liquidity:volume_24h:1000-10000"][11] == pytest.approx(1.0)
    assert rows_by_label["liquidity:volume_24h:1000-10000"][13] == 1
    assert rows_by_label["liquidity:volume_24h:1000-10000"][14] == 1
    assert rows_by_label["liquidity:volume_24h:1000-10000"][15] == 1


@pytest.mark.asyncio
async def test_insert_strategy_runs_atomically_persists_slice_rows() -> None:
    from pms.research.runner import _SliceDescriptor, _StrategyAccumulator

    journal: list[str] = []
    pool = _InsertTrackingPool(journal)
    runner = BacktestRunner(
        writable_pool=cast(Any, pool),
        readonly_pool=cast(Any, object()),
    )
    accumulator = _StrategyAccumulator(
        strategy_id="alpha",
        strategy_version_id="v1",
        execution_model=ExecutionModel.polymarket_paper(),
    )
    record_slice = getattr(accumulator, "record_slice_decision", None)
    assert callable(record_slice)
    record_slice(
        signal=_signal(datetime(2026, 4, 1, 12, 0, tzinfo=UTC)),
        opportunity=Opportunity(
            opportunity_id="opp-slice",
            market_id="research-runner-market",
            token_id="yes-token",
            side="yes",
            selected_factor_values={},
            expected_edge=0.1,
            rationale="slice-persist-check",
            target_size_usdc=10.0,
            expiry=datetime(2026, 4, 30, tzinfo=UTC),
            staleness_policy="research",
            strategy_id="alpha",
            strategy_version_id="v1",
            created_at=datetime(2026, 4, 1, 12, 0, tzinfo=UTC),
        ),
        decision=TradeDecision(
            decision_id="decision-slice",
            market_id="research-runner-market",
            token_id="yes-token",
            venue="polymarket",
            side="BUY",
            limit_price=0.41,
            notional_usdc=10.0,
            order_type="limit",
            max_slippage_bps=50,
            stop_conditions=[],
            prob_estimate=0.7,
            expected_edge=0.2,
            time_in_force=TimeInForce.IOC,
            opportunity_id="opp-slice",
            strategy_id="alpha",
            strategy_version_id="v1",
            model_id="rules",
        ),
        slice_descriptor=_SliceDescriptor(
            label="2026-04-01/2026-04-08",
            start=datetime(2026, 4, 1, tzinfo=UTC),
            end=datetime(2026, 4, 8, tzinfo=UTC),
            kind="walk_forward",
        ),
    )

    await runner._insert_strategy_runs_atomically(
        run_id="11111111-1111-1111-1111-111111111111",
        accumulators=[accumulator],
    )

    assert journal == [
        "BEGIN",
        "INSERT strategy_runs",
        "INSERT strategy_run_slices",
        "COMMIT",
    ]


@pytest.mark.asyncio
async def test_insert_strategy_runs_atomically_persists_execution_rows() -> None:
    pipeline = _PortfolioRecordingPipeline()
    runner = BacktestRunner(
        writable_pool=cast(Any, object()),
        readonly_pool=cast(Any, object()),
        replay_engine=_ReplayEngine([_signal(datetime(2026, 4, 20, 0, 0, tzinfo=UTC))]),
        controller_factory=_PipelineFactory(pipeline),
    )
    accumulator = await runner._run_strategy(
        strategy=_active_strategy(),
        spec=_spec(),
        exec_config=BacktestExecutionConfig(),
    )
    journal: list[str] = []
    insert_runner = BacktestRunner(
        writable_pool=cast(Any, _InsertTrackingPool(journal)),
        readonly_pool=cast(Any, object()),
    )

    await insert_runner._insert_strategy_runs_atomically(
        run_id="11111111-1111-1111-1111-111111111111",
        accumulators=[accumulator],
    )

    assert journal == [
        "BEGIN",
        "INSERT strategy_runs",
        "INSERT backtest_execution_rows",
        "INSERT strategy_run_slices",
        "INSERT strategy_run_slices",
        "COMMIT",
    ]


def _signal_without_resolution(ts: datetime) -> MarketSignal:
    base = _signal(ts)
    return MarketSignal(
        market_id=base.market_id,
        token_id=base.token_id,
        venue=base.venue,
        title=base.title,
        yes_price=base.yes_price,
        volume_24h=base.volume_24h,
        resolves_at=base.resolves_at,
        orderbook=base.orderbook,
        external_signal={"raw_event_type": "price_change"},  # no resolved_outcome
        fetched_at=base.fetched_at,
        market_status=base.market_status,
    )


@pytest.mark.asyncio
async def test_strategy_accumulator_emits_null_pnl_when_no_resolution_observed() -> None:
    """When the replay stream supplies no `resolved_outcome` (e.g. running on
    live-captured data for a window that hasn't resolved yet), cumulative_pnl
    must surface as NULL rather than a misleading 0.0."""
    pipeline = _PortfolioRecordingPipeline()
    runner = BacktestRunner(
        writable_pool=cast(Any, object()),
        readonly_pool=cast(Any, object()),
        replay_engine=_ReplayEngine(
            [_signal_without_resolution(datetime(2026, 4, 20, 0, 0, tzinfo=UTC))]
        ),
        controller_factory=_PipelineFactory(pipeline),
    )

    accumulator = await runner._run_strategy(
        strategy=_active_strategy(),
        spec=_spec(),
        exec_config=BacktestExecutionConfig(),
    )

    args = accumulator.as_insert_args(run_id="11111111-1111-1111-1111-111111111111")
    # args order: strategy_run_id, run_id, strategy_id, strategy_version_id,
    #             brier, pnl_cum, drawdown_max, fill_rate, slippage_bps, ...
    assert args[4] is None, "brier must be NULL when no resolutions observed"
    assert args[5] is None, "pnl_cum must be NULL when no resolutions observed"


@pytest.mark.asyncio
async def test_strategy_accumulator_emits_numeric_pnl_when_any_resolution_observed() -> None:
    """Calibration: pnl_cum is NOT nulled when at least one fill has a
    resolved_outcome. The `as_insert_args` nulling must not swallow real data."""
    pipeline = _PortfolioRecordingPipeline()
    runner = BacktestRunner(
        writable_pool=cast(Any, object()),
        readonly_pool=cast(Any, object()),
        replay_engine=_ReplayEngine(
            [_signal(datetime(2026, 4, 20, 0, 0, tzinfo=UTC))]
        ),
        controller_factory=_PipelineFactory(pipeline),
    )

    accumulator = await runner._run_strategy(
        strategy=_active_strategy(),
        spec=_spec(),
        exec_config=BacktestExecutionConfig(),
    )

    args = accumulator.as_insert_args(run_id="11111111-1111-1111-1111-111111111111")
    assert args[5] is not None, "pnl_cum must remain numeric when resolutions observed"


@pytest.mark.asyncio
async def test_backtest_runner_execution_model_staleness_budget_can_prevent_fill() -> None:
    pipeline = _PortfolioRecordingPipeline()
    runner = BacktestRunner(
        writable_pool=cast(Any, object()),
        readonly_pool=cast(Any, object()),
        replay_engine=_ReplayEngine([_signal(datetime(2026, 4, 20, 0, 0, tzinfo=UTC))]),
        controller_factory=_PipelineFactory(pipeline),
    )
    spec = BacktestSpec(
        strategy_versions=(("alpha", "alpha-v1"),),
        dataset=_spec().dataset,
        execution_model=ExecutionModel(
            fee_rate=0.0,
            slippage_bps=0.0,
            latency_ms=250.0,
            staleness_ms=100.0,
            fill_policy="immediate_or_cancel",
        ),
        risk_policy=_spec().risk_policy,
        date_range_start=_spec().date_range_start,
        date_range_end=_spec().date_range_end,
    )

    accumulator = await runner._run_strategy(
        strategy=_active_strategy(),
        spec=spec,
        exec_config=BacktestExecutionConfig(),
    )

    assert accumulator.decision_count == 1
    assert accumulator.fill_count == 0


@pytest.mark.asyncio
async def test_backtest_runner_records_resolution_cancellation_for_gtc_order() -> None:
    start = datetime(2026, 4, 20, tzinfo=UTC)
    cutoff = start + timedelta(seconds=1)
    runner = BacktestRunner(
        writable_pool=cast(Any, object()),
        readonly_pool=cast(Any, object()),
        replay_engine=_ReplayEngine([
            replace(_signal(start), resolves_at=cutoff),
            replace(_signal(cutoff), resolves_at=cutoff),
        ]),
        controller_factory=_PipelineFactory(_GtcUnfilledPipeline()),
    )
    spec = _spec()
    spec = replace(spec, execution_model=replace(
        spec.execution_model, fill_policy="good_til_cancelled",
    ))
    accumulator = await runner._run_strategy(
        strategy=_active_strategy(), spec=spec,
        exec_config=BacktestExecutionConfig(),
    )
    assert accumulator.fill_count == 0
    first = next(row for row in accumulator.execution_rows if row.decision_id == "decision-1")
    assert first.rejection_reason == "market_resolved_before_execution"


@pytest.mark.asyncio
async def test_backtest_runner_records_session_end_rejection_for_unfilled_gtc_order() -> None:
    pipeline = _GtcUnfilledPipeline()
    runner = BacktestRunner(
        writable_pool=cast(Any, object()),
        readonly_pool=cast(Any, object()),
        replay_engine=_ReplayEngine([_signal(datetime(2026, 4, 20, 0, 0, tzinfo=UTC))]),
        controller_factory=_PipelineFactory(pipeline),
    )
    spec = _spec()
    spec = replace(
        spec,
        execution_model=replace(
            spec.execution_model,
            fill_policy="good_til_cancelled",
        ),
    )

    accumulator = await runner._run_strategy(
        strategy=_active_strategy(),
        spec=spec,
        exec_config=BacktestExecutionConfig(),
    )

    assert accumulator.decision_count == 1
    assert accumulator.fill_count == 0
    assert accumulator.execution_rows[0].decision_id == "decision-1"
    assert accumulator.execution_rows[0].status == "rejected"
    assert accumulator.execution_rows[0].rejection_reason == "cancelled_session_end"
