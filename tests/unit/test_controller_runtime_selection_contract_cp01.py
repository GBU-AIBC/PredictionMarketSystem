from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
import importlib
from typing import Any

import pytest

from pms.config import ControllerSettings, PMSSettings, RiskSettings
from pms.controller.calibrators.netcal import NetcalCalibrator
from pms.controller.pipeline import ControllerPipeline
from pms.controller.router import Router
from pms.controller.sizers.kelly import KellySizer
from pms.core.models import MarketSignal, Portfolio
from pms.core.enums import RunMode
from pms.strategies.projections import (
    ActiveStrategy,
    EvalSpec,
    FactorCompositionStep,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - red phase
        pytest.fail(f"{module_name} is missing: {exc}")
    return getattr(module, symbol_name)


def _signal(
    *,
    yes_price: float = 0.4,
    external_signal: dict[str, Any] | None = None,
) -> MarketSignal:
    return MarketSignal(
        market_id="market-runtime-contract",
        token_id="yes-token",
        venue="polymarket",
        title="Will the runtime contract close?",
        yes_price=yes_price,
        volume_24h=1_000.0,
        resolves_at=datetime(2026, 4, 30, tzinfo=UTC),
        orderbook={
            "bids": [{"price": 0.39, "size": 10.0}],
            "asks": [{"price": 0.41, "size": 10.0}],
        },
        external_signal=(
            external_signal if external_signal is not None else {"fair_value": 0.58}
        ),
        fetched_at=datetime(2026, 4, 20, 12, 0, tzinfo=UTC),
        market_status="open",
    )


def _portfolio() -> Portfolio:
    return Portfolio(
        total_usdc=1_000.0,
        free_usdc=1_000.0,
        locked_usdc=0.0,
        open_positions=[],
    )


def _step(
    factor_id: str,
    *,
    role: str,
    weight: float = 1.0,
    param: str = "",
    threshold: float | None = None,
) -> FactorCompositionStep:
    return FactorCompositionStep(
        factor_id=factor_id,
        role=role,
        param=param,
        weight=weight,
        threshold=threshold,
    )


def _active_strategy(
    *,
    strategy_id: str = "alpha",
    strategy_version_id: str = "alpha-v1",
    factor_composition: tuple[FactorCompositionStep, ...],
    forecaster_names: tuple[str, ...],
) -> ActiveStrategy:
    return ActiveStrategy(
        strategy_id=strategy_id,
        strategy_version_id=strategy_version_id,
        config=StrategyConfig(
            strategy_id=strategy_id,
            factor_composition=factor_composition,
            metadata=(("owner", "test"),),
        ),
        risk=RiskParams(
            max_position_notional_usdc=100.0,
            max_daily_drawdown_pct=2.5,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(metrics=("brier", "pnl")),
        forecaster=ForecasterSpec(
            forecasters=tuple((name, ()) for name in forecaster_names)
        ),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=500.0,
        ),
    )


class StaticForecaster:
    def __init__(self, probability: float) -> None:
        self._probability = probability

    def predict(self, signal: MarketSignal) -> tuple[float, float, str]:
        del signal
        return (self._probability, 0.9, f"static-{self._probability}")

    async def forecast(self, signal: MarketSignal) -> float:
        del signal
        return self._probability


class RecordingFactorReader:
    def __init__(self, snapshot: Any) -> None:
        self.snapshot_value = snapshot
        self.calls: list[dict[str, Any]] = []

    async def snapshot(
        self,
        *,
        market_id: str,
        as_of: datetime,
        required: Sequence[FactorCompositionStep],
        strategy_id: str,
        strategy_version_id: str,
    ) -> Any:
        self.calls.append(
            {
                "market_id": market_id,
                "as_of": as_of,
                "required": tuple(required),
                "strategy_id": strategy_id,
                "strategy_version_id": strategy_version_id,
            }
        )
        return self.snapshot_value


@pytest.mark.asyncio
async def test_controller_pipeline_uses_strategy_factor_composition_not_forecaster_mean() -> None:
    factor_snapshot_cls = _load_symbol(
        "pms.controller.factor_snapshot",
        "FactorSnapshot",
    )
    strategy = _active_strategy(
        factor_composition=(
            _step("snapshot_probability", role="runtime_probability"),
        ),
        forecaster_names=("rules", "llm"),
    )
    factor_reader = RecordingFactorReader(
        factor_snapshot_cls(
            values={( "snapshot_probability", ""): 0.72},
            missing_factors=(),
            snapshot_hash="snapshot-72",
        )
    )
    pipeline = ControllerPipeline(
        strategy=strategy,
        factor_reader=factor_reader,
        forecasters=(StaticForecaster(0.10), StaticForecaster(0.90)),
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0)),
    )

    emission = await pipeline.on_signal(_signal(), portfolio=_portfolio())

    assert emission is not None
    opportunity, decision = emission
    assert decision.prob_estimate == pytest.approx(0.72)
    assert decision.expected_edge == pytest.approx(0.32)
    assert opportunity.expected_edge == pytest.approx(0.32)
    assert opportunity.selected_factor_values == {
        "orderbook_imbalance": pytest.approx(0.0),
        "yes_price": pytest.approx(0.4),
        "rules": pytest.approx(0.10),
        "llm": pytest.approx(0.90),
        "snapshot_probability": pytest.approx(0.72),
    }
    assert opportunity.factor_snapshot_hash == "snapshot-72"
    assert opportunity.composition_trace == {
        "selected_probability": pytest.approx(0.72),
        "expected_edge": pytest.approx(0.32),
        "factor_snapshot_hash": "snapshot-72",
        "missing_factors": [],
        "branch_probabilities": {"snapshot_probability": pytest.approx(0.72)},
        "yes_probability": pytest.approx(0.72),
        "yes_reference_price": pytest.approx(0.4),
        "yes_edge": pytest.approx(0.32),
        "traded_outcome": "YES",
        "traded_probability": pytest.approx(0.72),
        "traded_price": pytest.approx(0.4),
        "traded_edge": pytest.approx(0.32),
        "gross_edge": pytest.approx(0.32),
        "spread_bps_at_decision": 500,
        # yes_price frame (no price_reference metadata): the executable ask is
        # unknown, so the full spread stays a charged entry cost.
        "spread_edge": pytest.approx(0.02),
        "spread_already_in_price": False,
        "fee_rate": pytest.approx(0.07),
        "fee_edge": pytest.approx(0.042),
        "max_slippage_bps": 50,
        "slippage_edge": pytest.approx(0.002),
        "net_edge_after_costs": pytest.approx(0.256),
    }
    assert factor_reader.calls == [
        {
            "market_id": "market-runtime-contract",
            "as_of": datetime(2026, 4, 20, 12, 0, tzinfo=UTC),
            "required": strategy.config.factor_composition,
            "strategy_id": "alpha",
            "strategy_version_id": "alpha-v1",
        }
    ]


@pytest.mark.asyncio
async def test_controller_pipeline_drops_non_finite_factor_without_decision() -> None:
    factor_snapshot_cls = _load_symbol(
        "pms.controller.factor_snapshot",
        "FactorSnapshot",
    )
    strategy = _active_strategy(
        factor_composition=(
            _step("snapshot_probability", role="runtime_probability"),
        ),
        forecaster_names=("rules",),
    )
    pipeline = ControllerPipeline(
        strategy=strategy,
        factor_reader=RecordingFactorReader(
            factor_snapshot_cls(
                values={("snapshot_probability", ""): float("nan")},
                missing_factors=(),
                snapshot_hash="nan-factor",
            )
        ),
        forecasters=(StaticForecaster(0.7),),
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=100.0)),
        router=Router(ControllerSettings(min_volume=0.0)),
        settings=PMSSettings(mode=RunMode.PAPER),
    )

    emission = await pipeline.on_signal(_signal(), portfolio=_portfolio())

    assert emission is None
    assert pipeline.last_diagnostic is not None
    assert pipeline.last_diagnostic.code == "composition_resolution_failed"


@pytest.mark.asyncio
async def test_controller_uses_signal_fee_rate_bps_for_decision_cost_evidence() -> None:
    factor_snapshot_cls = _load_symbol("pms.controller.factor_snapshot", "FactorSnapshot")
    strategy = _active_strategy(
        factor_composition=(
            _step("snapshot_probability", role="runtime_probability"),
        ),
        forecaster_names=("rules",),
    )
    factor_reader = RecordingFactorReader(
        factor_snapshot_cls(
            values={("snapshot_probability", ""): 0.72},
            missing_factors=(),
            snapshot_hash="snapshot-72",
        )
    )
    pipeline = ControllerPipeline(
        strategy=strategy,
        factor_reader=factor_reader,
        forecasters=(StaticForecaster(0.10),),
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0)),
    )

    emission = await pipeline.on_signal(
        _signal(external_signal={"fair_value": 0.58, "fee_rate_bps": "300"}),
        portfolio=_portfolio(),
    )

    assert emission is not None
    opportunity, _ = emission
    assert opportunity.composition_trace["fee_rate"] == pytest.approx(0.03)
    assert opportunity.composition_trace["fee_edge"] == pytest.approx(0.018)
    assert opportunity.composition_trace["net_edge_after_costs"] == pytest.approx(
        0.28
    )


@pytest.mark.asyncio
async def test_controller_pipeline_maps_stats_forecaster_to_statistical_runtime_factor() -> None:
    strategy = _active_strategy(
        factor_composition=(
            _step("statistical", role="runtime_probability"),
        ),
        forecaster_names=("stats",),
    )
    factor_snapshot_cls = _load_symbol(
        "pms.controller.factor_snapshot",
        "FactorSnapshot",
    )
    pipeline = ControllerPipeline(
        strategy=strategy,
        factor_reader=RecordingFactorReader(
            factor_snapshot_cls(values={}, missing_factors=(), snapshot_hash=None)
        ),
        forecasters=(StaticForecaster(0.81),),
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0)),
    )

    emission = await pipeline.on_signal(_signal(yes_price=0.4), portfolio=_portfolio())

    assert emission is not None
    _, decision = emission
    assert decision.prob_estimate == pytest.approx(0.81)


def test_controller_pipeline_factory_passes_strategy_and_factor_reader() -> None:
    factor_snapshot_cls = _load_symbol(
        "pms.controller.factor_snapshot",
        "FactorSnapshot",
    )
    factory_cls = _load_symbol("pms.controller.factory", "ControllerPipelineFactory")
    strategy = _active_strategy(
        factor_composition=(
            _step("snapshot_probability", role="runtime_probability"),
        ),
        forecaster_names=("rules",),
    )
    factor_reader = RecordingFactorReader(
        factor_snapshot_cls(values={}, missing_factors=(), snapshot_hash="snapshot")
    )

    direct_book_reader = object()
    factory = factory_cls(
        factor_reader=factor_reader,
        direct_book_reader=direct_book_reader,
    )
    pipeline = factory.build(strategy)

    assert pipeline.strategy == strategy
    assert pipeline.factor_reader is factor_reader
    assert pipeline.direct_book_reader is direct_book_reader


def test_snapshot_hash_distinguishes_missing_factors_and_strategy_identity() -> None:
    snapshot_hash = _load_symbol("pms.controller.factor_snapshot", "_snapshot_hash")

    with_missing = snapshot_hash(
        market_id="market-runtime-contract",
        as_of=datetime(2026, 4, 20, 12, 0, tzinfo=UTC),
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        required_keys=(("rules", ""), ("llm", "")),
        values={("rules", ""): 0.55},
        missing_factors=(("llm", ""),),
    )
    without_missing = snapshot_hash(
        market_id="market-runtime-contract",
        as_of=datetime(2026, 4, 20, 12, 0, tzinfo=UTC),
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        required_keys=(("rules", ""),),
        values={("rules", ""): 0.55},
        missing_factors=(),
    )
    other_strategy = snapshot_hash(
        market_id="market-runtime-contract",
        as_of=datetime(2026, 4, 20, 12, 0, tzinfo=UTC),
        strategy_id="beta",
        strategy_version_id="beta-v2",
        required_keys=(("rules", ""), ("llm", "")),
        values={("rules", ""): 0.55},
        missing_factors=(("llm", ""),),
    )

    assert with_missing != without_missing
    assert with_missing != other_strategy


def test_trade_decision_asdict_exposes_normalized_order_intent_fields() -> None:
    decision = asdict(
        _load_symbol("pms.core.models", "TradeDecision")(
            decision_id="decision-runtime-contract",
            market_id="market-runtime-contract",
            token_id="yes-token",
            venue="polymarket",
            side="BUY",
            limit_price=0.4,
            notional_usdc=10.0,
            order_type="limit",
            max_slippage_bps=50,
            stop_conditions=[],
            prob_estimate=0.72,
            expected_edge=0.32,
            time_in_force="GTC",
            opportunity_id="opportunity-runtime-contract",
            strategy_id="alpha",
            strategy_version_id="alpha-v1",
            model_id="ensemble",
        )
    )

    assert decision["side"] == "BUY"
    assert decision["limit_price"] == pytest.approx(0.4)
    assert decision["notional_usdc"] == pytest.approx(10.0)
    assert decision["action"] is None
