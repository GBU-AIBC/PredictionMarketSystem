from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in red phase
        pytest.fail(f"{module_name} is missing: {exc}")

    return getattr(module, symbol_name)


def _build_projections() -> dict[str, Any]:
    factor_composition_step = _load_symbol("pms.strategies.projections", "FactorCompositionStep")
    strategy_config = _load_symbol("pms.strategies.projections", "StrategyConfig")
    risk_params = _load_symbol("pms.strategies.projections", "RiskParams")
    eval_spec = _load_symbol("pms.strategies.projections", "EvalSpec")
    forecaster_spec = _load_symbol("pms.strategies.projections", "ForecasterSpec")
    market_selection_spec = _load_symbol(
        "pms.strategies.projections",
        "MarketSelectionSpec",
    )
    calibration_spec = _load_symbol("pms.strategies.projections", "CalibrationSpec")

    return {
        "config": strategy_config(
            strategy_id="default",
            factor_composition=(
                factor_composition_step(
                    factor_id="yes_price",
                    role="weighted",
                    param="",
                    weight=1.0,
                    threshold=None,
                ),
            ),
            metadata=(("owner", "system"),),
        ),
        "risk": risk_params(
            max_position_notional_usdc=100.0,
            max_daily_drawdown_pct=2.5,
            min_order_size_usdc=1.0,
        ),
        "eval_spec": eval_spec(metrics=("brier", "pnl")),
        "forecaster": forecaster_spec(
            forecasters=(("rules", (("threshold", "0.55"),)),),
        ),
        "market_selection": market_selection_spec(
            venue="polymarket",
            resolution_time_max_horizon_days=14,
            volume_min_usdc=250.0,
        ),
        "calibration": calibration_spec(),
    }


def test_strategy_init_requires_keyword_only_projection_args() -> None:
    strategy = _load_symbol("pms.strategies.aggregate", "Strategy")
    signature = inspect.signature(strategy)

    assert list(signature.parameters) == [
        "config",
        "risk",
        "eval_spec",
        "forecaster",
        "market_selection",
        "calibration",
    ]
    for parameter in signature.parameters.values():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        if parameter.name == "calibration":
            assert parameter.default is None
        else:
            assert parameter.default is inspect.Parameter.empty


def test_strategy_returns_cached_projection_references_and_fixed_snapshot_order() -> None:
    strategy = _load_symbol("pms.strategies.aggregate", "Strategy")
    projections = _build_projections()

    aggregate = strategy(**projections)

    assert aggregate.config is projections["config"]
    assert aggregate.risk is projections["risk"]
    assert aggregate.eval_spec is projections["eval_spec"]
    assert aggregate.forecaster is projections["forecaster"]
    assert aggregate.market_selection is projections["market_selection"]
    assert aggregate.calibration is projections["calibration"]
    assert aggregate.snapshot() == (
        projections["config"],
        projections["risk"],
        projections["eval_spec"],
        projections["forecaster"],
        projections["market_selection"],
        projections["calibration"],
    )


@pytest.mark.parametrize(
    "missing_kwarg",
    ["config", "risk", "eval_spec", "forecaster", "market_selection"],
)
def test_strategy_missing_projection_kwarg_raises_type_error(
    missing_kwarg: str,
) -> None:
    strategy = _load_symbol("pms.strategies.aggregate", "Strategy")
    projections = _build_projections()
    projections.pop(missing_kwarg)

    with pytest.raises(TypeError, match=missing_kwarg):
        strategy(**projections)


@pytest.mark.parametrize(
    "none_kwarg",
    ["config", "risk", "eval_spec", "forecaster", "market_selection"],
)
def test_strategy_none_projection_kwarg_raises_type_error(none_kwarg: str) -> None:
    strategy = _load_symbol("pms.strategies.aggregate", "Strategy")
    projections = _build_projections()
    projections[none_kwarg] = None

    with pytest.raises(TypeError, match=none_kwarg):
        strategy(**projections)


def test_strategy_surface_excludes_select_markets() -> None:
    aggregate_module = importlib.import_module("pms.strategies.aggregate")
    strategy = getattr(aggregate_module, "Strategy")

    assert not hasattr(strategy, "select_markets")
    assert "select_markets" not in (inspect.getdoc(aggregate_module) or "")
    assert "select_markets" not in (inspect.getdoc(strategy) or "")
