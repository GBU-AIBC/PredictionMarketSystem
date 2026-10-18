from __future__ import annotations

import json

import pytest

from pms.storage.strategy_registry import (
    _strategy_from_config_json,
    _strategy_from_versioned_config_json,
    _strategy_to_config_json,
)
from pms.strategies.aggregate import Strategy
from pms.strategies.versioning import (
    compute_legacy_strategy_version_id,
    compute_strategy_version_id,
)
from pms.strategies.projections import (
    EvalSpec,
    FactorCompositionStep,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)


def _strategy() -> Strategy:
    return Strategy(
        config=StrategyConfig(
            strategy_id="default",
            factor_composition=(
                FactorCompositionStep(
                    factor_id="fair_value_spread",
                    role="threshold_edge",
                    param="",
                    weight=1.0,
                    threshold=0.02,
                ),
                FactorCompositionStep(
                    factor_id="yes_count",
                    role="posterior_success",
                    param="",
                    weight=1.0,
                    threshold=None,
                ),
            ),
            metadata=(("owner", "system"), ("tier", "default")),
        ),
        risk=RiskParams(
            max_position_notional_usdc=100.0,
            max_daily_drawdown_pct=2.5,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(metrics=("brier", "pnl", "fill_rate")),
        forecaster=ForecasterSpec(
            forecasters=(
                ("rules", (("threshold", "0.55"),)),
                ("stats", (("window", "15m"),)),
            )
        ),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=500.0,
        ),
    )


def test_strategy_registry_rehydrates_legacy_factor_composition_shape() -> None:
    payload = {
        "config": {
            "strategy_id": "default",
            "factor_composition": [["factor-a", 0.6], ["factor-b", 0.4]],
            "metadata": [["owner", "system"], ["tier", "default"]],
        },
        "risk": {
            "max_position_notional_usdc": 100.0,
            "max_daily_drawdown_pct": 2.5,
            "min_order_size_usdc": 1.0,
        },
        "eval_spec": {"metrics": ["brier", "pnl", "fill_rate"]},
        "forecaster": {
            "forecasters": [["rules", [["threshold", "0.55"]]], ["stats", [["window", "15m"]]]]
        },
        "market_selection": {
            "venue": "polymarket",
            "resolution_time_max_horizon_days": 7,
            "volume_min_usdc": 500.0,
        },
    }

    strategy = _strategy_from_config_json(payload)

    assert strategy.config.factor_composition == (
        FactorCompositionStep(
            factor_id="factor-a",
            role="weighted",
            param="",
            weight=0.6,
            threshold=None,
        ),
        FactorCompositionStep(
            factor_id="factor-b",
            role="weighted",
            param="",
            weight=0.4,
            threshold=None,
        ),
    )
    assert strategy.eval_spec == EvalSpec(
        metrics=("brier", "pnl", "fill_rate"),
        max_brier_score=0.30,
        slippage_threshold_bps=50.0,
        min_win_rate=0.50,
    )


def test_strategy_registry_round_trips_new_factor_composition_shape() -> None:
    serialized = _strategy_to_config_json(_strategy())
    strategy = _strategy_from_config_json(json.loads(serialized))

    assert strategy == _strategy()


def test_versioned_reader_accepts_legacy_default_v1_bootstrap_id() -> None:
    # The schema.sql bootstrap row labels the default strategy 'default-v1',
    # a load-bearing legacy sentinel that predates content-addressed version
    # ids. The reader must accept it so a runner can boot against a freshly
    # bootstrapped database when auto_migrate_default_v2 is disabled.
    strategy = _strategy()
    config_json = json.loads(_strategy_to_config_json(strategy))

    loaded = _strategy_from_versioned_config_json(
        config_json,
        strategy_id="default",
        strategy_version_id="default-v1",
    )

    assert loaded == strategy


def test_versioned_reader_accepts_matching_content_hash_id() -> None:
    strategy = _strategy()
    config_json = json.loads(_strategy_to_config_json(strategy))
    version_id = compute_strategy_version_id(*strategy.snapshot())

    loaded = _strategy_from_versioned_config_json(
        config_json,
        strategy_id="default",
        strategy_version_id=version_id,
    )

    assert loaded == strategy


def test_versioned_reader_flags_legacy_hash_definition_drift(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    strategy = _strategy()
    config_json = json.loads(_strategy_to_config_json(strategy))
    config_json.pop("factor_definition_versions", None)
    legacy_id = compute_legacy_strategy_version_id(*strategy.snapshot())
    monkeypatch.setattr(
        "pms.storage.strategy_registry.compute_strategy_version_id",
        lambda *args: "new-definition-hash",
    )

    with caplog.at_level("WARNING", logger="pms.storage.strategy_registry"):
        loaded = _strategy_from_versioned_config_json(
            config_json,
            strategy_id="default",
            strategy_version_id=legacy_id,
        )

    assert loaded == strategy
    assert "factor definition drift" in caplog.text
    assert any(
        record.__dict__.get("event") == "factor_definition_version_drift"
        for record in caplog.records
    )


def test_versioned_reader_rejects_non_legacy_id_that_mismatches_config_hash() -> None:
    # Every non-legacy version id must equal its config hash (Invariant 3:
    # immutable, content-addressed versions). A drifted id signals tampering
    # or corruption and must fail closed.
    strategy = _strategy()
    config_json = json.loads(_strategy_to_config_json(strategy))

    with pytest.raises(ValueError, match="strategy_version_id does not match"):
        _strategy_from_versioned_config_json(
            config_json,
            strategy_id="default",
            strategy_version_id="not-the-content-hash",
        )


def test_strategy_registry_rehydrates_step_factor_composition_without_threshold_field() -> None:
    payload = {
        "config": {
            "strategy_id": "default",
            "factor_composition": [
                {
                    "factor_id": "factor-a",
                    "role": "weighted",
                    "param": "",
                    "weight": 0.6,
                },
                {
                    "factor_id": "factor-b",
                    "role": "threshold_edge",
                    "param": "",
                    "weight": 1.0,
                    "threshold": 0.02,
                },
            ],
            "metadata": [["owner", "system"], ["tier", "default"]],
        },
        "risk": {
            "max_position_notional_usdc": 100.0,
            "max_daily_drawdown_pct": 2.5,
            "min_order_size_usdc": 1.0,
        },
        "eval_spec": {"metrics": ["brier", "pnl", "fill_rate"]},
        "forecaster": {
            "forecasters": [["rules", [["threshold", "0.55"]]], ["stats", [["window", "15m"]]]]
        },
        "market_selection": {
            "venue": "polymarket",
            "resolution_time_max_horizon_days": 7,
            "volume_min_usdc": 500.0,
        },
    }

    strategy = _strategy_from_config_json(payload)

    assert strategy.config.factor_composition == (
        FactorCompositionStep(
            factor_id="factor-a",
            role="weighted",
            param="",
            weight=0.6,
            threshold=None,
        ),
        FactorCompositionStep(
            factor_id="factor-b",
            role="threshold_edge",
            param="",
            weight=1.0,
            threshold=0.02,
        ),
    )
