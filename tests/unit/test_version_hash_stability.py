from __future__ import annotations

import importlib
import inspect
import os
import subprocess
from pathlib import Path
import sys
from typing import Any

import pytest


EXPECTED_DEFAULT_VERSION_ID = (
    "239f947ac755ee51b72b787d707dd207d7319b28e564047d2f3d956ea457ca29"
)


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in red phase
        pytest.fail(f"{module_name} is missing: {exc}")

    return getattr(module, symbol_name)


def _build_projection_inputs(
    *,
    metadata: tuple[tuple[str, str], ...] = (("owner", "system"), ("tier", "default")),
    max_daily_drawdown_pct: float = 2.5,
) -> dict[str, Any]:
    strategy_config = _load_symbol("pms.strategies.projections", "StrategyConfig")
    factor_composition_step = _load_symbol("pms.strategies.projections", "FactorCompositionStep")
    risk_params = _load_symbol("pms.strategies.projections", "RiskParams")
    eval_spec = _load_symbol("pms.strategies.projections", "EvalSpec")
    forecaster_spec = _load_symbol("pms.strategies.projections", "ForecasterSpec")
    market_selection_spec = _load_symbol(
        "pms.strategies.projections",
        "MarketSelectionSpec",
    )

    return {
        "config": strategy_config(
            strategy_id="default",
            factor_composition=(
                factor_composition_step(
                    factor_id="factor-a",
                    role="weighted",
                    param="",
                    weight=0.6,
                    threshold=None,
                ),
                factor_composition_step(
                    factor_id="factor-b",
                    role="weighted",
                    param="",
                    weight=0.4,
                    threshold=None,
                ),
            ),
            metadata=metadata,
        ),
        "risk": risk_params(
            max_position_notional_usdc=100.0,
            max_daily_drawdown_pct=max_daily_drawdown_pct,
            min_order_size_usdc=1.0,
        ),
        "eval_spec": eval_spec(metrics=("brier", "pnl", "fill_rate")),
        "forecaster": forecaster_spec(
            forecasters=(
                ("rules", (("threshold", "0.55"),)),
                ("stats", (("window", "15m"),)),
            )
        ),
        "market_selection": market_selection_spec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=500.0,
        ),
    }


def test_versioning_module_docstring_documents_canonicalization_contract() -> None:
    versioning = importlib.import_module("pms.strategies.versioning")
    docstring = inspect.getdoc(versioning)

    assert docstring is not None
    assert 'json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=True)' in docstring
    assert "Enum" in docstring
    assert "plain builtins" in docstring
    assert "byte-identical across Python minor-version bumps and process restarts" in docstring


def test_default_strategy_version_id_matches_locked_fixture() -> None:
    compute_strategy_version_id = _load_symbol(
        "pms.strategies.versioning",
        "compute_strategy_version_id",
    )

    version_id = compute_strategy_version_id(**_build_projection_inputs())

    assert version_id == EXPECTED_DEFAULT_VERSION_ID


def test_metadata_ordering_does_not_change_strategy_version_id() -> None:
    compute_strategy_version_id = _load_symbol(
        "pms.strategies.versioning",
        "compute_strategy_version_id",
    )

    canonical_hash = compute_strategy_version_id(
        **_build_projection_inputs(
            metadata=(("owner", "system"), ("tier", "default")),
        )
    )
    reordered_hash = compute_strategy_version_id(
        **_build_projection_inputs(
            metadata=(("tier", "default"), ("owner", "system")),
        )
    )

    assert reordered_hash == canonical_hash


def test_forecaster_ordering_does_not_change_strategy_version_id() -> None:
    compute_strategy_version_id = _load_symbol(
        "pms.strategies.versioning",
        "compute_strategy_version_id",
    )

    canonical_hash = compute_strategy_version_id(**_build_projection_inputs())
    reordered_hash = compute_strategy_version_id(
        **{
            **_build_projection_inputs(),
            "forecaster": _load_symbol("pms.strategies.projections", "ForecasterSpec")(
                forecasters=(
                    ("stats", (("window", "15m"),)),
                    ("rules", (("threshold", "0.55"),)),
                )
            ),
        }
    )

    assert reordered_hash == canonical_hash


def test_pair_element_swap_produces_distinct_strategy_version_id() -> None:
    # Pair records carry (key, value) semantics — swapping their elements
    # must produce a distinct hash, even though the outer tuple is treated
    # as an unordered collection.
    compute_strategy_version_id = _load_symbol(
        "pms.strategies.versioning",
        "compute_strategy_version_id",
    )

    baseline_hash = compute_strategy_version_id(
        **_build_projection_inputs(metadata=(("owner", "alice"),))
    )
    swapped_hash = compute_strategy_version_id(
        **_build_projection_inputs(metadata=(("alice", "owner"),))
    )

    assert swapped_hash != baseline_hash


def test_risk_param_changes_produce_a_new_strategy_version_id() -> None:
    compute_strategy_version_id = _load_symbol(
        "pms.strategies.versioning",
        "compute_strategy_version_id",
    )

    baseline_hash = compute_strategy_version_id(**_build_projection_inputs())
    changed_hash = compute_strategy_version_id(
        **_build_projection_inputs(max_daily_drawdown_pct=4.0)
    )

    assert changed_hash != baseline_hash


def test_strategy_version_id_is_stable_across_subprocesses() -> None:
    compute_strategy_version_id = _load_symbol(
        "pms.strategies.versioning",
        "compute_strategy_version_id",
    )

    expected_hash = compute_strategy_version_id(**_build_projection_inputs())
    repo_root = Path(__file__).resolve().parents[2]
    script = """
from pms.strategies.projections import (
    EvalSpec,
    FactorCompositionStep,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)
from pms.strategies.versioning import compute_strategy_version_id

print(
    compute_strategy_version_id(
        config=StrategyConfig(
            strategy_id="default",
            factor_composition=(
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
)
""".strip()

    env = os.environ.copy()
    pythonpath_entries = [str(repo_root / "src"), str(repo_root)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )

    assert result.stdout.strip() == expected_hash
