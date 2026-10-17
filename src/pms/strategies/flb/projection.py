from __future__ import annotations

from pms.strategies.aggregate import Strategy
from pms.strategies.projections import (
    CalibrationSpec,
    EvalSpec,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)

H1_FLB_STRATEGY_ID = "h1_flb"
H1_FLB_ECONOMICS_VERSION = "h1-flb-economics-v1"
H1_FLB_ENTRY_EXECUTION_COST_BPS = 15.0
H1_FLB_FEE_RATE = 0.07
H1_FLB_MAX_SLIPPAGE_BPS = 50
H1_FLB_MIN_EXPECTED_EDGE = 0.02
H1_FLB_LONGSHOT_THRESHOLD = 0.10
H1_FLB_FAVORITE_THRESHOLD = 0.90


def build_h1_flb_strategy(
    *,
    calibration_csv_sha256: str | None = None,
    calibration_source_label: str | None = None,
    time_in_force: str = "GTC",
) -> Strategy:
    metadata: tuple[tuple[str, str], ...] = (
        ("owner", "system"),
        ("tier", "live_candidate"),
        ("phase", "H1"),
        ("purpose", "calibrated_h1_flb_paper_soak"),
        ("price_reference", "best_ask"),
        ("live_allowed", "true"),
        ("alpha_source", "warehouse_flb_decile_model_v1"),
        ("edge_model_source", "flb_calibration_model_v1"),
        ("calibration_source", "warehouse_flb_v1"),
        ("evidence_source", "paper_soak_go_report_v1"),
        # Economics and gate thresholds are part of the content-addressed
        # strategy projection.  Runtime wiring must use these values for this
        # strategy version rather than mutable global settings.
        ("economics_version", H1_FLB_ECONOMICS_VERSION),
        ("entry_execution_cost_bps", str(H1_FLB_ENTRY_EXECUTION_COST_BPS)),
        ("fee_rate", str(H1_FLB_FEE_RATE)),
        ("max_slippage_bps", str(H1_FLB_MAX_SLIPPAGE_BPS)),
        ("time_in_force", time_in_force.upper()),
        ("min_expected_edge", str(H1_FLB_MIN_EXPECTED_EDGE)),
        ("longshot_yes_threshold", str(H1_FLB_LONGSHOT_THRESHOLD)),
        ("favorite_yes_threshold", str(H1_FLB_FAVORITE_THRESHOLD)),
    )
    if calibration_csv_sha256 is not None:
        metadata = (
            *metadata,
            ("flb_calibration_csv_sha256", calibration_csv_sha256),
        )
    if calibration_source_label is not None:
        metadata = (
            *metadata,
            ("flb_calibration_source_label", calibration_source_label),
        )
    return Strategy(
        config=StrategyConfig(
            strategy_id=H1_FLB_STRATEGY_ID,
            factor_composition=(),
            metadata=metadata,
        ),
        risk=RiskParams(
            max_position_notional_usdc=1.0,
            max_daily_drawdown_pct=20.0,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(
            metrics=("brier", "pnl", "fill_rate"),
            max_brier_score=0.20,
            slippage_threshold_bps=50.0,
            min_win_rate=0.45,
        ),
        forecaster=ForecasterSpec(forecasters=(("flb", ()),)),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=500.0,
            yes_price_min=0.01,
            yes_price_max=0.99,
        ),
        calibration=CalibrationSpec(
            enabled=True,
            shrinkage_factor=1.0,
            shrinkage_bias=0.0,
            extreme_clamp_low=0.001,
            extreme_clamp_high=0.999,
            min_resolved_for_extreme=100,
        ),
    )
