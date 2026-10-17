from __future__ import annotations

from pms.strategies.aggregate import Strategy
from pms.strategies.projections import (
    CalibrationSpec,
    EvalSpec,
    FactorCompositionStep,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)

PAPER_MULTI_FACTOR_STRATEGY_ID = "paper_multi_factor_v1"
_FACTOR_FRESHNESS_S = 300.0
_ORDERBOOK_IMBALANCE_MIN_ABS = 0.80
_ORDERBOOK_IMBALANCE_EDGE_SCALE = 0.25
_RESOLUTION_HORIZON_DAYS = 31
_ROUTER_MIN_YES_PRICE = 0.02
_ROUTER_MAX_YES_PRICE = 0.98
_CALIBRATION_SHRINKAGE_FACTOR = 1.0


def build_paper_multi_factor_strategy() -> Strategy:
    return Strategy(
        config=StrategyConfig(
            strategy_id=PAPER_MULTI_FACTOR_STRATEGY_ID,
            factor_composition=(
                FactorCompositionStep(
                    factor_id="orderbook_imbalance",
                    role="rule_delta",
                    param="",
                    weight=_ORDERBOOK_IMBALANCE_EDGE_SCALE,
                    threshold=_ORDERBOOK_IMBALANCE_MIN_ABS,
                    required=True,
                    freshness_sla_s=_FACTOR_FRESHNESS_S,
                    allow_neutral_fallback=False,
                ),
                FactorCompositionStep(
                    factor_id="metaculus_prior",
                    role="rule_delta",
                    param="",
                    weight=0.3,
                    threshold=None,
                    required=False,
                    freshness_sla_s=_FACTOR_FRESHNESS_S,
                    allow_neutral_fallback=True,
                    enabled=True,
                ),
                FactorCompositionStep(
                    factor_id="favorite_longshot_bias",
                    role="rule_delta",
                    param="",
                    weight=0.2,
                    threshold=None,
                    required=False,
                    freshness_sla_s=_FACTOR_FRESHNESS_S,
                    allow_neutral_fallback=True,
                    enabled=True,
                ),
                FactorCompositionStep(
                    factor_id="rules",
                    role="blend_weighted",
                    param="",
                    weight=1.0,
                    threshold=None,
                    required=False,
                    allow_neutral_fallback=True,
                ),
                FactorCompositionStep(
                    factor_id="llm",
                    role="runtime_probability",
                    param="",
                    weight=1.0,
                    threshold=None,
                    required=False,
                    allow_neutral_fallback=True,
                ),
                FactorCompositionStep(
                    factor_id="llm",
                    role="blend_weighted",
                    param="",
                    weight=1.0,
                    threshold=None,
                    required=False,
                    allow_neutral_fallback=True,
                ),
            ),
            metadata=(
                ("owner", "Researcher-Ciga"),
                ("tier", "paper"),
                ("phase", "A"),
                ("purpose", "paper_multi_factor_phase_a"),
                ("price_reference", "best_ask"),
                ("live_allowed", "false"),
            ),
        ),
        risk=RiskParams(
            max_position_notional_usdc=1.0,
            max_daily_drawdown_pct=50.0,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(
            metrics=("brier", "pnl", "fill_rate"),
            max_brier_score=0.30,
            slippage_threshold_bps=50.0,
            min_win_rate=0.45,
        ),
        forecaster=ForecasterSpec(
            forecasters=(
                ("rules", (("threshold", "0.55"),)),
                ("stats", (("window", "15m"),)),
                ("llm", ()),
            )
        ),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=_RESOLUTION_HORIZON_DAYS,
            volume_min_usdc=100.0,
            yes_price_min=_ROUTER_MIN_YES_PRICE,
            yes_price_max=_ROUTER_MAX_YES_PRICE,
        ),
        calibration=CalibrationSpec(
            enabled=True,
            shrinkage_factor=_CALIBRATION_SHRINKAGE_FACTOR,
            shrinkage_bias=0.0,
            extreme_clamp_low=0.08,
            extreme_clamp_high=0.92,
            min_resolved_for_extreme=20,
        ),
    )
