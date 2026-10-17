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

WC_FLB_TAIL_FADE_STRATEGY_ID = "wc_flb_tail_fade_v1"
_FLB_FRESHNESS_S = 300.0
_FLB_EDGE_SCALE = 2.0
_FLB_MIN_ABS_DELTA = 0.01
_ORDERBOOK_IMBALANCE_FRESHNESS_S = 60.0
_ORDERBOOK_IMBALANCE_EDGE_SCALE = 0.05
_ORDERBOOK_IMBALANCE_MIN_ABS = 0.50
_RESOLUTION_HORIZON_DAYS = 4
_ROUTER_MIN_YES_PRICE = 0.02
_ROUTER_MAX_YES_PRICE = 0.98


def build_wc_flb_tail_fade_strategy() -> Strategy:
    """Favorite-longshot-bias tail fade for World Cup 2026 markets.

    Thesis: tournament retail flow overprices longshot YES tails and
    underprices favorites. The favorite_longshot_bias factor emits signed
    values only in the tails (negative below yes_price 0.10 — fade the
    longshot via a negative delta; positive above 0.90 — back the
    favorite), so the strategy is inert across the mid-range by
    construction.
    """
    return Strategy(
        config=StrategyConfig(
            strategy_id=WC_FLB_TAIL_FADE_STRATEGY_ID,
            factor_composition=(
                FactorCompositionStep(
                    factor_id="favorite_longshot_bias",
                    role="rule_delta",
                    param="",
                    weight=_FLB_EDGE_SCALE,
                    threshold=_FLB_MIN_ABS_DELTA,
                    required=True,
                    freshness_sla_s=_FLB_FRESHNESS_S,
                    allow_neutral_fallback=False,
                ),
                FactorCompositionStep(
                    factor_id="orderbook_imbalance",
                    role="rule_delta",
                    param="",
                    weight=_ORDERBOOK_IMBALANCE_EDGE_SCALE,
                    threshold=_ORDERBOOK_IMBALANCE_MIN_ABS,
                    required=False,
                    freshness_sla_s=_ORDERBOOK_IMBALANCE_FRESHNESS_S,
                    allow_neutral_fallback=True,
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
            ),
            metadata=(
                ("owner", "Researcher-Ciga"),
                ("tier", "paper"),
                ("purpose", "wc2026_flb_tail_fade_paper_soak"),
                ("price_reference", "best_ask"),
                ("live_allowed", "false"),
            ),
        ),
        risk=RiskParams(
            max_position_notional_usdc=1.0,
            max_daily_drawdown_pct=20.0,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(
            metrics=("brier", "pnl", "fill_rate"),
            max_brier_score=0.30,
            slippage_threshold_bps=50.0,
            min_win_rate=0.45,
        ),
        forecaster=ForecasterSpec(
            forecasters=(("rules", (("threshold", "0.55"),)),)
        ),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=_RESOLUTION_HORIZON_DAYS,
            volume_min_usdc=1_000.0,
            yes_price_min=_ROUTER_MIN_YES_PRICE,
            yes_price_max=_ROUTER_MAX_YES_PRICE,
        ),
        # Wide clamp mirrors h1_flb (src/pms/strategies/flb/projection.py).
        # The whole edge lives in the tails (<0.10 / >0.90), so the default
        # [0.08, 0.92] clamp would reject every initial tail forecast before
        # the strategy has accumulated resolved calibration samples.
        calibration=CalibrationSpec(
            enabled=True,
            shrinkage_factor=1.0,
            shrinkage_bias=0.0,
            extreme_clamp_low=0.001,
            extreme_clamp_high=0.999,
            min_resolved_for_extreme=20,
        ),
    )
