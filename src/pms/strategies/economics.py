"""Accessors for immutable strategy economics and runtime risk projection."""

from __future__ import annotations

from math import isfinite

from pms.config import RiskSettings

from .projections import ActiveStrategy


def strategy_metadata(strategy: ActiveStrategy | None) -> dict[str, str]:
    if strategy is None:
        return {}
    return dict(strategy.config.metadata)


def strategy_economic_float(
    strategy: ActiveStrategy | None,
    key: str,
    fallback: float,
) -> float:
    raw = strategy_metadata(strategy).get(key)
    if raw is None:
        return fallback
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid strategy economic value {key!r}: {raw!r}") from exc
    if not isfinite(value):
        raise ValueError(f"invalid strategy economic value {key!r}: {raw!r}")
    return value


def strategy_economic_int(
    strategy: ActiveStrategy | None,
    key: str,
    fallback: int,
) -> int:
    value = strategy_economic_float(strategy, key, float(fallback))
    if value < 0.0 or not value.is_integer():
        raise ValueError(f"invalid strategy economic integer {key!r}: {value!r}")
    return int(value)


def strategy_economic_str(
    strategy: ActiveStrategy | None,
    key: str,
    fallback: str,
) -> str:
    raw = strategy_metadata(strategy).get(key)
    return fallback if raw is None else raw


def strategy_risk_settings(
    strategy: ActiveStrategy | None,
    fallback: RiskSettings,
) -> RiskSettings:
    if strategy is None:
        return fallback
    return RiskSettings(
        max_position_per_market=strategy.risk.max_position_notional_usdc,
        max_total_exposure=fallback.max_total_exposure,
        max_drawdown_pct=strategy.risk.max_daily_drawdown_pct,
        max_daily_loss_usdc=fallback.max_daily_loss_usdc,
        max_open_positions=fallback.max_open_positions,
        max_exposure_per_risk_group=fallback.max_exposure_per_risk_group,
        min_order_usdc=max(
            strategy.risk.min_order_size_usdc,
            fallback.min_order_usdc,
        ),
        slippage_threshold_bps=strategy_economic_float(
            strategy,
            "max_slippage_bps",
            fallback.slippage_threshold_bps,
        ),
        max_quantity_shares=fallback.max_quantity_shares,
    )
