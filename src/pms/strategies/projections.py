"""Immutable strategy projection value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import ClassVar


DEFAULT_MAX_BRIER_SCORE = 0.30
DEFAULT_SLIPPAGE_THRESHOLD_BPS = 50.0
DEFAULT_MIN_WIN_RATE = 0.50


@dataclass(frozen=True, slots=True)
class FactorCompositionStep:
    factor_id: str
    role: str
    param: str
    weight: float
    threshold: float | None
    required: bool = True
    freshness_sla_s: float | None = None
    allow_neutral_fallback: bool = False
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    strategy_id: str
    factor_composition: tuple[FactorCompositionStep, ...]
    metadata: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class RiskParams:
    max_position_notional_usdc: float
    max_daily_drawdown_pct: float
    min_order_size_usdc: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.max_position_notional_usdc)
            or self.max_position_notional_usdc < 0.0
            or not isfinite(self.max_daily_drawdown_pct)
            or self.max_daily_drawdown_pct < 0.0
            or not isfinite(self.min_order_size_usdc)
            or self.min_order_size_usdc < 0.0
        ):
            raise ValueError("RiskParams values must be finite and non-negative")


@dataclass(frozen=True)
class EvalSpec:
    __slots__ = ("__dict__",)
    metrics: tuple[str, ...]
    max_brier_score: float = DEFAULT_MAX_BRIER_SCORE
    slippage_threshold_bps: float = DEFAULT_SLIPPAGE_THRESHOLD_BPS
    min_win_rate: float = DEFAULT_MIN_WIN_RATE
    min_brier_samples: ClassVar[int] = 20
    # Feedback thresholds must not react to a single noisy fill by default.
    min_slippage_samples: ClassVar[int] = 20
    min_win_rate_samples: ClassVar[int] = 20

    def __init__(
        self,
        metrics: tuple[str, ...],
        max_brier_score: float = DEFAULT_MAX_BRIER_SCORE,
        slippage_threshold_bps: float = DEFAULT_SLIPPAGE_THRESHOLD_BPS,
        min_win_rate: float = DEFAULT_MIN_WIN_RATE,
        *,
        min_brier_samples: int = 20,
        min_slippage_samples: int = 20,
        min_win_rate_samples: int = 20,
    ) -> None:
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "max_brier_score", max_brier_score)
        object.__setattr__(self, "slippage_threshold_bps", slippage_threshold_bps)
        object.__setattr__(self, "min_win_rate", min_win_rate)
        object.__setattr__(self, "min_brier_samples", min_brier_samples)
        object.__setattr__(self, "min_slippage_samples", min_slippage_samples)
        object.__setattr__(self, "min_win_rate_samples", min_win_rate_samples)


@dataclass(frozen=True, slots=True)
class ForecasterSpec:
    forecasters: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]


@dataclass(frozen=True, slots=True)
class CalibrationSpec:
    enabled: bool = False
    shrinkage_factor: float = 0.35
    shrinkage_bias: float = 0.0
    extreme_clamp_low: float = 0.08
    extreme_clamp_high: float = 0.92
    min_resolved_for_extreme: int = 500


@dataclass(frozen=True, slots=True)
class CalibrationContext:
    resolved_sample_count: int
    model_id: str


@dataclass(frozen=True, slots=True)
class MarketSelectionSpec:
    venue: str
    resolution_time_max_horizon_days: int | None
    volume_min_usdc: float
    spread_max_bps: float | None = None
    depth_min_usdc: float | None = None
    liquidity_min_usdc: float | None = None
    accepting_orders: bool = True
    yes_price_min: float | None = None
    yes_price_max: float | None = None

    def __post_init__(self) -> None:
        _validate_optional_probability_bound(
            self.yes_price_min,
            "MarketSelectionSpec.yes_price_min",
        )
        _validate_optional_probability_bound(
            self.yes_price_max,
            "MarketSelectionSpec.yes_price_max",
        )
        if (
            self.yes_price_min is not None
            and self.yes_price_max is not None
            and self.yes_price_min > self.yes_price_max
        ):
            msg = "MarketSelectionSpec.yes_price_min must be <= yes_price_max"
            raise ValueError(msg)


def _validate_optional_probability_bound(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isfinite(value) or value < 0.0 or value > 1.0:
        msg = f"{field_name} must be finite and within [0.0, 1.0]"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StrategyRow:
    strategy_id: str
    active_version_id: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StrategyVersion:
    strategy_id: str
    strategy_version_id: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ActiveStrategy:
    strategy_id: str
    strategy_version_id: str
    config: StrategyConfig
    risk: RiskParams
    eval_spec: EvalSpec
    forecaster: ForecasterSpec
    market_selection: MarketSelectionSpec
    calibration: CalibrationSpec = field(default_factory=CalibrationSpec)

    def __post_init__(self) -> None:
        from .economics import validate_strategy_economics

        validate_strategy_economics(self.config.metadata)
        if not self.strategy_id:
            msg = "ActiveStrategy.strategy_id must be non-empty"
            raise ValueError(msg)
        if not self.strategy_version_id:
            msg = "ActiveStrategy.strategy_version_id must be non-empty"
            raise ValueError(msg)
        if self.config.strategy_id != self.strategy_id:
            msg = "ActiveStrategy.strategy_id must match config.strategy_id"
            raise ValueError(msg)
