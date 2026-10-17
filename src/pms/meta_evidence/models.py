from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal


ValidationRegime = Literal[
    "low_vol_bull",
    "high_vol_bear",
    "election",
    "range_bound",
    "other",
]
RegimeSource = Literal["eval_records", "price_changes", "insufficient_data"]
DecayStatusValue = Literal[
    "healthy",
    "degraded",
    "negative",
    "insufficient_resolved_outcomes",
    "insufficient_peak_data",
    "insufficient_data",
    "producer_not_wired",
]
TrendStatus = Literal["warming_up", "active"]


@dataclass(frozen=True)
class RegimeClassification:
    validation_regime: ValidationRegime
    regime_source: RegimeSource
    regime_sample_count: int
    volatility: float | None
    drift: float | None


@dataclass(frozen=True)
class PerformancePeak:
    strategy_id: str
    strategy_version_id: str
    peak_sharpe_7d: float
    peak_sharpe_30d: float
    peak_hit_rate: float
    recorded_at: datetime


@dataclass(frozen=True)
class DecayStatus:
    strategy_id: str
    strategy_version_id: str
    decay_status: DecayStatusValue
    rolling_sharpe_7d: float | None
    peak_sharpe_7d: float | None
    sharpe_ratio_vs_peak: float | None
    rolling_sharpe_30d: float | None
    hit_rate_7d: float | None
    peak_hit_rate: float | None
    trading_days_in_window: int
    resolved_sample_count: int
    min_resolved_samples: int
    last_updated: datetime


@dataclass(frozen=True)
class CompetitionSnapshot:
    snapshot_id: str
    strategy_id: str
    strategy_version_id: str
    snapshot_date: date
    mean_edge_30d: float | None
    mean_spread_bps_30d: float | None
    edge_trend_slope_90d: float | None
    spread_trend_slope_90d: float | None
    sample_count_30d: int
    trend_status: TrendStatus
    days_collected: int
    short_term_slope_30d: float | None
    short_term_slope_60d: float | None
    interpretation: str
    created_at: datetime
