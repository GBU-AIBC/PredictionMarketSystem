from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from math import sqrt

from pms.core.models import EvalRecord
from pms.meta_evidence.models import DecayStatus, DecayStatusValue, PerformancePeak


def compute_decay_status(
    records: list[EvalRecord],
    *,
    strategy_id: str,
    strategy_version_id: str,
    now: datetime,
    min_resolved_samples: int = 10,
    existing_peak: PerformancePeak | None = None,
) -> DecayStatus:
    if not records and existing_peak is None:
        return DecayStatus(
            strategy_id=strategy_id,
            strategy_version_id=strategy_version_id,
            decay_status="producer_not_wired",
            rolling_sharpe_7d=None,
            peak_sharpe_7d=None,
            sharpe_ratio_vs_peak=None,
            rolling_sharpe_30d=None,
            hit_rate_7d=None,
            peak_hit_rate=None,
            trading_days_in_window=0,
            resolved_sample_count=0,
            min_resolved_samples=min_resolved_samples,
            last_updated=now,
        )
    resolved_records = [record for record in records if record.filled]
    if len(resolved_records) < min_resolved_samples:
        return DecayStatus(
            strategy_id=strategy_id,
            strategy_version_id=strategy_version_id,
            decay_status="insufficient_resolved_outcomes",
            rolling_sharpe_7d=None,
            peak_sharpe_7d=None if existing_peak is None else existing_peak.peak_sharpe_7d,
            sharpe_ratio_vs_peak=None,
            rolling_sharpe_30d=None,
            hit_rate_7d=None,
            peak_hit_rate=None if existing_peak is None else existing_peak.peak_hit_rate,
            trading_days_in_window=0,
            resolved_sample_count=len(resolved_records),
            min_resolved_samples=min_resolved_samples,
            last_updated=now,
        )

    daily_pnl = _daily_pnl(resolved_records)
    rolling_7d = _window_values(daily_pnl, end_date=now.date(), window_days=7)
    rolling_30d = _window_values(daily_pnl, end_date=now.date(), window_days=30)
    sharpe_7d = _sharpe(rolling_7d)
    sharpe_30d = _sharpe(rolling_30d)
    hit_rate_7d = _hit_rate(resolved_records, end_date=now.date(), window_days=7)

    if existing_peak is None or existing_peak.peak_sharpe_7d <= 0.0:
        status: DecayStatusValue = "insufficient_peak_data"
        ratio = None
    elif sharpe_7d is None:
        status = "insufficient_resolved_outcomes"
        ratio = None
    else:
        ratio = sharpe_7d / existing_peak.peak_sharpe_7d
        if sharpe_7d < 0.0:
            status = "negative"
        elif ratio < 0.5:
            status = "degraded"
        else:
            status = "healthy"

    return DecayStatus(
        strategy_id=strategy_id,
        strategy_version_id=strategy_version_id,
        decay_status=status,
        rolling_sharpe_7d=sharpe_7d,
        peak_sharpe_7d=None if existing_peak is None else existing_peak.peak_sharpe_7d,
        sharpe_ratio_vs_peak=ratio,
        rolling_sharpe_30d=sharpe_30d,
        hit_rate_7d=hit_rate_7d,
        peak_hit_rate=None if existing_peak is None else existing_peak.peak_hit_rate,
        trading_days_in_window=len(rolling_7d),
        resolved_sample_count=len(resolved_records),
        min_resolved_samples=min_resolved_samples,
        last_updated=now,
    )


def _daily_pnl(records: list[EvalRecord]) -> dict[date, float]:
    grouped: dict[date, float] = defaultdict(float)
    for record in records:
        grouped[record.recorded_at.date()] += record.pnl
    return dict(grouped)


def _window_values(
    daily_pnl: dict[date, float],
    *,
    end_date: date,
    window_days: int,
) -> list[float]:
    start_date = end_date - timedelta(days=window_days - 1)
    return [
        pnl
        for day, pnl in sorted(daily_pnl.items())
        if start_date <= day <= end_date
    ]


def _sharpe(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    if variance == 0.0:
        return mean if mean >= 0.0 else -abs(mean)
    return mean / sqrt(variance)


def _hit_rate(records: list[EvalRecord], *, end_date: date, window_days: int) -> float | None:
    start_date = end_date - timedelta(days=window_days - 1)
    window_records = [
        record
        for record in records
        if start_date <= record.recorded_at.date() <= end_date
    ]
    if not window_records:
        return None
    wins = sum(1 for record in window_records if record.pnl > 0.0)
    return wins / len(window_records)
