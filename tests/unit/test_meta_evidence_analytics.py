from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from pms.core.enums import OrderStatus
from pms.core.models import EvalRecord, PriceChange
from pms.meta_evidence.competition import (
    compute_competition_snapshot,
    interpret_trends,
)
from pms.meta_evidence.decay import compute_decay_status
from pms.meta_evidence.models import PerformancePeak
from pms.meta_evidence.regime import classify_regime


def _eval_record(
    *,
    decision_id: str,
    recorded_at: datetime,
    pnl: float,
    edge_at_decision: float = 0.05,
    spread_bps_at_decision: int | None = 200,
    strategy_id: str = "meta-strategy",
    strategy_version_id: str = "meta-v1",
) -> EvalRecord:
    return EvalRecord(
        market_id="market-meta",
        decision_id=decision_id,
        strategy_id=strategy_id,
        strategy_version_id=strategy_version_id,
        prob_estimate=0.7,
        resolved_outcome=1.0,
        brier_score=0.09,
        fill_status=OrderStatus.MATCHED.value,
        recorded_at=recorded_at,
        citations=["unit"],
        pnl=pnl,
        slippage_bps=10.0,
        filled=True,
        edge_at_decision=edge_at_decision,
        spread_bps_at_decision=spread_bps_at_decision,
    )


def _price_change(index: int, *, price: float) -> PriceChange:
    return PriceChange(
        id=index,
        market_id="market-meta",
        token_id="token-meta",
        ts=datetime(2026, 5, 1 + index, tzinfo=UTC),
        side="BUY",
        price=price,
        size=10.0,
        best_bid=price - 0.01,
        best_ask=price + 0.01,
        hash=None,
    )


def test_regime_classifier_uses_price_changes_fallback_for_sparse_eval_records() -> None:
    result = classify_regime(
        eval_records=[],
        price_changes=[
            _price_change(0, price=0.40),
            _price_change(1, price=0.405),
            _price_change(2, price=0.41),
        ],
        volatility_threshold=0.15,
        drift_threshold=0.005,
        min_resolved_samples=5,
    )

    assert result.validation_regime == "low_vol_bull"
    assert result.regime_source == "price_changes"
    assert result.regime_sample_count == 3
    assert result.drift == pytest.approx(0.024845679012345644)


def test_decay_status_reports_insufficient_resolved_outcomes_before_sample_gate() -> None:
    now = datetime(2026, 5, 30, tzinfo=UTC)
    records = [
        _eval_record(decision_id=f"record-{index}", recorded_at=now, pnl=1.0)
        for index in range(3)
    ]

    status = compute_decay_status(
        records,
        strategy_id="meta-strategy",
        strategy_version_id="meta-v1",
        now=now,
        min_resolved_samples=10,
    )

    assert status.decay_status == "insufficient_resolved_outcomes"
    assert status.resolved_sample_count == 3
    assert status.min_resolved_samples == 10


def test_decay_status_marks_unwired_producer_when_no_evidence_exists() -> None:
    status = compute_decay_status(
        [],
        strategy_id="meta-strategy",
        strategy_version_id="meta-v1",
        now=datetime(2026, 5, 30, tzinfo=UTC),
    )

    assert status.decay_status == "producer_not_wired"


def test_decay_status_flags_degraded_when_rolling_sharpe_halves_from_peak() -> None:
    now = datetime(2026, 5, 30, tzinfo=UTC)
    records = [
        _eval_record(
            decision_id=f"record-{index}",
            recorded_at=now - timedelta(days=19 - index),
            pnl=10.0 if index < 10 else -1.0,
        )
        for index in range(20)
    ]
    peak = PerformancePeak(
        strategy_id="meta-strategy",
        strategy_version_id="meta-v1",
        peak_sharpe_7d=4.0,
        peak_sharpe_30d=4.0,
        peak_hit_rate=1.0,
        recorded_at=now - timedelta(days=1),
    )

    status = compute_decay_status(
        records,
        strategy_id="meta-strategy",
        strategy_version_id="meta-v1",
        now=now,
        min_resolved_samples=3,
        existing_peak=peak,
    )

    assert status.decay_status in {"degraded", "negative"}
    assert status.sharpe_ratio_vs_peak is not None
    assert status.sharpe_ratio_vs_peak < 0.5


def test_decay_status_reports_insufficient_when_trailing_window_is_empty() -> None:
    now = datetime(2026, 5, 30, tzinfo=UTC)
    records = [
        _eval_record(
            decision_id=f"record-{index}",
            recorded_at=now - timedelta(days=30 + index),
            pnl=1.0,
        )
        for index in range(12)
    ]
    peak = PerformancePeak(
        strategy_id="meta-strategy",
        strategy_version_id="meta-v1",
        peak_sharpe_7d=4.0,
        peak_sharpe_30d=4.0,
        peak_hit_rate=1.0,
        recorded_at=now - timedelta(days=10),
    )

    status = compute_decay_status(
        records,
        strategy_id="meta-strategy",
        strategy_version_id="meta-v1",
        now=now,
        min_resolved_samples=10,
        existing_peak=peak,
    )

    assert status.decay_status == "insufficient_resolved_outcomes"
    assert status.rolling_sharpe_7d is None
    assert status.sharpe_ratio_vs_peak is None
    assert status.trading_days_in_window == 0


def test_competition_snapshot_warmup_and_interpretation_matrix() -> None:
    snapshot_date = date(2026, 5, 30)
    records = [
        _eval_record(
            decision_id=f"record-{index}",
            recorded_at=datetime(2026, 5, 1 + index, tzinfo=UTC),
            pnl=1.0,
            edge_at_decision=0.05 - index * 0.001,
            spread_bps_at_decision=200 - index,
        )
        for index in range(10)
    ]

    snapshot = compute_competition_snapshot(
        records,
        strategy_id="meta-strategy",
        strategy_version_id="meta-v1",
        snapshot_date=snapshot_date,
    )

    assert snapshot.trend_status == "warming_up"
    assert snapshot.sample_count_30d == 10
    assert snapshot.mean_edge_30d == pytest.approx(0.0455)
    assert snapshot.mean_spread_bps_30d == pytest.approx(195.5)
    assert (
        interpret_trends(edge_trend_slope_90d=-0.01, spread_trend_slope_90d=-0.01)
        == "market_getting_efficient_edge_compressing"
    )


def test_competition_snapshot_excludes_legacy_defaulted_edges() -> None:
    snapshot_date = date(2026, 5, 30)
    records = [
        _eval_record(
            decision_id="legacy-default-edge",
            recorded_at=datetime(2026, 5, 29, tzinfo=UTC),
            pnl=1.0,
            edge_at_decision=0.0,
            spread_bps_at_decision=500,
        ),
        _eval_record(
            decision_id="captured-edge",
            recorded_at=datetime(2026, 5, 30, tzinfo=UTC),
            pnl=1.0,
            edge_at_decision=0.08,
            spread_bps_at_decision=120,
        ),
    ]

    snapshot = compute_competition_snapshot(
        records,
        strategy_id="meta-strategy",
        strategy_version_id="meta-v1",
        snapshot_date=snapshot_date,
    )

    assert snapshot.sample_count_30d == 1
    assert snapshot.mean_edge_30d == pytest.approx(0.08)
    assert snapshot.mean_spread_bps_30d == pytest.approx(120.0)


def test_regime_classifier_uses_cumulative_pnl_drift_for_eval_records() -> None:
    result = classify_regime(
        eval_records=[
            _eval_record(
                decision_id="first-loss",
                recorded_at=datetime(2026, 5, 1, tzinfo=UTC),
                pnl=-1.0,
            ),
            _eval_record(
                decision_id="second-win",
                recorded_at=datetime(2026, 5, 2, tzinfo=UTC),
                pnl=1.0,
            ),
            _eval_record(
                decision_id="third-win",
                recorded_at=datetime(2026, 5, 3, tzinfo=UTC),
                pnl=1.0,
            ),
        ],
        price_changes=[],
        volatility_threshold=2.0,
        drift_threshold=0.5,
        min_resolved_samples=3,
    )

    assert result.regime_source == "eval_records"
    assert result.drift == pytest.approx(1.0)
    assert result.validation_regime == "low_vol_bull"
