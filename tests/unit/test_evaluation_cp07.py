from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from pms.core.enums import FeedbackSource, FeedbackTarget, OrderStatus, Side, TimeInForce
from pms.core.models import (
    BookSide,
    BookSummary,
    EvalRecord,
    Feedback,
    FillRecord,
    QuoteEvalRecord,
    TradeDecision,
)
from pms.evaluation.adapters.scoring import Scorer
from pms.evaluation.feedback import EvaluatorFeedback
from pms.evaluation.metrics import (
    MetricsCollector,
    StrategyMetricsSnapshot,
)
from pms.evaluation.quote_scoring import QuoteScorer
from pms.evaluation.spool import EvalSpool
from pms.storage.eval_store import EvalStore
from pms.storage.feedback_store import FeedbackStore
from pms.storage.quote_eval_store import QuoteEvalStore
from pms.strategies.projections import EvalSpec
from tests.support.fake_stores import InMemoryEvalStore, InMemoryFeedbackStore


def _decision(
    *,
    decision_id: str = "d-cp07",
    prob: float = 0.7,
    price: float = 0.4,
    side: BookSide = Side.BUY.value,
    strategy_id: str = "default",
    strategy_version_id: str = "default-v1",
) -> TradeDecision:
    return TradeDecision(
        decision_id=decision_id,
        market_id="m-cp07",
        token_id="t-yes",
        venue="polymarket",
        side=side,
        limit_price=price,
        notional_usdc=price * 10.0,
        order_type="limit",
        max_slippage_bps=100,
        stop_conditions=["min_volume:100.00"],
        prob_estimate=prob,
        expected_edge=prob - price,
        time_in_force=TimeInForce.GTC,
        opportunity_id=f"op-{decision_id}",
        strategy_id=strategy_id,
        strategy_version_id=strategy_version_id,
        model_id="model-a",
    )


def _fill(
    *,
    decision_id: str = "d-cp07",
    resolved_outcome: float | None = 1.0,
    fill_price: float = 0.42,
    status: str = OrderStatus.MATCHED.value,
    side: str = Side.BUY.value,
    fees: float | None = None,
    strategy_id: str = "default",
    strategy_version_id: str = "default-v1",
) -> FillRecord:
    now = datetime(2026, 4, 14, tzinfo=UTC)
    return FillRecord(
        trade_id=f"trade-{decision_id}",
        order_id=f"order-{decision_id}",
        decision_id=decision_id,
        market_id="m-cp07",
        token_id="t-yes",
        venue="polymarket",
        side=side,
        fill_price=fill_price,
        fill_notional_usdc=fill_price * 10.0,
        fill_quantity=10.0,
        executed_at=now,
        filled_at=now,
        status=status,
        anomaly_flags=[],
        strategy_id=strategy_id,
        strategy_version_id=strategy_version_id,
        resolved_outcome=resolved_outcome,
        fees=fees,
    )


def _eval_record(
    *,
    decision_id: str = "d-cp07",
    strategy_id: str = "default",
    strategy_version_id: str = "default-v1",
    brier_score: float = 0.09,
    baseline_brier_score: float | None = None,
    category: str = "model-a",
    model_id: str = "model-a",
    pnl: float = 1.0,
    slippage_bps: float = 10.0,
    filled: bool = True,
) -> EvalRecord:
    return EvalRecord(
        market_id="m-cp07",
        decision_id=decision_id,
        strategy_id=strategy_id,
        strategy_version_id=strategy_version_id,
        prob_estimate=0.7,
        resolved_outcome=1.0,
        brier_score=brier_score,
        baseline_prob_estimate=(
            None if baseline_brier_score is None else 0.4
        ),
        baseline_brier_score=baseline_brier_score,
        fill_status=OrderStatus.MATCHED.value if filled else OrderStatus.INVALID.value,
        recorded_at=datetime(2026, 4, 14, tzinfo=UTC),
        citations=["unit-test"],
        category=category,
        model_id=model_id,
        pnl=pnl,
        slippage_bps=slippage_bps,
        filled=filled,
    )


class _InMemoryQuoteEvalStore:
    def __init__(self) -> None:
        self.records: list[QuoteEvalRecord] = []

    async def append(self, record: QuoteEvalRecord) -> None:
        self.records.append(record)


class _TransientFailureEvalStore(InMemoryEvalStore):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self._failures_remaining = failures

    async def append(self, record: EvalRecord) -> bool:
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            raise RuntimeError("transient eval-store failure")
        return await super().append(record)


class _StaticQuoteReader:
    def __init__(self, quote: BookSummary | None) -> None:
        self.quote = quote
        self.calls: list[tuple[str, str | None]] = []

    async def latest_book_summary(
        self,
        market_id: str,
        token_id: str | None,
    ) -> BookSummary | None:
        self.calls.append((market_id, token_id))
        return self.quote


def _strategy_snapshot(
    *,
    strategy_id: str = "default",
    strategy_version_id: str = "default-v1",
    brier_score: float = 0.31,
    sample_count: int = 20,
    slippage_bps: float = 51.0,
    win_rate: float = 0.54,
) -> StrategyMetricsSnapshot:
    return StrategyMetricsSnapshot(
        strategy_id=strategy_id,
        strategy_version_id=strategy_version_id,
        brier_overall=brier_score,
        baseline_brier_overall=None,
        brier_improvement_overall=None,
        brier_by_category={"model-a": brier_score},
        brier_samples={"model-a": sample_count},
        record_count=sample_count,
        pnl=0.0,
        slippage_bps=slippage_bps,
        fill_rate=1.0,
        win_rate=win_rate,
        calibration_samples={"model-a": sample_count},
    )


def _feedback(feedback_id: str) -> Feedback:
    return Feedback(
        feedback_id=feedback_id,
        target=FeedbackTarget.CONTROLLER.value,
        source=FeedbackSource.EVALUATOR.value,
        message="threshold crossed",
        severity="warning",
        created_at=datetime(2026, 4, 14, tzinfo=UTC),
        category="brier",
    )


def test_scorer_brier_known_values() -> None:
    scorer = Scorer()

    first = scorer.score(_fill(resolved_outcome=1.0), _decision(prob=0.7))
    second = scorer.score(_fill(resolved_outcome=0.0), _decision(prob=0.5))

    assert first.brier_score == pytest.approx(0.09)
    assert first.baseline_prob_estimate == pytest.approx(0.4)
    assert first.baseline_brier_score == pytest.approx(0.36)
    assert first.brier_improvement == pytest.approx(0.27)
    assert second.brier_score == pytest.approx(0.25)


def test_quote_scorer_uses_midpoint_for_calibration_and_bid_for_buy_mtm() -> None:
    quote = BookSummary(
        best_bid=0.62,
        best_ask=0.66,
        spread_bps=625.0,
        depth_usdc=250.0,
        timestamp=datetime(2026, 4, 14, 11, 0, tzinfo=UTC),
    )

    record = QuoteScorer().score(
        _fill(decision_id="d-quote", resolved_outcome=None, fill_price=0.50),
        _decision(decision_id="d-quote", prob=0.70, price=0.50),
        quote,
        quote_lag_seconds=3600,
        recorded_at=datetime(2026, 4, 14, 11, 0, tzinfo=UTC),
    )

    assert record.fill_id == "trade-d-quote"
    assert record.quote_lag_seconds == 3600
    assert record.prob_estimate == pytest.approx(0.70)
    assert record.quote_price == pytest.approx(0.64)
    assert record.quote_score == pytest.approx((0.70 - 0.64) ** 2)
    assert record.mtm_pnl == pytest.approx((0.62 - 0.50) * 10.0)
    assert record.category == "model-a"


def test_quote_scorer_subtracts_fill_fees_from_buy_mtm() -> None:
    quote = BookSummary(
        best_bid=0.62,
        best_ask=0.66,
        spread_bps=625.0,
        depth_usdc=250.0,
        timestamp=datetime(2026, 4, 14, 11, 0, tzinfo=UTC),
    )

    record = QuoteScorer().score(
        _fill(
            decision_id="d-quote-fee",
            resolved_outcome=None,
            fill_price=0.50,
            fees=0.15,
        ),
        _decision(decision_id="d-quote-fee", prob=0.70, price=0.50),
        quote,
        quote_lag_seconds=3600,
        recorded_at=datetime(2026, 4, 14, 11, 0, tzinfo=UTC),
    )

    assert record.mtm_pnl == pytest.approx((0.62 - 0.50) * 10.0 - 0.15)


def test_quote_scorer_subtracts_fill_fees_from_sell_mtm() -> None:
    quote = BookSummary(
        best_bid=0.62,
        best_ask=0.66,
        spread_bps=625.0,
        depth_usdc=250.0,
        timestamp=datetime(2026, 4, 14, 11, 0, tzinfo=UTC),
    )

    record = QuoteScorer().score(
        _fill(
            decision_id="d-quote-sell-fee",
            resolved_outcome=None,
            fill_price=0.70,
            side=Side.SELL.value,
            fees=0.12,
        ),
        _decision(
            decision_id="d-quote-sell-fee",
            prob=0.30,
            price=0.70,
            side=Side.SELL.value,
        ),
        quote,
        quote_lag_seconds=3600,
        recorded_at=datetime(2026, 4, 14, 11, 0, tzinfo=UTC),
    )

    assert record.mtm_pnl == pytest.approx((0.70 - 0.66) * 10.0 - 0.12)


def test_quote_scorer_uses_decimal_internals_for_boundary_mtm() -> None:
    quote = BookSummary(
        best_bid=0.30,
        best_ask=0.31,
        spread_bps=327.0,
        depth_usdc=250.0,
        timestamp=datetime(2026, 4, 14, 11, 0, tzinfo=UTC),
    )

    record = QuoteScorer().score(
        _fill(
            decision_id="d-quote-decimal",
            resolved_outcome=None,
            fill_price=0.20,
        ),
        _decision(decision_id="d-quote-decimal", prob=0.70, price=0.20),
        quote,
        quote_lag_seconds=3600,
        recorded_at=datetime(2026, 4, 14, 11, 0, tzinfo=UTC),
    )

    assert record.mtm_pnl == 1.0


def test_scorer_scores_secondary_baselines_from_decision_evidence() -> None:
    record = Scorer().score(
        _fill(resolved_outcome=1.0),
        _decision(prob=0.7, price=0.4),
        baseline_prob_estimates={
            "market_implied": 0.4,
            "mid_quote": 0.42,
            "last_trade": 0.43,
            "category_prior": 0.52,
            "invalid": 1.2,
        },
    )

    assert record.baseline_prob_estimates == {
        "market_implied": pytest.approx(0.4),
        "mid_quote": pytest.approx(0.42),
        "last_trade": pytest.approx(0.43),
        "category_prior": pytest.approx(0.52),
    }
    assert record.baseline_brier_scores == {
        "market_implied": pytest.approx(0.36),
        "mid_quote": pytest.approx(0.3364),
        "last_trade": pytest.approx(0.3249),
        "category_prior": pytest.approx(0.2304),
    }
    assert record.baseline_prob_estimate == pytest.approx(0.4)
    assert record.baseline_brier_score == pytest.approx(0.36)


@pytest.mark.asyncio
async def test_eval_spool_scores_category_prior_from_decision_evidence() -> None:
    store = cast(EvalStore, InMemoryEvalStore())
    spool = EvalSpool(store=store, scorer=Scorer())
    await spool.start()
    try:
        spool.enqueue(
            _fill(decision_id="d-category-prior", resolved_outcome=1.0),
            _decision(decision_id="d-category-prior", prob=0.7, price=0.4),
            decision_evidence={
                "category_prior_baseline_prob_estimate": 0.52,
            },
        )

        await asyncio.wait_for(spool.join(), timeout=1.0)
    finally:
        await spool.stop()

    records = await cast(InMemoryEvalStore, store).all()
    assert records[0].baseline_prob_estimates["category_prior"] == pytest.approx(0.52)
    assert records[0].baseline_brier_scores["category_prior"] == pytest.approx(0.2304)


def test_scorer_pnl_for_buy_no_uses_complementary_contract_outcome() -> None:
    scorer = Scorer()
    decision = TradeDecision(
        decision_id="d-buy-no",
        market_id="m-cp07",
        token_id="t-no",
        venue="polymarket",
        side=Side.BUY.value,
        limit_price=0.38,
        notional_usdc=3.8,
        order_type="limit",
        max_slippage_bps=100,
        stop_conditions=["min_volume:100.00"],
        prob_estimate=0.62,
        expected_edge=0.24,
        time_in_force=TimeInForce.GTC,
        opportunity_id="op-d-buy-no",
        strategy_id="default",
        strategy_version_id="default-v1",
        model_id="model-a",
        outcome="NO",
    )

    record = scorer.score(
        _fill(decision_id="d-buy-no", resolved_outcome=0.0, fill_price=0.38),
        decision,
    )

    assert record.pnl == pytest.approx(6.2)


def test_scorer_realized_pnl_is_net_of_fill_fees_for_buy() -> None:
    record = Scorer().score(
        _fill(resolved_outcome=1.0, fill_price=0.42, fees=0.15),
        _decision(prob=0.7, price=0.4),
    )

    # Realized PnL must be net of fees: (1.0 - 0.42) * 10 - 0.15, not 5.8.
    assert record.pnl == pytest.approx(5.65)


def test_scorer_realized_pnl_is_net_of_fill_fees_for_sell() -> None:
    record = Scorer().score(
        _fill(
            resolved_outcome=0.0,
            fill_price=0.42,
            side=Side.SELL.value,
            fees=0.12,
        ),
        _decision(prob=0.3, price=0.45, side=Side.SELL.value),
    )

    # Realized PnL must be net of fees: (0.42 - 0.0) * 10 - 0.12, not 4.2.
    assert record.pnl == pytest.approx(4.08)


def test_scorer_scores_fee_dominated_fill_as_loss() -> None:
    record = Scorer().score(
        _fill(resolved_outcome=1.0, fill_price=0.99, fees=0.25),
        _decision(prob=0.995, price=0.99),
    )

    # Gross edge (0.10) is smaller than fees (0.25): the trade is a loss and
    # must not count as a win in win_rate (metrics count record.pnl > 0.0).
    assert record.pnl == pytest.approx(-0.15)
    assert record.pnl < 0.0


def test_scorer_brier_for_buy_no_uses_yes_probability_basis() -> None:
    scorer = Scorer()
    decision = TradeDecision(
        decision_id="d-buy-no-brier",
        market_id="m-cp07",
        token_id="t-no",
        venue="polymarket",
        side=Side.BUY.value,
        limit_price=0.38,
        notional_usdc=3.8,
        order_type="limit",
        max_slippage_bps=100,
        stop_conditions=["min_volume:100.00"],
        prob_estimate=0.7,
        expected_edge=0.32,
        time_in_force=TimeInForce.GTC,
        opportunity_id="op-d-buy-no-brier",
        strategy_id="default",
        strategy_version_id="default-v1",
        model_id="model-a",
        outcome="NO",
    )

    record = scorer.score(
        _fill(
            decision_id="d-buy-no-brier",
            resolved_outcome=0.0,
            fill_price=0.38,
        ),
        decision,
    )

    assert record.prob_estimate == pytest.approx(0.3)
    assert record.baseline_prob_estimate == pytest.approx(0.62)
    assert record.baseline_brier_score == pytest.approx(0.3844)
    assert record.brier_improvement == pytest.approx(0.2944)
    assert record.brier_score == pytest.approx(0.09)


def test_metrics_collector_rolls_up_secondary_baseline_improvement() -> None:
    records = [
        _eval_record(
            decision_id="d-secondary-1",
            brier_score=0.09,
            baseline_brier_score=0.36,
        ),
        _eval_record(
            decision_id="d-secondary-2",
            brier_score=0.16,
            baseline_brier_score=0.25,
        ),
    ]
    records[0] = records[0].with_secondary_baselines(
        baseline_prob_estimates={"market_implied": 0.4, "mid_quote": 0.42},
        baseline_brier_scores={"market_implied": 0.36, "mid_quote": 0.3364},
    )
    records[1] = records[1].with_secondary_baselines(
        baseline_prob_estimates={"market_implied": 0.5, "mid_quote": 0.55},
        baseline_brier_scores={"market_implied": 0.25, "mid_quote": 0.2025},
    )

    snapshot = MetricsCollector(records).global_ops_snapshot()

    assert snapshot.baseline_brier_by_source == {
        "market_implied": pytest.approx(0.305),
        "mid_quote": pytest.approx(0.26945),
    }
    assert snapshot.brier_improvement_by_source == {
        "market_implied": pytest.approx(0.18),
        "mid_quote": pytest.approx(0.14445),
    }


def test_scorer_rejects_fill_and_decision_strategy_identity_mismatch() -> None:
    scorer = Scorer()

    with pytest.raises(ValueError, match="strategy identity must match"):
        scorer.score(
            _fill(strategy_id="alpha", strategy_version_id="alpha-v1"),
            _decision(strategy_id="beta", strategy_version_id="beta-v1"),
        )


@pytest.mark.asyncio
async def test_eval_spool_enqueue_is_non_blocking_and_scores_in_background(
) -> None:
    store = cast(EvalStore, InMemoryEvalStore())
    spool = EvalSpool(store=store, scorer=Scorer())
    await spool.start()
    try:
        for index in range(100):
            spool.enqueue(
                _fill(decision_id=f"d-{index}", resolved_outcome=1.0),
                _decision(prob=0.7),
            )
        assert await cast(InMemoryEvalStore, store).all() == []

        await spool.join()
    finally:
        await spool.stop()

    assert len(await cast(InMemoryEvalStore, store).all()) == 100


@pytest.mark.asyncio
async def test_eval_spool_skips_unresolved_fills_and_keeps_running(
) -> None:
    store = cast(EvalStore, InMemoryEvalStore())
    spool = EvalSpool(store=store, scorer=Scorer())
    await spool.start()
    try:
        spool.enqueue(
            _fill(decision_id="d-unresolved", resolved_outcome=None),
            _decision(decision_id="d-unresolved"),
        )
        spool.enqueue(
            _fill(decision_id="d-resolved", resolved_outcome=1.0),
            _decision(decision_id="d-resolved"),
        )

        await asyncio.wait_for(spool.join(), timeout=1.0)
    finally:
        await spool.stop()

    assert [record.decision_id for record in await cast(InMemoryEvalStore, store).all()] == ["d-resolved"]


@pytest.mark.asyncio
async def test_eval_spool_prefers_fresh_quote_reader_for_unresolved_fill_mtm(
) -> None:
    quote_store = _InMemoryQuoteEvalStore()
    fresh_reader = _StaticQuoteReader(
        BookSummary(
            best_bid=0.10,
            best_ask=0.20,
            spread_bps=6_666.0,
            depth_usdc=1.0,
            timestamp=datetime(2026, 4, 14, 10, 59, tzinfo=UTC),
        )
    )
    fill = _fill(
        decision_id="d-evidence-quote",
        resolved_outcome=None,
        fill_price=0.50,
    )
    spool = EvalSpool(
        store=cast(EvalStore, InMemoryEvalStore()),
        scorer=Scorer(),
        quote_store=cast(QuoteEvalStore, quote_store),
        quote_reader=fresh_reader,
    )
    await spool.start()
    try:
        spool.enqueue(
            fill,
            _decision(decision_id="d-evidence-quote", prob=0.70, price=0.50),
            decision_evidence={
                "direct_outcome_book_source": "venue_direct",
                "book_age_ms": 25.0,
                "book_top_levels": {
                    "bids": [{"price": 0.62, "size": 100.0}],
                    "asks": [{"price": 0.64, "size": 100.0}],
                },
            },
        )
        await asyncio.wait_for(spool.join(), timeout=1.0)
    finally:
        await spool.stop()

    assert fresh_reader.calls == [("m-cp07", "t-yes")]
    assert len(quote_store.records) == 1
    record = quote_store.records[0]
    assert record.quote_source == "postgres_snapshot"
    assert record.quote_price == pytest.approx(0.15)
    assert record.mtm_pnl == pytest.approx((0.10 - 0.50) * 10.0)
    assert record.book_ts == datetime(2026, 4, 14, 10, 59, tzinfo=UTC)


@pytest.mark.asyncio
async def test_eval_spool_falls_back_to_decision_time_book_evidence(
) -> None:
    quote_store = _InMemoryQuoteEvalStore()
    fill = _fill(
        decision_id="d-evidence-quote",
        resolved_outcome=None,
        fill_price=0.50,
    )
    spool = EvalSpool(
        store=cast(EvalStore, InMemoryEvalStore()),
        scorer=Scorer(),
        quote_store=cast(QuoteEvalStore, quote_store),
    )
    await spool.start()
    try:
        spool.enqueue(
            fill,
            _decision(decision_id="d-evidence-quote", prob=0.70, price=0.50),
            decision_evidence={
                "direct_outcome_book_source": "venue_direct",
                "book_age_ms": 25.0,
                "book_top_levels": {
                    "bids": [{"price": 0.62, "size": 100.0}],
                    "asks": [{"price": 0.64, "size": 100.0}],
                },
            },
        )
        await asyncio.wait_for(spool.join(), timeout=1.0)
    finally:
        await spool.stop()

    assert len(quote_store.records) == 1
    record = quote_store.records[0]
    assert record.quote_source == "venue_direct"
    assert record.quote_price == pytest.approx(0.63)
    assert record.mtm_pnl == pytest.approx((0.62 - 0.50) * 10.0)
    assert record.book_ts == fill.filled_at - timedelta(milliseconds=25.0)


def test_metrics_snapshot_empty_and_aggregated_records() -> None:
    empty = MetricsCollector([]).global_ops_snapshot()

    assert empty.brier_overall is None
    assert empty.baseline_brier_overall is None
    assert empty.brier_improvement_overall is None

    snapshot = MetricsCollector(
        [
            _eval_record(
                decision_id="d1",
                brier_score=0.09,
                baseline_brier_score=0.36,
                pnl=2.0,
                slippage_bps=10.0,
            ),
            _eval_record(
                decision_id="d2",
                brier_score=0.25,
                baseline_brier_score=0.16,
                category="model-b",
                model_id="model-b",
                pnl=-1.0,
                slippage_bps=20.0,
                filled=False,
            ),
        ]
    ).global_ops_snapshot()

    assert snapshot.brier_overall == pytest.approx(0.17)
    assert snapshot.baseline_brier_overall == pytest.approx(0.26)
    assert snapshot.brier_improvement_overall == pytest.approx(0.09)
    assert snapshot.brier_by_category == {"model-a": 0.09, "model-b": 0.25}
    assert snapshot.pnl == 1.0
    assert snapshot.slippage_bps == pytest.approx(15.0)
    assert snapshot.fill_rate == pytest.approx(0.5)
    assert snapshot.calibration_samples == {"model-a": 1, "model-b": 1}


@pytest.mark.asyncio
async def test_evaluator_feedback_uses_per_strategy_eval_thresholds() -> None:
    generator = EvaluatorFeedback(cast(FeedbackStore, InMemoryFeedbackStore()))
    alpha_key = ("alpha", "alpha-v1")
    beta_key = ("beta", "beta-v1")

    feedback = await generator.generate(
        {
            alpha_key: (
                _strategy_snapshot(
                    strategy_id=alpha_key[0],
                    strategy_version_id=alpha_key[1],
                ),
                EvalSpec(
                    metrics=("brier", "pnl", "fill_rate"),
                    max_brier_score=0.30,
                    slippage_threshold_bps=50.0,
                    min_win_rate=0.55,
                ),
            ),
            beta_key: (
                _strategy_snapshot(
                    strategy_id=beta_key[0],
                    strategy_version_id=beta_key[1],
                ),
                EvalSpec(
                    metrics=("brier", "pnl", "fill_rate"),
                    max_brier_score=0.40,
                    slippage_threshold_bps=60.0,
                    min_win_rate=0.50,
                ),
            ),
        }
    )

    assert {item.category for item in feedback} == {
        "brier:model-a",
        "slippage",
        "win_rate",
    }
    assert all(item.metadata["strategy_id"] == "alpha" for item in feedback)
    assert all(item.metadata["strategy_version_id"] == "alpha-v1" for item in feedback)


@pytest.mark.asyncio
async def test_eval_spool_generates_deduped_feedback_from_runtime_metrics() -> None:
    store = cast(EvalStore, InMemoryEvalStore())
    feedback_store = cast(FeedbackStore, InMemoryFeedbackStore())

    async def metrics_provider() -> dict[tuple[str, str], tuple[StrategyMetricsSnapshot, EvalSpec]]:
        snapshots = MetricsCollector(
            await cast(InMemoryEvalStore, store).all()
        ).snapshot_by_strategy()
        return {
            key: (
                snapshot,
                EvalSpec(
                    metrics=("brier", "pnl", "fill_rate"),
                    max_brier_score=0.30,
                    slippage_threshold_bps=50.0,
                    min_win_rate=0.50,
                        min_slippage_samples=1,
                        min_win_rate_samples=1,
                ),
            )
            for key, snapshot in snapshots.items()
        }

    spool = EvalSpool(
        store=store,
        scorer=Scorer(),
        feedback_generator=EvaluatorFeedback(feedback_store),
        metrics_provider=metrics_provider,
    )
    await spool.start()
    try:
        spool.enqueue(
            _fill(decision_id="d-feedback-1", resolved_outcome=1.0, fill_price=0.42),
            _decision(decision_id="d-feedback-1", price=0.4),
        )
        spool.enqueue(
            _fill(decision_id="d-feedback-2", resolved_outcome=1.0, fill_price=0.42),
            _decision(decision_id="d-feedback-2", price=0.4),
        )

        await asyncio.wait_for(spool.join(), timeout=1.0)
    finally:
        await spool.stop()

    feedback = await cast(InMemoryFeedbackStore, feedback_store).all()

    assert [item.category for item in feedback] == ["slippage"]
    assert feedback[0].metadata["strategy_id"] == "default"
    assert feedback[0].metadata["strategy_version_id"] == "default-v1"
    assert feedback[0].metadata["sample_size"] == 1
    assert feedback[0].metadata["market_cohort"] == "all_scored_fills"


@pytest.mark.asyncio
async def test_eval_spool_logs_feedback_errors_and_keeps_scoring(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = cast(EvalStore, InMemoryEvalStore())

    async def _failing_metrics_provider() -> dict[tuple[str, str], tuple[StrategyMetricsSnapshot, EvalSpec]]:
        raise RuntimeError("metrics boom")

    spool = EvalSpool(
        store=store,
        scorer=Scorer(),
        feedback_generator=EvaluatorFeedback(cast(FeedbackStore, InMemoryFeedbackStore())),
        metrics_provider=_failing_metrics_provider,
    )
    await spool.start()
    try:
        caplog.set_level("ERROR", logger="pms.evaluation.spool")
        spool.enqueue(
            _fill(decision_id="d-feedback-error", resolved_outcome=1.0),
            _decision(decision_id="d-feedback-error"),
        )
        spool.enqueue(
            _fill(decision_id="d-after-error", resolved_outcome=1.0),
            _decision(decision_id="d-after-error"),
        )
        await asyncio.wait_for(spool.join(), timeout=1.0)
    finally:
        await spool.stop()

    records = await cast(InMemoryEvalStore, store).all()
    assert [record.decision_id for record in records] == [
        "d-feedback-error",
        "d-after-error",
    ]
    assert "feedback generation failed in evaluator spool" in caplog.text


@pytest.mark.asyncio
async def test_eval_spool_survives_transient_store_failure_and_keeps_scoring(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _TransientFailureEvalStore(failures=1)
    spool = EvalSpool(store=cast(EvalStore, store), scorer=Scorer())
    await spool.start()
    try:
        caplog.set_level("ERROR", logger="pms.evaluation.spool")
        spool.enqueue(
            _fill(decision_id="d-store-error", resolved_outcome=1.0),
            _decision(decision_id="d-store-error"),
        )
        spool.enqueue(
            _fill(decision_id="d-after-store-error", resolved_outcome=1.0),
            _decision(decision_id="d-after-store-error"),
        )
        await asyncio.wait_for(spool.join(), timeout=1.0)
    finally:
        await spool.stop()

    records = await store.all()
    assert [record.decision_id for record in records] == ["d-after-store-error"]
    assert "fill evaluation failed in evaluator spool" in caplog.text


@pytest.mark.asyncio
async def test_eval_spool_survives_scoring_error_and_keeps_scoring(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = cast(EvalStore, InMemoryEvalStore())
    spool = EvalSpool(store=store, scorer=Scorer())
    await spool.start()
    try:
        caplog.set_level("ERROR", logger="pms.evaluation.spool")
        spool.enqueue(
            _fill(
                decision_id="d-poison",
                resolved_outcome=1.0,
                strategy_id="alpha",
                strategy_version_id="alpha-v1",
            ),
            _decision(
                decision_id="d-poison",
                strategy_id="beta",
                strategy_version_id="beta-v1",
            ),
        )
        spool.enqueue(
            _fill(decision_id="d-after-poison", resolved_outcome=1.0),
            _decision(decision_id="d-after-poison"),
        )
        await asyncio.wait_for(spool.join(), timeout=1.0)
    finally:
        await spool.stop()

    records = await cast(InMemoryEvalStore, store).all()
    assert [record.decision_id for record in records] == ["d-after-poison"]
    assert "fill evaluation failed in evaluator spool" in caplog.text


def test_metrics_snapshot_by_strategy_returns_only_present_keys() -> None:
    snapshots = MetricsCollector(
        [
            _eval_record(
                decision_id="alpha-1",
                strategy_id="alpha",
                strategy_version_id="alpha-v1",
                brier_score=0.09,
            ),
            _eval_record(
                decision_id="beta-1",
                strategy_id="beta",
                strategy_version_id="beta-v1",
                brier_score=0.25,
                category="model-b",
                model_id="model-b",
            ),
        ]
    ).snapshot_by_strategy()

    assert set(snapshots) == {("alpha", "alpha-v1"), ("beta", "beta-v1")}
    assert snapshots[("alpha", "alpha-v1")].brier_overall == pytest.approx(0.09)
    assert snapshots[("beta", "beta-v1")].brier_by_category == {"model-b": 0.25}
