from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import timedelta
from math import isfinite
from typing import Protocol, cast

from pms.core.models import BookSummary
from pms.core.models import EvalRecord, FillRecord, TradeDecision
from pms.evaluation.adapters.scoring import Scorer
from pms.evaluation.feedback import EvaluatorFeedback
from pms.evaluation.metrics import StrategyMetricsSnapshot, StrategyVersionKey
from pms.evaluation.quote_scoring import QuoteScorer
from pms.storage.eval_store import EvalStore
from pms.storage.quote_eval_store import QuoteEvalStore
from pms.strategies.projections import EvalSpec


logger = logging.getLogger(__name__)

StrategyMetricsProvider = Callable[
    [],
    Awaitable[
        Mapping[
            StrategyVersionKey,
            tuple[StrategyMetricsSnapshot, EvalSpec],
        ]
    ],
]


class QuoteReader(Protocol):
    async def latest_book_summary(
        self,
        market_id: str,
        token_id: str | None,
    ) -> BookSummary | None: ...


@dataclass
class EvalSpool:
    store: EvalStore
    scorer: Scorer
    feedback_generator: EvaluatorFeedback | None = None
    metrics_provider: StrategyMetricsProvider | None = None
    quote_store: QuoteEvalStore | None = None
    quote_reader: QuoteReader | None = None
    quote_scorer: QuoteScorer = field(default_factory=QuoteScorer)
    quote_lag_seconds: int = 0
    calibration_sink: Callable[[EvalRecord], None] | None = None
    _queue: asyncio.Queue[
        tuple[FillRecord, TradeDecision, Mapping[str, object] | None]
    ] = field(
        default_factory=asyncio.Queue,
    )
    _task: asyncio.Task[None] | None = None
    _fills_by_decision: dict[str, FillRecord] = field(default_factory=dict)

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    def enqueue(
        self,
        fill: FillRecord,
        decision: TradeDecision,
        *,
        decision_evidence: Mapping[str, object] | None = None,
    ) -> None:
        self._queue.put_nowait((fill, decision, decision_evidence))

    async def join(self) -> None:
        await self._queue.join()

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def process(
        self,
        fill: FillRecord,
        decision: TradeDecision,
        *,
        decision_evidence: Mapping[str, object] | None = None,
    ) -> bool:
        """Score one fill and commit its evaluation evidence synchronously.

        Resolved fills are scored and committed to the eval store before
        post-append hooks (calibration and feedback generation) fire
        best-effort. Unresolved fills route to quote evaluation instead.
        Returns ``True`` when a new eval record was inserted; duplicate
        decision IDs can be accepted durably by the store without inserting
        a second row. Scoring and append failures propagate to the caller:
        the queue worker (``_run``) and the resolution sweeper each own
        their failure policy.
        """
        if fill.resolved_outcome is None:
            await self._record_quote_eval(
                fill,
                decision,
                decision_evidence,
            )
            return False
        fill = self._aggregate_fill(fill)
        if fill.status == "partial":
            # Hold partial execution until the terminal fill arrives so the
            # durable row is written once with the decision's full quantity.
            return False
        scored_record = self.scorer.score(
            fill,
            decision,
            baseline_prob_estimates=_baseline_prob_estimates_from_evidence(
                decision_evidence,
            ),
            category=_category_from_evidence(decision_evidence),
        )
        # Persist before pushing: restart re-hydration reads the eval store,
        # so a record must never feed a calibrator unless it is also durable.
        #
        # Duplicate delivery for one decision_id (sweep retry, or an
        # ON CONFLICT-deduped store append) is safe: the sink's calibrator
        # dedups per (model_id, decision_id).
        inserted = await self.store.append(scored_record)
        if self.calibration_sink is not None:
            try:
                self.calibration_sink(scored_record)
            except Exception:  # noqa: BLE001
                logger.exception("calibration sink failed in evaluator spool")
        try:
            await self._generate_feedback()
        except Exception:  # noqa: BLE001
            logger.exception("feedback generation failed in evaluator spool")
        return inserted

    def _aggregate_fill(self, fill: FillRecord) -> FillRecord:
        """Coalesce partial fills so one decision is scored on total execution."""
        prior = self._fills_by_decision.get(fill.decision_id)
        if prior is None:
            self._fills_by_decision[fill.decision_id] = fill
            return fill
        total_notional = prior.fill_notional_usdc + fill.fill_notional_usdc
        total_quantity = prior.fill_quantity + fill.fill_quantity
        if total_quantity <= 0.0:
            return prior
        weighted_price = (
            prior.fill_price * prior.fill_quantity
            + fill.fill_price * fill.fill_quantity
        ) / total_quantity
        fees = (prior.fees or 0.0) + (fill.fees or 0.0)
        aggregated = replace(
            fill,
            fill_price=weighted_price,
            fill_notional_usdc=total_notional,
            fill_quantity=total_quantity,
            fees=fees,
            fee_bps=fill.fee_bps or prior.fee_bps,
            status=(
                fill.status
                if fill.status == "matched" or prior.status != "matched"
                else prior.status
            ),
            trade_id=f"{prior.trade_id},{fill.trade_id}",
        )
        self._fills_by_decision[fill.decision_id] = aggregated
        return aggregated

    async def _run(self) -> None:
        while True:
            fill, decision, decision_evidence = await self._queue.get()
            try:
                try:
                    await self.process(
                        fill,
                        decision,
                        decision_evidence=decision_evidence,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "fill evaluation failed in evaluator spool: %s",
                        fill.trade_id,
                    )
                    continue
            finally:
                self._queue.task_done()

    async def _generate_feedback(self) -> None:
        if self.feedback_generator is None or self.metrics_provider is None:
            return
        metrics_by_strategy = await self.metrics_provider()
        if not metrics_by_strategy:
            return
        await self.feedback_generator.generate(metrics_by_strategy)

    async def _record_quote_eval(
        self,
        fill: FillRecord,
        decision: TradeDecision,
        decision_evidence: Mapping[str, object] | None,
    ) -> None:
        if self.quote_store is None:
            logger.info(
                "skipping unresolved fill in evaluator spool: %s",
                fill.trade_id,
            )
            return
        try:
            quote = None
            quote_source = None
            if self.quote_reader is not None:
                quote = await self.quote_reader.latest_book_summary(
                    fill.market_id,
                    fill.token_id,
                )
            if quote is None:
                quote = _quote_from_decision_evidence(decision_evidence, fill)
                quote_source = _quote_source_from_decision_evidence(decision_evidence)
            if quote is None:
                logger.info(
                    "skipping unresolved fill without quote in evaluator spool: %s",
                    fill.trade_id,
                )
                return
            await self.quote_store.append(
                self.quote_scorer.score(
                    fill,
                    decision,
                    quote,
                    quote_lag_seconds=self.quote_lag_seconds,
                    quote_source=quote_source,
                )
            )
        except Exception:  # noqa: BLE001
            logger.exception("quote evaluation failed in evaluator spool")


def _baseline_prob_estimates_from_evidence(
    decision_evidence: Mapping[str, object] | None,
) -> dict[str, float]:
    if decision_evidence is None:
        return {}

    baselines: dict[str, float] = {}
    for evidence_key, baseline_source in (
        ("market_implied_baseline_prob_estimate", "market_implied"),
        ("mid_quote_baseline_prob_estimate", "mid_quote"),
        ("last_trade_baseline_prob_estimate", "last_trade"),
        ("category_prior_baseline_prob_estimate", "category_prior"),
    ):
        probability = _probability_or_none(decision_evidence.get(evidence_key))
        if probability is not None:
            baselines[baseline_source] = probability
    return baselines


def _category_from_evidence(
    decision_evidence: Mapping[str, object] | None,
) -> str | None:
    if decision_evidence is None:
        return None
    for key in ("category", "market_category"):
        value = decision_evidence.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _probability_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        probability = float(cast(int | float | str, value))
    except (TypeError, ValueError):
        return None
    if not isfinite(probability) or probability < 0.0 or probability > 1.0:
        return None
    return probability


def _quote_from_decision_evidence(
    decision_evidence: Mapping[str, object] | None,
    fill: FillRecord,
) -> BookSummary | None:
    if decision_evidence is None:
        return None
    raw_top_levels = decision_evidence.get("book_top_levels")
    if not isinstance(raw_top_levels, Mapping):
        return None
    top_levels = cast(Mapping[str, object], raw_top_levels)
    bids = _price_size_levels(top_levels.get("bids"))
    asks = _price_size_levels(top_levels.get("asks"))
    if not bids or not asks:
        return None

    best_bid = max(price for price, _size in bids)
    best_ask = min(price for price, _size in asks)
    midpoint = (best_bid + best_ask) / 2.0
    if midpoint <= 0.0:
        return None
    top_bid_depth = sum(price * size for price, size in bids if price == best_bid)
    top_ask_depth = sum(price * size for price, size in asks if price == best_ask)
    book_age_ms = _nonnegative_float(decision_evidence.get("book_age_ms"))
    book_ts = (
        fill.filled_at - timedelta(milliseconds=book_age_ms)
        if book_age_ms is not None
        else fill.filled_at
    )
    return BookSummary(
        best_bid=best_bid,
        best_ask=best_ask,
        spread_bps=((best_ask - best_bid) / midpoint) * 10_000.0,
        depth_usdc=top_bid_depth + top_ask_depth,
        timestamp=book_ts,
    )


def _price_size_levels(raw_levels: object) -> list[tuple[float, float]]:
    if not isinstance(raw_levels, list):
        return []
    parsed: list[tuple[float, float]] = []
    for raw_level in raw_levels:
        if not isinstance(raw_level, Mapping):
            return []
        level = cast(Mapping[str, object], raw_level)
        price = _positive_float(level.get("price"))
        size = _positive_float(level.get("size"))
        if price is None or size is None:
            return []
        parsed.append((price, size))
    return parsed


def _positive_float(value: object) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None or parsed <= 0.0:
        return None
    return parsed


def _nonnegative_float(value: object) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None or parsed < 0.0:
        return None
    return parsed


def _float_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(cast(int | float | str, value))
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return parsed


def _quote_source_from_decision_evidence(
    decision_evidence: Mapping[str, object] | None,
) -> str | None:
    if decision_evidence is None:
        return None
    for key in ("direct_outcome_book_source", "quote_source"):
        value = decision_evidence.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized:
            return normalized
    return "decision_evidence"
