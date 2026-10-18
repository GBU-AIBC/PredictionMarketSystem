from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from pms.core.enums import FeedbackSource, FeedbackTarget
from pms.core.models import Feedback
from pms.evaluation.metrics import StrategyMetricsSnapshot, StrategyVersionKey
from pms.storage.feedback_store import FeedbackStore
from pms.strategies.projections import EvalSpec


@dataclass(frozen=True)
class EvaluatorFeedback:
    store: FeedbackStore

    async def generate(
        self,
        metrics_by_strategy: Mapping[
            StrategyVersionKey,
            tuple[StrategyMetricsSnapshot, EvalSpec],
        ],
    ) -> list[Feedback]:
        existing_feedback = await self.store.all()
        existing_keys = {
            _feedback_key(item)
            for item in existing_feedback
            if not item.resolved
        }
        feedback: list[Feedback] = []

        for (strategy_id, strategy_version_id), (metrics, eval_spec) in metrics_by_strategy.items():
            for item in self._threshold_feedback(
                metrics=metrics,
                eval_spec=eval_spec,
                strategy_id=strategy_id,
                strategy_version_id=strategy_version_id,
            ):
                item_key = _feedback_key(item)
                if item_key in existing_keys:
                    continue
                feedback.append(item)
                existing_keys.add(item_key)

        for item in feedback:
            await self.store.append(
                item,
                strategy_id=str(item.metadata["strategy_id"]),
                strategy_version_id=str(item.metadata["strategy_version_id"]),
            )
        return feedback

    def _threshold_feedback(
        self,
        *,
        metrics: StrategyMetricsSnapshot,
        eval_spec: EvalSpec,
        strategy_id: str,
        strategy_version_id: str,
    ) -> list[Feedback]:
        feedback: list[Feedback] = []

        for category, brier_score in metrics.brier_by_category.items():
            if (
                metrics.brier_samples.get(category, 0) >= eval_spec.min_brier_samples
                and brier_score > eval_spec.max_brier_score
            ):
                feedback.append(
                    self._build_feedback(
                        category=f"brier:{category}",
                        message=(
                            f"Brier score {brier_score:.4f} exceeded "
                            f"{eval_spec.max_brier_score:.4f} for {category}"
                        ),
                        metadata={
                            "strategy_id": strategy_id,
                            "strategy_version_id": strategy_version_id,
                            "metric": "brier",
                            "category": category,
                            "observed_value": brier_score,
                            "threshold": eval_spec.max_brier_score,
                            "sample_size": metrics.brier_samples.get(category, 0),
                            "market_cohort": category,
                        },
                    )
                )

        if (
            metrics.record_count >= eval_spec.min_slippage_samples
            and metrics.slippage_bps > eval_spec.slippage_threshold_bps
        ):
            feedback.append(
                self._build_feedback(
                    category="slippage",
                    message=(
                        f"Slippage {metrics.slippage_bps:.2f} bps exceeded "
                        f"{eval_spec.slippage_threshold_bps:.2f} bps"
                    ),
                    metadata={
                        "strategy_id": strategy_id,
                        "strategy_version_id": strategy_version_id,
                        "metric": "slippage_bps",
                        "observed_value": metrics.slippage_bps,
                        "threshold": eval_spec.slippage_threshold_bps,
                        "sample_size": metrics.record_count,
                        "market_cohort": "all_scored_fills",
                    },
                )
            )

        if (
            metrics.record_count >= eval_spec.min_win_rate_samples
            and metrics.win_rate < eval_spec.min_win_rate
        ):
            feedback.append(
                self._build_feedback(
                    category="win_rate",
                    message=(
                        f"Win rate {metrics.win_rate:.4f} fell below "
                        f"{eval_spec.min_win_rate:.4f}"
                    ),
                    metadata={
                        "strategy_id": strategy_id,
                        "strategy_version_id": strategy_version_id,
                        "metric": "win_rate",
                        "observed_value": metrics.win_rate,
                        "threshold": eval_spec.min_win_rate,
                        "sample_size": metrics.record_count,
                        "market_cohort": "all_scored_fills",
                    },
                )
            )
        return feedback

    def _build_feedback(
        self,
        *,
        category: str,
        message: str,
        metadata: dict[str, object],
    ) -> Feedback:
        return Feedback(
            feedback_id=f"evaluator-{uuid4().hex}",
            target=FeedbackTarget.CONTROLLER.value,
            source=FeedbackSource.EVALUATOR.value,
            message=message,
            severity="warning",
            created_at=datetime.now(tz=UTC),
            category=category,
            metadata=metadata,
        )


def _feedback_key(feedback: Feedback) -> tuple[str, str, str]:
    strategy_id = str(feedback.metadata.get("strategy_id", ""))
    strategy_version_id = str(feedback.metadata.get("strategy_version_id", ""))
    return (
        strategy_id,
        strategy_version_id,
        feedback.category or "",
    )
