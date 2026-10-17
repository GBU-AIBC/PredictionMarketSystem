from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from math import isfinite
from typing import cast

from pms.core.enums import OrderStatus, Side
from pms.core.models import EvalRecord, FillRecord, TradeDecision


@dataclass(frozen=True)
class Scorer:
    def score(
        self,
        fill: FillRecord,
        decision: TradeDecision,
        *,
        baseline_prob_estimates: Mapping[str, object] | None = None,
        category: str | None = None,
    ) -> EvalRecord:
        if fill.resolved_outcome is None:
            raise ValueError("FillRecord.resolved_outcome is required for scoring")
        if (
            fill.strategy_id != decision.strategy_id
            or fill.strategy_version_id != decision.strategy_version_id
        ):
            msg = (
                "FillRecord and TradeDecision strategy identity must match for scoring"
            )
            raise ValueError(msg)

        yes_prob_estimate = _yes_probability(decision)
        baseline_prob_estimate = _baseline_yes_probability(decision)
        brier_score = (yes_prob_estimate - fill.resolved_outcome) ** 2
        secondary_baseline_prob_estimates = _secondary_baseline_prob_estimates(
            primary_baseline=baseline_prob_estimate,
            raw_values=baseline_prob_estimates,
        )
        secondary_baseline_brier_scores = {
            source: (probability - fill.resolved_outcome) ** 2
            for source, probability in secondary_baseline_prob_estimates.items()
        }
        baseline_brier_score = secondary_baseline_brier_scores["market_implied"]
        model_id = _model_id(decision)
        return EvalRecord(
            market_id=fill.market_id,
            decision_id=decision.decision_id,
            strategy_id=fill.strategy_id,
            strategy_version_id=fill.strategy_version_id,
            prob_estimate=yes_prob_estimate,
            resolved_outcome=fill.resolved_outcome,
            brier_score=brier_score,
            baseline_prob_estimate=baseline_prob_estimate,
            baseline_brier_score=baseline_brier_score,
            baseline_prob_estimates=secondary_baseline_prob_estimates,
            baseline_brier_scores=secondary_baseline_brier_scores,
            fill_status=fill.status,
            recorded_at=datetime.now(tz=UTC),
            citations=[fill.trade_id],
            category=category or "unknown",
            model_id=model_id,
            pnl=_pnl(fill, decision),
            slippage_bps=_slippage_bps(fill, decision),
            filled=fill.status == OrderStatus.MATCHED.value,
            edge_at_decision=decision.expected_edge,
            spread_bps_at_decision=decision.spread_bps_at_decision,
        )


def _model_id(decision: TradeDecision) -> str:
    return "unknown" if decision.model_id is None else decision.model_id


def _yes_probability(decision: TradeDecision) -> float:
    if decision.outcome == "NO":
        return 1.0 - decision.prob_estimate
    return decision.prob_estimate


def _baseline_yes_probability(decision: TradeDecision) -> float:
    if decision.outcome == "NO":
        return 1.0 - decision.limit_price
    return decision.limit_price


def _secondary_baseline_prob_estimates(
    *,
    primary_baseline: float,
    raw_values: Mapping[str, object] | None,
) -> dict[str, float]:
    baselines = {"market_implied": primary_baseline}
    if raw_values is None:
        return baselines
    for source, raw_value in raw_values.items():
        probability = _probability_or_none(raw_value)
        if probability is None:
            continue
        baselines[str(source)] = probability
    baselines["market_implied"] = primary_baseline
    return baselines


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


def _pnl(fill: FillRecord, decision: TradeDecision) -> float:
    if fill.resolved_outcome is None:
        return 0.0
    contract_outcome = Decimal(
        str(_contract_outcome(fill.resolved_outcome, decision))
    )
    fill_price = Decimal(str(fill.fill_price))
    fill_quantity = Decimal(str(fill.fill_quantity))
    fees = Decimal("0") if fill.fees is None else Decimal(str(fill.fees))
    action = decision.action if decision.action is not None else decision.side
    if action == Side.SELL.value:
        pnl = (fill_price - contract_outcome) * fill_quantity - fees
    else:
        pnl = (contract_outcome - fill_price) * fill_quantity - fees
    return float(pnl)


def _slippage_bps(fill: FillRecord, decision: TradeDecision) -> float:
    limit_price = decision.limit_price
    if limit_price == 0.0:
        return 0.0
    action = decision.action if decision.action is not None else decision.side
    if action == Side.SELL.value:
        slippage = limit_price - fill.fill_price
    else:
        slippage = fill.fill_price - limit_price
    return max(0.0, slippage / limit_price * 10_000)


def _contract_outcome(resolved_outcome: float, decision: TradeDecision) -> float:
    if decision.outcome == "NO":
        return 1.0 - resolved_outcome
    return resolved_outcome
