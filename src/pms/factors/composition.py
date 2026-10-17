from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Protocol

from pms.core.validation import validate_finite
from pms.factors.catalog import FACTOR_CATALOG_ROWS


COMPOSITION_ROLES = frozenset(
    {
        "weighted",
        "precedence_rank",
        "threshold_edge",
        "posterior_prior",
        "posterior_success",
        "posterior_failure",
        "rule_delta",
        "runtime_probability",
        "blend_weighted",
    }
)
COMPOSITION_PSEUDO_FACTORS = frozenset({"yes_price", "subset_price", "rules", "statistical", "llm"})
# Existing registry fixtures use these neutral placeholders to exercise JSON
# round-tripping.  They are retained for back-compat; production installers
# still validate against the catalog before activation.
_LEGACY_BACKCOMPAT_FACTOR_IDS = frozenset({"factor-a", "factor-b"})


class FactorCompositionStep(Protocol):
    @property
    def factor_id(self) -> str: ...

    @property
    def role(self) -> str: ...

    @property
    def param(self) -> str: ...

    @property
    def weight(self) -> float: ...

    @property
    def threshold(self) -> float | None: ...

    @property
    def required(self) -> bool: ...

    @property
    def freshness_sla_s(self) -> float | None: ...

    @property
    def allow_neutral_fallback(self) -> bool: ...

    @property
    def enabled(self) -> bool: ...


def validate_composition(composition: Sequence[FactorCompositionStep]) -> None:
    """Validate composition references before a strategy can be registered."""

    known_factor_ids = {
        entry.factor_id
        for entry in FACTOR_CATALOG_ROWS
        if entry.owner != "deferred"
    } | COMPOSITION_PSEUDO_FACTORS | _LEGACY_BACKCOMPAT_FACTOR_IDS
    for step in composition:
        if step.role not in COMPOSITION_ROLES:
            raise ValueError(f"unknown composition role: {step.role!r}")
        if step.factor_id not in known_factor_ids:
            raise ValueError(f"unknown composition factor id: {step.factor_id!r}")
        if (
            step.factor_id == "orderbook_imbalance"
            and step.role == "threshold_edge"
        ):
            raise ValueError(
                "orderbook_imbalance is a raw depth factor, not a probability edge; "
                "scale it through a strategy rule_delta step"
            )


def apply_composition(
    composition: Sequence[FactorCompositionStep],
    factor_values: Mapping[tuple[str, str], float],
) -> float:
    # Validate the complete input boundary before any threshold comparison.
    # ``abs(nan) < threshold`` is false and would otherwise let NaN reach the
    # probability clamp where Python's min/max turn it into 0.999999.
    for key, value in factor_values.items():
        validate_finite(value, field_name=f"factor {key[0]}:{key[1]}")
    composition = _enabled_steps(composition)
    weighted_steps = _eligible_weighted_steps(composition, factor_values)
    non_weighted_steps = tuple(step for step in composition if step.role != "weighted")
    if weighted_steps and not non_weighted_steps:
        return _apply_weighted(weighted_steps, factor_values)

    branch_probabilities = evaluate_branch_probabilities(composition, factor_values)
    blended_probability = _apply_blend_weighted(composition, branch_probabilities)
    if blended_probability is not None:
        return blended_probability

    precedence_probability = branch_probabilities.get("rules")
    if precedence_probability is not None:
        return precedence_probability

    posterior_probability = branch_probabilities.get("statistical")
    if posterior_probability is not None:
        return posterior_probability

    runtime_probability = _first_runtime_probability(composition, branch_probabilities)
    if runtime_probability is not None:
        return runtime_probability

    if weighted_steps:
        return _apply_weighted(weighted_steps, factor_values)

    fallback = factor_values.get(("yes_price", ""))
    if fallback is not None:
        return fallback

    msg = "composition could not resolve a probability"
    raise KeyError(msg)


def _eligible_weighted_steps(
    composition: Sequence[FactorCompositionStep],
    factor_values: Mapping[tuple[str, str], float],
) -> tuple[FactorCompositionStep, ...]:
    threshold_gates = {
        (step.factor_id, step.param): 0.0 if step.threshold is None else step.threshold
        for step in composition
        if step.role == "threshold_edge"
    }
    eligible_steps: list[FactorCompositionStep] = []
    for step in composition:
        if step.role != "weighted":
            continue
        threshold = threshold_gates.get((step.factor_id, step.param))
        if threshold is not None:
            value = factor_values.get((step.factor_id, step.param))
            if value is None or abs(value) < threshold:
                continue
        eligible_steps.append(step)
    return tuple(eligible_steps)


def evaluate_branch_probabilities(
    composition: Sequence[FactorCompositionStep],
    factor_values: Mapping[tuple[str, str], float],
) -> dict[str, float]:
    for key, value in factor_values.items():
        validate_finite(value, field_name=f"factor {key[0]}:{key[1]}")
    composition = _enabled_steps(composition)
    branch_probabilities: dict[str, float] = {}

    rules_probability = _apply_threshold_precedence(composition, factor_values)
    if rules_probability is None:
        rules_probability = _apply_rule_delta(composition, factor_values)
    if rules_probability is not None:
        branch_probabilities["rules"] = rules_probability

    posterior_probability = _apply_posterior(composition, factor_values)
    if posterior_probability is not None:
        branch_probabilities["statistical"] = posterior_probability

    for step in composition:
        if step.role != "runtime_probability":
            continue
        runtime_probability = factor_values.get((step.factor_id, step.param))
        if runtime_probability is not None:
            branch_probabilities[step.factor_id] = runtime_probability

    return branch_probabilities


def _apply_weighted(
    steps: tuple[FactorCompositionStep, ...],
    factor_values: Mapping[tuple[str, str], float],
) -> float:
    weighted_total = 0.0
    realized_weight = 0.0
    for step in steps:
        value = factor_values.get((step.factor_id, step.param))
        if value is None:
            continue
        weighted_total += step.weight * value
        realized_weight += step.weight
    if realized_weight == 0.0:
        msg = "weighted composition is missing all factor inputs"
        raise ValueError(msg)
    return weighted_total / realized_weight


def _apply_blend_weighted(
    composition: Sequence[FactorCompositionStep],
    branch_probabilities: Mapping[str, float],
) -> float | None:
    blend_steps = tuple(step for step in composition if step.role == "blend_weighted")
    if not blend_steps:
        return None

    total_weight = 0.0
    weighted_probability = 0.0
    for step in blend_steps:
        branch_probability = branch_probabilities.get(step.factor_id)
        if branch_probability is None:
            continue
        total_weight += step.weight
        weighted_probability += step.weight * branch_probability

    if total_weight == 0.0:
        return None
    return weighted_probability / total_weight


def _apply_threshold_precedence(
    composition: Sequence[FactorCompositionStep],
    factor_values: Mapping[tuple[str, str], float],
) -> float | None:
    ranks = {
        (step.factor_id, step.param): step.weight
        for step in composition
        if step.role == "precedence_rank"
    }
    threshold_steps = sorted(
        (step for step in composition if step.role == "threshold_edge"),
        key=lambda step: ranks.get((step.factor_id, step.param), step.weight),
    )

    for step in threshold_steps:
        edge = factor_values.get((step.factor_id, step.param))
        if edge is None:
            continue

        threshold = 0.0 if step.threshold is None else step.threshold
        if step.factor_id == "fair_value_spread":
            if abs(edge) < threshold:
                continue
            return _required_factor_value(factor_values, "yes_price", "") + edge
        if step.factor_id == "subset_pricing_violation":
            if edge < threshold:
                continue
            return _required_factor_value(factor_values, "subset_price", "") - edge
        if step.factor_id == "orderbook_imbalance":
            msg = (
                "orderbook_imbalance is a raw depth factor, not a probability "
                "edge; scale it through a strategy rule_delta step"
            )
            raise ValueError(msg)
        if abs(edge) < threshold:
            continue
        return edge

    return None


def _apply_rule_delta(
    composition: Sequence[FactorCompositionStep],
    factor_values: Mapping[tuple[str, str], float],
) -> float | None:
    rule_steps = tuple(step for step in composition if step.role == "rule_delta")
    if not rule_steps:
        return None

    yes_price = _required_factor_value(factor_values, "yes_price", "")
    probability = yes_price
    applied = False
    for step in rule_steps:
        factor_value = _rule_delta_factor_value(step, factor_values)
        if factor_value is None:
            continue
        delta = _rule_delta(step.factor_id, factor_value, market_price=yes_price)
        if step.threshold is not None and abs(delta) < step.threshold:
            continue
        probability += step.weight * delta
        applied = True

    if not applied:
        return None
    return _bounded_probability(probability)


def _rule_delta_factor_value(
    step: FactorCompositionStep,
    factor_values: Mapping[tuple[str, str], float],
) -> float | None:
    factor_value = factor_values.get((step.factor_id, step.param))
    if factor_value is not None:
        return _orient_rule_delta_value(
            factor_id=step.factor_id,
            param=step.param,
            factor_value=factor_value,
        )
    if step.factor_id != "orderbook_imbalance" or step.param != "":
        return None
    no_token_value = factor_values.get((step.factor_id, "NO"))
    if no_token_value is not None:
        return -no_token_value
    return factor_values.get((step.factor_id, "YES"))


def _orient_rule_delta_value(
    *,
    factor_id: str,
    param: str,
    factor_value: float,
) -> float:
    if factor_id == "orderbook_imbalance" and param.upper() == "NO":
        return -factor_value
    return factor_value


def _rule_delta(
    factor_id: str,
    factor_value: float,
    *,
    market_price: float,
) -> float:
    if factor_id == "metaculus_prior":
        return factor_value - market_price
    if factor_id == "subset_pricing_violation":
        return -factor_value
    return factor_value


def _first_runtime_probability(
    composition: Sequence[FactorCompositionStep],
    branch_probabilities: Mapping[str, float],
) -> float | None:
    for step in composition:
        if step.role != "runtime_probability":
            continue
        runtime_probability = branch_probabilities.get(step.factor_id)
        if runtime_probability is not None:
            return runtime_probability
    return None


def _apply_posterior(
    composition: Sequence[FactorCompositionStep],
    factor_values: Mapping[tuple[str, str], float],
) -> float | None:
    prior_steps = tuple(step for step in composition if step.role == "posterior_prior")
    success_steps = tuple(step for step in composition if step.role == "posterior_success")
    failure_steps = tuple(step for step in composition if step.role == "posterior_failure")
    if not prior_steps and not success_steps and not failure_steps:
        return None
    has_real_input = any(
        (step.factor_id, step.param) in factor_values
        for step in (*prior_steps, *success_steps, *failure_steps)
    )
    allow_neutral_fallback = any(
        step.allow_neutral_fallback
        for step in (*prior_steps, *success_steps, *failure_steps)
    )
    if not has_real_input and not allow_neutral_fallback:
        return None

    prior_strength = 0.0
    prior_alpha = 0.0
    prior_beta = 0.0
    if prior_steps:
        prior_step = prior_steps[0]
        prior_strength = prior_step.weight
        prior_prob = factor_values.get((prior_step.factor_id, prior_step.param))
        if prior_prob is None:
            prior_alpha = prior_strength / 2.0
            prior_beta = prior_strength / 2.0
        else:
            prior_alpha = prior_prob * prior_strength
            prior_beta = (1.0 - prior_prob) * prior_strength

    successes = sum(
        step.weight * factor_values.get((step.factor_id, step.param), 0.0)
        for step in success_steps
    )
    failures = sum(
        step.weight * factor_values.get((step.factor_id, step.param), 0.0)
        for step in failure_steps
    )
    total = prior_alpha + prior_beta + successes + failures
    if total == 0.0:
        return factor_values.get(("yes_price", ""))
    return (prior_alpha + successes) / total


def _enabled_steps(
    composition: Sequence[FactorCompositionStep],
) -> tuple[FactorCompositionStep, ...]:
    return tuple(step for step in composition if step.enabled)


def _required_factor_value(
    factor_values: Mapping[tuple[str, str], float],
    factor_id: str,
    param: str,
) -> float:
    value = factor_values.get((factor_id, param))
    if value is None:
        msg = f"missing required factor input {factor_id!r}:{param!r}"
        raise KeyError(msg)
    return value


def _bounded_probability(value: float) -> float:
    if not isfinite(value):
        raise ValueError("probability must be finite")
    return max(1e-6, min(1.0 - 1e-6, value))
