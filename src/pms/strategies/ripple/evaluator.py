"""Deterministic evidence assessment for ripple candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from math import isfinite, sqrt
from typing import Any, cast

from pms.strategies.intents import StrategyCandidate


DEFAULT_PRIOR_STRENGTH = 2.0
DEFAULT_MIN_EXPECTED_EDGE = 0.02
NEAR_RESOLUTION_MIN_DAYS = 1.0


@dataclass(frozen=True, slots=True)
class RippleEvidenceAssessment:
    approved: bool
    posterior_probability: float
    expected_edge: float
    confidence: float
    entry_edge_threshold: float
    rationale: str
    evidence_refs: tuple[str, ...]
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RippleEvidenceEvaluator:
    min_evidence_refs: int = 2
    min_confidence: float = 0.6
    min_expected_edge: float = DEFAULT_MIN_EXPECTED_EDGE

    def assess(self, candidate: StrategyCandidate) -> RippleEvidenceAssessment:
        posterior = _posterior_from_candidate(
            candidate,
            min_expected_edge=self.min_expected_edge,
        )
        contradiction_refs = _metadata_tuple(candidate.metadata, "contradiction_refs")
        if len(candidate.evidence_refs) < self.min_evidence_refs:
            return RippleEvidenceAssessment(
                approved=False,
                posterior_probability=posterior.posterior_probability,
                expected_edge=posterior.expected_edge,
                confidence=min(posterior.confidence, self.min_confidence - 0.01),
                entry_edge_threshold=posterior.entry_edge_threshold,
                rationale="rejected: insufficient ripple evidence",
                evidence_refs=candidate.evidence_refs,
                failure_reasons=("insufficient_evidence",),
            )
        if contradiction_refs:
            return RippleEvidenceAssessment(
                approved=False,
                posterior_probability=posterior.posterior_probability,
                expected_edge=posterior.expected_edge,
                confidence=min(posterior.confidence, self.min_confidence - 0.01),
                entry_edge_threshold=posterior.entry_edge_threshold,
                rationale="rejected: ripple evidence contains a contradiction",
                evidence_refs=(*candidate.evidence_refs, *contradiction_refs),
                failure_reasons=("contradiction",),
            )
        if posterior.confidence < self.min_confidence:
            return RippleEvidenceAssessment(
                approved=False,
                posterior_probability=posterior.posterior_probability,
                expected_edge=posterior.expected_edge,
                confidence=posterior.confidence,
                entry_edge_threshold=posterior.entry_edge_threshold,
                rationale="rejected: confidence below posterior threshold",
                evidence_refs=candidate.evidence_refs,
                failure_reasons=("low_confidence",),
            )
        if posterior.expected_edge < posterior.entry_edge_threshold:
            return RippleEvidenceAssessment(
                approved=False,
                posterior_probability=posterior.posterior_probability,
                expected_edge=posterior.expected_edge,
                confidence=posterior.confidence,
                entry_edge_threshold=posterior.entry_edge_threshold,
                rationale=(
                    "rejected: posterior edge below entry threshold "
                    f"{posterior.entry_edge_threshold:.4f}"
                ),
                evidence_refs=candidate.evidence_refs,
                failure_reasons=("insufficient_expected_edge",),
            )
        return RippleEvidenceAssessment(
            approved=True,
            posterior_probability=posterior.posterior_probability,
            expected_edge=posterior.expected_edge,
            confidence=posterior.confidence,
            entry_edge_threshold=posterior.entry_edge_threshold,
            rationale=(
                "approved: beta-binomial posterior edge is sufficient "
                f"({posterior.expected_edge:.4f})"
            ),
            evidence_refs=candidate.evidence_refs,
            failure_reasons=(),
        )


@dataclass(frozen=True, slots=True)
class RipplePosterior:
    posterior_probability: float
    expected_edge: float
    confidence: float
    entry_edge_threshold: float


def beta_binomial_posterior_probability(
    *,
    prior_probability: float,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    yes_count: float = 0.0,
    no_count: float = 0.0,
) -> float:
    _require_probability(prior_probability, "prior_probability")
    _require_positive(prior_strength, "prior_strength")
    _require_non_negative(yes_count, "yes_count")
    _require_non_negative(no_count, "no_count")
    posterior_alpha = prior_probability * prior_strength + yes_count
    posterior_beta = (1.0 - prior_probability) * prior_strength + no_count
    return posterior_alpha / (posterior_alpha + posterior_beta)


def posterior_confidence(
    *,
    prior_strength: float,
    yes_count: float,
    no_count: float,
    degraded: bool = False,
) -> float:
    _require_positive(prior_strength, "prior_strength")
    _require_non_negative(yes_count, "yes_count")
    _require_non_negative(no_count, "no_count")
    evidence_mass = yes_count + no_count
    confidence = 0.5 + (0.4 * evidence_mass / (prior_strength + evidence_mass))
    if degraded:
        confidence = min(confidence, 0.55)
    return min(confidence, 0.95)


def entry_edge_threshold(
    *,
    as_of: datetime,
    resolves_at: datetime | None,
    min_expected_edge: float = DEFAULT_MIN_EXPECTED_EDGE,
) -> float:
    _require_positive(min_expected_edge, "min_expected_edge")
    if resolves_at is None:
        return min_expected_edge
    remaining_s = (resolves_at - as_of).total_seconds()
    if remaining_s <= 0.0:
        return 1.0
    remaining_days = remaining_s / 86_400.0
    if remaining_days >= NEAR_RESOLUTION_MIN_DAYS:
        return min_expected_edge
    return min_expected_edge / sqrt(max(remaining_days, 1.0 / 24.0))


def _metadata_tuple(metadata: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    value = metadata[field_name]
    if not isinstance(value, tuple):
        msg = f"{field_name} must be a tuple"
        raise TypeError(msg)
    if any(not isinstance(item, str) for item in value):
        msg = f"{field_name} must contain strings"
        raise TypeError(msg)
    return cast(tuple[str, ...], value)


def _posterior_from_candidate(
    candidate: StrategyCandidate,
    *,
    min_expected_edge: float,
) -> RipplePosterior:
    prior_probability = _optional_metadata_float(
        candidate.metadata,
        "metaculus_prior",
    )
    has_metaculus_prior = prior_probability is not None
    missing_factors = _metadata_string_tuple(candidate.metadata, "missing_factors")
    stale_factors = _metadata_string_tuple(candidate.metadata, "stale_factors")
    source = candidate.metadata.get("source")
    has_provenance = (
        source != "live_factor_service"
        or bool(candidate.metadata.get("factor_snapshot_hash"))
        or any(ref.startswith("factor_snapshot:") for ref in candidate.evidence_refs)
    )
    prior_strength = _optional_metadata_float(
        candidate.metadata,
        "prior_strength",
    )
    yes_count = _optional_metadata_float(candidate.metadata, "yes_count")
    no_count = _optional_metadata_float(candidate.metadata, "no_count")
    limit_price = _optional_metadata_float(candidate.metadata, "limit_price")
    threshold = _optional_metadata_float(candidate.metadata, "entry_edge_threshold")
    if prior_probability is None:
        prior_probability = candidate.probability_estimate
    if prior_strength is None:
        prior_strength = DEFAULT_PRIOR_STRENGTH
    if yes_count is None:
        yes_count = 0.0
    if no_count is None:
        no_count = 0.0
    if limit_price is None:
        limit_price = candidate.probability_estimate - candidate.expected_edge
    if threshold is None:
        threshold = min_expected_edge
    if yes_count == 0.0 and no_count == 0.0 and not has_metaculus_prior:
        posterior_probability = candidate.probability_estimate
    else:
        posterior_probability = beta_binomial_posterior_probability(
            prior_probability=prior_probability,
            prior_strength=prior_strength,
            yes_count=yes_count,
            no_count=no_count,
        )
    confidence = posterior_confidence(
        prior_strength=prior_strength,
        yes_count=yes_count,
        no_count=no_count,
        degraded=(
            not has_metaculus_prior
            or bool(missing_factors)
            or bool(stale_factors)
            or not has_provenance
        ),
    )
    if (
        has_metaculus_prior
        and (
            bool(missing_factors)
            or bool(stale_factors)
            or not has_provenance
        )
    ):
        # A degraded posterior is never eligible for a trade.  Keep the
        # diagnostic confidence value, but force the evaluator's existing
        # confidence gate to reject it.
        confidence = 0.0
    return RipplePosterior(
        posterior_probability=posterior_probability,
        expected_edge=posterior_probability - limit_price,
        confidence=confidence,
        entry_edge_threshold=threshold,
    )


def _metadata_string_tuple(
    metadata: Mapping[str, Any],
    field_name: str,
) -> tuple[str, ...]:
    value = metadata.get(field_name, ())
    if value is None:
        return ()
    if not isinstance(value, tuple) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a tuple of strings")
    return cast(tuple[str, ...], value)


def _optional_metadata_float(
    metadata: Mapping[str, Any],
    field_name: str,
) -> float | None:
    value = metadata.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{field_name} must be numeric"
        raise TypeError(msg)
    return float(value)


def _require_probability(value: float, field_name: str) -> None:
    if not isfinite(value) or value < 0.0 or value > 1.0:
        msg = f"{field_name} must satisfy 0.0 <= value <= 1.0"
        raise ValueError(msg)


def _require_positive(value: float, field_name: str) -> None:
    if not isfinite(value) or value <= 0.0:
        msg = f"{field_name} must be positive"
        raise ValueError(msg)


def _require_non_negative(value: float, field_name: str) -> None:
    if not isfinite(value) or value < 0.0:
        msg = f"{field_name} must be non-negative"
        raise ValueError(msg)
