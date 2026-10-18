"""Candidate proposal for ripple fixture observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from pms.strategies.intents import (
    StrategyCandidate,
    StrategyContext,
    StrategyObservation,
)


@dataclass(frozen=True, slots=True)
class RippleController:
    async def propose(
        self,
        context: StrategyContext,
        observations: Sequence[StrategyObservation],
    ) -> Sequence[StrategyCandidate]:
        del context
        return tuple(_candidate_from_observation(observation) for observation in observations)


def _candidate_from_observation(observation: StrategyObservation) -> StrategyCandidate:
    payload = observation.payload
    payload_metadata = _mapping(payload.get("metadata", {}))
    metadata: dict[str, Any] = {
        "observation_id": observation.observation_id,
        "source": observation.source,
        "confidence": _required_float(payload, "confidence"),
        "token_id": _required_str(payload, "token_id"),
        "venue": _required_str(payload, "venue"),
        "side": _required_str(payload, "side"),
        "outcome": _required_str(payload, "outcome"),
        "limit_price": _required_float(payload, "limit_price"),
        "notional_usdc": _required_float(payload, "notional_usdc"),
        "expected_price": _required_float(payload, "expected_price"),
        "probability_estimate": _required_float(payload, "probability_estimate"),
        "max_slippage_bps": _required_int(payload, "max_slippage_bps"),
        "time_in_force": _required_str(payload, "time_in_force"),
        "contradiction_refs": _string_tuple(payload.get("contradiction_refs", ())),
        "fixture_metadata": payload_metadata,
    }
    for field_name in (
        "economics_version",
        "min_expected_edge",
        "entry_edge_threshold",
        "metaculus_prior",
        "no_count",
        "posterior_probability",
        "prior_strength",
        "yes_count",
    ):
        if field_name in payload_metadata:
            metadata[field_name] = payload_metadata[field_name]
    for field_name in ("missing_factors", "stale_factors", "factor_snapshot_hash"):
        if field_name in payload_metadata:
            metadata[field_name] = payload_metadata[field_name]
    return StrategyCandidate(
        candidate_id=f"candidate-{observation.observation_id}",
        strategy_id=observation.strategy_id,
        strategy_version_id=observation.strategy_version_id,
        market_id=_required_str(payload, "market_id"),
        title=_required_str(payload, "title"),
        thesis=_required_str(payload, "thesis"),
        probability_estimate=_required_float(payload, "probability_estimate"),
        expected_edge=_required_float(payload, "expected_edge"),
        evidence_refs=observation.evidence_refs,
        created_at=observation.observed_at,
        metadata=metadata,
    )


def _required_str(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload[field_name]
    if not isinstance(value, str):
        msg = f"{field_name} must be a string"
        raise TypeError(msg)
    return value


def _required_float(payload: Mapping[str, Any], field_name: str) -> float:
    value = payload[field_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{field_name} must be numeric"
        raise TypeError(msg)
    return float(value)


def _required_int(payload: Mapping[str, Any], field_name: str) -> int:
    value = payload[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{field_name} must be an integer"
        raise TypeError(msg)
    return cast(int, value)


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        msg = "contradiction_refs must be a tuple"
        raise TypeError(msg)
    if any(not isinstance(item, str) for item in value):
        msg = "contradiction_refs must contain strings"
        raise TypeError(msg)
    return cast(tuple[str, ...], value)


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = "metadata must be a mapping"
        raise TypeError(msg)
    return cast(Mapping[str, Any], value)
