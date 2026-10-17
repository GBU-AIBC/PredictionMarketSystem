"""Deterministic strategy-version hashing.

Canonicalization first normalizes frozen projection dataclasses,
``Enum`` values, and nested containers into plain builtins. The
normalized payload is then serialized with
``json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=True)``.

The resulting SHA-256 hex digest must remain byte-identical across Python minor-version bumps and process restarts. Any accidental drift is an Invariant 3 violation.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Any

from pms.factors.definitions import REGISTERED

from .projections import (
    CalibrationSpec,
    EvalSpec,
    FactorCompositionStep,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)


def _sorted_dict_factory(items: list[tuple[str, Any]]) -> dict[str, Any]:
    return {key: value for key, value in sorted(items)}


def _json_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _payload_value(value: Any) -> Any:
    # Stored config_json must preserve sequence order so the row can round-trip
    # back into the original projection tuple.
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, FactorCompositionStep):
        payload = asdict(value, dict_factory=_sorted_dict_factory)
        if payload.get("enabled") is True:
            del payload["enabled"]
        return _payload_value(payload)
    if isinstance(value, MarketSelectionSpec):
        payload = asdict(value, dict_factory=_sorted_dict_factory)
        if payload.get("yes_price_min") is None:
            del payload["yes_price_min"]
        if payload.get("yes_price_max") is None:
            del payload["yes_price_max"]
        return _payload_value(payload)
    if is_dataclass(value) and not isinstance(value, type):
        return _payload_value(asdict(value, dict_factory=_sorted_dict_factory))
    if isinstance(value, dict):
        payload = {
            key: _payload_value(value[key])
            for key in sorted(value)
        }
        if _is_factor_composition_step_record(payload) and payload.get("enabled") is True:
            del payload["enabled"]
        return payload
    if isinstance(value, tuple):
        return [_payload_value(item) for item in value]
    if isinstance(value, list):
        return [_payload_value(item) for item in value]
    if isinstance(value, frozenset):
        normalized_items = [_payload_value(item) for item in value]
        return sorted(normalized_items, key=_json_sort_key)
    return value


def _is_pair_record(item: Any) -> bool:
    # Pair records — 2-element (tuple|list) — model ordered (key, value)
    # structure throughout the projection layer (metadata entries,
    # factor composition weights, forecaster specs). Their element order
    # carries meaning and must not be sorted, or swapping key/value
    # produces a hash collision (Invariant 3 violation).
    return isinstance(item, (tuple, list)) and len(item) == 2


def _is_factor_composition_step_record(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    required_keys = {"factor_id", "role", "param", "weight", "threshold"}
    return required_keys.issubset(item)


def _normalize_value(value: Any) -> Any:
    # Version ids intentionally normalize sequence ordering so semantically
    # equivalent projection payloads hash identically across processes.
    if isinstance(value, dict):
        return {
            key: _normalize_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        if value and all(_is_factor_composition_step_record(item) for item in value):
            return [_normalize_value(item) for item in value]
        if value and all(_is_pair_record(item) for item in value):
            normalized_pairs = [
                [_normalize_value(item[0]), _normalize_value(item[1])]
                for item in value
            ]
            return sorted(normalized_pairs, key=_json_sort_key)
        normalized_items = [_normalize_value(item) for item in value]
        return sorted(normalized_items, key=_json_sort_key)
    if isinstance(value, frozenset):
        normalized_items = [_normalize_value(item) for item in value]
        return sorted(normalized_items, key=_json_sort_key)
    return value


def _strategy_payload(
    config: StrategyConfig,
    risk: RiskParams,
    eval_spec: EvalSpec,
    forecaster: ForecasterSpec,
    market_selection: MarketSelectionSpec,
    calibration: CalibrationSpec | None = None,
    include_factor_definition_versions: bool = True,
) -> dict[str, Any]:
    payload = {
        "config": _payload_value(config),
        "risk": _payload_value(risk),
        "eval_spec": _payload_value(eval_spec),
        "forecaster": _payload_value(forecaster),
        "market_selection": _payload_value(market_selection),
    }
    # Preserve the pre-W1B wire/hash contract for the omitted default. Existing
    # strategy rows intentionally omit calibration, and changing that hash
    # would make every installed default version fail rehydration. Explicit
    # non-default calibration remains part of the immutable version payload.
    if calibration is not None and calibration != CalibrationSpec():
        payload["calibration"] = _payload_value(calibration)
    if include_factor_definition_versions:
        factor_versions = _factor_definition_versions(config)
        if factor_versions:
            payload["factor_definition_versions"] = factor_versions
    return payload


def _factor_definition_versions(config: StrategyConfig) -> dict[str, str]:
    versions_by_id = {
        factor_cls.factor_id: factor_cls.definition_version
        for factor_cls in REGISTERED
    }
    versions = {
        step.factor_id: versions_by_id[step.factor_id]
        for step in config.factor_composition
        if step.factor_id in versions_by_id
        and versions_by_id[step.factor_id] != "1"
    }
    return dict(sorted(versions.items()))


def serialize_strategy_config_json(
    config: StrategyConfig,
    risk: RiskParams,
    eval_spec: EvalSpec,
    forecaster: ForecasterSpec,
    market_selection: MarketSelectionSpec,
    calibration: CalibrationSpec | None = None,
) -> str:
    return json.dumps(
        _strategy_payload(
            config=config,
            risk=risk,
            eval_spec=eval_spec,
            forecaster=forecaster,
            market_selection=market_selection,
            calibration=calibration,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def compute_strategy_version_id(
    config: StrategyConfig,
    risk: RiskParams,
    eval_spec: EvalSpec,
    forecaster: ForecasterSpec,
    market_selection: MarketSelectionSpec,
    calibration: CalibrationSpec | None = None,
) -> str:
    canonical_payload = _normalize_value(
        _strategy_payload(
            config=config,
            risk=risk,
            eval_spec=eval_spec,
            forecaster=forecaster,
            market_selection=market_selection,
            calibration=calibration,
        )
    )
    canonical_json = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_legacy_strategy_version_id(
    config: StrategyConfig,
    risk: RiskParams,
    eval_spec: EvalSpec,
    forecaster: ForecasterSpec,
    market_selection: MarketSelectionSpec,
    calibration: CalibrationSpec | None = None,
) -> str:
    """Compute the pre-factor-definition-version hash for old registry rows."""

    canonical_payload = _normalize_value(
        _strategy_payload(
            config=config,
            risk=risk,
            eval_spec=eval_spec,
            forecaster=forecaster,
            market_selection=market_selection,
            calibration=calibration,
            include_factor_definition_versions=False,
        )
    )
    canonical_json = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(canonical_json.encode("utf-8")).hexdigest()
