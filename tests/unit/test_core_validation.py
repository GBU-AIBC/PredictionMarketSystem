from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from pms.core.enums import ReasonCode, Venue
from pms.core.exceptions import (
    DomainValidationError,
    NonFiniteValueError,
    OutOfDomainValueError,
)
from pms.core.validation import (
    normalize_resolved_outcome,
    validate_notional,
    validate_price,
    validate_probability,
    validate_quantity,
    validate_venue_price,
)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_probability_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(NonFiniteValueError):
        validate_probability(value)


@pytest.mark.parametrize("value", [math.nextafter(0.0, -math.inf), 1.0 + math.ulp(1.0)])
def test_probability_rejects_just_outside_values(value: float) -> None:
    with pytest.raises(OutOfDomainValueError):
        validate_probability(value)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_probability_accepts_closed_boundaries(value: float) -> None:
    assert validate_probability(value) == value


@pytest.mark.parametrize("value", ["not-a-number", None, True])
def test_probability_rejects_non_numeric_values(value: object) -> None:
    with pytest.raises(DomainValidationError):
        validate_probability(value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_generic_price_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(NonFiniteValueError):
        validate_price(value)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.01, 1.01])
def test_generic_price_requires_open_unit_interval(value: float) -> None:
    with pytest.raises(OutOfDomainValueError):
        validate_price(value)


@pytest.mark.parametrize("value", ["0.5", None, "invalid"])
def test_price_handles_strings_and_missing_values(value: object) -> None:
    if value == "0.5":
        assert validate_price(value) == 0.5
    else:
        with pytest.raises(DomainValidationError):
            validate_price(value)


def test_venue_price_applies_wire_bounds() -> None:
    assert validate_venue_price(1.0, Venue.POLYMARKET) == 1.0
    assert validate_venue_price("0.01", Venue.KALSHI) == 0.01
    assert validate_price(0.99, venue="kalshi") == 0.99

    with pytest.raises(OutOfDomainValueError):
        validate_venue_price(0.0, "polymarket")
    with pytest.raises(OutOfDomainValueError):
        validate_venue_price(0.005, "kalshi")
    with pytest.raises(DomainValidationError):
        validate_venue_price(0.5, "unknown")


@pytest.mark.parametrize("validator", [validate_notional, validate_quantity])
def test_non_negative_domains_accept_zero_and_reject_negative(
    validator: Callable[[object], float],
) -> None:
    assert validator(0.0) == 0.0
    assert validator(1.0) == 1.0
    assert validator("2.5") == 2.5
    with pytest.raises(OutOfDomainValueError):
        validator(-math.ulp(1.0))
    with pytest.raises(DomainValidationError):
        validator("not-a-number")
    with pytest.raises(DomainValidationError):
        validator(None)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("validator", [validate_notional, validate_quantity])
def test_non_negative_domains_reject_non_finite_values(
    validator: Callable[[object], float],
    value: float,
) -> None:
    with pytest.raises(NonFiniteValueError):
        validator(value)


@pytest.mark.parametrize(
    "raw", [None, "not-a-number", math.nan, math.inf, -math.inf, -0.1, 1.1, True]
)
def test_normalize_resolved_outcome_returns_none_for_invalid_evidence(raw: object) -> None:
    assert normalize_resolved_outcome(raw) is None


@pytest.mark.parametrize("raw, expected", [(0, 0.0), (1, 1.0), (0.5, 0.5), ("0.25", 0.25)])
def test_normalize_resolved_outcome_returns_float_for_valid_evidence(
    raw: object,
    expected: float,
) -> None:
    normalized = normalize_resolved_outcome(raw)
    assert normalized == expected
    assert isinstance(normalized, float)


def test_reason_codes_are_stable_strings() -> None:
    assert {reason.value for reason in ReasonCode} == {
        "submission_unknown",
        "operator_gate_denied",
        "live_disabled",
        "stale_factor",
        "non_finite",
    }
