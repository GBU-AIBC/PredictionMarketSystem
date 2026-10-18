from __future__ import annotations

import math
from collections.abc import Callable
from decimal import Decimal

import pytest

from pms.core.enums import ReasonCode
from pms.core.exceptions import (
    DomainValidationError,
    NonFiniteValueError,
    OutOfDomainValueError,
)
from pms.core.validation import (
    normalize_resolved_outcome,
    validate_finite,
    validate_notional,
    validate_positive,
    validate_price,
    validate_probability,
    validate_quantity,
)


class FloatWrapper:
    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return self.value


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
        "nan",
        "1e400",
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        10**400,
    ],
)
def test_validate_finite_rejects_all_non_finite_inputs(value: object) -> None:
    with pytest.raises(NonFiniteValueError):
        validate_finite(value)


@pytest.mark.parametrize("value", [None, True, False, b"0.5", bytearray(b"0.5"), "invalid"])
def test_validate_finite_rejects_non_numeric_and_byte_inputs(value: object) -> None:
    with pytest.raises(DomainValidationError):
        validate_finite(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        (-0.3, -0.3),
        (Decimal("0.3"), 0.3),
        (FloatWrapper(0.25), 0.25),
        (" 0.5 ", 0.5),
        (-0.0, 0.0),
        (Decimal("-0.0"), 0.0),
    ],
)
def test_validate_finite_returns_float_and_canonicalizes_signed_zero(
    value: object,
    expected: float,
) -> None:
    result = validate_finite(value)
    assert result == expected
    assert isinstance(result, float)
    if expected == 0.0:
        assert math.copysign(1.0, result) == 1.0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, Decimal("sNaN"), 10**400])
def test_probability_rejects_non_finite_values(value: object) -> None:
    with pytest.raises(NonFiniteValueError):
        validate_probability(value)


@pytest.mark.parametrize("value", [0.0, 1.0, Decimal("0"), Decimal("1")])
def test_probability_accepts_closed_boundaries(value: object) -> None:
    assert validate_probability(value) in (0.0, 1.0)


@pytest.mark.parametrize("value", [math.nextafter(0.0, -math.inf), 1.0 + math.ulp(1.0), -0.3])
def test_probability_rejects_out_of_domain_values(value: float) -> None:
    with pytest.raises(OutOfDomainValueError):
        validate_probability(value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, Decimal("NaN"), 10**400])
def test_price_rejects_non_finite_values(value: object) -> None:
    with pytest.raises(NonFiniteValueError):
        validate_price(value)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.01, 1.01])
def test_price_requires_open_unit_interval(value: float) -> None:
    with pytest.raises(OutOfDomainValueError):
        validate_price(value)


@pytest.mark.parametrize("value", ["0.5", None, b"0.5", "invalid"])
def test_price_handles_strings_and_missing_values(value: object) -> None:
    if value == "0.5":
        assert validate_price(value) == 0.5
    else:
        with pytest.raises(DomainValidationError):
            validate_price(value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, Decimal("sNaN"), 10**400])
def test_positive_rejects_non_finite_values(value: object) -> None:
    with pytest.raises(NonFiniteValueError):
        validate_positive(value)


@pytest.mark.parametrize("value", [0.0, -0.0, -0.3, math.nextafter(0.0, -math.inf)])
def test_positive_requires_strictly_positive_values(value: float) -> None:
    with pytest.raises(OutOfDomainValueError):
        validate_positive(value)


def test_positive_accepts_smallest_positive_float() -> None:
    assert validate_positive(math.nextafter(0.0, math.inf)) > 0.0


@pytest.mark.parametrize("validator", [validate_notional, validate_quantity])
def test_non_negative_domains_accept_zero_and_canonicalize_signed_zero(
    validator: Callable[[object], float],
) -> None:
    assert validator(0.0) == 0.0
    assert validator(-0.0) == 0.0
    assert validator(1.0) == 1.0
    assert math.copysign(1.0, validator(-0.0)) == 1.0


@pytest.mark.parametrize("validator", [validate_notional, validate_quantity])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, Decimal("sNaN"), 10**400])
def test_non_negative_domains_reject_non_finite_values(
    validator: Callable[[object], float],
    value: object,
) -> None:
    with pytest.raises(NonFiniteValueError):
        validator(value)


@pytest.mark.parametrize("validator", [validate_notional, validate_quantity])
@pytest.mark.parametrize("value", [-math.ulp(1.0), "not-a-number", None, b"1"])
def test_non_negative_domains_reject_invalid_values(
    validator: Callable[[object], float],
    value: object,
) -> None:
    with pytest.raises(DomainValidationError):
        validator(value)


@pytest.mark.parametrize(
    "raw",
    [None, "not-a-number", b"1", bytearray(b"1"), math.nan, math.inf, -math.inf, Decimal("sNaN"), -0.1, 1.1, True],
)
def test_normalize_resolved_outcome_returns_none_for_invalid_evidence(raw: object) -> None:
    assert normalize_resolved_outcome(raw) is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        (0, 0.0),
        (1, 1.0),
        (0.5, 0.5),
        ("0.25", 0.25),
        (Decimal("0.3"), 0.3),
        (FloatWrapper(0.75), 0.75),
        (-0.0, 0.0),
    ],
)
def test_normalize_resolved_outcome_returns_float_for_valid_evidence(
    raw: object,
    expected: float,
) -> None:
    normalized = normalize_resolved_outcome(raw)
    assert normalized == expected
    assert isinstance(normalized, float)
    if expected == 0.0:
        assert math.copysign(1.0, normalized) == 1.0


def test_reason_codes_are_stable_strings() -> None:
    assert {reason.value for reason in ReasonCode} == {
        "submission_unknown",
        "operator_gate_denied",
        "live_disabled",
        "stale_factor",
        "non_finite",
    }
