"""Small, dependency-free validation contracts shared across runtime layers."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import SupportsFloat, SupportsIndex, cast

from pms.core.exceptions import (
    DomainValidationError,
    NonFiniteValueError,
    OutOfDomainValueError,
)


def validate_finite(value: object, *, field_name: str = "value") -> float:
    """Return ``value`` as a finite, canonicalized float.

    ``bool``, bytes, bytearrays, missing values, and non-numeric values are
    rejected as domain errors. Numeric values that cannot be represented as a
    finite float, including overflow and Decimal NaN variants, are rejected as
    :class:`NonFiniteValueError` so callers can report one stable reason.
    """

    if value is None or isinstance(value, (bool, bytes, bytearray)):
        msg = f"{field_name} must be numeric"
        raise DomainValidationError(msg)

    if isinstance(value, Decimal) and not value.is_finite():
        msg = f"{field_name} must be finite"
        raise NonFiniteValueError(msg)

    try:
        parsed = float(cast(str | SupportsFloat | SupportsIndex, value))
    except OverflowError as exc:
        msg = f"{field_name} must be finite"
        raise NonFiniteValueError(msg) from exc
    except (TypeError, ValueError) as exc:
        msg = f"{field_name} must be numeric"
        raise DomainValidationError(msg) from exc

    if not math.isfinite(parsed):
        msg = f"{field_name} must be finite"
        raise NonFiniteValueError(msg)
    return 0.0 if parsed == 0.0 else parsed


def _within_bounds(
    value: float,
    *,
    field_name: str,
    lower: float,
    upper: float,
    include_lower: bool,
    include_upper: bool,
) -> float:
    lower_ok = value >= lower if include_lower else value > lower
    upper_ok = value <= upper if include_upper else value < upper
    if not lower_ok or not upper_ok:
        left = "[" if include_lower else "("
        right = "]" if include_upper else ")"
        msg = f"{field_name} must be in {left}{lower}, {upper}{right}"
        raise OutOfDomainValueError(msg)
    return value


def validate_probability(value: object, *, field_name: str = "probability") -> float:
    """Return a finite probability in the closed interval ``[0, 1]``."""

    parsed = validate_finite(value, field_name=field_name)
    return _within_bounds(
        parsed,
        field_name=field_name,
        lower=0.0,
        upper=1.0,
        include_lower=True,
        include_upper=True,
    )


def validate_price(value: object, *, field_name: str = "price") -> float:
    """Return a finite price in the open interval ``(0, 1)``."""

    parsed = validate_finite(value, field_name=field_name)
    return _within_bounds(
        parsed,
        field_name=field_name,
        lower=0.0,
        upper=1.0,
        include_lower=False,
        include_upper=False,
    )


def validate_positive(value: object, *, field_name: str = "value") -> float:
    """Return a finite value strictly greater than zero."""

    parsed = validate_finite(value, field_name=field_name)
    return _within_bounds(
        parsed,
        field_name=field_name,
        lower=0.0,
        upper=math.inf,
        include_lower=False,
        include_upper=False,
    )


def _non_negative(value: object, *, field_name: str) -> float:
    parsed = validate_finite(value, field_name=field_name)
    return _within_bounds(
        parsed,
        field_name=field_name,
        lower=0.0,
        upper=math.inf,
        include_lower=True,
        include_upper=False,
    )


def validate_notional(value: object, *, field_name: str = "notional") -> float:
    """Return a finite non-negative notional."""

    return _non_negative(value, field_name=field_name)


def validate_quantity(value: object, *, field_name: str = "quantity") -> float:
    """Return a finite non-negative quantity."""

    return _non_negative(value, field_name=field_name)


def normalize_resolved_outcome(raw: object) -> float | None:
    """Normalize optional resolution evidence to a finite probability.

    Missing evidence returns ``None``. Invalid, non-finite, and out-of-range
    evidence is also treated as unresolved so optional metadata cannot abort
    fill or evaluation handling. Callers that need an anomaly can distinguish
    the two cases with ``raw is not None and normalized is None``. Numeric
    strings are accepted for compatibility with JSON and CSV inputs; booleans
    are deliberately excluded even though ``float(bool)`` succeeds.
    """

    if raw is None:
        return None
    try:
        return validate_probability(raw, field_name="resolved_outcome")
    except DomainValidationError:
        return None
