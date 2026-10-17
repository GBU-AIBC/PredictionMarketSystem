"""Small, dependency-free validation contracts shared across runtime layers."""

from __future__ import annotations

import math
from typing import Final, cast

from pms.core.enums import Venue
from pms.core.exceptions import (
    DomainValidationError,
    NonFiniteValueError,
    OutOfDomainValueError,
)

_POLYMARKET_BOUNDS: Final[tuple[float, float, bool, bool]] = (0.0, 1.0, False, True)
_KALSHI_BOUNDS: Final[tuple[float, float, bool, bool]] = (0.01, 0.99, True, True)


def _as_finite_float(value: object, *, field_name: str) -> float:
    if value is None or isinstance(value, bool):
        msg = f"{field_name} must be numeric"
        raise DomainValidationError(msg)

    try:
        parsed = float(cast(int | float | str, value))
    except (TypeError, ValueError, OverflowError) as exc:
        msg = f"{field_name} must be numeric"
        raise DomainValidationError(msg) from exc

    if not math.isfinite(parsed):
        msg = f"{field_name} must be finite"
        raise NonFiniteValueError(msg)
    return parsed


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

    parsed = _as_finite_float(value, field_name=field_name)
    return _within_bounds(
        parsed,
        field_name=field_name,
        lower=0.0,
        upper=1.0,
        include_lower=True,
        include_upper=True,
    )


def validate_price(
    value: object,
    *,
    venue: Venue | str | None = None,
    field_name: str = "price",
) -> float:
    """Return a finite open-interval price, or a venue-bounded price.

    The generic contract is ``0 < price < 1``.  Passing ``venue`` applies the
    venue's wire bounds instead: Polymarket accepts ``(0, 1]`` and the
    reserved Kalshi contract uses ``[0.01, 0.99]``.
    """

    if venue is None:
        bounds = (0.0, 1.0, False, False)
    else:
        bounds = _venue_price_bounds(venue)
    parsed = _as_finite_float(value, field_name=field_name)
    return _within_bounds(
        parsed,
        field_name=field_name,
        lower=bounds[0],
        upper=bounds[1],
        include_lower=bounds[2],
        include_upper=bounds[3],
    )


def validate_venue_price(
    value: object,
    venue: Venue | str,
    *,
    field_name: str = "price",
) -> float:
    """Return ``value`` after applying the selected venue's price bounds."""

    return validate_price(value, venue=venue, field_name=field_name)


def _venue_price_bounds(
    venue: Venue | str,
) -> tuple[float, float, bool, bool]:
    if isinstance(venue, Venue):
        venue_name = venue.value
    else:
        venue_name = str(venue).strip().lower()

    if venue_name == Venue.POLYMARKET.value:
        return _POLYMARKET_BOUNDS
    if venue_name == Venue.KALSHI.value:
        return _KALSHI_BOUNDS

    msg = f"unsupported venue {venue!r}"
    raise DomainValidationError(msg)


def validate_notional(value: object, *, field_name: str = "notional") -> float:
    """Return a finite non-negative notional."""

    parsed = _as_finite_float(value, field_name=field_name)
    return _within_bounds(
        parsed,
        field_name=field_name,
        lower=0.0,
        upper=math.inf,
        include_lower=True,
        include_upper=False,
    )


def validate_quantity(value: object, *, field_name: str = "quantity") -> float:
    """Return a finite non-negative quantity."""

    parsed = _as_finite_float(value, field_name=field_name)
    return _within_bounds(
        parsed,
        field_name=field_name,
        lower=0.0,
        upper=math.inf,
        include_lower=True,
        include_upper=False,
    )


def normalize_resolved_outcome(raw: object) -> float | None:
    """Normalize optional resolution evidence to a finite probability.

    Invalid, missing, non-finite, and out-of-range evidence is treated as
    unresolved so optional metadata cannot abort fill or evaluation handling.
    Numeric strings are accepted for compatibility with JSON and CSV inputs;
    booleans are deliberately excluded even though ``float(bool)`` succeeds.
    """

    if raw is None or isinstance(raw, bool):
        return None
    try:
        parsed = float(cast(int | float | str, raw))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        return None
    return parsed
