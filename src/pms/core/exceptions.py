from __future__ import annotations


class PMSBootError(RuntimeError):
    """Raised when a PMS process cannot complete mandatory startup checks."""


class KalshiStubError(NotImplementedError):
    """Raised when v1 runtime paths encounter the reserved Kalshi venue."""


class SensorDataQualityError(RuntimeError):
    """Raised when sensor input quality degrades beyond the configured threshold."""


class DomainValidationError(ValueError):
    """Raised when a numeric value cannot satisfy a core domain contract."""


class NonFiniteValueError(DomainValidationError):
    """Raised when a numeric value is NaN or infinite."""


class OutOfDomainValueError(DomainValidationError):
    """Raised when a finite numeric value is outside its allowed domain."""
