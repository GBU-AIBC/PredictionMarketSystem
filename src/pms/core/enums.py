from __future__ import annotations

from enum import StrEnum


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class Venue(StrEnum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class OrderStatus(StrEnum):
    INVALID = "invalid"
    LIVE = "live"
    DELAYED = "delayed"
    MATCHED = "matched"
    PARTIAL = "partial"
    UNMATCHED = "unmatched"
    CANCELLED = "cancelled"
    CANCELED = CANCELLED
    CANCELED_MARKET_RESOLVED = "canceled_market_resolved"


class MarketStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"
    UNOPENED = "unopened"


class FeedbackTarget(StrEnum):
    SENSOR = "sensor"
    CONTROLLER = "controller"
    ACTUATOR = "actuator"
    EVALUATOR = "evaluator"


class FeedbackSource(StrEnum):
    ACTUATOR = "actuator"
    EVALUATOR = "evaluator"
    HUMAN = "human"


class RunMode(StrEnum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class TimeInForce(StrEnum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class ReasonCode(StrEnum):
    """Stable reason/status names shared by decision and execution paths."""

    SUBMISSION_UNKNOWN = "submission_unknown"
    OPERATOR_GATE_DENIED = "operator_gate_denied"
    LIVE_DISABLED = "live_disabled"
    STALE_FACTOR = "stale_factor"
    NON_FINITE = "non_finite"
