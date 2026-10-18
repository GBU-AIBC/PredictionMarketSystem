from __future__ import annotations

from datetime import datetime

from pms.core.models import MarketSignal
from pms.factors.base import FactorDefinition, FactorValueRow, OuterRingReader


DEFAULT_DIVERGENCE_THRESHOLD = 0.15
DEFAULT_DECAY_WINDOW_HOURS = 24.0


class AnchoringLagDivergence(FactorDefinition):
    """Signed H2 divergence between LLM posterior and market YES price.

    Positive values mean the market appears to underreact to positive news
    (buy YES). Negative values mean the market appears to underreact to negative
    news (buy NO). Values linearly decay to zero after the configured news
    window.
    """

    factor_id = "anchoring_lag_divergence"
    definition_version = "2"
    required_inputs = (
        "yes_price",
        "external_signal.llm_posterior",
        "external_signal.news_timestamp",
    )

    def __init__(
        self,
        *,
        divergence_threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
        decay_window_hours: float = DEFAULT_DECAY_WINDOW_HOURS,
    ) -> None:
        if divergence_threshold <= 0.0:
            msg = "divergence_threshold must be > 0.0"
            raise ValueError(msg)
        if decay_window_hours <= 0.0:
            msg = "decay_window_hours must be > 0.0"
            raise ValueError(msg)
        self._divergence_threshold = divergence_threshold
        self._decay_window_hours = decay_window_hours

    def compute(
        self,
        signal: MarketSignal,
        outer_ring: OuterRingReader,
    ) -> FactorValueRow | None:
        del outer_ring

        if (
            "llm_posterior" not in signal.external_signal
            or "news_timestamp" not in signal.external_signal
            or signal.external_signal["llm_posterior"] is None
            or signal.external_signal["news_timestamp"] is None
        ):
            return None
        yes_price = _require_open_probability(signal.yes_price, "yes_price")
        llm_posterior = _require_open_probability(
            signal.external_signal.get("llm_posterior"),
            "llm_posterior",
        )
        news_timestamp = _require_datetime(
            signal.external_signal.get("news_timestamp"),
            "news_timestamp",
        )
        decay = _linear_decay(
            now=signal.timestamp,
            news_timestamp=news_timestamp,
            decay_window_hours=self._decay_window_hours,
        )
        delta_effective = (llm_posterior - yes_price) * decay
        if abs(delta_effective) <= self._divergence_threshold:
            # Preserve a current, neutral observation so the factor panel can
            # never fall back to a stale divergence after repricing.
            delta_effective = 0.0

        return FactorValueRow(
            factor_id=self.factor_id,
            param="",
            market_id=signal.market_id,
            ts=signal.timestamp,
            value=delta_effective,
        )


def _linear_decay(
    *,
    now: datetime,
    news_timestamp: datetime,
    decay_window_hours: float,
) -> float:
    elapsed_hours = (now - news_timestamp).total_seconds() / 3600.0
    if elapsed_hours < 0.0:
        return 0.0
    return max(0.0, 1.0 - elapsed_hours / decay_window_hours)


def _require_open_probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{field_name} must be numeric"
        raise TypeError(msg)
    probability = float(value)
    if probability <= 0.0 or probability >= 1.0 or probability != probability:
        msg = f"{field_name} must satisfy 0.0 < value < 1.0"
        raise ValueError(msg)
    return probability


def _require_datetime(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as error:
            msg = f"{field_name} must be an ISO datetime"
            raise ValueError(msg) from error
    msg = f"{field_name} must be an ISO datetime"
    raise TypeError(msg)
