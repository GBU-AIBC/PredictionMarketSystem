from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from pms.core.enums import MarketStatus
from pms.core.models import MarketSignal
from pms.factors.base import EMPTY_OUTER_RING
from pms.factors.catalog import FACTOR_CATALOG_ROWS
from pms.factors.definitions import AnchoringLagDivergence, REGISTERED


NOW = datetime(2026, 5, 4, 1, 0, tzinfo=UTC)


def _signal(
    *,
    yes_price: float = 0.50,
    llm_posterior: float = 0.80,
    news_timestamp: datetime | None = None,
) -> MarketSignal:
    return MarketSignal(
        market_id="market-h2",
        token_id="token-h2-yes",
        venue="polymarket",
        title="Will H2 anchoring lag detect delayed repricing?",
        yes_price=yes_price,
        volume_24h=1000.0,
        resolves_at=NOW + timedelta(days=5),
        orderbook={"bids": [], "asks": []},
        external_signal={
            "llm_posterior": llm_posterior,
            "news_timestamp": (news_timestamp or NOW).isoformat(),
        },
        fetched_at=NOW,
        market_status=MarketStatus.OPEN.value,
    )


def test_registered_includes_anchoring_lag_divergence() -> None:
    assert AnchoringLagDivergence in REGISTERED


def test_catalog_includes_anchoring_lag_as_neutral_signed_factor() -> None:
    entry = next(row for row in FACTOR_CATALOG_ROWS if row.factor_id == "anchoring_lag_divergence")

    assert entry.output_type == "scalar"
    assert entry.direction == "neutral"


def test_anchoring_lag_factor_outputs_positive_decayed_divergence() -> None:
    row = AnchoringLagDivergence().compute(
        _signal(
            yes_price=0.50,
            llm_posterior=0.80,
            news_timestamp=NOW - timedelta(hours=6),
        ),
        EMPTY_OUTER_RING,
    )

    assert row is not None
    assert row.factor_id == "anchoring_lag_divergence"
    assert row.param == ""
    assert row.value == pytest.approx(0.225)


def test_anchoring_lag_factor_outputs_negative_decayed_divergence() -> None:
    row = AnchoringLagDivergence().compute(
        _signal(
            yes_price=0.80,
            llm_posterior=0.40,
            news_timestamp=NOW - timedelta(hours=12),
        ),
        EMPTY_OUTER_RING,
    )

    assert row is not None
    assert row.value == pytest.approx(-0.20)


def test_anchoring_lag_factor_emits_neutral_below_threshold_or_after_decay_window() -> None:
    factor = AnchoringLagDivergence()

    below_threshold = factor.compute(
        _signal(yes_price=0.50, llm_posterior=0.60, news_timestamp=NOW),
        EMPTY_OUTER_RING,
    )
    assert below_threshold is not None
    assert below_threshold.value == pytest.approx(0.0)
    after_decay = factor.compute(
        _signal(yes_price=0.50, llm_posterior=0.90, news_timestamp=NOW - timedelta(hours=24)),
        EMPTY_OUTER_RING,
    )
    assert after_decay is not None
    assert after_decay.value == pytest.approx(0.0)


def test_anchoring_lag_factor_returns_none_when_llm_news_inputs_are_absent() -> None:
    signal = _signal()
    signal.external_signal.clear()

    assert AnchoringLagDivergence().compute(signal, EMPTY_OUTER_RING) is None


def test_anchoring_lag_factor_ignores_book_signal_without_llm_news_inputs() -> None:
    signal = _signal(yes_price=0.0)
    signal.external_signal.clear()

    assert AnchoringLagDivergence().compute(signal, EMPTY_OUTER_RING) is None


def test_anchoring_lag_factor_returns_none_for_null_llm_news_inputs() -> None:
    signal = _signal()
    signal.external_signal["llm_posterior"] = None
    assert AnchoringLagDivergence().compute(signal, EMPTY_OUTER_RING) is None

    signal = _signal()
    signal.external_signal["news_timestamp"] = None
    assert AnchoringLagDivergence().compute(signal, EMPTY_OUTER_RING) is None


def test_anchoring_lag_factor_emits_neutral_for_future_news_timestamp() -> None:
    row = AnchoringLagDivergence().compute(
        _signal(news_timestamp=NOW + timedelta(seconds=1)),
        EMPTY_OUTER_RING,
    )
    assert row is not None
    assert row.value == pytest.approx(0.0)


def test_anchoring_lag_factor_rejects_invalid_probability_inputs() -> None:
    with pytest.raises(ValueError, match="llm_posterior"):
        AnchoringLagDivergence().compute(
            _signal(llm_posterior=1.0),
            EMPTY_OUTER_RING,
        )
