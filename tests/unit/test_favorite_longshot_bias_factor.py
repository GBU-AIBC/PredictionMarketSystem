from __future__ import annotations

from datetime import UTC, datetime

import pytest

from pms.core.enums import MarketStatus
from pms.core.models import MarketSignal
from pms.factors.base import EMPTY_OUTER_RING
from pms.factors.catalog import (
    FACTOR_CATALOG_ROWS,
    FactorCatalogEntry,
    validate_registered_factor_catalog,
)
import pms.factors.catalog as catalog
from pms.factors.definitions import FavoriteLongshotBias, REGISTERED


def _signal(*, yes_price: float) -> MarketSignal:
    return MarketSignal(
        market_id="market-flb",
        token_id="token-flb-yes",
        venue="polymarket",
        title="Will FLB identify the contrarian side?",
        yes_price=yes_price,
        volume_24h=1000.0,
        resolves_at=datetime(2026, 5, 30, tzinfo=UTC),
        orderbook={"bids": [], "asks": []},
        external_signal={},
        fetched_at=datetime(2026, 5, 3, 12, 0, tzinfo=UTC),
        market_status=MarketStatus.OPEN.value,
    )


def test_registered_includes_favorite_longshot_bias() -> None:
    assert FavoriteLongshotBias in REGISTERED


def test_registered_factor_catalog_has_no_drift() -> None:
    validate_registered_factor_catalog()


def test_catalog_factor_without_registered_definition_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        catalog,
        "FACTOR_CATALOG_ROWS",
        (
            *FACTOR_CATALOG_ROWS,
            FactorCatalogEntry(
                factor_id="unregistered_factor",
                name="Unregistered",
                description="test",
                input_schema_hash="test",
                output_type="scalar",
                direction="neutral",
                owner="system",
            ),
        ),
    )

    with pytest.raises(ValueError, match="missing from REGISTERED"):
        validate_registered_factor_catalog()


def test_catalog_includes_flb_factor_as_schema_compatible_neutral_direction() -> None:
    entry = next(row for row in FACTOR_CATALOG_ROWS if row.factor_id == "favorite_longshot_bias")

    assert entry.output_type == "scalar"
    assert entry.direction == "neutral"


def test_flb_factor_marks_low_yes_longshot_as_buy_no_signal() -> None:
    factor = FavoriteLongshotBias()

    row = factor.compute(_signal(yes_price=0.05), EMPTY_OUTER_RING)

    assert row is not None
    assert factor.required_inputs == ("yes_price",)
    assert row.factor_id == "favorite_longshot_bias"
    assert row.param == ""
    assert row.value == pytest.approx(-0.05)


def test_flb_factor_marks_high_yes_favorite_as_buy_yes_signal() -> None:
    row = FavoriteLongshotBias().compute(_signal(yes_price=0.95), EMPTY_OUTER_RING)

    assert row is not None
    assert row.value == pytest.approx(0.05)


def test_flb_factor_emits_explicit_neutral_for_middle_deciles() -> None:
    row = FavoriteLongshotBias().compute(_signal(yes_price=0.50), EMPTY_OUTER_RING)
    assert row is not None
    assert row.value == pytest.approx(0.0)
