from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from pms.controller.factor_snapshot import PostgresFactorSnapshotReader, _freshness_slas
from pms.strategies.projections import FactorCompositionStep


NOW = datetime(2026, 5, 4, 12, 0, tzinfo=UTC)


class _Connection:
    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        del query, args
        return [
            {
                "factor_id": "metaculus_prior",
                "param": "",
                "value": 0.7,
                "ts": NOW - timedelta(seconds=11),
            }
        ]


class _Acquire:
    async def __aenter__(self) -> _Connection:
        return _Connection()

    async def __aexit__(self, *_: object) -> None:
        return None


class _Pool:
    def acquire(self) -> _Acquire:
        return _Acquire()


@pytest.mark.asyncio
async def test_optional_factor_sla_drops_stale_value_from_snapshot() -> None:
    reader = PostgresFactorSnapshotReader(cast(Any, _Pool()))
    snapshot = await reader.snapshot(
        market_id="market-1",
        as_of=NOW,
        required=(
            FactorCompositionStep(
                factor_id="metaculus_prior",
                role="rule_delta",
                param="",
                weight=1.0,
                threshold=None,
                required=False,
                freshness_sla_s=10.0,
            ),
        ),
        strategy_id="strategy",
        strategy_version_id="version",
    )

    assert snapshot.values == {}
    assert snapshot.missing_factors == ()
    assert snapshot.stale_factors == ()


def test_freshness_sla_none_does_not_overwrite_numeric_sla() -> None:
    steps = (
        FactorCompositionStep(
            factor_id="metaculus_prior",
            role="rule_delta",
            param="",
            weight=1.0,
            threshold=None,
            required=False,
            freshness_sla_s=10.0,
        ),
        FactorCompositionStep(
            factor_id="metaculus_prior",
            role="rule_delta",
            param="",
            weight=1.0,
            threshold=None,
            required=False,
            freshness_sla_s=None,
        ),
    )

    assert _freshness_slas(steps) == {("metaculus_prior", ""): 10.0}
