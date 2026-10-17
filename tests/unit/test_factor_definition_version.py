from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest

from pms.factors.base import FactorValueRow
from pms.factors.service import persist_factor_value


class _Connection:
    def __init__(self) -> None:
        self.args: tuple[object, ...] | None = None

    async def execute(self, query: str, *args: object) -> str:
        assert "definition_version" in query
        self.args = args
        return "INSERT 0 1"


class _Acquire:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _Connection:
        return self.connection

    async def __aexit__(self, *_: object) -> None:
        return None


class _Pool:
    def __init__(self) -> None:
        self.connection = _Connection()

    def acquire(self) -> _Acquire:
        return _Acquire(self.connection)


@pytest.mark.asyncio
async def test_factor_definition_version_is_written_with_observation() -> None:
    pool = _Pool()
    row = FactorValueRow(
        factor_id="orderbook_imbalance",
        param="",
        market_id="market-1",
        ts=datetime(2026, 5, 4, tzinfo=UTC),
        value=0.2,
    )

    await persist_factor_value(
        cast(Any, pool),
        row,
        definition_version="orderbook-v2",
    )

    assert pool.connection.args is not None
    assert pool.connection.args[-1] == "orderbook-v2"
