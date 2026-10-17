from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import asyncpg
import pytest

from pms.core.models import FillRecord, OrderState, Position
from pms.storage.fill_store import FillStore, _json_object as fill_json_object
from pms.storage.fill_store import StoredTradeRow, _string_list
from pms.storage.order_store import OrderStore, _json_object as order_json_object


class _TransactionRecorder:
    def __init__(self, connection: "_RecordingConnection") -> None:
        self._connection = connection

    async def __aenter__(self) -> "_TransactionRecorder":
        self._connection.in_transaction = True
        self._connection.transaction_entries += 1
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self._connection.in_transaction = False


class _RecordingConnection:
    def __init__(self) -> None:
        self.in_transaction = False
        self.transaction_entries = 0
        self.execute_flags: list[bool] = []
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetch_rows: list[object] = []

    async def execute(self, query: str, *args: object) -> str:
        self.execute_flags.append(self.in_transaction)
        self.execute_calls.append((query, args))
        return "OK"

    async def fetch(self, query: str, *args: object) -> list[object]:
        self.fetch_calls.append((query, args))
        return list(self.fetch_rows)

    def transaction(self) -> _TransactionRecorder:
        return _TransactionRecorder(self)


class _AcquireContext:
    def __init__(self, connection: _RecordingConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _RecordingConnection:
        return self._connection

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb


class _RecordingPool:
    def __init__(self, connection: _RecordingConnection) -> None:
        self._connection = connection

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self._connection)


def _order_state() -> OrderState:
    return OrderState(
        order_id="order-unit-cp10-1",
        decision_id="decision-unit-cp10-1",
        status="matched",
        market_id="market-unit-cp10-1",
        token_id="token-unit-cp10-1",
        venue="polymarket",
        requested_notional_usdc=100.0,
        filled_notional_usdc=100.0,
        remaining_notional_usdc=0.0,
        fill_price=0.25,
        submitted_at=datetime(2026, 4, 21, 10, 0, tzinfo=UTC),
        last_updated_at=datetime(2026, 4, 21, 10, 0, tzinfo=UTC),
        raw_status="matched",
        strategy_id="default",
        strategy_version_id="default-v2",
        filled_quantity=400.0,
    )


def _fill_record() -> FillRecord:
    return FillRecord(
        trade_id="trade-unit-cp10-1",
        fill_id="fill-unit-cp10-1",
        order_id="order-unit-cp10-1",
        decision_id="decision-unit-cp10-1",
        market_id="market-unit-cp10-1",
        token_id="token-unit-cp10-1",
        venue="polymarket",
        side="yes",
        fill_price=0.25,
        fill_notional_usdc=100.0,
        fill_quantity=400.0,
        executed_at=datetime(2026, 4, 21, 10, 0, tzinfo=UTC),
        filled_at=datetime(2026, 4, 21, 10, 0, tzinfo=UTC),
        status="filled",
        anomaly_flags=["checked"],
        strategy_id="default",
        strategy_version_id="default-v2",
        risk_group_id="event:unit-cp10",
    )


def _position_fill_row(
    *,
    fill_id: str,
    side: str,
    fill_price: float,
    fill_quantity: float,
    filled_at: datetime | None = None,
    market_id: str = "market-unit-cp10-1",
    token_id: str = "token-unit-cp10-1",
    strategy_id: str = "default",
    strategy_version_id: str = "default-v2",
    current_price: float | None = 0.31,
    mark_source: str | None = "clob",
    mark_age_seconds: float | None = 12.5,
    risk_group_id: str | None = None,
) -> dict[str, object]:
    return {
        "fill_id": fill_id,
        "market_id": market_id,
        "ts": filled_at or datetime(2026, 4, 21, 10, 0, tzinfo=UTC),
        "fill_notional_usdc": fill_price * fill_quantity,
        "fill_quantity": fill_quantity,
        "strategy_id": strategy_id,
        "strategy_version_id": strategy_version_id,
        "token_id": token_id,
        "venue": "polymarket",
        "side": side,
        "risk_group_id": risk_group_id,
        "fill_price": fill_price,
        "current_price": current_price,
        "mark_source": mark_source,
        "mark_age_seconds": mark_age_seconds,
    }


def test_bind_pool_sets_store_pool_reference() -> None:
    pool = cast(asyncpg.Pool, object())

    order_store = OrderStore()
    fill_store = FillStore()

    order_store.bind_pool(pool)
    fill_store.bind_pool(pool)

    assert order_store.pool is pool
    assert fill_store.pool is pool


@pytest.mark.asyncio
async def test_order_store_requires_bound_pool_for_insert() -> None:
    with pytest.raises(RuntimeError, match="OrderStore pool is not bound"):
        await OrderStore().insert(_order_state())


@pytest.mark.asyncio
async def test_fill_store_requires_bound_pool_for_insert() -> None:
    with pytest.raises(RuntimeError, match="FillStore pool is not bound"):
        await FillStore().insert(_fill_record())


@pytest.mark.asyncio
async def test_store_get_short_circuits_without_pool_or_fill_id() -> None:
    assert await OrderStore().get("missing") is None
    assert await FillStore().get(None) is None


def test_payload_helpers_cover_dict_and_error_branches() -> None:
    assert order_json_object({"status": "matched"}) == {"status": "matched"}
    assert order_json_object('{"status": "matched"}') == {"status": "matched"}
    assert fill_json_object({"status": "filled"}) == {"status": "filled"}
    assert _string_list(["a", 2]) == ["a", "2"]
    assert _string_list(None) == []

    with pytest.raises(RuntimeError, match="order payload must be a JSON object"):
        order_json_object(123)

    with pytest.raises(RuntimeError, match="fill payload must be a JSON object"):
        fill_json_object(123)


@pytest.mark.asyncio
async def test_order_store_insert_wraps_shell_and_payload_writes_in_transaction() -> None:
    connection = _RecordingConnection()
    store = OrderStore(cast(asyncpg.Pool, _RecordingPool(connection)))

    await store.insert(_order_state())

    assert connection.transaction_entries == 1
    assert connection.execute_flags == [False, True, True]
    orders_query, orders_args = connection.execute_calls[1]
    assert "$19" not in orders_query
    assert len(orders_args) == 18


@pytest.mark.asyncio
async def test_fill_store_insert_wraps_shell_and_payload_writes_in_transaction() -> None:
    connection = _RecordingConnection()
    store = FillStore(cast(asyncpg.Pool, _RecordingPool(connection)))

    await store.insert(_fill_record())

    assert connection.transaction_entries == 1
    assert connection.execute_flags == [False, True, True]
    payload = str(connection.execute_calls[2][1][1])
    assert '"risk_group_id": "event:unit-cp10"' in payload


@pytest.mark.real_fill_store
@pytest.mark.asyncio
async def test_fill_store_read_positions_marks_with_latest_clob_best_bid() -> None:
    connection = _RecordingConnection()
    store = FillStore(cast(asyncpg.Pool, _RecordingPool(connection)))

    await store.read_positions()

    query = connection.fetch_calls[0][0]
    assert "LEFT JOIN LATERAL" in query
    assert "book_snapshots" in query
    assert "book_levels" in query
    assert "MAX(book_levels.price)" in query
    assert "book_levels.side = 'BUY'" in query
    assert "book_snapshots.token_id = fill_payloads.payload->>'token_id'" in query
    assert "book_snapshots.ts > NOW() - INTERVAL '60 seconds'" in query
    assert "book_snapshots.ts AS snapshot_ts" in query
    assert "mark_source" in query
    assert "mark_age_seconds" in query
    assert "GREATEST(\n                            EXTRACT(EPOCH" in query
    assert "ORDER BY book_snapshots.ts DESC, book_snapshots.id DESC" in query
    assert query.index("clob_marks.best_bid") < query.index("markets.yes_price")


@pytest.mark.real_fill_store
@pytest.mark.asyncio
async def test_fill_store_read_positions_nets_opposing_buy_sell_fills() -> None:
    connection = _RecordingConnection()
    connection.fetch_rows = [
        _position_fill_row(
            fill_id="fill-net-buy",
            side="BUY",
            fill_price=0.30,
            fill_quantity=10.0,
        ),
        _position_fill_row(
            fill_id="fill-net-sell",
            side="SELL",
            fill_price=0.40,
            fill_quantity=10.0,
        ),
    ]
    store = FillStore(cast(asyncpg.Pool, _RecordingPool(connection)))

    positions = await store.read_positions()

    assert positions == []


@pytest.mark.real_fill_store
@pytest.mark.asyncio
async def test_fill_store_read_positions_preserves_cost_basis_after_partial_exit() -> None:
    connection = _RecordingConnection()
    connection.fetch_rows = [
        _position_fill_row(
            fill_id="fill-lot-buy-1",
            side="BUY",
            fill_price=0.40,
            fill_quantity=10.0,
            filled_at=datetime(2026, 4, 21, 10, 0, tzinfo=UTC),
            current_price=0.50,
        ),
        _position_fill_row(
            fill_id="fill-lot-sell",
            side="SELL",
            fill_price=0.60,
            fill_quantity=5.0,
            filled_at=datetime(2026, 4, 21, 10, 1, tzinfo=UTC),
            current_price=0.50,
        ),
        _position_fill_row(
            fill_id="fill-lot-buy-2",
            side="BUY",
            fill_price=0.50,
            fill_quantity=5.0,
            filled_at=datetime(2026, 4, 21, 10, 2, tzinfo=UTC),
            current_price=0.50,
        ),
    ]
    store = FillStore(cast(asyncpg.Pool, _RecordingPool(connection)))

    positions = await store.read_positions()

    assert len(positions) == 1
    assert positions[0].shares_held == pytest.approx(10.0)
    assert positions[0].avg_entry_price == pytest.approx(0.45)
    assert positions[0].locked_usdc == pytest.approx(4.50)


@pytest.mark.real_fill_store
@pytest.mark.asyncio
async def test_fill_store_read_positions_maps_aggregated_rows() -> None:
    connection = _RecordingConnection()
    connection.fetch_rows = [
        _position_fill_row(
            fill_id="fill-map",
            side="yes",
            fill_price=0.25,
            fill_quantity=400.0,
            risk_group_id="event:unit-cp10",
        )
    ]
    store = FillStore(cast(asyncpg.Pool, _RecordingPool(connection)))

    positions = await store.read_positions()

    assert positions == [
        Position(
            market_id="market-unit-cp10-1",
            token_id="token-unit-cp10-1",
            venue="polymarket",
            side="yes",
            shares_held=400.0,
            avg_entry_price=0.25,
            unrealized_pnl=24.0,
            locked_usdc=100.0,
            mark_source="clob",
            mark_age_seconds=12.5,
            current_price=0.31,
            opened_at=datetime(2026, 4, 21, 10, 0, tzinfo=UTC),
            strategy_id="default",
            strategy_version_id="default-v2",
            risk_group_id="event:unit-cp10",
        )
    ]
    assert "ORDER BY" in connection.fetch_calls[0][0]
    assert "LEFT JOIN tokens" in connection.fetch_calls[0][0]
    assert "LEFT JOIN markets" in connection.fetch_calls[0][0]


@pytest.mark.real_fill_store
@pytest.mark.asyncio
async def test_fill_store_read_positions_maps_missing_market_price_to_zero_pnl() -> None:
    connection = _RecordingConnection()
    connection.fetch_rows = [
        _position_fill_row(
            fill_id="fill-missing-price",
            market_id="market-unit-cp10-2",
            token_id="token-unit-cp10-2",
            side="BUY",
            fill_price=0.42,
            fill_quantity=50.0,
            current_price=None,
            mark_source="gamma",
            mark_age_seconds=None,
        )
    ]
    store = FillStore(cast(asyncpg.Pool, _RecordingPool(connection)))

    positions = await store.read_positions()

    assert positions == [
        Position(
            market_id="market-unit-cp10-2",
            token_id="token-unit-cp10-2",
            venue="polymarket",
            side="BUY",
            shares_held=50.0,
            avg_entry_price=0.42,
            unrealized_pnl=0.0,
            locked_usdc=21.0,
            mark_source="gamma",
            mark_age_seconds=None,
            opened_at=datetime(2026, 4, 21, 10, 0, tzinfo=UTC),
            strategy_id="default",
            strategy_version_id="default-v2",
        )
    ]


@pytest.mark.real_fill_store
@pytest.mark.asyncio
async def test_fill_store_read_positions_nets_across_risk_group_id_change() -> None:
    connection = _RecordingConnection()
    connection.fetch_rows = [
        _position_fill_row(
            fill_id="fill-open-rg1",
            side="BUY",
            fill_price=0.30,
            fill_quantity=10.0,
            risk_group_id="rg-1",
        ),
        _position_fill_row(
            fill_id="fill-partial-close-rg2",
            side="SELL",
            fill_price=0.40,
            fill_quantity=4.0,
            risk_group_id="rg-2",
            filled_at=datetime(2026, 4, 21, 11, 0, tzinfo=UTC),
        ),
    ]
    store = FillStore(cast(asyncpg.Pool, _RecordingPool(connection)))

    positions = await store.read_positions()

    assert len(positions) == 1
    assert positions[0].shares_held == pytest.approx(6.0)
    assert positions[0].risk_group_id == "rg-2"


@pytest.mark.real_fill_store
@pytest.mark.asyncio
async def test_fill_store_read_positions_keeps_netting_when_risk_group_metadata_appears() -> None:
    connection = _RecordingConnection()
    connection.fetch_rows = [
        _position_fill_row(
            fill_id="fill-historical-buy",
            side="BUY",
            fill_price=0.30,
            fill_quantity=10.0,
            risk_group_id=None,
        ),
        _position_fill_row(
            fill_id="fill-later-buy",
            side="BUY",
            fill_price=0.32,
            fill_quantity=5.0,
            risk_group_id="rg-newly-set",
            filled_at=datetime(2026, 4, 21, 11, 0, tzinfo=UTC),
        ),
    ]
    store = FillStore(cast(asyncpg.Pool, _RecordingPool(connection)))

    positions = await store.read_positions()

    assert len(positions) == 1
    assert positions[0].shares_held == pytest.approx(15.0)
    assert positions[0].risk_group_id == "rg-newly-set"


@pytest.mark.asyncio
async def test_fill_store_read_trades_fails_closed_on_missing_payloads() -> None:
    connection = _RecordingConnection()
    connection.fetch_rows = [
        {
            "fill_id": "fill-unit-cp10-1",
            "order_id": "order-unit-cp10-1",
            "market_id": "market-unit-cp10-1",
            "ts": datetime(2026, 4, 21, 10, 0, tzinfo=UTC),
            "fill_notional_usdc": 100.0,
            "fill_quantity": 400.0,
            "strategy_id": "default",
            "strategy_version_id": "default-v2",
            "question": "Will CP10 fill rows persist?",
            "payload": {
                "trade_id": "trade-unit-cp10-1",
                "decision_id": "decision-unit-cp10-1",
                "token_id": "token-unit-cp10-1",
                "venue": "polymarket",
                "side": "yes",
                "fill_price": 0.25,
                "executed_at": "2026-04-21T10:00:00+00:00",
                "status": "filled",
                "fee_bps": 10,
                "fees": 0.1,
                "risk_group_id": "event:unit-cp10",
            },
        },
        {
            "fill_id": "fill-unit-cp10-2",
            "order_id": "order-unit-cp10-2",
            "market_id": "market-unit-cp10-2",
            "ts": datetime(2026, 4, 21, 10, 0, tzinfo=UTC),
            "fill_notional_usdc": 50.0,
            "fill_quantity": 200.0,
            "strategy_id": "default",
            "strategy_version_id": "default-v2",
            "question": "skip me",
            "payload": None,
        },
    ]
    store = FillStore(cast(asyncpg.Pool, _RecordingPool(connection)))

    with pytest.raises(RuntimeError, match="fill-unit-cp10-2"):
        await store.read_trades(limit=10)
    assert connection.fetch_calls[0][1] == (None, 10, 0)


@pytest.mark.asyncio
async def test_fill_store_read_trades_uses_half_open_until_cutoff() -> None:
    connection = _RecordingConnection()
    store = FillStore(cast(asyncpg.Pool, _RecordingPool(connection)))
    until = datetime(2026, 5, 31, 0, 0, tzinfo=UTC)

    await store.read_trades(limit=10, until=until)

    query, args = connection.fetch_calls[0]
    assert "fills.ts < $1" in query
    assert "fills.ts <= $1" not in query
    assert args == (until, 10, 0)
