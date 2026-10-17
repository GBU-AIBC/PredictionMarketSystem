from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import json
import math
from typing import Any, cast

import asyncpg

from pms.core.models import FillRecord, Position, Venue


_CREATE_FILL_PAYLOADS_TABLE = """
CREATE TABLE IF NOT EXISTS fill_payloads (
    fill_id TEXT PRIMARY KEY REFERENCES fills(fill_id) ON DELETE CASCADE,
    payload JSONB NOT NULL
)
"""


@dataclass
class FillStore:
    pool: asyncpg.Pool | None = None

    def bind_pool(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def insert(self, fill: FillRecord) -> None:
        fill_id = fill.fill_id or fill.trade_id
        async with self._pool().acquire() as connection:
            await _ensure_fill_payloads_table(connection)
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO fills (
                        fill_id,
                        order_id,
                        market_id,
                        ts,
                        fill_notional_usdc,
                        fill_quantity,
                        strategy_id,
                        strategy_version_id
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8
                    )
                    ON CONFLICT (fill_id) DO UPDATE
                    SET order_id = EXCLUDED.order_id,
                        market_id = EXCLUDED.market_id,
                        ts = EXCLUDED.ts,
                        fill_notional_usdc = EXCLUDED.fill_notional_usdc,
                        fill_quantity = EXCLUDED.fill_quantity,
                        strategy_id = EXCLUDED.strategy_id,
                        strategy_version_id = EXCLUDED.strategy_version_id
                    """,
                    fill_id,
                    fill.order_id,
                    fill.market_id,
                    fill.filled_at,
                    fill.fill_notional_usdc,
                    fill.fill_quantity,
                    fill.strategy_id,
                    fill.strategy_version_id,
                )
                await connection.execute(
                    """
                    INSERT INTO fill_payloads (fill_id, payload)
                    VALUES ($1, $2::jsonb)
                    ON CONFLICT (fill_id) DO UPDATE
                    SET payload = EXCLUDED.payload
                    """,
                    fill_id,
                    json.dumps(_fill_payload(fill)),
                )

    async def get(self, fill_id: str | None) -> FillRecord | None:
        if self.pool is None or fill_id is None:
            return None

        async with self.pool.acquire() as connection:
            await _ensure_fill_payloads_table(connection)
            row = await connection.fetchrow(
                """
                SELECT
                    fills.fill_id,
                    fills.order_id,
                    fills.market_id,
                    fills.ts,
                    fills.fill_notional_usdc,
                    fills.fill_quantity,
                    fills.strategy_id,
                    fills.strategy_version_id,
                    fill_payloads.payload
                FROM fills
                LEFT JOIN fill_payloads
                    ON fill_payloads.fill_id = fills.fill_id
                WHERE fills.fill_id = $1
                """,
                fill_id,
            )
        if row is None or row["payload"] is None:
            return None
        return _fill_from_row(row)

    async def read_unresolved_fills(self) -> list[FillRecord]:
        async with self._pool().acquire() as connection:
            await _ensure_fill_payloads_table(connection)
            rows = await connection.fetch(
                """
                SELECT
                    fills.fill_id,
                    fills.order_id,
                    fills.market_id,
                    fills.ts,
                    fills.fill_notional_usdc,
                    fills.fill_quantity,
                    fills.strategy_id,
                    fills.strategy_version_id,
                    fill_payloads.payload
                FROM fills
                INNER JOIN fill_payloads
                    ON fill_payloads.fill_id = fills.fill_id
                WHERE fill_payloads.payload->>'resolved_outcome' IS NULL
                ORDER BY fills.ts ASC, fills.fill_id ASC
                """
            )
        return [_fill_from_row(row) for row in rows]

    async def resolve_fill(
        self,
        fill_id: str,
        *,
        resolved_outcome: float,
    ) -> bool:
        async with self._pool().acquire() as connection:
            await _ensure_fill_payloads_table(connection)
            # The NULL guard makes the update idempotent: a second sweep (or
            # a concurrent writer) observing the same resolution matches zero
            # rows, so callers can gate eval-record appends on the returned
            # flag without double-scoring.
            row = await connection.fetchrow(
                """
                UPDATE fill_payloads
                SET payload = jsonb_set(
                    payload,
                    '{resolved_outcome}',
                    to_jsonb($2::double precision)
                )
                WHERE fill_id = $1
                  AND payload->>'resolved_outcome' IS NULL
                RETURNING fill_id
                """,
                fill_id,
                resolved_outcome,
            )
        return row is not None

    async def reconcile_fee(
        self,
        fill_id: str,
        *,
        fees: float | None,
        fee_bps: int | None = None,
        slippage_bps: float | None = None,
        reconciled_by: str | None = None,
        evidence: str | None = None,
    ) -> bool:
        """Record verified venue fee telemetry for an unresolved fill once."""
        if not fill_id.strip():
            raise ValueError("fill_id is required")
        if fees is not None and (not math.isfinite(fees) or fees < 0.0):
            raise ValueError("fees must be finite and non-negative")
        if fee_bps is not None and (
            isinstance(fee_bps, bool)
            or not isinstance(fee_bps, int)
            or fee_bps < 0
        ):
            raise ValueError("fee_bps must be a non-negative integer")
        if slippage_bps is not None and not math.isfinite(slippage_bps):
            raise ValueError("slippage_bps must be finite")
        if (reconciled_by is None) != (evidence is None):
            raise ValueError("reconciled_by and evidence must be provided together")
        if reconciled_by is not None and not reconciled_by.strip():
            raise ValueError("reconciled_by is required")
        if evidence is not None and not evidence.strip():
            raise ValueError("evidence is required")
        async with self._pool().acquire() as connection:
            await _ensure_fill_payloads_table(connection)
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT payload
                    FROM fill_payloads
                    WHERE fill_id = $1
                    FOR UPDATE
                    """,
                    fill_id,
                )
                if row is None:
                    return False
                payload = _json_object(row["payload"])
                existing_fees = payload.get("fees")
                if existing_fees is not None and slippage_bps is None:
                    return False
                if existing_fees is None and fees is None:
                    raise ValueError("fees are required for an unresolved fee")
                if fees is not None:
                    payload["fees"] = fees
                    payload["fee_bps"] = fee_bps
                if reconciled_by is not None and evidence is not None:
                    payload["fee_reconciled_by"] = reconciled_by.strip()
                    payload["fee_reconcile_evidence"] = evidence.strip()
                flags = [
                    flag
                    for flag in _string_list(payload.get("anomaly_flags"))
                    if flag != "fee_unreconciled"
                    and not flag.startswith("fee_reconciled")
                    and (
                        slippage_bps is None
                        or not flag.startswith("execution_slippage_bps=")
                    )
                ]
                flags.append(
                    "fee_reconciled" if existing_fees is None else "slippage_reconciled"
                )
                if slippage_bps is not None:
                    flags.append(f"execution_slippage_bps={slippage_bps:.9f}")
                payload["anomaly_flags"] = flags
                await connection.execute(
                    """
                    UPDATE fill_payloads
                    SET payload = $2::jsonb
                    WHERE fill_id = $1
                    """,
                    fill_id,
                    json.dumps(payload),
                )
        return True

    async def read_positions(self) -> list[Position]:
        async with self._pool().acquire() as connection:
            await _ensure_fill_payloads_table(connection)
            rows = await connection.fetch(
                """
                SELECT
                    fills.fill_id,
                    fills.market_id,
                    fills.ts,
                    fills.fill_notional_usdc,
                    fills.fill_quantity,
                    fills.strategy_id,
                    fills.strategy_version_id,
                    fill_payloads.payload->>'token_id' AS token_id,
                    fill_payloads.payload->>'venue' AS venue,
                    fill_payloads.payload->>'side' AS side,
                    fill_payloads.payload->>'risk_group_id' AS risk_group_id,
                    (fill_payloads.payload->>'fill_price')::double precision
                        AS fill_price,
                    COALESCE(
                        clob_marks.best_bid,
                        CASE
                            WHEN tokens.outcome = 'YES' THEN COALESCE(
                                markets.yes_price::double precision,
                                CASE
                                    WHEN markets.no_price IS NULL THEN NULL
                                    ELSE (1 - markets.no_price)::double precision
                                END
                            )
                            WHEN tokens.outcome = 'NO' THEN COALESCE(
                                markets.no_price::double precision,
                                CASE
                                    WHEN markets.yes_price IS NULL THEN NULL
                                    ELSE (1 - markets.yes_price)::double precision
                                END
                            )
                            ELSE markets.yes_price::double precision
                        END
                    ) AS current_price,
                    CASE
                        WHEN clob_marks.best_bid IS NOT NULL THEN 'clob'
                        ELSE 'gamma'
                    END AS mark_source,
                    CASE
                        WHEN clob_marks.snapshot_ts IS NOT NULL
                        THEN GREATEST(
                            EXTRACT(EPOCH FROM NOW() - clob_marks.snapshot_ts),
                            0.0
                        )
                        ELSE NULL
                    END AS mark_age_seconds
                FROM fills
                INNER JOIN fill_payloads
                    ON fill_payloads.fill_id = fills.fill_id
                LEFT JOIN LATERAL (
                    SELECT
                        MAX(book_levels.price) AS best_bid,
                        latest_snapshot.snapshot_ts
                    FROM (
                        SELECT
                            book_snapshots.id,
                            book_snapshots.ts AS snapshot_ts
                        FROM book_snapshots
                        WHERE book_snapshots.market_id = fills.market_id
                          AND book_snapshots.token_id = fill_payloads.payload->>'token_id'
                          AND book_snapshots.ts > NOW() - INTERVAL '60 seconds'
                        ORDER BY book_snapshots.ts DESC, book_snapshots.id DESC
                        LIMIT 1
                    ) AS latest_snapshot
                    INNER JOIN book_levels
                       ON book_levels.snapshot_id = latest_snapshot.id
                      AND book_levels.market_id = fills.market_id
                      AND book_levels.side = 'BUY'
                    GROUP BY latest_snapshot.snapshot_ts
                ) AS clob_marks ON TRUE
                LEFT JOIN tokens
                    ON tokens.token_id = fill_payloads.payload->>'token_id'
                LEFT JOIN markets
                    ON markets.condition_id = fills.market_id
                ORDER BY
                    fills.market_id ASC,
                    fill_payloads.payload->>'token_id' ASC,
                    fill_payloads.payload->>'venue' ASC,
                    fills.strategy_id ASC,
                    fills.strategy_version_id ASC,
                    fills.ts ASC,
                    fills.fill_id ASC
                """
            )
        return _positions_from_fill_rows(rows)

    async def read_trades(
        self,
        *,
        limit: int,
        offset: int = 0,
        until: datetime | None = None,
    ) -> list["StoredTradeRow"]:
        async with self._pool().acquire() as connection:
            await _ensure_fill_payloads_table(connection)
            rows = await connection.fetch(
                """
                SELECT
                    fills.fill_id,
                    fills.order_id,
                    fills.market_id,
                    fills.ts,
                    fills.fill_notional_usdc,
                    fills.fill_quantity,
                    fills.strategy_id,
                    fills.strategy_version_id,
                    fill_payloads.payload,
                    COALESCE(markets.question, fills.market_id) AS question
                FROM fills
                LEFT JOIN fill_payloads
                    ON fill_payloads.fill_id = fills.fill_id
                LEFT JOIN markets
                    ON markets.condition_id = fills.market_id
                WHERE ($1::timestamptz IS NULL OR fills.ts < $1)
                ORDER BY
                    fills.ts ASC,
                    fill_payloads.payload->>'executed_at' ASC NULLS LAST,
                    fills.order_id ASC,
                    fills.fill_id ASC
                LIMIT $2
                OFFSET $3
                """,
                until,
                limit,
                offset,
            )
        return [_trade_from_row(row) for row in rows]

    async def count_trades(self) -> int:
        """Return the complete durable fill count for replay truncation checks."""
        async with self._pool().acquire() as connection:
            row = await connection.fetchrow("SELECT COUNT(*) AS fill_count FROM fills")
        return int(row["fill_count"])

    def _pool(self) -> asyncpg.Pool:
        if self.pool is None:
            msg = "FillStore pool is not bound"
            raise RuntimeError(msg)
        return self.pool


@dataclass(frozen=True)
class StoredTradeRow:
    trade_id: str
    fill_id: str
    order_id: str
    decision_id: str
    market_id: str
    question: str
    token_id: str | None
    venue: Venue
    side: str
    fill_price: float
    fill_notional_usdc: float
    fill_quantity: float
    executed_at: datetime
    filled_at: datetime
    status: str
    strategy_id: str
    strategy_version_id: str
    fee_bps: int | None = None
    fees: float | None = None
    slippage_bps: float | None = None
    risk_group_id: str | None = None


class _PositionAccumulator:
    def __init__(
        self,
        *,
        market_id: str,
        token_id: str | None,
        venue: str,
        risk_group_id: str | None,
        strategy_id: str,
        strategy_version_id: str,
    ) -> None:
        self.market_id = market_id
        self.token_id = token_id
        self.venue = venue
        self.risk_group_id = risk_group_id
        self.strategy_id = strategy_id
        self.strategy_version_id = strategy_version_id
        self.side: str | None = None
        self.shares_held = Decimal("0")
        self.locked_usdc = Decimal("0")
        self.opened_at: datetime | None = None
        self.last_fill_at: datetime | None = None
        self.current_price: float | None = None
        self.mark_source: str | None = None
        self.mark_age_seconds: float | None = None


async def _ensure_fill_payloads_table(connection: asyncpg.Connection) -> None:
    # The current branch still uses shell rows in `fills`; the sidecar preserves
    # the full runtime object until a later schema checkpoint widens the table.
    await connection.execute(_CREATE_FILL_PAYLOADS_TABLE)


def _fill_payload(fill: FillRecord) -> dict[str, object]:
    return {
        "trade_id": fill.trade_id,
        "decision_id": fill.decision_id,
        "token_id": fill.token_id,
        "venue": fill.venue,
        "side": fill.side,
        "fill_price": fill.fill_price,
        "executed_at": fill.executed_at.isoformat(),
        "status": fill.status,
        "anomaly_flags": list(fill.anomaly_flags),
        "fee_bps": fill.fee_bps,
        "fees": fill.fees,
        "liquidity_side": fill.liquidity_side,
        "transaction_ref": fill.transaction_ref,
        "resolved_outcome": fill.resolved_outcome,
        "risk_group_id": fill.risk_group_id,
    }


def _fill_from_row(row: asyncpg.Record) -> FillRecord:
    payload = _json_object(row["payload"])
    return FillRecord(
        trade_id=cast(str, payload["trade_id"]),
        fill_id=cast(str, row["fill_id"]),
        order_id=cast(str, row["order_id"]),
        decision_id=cast(str, payload["decision_id"]),
        market_id=cast(str, row["market_id"]),
        token_id=cast(str | None, payload.get("token_id")),
        venue=cast(Venue, payload["venue"]),
        side=cast(str, payload["side"]),
        fill_price=cast(float, payload["fill_price"]),
        fill_notional_usdc=cast(float, row["fill_notional_usdc"]),
        fill_quantity=cast(float, row["fill_quantity"]),
        executed_at=datetime.fromisoformat(cast(str, payload["executed_at"])),
        filled_at=cast(datetime, row["ts"]),
        status=cast(str, payload["status"]),
        anomaly_flags=_string_list(payload.get("anomaly_flags")),
        strategy_id=cast(str, row["strategy_id"]),
        strategy_version_id=cast(str, row["strategy_version_id"]),
        fee_bps=cast(int | None, payload.get("fee_bps")),
        fees=cast(float | None, payload.get("fees")),
        liquidity_side=cast(str | None, payload.get("liquidity_side")),
        transaction_ref=cast(str | None, payload.get("transaction_ref")),
        resolved_outcome=cast(float | None, payload.get("resolved_outcome")),
        risk_group_id=cast(str | None, payload.get("risk_group_id")),
    )


def _positions_from_fill_rows(rows: Sequence[asyncpg.Record]) -> list[Position]:
    accumulators: dict[
        tuple[str, str | None, str, str, str],
        _PositionAccumulator,
    ] = {}
    for row in rows:
        key = (
            cast(str, row["market_id"]),
            cast(str | None, row["token_id"]),
            cast(str, row["venue"]),
            cast(str, _optional_row_value(row, "strategy_id") or "default"),
            cast(str, _optional_row_value(row, "strategy_version_id") or "default-v1"),
        )
        accumulator = accumulators.get(key)
        if accumulator is None:
            accumulator = _PositionAccumulator(
                market_id=key[0],
                token_id=key[1],
                venue=key[2],
                risk_group_id=cast(
                    str | None, _optional_row_value(row, "risk_group_id")
                ),
                strategy_id=key[3],
                strategy_version_id=key[4],
            )
            accumulators[key] = accumulator
        _apply_fill_row(accumulator, row)
    return [
        _position_from_accumulator(accumulator)
        for accumulator in accumulators.values()
        if accumulator.shares_held > Decimal("1e-9") and accumulator.side is not None
    ]


def _apply_fill_row(accumulator: _PositionAccumulator, row: asyncpg.Record) -> None:
    fill_quantity = _decimal_or_none(row["fill_quantity"])
    if fill_quantity is None or fill_quantity <= 0:
        return
    fill_notional = _decimal_or_none(row["fill_notional_usdc"])
    if fill_notional is None:
        fill_notional = Decimal("0")
    fill_price = _decimal_or_none(_optional_row_value(row, "fill_price"))
    if fill_price is None:
        fill_price = (
            Decimal("0") if fill_quantity == 0 else fill_notional / fill_quantity
        )
    fill_side = str(row["side"])
    filled_at = cast(datetime, row["ts"])
    accumulator.current_price = _float_or_none(_optional_row_value(row, "current_price"))
    accumulator.mark_source = cast(str | None, _optional_row_value(row, "mark_source"))
    accumulator.mark_age_seconds = _float_or_none(
        _optional_row_value(row, "mark_age_seconds")
    )
    accumulator.risk_group_id = cast(
        str | None, _optional_row_value(row, "risk_group_id")
    )
    accumulator.last_fill_at = filled_at

    if accumulator.shares_held <= Decimal("1e-9") or accumulator.side is None:
        _open_accumulator(
            accumulator,
            side=fill_side,
            quantity=fill_quantity,
            notional=fill_notional,
            filled_at=filled_at,
        )
        return

    if _same_fill_side(accumulator.side, fill_side):
        accumulator.locked_usdc += fill_notional
        accumulator.shares_held += fill_quantity
        return

    avg_entry_price = accumulator.locked_usdc / accumulator.shares_held
    closing_quantity = min(accumulator.shares_held, fill_quantity)
    accumulator.shares_held -= closing_quantity
    accumulator.locked_usdc = max(
        Decimal("0"),
        accumulator.locked_usdc - (avg_entry_price * closing_quantity),
    )
    residual_quantity = fill_quantity - closing_quantity
    if accumulator.shares_held <= Decimal("1e-9"):
        accumulator.side = None
        accumulator.shares_held = Decimal("0")
        accumulator.locked_usdc = Decimal("0")
        accumulator.opened_at = None
    if residual_quantity > Decimal("1e-9"):
        _open_accumulator(
            accumulator,
            side=fill_side,
            quantity=residual_quantity,
            notional=fill_price * residual_quantity,
            filled_at=filled_at,
        )


def _open_accumulator(
    accumulator: _PositionAccumulator,
    *,
    side: str,
    quantity: Decimal,
    notional: Decimal,
    filled_at: datetime,
) -> None:
    accumulator.side = side
    accumulator.shares_held += quantity
    accumulator.locked_usdc += notional
    if accumulator.opened_at is None:
        accumulator.opened_at = filled_at


def _same_fill_side(left: str, right: str) -> bool:
    return left.upper() == right.upper()


def _position_from_accumulator(accumulator: _PositionAccumulator) -> Position:
    avg_entry_price = accumulator.locked_usdc / accumulator.shares_held
    return Position(
        market_id=accumulator.market_id,
        token_id=accumulator.token_id,
        venue=cast(Venue, accumulator.venue),
        side=cast(str, accumulator.side),
        shares_held=float(accumulator.shares_held),
        avg_entry_price=float(avg_entry_price),
        unrealized_pnl=_unrealized_pnl_from_values(
            side=cast(str, accumulator.side),
            shares_held=accumulator.shares_held,
            avg_entry_price=avg_entry_price,
            current_price=accumulator.current_price,
        ),
        locked_usdc=float(accumulator.locked_usdc),
        mark_source=accumulator.mark_source,
        mark_age_seconds=accumulator.mark_age_seconds,
        current_price=accumulator.current_price,
        opened_at=accumulator.opened_at,
        strategy_id=accumulator.strategy_id,
        strategy_version_id=accumulator.strategy_version_id,
        risk_group_id=accumulator.risk_group_id,
    )


def _position_from_row(row: asyncpg.Record) -> Position:
    return Position(
        market_id=cast(str, row["market_id"]),
        token_id=cast(str | None, row["token_id"]),
        venue=cast(Venue, row["venue"]),
        side=cast(str, row["side"]),
        shares_held=float(cast(float, row["shares_held"])),
        avg_entry_price=float(cast(float, row["avg_entry_price"])),
        unrealized_pnl=_unrealized_pnl_from_row(row),
        locked_usdc=float(cast(float, row["locked_usdc"])),
        mark_source=cast(str | None, _optional_row_value(row, "mark_source")),
        mark_age_seconds=_float_or_none(_optional_row_value(row, "mark_age_seconds")),
        current_price=_float_or_none(_optional_row_value(row, "current_price")),
        opened_at=cast(datetime | None, _optional_row_value(row, "opened_at")),
        strategy_id=cast(str, _optional_row_value(row, "strategy_id") or "default"),
        strategy_version_id=cast(
            str,
            _optional_row_value(row, "strategy_version_id") or "default-v1",
        ),
        risk_group_id=cast(str | None, _optional_row_value(row, "risk_group_id")),
    )


def _unrealized_pnl_from_row(row: asyncpg.Record) -> float:
    current_price = _decimal_or_none(_optional_row_value(row, "current_price"))
    if current_price is None:
        return 0.0

    shares_held = Decimal(str(row["shares_held"]))
    avg_entry_price = Decimal(str(row["avg_entry_price"]))
    return _unrealized_pnl_from_values(
        side=str(row["side"]),
        shares_held=shares_held,
        avg_entry_price=avg_entry_price,
        current_price=float(current_price),
    )


def _unrealized_pnl_from_values(
    *,
    side: str,
    shares_held: Decimal,
    avg_entry_price: Decimal,
    current_price: float | None,
) -> float:
    if current_price is None:
        return 0.0
    current_price_dec = Decimal(str(current_price))
    if side.upper() == "SELL":
        return float((avg_entry_price - current_price_dec) * shares_held)
    return float((current_price_dec - avg_entry_price) * shares_held)


def _optional_row_value(row: asyncpg.Record, key: str) -> object | None:
    try:
        return cast(object, row[key])
    except KeyError:
        return None


def _decimal_or_none(value: object | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _float_or_none(value: object | None) -> float | None:
    if value is None:
        return None
    return float(cast(float, value))


def _trade_from_row(row: asyncpg.Record) -> StoredTradeRow:
    fill_id = str(row["fill_id"])
    try:
        payload = _json_object(row["payload"])
    except (RuntimeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"fill {fill_id} has missing or malformed fill_payloads sidecar"
        ) from error
    return StoredTradeRow(
        trade_id=cast(str, payload["trade_id"]),
        fill_id=cast(str, row["fill_id"]),
        order_id=cast(str, row["order_id"]),
        decision_id=cast(str, payload["decision_id"]),
        market_id=cast(str, row["market_id"]),
        question=cast(str, row["question"]),
        token_id=cast(str | None, payload.get("token_id")),
        venue=cast(Venue, payload["venue"]),
        side=cast(str, payload["side"]),
        fill_price=cast(float, payload["fill_price"]),
        fill_notional_usdc=cast(float, row["fill_notional_usdc"]),
        fill_quantity=cast(float, row["fill_quantity"]),
        executed_at=datetime.fromisoformat(cast(str, payload["executed_at"])),
        filled_at=cast(datetime, row["ts"]),
        status=cast(str, payload["status"]),
        strategy_id=cast(str, row["strategy_id"]),
        strategy_version_id=cast(str, row["strategy_version_id"]),
        fee_bps=cast(int | None, payload.get("fee_bps")),
        fees=cast(float | None, payload.get("fees")),
        slippage_bps=_slippage_from_anomaly_flags(payload.get("anomaly_flags")),
        risk_group_id=cast(str | None, payload.get("risk_group_id")),
    )


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    if isinstance(value, str):
        loaded = json.loads(value)
        if isinstance(loaded, dict):
            return cast(dict[str, Any], loaded)
    msg = "fill payload must be a JSON object"
    raise RuntimeError(msg)


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _slippage_from_anomaly_flags(value: object) -> float | None:
    for flag in _string_list(value):
        prefix = "execution_slippage_bps="
        if not flag.startswith(prefix):
            continue
        try:
            return float(flag[len(prefix) :])
        except ValueError:
            return None
    return None
