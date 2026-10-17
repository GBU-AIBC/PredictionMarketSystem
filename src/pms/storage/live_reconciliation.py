from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import math
from typing import Literal
from typing import Any, cast

import asyncpg

from pms.core.validation import validate_finite, validate_positive, validate_price


_CREATE_FILL_PAYLOADS_TABLE = """
CREATE TABLE IF NOT EXISTS fill_payloads (
    fill_id TEXT PRIMARY KEY REFERENCES fills(fill_id) ON DELETE CASCADE,
    payload JSONB NOT NULL
)
"""


SubmissionUnknownResolutionStatus = Literal["filled", "not_found", "open"]
_VENUE_ORDER_ID_REQUIRED_STATUSES: frozenset[SubmissionUnknownResolutionStatus] = frozenset(
    ("filled", "open")
)


def normalize_submission_unknown_decision_id(decision_id: str) -> str:
    normalized = decision_id.strip()
    if normalized == "":
        raise ValueError("decision_id is required")
    if _looks_like_placeholder(normalized):
        raise ValueError("decision_id must not contain a placeholder")
    return normalized


def normalize_submission_unknown_venue_order_id(
    *,
    status: SubmissionUnknownResolutionStatus,
    venue_order_id: str | None,
) -> str | None:
    normalized = venue_order_id.strip() if venue_order_id is not None else None
    if normalized == "":
        normalized = None
    if status in _VENUE_ORDER_ID_REQUIRED_STATUSES and normalized is None:
        raise ValueError("venue_order_id is required when status is filled or open")
    if normalized is not None and _looks_like_placeholder(normalized):
        raise ValueError("venue_order_id must not contain a placeholder")
    return normalized


def normalize_submission_unknown_reconciled_by(reconciled_by: str) -> str:
    normalized = reconciled_by.strip()
    if normalized == "":
        raise ValueError("reconciled_by is required")
    if _looks_like_placeholder(normalized):
        raise ValueError("reconciled_by must not contain a placeholder")
    if any(character in normalized for character in ("|", "\n", "\r")):
        raise ValueError("reconciled_by must not contain delimiters or newlines")
    return normalized


def _looks_like_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "":
        return False
    placeholder_markers = (
        "fill_in",
        "__fill",
        "<",
        ">",
        "todo",
        "replace",
        "placeholder",
    )
    return any(marker in normalized for marker in placeholder_markers)


@dataclass(frozen=True, slots=True)
class SubmissionUnknownReconciliationStore:
    pool: asyncpg.Pool

    async def reconcile_submission_unknown(
        self,
        *,
        decision_id: str,
        venue_order_id: str | None,
        status: SubmissionUnknownResolutionStatus,
        reconciled_by: str,
        note: str | None = None,
        fill_price: float | None = None,
        fill_notional_usdc: float | None = None,
        fill_quantity: float | None = None,
        filled_at: datetime | None = None,
        fees: float | None = None,
        fee_bps: int | None = None,
    ) -> bool:
        normalized_decision_id = normalize_submission_unknown_decision_id(decision_id)
        normalized_venue_order_id = normalize_submission_unknown_venue_order_id(
            status=status,
            venue_order_id=venue_order_id,
        )
        normalized_reconciled_by = normalize_submission_unknown_reconciled_by(
            reconciled_by
        )
        has_fill_evidence = any(
            value is not None
            for value in (
                fill_price,
                fill_notional_usdc,
                fill_quantity,
                filled_at,
                fees,
                fee_bps,
            )
        )
        if status == "filled" and not has_fill_evidence and isinstance(
            self.pool,
            asyncpg.Pool,
        ):
            raise ValueError(
                "filled reconciliation requires fill_price, "
                "fill_notional_usdc, and fill_quantity"
            )
        if status == "filled" and has_fill_evidence:
            if fill_price is None or fill_notional_usdc is None or fill_quantity is None:
                raise ValueError(
                    "filled reconciliation requires fill_price, "
                    "fill_notional_usdc, and fill_quantity"
                )
            fill_price = validate_price(fill_price, field_name="fill_price")
            fill_notional_usdc = validate_positive(
                fill_notional_usdc,
                field_name="fill_notional_usdc",
            )
            fill_quantity = validate_positive(
                fill_quantity,
                field_name="fill_quantity",
            )
            if fees is not None:
                fees = validate_finite(fees, field_name="fees")
                if fees < 0.0:
                    raise ValueError("fees must be >= 0")
            if fee_bps is not None:
                fee_bps = int(validate_finite(fee_bps, field_name="fee_bps"))
                if fee_bps < 0:
                    raise ValueError("fee_bps must be >= 0")
            filled_at = filled_at or datetime.now(tz=UTC)
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    UPDATE order_intents
                    SET reconciled_at = now(),
                        venue_order_id = $2,
                        reconciliation_status = $3,
                        reconciled_by = $4,
                        reconciliation_note = $5
                    WHERE decision_id = $1
                      AND outcome = 'submission_unknown'
                      AND reconciled_at IS NULL
                      AND EXISTS (
                          SELECT 1
                          FROM decisions
                          WHERE decisions.decision_id = $1
                            AND decisions.status = 'submission_unknown'
                      )
                    RETURNING decision_id
                    """,
                    normalized_decision_id,
                    normalized_venue_order_id,
                    status,
                    normalized_reconciled_by,
                    note,
                )
                if row is None:
                    return False
                if status == "filled" and has_fill_evidence:
                    assert venue_order_id is not None
                    assert fill_price is not None
                    assert fill_notional_usdc is not None
                    assert fill_quantity is not None
                    await _insert_reconciled_fill(
                        connection,
                        decision_id=normalized_decision_id,
                        venue_order_id=venue_order_id,
                        fill_price=fill_price,
                        fill_notional_usdc=fill_notional_usdc,
                        fill_quantity=fill_quantity,
                        filled_at=filled_at or datetime.now(tz=UTC),
                        fees=fees,
                        fee_bps=fee_bps,
                    )
                decision_status = await connection.execute(
                    """
                    UPDATE decisions
                    SET status = 'reconciled',
                        updated_at = now()
                    WHERE decision_id = $1
                      AND status = 'submission_unknown'
                    """,
                    normalized_decision_id,
                )
                if _command_tag_row_count(decision_status) != 1:
                    msg = (
                        "submission_unknown decision row was not reconciled; "
                        "incident state rolled back"
                    )
                    raise RuntimeError(msg)
        return row is not None


@dataclass(frozen=True, slots=True)
class LiveOrderReconciliationRecord:
    decision_id: str
    decision_status: str
    order_id: str
    order_status: str
    order_raw_status: str
    market_id: str
    token_id: str
    venue: str
    strategy_id: str
    strategy_version_id: str
    requested_notional_usdc: float
    filled_notional_usdc: float
    remaining_notional_usdc: float
    filled_quantity: float
    fill_price: float
    submitted_at: datetime
    last_updated_at: datetime
    time_in_force: str
    action: str | None
    outcome: str | None
    intent_key: str | None
    pre_submit_quote_fingerprint: str
    pre_submit_quote_hash: str
    pre_submit_quote_source: str | None
    fill_id: str
    fill_status: str
    fill_notional_usdc: float
    fill_quantity: float
    filled_at: datetime

    def as_artifact_payload(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "decision_status": self.decision_status,
            "order": {
                "order_id": self.order_id,
                "status": self.order_status,
                "raw_status": self.order_raw_status,
                "market_id": self.market_id,
                "token_id": self.token_id,
                "venue": self.venue,
                "strategy_id": self.strategy_id,
                "strategy_version_id": self.strategy_version_id,
                "requested_notional_usdc": self.requested_notional_usdc,
                "filled_notional_usdc": self.filled_notional_usdc,
                "remaining_notional_usdc": self.remaining_notional_usdc,
                "filled_quantity": self.filled_quantity,
                "fill_price": self.fill_price,
                "submitted_at": self.submitted_at.isoformat(),
                "last_updated_at": self.last_updated_at.isoformat(),
                "time_in_force": self.time_in_force,
                "action": self.action,
                "outcome": self.outcome,
                "intent_key": self.intent_key,
                "pre_submit_quote_fingerprint": self.pre_submit_quote_fingerprint,
                "pre_submit_quote_hash": self.pre_submit_quote_hash,
                "pre_submit_quote_source": self.pre_submit_quote_source,
            },
            "fill": {
                "fill_id": self.fill_id,
                "status": self.fill_status,
                "fill_notional_usdc": self.fill_notional_usdc,
                "fill_quantity": self.fill_quantity,
                "filled_at": self.filled_at.isoformat(),
            },
        }


@dataclass(frozen=True, slots=True)
class LiveOrderReconciliationStore:
    pool: asyncpg.Pool

    async def load_live_order_record(
        self,
        *,
        decision_id: str,
    ) -> LiveOrderReconciliationRecord | None:
        normalized_decision_id = normalize_submission_unknown_decision_id(decision_id)
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT
                    decisions.decision_id,
                    decisions.status AS decision_status,
                    orders.order_id,
                    orders.status AS order_status,
                    orders.raw_status AS order_raw_status,
                    orders.market_id,
                    orders.token_id,
                    orders.venue,
                    orders.strategy_id,
                    orders.strategy_version_id,
                    orders.requested_notional_usdc,
                    orders.filled_notional_usdc,
                    orders.remaining_notional_usdc,
                    orders.filled_quantity,
                    orders.ts AS submitted_at,
                    orders.time_in_force,
                    orders.action,
                    orders.outcome,
                    orders.intent_key,
                    orders.pre_submit_quote_json,
                    order_payloads.payload AS order_payload,
                    fills.fill_id,
                    fills.fill_notional_usdc AS fill_notional_usdc,
                    fills.fill_quantity AS fill_quantity,
                    fills.ts AS filled_at,
                    fill_payloads.payload AS fill_payload
                FROM decisions
                INNER JOIN order_payloads
                    ON order_payloads.payload->>'decision_id' = decisions.decision_id
                INNER JOIN orders
                    ON orders.order_id = order_payloads.order_id
                INNER JOIN fills
                    ON fills.order_id = orders.order_id
                INNER JOIN fill_payloads
                    ON fill_payloads.fill_id = fills.fill_id
                   AND fill_payloads.payload->>'decision_id' = decisions.decision_id
                WHERE decisions.decision_id = $1
                ORDER BY fills.ts DESC, fills.fill_id DESC
                LIMIT 1
                """,
                normalized_decision_id,
            )
        if row is None:
            return None
        return _live_order_record_from_row(row)


def _command_tag_row_count(status: str) -> int | None:
    parts = status.split()
    if not parts:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


async def _insert_reconciled_fill(
    connection: asyncpg.Connection,
    *,
    decision_id: str,
    venue_order_id: str,
    fill_price: float,
    fill_notional_usdc: float,
    fill_quantity: float,
    filled_at: datetime,
    fees: float | None,
    fee_bps: int | None,
) -> None:
    """Insert the venue-confirmed fill exactly once inside reconciliation."""
    await connection.execute(_CREATE_FILL_PAYLOADS_TABLE)
    fill_id = f"reconciled:{decision_id}:{venue_order_id}"
    await connection.execute(
        """
        INSERT INTO fills (
            fill_id, order_id, market_id, ts, fill_notional_usdc,
            fill_quantity, strategy_id, strategy_version_id
        )
        SELECT
            $1::text,
            $2::text,
            decision_payloads.payload->>'market_id',
            $3::timestamptz,
            $4::double precision,
            $5::double precision,
            decisions.strategy_id,
            decisions.strategy_version_id
        FROM decisions
        LEFT JOIN decision_payloads
            ON decision_payloads.decision_id = decisions.decision_id
        WHERE decisions.decision_id = $6::text
        ON CONFLICT (fill_id) DO NOTHING
        """,
        fill_id,
        venue_order_id,
        filled_at,
        fill_notional_usdc,
        fill_quantity,
        decision_id,
    )
    await connection.execute(
        """
        INSERT INTO fill_payloads (fill_id, payload)
        SELECT
            $1::text,
            jsonb_build_object(
                'trade_id', $1::text,
                'decision_id', $2::text,
                'token_id', decision_payloads.payload->>'token_id',
                'venue', COALESCE(decision_payloads.payload->>'venue', 'polymarket'),
                'side', COALESCE(
                    decision_payloads.payload->>'action',
                    decision_payloads.payload->>'side',
                    'BUY'
                ),
                'fill_price', $3::double precision,
                'executed_at', $4::text,
                'status', 'filled',
                'anomaly_flags', jsonb_build_array('reconciled_from_submission_unknown'),
                'fee_bps', $5::integer,
                'fees', $6::double precision,
                'liquidity_side', NULL,
                'transaction_ref', $7::text,
                'resolved_outcome', NULL,
                'risk_group_id', decision_payloads.payload->'risk_group_id'
            )
        FROM decision_payloads
        WHERE decision_payloads.decision_id = $2::text
        ON CONFLICT (fill_id) DO NOTHING
        """,
        fill_id,
        decision_id,
        fill_price,
        filled_at.isoformat(),
        fee_bps,
        fees,
        venue_order_id,
    )


def _live_order_record_from_row(row: asyncpg.Record) -> LiveOrderReconciliationRecord:
    order_payload = _json_object(_row_value(row, "order_payload"))
    fill_payload = _json_object(_row_value(row, "fill_payload"))
    pre_submit_quote = _json_object(_row_value(row, "pre_submit_quote_json"))
    fill_price = _required_positive_price(order_payload.get("fill_price"), "fill_price")
    requested_notional = _required_positive_finite(
        _row_value(row, "requested_notional_usdc"),
        "requested_notional_usdc",
    )
    filled_notional = _required_positive_finite(
        _row_value(row, "filled_notional_usdc"),
        "filled_notional_usdc",
    )
    filled_quantity = _required_positive_finite(
        _row_value(row, "filled_quantity"),
        "filled_quantity",
    )
    fill_notional = _required_positive_finite(
        _row_value(row, "fill_notional_usdc"),
        "fill_notional_usdc",
    )
    fill_quantity = _required_positive_finite(
        _row_value(row, "fill_quantity"),
        "fill_quantity",
    )
    if not math.isclose(fill_notional, filled_notional, abs_tol=1e-9):
        msg = "live order reconciliation fill_notional_usdc does not match order"
        raise RuntimeError(msg)
    if not math.isclose(fill_quantity, filled_quantity, abs_tol=1e-9):
        msg = "live order reconciliation fill_quantity does not match order"
        raise RuntimeError(msg)
    quote_hash = _required_nonempty_text(
        pre_submit_quote.get("quote_hash"),
        "pre_submit_quote.quote_hash",
    )
    token_id = _required_nonempty_text(_row_value(row, "token_id"), "token_id")
    venue = _required_nonempty_text(_row_value(row, "venue"), "venue")
    if venue != "polymarket":
        msg = f"live order reconciliation only supports polymarket; got {venue!r}"
        raise RuntimeError(msg)
    time_in_force = _required_nonempty_text(
        _row_value(row, "time_in_force"),
        "time_in_force",
    )
    if time_in_force not in {"IOC", "FOK"}:
        msg = "live order reconciliation requires IOC/FOK order evidence"
        raise RuntimeError(msg)
    return LiveOrderReconciliationRecord(
        decision_id=_required_nonempty_text(_row_value(row, "decision_id"), "decision_id"),
        decision_status=_required_nonempty_text(
            _row_value(row, "decision_status"),
            "decision_status",
        ),
        order_id=_required_nonempty_text(_row_value(row, "order_id"), "order_id"),
        order_status=_required_nonempty_text(
            _row_value(row, "order_status"),
            "order_status",
        ),
        order_raw_status=_required_nonempty_text(
            _row_value(row, "order_raw_status"),
            "order_raw_status",
        ),
        market_id=_required_nonempty_text(_row_value(row, "market_id"), "market_id"),
        token_id=token_id,
        venue=venue,
        strategy_id=_required_nonempty_text(_row_value(row, "strategy_id"), "strategy_id"),
        strategy_version_id=_required_nonempty_text(
            _row_value(row, "strategy_version_id"),
            "strategy_version_id",
        ),
        requested_notional_usdc=requested_notional,
        filled_notional_usdc=filled_notional,
        remaining_notional_usdc=_required_nonnegative_finite(
            _row_value(row, "remaining_notional_usdc"),
            "remaining_notional_usdc",
        ),
        filled_quantity=filled_quantity,
        fill_price=fill_price,
        submitted_at=_required_datetime(_row_value(row, "submitted_at"), "submitted_at"),
        last_updated_at=_required_datetime(
            order_payload.get("last_updated_at"),
            "last_updated_at",
        ),
        time_in_force=time_in_force,
        action=_optional_nonempty_text(_row_value(row, "action"), "action"),
        outcome=_optional_nonempty_text(_row_value(row, "outcome"), "outcome"),
        intent_key=_optional_nonempty_text(_row_value(row, "intent_key"), "intent_key"),
        pre_submit_quote_fingerprint=_canonical_json_fingerprint(pre_submit_quote),
        pre_submit_quote_hash=quote_hash,
        pre_submit_quote_source=_optional_nonempty_text(
            pre_submit_quote.get("source"),
            "pre_submit_quote.source",
        ),
        fill_id=_required_nonempty_text(_row_value(row, "fill_id"), "fill_id"),
        fill_status=_required_nonempty_text(fill_payload.get("status"), "fill_status"),
        fill_notional_usdc=fill_notional,
        fill_quantity=fill_quantity,
        filled_at=_required_datetime(_row_value(row, "filled_at"), "filled_at"),
    )


def _row_value(row: asyncpg.Record, key: str) -> object:
    return cast(object, row[key])


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    if isinstance(value, str):
        loaded = json.loads(value, object_pairs_hook=_reject_duplicate_json_keys)
        if isinstance(loaded, dict):
            return cast(dict[str, Any], loaded)
    msg = f"live order reconciliation JSON field is not an object: {value!r}"
    raise RuntimeError(msg)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen: set[str] = set()
    result: dict[str, object] = {}
    for key, pair_value in pairs:
        if key in seen:
            msg = f"live order reconciliation duplicate JSON key: {key}"
            raise RuntimeError(msg)
        seen.add(key)
        result[key] = pair_value
    return result


def _required_nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        msg = f"live order reconciliation field {field_name} is required"
        raise RuntimeError(msg)
    normalized = value.strip()
    if normalized == "":
        msg = f"live order reconciliation field {field_name} is required"
        raise RuntimeError(msg)
    if _looks_like_placeholder(normalized):
        msg = f"live order reconciliation field {field_name} contains a placeholder"
        raise RuntimeError(msg)
    return normalized


def _optional_nonempty_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_nonempty_text(value, field_name)


def _required_positive_finite(value: object, field_name: str) -> float:
    number = _float_value(value, field_name)
    if number <= 0.0:
        msg = f"live order reconciliation field {field_name} must be > 0"
        raise RuntimeError(msg)
    return number


def _required_nonnegative_finite(value: object, field_name: str) -> float:
    number = _float_value(value, field_name)
    if number < 0.0:
        msg = f"live order reconciliation field {field_name} must be >= 0"
        raise RuntimeError(msg)
    return number


def _required_positive_price(value: object, field_name: str) -> float:
    number = _required_positive_finite(value, field_name)
    if number > 1.0:
        msg = f"live order reconciliation field {field_name} must be <= 1"
        raise RuntimeError(msg)
    return number


def _float_value(value: object, field_name: str) -> float:
    if isinstance(value, (int, float, str)):
        number = float(value)
        if math.isfinite(number):
            return number
    msg = f"live order reconciliation field {field_name} must be finite"
    raise RuntimeError(msg)


def _required_datetime(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    msg = f"live order reconciliation field {field_name} must be a timestamp"
    raise RuntimeError(msg)


def _canonical_json_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()
