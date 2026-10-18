from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
import json
from math import isfinite
from typing import Any, Final, Literal, cast

import asyncpg

from pms.core.enums import TimeInForce
from pms.core.models import Opportunity, TradeDecision


DecisionStatus = Literal[
    "pending",
    "accepted",
    "queued",
    "submitted",
    "partially_filled",
    "filled",
    "rejected",
    "venue_rejected",
    "cancelled",
    "expired",
    "submission_unknown",
    "reconciled",
]

DECISION_STATUSES: Final[tuple[DecisionStatus, ...]] = (
    "pending",
    "accepted",
    "queued",
    "submitted",
    "partially_filled",
    "filled",
    "rejected",
    "venue_rejected",
    "cancelled",
    "expired",
    "submission_unknown",
    "reconciled",
)
_VALID_STATUS_TRANSITIONS: Final[dict[DecisionStatus, frozenset[DecisionStatus]]] = {
    "pending": frozenset({"pending", "accepted", "rejected", "expired"}),
    "accepted": frozenset({"accepted", "queued", "submitted", "rejected"}),
    "queued": frozenset({"queued", "submitted", "rejected", "cancelled"}),
    "submitted": frozenset(
        {
            "submitted",
            "partially_filled",
            "filled",
            "rejected",
            "venue_rejected",
            "cancelled",
            "submission_unknown",
        }
    ),
    "partially_filled": frozenset(
        {
            "partially_filled",
            "filled",
            "cancelled",
            "submission_unknown",
        }
    ),
    "filled": frozenset({"filled", "reconciled"}),
    "rejected": frozenset({"rejected"}),
    "venue_rejected": frozenset({"venue_rejected"}),
    "cancelled": frozenset({"cancelled", "reconciled"}),
    "expired": frozenset({"expired"}),
    "submission_unknown": frozenset({"submission_unknown", "reconciled"}),
    "reconciled": frozenset({"reconciled"}),
}
_CREATE_DECISION_PAYLOADS_TABLE = """
CREATE TABLE IF NOT EXISTS decision_payloads (
    decision_id TEXT PRIMARY KEY REFERENCES decisions(decision_id) ON DELETE CASCADE,
    payload JSONB NOT NULL
)
"""
_DECISION_SELECT = """
SELECT
    decisions.decision_id,
    decisions.opportunity_id,
    decisions.strategy_id,
    decisions.strategy_version_id,
    decisions.status,
    decisions.factor_snapshot_hash,
    decisions.created_at,
    decisions.updated_at,
    decisions.expires_at,
    decision_payloads.payload,
    opportunities.opportunity_id AS opportunity_row_id,
    opportunities.market_id AS opportunity_market_id,
    opportunities.token_id AS opportunity_token_id,
    opportunities.side AS opportunity_side,
    opportunities.selected_factor_values,
    opportunities.expected_edge AS opportunity_expected_edge,
    opportunities.rationale,
    opportunities.target_size_usdc,
    opportunities.expiry AS opportunity_expiry,
    opportunities.staleness_policy,
    opportunities.strategy_id AS opportunity_strategy_id,
    opportunities.strategy_version_id AS opportunity_strategy_version_id,
    opportunities.created_at AS opportunity_created_at,
    opportunities.factor_snapshot_hash AS opportunity_factor_snapshot_hash,
    opportunities.composition_trace
FROM decisions
LEFT JOIN decision_payloads
    ON decision_payloads.decision_id = decisions.decision_id
LEFT JOIN opportunities
    ON opportunities.opportunity_id = decisions.opportunity_id
"""


@dataclass(frozen=True)
class DecisionPruneResult:
    decisions_deleted: int
    order_intents_deleted: int
    opportunities_deleted: int


@dataclass(frozen=True)
class StoredDecisionRow:
    decision: TradeDecision
    status: DecisionStatus
    factor_snapshot_hash: str | None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    decision_evidence: Mapping[str, Any] = field(default_factory=dict)
    opportunity: Opportunity | None = None


@dataclass
class DecisionStore:
    pool: asyncpg.Pool | None = None

    def bind_pool(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def insert(
        self,
        decision: TradeDecision,
        *,
        factor_snapshot_hash: str | None,
        created_at: datetime,
        expires_at: datetime,
        status: DecisionStatus = "pending",
        decision_evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if self.pool is None or not hasattr(self.pool, "acquire"):
            return

        validate_decision_status_transition("pending", status)
        async with self.pool.acquire() as connection:
            await _ensure_decision_payloads_table(connection)
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO decisions (
                        decision_id,
                        opportunity_id,
                        strategy_id,
                        strategy_version_id,
                        status,
                        factor_snapshot_hash,
                        created_at,
                        updated_at,
                        expires_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9
                    )
                    ON CONFLICT (decision_id) DO UPDATE
                    SET opportunity_id = EXCLUDED.opportunity_id,
                        strategy_id = EXCLUDED.strategy_id,
                        strategy_version_id = EXCLUDED.strategy_version_id,
                        status = EXCLUDED.status,
                        factor_snapshot_hash = EXCLUDED.factor_snapshot_hash,
                        updated_at = EXCLUDED.updated_at,
                        expires_at = EXCLUDED.expires_at
                    """,
                    decision.decision_id,
                    decision.opportunity_id,
                    decision.strategy_id,
                    decision.strategy_version_id,
                    status,
                    factor_snapshot_hash,
                    created_at,
                    created_at,
                    expires_at,
                )
                # Keep the full decision object durable without widening the
                # shell table ahead of the accept-flow checkpoint.
                await connection.execute(
                    """
                    INSERT INTO decision_payloads (decision_id, payload)
                    VALUES ($1, $2::jsonb)
                    ON CONFLICT (decision_id) DO UPDATE
                    SET payload = EXCLUDED.payload
                    """,
                    decision.decision_id,
                    json.dumps(
                        _decision_payload(
                            decision,
                            decision_evidence=decision_evidence,
                        ),
                        allow_nan=False,
                    ),
                )

    async def read_decisions(
        self,
        *,
        limit: int,
        offset: int = 0,
        status: str | None = None,
        include_opportunity: bool = False,
        until: datetime | None = None,
    ) -> Sequence[StoredDecisionRow]:
        if self.pool is None or not hasattr(self.pool, "acquire"):
            return []

        normalized_status = None if status is None else _coerce_decision_status(status)
        async with self.pool.acquire() as connection:
            await _ensure_decision_payloads_table(connection)
            rows = await connection.fetch(
                _DECISION_SELECT
                + """
                WHERE ($1::text IS NULL OR decisions.status = $1)
                  AND ($2::timestamptz IS NULL OR decisions.created_at <= $2)
                ORDER BY decisions.created_at DESC, decisions.decision_id DESC
                LIMIT $3
                OFFSET $4
                """,
                normalized_status,
                until,
                limit,
                offset,
            )
        return [
            _stored_decision_from_row(row, include_opportunity=include_opportunity)
            for row in rows
            if row["payload"] is not None
        ]

    async def get_decision(
        self,
        decision_id: str,
        *,
        include_opportunity: bool = False,
    ) -> StoredDecisionRow | None:
        if self.pool is None or not hasattr(self.pool, "acquire"):
            return None

        async with self.pool.acquire() as connection:
            await _ensure_decision_payloads_table(connection)
            row = await connection.fetchrow(
                _DECISION_SELECT
                + """
                WHERE decisions.decision_id = $1
                """,
                decision_id,
            )
        if row is None or row["payload"] is None:
            return None
        return _stored_decision_from_row(row, include_opportunity=include_opportunity)

    async def update_status(
        self,
        decision_id: str,
        *,
        current_status: str,
        next_status: str,
        updated_at: datetime,
    ) -> bool:
        if self.pool is None or not hasattr(self.pool, "acquire"):
            return False

        normalized_current = _coerce_decision_status(current_status)
        normalized_next = _coerce_decision_status(next_status)
        validate_decision_status_transition(normalized_current, normalized_next)
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                UPDATE decisions
                SET status = $3,
                    updated_at = $4
                WHERE decision_id = $1
                  AND status = $2
                RETURNING decision_id
                """,
                decision_id,
                normalized_current,
                normalized_next,
                updated_at,
            )
        return row is not None

    async def expire_pending(self, *, before: datetime) -> int:
        if self.pool is None or not hasattr(self.pool, "acquire"):
            return 0

        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                UPDATE decisions
                SET status = 'expired',
                    updated_at = $1
                WHERE status IN ('pending', 'accepted', 'queued')
                  AND expires_at < $1
                RETURNING decision_id
                """,
                before,
            )
        return len(rows)

    async def prune_expired(
        self,
        *,
        before: datetime,
        limit: int = 10_000,
    ) -> DecisionPruneResult:
        if self.pool is None or not hasattr(self.pool, "acquire"):
            return DecisionPruneResult(
                decisions_deleted=0,
                order_intents_deleted=0,
                opportunities_deleted=0,
            )
        if limit <= 0:
            return DecisionPruneResult(
                decisions_deleted=0,
                order_intents_deleted=0,
                opportunities_deleted=0,
            )

        async with self.pool.acquire() as connection:
            async with connection.transaction():
                rows = await connection.fetch(
                    """
                    SELECT decision_id, opportunity_id
                    FROM decisions
                    WHERE status = 'expired'
                      AND updated_at < $1
                    ORDER BY updated_at ASC, decision_id ASC
                    LIMIT $2
                    """,
                    before,
                    limit,
                )
                if not rows:
                    return DecisionPruneResult(
                        decisions_deleted=0,
                        order_intents_deleted=0,
                        opportunities_deleted=0,
                    )

                decision_ids = tuple(cast(str, row["decision_id"]) for row in rows)
                opportunity_ids = tuple(
                    sorted(
                        {
                            cast(str, row["opportunity_id"])
                            for row in rows
                            if row["opportunity_id"] is not None
                        }
                    )
                )
                intent_rows = await connection.fetch(
                    """
                    DELETE FROM order_intents
                    WHERE decision_id = ANY($1::text[])
                    RETURNING decision_id
                    """,
                    list(decision_ids),
                )
                decision_rows = await connection.fetch(
                    """
                    DELETE FROM decisions
                    WHERE decision_id = ANY($1::text[])
                    RETURNING decision_id
                    """,
                    list(decision_ids),
                )
                opportunity_rows: Sequence[asyncpg.Record] = []
                if opportunity_ids:
                    opportunity_rows = await connection.fetch(
                        """
                        DELETE FROM opportunities
                        WHERE opportunity_id = ANY($1::text[])
                          AND NOT EXISTS (
                              SELECT 1
                              FROM decisions
                              WHERE decisions.opportunity_id = opportunities.opportunity_id
                          )
                        RETURNING opportunity_id
                        """,
                        list(opportunity_ids),
                    )

        return DecisionPruneResult(
            decisions_deleted=len(decision_rows),
            order_intents_deleted=len(intent_rows),
            opportunities_deleted=len(opportunity_rows),
        )


def validate_decision_status_transition(
    current_status: str,
    next_status: str,
) -> None:
    normalized_current = _coerce_decision_status(current_status)
    normalized_next = _coerce_decision_status(next_status)
    if normalized_next not in _VALID_STATUS_TRANSITIONS[normalized_current]:
        msg = (
            "invalid decision status transition: "
            f"{normalized_current} -> {normalized_next}"
        )
        raise ValueError(msg)


async def _ensure_decision_payloads_table(connection: asyncpg.Connection) -> None:
    await connection.execute(_CREATE_DECISION_PAYLOADS_TABLE)


def _coerce_decision_status(status: str) -> DecisionStatus:
    if status not in DECISION_STATUSES:
        msg = f"unknown decision status: {status}"
        raise ValueError(msg)
    return status


def _decision_payload(
    decision: TradeDecision,
    *,
    decision_evidence: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    return {
        "decision_id": decision.decision_id,
        "market_id": decision.market_id,
        "token_id": decision.token_id,
        "venue": decision.venue,
        "side": decision.side,
        "notional_usdc": decision.notional_usdc,
        "order_type": decision.order_type,
        "max_slippage_bps": decision.max_slippage_bps,
        "stop_conditions": list(decision.stop_conditions),
        "prob_estimate": decision.prob_estimate,
        "expected_edge": decision.expected_edge,
        "time_in_force": decision.time_in_force.value,
        "opportunity_id": decision.opportunity_id,
        "strategy_id": decision.strategy_id,
        "strategy_version_id": decision.strategy_version_id,
        "limit_price": decision.limit_price,
        "action": decision.action,
        "outcome": decision.outcome,
        "model_id": decision.model_id,
        "intent_key": decision.intent_key,
        "risk_group_id": decision.risk_group_id,
        "spread_bps_at_decision": decision.spread_bps_at_decision,
        "calibration_prob_estimate": decision.calibration_prob_estimate,
        "decision_evidence": (
            cast(dict[str, object], _json_safe(decision_evidence))
            if decision_evidence is not None
            else {}
        ),
    }


def _stored_decision_from_row(
    row: asyncpg.Record,
    *,
    include_opportunity: bool,
) -> StoredDecisionRow:
    payload = _json_object(row["payload"])
    decision = _decision_from_payload(payload)
    return StoredDecisionRow(
        decision=decision,
        status=_coerce_decision_status(cast(str, row["status"])),
        factor_snapshot_hash=cast(str | None, row["factor_snapshot_hash"]),
        created_at=cast(datetime, row["created_at"]),
        updated_at=cast(datetime, row["updated_at"]),
        expires_at=cast(datetime, row["expires_at"]),
        decision_evidence=_json_object(payload.get("decision_evidence")),
        opportunity=(
            _opportunity_from_row(row)
            if include_opportunity and row["opportunity_row_id"] is not None
            else None
        ),
    )


def _decision_from_payload(payload: Mapping[str, Any]) -> TradeDecision:
    return TradeDecision(
        decision_id=cast(str, payload["decision_id"]),
        market_id=cast(str, payload["market_id"]),
        token_id=cast(str | None, payload.get("token_id")),
        venue=cast(Literal["polymarket", "kalshi"], payload["venue"]),
        side=cast(Literal["BUY", "SELL"], payload["side"]),
        notional_usdc=float(cast(float, payload["notional_usdc"])),
        order_type=cast(str, payload["order_type"]),
        max_slippage_bps=int(cast(int, payload["max_slippage_bps"])),
        stop_conditions=_string_list(payload.get("stop_conditions")),
        prob_estimate=float(cast(float, payload["prob_estimate"])),
        expected_edge=float(cast(float, payload["expected_edge"])),
        time_in_force=TimeInForce(cast(str, payload["time_in_force"])),
        opportunity_id=cast(str, payload["opportunity_id"]),
        strategy_id=cast(str, payload["strategy_id"]),
        strategy_version_id=cast(str, payload["strategy_version_id"]),
        limit_price=float(cast(float, payload["limit_price"])),
        action=cast(Literal["BUY", "SELL"] | None, payload.get("action")),
        outcome=cast(Literal["YES", "NO"], payload.get("outcome", "YES")),
        model_id=cast(str | None, payload.get("model_id")),
        intent_key=cast(str | None, payload.get("intent_key")),
        risk_group_id=cast(str | None, payload.get("risk_group_id")),
        spread_bps_at_decision=_optional_int(payload.get("spread_bps_at_decision")),
        calibration_prob_estimate=_optional_float(
            payload.get("calibration_prob_estimate")
        ),
    )


def _opportunity_from_row(row: asyncpg.Record) -> Opportunity:
    return Opportunity(
        opportunity_id=cast(str, row["opportunity_row_id"]),
        market_id=cast(str, row["opportunity_market_id"]),
        token_id=cast(str, row["opportunity_token_id"]),
        side=cast(Literal["yes", "no"], row["opportunity_side"]),
        selected_factor_values=_numeric_mapping(row["selected_factor_values"]),
        expected_edge=float(cast(float, row["opportunity_expected_edge"])),
        rationale=cast(str, row["rationale"]),
        target_size_usdc=float(cast(float, row["target_size_usdc"])),
        expiry=cast(datetime | None, row["opportunity_expiry"]),
        staleness_policy=cast(str, row["staleness_policy"]),
        strategy_id=cast(str, row["opportunity_strategy_id"]),
        strategy_version_id=cast(str, row["opportunity_strategy_version_id"]),
        created_at=cast(datetime, row["opportunity_created_at"]),
        factor_snapshot_hash=cast(str | None, row["opportunity_factor_snapshot_hash"]),
        composition_trace=_json_object(row["composition_trace"]),
    )


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    if isinstance(value, str):
        loaded = json.loads(value)
        if isinstance(loaded, dict):
            return cast(dict[str, Any], loaded)
    return {}


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if isfinite(value):
            return value
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(raw_value) for key, raw_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _numeric_mapping(value: object) -> dict[str, float]:
    payload = _json_object(value)
    return {
        str(key): float(raw_value)
        for key, raw_value in payload.items()
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool)
    }


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        numeric = float(cast(int | float | str, value))
    except (TypeError, ValueError):
        return None
    return int(round(numeric))


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(cast(int | float | str, value))
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None
