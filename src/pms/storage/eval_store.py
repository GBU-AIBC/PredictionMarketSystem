from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import SupportsFloat, cast

import asyncpg

from pms.core.models import EvalRecord


@dataclass
class EvalStore:
    pool: asyncpg.Pool | None = None

    def bind_pool(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def append(self, record: EvalRecord) -> bool:
        async with self._pool().acquire() as connection:
            return await insert_eval_record_row(connection, record)

    async def all(self) -> list[EvalRecord]:
        if self.pool is None:
            return []

        async with self.pool.acquire() as connection:
            rows = await connection.fetch(_SELECT_ALL_QUERY)
        return [_eval_record_from_row(row) for row in rows]

    async def all_for_strategy(
        self,
        strategy_id: str,
        strategy_version_id: str,
    ) -> list[EvalRecord]:
        if self.pool is None:
            return []

        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                _SELECT_BY_STRATEGY_QUERY,
                strategy_id,
                strategy_version_id,
            )
        return [_eval_record_from_row(row) for row in rows]

    def _pool(self) -> asyncpg.Pool:
        if self.pool is None:
            msg = "EvalStore pool is not bound"
            raise RuntimeError(msg)
        return self.pool


async def insert_eval_record_row(
    connection: asyncpg.Connection,
    record: EvalRecord,
) -> bool:
    result = await connection.execute(
        """
        INSERT INTO eval_records (
            decision_id,
            market_id,
            prob_estimate,
            resolved_outcome,
            brier_score,
            fill_status,
            recorded_at,
            citations,
            baseline_prob_estimate,
            baseline_brier_score,
            baseline_prob_estimates,
            baseline_brier_scores,
            category,
            model_id,
            pnl,
            slippage_bps,
            filled,
            strategy_id,
            strategy_version_id,
            edge_at_decision,
            spread_bps_at_decision
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11::jsonb, $12::jsonb,
            $13, $14, $15, $16, $17, $18, $19, $20, $21
        )
        ON CONFLICT (decision_id) DO NOTHING
        """,
        record.decision_id,
        record.market_id,
        record.prob_estimate,
        record.resolved_outcome,
        record.brier_score,
        record.fill_status,
        record.recorded_at,
        json.dumps(record.citations),
        record.baseline_prob_estimate,
        record.baseline_brier_score,
        json.dumps(dict(record.baseline_prob_estimates), allow_nan=False),
        json.dumps(dict(record.baseline_brier_scores), allow_nan=False),
        record.category,
        record.model_id,
        record.pnl,
        record.slippage_bps,
        record.filled,
        record.strategy_id,
        record.strategy_version_id,
        record.edge_at_decision,
        record.spread_bps_at_decision,
    )
    return cast(str, result).rsplit(" ", maxsplit=1)[-1] == "1"


_SELECT_ALL_QUERY = """
SELECT
    market_id,
    decision_id,
    prob_estimate,
    resolved_outcome,
    brier_score,
    fill_status,
    recorded_at,
    citations,
    baseline_prob_estimate,
    baseline_brier_score,
    baseline_prob_estimates,
    baseline_brier_scores,
    strategy_id,
    strategy_version_id,
    category,
    model_id,
    pnl,
    slippage_bps,
    filled,
    edge_at_decision,
    spread_bps_at_decision
FROM eval_records
ORDER BY recorded_at ASC, decision_id ASC
"""


_SELECT_BY_STRATEGY_QUERY = """
SELECT
    market_id,
    decision_id,
    prob_estimate,
    resolved_outcome,
    brier_score,
    fill_status,
    recorded_at,
    citations,
    baseline_prob_estimate,
    baseline_brier_score,
    baseline_prob_estimates,
    baseline_brier_scores,
    strategy_id,
    strategy_version_id,
    category,
    model_id,
    pnl,
    slippage_bps,
    filled,
    edge_at_decision,
    spread_bps_at_decision
FROM eval_records
WHERE strategy_id = $1 AND strategy_version_id = $2
ORDER BY recorded_at ASC, decision_id ASC
"""


def _eval_record_from_row(row: asyncpg.Record) -> EvalRecord:
    citations_value = row["citations"]
    citations: list[str]
    if isinstance(citations_value, list):
        citations = [str(item) for item in citations_value]
    elif isinstance(citations_value, str):
        loaded = json.loads(citations_value)
        citations = [str(item) for item in loaded] if isinstance(loaded, list) else []
    else:
        citations = []

    return EvalRecord(
        market_id=cast(str, row["market_id"]),
        decision_id=cast(str, row["decision_id"]),
        strategy_id=cast(str, row["strategy_id"]),
        strategy_version_id=cast(str, row["strategy_version_id"]),
        prob_estimate=cast(float, row["prob_estimate"]),
        resolved_outcome=cast(float, row["resolved_outcome"]),
        brier_score=cast(float, row["brier_score"]),
        baseline_prob_estimate=cast(
            float | None,
            _row_value(row, "baseline_prob_estimate", None),
        ),
        baseline_brier_score=cast(
            float | None,
            _row_value(row, "baseline_brier_score", None),
        ),
        baseline_prob_estimates=_json_numeric_mapping(
            _row_value(row, "baseline_prob_estimates", {}),
        ),
        baseline_brier_scores=_json_numeric_mapping(
            _row_value(row, "baseline_brier_scores", {}),
        ),
        fill_status=cast(str, row["fill_status"]),
        recorded_at=cast(datetime, row["recorded_at"]),
        citations=citations,
        category=cast(str | None, row["category"]),
        model_id=cast(str | None, row["model_id"]),
        pnl=cast(float, row["pnl"]),
        slippage_bps=cast(float, row["slippage_bps"]),
        filled=cast(bool, row["filled"]),
        edge_at_decision=float(
            cast(SupportsFloat, _row_value(row, "edge_at_decision", 0.0))
        ),
        spread_bps_at_decision=cast(
            int | None,
            _row_value(row, "spread_bps_at_decision", None),
        ),
    )


def _row_value(row: asyncpg.Record, key: str, default: object) -> object:
    try:
        value = row[key]
    except (KeyError, IndexError):
        return default
    return default if value is None and default is not None else value


def _json_numeric_mapping(value: object) -> dict[str, float]:
    payload: object = value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(payload, dict):
        return {}
    mapping: dict[str, float] = {}
    for key, raw_value in payload.items():
        try:
            mapping[str(key)] = float(cast(SupportsFloat, raw_value))
        except (TypeError, ValueError):
            continue
    return mapping
