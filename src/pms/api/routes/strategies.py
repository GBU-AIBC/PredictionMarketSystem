from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

import asyncpg
from pydantic import BaseModel, field_serializer

from pms.core.models import EvalRecord
from pms.config import PMSSettings
from pms.evaluation.metrics import (
    MetricsCollector,
    StrategyMetricsSnapshot,
    StrategyVersionKey,
)
from pms.storage.eval_store import EvalStore
from pms.storage.strategy_registry import (
    PostgresStrategyRegistry,
    llm_identity_for_settings,
)
from pms.strategies.projections import StrategyRow


class StrategyRowResponse(BaseModel):
    strategy_id: str
    active_version_id: str | None
    created_at: datetime

    @field_serializer("created_at")
    def _serialize_created_at(self, created_at: datetime) -> str:
        return created_at.isoformat()


class StrategiesResponse(BaseModel):
    strategies: list[StrategyRowResponse]


class StrategyMetricsRowResponse(BaseModel):
    strategy_id: str
    strategy_version_id: str
    created_at: datetime
    record_count: int
    insufficient_samples: bool
    brier_overall: float | None
    baseline_brier_overall: float | None
    brier_improvement_overall: float | None
    pnl: float
    pnl_source: Literal["final_eval", "quote_mtm", "none"]
    fill_rate: float
    slippage_bps: float
    drawdown: float
    decision_count: int
    fill_count: int
    execution_fill_rate: float
    executed_notional_usdc: float
    quote_record_count: int
    quote_score_overall: float | None
    quote_mtm_pnl: float

    @field_serializer("created_at")
    def _serialize_created_at(self, created_at: datetime) -> str:
        return created_at.isoformat()


class StrategyMetricsResponse(BaseModel):
    strategies: list[StrategyMetricsRowResponse]


@dataclass(frozen=True)
class StrategyExecutionSnapshot:
    decision_count: int
    fill_count: int
    executed_notional_usdc: float
    quote_record_count: int
    quote_score_overall: float | None
    quote_mtm_pnl: float

    @property
    def execution_fill_rate(self) -> float:
        if self.decision_count <= 0:
            return 0.0
        return self.fill_count / self.decision_count


async def list_strategies(
    pg_pool: asyncpg.Pool,
    *,
    settings: PMSSettings | None = None,
) -> dict[str, Any]:
    registry = PostgresStrategyRegistry(
        pg_pool,
        llm_identity=None if settings is None else llm_identity_for_settings(settings),
    )
    payload = StrategiesResponse(
        strategies=[
            StrategyRowResponse(
                strategy_id=row.strategy_id,
                active_version_id=row.active_version_id,
                created_at=row.created_at,
            )
            for row in await registry.list_strategies()
        ]
    )
    return payload.model_dump(mode="json")


async def list_strategy_metrics(
    pg_pool: asyncpg.Pool,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    settings: PMSSettings | None = None,
) -> dict[str, Any]:
    since = None if since is None else _coerce_aware_datetime(since)
    until = None if until is None else _coerce_aware_datetime(until)
    registry = PostgresStrategyRegistry(
        pg_pool,
        llm_identity=None if settings is None else llm_identity_for_settings(settings),
    )
    strategy_rows = await registry.list_active_strategy_rows()
    eval_store = EvalStore(pg_pool)
    records_by_strategy = _filter_records_by_window(
        await _load_records_by_strategy(eval_store, strategy_rows),
        since=since,
        until=until,
    )
    execution_by_strategy = await _load_execution_snapshots(
        pg_pool,
        since=since,
        until=until,
    )
    grouped_snapshots = MetricsCollector(
        record
        for strategy_records in records_by_strategy.values()
        for record in strategy_records
    ).snapshot_by_strategy()

    payload = StrategyMetricsResponse(
        strategies=[
            _strategy_metrics_row(
                row,
                records_by_strategy.get(
                    (row.strategy_id, _active_version_id(row)),
                    [],
                ),
                grouped_snapshots,
                execution_by_strategy.get(
                    (row.strategy_id, _active_version_id(row)),
                    _empty_execution_snapshot(),
                ),
            )
            for row in strategy_rows
        ]
    )
    return payload.model_dump(mode="json")


async def _load_records_by_strategy(
    eval_store: EvalStore,
    strategy_rows: list[StrategyRow],
) -> dict[StrategyVersionKey, list[EvalRecord]]:
    if not strategy_rows:
        return {}

    keys = [
        (row.strategy_id, _active_version_id(row))
        for row in strategy_rows
    ]
    results = await asyncio.gather(
        *(
            eval_store.all_for_strategy(strategy_id, strategy_version_id)
            for strategy_id, strategy_version_id in keys
        )
    )
    return dict(zip(keys, results, strict=True))


async def _load_execution_snapshots(
    pg_pool: asyncpg.Pool,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[StrategyVersionKey, StrategyExecutionSnapshot]:
    async with pg_pool.acquire() as connection:
        rows = await connection.fetch(
            """
            WITH active_strategies AS (
                SELECT strategy_id, active_version_id AS strategy_version_id
                FROM strategies
                WHERE active_version_id IS NOT NULL
                  AND archived IS NOT TRUE
            ),
            decision_stats AS (
                SELECT strategy_id, strategy_version_id, COUNT(*)::integer AS decision_count
                FROM decisions
                WHERE ($1::timestamptz IS NULL OR created_at >= $1)
                  AND ($2::timestamptz IS NULL OR created_at < $2)
                GROUP BY strategy_id, strategy_version_id
            ),
            fill_stats AS (
                SELECT
                    strategy_id,
                    strategy_version_id,
                    COUNT(*)::integer AS fill_count,
                    COALESCE(SUM(fill_notional_usdc), 0.0)::double precision
                        AS executed_notional_usdc
                FROM fills
                WHERE ($1::timestamptz IS NULL OR ts >= $1)
                  AND ($2::timestamptz IS NULL OR ts < $2)
                GROUP BY strategy_id, strategy_version_id
            ),
            quote_rows AS (
                SELECT
                    strategy_id,
                    strategy_version_id,
                    fill_id,
                    quote_score,
                    mtm_pnl,
                    ROW_NUMBER() OVER (
                        PARTITION BY strategy_id, strategy_version_id, fill_id
                        ORDER BY recorded_at DESC, quote_lag_seconds DESC
                    ) AS fill_mark_rank
                FROM quote_eval_records
                WHERE ($1::timestamptz IS NULL OR recorded_at >= $1)
                  AND ($2::timestamptz IS NULL OR recorded_at < $2)
            ),
            quote_stats AS (
                SELECT
                    strategy_id,
                    strategy_version_id,
                    COUNT(*)::integer AS quote_record_count,
                    AVG(quote_score)::double precision AS quote_score_overall,
                    COALESCE(
                        SUM(mtm_pnl) FILTER (WHERE fill_mark_rank = 1),
                        0.0
                    )::double precision AS quote_mtm_pnl
                FROM quote_rows
                GROUP BY strategy_id, strategy_version_id
            )
            SELECT
                active_strategies.strategy_id,
                active_strategies.strategy_version_id,
                COALESCE(decision_stats.decision_count, 0)::integer AS decision_count,
                COALESCE(fill_stats.fill_count, 0)::integer AS fill_count,
                COALESCE(fill_stats.executed_notional_usdc, 0.0)::double precision
                    AS executed_notional_usdc,
                COALESCE(quote_stats.quote_record_count, 0)::integer AS quote_record_count,
                quote_stats.quote_score_overall,
                COALESCE(quote_stats.quote_mtm_pnl, 0.0)::double precision AS quote_mtm_pnl
            FROM active_strategies
            LEFT JOIN decision_stats
                ON decision_stats.strategy_id = active_strategies.strategy_id
               AND decision_stats.strategy_version_id = active_strategies.strategy_version_id
            LEFT JOIN fill_stats
                ON fill_stats.strategy_id = active_strategies.strategy_id
               AND fill_stats.strategy_version_id = active_strategies.strategy_version_id
            LEFT JOIN quote_stats
                ON quote_stats.strategy_id = active_strategies.strategy_id
               AND quote_stats.strategy_version_id = active_strategies.strategy_version_id
            ORDER BY active_strategies.strategy_id ASC
            """,
            since,
            until,
        )
    return {
        (str(row["strategy_id"]), str(row["strategy_version_id"])): (
            StrategyExecutionSnapshot(
                decision_count=int(row["decision_count"]),
                fill_count=int(row["fill_count"]),
                executed_notional_usdc=float(row["executed_notional_usdc"]),
                quote_record_count=int(row["quote_record_count"]),
                quote_score_overall=(
                    None
                    if row["quote_score_overall"] is None
                    else float(row["quote_score_overall"])
                ),
                quote_mtm_pnl=float(row["quote_mtm_pnl"]),
            )
        )
        for row in rows
    }


def _filter_records_by_window(
    records_by_strategy: dict[StrategyVersionKey, list[EvalRecord]],
    *,
    since: datetime | None,
    until: datetime | None,
) -> dict[StrategyVersionKey, list[EvalRecord]]:
    if since is None and until is None:
        return records_by_strategy
    return {
        key: [
            record
            for record in records
            if (since is None or record.recorded_at >= since)
            and (until is None or record.recorded_at < until)
        ]
        for key, records in records_by_strategy.items()
    }


def _coerce_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _empty_execution_snapshot() -> StrategyExecutionSnapshot:
    return StrategyExecutionSnapshot(
        decision_count=0,
        fill_count=0,
        executed_notional_usdc=0.0,
        quote_record_count=0,
        quote_score_overall=None,
        quote_mtm_pnl=0.0,
    )


def _strategy_metrics_row(
    row: StrategyRow,
    records: list[EvalRecord],
    grouped_snapshots: Mapping[StrategyVersionKey, StrategyMetricsSnapshot],
    execution: StrategyExecutionSnapshot,
) -> StrategyMetricsRowResponse:
    strategy_version_id = _active_version_id(row)
    snapshot = grouped_snapshots.get((row.strategy_id, strategy_version_id))
    pnl_source: Literal["final_eval", "quote_mtm", "none"] = "none"
    pnl = 0.0
    if snapshot is not None:
        pnl_source = "final_eval"
        pnl = snapshot.pnl
    elif execution.quote_record_count > 0:
        pnl_source = "quote_mtm"
        pnl = execution.quote_mtm_pnl
    fill_rate = (
        snapshot.fill_rate
        if snapshot is not None
        else execution.execution_fill_rate
    )
    return StrategyMetricsRowResponse(
        strategy_id=row.strategy_id,
        strategy_version_id=strategy_version_id,
        created_at=row.created_at,
        record_count=len(records),
        insufficient_samples=len(records) == 0,
        brier_overall=None if snapshot is None else snapshot.brier_overall,
        baseline_brier_overall=(
            None if snapshot is None else snapshot.baseline_brier_overall
        ),
        brier_improvement_overall=(
            None if snapshot is None else snapshot.brier_improvement_overall
        ),
        pnl=pnl,
        pnl_source=pnl_source,
        fill_rate=fill_rate,
        slippage_bps=0.0 if snapshot is None else snapshot.slippage_bps,
        drawdown=_max_drawdown(records),
        decision_count=execution.decision_count,
        fill_count=execution.fill_count,
        execution_fill_rate=execution.execution_fill_rate,
        executed_notional_usdc=execution.executed_notional_usdc,
        quote_record_count=execution.quote_record_count,
        quote_score_overall=execution.quote_score_overall,
        quote_mtm_pnl=execution.quote_mtm_pnl,
    )


def _active_version_id(row: StrategyRow) -> str:
    if row.active_version_id is None:
        msg = f"strategy {row.strategy_id} is missing an active version"
        raise ValueError(msg)
    return row.active_version_id


def _max_drawdown(records: list[EvalRecord]) -> float:
    cumulative_pnl = Decimal("0")
    peak_equity = Decimal("0")
    max_drawdown = Decimal("0")
    for record in records:
        cumulative_pnl += Decimal(str(record.pnl))
        peak_equity = max(peak_equity, cumulative_pnl)
        max_drawdown = max(max_drawdown, peak_equity - cumulative_pnl)
    return float(max_drawdown)
