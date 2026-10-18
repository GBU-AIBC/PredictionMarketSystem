from __future__ import annotations

from dataclasses import asdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
import json
import os
from typing import cast
from uuid import UUID

import asyncpg
import yaml
from yaml import YAMLError

from pms.research.comparison import BacktestLiveComparisonTool
from pms.research.policies import (
    SelectionDenominator,
    SymbolNormalizationPolicy,
    TimeAlignmentPolicy,
)
from pms.research.spec_codec import deserialize_backtest_spec, deserialize_execution_config
from pms.research.sweep import ParameterSweep, QueuedSweepRun
from pms.core.validation import validate_finite


_ORPHANED_FAILURE_REASON = "orphaned (worker process gone)"
_JSON_COLUMN_NAMES = frozenset(
    {
        "benchmark_rows",
        "equity_delta_json",
        "exec_config_json",
        "portfolio_target_json",
        "ranked_strategies",
        "spec_json",
        "symbol_normalization_policy_json",
        "time_alignment_policy_json",
        "warnings",
    }
)
PidProbe = Callable[[int], None]


async def enqueue_backtest_runs(
    pg_pool: asyncpg.Pool,
    sweep_yaml: str,
) -> dict[str, object]:
    payload = _load_yaml_payload(sweep_yaml)
    base_spec = deserialize_backtest_spec(payload["base_spec"])
    exec_config = deserialize_execution_config(payload.get("exec_config", {}))
    parameter_grid = _parameter_grid(payload.get("parameter_grid", {}))

    sweep = ParameterSweep(pool=pg_pool)
    variants = sweep.enumerate_variants(base_spec, parameter_grid)
    queued_runs = await sweep.enqueue(variants, exec_config)

    return {
        "run_ids": [queued_run.run_id for queued_run in queued_runs],
        "unique_run_count": len({queued_run.spec_hash for queued_run in queued_runs}),
        "runs": [_serialize_queued_run(queued_run) for queued_run in queued_runs],
    }


async def fetch_backtest_run(
    pg_pool: asyncpg.Pool,
    run_id: str,
) -> dict[str, object] | None:
    async with pg_pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT
                run_id,
                spec_hash,
                status,
                strategy_ids,
                date_range_start,
                date_range_end,
                exec_config_json,
                spec_json,
                queued_at,
                started_at,
                finished_at,
                failure_reason,
                worker_pid,
                worker_host
            FROM backtest_runs
            WHERE run_id = $1::uuid
            """,
            run_id,
        )
    return None if row is None else _record_to_json(row)


async def list_backtest_runs(
    pg_pool: asyncpg.Pool,
    *,
    limit: int = 25,
) -> list[dict[str, object]]:
    async with pg_pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT
                run_id,
                spec_hash,
                status,
                strategy_ids,
                date_range_start,
                date_range_end,
                exec_config_json,
                spec_json,
                queued_at,
                started_at,
                finished_at,
                failure_reason,
                worker_pid,
                worker_host
            FROM backtest_runs
            ORDER BY queued_at DESC, run_id DESC
            LIMIT $1
            """,
            limit,
        )
    return [_record_to_json(row) for row in rows]


async def list_backtest_strategy_runs(
    pg_pool: asyncpg.Pool,
    run_id: str,
) -> list[dict[str, object]]:
    async with pg_pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT
                strategy_run_id,
                run_id,
                strategy_id,
                strategy_version_id,
                brier,
                pnl_cum,
                drawdown_max,
                fill_rate,
                slippage_bps,
                opportunity_count,
                decision_count,
                fill_count,
                portfolio_target_json,
                started_at,
                finished_at
            FROM strategy_runs
            WHERE run_id = $1::uuid
            ORDER BY started_at ASC NULLS LAST, strategy_id ASC, strategy_version_id ASC
            """,
            run_id,
        )
    return [_record_to_json(row) for row in rows]


async def compute_backtest_live_comparison(
    pg_pool: asyncpg.Pool,
    run_id: str,
    raw_body: object,
) -> dict[str, object]:
    if not isinstance(raw_body, Mapping):
        msg = "comparison request body must decode to a mapping"
        raise TypeError(msg)
    live_window_start = _required_datetime(raw_body, "live_window_start")
    live_window_end = _required_datetime(raw_body, "live_window_end")
    denominator = _required_denominator(raw_body)
    strategy_id, strategy_version_id = await _strategy_identity_for_compare(
        pg_pool,
        run_id=run_id,
        raw_strategy_id=raw_body.get("strategy_id"),
        raw_strategy_version_id=raw_body.get("strategy_version_id"),
    )
    tool = BacktestLiveComparisonTool(
        pool=pg_pool,
        time_alignment_policy=_time_alignment_policy(raw_body.get("time_alignment_policy")),
        symbol_normalization_policy=_symbol_normalization_policy(
            raw_body.get("symbol_normalization_policy")
        ),
    )
    comparison = await tool.compute(
        run_id=run_id,
        strategy_id=strategy_id,
        strategy_version_id=strategy_version_id,
        live_window_start=live_window_start,
        live_window_end=live_window_end,
        denominator=denominator,
    )
    payload = _jsonify(asdict(comparison))
    if not isinstance(payload, dict):
        msg = "comparison payload must encode to a mapping"
        raise TypeError(msg)
    return cast(dict[str, object], payload)


async def scan_orphaned_backtest_runs(
    pg_pool: asyncpg.Pool,
    *,
    pid_probe: PidProbe | None = None,
) -> int:
    probe = _pid_exists if pid_probe is None else pid_probe
    async with pg_pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT run_id, worker_pid
            FROM backtest_runs
            WHERE status = 'running' AND worker_pid IS NOT NULL
            ORDER BY started_at ASC NULLS LAST, queued_at ASC, run_id ASC
            """
        )
        orphaned_run_ids: list[str] = []
        for row in rows:
            worker_pid = row["worker_pid"]
            if not isinstance(worker_pid, int):
                continue
            if _pid_missing(worker_pid, probe):
                orphaned_run_ids.append(str(row["run_id"]))

        for orphaned_run_id in orphaned_run_ids:
            await connection.execute(
                """
                UPDATE backtest_runs
                SET
                    status = 'failed',
                    failure_reason = $1,
                    finished_at = COALESCE(finished_at, now())
                WHERE run_id = $2::uuid AND status = 'running'
                """,
                _ORPHANED_FAILURE_REASON,
                orphaned_run_id,
            )

    return len(orphaned_run_ids)


def orphaned_failure_reason() -> str:
    return _ORPHANED_FAILURE_REASON


def _required_datetime(payload: Mapping[str, object], key: str) -> datetime:
    raw_value = payload.get(key)
    if not isinstance(raw_value, str) or not raw_value:
        msg = f"{key} must be a non-empty ISO8601 string"
        raise TypeError(msg)
    timestamp = datetime.fromisoformat(raw_value)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        msg = f"{key} must be timezone-aware"
        raise ValueError(msg)
    return timestamp


def _required_denominator(payload: Mapping[str, object]) -> SelectionDenominator:
    raw_value = payload.get("denominator")
    if raw_value not in ("backtest_set", "live_set", "union"):
        msg = "denominator must be one of 'backtest_set', 'live_set', or 'union'"
        raise ValueError(msg)
    return raw_value


async def _strategy_identity_for_compare(
    pg_pool: asyncpg.Pool,
    *,
    run_id: str,
    raw_strategy_id: object,
    raw_strategy_version_id: object,
) -> tuple[str, str]:
    if raw_strategy_id is None and raw_strategy_version_id is None:
        async with pg_pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT DISTINCT strategy_id, strategy_version_id
                FROM strategy_runs
                WHERE run_id = $1::uuid
                ORDER BY strategy_id ASC, strategy_version_id ASC
                """,
                run_id,
            )
        if not rows:
            msg = f"Backtest run {run_id} has no strategy rows to compare"
            raise LookupError(msg)
        if len(rows) != 1:
            msg = (
                "strategy_id and strategy_version_id are required when a run "
                "contains multiple strategy rows"
            )
            raise ValueError(msg)
        return (cast(str, rows[0]["strategy_id"]), cast(str, rows[0]["strategy_version_id"]))
    if not isinstance(raw_strategy_id, str) or not raw_strategy_id:
        msg = "strategy_id must be a non-empty string when provided"
        raise TypeError(msg)
    if not isinstance(raw_strategy_version_id, str) or not raw_strategy_version_id:
        msg = "strategy_version_id must be a non-empty string when provided"
        raise TypeError(msg)
    return (raw_strategy_id, raw_strategy_version_id)


def _time_alignment_policy(raw_value: object) -> TimeAlignmentPolicy:
    if raw_value is None:
        return TimeAlignmentPolicy()
    if not isinstance(raw_value, Mapping):
        msg = "time_alignment_policy must decode to a mapping"
        raise TypeError(msg)
    return TimeAlignmentPolicy(
        generated_offset_s=_float_field(raw_value, "generated_offset_s", default=0.0),
        exchange_offset_s=_float_field(raw_value, "exchange_offset_s", default=0.0),
        ingest_offset_s=_float_field(raw_value, "ingest_offset_s", default=0.0),
        evaluation_offset_s=_float_field(raw_value, "evaluation_offset_s", default=0.0),
    )


def _symbol_normalization_policy(raw_value: object) -> SymbolNormalizationPolicy:
    if raw_value is None:
        return SymbolNormalizationPolicy()
    if not isinstance(raw_value, Mapping):
        msg = "symbol_normalization_policy must decode to a mapping"
        raise TypeError(msg)
    return SymbolNormalizationPolicy(
        token_id_aliases=_alias_map(raw_value.get("token_id_aliases")),
        market_id_aliases=_alias_map(raw_value.get("market_id_aliases")),
    )


def _alias_map(raw_value: object) -> Mapping[str, str]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, Mapping):
        msg = "alias maps must decode to string-to-string mappings"
        raise TypeError(msg)
    normalized: dict[str, str] = {}
    for key, value in raw_value.items():
        if not isinstance(key, str) or not isinstance(value, str):
            msg = "alias maps must decode to string-to-string mappings"
            raise TypeError(msg)
        normalized[key] = value
    return normalized


def _float_field(
    payload: Mapping[str, object],
    key: str,
    *,
    default: float,
) -> float:
    raw_value = payload.get(key, default)
    if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
        msg = f"{key} must be numeric"
        raise TypeError(msg)
    return validate_finite(raw_value, field_name=key)


def _load_yaml_payload(sweep_yaml: str) -> dict[str, object]:
    try:
        loaded = yaml.safe_load(sweep_yaml)
    except YAMLError as exc:
        raise ValueError(f"sweep spec is malformed YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        msg = "sweep spec must decode to a mapping"
        raise TypeError(msg)
    return dict(loaded)


def _parameter_grid(raw_value: object) -> dict[str, Sequence[object]]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        msg = "parameter_grid must decode to a mapping"
        raise TypeError(msg)
    normalized: dict[str, Sequence[object]] = {}
    for key, value in raw_value.items():
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            msg = f"parameter_grid[{key!r}] must be a sequence"
            raise TypeError(msg)
        normalized[str(key)] = tuple(value)
    return normalized


def _serialize_queued_run(queued_run: QueuedSweepRun) -> dict[str, object]:
    return {
        "run_id": queued_run.run_id,
        "spec_hash": queued_run.spec_hash,
        "inserted": queued_run.inserted,
    }


def _pid_exists(pid: int) -> None:
    os.kill(pid, 0)


def _pid_missing(pid: int, probe: PidProbe) -> bool:
    try:
        probe(pid)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _record_to_json(record: Mapping[str, object]) -> dict[str, object]:
    return {
        key: _jsonify(_decode_json_column(key, value))
        for key, value in dict(record).items()
    }


def _jsonify(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    return value


def _decode_json_column(key: str, value: object) -> object:
    if key not in _JSON_COLUMN_NAMES or not isinstance(value, str):
        return value
    return json.loads(value)


__all__ = [
    "compute_backtest_live_comparison",
    "enqueue_backtest_runs",
    "fetch_backtest_run",
    "list_backtest_runs",
    "list_backtest_strategy_runs",
    "orphaned_failure_reason",
    "scan_orphaned_backtest_runs",
]
