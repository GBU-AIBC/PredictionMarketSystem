"""Command-line entry points for research sweeps and workers."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import signal
import stat
import sys
from typing import Callable

import asyncpg
import httpx
import yaml

from pms.config import PMSSettings
from pms.core.exceptions import PMSBootError
from pms.evaluation.resolution import GammaResolutionSource
from pms.research.report import EvaluationReportGenerator
from pms.research.runner import BacktestRunner, CancelProbe
from pms.research.sweep import (
    ParameterSweep,
    QueuedSweepRun,
    cache_gate_threshold,
    deserialize_backtest_spec,
    deserialize_execution_config,
)
from pms.storage.schema_check import ensure_schema_current
from pms.storage.market_resolution_store import MarketResolutionStore


@dataclass(frozen=True, slots=True)
class _SweepArgs:
    spec_path_or_stdin: str
    database_url: str | None
    wait: bool
    wait_timeout: float
    poll_interval: float
    no_cache: bool


@dataclass(frozen=True, slots=True)
class _WorkerArgs:
    database_url: str | None
    poll_interval: float
    max_runs: int | None


@dataclass(frozen=True, slots=True)
class _ResolutionBackfillArgs:
    database_url: str | None
    market_ids: tuple[str, ...]
    start: datetime | None
    end: datetime | None


@dataclass(slots=True)
class _WorkerState:
    stop_requested: bool = False
    active_run_id: str | None = None
    processed_runs: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pms-research")
    subparsers = parser.add_subparsers(dest="command")

    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("spec_path_or_stdin")
    sweep_parser.add_argument("--database-url")
    sweep_parser.add_argument("--wait", action="store_true")
    sweep_parser.add_argument("--wait-timeout", type=float, default=600.0)
    sweep_parser.add_argument("--poll-interval", type=float, default=1.0)
    sweep_parser.add_argument("--no-cache", action="store_true")

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--database-url")
    worker_parser.add_argument("--poll-interval", type=float, default=1.0)
    worker_parser.add_argument("--max-runs", type=int)
    resolution_parser = subparsers.add_parser("resolution-backfill")
    resolution_parser.add_argument("--database-url")
    resolution_parser.add_argument("--market-id", action="append", default=[])
    resolution_parser.add_argument("--start")
    resolution_parser.add_argument("--end")
    return parser


def main() -> None:
    parser = build_parser()
    namespace = parser.parse_args()

    if namespace.command is None:
        parser.print_help()
        raise SystemExit(0)

    if namespace.command == "sweep":
        sweep_args = _SweepArgs(
            spec_path_or_stdin=str(namespace.spec_path_or_stdin),
            database_url=_optional_string(namespace.database_url),
            wait=bool(namespace.wait),
            wait_timeout=float(namespace.wait_timeout),
            poll_interval=float(namespace.poll_interval),
            no_cache=bool(namespace.no_cache),
        )
        raise SystemExit(asyncio.run(_run_sweep(sweep_args)))

    if namespace.command == "worker":
        worker_args = _WorkerArgs(
            database_url=_optional_string(namespace.database_url),
            poll_interval=float(namespace.poll_interval),
            max_runs=_optional_int(namespace.max_runs),
        )
        try:
            raise SystemExit(asyncio.run(_run_worker(worker_args)))
        except PMSBootError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc

    if namespace.command == "resolution-backfill":
        args = _ResolutionBackfillArgs(
            database_url=_optional_string(namespace.database_url),
            market_ids=tuple(str(item) for item in namespace.market_id),
            start=_parse_cli_datetime(namespace.start),
            end=_parse_cli_datetime(namespace.end),
        )
        raise SystemExit(asyncio.run(_run_resolution_backfill(args)))

    raise SystemExit(f"unknown command: {namespace.command}")


async def _run_sweep(args: _SweepArgs) -> int:
    payload = _load_yaml_payload(args.spec_path_or_stdin)
    base_spec = deserialize_backtest_spec(payload["base_spec"])
    exec_config = deserialize_execution_config(payload.get("exec_config", {}))
    parameter_grid = _parameter_grid(payload.get("parameter_grid", {}))

    pool = await _create_pool(args.database_url)
    try:
        sweep = ParameterSweep(pool=pool, cache_enabled=not args.no_cache)
        variants = sweep.enumerate_variants(base_spec, parameter_grid)
        await sweep.warm_cache(variants)
        cache_hit_rate = sweep.cache_hit_rate()
        if not args.no_cache and cache_hit_rate <= cache_gate_threshold():
            _print_json(
                {
                    "error": "cache_gate_failed",
                    "cache_hit_rate": cache_hit_rate,
                    "required_hit_rate": cache_gate_threshold(),
                    "variant_count": len(variants),
                }
            )
            return 1

        queued_runs = await sweep.enqueue(variants, exec_config)
        response: dict[str, object] = {
            "run_ids": [queued_run.run_id for queued_run in queued_runs],
            "unique_run_count": len({queued_run.spec_hash for queued_run in queued_runs}),
            "cache_hit_rate": cache_hit_rate,
            "runs": [_serialize_queued_run(queued_run) for queued_run in queued_runs],
        }
        if args.wait:
            response["results"] = await _wait_for_runs(
                pool=pool,
                run_ids=[queued_run.run_id for queued_run in queued_runs],
                poll_interval=args.poll_interval,
                timeout_s=args.wait_timeout,
            )

        _print_json(response)
        return 0
    finally:
        await pool.close()


async def _run_worker(args: _WorkerArgs) -> int:
    pool = await _create_pool(args.database_url)
    loop = asyncio.get_running_loop()
    state = _WorkerState()

    def request_stop() -> None:
        state.stop_requested = True

    handlers = _install_signal_handlers(loop, request_stop)
    try:
        await ensure_schema_current(pool)
        runner = BacktestRunner(
            writable_pool=pool,
            readonly_pool=pool,
            cancel_probe=_cancel_probe_from_env(),
        )
        report_generator = EvaluationReportGenerator(pool)

        while True:
            if state.stop_requested and state.active_run_id is None:
                break
            if args.max_runs is not None and state.processed_runs >= args.max_runs:
                break

            run_id = await _next_queued_run_id(pool)
            if run_id is None:
                if state.stop_requested:
                    break
                await asyncio.sleep(args.poll_interval)
                continue

            state.active_run_id = run_id
            try:
                succeeded = await runner.execute(run_id)
                if succeeded:
                    await report_generator.generate(run_id)
                    state.processed_runs += 1
                elif await _run_finished(pool, run_id):
                    state.processed_runs += 1
            finally:
                state.active_run_id = None

            if state.stop_requested:
                break
    finally:
        for handled_signal in handlers:
            with suppress(NotImplementedError):
                loop.remove_signal_handler(handled_signal)
        await pool.close()

    _print_json(
        {
            "processed_runs": state.processed_runs,
            "stop_requested": state.stop_requested,
        }
    )
    return 0


async def _run_resolution_backfill(args: _ResolutionBackfillArgs) -> int:
    pool = await _create_pool(args.database_url)
    try:
        await ensure_schema_current(pool)
        market_ids = list(args.market_ids)
        if not market_ids:
            rows = await pool.fetch(
                """
                SELECT condition_id FROM markets
                WHERE ($1::timestamptz IS NULL OR resolves_at >= $1)
                  AND ($2::timestamptz IS NULL OR resolves_at <= $2)
                ORDER BY condition_id
                """,
                args.start,
                args.end,
            )
            market_ids = [str(row["condition_id"]) for row in rows]
        settings = PMSSettings.load()
        timeout = httpx.Timeout(settings.sensor.discovery_http_timeout_s)
        async with httpx.AsyncClient(
            base_url="https://gamma-api.polymarket.com",
            timeout=timeout,
            trust_env=False,
        ) as client:
            source = GammaResolutionSource(http_client=client)
            resolutions = await source.fetch_resolutions(market_ids)
        store = MarketResolutionStore(pool)
        for market_id, resolved_outcome in resolutions.items():
            await store.upsert(
                market_id,
                resolved_outcome=resolved_outcome,
                resolved_at=source.resolved_at_by_market.get(market_id),
                source="gamma",
            )
        _print_json(
            {
                "requested_market_count": len(market_ids),
                "resolved_market_count": len(resolutions),
            }
        )
        return 0
    finally:
        await pool.close()


async def _create_pool(database_url: str | None) -> asyncpg.Pool:
    settings = PMSSettings.load()
    dsn = database_url or settings.database.dsn
    return await asyncpg.create_pool(
        dsn=dsn,
        min_size=max(1, settings.database.pool_min_size),
        max_size=max(1, settings.database.pool_max_size),
    )


def _load_yaml_payload(path_or_stdin: str) -> dict[str, object]:
    raw_text = sys.stdin.read() if path_or_stdin == "-" else _read_text(path_or_stdin)
    loaded = yaml.safe_load(raw_text)
    if not isinstance(loaded, dict):
        msg = "sweep spec must decode to a mapping"
        raise TypeError(msg)
    return dict(loaded)


def _parse_cli_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


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


def _read_text(path: str) -> str:
    spec_path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        fd = os.open(spec_path, flags)
        path_stat = os.fstat(fd)
        if not stat.S_ISREG(path_stat.st_mode):
            raise OSError("not a regular file")
        if path_stat.st_nlink != 1:
            raise OSError("multiple hardlinks")
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            return handle.read()
    except OSError as exc:
        msg = f"research sweep spec cannot be read safely: {spec_path}"
        raise ValueError(msg) from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return int(str(value))


def _serialize_queued_run(queued_run: QueuedSweepRun) -> dict[str, object]:
    return {
        "run_id": queued_run.run_id,
        "spec_hash": queued_run.spec_hash,
        "inserted": queued_run.inserted,
    }


async def _wait_for_runs(
    *,
    pool: asyncpg.Pool,
    run_ids: Sequence[str],
    poll_interval: float,
    timeout_s: float,
) -> list[dict[str, object]]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    report_generator = EvaluationReportGenerator(pool)
    terminal_rows: dict[str, dict[str, object]] = {}
    pending_run_ids = set(run_ids)

    while pending_run_ids:
        if asyncio.get_running_loop().time() > deadline:
            msg = f"sweep wait timed out after {timeout_s:.1f}s"
            raise TimeoutError(msg)
        for run_id in tuple(pending_run_ids):
            snapshot = await _run_snapshot(pool, run_id)
            if snapshot["status"] not in {"completed", "failed", "cancelled"}:
                continue
            if snapshot["status"] == "completed":
                report = await report_generator.generate(run_id)
                snapshot["ranking_metric"] = report.ranking_metric
                snapshot["top_ranked_strategy"] = (
                    None
                    if not report.ranked_strategies
                    else {
                        "strategy_id": report.ranked_strategies[0].strategy_id,
                        "strategy_version_id": report.ranked_strategies[0].strategy_version_id,
                        "metric_value": report.ranked_strategies[0].metric_value,
                    }
                )
            terminal_rows[run_id] = snapshot
            pending_run_ids.remove(run_id)
        if pending_run_ids:
            await asyncio.sleep(poll_interval)
    return [terminal_rows[run_id] for run_id in run_ids]


async def _next_queued_run_id(pool: asyncpg.Pool) -> str | None:
    async with pool.acquire() as connection:
        value = await connection.fetchval(
            """
            SELECT run_id
            FROM backtest_runs
            WHERE status = 'queued'
            ORDER BY queued_at ASC, run_id ASC
            LIMIT 1
            """
        )
    return None if value is None else str(value)


async def _run_finished(pool: asyncpg.Pool, run_id: str) -> bool:
    snapshot = await _run_snapshot(pool, run_id)
    return snapshot["status"] in {"completed", "failed", "cancelled"}


async def _run_snapshot(pool: asyncpg.Pool, run_id: str) -> dict[str, object]:
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT run_id, status, failure_reason, started_at, finished_at
            FROM backtest_runs
            WHERE run_id = $1::uuid
            """,
            run_id,
        )
    if row is None:
        msg = f"backtest run {run_id} does not exist"
        raise LookupError(msg)
    return {
        "run_id": str(row["run_id"]),
        "status": str(row["status"]),
        "failure_reason": row["failure_reason"],
        "started_at": _serialize_optional_datetime(row["started_at"]),
        "finished_at": _serialize_optional_datetime(row["finished_at"]),
    }


def _serialize_optional_datetime(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        msg = "expected datetime database field"
        raise TypeError(msg)
    return value.astimezone(UTC).isoformat()


def _cancel_probe_from_env() -> CancelProbe | None:
    raw_delay = os.environ.get("PMS_RESEARCH_WORKER_CANCEL_PROBE_DELAY_S")
    if raw_delay is None:
        return None
    delay_s = float(raw_delay)
    if delay_s <= 0.0:
        return None

    async def probe(_point: str) -> None:
        await asyncio.sleep(delay_s)

    return probe


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    request_stop: Callable[[], None],
) -> list[signal.Signals]:
    handled: list[signal.Signals] = []
    for handled_signal in (signal.SIGTERM, signal.SIGINT):
        with suppress(NotImplementedError):
            loop.add_signal_handler(handled_signal, request_stop)
            handled.append(handled_signal)
    return handled


def _print_json(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
