from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import asyncpg

from pms.api.app import create_app
from pms.api.research_routes import (
    _record_to_json,
    compute_backtest_live_comparison,
    enqueue_backtest_runs,
    fetch_backtest_run,
    scan_orphaned_backtest_runs,
)
from pms.research.sweep import QueuedSweepRun
from pms.runner import Runner


_SWEEP_YAML = """
base_spec:
  strategy_versions:
    - ["alpha", "alpha-v1"]
  dataset:
    source: "fixture"
    version: "v1"
    coverage_start: "2026-04-01T00:00:00+00:00"
    coverage_end: "2026-04-30T00:00:00+00:00"
    market_universe_filter:
      market_ids: ["market-a"]
    data_quality_gaps: []
  execution_model:
    fee_rate: 0.0
    slippage_bps: 5.0
    latency_ms: 0.0
    staleness_ms: 60000.0
    fill_policy: "immediate_or_cancel"
  risk_policy:
    max_position_notional_usdc: 100.0
    max_daily_drawdown_pct: 2.5
    min_order_size_usdc: 1.0
  date_range_start: "2026-04-01T00:00:00+00:00"
  date_range_end: "2026-04-30T00:00:00+00:00"
exec_config:
  chunk_days: 3
  time_budget: 900
parameter_grid:
  strategy_versions:
    - [["alpha", "alpha-v1"]]
    - [["beta", "beta-v1"]]
""".strip()


@pytest.mark.asyncio
async def test_enqueue_backtest_runs_parses_yaml_and_returns_run_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    fake_pool = object()

    class FakeSweep:
        def __init__(self, *, pool: object) -> None:
            observed["pool"] = pool

        def enumerate_variants(
            self,
            base_spec: object,
            parameter_grid: dict[str, object],
        ) -> list[object]:
            observed["base_spec"] = base_spec
            observed["parameter_grid"] = parameter_grid
            return [base_spec, base_spec]

        async def enqueue(
            self,
            specs: list[object],
            exec_config: object,
        ) -> list[QueuedSweepRun]:
            observed["specs"] = specs
            observed["exec_config"] = exec_config
            return [
                QueuedSweepRun(run_id="run-1", spec_hash="hash-1", inserted=True),
                QueuedSweepRun(run_id="run-2", spec_hash="hash-2", inserted=True),
            ]

    monkeypatch.setattr("pms.api.research_routes.ParameterSweep", FakeSweep)

    payload = await enqueue_backtest_runs(fake_pool, _SWEEP_YAML)

    assert observed["pool"] is fake_pool
    assert observed["base_spec"].strategy_versions == (("alpha", "alpha-v1"),)
    assert observed["parameter_grid"] == {
        "strategy_versions": (
            [["alpha", "alpha-v1"]],
            [["beta", "beta-v1"]],
        )
    }
    assert observed["exec_config"].chunk_days == 3
    assert observed["exec_config"].time_budget == 900
    assert observed["specs"] == [observed["base_spec"], observed["base_spec"]]
    assert payload == {
        "run_ids": ["run-1", "run-2"],
        "unique_run_count": 2,
        "runs": [
            {"run_id": "run-1", "spec_hash": "hash-1", "inserted": True},
            {"run_id": "run-2", "spec_hash": "hash-2", "inserted": True},
        ],
    }


@dataclass
class _FakeConnection:
    running_rows: list[dict[str, object]]
    executed: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        del args
        assert "FROM backtest_runs" in query
        return list(self.running_rows)

    async def execute(self, query: str, *args: object) -> str:
        self.executed.append((query, args))
        return "UPDATE 1"


class _FakeAcquireContext:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _FakeConnection:
        return self._connection

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


@dataclass
class _FakePool:
    connection: _FakeConnection

    def acquire(self) -> _FakeAcquireContext:
        return _FakeAcquireContext(self.connection)


@pytest.mark.asyncio
async def test_scan_orphaned_backtest_runs_marks_dead_workers_failed() -> None:
    connection = _FakeConnection(
        running_rows=[
            {"run_id": "run-dead", "worker_pid": 999999},
            {"run_id": "run-alive", "worker_pid": 1234},
        ]
    )
    seen_pids: list[int] = []

    def fake_pid_probe(pid: int) -> None:
        seen_pids.append(pid)
        if pid == 999999:
            raise ProcessLookupError(pid)

    await scan_orphaned_backtest_runs(
        _FakePool(connection),
        pid_probe=fake_pid_probe,
    )

    assert seen_pids == [999999, 1234]
    assert len(connection.executed) == 1
    update_sql, update_args = connection.executed[0]
    assert "UPDATE backtest_runs" in update_sql
    assert update_args == ("orphaned (worker process gone)", "run-dead")


def test_api_package_does_not_reference_backtest_runner() -> None:
    api_sources = sorted(Path("src/pms/api").rglob("*.py"))
    assert api_sources
    for path in api_sources:
        assert "BacktestRunner" not in path.read_text(encoding="utf-8")


def test_record_to_json_only_decodes_known_json_columns() -> None:
    payload = _record_to_json(
        {
            "spec_json": '{"strategy_versions":[["alpha","alpha-v1"]]}',
            "failure_reason": '{"unterminated"',
            "queued_at": datetime(2026, 4, 20, 12, 0, tzinfo=UTC),
        }
    )

    assert payload == {
        "spec_json": {"strategy_versions": [["alpha", "alpha-v1"]]},
        "failure_reason": '{"unterminated"',
        "queued_at": "2026-04-20T12:00:00+00:00",
    }


@pytest.mark.asyncio
async def test_enqueue_backtest_runs_rejects_non_mapping_yaml() -> None:
    # malformed YAML (empty) — yaml.safe_load returns None, not a mapping
    with pytest.raises(TypeError, match="mapping"):
        await enqueue_backtest_runs(cast(Any, object()), "")


@pytest.mark.asyncio
async def test_enqueue_backtest_runs_rejects_list_top_level() -> None:
    with pytest.raises(TypeError, match="mapping"):
        await enqueue_backtest_runs(cast(Any, object()), "- a\n- b\n")


@pytest.mark.asyncio
async def test_compute_backtest_live_comparison_rejects_non_mapping_body() -> None:
    with pytest.raises(TypeError, match="mapping"):
        await compute_backtest_live_comparison(cast(Any, object()), "run-1", [])


@pytest.mark.asyncio
async def test_compute_backtest_live_comparison_rejects_missing_denominator() -> None:
    body = {
        "live_window_start": "2026-04-01T00:00:00+00:00",
        "live_window_end": "2026-04-02T00:00:00+00:00",
    }
    with pytest.raises(ValueError, match="denominator"):
        await compute_backtest_live_comparison(cast(Any, object()), "run-1", body)


@pytest.mark.asyncio
async def test_compute_backtest_live_comparison_rejects_tz_naive_window() -> None:
    body = {
        "live_window_start": "2026-04-01T00:00:00",
        "live_window_end": "2026-04-02T00:00:00+00:00",
        "denominator": "backtest_set",
    }
    with pytest.raises(ValueError, match="timezone-aware"):
        await compute_backtest_live_comparison(cast(Any, object()), "run-1", body)


class _FetchRowConnection:
    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        del query, args
        return None


class _FetchRowContext:
    async def __aenter__(self) -> _FetchRowConnection:
        return _FetchRowConnection()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class _FetchRowPool:
    def acquire(self) -> _FetchRowContext:
        return _FetchRowContext()


@pytest.mark.asyncio
async def test_fetch_backtest_run_returns_none_for_unknown_run_id() -> None:
    result = await fetch_backtest_run(cast(Any, _FetchRowPool()), "missing-run")
    assert result is None


@pytest.mark.asyncio
async def test_create_backtest_run_returns_503_when_pg_pool_unset() -> None:
    # No pg_pool bound and auto_start=False means the lifespan contextmanager
    # is the only codepath that might create one. httpx.ASGITransport does not
    # invoke lifespan by default, so pg_pool stays None and the route yields 503.
    runner = Runner()
    app = create_app(runner, auto_start=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/research/backtest", content="")
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_create_backtest_run_returns_422_for_malformed_yaml_or_json() -> None:
    runner = Runner()
    runner.bind_pg_pool(cast(asyncpg.Pool, object()))
    app = create_app(runner, auto_start=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yaml_response = await client.post(
            "/research/backtest",
            content="base_spec: [",
            headers={"content-type": "text/plain"},
        )
        json_response = await client.post(
            "/research/backtest",
            content='{"sweep_yaml":',
            headers={"content-type": "application/json"},
        )
    assert yaml_response.status_code == 422
    assert json_response.status_code == 422


@pytest.mark.asyncio
async def test_get_backtest_run_returns_503_when_pg_pool_unset() -> None:
    runner = Runner()
    app = create_app(runner, auto_start=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/research/backtest/11111111-1111-1111-1111-111111111111"
        )
    assert response.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/research/backtest/not-a-uuid",
        "/research/backtest/not-a-uuid/strategies",
        "/research/backtest/not-a-uuid/compare",
    ],
)
async def test_research_run_routes_return_422_for_invalid_run_id(path: str) -> None:
    runner = Runner()
    runner.bind_pg_pool(cast(asyncpg.Pool, object()))
    app = create_app(runner, auto_start=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(path) if path.endswith("strategies") else await client.post(path, json={}) if path.endswith("compare") else await client.get(path)
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/research/backtest/11111111-1111-1111-1111-111111111111/strategies",
        "/research/backtest/11111111-1111-1111-1111-111111111111/compare",
    ],
)
async def test_research_run_routes_return_404_for_unknown_run_id(path: str) -> None:
    runner = Runner()
    runner.bind_pg_pool(cast(asyncpg.Pool, _FetchRowPool()))
    app = create_app(runner, auto_start=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(path) if path.endswith("strategies") else await client.post(path, json={})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_compare_backtest_run_returns_503_when_pg_pool_unset() -> None:
    runner = Runner()
    app = create_app(runner, auto_start=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/research/backtest/11111111-1111-1111-1111-111111111111/compare",
            json={"denominator": "backtest_set"},
        )
    assert response.status_code == 503
