from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import asyncpg
import httpx
import pytest

from pms.api.app import create_app
from pms.config import ControllerSettings, DatabaseSettings, PMSSettings
from pms.controller.factory import ControllerPipelineFactory
from pms.research.runner import BacktestRunner
from pms.runner import Runner
from tests.integration.test_research_runner_cp04b import (
    StaticReplayEngine,
    _active_strategy,
    _load_strategies,
    _signal,
)


PMS_TEST_DATABASE_URL = os.environ.get("PMS_TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("PMS_RUN_INTEGRATION") != "1",
        reason="set PMS_RUN_INTEGRATION=1 to run PostgreSQL integration tests",
    ),
    pytest.mark.skipif(
        PMS_TEST_DATABASE_URL is None,
        reason="set PMS_TEST_DATABASE_URL to the compose-backed PostgreSQL URI",
    ),
]


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
  chunk_days: 7
  time_budget: 1800
parameter_grid:
  strategy_versions:
    - [["alpha", "alpha-v1"]]
    - [["beta", "beta-v1"]]
""".strip()


def _settings() -> PMSSettings:
    return PMSSettings(
        database=DatabaseSettings(dsn=cast(str, PMS_TEST_DATABASE_URL)),
    )


def _app_client(pg_pool: asyncpg.Pool) -> httpx.AsyncClient:
    runner = Runner(config=_settings())
    runner.bind_pg_pool(pg_pool)
    app = create_app(runner)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


async def _insert_strategy_run(
    pool: asyncpg.Pool, *, run_id: str, rejection_counts: dict[str, int]
) -> None:
    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO strategy_runs (
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
                finished_at,
                controller_rejection_counts_json
            ) VALUES (
                $1::uuid,
                $2::uuid,
                $3,
                $4,
                $5,
                $6,
                $7,
                $8,
                $9,
                $10,
                $11,
                $12,
                $13::jsonb,
                $14,
                $15,
                $16::jsonb
            )
            """,
            str(uuid4()),
            run_id,
            "alpha",
            "alpha-v1",
            0.14,
            12.5,
            1.0,
            0.75,
            8.0,
            4,
            3,
            2,
            json.dumps([]),
            datetime(2026, 4, 10, 9, 0, tzinfo=UTC),
            datetime(2026, 4, 10, 10, 0, tzinfo=UTC),
            json.dumps(rejection_counts),
        )


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("rejection_counts", [
    {},
    {"router_gate:yes_price_out_of_band": 57, "calibration_clamp_rejected": 90},
])
async def test_research_backtest_post_and_get_routes_roundtrip(
    pg_pool: asyncpg.Pool,
    rejection_counts: dict[str, int],
) -> None:
    async with _app_client(pg_pool) as client:
        create_response = await client.post(
            "/research/backtest",
            content=_SWEEP_YAML,
            headers={"content-type": "application/x-yaml"},
        )

    assert create_response.status_code == 200
    payload = create_response.json()
    assert len(payload["run_ids"]) == 2
    run_id = payload["run_ids"][0]

    async with pg_pool.acquire() as connection:
        queued_count = await connection.fetchval(
            "SELECT COUNT(*) FROM backtest_runs WHERE run_id = ANY($1::uuid[])",
            payload["run_ids"],
        )
    assert queued_count == 2

    async with _app_client(pg_pool) as client:
        run_response = await client.get(f"/research/backtest/{run_id}")

    assert run_response.status_code == 200
    assert run_response.json()["run_id"] == run_id
    assert run_response.json()["status"] == "queued"
    assert run_response.json()["started_at"] is None
    assert run_response.json()["finished_at"] is None
    assert run_response.json()["failure_reason"] is None

    await _insert_strategy_run(pg_pool, run_id=run_id, rejection_counts=rejection_counts)

    async with _app_client(pg_pool) as client:
        strategies_response = await client.get(f"/research/backtest/{run_id}/strategies")

    assert strategies_response.status_code == 200
    assert strategies_response.json() == [
        {
            "strategy_run_id": strategies_response.json()[0]["strategy_run_id"],
            "run_id": run_id,
            "strategy_id": "alpha",
            "strategy_version_id": "alpha-v1",
            "brier": 0.14,
            "pnl_cum": 12.5,
            "drawdown_max": 1.0,
            "fill_rate": 0.75,
            "slippage_bps": 8.0,
            "opportunity_count": 4,
            "decision_count": 3,
            "fill_count": 2,
            "portfolio_target_json": [],
            "controller_rejection_counts": rejection_counts,
            "started_at": "2026-04-10T09:00:00+00:00",
            "finished_at": "2026-04-10T10:00:00+00:00",
        }
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_zero_trade_backtest_persists_controller_rejections_through_api(
    pg_pool: asyncpg.Pool,
) -> None:
    async with _app_client(pg_pool) as client:
        response = await client.post(
            "/research/backtest",
            content=_SWEEP_YAML,
            headers={"content-type": "application/x-yaml"},
        )
    assert response.status_code == 200
    run_id = response.json()["run_ids"][0]
    strategy_key = ("alpha", "alpha-v1")
    strategies = {
        strategy_key: _active_strategy(
            strategy_id=strategy_key[0], strategy_version_id=strategy_key[1],
        ),
    }
    backtest = BacktestRunner(
        writable_pool=pg_pool,
        readonly_pool=pg_pool,
        replay_engine=StaticReplayEngine([
            replace(_signal(), volume_24h=0.0),
            replace(_signal(), volume_24h=0.0),
            replace(_signal(), market_status="closed"),
        ]),
        strategy_loader=lambda keys: _load_strategies(strategies, keys),
        controller_factory=ControllerPipelineFactory(
            settings=_settings().model_copy(
                update={"controller": ControllerSettings(min_volume=500.0)},
            ),
        ),
    )
    assert await backtest.execute(run_id) is True
    expected_counts = {
        "router_gate:min_volume_too_low": 2,
        "router_gate:market_status_not_open": 1,
    }
    async with pg_pool.acquire() as connection:
        rows = await connection.fetch(
            "SELECT controller_rejection_counts_json FROM strategy_runs "
            "WHERE run_id = $1::uuid",
            run_id,
        )
        execution_count = await connection.fetchval(
            "SELECT COUNT(*) FROM backtest_execution_rows WHERE run_id = $1::uuid",
            run_id,
        )
    assert len(rows) == 1
    assert json.loads(rows[0]["controller_rejection_counts_json"]) == expected_counts
    assert execution_count == 0

    async with _app_client(pg_pool) as client:
        run_response = await client.get(f"/research/backtest/{run_id}")
        strategies_response = await client.get(f"/research/backtest/{run_id}/strategies")
    assert run_response.status_code == 200
    assert run_response.json()["status"] == "completed"
    assert run_response.json()["failure_reason"] is None
    assert strategies_response.status_code == 200
    results = strategies_response.json()
    assert len(results) == 1
    result = results[0]
    assert result["controller_rejection_counts"] == expected_counts
    assert result["opportunity_count"] == 0
    assert result["decision_count"] == 0
    assert result["fill_count"] == 0
    assert result["brier"] is None
    assert result["pnl_cum"] is None


@pytest.mark.asyncio(loop_scope="session")
async def test_research_backtest_orphan_scan_marks_dead_workers_failed(
    pg_pool: asyncpg.Pool,
) -> None:
    run_id = str(uuid4())
    async with pg_pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO backtest_runs (
                run_id,
                spec_hash,
                status,
                strategy_ids,
                date_range_start,
                date_range_end,
                exec_config_json,
                spec_json,
                started_at,
                worker_pid
            ) VALUES (
                $1::uuid,
                $2,
                'running',
                $3::text[],
                $4,
                $5,
                $6::jsonb,
                $7::jsonb,
                $8,
                $9
            )
            """,
            run_id,
            f"spec-{run_id}",
            ["alpha"],
            datetime(2026, 4, 1, tzinfo=UTC),
            datetime(2026, 4, 30, tzinfo=UTC),
            json.dumps({"chunk_days": 7, "time_budget": 1800}),
            json.dumps({"strategy_versions": [["alpha", "alpha-v1"]]}),
            datetime(2026, 4, 5, 12, 0, tzinfo=UTC),
            999999,
        )

    runner = Runner(config=_settings())
    runner.bind_pg_pool(pg_pool)
    app = create_app(runner, auto_start=False)

    async with app.router.lifespan_context(app):
        async with pg_pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT status, failure_reason
                FROM backtest_runs
                WHERE run_id = $1::uuid
                """,
                run_id,
            )

    assert row is not None
    assert row["status"] == "failed"
    assert row["failure_reason"] == "orphaned (worker process gone)"
