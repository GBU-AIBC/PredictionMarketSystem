from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import cast

import asyncpg
import httpx
import pytest

from pms.api.app import create_app
from pms.config import DatabaseSettings, PMSSettings
from pms.runner import Runner


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


async def _insert_backtest_run(
    pool: asyncpg.Pool,
    *,
    run_id: str,
    strategy_ids: list[str],
    queued_at: datetime,
) -> None:
    async with pool.acquire() as connection:
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
                queued_at,
                started_at,
                finished_at
            ) VALUES (
                $1::uuid,
                $2,
                'completed',
                $3::text[],
                $4,
                $5,
                $6::jsonb,
                $7::jsonb,
                $8,
                $9,
                $10
            )
            """,
            run_id,
            f"spec-{run_id}",
            strategy_ids,
            datetime(2026, 4, 1, tzinfo=UTC),
            datetime(2026, 4, 30, tzinfo=UTC),
            json.dumps({"chunk_days": 7, "time_budget": 1800}),
            json.dumps(
                {
                    "strategy_versions": [[strategy_id, f"{strategy_id}-v1"] for strategy_id in strategy_ids]
                }
            ),
            queued_at,
            queued_at,
            queued_at,
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_research_backtest_list_route_returns_recent_runs(
    pg_pool: asyncpg.Pool,
) -> None:
    older_run_id = "11111111-1111-1111-1111-111111111111"
    newer_run_id = "22222222-2222-2222-2222-222222222222"
    await _insert_backtest_run(
        pg_pool,
        run_id=older_run_id,
        strategy_ids=["alpha"],
        queued_at=datetime(2026, 4, 10, 10, 0, tzinfo=UTC),
    )
    await _insert_backtest_run(
        pg_pool,
        run_id=newer_run_id,
        strategy_ids=["alpha", "beta"],
        queued_at=datetime(2026, 4, 11, 10, 0, tzinfo=UTC),
    )

    async with _app_client(pg_pool) as client:
        response = await client.get("/research/backtest")

    assert response.status_code == 200
    assert response.json() == [
        {
            "run_id": newer_run_id,
            "spec_hash": f"spec-{newer_run_id}",
            "status": "completed",
            "strategy_ids": ["alpha", "beta"],
            "date_range_start": "2026-04-01T00:00:00+00:00",
            "date_range_end": "2026-04-30T00:00:00+00:00",
            "exec_config_json": {"chunk_days": 7, "time_budget": 1800},
            "spec_json": {"strategy_versions": [["alpha", "alpha-v1"], ["beta", "beta-v1"]]},
            "queued_at": "2026-04-11T10:00:00+00:00",
            "started_at": "2026-04-11T10:00:00+00:00",
            "finished_at": "2026-04-11T10:00:00+00:00",
            "failure_reason": None,
            "worker_pid": None,
            "worker_host": None,
        },
        {
            "run_id": older_run_id,
            "spec_hash": f"spec-{older_run_id}",
            "status": "completed",
            "strategy_ids": ["alpha"],
            "date_range_start": "2026-04-01T00:00:00+00:00",
            "date_range_end": "2026-04-30T00:00:00+00:00",
            "exec_config_json": {"chunk_days": 7, "time_budget": 1800},
            "spec_json": {"strategy_versions": [["alpha", "alpha-v1"]]},
            "queued_at": "2026-04-10T10:00:00+00:00",
            "started_at": "2026-04-10T10:00:00+00:00",
            "finished_at": "2026-04-10T10:00:00+00:00",
            "failure_reason": None,
            "worker_pid": None,
            "worker_host": None,
        },
    ]


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "run_id",
    [
        "{99999999-9999-9999-9999-999999999999}",
        "urn:uuid:99999999-9999-9999-9999-999999999999",
        "uuid:99999999999999999999999999999999",
        "not-a-uuid",
    ],
)
async def test_research_run_id_spellings_are_typed_4xx(
    pg_pool: asyncpg.Pool,
    run_id: str,
) -> None:
    async with _app_client(pg_pool) as client:
        detail = await client.get(f"/research/backtest/{run_id}")
        strategies = await client.get(f"/research/backtest/{run_id}/strategies")
        comparison = await client.post(
            f"/research/backtest/{run_id}/compare",
            json={},
        )

    expected_status = 404 if run_id.startswith(("{", "urn:")) else 422
    assert detail.status_code == expected_status
    assert strategies.status_code == expected_status
    assert comparison.status_code == expected_status
