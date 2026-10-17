from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from uuid import uuid4

import asyncpg
import httpx
import pytest

from pms.api.app import create_app
from pms.config import PMSSettings
from pms.runner import Runner


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("PMS_RUN_INTEGRATION") != "1",
        reason="set PMS_RUN_INTEGRATION=1 to run PostgreSQL integration tests",
    ),
    pytest.mark.skipif(
        os.environ.get("PMS_TEST_DATABASE_URL") is None,
        reason="set PMS_TEST_DATABASE_URL to the compose-backed PostgreSQL URI",
    ),
]


@pytest.mark.asyncio(loop_scope="session")
async def test_evaluation_report_routes_decode_jsonb_rows(pg_pool: asyncpg.Pool) -> None:
    run_id = uuid4()
    report_id = uuid4()
    async with pg_pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO backtest_runs (
                run_id, spec_hash, status, strategy_ids,
                date_range_start, date_range_end, exec_config_json, spec_json,
                started_at, finished_at
            ) VALUES ($1, 'w2a-report', 'completed', ARRAY['default']::text[],
                      $2, $3, '{}'::jsonb, '{}'::jsonb, $2, $3)
            """,
            run_id,
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
        )
        await connection.execute(
            """
            INSERT INTO evaluation_reports (
                report_id, run_id, ranking_metric, ranked_strategies,
                benchmark_rows, attribution_commentary, warnings, next_action,
                generated_at
            ) VALUES ($1, $2, 'brier', $3::jsonb, $4::jsonb, 'ok', $5::jsonb,
                      'observe', now())
            """,
            report_id,
            run_id,
            json.dumps([{"strategy_id": "default", "score": 0.2}]),
            json.dumps([]),
            json.dumps(["none"]),
        )
    runner = Runner(config=PMSSettings(auto_migrate_default_v2=False))
    runner.bind_pg_pool(pg_pool)
    app = create_app(runner, auto_start=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        listing = await client.get("/evaluation/reports")
        detail = await client.get(f"/evaluation/reports/{report_id}")
    assert listing.status_code == 200
    assert detail.status_code == 200
    assert detail.json()["ranked_strategies"][0]["strategy_id"] == "default"
    assert detail.json()["warnings"] == ["none"]
