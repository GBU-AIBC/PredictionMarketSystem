from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import asyncpg
from fastapi import FastAPI, Request
import httpx
import pytest
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from pms.api.routes.agent_evidence import router as evidence_router
from pms.api.routes.agent_factors import router as factor_router
from pms.config import DatabaseSettings, LLMSettings, PMSSettings
from pms.controller.factor_snapshot import PostgresFactorSnapshotReader
from pms.controller.factory import ControllerPipelineFactory
from pms.core.enums import RunMode
from pms.core.models import MarketSignal, Portfolio
from pms.factors.catalog import seed_factor_catalog
from pms.storage.agent_evidence_store import (
    AgentEvidenceStore,
    canonical_content_hash,
)
from pms.runner import Runner
from pms.strategies.intents import StrategyContext
from pms.strategies.paper_multifactor import build_paper_agent_probability_strategy


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        __import__("os").environ.get("PMS_RUN_INTEGRATION") != "1",
        reason="set PMS_RUN_INTEGRATION=1 to run PostgreSQL integration tests",
    ),
]


def _api(pool: asyncpg.Pool) -> FastAPI:
    app = FastAPI()
    app.state.runner = SimpleNamespace(pg_pool=pool)

    @app.middleware("http")
    async def token_context(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request.state.token_id = "integration-token"
        return await call_next(request)

    app.include_router(evidence_router)
    app.include_router(factor_router)
    return app


async def _seed_market(pool: asyncpg.Pool, market_id: str) -> None:
    now = datetime.now(tz=UTC)
    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO markets (condition_id, slug, question, venue, created_at, last_seen_at)
            VALUES ($1, $1, $2, 'polymarket', $3, $3)
            """,
            market_id,
            "Will the agent path work?",
            now,
        )
        await seed_factor_catalog(connection, factor_ids=("agent_probability",))


@pytest.mark.asyncio(loop_scope="session")
async def test_agent_evidence_and_factor_routes_round_trip_with_pit_and_provenance(
    pg_pool: asyncpg.Pool,
) -> None:
    market_id = "w2b-route-market"
    await _seed_market(pg_pool, market_id)
    now = datetime.now(tz=UTC).replace(microsecond=0)
    captured_at = now - timedelta(seconds=1)
    expires_at = now + timedelta(minutes=5)
    payload = {"probability": 0.8}
    content_hash = canonical_content_hash(
        market_id=market_id,
        source="agent",
        model="model-v1",
        url="https://example.test/evidence",
        summary="agent summary",
        payload=payload,
        captured_at=captured_at,
        expires_at=expires_at,
    )
    body = {
        "market_id": market_id,
        "source": "agent",
        "model": "model-v1",
        "url": "https://example.test/evidence",
        "summary": "agent summary",
        "payload": payload,
        "content_hash": content_hash,
        "captured_at": captured_at.isoformat(),
        "expires_at": expires_at.isoformat(),
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_api(pg_pool)),
        base_url="http://test",
    ) as client:
        evidence_response = await client.post("/evidence", json=body)
        assert evidence_response.status_code == 201, evidence_response.text
        duplicate_response = await client.post("/evidence", json=body)
        assert duplicate_response.status_code == 201
        assert duplicate_response.json()["evidence_id"] == evidence_response.json()["evidence_id"]
        listed = await client.get("/evidence", params={"market_id": market_id})
        assert listed.status_code == 200
        assert len(listed.json()["evidence"]) == 1

        factor_response = await client.post(
            "/factors/agent_probability/observations",
            json={
                "market_id": market_id,
                "value": 0.8,
                "ts": now.isoformat(),
                "model": "model-v1",
                "prompt_hash": "a" * 64,
                "evidence_ids": [evidence_response.json()["evidence_id"]],
            },
        )
        assert factor_response.status_code == 201, factor_response.text
        conflicting_provenance = await client.post(
            "/factors/agent_probability/observations",
            json={
                "market_id": market_id,
                "value": 0.8,
                "ts": now.isoformat(),
                "model": "model-other",
                "prompt_hash": "e" * 64,
            },
        )
        assert conflicting_provenance.status_code == 201
        assert conflicting_provenance.json()["provenance_written"] is False
        provenance = await client.get(
            "/factors/agent_probability/observations/provenance",
            params={"market_id": market_id, "ts": now.isoformat()},
        )
        assert provenance.status_code == 200
        assert provenance.json()["model"] == "model-v1"
        future_factor = await client.post(
            "/factors/agent_probability/observations",
            json={
                "market_id": market_id,
                "value": 0.7,
                "ts": (now + timedelta(seconds=1)).isoformat(),
                "model": "model-v1",
                "prompt_hash": "b" * 64,
            },
        )
        assert future_factor.status_code == 422
        stale_factor = await client.post(
            "/factors/agent_probability/observations",
            json={
                "market_id": market_id,
                "value": 0.7,
                "ts": (now - timedelta(seconds=121)).isoformat(),
                "model": "model-v1",
                "prompt_hash": "c" * 64,
            },
        )
        assert stale_factor.status_code == 422
        missing_factor = await client.post(
            "/factors/agent_probability/observations",
            json={
                "market_id": "missing-market",
                "value": 0.7,
                "ts": now.isoformat(),
                "model": "model-v1",
                "prompt_hash": "f" * 64,
            },
        )
        assert missing_factor.status_code == 422
        missing_evidence = dict(body)
        missing_evidence["market_id"] = "missing-market"
        missing_evidence["content_hash"] = canonical_content_hash(
            market_id="missing-market",
            source="agent",
            model="model-v1",
            url="https://example.test/evidence",
            summary="agent summary",
            payload=payload,
            captured_at=captured_at,
            expires_at=expires_at,
        )
        assert (await client.post("/evidence", json=missing_evidence)).status_code == 422

    store = AgentEvidenceStore(pg_pool)
    assert await store.list_for_market(market_id, as_of=expires_at) == []
    assert len(await store.list_for_market(market_id, as_of=datetime.now(tz=UTC))) == 1
    runner = Runner(
        config=PMSSettings(
            database=DatabaseSettings(
                dsn="postgres://postgres:postgres@127.0.0.1:55432/pms_w2b"
            )
        )
    )
    runner.bind_pg_pool(pg_pool)
    observations = await runner.build_agent_evidence_source((market_id,)).observe(
        StrategyContext(
            strategy_id="paper-agent",
            strategy_version_id="paper-agent-v1",
            as_of=datetime.now(tz=UTC),
        )
    )
    assert len(observations) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_agent_factor_paper_decision_changes_and_live_refuses(
    pg_pool: asyncpg.Pool,
) -> None:
    market_id = "w2b-decision-market"
    await _seed_market(pg_pool, market_id)
    now = datetime.now(tz=UTC).replace(microsecond=0)

    strategy = build_paper_agent_probability_strategy()
    active = strategy.to_active(strategy_version_id="paper-agent-v1")
    signal = MarketSignal(
        market_id=market_id,
        token_id="token-yes",
        venue="polymarket",
        title="Will the paper decision change?",
        yes_price=0.5,
        volume_24h=1_000.0,
        resolves_at=None,
        orderbook={},
        external_signal={},
        fetched_at=now,
        market_status="open",
    )
    settings = PMSSettings(
        mode=RunMode.PAPER,
        database=DatabaseSettings(dsn="postgres://postgres:postgres@127.0.0.1:55432/pms_w2b"),
        llm=LLMSettings(enabled=False),
    )
    reader = PostgresFactorSnapshotReader(pg_pool)
    pipeline = ControllerPipelineFactory(settings=settings, factor_reader=reader).build(active)
    portfolio = Portfolio(
        total_usdc=1_000.0,
        free_usdc=1_000.0,
        locked_usdc=0.0,
        open_positions=[],
    )
    before = await pipeline.on_signal(signal, portfolio=portfolio)

    async with pg_pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO factor_values (factor_id, param, market_id, ts, value, definition_version)
            VALUES ('agent_probability', '', $1, $2, 0.8, '1')
            """,
            market_id,
            now,
        )
        await connection.execute(
            """
            INSERT INTO agent_factor_provenance (
                factor_id, param, market_id, ts, definition_version,
                model, prompt_hash, evidence_ids, submitted_by_token_id, created_at
            ) VALUES ('agent_probability', '', $1, $2, '1', 'model-v1', $3, '[]'::jsonb, 'integration-token', $4)
            """,
            market_id,
            now,
            "d" * 64,
            now - timedelta(seconds=1),
        )
    after = await pipeline.on_signal(signal, portfolio=portfolio)
    assert before is None
    assert after is not None

    live_settings = settings.model_copy(update={"mode": RunMode.LIVE})
    with pytest.raises(ValueError, match="agent-owned factors"):
        ControllerPipelineFactory(settings=live_settings, factor_reader=reader).build(active)
