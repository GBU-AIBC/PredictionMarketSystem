from __future__ import annotations

from datetime import UTC, datetime, timedelta
import os
from typing import Any, cast
from uuid import uuid4

import asyncpg
import httpx
import pytest

from pms.agent_mcp.server import MCPConfig, MCPError, PMSMCPClient, create_server
from pms.api.app import create_app
from pms.config import PMSSettings
from pms.core.enums import RunMode
from pms.factors.catalog import seed_factor_catalog
from pms.runner import Runner
from pms.storage.agent_evidence_store import canonical_content_hash


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("PMS_RUN_INTEGRATION") != "1",
        reason="set PMS_RUN_INTEGRATION=1 to run PostgreSQL integration tests",
    ),
]


async def _seed_market(pool: asyncpg.Pool, market_id: str) -> None:
    now = datetime.now(tz=UTC)
    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO markets (condition_id, slug, question, venue, created_at, last_seen_at)
            VALUES ($1, $1, $2, 'polymarket', $3, $3)
            """,
            market_id,
            "Will the MCP contribution path work?",
            now,
        )
        await seed_factor_catalog(connection, factor_ids=("agent_probability",))


def _app(pool: asyncpg.Pool, mode: RunMode, *, token_env: str) -> Any:
    settings = PMSSettings(
        mode=mode,
        api_tokens=cast(Any, [{"token_id": "agent", "secret": token_env, "scopes": ["paper_operate"]}]),
        auto_migrate_default_v2=False,
    )
    runner = Runner(config=settings)
    runner.bind_pg_pool(pool)
    return create_app(runner, auto_start=False)


@pytest.mark.asyncio(loop_scope="session")
async def test_mcp_contribution_tools_are_scoped_and_live_refused(
    pg_pool: asyncpg.Pool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mcp")
    monkeypatch.setenv("MCP_PAPER_SECRET", "paper-secret")
    monkeypatch.setenv("MCP_READ_SECRET", "read-secret")
    market_id = f"mcp-w2c-{uuid4().hex}"
    await _seed_market(pg_pool, market_id)
    paper_app = _app(pg_pool, RunMode.PAPER, token_env="MCP_PAPER_SECRET")
    now = datetime.now(tz=UTC).replace(microsecond=0)
    captured_at = now - timedelta(seconds=1)
    expires_at = now + timedelta(minutes=5)
    payload = {"probability": 0.8}
    evidence_body = {
        "market_id": market_id,
        "source": "mcp-agent",
        "model": "model-v1",
        "url": "https://example.test/mcp",
        "summary": "MCP contribution",
        "payload": payload,
        "content_hash": canonical_content_hash(
            market_id=market_id,
            source="mcp-agent",
            model="model-v1",
            url="https://example.test/mcp",
            summary="MCP contribution",
            payload=payload,
            captured_at=captured_at,
            expires_at=expires_at,
        ),
        "captured_at": captured_at,
        "expires_at": expires_at,
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=paper_app), base_url="http://test") as paper_http:
        paper_client = PMSMCPClient(MCPConfig("http://test", "paper-secret"), http_client=paper_http)
        openapi = await paper_client.fetch_openapi()
        server = await create_server(paper_client, openapi=openapi)
        assert {tool.name for tool in await server.list_tools()} >= {"submit_evidence", "submit_agent_factor"}
        evidence = await paper_client.submit_evidence(**evidence_body)
        assert evidence["market_id"] == market_id
        factor_ts = datetime.now(tz=UTC)
        factor = await paper_client.submit_agent_factor(
            factor_id="agent_probability",
            market_id=market_id,
            value=0.8,
            ts=factor_ts,
            model="model-v1",
            prompt_hash="a" * 64,
            evidence_ids=[evidence["evidence_id"]],
        )
        assert factor["factor_id"] == "agent_probability"

    read_settings = PMSSettings(
        mode=RunMode.PAPER,
        api_tokens=cast(Any, [{"token_id": "read", "secret": "MCP_READ_SECRET", "scopes": ["read"]}]),
        auto_migrate_default_v2=False,
    )
    read_runner = Runner(config=read_settings)
    read_runner.bind_pg_pool(pg_pool)
    read_app = create_app(read_runner, auto_start=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=read_app), base_url="http://test") as read_http:
        read_client = PMSMCPClient(MCPConfig("http://test", "read-secret"), http_client=read_http)
        with pytest.raises(MCPError) as evidence_forbidden:
            await read_client.submit_evidence(**evidence_body)
        assert evidence_forbidden.value.status_code == 403
        with pytest.raises(MCPError) as factor_forbidden:
            await read_client.submit_agent_factor(
                factor_id="agent_probability",
                market_id=market_id,
                value=0.8,
                ts=factor_ts,
                model="model-v1",
                prompt_hash="b" * 64,
            )
        assert factor_forbidden.value.status_code == 403

    live_app = _app(pg_pool, RunMode.LIVE, token_env="MCP_PAPER_SECRET")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=live_app), base_url="http://test") as live_http:
        live_client = PMSMCPClient(MCPConfig("http://test", "paper-secret"), http_client=live_http)
        with pytest.raises(MCPError) as live_evidence:
            await live_client.submit_evidence(**evidence_body)
        assert live_evidence.value.status_code == 403
        with pytest.raises(MCPError) as live_factor:
            await live_client.submit_agent_factor(
                factor_id="agent_probability",
                market_id=market_id,
                value=0.8,
                ts=factor_ts,
                model="model-v1",
                prompt_hash="c" * 64,
            )
        assert live_factor.value.status_code == 403
