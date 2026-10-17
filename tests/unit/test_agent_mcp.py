from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any

import httpx
import pytest

from pms.agent_mcp.server import (
    MCPConfig,
    PMSMCPClient,
    TOOL_SPECS,
    create_server,
)


NOW = datetime(2026, 10, 18, 0, 0, tzinfo=UTC)


def _status() -> dict[str, Any]:
    return {
        "mode": "paper",
        "running": True,
        "healthy": True,
        "workers": {},
        "autostart_attempted": False,
        "controller": {},
        "actuator": {},
        "evaluator": {},
        "supervision": {},
        "quality": {},
    }


def _market() -> dict[str, Any]:
    return {
        "market_id": "m1",
        "question": "Will it happen?",
        "venue": "polymarket",
        "volume_24h": 1.0,
        "updated_at": NOW.isoformat(),
        "yes_token_id": "yes",
        "no_token_id": "no",
        "subscribed": False,
    }


def _signal() -> dict[str, Any]:
    return {
        "market_id": "m1",
        "token_id": "yes",
        "venue": "polymarket",
        "title": "Will it happen?",
        "yes_price": 0.5,
        "volume_24h": 1.0,
        "resolves_at": None,
        "orderbook": {},
        "external_signal": {},
        "fetched_at": NOW.isoformat(),
        "market_status": "open",
    }


def _handler(requests: list[httpx.Request]) -> Any:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/status":
            return httpx.Response(200, json=_status())
        if path == "/markets":
            return httpx.Response(200, json={"markets": [_market()], "limit": 20, "offset": 0, "total": 1})
        if path == "/selection/explain":
            return httpx.Response(200, json={"market_id": "m1", "available": True})
        if path == "/signals":
            return httpx.Response(200, json=[_signal()])
        if path == "/factors/catalog":
            return httpx.Response(200, json={"catalog": []})
        if path == "/factors":
            return httpx.Response(200, json={"factor_id": "agent_probability", "param": "", "market_id": "m1", "points": []})
        if path == "/evidence":
            return httpx.Response(200, json={
                "evidence_id": "e1", "market_id": "m1", "source": "agent", "model": "m",
                "url": "https://example.test", "summary": "s", "payload": {}, "content_hash": "a" * 64,
                "captured_at": NOW.isoformat(), "expires_at": None, "submitted_by_token_id": "agent",
                "created_at": NOW.isoformat(),
            })
        if path == "/factors/agent_probability/observations":
            return httpx.Response(200, json={
                "factor_id": "agent_probability", "param": "", "market_id": "m1", "ts": NOW.isoformat(),
                "value": 0.7, "definition_version": "1", "provenance_written": True,
            })
        if path == "/research/backtest":
            return httpx.Response(200, json={"run_id": "run-1", "status": "queued"})
        if path == "/research/backtest/run-1":
            return httpx.Response(200, json={"run_id": "run-1", "status": "complete"})
        if path == "/evaluation/reports/report-1":
            return httpx.Response(200, json={
                "report_id": "report-1", "run_id": "run-1", "ranking_metric": "brier",
                "ranked_strategies": [], "benchmark_rows": [], "warnings": [],
                "generated_at": NOW.isoformat(),
            })
        if path == "/decisions/d1/explain":
            return httpx.Response(200, json={"decision_id": "d1", "available": False})
        if path == "/decisions/d1/risk-check":
            return httpx.Response(200, json={"decision_id": "d1", "approved": False, "reason": "pending"})
        if path == "/approvals/pending":
            return httpx.Response(200, json={"approvals": []})
        if path == "/positions":
            return httpx.Response(200, json={"positions": []})
        if path == "/run/kill":
            return httpx.Response(200, json={"status": "killed"})
        return httpx.Response(404, json={"detail": path})

    return handle


@pytest.mark.asyncio
async def test_tools_forward_typed_requests_with_bearer_token() -> None:
    requests: list[httpx.Request] = []
    transport = httpx.MockTransport(_handler(requests))
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = PMSMCPClient(MCPConfig("http://test", "agent-secret"), http_client=http_client)
    try:
        await client.get_status()
        await client.list_markets(q="election")
        await client.explain_selection("m1")
        await client.list_signals(limit=3)
        await client.get_factor_catalog()
        await client.get_factor_series(factor_id="agent_probability", market_id="m1")
        await client.submit_evidence({
            "market_id": "m1", "source": "agent", "model": "m", "url": "https://example.test",
            "summary": "s", "payload": {}, "content_hash": "a" * 64, "captured_at": NOW,
        })
        await client.submit_agent_factor("agent_probability", {
            "market_id": "m1", "value": 0.7, "ts": NOW, "model": "m", "prompt_hash": "b" * 64,
        })
        await client.run_backtest("sweep: []")
        await client.get_research_report("run-1")
        await client.get_research_report(report_id="report-1")
        await client.explain_decision("d1")
        await client.risk_check_decision("d1")
        await client.list_pending_approvals()
        await client.list_positions()
        await client.kill_switch()
    finally:
        await http_client.aclose()

    assert [request.method for request in requests] == [
        "GET", "GET", "GET", "GET", "GET", "GET", "POST", "POST", "POST", "GET",
        "GET", "GET", "POST", "GET", "GET", "POST",
    ]
    assert requests[0].headers["authorization"] == "Bearer agent-secret"
    assert requests[1].url.params["q"] == "election"
    assert json.loads(requests[6].content)["market_id"] == "m1"
    assert json.loads(requests[7].content)["prompt_hash"] == "b" * 64
    assert json.loads(requests[8].content) == {"sweep_yaml": "sweep: []"}


def test_tool_specs_are_curated_and_do_not_expose_live_mutations() -> None:
    names = {spec.name for spec in TOOL_SPECS}
    assert len(names) == 15
    assert "kill_switch" in names
    forbidden = ("start", "live", "switch_mode", "accept", "clear", "resolve", "reconcile")
    assert not any(any(word in name for word in forbidden) for name in names if name != "kill_switch")
    assert next(spec for spec in TOOL_SPECS if spec.name == "kill_switch").scope == "kill"
    assert {spec.name for spec in TOOL_SPECS if spec.requires_routes} == {
        "submit_evidence",
        "submit_agent_factor",
    }


@pytest.mark.asyncio
async def test_mcp_server_registers_exact_curated_tool_list() -> None:
    pytest.importorskip("mcp")
    requests: list[httpx.Request] = []
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler(requests)),
        base_url="http://test",
    )
    server = create_server(PMSMCPClient(MCPConfig("http://test", "secret"), http_client=http_client))
    tools = await server.list_tools()
    assert {tool.name for tool in tools} == {spec.name for spec in TOOL_SPECS}
    await http_client.aclose()
