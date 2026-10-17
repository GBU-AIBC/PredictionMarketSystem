from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import httpx
import pytest

from pms.agent_mcp.server import (
    MCPConfig,
    MCPError,
    MCPStartupError,
    PMSMCPClient,
    TOOL_SPECS,
    ToolSpec,
    _assert_safe_tool,
    _input_schema,
    create_server,
)
from pms.api.app import create_app
from pms.config import PMSSettings
from pms.core.enums import RunMode
from pms.runner import Runner


def _app_openapi() -> dict[str, Any]:
    app = create_app(Runner(config=PMSSettings(auto_migrate_default_v2=False)), auto_start=False)
    return app.openapi()


def _add_2b_routes(document: dict[str, Any]) -> None:
    evidence_fields = {
        "market_id": {"type": "string"},
        "source": {"type": "string"},
        "model": {"type": "string"},
        "url": {"type": "string"},
        "summary": {"type": "string"},
        "payload": {"type": "object"},
        "content_hash": {"type": "string"},
        "captured_at": {"type": "string", "format": "date-time"},
        "expires_at": {"anyOf": [{"type": "string", "format": "date-time"}, {"type": "null"}]},
        "evidence_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    }
    factor_fields = {
        "market_id": {"type": "string"},
        "param": {"type": "string", "default": ""},
        "value": {},
        "ts": {"type": "string", "format": "date-time"},
        "model": {"type": "string"},
        "prompt_hash": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}, "default": []},
    }

    def operation(fields: dict[str, Any]) -> dict[str, Any]:
        return {
            "parameters": [],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object", "properties": fields, "required": list(fields)}}},
            },
            "responses": {"201": {"content": {"application/json": {"schema": {"type": "object"}}}}},
        }

    document["paths"]["/evidence"] = {"post": operation(evidence_fields)}
    document["paths"]["/factors/{factor_id}/observations"] = {
        "post": {
            **operation(factor_fields),
            "parameters": [{"name": "factor_id", "in": "path", "required": True, "schema": {"type": "string"}}],
        }
    }


def _client(handler: Any, *, token: str = "agent-secret", timeout: float = 1.0) -> tuple[PMSMCPClient, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    return PMSMCPClient(MCPConfig("http://test", token, timeout), http_client=http_client, timeout=timeout), http_client


@pytest.mark.asyncio
async def test_openapi_drives_active_tool_schemas_and_feature_detects_2b() -> None:
    pytest.importorskip("mcp")
    without_2b = _app_openapi()
    client, transport_client = _client(lambda request: httpx.Response(200, json={"ok": True}))
    try:
        server = await create_server(client, openapi=without_2b)
        names = {tool.name for tool in await server.list_tools()}
        assert "submit_evidence" not in names
        assert "submit_agent_factor" not in names
        assert len(names) == 13

        with_2b = deepcopy(without_2b)
        _add_2b_routes(with_2b)
        server = await create_server(client, openapi=with_2b)
        tools = {tool.name: tool for tool in await server.list_tools()}
        assert {name for name in tools if name.startswith("submit_")} == {"submit_evidence", "submit_agent_factor"}
        for spec in TOOL_SPECS:
            if spec.name not in tools:
                continue
            operation = with_2b["paths"][spec.path][spec.method.lower()]
            assert tools[spec.name].inputSchema == _input_schema(spec, operation, with_2b)
            assert tools[spec.name].outputSchema is not None
    finally:
        await transport_client.aclose()


@pytest.mark.asyncio
async def test_missing_required_operation_and_malformed_openapi_fail_closed() -> None:
    pytest.importorskip("mcp")
    document = _app_openapi()
    document["paths"].pop("/status")
    client, transport_client = _client(lambda request: httpx.Response(200, json={}))
    try:
        with pytest.raises(MCPStartupError, match="missing_operation"):
            await create_server(client, openapi=document)
        with pytest.raises(MCPStartupError, match="paths object"):
            await client.prepare({"openapi": "3.1.0"})
    finally:
        await transport_client.aclose()


def test_endpoint_guard_rejects_mutated_live_operation() -> None:
    with pytest.raises(MCPStartupError, match="forbidden_operation"):
        _assert_safe_tool(ToolSpec("approve_decision", "POST", "/decisions/{decision_id}/accept", "paper_operate"))
    with pytest.raises(MCPStartupError):
        _assert_safe_tool(ToolSpec("resume", "POST", "/run/start", "paper_operate"))
    _assert_safe_tool(next(spec for spec in TOOL_SPECS if spec.name == "kill_switch"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error_code", "status"),
    [
        (httpx.Response(200, text="not-json"), "bad_json", 200),
        (httpx.Response(401, text="agent-secret should not be copied"), "http_error", 401),
        (httpx.Response(403, json={"detail": "forbidden"}), "http_error", 403),
        (httpx.Response(503, text="x" * 2000), "http_error", 503),
    ],
)
async def test_typed_errors_are_bounded_and_status_aware(response: httpx.Response, error_code: str, status: int) -> None:
    client, transport_client = _client(lambda request: response)
    try:
        with pytest.raises(MCPError) as raised:
            await client.get_status()
        assert raised.value.code == error_code
        assert raised.value.status_code == status
        assert len(raised.value.detail) <= 513
        assert "agent-secret" not in str(raised.value)
    finally:
        await transport_client.aclose()


@pytest.mark.asyncio
async def test_timeout_and_owned_client_shutdown_are_typed() -> None:
    def timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("upstream timed out")

    client, transport_client = _client(timeout, timeout=0.1)
    try:
        with pytest.raises(MCPError, match="timeout"):
            await client.get_status()
    finally:
        await transport_client.aclose()
    owned = PMSMCPClient(MCPConfig("http://test", "secret"))
    assert not owned._client.is_closed
    await owned.aclose()
    assert owned._client.is_closed


@pytest.mark.asyncio
async def test_real_scoped_tokens_through_asgi_mcp_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("READ_SECRET", "read-secret")
    monkeypatch.setenv("KILL_SECRET", "kill-secret")
    settings = PMSSettings(
        mode=RunMode.PAPER,
        api_tokens=cast(Any, [
            {"token_id": "read", "secret": "READ_SECRET", "scopes": ["read"]},
            {"token_id": "kill", "secret": "KILL_SECRET", "scopes": ["kill"]},
        ]),
        auto_migrate_default_v2=False,
    )
    runner = Runner(config=settings)

    async def stop(*, close_pool: bool = True) -> None:
        del close_pool

    runner.stop = stop  # type: ignore[method-assign]
    app = create_app(runner, auto_start=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as read_http:
        read_client = PMSMCPClient(MCPConfig("http://test", "read-secret"), http_client=read_http)
        await read_client.get_status()
        with pytest.raises(MCPError) as forbidden:
            await read_client.kill_switch()
        assert forbidden.value.status_code == 403
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as bad_http:
        bad_client = PMSMCPClient(MCPConfig("http://test", "bad-secret"), http_client=bad_http)
        with pytest.raises(MCPError) as unauthorized:
            await bad_client.get_status()
        assert unauthorized.value.status_code == 401
        assert "bad-secret" not in str(unauthorized.value)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as kill_http:
        kill_client = PMSMCPClient(MCPConfig("http://test", "kill-secret"), http_client=kill_http)
        assert (await kill_client.kill_switch())["status"] == "killed"


def test_env_requires_exact_nonblank_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PMS_AGENT_API_BASE_URL", " ")
    monkeypatch.setenv("PMS_AGENT_TOKEN", "token")
    with pytest.raises(ValueError, match="API_BASE_URL"):
        MCPConfig.from_env()
    monkeypatch.setenv("PMS_AGENT_API_BASE_URL", "http://test")
    monkeypatch.setenv("PMS_AGENT_TOKEN", "  ")
    with pytest.raises(ValueError, match="AGENT_TOKEN"):
        MCPConfig.from_env()
