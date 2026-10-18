from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import httpx
import pytest
from fastapi.routing import APIRoute

from pms.agent_mcp.server import (
    MCPConfig,
    MCPError,
    MCPStartupError,
    PMSMCPClient,
    TOOL_SPECS,
    ToolSpec,
    _assert_safe_tool,
    _input_schema,
    _path_segment,
    _redact,
    create_server,
)
from pms.api.app import create_app
from pms.config import PMSSettings
from pms.core.enums import RunMode
from pms.runner import Runner


def _app_openapi() -> dict[str, Any]:
    app = create_app(Runner(config=PMSSettings(auto_migrate_default_v2=False)), auto_start=False)
    return app.openapi()


def _client(handler: Any, *, token: str = "agent-secret", timeout: float = 1.0) -> tuple[PMSMCPClient, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    return PMSMCPClient(MCPConfig("http://test", token, timeout), http_client=http_client, timeout=timeout), http_client


@pytest.mark.asyncio
async def test_openapi_drives_active_tool_schemas_and_feature_detects_2b() -> None:
    pytest.importorskip("mcp")
    real_openapi = _app_openapi()
    client, transport_client = _client(lambda request: httpx.Response(200, json={"ok": True}))
    try:
        server = await create_server(client, openapi=real_openapi)
        names = {tool.name for tool in await server.list_tools()}
        assert {spec.name for spec in TOOL_SPECS} == names

        tools = {tool.name: tool for tool in await server.list_tools()}
        for spec in TOOL_SPECS:
            operation = real_openapi["paths"][spec.path][spec.method.lower()]
            assert tools[spec.name].inputSchema == _input_schema(spec, operation, real_openapi)
            assert tools[spec.name].outputSchema is not None
            assert tools[spec.name].outputSchema["type"] == "object"

        without_2b = deepcopy(real_openapi)
        without_2b["paths"].pop("/evidence")
        without_2b["paths"].pop("/factors/{factor_id}/observations")
        server = await create_server(client, openapi=without_2b)
        names = {tool.name for tool in await server.list_tools()}
        assert "submit_evidence" not in names
        assert "submit_agent_factor" not in names
        assert len(names) == len(TOOL_SPECS) - 2
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
        drifted = deepcopy(_app_openapi())
        drifted["paths"]["/markets"]["get"]["parameters"].pop()
        with pytest.raises(MCPStartupError, match="incompatible_operation"):
            await create_server(client, openapi=drifted)
        drifted = deepcopy(_app_openapi())
        drifted["components"]["schemas"]["AgentEvidenceRequest"]["properties"].pop("summary")
        with pytest.raises(MCPStartupError, match="incompatible_operation"):
            await create_server(client, openapi=drifted)
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


def test_tool_scopes_match_actual_route_dependencies() -> None:
    app = create_app(Runner(config=PMSSettings(auto_migrate_default_v2=False)), auto_start=False)
    expected = {
        "read": "require_scope_read",
        "research": "require_scope_research",
        "kill": "require_scope_kill",
        "paper_operate|research": "require_any_scope_paper_operate_research",
    }
    for spec in TOOL_SPECS:
        route = next(
            route for route in app.routes
            if isinstance(route, APIRoute) and route.path == spec.path and spec.method in route.methods
        )
        dependency_names = {getattr(dependency.call, "__name__", "") for dependency in route.dependant.dependencies}
        assert expected[spec.scope] in dependency_names, spec.name
        assert "require_scope_live_admin" not in dependency_names, spec.name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error_code", "status"),
    [
        (httpx.Response(200, text="not-json"), "bad_json", 200),
        (httpx.Response(401, text="agent-secret should not be copied"), "http_error", 401),
        (httpx.Response(403, json={"detail": "forbidden"}), "http_error", 403),
        (httpx.Response(302, headers={"location": "/login"}), "http_error", 302),
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
async def test_real_mcp_session_wraps_non_object_outputs() -> None:
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/signals":
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={})

    client, transport_client = _client(handler)
    try:
        server = await create_server(client, openapi=_app_openapi())
        async with create_connected_server_and_client_session(server) as session:
            listed = await session.list_tools()
            assert all(tool.outputSchema is not None for tool in listed.tools)
            assert all(tool.outputSchema["type"] == "object" for tool in listed.tools if tool.outputSchema)
            result = await session.call_tool("list_signals", {})
            assert result.isError is False
            assert result.structuredContent == {"result": []}
        assert any(request.url.path == "/signals" for request in requests)
    finally:
        await transport_client.aclose()


def test_path_parameters_reject_traversal_and_slashes() -> None:
    for value in ("", ".", "..", "nested/value", "nested\\value"):
        with pytest.raises(MCPError, match="path parameters"):
            _path_segment(value)
    assert _path_segment("safe id") == "safe%20id"


def test_token_redaction_covers_adjacent_and_url_encoded_tokens() -> None:
    assert _redact("Bearer tok", "tok") == "Bearer [REDACTED]"
    assert _redact("tokenabc", "tok") == "[REDACTED]enabc"
    assert _redact("Bearer%20tok", "tok") == "Bearer%20[REDACTED]"
    assert _redact("Bearer%20abc", "abc") == "Bearer%20[REDACTED]"
    long_token = "long-secret/" + "x" * 64
    assert _redact(f"Bearer%20{long_token.replace('/', '%2F')}suffix", long_token) == (
        "Bearer%20[REDACTED]suffix"
    )
    encoded_token = "sec/ret tok"
    assert _redact("sec%2fret%20tok", encoded_token) == "[REDACTED]"
    assert _redact("sec%2Fret%20tok", encoded_token) == "[REDACTED]"
    assert _redact("sec%2Fret+tok", encoded_token) == "[REDACTED]"


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
