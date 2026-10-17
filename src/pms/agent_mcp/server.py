"""A thin, OpenAPI-driven MCP proxy for the PMS HTTP API."""

from __future__ import annotations

import asyncio
import inspect
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, cast
from urllib.parse import quote

import httpx
from pydantic import RootModel

DEFAULT_TIMEOUT_SECONDS = 20.0
MAX_ERROR_DETAIL = 512
QueryValue = str | int | float | bool | None | Sequence[str | int | float | bool | None]
QueryParams = Mapping[str, QueryValue]
FORBIDDEN_OPERATIONS = frozenset({
    ("POST", "/run/start"),
    ("POST", "/run/stop"),
    ("POST", "/config"),
    ("POST", "/decisions/{decision_id}/accept"),
})


class MCPError(RuntimeError):
    """A typed proxy or API error safe to return through MCP."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        status_code: int | None = None,
    ) -> None:
        self.code = code
        self.status_code = status_code
        self.detail = _truncate(detail)
        super().__init__(f"{code}: {self.detail}")


class MCPStartupError(MCPError):
    """The API contract cannot safely back the MCP tool surface."""


@dataclass(frozen=True, slots=True)
class MCPConfig:
    """A validated, single-source MCP connection configuration."""

    base_url: str
    token: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        base_url = self.base_url.strip()
        token = self.token.strip()
        if not base_url:
            raise ValueError("PMS_AGENT_API_BASE_URL must not be blank")
        if not token:
            raise ValueError("PMS_AGENT_TOKEN must not be blank")
        try:
            parsed = httpx.URL(base_url)
        except httpx.InvalidURL as exc:
            raise ValueError("PMS_AGENT_API_BASE_URL must be an absolute HTTP URL") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("PMS_AGENT_API_BASE_URL must be an absolute HTTP URL")
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 60:
            raise ValueError("MCP timeout must be between 0 and 60 seconds")
        object.__setattr__(self, "base_url", base_url.rstrip("/"))
        object.__setattr__(self, "token", token)

    @classmethod
    def from_env(cls) -> MCPConfig:
        base_url = os.environ.get("PMS_AGENT_API_BASE_URL")
        token = os.environ.get("PMS_AGENT_TOKEN")
        if base_url is None or not base_url.strip():
            raise ValueError("PMS_AGENT_API_BASE_URL must not be blank")
        if token is None or not token.strip():
            raise ValueError("PMS_AGENT_TOKEN must not be blank")
        return cls(base_url=base_url, token=token)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Static endpoint identity; all schemas come from OpenAPI."""

    name: str
    method: Literal["GET", "POST"]
    path: str
    scope: str
    query_params: tuple[str, ...] = ()
    path_params: tuple[str, ...] = ()
    body_fields: tuple[str, ...] = ()
    requires_routes: bool = False


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("get_status", "GET", "/status", "read"),
    ToolSpec(
        "list_markets", "GET", "/markets", "read",
        query_params=("limit", "offset", "q", "volume_min", "liquidity_min", "spread_max_bps", "yes_min", "yes_max", "resolves_within_days", "subscribed"),
    ),
    ToolSpec("explain_selection", "GET", "/selection/explain", "read", query_params=("market_id",)),
    ToolSpec("list_signals", "GET", "/signals", "read", query_params=("limit",)),
    ToolSpec("get_factor_catalog", "GET", "/factors/catalog", "read"),
    ToolSpec("get_factor_series", "GET", "/factors", "read", query_params=("factor_id", "market_id", "param", "since", "limit")),
    ToolSpec("submit_evidence", "POST", "/evidence", "paper_operate|research", body_fields=("market_id", "source", "model", "url", "summary", "payload", "content_hash", "captured_at", "expires_at", "evidence_id"), requires_routes=True),
    ToolSpec("submit_agent_factor", "POST", "/factors/{factor_id}/observations", "paper_operate|research", path_params=("factor_id",), body_fields=("market_id", "param", "value", "ts", "model", "prompt_hash", "evidence_ids"), requires_routes=True),
    ToolSpec("run_backtest", "POST", "/research/backtest", "research", body_fields=("sweep_yaml",)),
    ToolSpec("get_research_report", "GET", "/evaluation/reports/{report_id}", "read", path_params=("report_id",)),
    ToolSpec("explain_decision", "GET", "/decisions/{decision_id}/explain", "read", path_params=("decision_id",)),
    ToolSpec("risk_check_decision", "POST", "/decisions/{decision_id}/risk-check", "read", path_params=("decision_id",)),
    ToolSpec("list_pending_approvals", "GET", "/approvals/pending", "read"),
    ToolSpec("list_positions", "GET", "/positions", "read"),
    ToolSpec("kill_switch", "POST", "/run/kill", "kill"),
)


@dataclass(frozen=True, slots=True)
class OpenAPIOperation:
    spec: ToolSpec
    operation: Mapping[str, Any]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


class _AnyOutput(RootModel[Any]):
    """Runtime adapter; its advertised schema is replaced by OpenAPI."""


class PMSMCPClient:
    """HTTP-only client. It never imports PMS runtime modules."""

    def __init__(self, config: MCPConfig | None = None, *, http_client: httpx.AsyncClient | None = None, timeout: float | None = None) -> None:
        resolved = config or MCPConfig.from_env()
        effective_timeout = resolved.timeout_seconds if timeout is None else timeout
        if not math.isfinite(effective_timeout) or not 0 < effective_timeout <= 60:
            raise ValueError("MCP timeout must be between 0 and 60 seconds")
        if http_client is None:
            self._client = httpx.AsyncClient(base_url=resolved.base_url, headers={"Authorization": f"Bearer {resolved.token}"}, timeout=resolved.timeout_seconds)
            self._owns_client = True
        else:
            self._client = http_client
            if not str(http_client.base_url):
                http_client.base_url = httpx.URL(resolved.base_url)
            http_client.headers["Authorization"] = f"Bearer {resolved.token}"
            self._owns_client = False
        self.config = resolved
        self._timeout = effective_timeout

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch_openapi(self) -> dict[str, Any]:
        payload = await self._request("GET", "/openapi.json")
        if not isinstance(payload, dict) or not isinstance(payload.get("paths"), dict):
            raise MCPStartupError("bad_openapi", "OpenAPI response has no paths object")
        return payload

    async def prepare(self, document: Mapping[str, Any] | None = None) -> tuple[OpenAPIOperation, ...]:
        openapi = document if document is not None else await self.fetch_openapi()
        paths = openapi.get("paths")
        if not isinstance(paths, Mapping):
            raise MCPStartupError("bad_openapi", "OpenAPI document has no paths object")
        active: list[OpenAPIOperation] = []
        for spec in TOOL_SPECS:
            raw_path = paths.get(spec.path)
            operation = raw_path.get(spec.method.lower()) if isinstance(raw_path, Mapping) else None
            if not isinstance(operation, Mapping):
                if spec.requires_routes:
                    continue
                raise MCPStartupError("missing_operation", f"required MCP operation is missing: {spec.method} {spec.path}")
            active.append(_operation_from_openapi(spec, operation, openapi))
        return tuple(active)

    async def invoke(self, operation: OpenAPIOperation, arguments: Mapping[str, Any]) -> Any:
        return await self._invoke_spec(operation.spec, arguments)

    async def _invoke_spec(self, spec: ToolSpec, arguments: Mapping[str, Any]) -> Any:
        path = spec.path
        remaining = dict(arguments)
        for name in spec.path_params:
            if name not in remaining:
                raise MCPError("invalid_input", f"missing path parameter: {name}")
            path = path.replace("{" + name + "}", _path_segment(str(remaining.pop(name))))
        query: dict[str, object] = {}
        for name in spec.query_params:
            if name in remaining and remaining[name] is not None:
                query[name] = _wire_value(remaining.pop(name))
        body: dict[str, Any] | None = None
        if spec.body_fields:
            body = {name: _wire_value(remaining[name]) for name in spec.body_fields if name in remaining and remaining[name] is not None}
        return await self._request(spec.method, path, params=cast(QueryParams, query or None), json=body)

    async def _request(self, method: str, path: str, *, params: QueryParams | None = None, json: object | None = None) -> Any:
        try:
            response = await self._client.request(method, path, params=params, json=json, timeout=self._timeout)
        except httpx.TimeoutException as exc:
            raise MCPError("timeout", "PMS API request timed out") from exc
        except httpx.HTTPError as exc:
            raise MCPError("transport", "PMS API transport failed") from exc
        if response.is_error:
            raise MCPError("http_error", _response_detail(response, self.config.token), status_code=response.status_code)
        try:
            return response.json()
        except (TypeError, ValueError) as exc:
            raise MCPError("bad_json", f"PMS API returned malformed JSON ({response.status_code})", status_code=response.status_code) from exc

    async def get_status(self) -> Any:
        return await self._invoke_spec(_static_operation("get_status"), {})

    async def list_markets(self, **arguments: Any) -> Any:
        return await self._invoke_spec(_static_operation("list_markets"), arguments)

    async def explain_selection(self, market_id: str) -> Any:
        return await self._invoke_spec(_static_operation("explain_selection"), {"market_id": market_id})

    async def list_signals(self, *, limit: int = 50) -> Any:
        return await self._invoke_spec(_static_operation("list_signals"), {"limit": limit})

    async def get_factor_catalog(self) -> Any:
        return await self._invoke_spec(_static_operation("get_factor_catalog"), {})

    async def get_factor_series(self, **arguments: Any) -> Any:
        return await self._invoke_spec(_static_operation("get_factor_series"), arguments)

    async def submit_evidence(self, **arguments: Any) -> Any:
        return await self._invoke_spec(_static_operation("submit_evidence"), arguments)

    async def submit_agent_factor(self, **arguments: Any) -> Any:
        return await self._invoke_spec(_static_operation("submit_agent_factor"), arguments)

    async def run_backtest(self, sweep_yaml: str) -> Any:
        return await self._invoke_spec(_static_operation("run_backtest"), {"sweep_yaml": sweep_yaml})

    async def get_research_report(self, report_id: str) -> Any:
        return await self._invoke_spec(_static_operation("get_research_report"), {"report_id": report_id})

    async def explain_decision(self, decision_id: str) -> Any:
        return await self._invoke_spec(_static_operation("explain_decision"), {"decision_id": decision_id})

    async def risk_check_decision(self, decision_id: str) -> Any:
        return await self._invoke_spec(_static_operation("risk_check_decision"), {"decision_id": decision_id})

    async def list_pending_approvals(self) -> Any:
        return await self._invoke_spec(_static_operation("list_pending_approvals"), {})

    async def list_positions(self) -> Any:
        return await self._invoke_spec(_static_operation("list_positions"), {})

    async def kill_switch(self) -> Any:
        return await self._invoke_spec(_static_operation("kill_switch"), {})


async def create_server(client: PMSMCPClient | None = None, *, openapi: Mapping[str, Any] | None = None) -> Any:
    """Fetch and validate OpenAPI, then register only compatible tools."""

    api = client or PMSMCPClient()
    operations = await api.prepare(openapi)
    from mcp.server.fastmcp import FastMCP

    server: Any = FastMCP(name="pms-mcp", instructions="Typed, scoped proxy for the Prediction Market System API.")
    for operation in operations:
        _register_tool(server, api, operation)
    return server


async def _run_stdio() -> int:
    try:
        client = PMSMCPClient()
    except ValueError as exc:
        print(f"pms-mcp startup failed [config]: {exc}", file=sys.stderr)
        return 1
    try:
        server = await create_server(client)
        await server.run_stdio_async()
        return 0
    except MCPError as exc:
        print(f"pms-mcp startup failed [{exc.code}]: {exc.detail}", file=sys.stderr)
        return 1
    finally:
        await client.aclose()


def main() -> None:
    """Run the MCP server over stdio for Claude Code or Codex."""

    raise SystemExit(asyncio.run(_run_stdio()))


def _register_tool(server: Any, api: PMSMCPClient, operation: OpenAPIOperation) -> None:
    spec = operation.spec

    async def invoke(**kwargs: Any) -> Any:
        return await api.invoke(operation, kwargs)

    invoke.__name__ = spec.name
    invoke.__doc__ = f"{spec.method} {spec.path} ({spec.scope} scope)"
    setattr(invoke, "__signature__", _signature_for_schema(operation.input_schema))
    server.add_tool(invoke, name=spec.name, structured_output=False)
    tool = server._tool_manager._tools[spec.name]
    tool.parameters = operation.input_schema
    tool.fn_metadata.output_schema = operation.output_schema
    tool.fn_metadata.output_model = _AnyOutput


def _operation_from_openapi(spec: ToolSpec, operation: Mapping[str, Any], document: Mapping[str, Any]) -> OpenAPIOperation:
    _assert_safe_tool(spec)
    return OpenAPIOperation(spec, operation, _input_schema(spec, operation, document), _output_schema(operation, document))


def _input_schema(spec: ToolSpec, operation: Mapping[str, Any], document: Mapping[str, Any]) -> dict[str, Any]:
    parameters = operation.get("parameters", [])
    if not isinstance(parameters, list):
        raise MCPStartupError("incompatible_operation", f"{spec.name} parameters are not a list")
    props: dict[str, Any] = {}
    required: list[str] = []
    seen_query: list[str] = []
    seen_path: list[str] = []
    for parameter in parameters:
        if not isinstance(parameter, Mapping):
            raise MCPStartupError("incompatible_operation", f"{spec.name} has an invalid parameter")
        name = parameter.get("name")
        location = parameter.get("in")
        if not isinstance(name, str) or location not in {"query", "path"}:
            raise MCPStartupError("incompatible_operation", f"{spec.name} has an unsupported parameter")
        props[name] = _resolve_schema(parameter.get("schema"), document)
        (seen_query if location == "query" else seen_path).append(name)
        if parameter.get("required") is True:
            required.append(name)
    if tuple(seen_query) != spec.query_params or tuple(seen_path) != spec.path_params:
        raise MCPStartupError("incompatible_operation", f"{spec.name} parameter contract changed")
    request_body = operation.get("requestBody")
    if spec.body_fields:
        if not isinstance(request_body, Mapping):
            raise MCPStartupError("incompatible_operation", f"{spec.name} request body is missing")
        content = request_body.get("content")
        if not isinstance(content, Mapping) or not isinstance(content.get("application/json"), Mapping):
            raise MCPStartupError("incompatible_operation", f"{spec.name} has no JSON request body")
        body_schema = _resolve_schema(content["application/json"].get("schema"), document)
        body_props = body_schema.get("properties")
        if body_schema.get("type") != "object" or not isinstance(body_props, Mapping):
            raise MCPStartupError("incompatible_operation", f"{spec.name} body is not an object")
        if set(body_props) != set(spec.body_fields):
            raise MCPStartupError("incompatible_operation", f"{spec.name} body fields changed")
        props.update(body_props)
        for name in body_schema.get("required", []):
            if isinstance(name, str) and name not in required:
                required.append(name)
    elif request_body is not None:
        raise MCPStartupError("incompatible_operation", f"{spec.name} unexpectedly has a request body")
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def _output_schema(operation: Mapping[str, Any], document: Mapping[str, Any]) -> dict[str, Any]:
    responses = operation.get("responses")
    response = None
    if isinstance(responses, Mapping):
        response = responses.get("200") or responses.get("201")
        if response is None:
            response = next((item for key, item in responses.items() if str(key).startswith("2")), None)
    content = response.get("content") if isinstance(response, Mapping) else None
    schema = content.get("application/json", {}).get("schema") if isinstance(content, Mapping) else None
    if schema is None:
        raise MCPStartupError("incompatible_operation", "mapped operation has no JSON 2xx response schema")
    return _resolve_schema(schema, document)


def _resolve_schema(value: object, document: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MCPStartupError("bad_openapi", "OpenAPI schema is not an object")
    if "$ref" in value:
        ref = value.get("$ref")
        if not isinstance(ref, str) or not ref.startswith("#/components/schemas/"):
            raise MCPStartupError("bad_openapi", "unsupported OpenAPI schema reference")
        components = document.get("components")
        schemas = components.get("schemas") if isinstance(components, Mapping) else None
        target = schemas.get(ref.rsplit("/", 1)[-1]) if isinstance(schemas, Mapping) else None
        return _resolve_schema(target, document)
    resolved: dict[str, Any] = {}
    for key, item in value.items():
        if key == "properties" and isinstance(item, Mapping):
            resolved[key] = {name: _resolve_schema(schema, document) for name, schema in item.items()}
        elif key == "items" and isinstance(item, Mapping):
            resolved[key] = _resolve_schema(item, document)
        elif key in {"anyOf", "oneOf", "allOf"} and isinstance(item, list):
            resolved[key] = [_resolve_schema(schema, document) for schema in item]
        else:
            resolved[key] = item
    return resolved


def _assert_safe_tool(spec: ToolSpec) -> None:
    path = spec.path.lower()
    if (spec.method, spec.path) in FORBIDDEN_OPERATIONS:
        raise MCPStartupError("forbidden_operation", f"MCP tool maps to forbidden operation: {spec.method} {spec.path}")
    if path == "/run/kill" and spec.method != "POST":
        raise MCPStartupError("forbidden_operation", f"kill operation must be POST: {spec.method} {spec.path}")
    forbidden = (
        path == "/run/start",
        path == "/run/stop",
        path == "/config",
        "/accept" in path,
        path.startswith("/live/"),
        "/halt/clear" in path,
        "/reconcile" in path,
        "/resolve" in path,
    )
    if any(forbidden):
        raise MCPStartupError("forbidden_operation", f"MCP tool maps to forbidden operation: {spec.method} {spec.path}")


def _static_operation(name: str) -> ToolSpec:
    for spec in TOOL_SPECS:
        if spec.name == name:
            _assert_safe_tool(spec)
            return spec
    raise KeyError(name)


def _signature_for_schema(schema: Mapping[str, Any]) -> inspect.Signature:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    if not isinstance(properties, Mapping):
        properties = {}
    params = [inspect.Parameter(str(name), inspect.Parameter.KEYWORD_ONLY, annotation=Any, default=inspect.Parameter.empty if name in required else None) for name in properties]
    return inspect.Signature(params)


def _wire_value(value: object) -> object:
    return value.isoformat() if isinstance(value, datetime) else value


def _path_segment(value: str) -> str:
    return quote(value, safe="")


def _response_detail(response: httpx.Response, token: str) -> str:
    try:
        payload = response.json()
    except (TypeError, ValueError):
        return _truncate(_redact(response.text or f"HTTP {response.status_code}", token))
    if isinstance(payload, Mapping) and "detail" in payload:
        return _truncate(_redact(str(payload["detail"]), token))
    return _truncate(_redact(str(payload), token))


def _truncate(value: str) -> str:
    return value if len(value) <= MAX_ERROR_DETAIL else value[:MAX_ERROR_DETAIL] + "…"


def _redact(value: str, token: str) -> str:
    return value.replace(token, "[REDACTED]") if token else value
