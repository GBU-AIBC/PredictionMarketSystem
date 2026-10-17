"""A thin Model Context Protocol proxy for the PMS typed HTTP API.

The proxy deliberately owns no PMS runtime state.  Every tool is a small
HTTP request using the configured bearer token; the API remains the only
place that performs validation, authorization, and business decisions.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping, Protocol, TypeVar, cast
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field


class MCPError(RuntimeError):
    """An HTTP failure returned by the PMS API."""

    def __init__(self, status_code: int, detail: object) -> None:
        super().__init__(f"PMS API request failed ({status_code}): {detail}")
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class MCPConfig:
    """Connection settings for the proxy process."""

    base_url: str
    token: str

    @classmethod
    def from_env(cls) -> MCPConfig:
        base_url = _first_env(
            "PMS_AGENT_API_BASE_URL",
            "PMS_MCP_API_BASE_URL",
            "PMS_MCP_BASE_URL",
            "PMS_API_BASE_URL",
        ) or "http://127.0.0.1:8000"
        token = _first_env(
            "PMS_AGENT_TOKEN",
            "PMS_AGENT_API_TOKEN",
            "PMS_MCP_TOKEN",
            "PMS_MCP_API_TOKEN",
        )
        if not token:
            raise ValueError("PMS_AGENT_TOKEN (or PMS_MCP_TOKEN) is required")
        return cls(base_url=base_url, token=token)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """The HTTP contract and scope for one exposed MCP tool."""

    name: str
    method: Literal["GET", "POST"]
    path: str
    scope: str
    requires_routes: bool = False


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("get_status", "GET", "/status", "read"),
    ToolSpec("list_markets", "GET", "/markets", "read"),
    ToolSpec("explain_selection", "GET", "/selection/explain", "read"),
    ToolSpec("list_signals", "GET", "/signals", "read"),
    ToolSpec("get_factor_catalog", "GET", "/factors/catalog", "read"),
    ToolSpec("get_factor_series", "GET", "/factors", "read"),
    # These two routes are supplied by Wave 2B.  Keep the proxy contract in
    # place so the tools become usable as soon as that lane lands.
    ToolSpec("submit_evidence", "POST", "/evidence", "paper_operate|research", True),
    ToolSpec(
        "submit_agent_factor",
        "POST",
        "/factors/{factor_id}/observations",
        "paper_operate|research",
        True,
    ),
    ToolSpec("run_backtest", "POST", "/research/backtest", "research"),
    ToolSpec("get_research_report", "GET", "/research/backtest/{run_id}", "research"),
    ToolSpec("explain_decision", "GET", "/decisions/{decision_id}/explain", "read"),
    ToolSpec("risk_check_decision", "POST", "/decisions/{decision_id}/risk-check", "read"),
    ToolSpec("list_pending_approvals", "GET", "/approvals/pending", "read"),
    ToolSpec("list_positions", "GET", "/positions", "read"),
    ToolSpec("kill_switch", "POST", "/run/kill", "kill"),
)


class APIModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class StatusResponse(APIModel):
    mode: str
    runner_started_at: str | None = None
    running: bool
    healthy: bool
    workers: dict[str, Any]
    autostart_attempted: bool
    autostart_error: str | None = None
    runtime_continuity: Any = None
    sensors: Any = Field(default_factory=dict)
    controller: dict[str, Any]
    actuator: dict[str, Any]
    evaluator: dict[str, Any]
    supervision: dict[str, Any]
    quality: dict[str, Any]


class Market(APIModel):
    market_id: str
    question: str
    venue: str
    volume_24h: float | None
    updated_at: str
    resolves_at: str | None = None
    yes_token_id: str | None
    no_token_id: str | None
    yes_price: float | None = None
    no_price: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    last_trade_price: float | None = None
    liquidity: float | None = None
    spread_bps: int | None = None
    price_updated_at: str | None = None
    subscription_source: str | None = None
    subscribed: bool


class MarketsResponse(APIModel):
    markets: list[Market]
    limit: int
    offset: int
    total: int


class Signal(APIModel):
    market_id: str
    token_id: str | None
    venue: str
    title: str
    yes_price: float
    volume_24h: float | None
    resolves_at: datetime | None
    orderbook: dict[str, Any]
    external_signal: dict[str, Any]
    fetched_at: datetime
    market_status: str


class SelectionExplanation(APIModel):
    market_id: str
    available: bool
    reason: str | None = None
    reasons: list[str] = Field(default_factory=list)
    strategy_ids: list[str] = Field(default_factory=list)
    token_ids: list[str] = Field(default_factory=list)
    pinned_token_ids: list[str] = Field(default_factory=list)
    capped: bool = False
    strategy_reasons: dict[str, str] = Field(default_factory=dict)


class FactorCatalogEntry(APIModel):
    factor_id: str
    name: str
    description: str
    output_type: str
    direction: str


class FactorCatalogResponse(APIModel):
    catalog: list[FactorCatalogEntry]


class FactorPoint(APIModel):
    ts: datetime
    value: float


class FactorSeriesResponse(APIModel):
    factor_id: str
    param: str
    market_id: str
    points: list[FactorPoint]


class AgentEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_id: str = Field(min_length=1, max_length=256)
    source: str = Field(min_length=1, max_length=512)
    model: str = Field(min_length=1, max_length=256)
    url: str = Field(min_length=1, max_length=2048)
    summary: str = Field(min_length=1, max_length=8192)
    payload: dict[str, Any]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    captured_at: datetime
    expires_at: datetime | None = None
    evidence_id: str | None = Field(default=None, min_length=1, max_length=256)


class AgentEvidenceResponse(APIModel):
    evidence_id: str
    market_id: str
    source: str
    model: str
    url: str
    summary: str
    payload: dict[str, Any]
    content_hash: str
    captured_at: datetime
    expires_at: datetime | None
    submitted_by_token_id: str
    created_at: datetime


class AgentFactorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_id: str = Field(min_length=1, max_length=256)
    param: str = Field(default="", max_length=256)
    value: Any
    ts: datetime
    model: str = Field(min_length=1, max_length=256)
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_ids: list[str] = Field(default_factory=list, max_length=100)


class AgentFactorResponse(APIModel):
    factor_id: str
    param: str
    market_id: str
    ts: datetime
    value: float
    definition_version: str
    provenance_written: bool


class ResearchRunResponse(APIModel):
    run_ids: list[str] = Field(default_factory=list)
    unique_run_count: int | None = None
    runs: list[dict[str, Any]] = Field(default_factory=list)
    run_id: str | None = None
    status: str | None = None


class EvaluationReportResponse(APIModel):
    report_id: str
    run_id: str
    ranking_metric: Literal["brier", "sharpe", "pnl_cum"]
    ranked_strategies: list[dict[str, Any]]
    benchmark_rows: list[dict[str, Any]]
    attribution_commentary: str | None = None
    warnings: list[Any]
    next_action: str | None = None
    generated_at: datetime


class DecisionExplanation(APIModel):
    decision_id: str
    available: bool
    reason: str | None = None
    decision_trace: dict[str, Any] | None = None
    forecasts: dict[str, Any] | None = None
    gates: dict[str, Any] | None = None
    sizing: dict[str, Any] | None = None
    risk: dict[str, Any] | None = None


class RiskCheckResponse(APIModel):
    decision_id: str
    approved: bool
    reason: str


class PendingApprovalsResponse(APIModel):
    approvals: list[dict[str, Any]]


class Position(APIModel):
    market_id: str
    token_id: str | None
    venue: str
    side: str
    shares_held: float
    avg_entry_price: float
    unrealized_pnl: float
    locked_usdc: float
    mark_source: str | None = None
    mark_age_seconds: float | None = None
    strategy_id: str
    strategy_version_id: str
    risk_group_id: str | None = None


class PositionsResponse(APIModel):
    positions: list[Position]


class RunStopResponse(APIModel):
    status: str


ModelT = TypeVar("ModelT", bound=BaseModel)
QueryValue = str | int | float | bool | None | Sequence[str | int | float | bool | None]
QueryParams = Mapping[str, QueryValue]


class MCPServerLike(Protocol):
    def tool(
        self,
        *,
        name: str,
        structured_output: bool,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    async def run_stdio_async(self) -> None: ...


class PMSMCPClient:
    """Typed async client used by MCP tools and unit tests."""

    def __init__(
        self,
        config: MCPConfig | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 20.0,
    ) -> None:
        resolved = config or MCPConfig.from_env()
        self.config = resolved
        self._client = http_client or httpx.AsyncClient(
            base_url=resolved.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {resolved.token}"},
            timeout=timeout,
        )
        if http_client is not None:
            http_client.headers["Authorization"] = f"Bearer {resolved.token}"
        self._owns_client = http_client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        params: QueryParams | None = None,
        json: object | None = None,
    ) -> Any:
        response = await self._client.request(method, path, params=params, json=json)
        if response.is_error:
            try:
                detail: object = response.json()
            except ValueError:
                detail = response.text
            raise MCPError(response.status_code, detail)
        return response.json()

    async def _model(
        self,
        model: type[ModelT],
        method: Literal["GET", "POST"],
        path: str,
        *,
        params: QueryParams | None = None,
        json: object | None = None,
    ) -> ModelT:
        return model.model_validate(
            await self._request(method, path, params=params, json=json)
        )

    async def get_status(self) -> StatusResponse:
        return await self._model(StatusResponse, "GET", "/status")

    async def list_markets(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        q: str = "",
        volume_min: float = 0.0,
        liquidity_min: float = 0.0,
        spread_max_bps: int | None = None,
        yes_min: float = 0.0,
        yes_max: float = 1.0,
        resolves_within_days: int | None = None,
        subscribed: Literal["all", "only", "idle"] = "all",
    ) -> MarketsResponse:
        params = _compact_params(
            limit=limit,
            offset=offset,
            q=q,
            volume_min=volume_min,
            liquidity_min=liquidity_min,
            spread_max_bps=spread_max_bps,
            yes_min=yes_min,
            yes_max=yes_max,
            resolves_within_days=resolves_within_days,
            subscribed=subscribed,
        )
        return await self._model(MarketsResponse, "GET", "/markets", params=params)

    async def explain_selection(self, market_id: str) -> SelectionExplanation:
        return await self._model(
            SelectionExplanation,
            "GET",
            "/selection/explain",
            params={"market_id": market_id},
        )

    async def list_signals(self, *, limit: int = 50) -> list[Signal]:
        payload = await self._request("GET", "/signals", params={"limit": limit})
        return [Signal.model_validate(item) for item in payload]

    async def get_factor_catalog(self) -> FactorCatalogResponse:
        return await self._model(FactorCatalogResponse, "GET", "/factors/catalog")

    async def get_factor_series(
        self,
        *,
        factor_id: str,
        market_id: str,
        param: str = "",
        since: datetime | None = None,
        limit: int = 500,
    ) -> FactorSeriesResponse:
        return await self._model(
            FactorSeriesResponse,
            "GET",
            "/factors",
            params=_compact_params(
                factor_id=factor_id,
                market_id=market_id,
                param=param,
                since=since,
                limit=limit,
            ),
        )

    async def submit_evidence(
        self,
        request: AgentEvidenceRequest | Mapping[str, Any],
    ) -> AgentEvidenceResponse:
        body = AgentEvidenceRequest.model_validate(request)
        return await self._model(
            AgentEvidenceResponse,
            "POST",
            "/evidence",
            json=body.model_dump(mode="json", exclude_none=True),
        )

    async def submit_agent_factor(
        self,
        factor_id: str,
        request: AgentFactorRequest | Mapping[str, Any],
    ) -> AgentFactorResponse:
        body = AgentFactorRequest.model_validate(request)
        return await self._model(
            AgentFactorResponse,
            "POST",
            f"/factors/{_path_segment(factor_id)}/observations",
            json=body.model_dump(mode="json"),
        )

    async def run_backtest(self, sweep_yaml: str) -> ResearchRunResponse:
        return await self._model(
            ResearchRunResponse,
            "POST",
            "/research/backtest",
            json={"sweep_yaml": sweep_yaml},
        )

    async def get_research_report(
        self,
        run_id: str | None = None,
        *,
        report_id: str | None = None,
    ) -> ResearchRunResponse | EvaluationReportResponse:
        if report_id is not None:
            return await self._model(
                EvaluationReportResponse,
                "GET",
                f"/evaluation/reports/{_path_segment(report_id)}",
            )
        if run_id is None:
            raise ValueError("run_id or report_id is required")
        return await self._model(
            ResearchRunResponse,
            "GET",
            f"/research/backtest/{_path_segment(run_id)}",
        )

    async def explain_decision(self, decision_id: str) -> DecisionExplanation:
        return await self._model(
            DecisionExplanation,
            "GET",
            f"/decisions/{_path_segment(decision_id)}/explain",
        )

    async def risk_check_decision(self, decision_id: str) -> RiskCheckResponse:
        return await self._model(
            RiskCheckResponse,
            "POST",
            f"/decisions/{_path_segment(decision_id)}/risk-check",
        )

    async def list_pending_approvals(self) -> PendingApprovalsResponse:
        return await self._model(
            PendingApprovalsResponse,
            "GET",
            "/approvals/pending",
        )

    async def list_positions(self) -> PositionsResponse:
        return await self._model(PositionsResponse, "GET", "/positions")

    async def kill_switch(self) -> RunStopResponse:
        return await self._model(RunStopResponse, "POST", "/run/kill")


def create_server(client: PMSMCPClient | None = None) -> Any:
    """Build the stdio MCP server without importing PMS runtime modules."""

    api = client or PMSMCPClient()
    try:
        from mcp.server.fastmcp import FastMCP

        server_impl: Any = FastMCP(
            name="pms-mcp",
            instructions="Typed, scoped proxy for the Prediction Market System API.",
        )
    except ModuleNotFoundError:
        # MCP 2.x renamed FastMCP to MCPServer; keep the proxy source usable
        # for environments that already have the newer official SDK installed.
        from mcp.server.mcpserver import MCPServer

        server_impl = MCPServer(
            name="pms-mcp",
            version="0.1.0",
            description="Typed, scoped proxy for the Prediction Market System API.",
        )
    server = cast(MCPServerLike, server_impl)

    @server.tool(name="get_status", structured_output=True)
    async def get_status() -> StatusResponse:
        return await api.get_status()

    @server.tool(name="list_markets", structured_output=True)
    async def list_markets(
        limit: int = 20,
        offset: int = 0,
        q: str = "",
        volume_min: float = 0.0,
        liquidity_min: float = 0.0,
        spread_max_bps: int | None = None,
        yes_min: float = 0.0,
        yes_max: float = 1.0,
        resolves_within_days: int | None = None,
        subscribed: Literal["all", "only", "idle"] = "all",
    ) -> MarketsResponse:
        return await api.list_markets(
            limit=limit,
            offset=offset,
            q=q,
            volume_min=volume_min,
            liquidity_min=liquidity_min,
            spread_max_bps=spread_max_bps,
            yes_min=yes_min,
            yes_max=yes_max,
            resolves_within_days=resolves_within_days,
            subscribed=subscribed,
        )

    @server.tool(name="explain_selection", structured_output=True)
    async def explain_selection(market_id: str) -> SelectionExplanation:
        return await api.explain_selection(market_id)

    @server.tool(name="list_signals", structured_output=True)
    async def list_signals(limit: int = 50) -> list[Signal]:
        return await api.list_signals(limit=limit)

    @server.tool(name="get_factor_catalog", structured_output=True)
    async def get_factor_catalog() -> FactorCatalogResponse:
        return await api.get_factor_catalog()

    @server.tool(name="get_factor_series", structured_output=True)
    async def get_factor_series(
        factor_id: str,
        market_id: str,
        param: str = "",
        since: datetime | None = None,
        limit: int = 500,
    ) -> FactorSeriesResponse:
        return await api.get_factor_series(
            factor_id=factor_id,
            market_id=market_id,
            param=param,
            since=since,
            limit=limit,
        )

    @server.tool(name="submit_evidence", structured_output=True)
    async def submit_evidence(
        market_id: str,
        source: str,
        model: str,
        url: str,
        summary: str,
        payload: dict[str, Any],
        content_hash: str,
        captured_at: datetime,
        expires_at: datetime | None = None,
        evidence_id: str | None = None,
    ) -> AgentEvidenceResponse:
        return await api.submit_evidence(
            AgentEvidenceRequest(
                market_id=market_id,
                source=source,
                model=model,
                url=url,
                summary=summary,
                payload=payload,
                content_hash=content_hash,
                captured_at=captured_at,
                expires_at=expires_at,
                evidence_id=evidence_id,
            )
        )

    @server.tool(name="submit_agent_factor", structured_output=True)
    async def submit_agent_factor(
        factor_id: str,
        market_id: str,
        value: Any,
        ts: datetime,
        model: str,
        prompt_hash: str,
        param: str = "",
        evidence_ids: list[str] | None = None,
    ) -> AgentFactorResponse:
        return await api.submit_agent_factor(
            factor_id,
            AgentFactorRequest(
                market_id=market_id,
                param=param,
                value=value,
                ts=ts,
                model=model,
                prompt_hash=prompt_hash,
                evidence_ids=evidence_ids or [],
            ),
        )

    @server.tool(name="run_backtest", structured_output=True)
    async def run_backtest(sweep_yaml: str) -> ResearchRunResponse:
        return await api.run_backtest(sweep_yaml)

    @server.tool(name="get_research_report", structured_output=True)
    async def get_research_report(
        run_id: str | None = None,
        report_id: str | None = None,
    ) -> ResearchRunResponse | EvaluationReportResponse:
        return await api.get_research_report(run_id, report_id=report_id)

    @server.tool(name="explain_decision", structured_output=True)
    async def explain_decision(decision_id: str) -> DecisionExplanation:
        return await api.explain_decision(decision_id)

    @server.tool(name="risk_check_decision", structured_output=True)
    async def risk_check_decision(decision_id: str) -> RiskCheckResponse:
        return await api.risk_check_decision(decision_id)

    @server.tool(name="list_pending_approvals", structured_output=True)
    async def list_pending_approvals() -> PendingApprovalsResponse:
        return await api.list_pending_approvals()

    @server.tool(name="list_positions", structured_output=True)
    async def list_positions() -> PositionsResponse:
        return await api.list_positions()

    @server.tool(name="kill_switch", structured_output=True)
    async def kill_switch() -> RunStopResponse:
        return await api.kill_switch()

    return server


create_mcp_server = create_server


async def _run_stdio() -> None:
    client = PMSMCPClient()
    try:
        await create_server(client).run_stdio_async()
    finally:
        await client.aclose()


def main() -> None:
    """Run the MCP server over stdio for Claude Code or Codex."""

    asyncio.run(_run_stdio())


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _compact_params(**params: object) -> QueryParams:
    compact: dict[str, QueryValue] = {}
    for key, value in params.items():
        if value is None:
            continue
        compact[key] = value.isoformat() if isinstance(value, datetime) else value  # type: ignore[assignment]
    return compact


def _path_segment(value: str) -> str:
    return quote(value, safe="")
