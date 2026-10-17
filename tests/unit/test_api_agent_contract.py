from __future__ import annotations

from datetime import UTC, datetime, timedelta
import os
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from fastapi.routing import APIRoute
from pydantic import ValidationError

from pms.api.app import create_app
from pms.config import PMSSettings
from pms.config import _require_live_api_control_plane_auth
from pms.core.enums import RunMode, Side, TimeInForce, Venue
from pms.core.models import TradeDecision
from pms.runner import Runner


def _settings(*, mode: RunMode = RunMode.PAPER, api_token: str | None = None) -> PMSSettings:
    return PMSSettings(
        mode=mode,
        api_token=api_token,
        auto_migrate_default_v2=False,
    )


def _decision() -> TradeDecision:
    return TradeDecision(
        decision_id="d-agent-contract",
        market_id="m-agent-contract",
        token_id="t-agent-contract",
        venue=Venue.POLYMARKET.value,
        side=Side.BUY.value,
        notional_usdc=10.0,
        order_type="limit",
        max_slippage_bps=50,
        stop_conditions=[],
        prob_estimate=0.7,
        expected_edge=0.2,
        time_in_force=TimeInForce.GTC,
        opportunity_id="o-agent-contract",
        strategy_id="default",
        strategy_version_id="default-v1",
        limit_price=0.5,
        action=Side.BUY.value,
    )


def _client(app: Any, *, raise_app_exceptions: bool = True) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions),
        base_url="http://test",
    )


@pytest.mark.asyncio
async def test_scoped_tokens_enforce_dynamic_mode_and_kill_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PAPER_SECRET", "paper-secret")
    monkeypatch.setenv("AGENT_KILL_SECRET", "kill-secret")
    settings = PMSSettings(
        mode=RunMode.PAPER,
        api_tokens=cast(Any, [
            {"token_id": "agent-paper", "secret": "AGENT_PAPER_SECRET", "scopes": ["paper_operate"]},
            {"token_id": "agent-kill", "secret": "AGENT_KILL_SECRET", "scopes": ["kill"]},
        ]),
        auto_migrate_default_v2=False,
    )
    runner = Runner(config=settings)
    app = create_app(runner, auto_start=False)
    async with _client(app, raise_app_exceptions=False) as client:
        paper = {"Authorization": "Bearer paper-secret"}
        kill = {"Authorization": "Bearer kill-secret"}
        assert (await client.post("/run/start", headers=paper)).status_code == 200
        assert (await client.post("/run/stop", headers=kill)).status_code == 200
        assert (await client.post("/run/start", headers=kill)).status_code == 403
        assert (await client.post("/config", headers=kill, json={"mode": "paper"})).status_code == 403
        runner.config.mode = RunMode.LIVE
        runner.state.mode = RunMode.LIVE
        assert (await client.post("/run/start", headers=paper)).status_code == 403
        assert (await client.post("/config", headers=paper, json={"mode": "live"})).status_code == 403
        assert (await client.post("/decisions/d/accept", headers=paper, json={"factor_snapshot_hash": "x"})).status_code == 403
        assert (await client.post("/live/halt/clear", headers=paper)).status_code == 403
        assert (await client.post("/live/reconcile-submission-unknown", headers=paper, json={"decision_id": "d", "status": "filled", "reconciled_by": "agent"})).status_code == 403
        assert (await client.post("/feedback/f/resolve", headers=paper)).status_code == 403


def test_token_settings_reject_invalid_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_AGENT_SECRET", raising=False)
    with pytest.raises(ValidationError, match="missing"):
        PMSSettings(api_tokens=cast(Any, [{"token_id": "a", "secret": "MISSING_AGENT_SECRET", "scopes": ["read"]}]))
    monkeypatch.setenv("DUP_SECRET", "secret")
    with pytest.raises(ValidationError, match="unique"):
        PMSSettings(api_tokens=cast(Any, [
            {"token_id": "a", "secret": "DUP_SECRET", "scopes": ["read"]},
            {"token_id": "a", "secret": "DUP_SECRET", "scopes": ["read"]},
        ]))
    with pytest.raises(ValidationError):
        PMSSettings(api_tokens=cast(Any, [{"token_id": "a", "secret": "DUP_SECRET", "scopes": ["bogus"]}]))
    with pytest.raises(ValidationError):
        PMSSettings(api_tokens=cast(Any, [{"token_id": "a", "secret": "", "scopes": ["read"]}]))


def test_scoped_live_admin_token_satisfies_live_control_plane_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_ADMIN_SECRET", "live-admin-secret")
    settings = PMSSettings(
        mode=RunMode.LIVE,
        api_tokens=cast(Any, [{"token_id": "admin", "secret": "LIVE_ADMIN_SECRET", "scopes": ["live_admin"]}]),
    )
    _require_live_api_control_plane_auth(settings)


@pytest.mark.asyncio
async def test_legacy_token_fallback_and_token_id_spoofing_are_closed() -> None:
    app = create_app(Runner(config=_settings(api_token="legacy-secret")), auto_start=False)
    async with _client(app) as client:
        headers = {"Authorization": "Bearer legacy-secret"}
        assert (await client.post("/config?token_id=attacker", headers=headers, json={"mode": "paper"})).status_code == 422
        assert (await client.post("/config", headers=headers, json={"mode": "paper", "token_id": "attacker"})).status_code == 422
        assert (await client.post("/config", headers=headers, json={"mode": "paper"})).status_code == 200


def test_routes_have_explicit_auth_or_public_exception_and_typed_openapi() -> None:
    app = create_app(Runner(config=_settings()), auto_start=False)
    public = {"/health", "/readiness", "/share/{strategy_id}"}
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.include_in_schema:
            continue
        assert route.tags, route.path
        assert route.response_model is not None, route.path
        dependency_names = {
            getattr(dependency.call, "__name__", "")
            for dependency in route.dependant.dependencies
        }
        if route.path not in public:
            assert any(name.startswith("require_") for name in dependency_names), route.path
    openapi = app.openapi()
    operations = openapi["paths"]
    for path, methods in operations.items():
        for method, operation in methods.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            assert operation.get("tags"), (path, method)
            schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
            assert schema
            while "$ref" in schema:
                schema = openapi["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
            if schema.get("type") == "array":
                schema = schema["items"]
                while "$ref" in schema:
                    schema = openapi["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
            assert schema.get("properties") or schema.get("anyOf"), (path, method, schema)


def test_generated_authorization_matrix_covers_every_route_and_scope() -> None:
    app = create_app(Runner(config=_settings()), auto_start=False)
    scopes = ("none", "read", "research", "paper_operate", "kill", "live_admin")
    public = {"/health", "/readiness", "/share/{strategy_id}"}
    matrix: dict[tuple[str, str, str], str] = {}
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.include_in_schema:
            continue
        methods = sorted(route.methods or ())
        names = {
            getattr(dependency.call, "__name__", "")
            for dependency in route.dependant.dependencies
        }
        if route.path in public:
            required: set[str] = set(scopes)
        elif "require_any_scope_kill_paper_operate_live_admin" in names:
            required = {"kill", "paper_operate", "live_admin"}
        elif "require_any_scope_paper_operate_live_admin" in names:
            required = {"paper_operate", "live_admin"}
        else:
            required = {
                scope
                for scope in scopes
                if f"require_scope_{scope}" in names
            }
        assert required or route.path in public, route.path
        for method in methods:
            for scope in scopes:
                matrix[(method, route.path, scope)] = (
                    "allow"
                    if scope in required
                    else ("public" if route.path in public else "401" if scope == "none" else "403")
                )
    assert matrix
    assert all(value in {"allow", "public", "401", "403"} for value in matrix.values())
    assert any(key[1] == "/run/start" and key[2] == "paper_operate" for key in matrix)
    assert any(key[1] == "/run/start" and key[2] == "live_admin" for key in matrix)


@pytest.mark.asyncio
async def test_risk_check_honors_halt_without_mutating_state() -> None:
    runner = Runner(config=_settings(api_token="legacy-secret"))
    risk = runner.actuator_executor.risk
    risk.record_api_error(401)
    class FakeStore:
        async def read_decisions(self, **_kwargs: object) -> list[Any]:
            return []

        async def get_decision(self, decision_id: str, include_opportunity: bool = False) -> Any:
            return _stored_decision(_decision())

    fake_store = FakeStore()
    runner.decision_store = cast(Any, fake_store)
    app = create_app(runner, auto_start=False)
    before = (len(risk.halt_events), runner.actuator_executor.dedup_store)
    async with _client(app) as client:
        response = await client.post(
            "/decisions/d-agent-contract/risk-check",
            headers={"Authorization": "Bearer legacy-secret"},
        )
    assert response.status_code == 200
    assert response.json() == {
        "decision_id": "d-agent-contract",
        "approved": False,
        "reason": "credential_failure",
    }
    assert len(risk.halt_events) == before[0]
    assert runner.actuator_executor.dedup_store is before[1]


@pytest.mark.asyncio
async def test_explain_and_selection_explain_are_read_only_and_typed() -> None:
    runner = Runner(config=_settings(api_token="legacy-secret"))

    class FakeStore:
        async def read_decisions(self, **_kwargs: object) -> list[Any]:
            return []

        async def get_decision(self, decision_id: str, include_opportunity: bool = False) -> Any:
            return _stored_decision(_decision())

    runner.decision_store = cast(Any, FakeStore())
    before_decisions = list(runner.state.decisions)
    before_events = tuple(runner.event_bus._events)  # noqa: SLF001
    app = create_app(runner, auto_start=False)
    async with _client(app) as client:
        decision = await client.get(
            "/decisions/d-agent-contract/explain",
            headers={"Authorization": "Bearer legacy-secret"},
        )
        selection = await client.get(
            "/selection/explain?market_id=m-agent-contract",
            headers={"Authorization": "Bearer legacy-secret"},
        )
    assert decision.status_code == 200
    assert decision.json()["available"] is True
    assert decision.json()["decision_id"] == "d-agent-contract"
    assert selection.status_code == 503
    assert selection.json()["detail"] == "selector_not_running"
    assert runner.state.decisions == before_decisions
    assert tuple(runner.event_bus._events) == before_events  # noqa: SLF001


@pytest.mark.asyncio
async def test_approval_pending_reads_redacted_audit_events(tmp_path: Any) -> None:
    audit_path = tmp_path / "first-order-audit.jsonl"
    audit_path.write_text(
        '{"event":"approval_denied","market_id":"m","token_id":"t","side":"BUY","outcome":"YES","max_notional_usdc":10.0,"limit_price":0.4}\n'
        '{"event":"approval_matched","market_id":"m","token_id":"t","side":"BUY","outcome":"YES","max_notional_usdc":10.0,"limit_price":0.4,"question":"secret"}\n'
        '{"event":"approval_denied","market_id":"m2","token_id":"t2","side":"BUY","outcome":"NO","max_notional_usdc":4.0,"limit_price":0.6,"question":"private"}\n',
        encoding="utf-8",
    )
    settings = _settings(api_token="legacy-secret").model_copy(
        update={"live_first_order_audit_path": str(audit_path)},
    )
    app = create_app(Runner(config=settings), auto_start=False)
    async with _client(app) as client:
        response = await client.get(
            "/approvals/pending",
            headers={"Authorization": "Bearer legacy-secret"},
        )
    assert response.status_code == 200
    assert response.json()["approvals"] == [
        {
            "market_id": "m2",
            "token_id": "t2",
            "side": "BUY",
            "outcome": "NO",
            "max_notional_usdc": 4.0,
            "limit_price": 0.6,
            "question": "private",
            "status": "pending",
        }
    ]


def _stored_decision(decision: TradeDecision) -> Any:
    return SimpleNamespace(
        decision=decision,
        status="pending",
        factor_snapshot_hash=None,
        created_at=datetime.now(tz=UTC),
        updated_at=datetime.now(tz=UTC),
        expires_at=datetime.now(tz=UTC) + timedelta(minutes=5),
        opportunity=None,
        decision_evidence={"decision_trace": {"risk": {"approved": True}}},
    )


class _AuditConnection:
    async def execute(self, *_args: object) -> None:
        raise RuntimeError("audit sink down")


class _AuditAcquire:
    async def __aenter__(self) -> _AuditConnection:
        return _AuditConnection()

    async def __aexit__(self, *_args: object) -> None:
        return None


class _AuditPool:
    def acquire(self) -> _AuditAcquire:
        return _AuditAcquire()

    async def fetch(self, *_args: object) -> list[object]:
        return []

    async def fetchval(self, *_args: object) -> int:
        return 1


class _RecordingConnection:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows

    async def execute(self, _query: str, *args: object) -> None:
        self.rows.append(args)


class _RecordingAcquire:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows

    async def __aenter__(self) -> _RecordingConnection:
        return _RecordingConnection(self.rows)

    async def __aexit__(self, *_args: object) -> None:
        return None


class _RecordingPool:
    def __init__(self) -> None:
        self.rows: list[tuple[object, ...]] = []

    def acquire(self) -> _RecordingAcquire:
        return _RecordingAcquire(self.rows)

    async def fetch(self, *_args: object) -> list[object]:
        return []

    async def fetchval(self, *_args: object) -> int:
        return 1


@pytest.mark.asyncio
async def test_request_audit_records_success_auth_errors_validation_and_handler_failure() -> None:
    runner = Runner(config=_settings(api_token="legacy-secret"))
    pool = _RecordingPool()
    runner.bind_pg_pool(cast(Any, pool))
    app = create_app(runner, auto_start=False)
    async with _client(app, raise_app_exceptions=False) as client:
        assert (await client.post("/config", headers={"Authorization": "Bearer legacy-secret"}, json={"mode": "paper"})).status_code == 200
        assert (await client.post("/config")).status_code == 401
        assert (await client.post("/config", headers={"Authorization": "Bearer legacy-secret"}, json={"mode": "paper", "token_id": "spoof"})).status_code == 422
        runner.config.api_token = "legacy-secret"
        async def failing_start() -> None:
            raise RuntimeError("handler failed")
        runner.start = failing_start  # type: ignore[method-assign]
        response = await client.post("/run/start", headers={"Authorization": "Bearer legacy-secret"})
        assert response.status_code == 500
    os.environ["READ_ONLY_SECRET"] = "read-only-secret"
    read_runner = Runner(config=PMSSettings(
        mode=RunMode.PAPER,
        api_tokens=cast(Any, [{"token_id": "read", "secret": "READ_ONLY_SECRET", "scopes": ["read"]}]),
        auto_migrate_default_v2=False,
    ))
    read_pool = _RecordingPool()
    read_runner.bind_pg_pool(cast(Any, read_pool))
    read_app = create_app(read_runner, auto_start=False)
    async with _client(read_app) as client:
        assert (await client.post("/config", headers={"Authorization": "Bearer read-only-secret"}, json={"mode": "paper"})).status_code == 403
    assert {int(cast(int, row[4])) for row in pool.rows} >= {200, 401, 422, 500}
    assert {int(cast(int, row[4])) for row in read_pool.rows} == {403}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "path", "kwargs"),
    [
        (RunMode.LIVE, "/run/start", {}),
        (RunMode.LIVE, "/decisions/d/accept", {"json": {"factor_snapshot_hash": "x"}}),
        (RunMode.PAPER, "/config", {"json": {"mode": "live"}}),
    ],
)
async def test_live_control_fails_closed_when_audit_sink_is_down(
    mode: RunMode,
    path: str,
    kwargs: dict[str, Any],
) -> None:
    runner = Runner(config=_settings(mode=mode, api_token="legacy-secret"))
    runner.bind_pg_pool(cast(Any, _AuditPool()))
    app = create_app(runner, auto_start=False)
    async with _client(app) as client:
        response = await client.post(path, headers={"Authorization": "Bearer legacy-secret"}, **kwargs)
    assert response.status_code == 503
    if mode == RunMode.PAPER:
        assert runner.config.mode == RunMode.PAPER
