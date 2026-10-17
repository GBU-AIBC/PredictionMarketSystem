from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import time
from types import SimpleNamespace
from typing import cast

import asyncpg
from fastapi import FastAPI, Request
import httpx
import pytest
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from pms.api.routes.agent_evidence import router as evidence_router
from pms.api.routes.agent_factors import (
    AgentFactorObservationRequest,
    router as factor_router,
    submit_agent_factor_observation,
)
from pms.config import DatabaseSettings, LLMSettings, PMSSettings
from pms.controller.factor_snapshot import PostgresFactorSnapshotReader
from pms.controller.factory import ControllerPipelineFactory
from pms.controller.forecasters.llm import LLMForecaster
from pms.core.enums import RunMode
from pms.core.models import MarketSignal, Portfolio
from pms.factors.catalog import seed_factor_catalog
from pms.storage.agent_evidence_store import (
    AgentEvidenceStore,
    canonical_content_hash,
)
from pms.storage.llm_replay_store import PostgresLLMReplayStore
from pms.runner import Runner
from pms.research.runner import BacktestRunner
from pms.strategies.intents import StrategyContext
from pms.strategies.agent_evidence import AgentEvidenceObservationSource
from pms.strategies.paper_multifactor import build_paper_agent_probability_strategy
from pms.strategies.projections import ForecasterSpec


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

        factor_ts = datetime.now(tz=UTC)
        factor_response = await client.post(
            "/factors/agent_probability/observations",
            json={
                "market_id": market_id,
                "value": 0.8,
                "ts": factor_ts.isoformat(),
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
                "ts": factor_ts.isoformat(),
                "model": "model-other",
                "prompt_hash": "e" * 64,
            },
        )
        assert conflicting_provenance.status_code == 409
        knowledge_time_conflict = await client.post(
            "/factors/agent_probability/observations",
            json={
                "market_id": market_id,
                "value": 0.8,
                "ts": (now - timedelta(seconds=1)).isoformat(),
                "model": "model-v1",
                "prompt_hash": "a" * 64,
                "evidence_ids": [evidence_response.json()["evidence_id"]],
            },
        )
        assert knowledge_time_conflict.status_code == 422
        provenance = await client.get(
            "/factors/agent_probability/observations/provenance",
            params={"market_id": market_id, "ts": factor_ts.isoformat()},
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
    retry_ts = now - timedelta(seconds=1)
    retry_request = AgentFactorObservationRequest(
        market_id=market_id,
        value=0.7,
        ts=retry_ts,
        model="retry-model",
        prompt_hash="f" * 64,
    )
    await submit_agent_factor_observation(
        pg_pool,
        factor_id="agent_probability",
        request=retry_request,
        submitted_by_token_id="integration-token",
        now=now,
    )
    retry = await submit_agent_factor_observation(
        pg_pool,
        factor_id="agent_probability",
        request=retry_request,
        submitted_by_token_id="integration-token",
        now=now + timedelta(seconds=121),
    )
    assert retry.provenance_written is False
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


@pytest.mark.asyncio(loop_scope="session")
async def test_posted_evidence_changes_paper_llm_prompt_but_not_before_as_of(
    pg_pool: asyncpg.Pool,
) -> None:
    market_id = "w2b-prompt-market"
    await _seed_market(pg_pool, market_id)
    now = datetime.now(tz=UTC)
    as_of = now - timedelta(seconds=2)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_api(pg_pool)),
        base_url="http://test",
    ) as client:
        summary = "known-before-as-of"
        captured_at = now - timedelta(seconds=1)
        payload = {"summary": summary}
        expires_at = now + timedelta(minutes=5)
        content_hash = canonical_content_hash(
            market_id=market_id,
            source="prompt-agent",
            model="prompt-model",
            url="https://example.test/prompt",
            summary=summary,
            payload=payload,
            captured_at=captured_at,
            expires_at=expires_at,
        )
        response = await client.post(
            "/evidence",
            json={
                "market_id": market_id,
                "source": "prompt-agent",
                "model": "prompt-model",
                "url": "https://example.test/prompt",
                "summary": summary,
                "payload": payload,
                "content_hash": content_hash,
                "captured_at": captured_at.isoformat(),
                "expires_at": expires_at.isoformat(),
            },
        )
        assert response.status_code == 201, response.text
        as_of = datetime.now(tz=UTC) - timedelta(milliseconds=200)
        async with pg_pool.acquire() as connection:
            await connection.execute(
                "UPDATE agent_evidence SET created_at = $1 WHERE evidence_id = $2",
                as_of - timedelta(seconds=1),
                response.json()["evidence_id"],
            )
        summary = "captured-after-as-of"
        captured_at = as_of + timedelta(milliseconds=100)
        payload = {"summary": summary}
        content_hash = canonical_content_hash(
            market_id=market_id,
            source="prompt-agent",
            model="prompt-model",
            url="https://example.test/prompt",
            summary=summary,
            payload=payload,
            captured_at=captured_at,
            expires_at=expires_at,
        )
        response = await client.post(
            "/evidence",
            json={
                "market_id": market_id,
                "source": "prompt-agent",
                "model": "prompt-model",
                "url": "https://example.test/prompt",
                "summary": summary,
                "payload": payload,
                "content_hash": content_hash,
                "captured_at": captured_at.isoformat(),
                "expires_at": expires_at.isoformat(),
            },
        )
        assert response.status_code == 201, response.text

    class Messages:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                content='{"prob_estimate": 0.8, "confidence": 0.5, "rationale": "evidence"}'
            )

    messages = Messages()
    signal = MarketSignal(
        market_id=market_id,
        token_id="prompt-token",
        venue="polymarket",
        title="Prompt evidence",
        yes_price=0.5,
        volume_24h=1000.0,
        resolves_at=None,
        orderbook={},
        external_signal={},
        fetched_at=as_of,
        market_status="open",
    )
    forecaster = LLMForecaster(
        config=LLMSettings(
            enabled=True,
            provider="anthropic",
            api_key="test",
            model="claude-test",
            cache_ttl_s=30.0,
        ),
        client=SimpleNamespace(messages=messages),
        evidence_reader=AgentEvidenceObservationSource(
            AgentEvidenceStore(pg_pool)
        ),
    )
    assert await forecaster.apredict(signal) is not None
    first_prompt = str(messages.calls[0]["messages"])
    assert "known-before-as-of" in first_prompt
    assert "captured-after-as-of" not in first_prompt


@pytest.mark.asyncio(loop_scope="session")
async def test_paper_evidence_replays_through_backtest_factory_without_provider(
    pg_pool: asyncpg.Pool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pms.controller.factory.validate_llm_runtime_dependencies",
        lambda config: None,
    )
    market_id = "w2b-replay-evidence-market"
    await _seed_market(pg_pool, market_id)
    now = datetime.now(tz=UTC).replace(microsecond=0)
    captured_at = now - timedelta(seconds=2)
    expires_at = now + timedelta(minutes=5)
    payload = {"probability": 0.8}
    summary = "evidence used for replay"
    evidence = {
        "market_id": market_id,
        "source": "replay-agent",
        "model": "replay-model",
        "url": "https://example.test/replay",
        "summary": summary,
        "payload": payload,
        "content_hash": canonical_content_hash(
            market_id=market_id,
            source="replay-agent",
            model="replay-model",
            url="https://example.test/replay",
            summary=summary,
            payload=payload,
            captured_at=captured_at,
            expires_at=expires_at,
        ),
        "captured_at": captured_at.isoformat(),
        "expires_at": expires_at.isoformat(),
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_api(pg_pool)),
        base_url="http://test",
    ) as client:
        response = await client.post("/evidence", json=evidence)
        assert response.status_code == 201, response.text
    async with pg_pool.acquire() as connection:
        await connection.execute(
            "UPDATE agent_evidence SET created_at = $1 WHERE market_id = $2",
            now - timedelta(seconds=1),
            market_id,
        )
    signal = MarketSignal(
        market_id=market_id,
        token_id="replay-token",
        venue="polymarket",
        title="Replay evidence?",
        yes_price=0.5,
        volume_24h=1_000.0,
        resolves_at=None,
        orderbook={},
        external_signal={},
        fetched_at=now,
        market_status="open",
    )
    strategy = build_paper_agent_probability_strategy().to_active(
        strategy_version_id="paper-agent-v1"
    )
    strategy = replace(strategy, forecaster=ForecasterSpec(forecasters=(("llm", ()),)))
    llm_settings = LLMSettings(
        enabled=True,
        provider="openai",
        api_key="test",
        model="openai-test",
        cache_ttl_s=0.0,
    )
    settings = PMSSettings(
        mode=RunMode.PAPER,
        database=DatabaseSettings(
            dsn="postgres://postgres:postgres@127.0.0.1:55432/pms_w2b"
        ),
        llm=llm_settings,
    )
    paper_factory = ControllerPipelineFactory(
        settings=settings,
        evidence_reader=AgentEvidenceObservationSource(AgentEvidenceStore(pg_pool)),
        replay_store=PostgresLLMReplayStore(pg_pool),
    )
    paper_pipeline = paper_factory.build(strategy)
    assert paper_pipeline.forecasters is not None
    paper_forecaster = cast(LLMForecaster, paper_pipeline.forecasters[0])

    def paper_create(**kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"prob_estimate": 0.81, "confidence": 0.7, "rationale": "evidence"}'
                    )
                )
            ]
        )

    paper_forecaster.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=paper_create))
    )
    recorded = await paper_forecaster.apredict(signal)
    assert recorded is not None, paper_forecaster.last_failure_reason

    research_factory = ControllerPipelineFactory(
        settings=settings.model_copy(update={"mode": RunMode.BACKTEST})
    )
    BacktestRunner(
        writable_pool=pg_pool,
        readonly_pool=pg_pool,
        controller_factory=research_factory,
    )
    replay_pipeline = research_factory.build(strategy)
    assert replay_pipeline.forecasters is not None
    replay_forecaster = cast(LLMForecaster, replay_pipeline.forecasters[0])

    def fail_create(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("provider called during research replay")

    replay_forecaster.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail_create))
    )
    replayed = await replay_forecaster.apredict(signal)
    assert replayed == recorded


@pytest.mark.asyncio(loop_scope="session")
async def test_postgres_replay_store_indexed_lookup_and_first_record_wins(
    pg_pool: asyncpg.Pool,
) -> None:
    recorded_at = datetime.now(tz=UTC)
    rows = [
        (
            f"w2b-scale-{index}",
            "openai-test",
            "p" * 64,
            f"{index:064x}",
            json.dumps({"prob_estimate": 0.5, "confidence": 0.5, "rationale": "scale"}),
            recorded_at,
            "paper",
        )
        for index in range(100_000)
    ]
    async with pg_pool.acquire() as connection:
        await connection.executemany(
            """
            INSERT INTO llm_replay_records (
                replay_key, model, prompt_hash, input_hash, response,
                recorded_at, run_mode
            ) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
            ON CONFLICT (replay_key) DO NOTHING
            """,
            rows,
        )
    store = PostgresLLMReplayStore(pg_pool)
    started = time.monotonic()
    result = await store.get("w2b-scale-99999")
    assert time.monotonic() - started < 0.5
    assert result is not None
    await store.put(
        replay_key="w2b-scale-99999",
        model="openai-test",
        prompt_hash="p" * 64,
        input_hash="different",
        response={"prob_estimate": 0.1},
        recorded_at=recorded_at,
        run_mode="paper",
    )
    after = await store.get("w2b-scale-99999")
    assert after is not None
    assert after["prob_estimate"] == 0.5
