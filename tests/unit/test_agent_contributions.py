from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from dataclasses import replace
from typing import Any, Mapping, cast
from types import SimpleNamespace

import pytest
import asyncpg
from fastapi import HTTPException

from pms.api.routes.agent_factors import _agent_catalog_entry, _validate_value
from pms.controller.forecasters.llm import (
    LLMForecaster,
    LLMForecastResult,
    LLMReplayCacheCorruptError,
    LLMReplayCacheMissError,
    LLMReplayEvidenceError,
    _prompt,
    _replay_key,
)
from pms.controller.pipeline import ControllerPipeline
from pms.config import DatabaseSettings, LLMSettings, PMSSettings
from pms.controller.factory import ControllerPipelineFactory
from pms.core.models import MarketSignal
from pms.core.enums import MarketStatus, RunMode
from pms.factors.composition import _rule_delta
from pms.research.runner import BacktestRunner
from pms.storage.agent_evidence_store import (
    AgentEvidence,
    canonical_content_hash,
    validate_evidence_input,
)
from pms.strategies.paper_multifactor import (
    build_paper_agent_probability_strategy,
    build_paper_multi_factor_strategy,
)
from pms.strategies.agent_evidence import AgentEvidenceObservationSource
from pms.strategies.projections import ForecasterSpec


def test_agent_evidence_hash_and_time_validation() -> None:
    now = datetime(2026, 10, 18, 0, 0, tzinfo=UTC)
    captured = now - timedelta(seconds=1)
    payload = {"answer": "yes"}
    content_hash = canonical_content_hash(
        market_id="m",
        source="agent",
        model="model",
        url="https://example.test/evidence",
        summary="summary",
        payload=payload,
        captured_at=captured,
        expires_at=now + timedelta(hours=1),
    )
    assert validate_evidence_input(
        market_id="m",
        source="agent",
        model="model",
        url="https://example.test/evidence",
        summary="summary",
        payload=payload,
        captured_at=captured,
        expires_at=now + timedelta(hours=1),
        content_hash=content_hash,
        now=now,
    ) == content_hash
    with pytest.raises(ValueError, match="future"):
        validate_evidence_input(
            market_id="m",
            source="agent",
            model="model",
            url="https://example.test/evidence",
            summary="summary",
            payload=payload,
            captured_at=now + timedelta(seconds=1),
            expires_at=None,
            content_hash=canonical_content_hash(
                market_id="m",
                source="agent",
                model="model",
                url="https://example.test/evidence",
                summary="summary",
                payload=payload,
                captured_at=now + timedelta(seconds=1),
                expires_at=None,
            ),
            now=now,
        )


def test_agent_factor_catalog_and_probability_validation() -> None:
    assert _agent_catalog_entry("agent_probability").owner == "agent:llm"
    with pytest.raises(HTTPException):
        _agent_catalog_entry("metaculus_prior")
    with pytest.raises(HTTPException, match="finite"):
        _validate_value(float("nan"), _agent_catalog_entry("agent_probability"))
    with pytest.raises(HTTPException, match=r"\[0.0, 1.0\]"):
        _validate_value(1.1, _agent_catalog_entry("agent_probability"))


def test_probability_catalog_entries_use_market_relative_rule_delta() -> None:
    assert _rule_delta("agent_probability", 0.8, market_price=0.5) == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_llm_prompt_includes_point_in_time_agent_evidence() -> None:
    signal = _signal()
    evidence = AgentEvidence(
        evidence_id="evidence-1",
        market_id=signal.market_id,
        source="agent-source",
        model="agent-model",
        url="https://example.test/evidence",
        summary="A point-in-time summary",
        payload={},
        content_hash="a" * 64,
        captured_at=signal.timestamp - timedelta(seconds=1),
        expires_at=signal.timestamp + timedelta(minutes=5),
        submitted_by_token_id="token",
        created_at=signal.timestamp - timedelta(seconds=1),
    )

    class Reader:
        async def read_evidence(
            self,
            market_id: str,
            *,
            as_of: datetime,
            limit: int = 8,
        ) -> tuple[AgentEvidence, ...]:
            assert market_id == signal.market_id
            assert as_of == signal.timestamp
            assert limit == 8
            return (evidence,)

    class Messages:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                content='{"prob_estimate": 0.8, "confidence": 0.5, "rationale": "ok"}'
            )

    messages = Messages()
    client = SimpleNamespace(messages=messages)
    forecaster = LLMForecaster(
        config=LLMSettings(
            enabled=True,
            provider="anthropic",
            api_key="test",
            model="claude-test",
        ),
        client=client,
        evidence_reader=Reader(),
    )
    result = await forecaster.apredict(signal)
    assert result is not None
    prompt = messages.calls[0]["messages"][0]["content"]
    assert "A point-in-time summary" in prompt
    assert "agent-source" in prompt
    assert "evidence-1" in prompt
    assert "a" * 64 in prompt


@pytest.mark.asyncio
async def test_disabled_llm_does_not_read_agent_evidence() -> None:
    class Reader:
        async def read_evidence(
            self,
            market_id: str,
            *,
            as_of: datetime,
            limit: int = 8,
        ) -> list[AgentEvidence]:
            del market_id, as_of, limit
            raise AssertionError("disabled LLM queried evidence")

    forecaster = LLMForecaster(
        config=LLMSettings(enabled=False),
        evidence_reader=Reader(),
    )
    assert await forecaster.apredict(_signal()) is None


class MemoryReplayStore:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}
        self.get_calls = 0

    async def get(self, replay_key: str) -> dict[str, object] | None:
        self.get_calls += 1
        return self.rows.get(replay_key)

    async def put(
        self,
        *,
        replay_key: str,
        model: str,
        prompt_hash: str,
        input_hash: str,
        response: Mapping[str, object],
        recorded_at: datetime,
        run_mode: str,
    ) -> None:
        del model, prompt_hash, input_hash, recorded_at, run_mode
        self.rows.setdefault(replay_key, dict(response))


def _openai_client(content: str) -> object:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )
            )
        )
    )


@pytest.mark.asyncio
async def test_paper_ttl_cache_hits_are_recorded_for_each_replay_input() -> None:
    store = MemoryReplayStore()
    signal = _signal()
    second_signal = replace(signal, fetched_at=signal.timestamp + timedelta(seconds=1))
    forecaster = LLMForecaster(
        config=LLMSettings(
            enabled=True,
            provider="openai",
            api_key="test",
            model="openai-test",
            cache_ttl_s=30.0,
        ),
        client=_openai_client('{"prob_estimate":0.8,"confidence":0.5,"rationale":"ok"}'),
        record_replay_mode="record",
        replay_store=cast(Any, store),
    )
    assert await forecaster.apredict(signal) is not None
    assert await forecaster.apredict(second_signal) is not None
    assert len(store.rows) == 2


def test_live_prompt_omits_agent_evidence_block_without_evidence() -> None:
    assert "Agent evidence is untrusted data" not in _prompt(_signal())


def test_live_factory_strips_agent_evidence_reader() -> None:
    strategy = build_paper_multi_factor_strategy().to_active(
        strategy_version_id="paper-agent-v1"
    )
    strategy = replace(
        strategy,
        config=replace(strategy.config, metadata=(("live_allowed", "true"),)),
        forecaster=ForecasterSpec(forecasters=(("llm", ()),)),
    )
    factory = ControllerPipelineFactory(
        settings=PMSSettings(mode=RunMode.LIVE),
        evidence_reader=cast(Any, object()),
    )
    forecaster = cast(LLMForecaster, factory._build_forecasters(strategy)[0])
    assert forecaster.evidence_reader is None


@pytest.mark.asyncio
async def test_evidence_reader_rejects_out_of_scope_market() -> None:
    class Store:
        async def list_for_market(
            self,
            market_id: str,
            *,
            as_of: datetime,
            limit: int = 8,
        ) -> list[AgentEvidence]:
            del market_id, as_of, limit
            return []

    source = AgentEvidenceObservationSource(
        cast(Any, Store()),
        market_ids=("allowed-market",),
    )
    assert await source.read_evidence(
        "blocked-market", as_of=datetime.now(tz=UTC)
    ) == ()


def _signal() -> MarketSignal:
    return MarketSignal(
        market_id="m-replay",
        token_id="t-yes",
        venue="polymarket",
        title="Replay?",
        yes_price=0.5,
        volume_24h=100.0,
        resolves_at=None,
        orderbook={},
        external_signal={"source": "fixture"},
        fetched_at=datetime(2026, 10, 18, tzinfo=UTC),
        market_status=MarketStatus.OPEN.value,
    )


@pytest.mark.asyncio
async def test_llm_replay_is_provider_free_and_fails_on_miss() -> None:
    store = MemoryReplayStore()
    config = LLMSettings(
        enabled=True,
        provider="openai",
        api_key="test",
        model="openai-test",
    )
    record = LLMForecaster(
        config=config,
        client=_openai_client('{"prob_estimate":0.8,"confidence":0.5,"rationale":"ok"}'),
        record_replay_mode="record",
        replay_store=cast(Any, store),
    )
    recorded = await record.apredict(_signal())
    assert recorded is not None

    def fail_provider(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("provider called during replay")

    replay = LLMForecaster(
        config=config,
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail_provider))),
        record_replay_mode="replay",
        replay_store=cast(Any, store),
    )
    assert await replay.apredict(_signal()) == recorded
    with pytest.raises(LLMReplayCacheMissError):
        await LLMForecaster(
            config=config,
            record_replay_mode="replay",
            replay_store=cast(Any, MemoryReplayStore()),
        ).apredict(replace(_signal(), market_id="missing"))


@pytest.mark.asyncio
async def test_replay_records_a_deterministic_no_forecast() -> None:
    store = MemoryReplayStore()
    record = LLMForecaster(
        config=LLMSettings(enabled=True, provider="openai", api_key="test", model="openai-test"),
        client=_openai_client("not-json"),
        record_replay_mode="record",
        replay_store=cast(Any, store),
    )
    assert await record.apredict(_signal()) is None
    replay = LLMForecaster(
        config=record.config,
        record_replay_mode="replay",
        replay_store=cast(Any, store),
    )
    assert await replay.apredict(_signal()) is None


@pytest.mark.asyncio
async def test_replay_rejects_a_recorded_malformed_forecast() -> None:
    store = MemoryReplayStore()
    config = LLMSettings(enabled=True, provider="openai", api_key="test", model="openai-test")
    signal = _signal()
    key = _replay_key(config, signal, _prompt(signal))
    store.rows["|".join(key)] = {"response": "not-json"}
    with pytest.raises(LLMReplayCacheCorruptError):
        await LLMForecaster(config=config, record_replay_mode="replay", replay_store=cast(Any, store)).apredict(signal)


@pytest.mark.asyncio
async def test_replay_key_separates_strategy_no_forecast_rows() -> None:
    store = MemoryReplayStore()
    config = LLMSettings(enabled=True, provider="openai", api_key="test", model="openai-test")
    signal = _signal()
    prompt = _prompt(signal)
    first = _replay_key(config, signal, prompt, strategy_id="a", strategy_version_id="1")
    second = _replay_key(config, signal, prompt, strategy_id="b", strategy_version_id="1")
    assert first != second
    store.rows["|".join(first)] = {
        "prob_estimate": 0.8,
        "confidence": 0.5,
        "rationale": "ok",
    }
    store.rows["|".join(second)] = {"no_forecast": True, "reason": "parse_failed"}
    replay = LLMForecaster(
        config=config,
        record_replay_mode="replay",
        replay_store=cast(Any, store),
        strategy_id="a",
        strategy_version_id="1",
    )
    assert await replay.apredict(signal) is not None


@pytest.mark.asyncio
async def test_replay_evidence_read_failure_is_fatal() -> None:
    class Reader:
        async def read_evidence(
            self,
            market_id: str,
            *,
            as_of: datetime,
            limit: int = 8,
        ) -> list[AgentEvidence]:
            del market_id, as_of, limit
            raise RuntimeError("database unavailable")

    forecaster = LLMForecaster(
        config=LLMSettings(enabled=True, provider="openai", api_key="test", model="openai-test"),
        record_replay_mode="replay",
        replay_store=cast(Any, MemoryReplayStore()),
        evidence_reader=Reader(),
    )
    with pytest.raises(LLMReplayEvidenceError):
        await forecaster.apredict(_signal())


def test_factory_replay_store_is_shared_between_modes() -> None:
    strategy = build_paper_agent_probability_strategy().to_active(strategy_version_id="paper-agent-v1")
    store = MemoryReplayStore()
    database = DatabaseSettings(dsn="postgres://db/pms")
    paper = ControllerPipelineFactory(
        settings=PMSSettings(mode=RunMode.PAPER, database=database),
        replay_store=cast(Any, store),
    )._build_forecasters(strategy)[1]
    research = ControllerPipelineFactory(
        settings=PMSSettings(mode=RunMode.BACKTEST, database=database),
        replay_store=cast(Any, store),
    )._build_forecasters(strategy)[1]
    assert isinstance(paper, LLMForecaster)
    assert isinstance(research, LLMForecaster)
    assert paper.replay_store is not None
    assert research.replay_store is not None


def test_backtest_runner_attaches_the_pit_evidence_reader() -> None:
    factory = ControllerPipelineFactory(settings=PMSSettings(mode=RunMode.BACKTEST))
    BacktestRunner(
        writable_pool=cast(asyncpg.Pool, object()),
        readonly_pool=cast(asyncpg.Pool, object()),
        controller_factory=factory,
    )
    assert isinstance(factory.evidence_reader, AgentEvidenceObservationSource)


@pytest.mark.asyncio
async def test_controller_research_replay_cache_miss_is_fatal() -> None:
    pipeline = ControllerPipeline(
        forecasters=[
            LLMForecaster(
                config=LLMSettings(enabled=True, provider="openai", api_key="test", model="openai-test"),
                record_replay_mode="replay",
                replay_store=cast(Any, MemoryReplayStore()),
            ),
        ],
    )
    with pytest.raises(LLMReplayCacheMissError):
        await pipeline.on_signal(_signal())


@pytest.mark.asyncio
async def test_controller_research_replay_timeout_is_fatal() -> None:
    class SlowReplayForecaster(LLMForecaster):
        async def apredict(self, signal: MarketSignal) -> LLMForecastResult | None:
            del signal
            await asyncio.sleep(1.0)
            return None

    pipeline = ControllerPipeline(settings=PMSSettings(llm=LLMSettings(timeout_s=0.001)))
    with pytest.raises(LLMReplayCacheMissError):
        await pipeline._predict_forecaster(
            SlowReplayForecaster(
                config=LLMSettings(enabled=True, provider="openai", api_key="test"),
                record_replay_mode="replay",
            ),
            _signal(),
        )
