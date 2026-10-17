from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from pms.api.routes.agent_factors import _agent_catalog_entry, _validate_value
from pms.controller.forecasters.llm import LLMForecaster, LLMReplayCacheMissError
from pms.controller.pipeline import ControllerPipeline
from pms.config import LLMSettings
from pms.core.models import MarketSignal
from pms.core.enums import MarketStatus
from pms.factors.composition import _rule_delta
from pms.storage.agent_evidence_store import (
    canonical_content_hash,
    validate_evidence_input,
)


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


def test_llm_replay_is_provider_free_and_fails_on_miss(tmp_path: Path) -> None:
    cache_path = tmp_path / "llm-replay.json"
    config = LLMSettings(
        enabled=True,
        provider="openai",
        api_key="test",
        model="openai-test",
        record_replay_path=str(cache_path),
    )
    record = LLMForecaster(config=config, record_replay_mode="record")

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        class Chat:
            class Completions:
                def create(self, **kwargs: Any) -> object:
                    del kwargs
                    return type(
                        "Response",
                        (),
                        {
                            "choices": [
                                type(
                                    "Choice",
                                    (),
                                    {"message": type("Message", (), {"content": '{"prob_estimate": 0.8, "confidence": 0.5, "rationale": "ok"}'})()},
                                )
                            ]
                        },
                    )()

            completions = Completions()

        chat = Chat()

    record.client = Client()
    assert record.predict(_signal()) is not None

    replay = LLMForecaster(config=config, record_replay_mode="replay")
    assert replay.predict(_signal()) is not None
    missing = LLMForecaster(
        config=config.model_copy(update={"record_replay_path": str(tmp_path / "missing.json")}),
        record_replay_mode="replay",
    )
    with pytest.raises(LLMReplayCacheMissError):
        missing.predict(_signal())


@pytest.mark.asyncio
async def test_controller_research_replay_cache_miss_is_fatal(tmp_path: Path) -> None:
    config = LLMSettings(
        enabled=True,
        provider="openai",
        api_key="test",
        model="openai-test",
        record_replay_path=str(tmp_path / "missing.json"),
    )
    pipeline = ControllerPipeline(
        forecasters=[
            LLMForecaster(config=config, record_replay_mode="replay"),
        ],
    )
    with pytest.raises(LLMReplayCacheMissError):
        await pipeline.on_signal(_signal())
