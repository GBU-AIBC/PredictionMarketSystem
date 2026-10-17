from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from pms.config import PMSSettings, RiskSettings
from pms.core.enums import RunMode
from pms.runner import Runner, _verify_runtime_strategy_version
from pms.strategies.anchoring import (
    AnchoringAgent,
    AnchoringController,
    AnchoringLagStrategyModule,
    AnchoringMarketSnapshot,
    LiveAnchoringSource,
    build_anchoring_strategy,
)
from pms.strategies.intents import TradeIntent
from pms.strategies.versioning import compute_strategy_version_id


NOW = datetime(2026, 5, 4, 1, 0, tzinfo=UTC)


class _StaticAnchoringReader:
    def __init__(self, market: AnchoringMarketSnapshot | None) -> None:
        self.market = market
        self.calls: list[tuple[str, datetime]] = []

    async def latest(
        self,
        market_id: str,
        *,
        as_of: datetime,
    ) -> AnchoringMarketSnapshot | None:
        self.calls.append((market_id, as_of))
        return self.market


def _settings() -> PMSSettings:
    return PMSSettings(
        mode=RunMode.PAPER,
        risk=RiskSettings(max_position_per_market=5.0),
    )


def _market(**overrides: object) -> AnchoringMarketSnapshot:
    data = {
        "market_id": "market-h2-1",
        "title": "Will H2 anchoring lag production wiring emit the LLM side?",
        "yes_token_id": "token-yes",
        "no_token_id": "token-no",
        "yes_price": 0.50,
        "llm_posterior": 0.80,
        "llm_confidence": 0.70,
        "news_timestamp": NOW,
        "observed_at": NOW,
        "yes_best_ask": 0.51,
        "no_best_ask": 0.50,
        "resolves_at": NOW + timedelta(days=5),
        "news_ref": "news:headline-1",
    }
    data.update(overrides)
    return AnchoringMarketSnapshot(**data)  # type: ignore[arg-type]


def test_runner_builds_anchoring_strategy_module_with_live_components() -> None:
    runner = Runner(config=_settings())
    reader = _StaticAnchoringReader(_market())

    module = runner.build_anchoring_strategy_module(
        strategy_id="h2_anchoring_lag",
        strategy_version_id="h2-anchoring-lag-v1",
        market_ids=("market-h2-1",),
        market_reader=reader,
    )

    assert isinstance(module, AnchoringLagStrategyModule)
    assert isinstance(module.source, LiveAnchoringSource)
    assert isinstance(module.controller, AnchoringController)
    assert isinstance(module.agent, AnchoringAgent)
    assert module.strategy_id == "h2_anchoring_lag"
    assert module.strategy_version_id == "h2-anchoring-lag-v1"


def test_anchoring_runtime_requires_hashed_economics_version() -> None:
    strategy = build_anchoring_strategy("h2_anchoring_lag")
    version_id = compute_strategy_version_id(*strategy.snapshot())
    assert (
        _verify_runtime_strategy_version(strategy, version_id, legacy_ids=set())
        == version_id
    )
    with pytest.raises(ValueError, match="runtime strategy projection"):
        _verify_runtime_strategy_version(strategy, "arbitrary", legacy_ids=set())


@pytest.mark.asyncio
async def test_runner_calls_anchoring_module_without_live_capital() -> None:
    runner = Runner(config=_settings())
    reader = _StaticAnchoringReader(_market())

    results = await runner.run_anchoring_strategy_once(
        strategy_id="h2_anchoring_lag",
        strategy_version_id="h2-anchoring-lag-v1",
        market_ids=("market-h2-1",),
        as_of=NOW,
        market_reader=reader,
    )

    assert reader.calls == [("market-h2-1", NOW)]
    assert len(results) == 1
    judgement = results[0].judgement
    assert judgement is not None
    assert judgement.approved is True

    assert len(results[0].intents) == 1
    intent = results[0].intents[0]
    assert isinstance(intent, TradeIntent)
    assert intent.outcome == "YES"
    assert intent.token_id == "token-yes"
    assert intent.limit_price == pytest.approx(0.51)
    assert intent.expected_edge == pytest.approx(0.29)
    assert intent.notional_usdc <= 5.0

    assert runner.state.decisions == []
    assert runner._decision_queue.empty()  # noqa: SLF001


def test_runner_anchoring_module_requires_explicit_reader_until_news_source_lands() -> None:
    runner = Runner(config=_settings())

    with pytest.raises(RuntimeError, match="explicit H2 anchoring market reader"):
        runner.build_anchoring_strategy_module(
            strategy_id="h2_anchoring_lag",
            strategy_version_id="h2-anchoring-lag-v1",
            market_ids=("market-h2-1",),
        )
