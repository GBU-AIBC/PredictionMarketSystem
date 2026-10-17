from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
import tomllib

import pytest

from pms.config import PMSSettings, RiskSettings
from pms.core.enums import TimeInForce
from pms.core.models import Portfolio
from pms.controller.sizers.kelly import KellySizer
from pms.runner import Runner
from pms.strategies.base import (
    StrategyAgent,
    StrategyController,
    StrategyModule,
    StrategyObservationSource,
)
from pms.strategies.intents import (
    StrategyCandidate,
    StrategyContext,
    StrategyJudgement,
    StrategyObservation,
    TradeIntent,
)
from pms.strategies.ripple.agent import RippleAgent
from pms.strategies.ripple.controller import RippleController
from pms.strategies.ripple.evaluator import (
    RippleEvidenceEvaluator,
    posterior_confidence,
)
from pms.strategies.ripple.source import (
    LiveRippleSource,
    RIPPLE_FACTOR_REQUIREMENTS,
    RippleMarketSnapshot,
    RippleObservationFixture,
    RippleObservationSource,
    RipplePositionSizer,
)
from pms.strategies.ripple.strategy import RippleStrategyModule
from pms.strategies.projections import FactorCompositionStep


NOW = datetime(2026, 4, 28, 12, 0, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[2]


def _context() -> StrategyContext:
    return StrategyContext(
        strategy_id="ripple",
        strategy_version_id="ripple-v1",
        as_of=NOW,
    )


def _fixture(**overrides: object) -> RippleObservationFixture:
    data: dict[str, object] = {
        "observation_id": "ripple-observation-1",
        "market_id": "market-ripple-1",
        "title": "Will the deterministic ripple fixture resolve YES?",
        "thesis": "Fixture evidence supports a YES edge.",
        "probability_estimate": 0.63,
        "expected_edge": 0.09,
        "confidence": 0.78,
        "token_id": "token-ripple-yes",
        "limit_price": 0.56,
        "notional_usdc": 25.0,
        "expected_price": 0.64,
        "max_slippage_bps": 40,
        "evidence_refs": ("doc://ripple/a", "doc://ripple/b"),
        "contradiction_refs": (),
        "metadata": {
            "metaculus_prior": 0.63,
            "prior_strength": 2.0,
            "yes_count": 8.0,
            "no_count": 2.0,
            "entry_edge_threshold": 0.02,
        },
    }
    data.update(overrides)
    return RippleObservationFixture(**cast(Any, data))


class _FakeFactorSnapshot:
    def __init__(
        self,
        *,
        values: Mapping[tuple[str, str], float] | None = None,
        missing_factors: tuple[tuple[str, str], ...] = (),
        stale_factors: tuple[tuple[str, str], ...] = (),
        snapshot_hash: str | None = "factor-hash-1",
    ) -> None:
        self.values = values or {
            ("metaculus_prior", ""): 0.67,
            ("orderbook_imbalance", ""): 0.21,
            ("fair_value_spread", ""): 0.08,
            ("yes_count", ""): 4.0,
            ("no_count", ""): 1.0,
        }
        self.missing_factors = missing_factors
        self.stale_factors = stale_factors
        self.snapshot_hash = snapshot_hash


class _RecordingFactorReader:
    def __init__(self, snapshot: _FakeFactorSnapshot | None = None) -> None:
        self.snapshot_result = snapshot or _FakeFactorSnapshot()
        self.calls: list[dict[str, object]] = []

    async def snapshot(
        self,
        *,
        market_id: str,
        as_of: datetime,
        required: Sequence[FactorCompositionStep],
        strategy_id: str,
        strategy_version_id: str,
    ) -> _FakeFactorSnapshot:
        self.calls.append(
            {
                "market_id": market_id,
                "as_of": as_of,
                "required": required,
                "strategy_id": strategy_id,
                "strategy_version_id": strategy_version_id,
            }
        )
        return self.snapshot_result


class _LegacyRecordingFactorReader:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def snapshot(
        self,
        *,
        market_id: str,
        as_of: datetime,
        required: Sequence[FactorCompositionStep],
        strategy_id: str,
        strategy_version_id: str,
    ) -> _FakeFactorSnapshot:
        self.calls.append(
            {
                "market_id": market_id,
                "as_of": as_of,
                "required": required,
                "strategy_id": strategy_id,
                "strategy_version_id": strategy_version_id,
            }
        )
        return _FakeFactorSnapshot()


class _StaticMarketReader:
    def __init__(
        self,
        *,
        yes_price: float = 0.59,
        best_bid: float | None = 0.58,
        best_ask: float | None = 0.60,
        resolves_at: datetime | None = None,
    ) -> None:
        self.yes_price = yes_price
        self.best_bid = best_bid
        self.best_ask = best_ask
        self.resolves_at = resolves_at

    async def latest(
        self,
        market_id: str,
        *,
        as_of: datetime,
    ) -> RippleMarketSnapshot | None:
        del as_of
        return RippleMarketSnapshot(
            market_id=market_id,
            title="Will the live factor source resolve YES?",
            token_id="token-ripple-yes",
            yes_price=self.yes_price,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            observed_at=NOW,
            resolves_at=self.resolves_at,
        )


class _ZeroSizer:
    def size(
        self,
        *,
        prob: float,
        market_price: float,
        portfolio: Portfolio,
    ) -> float:
        del prob, market_price, portfolio
        return 0.0


class _FixedSizer:
    def __init__(self, notional_usdc: float) -> None:
        self.notional_usdc = notional_usdc

    def size(
        self,
        *,
        prob: float,
        market_price: float,
        portfolio: Portfolio,
    ) -> float:
        del prob, market_price, portfolio
        return self.notional_usdc


async def _candidate_from_fixture(
    fixture: RippleObservationFixture,
) -> StrategyCandidate:
    context = _context()
    observations = await RippleObservationSource([fixture]).observe(context)
    candidates = await RippleController().propose(context, observations)
    assert len(candidates) == 1
    return candidates[0]


def _live_candidate_with_metadata(
    *,
    probability_estimate: float = 0.76,
    expected_edge: float = 0.26,
    metadata: Mapping[str, object],
) -> StrategyCandidate:
    return StrategyCandidate(
        candidate_id="candidate-ripple-live",
        strategy_id="ripple",
        strategy_version_id="ripple-v1",
        market_id="market-ripple-1",
        title="Will posterior math stay deterministic?",
        thesis="Live evidence supports a posterior edge.",
        probability_estimate=probability_estimate,
        expected_edge=expected_edge,
        evidence_refs=("factor_snapshot:hash", "market_snapshot:market-ripple-1"),
        created_at=NOW,
        metadata=metadata,
    )


def _portfolio(free_usdc: float = 100.0) -> Portfolio:
    return Portfolio(
        total_usdc=free_usdc,
        free_usdc=free_usdc,
        locked_usdc=0.0,
        open_positions=[],
    )


def test_ripple_components_satisfy_strategy_protocols() -> None:
    source: StrategyObservationSource = RippleObservationSource([_fixture()])
    controller: StrategyController = RippleController()
    agent: StrategyAgent = RippleAgent()
    module: StrategyModule = RippleStrategyModule(
        source=source,
        controller=controller,
        agent=agent,
        strategy_id="ripple",
        strategy_version_id="ripple-v1",
    )

    assert module.strategy_id == "ripple"
    assert module.strategy_version_id == "ripple-v1"


@pytest.mark.asyncio
async def test_ripple_strategy_approves_fixture_candidate_and_emits_trade_intent() -> None:
    module = RippleStrategyModule(
        source=RippleObservationSource([_fixture()]),
        controller=RippleController(),
        agent=RippleAgent(),
        strategy_id="ripple",
        strategy_version_id="ripple-v1",
    )

    intents = await module.run(_context())

    assert len(intents) == 1
    intent = intents[0]
    assert isinstance(intent, TradeIntent)
    assert intent.strategy_id == "ripple"
    assert intent.strategy_version_id == "ripple-v1"
    assert intent.candidate_id == "candidate-ripple-observation-1"
    assert intent.token_id == "token-ripple-yes"
    assert intent.time_in_force is TimeInForce.GTC


@pytest.mark.asyncio
async def test_live_ripple_source_reads_factor_snapshot_and_market_price() -> None:
    factor_reader = _LegacyRecordingFactorReader()
    source = LiveRippleSource(
        market_ids=("market-ripple-1",),
        factor_reader=factor_reader,
        market_reader=_StaticMarketReader(),
        position_sizer=KellySizer(risk=RiskSettings(max_position_per_market=100.0)),
        portfolio=_portfolio(),
    )

    observations = await source.observe(_context())
    candidates = await RippleController().propose(_context(), observations)

    assert len(observations) == 1
    observation = observations[0]
    assert observation.source == "live_factor_service"
    assert observation.payload["probability_estimate"] == pytest.approx(
        (0.67 * 2.0 + 4.0) / (2.0 + 4.0 + 1.0)
    )
    assert observation.payload["expected_edge"] == pytest.approx(
        observation.payload["probability_estimate"] - 0.60
    )
    assert observation.payload["limit_price"] == 0.60
    assert observation.payload["expected_price"] == observation.payload["probability_estimate"]
    assert observation.payload["metadata"]["yes_price"] == 0.59
    assert observation.payload["metadata"]["factor_values"] == {
        "fair_value_spread": 0.08,
        "metaculus_prior": 0.67,
        "orderbook_imbalance": 0.21,
        "yes_count": 4.0,
        "no_count": 1.0,
    }
    assert observation.payload["metadata"]["posterior_probability"] == pytest.approx(
        observation.payload["probability_estimate"]
    )
    assert observation.payload["metadata"]["entry_edge_threshold"] == pytest.approx(0.02)
    assert observation.evidence_refs == (
        "factor_snapshot:factor-hash-1",
        "market_snapshot:market-ripple-1:2026-04-28T12:00:00+00:00",
    )
    assert len(factor_reader.calls) == 1
    call = factor_reader.calls[0]
    assert call["market_id"] == "market-ripple-1"
    assert call["as_of"] == NOW
    assert call["strategy_id"] == "ripple"
    assert call["strategy_version_id"] == "ripple-v1"
    required_factor_ids = {
        step.factor_id
        for step in cast(Any, call["required"])
    }
    assert required_factor_ids == {
        "fair_value_spread",
        "metaculus_prior",
        "orderbook_imbalance",
        "yes_count",
        "no_count",
    }
    assert candidates[0].metadata["source"] == "live_factor_service"
    assert candidates[0].metadata["fixture_metadata"]["factor_snapshot_hash"] == (
        "factor-hash-1"
    )


@pytest.mark.asyncio
async def test_live_ripple_fails_closed_without_fresh_metaculus_prior() -> None:
    source = LiveRippleSource(
        market_ids=("market-ripple-1",),
        factor_reader=_RecordingFactorReader(
            _FakeFactorSnapshot(
                values={
                    ("yes_count", ""): 15.0,
                    ("no_count", ""): 5.0,
                },
                missing_factors=(("metaculus_prior", ""),),
                stale_factors=(("yes_count", ""),),
            )
        ),
        market_reader=_StaticMarketReader(yes_price=0.50, best_ask=0.50),
        position_sizer=_FixedSizer(2.0),
        portfolio=_portfolio(),
    )

    assert await source.observe(_context()) == ()


def test_ripple_evaluator_rejects_provenance_marked_stale_prior() -> None:
    candidate = _live_candidate_with_metadata(
        metadata={
            "source": "live_factor_service",
            "metaculus_prior": 0.7,
            "prior_strength": 2.0,
            "yes_count": 15.0,
            "no_count": 5.0,
            "limit_price": 0.5,
            "factor_snapshot_hash": "stale-snapshot",
            "missing_factors": (),
            "stale_factors": ("metaculus_prior",),
            "contradiction_refs": (),
        },
    )

    assessment = RippleEvidenceEvaluator(min_confidence=0.6).assess(candidate)

    assert assessment.approved is False
    assert assessment.failure_reasons == ("low_confidence",)


def test_live_ripple_requires_beta_binomial_count_factors() -> None:
    required_flags = {
        step.factor_id: step.required
        for step in RIPPLE_FACTOR_REQUIREMENTS
    }

    assert required_flags["yes_count"] is True
    assert required_flags["no_count"] is True


def test_kelly_sizer_satisfies_ripple_position_sizer_protocol() -> None:
    sizer = KellySizer(risk=RiskSettings(max_position_per_market=100.0))

    assert isinstance(sizer, RipplePositionSizer)


@pytest.mark.parametrize(
    ("prob", "market_price", "expected_notional"),
    [
        (0.52, 0.50, 10.0),
        (0.90, 0.50, 100.0),
        (0.50, 0.50, 0.0),
        (0.49, 0.50, 0.0),
    ],
)
def test_kelly_sizer_scales_ripple_edges_and_respects_risk_cap(
    prob: float,
    market_price: float,
    expected_notional: float,
) -> None:
    sizer = KellySizer(risk=RiskSettings(max_position_per_market=100.0))

    assert sizer.size(
        prob=prob,
        market_price=market_price,
        portfolio=_portfolio(free_usdc=1000.0),
    ) == pytest.approx(expected_notional)


def test_runner_builds_live_ripple_source_with_fractional_kelly_sizer() -> None:
    risk = RiskSettings(max_position_per_market=100.0)
    runner = Runner(config=PMSSettings(risk=risk))

    module = runner.build_ripple_strategy_module(
        strategy_id="ripple",
        strategy_version_id="ripple-v1",
        market_ids=("market-ripple-1",),
        factor_reader=_RecordingFactorReader(),
        market_reader=_StaticMarketReader(),
    )

    assert isinstance(module.source, LiveRippleSource)
    assert isinstance(module.source.position_sizer, KellySizer)
    assert module.source.position_sizer.risk is not None
    assert module.source.position_sizer.risk.max_position_per_market == pytest.approx(1.0)
    assert module.source.position_sizer.fraction == Decimal("0.25")


def test_ripple_evaluator_emits_beta_binomial_posterior_edge_and_confidence() -> None:
    candidate = _live_candidate_with_metadata(
        metadata={
            "source": "live_factor_service",
            "metaculus_prior": 0.6,
            "prior_strength": 2.0,
            "yes_count": 8.0,
            "no_count": 2.0,
            "limit_price": 0.5,
            "confidence": 0.8,
            "contradiction_refs": (),
        },
    )

    assessment = RippleEvidenceEvaluator(min_confidence=0.6).assess(candidate)

    assert assessment.approved is True
    assert assessment.posterior_probability == pytest.approx((0.6 * 2.0 + 8.0) / 12.0)
    assert assessment.expected_edge == pytest.approx(assessment.posterior_probability - 0.5)
    assert assessment.confidence == pytest.approx(
        posterior_confidence(prior_strength=2.0, yes_count=8.0, no_count=2.0)
    )
    assert assessment.entry_edge_threshold == pytest.approx(0.02)


def test_ripple_evaluator_derives_confidence_from_counts_not_candidate_metadata() -> None:
    candidate = _live_candidate_with_metadata(
        metadata={
            "source": "live_factor_service",
            "metaculus_prior": 0.6,
            "prior_strength": 2.0,
            "yes_count": 8.0,
            "no_count": 2.0,
            "limit_price": 0.5,
            "confidence": 0.01,
            "contradiction_refs": (),
        },
    )

    assessment = RippleEvidenceEvaluator(min_confidence=0.6).assess(candidate)

    assert assessment.approved is True
    assert assessment.confidence == pytest.approx(
        posterior_confidence(prior_strength=2.0, yes_count=8.0, no_count=2.0)
    )
    assert assessment.confidence > 0.5


def test_ripple_evaluator_prior_only_confidence_stays_at_half() -> None:
    candidate = _live_candidate_with_metadata(
        metadata={
            "source": "live_factor_service",
            "metaculus_prior": 0.6,
            "prior_strength": 2.0,
            "limit_price": 0.5,
            "confidence": 0.99,
            "contradiction_refs": (),
        },
    )

    assessment = RippleEvidenceEvaluator(min_confidence=0.6).assess(candidate)

    assert assessment.approved is False
    assert assessment.failure_reasons == ("low_confidence",)
    assert assessment.confidence == pytest.approx(0.5)


def test_ripple_evaluator_caps_confidence_without_metaculus_prior() -> None:
    candidate = _live_candidate_with_metadata(
        probability_estimate=0.7,
        expected_edge=0.2,
        metadata={
            "source": "live_factor_service",
            "prior_strength": 2.0,
            "yes_count": 10.0,
            "no_count": 0.0,
            "limit_price": 0.5,
            "confidence": 0.99,
            "contradiction_refs": (),
        },
    )

    assessment = RippleEvidenceEvaluator(min_confidence=0.6).assess(candidate)

    assert assessment.approved is False
    assert assessment.failure_reasons == ("low_confidence",)
    assert assessment.confidence == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_live_ripple_uses_fractional_kelly_sizing_instead_of_fixture_notional() -> None:
    sizer = KellySizer(risk=RiskSettings(max_position_per_market=100.0))
    portfolio = _portfolio(free_usdc=100.0)
    source = LiveRippleSource(
        market_ids=("market-ripple-1",),
        factor_reader=_RecordingFactorReader(),
        market_reader=_StaticMarketReader(best_ask=0.60),
        position_sizer=sizer,
        portfolio=portfolio,
    )
    module = RippleStrategyModule(
        source=source,
        controller=RippleController(),
        agent=RippleAgent(),
        strategy_id="ripple",
        strategy_version_id="ripple-v1",
    )

    intents = await module.run(_context())

    assert len(intents) == 1
    intent = intents[0]
    assert isinstance(intent, TradeIntent)
    expected_probability = (0.67 * 2.0 + 4.0) / (2.0 + 4.0 + 1.0)
    assert intent.expected_price == pytest.approx(expected_probability)
    assert intent.expected_edge == pytest.approx(expected_probability - 0.60)
    assert intent.notional_usdc == pytest.approx(
        sizer.size(prob=expected_probability, market_price=0.60, portfolio=portfolio)
    )
    assert intent.notional_usdc != 25.0


@pytest.mark.asyncio
async def test_live_ripple_zero_kelly_size_is_clean_no_trade() -> None:
    source = LiveRippleSource(
        market_ids=("market-ripple-1",),
        factor_reader=_RecordingFactorReader(),
        market_reader=_StaticMarketReader(best_ask=0.60),
        position_sizer=_ZeroSizer(),
        portfolio=_portfolio(),
    )
    module = RippleStrategyModule(
        source=source,
        controller=RippleController(),
        agent=RippleAgent(),
        strategy_id="ripple",
        strategy_version_id="ripple-v1",
    )

    assert await source.observe(_context()) == ()
    assert await module.run(_context()) == ()


@pytest.mark.asyncio
async def test_live_ripple_rejects_zero_or_negative_edge_before_building_intent() -> None:
    source = LiveRippleSource(
        market_ids=("market-ripple-1",),
        factor_reader=_RecordingFactorReader(
            _FakeFactorSnapshot(
                values={
                    ("metaculus_prior", ""): 0.4,
                    ("yes_count", ""): 1.0,
                    ("no_count", ""): 9.0,
                }
            )
        ),
        market_reader=_StaticMarketReader(yes_price=0.55, best_ask=0.56),
        position_sizer=_FixedSizer(2.0),
        portfolio=_portfolio(),
    )
    observations = await source.observe(_context())
    candidates = await RippleController().propose(_context(), observations)
    agent = RippleAgent(min_confidence=0.6)

    judgement = await agent.judge(_context(), candidates[0])
    intents = await agent.build_intents(_context(), candidates[0], judgement)

    assert judgement.approved is False
    assert judgement.failure_reasons == ("insufficient_expected_edge",)
    assert intents == ()


@pytest.mark.asyncio
async def test_live_ripple_raises_entry_threshold_near_resolution() -> None:
    source = LiveRippleSource(
        market_ids=("market-ripple-1",),
        factor_reader=_RecordingFactorReader(
            _FakeFactorSnapshot(
                values={
                    ("metaculus_prior", ""): 0.58,
                    ("yes_count", ""): 9.0,
                    ("no_count", ""): 6.0,
                }
            )
        ),
        market_reader=_StaticMarketReader(
            yes_price=0.55,
            best_ask=0.56,
            resolves_at=datetime(2026, 4, 28, 16, 0, tzinfo=UTC),
        ),
        position_sizer=KellySizer(risk=RiskSettings(max_position_per_market=100.0)),
        portfolio=_portfolio(),
    )
    observations = await source.observe(_context())
    candidates = await RippleController().propose(_context(), observations)
    agent = RippleAgent(min_confidence=0.6)

    judgement = await agent.judge(_context(), candidates[0])

    assert candidates[0].expected_edge == pytest.approx(
        ((0.58 * 2.0 + 9.0) / (2.0 + 9.0 + 6.0)) - 0.56
    )
    assert candidates[0].expected_edge > 0.02
    assert candidates[0].metadata["entry_edge_threshold"] > 0.02
    assert judgement.approved is False
    assert judgement.failure_reasons == ("insufficient_expected_edge",)


@pytest.mark.parametrize(
    ("fixture", "expected_reason"),
    [
        (
            _fixture(evidence_refs=("doc://ripple/only-one",)),
            "insufficient_evidence",
        ),
        (
            _fixture(contradiction_refs=("doc://ripple/contradiction",)),
            "contradiction",
        ),
        (
            _fixture(
                metadata={
                    "metaculus_prior": 0.63,
                    "prior_strength": 2.0,
                    "entry_edge_threshold": 0.02,
                }
            ),
            "low_confidence",
        ),
    ],
)
@pytest.mark.asyncio
async def test_ripple_agent_emits_typed_rejections_without_trade_intents(
    fixture: RippleObservationFixture,
    expected_reason: str,
) -> None:
    context = _context()
    candidate = await _candidate_from_fixture(fixture)
    agent = RippleAgent(min_evidence_refs=2, min_confidence=0.6)

    judgement = await agent.judge(context, candidate)
    intents = await agent.build_intents(context, candidate, judgement)

    assert isinstance(judgement, StrategyJudgement)
    assert judgement.approved is False
    assert judgement.failure_reasons == (expected_reason,)
    assert judgement.strategy_id == "ripple"
    assert judgement.strategy_version_id == "ripple-v1"
    assert intents == ()


@pytest.mark.asyncio
async def test_ripple_observation_source_and_controller_parse_fixture_payload() -> None:
    context = _context()
    source = RippleObservationSource([_fixture(observation_id="obs-custom")])

    observations = await source.observe(context)
    candidates = await RippleController().propose(context, observations)

    assert len(observations) == 1
    observation = observations[0]
    assert isinstance(observation, StrategyObservation)
    assert observation.observation_id == "obs-custom"
    assert observation.source == "ripple-fixture"
    assert observation.evidence_refs == ("doc://ripple/a", "doc://ripple/b")
    assert len(candidates) == 1
    assert candidates[0].candidate_id == "candidate-obs-custom"
    assert candidates[0].metadata["token_id"] == "token-ripple-yes"


def test_ripple_agent_and_prompts_are_fixture_driven_only() -> None:
    checked_paths = [
        ROOT / "src" / "pms" / "strategies" / "ripple" / "agent.py",
        ROOT / "src" / "pms" / "strategies" / "ripple" / "prompts.py",
    ]
    forbidden_fragments = (
        "anthropic",
        "openai",
        "ollama",
        "litellm",
        "py_clob_client_v2",
    )

    for path in checked_paths:
        text = path.read_text(encoding="utf-8").lower()
        for fragment in forbidden_fragments:
            assert fragment not in text


def test_ripple_import_linter_contract_is_declared() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    contracts: Sequence[dict[str, object]] = pyproject["tool"]["importlinter"][
        "contracts"
    ]
    contract = next(
        candidate
        for candidate in contracts
        if candidate["name"] == "Strategy plugins: no actuator, controller, or venue adapter imports"
    )

    assert contract["type"] == "forbidden"
    assert contract["source_modules"] == [
        "pms.strategies.ripple",
        "pms.strategies.flb",
        "pms.strategies.anchoring",
    ]
    assert contract["forbidden_modules"] == [
        "pms.actuator",
        "pms.actuator.adapters",
        "pms.controller",
        "py_clob_client_v2",
    ]
