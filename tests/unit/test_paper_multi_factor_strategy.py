from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from pms.config import LLMSettings, PMSSettings, RiskSettings
from pms.controller.factor_snapshot import FactorSnapshot
from pms.controller.factory import ControllerPipelineFactory
from pms.controller.pipeline import ControllerPipeline
from pms.controller.sizers.kelly import KellySizer
from pms.core.enums import RunMode
from pms.core.models import MarketSignal, Portfolio, Position
from pms.factors.composition import apply_composition
from pms.storage.strategy_registry import _strategy_from_config_json
from pms.strategies.paper_multifactor import (
    PAPER_MULTI_FACTOR_STRATEGY_ID,
    build_paper_multi_factor_strategy,
)
from pms.strategies.projections import FactorCompositionStep
from pms.strategies.versioning import serialize_strategy_config_json


def _signal(
    *,
    yes_price: float = 0.50,
    orderbook: dict[str, object] | None = None,
) -> MarketSignal:
    return MarketSignal(
        market_id="phase-a-market",
        token_id="phase-a-token",
        venue="polymarket",
        title="Can Phase A produce a simple factor-backed decision?",
        yes_price=yes_price,
        volume_24h=1_000.0,
        resolves_at=datetime(2026, 6, 1, tzinfo=UTC),
        orderbook=orderbook
        or {
            "bids": [{"price": 0.49, "size": 100.0}],
            "asks": [{"price": 0.51, "size": 100.0}],
        },
        external_signal={"raw_event_type": "book"},
        fetched_at=datetime(2026, 5, 5, 12, 0, tzinfo=UTC),
        market_status="open",
    )


def _portfolio(open_positions: list[Position] | None = None) -> Portfolio:
    positions = [] if open_positions is None else open_positions
    locked_usdc = sum(position.locked_usdc for position in positions)
    return Portfolio(
        total_usdc=1_000.0,
        free_usdc=1_000.0 - locked_usdc,
        locked_usdc=locked_usdc,
        open_positions=positions,
    )


def _position(*, locked_usdc: float = 0.0) -> Position:
    return Position(
        market_id="phase-a-market",
        token_id="phase-a-token",
        venue="polymarket",
        side="BUY",
        shares_held=locked_usdc / 0.50 if locked_usdc > 0.0 else 0.0,
        avg_entry_price=0.50,
        unrealized_pnl=0.0,
        locked_usdc=locked_usdc,
        strategy_id=PAPER_MULTI_FACTOR_STRATEGY_ID,
        strategy_version_id="phase-a-v1",
    )


class StaticFactorReader:
    def __init__(self, snapshot: FactorSnapshot) -> None:
        self.snapshot_calls: list[Sequence[FactorCompositionStep]] = []
        self._snapshot = snapshot

    async def snapshot(
        self,
        *,
        market_id: str,
        as_of: datetime,
        required: Sequence[FactorCompositionStep],
        strategy_id: str,
        strategy_version_id: str,
    ) -> Any:
        del market_id, as_of, strategy_id, strategy_version_id
        self.snapshot_calls.append(required)
        return self._snapshot


def test_paper_multi_factor_strategy_matches_phase_a_contract() -> None:
    strategy = build_paper_multi_factor_strategy()

    assert strategy.config.strategy_id == PAPER_MULTI_FACTOR_STRATEGY_ID
    assert strategy.calibration.enabled is True
    assert strategy.calibration.shrinkage_factor == pytest.approx(1.0)
    assert strategy.calibration.shrinkage_bias == pytest.approx(0.0)
    assert strategy.calibration.extreme_clamp_low == pytest.approx(0.08)
    assert strategy.calibration.extreme_clamp_high == pytest.approx(0.92)
    assert strategy.calibration.min_resolved_for_extreme == 20
    assert dict(strategy.config.metadata) == {
        "owner": "Researcher-Ciga",
        "tier": "paper",
        "phase": "A",
        "purpose": "paper_multi_factor_phase_a",
        "price_reference": "best_ask",
        "live_allowed": "false",
    }
    assert strategy.risk.max_position_notional_usdc == pytest.approx(1.0)
    assert strategy.risk.max_daily_drawdown_pct == pytest.approx(50.0)
    assert strategy.risk.min_order_size_usdc == pytest.approx(1.0)
    assert strategy.eval_spec.metrics == ("brier", "pnl", "fill_rate")
    assert strategy.eval_spec.min_win_rate == pytest.approx(0.45)
    assert strategy.market_selection.venue == "polymarket"
    assert strategy.market_selection.resolution_time_max_horizon_days == 31
    assert strategy.market_selection.volume_min_usdc == pytest.approx(100.0)
    assert strategy.market_selection.spread_max_bps is None
    assert strategy.market_selection.yes_price_min == pytest.approx(0.02)
    assert strategy.market_selection.yes_price_max == pytest.approx(0.98)
    assert strategy.forecaster.forecasters == (
        ("rules", (("threshold", "0.55"),)),
        ("stats", (("window", "15m"),)),
        ("llm", ()),
    )
    assert tuple(
        (
            step.factor_id,
            step.role,
            step.weight,
            step.threshold,
            step.required,
            step.enabled,
        )
        for step in strategy.config.factor_composition
    ) == (
        ("orderbook_imbalance", "rule_delta", 0.25, 0.80, True, True),
        ("metaculus_prior", "rule_delta", 0.3, None, False, True),
        ("favorite_longshot_bias", "rule_delta", 0.2, None, False, True),
        ("rules", "blend_weighted", 1.0, None, False, True),
        ("llm", "runtime_probability", 1.0, None, False, True),
        ("llm", "blend_weighted", 1.0, None, False, True),
    )


def test_paper_multi_factor_blends_llm_runtime_probability_when_available() -> None:
    strategy = build_paper_multi_factor_strategy()
    factor_values = {
        ("yes_price", ""): 0.50,
        ("orderbook_imbalance", ""): 0.85,
        ("favorite_longshot_bias", ""): 0.00,
        ("rules", ""): 0.60,
        ("llm", ""): 0.70,
    }

    probability = apply_composition(strategy.config.factor_composition, factor_values)

    assert probability == pytest.approx(0.70625)


def test_paper_multi_factor_falls_back_to_rules_when_llm_is_unavailable() -> None:
    strategy = build_paper_multi_factor_strategy()
    factor_values = {
        ("yes_price", ""): 0.50,
        ("orderbook_imbalance", ""): 0.85,
        ("favorite_longshot_bias", ""): 0.00,
        ("rules", ""): 0.60,
    }

    probability = apply_composition(strategy.config.factor_composition, factor_values)

    assert probability == pytest.approx(0.7125)


def test_paper_multi_factor_config_json_round_trips_enabled_calibration() -> None:
    strategy = build_paper_multi_factor_strategy()

    config_json = serialize_strategy_config_json(*strategy.snapshot())
    payload = json.loads(config_json)
    round_tripped = _strategy_from_config_json(payload)

    assert payload["calibration"]["enabled"] is True
    assert payload["calibration"]["shrinkage_factor"] == pytest.approx(1.0)
    assert payload["calibration"]["shrinkage_bias"] == pytest.approx(0.0)
    assert payload["calibration"]["extreme_clamp_low"] == pytest.approx(0.08)
    assert payload["calibration"]["extreme_clamp_high"] == pytest.approx(0.92)
    assert payload["calibration"]["min_resolved_for_extreme"] == 20
    assert payload["market_selection"]["spread_max_bps"] is None
    assert payload["market_selection"]["yes_price_min"] == pytest.approx(0.02)
    assert payload["market_selection"]["yes_price_max"] == pytest.approx(0.98)
    assert round_tripped.calibration.enabled is True
    assert round_tripped.calibration.shrinkage_factor == pytest.approx(1.0)
    assert round_tripped.calibration.shrinkage_bias == pytest.approx(0.0)
    assert round_tripped.calibration.extreme_clamp_low == pytest.approx(0.08)
    assert round_tripped.calibration.extreme_clamp_high == pytest.approx(0.92)
    assert round_tripped.calibration.min_resolved_for_extreme == 20
    assert round_tripped.market_selection.spread_max_bps is None
    assert round_tripped.market_selection.yes_price_min == pytest.approx(0.02)
    assert round_tripped.market_selection.yes_price_max == pytest.approx(0.98)


def test_paper_multi_factor_strategy_is_rejected_outside_paper_mode() -> None:
    settings = PMSSettings(mode=RunMode.LIVE)
    strategy = build_paper_multi_factor_strategy()

    with pytest.raises(ValueError, match="paper_multi_factor_v1 is PAPER-only"):
        ControllerPipelineFactory(settings=settings).build(
            strategy.to_active(strategy_version_id="phase-a-v1")
        )


def test_paper_multi_factor_llm_enabled_requires_provider_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = build_paper_multi_factor_strategy()

    def missing_provider(module_name: str) -> object:
        raise ImportError(module_name)

    monkeypatch.setattr(
        "pms.controller.forecasters.llm.import_module",
        missing_provider,
    )
    settings = PMSSettings(
        mode=RunMode.PAPER,
        llm=LLMSettings(
            enabled=True,
            provider="anthropic",
            api_key="test-key",
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="LLM provider package 'anthropic' is required",
    ):
        ControllerPipelineFactory(settings=settings).build(
            strategy.to_active(strategy_version_id="phase-a-v1")
        )


def test_controller_factory_does_not_let_strategy_min_order_undercut_runtime_risk() -> None:
    strategy = build_paper_multi_factor_strategy()
    pipeline = ControllerPipelineFactory(
        settings=PMSSettings(
            mode=RunMode.PAPER,
            risk=RiskSettings(min_order_usdc=1.0),
        )
    ).build(strategy.to_active(strategy_version_id="phase-a-v1"))

    assert isinstance(pipeline.sizer, KellySizer)
    assert pipeline.sizer.risk is not None
    assert pipeline.sizer.risk.min_order_usdc == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_paper_multi_factor_can_decide_from_orderbook_imbalance_only() -> None:
    strategy = build_paper_multi_factor_strategy()
    factor_reader = StaticFactorReader(
        FactorSnapshot(
            values={("orderbook_imbalance", ""): 0.85},
            missing_factors=(),
            snapshot_hash="phase-a-snapshot",
        )
    )
    pipeline = ControllerPipelineFactory(
        settings=PMSSettings(mode=RunMode.PAPER),
        factor_reader=factor_reader,
    ).build(strategy.to_active(strategy_version_id="phase-a-v1"))

    decision = await pipeline.decide(
        _signal(
            orderbook={
                "bids": [{"price": 0.499, "size": 92.5}],
                "asks": [{"price": 0.501, "size": 7.5}],
            }
        ),
        portfolio=_portfolio(),
    )

    assert decision is not None
    assert decision.strategy_id == PAPER_MULTI_FACTOR_STRATEGY_ID
    assert decision.outcome == "YES"
    assert decision.limit_price == pytest.approx(0.501)
    assert decision.prob_estimate == pytest.approx(0.7125)
    assert decision.expected_edge > 0.05
    assert decision.notional_usdc == pytest.approx(1.0)
    assert factor_reader.snapshot_calls


@pytest.mark.asyncio
async def test_paper_multi_factor_does_not_create_longshot_edge_from_near_market_forecast() -> None:
    strategy = build_paper_multi_factor_strategy()
    factor_reader = StaticFactorReader(
        FactorSnapshot(
            values={},
            missing_factors=(),
            snapshot_hash="neutral-longshot-snapshot",
        )
    )
    pipeline = ControllerPipeline(
        strategy=strategy.to_active(strategy_version_id="phase-a-v1"),
        factor_reader=factor_reader,
        forecasters=(),
        settings=PMSSettings(mode=RunMode.PAPER),
    )

    decision = await pipeline.decide(
        _signal(
            yes_price=0.021,
            orderbook={
                "bids": [{"price": 0.02099, "size": 100.0}],
                "asks": [{"price": 0.02101, "size": 100.0}],
            },
        ),
        portfolio=_portfolio(),
    )

    assert decision is None
    assert pipeline.last_diagnostic is not None
    assert pipeline.last_diagnostic.code == "calibration_clamp_rejected"
    assert factor_reader.snapshot_calls


@pytest.mark.asyncio
async def test_paper_multi_factor_skips_remaining_capacity_below_runtime_min_order() -> None:
    strategy = build_paper_multi_factor_strategy()
    factor_reader = StaticFactorReader(
        FactorSnapshot(
            values={("orderbook_imbalance", ""): 0.85},
            missing_factors=(),
            snapshot_hash="phase-a-snapshot",
        )
    )
    pipeline = ControllerPipelineFactory(
        settings=PMSSettings(
            mode=RunMode.PAPER,
            risk=RiskSettings(min_order_usdc=1.0),
        ),
        factor_reader=factor_reader,
    ).build(strategy.to_active(strategy_version_id="phase-a-v1"))

    decision = await pipeline.decide(
        _signal(
            orderbook={
                "bids": [{"price": 0.499, "size": 92.5}],
                "asks": [{"price": 0.501, "size": 7.5}],
            }
        ),
        portfolio=_portfolio(open_positions=[_position(locked_usdc=1.31)]),
    )

    assert decision is None
    assert pipeline.last_diagnostic is not None
    assert pipeline.last_diagnostic.code == "market_position_capacity_below_minimum"


@pytest.mark.asyncio
async def test_paper_multi_factor_prefers_signal_local_orderbook_imbalance_over_stale_snapshot() -> None:
    strategy = build_paper_multi_factor_strategy()
    factor_reader = StaticFactorReader(
        FactorSnapshot(
            values={("orderbook_imbalance", ""): -1.0},
            missing_factors=(),
            stale_factors=(("orderbook_imbalance", ""),),
            snapshot_hash="stale-imbalance-snapshot",
        )
    )
    pipeline = ControllerPipelineFactory(
        settings=PMSSettings(mode=RunMode.PAPER),
        factor_reader=factor_reader,
    ).build(strategy.to_active(strategy_version_id="phase-a-v1"))

    decision = await pipeline.decide(
        _signal(
            orderbook={
                "bids": [{"price": 0.499, "size": 92.5}],
                "asks": [{"price": 0.501, "size": 7.5}],
            }
        ),
        portfolio=_portfolio(),
    )

    assert decision is not None
    assert decision.outcome == "YES"
    assert decision.prob_estimate > 0.50
