from __future__ import annotations

from datetime import UTC, datetime

import pytest

from pms.config import LLMSettings, RiskSettings
from pms.actuator.risk import InsufficientLiquidityError
from pms.controller.forecasters.llm import (
    LLMForecaster,
    _ProviderResponse,
    _prompt,
)
from pms.controller.sizers.kelly import KellySizer
from pms.core.enums import Side, TimeInForce
from pms.core.models import MarketSignal, Portfolio, TradeDecision
from pms.research.execution import _walk_book
from pms.research.replay import _market_resolution_label


def _signal(external_signal: dict[str, object]) -> MarketSignal:
    return MarketSignal(
        market_id="w1c-market",
        token_id="w1c-token",
        venue="polymarket",
        title="A market title",
        yes_price=0.4,
        volume_24h=100.0,
        resolves_at=datetime(2026, 1, 2, tzinfo=UTC),
        orderbook={"bids": [{"price": 0.39, "size": 1.0}]},
        external_signal=external_signal,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        market_status="open",
    )


def _decision() -> TradeDecision:
    return TradeDecision(
        decision_id="w1c-decision",
        market_id="w1c-market",
        token_id="w1c-token",
        venue="polymarket",
        side=Side.BUY.value,
        notional_usdc=10.0,
        order_type="limit",
        max_slippage_bps=50,
        stop_conditions=[],
        prob_estimate=0.7,
        expected_edge=0.3,
        time_in_force=TimeInForce.IOC,
        opportunity_id="w1c-opportunity",
        strategy_id="w1c",
        strategy_version_id="w1c-v1",
        limit_price=0.4,
    )


def test_llm_prompt_strips_nested_resolution_labels() -> None:
    prompt = _prompt(
        _signal(
            {
                "category": "politics",
                "nested": {"resolved_outcome": 1.0, "settlement": "yes"},
            }
        )
    )
    assert "resolved_outcome" not in prompt
    assert "settlement" not in prompt
    assert "category" in prompt


def test_llm_usage_cost_uses_provider_token_counts() -> None:
    forecaster = LLMForecaster(
        config=LLMSettings(
            enabled=True,
            provider="openai",
            api_key="test",
            input_price_usdc_per_million_tokens=2.0,
            output_price_usdc_per_million_tokens=8.0,
        )
    )
    cost = forecaster._usage_cost(_ProviderResponse("{}", 1_000, 500))
    assert float(cost) == pytest.approx(0.006)


def test_kelly_sizes_on_cost_adjusted_entry_price() -> None:
    sizer = KellySizer(risk=RiskSettings(max_position_per_market=1_000.0))
    portfolio = Portfolio(1_000.0, 1_000.0, 0.0, [])
    gross = sizer.size(prob=0.6, market_price=0.5, portfolio=portfolio)
    net = sizer.size(
        prob=0.6,
        market_price=0.5,
        cost_edge=0.05,
        portfolio=portfolio,
    )
    assert 0.0 < net < gross


def test_replay_resolution_label_accepts_closed_settlement_prices() -> None:
    assert _market_resolution_label(
        {"closed": True, "yes_price": 1.0, "no_price": 0.0}
    ) == 1.0
    assert _market_resolution_label(
        {"closed": True, "yes_price": 0.0, "no_price": 1.0}
    ) == 0.0


def test_replay_walk_book_rejects_non_finite_notional() -> None:
    with pytest.raises(InsufficientLiquidityError):
        _walk_book(
            orderbook={"asks": [{"price": 0.5, "size": 1.0}]},
            decision=_decision(),
            target_notional_usdc=float("nan"),
            slippage_bps=0.0,
            displayed_depth_fill_ratio=1.0,
            adverse_selection_bps=0.0,
        )
