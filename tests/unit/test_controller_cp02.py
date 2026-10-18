from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from datetime import UTC, datetime, timedelta

import pytest

from pms.config import ControllerSettings, PMSSettings, RiskSettings
from pms.controller.calibrators.netcal import NetcalCalibrator
from pms.controller.outcome_tokens import OutcomeTokens
from pms.controller.pipeline import (
    ControllerPipeline,
    _decision_cost_edges,
    _log_pipeline_funnel,
)
from pms.controller.router import Router
from pms.controller.sizers.kelly import KellySizer
from pms.core.enums import RunMode
from pms.core.models import BookLevel, BookSnapshot, MarketSignal, Portfolio, Position
from pms.metrics import (
    SELECTION_FUNNEL_CONTROLLER_EMITTED_TOTAL_METRIC,
    SELECTION_FUNNEL_FORECASTED_TOTAL_METRIC,
    SELECTION_FUNNEL_ROUTED_TOTAL_METRIC,
    get_metric,
)
from pms.strategies.projections import (
    ActiveStrategy,
    EvalSpec,
    FactorCompositionStep,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)


class StaticForecaster:
    def predict(self, signal: MarketSignal) -> tuple[float, float, str]:
        del signal
        return (0.67, 0.9, "factor-value edge")

    async def forecast(self, signal: MarketSignal) -> float:
        del signal
        return 0.67


class CountingForecaster(StaticForecaster):
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, signal: MarketSignal) -> tuple[float, float, str]:
        self.calls += 1
        return super().predict(signal)


class StaticSizer:
    def size(
        self,
        *,
        prob: float,
        market_price: float,
        portfolio: Portfolio,
        cost_edge: float = 0.0,
    ) -> float:
        del prob, market_price, portfolio, cost_edge
        return 1.0


def test_pipeline_funnel_log_updates_live_metrics() -> None:
    routed_before = get_metric(SELECTION_FUNNEL_ROUTED_TOTAL_METRIC) or 0.0
    forecasted_before = get_metric(SELECTION_FUNNEL_FORECASTED_TOTAL_METRIC) or 0.0
    emitted_before = get_metric(SELECTION_FUNNEL_CONTROLLER_EMITTED_TOTAL_METRIC) or 0.0

    _log_pipeline_funnel(_signal(), forecasted_count=2, emitted_count=1)

    assert (get_metric(SELECTION_FUNNEL_ROUTED_TOTAL_METRIC) or 0.0) == pytest.approx(
        routed_before + 1.0
    )
    assert (
        get_metric(SELECTION_FUNNEL_FORECASTED_TOTAL_METRIC) or 0.0
    ) == pytest.approx(forecasted_before + 2.0)
    assert (
        get_metric(SELECTION_FUNNEL_CONTROLLER_EMITTED_TOTAL_METRIC) or 0.0
    ) == pytest.approx(emitted_before + 1.0)


class BearishForecaster:
    def predict(self, signal: MarketSignal) -> tuple[float, float, str]:
        del signal
        return (0.47, -0.1, "bearish edge")

    async def forecast(self, signal: MarketSignal) -> float:
        del signal
        return 0.47


class VeryBearishForecaster:
    def predict(self, signal: MarketSignal) -> tuple[float, float, str]:
        del signal
        return (0.35, -0.2, "direct no edge")

    async def forecast(self, signal: MarketSignal) -> float:
        del signal
        return 0.35


class VeryBullishForecaster:
    def predict(self, signal: MarketSignal) -> tuple[float, float, str]:
        del signal
        return (0.80, 0.2, "direct yes edge")

    async def forecast(self, signal: MarketSignal) -> float:
        del signal
        return 0.80


class StaticOutcomeTokenResolver:
    async def resolve(
        self,
        *,
        market_id: str,
        signal_token_id: str | None,
    ) -> OutcomeTokens:
        del market_id, signal_token_id
        return OutcomeTokens(yes_token_id="token-yes", no_token_id="token-no")


class StaticDirectBookReader:
    def __init__(
        self,
        *,
        token_id: str,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        ts: datetime | None = None,
        fee_rate_bps: float | None = None,
    ) -> None:
        self.token_id = token_id
        self.fee_rate_bps = fee_rate_bps
        self.snapshot = BookSnapshot(
            id=101,
            market_id="market-cp02",
            token_id=token_id,
            ts=ts or datetime.now(tz=UTC),
            hash="direct-book",
            source="subscribe",
        )
        self.levels = [
            *(
                BookLevel(
                    snapshot_id=self.snapshot.id,
                    market_id="market-cp02",
                    side="BUY",
                    price=price,
                    size=size,
                )
                for price, size in bids
            ),
            *(
                BookLevel(
                    snapshot_id=self.snapshot.id,
                    market_id="market-cp02",
                    side="SELL",
                    price=price,
                    size=size,
                )
                for price, size in asks
            ),
        ]

    async def read_latest_snapshot(
        self,
        market_id: str,
        token_id: str,
    ) -> BookSnapshot | None:
        del market_id
        return self.snapshot if token_id == self.token_id else None

    async def read_levels_for_snapshot(self, snapshot_id: int) -> list[BookLevel]:
        return list(self.levels) if snapshot_id == self.snapshot.id else []

    async def read_fee_rate_bps(self, market_id: str, token_id: str) -> float | None:
        del market_id
        return self.fee_rate_bps if token_id == self.token_id else None


class MultiDirectBookReader:
    def __init__(
        self,
        books: dict[str, tuple[list[tuple[float, float]], list[tuple[float, float]]]],
    ) -> None:
        self.snapshots: dict[str, BookSnapshot] = {}
        self.levels: dict[int, list[BookLevel]] = {}
        for index, (token_id, (bids, asks)) in enumerate(books.items(), start=1):
            snapshot = BookSnapshot(
                id=200 + index,
                market_id="market-cp02",
                token_id=token_id,
                ts=datetime.now(tz=UTC),
                hash=f"direct-book-{token_id}",
                source="subscribe",
            )
            self.snapshots[token_id] = snapshot
            self.levels[snapshot.id] = [
                *(
                    BookLevel(
                        snapshot_id=snapshot.id,
                        market_id="market-cp02",
                        side="BUY",
                        price=price,
                        size=size,
                    )
                    for price, size in bids
                ),
                *(
                    BookLevel(
                        snapshot_id=snapshot.id,
                        market_id="market-cp02",
                        side="SELL",
                        price=price,
                        size=size,
                    )
                    for price, size in asks
                ),
            ]

    async def read_latest_snapshot(
        self,
        market_id: str,
        token_id: str,
    ) -> BookSnapshot | None:
        del market_id
        return self.snapshots.get(token_id)

    async def read_levels_for_snapshot(self, snapshot_id: int) -> list[BookLevel]:
        return list(self.levels.get(snapshot_id, ()))


class FailingFeeDirectBookReader(StaticDirectBookReader):
    async def read_fee_rate_bps(self, market_id: str, token_id: str) -> float | None:
        del market_id, token_id
        raise RuntimeError


class NullForecaster:
    """Forecaster that always returns None — used to exercise the
    no-forecaster-output branch of ``ControllerPipeline.on_signal``."""

    def predict(self, signal: MarketSignal) -> tuple[float, float, str] | None:
        del signal
        return None

    async def forecast(self, signal: MarketSignal) -> float:
        return signal.yes_price


def _signal(*, fetched_at: datetime | None = None) -> MarketSignal:
    return MarketSignal(
        market_id="market-cp02",
        token_id="token-cp02",
        venue="polymarket",
        title="Will CP02 emit opportunities?",
        yes_price=0.4,
        volume_24h=1_000.0,
        resolves_at=datetime(2026, 4, 30, tzinfo=UTC),
        orderbook={"bids": [], "asks": []},
        external_signal={"fair_value": 0.61, "confidence": 0.8, "label": "skip"},
        fetched_at=fetched_at or datetime(2026, 4, 19, tzinfo=UTC),
        market_status="open",
    )


def _position(
    *,
    market_id: str = "market-cp02",
    token_id: str = "token-cp02",
    locked_usdc: float = 0.0,
    risk_group_id: str | None = None,
) -> Position:
    return Position(
        market_id=market_id,
        token_id=token_id,
        venue="polymarket",
        side="BUY",
        shares_held=locked_usdc / 0.4 if locked_usdc > 0.0 else 0.0,
        avg_entry_price=0.4,
        unrealized_pnl=0.0,
        locked_usdc=locked_usdc,
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        risk_group_id=risk_group_id,
    )


def _portfolio(open_positions: list[Position] | None = None) -> Portfolio:
    locked_usdc = sum(position.locked_usdc for position in open_positions or [])
    return Portfolio(
        total_usdc=1_000.0,
        free_usdc=1_000.0 - locked_usdc,
        locked_usdc=locked_usdc,
        open_positions=[] if open_positions is None else open_positions,
    )


def _expected_kelly_notional() -> float:
    probability = Decimal("0.67")
    market_price = Decimal("0.4")
    payout_multiple = (Decimal("1.0") - market_price) / market_price
    kelly_fraction = (
        (probability * payout_multiple) - (Decimal("1.0") - probability)
    ) / payout_multiple
    scaled_fraction = kelly_fraction * Decimal("0.25")
    return float(Decimal("1000.0") * scaled_fraction)


def _best_ask_strategy() -> ActiveStrategy:
    return ActiveStrategy(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        config=StrategyConfig(
            strategy_id="alpha",
            factor_composition=(),
            metadata=(("price_reference", "best_ask"),),
        ),
        risk=RiskParams(
            max_position_notional_usdc=500.0,
            max_daily_drawdown_pct=2.5,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(metrics=("brier", "pnl")),
        forecaster=ForecasterSpec(forecasters=(("rules", ()),)),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=100.0,
        ),
    )


def _best_ask_imbalance_strategy() -> ActiveStrategy:
    return ActiveStrategy(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        config=StrategyConfig(
            strategy_id="alpha",
            factor_composition=(
                FactorCompositionStep(
                    factor_id="orderbook_imbalance",
                    role="rule_delta",
                    param="",
                    weight=0.25,
                    threshold=0.0,
                    required=True,
                ),
            ),
            metadata=(("price_reference", "best_ask"),),
        ),
        risk=RiskParams(
            max_position_notional_usdc=500.0,
            max_daily_drawdown_pct=2.5,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(metrics=("brier", "pnl")),
        forecaster=ForecasterSpec(forecasters=()),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=100.0,
        ),
    )


@pytest.mark.asyncio
async def test_controller_pipeline_on_signal_emits_opportunity_and_linked_decision() -> None:
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0)),
    )

    emission = await pipeline.on_signal(_signal(), portfolio=_portfolio())

    assert emission is not None
    opportunity, decision = emission
    assert opportunity.strategy_id == "alpha"
    assert opportunity.strategy_version_id == "alpha-v1"
    assert opportunity.side == "yes"
    assert opportunity.market_id == "market-cp02"
    assert opportunity.token_id == "token-cp02"
    assert opportunity.selected_factor_values == {"yes_price": 0.4}
    assert opportunity.rationale == "StaticForecaster:factor-value edge"
    assert decision.notional_usdc == pytest.approx(_expected_kelly_notional())
    assert opportunity.target_size_usdc == pytest.approx(decision.notional_usdc)
    assert opportunity.expiry == datetime(2026, 4, 30, tzinfo=UTC)
    assert opportunity.staleness_policy == "market_signal_freshness"
    assert decision.opportunity_id == opportunity.opportunity_id
    assert decision.model_id == "StaticForecaster"
    assert decision.strategy_id == "alpha"
    assert decision.strategy_version_id == "alpha-v1"
    assert decision.limit_price == pytest.approx(0.4)
    assert decision.stop_conditions


@pytest.mark.asyncio
async def test_backtest_pipeline_drops_zero_execution_price_before_decision_entity() -> None:
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=StaticSizer(),
        router=Router(ControllerSettings(min_volume=100.0), allow_extreme_prices=True),
        settings=PMSSettings(mode=RunMode.BACKTEST),
    )

    emission = await pipeline.on_signal(
        replace(_signal(), yes_price=0.0),
        portfolio=_portfolio(),
    )

    assert emission is None
    assert pipeline.last_diagnostic is not None
    assert pipeline.last_diagnostic.code == "decision_price_out_of_domain"


@pytest.mark.asyncio
async def test_controller_pipeline_decide_returns_notional_decision() -> None:
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0)),
    )

    decision = await pipeline.decide(_signal(), portfolio=_portfolio())

    assert decision is not None
    assert decision.notional_usdc == pytest.approx(_expected_kelly_notional())
    assert decision.limit_price == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_controller_pipeline_caps_size_to_remaining_market_capacity() -> None:
    risk = RiskSettings(max_position_per_market=5.0, min_order_usdc=1.0)
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=risk),
        router=Router(ControllerSettings(min_volume=100.0)),
        settings=PMSSettings(risk=risk),
    )

    emission = await pipeline.on_signal(
        _signal(),
        portfolio=_portfolio(open_positions=[_position(locked_usdc=3.5)]),
    )

    assert emission is not None
    opportunity, decision = emission
    assert decision.notional_usdc == pytest.approx(1.5)
    assert opportunity.target_size_usdc == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_controller_pipeline_skips_full_risk_group_before_forecasting() -> None:
    risk = RiskSettings(
        max_position_per_market=5.0,
        max_exposure_per_risk_group=1.0,
        min_order_usdc=1.0,
    )
    forecaster = CountingForecaster()
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[forecaster],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=risk),
        router=Router(ControllerSettings(min_volume=100.0)),
        settings=PMSSettings(risk=risk),
    )
    signal = replace(
        _signal(),
        external_signal={**_signal().external_signal, "risk_group_id": "event:cp02"},
    )

    emission = await pipeline.on_signal(
        signal,
        portfolio=_portfolio(
            open_positions=[
                _position(
                    market_id="other-market",
                    token_id="other-token",
                    locked_usdc=1.0,
                    risk_group_id="event:cp02",
                )
            ]
        ),
    )

    assert emission is None
    assert forecaster.calls == 0
    assert pipeline.last_diagnostic is not None
    assert pipeline.last_diagnostic.code == "risk_group_capacity_below_minimum"


@pytest.mark.asyncio
async def test_controller_pipeline_caps_size_to_remaining_risk_group_capacity() -> None:
    risk = RiskSettings(
        max_position_per_market=5.0,
        max_exposure_per_risk_group=2.0,
        min_order_usdc=1.0,
    )
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=risk),
        router=Router(ControllerSettings(min_volume=100.0)),
        settings=PMSSettings(risk=risk),
    )
    signal = replace(
        _signal(),
        external_signal={**_signal().external_signal, "risk_group_id": "event:cp02"},
    )

    emission = await pipeline.on_signal(
        signal,
        portfolio=_portfolio(
            open_positions=[
                _position(
                    market_id="other-market",
                    token_id="other-token",
                    locked_usdc=0.5,
                    risk_group_id="event:cp02",
                )
            ]
        ),
    )

    assert emission is not None
    opportunity, decision = emission
    assert decision.notional_usdc == pytest.approx(1.5)
    assert opportunity.target_size_usdc == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_controller_pipeline_caps_size_to_executable_book_depth() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=1_000.0)),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(max_spread_bps=1_000.0),
        ),
    )
    signal = replace(
        _signal(),
        orderbook={
            "bids": [{"price": 0.39, "size": 10.0}],
            "asks": [
                {"price": 0.4, "size": 3.0},
                {"price": 0.41, "size": 1_000.0},
            ],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is not None
    opportunity, decision = emission
    assert decision.notional_usdc == pytest.approx(1.2)
    assert opportunity.target_size_usdc == pytest.approx(1.2)


@pytest.mark.asyncio
async def test_controller_pipeline_skips_when_executable_depth_below_minimum() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=1_000.0)),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(max_spread_bps=1_000.0),
            risk=RiskSettings(min_order_usdc=1.0),
        ),
    )
    signal = replace(
        _signal(),
        orderbook={
            "bids": [{"price": 0.39, "size": 10.0}],
            "asks": [
                {"price": 0.4, "size": 2.0},
                {"price": 0.41, "size": 1_000.0},
            ],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "executable_depth_below_minimum"
    assert diagnostic.metadata["executable_depth_usdc"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_best_ask_strategy_rejects_stale_live_signal_orderbook() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(
            ControllerSettings(
                min_volume=100.0,
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            )
        ),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            ),
        ),
    )
    now = datetime.now(tz=UTC)
    signal = replace(
        _signal(fetched_at=now),
        resolves_at=now + timedelta(days=1),
        external_signal={
            **_signal().external_signal,
            "raw_event_type": "book",
            "book_received_at": (now - timedelta(seconds=5)).isoformat(),
        },
        orderbook={
            "bids": [{"price": 0.39, "size": 1_000.0}],
            "asks": [{"price": 0.40, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "router_gate:book_too_stale"
    assert diagnostic.metadata["gate_reason"] == "book_too_stale"
    assert diagnostic.metadata["book_age_ms"] > 1_000.0


@pytest.mark.asyncio
async def test_best_ask_strategy_refreshes_stale_signal_book_before_router_gate() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(
            ControllerSettings(
                min_volume=100.0,
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            )
        ),
        direct_book_reader=StaticDirectBookReader(
            token_id="token-cp02",
            bids=[(0.39, 1_000.0)],
            asks=[(0.40, 1_000.0)],
        ),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            ),
        ),
    )
    now = datetime.now(tz=UTC)
    signal = replace(
        _signal(fetched_at=now),
        resolves_at=now + timedelta(days=1),
        external_signal={
            **_signal().external_signal,
            "raw_event_type": "book",
            "book_received_at": (now - timedelta(seconds=5)).isoformat(),
        },
        orderbook={
            "bids": [{"price": 0.39, "size": 1_000.0}],
            "asks": [{"price": 0.40, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is not None
    assert pipeline.last_diagnostic is None


@pytest.mark.asyncio
async def test_best_ask_strategy_sorts_refreshed_direct_book_levels() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(
            ControllerSettings(
                min_volume=100.0,
                max_book_age_ms=1_000.0,
                max_spread_bps=10_000.0,
            )
        ),
        direct_book_reader=StaticDirectBookReader(
            token_id="token-cp02",
            bids=[(0.01, 10.0), (0.39, 1_000.0), (0.25, 20.0)],
            asks=[(0.99, 10.0), (0.40, 1_000.0), (0.75, 20.0)],
        ),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                max_book_age_ms=1_000.0,
                max_spread_bps=10_000.0,
            ),
        ),
    )
    now = datetime.now(tz=UTC)
    signal = replace(
        _signal(fetched_at=now),
        resolves_at=now + timedelta(days=1),
        external_signal={
            **_signal().external_signal,
            "raw_event_type": "book",
            "book_received_at": (now - timedelta(seconds=5)).isoformat(),
        },
        orderbook={
            "bids": [{"price": 0.01, "size": 10.0}],
            "asks": [{"price": 0.99, "size": 10.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is not None
    _, decision = emission
    assert decision.limit_price == pytest.approx(0.40)
    assert pipeline.last_execution_signal is not None
    assert pipeline.last_execution_signal.orderbook == {
        "bids": [
            {"price": 0.39, "size": 1_000.0},
            {"price": 0.25, "size": 20.0},
            {"price": 0.01, "size": 10.0},
        ],
        "asks": [
            {"price": 0.40, "size": 1_000.0},
            {"price": 0.75, "size": 20.0},
            {"price": 0.99, "size": 10.0},
        ],
    }


@pytest.mark.asyncio
async def test_best_ask_strategy_uses_direct_fee_rate_bps_for_costs() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(
            ControllerSettings(
                min_volume=100.0,
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            )
        ),
        direct_book_reader=StaticDirectBookReader(
            token_id="token-cp02",
            bids=[(0.39, 1_000.0)],
            asks=[(0.40, 1_000.0)],
            fee_rate_bps=300.0,
        ),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            ),
        ),
    )
    now = datetime.now(tz=UTC)
    signal = replace(
        _signal(fetched_at=now),
        resolves_at=now + timedelta(days=1),
        external_signal={
            **_signal().external_signal,
            "raw_event_type": "book",
            "book_received_at": (now - timedelta(seconds=5)).isoformat(),
        },
        orderbook={
            "bids": [{"price": 0.39, "size": 1_000.0}],
            "asks": [{"price": 0.40, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is not None
    opportunity, _ = emission
    assert pipeline.last_execution_signal is not None
    assert pipeline.last_execution_signal.external_signal["fee_rate_bps"] == 300.0
    assert opportunity.composition_trace["fee_rate"] == pytest.approx(0.03)
    assert opportunity.composition_trace["fee_edge"] == pytest.approx(0.018)


@pytest.mark.asyncio
async def test_direct_fee_rate_failure_logs_exception_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(
            ControllerSettings(
                min_volume=100.0,
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            )
        ),
        direct_book_reader=FailingFeeDirectBookReader(
            token_id="token-cp02",
            bids=[(0.39, 1_000.0)],
            asks=[(0.40, 1_000.0)],
        ),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            ),
        ),
    )
    now = datetime.now(tz=UTC)
    signal = replace(
        _signal(fetched_at=now),
        resolves_at=now + timedelta(days=1),
        external_signal={
            **_signal().external_signal,
            "raw_event_type": "book",
            "book_received_at": (now - timedelta(seconds=5)).isoformat(),
        },
        orderbook={
            "bids": [{"price": 0.39, "size": 1_000.0}],
            "asks": [{"price": 0.40, "size": 1_000.0}],
        },
    )

    with caplog.at_level("WARNING"):
        emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is not None
    opportunity, _ = emission
    assert opportunity.composition_trace["fee_rate"] == pytest.approx(0.07)
    assert any(
        "RuntimeError" in record.message and "(no message)" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_best_ask_strategy_accepts_freshly_received_live_orderbook_with_old_venue_timestamp() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(
            ControllerSettings(
                min_volume=100.0,
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            )
        ),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            ),
        ),
    )
    now = datetime.now(tz=UTC)
    signal = replace(
        _signal(fetched_at=now - timedelta(seconds=5)),
        resolves_at=now + timedelta(days=1),
        external_signal={
            **_signal().external_signal,
            "raw_event_type": "book",
            "book_received_at": now.isoformat(),
        },
        orderbook={
            "bids": [{"price": 0.39, "size": 1_000.0}],
            "asks": [{"price": 0.40, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is not None
    assert pipeline.last_diagnostic is None


@pytest.mark.asyncio
async def test_best_ask_strategy_rechecks_live_orderbook_age_before_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(
            ControllerSettings(
                min_volume=100.0,
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            )
        ),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            ),
        ),
    )
    age_samples_ms = [900.0, 1_100.0]

    def fake_decision_time_book_age_ms(
        signal: MarketSignal,
        *,
        allowed_clock_skew_ms: float,
        now: datetime | None = None,
    ) -> float:
        del signal, allowed_clock_skew_ms, now
        return age_samples_ms.pop(0) if age_samples_ms else 1_100.0

    monkeypatch.setattr(
        "pms.controller.pipeline._decision_time_book_age_ms",
        fake_decision_time_book_age_ms,
    )
    now = datetime.now(tz=UTC)
    signal = replace(
        _signal(fetched_at=now),
        resolves_at=now + timedelta(days=1),
        external_signal={
            **_signal().external_signal,
            "raw_event_type": "book",
            "book_received_at": now.isoformat(),
        },
        orderbook={
            "bids": [{"price": 0.39, "size": 1_000.0}],
            "asks": [{"price": 0.40, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "router_gate:book_too_stale"
    assert diagnostic.metadata["phase"] == "pre_emit"
    assert diagnostic.metadata["book_age_ms"] == pytest.approx(1_100.0)


@pytest.mark.asyncio
async def test_best_ask_strategy_refreshes_signal_book_before_pre_emit_staleness_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(
            ControllerSettings(
                min_volume=100.0,
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            )
        ),
        direct_book_reader=StaticDirectBookReader(
            token_id="token-cp02",
            bids=[(0.39, 1_000.0)],
            asks=[(0.40, 1_000.0)],
        ),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            ),
        ),
    )
    non_refreshed_age_samples_ms = [900.0, 1_100.0]

    def fake_decision_time_book_age_ms(
        signal: MarketSignal,
        *,
        allowed_clock_skew_ms: float,
        now: datetime | None = None,
    ) -> float:
        del allowed_clock_skew_ms, now
        if signal.external_signal.get("direct_outcome_book_source") == "subscribe":
            return 0.0
        return (
            non_refreshed_age_samples_ms.pop(0)
            if non_refreshed_age_samples_ms
            else 1_100.0
        )

    monkeypatch.setattr(
        "pms.controller.pipeline._decision_time_book_age_ms",
        fake_decision_time_book_age_ms,
    )
    now = datetime.now(tz=UTC)
    signal = replace(
        _signal(fetched_at=now),
        resolves_at=now + timedelta(days=1),
        external_signal={
            **_signal().external_signal,
            "raw_event_type": "book",
            "book_received_at": now.isoformat(),
        },
        orderbook={
            "bids": [{"price": 0.39, "size": 1_000.0}],
            "asks": [{"price": 0.40, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is not None
    _, decision = emission
    assert decision.limit_price == pytest.approx(0.40)
    assert pipeline.last_diagnostic is None


@pytest.mark.asyncio
async def test_best_ask_strategy_skips_synthetic_no_decision_without_direct_no_book() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[BearishForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0)),
        outcome_token_resolver=StaticOutcomeTokenResolver(),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(max_spread_bps=1_000.0),
        ),
    )
    signal = replace(
        _signal(),
        token_id="token-yes",
        yes_price=0.585,
        orderbook={
            "bids": [{"price": 0.58, "size": 1_000.0}],
            "asks": [{"price": 0.59, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "direct_outcome_orderbook_required"
    assert diagnostic.metadata["decision_outcome"] == "NO"
    assert diagnostic.metadata["signal_token_id"] == "token-yes"
    assert diagnostic.metadata["decision_token_id"] == "token-no"


@pytest.mark.asyncio
async def test_best_ask_strategy_reports_stale_direct_book_reason() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[BearishForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=1_000.0)),
        outcome_token_resolver=StaticOutcomeTokenResolver(),
        direct_book_reader=StaticDirectBookReader(
            token_id="token-no",
            bids=[(0.40, 1_000.0)],
            asks=[(0.41, 1_000.0)],
            ts=datetime.now(tz=UTC) - timedelta(seconds=5),
        ),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                max_book_age_ms=1_000.0,
                max_spread_bps=1_000.0,
            ),
        ),
    )
    signal = replace(
        _signal(fetched_at=datetime.now(tz=UTC)),
        resolves_at=datetime.now(tz=UTC) + timedelta(days=1),
        token_id="token-yes",
        yes_price=0.585,
        external_signal={
            **_signal().external_signal,
            "yes_token_id": "token-yes",
            "no_token_id": "token-no",
            "raw_event_type": "book",
            "book_received_at": datetime.now(tz=UTC).isoformat(),
        },
        orderbook={
            "bids": [{"price": 0.58, "size": 1_000.0}],
            "asks": [{"price": 0.59, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "direct_outcome_orderbook_required"
    assert diagnostic.metadata["direct_book_failure"] == "stale"
    assert diagnostic.metadata["direct_book_age_ms"] > 1_000.0


@pytest.mark.asyncio
async def test_best_ask_strategy_uses_cached_direct_no_book_for_bearish_yes_signal() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[BearishForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=1_000.0)),
        outcome_token_resolver=StaticOutcomeTokenResolver(),
        direct_book_reader=StaticDirectBookReader(
            token_id="token-no",
            bids=[(0.40, 1_000.0)],
            asks=[(0.41, 1_000.0)],
        ),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(max_spread_bps=1_000.0),
        ),
    )
    signal = replace(
        _signal(),
        token_id="token-yes",
        yes_price=0.585,
        external_signal={
            **_signal().external_signal,
            "yes_token_id": "token-yes",
            "no_token_id": "token-no",
        },
        orderbook={
            "bids": [{"price": 0.58, "size": 1_000.0}],
            "asks": [{"price": 0.59, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is not None
    opportunity, decision = emission
    assert opportunity.side == "no"
    assert decision.outcome == "NO"
    assert decision.token_id == "token-no"
    assert decision.limit_price == pytest.approx(0.41)
    assert decision.prob_estimate == pytest.approx(0.53)
    assert decision.expected_edge == pytest.approx(0.12)
    assert decision.spread_bps_at_decision == pytest.approx(247)


@pytest.mark.asyncio
async def test_best_ask_strategy_uses_direct_no_orderbook_price_for_no_signal() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[VeryBearishForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=1_000.0)),
        outcome_token_resolver=StaticOutcomeTokenResolver(),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(max_spread_bps=1_000.0),
        ),
    )
    signal = replace(
        _signal(),
        token_id="token-no",
        yes_price=0.585,
        external_signal={
            **_signal().external_signal,
            "yes_token_id": "token-yes",
            "no_token_id": "token-no",
            "signal_token_outcome": "NO",
        },
        orderbook={
            "bids": [{"price": 0.58, "size": 1_000.0}],
            "asks": [{"price": 0.59, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is not None
    opportunity, decision = emission
    assert opportunity.side == "no"
    assert decision.outcome == "NO"
    assert decision.token_id == "token-no"
    assert decision.limit_price == pytest.approx(0.59)
    assert decision.prob_estimate == pytest.approx(0.65)
    assert decision.expected_edge == pytest.approx(0.06)


@pytest.mark.asyncio
async def test_no_token_orderbook_imbalance_is_projected_to_canonical_yes_probability() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_imbalance_strategy(),
        forecasters=[],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=1_000.0)),
        outcome_token_resolver=StaticOutcomeTokenResolver(),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(max_spread_bps=1_000.0),
        ),
    )
    signal = replace(
        _signal(),
        token_id="token-no",
        yes_price=0.585,
        external_signal={
            **_signal().external_signal,
            "yes_token_id": "token-yes",
            "no_token_id": "token-no",
            "signal_token_outcome": "NO",
        },
        orderbook={
            "bids": [{"price": 0.58, "size": 900.0}],
            "asks": [{"price": 0.59, "size": 100.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is not None
    opportunity, decision = emission
    assert opportunity.side == "no"
    assert decision.outcome == "NO"
    assert decision.token_id == "token-no"
    assert decision.limit_price == pytest.approx(0.59)
    assert decision.prob_estimate == pytest.approx(0.785)
    assert decision.expected_edge == pytest.approx(0.195)


@pytest.mark.asyncio
async def test_no_token_signal_with_positive_yes_edge_does_not_force_no_trade() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[VeryBullishForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=1_000.0)),
        outcome_token_resolver=StaticOutcomeTokenResolver(),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(max_spread_bps=1_000.0),
        ),
    )
    signal = replace(
        _signal(),
        token_id="token-no",
        yes_price=0.30,
        external_signal={
            **_signal().external_signal,
            "yes_token_id": "token-yes",
            "no_token_id": "token-no",
            "signal_token_outcome": "NO",
        },
        orderbook={
            "bids": [{"price": 0.29, "size": 1_000.0}],
            "asks": [{"price": 0.30, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "direct_outcome_orderbook_required"
    assert diagnostic.metadata["decision_outcome"] == "YES"
    assert diagnostic.metadata["decision_token_id"] == "token-yes"


@pytest.mark.asyncio
async def test_no_token_signal_with_positive_yes_edge_requires_yes_token_metadata() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[VeryBullishForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=1_000.0)),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(max_spread_bps=1_000.0),
        ),
    )
    signal = replace(
        _signal(),
        token_id="token-no",
        yes_price=0.30,
        external_signal={
            **_signal().external_signal,
            "no_token_id": "token-no",
            "signal_token_outcome": "NO",
        },
        orderbook={
            "bids": [{"price": 0.29, "size": 1_000.0}],
            "asks": [{"price": 0.30, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "missing_yes_token"
    assert diagnostic.metadata["signal_token_id"] == "token-no"
    assert diagnostic.metadata["no_token_id"] == "token-no"
    assert diagnostic.metadata["outcome"] == "YES"


@pytest.mark.asyncio
async def test_best_ask_strategy_recomputes_spread_after_direct_book_switch() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[VeryBearishForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=1_000.0)),
        outcome_token_resolver=StaticOutcomeTokenResolver(),
        direct_book_reader=MultiDirectBookReader(
            {
                "token-no": (
                    [(0.20, 1_000.0)],
                    [(0.30, 1_000.0)],
                ),
            }
        ),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(max_spread_bps=1_000.0),
        ),
    )
    signal = replace(
        _signal(),
        token_id="token-yes",
        yes_price=0.80,
        external_signal={
            **_signal().external_signal,
            "yes_token_id": "token-yes",
            "no_token_id": "token-no",
            "spread_bps": 50,
        },
        orderbook={
            "bids": [{"price": 0.795, "size": 1_000.0}],
            "asks": [{"price": 0.800, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "router_gate:spread_too_wide"
    assert diagnostic.metadata["spread_bps_at_decision"] == 4000
    assert diagnostic.metadata["decision_token_id"] == "token-no"


@pytest.mark.asyncio
async def test_best_ask_strategy_refreshes_direct_book_token_price_evidence() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[BearishForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=1_000.0)),
        outcome_token_resolver=StaticOutcomeTokenResolver(),
        direct_book_reader=StaticDirectBookReader(
            token_id="token-no",
            bids=[(0.40, 1_000.0)],
            asks=[(0.41, 1_000.0)],
        ),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(max_spread_bps=1_000.0),
        ),
    )
    signal = replace(
        _signal(),
        token_id="token-yes",
        yes_price=0.585,
        external_signal={
            **_signal().external_signal,
            "yes_token_id": "token-yes",
            "no_token_id": "token-no",
        },
        orderbook={
            "bids": [{"price": 0.58, "size": 1_000.0}],
            "asks": [{"price": 0.59, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is not None
    _, decision = emission
    assert decision.outcome == "NO"
    assert decision.token_id == "token-no"
    assert decision.limit_price == pytest.approx(0.41)
    assert pipeline.last_execution_signal is not None
    assert pipeline.last_execution_signal.token_id == "token-no"
    assert pipeline.last_execution_signal.yes_price == pytest.approx(0.41)
    assert pipeline.last_execution_signal.external_signal["direct_token_price"] == pytest.approx(
        0.41
    )
    assert pipeline.last_execution_signal.external_signal["best_ask"] == pytest.approx(0.41)


@pytest.mark.asyncio
async def test_controller_pipeline_skips_when_remaining_market_capacity_below_minimum() -> None:
    risk = RiskSettings(max_position_per_market=5.0, min_order_usdc=1.0)
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=risk),
        router=Router(ControllerSettings(min_volume=100.0)),
        settings=PMSSettings(risk=risk),
    )

    emission = await pipeline.on_signal(
        _signal(),
        portfolio=_portfolio(open_positions=[_position(locked_usdc=4.25)]),
    )

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "market_position_capacity_below_minimum"
    assert diagnostic.metadata["market_exposure_usdc"] == pytest.approx(4.25)
    assert diagnostic.metadata["remaining_market_capacity_usdc"] == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_controller_pipeline_suppresses_duplicate_paper_decisions_within_cooldown() -> None:
    first_ts = datetime(2026, 4, 19, tzinfo=UTC)
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0)),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                min_volume=100.0,
                decision_cooldown_s=60.0,
            ),
        ),
    )

    first = await pipeline.on_signal(_signal(fetched_at=first_ts), portfolio=_portfolio())
    duplicate = await pipeline.on_signal(
        _signal(fetched_at=first_ts + timedelta(seconds=30)),
        portfolio=_portfolio(),
    )
    after_cooldown = await pipeline.on_signal(
        _signal(fetched_at=first_ts + timedelta(seconds=61)),
        portfolio=_portfolio(),
    )

    assert first is not None
    assert duplicate is None
    assert after_cooldown is not None


@pytest.mark.asyncio
async def test_on_signal_emits_diagnostic_when_router_gate_rejects_signal() -> None:
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=100.0)),
    )

    emission = await pipeline.on_signal(
        replace(_signal(), external_signal={"spread_bps": 250.0}),
        portfolio=_portfolio(),
    )

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None, "router-gate rejections must surface as a diagnostic"
    assert diagnostic.code == "router_gate:spread_too_wide"
    assert diagnostic.market_id == "market-cp02"
    assert diagnostic.strategy_id == "alpha"
    assert diagnostic.strategy_version_id == "alpha-v1"
    assert diagnostic.metadata.get("gate_reason") == "spread_too_wide"


@pytest.mark.asyncio
async def test_on_signal_emits_diagnostic_when_signal_lacks_token_id() -> None:
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0)),
    )

    emission = await pipeline.on_signal(
        replace(_signal(), token_id=None),
        portfolio=_portfolio(),
    )

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None, "missing-token-id rejections must surface as a diagnostic"
    assert diagnostic.code == "missing_token_id"
    assert diagnostic.market_id == "market-cp02"
    assert diagnostic.token_id is None


@pytest.mark.asyncio
async def test_on_signal_emits_diagnostic_when_no_forecaster_output_and_no_factor_composition() -> None:
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[NullForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0)),
    )

    emission = await pipeline.on_signal(_signal(), portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None, (
        "no-forecaster-output rejections must surface as a diagnostic so operators "
        "can see that signals reached the pipeline but no forecaster produced a probability"
    )
    assert diagnostic.code == "no_forecaster_output"
    assert diagnostic.market_id == "market-cp02"
    assert diagnostic.token_id == "token-cp02"


class _EqualToMarketForecaster:
    """Returns exactly the market YES price so the resulting edge is zero,
    exercising the decision_edge <= 0 silent-return branch."""

    def predict(self, signal: MarketSignal) -> tuple[float, float, str]:
        del signal
        return (0.4, 0.9, "no-edge")

    async def forecast(self, signal: MarketSignal) -> float:
        return 0.4


class _SmallGrossEdgeForecaster:
    """Returns a positive gross edge that is erased by configured costs."""

    def predict(self, signal: MarketSignal) -> tuple[float, float, str]:
        del signal
        return (0.43, 0.9, "small-gross-edge")

    async def forecast(self, signal: MarketSignal) -> float:
        del signal
        return 0.43


class _ModestEdgeForecaster:
    """Against a 0.403 entry price the 0.047 gross edge clears fee+slippage
    (0.043805) but not a second spread charge (+0.006045 at 150 bps)."""

    def predict(self, signal: MarketSignal) -> tuple[float, float, str]:
        del signal
        return (0.45, 0.9, "modest edge")

    async def forecast(self, signal: MarketSignal) -> float:
        del signal
        return 0.45


@pytest.mark.asyncio
async def test_on_signal_emits_diagnostic_when_decision_edge_not_positive() -> None:
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[_EqualToMarketForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0)),
    )

    # _signal() has yes_price=0.4; a forecast of 0.4 yields zero edge.
    emission = await pipeline.on_signal(_signal(), portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None, (
        "a zero/negative-edge drop must surface as a diagnostic so operators can "
        "distinguish 'no opportunity' from 'controller idle'"
    )
    assert diagnostic.code == "decision_edge_not_positive"
    assert diagnostic.severity == "info"
    assert diagnostic.market_id == "market-cp02"
    assert diagnostic.metadata.get("decision_edge") == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_on_signal_rejects_when_configured_costs_erase_gross_edge() -> None:
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[_SmallGrossEdgeForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0)),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                min_volume=100.0,
                max_slippage_bps=50,
            ),
        ),
    )

    emission = await pipeline.on_signal(_signal(), portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "decision_net_edge_not_positive"
    assert diagnostic.metadata["gross_edge"] == pytest.approx(0.03)
    assert diagnostic.metadata["fee_edge"] == pytest.approx(0.042)
    assert diagnostic.metadata["slippage_edge"] == pytest.approx(0.002)
    assert diagnostic.metadata["net_edge_after_costs"] < 0.0


def test_decision_cost_edges_convert_bps_to_price_space() -> None:
    costs = _decision_cost_edges(
        decision_price=0.40,
        spread_bps=500,
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(max_slippage_bps=50),
        ),
    )

    assert costs.slippage_edge == pytest.approx(0.002)
    assert costs.spread_edge == pytest.approx(0.02)


def test_decision_cost_edges_suppress_spread_when_already_in_price() -> None:
    costs = _decision_cost_edges(
        decision_price=0.40,
        spread_bps=500,
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(max_slippage_bps=50),
        ),
        spread_already_in_price=True,
    )

    assert costs.spread_edge == 0.0
    assert costs.slippage_edge == pytest.approx(0.002)
    assert costs.total_edge == pytest.approx(costs.fee_edge + 0.002)


@pytest.mark.asyncio
async def test_best_ask_decision_does_not_double_charge_spread() -> None:
    """best_ask frame: decision_price IS the executable ask, so the gross
    edge p - ask is already net of crossing the spread. Charging spread_bps
    again double-counts it and rejects a profitable decision."""
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[_ModestEdgeForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=1_000.0)),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                min_volume=100.0,
                max_spread_bps=1_000.0,
                max_slippage_bps=50,
            ),
        ),
    )
    signal = replace(
        _signal(),
        orderbook={
            "bids": [{"price": "0.397", "size": "100"}],
            "asks": [{"price": "0.403", "size": "100"}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is not None
    opportunity, decision = emission
    assert decision.limit_price == pytest.approx(0.403)
    trace = opportunity.composition_trace
    assert trace["spread_bps_at_decision"] == 150
    assert trace["spread_edge"] == 0.0
    assert trace["spread_already_in_price"] is True
    # net edge = gross - fee - slippage only; the spread is inside the gross.
    assert trace["net_edge_after_costs"] == pytest.approx(
        (0.45 - 0.403) - 0.07 * (1.0 - 0.403) - 0.005 * 0.403
    )


@pytest.mark.asyncio
async def test_yes_price_decision_still_charges_spread_on_identical_book() -> None:
    """yes_price frame control: with no price_reference metadata the decision
    price is the mid/last-trade reference, the executable ask is unknown, and
    the full spread stays a real worst-case entry cost."""
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[_ModestEdgeForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=1_000.0)),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                min_volume=100.0,
                max_spread_bps=1_000.0,
                max_slippage_bps=50,
            ),
        ),
    )
    signal = replace(
        _signal(),
        yes_price=0.403,
        orderbook={
            "bids": [{"price": "0.397", "size": "100"}],
            "asks": [{"price": "0.403", "size": "100"}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "decision_net_edge_not_positive"
    assert diagnostic.metadata["spread_edge"] == pytest.approx(0.015 * 0.403)
    assert diagnostic.metadata["spread_already_in_price"] is False
    assert diagnostic.metadata["net_edge_after_costs"] < 0.0


@pytest.mark.asyncio
async def test_no_side_direct_book_decision_does_not_charge_spread() -> None:
    """NO-side best_ask frame: the direct outcome book provides the
    executable NO ask, so the spread is already inside p_no - ask_no."""
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[BearishForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=1_000.0)),
        outcome_token_resolver=StaticOutcomeTokenResolver(),
        direct_book_reader=StaticDirectBookReader(
            token_id="token-no",
            bids=[(0.40, 1_000.0)],
            asks=[(0.41, 1_000.0)],
        ),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(max_spread_bps=1_000.0),
        ),
    )
    signal = replace(
        _signal(),
        token_id="token-yes",
        yes_price=0.585,
        external_signal={
            **_signal().external_signal,
            "yes_token_id": "token-yes",
            "no_token_id": "token-no",
        },
        orderbook={
            "bids": [{"price": 0.58, "size": 1_000.0}],
            "asks": [{"price": 0.59, "size": 1_000.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is not None
    opportunity, decision = emission
    assert decision.outcome == "NO"
    assert decision.limit_price == pytest.approx(0.41)
    trace = opportunity.composition_trace
    assert trace["spread_bps_at_decision"] == 247
    assert trace["spread_edge"] == 0.0
    assert trace["spread_already_in_price"] is True


@pytest.mark.asyncio
async def test_best_ask_strategy_without_ask_still_charges_external_spread() -> None:
    """Fallback hole: a best_ask strategy whose signal has no executable ask
    anywhere falls back to yes_price pricing, so external spread_bps remains
    a real entry cost (the price-equality half of the suppression condition)."""
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=1_000.0)),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                min_volume=100.0,
                max_spread_bps=1_000.0,
                max_slippage_bps=50,
            ),
        ),
    )
    signal = replace(
        _signal(),
        external_signal={**_signal().external_signal, "spread_bps": 50},
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is not None
    opportunity, decision = emission
    assert decision.limit_price == pytest.approx(0.4)
    trace = opportunity.composition_trace
    assert trace["spread_bps_at_decision"] == 50
    assert trace["spread_edge"] == pytest.approx(0.005 * 0.4)
    assert trace["spread_already_in_price"] is False


@pytest.mark.asyncio
async def test_best_ask_strategy_rejects_orderbook_spread_above_configured_limit() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=100.0)),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                min_volume=100.0,
                max_spread_bps=100.0,
                max_slippage_bps=50,
            ),
        ),
    )
    signal = replace(
        _signal(),
        orderbook={
            "bids": [{"price": "0.397", "size": "100"}],
            "asks": [{"price": "0.403", "size": "100"}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "router_gate:spread_too_wide"
    assert diagnostic.metadata["spread_bps_at_decision"] == 150
    assert diagnostic.metadata["max_spread_bps"] == 100.0


@pytest.mark.asyncio
async def test_best_ask_strategy_uses_orderbook_spread_over_stale_external_quote() -> None:
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0, max_spread_bps=100.0)),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(
                min_volume=100.0,
                max_spread_bps=100.0,
                max_slippage_bps=50,
            ),
        ),
    )
    signal = replace(
        _signal(),
        external_signal={
            **_signal().external_signal,
            "best_bid": 0.397,
            "best_ask": 0.400,
        },
        orderbook={
            "bids": [{"price": 0.30, "size": 100.0}],
            "asks": [{"price": 0.40, "size": 100.0}],
        },
    )

    emission = await pipeline.on_signal(signal, portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "router_gate:spread_too_wide"
    assert diagnostic.metadata["spread_bps_at_decision"] == 2857
    assert diagnostic.metadata["max_spread_bps"] == 100.0


@pytest.mark.asyncio
async def test_on_signal_emits_diagnostic_when_order_size_below_minimum() -> None:
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0)),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(min_volume=100.0),
            # Force the computed Kelly size (~$137) below the floor.
            risk=RiskSettings(min_order_usdc=500.0),
        ),
    )

    emission = await pipeline.on_signal(_signal(), portfolio=_portfolio())

    assert emission is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None, (
        "a sub-minimum order-size drop must surface as a diagnostic"
    )
    assert diagnostic.code == "order_size_below_minimum"
    assert diagnostic.severity == "info"
    assert diagnostic.metadata.get("min_order_usdc") == pytest.approx(500.0)


@pytest.mark.asyncio
async def test_on_signal_emits_diagnostic_when_within_decision_cooldown() -> None:
    first_ts = datetime(2026, 4, 19, tzinfo=UTC)
    pipeline = ControllerPipeline(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        forecasters=[StaticForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(min_volume=100.0)),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(min_volume=100.0, decision_cooldown_s=60.0),
        ),
    )

    first = await pipeline.on_signal(_signal(fetched_at=first_ts), portfolio=_portfolio())
    duplicate = await pipeline.on_signal(
        _signal(fetched_at=first_ts + timedelta(seconds=30)),
        portfolio=_portfolio(),
    )

    assert first is not None
    assert duplicate is None
    diagnostic = pipeline.last_diagnostic
    assert diagnostic is not None, (
        "a cooldown-suppressed duplicate must surface as a diagnostic so the "
        "suppression is observable, not silent"
    )
    assert diagnostic.code == "within_decision_cooldown"
    assert diagnostic.severity == "info"
