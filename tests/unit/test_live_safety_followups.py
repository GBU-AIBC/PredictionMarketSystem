from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Literal, cast

import asyncpg
import pytest
import httpx
from pydantic import SecretStr

from pms.api.app import create_app
from pms.actuator.adapters.polymarket import (
    FileFirstLiveOrderGate,
    LiveOrderPreview,
    LivePreSubmitQuote,
    LiveVenueBook,
    PolymarketActuator,
    PolymarketBookQuoteProvider,
    PolymarketDirectQuoteProvider,
    PolymarketOrderRequest,
    PolymarketOrderResult,
    PolymarketRoutingQuoteProvider,
    PolymarketSDKClient,
    PolymarketSubmissionUnknownError,
    PolymarketVenueAccountReconciler,
)
from pms.config import (
    ControllerSettings,
    DiscordSettings,
    PMSSettings,
    PolymarketSettings,
    RiskSettings,
    validate_live_mode_ready,
)
from pms.controller.calibrators.netcal import NetcalCalibrator
from pms.controller.factor_snapshot import FactorSnapshot
from pms.controller.pipeline import ControllerPipeline
from pms.controller.router import Router
from pms.core.enums import MarketStatus, OrderStatus, RunMode, Side, TimeInForce
from pms.core.models import (
    BookLevel,
    BookSnapshot,
    LiveTradingDisabledError,
    Market,
    MarketSignal,
    OrderState,
    Portfolio,
    Position,
    TradeDecision,
    VenueCredentials,
)
from pms.runner import (
    ReconciliationReport,
    Runner,
    VenueAccountSnapshot,
)
from pms.storage.decision_store import DECISION_STATUSES, validate_decision_status_transition
from pms.storage.live_reconciliation import (
    LiveOrderReconciliationStore,
    SubmissionUnknownReconciliationStore,
    SubmissionUnknownResolutionStatus,
)
from pms.live_cli import build_parser
from pms.strategies.projections import (
    ActiveStrategy,
    EvalSpec,
    FactorCompositionStep,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)
from tests.support.live_paths import (
    make_live_preflight_artifact_path,
    make_live_report_paths,
    make_private_live_paths,
)


AUTH_HEADERS = {"Authorization": "Bearer live-api-token"}


class ConstantForecaster:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict(self, signal: MarketSignal) -> tuple[float, float, str]:
        del signal
        return self.probability, 0.0, "constant"

    async def forecast(self, signal: MarketSignal) -> float:
        del signal
        return self.probability


class FixedSizer:
    def size(self, *, prob: float, market_price: float, portfolio: Portfolio) -> float:
        del prob, market_price, portfolio
        return 10.0


@dataclass(frozen=True)
class SnapshotReader:
    snapshot_value: FactorSnapshot

    async def snapshot(
        self,
        *,
        market_id: str,
        as_of: datetime,
        required: Sequence[FactorCompositionStep],
        strategy_id: str,
        strategy_version_id: str,
    ) -> FactorSnapshot:
        del market_id, as_of, required, strategy_id, strategy_version_id
        return self.snapshot_value


def _strategy(
    factor_id: str = "metaculus_prior",
) -> ActiveStrategy:
    return ActiveStrategy(
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        config=StrategyConfig(
            strategy_id="alpha",
            factor_composition=(
                FactorCompositionStep(
                    factor_id=factor_id,
                    role="weighted",
                    param="",
                    weight=1.0,
                    threshold=None,
                    freshness_sla_s=300.0,
                ),
            ),
            metadata=(),
        ),
        risk=RiskParams(
            max_position_notional_usdc=1000.0,
            max_daily_drawdown_pct=2.5,
            min_order_size_usdc=1.0,
        ),
        eval_spec=EvalSpec(metrics=("brier",)),
        forecaster=ForecasterSpec(forecasters=(("rules", ()),)),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=7,
            volume_min_usdc=0.0,
        ),
    )


def _signal(
    *,
    external_signal: dict[str, Any] | None = None,
    fetched_at: datetime | None = None,
) -> MarketSignal:
    return MarketSignal(
        market_id="m-live-safety",
        token_id="t-yes",
        venue="polymarket",
        title="Will live safety pass?",
        yes_price=0.4,
        volume_24h=10_000.0,
        resolves_at=datetime(2026, 5, 1, tzinfo=UTC),
        orderbook={
            "bids": [{"price": 0.39, "size": 100.0}],
            "asks": [{"price": 0.41, "size": 100.0}],
        },
        external_signal=external_signal or {},
        fetched_at=fetched_at or datetime(2026, 4, 27, tzinfo=UTC),
        market_status=MarketStatus.OPEN.value,
    )


def _portfolio() -> Portfolio:
    return Portfolio(
        total_usdc=1000.0,
        free_usdc=1000.0,
        locked_usdc=0.0,
        open_positions=[],
    )


def _live_settings(**overrides: object) -> PMSSettings:
    attested_at = datetime.now(tz=UTC)
    approval_path, audit_path = make_private_live_paths(prefix="pms-live-safety-")
    paper_report_path, rehearsal_report_path = make_live_report_paths(
        prefix="pms-live-safety-reports-"
    )
    values: dict[str, Any] = {
        "mode": RunMode.LIVE,
        "secret_source": "fly",
        "live_trading_enabled": True,
        "api_token": "live-api-token",
        "auto_migrate_default_v2": False,
        "live_emergency_audit_path": str(
            Path(approval_path).parent / "live-emergency-audit.jsonl"
        ),
        "live_first_order_audit_path": audit_path,
        "live_preflight_artifact_path": str(
            Path(approval_path).parent / "credentialed-preflight.json"
        ),
        "live_exit_criteria_ratified_by": "operator",
        "live_exit_criteria_ratified_at": attested_at,
        "live_compliance_reviewed_by": "counsel",
        "live_compliance_reviewed_at": attested_at,
        "live_compliance_jurisdiction": "US-operator-approved",
        "live_paper_soak_report_path": paper_report_path,
        "live_operator_rehearsal_report_path": rehearsal_report_path,
        "controller": ControllerSettings(
            time_in_force="IOC",
            min_volume=0.0,
            quote_source="dual",
        ),
        "discord": DiscordSettings(
            webhook_url=SecretStr("https://discord.example/webhooks/safety/unit"),
            alert_dir=str(Path(approval_path).parent / "discord-alerts"),
        ),
        "risk": RiskSettings(
            max_position_per_market=1000.0,
            max_total_exposure=10_000.0,
            max_drawdown_pct=20.0,
            max_daily_loss_usdc=20.0,
            max_open_positions=5,
            max_exposure_per_risk_group=5_000.0,
            max_quantity_shares=500.0,
            min_order_usdc=1.0,
        ),
        "polymarket": PolymarketSettings(
            private_key="private-key",
            api_key="api-key",
            api_secret="api-secret",
            api_passphrase="passphrase",
            signature_type=1,
            funder_address="0x1111111111111111111111111111111111111111",
            operator_approval_mode="every_order",
            first_live_order_approval_path=approval_path,
        ),
    }
    values.update(overrides)
    return PMSSettings(**values)


def test_live_mode_ready_requires_account_reconciliation_gate() -> None:
    settings = _live_settings(live_account_reconciliation_required=False)

    with pytest.raises(LiveTradingDisabledError, match="account reconciliation"):
        validate_live_mode_ready(settings)


def _decision(
    *,
    outcome: Literal["YES", "NO"] = "YES",
    time_in_force: TimeInForce = TimeInForce.IOC,
) -> TradeDecision:
    return TradeDecision(
        decision_id=f"d-{outcome.lower()}",
        market_id="m-live-safety",
        token_id="t-yes" if outcome == "YES" else "t-no",
        venue="polymarket",
        side=Side.BUY.value,
        notional_usdc=10.0,
        order_type="limit",
        max_slippage_bps=50,
        stop_conditions=["unit-test"],
        prob_estimate=0.7,
        expected_edge=0.3,
        time_in_force=time_in_force,
        opportunity_id=f"op-{outcome.lower()}",
        strategy_id="alpha",
        strategy_version_id="alpha-v1",
        limit_price=0.4,
        action=Side.BUY.value,
        outcome=outcome,
        intent_key="intent-live-safety",
    )


@dataclass
class FakeBookStore:
    market: Market | None
    snapshot: BookSnapshot | None
    levels: list[BookLevel]

    async def read_market(self, market_id: str) -> Market | None:
        del market_id
        return self.market

    async def read_latest_snapshot(
        self,
        market_id: str,
        token_id: str,
    ) -> BookSnapshot | None:
        del market_id, token_id
        return self.snapshot

    async def read_levels_for_snapshot(self, snapshot_id: int) -> list[BookLevel]:
        del snapshot_id
        return self.levels


@dataclass
class FakeVenueBookClient:
    book: LiveVenueBook
    calls: int = 0

    async def read_order_book(
        self,
        order: PolymarketOrderRequest,
        credentials: object,
    ) -> LiveVenueBook:
        del order, credentials
        self.calls += 1
        return self.book


@pytest.mark.asyncio
async def test_polymarket_book_quote_provider_uses_fresh_book_depth() -> None:
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    provider = PolymarketBookQuoteProvider(
        store=FakeBookStore(
            market=Market(
                condition_id="m-live-safety",
                slug="live-safety",
                question="Will live safety pass?",
                venue="polymarket",
                resolves_at=now + timedelta(days=1),
                created_at=now - timedelta(days=1),
                last_seen_at=now,
            ),
            snapshot=BookSnapshot(
                id=11,
                market_id="m-live-safety",
                token_id="t-yes",
                ts=now - timedelta(milliseconds=250),
                hash="book-hash",
                source="subscribe",
            ),
            levels=[
                BookLevel(
                    snapshot_id=11,
                    market_id="m-live-safety",
                    side="BUY",
                    price=0.39,
                    size=20.0,
                ),
                BookLevel(
                    snapshot_id=11,
                    market_id="m-live-safety",
                    side="SELL",
                    price=0.40,
                    size=10.0,
                ),
                BookLevel(
                    snapshot_id=11,
                    market_id="m-live-safety",
                    side="SELL",
                    price=0.41,
                    size=100.0,
                ),
            ],
        ),
        clock=lambda: now,
    )

    quote = await provider.quote(
        PolymarketOrderRequest(
            market_id="m-live-safety",
            token_id="t-yes",
            side="BUY",
            price=0.40,
            size=25.0,
            notional_usdc=10.0,
            estimated_quantity=25.0,
            order_type="limit",
            time_in_force="IOC",
            max_slippage_bps=50,
        ),
        _live_settings().polymarket.credentials(),
    )

    assert quote.market_status == "open"
    assert quote.book_age_ms == pytest.approx(250.0)
    assert quote.best_executable_price == pytest.approx(0.40)
    assert quote.executable_notional_usdc == pytest.approx(4.0)
    assert quote.spread_bps > 0.0
    assert quote.quote_hash == "book-hash"


@pytest.mark.asyncio
async def test_polymarket_direct_quote_provider_uses_venue_book() -> None:
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    client = FakeVenueBookClient(
        LiveVenueBook(
            market_id="m-live-safety",
            token_id="t-yes",
            bids=(
                BookLevel(
                    snapshot_id=0,
                    market_id="m-live-safety",
                    side=Side.BUY.value,
                    price=0.39,
                    size=100.0,
                ),
            ),
            asks=(
                BookLevel(
                    snapshot_id=0,
                    market_id="m-live-safety",
                    side=Side.SELL.value,
                    price=0.40,
                    size=40.0,
                ),
            ),
            book_ts=now - timedelta(milliseconds=100),
            quote_hash="venue-book",
            market_status="open",
        )
    )
    provider = PolymarketDirectQuoteProvider(book_client=client, clock=lambda: now)

    quote = await provider.quote(
        PolymarketOrderRequest(
            market_id="m-live-safety",
            token_id="t-yes",
            side="BUY",
            price=0.40,
            size=25.0,
            notional_usdc=10.0,
            estimated_quantity=25.0,
            order_type="limit",
            time_in_force="IOC",
            max_slippage_bps=50,
        ),
        _live_settings().polymarket.credentials(),
    )

    assert client.calls == 1
    assert quote.source == "venue_direct"
    assert quote.quote_hash == "venue-book"
    assert quote.book_age_ms == pytest.approx(100.0)
    assert quote.executable_notional_usdc == pytest.approx(16.0)


@pytest.mark.asyncio
async def test_polymarket_sdk_direct_quote_honors_venue_accepting_orders_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)

    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def get_order_book(self, token_id: str) -> dict[str, object]:
            assert token_id == "t-yes"
            return {
                "timestamp": now.isoformat(),
                "accepting_orders": False,
                "bids": [{"price": 0.39, "size": 100.0}],
                "asks": [{"price": 0.40, "size": 100.0}],
            }

    monkeypatch.setitem(
        sys.modules,
        "py_clob_client_v2",
        SimpleNamespace(ApiCreds=FakeApiCreds, ClobClient=FakeClobClient),
    )

    provider = PolymarketDirectQuoteProvider(
        book_client=PolymarketSDKClient(),
        clock=lambda: now,
    )

    quote = await provider.quote(
        PolymarketOrderRequest(
            market_id="m-live-safety",
            token_id="t-yes",
            side="BUY",
            price=0.40,
            size=25.0,
            notional_usdc=10.0,
            estimated_quantity=25.0,
            order_type="limit",
            time_in_force="IOC",
            max_slippage_bps=50,
        ),
        _live_settings().polymarket.credentials(),
    )

    assert quote.market_status == "not_accepting_orders"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_payload",
    [
        {"closed": "not-a-bool"},
        {"active": "not-a-bool"},
        {"accepting_orders": "not-a-bool"},
    ],
)
async def test_polymarket_sdk_direct_quote_rejects_unparseable_status_flags(
    monkeypatch: pytest.MonkeyPatch,
    status_payload: dict[str, object],
) -> None:
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)

    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def get_order_book(self, token_id: str) -> dict[str, object]:
            assert token_id == "t-yes"
            return {
                "timestamp": now.isoformat(),
                **status_payload,
                "bids": [{"price": 0.39, "size": 100.0}],
                "asks": [{"price": 0.40, "size": 100.0}],
            }

    monkeypatch.setitem(
        sys.modules,
        "py_clob_client_v2",
        SimpleNamespace(ApiCreds=FakeApiCreds, ClobClient=FakeClobClient),
    )

    with pytest.raises(LiveTradingDisabledError, match="status flag"):
        await PolymarketSDKClient().read_order_book(
            PolymarketOrderRequest(
                market_id="m-live-safety",
                token_id="t-yes",
                side="BUY",
                price=0.40,
                size=25.0,
                notional_usdc=10.0,
                estimated_quantity=25.0,
                order_type="limit",
                time_in_force="IOC",
                max_slippage_bps=50,
            ),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timestamp_payload",
    [
        {},
        {"timestamp": ""},
        {"timestamp": "not-a-timestamp"},
    ],
)
async def test_polymarket_sdk_direct_quote_rejects_missing_or_bad_venue_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    timestamp_payload: dict[str, object],
) -> None:
    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def get_order_book(self, token_id: str) -> dict[str, object]:
            assert token_id == "t-yes"
            return {
                **timestamp_payload,
                "bids": [{"price": 0.39, "size": 100.0}],
                "asks": [{"price": 0.40, "size": 100.0}],
            }

    monkeypatch.setitem(
        sys.modules,
        "py_clob_client_v2",
        SimpleNamespace(ApiCreds=FakeApiCreds, ClobClient=FakeClobClient),
    )

    with pytest.raises(LiveTradingDisabledError, match="timestamp"):
        await PolymarketSDKClient().read_order_book(
            PolymarketOrderRequest(
                market_id="m-live-safety",
                token_id="t-yes",
                side="BUY",
                price=0.40,
                size=25.0,
                notional_usdc=10.0,
                estimated_quantity=25.0,
                order_type="limit",
                time_in_force="IOC",
                max_slippage_bps=50,
            ),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_level",
    [
        {"price": 0.0, "size": 100.0},
        {"price": -0.01, "size": 100.0},
        {"price": 1.01, "size": 100.0},
        {"price": 0.40, "size": 0.0},
        {"price": 0.40, "size": -1.0},
    ],
)
async def test_polymarket_sdk_direct_quote_rejects_invalid_book_level_domain(
    monkeypatch: pytest.MonkeyPatch,
    bad_level: dict[str, float],
) -> None:
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)

    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def get_order_book(self, token_id: str) -> dict[str, object]:
            assert token_id == "t-yes"
            return {
                "timestamp": now.isoformat(),
                "bids": [{"price": 0.39, "size": 100.0}],
                "asks": [bad_level],
            }

    monkeypatch.setitem(
        sys.modules,
        "py_clob_client_v2",
        SimpleNamespace(ApiCreds=FakeApiCreds, ClobClient=FakeClobClient),
    )

    with pytest.raises(LiveTradingDisabledError, match="order book level"):
        await PolymarketSDKClient().read_order_book(
            PolymarketOrderRequest(
                market_id="m-live-safety",
                token_id="t-yes",
                side="BUY",
                price=0.40,
                size=25.0,
                notional_usdc=10.0,
                estimated_quantity=25.0,
                order_type="limit",
                time_in_force="IOC",
                max_slippage_bps=50,
            ),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("first_live_order", [False, True])
async def test_routing_quote_provider_dual_mode_fails_on_material_mismatch(
    first_live_order: bool,
) -> None:
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    snapshot_provider = PolymarketBookQuoteProvider(
        store=FakeBookStore(
            market=Market(
                condition_id="m-live-safety",
                slug="live-safety",
                question="Will live safety pass?",
                venue="polymarket",
                resolves_at=now + timedelta(days=1),
                created_at=now - timedelta(days=1),
                last_seen_at=now,
            ),
            snapshot=BookSnapshot(
                id=24,
                market_id="m-live-safety",
                token_id="t-yes",
                ts=now - timedelta(milliseconds=100),
                hash="snapshot-book",
                source="subscribe",
            ),
            levels=[
                BookLevel(
                    snapshot_id=24,
                    market_id="m-live-safety",
                    side=Side.SELL.value,
                    price=0.40,
                    size=100.0,
                ),
            ],
        ),
        clock=lambda: now,
    )
    direct_provider = PolymarketDirectQuoteProvider(
        book_client=FakeVenueBookClient(
            LiveVenueBook(
                market_id="m-live-safety",
                token_id="t-yes",
                bids=(),
                asks=(
                    BookLevel(
                        snapshot_id=0,
                        market_id="m-live-safety",
                        side=Side.SELL.value,
                        price=0.44,
                        size=100.0,
                    ),
                ),
                book_ts=now - timedelta(milliseconds=100),
                quote_hash="direct-book",
                market_status="open",
            )
        ),
        clock=lambda: now,
    )
    settings = _live_settings(
        controller=ControllerSettings(
            time_in_force="IOC",
            min_volume=0.0,
            quote_source="dual",
            dual_quote_max_price_delta_bps=25.0,
        )
    )
    provider = PolymarketRoutingQuoteProvider(
        snapshot_provider=snapshot_provider,
        direct_provider=direct_provider,
        settings=settings,
    )

    with pytest.raises(LiveTradingDisabledError, match="dual quote mismatch"):
        await provider.quote_for_order(
            PolymarketOrderRequest(
                market_id="m-live-safety",
                token_id="t-yes",
                side="BUY",
                price=0.45,
                size=25.0,
                notional_usdc=10.0,
                estimated_quantity=25.0,
                order_type="limit",
                time_in_force="IOC",
                max_slippage_bps=50,
            ),
            settings.polymarket.credentials(),
            first_live_order=first_live_order,
        )


@pytest.mark.asyncio
async def test_routing_quote_provider_uses_direct_quote_for_venue_direct_mode() -> None:
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    direct_client = FakeVenueBookClient(
        LiveVenueBook(
            market_id="m-live-safety",
            token_id="t-yes",
            bids=(),
            asks=(
                BookLevel(
                    snapshot_id=0,
                    market_id="m-live-safety",
                    side=Side.SELL.value,
                    price=0.40,
                    size=100.0,
                ),
            ),
            book_ts=now - timedelta(milliseconds=100),
            quote_hash="first-direct",
            market_status="open",
        )
    )
    provider = PolymarketRoutingQuoteProvider(
        snapshot_provider=PolymarketBookQuoteProvider(
            store=FakeBookStore(market=None, snapshot=None, levels=[]),
            clock=lambda: now,
        ),
        direct_provider=PolymarketDirectQuoteProvider(
            book_client=direct_client,
            clock=lambda: now,
        ),
        settings=_live_settings(
            controller=ControllerSettings(
                time_in_force="IOC",
                min_volume=0.0,
                quote_source="venue_direct",
            )
        ),
    )
    settings = _live_settings()

    quote = await provider.quote_for_order(
        PolymarketOrderRequest(
            market_id="m-live-safety",
            token_id="t-yes",
            side="BUY",
            price=0.40,
            size=25.0,
            notional_usdc=10.0,
            estimated_quantity=25.0,
            order_type="limit",
            time_in_force="IOC",
            max_slippage_bps=50,
        ),
        settings.polymarket.credentials(),
        first_live_order=True,
    )

    assert quote.source == "venue_direct"
    assert quote.quote_hash == "first-direct"
    assert direct_client.calls == 1


@pytest.mark.asyncio
async def test_future_dated_book_snapshot_fails_quote_guard() -> None:
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    provider = PolymarketBookQuoteProvider(
        store=FakeBookStore(
            market=Market(
                condition_id="m-live-safety",
                slug="live-safety",
                question="Will live safety pass?",
                venue="polymarket",
                resolves_at=now + timedelta(days=1),
                created_at=now - timedelta(days=1),
                last_seen_at=now,
            ),
            snapshot=BookSnapshot(
                id=22,
                market_id="m-live-safety",
                token_id="t-yes",
                ts=now + timedelta(seconds=5),
                hash="future-book",
                source="subscribe",
            ),
            levels=[
                BookLevel(
                    snapshot_id=22,
                    market_id="m-live-safety",
                    side=Side.SELL.value,
                    price=0.40,
                    size=100.0,
                ),
            ],
        ),
        clock=lambda: now,
    )

    with pytest.raises(LiveTradingDisabledError, match="timestamp is in the future"):
        await provider.quote(
            PolymarketOrderRequest(
                market_id="m-live-safety",
                token_id="t-yes",
                side="BUY",
                price=0.40,
                size=25.0,
                notional_usdc=10.0,
                estimated_quantity=25.0,
                order_type="limit",
                time_in_force="IOC",
                max_slippage_bps=50,
            ),
            _live_settings().polymarket.credentials(),
        )


def test_runner_live_adapter_uses_real_quote_provider_when_live_enabled() -> None:
    runner = Runner(config=_live_settings())
    runner._pg_pool = cast(asyncpg.Pool, object())  # noqa: SLF001

    adapter = runner._build_adapter(RunMode.LIVE)  # noqa: SLF001

    assert isinstance(adapter, PolymarketActuator)
    assert isinstance(adapter.quote_provider, PolymarketRoutingQuoteProvider)
    assert isinstance(adapter.quote_provider.snapshot_provider, PolymarketBookQuoteProvider)
    assert isinstance(adapter.quote_provider.direct_provider, PolymarketDirectQuoteProvider)


@pytest.mark.asyncio
async def test_external_signal_numeric_key_does_not_satisfy_required_raw_factor() -> None:
    pipeline = ControllerPipeline(
        strategy=_strategy("metaculus_prior"),
        factor_reader=SnapshotReader(
            FactorSnapshot(
                values={},
                missing_factors=(("metaculus_prior", ""),),
                snapshot_hash="missing-metaculus",
            )
        ),
        forecasters=(ConstantForecaster(0.7),),
        calibrator=NetcalCalibrator(),
        sizer=FixedSizer(),
        router=Router(ControllerSettings(min_volume=0.0)),
        settings=PMSSettings(
            mode=RunMode.LIVE,
            controller=ControllerSettings(min_volume=0.0),
        ),
    )

    emission = await pipeline.on_signal(
        _signal(external_signal={"metaculus_prior": 0.9}),
        portfolio=_portfolio(),
    )

    assert emission is None
    assert pipeline.last_diagnostic is not None
    assert pipeline.last_diagnostic.code == "missing_required_factors"


@pytest.mark.asyncio
async def test_stale_snapshot_factor_cannot_be_unstaled_by_untrusted_external_signal() -> None:
    pipeline = ControllerPipeline(
        strategy=_strategy("metaculus_prior"),
        factor_reader=SnapshotReader(
            FactorSnapshot(
                values={("metaculus_prior", ""): 0.5},
                stale_factors=(("metaculus_prior", ""),),
                snapshot_hash="stale-metaculus",
            )
        ),
        forecasters=(ConstantForecaster(0.7),),
        calibrator=NetcalCalibrator(),
        sizer=FixedSizer(),
        router=Router(ControllerSettings(min_volume=0.0)),
        settings=PMSSettings(
            mode=RunMode.LIVE,
            controller=ControllerSettings(min_volume=0.0),
        ),
    )

    emission = await pipeline.on_signal(
        _signal(external_signal={"metaculus_prior": 0.9}),
        portfolio=_portfolio(),
    )

    assert emission is None
    assert pipeline.last_diagnostic is not None
    assert pipeline.last_diagnostic.code == "stale_required_factors"


@pytest.mark.asyncio
async def test_paper_mode_strict_factor_gates_by_default() -> None:
    pipeline = ControllerPipeline(
        strategy=_strategy("metaculus_prior"),
        factor_reader=SnapshotReader(
            FactorSnapshot(
                values={},
                missing_factors=(("metaculus_prior", ""),),
                snapshot_hash="paper-missing",
            )
        ),
        forecasters=(ConstantForecaster(0.7),),
        calibrator=NetcalCalibrator(),
        sizer=FixedSizer(),
        router=Router(ControllerSettings(min_volume=0.0)),
        settings=PMSSettings(
            mode=RunMode.PAPER,
            controller=ControllerSettings(min_volume=0.0),
        ),
    )

    emission = await pipeline.on_signal(
        _signal(external_signal={"resolved_outcome": 1.0}),
        portfolio=_portfolio(),
    )

    assert emission is None
    assert pipeline.last_diagnostic is not None
    assert pipeline.last_diagnostic.code == "missing_required_factors"


@pytest.mark.asyncio
async def test_backtest_can_disable_strict_factor_gates_for_exploration() -> None:
    pipeline = ControllerPipeline(
        strategy=_strategy("metaculus_prior"),
        factor_reader=SnapshotReader(
            FactorSnapshot(
                values={},
                missing_factors=(("metaculus_prior", ""),),
                snapshot_hash="backtest-missing",
            )
        ),
        forecasters=(ConstantForecaster(0.7),),
        calibrator=NetcalCalibrator(),
        sizer=FixedSizer(),
        router=Router(ControllerSettings(min_volume=0.0, strict_factor_gates=False)),
        settings=PMSSettings(
            mode=RunMode.BACKTEST,
            controller=ControllerSettings(min_volume=0.0, strict_factor_gates=False),
        ),
    )

    emission = await pipeline.on_signal(
        _signal(external_signal={"resolved_outcome": 1.0}),
        portfolio=_portfolio(),
    )

    assert emission is not None


@dataclass
class AllowFirstOrderGate:
    previews: list[LiveOrderPreview] = field(default_factory=list)

    async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
        self.previews.append(preview)
        return True

    async def consume(self, preview: LiveOrderPreview) -> None:
        del preview


@dataclass(frozen=True)
class RecordingFileFirstOrderGate(FileFirstLiveOrderGate):
    previews: list[LiveOrderPreview] = field(default_factory=list)

    async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
        self.previews.append(preview)
        return await super().approve_first_order(preview)


def _strict_file_gate(
    settings: PMSSettings,
    decision: TradeDecision,
) -> FileFirstLiveOrderGate:
    approval_path = _write_operator_approval(settings, decision)
    return FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=settings.polymarket.operator_approval_max_age_s,
    )


def _strict_recording_file_gate(
    settings: PMSSettings,
    decision: TradeDecision,
) -> RecordingFileFirstOrderGate:
    approval_path = _write_operator_approval(settings, decision)
    return RecordingFileFirstOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=settings.polymarket.operator_approval_max_age_s,
    )


def _write_operator_approval(
    settings: PMSSettings,
    decision: TradeDecision,
) -> Path:
    assert decision.limit_price is not None
    approval_path = Path(cast(str, settings.polymarket.first_live_order_approval_path))
    approval_payload: dict[str, object] = {
        "approved": True,
        "max_notional_usdc": decision.notional_usdc,
        "venue": decision.venue,
        "market_id": decision.market_id,
        "token_id": decision.token_id,
        "side": decision.side,
        "outcome": decision.outcome,
        "limit_price": decision.limit_price,
        "max_slippage_bps": decision.max_slippage_bps,
    }
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps(
            {
                "approver_id": "test-operator",
                "approval_sha256": _approval_payload_hash(approval_payload),
                "ts": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return approval_path


def _approval_payload_hash(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _sidecar_path(approval_path: Path) -> Path:
    return Path(str(approval_path) + ".meta.json")


@dataclass
class AllowQuoteProvider:
    async def quote(
        self,
        order: PolymarketOrderRequest,
        credentials: object,
    ) -> LivePreSubmitQuote:
        del credentials
        return LivePreSubmitQuote(
            market_status="open",
            book_age_ms=10.0,
            executable_notional_usdc=order.notional_usdc,
            best_executable_price=order.price,
            spread_bps=5.0,
            quote_hash="quote-ok",
            book_ts=datetime(2026, 4, 27, tzinfo=UTC),
        )


@dataclass
class RestingNonGtcClient:
    async def submit_order(
        self,
        order: PolymarketOrderRequest,
        credentials: object,
    ) -> PolymarketOrderResult:
        del credentials
        return PolymarketOrderResult(
            order_id="pm-resting-ioc",
            status=OrderStatus.LIVE.value,
            raw_status="live",
            filled_notional_usdc=2.0,
            remaining_notional_usdc=8.0,
            fill_price=order.price,
            filled_quantity=5.0,
        )


@pytest.mark.asyncio
async def test_live_ioc_order_returning_live_with_remaining_triggers_reconciliation_halt() -> None:
    settings = _live_settings()
    settings.live_preflight_artifact_path = make_live_preflight_artifact_path(
        prefix="pms-live-safety-resting-ioc-preflight-",
        settings=settings,
    )
    decision = _decision(time_in_force=TimeInForce.IOC)
    actuator = PolymarketActuator(
        settings,
        client=RestingNonGtcClient(),
        operator_gate=_strict_file_gate(settings, decision),
        quote_provider=AllowQuoteProvider(),
        live_preflight_validated=True,
    )

    with pytest.raises(PolymarketSubmissionUnknownError) as exc_info:
        await actuator.execute(decision, _portfolio())

    assert "Non-GTC live order appears resting" in str(exc_info.value)
    assert exc_info.value.order_state is not None
    assert exc_info.value.order_state.order_id == "pm-resting-ioc"


@dataclass
class MatchedClient:
    async def submit_order(
        self,
        order: PolymarketOrderRequest,
        credentials: object,
    ) -> PolymarketOrderResult:
        del credentials
        return PolymarketOrderResult(
            order_id="pm-preview-outcome",
            status=OrderStatus.MATCHED.value,
            raw_status="matched",
            filled_notional_usdc=order.notional_usdc,
            remaining_notional_usdc=0.0,
            fill_price=order.price,
            filled_quantity=order.estimated_quantity,
        )


@pytest.mark.asyncio
async def test_first_live_order_preview_and_approval_require_outcome(
    tmp_path: Any,
) -> None:
    approval_path = tmp_path / "approval.json"
    gate = FileFirstLiveOrderGate(approval_path)
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-live-safety",
        token_id="t-no",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
        outcome="NO",
    )
    approval_path.write_text(
        '{"approved": true, "venue": "polymarket", "market_id": "m-live-safety", '
        '"token_id": "t-no", "side": "BUY", "max_notional_usdc": 10.0, '
        '"limit_price": 0.4, "max_slippage_bps": 50}',
        encoding="utf-8",
    )
    assert await gate.approve_first_order(preview) is False

    approval_path.write_text(
        '{"approved": true, "venue": "polymarket", "market_id": "m-live-safety", '
        '"token_id": "t-no", "side": "BUY", "outcome": "NO", '
        '"max_notional_usdc": 10.0, "limit_price": 0.4, "max_slippage_bps": 50}',
        encoding="utf-8",
    )
    assert await gate.approve_first_order(preview) is True

    settings = _live_settings()
    settings.live_preflight_artifact_path = make_live_preflight_artifact_path(
        prefix="pms-live-safety-preview-preflight-",
        settings=settings,
    )
    decision = _decision(outcome="NO")
    recording_gate = _strict_recording_file_gate(settings, decision)
    actuator = PolymarketActuator(
        settings,
        client=MatchedClient(),
        operator_gate=recording_gate,
        quote_provider=AllowQuoteProvider(),
        live_preflight_validated=True,
    )
    await actuator.execute(decision, _portfolio())

    assert recording_gate.previews[0].outcome == "NO"

    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    metadata_decision = _decision(outcome="NO")
    metadata_store = FakeBookStore(
        market=Market(
            condition_id="m-live-safety",
            slug="live-safety",
            question="Will live safety pass?",
            venue="polymarket",
            resolves_at=now + timedelta(days=1),
            created_at=now - timedelta(days=1),
            last_seen_at=now,
        ),
        snapshot=BookSnapshot(
            id=31,
            market_id="m-live-safety",
            token_id="t-no",
            ts=now - timedelta(milliseconds=10),
            hash="book-preview",
            source="subscribe",
        ),
        levels=[
            BookLevel(
                snapshot_id=31,
                market_id="m-live-safety",
                side=Side.BUY.value,
                price=0.399,
                size=25.0,
            ),
            BookLevel(
                snapshot_id=31,
                market_id="m-live-safety",
                side=Side.SELL.value,
                price=0.40,
                size=25.0,
            ),
        ],
    )
    settings_with_book = _live_settings()
    settings_with_book.live_preflight_artifact_path = make_live_preflight_artifact_path(
        prefix="pms-live-safety-preview-book-preflight-",
        settings=settings_with_book,
    )
    metadata_gate = _strict_recording_file_gate(settings_with_book, metadata_decision)
    actuator_with_book = PolymarketActuator(
        settings_with_book,
        client=MatchedClient(),
        operator_gate=metadata_gate,
        quote_provider=PolymarketBookQuoteProvider(
            store=metadata_store,
            clock=lambda: now,
        ),
        live_preflight_validated=True,
    )
    await actuator_with_book.execute(metadata_decision, _portfolio())

    assert metadata_gate.previews[0].outcome == "NO"
    assert metadata_gate.previews[0].market_slug == "live-safety"
    assert metadata_gate.previews[0].question == "Will live safety pass?"


class _Acquire:
    def __init__(self, connection: "_Connection") -> None:
        self.connection = connection

    async def __aenter__(self) -> "_Connection":
        return self.connection

    async def __aexit__(self, *_: object) -> None:
        return None


class _Transaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_: object) -> None:
        return None


class _Connection:
    def __init__(self) -> None:
        self.fetchval_result: int = 0
        self.latest_book_snapshot_age_s: float | None = 30.0
        self.latest_usable_book_snapshot_age_s: float | None = 30.0
        self.missing_market_risk_metadata_count = 0
        self.missing_launch_usable_token_count = 0
        self.missing_subscribed_usable_token_count = 0
        self.fetchrow_result: dict[str, object] | None = None
        self.decision_submission_unknown_exists = True
        self.order_intent_submission_unknown_exists = True
        self.order_intent_already_reconciled = False
        self.execute_result = "UPDATE 1"
        self.fetchval_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchval(self, query: str, *args: object) -> object:
        self.fetchval_calls.append((query, args))
        if "missing_launch_usable_tokens" in query:
            return self.missing_launch_usable_token_count
        if "missing_subscribed_usable_tokens" in query:
            return self.missing_subscribed_usable_token_count
        if "missing_market_risk_metadata" in query:
            return self.missing_market_risk_metadata_count
        if "usable_book_snapshots" in query:
            return self.latest_usable_book_snapshot_age_s
        if "book_snapshots" in query:
            return self.latest_book_snapshot_age_s
        return self.fetchval_result

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        self.fetchrow_calls.append((query, args))
        if "UPDATE order_intents" in query and "outcome = 'submission_unknown'" in query:
            if not self.order_intent_submission_unknown_exists:
                return None
            has_unresolved_guard = "reconciled_at IS NULL" in query
            if has_unresolved_guard and self.order_intent_already_reconciled:
                return None
            has_decision_guard = (
                "EXISTS" in query
                and "decisions" in query
                and "decisions.status = 'submission_unknown'" in query
            )
            if has_decision_guard and not self.decision_submission_unknown_exists:
                return None
        return self.fetchrow_result

    async def execute(self, query: str, *args: object) -> str:
        self.execute_calls.append((query, args))
        return self.execute_result

    def transaction(self) -> _Transaction:
        return _Transaction()


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self) -> _Acquire:
        return _Acquire(self.connection)


def _live_order_reconciliation_row() -> dict[str, object]:
    submitted_at = datetime(2026, 5, 26, 10, 0, tzinfo=UTC)
    filled_at = datetime(2026, 5, 26, 10, 0, 2, tzinfo=UTC)
    return {
        "decision_id": "d-live",
        "decision_status": "filled",
        "order_id": "venue-order-1",
        "order_status": "matched",
        "order_raw_status": "matched",
        "market_id": "m-live-safety",
        "token_id": "t-yes",
        "venue": "polymarket",
        "strategy_id": "alpha",
        "strategy_version_id": "alpha-v1",
        "requested_notional_usdc": 10.0,
        "filled_notional_usdc": 10.0,
        "remaining_notional_usdc": 0.0,
        "filled_quantity": 25.0,
        "submitted_at": submitted_at,
        "time_in_force": "IOC",
        "action": "BUY",
        "outcome": "YES",
        "intent_key": "intent-live",
        "pre_submit_quote_json": {
            "quote_hash": "quote-hash-1",
            "source": "dual",
        },
        "order_payload": {
            "decision_id": "d-live",
            "fill_price": 0.4,
            "last_updated_at": filled_at.isoformat(),
        },
        "fill_id": "fill-1",
        "fill_notional_usdc": 10.0,
        "fill_quantity": 25.0,
        "filled_at": filled_at,
        "fill_payload": {
            "decision_id": "d-live",
            "status": "matched",
        },
    }


@pytest.mark.asyncio
async def test_live_order_reconciliation_store_loads_persisted_order_and_fill() -> None:
    connection = _Connection()
    connection.fetchrow_result = _live_order_reconciliation_row()
    store = LiveOrderReconciliationStore(cast(asyncpg.Pool, _Pool(connection)))

    record = await store.load_live_order_record(decision_id="d-live")

    assert record is not None
    assert record.decision_id == "d-live"
    assert record.order_id == "venue-order-1"
    assert record.fill_id == "fill-1"
    assert record.pre_submit_quote_hash == "quote-hash-1"
    assert record.pre_submit_quote_source == "dual"
    assert record.filled_notional_usdc == 10.0
    query, args = connection.fetchrow_calls[0]
    assert args == ("d-live",)
    assert "order_payloads.payload->>'decision_id' = decisions.decision_id" in query
    assert "fill_payloads.payload->>'decision_id' = decisions.decision_id" in query
    assert "INNER JOIN fills" in query


@pytest.mark.asyncio
async def test_live_order_reconciliation_store_rejects_duplicate_json_keys() -> None:
    connection = _Connection()
    row = _live_order_reconciliation_row()
    row["pre_submit_quote_json"] = (
        '{"quote_hash": "forged-quote-hash", '
        '"quote_hash": "quote-hash-1", "source": "dual"}'
    )
    connection.fetchrow_result = row
    store = LiveOrderReconciliationStore(cast(asyncpg.Pool, _Pool(connection)))

    with pytest.raises(RuntimeError, match="duplicate JSON key: quote_hash"):
        await store.load_live_order_record(decision_id="d-live")


@pytest.mark.asyncio
async def test_live_start_refuses_unresolved_submission_unknown() -> None:
    connection = _Connection()
    connection.fetchval_result = 1
    runner = Runner(config=_live_settings())
    runner._pg_pool = cast(asyncpg.Pool, _Pool(connection))  # noqa: SLF001

    with pytest.raises(RuntimeError, match="unresolved submission_unknown"):
        await runner._assert_no_unresolved_submission_unknown_incidents()  # noqa: SLF001

    query, _ = connection.fetchval_calls[0]
    assert "reconciled_at IS NULL" in query


@pytest.mark.asyncio
async def test_reconciled_submission_unknown_allows_live_restart() -> None:
    connection = _Connection()
    connection.fetchval_result = 0
    runner = Runner(config=_live_settings())
    runner._pg_pool = cast(asyncpg.Pool, _Pool(connection))  # noqa: SLF001

    await runner._assert_no_unresolved_submission_unknown_incidents()  # noqa: SLF001


@pytest.mark.asyncio
async def test_submission_unknown_startup_guard_redacts_db_failure() -> None:
    class _FailingSubmissionUnknownConnection(_Connection):
        async def fetchval(self, query: str, *args: object) -> object:
            self.fetchval_calls.append((query, args))
            if "order_intents" in query and "submission_unknown" in query:
                raise RuntimeError(
                    _secret_bearing_reconciliation_detail(
                        "submission_unknown guard failed"
                    )
                )
            return await super().fetchval(query, *args)

    connection = _FailingSubmissionUnknownConnection()
    runner = Runner(config=_live_settings_with_secret_polymarket_credentials())
    runner._pg_pool = cast(asyncpg.Pool, _Pool(connection))  # noqa: SLF001

    with pytest.raises(RuntimeError) as exc_info:
        await runner._assert_no_unresolved_submission_unknown_incidents()  # noqa: SLF001

    rendered = str(exc_info.value)
    assert "LIVE submission_unknown guard failed" in rendered
    _assert_reconciliation_detail_redacted(rendered)


@pytest.mark.asyncio
async def test_live_start_refuses_missing_market_data_freshness() -> None:
    connection = _Connection()
    connection.latest_book_snapshot_age_s = None
    runner = Runner(config=_live_settings())
    runner._pg_pool = cast(asyncpg.Pool, _Pool(connection))  # noqa: SLF001

    with pytest.raises(RuntimeError, match="no book_snapshots"):
        await runner._assert_live_market_data_freshness()  # noqa: SLF001


@pytest.mark.asyncio
async def test_recent_two_sided_market_data_allows_live_start() -> None:
    connection = _Connection()
    connection.latest_book_snapshot_age_s = 30.0
    connection.latest_usable_book_snapshot_age_s = 30.0
    runner = Runner(config=_live_settings())
    runner._pg_pool = cast(asyncpg.Pool, _Pool(connection))  # noqa: SLF001

    await runner._assert_live_market_data_freshness()  # noqa: SLF001


@pytest.mark.asyncio
async def test_live_start_refuses_fresh_usable_market_without_risk_group_metadata() -> None:
    connection = _Connection()
    connection.latest_book_snapshot_age_s = 30.0
    connection.latest_usable_book_snapshot_age_s = 30.0
    connection.missing_market_risk_metadata_count = 1
    runner = Runner(config=_live_settings())
    runner._pg_pool = cast(asyncpg.Pool, _Pool(connection))  # noqa: SLF001

    with pytest.raises(RuntimeError, match="risk_group_id"):
        await runner._assert_live_market_data_freshness()  # noqa: SLF001


@pytest.mark.asyncio
async def test_submission_unknown_reconciliation_store_marks_incident_resolved() -> None:
    connection = _Connection()
    connection.fetchrow_result = {"decision_id": "d-unknown"}
    store = SubmissionUnknownReconciliationStore(cast(asyncpg.Pool, _Pool(connection)))

    updated = await store.reconcile_submission_unknown(
        decision_id="d-unknown",
        venue_order_id="pm-123",
        status="filled",
        reconciled_by="operator",
        note="matched venue fill",
    )

    assert updated is True
    query, args = connection.fetchrow_calls[0]
    assert "reconciled_at = now()" in query
    assert "reconciled_at IS NULL" in query
    assert "outcome = 'submission_unknown'" in query
    assert "EXISTS" in query
    assert "decisions.status = 'submission_unknown'" in query
    assert args == ("d-unknown", "pm-123", "filled", "operator", "matched venue fill")
    decision_query, decision_args = connection.execute_calls[0]
    assert "UPDATE decisions" in decision_query
    assert "status = 'reconciled'" in decision_query
    assert "status = 'submission_unknown'" in decision_query
    assert decision_args == ("d-unknown",)


@pytest.mark.asyncio
async def test_submission_unknown_reconciliation_rejects_inconsistent_fill_before_write() -> None:
    connection = _Connection()
    store = SubmissionUnknownReconciliationStore(cast(asyncpg.Pool, _Pool(connection)))

    with pytest.raises(ValueError, match="fill_notional_usdc does not match"):
        await store.reconcile_submission_unknown(
            decision_id="d-unknown",
            venue_order_id="pm-123",
            status="filled",
            reconciled_by="operator",
            fill_price=0.5,
            fill_notional_usdc=4.0,
            fill_quantity=10.0,
        )

    assert connection.fetchrow_calls == []
    assert connection.execute_calls == []


@pytest.mark.asyncio
async def test_submission_unknown_reconciliation_requires_matching_decision_state() -> None:
    connection = _Connection()
    connection.fetchrow_result = {"decision_id": "d-unknown"}
    connection.decision_submission_unknown_exists = False
    store = SubmissionUnknownReconciliationStore(cast(asyncpg.Pool, _Pool(connection)))

    updated = await store.reconcile_submission_unknown(
        decision_id="d-unknown",
        venue_order_id="pm-123",
        status="filled",
        reconciled_by="operator",
        note="matched venue fill",
    )

    assert updated is False
    assert connection.execute_calls == []
    query, _ = connection.fetchrow_calls[0]
    assert "EXISTS" in query
    assert "decisions.status = 'submission_unknown'" in query


@pytest.mark.asyncio
async def test_submission_unknown_reconciliation_rolls_back_when_decision_update_misses() -> None:
    connection = _Connection()
    connection.fetchrow_result = {"decision_id": "d-unknown"}
    connection.execute_result = "UPDATE 0"
    store = SubmissionUnknownReconciliationStore(cast(asyncpg.Pool, _Pool(connection)))

    with pytest.raises(RuntimeError, match="decision row was not reconciled"):
        await store.reconcile_submission_unknown(
            decision_id="d-unknown",
            venue_order_id="pm-123",
            status="filled",
            reconciled_by="operator",
            note="matched venue fill",
        )

    assert len(connection.fetchrow_calls) == 1
    assert len(connection.execute_calls) == 1


@pytest.mark.asyncio
async def test_submission_unknown_reconciliation_ignores_already_reconciled_intent() -> None:
    connection = _Connection()
    connection.fetchrow_result = {"decision_id": "d-unknown"}
    connection.order_intent_already_reconciled = True
    store = SubmissionUnknownReconciliationStore(cast(asyncpg.Pool, _Pool(connection)))

    updated = await store.reconcile_submission_unknown(
        decision_id="d-unknown",
        venue_order_id="pm-123",
        status="filled",
        reconciled_by="operator",
        note="duplicate operator attempt",
    )

    assert updated is False
    assert connection.execute_calls == []
    query, _ = connection.fetchrow_calls[0]
    assert "reconciled_at IS NULL" in query


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["filled", "open"])
async def test_submission_unknown_reconciliation_requires_venue_order_id_for_live_order_status(
    status: SubmissionUnknownResolutionStatus,
) -> None:
    connection = _Connection()
    store = SubmissionUnknownReconciliationStore(cast(asyncpg.Pool, _Pool(connection)))

    with pytest.raises(ValueError, match="venue_order_id is required"):
        await store.reconcile_submission_unknown(
            decision_id="d-unknown",
            venue_order_id=" ",
            status=status,
            reconciled_by="operator",
            note="matched venue state",
        )

    assert connection.fetchrow_calls == []
    assert connection.execute_calls == []


@pytest.mark.asyncio
async def test_submission_unknown_reconciliation_requires_reconciled_by() -> None:
    connection = _Connection()
    store = SubmissionUnknownReconciliationStore(cast(asyncpg.Pool, _Pool(connection)))

    with pytest.raises(ValueError, match="reconciled_by is required"):
        await store.reconcile_submission_unknown(
            decision_id="d-unknown",
            venue_order_id="pm-123",
            status="filled",
            reconciled_by=" ",
            note="matched venue state",
        )

    assert connection.fetchrow_calls == []
    assert connection.execute_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reconciled_by",
    (
        "operator|forged",
        "operator\nforged",
        "operator\rforged",
    ),
)
async def test_submission_unknown_reconciliation_rejects_forged_reconciled_by(
    reconciled_by: str,
) -> None:
    connection = _Connection()
    store = SubmissionUnknownReconciliationStore(cast(asyncpg.Pool, _Pool(connection)))

    with pytest.raises(ValueError, match="reconciled_by"):
        await store.reconcile_submission_unknown(
            decision_id="d-unknown",
            venue_order_id="pm-123",
            status="filled",
            reconciled_by=reconciled_by,
            note="matched venue state",
        )

    assert connection.fetchrow_calls == []
    assert connection.execute_calls == []


@pytest.mark.asyncio
async def test_api_reconciles_submission_unknown_incident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _Connection()
    connection.fetchrow_result = {"decision_id": "d-unknown"}
    runner = Runner(config=PMSSettings(auto_migrate_default_v2=False))
    runner.config.api_token = "live-api-token"
    runner._pg_pool = cast(asyncpg.Pool, _Pool(connection))  # noqa: SLF001

    async def fake_schema_check(pool: object) -> None:
        assert pool is runner.pg_pool

    monkeypatch.setattr("pms.api.app.ensure_schema_current", fake_schema_check)
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/live/reconcile-submission-unknown",
            headers=AUTH_HEADERS,
            json={
                "decision_id": "d-unknown",
                "venue_order_id": "pm-123",
                "status": "filled",
                "reconciled_by": "operator",
                "note": "matched venue fill",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "reconciled",
        "decision_id": "d-unknown",
        "resolution": "filled",
    }


@pytest.mark.asyncio
async def test_api_reconcile_submission_unknown_checks_schema_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _Connection()
    connection.fetchrow_result = {"decision_id": "d-unknown"}
    runner = Runner(config=PMSSettings(auto_migrate_default_v2=False))
    runner.config.api_token = "live-api-token"
    runner._pg_pool = cast(asyncpg.Pool, _Pool(connection))  # noqa: SLF001
    secret_dsn = "postgresql://admin:supersecret@db.internal.example.com:5432/pms_live"

    async def fake_schema_check(pool: object) -> None:
        assert pool is runner.pg_pool
        raise RuntimeError(
            f"schema stale for {secret_dsn} password=keyword-secret"
        )

    monkeypatch.setattr("pms.api.app.ensure_schema_current", fake_schema_check)
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/live/reconcile-submission-unknown",
            headers=AUTH_HEADERS,
            json={
                "decision_id": "d-unknown",
                "venue_order_id": "pm-123",
                "status": "filled",
                "reconciled_by": "operator",
                "note": "matched venue fill",
            },
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "<redacted-database-url>" in detail
    assert "password=<redacted>" in detail
    assert "supersecret" not in detail
    assert "keyword-secret" not in detail
    assert "admin" not in detail
    assert connection.fetchrow_calls == []
    assert connection.execute_calls == []


@pytest.mark.asyncio
async def test_api_reconcile_submission_unknown_redacts_live_credentials_from_schema_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_values = (
        "private-key-secret",
        "api-key-secret",
        "api-secret-secret",
        "passphrase-secret",
        "0x2222222222222222222222222222222222222222",
    )
    base_settings = _live_settings()
    runner = Runner(
        config=base_settings.model_copy(
            update={
                "polymarket": PolymarketSettings(
                    private_key=credential_values[0],
                    api_key=credential_values[1],
                    api_secret=credential_values[2],
                    api_passphrase=credential_values[3],
                    signature_type=1,
                    funder_address=credential_values[4],
                    operator_approval_mode="every_order",
                    first_live_order_approval_path=(
                        base_settings.polymarket.first_live_order_approval_path
                    ),
                )
            }
        )
    )
    connection = _Connection()
    connection.fetchrow_result = {"decision_id": "d-unknown"}
    runner._pg_pool = cast(asyncpg.Pool, _Pool(connection))  # noqa: SLF001
    secret_dsn = "postgresql://admin:supersecret@db.internal.example.com:5432/pms_live"

    async def fake_schema_check(pool: object) -> None:
        assert pool is runner.pg_pool
        raise RuntimeError(
            "schema stale "
            f"{credential_values[0]} {credential_values[1]} "
            f"{credential_values[2]} {credential_values[3]} {credential_values[4]} "
            f"{secret_dsn} password=keyword-secret"
        )

    monkeypatch.setattr("pms.api.app.ensure_schema_current", fake_schema_check)
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/live/reconcile-submission-unknown",
            headers=AUTH_HEADERS,
            json={
                "decision_id": "d-unknown",
                "venue_order_id": "pm-123",
                "status": "filled",
                "reconciled_by": "operator",
                "note": "matched venue fill",
            },
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "<redacted-polymarket-credential>" in detail
    assert "<redacted-database-url>" in detail
    assert "password=<redacted>" in detail
    for credential in credential_values:
        assert credential not in detail
    assert "supersecret" not in detail
    assert "keyword-secret" not in detail
    assert "admin" not in detail
    assert connection.fetchrow_calls == []
    assert connection.execute_calls == []


@pytest.mark.asyncio
async def test_api_rejects_open_submission_unknown_without_venue_order_id() -> None:
    connection = _Connection()
    connection.fetchrow_result = {"decision_id": "d-unknown"}
    runner = Runner(config=PMSSettings(auto_migrate_default_v2=False))
    runner.config.api_token = "live-api-token"
    runner._pg_pool = cast(asyncpg.Pool, _Pool(connection))  # noqa: SLF001
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/live/reconcile-submission-unknown",
            headers=AUTH_HEADERS,
            json={
                "decision_id": "d-unknown",
                "status": "open",
                "reconciled_by": "operator",
                "note": "venue still shows open exposure",
            },
        )

    assert response.status_code == 400
    assert "venue_order_id is required" in response.json()["detail"]
    assert connection.fetchrow_calls == []
    assert connection.execute_calls == []


@pytest.mark.asyncio
async def test_api_rejects_submission_unknown_blank_reconciled_by() -> None:
    connection = _Connection()
    connection.fetchrow_result = {"decision_id": "d-unknown"}
    runner = Runner(config=PMSSettings(auto_migrate_default_v2=False))
    runner.config.api_token = "live-api-token"
    runner._pg_pool = cast(asyncpg.Pool, _Pool(connection))  # noqa: SLF001
    app = create_app(runner, auto_start=False)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/live/reconcile-submission-unknown",
            headers=AUTH_HEADERS,
            json={
                "decision_id": "d-unknown",
                "venue_order_id": "pm-123",
                "status": "filled",
                "reconciled_by": " ",
                "note": "matched venue fill",
            },
        )

    assert response.status_code == 400
    assert "reconciled_by is required" in response.json()["detail"]
    assert connection.fetchrow_calls == []
    assert connection.execute_calls == []


def test_pms_live_cli_parses_submission_unknown_reconcile_command() -> None:
    args = build_parser().parse_args(
        [
            "reconcile-submission-unknown",
            "--decision-id",
            "d-unknown",
            "--venue-order-id",
            "pm-123",
            "--status",
            "filled",
            "--reconciled-by",
            "operator",
            "--note",
            "matched venue fill",
        ]
    )

    assert args.command == "reconcile-submission-unknown"
    assert args.decision_id == "d-unknown"
    assert args.venue_order_id == "pm-123"
    assert args.status == "filled"
    assert args.reconciled_by == "operator"
    assert args.note == "matched venue fill"


def test_decision_lifecycle_statuses_cover_execution_and_reconciliation() -> None:
    assert "submitted" in DECISION_STATUSES
    assert "submission_unknown" in DECISION_STATUSES
    assert "reconciled" in DECISION_STATUSES
    validate_decision_status_transition("accepted", "rejected")
    validate_decision_status_transition("accepted", "queued")
    validate_decision_status_transition("queued", "submitted")
    validate_decision_status_transition("submitted", "rejected")
    validate_decision_status_transition("submitted", "submission_unknown")
    validate_decision_status_transition("submission_unknown", "reconciled")


class MatchingVenueReconciler:
    async def snapshot(self, credentials: object) -> VenueAccountSnapshot:
        del credentials
        return VenueAccountSnapshot(
            balances={"USDC": 1000.0},
            open_orders=(),
            positions=(),
        )

    async def compare(
        self,
        db_portfolio: Portfolio,
        venue_snapshot: VenueAccountSnapshot,
    ) -> ReconciliationReport:
        del db_portfolio, venue_snapshot
        return ReconciliationReport(ok=True, mismatches=())


class MismatchingVenueReconciler(MatchingVenueReconciler):
    async def compare(
        self,
        db_portfolio: Portfolio,
        venue_snapshot: VenueAccountSnapshot,
    ) -> ReconciliationReport:
        del db_portfolio, venue_snapshot
        return ReconciliationReport(ok=False, mismatches=("position mismatch",))


_VENUE_RECONCILIATION_SECRET_VALUES = (
    "private-key-secret",
    "api-key-secret",
    "api-secret-secret",
    "passphrase-secret",
    "0x2222222222222222222222222222222222222222",
)
_VENUE_RECONCILIATION_SECRET_DSN = (
    "postgresql://admin:supersecret@db.internal.example.com:5432/pms_live"
)


def _live_settings_with_secret_polymarket_credentials() -> PMSSettings:
    return _live_settings(
        polymarket=PolymarketSettings(
            private_key=_VENUE_RECONCILIATION_SECRET_VALUES[0],
            api_key=_VENUE_RECONCILIATION_SECRET_VALUES[1],
            api_secret=_VENUE_RECONCILIATION_SECRET_VALUES[2],
            api_passphrase=_VENUE_RECONCILIATION_SECRET_VALUES[3],
            signature_type=1,
            funder_address=_VENUE_RECONCILIATION_SECRET_VALUES[4],
            operator_approval_mode="every_order",
            first_live_order_approval_path=make_private_live_paths(
                prefix="pms-live-secret-reconcile-"
            )[0],
        )
    )


def _secret_bearing_reconciliation_detail(prefix: str) -> str:
    return (
        f"{prefix} "
        f"{_VENUE_RECONCILIATION_SECRET_VALUES[0]} "
        f"{_VENUE_RECONCILIATION_SECRET_VALUES[1]} "
        f"{_VENUE_RECONCILIATION_SECRET_VALUES[2]} "
        f"{_VENUE_RECONCILIATION_SECRET_VALUES[3]} "
        f"{_VENUE_RECONCILIATION_SECRET_VALUES[4]} "
        f"{_VENUE_RECONCILIATION_SECRET_DSN} password=keyword-secret"
    )


def _assert_reconciliation_detail_redacted(rendered: str) -> None:
    assert "<redacted-polymarket-credential>" in rendered
    assert "<redacted-database-url>" in rendered
    assert "password=<redacted>" in rendered
    for credential in _VENUE_RECONCILIATION_SECRET_VALUES:
        assert credential not in rendered
    assert "supersecret" not in rendered
    assert "keyword-secret" not in rendered
    assert "admin" not in rendered


@dataclass
class FakeVenueAccountClient:
    snapshot_value: VenueAccountSnapshot
    calls: int = 0

    async def read_account_snapshot(
        self,
        credentials: VenueCredentials,
    ) -> VenueAccountSnapshot:
        assert credentials.venue == "polymarket"
        self.calls += 1
        return self.snapshot_value


def _venue_open_order() -> OrderState:
    now = datetime(2026, 4, 27, tzinfo=UTC)
    return OrderState(
        order_id="pm-open-1",
        decision_id="venue-open-pm-open-1",
        status=OrderStatus.LIVE.value,
        market_id="m-live-safety",
        token_id="t-yes",
        venue="polymarket",
        requested_notional_usdc=5.0,
        filled_notional_usdc=0.0,
        remaining_notional_usdc=5.0,
        fill_price=0.40,
        submitted_at=now,
        last_updated_at=now,
        raw_status="open",
        strategy_id="venue",
        strategy_version_id="venue",
    )


def _venue_position(*, shares: float = 25.0) -> Position:
    return Position(
        market_id="m-live-safety",
        token_id="t-yes",
        venue="polymarket",
        side=Side.BUY.value,
        shares_held=shares,
        avg_entry_price=0.4,
        unrealized_pnl=0.0,
        locked_usdc=shares * 0.4,
    )


@pytest.mark.asyncio
async def test_live_venue_account_reconciliation_blocks_mismatch() -> None:
    runner = Runner(
        config=_live_settings(live_account_reconciliation_required=True),
        venue_account_reconciler=MismatchingVenueReconciler(),
    )

    with pytest.raises(RuntimeError, match="venue account reconciliation mismatch"):
        await runner._reconcile_venue_account()  # noqa: SLF001


@pytest.mark.asyncio
async def test_live_venue_account_reconciliation_redacts_reconciler_failure() -> None:
    class _SecretFailingVenueReconciler(MatchingVenueReconciler):
        async def snapshot(self, credentials: object) -> VenueAccountSnapshot:
            del credentials
            raise RuntimeError(
                _secret_bearing_reconciliation_detail("venue snapshot failed")
            )

    runner = Runner(
        config=_live_settings_with_secret_polymarket_credentials(),
        venue_account_reconciler=_SecretFailingVenueReconciler(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await runner._reconcile_venue_account()  # noqa: SLF001

    rendered = str(exc_info.value)
    assert "LIVE venue account reconciliation failed" in rendered
    _assert_reconciliation_detail_redacted(rendered)


@pytest.mark.asyncio
async def test_live_venue_account_reconciliation_redacts_mismatch_details() -> None:
    class _SecretMismatchingVenueReconciler(MatchingVenueReconciler):
        async def compare(
            self,
            db_portfolio: Portfolio,
            venue_snapshot: VenueAccountSnapshot,
        ) -> ReconciliationReport:
            del db_portfolio, venue_snapshot
            return ReconciliationReport(
                ok=False,
                mismatches=(
                    _secret_bearing_reconciliation_detail("venue mismatch"),
                ),
            )

    runner = Runner(
        config=_live_settings_with_secret_polymarket_credentials(),
        venue_account_reconciler=_SecretMismatchingVenueReconciler(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await runner._reconcile_venue_account()  # noqa: SLF001

    rendered = str(exc_info.value)
    assert "LIVE venue account reconciliation mismatch" in rendered
    _assert_reconciliation_detail_redacted(rendered)


@pytest.mark.asyncio
async def test_live_venue_account_reconciliation_allows_matching_snapshot() -> None:
    runner = Runner(
        config=_live_settings(live_account_reconciliation_required=True),
        venue_account_reconciler=MatchingVenueReconciler(),
    )

    await runner._reconcile_venue_account()  # noqa: SLF001


@pytest.mark.asyncio
async def test_polymarket_venue_reconciler_reads_snapshot_client_and_blocks_open_orders() -> None:
    client = FakeVenueAccountClient(
        VenueAccountSnapshot(
            balances={"PUSD": 1000.0, "PUSD_ALLOWANCE": 1000.0},
            open_orders=(_venue_open_order(),),
            positions=(),
        )
    )
    reconciler = PolymarketVenueAccountReconciler(client=client)

    snapshot = await reconciler.snapshot(_live_settings().polymarket.credentials())
    report = await reconciler.compare(_portfolio(), snapshot)

    assert client.calls == 1
    assert report.ok is False
    assert report.mismatches == (
        "venue has 1 open orders; PMS has no durable live open-order ledger yet",
    )


@pytest.mark.asyncio
async def test_polymarket_venue_reconciler_blocks_when_venue_cash_below_db_free() -> None:
    snapshot = VenueAccountSnapshot(
        balances={"PUSD": 999.0, "PUSD_ALLOWANCE": 1000.0},
        open_orders=(),
        positions=(),
    )
    reconciler = PolymarketVenueAccountReconciler()

    report = await reconciler.compare(_portfolio(), snapshot)

    assert report.ok is False
    assert report.mismatches == (
        "venue pUSD balance below PMS free cash: venue=999.00000000 DB=1000.00000000",
    )
