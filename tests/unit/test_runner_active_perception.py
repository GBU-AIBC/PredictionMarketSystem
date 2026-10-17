from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import SecretStr
import pms.runner as runner_module

from pms.config import (
    ControllerSettings,
    DatabaseSettings,
    DiscordSettings,
    PMSSettings,
    PolymarketSettings,
    RiskSettings,
    SensorSettings,
)
from pms.core.enums import RunMode
from pms.core.models import (
    MarketSignal,
    Portfolio,
    Position,
    ReconciliationReport,
    VenueAccountSnapshot,
    VenueCredentials,
)
from pms.market_selection.merge import StrategyMarketSet
from pms.runner import Runner, _default_active_strategy
from tests.support.default_strategy_seed import build_default_v1_strategy
from tests.support.live_paths import (
    make_live_preflight_artifact_path,
    make_live_report_paths,
    make_private_live_paths,
)


FIXTURE_PATH = Path("tests/fixtures/polymarket_7day_synthetic.jsonl")


@dataclass
class FakePool:
    close_calls: int = 0
    closed: bool = False

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class IdleSensor:
    def __aiter__(self) -> AsyncIterator[MarketSignal]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MarketSignal]:
        while True:
            await asyncio.sleep(60.0)
            yield _signal()


@dataclass
class RecordingMarketDataSensor:
    asset_ids: list[str] = field(default_factory=list)
    updates: list[list[str]] = field(default_factory=list)
    update_started: asyncio.Event = field(default_factory=asyncio.Event)
    update_release: asyncio.Event = field(default_factory=asyncio.Event)
    close_calls: int = 0

    def __aiter__(self) -> AsyncIterator[MarketSignal]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MarketSignal]:
        while True:
            await asyncio.sleep(60.0)
            yield _signal()

    async def update_subscription(self, asset_ids: list[str]) -> None:
        self.asset_ids = list(asset_ids)
        self.updates.append(list(asset_ids))
        self.update_started.set()
        await self.update_release.wait()

    async def aclose(self) -> None:
        self.close_calls += 1


@dataclass
class RecordingDiscoverySensor:
    events: list[str]
    on_poll_complete: Callable[[], Any] | None = None
    poll_count: int = 0
    close_calls: int = 0
    poll_started: asyncio.Event = field(default_factory=asyncio.Event)
    poll_released: asyncio.Event = field(default_factory=asyncio.Event)
    poll_complete_called: asyncio.Event = field(default_factory=asyncio.Event)
    first_poll_can_finish: asyncio.Event = field(default_factory=asyncio.Event)

    def __aiter__(self) -> AsyncIterator[MarketSignal]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MarketSignal]:
        self.events.append("discovery-started")
        self.poll_started.set()
        await self.first_poll_can_finish.wait()
        self.poll_count += 1
        self.events.append("first-poll-completed")
        if self.on_poll_complete is not None:
            await self.on_poll_complete()
        self.poll_complete_called.set()
        await self.poll_released.wait()
        while True:
            await asyncio.sleep(60.0)
            yield _signal()

    async def aclose(self) -> None:
        self.close_calls += 1


@dataclass
class RecordingSelector:
    events: list[str]
    returned_asset_ids: list[str]
    calls: int = 0
    call_started: asyncio.Event = field(default_factory=asyncio.Event)
    call_release: asyncio.Event = field(default_factory=asyncio.Event)
    last_user_subscription_asset_ids: frozenset[str] = frozenset()

    async def select(self) -> Any:
        self.calls += 1
        self.call_started.set()
        self.events.append("first-selection-complete")
        await self.call_release.wait()
        return type(
            "MergeResult",
            (),
            {"asset_ids": tuple(self.returned_asset_ids)},
        )()

    async def select_per_strategy(self) -> list[StrategyMarketSet]:
        return [
            StrategyMarketSet(
                strategy_id="default",
                strategy_version_id="default-v1",
                asset_ids=frozenset(self.returned_asset_ids),
            )
        ]


@dataclass
class RecordingSubscriptionController:
    sink: RecordingMarketDataSensor
    events: list[str]
    calls: int = 0
    updates: list[list[str]] = field(default_factory=list)
    call_started: asyncio.Event = field(default_factory=asyncio.Event)
    call_release: asyncio.Event = field(default_factory=asyncio.Event)

    async def update(self, asset_ids: list[str]) -> bool:
        self.calls += 1
        self.updates.append(list(asset_ids))
        self.call_started.set()
        self.events.append("data-sensor-subscribed")
        await self.call_release.wait()
        await self.sink.update_subscription(asset_ids)
        return True


@dataclass
class MatchingVenueReconciler:
    async def snapshot(self, credentials: VenueCredentials) -> VenueAccountSnapshot:
        del credentials
        return VenueAccountSnapshot(balances={"USDC": 10_000.0}, open_orders=(), positions=())

    async def compare(
        self,
        db_portfolio: Portfolio,
        venue_snapshot: VenueAccountSnapshot,
    ) -> ReconciliationReport:
        del db_portfolio, venue_snapshot
        return ReconciliationReport(ok=True, mismatches=())


def _settings(mode: RunMode) -> PMSSettings:
    approval_path: str | None = None
    emergency_audit_path = ".data/live-emergency-audit.jsonl"
    audit_path = ".data/first-order-audit.jsonl"
    if mode == RunMode.LIVE:
        approval_path, audit_path = make_private_live_paths(
            prefix="pms-runner-active-"
        )
        emergency_audit_path = str(
            Path(approval_path).parent / "live-emergency-audit.jsonl"
        )
        paper_report_path, rehearsal_report_path = make_live_report_paths(
            prefix="pms-runner-active-reports-"
        )
    else:
        paper_report_path = None
        rehearsal_report_path = None
    attested_at = datetime.now(tz=UTC) if mode == RunMode.LIVE else None
    settings = PMSSettings(
        mode=mode,
        secret_source="fly" if mode == RunMode.LIVE else None,
        live_trading_enabled=mode == RunMode.LIVE,
        api_token="live-api-token" if mode == RunMode.LIVE else None,
        live_exit_criteria_ratified_by=(
            "test-operator" if mode == RunMode.LIVE else None
        ),
        live_exit_criteria_ratified_at=attested_at,
        live_compliance_reviewed_by=(
            "test-compliance" if mode == RunMode.LIVE else None
        ),
        live_compliance_reviewed_at=attested_at,
        live_compliance_jurisdiction="US" if mode == RunMode.LIVE else None,
        live_paper_soak_report_path=paper_report_path,
        live_operator_rehearsal_report_path=rehearsal_report_path,
        live_emergency_audit_path=emergency_audit_path,
        live_first_order_audit_path=audit_path,
        auto_migrate_default_v2=False,
        database=DatabaseSettings(
            dsn="postgresql://localhost/pms_test_runner",
            pool_min_size=2,
            pool_max_size=10,
        ),
        risk=RiskSettings(
            max_position_per_market=1000.0,
            max_total_exposure=10_000.0,
            max_drawdown_pct=20.0,
            max_daily_loss_usdc=20.0,
            max_open_positions=5,
            max_exposure_per_risk_group=5_000.0,
            max_quantity_shares=500.0,
        ),
        controller=ControllerSettings(
            time_in_force="IOC" if mode == RunMode.LIVE else "GTC",
            quote_source="dual" if mode == RunMode.LIVE else "postgres_snapshot",
        ),
        discord=(
            DiscordSettings(
                webhook_url=SecretStr(
                    "https://discord.example/webhooks/runner-active/unit"
                ),
                alert_dir=str(Path(cast(str, approval_path)).parent / "discord-alerts"),
            )
            if mode == RunMode.LIVE
            else DiscordSettings()
        ),
        polymarket=(
            _live_polymarket_settings(approval_path)
            if mode == RunMode.LIVE
            else PolymarketSettings()
        ),
    )
    if mode == RunMode.LIVE:
        settings.live_preflight_artifact_path = make_live_preflight_artifact_path(
            prefix="pms-runner-active-preflight-",
            settings=settings,
        )
    return settings


def _live_polymarket_settings(approval_path: str | None = None) -> PolymarketSettings:
    if approval_path is None:
        approval_path, _ = make_private_live_paths(
            prefix="pms-runner-active-polymarket-"
        )
    return PolymarketSettings(
        private_key="private-key",
        api_key="api-key",
        api_secret="api-secret",
        api_passphrase="passphrase",
        signature_type=1,
        funder_address="0x1111111111111111111111111111111111111111",
        operator_approval_mode="every_order",
        first_live_order_approval_path=approval_path,
    )


class _DefaultV2Connection:
    async def fetchval(self, query: str) -> str:
        del query
        return "default-v1"


class _DefaultV2AcquireContext:
    async def __aenter__(self) -> _DefaultV2Connection:
        return _DefaultV2Connection()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> bool:
        del exc_type, exc, tb
        return False


class _DefaultV2Pool:
    def acquire(self) -> _DefaultV2AcquireContext:
        return _DefaultV2AcquireContext()


@dataclass
class _RecordingDefaultV2Registry:
    created_strategy: Any | None = None
    active_version: str | None = None
    populated_version: str | None = None

    async def get_by_id(self, strategy_id: str) -> Any:
        assert strategy_id == "default"
        return build_default_v1_strategy()

    async def create_version(self, strategy: Any) -> Any:
        self.created_strategy = strategy
        return SimpleNamespace(strategy_version_id="default-v2-test")

    async def set_active(self, strategy_id: str, strategy_version_id: str) -> None:
        assert strategy_id == "default"
        self.active_version = strategy_version_id

    async def populate_strategy_factors(
        self,
        strategy_id: str,
        strategy_version_id: str,
        factor_ids: object,
    ) -> None:
        del factor_ids
        assert strategy_id == "default"
        self.populated_version = strategy_version_id


def _runner(mode: RunMode, **kwargs: Any) -> Runner:
    if mode == RunMode.LIVE and "venue_account_reconciler" not in kwargs:
        kwargs["venue_account_reconciler"] = MatchingVenueReconciler()
    return Runner(
        config=_settings(mode),
        historical_data_path=FIXTURE_PATH,
        **kwargs,
    )


def _signal() -> MarketSignal:
    return MarketSignal(
        market_id="runner-active-perception",
        token_id="yes-token",
        venue="polymarket",
        title="Will active perception tests pass?",
        yes_price=0.42,
        volume_24h=1_000.0,
        resolves_at=datetime(2026, 4, 30, tzinfo=UTC),
        orderbook={"bids": [], "asks": []},
        external_signal={"fair_value": 0.55},
        fetched_at=datetime(2026, 4, 16, tzinfo=UTC),
        market_status="open",
    )


@pytest.fixture(autouse=True)
def _stub_factor_catalog_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop_ensure_factor_catalog(pool: object, *, factor_ids: object = None) -> None:
        del pool, factor_ids

    monkeypatch.setattr("pms.runner.ensure_factor_catalog", _noop_ensure_factor_catalog)


@pytest.fixture(autouse=True)
def _stub_factor_service(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoopFactorService:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        async def run(self) -> None:
            return None

    monkeypatch.setattr("pms.runner.FactorService", _NoopFactorService)


@pytest.fixture(autouse=True)
def _stub_live_venue_reconciler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pms.runner.PolymarketVenueAccountReconciler",
        MatchingVenueReconciler,
    )


@pytest.fixture(autouse=True)
def _stub_live_active_strategy_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _accept_active_strategy_preflight(
        settings: PMSSettings,
        registry: object,
    ) -> None:
        del settings, registry

    monkeypatch.setattr(
        "pms.runner.require_live_preflight_active_strategies_artifact",
        _accept_active_strategy_preflight,
    )


def _install_live_doubles(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: list[str],
    returned_asset_ids: list[str],
) -> tuple[
    FakePool,
    RecordingDiscoverySensor,
    RecordingMarketDataSensor,
    RecordingSelector,
    RecordingSubscriptionController,
]:
    fake_pool = FakePool()
    discovery = RecordingDiscoverySensor(events=events)
    market_data = RecordingMarketDataSensor()
    selector = RecordingSelector(events=events, returned_asset_ids=returned_asset_ids)
    subscription_controller = RecordingSubscriptionController(
        sink=market_data,
        events=events,
    )

    async def fake_create_pool(*, dsn: str, min_size: int, max_size: int) -> FakePool:
        del dsn, min_size, max_size
        events.append("pool-ready")
        return fake_pool

    class EmptyRegistry:
        def __init__(self, pool: object) -> None:
            del pool

        def register_change_callback(self, callback: Any) -> None:
            del callback

        def unregister_change_callback(self, callback: Any) -> None:
            del callback

        async def list_active_strategies(self) -> list[object]:
            return []

    monkeypatch.setattr("pms.runner.asyncpg.create_pool", fake_create_pool)
    monkeypatch.setattr("pms.runner.MarketDiscoverySensor", lambda **kwargs: discovery)
    monkeypatch.setattr("pms.runner.MarketDataSensor", lambda **kwargs: market_data)
    monkeypatch.setattr("pms.runner.PostgresStrategyRegistry", EmptyRegistry)
    monkeypatch.setattr("pms.runner.MarketSelector", lambda *args, **kwargs: selector)
    monkeypatch.setattr(
        "pms.runner.SensorSubscriptionController",
        lambda sink: subscription_controller,
    )
    return fake_pool, discovery, market_data, selector, subscription_controller


@pytest.mark.asyncio
async def test_runner_active_perception_boot_order_and_first_poll_subscription_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _, discovery, market_data, selector, subscription_controller = _install_live_doubles(
        monkeypatch,
        events=events,
        returned_asset_ids=["no-10d", "yes-10d"],
    )

    runner = _runner(RunMode.LIVE)
    original_build_sensors = runner._build_sensors

    def wrapped_build_sensors() -> tuple[Any, ...]:
        events.append("sensors-built")
        return original_build_sensors()

    monkeypatch.setattr(runner, "_build_sensors", wrapped_build_sensors)
    start_task = asyncio.create_task(runner.start())
    try:
        await asyncio.wait_for(discovery.poll_started.wait(), timeout=2.0)
        assert events == ["pool-ready", "sensors-built", "discovery-started"]
        assert discovery.on_poll_complete is not None
        assert selector.calls == 0
        assert subscription_controller.calls == 0
        assert market_data.updates == []

        discovery.first_poll_can_finish.set()
        await asyncio.wait_for(selector.call_started.wait(), timeout=2.0)
        assert selector.calls == 1
        assert subscription_controller.calls == 0
        assert market_data.updates == []

        selector.call_release.set()
        await asyncio.wait_for(subscription_controller.call_started.wait(), timeout=2.0)
        assert subscription_controller.calls == 1
        assert market_data.updates == []

        subscription_controller.call_release.set()
        await asyncio.wait_for(market_data.update_started.wait(), timeout=2.0)
        assert market_data.asset_ids == ["no-10d", "yes-10d"]
        market_data.update_release.set()
        await asyncio.wait_for(start_task, timeout=2.0)
    finally:
        if not start_task.done():
            start_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await start_task

    assert events == [
        "pool-ready",
        "sensors-built",
        "discovery-started",
        "first-poll-completed",
        "first-selection-complete",
        "data-sensor-subscribed",
    ]
    assert market_data.asset_ids == ["no-10d", "yes-10d"]

    discovery.poll_released.set()
    await runner.stop()


@pytest.mark.asyncio
async def test_runner_market_selection_freshness_has_safety_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Discovery:
        on_poll_complete: Callable[[], Any] | None = None

        def __aiter__(self) -> AsyncIterator[MarketSignal]:
            return self._iterate()

        async def _iterate(self) -> AsyncIterator[MarketSignal]:
            await asyncio.Event().wait()
            yield _signal()

    class Data:
        def __aiter__(self) -> AsyncIterator[MarketSignal]:
            return self._iterate()

        async def _iterate(self) -> AsyncIterator[MarketSignal]:
            await asyncio.Event().wait()
            yield _signal()

        async def update_subscription(self, asset_ids: list[str]) -> None:
            del asset_ids

    class Registry:
        def register_change_callback(self, callback: Any) -> None:
            del callback

        def unregister_change_callback(self, callback: Any) -> None:
            del callback

    captured: dict[str, Any] = {}

    class Selector:
        def __init__(self, store: Any, *args: Any) -> None:
            del args
            captured["store"] = store

    runner = Runner(
        config=PMSSettings(
            mode=RunMode.PAPER,
            auto_migrate_default_v2=False,
            sensor=SensorSettings(poll_interval_s=5.0),
        )
    )
    runner.bind_pg_pool(cast(Any, object()))
    runner._strategy_registry = cast(Any, Registry())  # noqa: SLF001
    monkeypatch.setattr(runner, "_register_strategy_change_callbacks", lambda: None)
    monkeypatch.setattr(runner_module, "MarketSelector", Selector)

    runner._wire_active_perception((Discovery(), Data()))  # noqa: SLF001
    assert captured["store"]._market_freshness_s == pytest.approx(300.0)  # noqa: SLF001

    task = runner._reselection_task  # noqa: SLF001
    if task is not None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_default_active_strategy_uses_operational_market_selection_horizon() -> None:
    strategy = _default_active_strategy(_settings(RunMode.PAPER))

    assert strategy.market_selection.resolution_time_max_horizon_days == 90


@pytest.mark.asyncio
async def test_default_v2_migration_widens_legacy_bootstrap_horizon() -> None:
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.PAPER,
            auto_migrate_default_v2=True,
            database=DatabaseSettings(
                dsn="postgresql://localhost/pms_test_runner",
                pool_min_size=2,
                pool_max_size=10,
            ),
        ),
        historical_data_path=FIXTURE_PATH,
    )
    registry = _RecordingDefaultV2Registry()
    runner._pg_pool = cast(Any, _DefaultV2Pool())  # noqa: SLF001
    runner._strategy_registry = cast(Any, registry)  # noqa: SLF001

    await runner._ensure_default_v2_version()  # noqa: SLF001

    assert registry.created_strategy is not None
    assert (
        registry.created_strategy.market_selection.resolution_time_max_horizon_days
        == 90
    )
    assert registry.active_version == "default-v2-test"
    assert registry.populated_version == "default-v2-test"


@pytest.mark.asyncio
async def test_runner_active_perception_backtest_skips_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool = FakePool()

    async def fake_create_pool(*, dsn: str, min_size: int, max_size: int) -> FakePool:
        del dsn, min_size, max_size
        return fake_pool

    monkeypatch.setattr("pms.runner.asyncpg.create_pool", fake_create_pool)

    runner = _runner(RunMode.BACKTEST, sensors=[IdleSensor()])
    await runner.start()

    assert runner._market_selector is None
    assert runner._subscription_controller is None
    assert runner._reselection_task is None

    await runner.stop()


@pytest.mark.asyncio
async def test_runner_active_perception_stop_cleans_up_references_and_sensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _, discovery, market_data, _, _ = _install_live_doubles(
        monkeypatch,
        events=events,
        returned_asset_ids=["no-10d", "yes-10d"],
    )

    runner = _runner(RunMode.LIVE)
    await runner.start()
    await runner.stop()

    assert runner._market_selector is None
    assert runner._subscription_controller is None
    assert runner._reselection_task is None
    assert discovery.close_calls == 1
    assert market_data.close_calls == 1


@pytest.mark.asyncio
async def test_runner_active_perception_reselection_is_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _, discovery, market_data, selector, subscription_controller = _install_live_doubles(
        monkeypatch,
        events=events,
        returned_asset_ids=["no-10d", "yes-10d"],
    )

    runner = _runner(RunMode.LIVE)
    await runner.start()

    assert runner._reselection_task is not None

    first = asyncio.create_task(runner._reselect())
    second = asyncio.create_task(runner._reselect())
    await asyncio.wait_for(selector.call_started.wait(), timeout=2.0)
    assert selector.calls == 1
    assert subscription_controller.calls == 0

    selector.call_release.set()
    await asyncio.wait_for(subscription_controller.call_started.wait(), timeout=2.0)
    assert subscription_controller.calls == 1

    subscription_controller.call_release.set()
    market_data.update_release.set()
    await asyncio.wait_for(first, timeout=2.0)
    await asyncio.wait_for(second, timeout=2.0)

    assert selector.calls == 2
    assert subscription_controller.calls == 2

    discovery.poll_released.set()
    await runner.stop()


@pytest.mark.asyncio
async def test_discovery_poll_refreshes_controller_runtime_scopes() -> None:
    events: list[list[str]] = []

    class IdleController:
        async def decide(
            self,
            signal: MarketSignal,
            portfolio: object | None = None,
        ) -> None:
            del signal, portfolio
            return None

    class ActiveRegistry:
        async def list_active_strategies(self) -> list[Any]:
            return [
                SimpleNamespace(
                    strategy_id="alpha",
                    strategy_version_id="alpha-v1",
                )
            ]

    @dataclass
    class ScopeSelector:
        async def select(self) -> Any:
            return type("MergeResult", (), {"asset_ids": ("fresh-token",)})()

        async def select_per_strategy(self) -> list[StrategyMarketSet]:
            return [
                StrategyMarketSet(
                    strategy_id="alpha",
                    strategy_version_id="alpha-v1",
                    asset_ids=frozenset({"fresh-token"}),
                )
            ]

    @dataclass
    class RecordingSubscriptionController:
        async def update(self, asset_ids: list[str]) -> bool:
            events.append(list(asset_ids))
            return True

    class ControllerFactory:
        def build_many(self, active_strategies: list[Any]) -> dict[str, IdleController]:
            return {
                strategy.strategy_id: IdleController()
                for strategy in active_strategies
            }

    runner = _runner(RunMode.LIVE)
    runner._strategy_registry = cast(Any, ActiveRegistry())  # noqa: SLF001
    runner._market_selector = cast(Any, ScopeSelector())  # noqa: SLF001
    runner._subscription_controller = cast(Any, RecordingSubscriptionController())  # noqa: SLF001
    runner._controller_factory = cast(Any, ControllerFactory())  # noqa: SLF001

    try:
        await runner._handle_discovery_poll_complete()  # noqa: SLF001

        assert runner._controller_runtimes["alpha"].asset_ids == frozenset(  # noqa: SLF001
            {"fresh-token"}
        )
        assert events == [["fresh-token"]]
        assert runner._reselection_requested.is_set()  # noqa: SLF001
    finally:
        runner._stop_event.set()  # noqa: SLF001
        tasks = tuple(runner._controller_pipeline_tasks.values())  # noqa: SLF001
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_reselection_caps_subscription_asset_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _, discovery, market_data, selector, subscription_controller = _install_live_doubles(
        monkeypatch,
        events=events,
        returned_asset_ids=["asset-a", "asset-b", "asset-c"],
    )
    approval_path, audit_path = make_private_live_paths(
        prefix="pms-runner-active-capped-"
    )
    paper_report_path, rehearsal_report_path = make_live_report_paths(
        prefix="pms-runner-active-capped-reports-"
    )
    attested_at = datetime.now(tz=UTC)

    settings = PMSSettings(
        mode=RunMode.LIVE,
        secret_source="fly",
        live_trading_enabled=True,
        api_token="live-api-token",
        live_exit_criteria_ratified_by="test-operator",
        live_exit_criteria_ratified_at=attested_at,
        live_compliance_reviewed_by="test-compliance",
        live_compliance_reviewed_at=attested_at,
        live_compliance_jurisdiction="US",
        live_paper_soak_report_path=paper_report_path,
        live_operator_rehearsal_report_path=rehearsal_report_path,
        live_emergency_audit_path=str(
            Path(approval_path).parent / "live-emergency-audit.jsonl"
        ),
        live_first_order_audit_path=audit_path,
        auto_migrate_default_v2=False,
        database=DatabaseSettings(
            dsn="postgresql://localhost/pms_test_runner",
            pool_min_size=2,
            pool_max_size=10,
        ),
        sensor=SensorSettings(max_subscription_asset_ids=2),
        discord=DiscordSettings(
            webhook_url=SecretStr(
                "https://discord.example/webhooks/runner-active/capped"
            ),
            alert_dir=str(Path(approval_path).parent / "discord-alerts"),
        ),
        risk=RiskSettings(
            max_drawdown_pct=20.0,
            max_daily_loss_usdc=20.0,
            max_open_positions=5,
            max_exposure_per_risk_group=250.0,
            max_quantity_shares=500.0,
        ),
        controller=ControllerSettings(time_in_force="IOC", quote_source="dual"),
        polymarket=_live_polymarket_settings(approval_path),
    )
    settings.live_preflight_artifact_path = make_live_preflight_artifact_path(
        prefix="pms-runner-active-capped-preflight-",
        settings=settings,
    )
    runner = Runner(
        config=settings,
        historical_data_path=FIXTURE_PATH,
    )
    await runner.start()

    task = asyncio.create_task(runner._reselect())  # noqa: SLF001
    await asyncio.wait_for(selector.call_started.wait(), timeout=2.0)
    selector.call_release.set()
    await asyncio.wait_for(subscription_controller.call_started.wait(), timeout=2.0)
    subscription_controller.call_release.set()
    await asyncio.wait_for(market_data.update_started.wait(), timeout=2.0)
    market_data.update_release.set()
    await asyncio.wait_for(task, timeout=2.0)

    assert subscription_controller.updates[-1] == ["asset-a", "asset-b"]
    assert market_data.asset_ids == ["asset-a", "asset-b"]

    discovery.poll_released.set()
    await runner.stop()


@pytest.mark.asyncio
async def test_refresh_subscription_updates_wait_for_reselection_lock() -> None:
    events: list[list[str]] = []

    @dataclass
    class SerializingSubscriptionController:
        active_calls: int = 0
        max_active_calls: int = 0
        first_call_started: asyncio.Event = field(default_factory=asyncio.Event)
        release_first_call: asyncio.Event = field(default_factory=asyncio.Event)

        async def update(self, asset_ids: list[str]) -> bool:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            events.append(list(asset_ids))
            if not self.first_call_started.is_set():
                self.first_call_started.set()
                await self.release_first_call.wait()
            self.active_calls -= 1
            return True

    @dataclass
    class StaticSelector:
        asset_ids: list[str]

        async def select(self) -> Any:
            return type("MergeResult", (), {"asset_ids": tuple(self.asset_ids)})()

        async def select_per_strategy(self) -> list[StrategyMarketSet]:
            return [
                StrategyMarketSet(
                    strategy_id="alpha",
                    strategy_version_id="alpha-v1",
                    asset_ids=frozenset(self.asset_ids),
                )
            ]

    runner = _runner(RunMode.LIVE)
    subscription_controller = SerializingSubscriptionController()
    runner._market_selector = cast(Any, StaticSelector(["strategy-token"]))  # noqa: SLF001
    runner._subscription_controller = cast(Any, subscription_controller)  # noqa: SLF001
    runner._controller_runtimes = {  # noqa: SLF001
        "alpha": cast(Any, SimpleNamespace(asset_ids=frozenset({"refresh-token"})))
    }

    reselect_task = asyncio.create_task(runner._reselect())  # noqa: SLF001
    await asyncio.wait_for(subscription_controller.first_call_started.wait(), timeout=2.0)

    refresh_task = asyncio.create_task(runner._refresh_subscription_assets_locked())  # noqa: SLF001
    await asyncio.sleep(0)
    assert subscription_controller.max_active_calls == 1
    assert events == [["strategy-token"]]

    subscription_controller.release_first_call.set()
    await asyncio.wait_for(reselect_task, timeout=2.0)
    await asyncio.wait_for(refresh_task, timeout=2.0)

    assert subscription_controller.max_active_calls == 1
    assert events == [["strategy-token"], ["refresh-token"]]


@pytest.mark.asyncio
async def test_reselect_does_not_hold_lock_during_selection_io() -> None:
    class Selector:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.lock_was_held = True
            self.last_user_subscription_asset_ids: frozenset[str] = frozenset()

        async def select(self) -> Any:
            self.lock_was_held = runner._reselection_lock.locked()  # noqa: SLF001
            self.started.set()
            await self.release.wait()
            return SimpleNamespace(asset_ids=["selected-token"])

    @dataclass
    class SubscriptionController:
        updates: list[list[str]] = field(default_factory=list)

        async def update(self, asset_ids: list[str]) -> bool:
            self.updates.append(list(asset_ids))
            return True

    runner = _runner(RunMode.PAPER)
    selector = Selector()
    subscription_controller = SubscriptionController()
    runner._market_selector = cast(Any, selector)  # noqa: SLF001
    runner._subscription_controller = cast(Any, subscription_controller)  # noqa: SLF001

    task = asyncio.create_task(runner._reselect())  # noqa: SLF001
    await asyncio.wait_for(selector.started.wait(), timeout=1.0)
    assert selector.lock_was_held is False
    selector.release.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert subscription_controller.updates == [["selected-token"]]


@pytest.mark.asyncio
async def test_refresh_subscription_caps_asset_ids() -> None:
    events: list[list[str]] = []
    approval_path, audit_path = make_private_live_paths(
        prefix="pms-runner-active-refresh-"
    )
    paper_report_path, rehearsal_report_path = make_live_report_paths(
        prefix="pms-runner-active-refresh-reports-"
    )
    attested_at = datetime.now(tz=UTC)

    @dataclass
    class RecordingSubscriptionController:
        async def update(self, asset_ids: list[str]) -> bool:
            events.append(list(asset_ids))
            return True

    settings = PMSSettings(
        mode=RunMode.LIVE,
        secret_source="fly",
        live_trading_enabled=True,
        api_token="live-api-token",
        live_exit_criteria_ratified_by="test-operator",
        live_exit_criteria_ratified_at=attested_at,
        live_compliance_reviewed_by="test-compliance",
        live_compliance_reviewed_at=attested_at,
        live_compliance_jurisdiction="US",
        live_paper_soak_report_path=paper_report_path,
        live_operator_rehearsal_report_path=rehearsal_report_path,
        live_emergency_audit_path=str(
            Path(approval_path).parent / "live-emergency-audit.jsonl"
        ),
        live_first_order_audit_path=audit_path,
        auto_migrate_default_v2=False,
        database=DatabaseSettings(
            dsn="postgresql://localhost/pms_test_runner",
            pool_min_size=2,
            pool_max_size=10,
        ),
        sensor=SensorSettings(max_subscription_asset_ids=2),
        discord=DiscordSettings(
            webhook_url=SecretStr(
                "https://discord.example/webhooks/runner-active/refresh"
            ),
            alert_dir=str(Path(approval_path).parent / "discord-alerts"),
        ),
        risk=RiskSettings(
            max_drawdown_pct=20.0,
            max_daily_loss_usdc=20.0,
            max_open_positions=5,
            max_exposure_per_risk_group=250.0,
            max_quantity_shares=500.0,
        ),
        controller=ControllerSettings(time_in_force="IOC", quote_source="dual"),
        polymarket=_live_polymarket_settings(approval_path),
    )
    settings.live_preflight_artifact_path = make_live_preflight_artifact_path(
        prefix="pms-runner-active-refresh-preflight-",
        settings=settings,
    )
    runner = Runner(
        config=settings,
        historical_data_path=FIXTURE_PATH,
    )
    runner._subscription_controller = cast(Any, RecordingSubscriptionController())  # noqa: SLF001
    runner._controller_runtimes = {  # noqa: SLF001
        "alpha": cast(
            Any,
            SimpleNamespace(asset_ids=frozenset({"asset-c", "asset-a", "asset-b"})),
        )
    }

    await runner._refresh_subscription_assets_locked()  # noqa: SLF001

    assert events == [["asset-a", "asset-b"]]


@pytest.mark.asyncio
async def test_refresh_subscription_protects_user_pins_under_cap() -> None:
    events: list[list[str]] = []

    @dataclass
    class RecordingSubscriptionController:
        async def update(self, asset_ids: list[str]) -> bool:
            events.append(list(asset_ids))
            return True

    runner = Runner(
        config=PMSSettings(
            mode=RunMode.PAPER,
            auto_migrate_default_v2=False,
            sensor=SensorSettings(max_subscription_asset_ids=2),
        ),
        historical_data_path=FIXTURE_PATH,
    )
    runner._subscription_controller = cast(Any, RecordingSubscriptionController())  # noqa: SLF001
    runner._market_selector = cast(  # noqa: SLF001
        Any,
        SimpleNamespace(last_user_subscription_asset_ids=frozenset({"user-pin"})),
    )
    runner._controller_runtimes = {  # noqa: SLF001
        "alpha": cast(
            Any,
            SimpleNamespace(asset_ids=frozenset({"strategy-a", "strategy-b", "user-pin"})),
        )
    }

    await runner._refresh_subscription_assets_locked()  # noqa: SLF001

    assert events == [["user-pin", "strategy-a"]]


@pytest.mark.asyncio
async def test_refresh_subscription_protects_open_position_tokens_under_cap() -> None:
    events: list[list[str]] = []

    @dataclass
    class RecordingSubscriptionController:
        async def update(self, asset_ids: list[str]) -> bool:
            events.append(list(asset_ids))
            return True

    runner = Runner(
        config=PMSSettings(
            mode=RunMode.LIVE,
            secret_source="fly",
            live_trading_enabled=True,
            api_token="live-api-token",
            auto_migrate_default_v2=False,
            database=DatabaseSettings(
                dsn="postgresql://localhost/pms_test_runner",
                pool_min_size=2,
                pool_max_size=10,
            ),
            sensor=SensorSettings(max_subscription_asset_ids=2),
            controller=ControllerSettings(time_in_force="IOC"),
            polymarket=_live_polymarket_settings(),
        ),
        historical_data_path=FIXTURE_PATH,
        portfolio=Portfolio(
            total_usdc=1000.0,
            free_usdc=995.0,
            locked_usdc=5.0,
            open_positions=[
                Position(
                    market_id="held-market",
                    token_id="held-token",
                    venue="polymarket",
                    side="BUY",
                    shares_held=10.0,
                    avg_entry_price=0.5,
                    unrealized_pnl=0.0,
                    locked_usdc=5.0,
                    strategy_id="alpha",
                    strategy_version_id="alpha-v1",
                )
            ],
        ),
    )
    runner._subscription_controller = cast(Any, RecordingSubscriptionController())  # noqa: SLF001
    runner._controller_runtimes = {  # noqa: SLF001
        "alpha": cast(
            Any,
            SimpleNamespace(asset_ids=frozenset({"asset-c", "asset-a", "asset-b"})),
        )
    }

    await runner._refresh_subscription_assets_locked()  # noqa: SLF001

    assert events == [["held-token", "asset-a"]]


@pytest.mark.asyncio
async def test_event_triggered_reselection_failure_does_not_kill_loop(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[str] = []
    _, discovery, _, _, _ = _install_live_doubles(
        monkeypatch,
        events=events,
        returned_asset_ids=["no-10d", "yes-10d"],
    )

    runner = _runner(RunMode.LIVE)
    await runner.start()
    try:
        reselection_task = runner._reselection_task
        assert reselection_task is not None
        assert not reselection_task.done()

        raising_selector_called = asyncio.Event()

        class RaisingSelector:
            calls = 0

            async def select(self) -> Any:
                RaisingSelector.calls += 1
                raising_selector_called.set()
                raise RuntimeError("transient postgres error")

            async def select_per_strategy(self) -> list[StrategyMarketSet]:
                raise RuntimeError("transient postgres error")

        runner._market_selector = RaisingSelector()

        caplog.set_level(logging.WARNING, logger="pms.runner")
        await runner._request_reselection()
        await asyncio.wait_for(raising_selector_called.wait(), timeout=2.0)
        for _ in range(10):
            await asyncio.sleep(0)

        assert not reselection_task.done(), (
            "reselection task died when event-triggered _reselect raised; "
            f"task state: done={reselection_task.done()}"
        )
        assert RaisingSelector.calls == 1
        assert "periodic reselection failed" in caplog.text
    finally:
        discovery.poll_released.set()
        await runner.stop()
