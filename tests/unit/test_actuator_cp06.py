from __future__ import annotations

import asyncio
import os
import json
import inspect
import importlib.machinery
import importlib.util
import logging
import sys
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest
from pydantic import SecretStr

from pms.actuator import executor
from pms.actuator.adapters import backtest
from pms.actuator.adapters.backtest import BacktestActuator
from pms.actuator.adapters.paper import PaperActuator
from pms.actuator.adapters.polymarket import (
    FileFirstLiveOrderGate,
    LiveOrderPreview,
    LivePreSubmitQuote,
    MissingLiveQuoteProvider,
    OperatorApprovalRequiredError,
    PolymarketActuator,
    PolymarketOrderResult,
    PolymarketOrderRequest,
    PolymarketVenueAccountReconciler,
    PolymarketSDKClient,
    PolymarketSubmissionUnknownError,
)
from pms.actuator.feedback import ActuatorFeedback
from pms.actuator.risk import InsufficientLiquidityError, RiskManager
from pms.config import (
    ControllerSettings,
    DiscordSettings,
    PMSSettings,
    PolymarketSettings,
    RiskSettings,
)
from pms.core.enums import FeedbackSource, FeedbackTarget, OrderStatus, RunMode, Side, TimeInForce
from pms.core.models import (
    LiveTradingDisabledError,
    MarketSignal,
    OrderState,
    Portfolio,
    TradeDecision,
    VenueAccountSnapshot,
)
from pms.live_preflight_artifact import (
    live_preflight_readiness_reports_fingerprint,
    live_preflight_settings_fingerprint,
    validate_live_strategy_artifacts_for_submission,
)
from pms.storage.dedup_store import InMemoryDedupStore
from pms.storage.feedback_store import FeedbackStore
from tests.support.fake_stores import InMemoryFeedbackStore
from tests.support.live_paths import make_live_report_paths


def _decision(
    *,
    decision_id: str = "d-cp06",
    market_id: str = "m-cp06",
    token_id: str | None = None,
    side: Literal["BUY", "SELL"] = Side.BUY.value,
    notional_usdc: float = 10.0,
    limit_price: float = 0.4,
    action: Literal["BUY", "SELL"] | None = None,
    outcome: Literal["YES", "NO"] = "YES",
) -> TradeDecision:
    return TradeDecision(
        decision_id=decision_id,
        market_id=market_id,
        token_id=token_id or ("t-yes" if outcome == "YES" else "t-no"),
        venue="polymarket",
        side=side,
        notional_usdc=notional_usdc,
        order_type="limit",
        max_slippage_bps=50,
        stop_conditions=["unit-test"],
        prob_estimate=0.6,
        expected_edge=0.2,
        time_in_force=TimeInForce.IOC,
        opportunity_id=f"op-{decision_id}",
        strategy_id="default",
        strategy_version_id="default-v1",
        action=action or side,
        limit_price=limit_price,
        outcome=outcome,
    )


def _portfolio(
    *,
    locked_usdc: float = 0.0,
    max_drawdown_pct: float | None = None,
) -> Portfolio:
    return Portfolio(
        total_usdc=1000.0,
        free_usdc=1000.0 - locked_usdc,
        locked_usdc=locked_usdc,
        open_positions=[],
        max_drawdown_pct=max_drawdown_pct,
    )


def _order_state(
    decision: TradeDecision,
    *,
    status: str = OrderStatus.INVALID.value,
    raw_status: str = "rejected",
    fill_price: float | None = None,
    filled_notional_usdc: float = 0.0,
) -> OrderState:
    now = datetime(2026, 4, 14, tzinfo=UTC)
    filled_quantity = 0.0
    if fill_price is not None and fill_price != 0.0:
        filled_quantity = filled_notional_usdc / fill_price
    return OrderState(
        order_id=f"order-{decision.decision_id}",
        decision_id=decision.decision_id,
        status=status,
        market_id=decision.market_id,
        token_id=decision.token_id,
        venue=decision.venue,
        requested_notional_usdc=decision.notional_usdc,
        filled_notional_usdc=filled_notional_usdc,
        remaining_notional_usdc=decision.notional_usdc - filled_notional_usdc,
        fill_price=fill_price,
        submitted_at=now,
        last_updated_at=now,
        raw_status=raw_status,
        strategy_id=decision.strategy_id,
        strategy_version_id=decision.strategy_version_id,
        filled_quantity=filled_quantity,
        action=decision.action,
        outcome=decision.outcome,
        time_in_force=decision.time_in_force.value,
        intent_key=decision.intent_key,
        risk_group_id=decision.risk_group_id,
    )


@dataclass
class RecordingDedupStore:
    acquire_allowed: bool = True
    release_error: Exception | None = None
    acquire_calls: list[str] = field(default_factory=list)
    release_calls: list[tuple[str, str]] = field(default_factory=list)

    async def acquire(self, decision: TradeDecision) -> bool:
        self.acquire_calls.append(decision.decision_id)
        return self.acquire_allowed

    async def release(self, decision_id: str, outcome: str) -> None:
        self.release_calls.append((decision_id, outcome))
        if self.release_error is not None:
            raise self.release_error

    async def retention_scan(self, older_than: timedelta) -> int:
        del older_than
        return 0


@dataclass
class StaticAdapter:
    state: OrderState
    calls: int = 0

    async def execute(
        self,
        decision: TradeDecision,
        portfolio: Portfolio | None = None,
    ) -> OrderState:
        del decision, portfolio
        self.calls += 1
        return self.state


@dataclass
class FailingAdapter:
    error: RuntimeError
    calls: int = 0

    async def execute(
        self,
        decision: TradeDecision,
        portfolio: Portfolio | None = None,
    ) -> OrderState:
        del decision, portfolio
        self.calls += 1
        raise self.error


def test_backtest_adapter_documents_license_decision_before_internal_replay() -> None:
    source = inspect.getsource(backtest)

    assert "prediction-market-backtesting" in source
    assert "LGPL-3.0-or-later" in source
    assert "internal replay" in source


@pytest.mark.asyncio
async def test_polymarket_reconciler_rejects_missing_cash_balance() -> None:
    report = await PolymarketVenueAccountReconciler().compare(
        _portfolio(),
        VenueAccountSnapshot(balances={}, open_orders=(), positions=()),
    )

    assert report.ok is False
    assert report.mismatches == (
        "venue pUSD balance missing; cannot prove LIVE cash budget",
    )


@pytest.mark.asyncio
async def test_polymarket_reconciler_accepts_pusd_balance_and_allowance() -> None:
    report = await PolymarketVenueAccountReconciler().compare(
        _portfolio(),
        VenueAccountSnapshot(
            balances={"PUSD": 1000.0, "PUSD_ALLOWANCE": 1000.0},
            open_orders=(),
            positions=(),
        ),
    )

    assert report.ok is True
    assert report.mismatches == ()


@pytest.mark.asyncio
async def test_polymarket_reconciler_rejects_missing_pusd_allowance() -> None:
    report = await PolymarketVenueAccountReconciler().compare(
        _portfolio(),
        VenueAccountSnapshot(
            balances={"PUSD": 1000.0},
            open_orders=(),
            positions=(),
        ),
    )

    assert report.ok is False
    assert report.mismatches == (
        "venue pUSD allowance missing; cannot prove LIVE buy capacity",
    )


@pytest.mark.asyncio
async def test_polymarket_reconciler_rejects_insufficient_pusd_allowance() -> None:
    report = await PolymarketVenueAccountReconciler().compare(
        _portfolio(),
        VenueAccountSnapshot(
            balances={"PUSD": 1000.0, "PUSD_ALLOWANCE": 5.0},
            open_orders=(),
            positions=(),
        ),
    )

    assert report.ok is False
    assert report.mismatches == (
        "venue pUSD allowance below PMS free cash: venue=5.00000000 DB=1000.00000000",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_balance", [float("nan"), float("inf"), float("-inf")])
async def test_polymarket_reconciler_rejects_non_finite_cash_balance(
    bad_balance: float,
) -> None:
    report = await PolymarketVenueAccountReconciler().compare(
        _portfolio(),
        VenueAccountSnapshot(
            balances={"USDC": bad_balance},
            open_orders=(),
            positions=(),
        ),
    )

    assert report.ok is False
    assert report.mismatches == (
        "venue pUSD balance invalid; cannot prove LIVE cash budget",
    )


@pytest.mark.asyncio
async def test_polymarket_sdk_account_snapshot_parses_wrapped_collections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeBalanceAllowanceParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs: dict[str, object] = kwargs

    class FakeAssetType:
        COLLATERAL = "COLLATERAL"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def get_orders(self) -> dict[str, object]:
            return {
                "orders": [
                    {
                        "order_id": "pm-open-1",
                        "market_id": "m-open",
                        "token_id": "t-open",
                        "remaining": "7.50",
                        "price": "0.25",
                        "status": "open",
                    }
                ]
            }

        def get_positions(self, user: str) -> dict[str, object]:
            assert user == "0x1111111111111111111111111111111111111111"
            return {
                "positions": [
                    {
                        "market_id": "m-position",
                        "token_id": "t-position",
                        "shares": "12.5",
                        "avg_entry_price": "0.40",
                    }
                ]
            }

        def update_balance_allowance(self, **kwargs: object) -> None:
            del kwargs

        def get_balance_allowance(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {"balance": "1000.00", "allowance": "500.00"}

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        AssetType=FakeAssetType,
        BalanceAllowanceParams=FakeBalanceAllowanceParams,
        ClobClient=FakeClobClient,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    snapshot = await PolymarketSDKClient().read_account_snapshot(
        _live_settings().polymarket.credentials(),
    )

    assert snapshot.balances == {"PUSD": 1000.0, "PUSD_ALLOWANCE": 500.0}
    assert len(snapshot.open_orders) == 1
    assert snapshot.open_orders[0].order_id == "pm-open-1"
    assert snapshot.open_orders[0].remaining_notional_usdc == pytest.approx(7.5)
    assert len(snapshot.positions) == 1
    assert snapshot.positions[0].market_id == "m-position"
    assert snapshot.positions[0].locked_usdc == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_polymarket_sdk_account_snapshot_syncs_balance_allowance_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeBalanceAllowanceParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs: dict[str, object] = kwargs

    class FakeAssetType:
        COLLATERAL = "COLLATERAL"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def get_orders(self) -> list[object]:
            return []

        def get_positions(self, user: str) -> list[object]:
            assert user == "0x1111111111111111111111111111111111111111"
            return []

        def update_balance_allowance(self, **kwargs: object) -> None:
            calls.append(("update", kwargs["params"]))

        def get_balance_allowance(self, **kwargs: object) -> dict[str, object]:
            calls.append(("get", kwargs["params"]))
            return {"balance": "1000.00", "allowance": "1000.00"}

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        AssetType=FakeAssetType,
        BalanceAllowanceParams=FakeBalanceAllowanceParams,
        ClobClient=FakeClobClient,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    snapshot = await PolymarketSDKClient().read_account_snapshot(
        _live_settings().polymarket.credentials(),
    )

    assert snapshot.balances == {"PUSD": 1000.0, "PUSD_ALLOWANCE": 1000.0}
    assert [name for name, _ in calls] == ["update", "get"]
    assert [cast(FakeBalanceAllowanceParams, param).kwargs for _, param in calls] == [
        {"asset_type": "COLLATERAL"},
        {"asset_type": "COLLATERAL"},
    ]


@pytest.mark.asyncio
async def test_polymarket_sdk_account_snapshot_uses_camelcase_balance_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeBalanceAllowanceParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs: dict[str, object] = kwargs

    class FakeAssetType:
        COLLATERAL = "COLLATERAL"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def get_orders(self) -> list[object]:
            return []

        def get_positions(self, user: str) -> list[object]:
            assert user == "0x1111111111111111111111111111111111111111"
            return []

        def updateBalanceAllowance(self, **kwargs: object) -> None:  # noqa: N802
            params = cast(FakeBalanceAllowanceParams, kwargs["params"])
            assert params.kwargs == {"asset_type": "COLLATERAL"}
            calls.append("update")

        def get_balance_allowance(self, **kwargs: object) -> dict[str, object]:
            params = cast(FakeBalanceAllowanceParams, kwargs["params"])
            assert params.kwargs == {"asset_type": "COLLATERAL"}
            calls.append("get")
            return {"balance": "1000.00", "allowance": "1000.00"}

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        AssetType=FakeAssetType,
        BalanceAllowanceParams=FakeBalanceAllowanceParams,
        ClobClient=FakeClobClient,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    snapshot = await PolymarketSDKClient().read_account_snapshot(
        _live_settings().polymarket.credentials(),
    )

    assert snapshot.balances == {"PUSD": 1000.0, "PUSD_ALLOWANCE": 1000.0}
    assert calls == ["update", "get"]


@pytest.mark.asyncio
async def test_polymarket_sdk_account_snapshot_fails_when_balance_sync_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_attempted = False

    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeBalanceAllowanceParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs: dict[str, object] = kwargs

    class FakeAssetType:
        COLLATERAL = "COLLATERAL"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def get_orders(self) -> list[object]:
            return []

        def get_positions(self, user: str) -> list[object]:
            assert user == "0x1111111111111111111111111111111111111111"
            return []

        def update_balance_allowance(self, **kwargs: object) -> None:
            nonlocal sync_attempted
            params = cast(FakeBalanceAllowanceParams, kwargs["params"])
            assert params.kwargs == {"asset_type": "COLLATERAL"}
            sync_attempted = True
            raise RuntimeError("cache sync rejected")

        def get_balance_allowance(self, **kwargs: object) -> dict[str, object]:
            raise AssertionError("snapshot must not read stale cached balances")

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        AssetType=FakeAssetType,
        BalanceAllowanceParams=FakeBalanceAllowanceParams,
        ClobClient=FakeClobClient,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    with pytest.raises(LiveTradingDisabledError, match="account snapshot failed"):
        await PolymarketSDKClient().read_account_snapshot(
            _live_settings().polymarket.credentials(),
        )
    assert sync_attempted is True


@pytest.mark.asyncio
async def test_polymarket_sdk_account_snapshot_passes_deposit_wallet_signature_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_kwargs: dict[str, object] = {}
    balance_param_calls: list[dict[str, object]] = []

    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeSignatureTypeV2:
        POLY_1271 = "SDK_POLY_1271"

    class FakeBalanceAllowanceParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs: dict[str, object] = kwargs

    class FakeAssetType:
        COLLATERAL = "COLLATERAL"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            init_kwargs.update(kwargs)

        def get_orders(self) -> list[object]:
            return []

        def get_positions(self, user: str) -> list[object]:
            assert user == "0x1111111111111111111111111111111111111111"
            return []

        def update_balance_allowance(self, **kwargs: object) -> None:
            params = cast(FakeBalanceAllowanceParams, kwargs["params"])
            balance_param_calls.append(params.kwargs)

        def get_balance_allowance(self, **kwargs: object) -> dict[str, object]:
            params = cast(FakeBalanceAllowanceParams, kwargs["params"])
            balance_param_calls.append(params.kwargs)
            return {"balance": "1000.00", "allowance": "1000.00"}

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        AssetType=FakeAssetType,
        BalanceAllowanceParams=FakeBalanceAllowanceParams,
        ClobClient=FakeClobClient,
        SignatureTypeV2=FakeSignatureTypeV2,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    snapshot = await PolymarketSDKClient().read_account_snapshot(
        _live_settings(signature_type=3).polymarket.credentials(),
    )

    assert init_kwargs["signature_type"] == "SDK_POLY_1271"
    assert snapshot.balances == {"PUSD": 1000.0, "PUSD_ALLOWANCE": 1000.0}
    assert balance_param_calls == [
        {"asset_type": "COLLATERAL", "signature_type": "SDK_POLY_1271"},
        {"asset_type": "COLLATERAL", "signature_type": "SDK_POLY_1271"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("order_id", [None, "", "   ", "__FILL_IN_ORDER_ID__"])
async def test_polymarket_sdk_account_snapshot_rejects_open_order_without_concrete_id(
    monkeypatch: pytest.MonkeyPatch,
    order_id: object,
) -> None:
    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeBalanceAllowanceParams:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeAssetType:
        COLLATERAL = "COLLATERAL"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def get_orders(self) -> list[dict[str, object]]:
            order: dict[str, object] = {
                "market_id": "m-open",
                "token_id": "t-open",
                "remaining": "7.50",
                "price": "0.25",
                "status": "open",
            }
            if order_id is not None:
                order["order_id"] = order_id
            return [order]

        def get_positions(self, user: str) -> list[object]:
            assert user == "0x1111111111111111111111111111111111111111"
            return []

        def get_balance_allowance(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {"balance": "1000.00"}

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        AssetType=FakeAssetType,
        BalanceAllowanceParams=FakeBalanceAllowanceParams,
        ClobClient=FakeClobClient,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    with pytest.raises(LiveTradingDisabledError, match="account snapshot failed"):
        await PolymarketSDKClient().read_account_snapshot(
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "order_payload",
    [
        {"order_id": "pm-open-1", "market_id": "m-open", "token_id": "t-open"},
        {
            "order_id": "pm-open-1",
            "market_id": "m-open",
            "token_id": "t-open",
            "remaining": "",
        },
        {
            "order_id": "pm-open-1",
            "market_id": "m-open",
            "token_id": "t-open",
            "remaining": "nan",
        },
        {
            "order_id": "pm-open-1",
            "market_id": "m-open",
            "token_id": "t-open",
            "remaining": -1.0,
        },
    ],
)
async def test_polymarket_sdk_account_snapshot_rejects_open_order_with_bad_remaining(
    monkeypatch: pytest.MonkeyPatch,
    order_payload: dict[str, object],
) -> None:
    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeBalanceAllowanceParams:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeAssetType:
        COLLATERAL = "COLLATERAL"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def get_orders(self) -> list[dict[str, object]]:
            return [order_payload]

        def get_positions(self, user: str) -> list[object]:
            assert user == "0x1111111111111111111111111111111111111111"
            return []

        def get_balance_allowance(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {"balance": "1000.00"}

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        AssetType=FakeAssetType,
        BalanceAllowanceParams=FakeBalanceAllowanceParams,
        ClobClient=FakeClobClient,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    with pytest.raises(LiveTradingDisabledError, match="account snapshot failed"):
        await PolymarketSDKClient().read_account_snapshot(
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "position_payload",
    [
        {"market_id": "m-position", "token_id": "t-position"},
        {"market_id": "m-position", "token_id": "t-position", "shares": ""},
        {"market_id": "m-position", "token_id": "t-position", "shares": "nan"},
        {"market_id": "m-position", "token_id": "t-position", "shares": -1.0},
    ],
)
async def test_polymarket_sdk_account_snapshot_rejects_position_with_bad_shares(
    monkeypatch: pytest.MonkeyPatch,
    position_payload: dict[str, object],
) -> None:
    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeBalanceAllowanceParams:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeAssetType:
        COLLATERAL = "COLLATERAL"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def get_orders(self) -> list[object]:
            return []

        def get_positions(self, user: str) -> list[dict[str, object]]:
            assert user == "0x1111111111111111111111111111111111111111"
            return [position_payload]

        def get_balance_allowance(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {"balance": "1000.00"}

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        AssetType=FakeAssetType,
        BalanceAllowanceParams=FakeBalanceAllowanceParams,
        ClobClient=FakeClobClient,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    with pytest.raises(LiveTradingDisabledError, match="account snapshot failed"):
        await PolymarketSDKClient().read_account_snapshot(
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "position_payload",
    [
        {"market_id": "m-position", "token_id": "t-position", "shares": 12.5},
        {
            "market_id": "m-position",
            "token_id": "t-position",
            "shares": 12.5,
            "avg_entry_price": "",
        },
        {
            "market_id": "m-position",
            "token_id": "t-position",
            "shares": 12.5,
            "avg_entry_price": "nan",
        },
        {
            "market_id": "m-position",
            "token_id": "t-position",
            "shares": 12.5,
            "avg_entry_price": -0.01,
        },
        {
            "market_id": "m-position",
            "token_id": "t-position",
            "shares": 12.5,
            "avg_entry_price": 1.01,
        },
    ],
)
async def test_polymarket_sdk_account_snapshot_rejects_position_with_bad_price(
    monkeypatch: pytest.MonkeyPatch,
    position_payload: dict[str, object],
) -> None:
    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeBalanceAllowanceParams:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeAssetType:
        COLLATERAL = "COLLATERAL"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def get_orders(self) -> list[object]:
            return []

        def get_positions(self, user: str) -> list[dict[str, object]]:
            assert user == "0x1111111111111111111111111111111111111111"
            return [position_payload]

        def get_balance_allowance(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {"balance": "1000.00"}

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        AssetType=FakeAssetType,
        BalanceAllowanceParams=FakeBalanceAllowanceParams,
        ClobClient=FakeClobClient,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    with pytest.raises(LiveTradingDisabledError, match="account snapshot failed"):
        await PolymarketSDKClient().read_account_snapshot(
            _live_settings().polymarket.credentials(),
        )


def test_risk_manager_position_breakpoint_exact_limit_and_plus_one() -> None:
    manager = RiskManager(
        RiskSettings(max_position_per_market=100.0, max_total_exposure=1000.0)
    )

    assert manager.check(_decision(notional_usdc=100.0), _portfolio()).approved is True
    rejected = manager.check(_decision(notional_usdc=101.0), _portfolio())

    assert rejected.approved is False
    assert rejected.reason == "max_position_per_market"


def test_risk_manager_rejects_total_exposure_and_drawdown() -> None:
    manager = RiskManager(
        RiskSettings(
            max_position_per_market=1000.0,
            max_total_exposure=1000.0,
            max_drawdown_pct=0.2,
        )
    )

    assert (
        manager.check(
            _decision(notional_usdc=501.0),
            _portfolio(locked_usdc=500.0),
        ).reason
        == "max_total_exposure"
    )
    assert (
        manager.check(_decision(notional_usdc=10.0), _portfolio(max_drawdown_pct=0.21)).reason
        == "drawdown_circuit_breaker"
    )


def test_risk_manager_reserves_risk_group_exposure_for_live_open_orders() -> None:
    manager = RiskManager(
        RiskSettings(
            max_position_per_market=100.0,
            max_total_exposure=1000.0,
            max_exposure_per_risk_group=10.0,
        )
    )
    decision = replace(
        _decision(
            decision_id="risk-open-order",
            notional_usdc=6.0,
        ),
        risk_group_id="event:risk-group",
    )
    order = _order_state(
        decision,
        status=OrderStatus.LIVE.value,
        raw_status="open",
        filled_notional_usdc=0.0,
    )

    manager.record_open_order_state(order)

    rejected = manager.check(
        replace(
            _decision(
                decision_id="risk-next-order",
                notional_usdc=5.0,
            ),
            risk_group_id="event:risk-group",
        ),
        _portfolio(),
    )
    assert rejected.approved is False
    assert rejected.reason == "max_exposure_per_risk_group"

    manager.record_order_filled(order.order_id)

    approved = manager.check(
        replace(
            _decision(
                decision_id="risk-after-fill-order",
                notional_usdc=5.0,
            ),
            risk_group_id="event:risk-group",
        ),
        _portfolio(),
    )
    assert approved.approved is True


def test_risk_manager_reserves_total_exposure_for_live_open_orders() -> None:
    manager = RiskManager(
        RiskSettings(max_position_per_market=100.0, max_total_exposure=10.0)
    )
    first_decision = _decision(
        decision_id="total-open-order",
        market_id="m-total-a",
        notional_usdc=6.0,
    )
    first_order = _order_state(
        first_decision,
        status=OrderStatus.LIVE.value,
        raw_status="open",
        filled_notional_usdc=0.0,
    )

    manager.record_open_order_state(first_order)

    rejected = manager.check(
        _decision(
            decision_id="total-next-order",
            market_id="m-total-b",
            notional_usdc=5.0,
        ),
        _portfolio(),
    )
    assert rejected.approved is False
    assert rejected.reason == "max_total_exposure"


def test_risk_manager_reserves_market_exposure_for_live_open_orders() -> None:
    manager = RiskManager(
        RiskSettings(max_position_per_market=10.0, max_total_exposure=1000.0)
    )
    first_decision = _decision(
        decision_id="market-open-order",
        market_id="m-market-cap",
        notional_usdc=6.0,
    )
    first_order = _order_state(
        first_decision,
        status=OrderStatus.LIVE.value,
        raw_status="open",
        filled_notional_usdc=0.0,
    )

    manager.record_open_order_state(first_order)

    rejected = manager.check(
        _decision(
            decision_id="market-next-order",
            market_id="m-market-cap",
            notional_usdc=5.0,
        ),
        _portfolio(),
    )
    assert rejected.approved is False
    assert rejected.reason == "max_position_per_market"


def test_risk_manager_reserves_free_cash_for_live_open_orders() -> None:
    manager = RiskManager(
        RiskSettings(max_position_per_market=100.0, max_total_exposure=1000.0)
    )
    first_decision = _decision(
        decision_id="cash-open-order",
        market_id="m-cash-a",
        notional_usdc=6.0,
    )
    first_order = _order_state(
        first_decision,
        status=OrderStatus.LIVE.value,
        raw_status="open",
        filled_notional_usdc=0.0,
    )

    manager.record_open_order_state(first_order)

    rejected = manager.check(
        _decision(
            decision_id="cash-next-order",
            market_id="m-cash-b",
            notional_usdc=5.0,
        ),
        Portfolio(
            total_usdc=10.0,
            free_usdc=10.0,
            locked_usdc=0.0,
            open_positions=[],
        ),
    )
    assert rejected.approved is False
    assert rejected.reason == "insufficient_free_usdc"


def test_risk_manager_does_not_reserve_exposure_for_live_open_sell_orders() -> None:
    manager = RiskManager(
        RiskSettings(max_position_per_market=10.0, max_total_exposure=10.0)
    )
    exit_decision = _decision(
        decision_id="sell-open-order",
        market_id="m-open-sell",
        side=Side.SELL.value,
        action=Side.SELL.value,
        notional_usdc=6.0,
    )
    exit_order = _order_state(
        exit_decision,
        status=OrderStatus.LIVE.value,
        raw_status="open",
        filled_notional_usdc=0.0,
    )

    manager.record_open_order_state(exit_order)

    approved = manager.check(
        _decision(
            decision_id="buy-after-open-sell",
            market_id="m-open-sell",
            notional_usdc=10.0,
        ),
        _portfolio(),
    )
    assert approved.approved is True


@pytest.mark.asyncio
async def test_paper_actuator_fills_buy_at_best_ask() -> None:
    actuator = PaperActuator(
        orderbooks={
            "m-cp06": {
                "bids": [{"price": 0.39, "size": 100.0}],
                "asks": [{"price": 0.41, "size": 100.0}],
            }
        }
    )

    state = await actuator.execute(
        _decision(notional_usdc=10.0, limit_price=0.41),
        _portfolio(),
    )

    assert state.status == OrderStatus.MATCHED.value
    assert state.fill_price == 0.41
    assert state.filled_notional_usdc == pytest.approx(10.0)
    assert state.remaining_notional_usdc == 0.0


@pytest.mark.asyncio
async def test_paper_actuator_fills_no_from_no_token_ask() -> None:
    actuator = PaperActuator(
        orderbooks={
            "t-no": {
                "bids": [{"price": 0.36, "size": 100.0}],
                "asks": [{"price": 0.38, "size": 100.0}],
            }
        }
    )

    state = await actuator.execute(
        _decision(
            decision_id="d-no-cp06",
            notional_usdc=10.0,
            limit_price=0.38,
            outcome="NO",
        ),
        _portfolio(),
    )

    assert state.status == OrderStatus.MATCHED.value
    assert state.fill_price == pytest.approx(0.38)


@pytest.mark.asyncio
async def test_paper_actuator_empty_orderbook_raises_insufficient_liquidity() -> None:
    actuator = PaperActuator(orderbooks={"m-cp06": {"bids": [], "asks": []}})

    with pytest.raises(InsufficientLiquidityError):
        await actuator.execute(_decision(), _portfolio())


@pytest.mark.asyncio
async def test_backtest_actuator_replays_fill_from_fixture() -> None:
    fixture_path = Path("tests/fixtures/polymarket_7day_synthetic.jsonl")
    actuator = BacktestActuator(fixture_path)

    state = await actuator.execute(
        _decision(
            market_id="pm-synthetic-000",
            token_id="yes-token-000",
            notional_usdc=10.0,
            limit_price=0.31,
        ),
        _portfolio(),
    )

    assert state.status == OrderStatus.MATCHED.value
    assert state.fill_price == 0.31


@pytest.mark.asyncio
async def test_polymarket_actuator_raises_when_live_trading_disabled() -> None:
    actuator = PolymarketActuator(PMSSettings(live_trading_enabled=False))

    with pytest.raises(LiveTradingDisabledError):
        await actuator.execute(_decision(), _portfolio())


@dataclass
class RecordingPolymarketClient:
    requires_live_mode: bool = False
    submitted: list[PolymarketOrderRequest] = field(default_factory=list)

    async def submit_order(
        self,
        order: PolymarketOrderRequest,
        credentials: object,
    ) -> PolymarketOrderResult:
        del credentials
        self.submitted.append(order)
        return PolymarketOrderResult(
            order_id="pm-live-order-1",
            status=OrderStatus.MATCHED.value,
            raw_status="matched",
            filled_notional_usdc=order.notional_usdc,
            remaining_notional_usdc=0.0,
            fill_price=order.price,
            filled_quantity=order.estimated_quantity,
        )


@dataclass
class RecordingOperatorGate:
    approved: bool
    previews: list[LiveOrderPreview] = field(default_factory=list)
    consumed: list[LiveOrderPreview] = field(default_factory=list)

    async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
        self.previews.append(preview)
        return self.approved

    async def consume(self, preview: LiveOrderPreview) -> None:
        self.consumed.append(preview)


@dataclass
class BlockingOperatorGate:
    previews: list[LiveOrderPreview] = field(default_factory=list)
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
        self.previews.append(preview)
        self.entered.set()
        await self.release.wait()
        return True

    async def consume(self, preview: LiveOrderPreview) -> None:
        del preview


@dataclass(frozen=True)
class AllowQuoteProvider:
    async def quote(
        self,
        order: PolymarketOrderRequest,
        credentials: object,
    ) -> LivePreSubmitQuote:
        del credentials
        return LivePreSubmitQuote(
            market_status="open",
            book_age_ms=25.0,
            executable_notional_usdc=order.notional_usdc,
            best_executable_price=order.price,
            spread_bps=10.0,
            quote_hash="quote-unit",
            book_ts=datetime(2026, 4, 26, tzinfo=UTC),
        )


@dataclass(frozen=True)
class StaticQuoteProvider:
    quote_result: LivePreSubmitQuote

    async def quote(
        self,
        order: PolymarketOrderRequest,
        credentials: object,
    ) -> LivePreSubmitQuote:
        del order, credentials
        return self.quote_result


def _live_settings(
    *,
    operator_approval_mode: Literal["first_order", "every_order"] = "first_order",
    signature_type: int = 1,
) -> PMSSettings:
    return PMSSettings(
        live_trading_enabled=True,
        api_token="live-api-token",
        controller=ControllerSettings(time_in_force="IOC"),
        polymarket=PolymarketSettings(
            private_key="private-key",
            api_key="api-key",
            api_secret="api-secret",
            api_passphrase="passphrase",
            signature_type=signature_type,
            funder_address="0x1111111111111111111111111111111111111111",
            operator_approval_mode=operator_approval_mode,
        ),
    )


_SECRET_CREDENTIAL_VALUES = (
    "private-key-secret",
    "api-key-secret",
    "api-secret-secret",
    "passphrase-secret",
    "0x2222222222222222222222222222222222222222",
)
_SECRET_DSN = "postgresql://admin:supersecret@db.internal.example.com:5432/pms_live"
_LIVE_FIXTURE_STRATEGY_EVIDENCE = (
    "default@4d326514fa853b9278502ad43750b9648ac8f4f6ad8685ba522b2a4aa5f47d25"
)


def _live_settings_with_secret_credentials() -> PMSSettings:
    settings = _live_settings()
    settings.polymarket.private_key = _SECRET_CREDENTIAL_VALUES[0]
    settings.polymarket.api_key = _SECRET_CREDENTIAL_VALUES[1]
    settings.polymarket.api_secret = _SECRET_CREDENTIAL_VALUES[2]
    settings.polymarket.api_passphrase = _SECRET_CREDENTIAL_VALUES[3]
    settings.polymarket.funder_address = _SECRET_CREDENTIAL_VALUES[4]
    return settings


def _true_live_settings_without_preflight_artifact(tmp_path: Path) -> PMSSettings:
    approval_dir = tmp_path / "secure"
    approval_dir.mkdir(mode=0o700)
    paper_report_path, rehearsal_report_path = make_live_report_paths(
        prefix="pms-actuator-live-preflight-reports-"
    )
    attested_at = datetime.now(tz=UTC)
    return PMSSettings(
        mode=RunMode.LIVE,
        secret_source="fly",
        live_trading_enabled=True,
        api_token="live-api-token",
        live_account_reconciliation_required=True,
        live_emergency_audit_path=str(approval_dir / "live-emergency-audit.jsonl"),
        live_first_order_audit_path=str(approval_dir / "first-order-audit.jsonl"),
        live_exit_criteria_ratified_by="operator",
        live_exit_criteria_ratified_at=attested_at,
        live_compliance_reviewed_by="counsel",
        live_compliance_reviewed_at=attested_at,
        live_compliance_jurisdiction="US-operator-approved",
        live_paper_soak_report_path=paper_report_path,
        live_operator_rehearsal_report_path=rehearsal_report_path,
        controller=ControllerSettings(
            time_in_force="IOC",
            quote_source="dual",
            strict_factor_gates=True,
        ),
        risk=RiskSettings(
            max_position_per_market=5.0,
            max_total_exposure=50.0,
            max_drawdown_pct=20.0,
            max_daily_loss_usdc=20.0,
            max_open_positions=5,
            max_exposure_per_risk_group=15.0,
            max_quantity_shares=500.0,
            min_order_usdc=1.0,
        ),
        discord=DiscordSettings(
            webhook_url=SecretStr("https://discord.example/webhooks/actuator/unit"),
            alert_dir=str(approval_dir / "discord-alerts"),
        ),
        polymarket=PolymarketSettings(
            private_key="private-key",
            api_key="api-key",
            api_secret="api-secret",
            api_passphrase="passphrase",
            signature_type=1,
            funder_address="0x1111111111111111111111111111111111111111",
            first_live_order_approval_path=str(approval_dir / "first-order.json"),
            operator_approval_mode="every_order",
        ),
    )


def _stage_readiness_fingerprint_files(settings: PMSSettings, root: Path) -> None:
    execution_model_path = root / "execution-model.json"
    paper_backtest_diff_path = root / "paper-backtest-diff.json"
    category_prior_path = root / "category-prior.csv"
    flb_calibration_path = root / "flb-calibration.csv"
    generated_at = datetime.now(UTC) - timedelta(seconds=30)
    execution_model_path.write_text(
        json.dumps(
            {
                "artifact_mode": "telemetry_execution_model",
                "generated_at": generated_at.isoformat(),
                "generated_by": "scripts/execution_model_from_telemetry.py",
                "strategy_evidence": _LIVE_FIXTURE_STRATEGY_EVIDENCE,
                "fee_rate": 0.04,
                "slippage_bps": 6.0,
                "latency_ms": 500.0,
                "staleness_ms": 120_000.0,
                "fill_policy": "immediate_or_cancel",
                "displayed_depth_fill_ratio": 0.75,
                "adverse_selection_bps": 9.0,
                "order_ttl_ms": 60_000,
                "price_invalidation_streak": 10,
                "replay_window_ms": 86_400_000,
                "calibration_source": "telemetry_calibrated",
                "input_csv_sha256": "c" * 64,
                "min_samples": 10,
                "telemetry_sample_count": 10,
                "adverse_selection_sample_count": 10,
                "require_adverse_selection": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paper_backtest_diff_path.write_text(
        json.dumps(
            {
                "generated_by": "scripts/paper_backtest_execution_diff.py",
                "artifact_mode": "paper_backtest_execution_diff",
                "generated_at": generated_at.isoformat(),
                "strategy_evidence": _LIVE_FIXTURE_STRATEGY_EVIDENCE,
                "input_csv_sha256": {
                    "paper": "a" * 64,
                    "backtest": "b" * 64,
                },
                "final_go_no_go_valid": True,
                "thresholds": {
                    "min_matched_decisions": 10,
                    "max_fill_rate_delta": 0.05,
                    "max_rejection_rate_delta": 0.05,
                    "max_avg_slippage_bps_delta": 5.0,
                    "max_total_pnl_delta": 1.0,
                },
                "metrics": {
                    "paper_decision_count": 10,
                    "backtest_decision_count": 10,
                    "matched_decision_count": 10,
                    "fill_rate_delta_abs": 0.0,
                    "rejection_rate_delta_abs": 0.0,
                    "avg_slippage_bps_delta_abs": 0.0,
                    "total_pnl_delta_abs": 0.0,
                },
                "paper_only_decision_ids": [],
                "backtest_only_decision_ids": [],
                "status_mismatches": [],
                "failures": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    category_prior_rows = ["market_id,category,yes_payout,no_payout,resolved_at"]
    for index in range(1, 121):
        category = "politics" if index % 2 == 0 else "sports"
        yes_payout, no_payout = ("1", "0") if index % 3 == 0 else ("0", "1")
        category_prior_rows.append(
            f"m-{index},{category},{yes_payout},{no_payout},2026-05-{(index % 20) + 1:02d}T12:00:00Z"
        )
    category_prior_path.write_text(
        "\n".join(category_prior_rows) + "\n",
        encoding="utf-8",
    )
    flb_calibration_path.write_text(
        "\n".join(
            (
                "signal_name,probability_estimate,sample_count,source_label",
                "longshot_yes_overpriced_buy_no,0.99,150,warehouse-flb-v1",
                "favorite_yes_underpriced_buy_yes,0.97,151,warehouse-flb-v1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    Path(f"{flb_calibration_path}.provenance.json").write_text(
        json.dumps(
            {
                "artifact_type": "flb_calibration_provenance",
                "generated_by": "scripts/flb_data_feasibility.py",
                "source": "warehouse-csv",
                "generated_at": generated_at.isoformat(),
                "warehouse_csv_sha256": sha256(
                    b"unit warehouse provenance fixture"
                ).hexdigest(),
                "warehouse_market_count": 301,
                "warehouse_longshot_count": 150,
                "warehouse_favorite_count": 151,
                "calibration_csv_sha256": sha256(
                    flb_calibration_path.read_bytes()
                ).hexdigest(),
                "calibration_source_label": "warehouse-flb-v1",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    settings.live_execution_model_path = str(execution_model_path)
    settings.live_paper_backtest_diff_path = str(paper_backtest_diff_path)
    settings.controller.category_prior_observations_path = str(category_prior_path)
    settings.strategies.flb_calibration_path = str(flb_calibration_path)


def _replace_report_generated_at(path: str, generated_at: datetime) -> None:
    report_path = Path(path)
    replaced = False
    lines: list[str] = []
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| generated_at |"):
            lines.append(f"| generated_at | {generated_at.isoformat()} |")
            replaced = True
        else:
            lines.append(line)
    assert replaced
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_live_preflight_artifact(
    settings: PMSSettings,
    artifact_path: Path,
    *,
    generated_at: datetime | None = None,
) -> None:
    check_names = (
        "live_config",
        "runtime_dependencies",
        "operator_approval",
        "emergency_audit",
        "first_order_audit",
        "database_connection",
        "schema_current",
        "market_data_freshness",
        "submission_unknown",
        "live_open_orders",
        "active_strategies",
        "venue_reconciliation",
    )
    artifact_path.write_text(
        json.dumps(
            {
                "generated_by": "pms-live preflight",
                "artifact_mode": "credentialed_preflight",
                "final_go_no_go_valid": True,
                "skip_venue": False,
                "database_url_override_used": False,
                "settings_fingerprint": live_preflight_settings_fingerprint(settings),
                "readiness_reports_fingerprint": (
                    live_preflight_readiness_reports_fingerprint(settings)
                ),
                "active_strategies_fingerprint": "d" * 64,
                "output_path": str(artifact_path),
                "generated_at": (generated_at or datetime.now(UTC)).isoformat(),
                "result": {
                    "ok": True,
                    "checks": [
                        {
                            "name": name,
                            "ok": True,
                            "detail": "unit preflight passed",
                        }
                        for name in check_names
                    ],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


@pytest.fixture
def live_sdk_dependency_available(monkeypatch: pytest.MonkeyPatch) -> None:
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(
        name: str,
        package: str | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if name == "py_clob_client_v2":
            return importlib.machinery.ModuleSpec(name, loader=None)
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


def _secret_bearing_error_message(prefix: str) -> str:
    return (
        f"{prefix} "
        f"{_SECRET_CREDENTIAL_VALUES[0]} {_SECRET_CREDENTIAL_VALUES[1]} "
        f"{_SECRET_CREDENTIAL_VALUES[2]} {_SECRET_CREDENTIAL_VALUES[3]} "
        f"{_SECRET_CREDENTIAL_VALUES[4]} {_SECRET_DSN} password=keyword-secret"
    )


def _assert_live_secrets_redacted(rendered: str) -> None:
    assert "<redacted-polymarket-credential>" in rendered
    assert "<redacted-database-url>" in rendered
    assert "password=<redacted>" in rendered
    for credential in _SECRET_CREDENTIAL_VALUES:
        assert credential not in rendered
    assert "supersecret" not in rendered
    assert "keyword-secret" not in rendered
    assert "admin" not in rendered


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_missing_live_credentials() -> None:
    actuator = PolymarketActuator(
        PMSSettings(mode=RunMode.LIVE, live_trading_enabled=True)
    )

    with pytest.raises(LiveTradingDisabledError, match="Missing Polymarket credential fields"):
        await actuator.execute(_decision(), _portfolio())


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_real_client_outside_live_mode() -> None:
    settings = _live_settings()
    settings.mode = RunMode.PAPER
    actuator = PolymarketActuator(
        settings,
        client=PolymarketSDKClient(),
        operator_gate=RecordingOperatorGate(approved=True),
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(LiveTradingDisabledError, match="mode=live"):
        await actuator.execute(_decision(), _portfolio())


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_unmarked_custom_client_outside_live_mode() -> None:
    @dataclass
    class _CustomVenueClient:
        submitted: list[PolymarketOrderRequest] = field(default_factory=list)

        async def submit_order(
            self,
            order: PolymarketOrderRequest,
            credentials: object,
        ) -> PolymarketOrderResult:
            del credentials
            self.submitted.append(order)
            return PolymarketOrderResult(
                order_id="pm-custom-client-order",
                status=OrderStatus.MATCHED.value,
                raw_status="matched",
                filled_notional_usdc=10.0,
                remaining_notional_usdc=0.0,
                fill_price=0.4,
                filled_quantity=25.0,
            )

    settings = _live_settings()
    settings.mode = RunMode.PAPER
    client = _CustomVenueClient()
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=RecordingOperatorGate(approved=True),
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(LiveTradingDisabledError, match="mode=live"):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []


@pytest.mark.parametrize("order_id", ["", "   ", "__FILL_IN_ORDER_ID__"])
@pytest.mark.asyncio
async def test_polymarket_actuator_treats_missing_result_order_id_as_unknown(
    order_id: str,
) -> None:
    @dataclass
    class _MissingOrderIdClient:
        requires_live_mode: bool = False
        submitted: list[PolymarketOrderRequest] = field(default_factory=list)

        async def submit_order(
            self,
            order: PolymarketOrderRequest,
            credentials: object,
        ) -> PolymarketOrderResult:
            del credentials
            self.submitted.append(order)
            return PolymarketOrderResult(
                order_id=order_id,
                status=OrderStatus.MATCHED.value,
                raw_status="matched",
                filled_notional_usdc=10.0,
                remaining_notional_usdc=0.0,
                fill_price=0.4,
                filled_quantity=25.0,
            )

    client = _MissingOrderIdClient()
    actuator = PolymarketActuator(
        _live_settings(),
        client=client,
        operator_gate=RecordingOperatorGate(approved=True),
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(PolymarketSubmissionUnknownError, match="venue order id"):
        await actuator.execute(_decision(), _portfolio())

    assert len(client.submitted) == 1


@pytest.mark.parametrize(
    ("quote_overrides", "expected_detail"),
    [
        ({"book_age_ms": float("nan")}, "book_age_ms"),
        ({"book_age_ms": -1.0}, "book_age_ms"),
        ({"executable_notional_usdc": float("nan")}, "executable_notional"),
        ({"best_executable_price": float("nan")}, "best_executable_price"),
        ({"best_executable_price": 0.0}, "best_executable_price"),
        ({"spread_bps": float("nan")}, "spread_bps"),
        ({"spread_bps": -1.0}, "spread_bps"),
        ({"quote_hash": ""}, "quote_hash"),
    ],
)
@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_malformed_pre_submit_quote(
    quote_overrides: dict[str, object],
    expected_detail: str,
) -> None:
    client = RecordingPolymarketClient()
    quote = {
        "market_status": "open",
        "book_age_ms": 25.0,
        "executable_notional_usdc": 10.0,
        "best_executable_price": 0.4,
        "spread_bps": 10.0,
        "quote_hash": "quote-valid",
        "book_ts": datetime(2026, 4, 26, tzinfo=UTC),
    } | quote_overrides
    actuator = PolymarketActuator(
        _live_settings(),
        client=client,
        operator_gate=RecordingOperatorGate(approved=True),
        quote_provider=StaticQuoteProvider(
            LivePreSubmitQuote(
                market_status=cast(str, quote["market_status"]),
                book_age_ms=cast(float, quote["book_age_ms"]),
                executable_notional_usdc=cast(float, quote["executable_notional_usdc"]),
                best_executable_price=cast(float, quote["best_executable_price"]),
                spread_bps=cast(float, quote["spread_bps"]),
                quote_hash=cast(str, quote["quote_hash"]),
                book_ts=cast(datetime, quote["book_ts"]),
            )
        ),
    )

    with pytest.raises(LiveTradingDisabledError, match=expected_detail):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []


@pytest.mark.parametrize(
    ("result_overrides", "expected_detail"),
    [
        ({"filled_notional_usdc": float("nan")}, "filled_notional"),
        ({"filled_notional_usdc": -1.0}, "filled_notional"),
        ({"filled_notional_usdc": 11.0, "remaining_notional_usdc": 0.0}, "filled_notional"),
        ({"remaining_notional_usdc": -0.01}, "remaining_notional"),
        ({"filled_quantity": -1.0}, "filled_quantity"),
        ({"fill_price": 1.5}, "fill_price"),
        (
            {
                "filled_notional_usdc": 4.0,
                "remaining_notional_usdc": 6.0,
                "filled_quantity": 100.0,
                "fill_price": 0.5,
            },
            "fill accounting",
        ),
        (
            {
                "filled_notional_usdc": 4.0,
                "remaining_notional_usdc": 7.0,
                "filled_quantity": 10.0,
                "fill_price": 0.4,
            },
            "notional accounting",
        ),
    ],
)
@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_malformed_client_result_accounting(
    result_overrides: dict[str, object],
    expected_detail: str,
) -> None:
    @dataclass
    class _MalformedResultClient:
        requires_live_mode: bool = False

        async def submit_order(
            self,
            order: PolymarketOrderRequest,
            credentials: object,
        ) -> PolymarketOrderResult:
            del order, credentials
            result = {
                "order_id": "pm-malformed-result",
                "status": OrderStatus.MATCHED.value,
                "raw_status": "matched",
                "filled_notional_usdc": 10.0,
                "remaining_notional_usdc": 0.0,
                "fill_price": 0.4,
                "filled_quantity": 25.0,
            } | result_overrides
            return PolymarketOrderResult(
                order_id=cast(str, result["order_id"]),
                status=cast(str, result["status"]),
                raw_status=cast(str, result["raw_status"]),
                filled_notional_usdc=cast(float, result["filled_notional_usdc"]),
                remaining_notional_usdc=cast(
                    float,
                    result["remaining_notional_usdc"],
                ),
                fill_price=cast(float | None, result["fill_price"]),
                filled_quantity=cast(float, result["filled_quantity"]),
            )

    actuator = PolymarketActuator(
        _live_settings(),
        client=_MalformedResultClient(),
        operator_gate=RecordingOperatorGate(approved=True),
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(PolymarketSubmissionUnknownError, match=expected_detail):
        await actuator.execute(_decision(), _portfolio())


@pytest.mark.asyncio
async def test_polymarket_actuator_requires_preflight_artifact_for_direct_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        _true_live_settings_without_preflight_artifact(tmp_path),
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(LiveTradingDisabledError, match="preflight artifact"):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert gate.previews == []


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_runner_validated_live_submit_without_preflight_artifact_path(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        _true_live_settings_without_preflight_artifact(tmp_path),
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
        live_preflight_validated=True,
    )

    with pytest.raises(LiveTradingDisabledError, match="preflight artifact"):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert gate.previews == []


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_preflight_artifact_permissive_parent_for_direct_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_dir = tmp_path / "preflight"
    artifact_dir.mkdir(mode=0o755)
    readiness_dir = tmp_path / "readiness"
    readiness_dir.mkdir(mode=0o700)
    artifact_path = artifact_dir / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, readiness_dir)
    _write_final_live_preflight_artifact(settings, artifact_path)
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    try:
        with pytest.raises(
            LiveTradingDisabledError,
            match="preflight artifact parent",
        ):
            await actuator.execute(_decision(), _portfolio())
    finally:
        artifact_dir.chmod(0o700)

    assert client.submitted == []
    assert gate.previews == []


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_preflight_artifact_inside_working_tree_for_direct_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_dir = (
        Path.cwd()
        / f".pms-live-preflight-actuator-test-{os.getpid()}-{tmp_path.name}"
    )
    readiness_dir = tmp_path / "readiness"
    readiness_dir.mkdir(mode=0o700)
    artifact_path = artifact_dir / "credentialed-preflight.json"
    artifact_dir.mkdir(mode=0o700)
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, readiness_dir)
    _write_final_live_preflight_artifact(settings, artifact_path)
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    try:
        with pytest.raises(LiveTradingDisabledError, match="working tree"):
            await actuator.execute(_decision(), _portfolio())
    finally:
        artifact_path.unlink(missing_ok=True)
        artifact_dir.rmdir()

    assert client.submitted == []
    assert gate.previews == []


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_incomplete_preflight_artifact_for_direct_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    artifact_path.write_text(
        json.dumps(
            {
                "generated_by": "pms-live preflight",
                "artifact_mode": "credentialed_preflight",
                "final_go_no_go_valid": True,
                "generated_at": datetime.now(UTC).isoformat(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(LiveTradingDisabledError, match="preflight artifact"):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert gate.previews == []


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_naive_preflight_generated_at_for_direct_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    _write_final_live_preflight_artifact(
        settings,
        artifact_path,
        generated_at=datetime.now(UTC).replace(tzinfo=None),
    )
    approval_path = Path(cast(str, settings.polymarket.first_live_order_approval_path))
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )
    client = RecordingPolymarketClient()
    gate = FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=settings.polymarket.operator_approval_max_age_s,
    )
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    try:
        with pytest.raises(
            LiveTradingDisabledError,
            match="generated_at must include timezone",
        ):
            await actuator.execute(_decision(), _portfolio())
    finally:
        approval_path.unlink(missing_ok=True)
        _sidecar_path(approval_path).unlink(missing_ok=True)

    assert client.submitted == []


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_wrong_settings_preflight_artifact_for_direct_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    check_names = (
        "live_config",
        "runtime_dependencies",
        "operator_approval",
        "emergency_audit",
        "first_order_audit",
        "database_connection",
        "schema_current",
        "market_data_freshness",
        "submission_unknown",
        "live_open_orders",
        "active_strategies",
        "venue_reconciliation",
    )
    artifact_path.write_text(
        json.dumps(
            {
                "generated_by": "pms-live preflight",
                "artifact_mode": "credentialed_preflight",
                "final_go_no_go_valid": True,
                "skip_venue": False,
                "database_url_override_used": False,
                "settings_fingerprint": "b" * 64,
                "readiness_reports_fingerprint": "c" * 64,
                "active_strategies_fingerprint": "d" * 64,
                "output_path": str(artifact_path),
                "generated_at": datetime.now(UTC).isoformat(),
                "result": {
                    "ok": True,
                    "checks": [
                        {
                            "name": name,
                            "ok": True,
                            "detail": "unit preflight passed",
                        }
                        for name in check_names
                    ],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(LiveTradingDisabledError, match="settings fingerprint"):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert gate.previews == []


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_stale_readiness_preflight_artifact_for_direct_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    check_names = (
        "live_config",
        "runtime_dependencies",
        "operator_approval",
        "emergency_audit",
        "first_order_audit",
        "database_connection",
        "schema_current",
        "market_data_freshness",
        "submission_unknown",
        "live_open_orders",
        "active_strategies",
        "venue_reconciliation",
    )
    artifact_path.write_text(
        json.dumps(
            {
                "generated_by": "pms-live preflight",
                "artifact_mode": "credentialed_preflight",
                "final_go_no_go_valid": True,
                "skip_venue": False,
                "database_url_override_used": False,
                "settings_fingerprint": live_preflight_settings_fingerprint(settings),
                "readiness_reports_fingerprint": "c" * 64,
                "active_strategies_fingerprint": "d" * 64,
                "output_path": str(artifact_path),
                "generated_at": datetime.now(UTC).isoformat(),
                "result": {
                    "ok": True,
                    "checks": [
                        {
                            "name": name,
                            "ok": True,
                            "detail": "unit preflight passed",
                        }
                        for name in check_names
                    ],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(LiveTradingDisabledError, match="readiness reports fingerprint"):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert gate.previews == []


@pytest.mark.asyncio
async def test_polymarket_actuator_revalidates_strategy_artifacts_for_direct_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    execution_model_path = Path(cast(str, settings.live_execution_model_path))
    execution_model_path.write_text(
        json.dumps(
            {
                "generated_by": "scripts/execution_model_from_telemetry.py",
                "artifact_mode": "telemetry_execution_model",
                "generated_at": datetime.now(UTC).isoformat(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_final_live_preflight_artifact(settings, artifact_path)
    decision = _decision()
    approval_path = Path(cast(str, settings.polymarket.first_live_order_approval_path))
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )
    client = RecordingPolymarketClient()
    gate = FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=settings.polymarket.operator_approval_max_age_s,
    )
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    try:
        with pytest.raises(LiveTradingDisabledError, match="execution-model artifact"):
            await actuator.execute(decision, _portfolio())
    finally:
        approval_path.unlink(missing_ok=True)
        _sidecar_path(approval_path).unlink(missing_ok=True)

    assert client.submitted == []


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_duplicate_flb_calibration_for_direct_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    calibration_path = Path(cast(str, settings.strategies.flb_calibration_path))
    calibration_path.write_text(
        "\n".join(
            (
                "signal_name,probability_estimate,sample_count,source_label",
                "longshot_yes_overpriced_buy_no,0.99,150,warehouse-flb-v1",
                "longshot_yes_overpriced_buy_no,0.98,151,warehouse-flb-v2",
                "favorite_yes_underpriced_buy_yes,0.97,152,warehouse-flb-v1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _write_final_live_preflight_artifact(settings, artifact_path)
    decision = _decision()
    approval_path = Path(cast(str, settings.polymarket.first_live_order_approval_path))
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )
    client = RecordingPolymarketClient()
    gate = FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=settings.polymarket.operator_approval_max_age_s,
    )
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    try:
        with pytest.raises(LiveTradingDisabledError, match="FLB calibration artifact"):
            await actuator.execute(decision, _portfolio())
    finally:
        approval_path.unlink(missing_ok=True)
        _sidecar_path(approval_path).unlink(missing_ok=True)

    assert client.submitted == []


def test_direct_live_submit_artifact_validation_rejects_infinite_execution_staleness(
    tmp_path: Path,
) -> None:
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    model_path = Path(cast(str, settings.live_execution_model_path))
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    payload["staleness_ms"] = ".inf"
    model_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        LiveTradingDisabledError,
        match="execution-model artifact staleness_ms must be finite",
    ):
        validate_live_strategy_artifacts_for_submission(settings)


def test_direct_live_submit_artifact_validation_rejects_duplicate_execution_json_key(
    tmp_path: Path,
) -> None:
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    model_path = Path(cast(str, settings.live_execution_model_path))
    model_text = model_path.read_text(encoding="utf-8")
    model_text = model_text.replace(
        '"generated_by": "scripts/execution_model_from_telemetry.py"',
        '"generated_by": "forged-generator.py", '
        '"generated_by": "scripts/execution_model_from_telemetry.py"',
        1,
    )
    assert '"generated_by": "forged-generator.py"' in model_text
    model_path.write_text(model_text, encoding="utf-8")

    with pytest.raises(
        LiveTradingDisabledError,
        match="execution-model artifact duplicate JSON key: generated_by",
    ):
        validate_live_strategy_artifacts_for_submission(settings)


def test_direct_live_submit_artifact_validation_rejects_missing_execution_sample_contract(
    tmp_path: Path,
) -> None:
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    model_path = Path(cast(str, settings.live_execution_model_path))
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    payload.pop("min_samples", None)
    payload.pop("telemetry_sample_count", None)
    payload.pop("adverse_selection_sample_count", None)
    payload.pop("require_adverse_selection", None)
    model_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        LiveTradingDisabledError,
        match="execution-model artifact missing telemetry sample contract",
    ):
        validate_live_strategy_artifacts_for_submission(settings)


def test_direct_live_submit_artifact_validation_rejects_missing_paper_backtest_min_matched_threshold(
    tmp_path: Path,
) -> None:
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    diff_path = Path(cast(str, settings.live_paper_backtest_diff_path))
    payload = json.loads(diff_path.read_text(encoding="utf-8"))
    thresholds = cast(dict[str, object], payload["thresholds"])
    thresholds.pop("min_matched_decisions", None)
    diff_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        LiveTradingDisabledError,
        match="missing threshold: min_matched_decisions",
    ):
        validate_live_strategy_artifacts_for_submission(settings)


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_inconsistent_paper_backtest_diff_for_direct_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    diff_path = Path(cast(str, settings.live_paper_backtest_diff_path))
    payload = json.loads(diff_path.read_text(encoding="utf-8"))
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["matched_decision_count"] = 10
    metrics["paper_decision_count"] = 11
    metrics["backtest_decision_count"] = 10
    payload["paper_only_decision_ids"] = []
    payload["backtest_only_decision_ids"] = []
    payload["status_mismatches"] = []
    diff_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_final_live_preflight_artifact(settings, artifact_path)
    decision = _decision()
    approval_path = Path(cast(str, settings.polymarket.first_live_order_approval_path))
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )
    client = RecordingPolymarketClient()
    gate = FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=settings.polymarket.operator_approval_max_age_s,
    )
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    try:
        with pytest.raises(
            LiveTradingDisabledError,
            match="matched_decision_count must equal paper_decision_count",
        ):
            await actuator.execute(decision, _portfolio())
    finally:
        approval_path.unlink(missing_ok=True)
        _sidecar_path(approval_path).unlink(missing_ok=True)

    assert client.submitted == []


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_preflight_before_emergency_audit_for_direct_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    generated_at = datetime.now(UTC) - timedelta(seconds=20)
    emergency_audit_at = datetime.now(UTC) - timedelta(seconds=10)
    settings.live_exit_criteria_ratified_at = generated_at - timedelta(seconds=1)
    settings.live_compliance_reviewed_at = generated_at - timedelta(seconds=1)
    Path(settings.live_emergency_audit_path).write_text(
        json.dumps(
            {
                "event": "emergency_stop_completed",
                "operator_id": "operator",
                "timestamp": emergency_audit_at.isoformat(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    check_names = (
        "live_config",
        "runtime_dependencies",
        "operator_approval",
        "emergency_audit",
        "first_order_audit",
        "database_connection",
        "schema_current",
        "market_data_freshness",
        "submission_unknown",
        "live_open_orders",
        "active_strategies",
        "venue_reconciliation",
    )
    artifact_path.write_text(
        json.dumps(
            {
                "generated_by": "pms-live preflight",
                "artifact_mode": "credentialed_preflight",
                "final_go_no_go_valid": True,
                "skip_venue": False,
                "database_url_override_used": False,
                "settings_fingerprint": live_preflight_settings_fingerprint(settings),
                "readiness_reports_fingerprint": (
                    live_preflight_readiness_reports_fingerprint(settings)
                ),
                "active_strategies_fingerprint": "d" * 64,
                "output_path": str(artifact_path),
                "generated_at": generated_at.isoformat(),
                "result": {
                    "ok": True,
                    "checks": [
                        {
                            "name": name,
                            "ok": True,
                            "detail": "unit preflight passed",
                        }
                        for name in check_names
                    ],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(LiveTradingDisabledError, match="emergency audit"):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert gate.previews == []


@pytest.mark.asyncio
async def test_polymarket_actuator_rechecks_preflight_artifact_after_runner_start_for_direct_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    generated_at = datetime.now(UTC) - timedelta(seconds=20)
    emergency_audit_at = datetime.now(UTC) - timedelta(seconds=10)
    settings.live_exit_criteria_ratified_at = generated_at - timedelta(seconds=1)
    settings.live_compliance_reviewed_at = generated_at - timedelta(seconds=1)
    _write_final_live_preflight_artifact(
        settings,
        artifact_path,
        generated_at=generated_at,
    )
    Path(settings.live_emergency_audit_path).write_text(
        json.dumps(
            {
                "event": "emergency_stop_completed",
                "operator_id": "operator",
                "timestamp": emergency_audit_at.isoformat(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
        live_preflight_validated=True,
    )

    with pytest.raises(LiveTradingDisabledError, match="emergency audit"):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert gate.previews == []


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_preflight_active_strategy_fingerprint_changed_after_runner_validation(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    _write_final_live_preflight_artifact(settings, artifact_path)
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
        live_preflight_validated=True,
        live_preflight_active_strategies_fingerprint="e" * 64,
    )

    with pytest.raises(
        LiveTradingDisabledError,
        match="active strategies fingerprint mismatch",
    ):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert gate.previews == []


@pytest.mark.asyncio
async def test_polymarket_actuator_allows_pending_strict_sidecar_approval_during_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    _write_final_live_preflight_artifact(settings, artifact_path)
    approval_path = Path(cast(str, settings.polymarket.first_live_order_approval_path))
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )
    client = RecordingPolymarketClient()
    writer = RecordingFirstOrderAuditWriter()
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=FileFirstLiveOrderGate(
            approval_path,
            require_approver_sidecar=True,
            approval_max_age_s=settings.polymarket.operator_approval_max_age_s,
        ),
        quote_provider=AllowQuoteProvider(),
        audit_writer=writer,
    )

    state = await actuator.execute(_decision(), _portfolio())

    assert state.status == OrderStatus.MATCHED.value
    assert len(client.submitted) == 1
    assert approval_path.exists() is False
    assert _sidecar_path(approval_path).exists() is False
    assert [
        (event, approver_id) for event, _, approver_id in writer.events
    ] == [
        ("approval_matched", "operator-alice"),
        ("approval_consumed", "operator-alice"),
    ]


@pytest.mark.asyncio
async def test_polymarket_actuator_requires_strict_sidecar_gate_for_true_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    _write_final_live_preflight_artifact(settings, artifact_path)
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(LiveTradingDisabledError, match="strict sidecar"):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert gate.previews == []


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_spoofed_sidecar_gate_for_true_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available

    @dataclass
    class SpoofedSidecarGate:
        require_approver_sidecar: bool = True
        previews: list[LiveOrderPreview] = field(default_factory=list)

        async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
            self.previews.append(preview)
            return True

        async def consume(self, preview: LiveOrderPreview) -> None:
            del preview

        def read_approver_id(self) -> str:
            return "operator-spoof"

    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    _write_final_live_preflight_artifact(settings, artifact_path)
    client = RecordingPolymarketClient()
    gate = SpoofedSidecarGate()
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(LiveTradingDisabledError, match="FileFirstLiveOrderGate"):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert gate.previews == []


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_preflight_before_readiness_reports_for_direct_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    generated_at = datetime.now(UTC) - timedelta(seconds=20)
    report_generated_at = datetime.now(UTC) - timedelta(seconds=10)
    readiness_attested_at = datetime.now(UTC) - timedelta(seconds=5)
    settings.live_exit_criteria_ratified_at = readiness_attested_at
    settings.live_compliance_reviewed_at = readiness_attested_at
    assert settings.live_paper_soak_report_path is not None
    assert settings.live_operator_rehearsal_report_path is not None
    _replace_report_generated_at(
        settings.live_paper_soak_report_path,
        report_generated_at,
    )
    _replace_report_generated_at(
        settings.live_operator_rehearsal_report_path,
        report_generated_at,
    )
    check_names = (
        "live_config",
        "runtime_dependencies",
        "operator_approval",
        "emergency_audit",
        "first_order_audit",
        "database_connection",
        "schema_current",
        "market_data_freshness",
        "submission_unknown",
        "live_open_orders",
        "active_strategies",
        "venue_reconciliation",
    )
    artifact_path.write_text(
        json.dumps(
            {
                "generated_by": "pms-live preflight",
                "artifact_mode": "credentialed_preflight",
                "final_go_no_go_valid": True,
                "skip_venue": False,
                "database_url_override_used": False,
                "settings_fingerprint": live_preflight_settings_fingerprint(settings),
                "readiness_reports_fingerprint": (
                    live_preflight_readiness_reports_fingerprint(settings)
                ),
                "active_strategies_fingerprint": "d" * 64,
                "output_path": str(artifact_path),
                "generated_at": generated_at.isoformat(),
                "result": {
                    "ok": True,
                    "checks": [
                        {
                            "name": name,
                            "ok": True,
                            "detail": "unit preflight passed",
                        }
                        for name in check_names
                    ],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(LiveTradingDisabledError, match="readiness reports"):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert gate.previews == []


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_preflight_before_live_readiness_signoff_for_direct_live_submit(
    tmp_path: Path,
    live_sdk_dependency_available: None,
) -> None:
    del live_sdk_dependency_available
    settings = _true_live_settings_without_preflight_artifact(tmp_path)
    artifact_path = tmp_path / "secure" / "credentialed-preflight.json"
    settings.live_preflight_artifact_path = str(artifact_path)
    _stage_readiness_fingerprint_files(settings, artifact_path.parent)
    generated_at = datetime.now(UTC) - timedelta(seconds=20)
    report_generated_at = datetime.now(UTC) - timedelta(seconds=30)
    readiness_attested_at = datetime.now(UTC) - timedelta(seconds=10)
    settings.live_exit_criteria_ratified_at = readiness_attested_at
    settings.live_compliance_reviewed_at = readiness_attested_at
    assert settings.live_paper_soak_report_path is not None
    assert settings.live_operator_rehearsal_report_path is not None
    _replace_report_generated_at(
        settings.live_paper_soak_report_path,
        report_generated_at,
    )
    _replace_report_generated_at(
        settings.live_operator_rehearsal_report_path,
        report_generated_at,
    )
    check_names = (
        "live_config",
        "runtime_dependencies",
        "operator_approval",
        "emergency_audit",
        "first_order_audit",
        "database_connection",
        "schema_current",
        "market_data_freshness",
        "submission_unknown",
        "live_open_orders",
        "active_strategies",
        "venue_reconciliation",
    )
    artifact_path.write_text(
        json.dumps(
            {
                "generated_by": "pms-live preflight",
                "artifact_mode": "credentialed_preflight",
                "final_go_no_go_valid": True,
                "skip_venue": False,
                "database_url_override_used": False,
                "settings_fingerprint": live_preflight_settings_fingerprint(settings),
                "readiness_reports_fingerprint": (
                    live_preflight_readiness_reports_fingerprint(settings)
                ),
                "active_strategies_fingerprint": "d" * 64,
                "output_path": str(artifact_path),
                "generated_at": generated_at.isoformat(),
                "result": {
                    "ok": True,
                    "checks": [
                        {
                            "name": name,
                            "ok": True,
                            "detail": "unit preflight passed",
                        }
                        for name in check_names
                    ],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        settings,
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(LiveTradingDisabledError, match="LIVE readiness"):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert gate.previews == []


@pytest.mark.asyncio
async def test_polymarket_actuator_requires_operator_approval_for_first_live_order() -> None:
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=False)
    actuator = PolymarketActuator(
        _live_settings(),
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(OperatorApprovalRequiredError):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert len(gate.previews) == 1
    assert gate.previews[0].max_notional_usdc == 10.0
    assert gate.previews[0].venue == "polymarket"
    assert gate.previews[0].market_id == "m-cp06"
    assert gate.previews[0].token_id == "t-yes"
    assert gate.previews[0].side == Side.BUY.value
    assert gate.previews[0].limit_price == 0.4
    assert gate.previews[0].max_slippage_bps == 50


@pytest.mark.asyncio
async def test_missing_quote_provider_fails_closed_before_submit() -> None:
    client = RecordingPolymarketClient()
    actuator = PolymarketActuator(
        _live_settings(),
        client=client,
        operator_gate=RecordingOperatorGate(approved=True),
        quote_provider=MissingLiveQuoteProvider(),
    )

    with pytest.raises(
        LiveTradingDisabledError,
        match="pre-submit quote guard is not configured",
    ):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []


@pytest.mark.asyncio
async def test_polymarket_actuator_submits_mocked_live_order_after_first_order_gate() -> None:
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        _live_settings(),
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    state = await actuator.execute(_decision(), _portfolio())

    assert state.order_id == "pm-live-order-1"
    assert state.status == OrderStatus.MATCHED.value
    assert state.filled_notional_usdc == 10.0
    assert state.remaining_notional_usdc == 0.0
    assert state.fill_price == 0.4
    assert state.filled_quantity == 25.0
    assert state.strategy_id == "default"
    assert state.strategy_version_id == "default-v1"
    assert len(client.submitted) == 1
    assert len(gate.previews) == 1
    # After a successful first submit, the gate's approval artefact must be
    # consumed so it cannot be replayed by a future restart or concurrent
    # task. Verifies the consume-on-success contract.
    assert len(gate.consumed) == 1
    assert gate.consumed[0] == gate.previews[0]


@pytest.mark.asyncio
async def test_polymarket_actuator_default_first_order_mode_skips_gate_after_first_success() -> None:
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        _live_settings(operator_approval_mode="first_order"),
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    await actuator.execute(_decision(decision_id="d-first"), _portfolio())
    await actuator.execute(_decision(decision_id="d-second"), _portfolio())

    assert len(client.submitted) == 2
    assert [preview.market_id for preview in gate.previews] == ["m-cp06"]
    assert len(gate.consumed) == 1


@pytest.mark.asyncio
async def test_polymarket_actuator_every_order_mode_requires_approval_for_each_submit() -> None:
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        _live_settings(operator_approval_mode="every_order"),
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    await actuator.execute(_decision(decision_id="d-first"), _portfolio())
    await actuator.execute(
        _decision(decision_id="d-second", market_id="m-second"),
        _portfolio(),
    )

    assert len(client.submitted) == 2
    assert [preview.market_id for preview in gate.previews] == [
        "m-cp06",
        "m-second",
    ]
    assert [preview.market_id for preview in gate.consumed] == [
        "m-cp06",
        "m-second",
    ]
    assert actuator._first_order_approved() is True  # noqa: SLF001


@pytest.mark.asyncio
async def test_polymarket_actuator_every_order_mode_denial_blocks_later_submit() -> None:
    @dataclass
    class _SequencedOperatorGate:
        approvals: list[bool]
        previews: list[LiveOrderPreview] = field(default_factory=list)
        consumed: list[LiveOrderPreview] = field(default_factory=list)

        async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
            self.previews.append(preview)
            return self.approvals.pop(0)

        async def consume(self, preview: LiveOrderPreview) -> None:
            self.consumed.append(preview)

    client = RecordingPolymarketClient()
    gate = _SequencedOperatorGate(approvals=[True, False])
    actuator = PolymarketActuator(
        _live_settings(operator_approval_mode="every_order"),
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    await actuator.execute(_decision(decision_id="d-first"), _portfolio())
    with pytest.raises(OperatorApprovalRequiredError):
        await actuator.execute(
            _decision(decision_id="d-second", market_id="m-second"),
            _portfolio(),
        )

    assert len(client.submitted) == 1
    assert [preview.market_id for preview in gate.previews] == [
        "m-cp06",
        "m-second",
    ]
    assert [preview.market_id for preview in gate.consumed] == ["m-cp06"]


@pytest.mark.asyncio
async def test_polymarket_actuator_every_order_mode_blocks_after_consume_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @dataclass
    class _ConsumeRaisesGate:
        previews: list[LiveOrderPreview] = field(default_factory=list)

        async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
            self.previews.append(preview)
            return True

        async def consume(self, preview: LiveOrderPreview) -> None:
            del preview
            raise OSError("permission denied")

    client = RecordingPolymarketClient()
    gate = _ConsumeRaisesGate()
    actuator = PolymarketActuator(
        _live_settings(operator_approval_mode="every_order"),
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    with caplog.at_level(logging.ERROR, logger="pms.actuator.adapters.polymarket"):
        state = await actuator.execute(_decision(decision_id="d-first"), _portfolio())

    assert state.status == OrderStatus.MATCHED.value
    assert len(client.submitted) == 1
    assert any("consume" in record.message.lower() for record in caplog.records)

    with pytest.raises(LiveTradingDisabledError, match="approval consume failed"):
        await actuator.execute(
            _decision(decision_id="d-second", market_id="m-second"),
            _portfolio(),
        )

    assert len(client.submitted) == 1
    assert [preview.market_id for preview in gate.previews] == ["m-cp06"]


@pytest.mark.asyncio
async def test_every_order_consume_failure_blocks_waiting_concurrent_submit() -> None:
    @dataclass
    class _RaceClient:
        release_first: asyncio.Event = field(default_factory=asyncio.Event)
        first_entered: asyncio.Event = field(default_factory=asyncio.Event)
        submitted: list[PolymarketOrderRequest] = field(default_factory=list)
        requires_live_mode: bool = False

        async def submit_order(
            self,
            order: PolymarketOrderRequest,
            credentials: object,
        ) -> PolymarketOrderResult:
            del credentials
            self.submitted.append(order)
            if len(self.submitted) == 1:
                self.first_entered.set()
                await self.release_first.wait()
            return PolymarketOrderResult(
                order_id=f"race-{len(self.submitted)}",
                status=OrderStatus.MATCHED.value,
                raw_status="matched",
                filled_notional_usdc=order.notional_usdc,
                remaining_notional_usdc=0.0,
                fill_price=order.price,
                filled_quantity=order.estimated_quantity,
            )

    @dataclass
    class _ConsumeFailureGate:
        consume_calls: int = 0

        async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
            del preview
            return True

        async def consume(self, preview: LiveOrderPreview) -> None:
            del preview
            self.consume_calls += 1
            if self.consume_calls == 1:
                raise OSError("stale approval cleanup failed")

    client = _RaceClient()
    gate = _ConsumeFailureGate()
    actuator = PolymarketActuator(
        _live_settings(operator_approval_mode="every_order"),
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    first = asyncio.create_task(
        actuator.execute(_decision(decision_id="race-first"), _portfolio())
    )
    await asyncio.wait_for(client.first_entered.wait(), timeout=1.0)
    second = asyncio.create_task(
        actuator.execute(_decision(decision_id="race-second"), _portfolio())
    )
    # Let the second task pass the pre-lock check and wait for the first task.
    await asyncio.sleep(0)
    client.release_first.set()
    first_result, second_result = await asyncio.gather(
        first,
        second,
        return_exceptions=True,
    )

    assert isinstance(first_result, OrderState)
    assert isinstance(second_result, LiveTradingDisabledError)
    assert len(client.submitted) == 1
    assert gate.consume_calls == 1


@pytest.mark.asyncio
async def test_polymarket_actuator_timeout_is_submission_unknown_and_consumes_approval() -> None:
    @dataclass
    class _TimeoutClient:
        requires_live_mode: bool = False

        async def submit_order(
            self,
            order: PolymarketOrderRequest,
            credentials: object,
        ) -> PolymarketOrderResult:
            del order, credentials
            raise asyncio.TimeoutError("client request timed out")

    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        _live_settings(operator_approval_mode="every_order"),
        client=_TimeoutClient(),
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(PolymarketSubmissionUnknownError):
        await actuator.execute(_decision(), _portfolio())

    assert len(gate.consumed) == 1
    assert actuator._first_order_approved() is True  # noqa: SLF001


@pytest.mark.asyncio
async def test_polymarket_actuator_first_order_gate_is_serialized() -> None:
    # The lock around the gate must serialize gate calls — a second concurrent
    # task cannot observe an in-progress gate. After the live-hardening fix
    # the first-order flag is *not* set until after a successful submit, so
    # the second task may legitimately call the gate again once T1 exits the
    # critical section. What MUST hold:
    #   1. While T1 is blocked in the gate, T2 cannot have called it.
    #   2. Both tasks eventually submit (one operator approval is the floor,
    #      not the ceiling — this matches the original concurrent semantics).
    client = RecordingPolymarketClient()
    gate = BlockingOperatorGate()
    actuator = PolymarketActuator(
        _live_settings(),
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    first = asyncio.create_task(actuator.execute(_decision(decision_id="d-first"), _portfolio()))
    await asyncio.wait_for(gate.entered.wait(), timeout=1.0)
    second = asyncio.create_task(actuator.execute(_decision(decision_id="d-second"), _portfolio()))
    await asyncio.sleep(0)
    assert len(gate.previews) == 1

    gate.release.set()
    await asyncio.gather(first, second)

    # Lock invariant preserved (gate not entered concurrently). Whether T2's
    # gate call is observed is race-dependent — what matters is that T1 was
    # alone in the critical section while blocking.
    assert 1 <= len(gate.previews) <= 2
    assert len(client.submitted) == 2


@pytest.mark.asyncio
async def test_polymarket_actuator_converts_limit_notional_to_order_shares() -> None:
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        _live_settings(),
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    await actuator.execute(_decision(notional_usdc=10.0, limit_price=0.4), _portfolio())

    assert client.submitted[0].size == pytest.approx(25.0)
    assert client.submitted[0].notional_usdc == pytest.approx(10.0)
    assert client.submitted[0].price == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_polymarket_actuator_converts_market_sell_notional_to_shares() -> None:
    client = RecordingPolymarketClient()
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        _live_settings(),
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )
    decision = _decision(
        side=Side.SELL.value,
        action=Side.SELL.value,
        notional_usdc=12.0,
        limit_price=0.3,
    )
    object.__setattr__(decision, "order_type", "market")

    await actuator.execute(decision, _portfolio())

    assert client.submitted[0].size == pytest.approx(40.0)
    assert client.submitted[0].notional_usdc == pytest.approx(12.0)
    assert client.submitted[0].order_type == "market"


@pytest.mark.asyncio
async def test_file_first_live_order_gate_requires_exact_preview(tmp_path: Path) -> None:
    path = tmp_path / "approval.json"
    gate = FileFirstLiveOrderGate(path)
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
    )
    path.write_text(
        json.dumps(
            {
                "approved": True,
                "max_notional_usdc": 10.0,
                "venue": "polymarket",
                "market_id": "m-cp06",
                "token_id": "t-yes",
                "side": Side.BUY.value,
                "outcome": "YES",
                "limit_price": 0.4,
                "max_slippage_bps": 50,
            }
        ),
        encoding="utf-8",
    )

    assert await gate.approve_first_order(preview) is True

    path.write_text(
        json.dumps(
            {
                "approved": True,
                "max_notional_usdc": 11.0,
                "venue": "polymarket",
                "market_id": "m-cp06",
                "token_id": "t-yes",
                "side": Side.BUY.value,
                "outcome": "YES",
                "limit_price": 0.4,
                "max_slippage_bps": 50,
            }
        ),
        encoding="utf-8",
    )

    assert await gate.approve_first_order(preview) is False


@pytest.mark.asyncio
async def test_file_first_live_order_gate_rejects_duplicate_approval_json_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "approval.json"
    gate = FileFirstLiveOrderGate(path)
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
        outcome="YES",
    )
    path.write_text(
        (
            '{"approved": false, "approved": true, '
            '"max_notional_usdc": 10.0, "venue": "polymarket", '
            '"market_id": "m-cp06", "token_id": "t-yes", '
            f'"side": "{Side.BUY.value}", "outcome": "YES", '
            '"limit_price": 0.4, "max_slippage_bps": 50}'
        ),
        encoding="utf-8",
    )

    assert await gate.approve_first_order(preview) is False


@pytest.mark.asyncio
async def test_polymarket_sdk_client_posts_limit_order_through_v2_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    @dataclass
    class FakeApiCreds:
        api_key: str
        api_secret: str
        api_passphrase: str

    @dataclass
    class FakeOrderArgs:
        token_id: str
        price: float
        side: object
        size: float

    @dataclass
    class FakeMarketOrderArgs:
        token_id: str
        amount: float
        side: object
        price: float
        order_type: object

    @dataclass
    class FakePartialCreateOrderOptions:
        tick_size: object | None = None
        neg_risk: object | None = None

    class FakeOrderType:
        GTC = "GTC"
        FOK = "FOK"
        FAK = "FAK"
        GTD = "GTD"

    class FakeSide:
        BUY = "SDK_BUY"
        SELL = "SDK_SELL"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def create_and_post_order(
            self,
            *,
            order_args: FakeOrderArgs,
            options: FakePartialCreateOrderOptions,
            order_type: object,
        ) -> dict[str, object]:
            calls["limit_order"] = order_args
            calls["limit_options"] = options
            calls["limit_order_type"] = order_type
            return {"orderID": "sdk-order-1", "status": "matched", "errorMsg": ""}

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        ClobClient=FakeClobClient,
        OrderArgs=FakeOrderArgs,
        MarketOrderArgs=FakeMarketOrderArgs,
        OrderType=FakeOrderType,
        PartialCreateOrderOptions=FakePartialCreateOrderOptions,
        Side=FakeSide,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    request = PolymarketOrderRequest(
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        price=0.4,
        size=25.0,
        notional_usdc=10.0,
        estimated_quantity=25.0,
        order_type="limit",
        time_in_force=TimeInForce.GTC.value,
        max_slippage_bps=50,
    )
    result = await PolymarketSDKClient().submit_order(
        request,
        _live_settings().polymarket.credentials(),
    )

    init = cast(dict[str, object], calls["init"])
    creds = cast(FakeApiCreds, init["creds"])
    order_args = cast(FakeOrderArgs, calls["limit_order"])
    assert init["host"] == "https://clob.polymarket.com"
    assert init["chain_id"] == 137
    assert init["key"] == "private-key"
    assert init["signature_type"] == 1
    assert init["funder"] == "0x1111111111111111111111111111111111111111"
    assert creds.api_key == "api-key"
    assert creds.api_secret == "api-secret"
    assert creds.api_passphrase == "passphrase"
    assert order_args.token_id == "t-yes"
    assert order_args.price == 0.4
    assert order_args.side == "SDK_BUY"
    assert order_args.size == 25.0
    assert calls["limit_order_type"] == "GTC"
    assert result.order_id == "sdk-order-1"
    assert result.status == OrderStatus.MATCHED.value
    assert result.filled_notional_usdc == pytest.approx(10.0)
    assert result.filled_quantity == pytest.approx(25.0)


@pytest.mark.asyncio
async def test_polymarket_sdk_client_posts_market_order_with_consistent_order_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    @dataclass
    class FakeApiCreds:
        api_key: str
        api_secret: str
        api_passphrase: str

    @dataclass
    class FakeOrderArgs:
        token_id: str
        price: float
        side: object
        size: float

    @dataclass
    class FakeMarketOrderArgs:
        token_id: str
        amount: float
        side: object
        price: float
        order_type: object

    class FakePartialCreateOrderOptions:
        pass

    class FakeOrderType:
        GTC = "GTC"
        FOK = "FOK"
        FAK = "FAK"
        GTD = "GTD"

    class FakeSide:
        BUY = "SDK_BUY"
        SELL = "SDK_SELL"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def create_and_post_market_order(
            self,
            *,
            order_args: FakeMarketOrderArgs,
            options: FakePartialCreateOrderOptions,
            order_type: object,
        ) -> dict[str, object]:
            del options
            calls["market_order"] = order_args
            calls["market_order_type"] = order_type
            return {"orderID": "sdk-market-order-1", "status": "live", "errorMsg": ""}

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        ClobClient=FakeClobClient,
        OrderArgs=FakeOrderArgs,
        MarketOrderArgs=FakeMarketOrderArgs,
        OrderType=FakeOrderType,
        PartialCreateOrderOptions=FakePartialCreateOrderOptions,
        Side=FakeSide,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    await PolymarketSDKClient().submit_order(
        PolymarketOrderRequest(
            market_id="m-cp06",
            token_id="t-yes",
            side=Side.BUY.value,
            price=0.4,
            size=10.0,
            notional_usdc=10.0,
            estimated_quantity=25.0,
            order_type="market",
            time_in_force=TimeInForce.IOC.value,
            max_slippage_bps=50,
        ),
        _live_settings().polymarket.credentials(),
    )

    order_args = cast(FakeMarketOrderArgs, calls["market_order"])
    assert order_args.order_type == "FAK"
    assert calls["market_order_type"] == "FAK"


@pytest.mark.asyncio
async def test_polymarket_sdk_client_redacts_secrets_from_sdk_errors(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeApiCreds:
        def __init__(
            self,
            *,
            api_key: str,
            api_secret: str,
            api_passphrase: str,
        ) -> None:
            del api_key, api_secret, api_passphrase

    class FakeOrderArgs:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakePartialCreateOrderOptions:
        pass

    class FakeOrderType:
        GTC = "GTC"

    class FakeSide:
        BUY = "SDK_BUY"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def create_and_post_order(self, **kwargs: object) -> object:
            del kwargs
            raise RuntimeError(
                "venue rejected private-key api-key api-secret passphrase 0x1111111111111111111111111111111111111111 "
                "postgresql://admin:supersecret@db.internal.example.com/pms_live "
                "password=keyword-secret"
            )

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        ClobClient=FakeClobClient,
        OrderArgs=FakeOrderArgs,
        MarketOrderArgs=FakeOrderArgs,
        OrderType=FakeOrderType,
        PartialCreateOrderOptions=FakePartialCreateOrderOptions,
        Side=FakeSide,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)
    caplog.set_level(logging.WARNING, logger="pms.actuator.adapters.polymarket")

    with pytest.raises(LiveTradingDisabledError) as exc_info:
        await PolymarketSDKClient().submit_order(
            PolymarketOrderRequest(
                market_id="m-cp06",
                token_id="t-yes",
                side=Side.BUY.value,
                price=0.4,
                size=25.0,
                notional_usdc=10.0,
                estimated_quantity=25.0,
                order_type="limit",
                time_in_force=TimeInForce.GTC.value,
                max_slippage_bps=50,
            ),
            _live_settings().polymarket.credentials(),
        )

    message = str(exc_info.value)
    assert "private-key" not in message
    assert "api-key" not in message
    assert "api-secret" not in message
    assert "passphrase" not in message
    assert "0x1111111111111111111111111111111111111111" not in message
    assert "Polymarket live order submission failed" in message
    rendered_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "venue rejected" in rendered_logs
    assert "<redacted-database-url>" in rendered_logs
    assert "password=<redacted>" in rendered_logs
    assert "private-key" not in rendered_logs
    assert "api-key" not in rendered_logs
    assert "api-secret" not in rendered_logs
    assert "passphrase" not in rendered_logs
    assert "0x1111111111111111111111111111111111111111" not in rendered_logs
    assert "supersecret" not in rendered_logs
    assert "keyword-secret" not in rendered_logs
    assert "admin" not in rendered_logs


@pytest.mark.asyncio
async def test_actuator_feedback_appends_controller_feedback() -> None:
    store = cast(FeedbackStore, InMemoryFeedbackStore())
    generator = ActuatorFeedback(store)
    decision = _decision()

    feedback = await generator.generate(
        _order_state(decision, raw_status="insufficient_liquidity"),
        reason="insufficient_liquidity",
    )

    assert feedback.source == FeedbackSource.ACTUATOR.value
    assert feedback.target == FeedbackTarget.CONTROLLER.value
    assert feedback.category == "insufficient_liquidity"
    assert await cast(InMemoryFeedbackStore, store).all() == [feedback]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "raw_status", "expected_outcome"),
    [
        (OrderStatus.MATCHED.value, "matched", "matched"),
        ("partial", "matched", "matched"),
        ("cancelled", "ttl", "cancelled_ttl"),
        ("canceled", "limit_invalidated", "cancelled_limit_invalidated"),
        ("cancelled", "session_end", "cancelled_session_end"),
        (
            OrderStatus.CANCELED_MARKET_RESOLVED.value,
            "market_resolved_before_execution",
            "cancelled_market_resolved",
        ),
    ],
)
async def test_executor_releases_mapped_outcome_from_returned_order_state(
    status: str,
    raw_status: str,
    expected_outcome: str,
) -> None:
    store = cast(FeedbackStore, InMemoryFeedbackStore())
    decision = _decision(decision_id=f"d-{status}-{raw_status}".replace("_", "-"))
    dedup_store = RecordingDedupStore()
    returned_state = _order_state(
        decision,
        status=status,
        raw_status=raw_status,
        fill_price=decision.limit_price,
        filled_notional_usdc=decision.notional_usdc,
    )
    actuator = executor.ActuatorExecutor(
        adapter=StaticAdapter(returned_state),
        risk=RiskManager(
            RiskSettings(max_position_per_market=100.0, max_total_exposure=1000.0)
        ),
        feedback=ActuatorFeedback(store),
        dedup_store=dedup_store,
    )

    state = await actuator.execute(decision, _portfolio())

    assert state == returned_state
    assert dedup_store.acquire_calls == [decision.decision_id]
    assert dedup_store.release_calls == [(decision.decision_id, expected_outcome)]


@pytest.mark.asyncio
async def test_executor_tracks_open_status_as_live_order_lifecycle() -> None:
    store = cast(FeedbackStore, InMemoryFeedbackStore())
    decision = _decision(decision_id="d-open-lifecycle")
    returned_state = _order_state(
        decision,
        status="open",
        raw_status="open",
        filled_notional_usdc=0.0,
    )
    risk = RiskManager(
        RiskSettings(max_position_per_market=100.0, max_total_exposure=1000.0)
    )
    dedup_store = RecordingDedupStore()
    actuator = executor.ActuatorExecutor(
        adapter=StaticAdapter(returned_state),
        risk=risk,
        feedback=ActuatorFeedback(store),
        dedup_store=dedup_store,
    )

    await actuator.execute(decision, _portfolio())

    halt_state = risk.check_auto_halt(
        _portfolio(),
        now=returned_state.submitted_at + timedelta(minutes=31),
    )
    assert halt_state.halted is True
    assert halt_state.trigger_kind == "order_without_fill"
    assert dedup_store.release_calls == []


@pytest.mark.asyncio
async def test_executor_ioc_partial_fill_does_not_arm_stale_order_halt() -> None:
    """A PARTIAL fill on a non-GTC (IOC/FOK) order is terminal: the venue
    cancels the unfilled remainder on arrival, so no further fill will ever
    arrive for that order_id. Arming the 30-minute order-without-fill timer
    for it guarantees a spurious permanent auto-halt 30 minutes after any
    routine IOC partial fill."""
    store = cast(FeedbackStore, InMemoryFeedbackStore())
    decision = _decision(decision_id="d-ioc-partial")
    assert decision.time_in_force == TimeInForce.IOC
    returned_state = _order_state(
        decision,
        status=OrderStatus.PARTIAL.value,
        raw_status="partial",
        fill_price=decision.limit_price,
        filled_notional_usdc=decision.notional_usdc / 2,
    )
    risk = RiskManager(
        RiskSettings(max_position_per_market=100.0, max_total_exposure=1000.0)
    )
    actuator = executor.ActuatorExecutor(
        adapter=StaticAdapter(returned_state),
        risk=risk,
        feedback=ActuatorFeedback(store),
        dedup_store=RecordingDedupStore(),
    )

    await actuator.execute(decision, _portfolio())

    halt_state = risk.check_auto_halt(
        _portfolio(),
        now=returned_state.submitted_at + timedelta(minutes=31),
    )
    assert halt_state.halted is False, (
        "terminal IOC partial fill must not trigger order_without_fill auto-halt"
    )


@pytest.mark.asyncio
async def test_executor_gtc_partial_fill_still_arms_stale_order_halt() -> None:
    """A PARTIAL fill on a GTC order leaves the remainder resting at the
    venue, so the order-without-fill stale timer must still arm."""
    store = cast(FeedbackStore, InMemoryFeedbackStore())
    decision = replace(
        _decision(decision_id="d-gtc-partial"),
        time_in_force=TimeInForce.GTC,
    )
    returned_state = _order_state(
        decision,
        status=OrderStatus.PARTIAL.value,
        raw_status="partial",
        fill_price=decision.limit_price,
        filled_notional_usdc=decision.notional_usdc / 2,
    )
    risk = RiskManager(
        RiskSettings(max_position_per_market=100.0, max_total_exposure=1000.0)
    )
    actuator = executor.ActuatorExecutor(
        adapter=StaticAdapter(returned_state),
        risk=risk,
        feedback=ActuatorFeedback(store),
        dedup_store=RecordingDedupStore(),
    )

    await actuator.execute(decision, _portfolio())

    halt_state = risk.check_auto_halt(
        _portfolio(),
        now=returned_state.submitted_at + timedelta(minutes=31),
    )
    assert halt_state.halted is True
    assert halt_state.trigger_kind == "order_without_fill"


@pytest.mark.asyncio
async def test_executor_releases_invalid_for_risk_rejection() -> None:
    store = cast(FeedbackStore, InMemoryFeedbackStore())
    dedup_store = RecordingDedupStore()
    actuator = executor.ActuatorExecutor(
        adapter=StaticAdapter(_order_state(_decision(), status=OrderStatus.MATCHED.value)),
        risk=RiskManager(
            RiskSettings(max_position_per_market=5.0, max_total_exposure=1000.0)
        ),
        feedback=ActuatorFeedback(store),
        dedup_store=dedup_store,
    )

    state = await actuator.execute(_decision(), _portfolio())

    assert state.status == OrderStatus.INVALID.value
    assert state.raw_status == "max_position_per_market"
    assert dedup_store.release_calls == [("d-cp06", "invalid")]
    assert (await cast(InMemoryFeedbackStore, store).all())[-1].category == (
        "max_position_per_market"
    )


@pytest.mark.asyncio
async def test_executor_releases_rejected_for_insufficient_liquidity() -> None:
    store = cast(FeedbackStore, InMemoryFeedbackStore())
    dedup_store = RecordingDedupStore()
    actuator = executor.ActuatorExecutor(
        adapter=PaperActuator(orderbooks={"m-cp06": {"bids": [], "asks": []}}),
        risk=RiskManager(
            RiskSettings(max_position_per_market=100.0, max_total_exposure=1000.0)
        ),
        feedback=ActuatorFeedback(store),
        dedup_store=dedup_store,
    )

    state = await actuator.execute(_decision(decision_id="d-fail"), _portfolio())

    assert state.status == OrderStatus.INVALID.value
    assert state.raw_status == "insufficient_liquidity"
    assert dedup_store.release_calls == [("d-fail", "rejected")]
    assert (await cast(InMemoryFeedbackStore, store).all())[-1].category == (
        "insufficient_liquidity"
    )


@pytest.mark.asyncio
async def test_executor_release_failure_logs_without_masking_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = cast(FeedbackStore, InMemoryFeedbackStore())
    decision = _decision(decision_id="d-release-success")
    dedup_store = RecordingDedupStore(release_error=RuntimeError("release broke"))
    returned_state = _order_state(
        decision,
        status=OrderStatus.MATCHED.value,
        raw_status="matched",
        fill_price=decision.limit_price,
        filled_notional_usdc=decision.notional_usdc,
    )
    actuator = executor.ActuatorExecutor(
        adapter=StaticAdapter(returned_state),
        risk=RiskManager(
            RiskSettings(max_position_per_market=100.0, max_total_exposure=1000.0)
        ),
        feedback=ActuatorFeedback(store),
        dedup_store=dedup_store,
    )

    with caplog.at_level(logging.ERROR):
        state = await actuator.execute(decision, _portfolio())

    assert state == returned_state
    assert dedup_store.release_calls == [(decision.decision_id, "matched")]
    assert "Failed to release dedup state" in caplog.text


@pytest.mark.asyncio
async def test_executor_release_failure_logs_without_masking_original_adapter_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = cast(FeedbackStore, InMemoryFeedbackStore())
    decision = _decision(decision_id="d-release-error")
    dedup_store = RecordingDedupStore(release_error=RuntimeError("release broke"))
    actuator = executor.ActuatorExecutor(
        adapter=FailingAdapter(RuntimeError("venue rejected order")),
        risk=RiskManager(
            RiskSettings(max_position_per_market=100.0, max_total_exposure=1000.0)
        ),
        feedback=ActuatorFeedback(store),
        dedup_store=dedup_store,
    )

    with caplog.at_level(logging.ERROR), pytest.raises(
        RuntimeError,
        match="venue rejected order",
    ):
        await actuator.execute(decision, _portfolio())

    assert dedup_store.release_calls == [(decision.decision_id, "venue_rejection")]
    assert "Failed to release dedup state" in caplog.text


@pytest.mark.asyncio
async def test_executor_soft_release_keeps_decision_blocked_until_retention_scan() -> None:
    store = cast(FeedbackStore, InMemoryFeedbackStore())
    dedup_store = InMemoryDedupStore()
    actuator = executor.ActuatorExecutor(
        adapter=PaperActuator(orderbooks={"m-cp06": {"bids": [], "asks": []}}),
        risk=RiskManager(
            RiskSettings(max_position_per_market=100.0, max_total_exposure=1000.0)
        ),
        feedback=ActuatorFeedback(store),
        dedup_store=dedup_store,
    )

    first = await actuator.execute(_decision(decision_id="d-soft-release"), _portfolio())
    second = await actuator.execute(_decision(decision_id="d-soft-release"), _portfolio())

    assert first.raw_status == "insufficient_liquidity"
    assert second.raw_status == "duplicate_decision"
    assert dedup_store.contains("d-soft-release") is True


# --- live-readiness hardening tests (added by fix/live-readiness-hardening) ---


@dataclass
class _FailThenSucceedClient:
    """Polymarket client that raises on the first call and succeeds after."""

    requires_live_mode: bool = False
    fail_count: int = 1
    submitted: list[PolymarketOrderRequest] = field(default_factory=list)
    error: Exception = field(
        default_factory=lambda: RuntimeError("simulated network drop")
    )

    async def submit_order(
        self,
        order: PolymarketOrderRequest,
        credentials: object,
    ) -> PolymarketOrderResult:
        del credentials
        if self.fail_count > 0:
            self.fail_count -= 1
            raise self.error
        self.submitted.append(order)
        return PolymarketOrderResult(
            order_id="pm-live-recovered",
            status=OrderStatus.MATCHED.value,
            raw_status="matched",
            filled_notional_usdc=order.notional_usdc,
            remaining_notional_usdc=0.0,
            fill_price=order.price,
            filled_quantity=order.estimated_quantity,
        )


@pytest.mark.asyncio
async def test_polymarket_actuator_does_not_mark_approved_when_submit_fails() -> None:
    """SECURITY: a failed first submit must not permanently bypass the gate.

    Pre-fix: `_approval_state.approved = True` was set inside the gate flow,
    so a network failure on the first venue call left the gate permanently
    open — every subsequent decision skipped operator approval.
    """
    client = _FailThenSucceedClient(fail_count=1)
    gate = RecordingOperatorGate(approved=True)
    actuator = PolymarketActuator(
        _live_settings(),
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(RuntimeError, match="simulated network drop"):
        await actuator.execute(_decision(decision_id="d-fail"), _portfolio())

    # Gate was consulted, but submit failed — the floodgate must remain shut.
    assert len(gate.previews) == 1
    assert actuator._first_order_approved() is False
    # And no consume happened, since no submit succeeded.
    assert gate.consumed == []

    # Next attempt MUST go through the gate again.
    state = await actuator.execute(_decision(decision_id="d-retry"), _portfolio())
    assert state.status == OrderStatus.MATCHED.value
    assert len(gate.previews) == 2
    assert actuator._first_order_approved() is True
    # Now the gate has been consumed.
    assert len(gate.consumed) == 1


@pytest.mark.asyncio
async def test_file_first_live_order_gate_consumes_file_after_first_success(
    tmp_path: Path,
) -> None:
    """The approval JSON file must be removed after a successful first submit
    so an attacker / stale file cannot replay it."""
    approval_path = tmp_path / "approval.json"
    preview_payload = {
        "approved": True,
        "max_notional_usdc": 10.0,
        "venue": "polymarket",
        "market_id": "m-cp06",
        "token_id": "t-yes",
        "side": Side.BUY.value,
        "outcome": "YES",
        "limit_price": 0.4,
        "max_slippage_bps": 50,
    }
    approval_path.write_text(json.dumps(preview_payload), encoding="utf-8")

    client = RecordingPolymarketClient()
    actuator = PolymarketActuator(
        _live_settings(),
        client=client,
        operator_gate=FileFirstLiveOrderGate(approval_path),
        quote_provider=AllowQuoteProvider(),
    )

    state = await actuator.execute(_decision(), _portfolio())

    assert state.status == OrderStatus.MATCHED.value
    assert approval_path.exists() is False, "approval file must be unlinked after use"


@pytest.mark.asyncio
async def test_file_first_live_order_gate_uses_isclose_for_floats(
    tmp_path: Path,
) -> None:
    """Approval JSON copied from preview values may pick up float
    representation drift (e.g. 0.1+0.2). The gate must accept tiny drift,
    while still rejecting any meaningful difference."""
    path = tmp_path / "approval.json"
    gate = FileFirstLiveOrderGate(path)
    preview = LiveOrderPreview(
        max_notional_usdc=0.1 + 0.2,  # 0.30000000000000004
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
    )
    # Operator wrote the rounded value 0.3, the actuator computed 0.1 + 0.2.
    path.write_text(
        json.dumps(
            {
                "approved": True,
                "max_notional_usdc": 0.3,
                "venue": "polymarket",
                "market_id": "m-cp06",
                "token_id": "t-yes",
                "side": Side.BUY.value,
                "outcome": "YES",
                "limit_price": 0.4,
                "max_slippage_bps": 50,
            }
        ),
        encoding="utf-8",
    )
    assert await gate.approve_first_order(preview) is True

    # But a meaningful difference is still rejected.
    path.write_text(
        json.dumps(
            {
                "approved": True,
                "max_notional_usdc": 0.31,
                "venue": "polymarket",
                "market_id": "m-cp06",
                "token_id": "t-yes",
                "side": Side.BUY.value,
                "outcome": "YES",
                "limit_price": 0.4,
                "max_slippage_bps": 50,
            }
        ),
        encoding="utf-8",
    )
    assert await gate.approve_first_order(preview) is False


@pytest.mark.asyncio
async def test_polymarket_sdk_client_propagates_timeout_as_unknown_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout in the SDK call means the order MAY have reached the venue.
    Wrapping it as `LiveTradingDisabledError` ('disabled, nothing sent') is
    misleading. Surface as `PolymarketSubmissionUnknownError` so operators
    know to reconcile."""

    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeOrderArgs:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakePartialCreateOrderOptions:
        pass

    class FakeOrderType:
        GTC = "GTC"

    class FakeSide:
        BUY = "SDK_BUY"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def create_and_post_order(self, **kwargs: object) -> object:
            del kwargs
            raise asyncio.TimeoutError()

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        ClobClient=FakeClobClient,
        OrderArgs=FakeOrderArgs,
        MarketOrderArgs=FakeOrderArgs,
        OrderType=FakeOrderType,
        PartialCreateOrderOptions=FakePartialCreateOrderOptions,
        Side=FakeSide,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    with pytest.raises(PolymarketSubmissionUnknownError) as exc_info:
        await PolymarketSDKClient().submit_order(
            PolymarketOrderRequest(
                market_id="m-cp06",
                token_id="t-yes",
                side=Side.BUY.value,
                price=0.4,
                size=25.0,
                notional_usdc=10.0,
                estimated_quantity=25.0,
                order_type="limit",
                time_in_force=TimeInForce.GTC.value,
                max_slippage_bps=50,
            ),
            _live_settings().polymarket.credentials(),
        )

    message = str(exc_info.value)
    assert "reconcile" in message.lower()


@pytest.mark.asyncio
async def test_polymarket_sdk_client_maps_limit_ioc_to_fak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A limit order with TIF=IOC must map to FAK, not silently demote to
    GTC (which would rest in the book instead of being killed unfilled)."""
    calls: dict[str, object] = {}

    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    @dataclass
    class FakeOrderArgs:
        token_id: str
        price: float
        side: object
        size: float

    class FakePartialCreateOrderOptions:
        pass

    class FakeOrderType:
        GTC = "GTC"
        FOK = "FOK"
        FAK = "FAK"
        GTD = "GTD"

    class FakeSide:
        BUY = "SDK_BUY"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def create_and_post_order(
            self,
            *,
            order_args: FakeOrderArgs,
            options: FakePartialCreateOrderOptions,
            order_type: object,
        ) -> dict[str, object]:
            del options
            calls["order_args"] = order_args
            calls["order_type"] = order_type
            return {"orderID": "x", "status": "matched", "errorMsg": ""}

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        ClobClient=FakeClobClient,
        OrderArgs=FakeOrderArgs,
        MarketOrderArgs=FakeOrderArgs,
        OrderType=FakeOrderType,
        PartialCreateOrderOptions=FakePartialCreateOrderOptions,
        Side=FakeSide,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    await PolymarketSDKClient().submit_order(
        PolymarketOrderRequest(
            market_id="m-cp06",
            token_id="t-yes",
            side=Side.BUY.value,
            price=0.4,
            size=25.0,
            notional_usdc=10.0,
            estimated_quantity=25.0,
            order_type="limit",
            time_in_force=TimeInForce.IOC.value,
            max_slippage_bps=50,
        ),
        _live_settings().polymarket.credentials(),
    )

    assert calls["order_type"] == "FAK", "limit + IOC must map to FAK"


def test_risk_manager_rejects_above_max_quantity_shares() -> None:
    """RiskSettings.max_quantity_shares clamps the low-price-token blow-up
    case (e.g. limit_price=0.001 → notional/$10 = 10,000 shares)."""
    manager = RiskManager(
        RiskSettings(
            max_position_per_market=100.0,
            max_total_exposure=1000.0,
            max_quantity_shares=5_000.0,
        )
    )

    # 10 USDC at price 0.001 = 10,000 shares — above the 5,000 cap.
    over_cap = manager.check(
        _decision(notional_usdc=10.0, limit_price=0.001),
        _portfolio(),
    )
    assert over_cap.approved is False
    assert over_cap.reason == "max_quantity_shares"

    # Same notional at a normal price stays under the cap.
    in_cap = manager.check(
        _decision(notional_usdc=10.0, limit_price=0.4),
        _portfolio(),
    )
    assert in_cap.approved is True


def test_risk_manager_quantity_cap_disabled_by_default() -> None:
    """Default RiskSettings has no max_quantity_shares — preserves
    backward-compatible behaviour for users who haven't opted in."""
    manager = RiskManager(
        RiskSettings(max_position_per_market=100.0, max_total_exposure=1000.0)
    )
    decision = _decision(notional_usdc=10.0, limit_price=0.001)
    assert manager.check(decision, _portfolio()).approved is True


# --- review-loop round-1 follow-ups (codex findings f2 and f3) ---


@pytest.mark.asyncio
async def test_executor_categorizes_submission_unknown_distinctly_from_venue_rejection() -> None:
    """Codex finding f2: a `PolymarketSubmissionUnknownError` (timeout)
    must be released distinctly from `venue_rejection` so retries do not
    green-light a duplicate submission of an order that may already be
    on the venue."""
    from pms.actuator.adapters.polymarket import PolymarketSubmissionUnknownError

    @dataclass
    class _UnknownTimeoutAdapter:
        calls: int = 0

        async def execute(
            self,
            decision: TradeDecision,
            portfolio: Portfolio | None = None,
        ) -> OrderState:
            del decision, portfolio
            self.calls += 1
            raise PolymarketSubmissionUnknownError(
                "Polymarket live order submission timed out; reconcile"
            )

    in_memory_store = InMemoryFeedbackStore()
    store = cast(FeedbackStore, in_memory_store)
    dedup = InMemoryDedupStore()
    timeout_adapter = _UnknownTimeoutAdapter()
    actuator = executor.ActuatorExecutor(
        adapter=cast(executor.ActuatorAdapter, timeout_adapter),
        risk=RiskManager(
            RiskSettings(max_position_per_market=100.0, max_total_exposure=1000.0)
        ),
        feedback=ActuatorFeedback(store),
        dedup_store=dedup,
    )

    with pytest.raises(PolymarketSubmissionUnknownError):
        await actuator.execute(_decision(decision_id="d-unknown"), _portfolio())

    feedbacks = await in_memory_store.all()
    submission_unknown = [
        f for f in feedbacks if getattr(f, "category", None) == "submission_unknown"
    ]
    venue_rejection = [
        f for f in feedbacks if getattr(f, "category", None) == "venue_rejection"
    ]
    # The point of this test: the unknown timeout must NOT be logged as
    # venue_rejection; it gets its own category.
    assert len(submission_unknown) == 1
    assert len(venue_rejection) == 0


@pytest.mark.asyncio
async def test_polymarket_sdk_parses_partial_fill_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f3: an SDK response with explicit `filled_notional`
    and status != MATCHED (e.g. IOC limit with partial match) must
    surface positive fill values — not get silently dropped."""

    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeOrderArgs:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakePartialCreateOrderOptions:
        pass

    class FakeOrderType:
        FAK = "FAK"
        FOK = "FOK"
        GTC = "GTC"
        GTD = "GTD"

    class FakeSide:
        BUY = "SDK_BUY"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def create_and_post_order(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            # IOC limit that filled $4 of a $10 order, half a share at 0.5,
            # and was then cancelled. Status is `live`/`cancelled` post-IOC.
            return {
                "orderID": "sdk-partial-1",
                "status": "live",
                "errorMsg": "",
                "filled_notional_usdc": 4.0,
                "filled_quantity": 8.0,
                "fill_price": 0.5,
            }

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        ClobClient=FakeClobClient,
        OrderArgs=FakeOrderArgs,
        MarketOrderArgs=FakeOrderArgs,
        OrderType=FakeOrderType,
        PartialCreateOrderOptions=FakePartialCreateOrderOptions,
        Side=FakeSide,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    result = await PolymarketSDKClient().submit_order(
        PolymarketOrderRequest(
            market_id="m-cp06",
            token_id="t-yes",
            side=Side.BUY.value,
            price=0.5,
            size=20.0,
            notional_usdc=10.0,
            estimated_quantity=20.0,
            order_type="limit",
            time_in_force=TimeInForce.IOC.value,
            max_slippage_bps=50,
        ),
        _live_settings().polymarket.credentials(),
    )

    # Pre-fix: filled_notional would be 0 because status != "matched".
    assert result.filled_notional_usdc == pytest.approx(4.0)
    assert result.filled_quantity == pytest.approx(8.0)
    assert result.fill_price == pytest.approx(0.5)
    assert result.remaining_notional_usdc == pytest.approx(6.0)
    assert result.status == "live"


@pytest.mark.asyncio
async def test_polymarket_sdk_normalizes_open_status_to_live_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "sdk-open-resting-1",
            "status": "open",
            "success": True,
            "errorMsg": "",
        },
    )

    result = await PolymarketSDKClient().submit_order(
        PolymarketOrderRequest(
            market_id="m-cp06",
            token_id="t-yes",
            side=Side.BUY.value,
            price=0.5,
            size=20.0,
            notional_usdc=10.0,
            estimated_quantity=20.0,
            order_type="limit",
            time_in_force=TimeInForce.IOC.value,
            max_slippage_bps=50,
        ),
        _live_settings().polymarket.credentials(),
    )

    assert result.status == OrderStatus.LIVE.value
    assert result.raw_status == "open"
    assert result.remaining_notional_usdc == pytest.approx(10.0)


def test_fill_from_order_emits_fill_for_partial_status() -> None:
    """Codex finding f3, runner side: `_fill_from_order` must emit a
    FillRecord whenever filled_notional > 0 AND fill_price > 0 — not
    only on status == MATCHED."""
    from pms.runner import _fill_from_order

    decision = _decision(notional_usdc=10.0)
    partial_fill_state = OrderState(
        order_id="sdk-partial-1",
        decision_id=decision.decision_id,
        status="live",  # partial fill, not matched
        market_id=decision.market_id,
        token_id=decision.token_id,
        venue=decision.venue,
        requested_notional_usdc=decision.notional_usdc,
        filled_notional_usdc=4.0,
        remaining_notional_usdc=6.0,
        fill_price=0.5,
        submitted_at=datetime(2026, 4, 25, tzinfo=UTC),
        last_updated_at=datetime(2026, 4, 25, tzinfo=UTC),
        raw_status="live",
        strategy_id=decision.strategy_id,
        strategy_version_id=decision.strategy_version_id,
        filled_quantity=8.0,
    )

    fill = _fill_from_order(partial_fill_state, decision, None)

    assert fill is not None, "partial fill must produce a FillRecord"
    assert fill.fill_notional_usdc == pytest.approx(4.0)
    assert fill.fill_quantity == pytest.approx(8.0)
    assert fill.fill_price == pytest.approx(0.5)
    assert fill.status == "live"


def test_fill_from_order_drops_zero_filled_state() -> None:
    """Sanity: status=live with NO fill data must still drop (no fill
    record for a resting limit order that hasn't matched anything)."""
    from pms.runner import _fill_from_order

    decision = _decision(notional_usdc=10.0)
    no_fill_state = OrderState(
        order_id="sdk-resting-1",
        decision_id=decision.decision_id,
        status="live",
        market_id=decision.market_id,
        token_id=decision.token_id,
        venue=decision.venue,
        requested_notional_usdc=decision.notional_usdc,
        filled_notional_usdc=0.0,
        remaining_notional_usdc=decision.notional_usdc,
        fill_price=None,
        submitted_at=datetime(2026, 4, 25, tzinfo=UTC),
        last_updated_at=datetime(2026, 4, 25, tzinfo=UTC),
        raw_status="live",
        strategy_id=decision.strategy_id,
        strategy_version_id=decision.strategy_version_id,
        filled_quantity=0.0,
    )

    assert _fill_from_order(no_fill_state, decision, None) is None


def _filled_state(decision: object) -> OrderState:
    d = cast("TradeDecision", decision)
    return OrderState(
        order_id="paper-fee-1",
        decision_id=d.decision_id,
        status="matched",
        market_id=d.market_id,
        token_id=d.token_id,
        venue=d.venue,
        requested_notional_usdc=d.notional_usdc,
        filled_notional_usdc=4.0,
        remaining_notional_usdc=0.0,
        fill_price=0.5,
        submitted_at=datetime(2026, 4, 25, tzinfo=UTC),
        last_updated_at=datetime(2026, 4, 25, tzinfo=UTC),
        raw_status="matched",
        strategy_id=d.strategy_id,
        strategy_version_id=d.strategy_version_id,
        filled_quantity=8.0,
    )


def test_fill_from_order_applies_configured_fee_rate() -> None:
    """Paper/backtest fills must carry the configured Polymarket fee so the
    evaluator's net-edge gate (average_net_edge_bps) can compute instead of
    rendering N/A. Fee matches ExecutionModel.compute_fee:
    notional * fee_rate * (1 - fill_price); fee_bps is the nominal rate."""
    from pms.runner import _fill_from_order

    decision = _decision(notional_usdc=10.0)
    fill = _fill_from_order(
        _filled_state(decision), decision, None, fee_rate=0.04
    )

    assert fill is not None
    # 4.0 notional * 0.04 * (1 - 0.5) = 0.08
    assert fill.fees == pytest.approx(0.08)
    assert fill.fee_bps == 400


def test_fill_from_order_uses_signal_fee_rate_bps_when_available() -> None:
    """Paper fills should use market fee evidence carried by the signal.

    Polymarket fees are market/category specific, so a global fallback rate can
    corrupt paper P&L and net-edge evidence when the feed includes fee_rate_bps.
    """
    from pms.runner import _fill_from_order

    decision = _decision(notional_usdc=10.0)
    signal = MarketSignal(
        market_id=decision.market_id,
        token_id=decision.token_id,
        venue=decision.venue,
        title="Will signal fees be used?",
        yes_price=0.5,
        volume_24h=1000.0,
        resolves_at=datetime(2026, 4, 26, tzinfo=UTC),
        orderbook={},
        external_signal={"fee_rate_bps": "300"},
        fetched_at=datetime(2026, 4, 25, tzinfo=UTC),
        market_status="open",
    )
    fill = _fill_from_order(
        _filled_state(decision),
        decision,
        signal,
        fee_rate=0.07,
    )

    assert fill is not None
    # 4.0 notional * 0.03 * (1 - 0.5) = 0.06
    assert fill.fees == pytest.approx(0.06)
    assert fill.fee_bps == 300


def test_fill_from_order_omits_fee_when_rate_is_none() -> None:
    """LIVE fills (fee_rate=None) leave fees unset; venue reconciliation owns
    the real fee. Default preserves the prior fee-free behaviour."""
    from pms.runner import _fill_from_order

    decision = _decision(notional_usdc=10.0)
    fill = _fill_from_order(_filled_state(decision), decision, None)

    assert fill is not None
    assert fill.fees is None
    assert fill.fee_bps is None


# --- review-loop round-2 follow-ups (codex findings f7 and f8) ---


def test_coerce_float_or_none_rejects_nan_and_infinity() -> None:
    """Codex finding f8: a venue response surfacing `NaN` / `inf` (in
    floats or strings) must produce `None`, not corrupt the persisted
    fill with a non-finite numeric value."""
    from pms.actuator.adapters.polymarket import _coerce_float_or_none

    assert _coerce_float_or_none(float("nan")) is None
    assert _coerce_float_or_none(float("inf")) is None
    assert _coerce_float_or_none(float("-inf")) is None
    assert _coerce_float_or_none("nan") is None
    assert _coerce_float_or_none("inf") is None
    assert _coerce_float_or_none("Infinity") is None
    # Sanity: well-formed values still pass through.
    assert _coerce_float_or_none(0.0) == 0.0
    assert _coerce_float_or_none(0.5) == 0.5
    assert _coerce_float_or_none("0.5") == 0.5
    assert _coerce_float_or_none(True) is None  # bool subclass of int
    assert _coerce_float_or_none("nope") is None


def _partial_fill_request() -> PolymarketOrderRequest:
    return PolymarketOrderRequest(
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        price=0.5,
        size=20.0,
        notional_usdc=10.0,
        estimated_quantity=20.0,
        order_type="limit",
        time_in_force=TimeInForce.GTC.value,
        max_slippage_bps=50,
    )


@pytest.mark.asyncio
async def test_polymarket_partial_fill_rejects_overfilled_notional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f8: filled_notional > requested_notional must be
    rejected — the venue cannot fill more than was ordered."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "live",
            "filled_notional_usdc": 15.0,  # > 10.0 requested
            "filled_quantity": 30.0,
            "fill_price": 0.5,
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="exceeds requested"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
async def test_polymarket_partial_fill_rejects_negative_filled_quantity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f8: negative filled_quantity is malformed and must
    be rejected."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "live",
            "filled_notional_usdc": 4.0,
            "filled_quantity": -8.0,
            "fill_price": 0.5,
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="negative filled_quantity"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
async def test_polymarket_partial_fill_rejects_out_of_range_fill_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f8: fill_price outside (0, 1] is invalid for a
    Polymarket probability market and must be rejected."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "live",
            "filled_notional_usdc": 4.0,
            "filled_quantity": 8.0,
            "fill_price": 1.5,  # impossible probability
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="outside"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


# --- review-loop round-3 follow-ups (codex findings f9 and f10) ---


@pytest.mark.asyncio
async def test_polymarket_matched_status_rejects_invalid_explicit_quantity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f9: even when status is `matched` (the backwards-
    compat fallback), a venue response with an invalid explicit field
    must be rejected — not silently persisted via the fallback path."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "matched",
            # No explicit_filled_notional → matched fallback would
            # synthesize a full fill, but the explicit (negative)
            # quantity must still trigger validation.
            "filled_quantity": -8.0,
            "fill_price": 0.5,
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="negative filled_quantity"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
async def test_polymarket_matched_status_rejects_invalid_explicit_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f9: matched fallback must also reject an explicit
    out-of-range fill_price."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "matched",
            "fill_price": 1.5,  # impossible probability
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="outside"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
async def test_polymarket_matched_status_rejects_negative_explicit_notional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f9: matched fallback must reject negative explicit
    filled_notional (which previously fell through the `> 0.0` filter
    into the matched-fallback branch and was silently ignored)."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "matched",
            "filled_notional_usdc": -4.0,
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="negative filled_notional"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


# --- review-loop round-4 follow-ups (codex finding f11) ---


@pytest.mark.asyncio
async def test_polymarket_partial_fill_derives_quantity_when_only_notional_and_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f11: a venue response with filled_notional > 0 and
    fill_price > 0 but NO filled_quantity must derive quantity from
    notional/price — not silently persist filled_quantity=0 (which
    would corrupt share accounting)."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "live",
            "filled_notional_usdc": 4.0,
            "fill_price": 0.5,
            # filled_quantity intentionally omitted
        },
    )

    result = await PolymarketSDKClient().submit_order(
        _partial_fill_request(),
        _live_settings().polymarket.credentials(),
    )

    assert result.filled_notional_usdc == pytest.approx(4.0)
    # 4.0 / 0.5 = 8.0 shares — derived, not silent zero.
    assert result.filled_quantity == pytest.approx(8.0)
    assert result.fill_price == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_polymarket_partial_fill_rejects_zero_quantity_with_positive_notional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f11: an explicit `filled_quantity == 0` paired
    with positive `filled_notional` is contradictory (a $4 fill of zero
    shares) and must be rejected, not persisted."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "live",
            "filled_notional_usdc": 4.0,
            "filled_quantity": 0.0,  # contradicts positive notional
            "fill_price": 0.5,
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="filled_quantity == 0"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


# --- review-loop round-5 follow-ups (codex finding f12) ---


@pytest.mark.asyncio
async def test_polymarket_matched_status_rejects_explicit_zero_notional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f12: status=matched with `filled_notional_usdc: 0`
    is contradictory — the matched-fallback heuristic must NOT
    synthesize a $10 / fully-filled response from an explicit zero."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "matched",
            "filled_notional_usdc": 0.0,
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="inconsistent"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
async def test_polymarket_matched_status_rejects_partial_fields_without_notional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f12: status=matched with explicit
    filled_quantity / fill_price but NO filled_notional must be
    rejected — the matched-fallback would otherwise synthesize a
    notional from the order while using the explicit zero quantity,
    persisting a $10 fill of 0 shares."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "matched",
            "filled_quantity": 0.0,
            "fill_price": 0.5,
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="inconsistent"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
async def test_polymarket_matched_status_synthesizes_full_fill_only_when_no_explicit_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f12 (positive case): status=matched WITHOUT any
    explicit fill fields IS the legitimate backwards-compat path — the
    full-fill heuristic should still fire to support venues that
    surface only `status: matched` without per-fill detail."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "matched",
            # No explicit fill fields at all.
        },
    )

    result = await PolymarketSDKClient().submit_order(
        _partial_fill_request(),
        _live_settings().polymarket.credentials(),
    )

    assert result.filled_notional_usdc == pytest.approx(10.0)
    assert result.filled_quantity == pytest.approx(20.0)  # 10 / 0.5
    assert result.fill_price == pytest.approx(0.5)


# --- review-loop round-6 follow-ups (codex finding f13) ---


@pytest.mark.asyncio
async def test_polymarket_matched_status_rejects_malformed_filled_notional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f13: a venue response with a PRESENT-BUT-MALFORMED
    fill field (e.g. `"nan"`, `"inf"`, `null`) must be rejected — not
    treated as 'no explicit fields' and silently routed into the
    matched-fallback full-fill synthesis path."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "matched",
            "filled_notional_usdc": "nan",  # present but unparseable
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="unparseable filled_notional"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
async def test_polymarket_matched_status_rejects_malformed_filled_quantity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f13: malformed filled_quantity (here `"inf"`)
    must be rejected, not silently dropped into the matched fallback."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "matched",
            "filled_quantity": "inf",
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="unparseable filled_quantity"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
async def test_polymarket_matched_status_rejects_null_fill_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f13: a JSON null in a present field is also
    'present but unparseable' and must be rejected."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "matched",
            "fill_price": None,  # JSON null
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="unparseable fill_price"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


# --- review-loop round-7 follow-ups (codex finding f14, partial accept) ---


@pytest.mark.asyncio
async def test_polymarket_recognizes_fill_count_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex finding f14: `fill_count` is documented at
    `docs/research/schema-spec.md:284,307` as a venue-side contract
    count alias for filled_quantity. Recognising it ensures malformed
    `fill_count` values (e.g. "nan") get routed through the
    raw-presence rejection path rather than being silently ignored."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "matched",
            "fill_count": "nan",
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="unparseable filled_quantity"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
async def test_polymarket_recognizes_fill_count_as_valid_quantity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: a well-formed `fill_count` is accepted as the
    canonical filled_quantity, just like `filled_quantity` itself."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "live",
            "filled_notional_usdc": 4.0,
            "fill_count": 8.0,
            "fill_price": 0.5,
        },
    )

    result = await PolymarketSDKClient().submit_order(
        _partial_fill_request(),
        _live_settings().polymarket.credentials(),
    )

    assert result.filled_quantity == pytest.approx(8.0)
    assert result.filled_notional_usdc == pytest.approx(4.0)


# --- review-loop fresh-final follow-up (codex SDK-wrapped timeout finding) ---


@pytest.mark.asyncio
async def test_polymarket_sdk_polyapiexception_no_resp_routes_as_submission_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh-final consensus finding: `py_clob_client_v2` wraps httpx
    request-level timeouts into `PolyApiException(resp=None, ...)`.
    The bare `TimeoutError` catch alone misses these — so a REAL POST
    timeout would still flow as `LiveTradingDisabledError` =
    venue_rejection, not `submission_unknown`. This test pins the
    correct routing."""

    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeOrderArgs:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakePartialCreateOrderOptions:
        pass

    class FakeOrderType:
        FAK = "FAK"
        FOK = "FOK"
        GTC = "GTC"
        GTD = "GTD"

    class FakeSide:
        BUY = "SDK_BUY"

    class PolyApiException(Exception):
        """Mirror the REAL SDK class shape — verified against
        `py_clob_client_v2==1.0.0`: stores `status_code` (extracted from
        `resp.status_code`, or None if resp is None) plus `error_msg`.
        Crucially, does NOT retain `resp` as an attribute. An earlier
        version of this fake exposed `self.resp` and silently diverged
        from the real SDK — caught by the fresh-final consensus pass."""

        def __init__(
            self,
            resp: object | None = None,
            error_msg: str | None = None,
        ) -> None:
            super().__init__(error_msg or "Request exception!")
            # Match real SDK: extract status_code from resp (if any),
            # do NOT store resp itself.
            self.status_code: int | None = (
                getattr(resp, "status_code", None) if resp is not None else None
            )
            self.error_msg = error_msg

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def create_and_post_order(self, **kwargs: object) -> object:
            del kwargs
            raise PolyApiException(resp=None, error_msg="Request exception!")

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        ClobClient=FakeClobClient,
        OrderArgs=FakeOrderArgs,
        MarketOrderArgs=FakeOrderArgs,
        OrderType=FakeOrderType,
        PartialCreateOrderOptions=FakePartialCreateOrderOptions,
        Side=FakeSide,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    with pytest.raises(PolymarketSubmissionUnknownError, match="transport failure"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
async def test_polymarket_sdk_polyapiexception_with_resp_routes_as_venue_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inverse control: PolyApiException WITH a populated `resp` (i.e.
    venue actually responded with an HTTP error) is a real
    venue_rejection, NOT submission_unknown. Routing must distinguish."""

    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeOrderArgs:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakePartialCreateOrderOptions:
        pass

    class FakeOrderType:
        GTC = "GTC"
        FOK = "FOK"
        FAK = "FAK"
        GTD = "GTD"

    class FakeSide:
        BUY = "SDK_BUY"

    class FakeHttpResponse:
        status_code = 400
        text = "venue rejected"

    class PolyApiException(Exception):
        """Same real-SDK-mirror shape — `status_code` populated from
        `resp.status_code` when resp is present, plus `error_msg`."""

        def __init__(
            self,
            resp: object | None = None,
            error_msg: str | None = None,
        ) -> None:
            super().__init__(error_msg or "")
            self.status_code: int | None = (
                getattr(resp, "status_code", None) if resp is not None else None
            )
            self.error_msg = error_msg

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def create_and_post_order(self, **kwargs: object) -> object:
            del kwargs
            raise PolyApiException(resp=FakeHttpResponse(), error_msg="rejected")

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        ClobClient=FakeClobClient,
        OrderArgs=FakeOrderArgs,
        MarketOrderArgs=FakeOrderArgs,
        OrderType=FakeOrderType,
        PartialCreateOrderOptions=FakePartialCreateOrderOptions,
        Side=FakeSide,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)

    # Should raise LiveTradingDisabledError (venue_rejection), NOT
    # PolymarketSubmissionUnknownError. Use the broader exception type
    # to confirm that and assert the message hints at venue error.
    with pytest.raises(LiveTradingDisabledError) as exc_info:
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )
    assert not isinstance(exc_info.value, PolymarketSubmissionUnknownError)
    assert "venue error redacted" in str(exc_info.value)


@pytest.mark.parametrize("success_value", ["false", "False", "0", 0])
@pytest.mark.asyncio
async def test_polymarket_sdk_rejects_non_boolean_false_success_flags(
    monkeypatch: pytest.MonkeyPatch,
    success_value: object,
) -> None:
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "sdk-rejected-flag",
            "status": "matched",
            "success": success_value,
            "errorMsg": "",
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="rejected by venue"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
async def test_polymarket_sdk_rejects_unparseable_success_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "sdk-unparseable-success",
            "status": "matched",
            "success": "definitely",
            "errorMsg": "",
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="unparseable success flag"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.parametrize(
    "response_overrides",
    [
        {},
        {"orderID": ""},
        {"orderID": "   "},
        {"orderID": "__FILL_IN_ORDER_ID__"},
    ],
)
@pytest.mark.asyncio
async def test_polymarket_sdk_success_without_concrete_venue_order_id_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
    response_overrides: dict[str, object],
) -> None:
    response = {
        "status": "matched",
        "success": True,
        "errorMsg": "",
    } | response_overrides
    _install_partial_fill_sdk(monkeypatch, response=response)

    with pytest.raises(PolymarketSubmissionUnknownError, match="venue order id"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


def _install_partial_fill_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: dict[str, object],
) -> None:
    """Wires a fake `py_clob_client_v2` whose limit-order endpoint
    returns the given response. Used by the f8 validation tests."""

    class FakeApiCreds:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakeOrderArgs:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    class FakePartialCreateOrderOptions:
        pass

    class FakeOrderType:
        FAK = "FAK"
        FOK = "FOK"
        GTC = "GTC"
        GTD = "GTD"

    class FakeSide:
        BUY = "SDK_BUY"

    class FakeClobClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def create_and_post_order(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return response

    fake_module = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        ClobClient=FakeClobClient,
        OrderArgs=FakeOrderArgs,
        MarketOrderArgs=FakeOrderArgs,
        OrderType=FakeOrderType,
        PartialCreateOrderOptions=FakePartialCreateOrderOptions,
        Side=FakeSide,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake_module)


# ---------------------------------------------------------------------------
# Regression: first-order approval race (codex P2 / CodeRabbit Critical)
# ---------------------------------------------------------------------------


@dataclass
class _SlowPolymarketClient:
    """Client whose `submit_order` blocks on an `asyncio.Event` until
    `release` is fired. Used to widen the post-approval / pre-commit
    window so a second concurrent task can attempt to re-use the same
    approval.
    """

    requires_live_mode: bool = False
    release: asyncio.Event = field(default_factory=asyncio.Event)
    entered_submit: asyncio.Event = field(default_factory=asyncio.Event)
    submitted: list[PolymarketOrderRequest] = field(default_factory=list)

    async def submit_order(
        self,
        order: PolymarketOrderRequest,
        credentials: object,
    ) -> PolymarketOrderResult:
        del credentials
        self.submitted.append(order)
        self.entered_submit.set()
        await self.release.wait()
        return PolymarketOrderResult(
            order_id=f"pm-live-{len(self.submitted)}",
            status=OrderStatus.MATCHED.value,
            raw_status="matched",
            filled_notional_usdc=10.0,
            remaining_notional_usdc=0.0,
            fill_price=0.4,
            filled_quantity=25.0,
        )


@dataclass
class _SequencedQuoteProvider:
    quotes: list[LivePreSubmitQuote]
    calls: int = 0

    async def quote(
        self,
        order: PolymarketOrderRequest,
        credentials: object,
    ) -> LivePreSubmitQuote:
        del order, credentials
        index = min(self.calls, len(self.quotes) - 1)
        self.calls += 1
        return self.quotes[index]


@pytest.mark.asyncio
async def test_polymarket_actuator_lock_held_through_submit_blocks_concurrent_reuse(
    tmp_path: Path,
) -> None:
    """Regression for codex P2 / CodeRabbit Critical: prior implementation
    released `_approval_lock` before `submit_order()` returned and only
    set `_approval_state.approved=True` afterwards, leaving a window
    where a second concurrent task could re-read the same approval file
    (which `consume()` had not yet unlinked) and submit a parallel
    first-order trade with one operator approval.

    The fix holds the lock across approval + submit + commit + consume.
    With the fix:
      * T1 enters, takes lock, gets gate approval, blocks in submit.
      * T2 enters, blocks on the lock.
      * T1 completes submit, sets approved=True, consumes the file,
        releases the lock.
      * T2 acquires the lock, the double-check sees approved=True, and
        T2 submits via the post-approval path WITHOUT re-reading the
        (now-unlinked) approval file.
    Net: gate.approve_first_order is called exactly once even though
    both tasks make a first-order submit.
    """
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "approved": True,
                "max_notional_usdc": 10.0,
                "venue": "polymarket",
                "market_id": "m-cp06",
                "token_id": "t-yes",
                "side": Side.BUY.value,
                "outcome": "YES",
                "limit_price": 0.4,
                "max_slippage_bps": 50,
            }
        ),
        encoding="utf-8",
    )

    @dataclass(frozen=True)
    class _CountingFileGate:
        """Wrapper that delegates to FileFirstLiveOrderGate while
        counting approve calls. Frozen dataclass to satisfy the
        FirstLiveOrderGate protocol the actuator expects."""

        inner: FileFirstLiveOrderGate
        calls: list[int] = field(default_factory=list)

        async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
            self.calls.append(1)
            return await self.inner.approve_first_order(preview)

        async def consume(self, preview: LiveOrderPreview) -> None:
            await self.inner.consume(preview)

    gate = _CountingFileGate(inner=FileFirstLiveOrderGate(approval_path))

    client = _SlowPolymarketClient()
    actuator = PolymarketActuator(
        _live_settings(),
        client=client,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    first = asyncio.create_task(
        actuator.execute(_decision(decision_id="d-first"), _portfolio())
    )
    # Wait for T1 to enter submit_order — it now holds the approval lock
    # *and* is blocked inside the venue call.
    await asyncio.wait_for(client.entered_submit.wait(), timeout=1.0)

    second = asyncio.create_task(
        actuator.execute(_decision(decision_id="d-second"), _portfolio())
    )
    # Give the event loop a chance to run T2; with the lock-through-submit
    # fix it MUST block on `_approval_lock` and not have submitted yet.
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(client.submitted) == 1, (
        "second task must not submit while first holds the approval lock"
    )

    # Now allow T1 to finish submit. It will set approved=True, consume
    # the file, release the lock, and T2 will fast-path through.
    client.release.set()
    await asyncio.gather(first, second)

    # Both submits eventually happened.
    assert len(client.submitted) == 2
    # And the gate was called exactly once — T2 must NOT have re-read
    # the approval file. This is the load-bearing assertion.
    assert len(gate.calls) == 1
    # The approval file was consumed by T1's success.
    assert approval_path.exists() is False


@pytest.mark.asyncio
async def test_second_waiter_after_first_order_approval_still_uses_quote_guard(
    tmp_path: Path,
) -> None:
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "approved": True,
                "max_notional_usdc": 10.0,
                "venue": "polymarket",
                "market_id": "m-cp06",
                "token_id": "t-yes",
                "side": Side.BUY.value,
                "outcome": "YES",
                "limit_price": 0.4,
                "max_slippage_bps": 50,
            }
        ),
        encoding="utf-8",
    )
    valid_quote = LivePreSubmitQuote(
        market_status="open",
        book_age_ms=25.0,
        executable_notional_usdc=10.0,
        best_executable_price=0.4,
        spread_bps=10.0,
        quote_hash="quote-first",
        book_ts=datetime(2026, 4, 26, tzinfo=UTC),
    )
    stale_quote = LivePreSubmitQuote(
        market_status="open",
        book_age_ms=10_000.0,
        executable_notional_usdc=10.0,
        best_executable_price=0.4,
        spread_bps=10.0,
        quote_hash="quote-second-stale",
        book_ts=datetime(2026, 4, 26, tzinfo=UTC),
    )
    quote_provider = _SequencedQuoteProvider([valid_quote, stale_quote])
    client = _SlowPolymarketClient()
    actuator = PolymarketActuator(
        _live_settings(),
        client=client,
        operator_gate=FileFirstLiveOrderGate(approval_path),
        quote_provider=quote_provider,
    )

    first = asyncio.create_task(
        actuator.execute(_decision(decision_id="d-first"), _portfolio())
    )
    await asyncio.wait_for(client.entered_submit.wait(), timeout=1.0)

    second = asyncio.create_task(
        actuator.execute(_decision(decision_id="d-second"), _portfolio())
    )
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(client.submitted) == 1

    client.release.set()
    first_state = await first
    with pytest.raises(LiveTradingDisabledError, match="book is stale"):
        await second

    assert first_state.pre_submit_quote["quote_hash"] == "quote-first"
    assert quote_provider.calls == 2
    assert len(client.submitted) == 1


@pytest.mark.asyncio
async def test_polymarket_actuator_failed_first_submit_re_prompts_gate(
    tmp_path: Path,
) -> None:
    """Consume-on-success preservation under the new lock scope.

    The lock-through-submit fix must NOT regress the consume-on-success
    semantic: if `submit_order()` raises while we hold the lock, the
    approval flag stays False and a subsequent caller must be re-
    prompted for operator approval (rather than silently re-using a
    stale flag or being permanently locked out).
    """
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "approved": True,
                "max_notional_usdc": 10.0,
                "venue": "polymarket",
                "market_id": "m-cp06",
                "token_id": "t-yes",
                "side": Side.BUY.value,
                "outcome": "YES",
                "limit_price": 0.4,
                "max_slippage_bps": 50,
            }
        ),
        encoding="utf-8",
    )
    gate = FileFirstLiveOrderGate(approval_path)

    @dataclass
    class _FailingClient:
        requires_live_mode: bool = False
        attempts: int = 0

        async def submit_order(
            self,
            order: PolymarketOrderRequest,
            credentials: object,
        ) -> PolymarketOrderResult:
            del order, credentials
            self.attempts += 1
            raise LiveTradingDisabledError("simulated venue rejection")

    failing = _FailingClient()
    actuator = PolymarketActuator(
        _live_settings(),
        client=failing,
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )

    with pytest.raises(LiveTradingDisabledError):
        await actuator.execute(_decision(decision_id="d-first"), _portfolio())
    # Approval flag must still be False — submit failed, do not commit.
    assert actuator._first_order_approved() is False  # noqa: SLF001
    # Approval file was NOT consumed because the submit failed (the
    # operator can retry without re-writing the file).
    assert approval_path.exists() is True

    # Second attempt must succeed by re-reading the approval file.
    @dataclass
    class _PassingClient:
        requires_live_mode: bool = False

        async def submit_order(
            self,
            order: PolymarketOrderRequest,
            credentials: object,
        ) -> PolymarketOrderResult:
            del order, credentials
            return PolymarketOrderResult(
                order_id="pm-live-retry",
                status=OrderStatus.MATCHED.value,
                raw_status="matched",
                filled_notional_usdc=10.0,
                remaining_notional_usdc=0.0,
                fill_price=0.4,
                filled_quantity=25.0,
            )

    actuator2 = PolymarketActuator(
        _live_settings(),
        client=_PassingClient(),
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
    )
    state = await actuator2.execute(_decision(decision_id="d-retry"), _portfolio())
    assert state.order_id == "pm-live-retry"
    assert actuator2._first_order_approved() is True  # noqa: SLF001
    # And now consumed.
    assert approval_path.exists() is False


# ---------------------------------------------------------------------------
# Regression: cross-field fill consistency (codex P1 / CodeRabbit Major)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_polymarket_partial_fill_rejects_inconsistent_notional_quantity_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for codex P1 / CodeRabbit Major: a venue response with
    all three explicit fill fields set must satisfy
    `notional ≈ quantity * price` within rounding tolerance. Without
    cross-field validation the example
        (filled_notional_usdc=4, filled_quantity=100, fill_price=0.5)
    where the true notional should be 50 — a 92% miss — passes
    individual range validation and corrupts share accounting.
    """
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "matched",
            "filled_notional_usdc": 4.0,
            "filled_quantity": 100.0,
            "fill_price": 0.5,  # 100 * 0.5 == 50, NOT 4
        },
    )

    with pytest.raises(LiveTradingDisabledError, match="inconsistent fill triple"):
        await PolymarketSDKClient().submit_order(
            _partial_fill_request(),
            _live_settings().polymarket.credentials(),
        )


@pytest.mark.asyncio
async def test_polymarket_partial_fill_accepts_consistent_triple_within_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive path: a triple where notional ≈ quantity * price within
    1% / $0.01 tolerance must pass the cross-field check. Polymarket
    CLOB rounding can introduce sub-cent drift; the validator must not
    reject those."""
    # 20 shares * 0.5 = 10.0 exactly — well within tolerance.
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "matched",
            "filled_notional_usdc": 10.0,
            "filled_quantity": 20.0,
            "fill_price": 0.5,
        },
    )

    result = await PolymarketSDKClient().submit_order(
        _partial_fill_request(),
        _live_settings().polymarket.credentials(),
    )
    assert result.filled_notional_usdc == pytest.approx(10.0)
    assert result.filled_quantity == pytest.approx(20.0)
    assert result.fill_price == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_polymarket_partial_fill_accepts_sub_cent_rounding_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sub-cent rounding drift in the venue response (e.g. quantity
    rounded to whole shares) must not be rejected. 20.001 shares at
    0.5 = 10.0005, reported as notional=10.0 — well within $0.01
    absolute tolerance."""
    _install_partial_fill_sdk(
        monkeypatch,
        response={
            "orderID": "x",
            "status": "matched",
            "filled_notional_usdc": 10.0,
            "filled_quantity": 20.001,
            "fill_price": 0.5,
        },
    )

    result = await PolymarketSDKClient().submit_order(
        _partial_fill_request(),
        _live_settings().polymarket.credentials(),
    )
    assert result.filled_notional_usdc == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# STO-10 cp-02: first-order audit emission
# ---------------------------------------------------------------------------


@dataclass
class RecordingFirstOrderAuditWriter:
    events: list[tuple[str, LiveOrderPreview, str | None]] = field(default_factory=list)
    raise_on_event: str | None = None
    error_message: str | None = None

    async def record_event(
        self,
        *,
        event: str,
        preview: LiveOrderPreview,
        approver_id: str | None = None,
    ) -> None:
        self.events.append((event, preview, approver_id))
        if self.raise_on_event == event:
            message = self.error_message or f"audit write failed: {event}"
            raise RuntimeError(message)


@pytest.mark.asyncio
async def test_polymarket_actuator_emits_audit_on_gate_match_and_consume() -> None:
    """STO-10 cp-02: happy path emits approval_matched then approval_consumed
    with the same preview, so a forensic walker can reconstruct what was
    authorized and confirm the consume completed."""
    writer = RecordingFirstOrderAuditWriter()
    actuator = PolymarketActuator(
        _live_settings(),
        client=RecordingPolymarketClient(),
        operator_gate=RecordingOperatorGate(approved=True),
        quote_provider=AllowQuoteProvider(),
        audit_writer=writer,
    )

    state = await actuator.execute(_decision(), _portfolio())

    assert state.status == OrderStatus.MATCHED.value
    assert [event for event, _, _ in writer.events] == [
        "approval_matched",
        "approval_consumed",
    ]
    matched_preview = writer.events[0][1]
    consumed_preview = writer.events[1][1]
    assert matched_preview == consumed_preview
    assert matched_preview.market_id == "m-cp06"
    assert matched_preview.token_id == "t-yes"
    assert matched_preview.side == Side.BUY.value
    assert matched_preview.max_notional_usdc == 10.0


@pytest.mark.asyncio
async def test_polymarket_actuator_emits_audit_on_gate_denial() -> None:
    """STO-10 cp-02: denial path emits approval_denied with the preview
    that was rejected, so the audit log records every gate consultation
    even when the gate refuses."""
    writer = RecordingFirstOrderAuditWriter()
    actuator = PolymarketActuator(
        _live_settings(),
        client=RecordingPolymarketClient(),
        operator_gate=RecordingOperatorGate(approved=False),
        quote_provider=AllowQuoteProvider(),
        audit_writer=writer,
    )

    with pytest.raises(OperatorApprovalRequiredError):
        await actuator.execute(_decision(), _portfolio())

    assert [event for event, _, _ in writer.events] == ["approval_denied"]
    assert writer.events[0][1].market_id == "m-cp06"


@pytest.mark.asyncio
async def test_polymarket_actuator_audit_writer_failure_does_not_break_submit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """STO-10 cp-02: an audit-writer failure must NOT block the submit
    or leave the gate in an inconsistent state. Audit failure logs WARN
    and the order proceeds. Mirrors precedent at runner.py:1319-1320 where
    LiveEmergencyAuditWriter failures degrade gracefully rather than
    interrupting the trading hot path."""
    writer = RecordingFirstOrderAuditWriter(raise_on_event="approval_matched")
    actuator = PolymarketActuator(
        _live_settings(),
        client=RecordingPolymarketClient(),
        operator_gate=RecordingOperatorGate(approved=True),
        quote_provider=AllowQuoteProvider(),
        audit_writer=writer,
    )

    with caplog.at_level(logging.WARNING, logger="pms.actuator.adapters.polymarket"):
        state = await actuator.execute(_decision(), _portfolio())

    assert state.status == OrderStatus.MATCHED.value
    # approval_matched emit raised; approval_consumed must still fire so the
    # audit log records the eventual consume even when the matched-event
    # write failed transiently.
    assert [event for event, _, _ in writer.events] == [
        "approval_matched",
        "approval_consumed",
    ]
    assert any(
        "first-order audit" in record.message and "approval_matched" in record.message
        for record in caplog.records
    ), (
        "expected WARN log mentioning approval_matched failure, got: "
        f"{[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_polymarket_actuator_redacts_audit_writer_failure_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    writer = RecordingFirstOrderAuditWriter(
        raise_on_event="approval_matched",
        error_message=_secret_bearing_error_message("audit sink failed"),
    )
    actuator = PolymarketActuator(
        _live_settings_with_secret_credentials(),
        client=RecordingPolymarketClient(),
        operator_gate=RecordingOperatorGate(approved=True),
        quote_provider=AllowQuoteProvider(),
        audit_writer=writer,
    )

    with caplog.at_level(logging.WARNING, logger="pms.actuator.adapters.polymarket"):
        state = await actuator.execute(_decision(), _portfolio())

    assert state.status == OrderStatus.MATCHED.value
    rendered = "\n".join(record.message for record in caplog.records)
    _assert_live_secrets_redacted(rendered)


@pytest.mark.asyncio
async def test_polymarket_actuator_redacts_read_approver_id_failure_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @dataclass
    class _ApproverReaderRaisesGate:
        previews: list[LiveOrderPreview] = field(default_factory=list)

        async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
            self.previews.append(preview)
            return True

        async def consume(self, preview: LiveOrderPreview) -> None:
            del preview

        def read_approver_id(self) -> str:
            raise RuntimeError(
                _secret_bearing_error_message("approver sidecar read failed")
            )

    actuator = PolymarketActuator(
        _live_settings_with_secret_credentials(),
        client=RecordingPolymarketClient(),
        operator_gate=_ApproverReaderRaisesGate(),
        quote_provider=AllowQuoteProvider(),
    )

    with caplog.at_level(logging.WARNING, logger="pms.actuator.adapters.polymarket"):
        state = await actuator.execute(_decision(), _portfolio())

    assert state.status == OrderStatus.MATCHED.value
    rendered = "\n".join(record.message for record in caplog.records)
    _assert_live_secrets_redacted(rendered)


# ---------------------------------------------------------------------------
# STO-10 follow-up: sidecar approver_id reader on FileFirstLiveOrderGate
# ---------------------------------------------------------------------------


def _approval_payload(*, market_id: str = "m-cp06") -> dict[str, object]:
    return {
        "approved": True,
        "max_notional_usdc": 10.0,
        "venue": "polymarket",
        "market_id": market_id,
        "token_id": "t-yes",
        "side": Side.BUY.value,
        "outcome": "YES",
        "limit_price": 0.4,
        "max_slippage_bps": 50,
    }


def _approval_payload_hash(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _approval_sidecar_payload(
    approval_payload: dict[str, object],
    *,
    approver_id: str = "operator-alice",
    ts: datetime | None = None,
) -> dict[str, object]:
    return {
        "approver_id": approver_id,
        "approval_sha256": _approval_payload_hash(approval_payload),
        "ts": (ts or datetime.now(tz=UTC)).isoformat(),
    }


def _sidecar_path(approval_path: Path) -> Path:
    return Path(str(approval_path) + ".meta.json")


def test_file_first_live_order_gate_reads_approver_id_from_sidecar(
    tmp_path: Path,
) -> None:
    """STO-10 sidecar: approver_id from <path>.meta.json is captured so
    the audit log records who authorized the order."""
    approval_path = tmp_path / "approval.json"
    _sidecar_path(approval_path).write_text(
        json.dumps({"approver_id": "operator-alice", "ts": "2026-05-07T00:00:00Z"}),
        encoding="utf-8",
    )

    gate = FileFirstLiveOrderGate(approval_path)

    assert gate.read_approver_id() == "operator-alice"


def test_file_first_live_order_gate_returns_none_when_sidecar_missing(
    tmp_path: Path,
) -> None:
    """No sidecar → approver_id is None (degraded but unblocked: the
    runbook's gate's job is to authorize the trade, not to enforce
    metadata hygiene)."""
    gate = FileFirstLiveOrderGate(tmp_path / "approval.json")

    assert gate.read_approver_id() is None


def test_file_first_live_order_gate_returns_none_when_sidecar_malformed(
    tmp_path: Path,
) -> None:
    """Malformed sidecar (invalid JSON, wrong shape, non-string
    approver_id) must degrade to None without raising."""
    approval_path = tmp_path / "approval.json"

    _sidecar_path(approval_path).write_text("not-json", encoding="utf-8")
    assert FileFirstLiveOrderGate(approval_path).read_approver_id() is None

    _sidecar_path(approval_path).write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert FileFirstLiveOrderGate(approval_path).read_approver_id() is None

    _sidecar_path(approval_path).write_text(
        json.dumps({"approver_id": 42}), encoding="utf-8"
    )
    assert FileFirstLiveOrderGate(approval_path).read_approver_id() is None


def test_file_first_live_order_gate_returns_none_for_duplicate_sidecar_json_key(
    tmp_path: Path,
) -> None:
    approval_path = tmp_path / "approval.json"
    _sidecar_path(approval_path).write_text(
        (
            '{"approver_id": "operator-mallory", '
            '"approver_id": "operator-alice", '
            '"ts": "2026-05-07T00:00:00Z"}'
        ),
        encoding="utf-8",
    )

    assert FileFirstLiveOrderGate(approval_path).read_approver_id() is None


@pytest.mark.asyncio
async def test_file_first_live_order_gate_requires_sidecar_when_configured(
    tmp_path: Path,
) -> None:
    approval_path = tmp_path / "approval.json"
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
        outcome="YES",
    )
    gate = FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=300.0,
    )

    assert await gate.approve_first_order(preview) is False

    _sidecar_path(approval_path).write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )

    assert await gate.approve_first_order(preview) is True


@pytest.mark.asyncio
async def test_file_first_live_order_gate_opens_artifacts_with_no_follow_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    if no_follow_flag == 0:
        pytest.skip("os.O_NOFOLLOW is unavailable on this platform")

    approval_path = tmp_path / "approval.json"
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )
    observed_flags: list[int] = []
    real_open = os.open

    def recording_open(
        path_arg: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        observed_flags.append(flags)
        return real_open(path_arg, flags, mode)

    monkeypatch.setattr(os, "open", recording_open)
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
        outcome="YES",
    )
    gate = FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=300.0,
    )

    assert await gate.approve_first_order(preview) is True
    assert len(observed_flags) == 2
    assert all(flags & no_follow_flag for flags in observed_flags)


@pytest.mark.asyncio
async def test_file_first_live_order_gate_rejects_permissive_approval_parent(
    tmp_path: Path,
) -> None:
    approval_dir = tmp_path / "permissive-approval"
    approval_dir.mkdir(mode=0o700)
    approval_dir.chmod(0o755)
    approval_path = approval_dir / "approval.json"
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
        outcome="YES",
    )
    gate = FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=300.0,
    )

    try:
        assert await gate.approve_first_order(preview) is False
    finally:
        approval_dir.chmod(0o700)


@pytest.mark.asyncio
async def test_file_first_live_order_gate_rejects_symlink_approval_parent(
    tmp_path: Path,
) -> None:
    approval_dir = tmp_path / "approval-target"
    approval_dir.mkdir(mode=0o700)
    approval_path = approval_dir / "approval.json"
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )
    symlink_parent = tmp_path / "approval-parent-link"
    symlink_parent.symlink_to(approval_dir, target_is_directory=True)
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
        outcome="YES",
    )
    gate = FileFirstLiveOrderGate(
        symlink_parent / "approval.json",
        require_approver_sidecar=True,
        approval_max_age_s=300.0,
    )

    assert await gate.approve_first_order(preview) is False


@pytest.mark.asyncio
async def test_file_first_live_order_gate_rejects_symlink_approval_file(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "target-approval.json"
    target_path.write_text(json.dumps(_approval_payload()), encoding="utf-8")
    approval_path = tmp_path / "approval.json"
    approval_path.symlink_to(target_path)
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
        outcome="YES",
    )
    gate = FileFirstLiveOrderGate(approval_path)

    assert await gate.approve_first_order(preview) is False


@pytest.mark.asyncio
async def test_file_first_live_order_gate_rejects_hardlinked_approval_file(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "target-approval.json"
    approval_payload = _approval_payload()
    target_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    approval_path = tmp_path / "approval.json"
    os.link(target_path, approval_path)
    _sidecar_path(approval_path).write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
        outcome="YES",
    )
    gate = FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=300.0,
    )

    assert await gate.approve_first_order(preview) is False


@pytest.mark.asyncio
async def test_file_first_live_order_gate_rejects_symlink_sidecar(
    tmp_path: Path,
) -> None:
    approval_path = tmp_path / "approval.json"
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    sidecar_target = tmp_path / "sidecar-target.json"
    sidecar_target.write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )
    _sidecar_path(approval_path).symlink_to(sidecar_target)
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
        outcome="YES",
    )
    gate = FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=300.0,
    )

    assert await gate.approve_first_order(preview) is False


@pytest.mark.asyncio
async def test_file_first_live_order_gate_rejects_hardlinked_sidecar(
    tmp_path: Path,
) -> None:
    approval_path = tmp_path / "approval.json"
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    sidecar_target = tmp_path / "sidecar-target.json"
    sidecar_target.write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )
    os.link(sidecar_target, _sidecar_path(approval_path))
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
        outcome="YES",
    )
    gate = FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=300.0,
    )

    assert await gate.approve_first_order(preview) is False


@pytest.mark.asyncio
async def test_file_first_live_order_gate_rejects_stale_sidecar_when_configured(
    tmp_path: Path,
) -> None:
    approval_path = tmp_path / "approval.json"
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps(
            _approval_sidecar_payload(
                approval_payload,
                ts=datetime.now(tz=UTC) - timedelta(seconds=301),
            )
        ),
        encoding="utf-8",
    )
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
        outcome="YES",
    )
    gate = FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=300.0,
    )

    assert await gate.approve_first_order(preview) is False


@pytest.mark.asyncio
async def test_file_first_live_order_gate_rejects_placeholder_approver_id(
    tmp_path: Path,
) -> None:
    approval_path = tmp_path / "approval.json"
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps(
            _approval_sidecar_payload(
                approval_payload,
                approver_id="__FILL_IN_OPERATOR_ID__",
            )
        ),
        encoding="utf-8",
    )
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
        outcome="YES",
    )
    gate = FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=300.0,
    )

    assert await gate.approve_first_order(preview) is False


@pytest.mark.asyncio
async def test_file_first_live_order_gate_requires_sidecar_hash_match(
    tmp_path: Path,
) -> None:
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(_approval_payload()), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps(
            {
                "approver_id": "operator-alice",
                "approval_sha256": "not-the-approval-payload-hash",
                "ts": datetime.now(tz=UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
        outcome="YES",
    )
    gate = FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=300.0,
    )

    assert await gate.approve_first_order(preview) is False


@pytest.mark.asyncio
async def test_file_first_live_order_gate_consume_unlinks_sidecar(
    tmp_path: Path,
) -> None:
    """STO-10 sidecar: consume() must unlink the sidecar alongside the
    approval JSON so a stale `<path>.meta.json` cannot linger and
    misattribute a future authorization to the previous approver."""
    approval_path = tmp_path / "approval.json"
    sidecar = _sidecar_path(approval_path)
    approval_path.write_text(json.dumps(_approval_payload()), encoding="utf-8")
    sidecar.write_text(json.dumps({"approver_id": "operator-alice"}), encoding="utf-8")

    gate = FileFirstLiveOrderGate(approval_path)
    preview = LiveOrderPreview(
        max_notional_usdc=10.0,
        venue="polymarket",
        market_id="m-cp06",
        token_id="t-yes",
        side=Side.BUY.value,
        limit_price=0.4,
        max_slippage_bps=50,
    )

    await gate.consume(preview)

    assert approval_path.exists() is False
    assert sidecar.exists() is False


@pytest.mark.asyncio
async def test_polymarket_actuator_audit_includes_approver_id_from_sidecar(
    tmp_path: Path,
) -> None:
    """STO-10 follow-up: end-to-end the approver_id flows from the
    operator's sidecar file into every audit record (matched and
    consumed). This closes the loop on 'who authorized this order' for
    forensic walkers."""
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(_approval_payload()), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps({"approver_id": "operator-alice"}), encoding="utf-8"
    )

    writer = RecordingFirstOrderAuditWriter()
    actuator = PolymarketActuator(
        _live_settings(),
        client=RecordingPolymarketClient(),
        operator_gate=FileFirstLiveOrderGate(approval_path),
        quote_provider=AllowQuoteProvider(),
        audit_writer=writer,
    )

    state = await actuator.execute(_decision(), _portfolio())

    assert state.status == OrderStatus.MATCHED.value
    # Both matched and consumed events must carry the approver_id;
    # capturing once before consume() unlinks the sidecar guarantees
    # consumed sees the same approver as matched.
    assert [
        (event, approver_id) for event, _, approver_id in writer.events
    ] == [
        ("approval_matched", "operator-alice"),
        ("approval_consumed", "operator-alice"),
    ]


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_when_strict_sidecar_disappears(
    tmp_path: Path,
) -> None:
    """If strict sidecar provenance disappears after approve() returns
    but before audit attribution, the actuator must fail closed instead
    of submitting an unattributed live order."""

    @dataclass(frozen=True)
    class _DeletingSidecarGate(FileFirstLiveOrderGate):
        async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
            approved = await super().approve_first_order(preview)
            if approved:
                self._sidecar_path().unlink()
            return approved

    approval_path = tmp_path / "approval.json"
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )
    client = RecordingPolymarketClient()
    writer = RecordingFirstOrderAuditWriter()
    actuator = PolymarketActuator(
        _live_settings(operator_approval_mode="every_order"),
        client=client,
        operator_gate=_DeletingSidecarGate(
            approval_path,
            require_approver_sidecar=True,
            approval_max_age_s=300.0,
        ),
        quote_provider=AllowQuoteProvider(),
        audit_writer=writer,
    )

    with pytest.raises(OperatorApprovalRequiredError, match="approval"):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert [(event, approver_id) for event, _, approver_id in writer.events] == [
        ("approval_denied", None)
    ]


@pytest.mark.asyncio
async def test_polymarket_actuator_rejects_when_strict_sidecar_is_replaced(
    tmp_path: Path,
) -> None:
    """If strict sidecar provenance changes after approve() returns,
    the actuator must fail closed instead of attributing a live order to
    an operator id that was not part of the validated authorization."""

    @dataclass(frozen=True)
    class _ReplacingSidecarGate(FileFirstLiveOrderGate):
        async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
            approved = await super().approve_first_order(preview)
            if approved:
                self._sidecar_path().write_text(
                    json.dumps(
                        _approval_sidecar_payload(
                            _approval_payload(),
                            approver_id="operator-mallory",
                        )
                    ),
                    encoding="utf-8",
                )
            return approved

    approval_path = tmp_path / "approval.json"
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    _sidecar_path(approval_path).write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )
    client = RecordingPolymarketClient()
    writer = RecordingFirstOrderAuditWriter()
    actuator = PolymarketActuator(
        _live_settings(operator_approval_mode="every_order"),
        client=client,
        operator_gate=_ReplacingSidecarGate(
            approval_path,
            require_approver_sidecar=True,
            approval_max_age_s=300.0,
        ),
        quote_provider=AllowQuoteProvider(),
        audit_writer=writer,
    )

    with pytest.raises(OperatorApprovalRequiredError, match="approval"):
        await actuator.execute(_decision(), _portfolio())

    assert client.submitted == []
    assert [(event, approver_id) for event, _, approver_id in writer.events] == [
        ("approval_denied", None)
    ]


@pytest.mark.asyncio
async def test_polymarket_actuator_emits_consume_failed_when_consume_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """STO-10 review-loop f1: when `consume()` raises after a successful
    submit, the audit log must NOT claim the approval was cleanly
    consumed — the artifact may still be on disk and could replay on
    restart. Emit a distinct `approval_consume_failed` event instead so
    forensic walkers can separate "successfully consumed" from "consumed
    but cleanup failed", and log at ERROR rather than WARN.

    The fast path (`_approval_state.approved=True`) MUST stay set after
    submit succeeds — flipping it back would cause the *next* in-process
    decision to re-prompt the gate and double-submit on the same
    approval file. The cleanup-failure handling is forensic, not
    transactional."""

    @dataclass
    class _ConsumeRaisesGate:
        consume_error: Exception
        previews: list[LiveOrderPreview] = field(default_factory=list)

        async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
            self.previews.append(preview)
            return True

        async def consume(self, preview: LiveOrderPreview) -> None:
            del preview
            raise self.consume_error

    writer = RecordingFirstOrderAuditWriter()
    gate = _ConsumeRaisesGate(
        consume_error=OSError(
            _secret_bearing_error_message("approval consume failed")
        )
    )
    actuator = PolymarketActuator(
        _live_settings_with_secret_credentials(),
        client=RecordingPolymarketClient(),
        operator_gate=gate,
        quote_provider=AllowQuoteProvider(),
        audit_writer=writer,
    )

    with caplog.at_level(logging.ERROR, logger="pms.actuator.adapters.polymarket"):
        state = await actuator.execute(_decision(), _portfolio())

    # Submit succeeded; the in-process fast path must stay open so the
    # next decision doesn't re-trigger the gate.
    assert state.status == OrderStatus.MATCHED.value
    assert actuator._first_order_approved() is True

    # Audit log records the failure honestly, not a clean consume.
    events = [event for event, _, _ in writer.events]
    assert events == ["approval_matched", "approval_consume_failed"]
    assert "approval_consumed" not in events

    # ERROR-level log fires on consume failure (not the previous WARN).
    assert any(
        record.levelname == "ERROR" and "consume" in record.message.lower()
        for record in caplog.records
    ), (
        "expected ERROR log on consume failure, got: "
        f"{[(r.levelname, r.message) for r in caplog.records]}"
    )
    _assert_live_secrets_redacted(
        "\n".join(record.message for record in caplog.records)
    )


@pytest.mark.asyncio
async def test_polymarket_actuator_audit_approver_id_is_none_when_sidecar_absent(
    tmp_path: Path,
) -> None:
    """Regression: actuator must still emit cleanly when no sidecar
    exists (the unattributed-but-authorized fallback path)."""
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(_approval_payload()), encoding="utf-8")

    writer = RecordingFirstOrderAuditWriter()
    actuator = PolymarketActuator(
        _live_settings(),
        client=RecordingPolymarketClient(),
        operator_gate=FileFirstLiveOrderGate(approval_path),
        quote_provider=AllowQuoteProvider(),
        audit_writer=writer,
    )

    state = await actuator.execute(_decision(), _portfolio())

    assert state.status == OrderStatus.MATCHED.value
    assert [
        (event, approver_id) for event, _, approver_id in writer.events
    ] == [
        ("approval_matched", None),
        ("approval_consumed", None),
    ]
