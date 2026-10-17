from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
from typing import Any, cast

import httpx
import pytest

from pms.api.app import create_app
from pms.config import PMSSettings
from pms.core.enums import RunMode
from pms.runner import Runner
from pms.storage.market_subscription_store import MarketSubscriptionRow


@dataclass
class _SubscriptionStoreDouble:
    upsert_results: list[MarketSubscriptionRow | None] = field(default_factory=list)
    delete_results: list[bool] = field(default_factory=list)
    user_subscription_results: list[set[str]] = field(default_factory=list)
    upsert_calls: list[str] = field(default_factory=list)
    delete_calls: list[str] = field(default_factory=list)

    async def upsert_user_subscription(
        self,
        token_id: str,
    ) -> MarketSubscriptionRow | None:
        self.upsert_calls.append(token_id)
        return self.upsert_results.pop(0)

    async def delete_user_subscription(self, token_id: str) -> bool:
        self.delete_calls.append(token_id)
        return self.delete_results.pop(0)

    async def read_user_subscriptions(self) -> set[str]:
        if self.user_subscription_results:
            return self.user_subscription_results.pop(0)
        return set()


def _row(
    *,
    token_id: str = "token-yes",
    condition_id: str = "market-1",
    created_at: datetime | None = None,
) -> MarketSubscriptionRow:
    return MarketSubscriptionRow(
        token_id=token_id,
        condition_id=condition_id,
        source="user",
        created_at=created_at or datetime(2026, 4, 24, 9, 0, tzinfo=UTC),
    )


def _app(
    monkeypatch: pytest.MonkeyPatch,
    store: _SubscriptionStoreDouble,
) -> Any:
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.BACKTEST,
            auto_migrate_default_v2=False,
        )
    )
    runner._pg_pool = cast(Any, object())  # noqa: SLF001
    monkeypatch.setattr(
        "pms.api.app.PostgresMarketSubscriptionStore",
        lambda _: store,
    )
    return create_app(runner, auto_start=False)


@pytest.mark.asyncio
async def test_subscribe_endpoint_upserts_user_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _SubscriptionStoreDouble(upsert_results=[_row()])
    reselection_calls: list[str] = []

    async def request_reselection(self: Runner) -> None:
        del self
        reselection_calls.append("subscribe")

    monkeypatch.setattr(Runner, "_request_reselection", request_reselection)
    app = _app(monkeypatch, store)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/markets/token-yes/subscribe")

    assert response.status_code == 200
    assert response.json() == {
        "token_id": "token-yes",
        "source": "user",
        "created_at": "2026-04-24T09:00:00+00:00",
        "effective": True,
    }
    assert store.upsert_calls == ["token-yes"]
    assert reselection_calls == ["subscribe"]


@pytest.mark.asyncio
async def test_subscribe_endpoint_idempotent_returns_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row()
    store = _SubscriptionStoreDouble(upsert_results=[row, row])
    app = _app(monkeypatch, store)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        first = await client.post("/markets/token-yes/subscribe")
        second = await client.post("/markets/token-yes/subscribe")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == {
        "token_id": "token-yes",
        "source": "user",
        "created_at": "2026-04-24T09:00:00+00:00",
        "effective": True,
    }
    assert store.upsert_calls == ["token-yes", "token-yes"]


@pytest.mark.asyncio
async def test_subscribe_endpoint_404_on_unknown_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _SubscriptionStoreDouble(upsert_results=[None])
    app = _app(monkeypatch, store)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/markets/missing-token/subscribe")

    assert response.status_code == 404
    assert response.json() == {"detail": "Token not found"}
    assert store.upsert_calls == ["missing-token"]


@pytest.mark.asyncio
async def test_unsubscribe_endpoint_deletes_row_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _SubscriptionStoreDouble(delete_results=[True])
    reselection_calls: list[str] = []

    async def request_reselection(self: Runner) -> None:
        del self
        reselection_calls.append("unsubscribe")

    monkeypatch.setattr(Runner, "_request_reselection", request_reselection)
    app = _app(monkeypatch, store)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.delete("/markets/token-yes/subscribe")

    assert response.status_code == 200
    assert response.json() == {"token_id": "token-yes", "deleted": True}
    assert store.delete_calls == ["token-yes"]
    assert reselection_calls == ["unsubscribe"]


@pytest.mark.asyncio
async def test_unsubscribe_endpoint_returns_false_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _SubscriptionStoreDouble(delete_results=[False])
    app = _app(monkeypatch, store)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.delete("/markets/token-yes/subscribe")

    assert response.status_code == 200
    assert response.json() == {"token_id": "token-yes", "deleted": False}
    assert store.delete_calls == ["token-yes"]


@pytest.mark.asyncio
async def test_subscribe_endpoint_emits_structured_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _SubscriptionStoreDouble(upsert_results=[_row()])
    app = _app(monkeypatch, store)
    caplog.set_level(logging.INFO, logger="pms.api.routes.market_subscriptions")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/markets/token-yes/subscribe")

    assert response.status_code == 200
    records = [
        record
        for record in caplog.records
        if record.name == "pms.api.routes.market_subscriptions"
    ]
    assert len(records) == 1
    assert records[0].message == "subscription.user_add"
    assert getattr(records[0], "token_id") == "token-yes"
    assert getattr(records[0], "condition_id") == "market-1"
    assert getattr(records[0], "request_method") == "POST"
    assert getattr(records[0], "request_path") == "/markets/token-yes/subscribe"


@pytest.mark.asyncio
async def test_subscribe_endpoint_surfaces_pin_dropped_by_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = {f"token-{index}" for index in range(3)}
    store = _SubscriptionStoreDouble(
        upsert_results=[_row(token_id="token-yes")],
        user_subscription_results=[rows],
    )
    app = _app(monkeypatch, store)
    app.state.runner.config.sensor.max_subscription_asset_ids = 2

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/markets/token-yes/subscribe")

    assert response.status_code == 200
    assert response.json()["effective"] is False
