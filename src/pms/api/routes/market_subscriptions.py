from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol, cast

from pydantic import BaseModel

from pms.storage.market_subscription_store import MarketSubscriptionRow


logger = logging.getLogger(__name__)


class UnknownSubscriptionTokenError(Exception):
    def __init__(self, token_id: str) -> None:
        super().__init__(token_id)
        self.token_id = token_id


class SubscriptionStore(Protocol):
    async def upsert_user_subscription(
        self,
        token_id: str,
    ) -> MarketSubscriptionRow | None: ...

    async def delete_user_subscription(self, token_id: str) -> bool: ...

    async def read_user_subscriptions(self) -> set[str]: ...


class SubscribeMarketResponse(BaseModel):
    token_id: str
    source: str
    created_at: str
    effective: bool


class UnsubscribeMarketResponse(BaseModel):
    token_id: str
    deleted: bool


async def subscribe_market(
    store: SubscriptionStore,
    *,
    token_id: str,
    request_metadata: Mapping[str, object],
    max_subscription_asset_ids: int | None = None,
) -> SubscribeMarketResponse:
    row = await store.upsert_user_subscription(token_id)
    if row is None:
        raise UnknownSubscriptionTokenError(token_id)

    _log_subscription_event(
        "subscription.user_add",
        token_id=row.token_id,
        condition_id=row.condition_id,
        request_metadata=request_metadata,
    )
    effective = True
    if max_subscription_asset_ids is not None:
        active_ids = sorted(await store.read_user_subscriptions())
        effective = (
            not active_ids
            or token_id in active_ids[:max_subscription_asset_ids]
        )
    return SubscribeMarketResponse(
        token_id=row.token_id,
        source=row.source,
        created_at=row.created_at.isoformat(),
        effective=effective,
    )


async def unsubscribe_market(
    store: SubscriptionStore,
    *,
    token_id: str,
    request_metadata: Mapping[str, object],
) -> UnsubscribeMarketResponse:
    condition_id = await _read_condition_id_for_log(store, token_id)
    deleted = await store.delete_user_subscription(token_id)
    if deleted:
        _log_subscription_event(
            "subscription.user_remove",
            token_id=token_id,
            condition_id=condition_id,
            request_metadata=request_metadata,
        )
    return UnsubscribeMarketResponse(token_id=token_id, deleted=deleted)


async def _read_condition_id_for_log(
    store: SubscriptionStore,
    token_id: str,
) -> str | None:
    read_token_condition_id = getattr(store, "read_token_condition_id", None)
    if not callable(read_token_condition_id):
        return None
    return cast(str | None, await read_token_condition_id(token_id))


def _log_subscription_event(
    event: str,
    *,
    token_id: str,
    condition_id: str | None,
    request_metadata: Mapping[str, object],
) -> None:
    extra: dict[str, Any] = {
        "token_id": token_id,
        "condition_id": condition_id,
    }
    extra.update(request_metadata)
    logger.info(event, extra=extra)
