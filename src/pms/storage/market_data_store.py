from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

import asyncpg

from pms.core.models import (
    BookLevel,
    BookSnapshot,
    BookSummary,
    Market,
    Outcome,
    PriceChange,
    Token,
    Trade,
    Venue,
)

SubscribedMarketFilter = Literal["all", "only", "idle"]


@dataclass(frozen=True)
class MarketFilters:
    q: str = ""
    volume_min: float = 0.0
    liquidity_min: float = 0.0
    spread_max_bps: int | None = None
    yes_min: float = 0.0
    yes_max: float = 1.0
    resolves_within_days: int | None = None
    subscribed: SubscribedMarketFilter = "all"


@dataclass(frozen=True)
class MarketCatalogRow:
    market_id: str
    question: str
    venue: Venue
    volume_24h: float | None
    updated_at: datetime
    yes_token_id: str | None
    no_token_id: str | None
    resolves_at: datetime | None = None
    yes_price: float | None = None
    no_price: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    last_trade_price: float | None = None
    liquidity: float | None = None
    spread_bps: int | None = None
    price_updated_at: datetime | None = None
    subscription_source: str | None = None
    active: bool | None = None
    closed: bool | None = None
    accepting_orders: bool | None = None
    status_updated_at: datetime | None = None


@dataclass(frozen=True)
class MarketPriceSnapshotRow:
    snapshot_at: datetime
    yes_price: float | None
    no_price: float | None
    best_bid: float | None
    best_ask: float | None
    last_trade_price: float | None
    liquidity: float | None
    volume_24h: float | None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(cast(Any, value))


def _row_value(row: object, key: str) -> object | None:
    try:
        value: object = cast(Any, row)[key]
    except KeyError:
        return None
    return value


class PostgresMarketDataStore:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        market_freshness_s: float = 900.0,
    ) -> None:
        self._pool = pool
        self._market_freshness_s = market_freshness_s

    @property
    def pool(self) -> asyncpg.Pool:
        return self._pool

    async def read_market(self, market_id: str) -> Market | None:
        query = """
        SELECT
            condition_id,
            slug,
            question,
            venue,
            resolves_at,
            created_at,
            last_seen_at,
            volume_24h,
            risk_group_id,
            category,
            event_id,
            active,
            closed,
            accepting_orders,
            status_updated_at
        FROM markets
        WHERE condition_id = $1
        """
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(query, market_id)
        if row is None:
            return None
        return Market(
            condition_id=row["condition_id"],
            slug=row["slug"],
            question=row["question"],
            venue=row["venue"],
            resolves_at=row["resolves_at"],
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
            volume_24h=row["volume_24h"],
            risk_group_id=cast(str | None, _row_value(row, "risk_group_id")),
            category=cast(str | None, _row_value(row, "category")),
            event_id=cast(str | None, _row_value(row, "event_id")),
            active=cast(bool | None, _row_value(row, "active")),
            closed=cast(bool | None, _row_value(row, "closed")),
            accepting_orders=cast(
                bool | None,
                _row_value(row, "accepting_orders"),
            ),
            status_updated_at=cast(
                datetime | None,
                _row_value(row, "status_updated_at"),
            ),
        )

    async def read_tokens_for_market(self, market_id: str) -> list[Token]:
        query = """
        SELECT token_id, condition_id, outcome
        FROM tokens
        WHERE condition_id = $1
        ORDER BY CASE outcome WHEN 'YES' THEN 0 ELSE 1 END, token_id ASC
        """
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(query, market_id)
        return [
            Token(
                token_id=row["token_id"],
                condition_id=row["condition_id"],
                outcome=row["outcome"],
            )
            for row in rows
        ]

    async def read_markets(
        self,
        *,
        limit: int,
        offset: int,
        filters: MarketFilters | None = None,
        current_asset_ids: frozenset[str] = frozenset(),
        now: datetime | None = None,
        market_id: str | None = None,
    ) -> tuple[list[MarketCatalogRow], int]:
        query = """
        SELECT
            markets.condition_id AS market_id,
            markets.question,
            markets.venue,
            markets.volume_24h,
            markets.last_seen_at AS updated_at,
            markets.resolves_at,
            markets.yes_price,
            markets.no_price,
            markets.best_bid,
            markets.best_ask,
            markets.last_trade_price,
            markets.liquidity,
            markets.spread_bps,
            markets.price_updated_at,
            markets.active,
            markets.closed,
            markets.accepting_orders,
            markets.status_updated_at,
            MAX(CASE WHEN tokens.outcome = 'YES' THEN tokens.token_id END) AS yes_token_id,
            MAX(CASE WHEN tokens.outcome = 'NO' THEN tokens.token_id END) AS no_token_id,
            MAX(market_subscriptions.source) AS subscription_source,
            COUNT(*) OVER() AS total_count
        FROM markets
        LEFT JOIN tokens
            ON tokens.condition_id = markets.condition_id
        LEFT JOIN market_subscriptions
            ON market_subscriptions.token_id = tokens.token_id
        WHERE (markets.resolves_at IS NULL OR markets.resolves_at > $1)
          AND ($13::text IS NULL OR markets.condition_id = $13)
          AND ($2 = '' OR markets.question ILIKE '%' || $2 || '%')
          AND (
              $3 = 0
              OR (markets.volume_24h IS NOT NULL AND markets.volume_24h >= $3)
          )
          AND (
              $4 = 0
              OR (markets.liquidity IS NOT NULL AND markets.liquidity >= $4)
          )
          AND (
              $5::integer IS NULL
              OR (markets.spread_bps IS NOT NULL AND markets.spread_bps <= $5)
          )
          AND (
              $6 = 0
              OR (markets.yes_price IS NOT NULL AND markets.yes_price >= $6)
          )
          AND (
              $7 = 1
              OR (markets.yes_price IS NOT NULL AND markets.yes_price <= $7)
          )
          AND (
              $8::timestamptz IS NULL
              OR (markets.resolves_at IS NOT NULL AND markets.resolves_at <= $8)
          )
          AND (
              $9 = 'all'
              OR (
                  $9 = 'only'
                  AND (
                      EXISTS (
                          SELECT 1
                          FROM tokens AS subscribed_tokens
                          WHERE subscribed_tokens.condition_id = markets.condition_id
                            AND subscribed_tokens.token_id = ANY($10::text[])
                      )
                      OR EXISTS (
                          SELECT 1
                          FROM tokens AS user_subscription_tokens
                          JOIN market_subscriptions AS user_subscriptions
                              ON user_subscriptions.token_id = user_subscription_tokens.token_id
                          WHERE user_subscription_tokens.condition_id = markets.condition_id
                      )
                  )
              )
              OR (
                  $9 = 'idle'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM tokens AS subscribed_tokens
                      WHERE subscribed_tokens.condition_id = markets.condition_id
                        AND subscribed_tokens.token_id = ANY($10::text[])
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM tokens AS user_subscription_tokens
                      JOIN market_subscriptions AS user_subscriptions
                          ON user_subscriptions.token_id = user_subscription_tokens.token_id
                      WHERE user_subscription_tokens.condition_id = markets.condition_id
                  )
              )
          )
        GROUP BY
            markets.condition_id,
            markets.question,
            markets.venue,
            markets.volume_24h,
            markets.last_seen_at,
            markets.resolves_at,
            markets.yes_price,
            markets.no_price,
            markets.best_bid,
            markets.best_ask,
            markets.last_trade_price,
            markets.liquidity,
            markets.spread_bps,
            markets.price_updated_at,
            markets.active,
            markets.closed,
            markets.accepting_orders,
            markets.status_updated_at
        ORDER BY
            COALESCE(markets.volume_24h, 0) DESC,
            markets.last_seen_at DESC,
            markets.condition_id ASC
        LIMIT $11
        OFFSET $12
        """
        reference_now = now or datetime.now(tz=UTC)
        active_filters = filters or MarketFilters()
        resolves_before = (
            None
            if active_filters.resolves_within_days is None
            else reference_now + timedelta(days=active_filters.resolves_within_days)
        )
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                query,
                reference_now,
                active_filters.q.strip(),
                active_filters.volume_min,
                active_filters.liquidity_min,
                active_filters.spread_max_bps,
                active_filters.yes_min,
                active_filters.yes_max,
                resolves_before,
                active_filters.subscribed,
                sorted(current_asset_ids),
                limit,
                offset,
                market_id,
            )
        if not rows:
            return [], 0
        total = cast(int, rows[0]["total_count"])
        return [
            MarketCatalogRow(
                market_id=cast(str, row["market_id"]),
                question=cast(str, row["question"]),
                venue=cast(Venue, row["venue"]),
                volume_24h=cast(float | None, row["volume_24h"]),
                updated_at=cast(datetime, row["updated_at"]),
                yes_token_id=cast(str | None, row["yes_token_id"]),
                no_token_id=cast(str | None, row["no_token_id"]),
                resolves_at=cast(datetime | None, row["resolves_at"]),
                yes_price=_optional_float(row["yes_price"]),
                no_price=_optional_float(row["no_price"]),
                best_bid=_optional_float(row["best_bid"]),
                best_ask=_optional_float(row["best_ask"]),
                last_trade_price=_optional_float(row["last_trade_price"]),
                liquidity=_optional_float(row["liquidity"]),
                spread_bps=cast(int | None, row["spread_bps"]),
                price_updated_at=cast(datetime | None, row["price_updated_at"]),
                subscription_source=cast(str | None, row["subscription_source"]),
                active=cast(bool | None, _row_value(row, "active")),
                closed=cast(bool | None, _row_value(row, "closed")),
                accepting_orders=cast(
                    bool | None,
                    _row_value(row, "accepting_orders"),
                ),
                status_updated_at=cast(
                    datetime | None,
                    _row_value(row, "status_updated_at"),
                ),
            )
            for row in rows
        ], total

    async def read_market_by_id(
        self,
        *,
        market_id: str,
        current_asset_ids: frozenset[str] = frozenset(),
        now: datetime | None = None,
    ) -> MarketCatalogRow | None:
        rows, _ = await self.read_markets(
            limit=1,
            offset=0,
            filters=MarketFilters(),
            current_asset_ids=current_asset_ids,
            now=now,
            market_id=market_id,
        )
        return rows[0] if rows else None

    async def read_price_history(
        self,
        *,
        condition_id: str,
        since: datetime,
        limit: int,
    ) -> list[MarketPriceSnapshotRow] | None:
        snapshots_query = """
        SELECT
            snapshot_at,
            yes_price,
            no_price,
            best_bid,
            best_ask,
            last_trade_price,
            liquidity,
            volume_24h
        FROM market_price_snapshots
        WHERE condition_id = $1
          AND snapshot_at >= $2
        ORDER BY snapshot_at ASC
        LIMIT $3
        """
        async with self._pool.acquire() as connection:
            market_exists = await connection.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM markets
                    WHERE condition_id = $1
                )
                """,
                condition_id,
            )
            if not market_exists:
                return None
            rows = await connection.fetch(
                snapshots_query,
                condition_id,
                since,
                limit,
            )
        return [
            MarketPriceSnapshotRow(
                snapshot_at=cast(datetime, row["snapshot_at"]),
                yes_price=_optional_float(row["yes_price"]),
                no_price=_optional_float(row["no_price"]),
                best_bid=_optional_float(row["best_bid"]),
                best_ask=_optional_float(row["best_ask"]),
                last_trade_price=_optional_float(row["last_trade_price"]),
                liquidity=_optional_float(row["liquidity"]),
                volume_24h=_optional_float(row["volume_24h"]),
            )
            for row in rows
        ]

    async def read_eligible_markets(
        self,
        venue: str,
        max_horizon_days: int | None,
        min_volume_usdc: float,
    ) -> list[tuple[Market, list[Token]]]:
        now = datetime.now(tz=UTC)
        upper_bound = (
            None
            if max_horizon_days is None
            else now + timedelta(days=max_horizon_days)
        )
        query = """
        SELECT
            markets.condition_id,
            markets.slug,
            markets.question,
            markets.venue,
            markets.resolves_at,
            markets.created_at,
            markets.last_seen_at,
            markets.volume_24h,
            markets.risk_group_id,
            markets.category,
            markets.event_id,
            markets.yes_price,
            markets.no_price,
            markets.best_bid,
            markets.best_ask,
            markets.last_trade_price,
            markets.liquidity,
            markets.spread_bps,
            markets.price_updated_at,
            markets.active,
            markets.closed,
            markets.accepting_orders,
            markets.status_updated_at,
            tokens.token_id,
            tokens.outcome
        FROM markets
        LEFT JOIN tokens
            ON tokens.condition_id = markets.condition_id
        WHERE markets.venue = $1
          AND (
                $4::double precision <= 0
                OR (
                    markets.volume_24h IS NOT NULL
                    AND markets.volume_24h >= $4
                )
          )
          AND COALESCE(markets.closed, false) = false
          AND COALESCE(markets.active, true) = true
          AND markets.last_seen_at >= (
                $2::timestamptz - ($5::double precision * INTERVAL '1 second')
          )
          AND (
                (
                    $3::timestamptz IS NULL
                    AND (
                        markets.resolves_at IS NULL
                        OR markets.resolves_at > $2
                    )
                )
                OR (
                    $3::timestamptz IS NOT NULL
                    AND markets.resolves_at IS NOT NULL
                    AND markets.resolves_at > $2
                    AND markets.resolves_at <= $3
                )
          )
        ORDER BY
            COALESCE(markets.volume_24h, 0) DESC,
            COALESCE(markets.liquidity, 0) DESC,
            markets.condition_id ASC,
            CASE tokens.outcome WHEN 'YES' THEN 0 WHEN 'NO' THEN 1 ELSE 2 END,
            tokens.token_id ASC
        """
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                query,
                venue,
                now,
                upper_bound,
                min_volume_usdc,
                self._market_freshness_s,
            )

        eligible_markets: list[tuple[Market, list[Token]]] = []
        current_market_id: str | None = None
        current_tokens: list[Token] | None = None

        for row in rows:
            market_id = cast(str, row["condition_id"])
            if market_id != current_market_id:
                current_market_id = market_id
                current_tokens = []
                eligible_markets.append(
                    (
                        Market(
                            condition_id=market_id,
                            slug=cast(str, row["slug"]),
                            question=cast(str, row["question"]),
                            venue=cast(Venue, row["venue"]),
                            resolves_at=cast(datetime | None, row["resolves_at"]),
                            created_at=cast(datetime, row["created_at"]),
                            last_seen_at=cast(datetime, row["last_seen_at"]),
                            volume_24h=cast(float | None, row["volume_24h"]),
                            risk_group_id=cast(
                                str | None,
                                _row_value(row, "risk_group_id"),
                            ),
                            category=cast(str | None, _row_value(row, "category")),
                            event_id=cast(str | None, _row_value(row, "event_id")),
                            yes_price=_optional_float(_row_value(row, "yes_price")),
                            no_price=_optional_float(_row_value(row, "no_price")),
                            best_bid=_optional_float(_row_value(row, "best_bid")),
                            best_ask=_optional_float(_row_value(row, "best_ask")),
                            last_trade_price=_optional_float(
                                _row_value(row, "last_trade_price")
                            ),
                            liquidity=_optional_float(row["liquidity"]),
                            spread_bps=cast(int | None, _row_value(row, "spread_bps")),
                            price_updated_at=cast(
                                datetime | None,
                                _row_value(row, "price_updated_at"),
                            ),
                            active=cast(bool | None, _row_value(row, "active")),
                            closed=cast(bool | None, _row_value(row, "closed")),
                            accepting_orders=cast(
                                bool | None,
                                _row_value(row, "accepting_orders"),
                            ),
                            status_updated_at=cast(
                                datetime | None,
                                _row_value(row, "status_updated_at"),
                            ),
                        ),
                        current_tokens,
                    )
                )

            token_id = row["token_id"]
            outcome = row["outcome"]
            if (
                current_tokens is not None
                and token_id is not None
                and outcome is not None
            ):
                current_tokens.append(
                    Token(
                        token_id=cast(str, token_id),
                        condition_id=market_id,
                        outcome=cast(Outcome, outcome),
                    )
                )

        return eligible_markets

    async def write_market(self, market: Market) -> None:
        query = """
        INSERT INTO markets (
            condition_id,
            slug,
            question,
            venue,
            resolves_at,
            created_at,
            last_seen_at,
            volume_24h,
            risk_group_id,
            category,
            event_id,
            yes_price,
            no_price,
            best_bid,
            best_ask,
            last_trade_price,
            liquidity,
            spread_bps,
            price_updated_at,
            active,
            closed,
            accepting_orders,
            status_updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8,
            $9, $10, $11, $12, $13, $14, $15, $16,
            $17, $18, $19, $20, $21, $22, $23
        )
        ON CONFLICT (condition_id) DO UPDATE
        SET slug = EXCLUDED.slug,
            question = EXCLUDED.question,
            venue = EXCLUDED.venue,
            resolves_at = EXCLUDED.resolves_at,
            last_seen_at = EXCLUDED.last_seen_at,
            volume_24h = EXCLUDED.volume_24h,
            risk_group_id = EXCLUDED.risk_group_id,
            category = EXCLUDED.category,
            event_id = EXCLUDED.event_id,
            yes_price = EXCLUDED.yes_price,
            no_price = EXCLUDED.no_price,
            best_bid = EXCLUDED.best_bid,
            best_ask = EXCLUDED.best_ask,
            last_trade_price = EXCLUDED.last_trade_price,
            liquidity = EXCLUDED.liquidity,
            spread_bps = EXCLUDED.spread_bps,
            price_updated_at = EXCLUDED.price_updated_at,
            active = EXCLUDED.active,
            closed = EXCLUDED.closed,
            accepting_orders = EXCLUDED.accepting_orders,
            status_updated_at = EXCLUDED.status_updated_at
        """
        async with self._pool.acquire() as connection:
            await connection.execute(
                query,
                market.condition_id,
                market.slug,
                market.question,
                market.venue,
                market.resolves_at,
                market.created_at,
                market.last_seen_at,
                market.volume_24h,
                market.risk_group_id,
                market.category,
                market.event_id,
                market.yes_price,
                market.no_price,
                market.best_bid,
                market.best_ask,
                market.last_trade_price,
                market.liquidity,
                market.spread_bps,
                market.price_updated_at,
                market.active,
                market.closed,
                market.accepting_orders,
                market.status_updated_at,
            )

    async def read_market_signal_metadata(self, market_id: str) -> dict[str, str]:
        query = """
        SELECT
            markets.risk_group_id,
            markets.category,
            markets.event_id,
            MAX(CASE WHEN tokens.outcome = 'YES' THEN tokens.token_id END) AS yes_token_id,
            MAX(CASE WHEN tokens.outcome = 'NO' THEN tokens.token_id END) AS no_token_id
        FROM markets
        LEFT JOIN tokens
            ON tokens.condition_id = markets.condition_id
        WHERE markets.condition_id = $1
        GROUP BY markets.risk_group_id, markets.category, markets.event_id
        """
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(query, market_id)
        if row is None:
            return {}

        metadata: dict[str, str] = {}
        for field_name in (
            "risk_group_id",
            "category",
            "event_id",
            "yes_token_id",
            "no_token_id",
        ):
            value = cast(str | None, _row_value(row, field_name))
            if value is not None and value.strip() != "":
                metadata[field_name] = value
        return metadata

    async def write_price_snapshot(
        self,
        *,
        condition_id: str,
        snapshot_at: datetime,
        yes_price: float | None,
        no_price: float | None,
        best_bid: float | None,
        best_ask: float | None,
        last_trade_price: float | None,
        liquidity: float | None,
        volume_24h: float | None,
    ) -> None:
        query = """
        INSERT INTO market_price_snapshots (
            condition_id,
            snapshot_at,
            yes_price,
            no_price,
            best_bid,
            best_ask,
            last_trade_price,
            liquidity,
            volume_24h
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        async with self._pool.acquire() as connection:
            await connection.execute(
                query,
                condition_id,
                snapshot_at,
                yes_price,
                no_price,
                best_bid,
                best_ask,
                last_trade_price,
                liquidity,
                volume_24h,
            )

    async def read_snapshot_lag_seconds_max(self) -> float:
        query = """
        WITH latest_snapshots AS (
            SELECT condition_id, MAX(snapshot_at) AS snapshot_at
            FROM market_price_snapshots
            GROUP BY condition_id
        )
        SELECT COALESCE(
            MAX(
                GREATEST(
                    EXTRACT(EPOCH FROM markets.price_updated_at - latest_snapshots.snapshot_at),
                    0
                )
            ),
            0
        ) AS snapshot_lag_seconds_max
        FROM markets
        INNER JOIN latest_snapshots
            ON latest_snapshots.condition_id = markets.condition_id
        WHERE markets.price_updated_at IS NOT NULL
        """
        async with self._pool.acquire() as connection:
            value = await connection.fetchval(query)
        if value is None:
            return 0.0
        return float(value)

    async def write_token(self, token: Token) -> None:
        query = """
        INSERT INTO tokens (
            token_id,
            condition_id,
            outcome
        ) VALUES ($1, $2, $3)
        ON CONFLICT (token_id) DO UPDATE
        SET condition_id = EXCLUDED.condition_id,
            outcome = EXCLUDED.outcome
        """
        async with self._pool.acquire() as connection:
            await connection.execute(
                query,
                token.token_id,
                token.condition_id,
                token.outcome,
            )

    async def write_book_snapshot(
        self,
        snapshot: BookSnapshot,
        levels: list[BookLevel],
    ) -> int:
        snapshot_query = """
        INSERT INTO book_snapshots (
            market_id,
            token_id,
            ts,
            hash,
            source
        ) VALUES ($1, $2, $3, $4, $5)
        RETURNING id
        """
        level_query = """
        INSERT INTO book_levels (
            snapshot_id,
            market_id,
            side,
            price,
            size
        ) VALUES ($1, $2, $3, $4, $5)
        """
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                snapshot_id = await connection.fetchval(
                    snapshot_query,
                    snapshot.market_id,
                    snapshot.token_id,
                    snapshot.ts,
                    snapshot.hash,
                    snapshot.source,
                )
                if not isinstance(snapshot_id, int):
                    msg = "book_snapshots.id did not return an integer"
                    raise TypeError(msg)
                for level in levels:
                    await connection.execute(
                        level_query,
                        snapshot_id,
                        level.market_id,
                        level.side,
                        level.price,
                        level.size,
                    )
        return snapshot_id

    async def get_latest_book_summary(self, market_id: str) -> BookSummary | None:
        query = """
        WITH latest_snapshot AS (
            SELECT id, ts
            FROM book_snapshots
            INNER JOIN tokens
                ON tokens.token_id = book_snapshots.token_id
               AND tokens.condition_id = book_snapshots.market_id
               AND tokens.outcome = 'YES'
            WHERE book_snapshots.market_id = $1
            ORDER BY ts DESC, id DESC
            LIMIT 1
        )
        SELECT
            latest_snapshot.ts,
            book_levels.side,
            book_levels.price,
            book_levels.size
        FROM latest_snapshot
        JOIN book_levels
            ON book_levels.snapshot_id = latest_snapshot.id
        WHERE book_levels.market_id = $1
        """
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(query, market_id)
        if not rows:
            return None

        bids = [
            (float(row["price"]), float(row["size"]))
            for row in rows
            if row["side"] == "BUY"
        ]
        asks = [
            (float(row["price"]), float(row["size"]))
            for row in rows
            if row["side"] == "SELL"
        ]
        if not bids or not asks:
            return None

        best_bid = max(price for price, _ in bids)
        best_ask = min(price for price, _ in asks)
        midpoint = (best_bid + best_ask) / 2.0
        if midpoint <= 0.0:
            return None
        top_bid_depth = sum(price * size for price, size in bids if price == best_bid)
        top_ask_depth = sum(price * size for price, size in asks if price == best_ask)
        return BookSummary(
            best_bid=best_bid,
            best_ask=best_ask,
            spread_bps=((best_ask - best_bid) / midpoint) * 10_000.0,
            depth_usdc=top_bid_depth + top_ask_depth,
            timestamp=cast(datetime, rows[0]["ts"]),
        )

    async def write_price_change(self, price_change: PriceChange) -> None:
        query = """
        INSERT INTO price_changes (
            market_id,
            token_id,
            ts,
            side,
            price,
            size,
            best_bid,
            best_ask,
            hash
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        async with self._pool.acquire() as connection:
            await connection.execute(
                query,
                price_change.market_id,
                price_change.token_id,
                price_change.ts,
                price_change.side,
                price_change.price,
                price_change.size,
                price_change.best_bid,
                price_change.best_ask,
                price_change.hash,
            )

    async def write_trade(self, trade: Trade) -> None:
        query = """
        INSERT INTO trades (
            market_id,
            token_id,
            ts,
            price
        ) VALUES ($1, $2, $3, $4)
        """
        async with self._pool.acquire() as connection:
            await connection.execute(
                query,
                trade.market_id,
                trade.token_id,
                trade.ts,
                trade.price,
            )

    async def read_latest_snapshot(
        self,
        market_id: str,
        token_id: str,
    ) -> BookSnapshot | None:
        query = """
        SELECT id, market_id, token_id, ts, hash, source
        FROM book_snapshots
        WHERE market_id = $1 AND token_id = $2
        ORDER BY ts DESC, id DESC
        LIMIT 1
        """
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(query, market_id, token_id)
        if row is None:
            return None
        return BookSnapshot(
            id=row["id"],
            market_id=row["market_id"],
            token_id=row["token_id"],
            ts=row["ts"],
            hash=row["hash"],
            source=row["source"],
        )

    async def read_latest_book_snapshot(self, market_id: str) -> BookSnapshot | None:
        query = """
        SELECT id, market_id, token_id, ts, hash, source
        FROM book_snapshots
        WHERE market_id = $1
        ORDER BY ts DESC, id DESC
        LIMIT 1
        """
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(query, market_id)
        if row is None:
            return None
        return BookSnapshot(
            id=row["id"],
            market_id=row["market_id"],
            token_id=row["token_id"],
            ts=row["ts"],
            hash=row["hash"],
            source=row["source"],
        )

    async def read_levels_for_snapshot(self, snapshot_id: int) -> list[BookLevel]:
        query = """
        SELECT snapshot_id, market_id, side, price, size
        FROM book_levels
        WHERE snapshot_id = $1
        ORDER BY
            CASE side WHEN 'BUY' THEN 0 ELSE 1 END,
            price DESC
        """
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(query, snapshot_id)
        return [
            BookLevel(
                snapshot_id=row["snapshot_id"],
                market_id=row["market_id"],
                side=row["side"],
                price=row["price"],
                size=row["size"],
            )
            for row in rows
        ]

    async def read_price_changes_since(
        self,
        market_id: str,
        token_id: str,
        since_ts: object,
    ) -> list[PriceChange]:
        query = """
        SELECT id, market_id, token_id, ts, side, price, size, best_bid, best_ask, hash
        FROM price_changes
        WHERE market_id = $1 AND token_id = $2 AND ts >= $3
        ORDER BY ts ASC, id ASC
        """
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(query, market_id, token_id, since_ts)
        return [
            PriceChange(
                id=row["id"],
                market_id=row["market_id"],
                token_id=row["token_id"],
                ts=row["ts"],
                side=row["side"],
                price=row["price"],
                size=row["size"],
                best_bid=row["best_bid"],
                best_ask=row["best_ask"],
                hash=row["hash"],
            )
            for row in rows
        ]
