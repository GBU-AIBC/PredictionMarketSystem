from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from collections.abc import Mapping

import asyncpg

from pms.core.validation import normalize_resolved_outcome


@dataclass
class MarketResolutionStore:
    pool: asyncpg.Pool

    async def upsert(
        self,
        market_id: str,
        *,
        resolved_outcome: object,
        resolved_at: datetime | None = None,
        source: str,
        observed_at: datetime | None = None,
    ) -> None:
        normalized = normalize_resolved_outcome(resolved_outcome)
        if normalized is None:
            raise ValueError("resolved_outcome must be finite and within [0, 1]")
        if not source.strip():
            raise ValueError("resolution source must be non-empty")
        observed = observed_at or datetime.now(tz=UTC)
        await self.pool.execute(
            """
            INSERT INTO market_resolutions (
                market_id, resolved_outcome, resolved_at, source, observed_at
            ) VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (market_id) DO UPDATE SET
                resolved_outcome = EXCLUDED.resolved_outcome,
                resolved_at = EXCLUDED.resolved_at,
                source = EXCLUDED.source,
                observed_at = EXCLUDED.observed_at
            """,
            market_id,
            normalized,
            resolved_at,
            source,
            observed,
        )

    async def get_many(self, market_ids: list[str]) -> Mapping[str, float]:
        if not market_ids:
            return {}
        rows = await self.pool.fetch(
            """
            SELECT market_id, resolved_outcome
            FROM market_resolutions
            WHERE market_id = ANY($1::text[])
            """,
            market_ids,
        )
        return {
            str(row["market_id"]): float(row["resolved_outcome"])
            for row in rows
        }
