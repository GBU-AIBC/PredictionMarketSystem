from __future__ import annotations

from datetime import UTC, datetime

import asyncpg


async def append_request_audit(
    pool: asyncpg.Pool,
    *,
    token_id: str,
    scope: str,
    method: str,
    path: str,
    outcome: int,
    created_at: datetime | None = None,
) -> None:
    """Append one redacted request outcome; bearer values never reach this sink."""
    timestamp = created_at or datetime.now(tz=UTC)
    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO api_request_audit
                (token_id, scope, method, path, outcome, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            token_id,
            scope,
            method,
            path,
            outcome,
            timestamp,
        )
