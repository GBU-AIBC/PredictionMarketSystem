"""PostgreSQL persistence for deterministic LLM record/replay responses.

Rows remain available for deterministic replay and grow with distinct prompt
keys; operators may call :meth:`prune_before` during maintenance.
"""

from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Mapping

import asyncpg


class PostgresLLMReplayStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def get(self, replay_key: str) -> Mapping[str, Any] | None:
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT model, prompt_hash, input_hash, response, recorded_at, run_mode
                FROM llm_replay_records
                WHERE replay_key = $1
                """,
                replay_key,
            )
        if row is None:
            return None
        response = row["response"]
        if isinstance(response, str):
            response = json.loads(response)
        if not isinstance(response, Mapping):
            raise ValueError("LLM replay response JSONB is not an object")
        return dict(response)

    async def put(
        self,
        *,
        replay_key: str,
        model: str,
        prompt_hash: str,
        input_hash: str,
        prompt: str,
        response: Mapping[str, Any],
        recorded_at: datetime,
        run_mode: str,
    ) -> None:
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO llm_replay_records (
                    replay_key, model, prompt_hash, input_hash, prompt, response,
                    recorded_at, run_mode
                ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8)
                ON CONFLICT (replay_key) DO NOTHING
                """,
                replay_key,
                model,
                prompt_hash,
                input_hash,
                prompt,
                _json_response(response),
                recorded_at,
                run_mode,
            )

    async def prune_before(self, before: datetime) -> int:
        """Delete old replay rows when an operator explicitly requests pruning."""
        async with self.pool.acquire() as connection:
            result = await connection.execute(
                "DELETE FROM llm_replay_records WHERE recorded_at < $1",
                before,
            )
        return int(result.rsplit(" ", 1)[-1])

    async def count_before(self, before: datetime) -> int:
        """Count rows eligible for an explicit prune without mutating storage."""
        async with self.pool.acquire() as connection:
            count = await connection.fetchval(
                "SELECT COUNT(*) FROM llm_replay_records WHERE recorded_at < $1",
                before,
            )
        return int(count or 0)


def _json_response(response: Mapping[str, Any]) -> str:
    return json.dumps(dict(response), sort_keys=True, separators=(",", ":"))
