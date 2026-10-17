"""Shared request context helpers for the 2A-scoped agent routes."""

from __future__ import annotations

import asyncpg
from fastapi import HTTPException, Request


AGENT_CONTRIBUTION_SCOPES = frozenset({"paper_operate", "research"})


def pool_from_request(request: Request) -> asyncpg.Pool:
    runner = getattr(request.app.state, "runner", None)
    pool = getattr(runner, "pg_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="PostgreSQL pool is not initialized")
    return pool


def token_id_from_request(request: Request) -> str:
    token_id = getattr(request.state, "token_id", None)
    if not isinstance(token_id, str) or not token_id:
        raise HTTPException(status_code=401, detail="Authenticated token identity is required")
    return token_id
