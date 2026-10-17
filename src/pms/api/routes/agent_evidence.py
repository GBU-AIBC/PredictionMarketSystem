"""Agent evidence request/response models and storage handlers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from pms.storage.agent_evidence_store import (
    AgentEvidence,
    AgentEvidenceStore,
)
from pms.api.auth import require_any_scope, require_scope
from pms.api.routes.agent_common import (
    pool_from_request,
    reject_live_agent_write,
    token_id_from_request,
)


router = APIRouter(tags=["agent"])


class AgentEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_id: str = Field(min_length=1, max_length=256)
    source: str = Field(min_length=1, max_length=512)
    model: str = Field(min_length=1, max_length=256)
    url: str = Field(min_length=1, max_length=2048)
    summary: str = Field(min_length=1, max_length=8192)
    payload: dict[str, Any]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    captured_at: datetime
    expires_at: datetime | None = None
    evidence_id: str | None = Field(default=None, min_length=1, max_length=256)


class AgentEvidenceResponse(BaseModel):
    evidence_id: str
    market_id: str
    source: str
    model: str
    url: str
    summary: str
    payload: dict[str, Any]
    content_hash: str
    captured_at: datetime
    expires_at: datetime | None
    submitted_by_token_id: str
    created_at: datetime


class AgentEvidenceListResponse(BaseModel):
    evidence: list[AgentEvidenceResponse]


async def submit_evidence(
    pool: asyncpg.Pool,
    request: AgentEvidenceRequest,
    *,
    submitted_by_token_id: str,
    allow_backfill: bool = False,
    now: datetime | None = None,
) -> AgentEvidenceResponse:
    try:
        row = await AgentEvidenceStore(pool).insert(
            evidence_id=request.evidence_id,
            market_id=request.market_id,
            source=request.source,
            model=request.model,
            url=request.url,
            summary=request.summary,
            payload=request.payload,
            content_hash=request.content_hash,
            captured_at=request.captured_at,
            expires_at=request.expires_at,
            submitted_by_token_id=submitted_by_token_id,
            allow_backfill=allow_backfill,
            now=now,
        )
    except (ValueError, asyncpg.ForeignKeyViolationError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except asyncpg.UniqueViolationError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="evidence_id already exists") from exc
    return _evidence_response(row)


async def list_evidence(
    pool: asyncpg.Pool,
    *,
    market_id: str,
    as_of: datetime | None = None,
    limit: int = 100,
) -> AgentEvidenceListResponse:
    rows = await AgentEvidenceStore(pool).list_for_market(
        market_id,
        as_of=as_of or datetime.now(tz=UTC),
        limit=limit,
    )
    return AgentEvidenceListResponse(evidence=[_evidence_response(row) for row in rows])


@router.post(
    "/evidence",
    response_model=AgentEvidenceResponse,
    status_code=201,
    dependencies=[
        Depends(require_any_scope("paper_operate", "research")),
        Depends(reject_live_agent_write),
    ],
)
async def post_evidence(request: Request, body: AgentEvidenceRequest) -> AgentEvidenceResponse:
    pool = pool_from_request(request)
    token_id = token_id_from_request(request)
    return await submit_evidence(pool, body, submitted_by_token_id=token_id)


@router.get(
    "/evidence",
    response_model=AgentEvidenceListResponse,
    dependencies=[Depends(require_scope("read"))],
)
async def get_evidence(
    request: Request,
    market_id: str = Query(min_length=1, max_length=256),
    limit: int = Query(default=100, ge=1, le=1000),
) -> AgentEvidenceListResponse:
    return await list_evidence(pool_from_request(request), market_id=market_id, limit=limit)


def _evidence_response(row: AgentEvidence) -> AgentEvidenceResponse:
    return AgentEvidenceResponse(
        evidence_id=row.evidence_id,
        market_id=row.market_id,
        source=row.source,
        model=row.model,
        url=row.url,
        summary=row.summary,
        payload=dict(row.payload),
        content_hash=row.content_hash,
        captured_at=row.captured_at,
        expires_at=row.expires_at,
        submitted_by_token_id=row.submitted_by_token_id,
        created_at=row.created_at,
    )
