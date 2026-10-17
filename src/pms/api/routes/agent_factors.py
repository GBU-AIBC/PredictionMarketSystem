"""Agent-owned raw factor observation ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import re
from typing import Any

import asyncpg
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from pms.core.validation import validate_finite, validate_probability
from pms.factors.catalog import FactorCatalogEntry, catalog_entry
from pms.api.routes.agent_common import pool_from_request, token_id_from_request


router = APIRouter(tags=["agent"])
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_EVIDENCE_IDS = 100
MAX_FACTOR_AGE_S = 120.0


class AgentFactorObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_id: str = Field(min_length=1, max_length=256)
    param: str = Field(default="", max_length=256)
    value: Any
    ts: datetime
    model: str = Field(min_length=1, max_length=256)
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_ids: list[str] = Field(default_factory=list, max_length=MAX_EVIDENCE_IDS)


class AgentFactorObservationResponse(BaseModel):
    factor_id: str
    param: str
    market_id: str
    ts: datetime
    value: float
    definition_version: str
    provenance_written: bool


class AgentFactorProvenanceResponse(BaseModel):
    factor_id: str
    param: str
    market_id: str
    ts: datetime
    model: str
    prompt_hash: str
    evidence_ids: list[str]
    submitted_by_token_id: str
    created_at: datetime


async def submit_agent_factor_observation(
    pool: asyncpg.Pool,
    *,
    factor_id: str,
    request: AgentFactorObservationRequest,
    submitted_by_token_id: str,
    now: datetime | None = None,
) -> AgentFactorObservationResponse:
    entry = _agent_catalog_entry(factor_id)
    reference_now = _aware(now or datetime.now(tz=UTC), "now")
    ts = _aware(request.ts, "ts")
    if ts > reference_now:
        raise HTTPException(status_code=422, detail="ts must not be in the future")
    age = (reference_now - ts).total_seconds()
    if age > MAX_FACTOR_AGE_S:
        raise HTTPException(status_code=422, detail="ts is too old for a contribution")
    value = _validate_value(request.value, entry)
    if not request.model.strip() or not _HASH_RE.fullmatch(request.prompt_hash):
        raise HTTPException(status_code=422, detail="invalid provenance")
    if any(not evidence_id.strip() or len(evidence_id) > 256 for evidence_id in request.evidence_ids):
        raise HTTPException(status_code=422, detail="invalid evidence_ids")
    provenance_written = False
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                inserted = await connection.fetchrow(
                    """
                    INSERT INTO factor_values (
                        factor_id, param, market_id, ts, value, definition_version
                    ) VALUES ($1, $2, $3, $4, $5, '1')
                    ON CONFLICT (factor_id, param, market_id, ts, definition_version)
                    DO NOTHING
                    RETURNING factor_id
                    """,
                    factor_id,
                    request.param,
                    request.market_id,
                    ts,
                    value,
                )
                if inserted is None:
                    existing = await connection.fetchrow(
                        """
                        SELECT value FROM factor_values
                        WHERE factor_id = $1 AND param = $2 AND market_id = $3
                          AND ts = $4 AND definition_version = '1'
                        """,
                        factor_id,
                        request.param,
                        request.market_id,
                        ts,
                    )
                    if existing is None or float(existing["value"]) != value:
                        raise HTTPException(status_code=409, detail="factor observation conflicts with existing row")
                if request.evidence_ids:
                    evidence_count = await connection.fetchval(
                        """
                        SELECT COUNT(*)
                        FROM agent_evidence
                        WHERE evidence_id = ANY($1::text[])
                          AND market_id = $2
                          AND captured_at <= $3
                          AND created_at <= $3
                          AND (expires_at IS NULL OR expires_at > $3)
                        """,
                        request.evidence_ids,
                        request.market_id,
                        ts,
                    )
                    if int(evidence_count or 0) != len(set(request.evidence_ids)):
                        raise HTTPException(
                            status_code=422,
                            detail="evidence_ids must reference current evidence for the market",
                        )
                provenance = await connection.fetchrow(
                    """
                    INSERT INTO agent_factor_provenance (
                        factor_id, param, market_id, ts, definition_version,
                        model, prompt_hash, evidence_ids, submitted_by_token_id
                    ) VALUES ($1, $2, $3, $4, '1', $5, $6, $7::jsonb, $8)
                    ON CONFLICT (factor_id, param, market_id, ts, definition_version)
                    DO NOTHING
                    RETURNING factor_id
                    """,
                    factor_id,
                    request.param,
                    request.market_id,
                    ts,
                    request.model,
                    request.prompt_hash,
                    json.dumps(request.evidence_ids, allow_nan=False),
                    submitted_by_token_id,
                )
                provenance_written = provenance is not None
                if not provenance_written:
                    existing_provenance = await connection.fetchrow(
                        """
                        SELECT model, prompt_hash, evidence_ids, submitted_by_token_id
                        FROM agent_factor_provenance
                        WHERE factor_id = $1 AND param = $2 AND market_id = $3
                          AND ts = $4 AND definition_version = '1'
                        """,
                        factor_id,
                        request.param,
                        request.market_id,
                        ts,
                    )
                    if existing_provenance is None:
                        raise HTTPException(status_code=409, detail="factor provenance was not persisted")
                    existing_evidence_ids = existing_provenance["evidence_ids"]
                    if isinstance(existing_evidence_ids, str):
                        existing_evidence_ids = json.loads(existing_evidence_ids)
                    if (
                        str(existing_provenance["model"]) != request.model
                        or str(existing_provenance["prompt_hash"]) != request.prompt_hash
                        or [str(value) for value in existing_evidence_ids]
                        != request.evidence_ids
                        or str(existing_provenance["submitted_by_token_id"])
                        != submitted_by_token_id
                    ):
                        raise HTTPException(
                            status_code=409,
                            detail="factor provenance conflicts with existing row",
                        )
    except asyncpg.ForeignKeyViolationError as exc:
        raise HTTPException(status_code=422, detail="factor catalog or market does not exist") from exc
    return AgentFactorObservationResponse(
        factor_id=factor_id,
        param=request.param,
        market_id=request.market_id,
        ts=ts,
        value=value,
        definition_version="1",
        provenance_written=provenance_written,
    )


async def get_agent_factor_provenance(
    pool: asyncpg.Pool,
    *,
    factor_id: str,
    param: str,
    market_id: str,
    ts: datetime,
) -> AgentFactorProvenanceResponse | None:
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT factor_id, param, market_id, ts, model, prompt_hash,
                   evidence_ids, submitted_by_token_id, created_at
            FROM agent_factor_provenance
            WHERE factor_id = $1 AND param = $2 AND market_id = $3
              AND ts = $4 AND definition_version = '1'
            """,
            factor_id,
            param,
            market_id,
            _aware(ts, "ts"),
        )
    if row is None:
        return None
    evidence_ids = row["evidence_ids"]
    if isinstance(evidence_ids, str):
        evidence_ids = json.loads(evidence_ids)
    return AgentFactorProvenanceResponse(
        factor_id=str(row["factor_id"]),
        param=str(row["param"]),
        market_id=str(row["market_id"]),
        ts=row["ts"],
        model=str(row["model"]),
        prompt_hash=str(row["prompt_hash"]),
        evidence_ids=[str(value) for value in evidence_ids],
        submitted_by_token_id=str(row["submitted_by_token_id"]),
        created_at=row["created_at"],
    )


@router.post("/factors/{factor_id}/observations", response_model=AgentFactorObservationResponse, status_code=201)
async def post_agent_factor(
    factor_id: str,
    request: Request,
    body: AgentFactorObservationRequest,
) -> AgentFactorObservationResponse:
    pool = pool_from_request(request)
    token_id = token_id_from_request(request)
    return await submit_agent_factor_observation(
        pool,
        factor_id=factor_id,
        request=body,
        submitted_by_token_id=token_id,
    )


@router.get(
    "/factors/{factor_id}/observations/provenance",
    response_model=AgentFactorProvenanceResponse,
    response_model_exclude_none=True,
)
async def get_agent_factor_provenance_route(
    factor_id: str,
    request: Request,
    market_id: str = Query(min_length=1, max_length=256),
    ts: datetime = Query(),
    param: str = Query(default="", max_length=256),
) -> AgentFactorProvenanceResponse:
    row = await get_agent_factor_provenance(
        pool_from_request(request),
        factor_id=factor_id,
        param=param,
        market_id=market_id,
        ts=ts,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="factor provenance not found")
    return row


def _agent_catalog_entry(factor_id: str) -> FactorCatalogEntry:
    entry = catalog_entry(factor_id)
    if entry is None or not entry.owner.startswith("agent:"):
        raise HTTPException(status_code=422, detail="factor_id is not an agent-owned catalog factor")
    return entry


def _validate_value(value: object, entry: FactorCatalogEntry) -> float:
    try:
        finite = validate_finite(value, field_name="value")
        if entry.output_type == "probability":
            return validate_probability(finite, field_name="value")
        return finite
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _aware(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise HTTPException(status_code=422, detail=f"{field} must be timezone-aware")
    return value.astimezone(UTC)
