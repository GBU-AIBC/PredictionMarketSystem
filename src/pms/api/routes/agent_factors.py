"""Agent-owned raw factor observation ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import re
from typing import Any

import asyncpg
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from pms.core.validation import validate_finite, validate_probability
from pms.factors.catalog import FactorCatalogEntry
from pms.factors.catalog import FACTOR_CATALOG_ROWS


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
            await connection.execute(
                """
                INSERT INTO agent_factor_provenance (
                    factor_id, param, market_id, ts, definition_version,
                    model, prompt_hash, evidence_ids, submitted_by_token_id
                ) VALUES ($1, $2, $3, $4, '1', $5, $6, $7::jsonb, $8)
                ON CONFLICT (factor_id, param, market_id, ts, definition_version)
                DO NOTHING
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
    return AgentFactorObservationResponse(
        factor_id=factor_id,
        param=request.param,
        market_id=request.market_id,
        ts=ts,
        value=value,
        definition_version="1",
        provenance_written=True,
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
    pool = _pool_from_request(request)
    token_id = next(
        (
            getattr(request.state, attribute)
            for attribute in ("token_id", "auth_token_id", "api_token_id")
            if isinstance(getattr(request.state, attribute, None), str)
            and getattr(request.state, attribute)
        ),
        None,
    )
    if not isinstance(token_id, str):
        raise HTTPException(status_code=401, detail="Authenticated token identity is required")
    return await submit_agent_factor_observation(
        pool,
        factor_id=factor_id,
        request=body,
        submitted_by_token_id=token_id,
    )


def _agent_catalog_entry(factor_id: str) -> FactorCatalogEntry:
    entry = next((item for item in FACTOR_CATALOG_ROWS if item.factor_id == factor_id), None)
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


def _pool_from_request(request: Request) -> asyncpg.Pool:
    runner = getattr(request.app.state, "runner", None)
    pool = getattr(runner, "pg_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="PostgreSQL pool is not initialized")
    return pool
