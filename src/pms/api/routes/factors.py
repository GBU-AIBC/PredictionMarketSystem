from __future__ import annotations

from datetime import datetime
from typing import Any

import asyncpg
from pydantic import BaseModel, field_serializer


class FactorCatalogRowResponse(BaseModel):
    factor_id: str
    name: str
    description: str
    output_type: str
    direction: str


class FactorCatalogResponse(BaseModel):
    catalog: list[FactorCatalogRowResponse]


class FactorPointResponse(BaseModel):
    ts: datetime
    value: float

    @field_serializer("ts")
    def _serialize_ts(self, ts: datetime) -> str:
        return ts.isoformat()


class FactorSeriesResponse(BaseModel):
    factor_id: str
    param: str
    market_id: str
    points: list[FactorPointResponse]


MAX_FACTOR_SERIES_LIMIT = 2000


async def list_factor_catalog(pg_pool: asyncpg.Pool) -> dict[str, Any]:
    query = """
    SELECT factor_id, name, description, output_type, direction
    FROM factors
    ORDER BY factor_id ASC
    """
    async with pg_pool.acquire() as connection:
        rows = await connection.fetch(query)
    payload = FactorCatalogResponse(
        catalog=[
            FactorCatalogRowResponse(
                factor_id=row["factor_id"],
                name=row["name"],
                description=row["description"],
                output_type=row["output_type"],
                direction=row["direction"],
            )
            for row in rows
        ]
    )
    return payload.model_dump(mode="json")


async def list_factor_series(
    pg_pool: asyncpg.Pool,
    *,
    factor_id: str,
    market_id: str,
    param: str = "",
    since: datetime | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    if limit <= 0:
        msg = "limit must be positive"
        raise ValueError(msg)
    effective_limit = min(limit, MAX_FACTOR_SERIES_LIMIT)
    query = """
    WITH limited AS (
        SELECT factor_id, param, market_id, ts, value
        FROM factor_values
        WHERE factor_id = $1
          AND param = $2
          AND market_id = $3
          AND ($4::timestamptz IS NULL OR ts >= $4)
        ORDER BY ts DESC
        LIMIT $5
    )
    SELECT factor_id, param, market_id, ts, value
    FROM limited
    ORDER BY ts ASC
    """
    async with pg_pool.acquire() as connection:
        rows = await connection.fetch(
            query,
            factor_id,
            param,
            market_id,
            since,
            effective_limit,
        )
    payload = FactorSeriesResponse(
        factor_id=factor_id,
        param=param,
        market_id=market_id,
        points=[
            FactorPointResponse(
                ts=row["ts"],
                value=float(row["value"]),
            )
            for row in rows
        ],
    )
    return payload.model_dump(mode="json")
