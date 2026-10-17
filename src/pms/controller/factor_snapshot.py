from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
import json
from typing import Protocol

import asyncpg
from pms.core.validation import validate_finite

from pms.strategies.projections import FactorCompositionStep


FactorKey = tuple[str, str]
RAW_FACTOR_ROLES = frozenset(
    {
        "weighted",
        "precedence_rank",
        "threshold_edge",
        "posterior_prior",
        "posterior_success",
        "posterior_failure",
        "rule_delta",
    }
)


@dataclass(frozen=True, slots=True)
class FactorSnapshot:
    values: Mapping[FactorKey, float]
    missing_factors: tuple[FactorKey, ...] = ()
    timestamps: Mapping[FactorKey, datetime] = field(default_factory=dict)
    ages_ms: Mapping[FactorKey, float] = field(default_factory=dict)
    stale_factors: tuple[FactorKey, ...] = ()
    snapshot_hash: str | None = None


class FactorSnapshotReader(Protocol):
    async def snapshot(
        self,
        *,
        market_id: str,
        as_of: datetime,
        required: Sequence[FactorCompositionStep],
        strategy_id: str,
        strategy_version_id: str,
    ) -> FactorSnapshot: ...


@dataclass(frozen=True, slots=True)
class NullFactorSnapshotReader:
    async def snapshot(
        self,
        *,
        market_id: str,
        as_of: datetime,
        required: Sequence[FactorCompositionStep],
        strategy_id: str,
        strategy_version_id: str,
    ) -> FactorSnapshot:
        snapshot_keys = snapshot_factor_keys(required)
        missing_factors = required_factor_keys(required)
        return FactorSnapshot(
            values={},
            missing_factors=missing_factors,
            snapshot_hash=_snapshot_hash(
                market_id=market_id,
                as_of=as_of,
                strategy_id=strategy_id,
                strategy_version_id=strategy_version_id,
                required_keys=snapshot_keys,
                values={},
                missing_factors=missing_factors,
                stale_factors=(),
            ),
        )


@dataclass(frozen=True, slots=True)
class PostgresFactorSnapshotReader:
    pool: asyncpg.Pool

    async def snapshot(
        self,
        *,
        market_id: str,
        as_of: datetime,
        required: Sequence[FactorCompositionStep],
        strategy_id: str,
        strategy_version_id: str,
    ) -> FactorSnapshot:
        snapshot_keys = snapshot_factor_keys(required)
        required_keys = required_factor_keys(required)
        if not snapshot_keys:
            return FactorSnapshot(
                values={},
                missing_factors=(),
                snapshot_hash=_snapshot_hash(
                    market_id=market_id,
                    as_of=as_of,
                    strategy_id=strategy_id,
                    strategy_version_id=strategy_version_id,
                    required_keys=(),
                    values={},
                    missing_factors=(),
                    stale_factors=(),
                ),
            )

        factor_ids = [factor_id for factor_id, _ in snapshot_keys]
        params = [param for _, param in snapshot_keys]
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                WITH required(factor_id, param) AS (
                    SELECT * FROM unnest($3::text[], $4::text[])
                ),
                latest AS (
                    SELECT DISTINCT ON (fv.factor_id, fv.param)
                        fv.factor_id,
                        fv.param,
                        fv.value,
                        fv.ts
                    FROM factor_values AS fv
                    INNER JOIN required
                        ON required.factor_id = fv.factor_id
                       AND required.param = fv.param
                    WHERE fv.market_id = $1
                      AND fv.ts <= $2
                    ORDER BY fv.factor_id ASC, fv.param ASC, fv.ts DESC
                )
                SELECT
                    required.factor_id,
                    required.param,
                    latest.value,
                    latest.ts
                FROM required
                LEFT JOIN latest
                    ON latest.factor_id = required.factor_id
                   AND latest.param = required.param
                ORDER BY required.factor_id ASC, required.param ASC
                """,
                market_id,
                as_of,
                factor_ids,
                params,
            )
        values: dict[FactorKey, float] = {}
        timestamps: dict[FactorKey, datetime] = {}
        ages_ms: dict[FactorKey, float] = {}
        missing_factors: list[FactorKey] = []
        stale_factors: list[FactorKey] = []
        freshness_slas = _freshness_slas(required)
        required_key_set = set(required_keys)
        for row in rows:
            key = (row["factor_id"], row["param"])
            value = row["value"]
            if value is None:
                if key in required_key_set:
                    missing_factors.append(key)
                continue
            try:
                values[key] = validate_finite(value, field_name=f"factor {key[0]} value")
            except (TypeError, ValueError):
                if key in required_key_set:
                    missing_factors.append(key)
                continue
            ts = row["ts"]
            if isinstance(ts, datetime):
                timestamps[key] = ts
                age_ms = max(0.0, (as_of - ts).total_seconds() * 1000.0)
                ages_ms[key] = age_ms
                freshness_sla_s = freshness_slas.get(key)
                if freshness_sla_s is not None and age_ms > freshness_sla_s * 1000.0:
                    if key in required_key_set:
                        stale_factors.append(key)
                    # A stale optional value is still unsafe to compose.  Keep
                    # its identity for diagnostics while removing its value
                    # from the usable snapshot.
                    values.pop(key, None)
        return FactorSnapshot(
            values=values,
            missing_factors=tuple(missing_factors),
            timestamps=timestamps,
            ages_ms=ages_ms,
            stale_factors=tuple(stale_factors),
            snapshot_hash=_snapshot_hash(
                market_id=market_id,
                as_of=as_of,
                strategy_id=strategy_id,
                strategy_version_id=strategy_version_id,
                required_keys=snapshot_keys,
                values=values,
                missing_factors=tuple(missing_factors),
                stale_factors=tuple(stale_factors),
            ),
        )


def _snapshot_hash(
    *,
    market_id: str,
    as_of: datetime,
    strategy_id: str,
    strategy_version_id: str,
    required_keys: Sequence[FactorKey],
    values: Mapping[FactorKey, float],
    missing_factors: Sequence[FactorKey],
    stale_factors: Sequence[FactorKey] = (),
) -> str:
    payload = {
        "market_id": market_id,
        "as_of": as_of.isoformat(),
        "strategy_id": strategy_id,
        "strategy_version_id": strategy_version_id,
        "required_keys": [
            [factor_id, param]
            for factor_id, param in sorted(required_keys)
        ],
        "values": [
            [factor_id, param, values[(factor_id, param)]]
            for factor_id, param in sorted(values)
        ],
        "missing_factors": [
            [factor_id, param]
            for factor_id, param in sorted(missing_factors)
        ],
        "stale_factors": [
            [factor_id, param]
            for factor_id, param in sorted(stale_factors)
        ],
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def required_factor_keys(
    steps: Sequence[FactorCompositionStep],
) -> tuple[FactorKey, ...]:
    return tuple(
        dict.fromkeys(
            (step.factor_id, step.param)
            for step in steps
            if _requires_raw_factor(step)
        )
    )


def snapshot_factor_keys(
    steps: Sequence[FactorCompositionStep],
) -> tuple[FactorKey, ...]:
    return tuple(
        dict.fromkeys(
            (step.factor_id, step.param)
            for step in steps
            if _uses_raw_factor(step)
        )
    )


def _uses_raw_factor(step: FactorCompositionStep) -> bool:
    if not step.enabled:
        return False
    return step.role in RAW_FACTOR_ROLES


def _requires_raw_factor(step: FactorCompositionStep) -> bool:
    if not step.enabled:
        return False
    if getattr(step, "required", True) is False:
        return False
    return step.role in RAW_FACTOR_ROLES


def _freshness_slas(
    steps: Sequence[FactorCompositionStep],
) -> dict[FactorKey, float | None]:
    slas: dict[FactorKey, float | None] = {}
    for step in steps:
        if not _uses_raw_factor(step):
            continue
        raw_sla = getattr(step, "freshness_sla_s", None)
        if raw_sla is None:
            continue
        freshness_sla_s = float(raw_sla)
        key = (step.factor_id, step.param)
        current = slas.get(key)
        if current is None or freshness_sla_s < current:
            slas[key] = freshness_sla_s
    return slas
