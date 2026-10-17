"""Durable, point-in-time storage for agent supplied evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from typing import Any, cast
from urllib.parse import urlparse
from uuid import uuid4

import asyncpg


MAX_PAYLOAD_BYTES = 64 * 1024
MAX_PAYLOAD_DEPTH = 8
MAX_EVIDENCE_AGE_S = 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class AgentEvidence:
    evidence_id: str
    market_id: str
    source: str
    model: str
    url: str
    summary: str
    payload: Mapping[str, Any]
    content_hash: str
    captured_at: datetime
    expires_at: datetime | None
    submitted_by_token_id: str
    created_at: datetime


def canonical_content_hash(
    *,
    market_id: str,
    source: str,
    model: str,
    url: str,
    summary: str,
    payload: Mapping[str, Any],
    captured_at: datetime,
    expires_at: datetime | None,
) -> str:
    canonical = {
        "captured_at": _utc(captured_at).isoformat(),
        "expires_at": None if expires_at is None else _utc(expires_at).isoformat(),
        "market_id": market_id,
        "model": model,
        "payload": payload,
        "source": source,
        "summary": summary,
        "url": url,
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_evidence_input(
    *,
    market_id: str,
    source: str,
    model: str,
    url: str,
    summary: str,
    payload: Mapping[str, Any],
    captured_at: datetime,
    expires_at: datetime | None,
    content_hash: str,
    now: datetime | None = None,
    allow_backfill: bool = False,
) -> str:
    if not market_id.strip():
        raise ValueError("market_id must not be blank")
    for value, field, limit in (
        (source, "source", 512),
        (model, "model", 256),
        (summary, "summary", 8192),
    ):
        if not value.strip() or len(value) > limit:
            raise ValueError(f"{field} must be non-empty and at most {limit} characters")
    parsed_url = urlparse(url)
    if len(url) > 2048 or parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("url must be an absolute http(s) URL")
    _validate_payload(payload)
    captured_at = _aware(captured_at, "captured_at")
    expires_at = None if expires_at is None else _aware(expires_at, "expires_at")
    reference_now = _aware(now or datetime.now(tz=UTC), "now")
    if captured_at > reference_now:
        raise ValueError("captured_at must not be in the future")
    if not allow_backfill and (reference_now - captured_at).total_seconds() > MAX_EVIDENCE_AGE_S:
        raise ValueError("captured_at is too old for a paper contribution")
    if expires_at is not None and expires_at <= captured_at:
        raise ValueError("expires_at must be after captured_at")
    if expires_at is not None and expires_at <= reference_now and not allow_backfill:
        raise ValueError("evidence is already expired")
    expected_hash = canonical_content_hash(
        market_id=market_id,
        source=source,
        model=model,
        url=url,
        summary=summary,
        payload=payload,
        captured_at=captured_at,
        expires_at=expires_at,
    )
    if content_hash != expected_hash:
        raise ValueError("content_hash does not match canonical evidence")
    return expected_hash


@dataclass
class AgentEvidenceStore:
    pool: asyncpg.Pool

    async def insert(
        self,
        *,
        evidence_id: str | None = None,
        market_id: str,
        source: str,
        model: str,
        url: str,
        summary: str,
        payload: Mapping[str, Any],
        content_hash: str,
        captured_at: datetime,
        expires_at: datetime | None,
        submitted_by_token_id: str,
        now: datetime | None = None,
        allow_backfill: bool = False,
    ) -> AgentEvidence:
        # A retry must remain idempotent after the submit-time freshness window
        # closes.  Verify the canonical identity first, then return the
        # immutable row before applying current-time age/expiry checks.
        try:
            canonical_hash = canonical_content_hash(
                market_id=market_id,
                source=source,
                model=model,
                url=url,
                summary=summary,
                payload=payload,
                captured_at=captured_at,
                expires_at=expires_at,
            )
        except (TypeError, ValueError):
            canonical_hash = ""
        if canonical_hash == content_hash:
            async with self.pool.acquire() as connection:
                existing_row = await connection.fetchrow(
                    """
                    SELECT evidence_id, market_id, source, model, url, summary, payload,
                           content_hash, captured_at, expires_at, submitted_by_token_id, created_at
                    FROM agent_evidence
                    WHERE content_hash = $1
                    """,
                    content_hash,
                )
            if existing_row is not None:
                return evidence_from_row(existing_row)
        validate_evidence_input(
            market_id=market_id,
            source=source,
            model=model,
            url=url,
            summary=summary,
            payload=payload,
            captured_at=captured_at,
            expires_at=expires_at,
            content_hash=content_hash,
            now=now,
            allow_backfill=allow_backfill,
        )
        created_at = _aware(now or datetime.now(tz=UTC), "now")
        resolved_evidence_id = evidence_id or str(uuid4())
        if len(resolved_evidence_id) > 256 or not resolved_evidence_id.strip():
            raise ValueError("evidence_id must be non-empty and at most 256 characters")
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO agent_evidence (
                    evidence_id, market_id, source, model, url, summary, payload,
                    content_hash, captured_at, expires_at, submitted_by_token_id, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12)
                ON CONFLICT (content_hash) DO UPDATE SET content_hash = EXCLUDED.content_hash
                RETURNING evidence_id, market_id, source, model, url, summary, payload,
                    content_hash, captured_at, expires_at, submitted_by_token_id, created_at
                """,
                resolved_evidence_id,
                market_id,
                source,
                model,
                url,
                summary,
                json.dumps(payload, ensure_ascii=False, allow_nan=False),
                content_hash,
                _aware(captured_at, "captured_at"),
                None if expires_at is None else _aware(expires_at, "expires_at"),
                submitted_by_token_id,
                created_at,
            )
        if row is None:
            raise RuntimeError("evidence insert did not return a row")
        return evidence_from_row(row)

    async def list_for_market(
        self,
        market_id: str,
        *,
        as_of: datetime | None = None,
        limit: int = 100,
    ) -> list[AgentEvidence]:
        point_in_time = _aware(as_of or datetime.now(tz=UTC), "as_of")
        if limit <= 0 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT evidence_id, market_id, source, model, url, summary, payload,
                       content_hash, captured_at, expires_at, submitted_by_token_id, created_at
                FROM agent_evidence
                WHERE market_id = $1
                  AND GREATEST(captured_at, created_at) <= $2
                  AND (expires_at IS NULL OR expires_at > $2)
                ORDER BY captured_at DESC, evidence_id ASC
                LIMIT $3
                """,
                market_id,
                point_in_time,
                limit,
            )
        return [evidence_from_row(row) for row in rows]


def evidence_from_row(row: Mapping[str, object]) -> AgentEvidence:
    raw_payload = row["payload"]
    payload = json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
    return AgentEvidence(
        evidence_id=str(row["evidence_id"]),
        market_id=str(row["market_id"]),
        source=str(row["source"]),
        model=str(row["model"]),
        url=str(row["url"]),
        summary=str(row["summary"]),
        payload=cast(Mapping[str, Any], payload),
        content_hash=str(row["content_hash"]),
        captured_at=_aware(cast(datetime, row["captured_at"]), "captured_at"),
        expires_at=(
            None
            if row["expires_at"] is None
            else _aware(cast(datetime, row["expires_at"]), "expires_at")
        ),
        submitted_by_token_id=str(row["submitted_by_token_id"]),
        created_at=_aware(cast(datetime, row["created_at"]), "created_at"),
    )


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a JSON object")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ValueError("payload exceeds the maximum size")

    def visit(value: object, depth: int) -> None:
        if depth > MAX_PAYLOAD_DEPTH:
            raise ValueError("payload is nested too deeply")
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                visit(item, depth + 1)

    visit(payload, 0)


def _aware(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _utc(value: datetime) -> datetime:
    return _aware(value, "timestamp")
