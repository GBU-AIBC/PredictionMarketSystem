from __future__ import annotations

from pathlib import Path

import asyncpg
from alembic.config import Config
from alembic.script import ScriptDirectory

from pms.core.exceptions import PMSBootError


ROOT = Path(__file__).resolve().parents[3]
ALEMBIC_INI_PATH = ROOT / "alembic.ini"
EXPECTED_SCHEMA_HEAD: str | None
EXPECTED_SCHEMA_HEAD_ERROR: Exception | None


class SchemaVersionMismatchError(PMSBootError):
    """Raised when the runtime database schema does not match the Alembic head."""


def _alembic_config() -> Config:
    return Config(str(ALEMBIC_INI_PATH))


def _load_expected_schema_head() -> str:
    head = ScriptDirectory.from_config(_alembic_config()).get_current_head()
    if head is None:
        msg = "Alembic did not report a current head revision"
        raise RuntimeError(msg)
    return head


try:
    EXPECTED_SCHEMA_HEAD = _load_expected_schema_head()
    EXPECTED_SCHEMA_HEAD_ERROR = None
except Exception as exc:  # pragma: no cover - exercised via reload test
    EXPECTED_SCHEMA_HEAD = None
    EXPECTED_SCHEMA_HEAD_ERROR = exc


async def ensure_schema_current(pool: asyncpg.Pool) -> None:
    if EXPECTED_SCHEMA_HEAD_ERROR is not None:
        msg = (
            "migration graph inconsistent — check `alembic/versions/` "
            "for typos in `down_revision`"
        )
        raise SchemaVersionMismatchError(msg) from EXPECTED_SCHEMA_HEAD_ERROR

    connection = await pool.acquire()
    try:
        try:
            observed_head = await connection.fetchval(
                "SELECT version_num FROM alembic_version LIMIT 1"
            )
        except asyncpg.UndefinedTableError as exc:
            msg = "schema not initialized — run `uv run alembic upgrade head`"
            raise SchemaVersionMismatchError(msg) from exc
    finally:
        await pool.release(connection)

    observed_version = "" if observed_head is None else str(observed_head).strip()
    if not observed_version:
        msg = "schema not initialized — run `uv run alembic upgrade head`"
        raise SchemaVersionMismatchError(msg)

    expected_head = EXPECTED_SCHEMA_HEAD
    if expected_head is None:
        msg = (
            "migration graph inconsistent — check `alembic/versions/` "
            "for typos in `down_revision`"
        )
        raise SchemaVersionMismatchError(msg)

    if observed_version != expected_head:
        msg = (
            "schema out of date — observed "
            f"{observed_version}, expected {expected_head}; "
            "run `uv run alembic upgrade head`"
        )
        raise SchemaVersionMismatchError(msg)
