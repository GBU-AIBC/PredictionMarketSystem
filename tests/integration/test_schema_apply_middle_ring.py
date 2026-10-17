from __future__ import annotations

import os
import re
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest


SCHEMA_PATH = Path("schema.sql")
PMS_TEST_DATABASE_URL = os.environ.get("PMS_TEST_DATABASE_URL")

EXPECTED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "factors": [
        ("factor_id", "text"),
        ("name", "text"),
        ("description", "text"),
        ("input_schema_hash", "text"),
        ("default_params", "jsonb"),
        ("output_type", "text"),
        ("direction", "text"),
        ("owner", "text"),
    ],
    "factor_values": [
        ("id", "bigint"),
        ("factor_id", "text"),
        ("param", "text"),
        ("market_id", "text"),
        ("ts", "timestamp with time zone"),
        ("value", "double precision"),
        ("definition_version", "text"),
    ],
}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("PMS_RUN_INTEGRATION") != "1",
        reason="set PMS_RUN_INTEGRATION=1 to run PostgreSQL schema integration tests",
    ),
    pytest.mark.skipif(
        PMS_TEST_DATABASE_URL is None,
        reason="set PMS_TEST_DATABASE_URL to a PostgreSQL URI with CREATE DATABASE privileges",
    ),
]


def _replace_database(database_url: str, database_name: str) -> str:
    parts = urlsplit(database_url)
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            f"/{database_name}",
            parts.query,
            parts.fragment,
        )
    )


def _run_psql(
    database_url: str,
    *args: str,
    input_sql: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["psql", database_url, "--set", "ON_ERROR_STOP=1", *args],
        input=input_sql,
        text=True,
        capture_output=True,
        check=True,
    )


def _extract_block(schema_text: str, begin: str, end: str) -> str:
    match = re.search(rf"(?s){re.escape(begin)}\s*(.*?)\s*{re.escape(end)}", schema_text)
    assert match is not None, f"{begin} / {end} block not found"
    return match.group(1)


@pytest.mark.asyncio(loop_scope="session")
async def test_schema_sql_applies_middle_ring_tables() -> None:
    assert PMS_TEST_DATABASE_URL is not None

    schema_text = SCHEMA_PATH.read_text()
    middle_ring = _extract_block(
        schema_text,
        "-- BEGIN MIDDLE RING",
        "-- END MIDDLE RING",
    )

    assert "CREATE TABLE IF NOT EXISTS factors" in middle_ring
    assert "CREATE TABLE IF NOT EXISTS factor_values" in middle_ring
    assert "strategy_id" not in middle_ring
    assert "strategy_version_id" not in middle_ring

    admin_database_url = _replace_database(PMS_TEST_DATABASE_URL, "postgres")
    temp_database = f"pms_cp03_{uuid.uuid4().hex[:12]}"
    temp_database_url = _replace_database(PMS_TEST_DATABASE_URL, temp_database)

    try:
        _run_psql(admin_database_url, "-c", f"CREATE DATABASE {temp_database}")
        _run_psql(temp_database_url, "-f", str(SCHEMA_PATH))
        _run_psql(temp_database_url, "-f", str(SCHEMA_PATH))

        columns_result = _run_psql(
            temp_database_url,
            "-At",
            "-F",
            "|",
            "-c",
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name IN ('factors', 'factor_values')
            ORDER BY table_name, ordinal_position
            """,
        )

        actual_columns: dict[str, list[tuple[str, str]]] = {}
        for line in columns_result.stdout.splitlines():
            table_name, column_name, data_type = line.split("|")
            actual_columns.setdefault(table_name, []).append((column_name, data_type))

        assert actual_columns == EXPECTED_COLUMNS

        indexes_result = _run_psql(
            temp_database_url,
            "-At",
            "-F",
            "|",
            "-c",
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename = 'factor_values'
            ORDER BY indexname
            """,
        )
        assert "idx_factor_values_factor_param_market_ts" in indexes_result.stdout.splitlines()
        unique_index_result = _run_psql(
            temp_database_url,
            "-At",
            "-F",
            "|",
            "-c",
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND indexname = 'idx_factor_values_factor_param_market_ts'
            """,
        )
        assert "CREATE UNIQUE INDEX" in unique_index_result.stdout

        connection = await asyncpg.connect(temp_database_url)
        try:
            with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
                await connection.execute(
                    """
                    INSERT INTO strategy_factors (
                        strategy_id,
                        strategy_version_id,
                        factor_id,
                        param,
                        weight,
                        direction
                    ) VALUES (
                        'default',
                        'default-v1',
                        'missing-factor',
                        '{}'::jsonb,
                        1.0,
                        'long'
                    )
                    """
                )
        finally:
            await connection.close()
    finally:
        _run_psql(
            admin_database_url,
            "-c",
            f"DROP DATABASE IF EXISTS {temp_database} WITH (FORCE)",
        )
