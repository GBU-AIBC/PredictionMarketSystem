from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

SCHEMA_PATH = Path("schema.sql")
PMS_TEST_DATABASE_URL = os.environ.get("PMS_TEST_DATABASE_URL")

EXPECTED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "strategies": [
        ("strategy_id", "text"),
        ("active_version_id", "text"),
        ("created_at", "timestamp with time zone"),
        ("metadata_json", "jsonb"),
        ("title", "text"),
        ("description", "text"),
        ("archived", "boolean"),
        ("share_enabled", "boolean"),
    ],
    "strategy_versions": [
        ("strategy_version_id", "text"),
        ("strategy_id", "text"),
        ("config_json", "jsonb"),
        ("metadata_json", "jsonb"),
        ("created_at", "timestamp with time zone"),
    ],
    "strategy_factors": [
        ("strategy_id", "text"),
        ("strategy_version_id", "text"),
        ("factor_id", "text"),
        ("param", "jsonb"),
        ("weight", "double precision"),
        ("direction", "text"),
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


def _default_strategy_config_json() -> str:
    return (
        '{"config":{"factor_composition":[["factor-a",0.6],["factor-b",0.4]],'
        '"metadata":[["owner","system"],["tier","default"]],"strategy_id":"default"},'
        '"eval_spec":{"max_brier_score":0.3,"metrics":["brier","pnl","fill_rate"],'
        '"min_win_rate":0.5,"slippage_threshold_bps":50.0},'
        '"forecaster":{"forecasters":[["rules",[["threshold","0.55"]]],'
        '["stats",[["window","15m"]]],["llm",[]]]},'
        '"market_selection":{"resolution_time_max_horizon_days":7,'
        '"venue":"polymarket","volume_min_usdc":500.0},'
        '"risk":{"max_daily_drawdown_pct":2.5,'
        '"max_position_notional_usdc":100.0,"min_order_size_usdc":1.0}}'
    )


def test_schema_sql_applies_strategy_identity_tables() -> None:
    assert PMS_TEST_DATABASE_URL is not None

    schema_text = SCHEMA_PATH.read_text()
    assert "-- strategies: inner-ring identity table (Invariants 3, 8)" in schema_text
    assert "-- strategy_versions: immutable hash-keyed version rows (Invariant 3)" in schema_text
    assert (
        "-- strategy_factors: link table (empty in S2; populated by S3). Columns "
        "declared so S3 inserts do not require a schema change (Invariants 2, 4, 8)"
        in schema_text
    )

    admin_database_url = _replace_database(PMS_TEST_DATABASE_URL, "postgres")
    temp_database = f"pms_cp02_{uuid.uuid4().hex[:12]}"
    temp_database_url = _replace_database(PMS_TEST_DATABASE_URL, temp_database)

    try:
        _run_psql(admin_database_url, "-c", f"CREATE DATABASE {temp_database}")
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
              AND table_name IN ('strategies', 'strategy_versions', 'strategy_factors')
            ORDER BY table_name, ordinal_position
            """,
        )

        actual_columns: dict[str, list[tuple[str, str]]] = {}
        for line in columns_result.stdout.splitlines():
            table_name, column_name, data_type = line.split("|")
            actual_columns.setdefault(table_name, []).append((column_name, data_type))

        assert actual_columns == EXPECTED_COLUMNS

        share_default = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            "INSERT INTO strategies (strategy_id) VALUES ('private') RETURNING share_enabled",
        )
        assert share_default.stdout.splitlines()[0] == "f"
        _run_psql(
            temp_database_url,
            "-c",
            "DELETE FROM strategies WHERE strategy_id = 'private'",
        )

        counts_result = _run_psql(
            temp_database_url,
            "-At",
            "-F",
            "|",
            "-c",
            """
            SELECT 'strategies', COUNT(*) FROM strategies
            UNION ALL
            SELECT 'strategy_versions', COUNT(*) FROM strategy_versions
            UNION ALL
            SELECT 'strategy_factors', COUNT(*) FROM strategy_factors
            ORDER BY 1
            """,
        )

        actual_counts = dict(
            (table_name, int(count))
            for table_name, count in (
                line.split("|") for line in counts_result.stdout.splitlines()
            )
        )

        assert actual_counts == {
            "strategy_factors": 0,
            "strategies": 1,
            "strategy_versions": 1,
        }

        seed_result = _run_psql(
            temp_database_url,
            "-At",
            "-F",
            "|",
            "-c",
            """
            SELECT
                strategy_id,
                active_version_id,
                (
                    SELECT config_json::text
                    FROM strategy_versions
                    WHERE strategy_version_id = 'default-v1'
                )
            FROM strategies
            WHERE strategy_id = 'default'
            """,
        )

        strategy_id, active_version_id, config_json = seed_result.stdout.strip().split("|", 2)
        assert strategy_id == "default"
        assert active_version_id == "default-v1"
        assert json.loads(config_json) == json.loads(_default_strategy_config_json())
    finally:
        _run_psql(
            admin_database_url,
            "-c",
            f"DROP DATABASE IF EXISTS {temp_database} WITH (FORCE)",
        )
