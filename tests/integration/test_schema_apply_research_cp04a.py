from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest


SCHEMA_PATH = Path("schema.sql")
PMS_TEST_DATABASE_URL = os.environ.get("PMS_TEST_DATABASE_URL")

EXPECTED_NULLABILITY: dict[str, list[tuple[str, str]]] = {
    "backtest_live_comparisons": [
        ("comparison_id", "NO"),
        ("run_id", "NO"),
        ("strategy_id", "NO"),
        ("strategy_version_id", "NO"),
        ("live_window_start", "NO"),
        ("live_window_end", "NO"),
        ("denominator", "NO"),
        ("equity_delta_json", "NO"),
        ("overlap_ratio", "NO"),
        ("backtest_only_symbols", "NO"),
        ("live_only_symbols", "NO"),
        ("time_alignment_policy_json", "NO"),
        ("symbol_normalization_policy_json", "NO"),
        ("computed_at", "NO"),
    ],
    "backtest_runs": [
        ("run_id", "NO"),
        ("spec_hash", "NO"),
        ("status", "NO"),
        ("strategy_ids", "NO"),
        ("date_range_start", "NO"),
        ("date_range_end", "NO"),
        ("exec_config_json", "NO"),
        ("spec_json", "NO"),
        ("queued_at", "NO"),
        ("started_at", "YES"),
        ("finished_at", "YES"),
        ("failure_reason", "YES"),
        ("worker_pid", "YES"),
        ("worker_host", "YES"),
    ],
    "evaluation_reports": [
        ("report_id", "NO"),
        ("run_id", "NO"),
        ("ranking_metric", "NO"),
        ("ranked_strategies", "NO"),
        ("benchmark_rows", "NO"),
        ("attribution_commentary", "YES"),
        ("warnings", "NO"),
        ("next_action", "YES"),
        ("generated_at", "NO"),
    ],
    "strategy_run_slices": [
        ("strategy_run_slice_id", "NO"),
        ("run_id", "NO"),
        ("strategy_id", "NO"),
        ("strategy_version_id", "NO"),
        ("slice_label", "NO"),
        ("slice_start", "NO"),
        ("slice_end", "NO"),
        ("slice_kind", "NO"),
        ("brier", "YES"),
        ("pnl_cum", "YES"),
        ("drawdown_max", "YES"),
        ("fill_rate", "YES"),
        ("slippage_bps", "YES"),
        ("opportunity_count", "NO"),
        ("decision_count", "NO"),
        ("fill_count", "NO"),
        ("created_at", "NO"),
    ],
    "strategy_runs": [
        ("strategy_run_id", "NO"),
        ("run_id", "NO"),
        ("strategy_id", "NO"),
        ("strategy_version_id", "NO"),
        ("brier", "YES"),
        ("pnl_cum", "YES"),
        ("drawdown_max", "YES"),
        ("fill_rate", "YES"),
        ("slippage_bps", "YES"),
        ("opportunity_count", "YES"),
        ("decision_count", "YES"),
        ("fill_count", "YES"),
        ("portfolio_target_json", "YES"),
        ("started_at", "NO"),
        ("finished_at", "YES"),
        ("controller_rejection_counts_json", "NO"),
    ],
}

EXPECTED_INDEXES = [
    "idx_backtest_live_comparisons_run_strategy_identity",
    "idx_backtest_runs_queued_at_desc",
    "idx_backtest_runs_status",
    "idx_strategy_run_slices_run_strategy_identity",
    "idx_strategy_runs_run_id",
]

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


def test_schema_sql_applies_research_backtest_tables() -> None:
    assert PMS_TEST_DATABASE_URL is not None

    schema_text = SCHEMA_PATH.read_text()
    for table_name in EXPECTED_NULLABILITY:
        assert f"CREATE TABLE IF NOT EXISTS {table_name}" in schema_text

    admin_database_url = _replace_database(PMS_TEST_DATABASE_URL, "postgres")
    temp_database = f"pms_cp04a_{uuid.uuid4().hex[:12]}"
    temp_database_url = _replace_database(PMS_TEST_DATABASE_URL, temp_database)

    try:
        _run_psql(admin_database_url, "-c", f"CREATE DATABASE {temp_database}")
        _run_psql(temp_database_url, "-f", str(SCHEMA_PATH))
        _run_psql(temp_database_url, "-f", str(SCHEMA_PATH))

        tables_result = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN (
                'backtest_runs',
                'strategy_runs',
                'strategy_run_slices',
                'evaluation_reports',
                'backtest_live_comparisons'
              )
            ORDER BY table_name
            """,
        )
        assert tables_result.stdout.splitlines() == sorted(EXPECTED_NULLABILITY)

        columns_result = _run_psql(
            temp_database_url,
            "-At",
            "-F",
            "|",
            "-c",
            """
            SELECT table_name, column_name, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name IN (
                'backtest_runs',
                'strategy_runs',
                'strategy_run_slices',
                'evaluation_reports',
                'backtest_live_comparisons'
              )
            ORDER BY table_name, ordinal_position
            """,
        )

        actual_columns: dict[str, list[tuple[str, str]]] = {}
        for line in columns_result.stdout.splitlines():
            table_name, column_name, is_nullable = line.split("|")
            actual_columns.setdefault(table_name, []).append((column_name, is_nullable))

        assert actual_columns == EXPECTED_NULLABILITY

        check_constraints_result = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            """
            SELECT conname
            FROM pg_constraint
            WHERE conname IN (
                'strategy_runs_strategy_identity_check',
                'strategy_run_slices_strategy_identity_check',
                'strategy_run_slices_window_check',
                'strategy_run_slices_counts_check',
                'backtest_live_comparisons_strategy_identity_check'
            )
            ORDER BY conname
            """,
        )
        assert check_constraints_result.stdout.splitlines() == [
            "backtest_live_comparisons_strategy_identity_check",
            "strategy_run_slices_counts_check",
            "strategy_run_slices_strategy_identity_check",
            "strategy_run_slices_window_check",
            "strategy_runs_strategy_identity_check",
        ]

        unique_result = _run_psql(
            temp_database_url,
            "-At",
            "-F",
            "|",
            "-c",
            """
            SELECT constraint_name, constraint_type
            FROM information_schema.table_constraints
            WHERE table_schema = 'public'
              AND table_name = 'evaluation_reports'
              AND constraint_name = 'evaluation_reports_run_id_ranking_metric_key'
            """,
        )
        assert unique_result.stdout.strip() == (
            "evaluation_reports_run_id_ranking_metric_key|UNIQUE"
        )

        indexes_result = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND indexname IN (
                'idx_backtest_runs_status',
                'idx_backtest_runs_queued_at_desc',
                'idx_strategy_runs_run_id',
                'idx_strategy_run_slices_run_strategy_identity',
                'idx_backtest_live_comparisons_run_strategy_identity'
              )
            ORDER BY indexname
            """,
        )
        assert indexes_result.stdout.splitlines() == EXPECTED_INDEXES

        _run_psql(
            temp_database_url,
            "-c",
            """
            INSERT INTO backtest_runs (
                run_id,
                spec_hash,
                status,
                strategy_ids,
                date_range_start,
                date_range_end,
                exec_config_json,
                spec_json
            ) VALUES (
                '11111111-1111-1111-1111-111111111111',
                'spec-hash',
                'queued',
                '{}'::TEXT[],
                '2026-01-01T00:00:00Z'::timestamptz,
                '2026-01-31T00:00:00Z'::timestamptz,
                '{"chunk_days": 7, "time_budget": 1800}'::jsonb,
                '{"strategy_versions": []}'::jsonb
            );
            """,
        )

        empty_strategy_ids_result = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            """
            SELECT cardinality(strategy_ids)
            FROM backtest_runs
            WHERE run_id = '11111111-1111-1111-1111-111111111111'::uuid
            """,
        )
        assert empty_strategy_ids_result.stdout.strip() == "0"

        _run_psql(
            temp_database_url,
            "-c",
            """
            INSERT INTO strategy_runs (
                strategy_run_id,
                run_id,
                strategy_id,
                strategy_version_id
            ) VALUES (
                '22222222-2222-2222-2222-222222222222',
                '11111111-1111-1111-1111-111111111111'::uuid,
                'alpha',
                'alpha-v1'
            );
            """,
        )

        rejection_counts = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            "SELECT controller_rejection_counts_json FROM strategy_runs",
        )
        assert rejection_counts.stdout.strip() == "{}"

        explain_result = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            """
            SET enable_seqscan = off;
            EXPLAIN (COSTS OFF)
            SELECT strategy_run_id
            FROM strategy_runs
            WHERE run_id = '11111111-1111-1111-1111-111111111111'::uuid;
            """,
        )
        assert "idx_strategy_runs_run_id" in explain_result.stdout
        assert "Seq Scan on strategy_runs" not in explain_result.stdout
    finally:
        _run_psql(
            admin_database_url,
            "-c",
            f"DROP DATABASE IF EXISTS {temp_database} WITH (FORCE)",
        )


def test_controller_rejections_migration_preserves_research_inputs() -> None:
    assert PMS_TEST_DATABASE_URL is not None
    admin_database_url = _replace_database(PMS_TEST_DATABASE_URL, "postgres")
    temp_database = f"pms_rejection_counts_{uuid.uuid4().hex[:8]}"
    temp_database_url = _replace_database(PMS_TEST_DATABASE_URL, temp_database)

    def run_alembic(*args: str) -> None:
        env = os.environ.copy()
        env["DATABASE_URL"] = temp_database_url
        env.pop("PMS_DATABASE_URL", None)
        result = subprocess.run(
            ["uv", "run", "alembic", *args],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    snapshot_sql = """
        SELECT to_jsonb(s) - 'controller_rejection_counts_json', to_jsonb(b)
        FROM strategy_runs s JOIN backtest_runs b USING (run_id)
    """

    try:
        _run_psql(admin_database_url, "-c", f"CREATE DATABASE {temp_database}")
        run_alembic("upgrade", "0028_strategy_share_opt_in")
        # Reconstruct the deployed schema before the new output column existed.
        _run_psql(
            temp_database_url,
            "-c",
            """
            ALTER TABLE strategy_runs DROP COLUMN IF EXISTS controller_rejection_counts_json;
            INSERT INTO backtest_runs (
                run_id, spec_hash, status, strategy_ids, date_range_start, date_range_end,
                exec_config_json, spec_json
            ) VALUES (
                '11111111-1111-1111-1111-111111111111', 'retained', 'completed', ARRAY['alpha'],
                '2026-01-01', '2026-01-31', '{"chunk_days":7}', '{}'
            );
            INSERT INTO strategy_runs (
                strategy_run_id, run_id, strategy_id, strategy_version_id, portfolio_target_json
            ) VALUES (
                '22222222-2222-2222-2222-222222222222', '11111111-1111-1111-1111-111111111111',
                'alpha', 'alpha-v1', '[{"market_id":"retained"}]'
            );
            """,
        )
        before = _run_psql(temp_database_url, "-At", "-c", snapshot_sql).stdout
        run_alembic("upgrade", "head")
        assert _run_psql(temp_database_url, "-At", "-c", snapshot_sql).stdout == before
        counts = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            "SELECT controller_rejection_counts_json FROM strategy_runs",
        )
        assert counts.stdout.strip() == "{}"
        _run_psql(
            temp_database_url,
            "-c",
            """
            UPDATE strategy_runs SET controller_rejection_counts_json = '{"no_positive_edge":3}';
            """,
        )
        run_alembic("upgrade", "head")
        counts = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            "SELECT controller_rejection_counts_json FROM strategy_runs",
        )
        assert counts.stdout.strip() == '{"no_positive_edge": 3}'
        run_alembic("downgrade", "0028_strategy_share_opt_in")
        assert _run_psql(temp_database_url, "-At", "-c", snapshot_sql).stdout == before
    finally:
        _run_psql(
            admin_database_url,
            "-c",
            f"DROP DATABASE IF EXISTS {temp_database} WITH (FORCE)",
        )
