from __future__ import annotations

import os
import subprocess
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest


PMS_TEST_DATABASE_URL = os.environ.get("PMS_TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("PMS_RUN_INTEGRATION") != "1",
        reason="set PMS_RUN_INTEGRATION=1 to run PostgreSQL integration tests",
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


def _run_psql(database_url: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["psql", database_url, "--set", "ON_ERROR_STOP=1", *args],
        text=True,
        capture_output=True,
        check=True,
    )


def _run_alembic(database_url: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    env.pop("PMS_DATABASE_URL", None)
    return subprocess.run(
        ["uv", "run", "alembic", *args],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_alembic_share_projection_upgrade_and_downgrade_round_trip() -> None:
    assert PMS_TEST_DATABASE_URL is not None

    admin_database_url = _replace_database(PMS_TEST_DATABASE_URL, "postgres")
    temp_database = f"pms_cp11_share_{uuid.uuid4().hex[:8]}"
    temp_database_url = _replace_database(PMS_TEST_DATABASE_URL, temp_database)

    try:
        _run_psql(admin_database_url, "-c", f"CREATE DATABASE {temp_database}")
        previous = _run_alembic(temp_database_url, "upgrade", "0004_decisions_table")
        assert previous.returncode == 0, previous.stderr
        # The baseline reads today's schema; reconstruct the pre-sharing table.
        _run_psql(
            temp_database_url,
            "-c",
            """
            ALTER TABLE strategies
                DROP COLUMN title,
                DROP COLUMN description,
                DROP COLUMN archived,
                DROP COLUMN share_enabled
            """,
        )
        upgrade = _run_alembic(
            temp_database_url,
            "upgrade",
            "0005_strategies_share_metadata",
        )
        assert upgrade.returncode == 0, upgrade.stderr

        upgraded_columns = _run_psql(
            temp_database_url,
            "-At",
            "-F",
            "|",
            "-c",
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'strategies'
            ORDER BY ordinal_position
            """,
        )
        assert upgraded_columns.stdout.splitlines() == [
            "strategy_id|text|NO|",
            "active_version_id|text|YES|",
            "created_at|timestamp with time zone|NO|now()",
            "metadata_json|jsonb|NO|'{}'::jsonb",
            "title|text|YES|",
            "description|text|YES|",
            "archived|boolean|NO|false",
            "share_enabled|boolean|NO|true",
        ]

        downgrade = _run_alembic(temp_database_url, "downgrade", "-1")
        assert downgrade.returncode == 0, downgrade.stderr

        downgraded_columns = _run_psql(
            temp_database_url,
            "-At",
            "-F",
            "|",
            "-c",
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'strategies'
            ORDER BY ordinal_position
            """,
        )
        assert downgraded_columns.stdout.splitlines() == [
            "strategy_id|text|NO|",
            "active_version_id|text|YES|",
            "created_at|timestamp with time zone|NO|now()",
            "metadata_json|jsonb|NO|'{}'::jsonb",
        ]

        reupgrade = _run_alembic(temp_database_url, "upgrade", "head")
        assert reupgrade.returncode == 0, reupgrade.stderr
    finally:
        _run_psql(
            admin_database_url,
            "-c",
            f"DROP DATABASE IF EXISTS {temp_database} WITH (FORCE)",
        )


def test_share_default_migration_preserves_existing_rows_in_both_directions() -> None:
    assert PMS_TEST_DATABASE_URL is not None
    admin_database_url = _replace_database(PMS_TEST_DATABASE_URL, "postgres")
    temp_database = f"pms_share_default_{uuid.uuid4().hex[:8]}"
    temp_database_url = _replace_database(PMS_TEST_DATABASE_URL, temp_database)
    snapshot_sql = """
        SELECT jsonb_build_object(
            'strategies', (SELECT jsonb_agg(to_jsonb(s) ORDER BY strategy_id) FROM strategies s),
            'versions', (SELECT jsonb_agg(to_jsonb(v) ORDER BY strategy_version_id) FROM strategy_versions v)
        )
    """

    try:
        _run_psql(admin_database_url, "-c", f"CREATE DATABASE {temp_database}")
        previous = _run_alembic(temp_database_url, "upgrade", "0027_merge_wave2_heads")
        assert previous.returncode == 0, previous.stderr
        # Model an existing deployment, independently of the fresh schema default.
        _run_psql(
            temp_database_url,
            "-c",
            """
            ALTER TABLE strategies ALTER COLUMN share_enabled SET DEFAULT TRUE;
            BEGIN;
            SET CONSTRAINTS ALL DEFERRED;
            INSERT INTO strategies
                (strategy_id, active_version_id, title, description, metadata_json, share_enabled)
            VALUES
                ('published', 'published-v1', 'Public title', 'Public description', '{"private_note":"keep"}', TRUE),
                ('private', 'private-v1', 'Private title', 'Private description', '{}', FALSE);
            INSERT INTO strategies (strategy_id) VALUES ('legacy-default');
            INSERT INTO strategy_versions (strategy_id, strategy_version_id, config_json)
            VALUES ('published', 'published-v1', '{"private_config":"keep"}'),
                   ('private', 'private-v1', '{}');
            COMMIT;
            """,
        )
        before = _run_psql(temp_database_url, "-At", "-c", snapshot_sql).stdout

        upgrade = _run_alembic(temp_database_url, "upgrade", "head")
        assert upgrade.returncode == 0, upgrade.stderr
        assert _run_psql(temp_database_url, "-At", "-c", snapshot_sql).stdout == before

        new_rows = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            """
            INSERT INTO strategies (strategy_id) VALUES ('new-default');
            INSERT INTO strategies (strategy_id, share_enabled) VALUES ('new-opt-in', TRUE);
            SELECT strategy_id, share_enabled FROM strategies
            WHERE strategy_id IN ('published', 'private', 'legacy-default', 'new-default', 'new-opt-in')
            ORDER BY strategy_id;
            """,
        )
        assert new_rows.stdout.splitlines()[-5:] == [
            "legacy-default|t",
            "new-default|f",
            "new-opt-in|t",
            "private|f",
            "published|t",
        ]

        before_downgrade = _run_psql(temp_database_url, "-At", "-c", snapshot_sql).stdout
        downgrade = _run_alembic(temp_database_url, "downgrade", "0027_merge_wave2_heads")
        assert downgrade.returncode == 0, downgrade.stderr
        assert _run_psql(temp_database_url, "-At", "-c", snapshot_sql).stdout == before_downgrade
        old_default = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            "INSERT INTO strategies (strategy_id) VALUES ('after-downgrade') RETURNING share_enabled",
        )
        assert old_default.stdout.splitlines()[0] == "t"

        before_reupgrade = _run_psql(temp_database_url, "-At", "-c", snapshot_sql).stdout
        reupgrade = _run_alembic(temp_database_url, "upgrade", "head")
        assert reupgrade.returncode == 0, reupgrade.stderr
        assert _run_psql(temp_database_url, "-At", "-c", snapshot_sql).stdout == before_reupgrade
        new_default = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            "INSERT INTO strategies (strategy_id) VALUES ('after-reupgrade') RETURNING share_enabled",
        )
        assert new_default.stdout.splitlines()[0] == "f"
    finally:
        _run_psql(
            admin_database_url,
            "-c",
            f"DROP DATABASE IF EXISTS {temp_database} WITH (FORCE)",
        )
