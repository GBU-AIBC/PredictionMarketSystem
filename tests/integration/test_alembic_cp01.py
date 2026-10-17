from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PMS_TEST_DATABASE_URL = os.environ.get("PMS_TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("PMS_RUN_INTEGRATION") != "1",
        reason="set PMS_RUN_INTEGRATION=1 to run PostgreSQL integration tests",
    ),
    pytest.mark.skipif(
        not PMS_TEST_DATABASE_URL,
        reason="set PMS_TEST_DATABASE_URL to the compose-backed PostgreSQL URI",
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
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )


def _alembic_command() -> list[str]:
    executable = REPO_ROOT / ".venv" / "bin" / "alembic"
    if executable.exists():
        return [str(executable)]

    resolved = shutil.which("alembic")
    if resolved is None:
        pytest.fail("alembic executable is not available in the active environment")
    return [resolved]


def _run_alembic(database_url: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    env.pop("PMS_DATABASE__DSN", None)
    env.pop("PMS_DATABASE_URL", None)
    return subprocess.run(
        [*_alembic_command(), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=60,
    )


def test_alembic_cp01_reports_version_and_repository_head() -> None:
    assert PMS_TEST_DATABASE_URL
    version = _run_alembic(PMS_TEST_DATABASE_URL, "--version")
    heads = _run_alembic(PMS_TEST_DATABASE_URL, "heads")

    assert version.returncode == 0, version.stdout + version.stderr
    assert "alembic" in version.stdout.lower()
    assert heads.returncode == 0, heads.stdout + heads.stderr
    head_lines = [line for line in heads.stdout.splitlines() if line.strip()]
    assert len(head_lines) == 1
    assert head_lines[0].endswith("(head)")


def test_alembic_cp01_migration_commands_roundtrip_on_temp_database() -> None:
    assert PMS_TEST_DATABASE_URL
    admin_database_url = _replace_database(PMS_TEST_DATABASE_URL, "postgres")
    temp_database = f"pms_alembic_cp01_{uuid.uuid4().hex[:12]}"
    temp_database_url = _replace_database(PMS_TEST_DATABASE_URL, temp_database)

    try:
        _run_psql(admin_database_url, "-c", f"CREATE DATABASE {temp_database}")

        current = _run_alembic(temp_database_url, "current")
        upgrade = _run_alembic(temp_database_url, "upgrade", "head")
        downgrade = _run_alembic(temp_database_url, "downgrade", "base")

        assert current.returncode == 0, current.stdout + current.stderr
        assert current.stdout.strip() == ""
        assert upgrade.returncode == 0, upgrade.stdout + upgrade.stderr
        assert downgrade.returncode == 0, downgrade.stdout + downgrade.stderr
    finally:
        _run_psql(
            admin_database_url,
            "-c",
            f"DROP DATABASE IF EXISTS {temp_database} WITH (FORCE)",
        )


def test_factor_definition_version_migration_preserves_historical_rows() -> None:
    """0024 adds provenance without deleting immutable factor history."""
    assert PMS_TEST_DATABASE_URL
    admin_database_url = _replace_database(PMS_TEST_DATABASE_URL, "postgres")
    temp_database = f"pms_alembic_factor_rows_{uuid.uuid4().hex[:12]}"
    temp_database_url = _replace_database(PMS_TEST_DATABASE_URL, temp_database)

    seed_sql = """
    INSERT INTO markets (
        condition_id, slug, question, venue, created_at, last_seen_at
    ) VALUES (
        'factor-migration-market', 'factor-migration-market',
        'Will migration preserve this market?', 'polymarket', now(), now()
    );
    INSERT INTO factors (
        factor_id, name, description, input_schema_hash, default_params,
        output_type, direction, owner
    ) VALUES (
        'subset_relation', 'Subset relation', 'legacy relation factor', 'hash',
        '{}'::jsonb, 'scalar', 'neutral', 'system'
    );
    INSERT INTO factor_values (
        factor_id, param, market_id, ts, value
    ) VALUES (
        'subset_relation', '', 'factor-migration-market', now(), 0.42
    );
    INSERT INTO strategies (strategy_id) VALUES ('factor-migration-strategy');
    INSERT INTO strategy_versions (
        strategy_version_id, strategy_id, config_json
    ) VALUES (
        'factor-migration-version', 'factor-migration-strategy', '{}'::jsonb
    );
    INSERT INTO strategy_factors (
        strategy_id, strategy_version_id, factor_id, param, weight, direction
    ) VALUES (
        'factor-migration-strategy', 'factor-migration-version',
        'subset_relation', '{}'::jsonb, 1.0, 'long'
    );
    """
    count_sql = """
    SELECT 'factors', COUNT(*) FROM factors
    UNION ALL SELECT 'factor_values', COUNT(*) FROM factor_values
    UNION ALL SELECT 'strategy_factors', COUNT(*) FROM strategy_factors
    ORDER BY 1
    """

    try:
        _run_psql(admin_database_url, "-c", f"CREATE DATABASE {temp_database}")
        upgrade_0023 = _run_alembic(
            temp_database_url,
            "upgrade",
            "0023_backtest_execution_rows",
        )
        assert upgrade_0023.returncode == 0, upgrade_0023.stderr
        _run_psql(temp_database_url, "-c", seed_sql)

        before = _run_psql(temp_database_url, "-At", "-c", count_sql).stdout
        upgrade_head = _run_alembic(temp_database_url, "upgrade", "head")
        assert upgrade_head.returncode == 0, upgrade_head.stderr
        after = _run_psql(temp_database_url, "-At", "-c", count_sql).stdout
        assert after == before
        definition_column = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_name = 'factor_values' AND column_name = 'definition_version'",
        ).stdout.strip()
        assert definition_column == "1"

        downgrade = _run_alembic(
            temp_database_url,
            "downgrade",
            "0023_backtest_execution_rows",
        )
        assert downgrade.returncode == 0, downgrade.stderr
        assert _run_psql(temp_database_url, "-At", "-c", count_sql).stdout == before
        definition_column = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_name = 'factor_values' AND column_name = 'definition_version'",
        ).stdout.strip()
        assert definition_column == "0"
    finally:
        _run_psql(
            admin_database_url,
            "-c",
            f"DROP DATABASE IF EXISTS {temp_database} WITH (FORCE)",
        )


def test_factor_definition_version_downgrade_rejects_duplicate_versions() -> None:
    assert PMS_TEST_DATABASE_URL
    admin_database_url = _replace_database(PMS_TEST_DATABASE_URL, "postgres")
    temp_database = f"pms_alembic_factor_dupes_{uuid.uuid4().hex[:12]}"
    temp_database_url = _replace_database(PMS_TEST_DATABASE_URL, temp_database)
    seed_sql = """
    INSERT INTO markets (
        condition_id, slug, question, venue, created_at, last_seen_at
    ) VALUES (
        'factor-duplicate-market', 'factor-duplicate-market',
        'Will duplicate definitions survive downgrade?', 'polymarket', now(), now()
    );
    INSERT INTO factors (
        factor_id, name, description, input_schema_hash, default_params,
        output_type, direction, owner
    ) VALUES (
        'orderbook_imbalance', 'Orderbook', 'factor', 'hash', '{}'::jsonb,
        'scalar', 'neutral', 'system'
        ) ON CONFLICT (factor_id) DO NOTHING;
    """
    try:
        _run_psql(admin_database_url, "-c", f"CREATE DATABASE {temp_database}")
        upgrade_0023 = _run_alembic(
            temp_database_url,
            "upgrade",
            "0023_backtest_execution_rows",
        )
        assert upgrade_0023.returncode == 0, upgrade_0023.stderr
        _run_psql(temp_database_url, "-c", seed_sql)
        upgrade_head = _run_alembic(temp_database_url, "upgrade", "head")
        assert upgrade_head.returncode == 0, upgrade_head.stderr
        _run_psql(
            temp_database_url,
            "-c",
            """
            INSERT INTO factor_values (
                factor_id, param, market_id, ts, value, definition_version
            ) VALUES
                ('orderbook_imbalance', '', 'factor-duplicate-market',
                 '2026-01-01T00:00:00Z', 0.1, '1'),
                ('orderbook_imbalance', '', 'factor-duplicate-market',
                 '2026-01-01T00:00:00Z', 0.9, '2')
            """,
        )
        downgrade = _run_alembic(
            temp_database_url,
            "downgrade",
            "0023_backtest_execution_rows",
        )
        assert downgrade.returncode != 0
        assert "coexisting definition versions" in downgrade.stderr
        current = _run_alembic(temp_database_url, "current")
        # The failed guarded downgrade rolls back as one transaction, so the
        # database stays at the current integration head.
        heads = _run_alembic(temp_database_url, "heads")
        current_revision = current.stdout.strip().split(maxsplit=1)[0]
        assert current_revision in heads.stdout
        column = _run_psql(
            temp_database_url,
            "-At",
            "-c",
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_name = 'factor_values' AND column_name = 'definition_version'",
        ).stdout.strip()
        assert column == "1"
    finally:
        _run_psql(
            admin_database_url,
            "-c",
            f"DROP DATABASE IF EXISTS {temp_database} WITH (FORCE)",
        )
