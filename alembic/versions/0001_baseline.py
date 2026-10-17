from __future__ import annotations

from pathlib import Path
import re
from typing import Final

from alembic import op
from sqlalchemy.engine import Connection


revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None

_CREATED_TABLES: Final[tuple[str, ...]] = (
    "backtest_live_comparisons",
    "runtime_heartbeats",
    "evaluation_reports",
    "strategy_run_slices",
    "backtest_execution_rows",
    "strategy_runs",
    "backtest_runs",
    "alpha_competition_snapshots",
    "strategy_performance_peaks",
    "opportunities",
    "fills",
    "order_intents",
    "orders",
    "decisions",
    "quote_eval_records",
    "eval_records",
    "feedback",
    "strategy_factors",
    "strategy_versions",
    "strategies",
    "market_relations",
    "market_resolutions",
    "factor_values",
    "factors",
    "trades",
    "price_changes",
    "book_levels",
    "book_snapshots",
    "market_subscriptions",
    "tokens",
    "market_price_snapshots",
    "markets",
)

_DEFERRED_SCHEMA_STATEMENT_PREFIXES: Final[tuple[str, ...]] = (
    # Wave 2 contribution tables are created by their own revision so an
    # upgrade from the pre-Wave-2 head and a fresh schema stay equivalent.
    "CREATE TABLE IF NOT EXISTS agent_evidence",
    "CREATE INDEX IF NOT EXISTS idx_agent_evidence_market_captured",
    "CREATE TABLE IF NOT EXISTS agent_factor_provenance",
    "CREATE INDEX IF NOT EXISTS idx_agent_factor_provenance_market_ts",
    "CREATE TABLE IF NOT EXISTS llm_replay_records",
    "CREATE INDEX IF NOT EXISTS idx_llm_replay_records_recorded_at",
    "CREATE TABLE IF NOT EXISTS strategy_judgement_artifacts",
    "CREATE TABLE IF NOT EXISTS strategy_execution_artifacts",
    "CREATE INDEX IF NOT EXISTS idx_strategy_judgement_artifacts",
    "CREATE INDEX IF NOT EXISTS idx_strategy_execution_artifacts",
)
_DOLLAR_QUOTE_RE: Final[re.Pattern[str]] = re.compile(
    r"\$[A-Za-z_][A-Za-z_0-9]*\$|\$\$"
)


def _schema_sql_path() -> Path:
    return Path(__file__).resolve().parents[2] / "schema.sql"


def _migration_schema_sql() -> str:
    schema_sql = _schema_sql_path().read_text(encoding="utf-8").strip()
    lines = schema_sql.splitlines()
    if lines and lines[0].strip() == "BEGIN;":
        lines = lines[1:]
    if lines and lines[-1].strip() == "COMMIT;":
        lines = lines[:-1]
    statements = [
        f"{statement};"
        for statement in _split_sql_statements("\n".join(lines))
        if not _is_deferred_schema_statement(statement)
    ]
    return "\n\n".join(statements).strip() + "\n"


def _split_sql_statements(schema_sql: str) -> tuple[str, ...]:
    statements: list[str] = []
    start = 0
    index = 0
    in_single_quote = False
    dollar_quote_tag: str | None = None
    while index < len(schema_sql):
        if dollar_quote_tag is not None:
            if schema_sql.startswith(dollar_quote_tag, index):
                index += len(dollar_quote_tag)
                dollar_quote_tag = None
                continue
            index += 1
            continue

        char = schema_sql[index]
        if in_single_quote:
            if char == "'" and schema_sql[index : index + 2] == "''":
                index += 2
                continue
            if char == "'":
                in_single_quote = False
            index += 1
            continue

        if char == "'":
            in_single_quote = True
            index += 1
            continue
        if schema_sql.startswith("--", index):
            newline_index = schema_sql.find("\n", index + 2)
            if newline_index == -1:
                break
            index = newline_index + 1
            continue
        if schema_sql.startswith("/*", index):
            comment_end = schema_sql.find("*/", index + 2)
            if comment_end == -1:
                break
            index = comment_end + 2
            continue
        if char == "$":
            match = _DOLLAR_QUOTE_RE.match(schema_sql, index)
            if match is not None:
                dollar_quote_tag = match.group(0)
                index = match.end()
                continue
        if char == ";":
            statement = schema_sql[start:index].strip()
            if statement:
                statements.append(statement)
            start = index + 1
        index += 1

    trailing = schema_sql[start:].strip()
    if trailing:
        statements.append(trailing)
    return tuple(statements)


def _is_deferred_schema_statement(statement: str) -> bool:
    normalized = " ".join(statement.split()).upper()
    return any(
        normalized.startswith(prefix.upper())
        for prefix in _DEFERRED_SCHEMA_STATEMENT_PREFIXES
    )


def upgrade() -> None:
    connection: Connection = op.get_bind()
    connection.exec_driver_sql(_migration_schema_sql())


def downgrade() -> None:
    connection: Connection = op.get_bind()
    for table_name in _CREATED_TABLES:
        connection.exec_driver_sql(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
