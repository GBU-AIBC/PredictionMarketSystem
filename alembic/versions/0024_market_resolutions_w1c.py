"""Persist point-in-time market resolution labels for research replay."""

from __future__ import annotations

from alembic import op
from sqlalchemy.engine import Connection


revision = "0024_market_resolutions_w1c"
down_revision = "0023_backtest_execution_rows"
branch_labels = ("w1c",)
depends_on = None


def upgrade() -> None:
    connection: Connection = op.get_bind()
    connection.exec_driver_sql(
        "ALTER TABLE eval_records ADD COLUMN IF NOT EXISTS calibration_prob_estimate DOUBLE PRECISION"
    )
    connection.exec_driver_sql(
        """
        CREATE TABLE IF NOT EXISTS market_resolutions (
            market_id TEXT PRIMARY KEY REFERENCES markets(condition_id) ON DELETE CASCADE,
            resolved_outcome DOUBLE PRECISION NOT NULL
                CHECK (resolved_outcome >= 0.0 AND resolved_outcome <= 1.0),
            resolved_at TIMESTAMPTZ,
            source TEXT NOT NULL CHECK (source <> ''),
            observed_at TIMESTAMPTZ NOT NULL
        )
        """
    )


def downgrade() -> None:
    connection: Connection = op.get_bind()
    connection.exec_driver_sql("DROP TABLE IF EXISTS market_resolutions")
