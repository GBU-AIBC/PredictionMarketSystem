"""Record the raw factor definition version beside each observation."""

from __future__ import annotations

from alembic import op
from sqlalchemy.engine import Connection


revision = "0024_factor_def_version_w1b"
down_revision = "0023_backtest_execution_rows"
branch_labels = ("w1b",)
depends_on = None


def upgrade() -> None:
    connection: Connection = op.get_bind()
    connection.exec_driver_sql(
        """
        ALTER TABLE factor_values
            ADD COLUMN IF NOT EXISTS definition_version TEXT NOT NULL DEFAULT '1'
        """
    )
    connection.exec_driver_sql(
        """
        DROP INDEX IF EXISTS idx_factor_values_factor_param_market_ts;
        CREATE UNIQUE INDEX idx_factor_values_factor_param_market_ts
            ON factor_values(factor_id, param, market_id, ts DESC, definition_version)
        """
    )


def downgrade() -> None:
    connection: Connection = op.get_bind()
    connection.exec_driver_sql(
        "ALTER TABLE factor_values DROP COLUMN IF EXISTS definition_version"
    )
    connection.exec_driver_sql(
        """
        DROP INDEX IF EXISTS idx_factor_values_factor_param_market_ts;
        CREATE UNIQUE INDEX idx_factor_values_factor_param_market_ts
            ON factor_values(factor_id, param, market_id, ts DESC)
        """
    )
