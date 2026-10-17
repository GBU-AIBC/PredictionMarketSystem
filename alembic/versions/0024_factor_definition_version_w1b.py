"""Record the raw factor definition version beside each observation."""

from __future__ import annotations

from alembic import op
from sqlalchemy.engine import Connection


revision = "0024_factor_definition_version_w1b"
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


def downgrade() -> None:
    connection: Connection = op.get_bind()
    connection.exec_driver_sql(
        "ALTER TABLE factor_values DROP COLUMN IF EXISTS definition_version"
    )
