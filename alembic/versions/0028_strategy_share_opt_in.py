"""Make new strategies private without changing existing sharing choices."""

from __future__ import annotations

from alembic import op


revision = "0028_strategy_share_opt_in"
down_revision = "0027_merge_wave2_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    # Fresh baselines seed a strategy with a pending deferred FK check.
    connection.exec_driver_sql("SET CONSTRAINTS strategies_active_version_id_fkey IMMEDIATE")
    connection.exec_driver_sql(
        "ALTER TABLE strategies ALTER COLUMN share_enabled SET DEFAULT FALSE"
    )
    connection.exec_driver_sql("SET CONSTRAINTS strategies_active_version_id_fkey DEFERRED")


def downgrade() -> None:
    op.get_bind().exec_driver_sql(
        "ALTER TABLE strategies ALTER COLUMN share_enabled SET DEFAULT TRUE"
    )
