"""Store controller rejection counts separately from research inputs."""

from __future__ import annotations

from alembic import op


revision = "0029_controller_rejections"
down_revision = "0028_strategy_share_opt_in"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.get_bind().exec_driver_sql(
        "ALTER TABLE strategy_runs ADD COLUMN IF NOT EXISTS "
        "controller_rejection_counts_json JSONB NOT NULL DEFAULT '{}'::jsonb"
    )


def downgrade() -> None:
    op.get_bind().exec_driver_sql(
        "ALTER TABLE strategy_runs DROP COLUMN IF EXISTS controller_rejection_counts_json"
    )
