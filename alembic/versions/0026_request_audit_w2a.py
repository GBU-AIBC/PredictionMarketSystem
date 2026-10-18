"""Persist scoped API request outcomes for operator audit."""

from alembic import op

revision = "0026_request_audit_w2a"
down_revision = "0025_merge_wave1_heads"
branch_labels = ("w2a",)
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS api_request_audit (
            audit_id BIGSERIAL PRIMARY KEY,
            token_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            method TEXT NOT NULL,
            path TEXT NOT NULL,
            outcome INTEGER NOT NULL CHECK (outcome BETWEEN 100 AND 599),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_api_request_audit_created_at "
        "ON api_request_audit (created_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS api_request_audit")
