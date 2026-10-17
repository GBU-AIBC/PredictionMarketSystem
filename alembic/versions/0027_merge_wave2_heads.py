"""Merge the Wave 2 request-audit and agent-contribution migration heads."""

from __future__ import annotations


revision = "0027_merge_wave2_heads"
down_revision = ("0026_request_audit_w2a", "0026_agent_contrib_w2b")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
