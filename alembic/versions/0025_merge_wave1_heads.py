"""Merge the Wave 1 factor-version and market-resolution migration heads."""

from __future__ import annotations


revision = "0025_merge_wave1_heads"
down_revision = ("0024_factor_def_version_w1b", "0024_market_resolutions_w1c")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
