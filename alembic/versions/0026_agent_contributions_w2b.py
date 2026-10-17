"""Persist agent evidence and factor provenance."""

from __future__ import annotations

from alembic import op


revision = "0026_agent_contrib_w2b"
down_revision = "0025_merge_wave1_heads"
branch_labels = ("w2b",)
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_evidence (
            evidence_id TEXT PRIMARY KEY,
            market_id TEXT NOT NULL REFERENCES markets(condition_id) ON DELETE CASCADE,
            source TEXT NOT NULL CHECK (length(source) BETWEEN 1 AND 512),
            model TEXT NOT NULL CHECK (length(model) BETWEEN 1 AND 256),
            url TEXT NOT NULL CHECK (length(url) BETWEEN 1 AND 2048),
            summary TEXT NOT NULL CHECK (length(summary) BETWEEN 1 AND 8192),
            payload JSONB NOT NULL,
            content_hash TEXT NOT NULL UNIQUE CHECK (content_hash ~ '^[0-9a-f]{64}$'),
            captured_at TIMESTAMPTZ NOT NULL,
            expires_at TIMESTAMPTZ,
            submitted_by_token_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CHECK (expires_at IS NULL OR expires_at > captured_at)
        );
        CREATE INDEX IF NOT EXISTS idx_agent_evidence_market_captured
            ON agent_evidence (market_id, captured_at DESC);

        CREATE TABLE IF NOT EXISTS agent_factor_provenance (
            factor_id TEXT NOT NULL,
            param TEXT NOT NULL DEFAULT '',
            market_id TEXT NOT NULL REFERENCES markets(condition_id) ON DELETE CASCADE,
            ts TIMESTAMPTZ NOT NULL,
            definition_version TEXT NOT NULL DEFAULT '1',
            model TEXT NOT NULL CHECK (length(model) BETWEEN 1 AND 256),
            prompt_hash TEXT NOT NULL CHECK (prompt_hash ~ '^[0-9a-f]{64}$'),
            evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
            submitted_by_token_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (factor_id, param, market_id, ts, definition_version)
        );
        CREATE INDEX IF NOT EXISTS idx_agent_factor_provenance_market_ts
            ON agent_factor_provenance (market_id, ts DESC);

        INSERT INTO factors (
            factor_id, name, description, input_schema_hash, output_type, direction, owner
        ) VALUES (
            'agent_probability',
            'Agent Probability',
            'Externally submitted point-in-time probability from an agent.',
            '3e8b3b26e3d6a9f3f7db8eb5d651c7adf3a2d3cf9f6f21f4b5b11f4aa2a2c5ef',
            'probability', 'neutral', 'agent:llm'
        ) ON CONFLICT (factor_id) DO UPDATE SET
            name = EXCLUDED.name,
            description = EXCLUDED.description,
            input_schema_hash = EXCLUDED.input_schema_hash,
            output_type = EXCLUDED.output_type,
            direction = EXCLUDED.direction,
            owner = EXCLUDED.owner;
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_factor_provenance")
    op.execute("DROP TABLE IF EXISTS agent_evidence")
