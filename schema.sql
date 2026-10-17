BEGIN;

-- BEGIN OUTER RING

CREATE TABLE IF NOT EXISTS markets (
    condition_id TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    question TEXT NOT NULL,
    -- Kalshi is a reserved venue enum: schema can represent it, while all
    -- runtime sensor/actuator paths fail closed until a real adapter ships.
    venue TEXT NOT NULL CHECK (venue IN ('polymarket', 'kalshi')),
    resolves_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    volume_24h DOUBLE PRECISION
);

ALTER TABLE markets
    ADD COLUMN IF NOT EXISTS volume_24h DOUBLE PRECISION;

ALTER TABLE markets
    ADD COLUMN IF NOT EXISTS yes_price NUMERIC(6,4),
    ADD COLUMN IF NOT EXISTS no_price NUMERIC(6,4),
    ADD COLUMN IF NOT EXISTS best_bid NUMERIC(6,4),
    ADD COLUMN IF NOT EXISTS best_ask NUMERIC(6,4),
    ADD COLUMN IF NOT EXISTS last_trade_price NUMERIC(6,4),
    ADD COLUMN IF NOT EXISTS liquidity NUMERIC,
    ADD COLUMN IF NOT EXISTS spread_bps INTEGER,
    ADD COLUMN IF NOT EXISTS price_updated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS active BOOLEAN,
    ADD COLUMN IF NOT EXISTS closed BOOLEAN,
    ADD COLUMN IF NOT EXISTS accepting_orders BOOLEAN,
    ADD COLUMN IF NOT EXISTS status_updated_at TIMESTAMPTZ;

-- Risk metadata (migration 0021) is applied last so the fresh-apply column
-- order in this canonical schema matches a database migrated to head.
ALTER TABLE markets
    ADD COLUMN IF NOT EXISTS risk_group_id TEXT,
    ADD COLUMN IF NOT EXISTS category TEXT,
    ADD COLUMN IF NOT EXISTS event_id TEXT;

CREATE INDEX IF NOT EXISTS idx_markets_price_updated_at
    ON markets (price_updated_at DESC)
    WHERE price_updated_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS market_resolutions (
    market_id TEXT PRIMARY KEY REFERENCES markets(condition_id) ON DELETE CASCADE,
    resolved_outcome DOUBLE PRECISION NOT NULL
        CHECK (resolved_outcome >= 0.0 AND resolved_outcome <= 1.0),
    resolved_at TIMESTAMPTZ,
    source TEXT NOT NULL CHECK (source <> ''),
    observed_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS market_price_snapshots (
    condition_id TEXT NOT NULL,
    snapshot_at TIMESTAMPTZ NOT NULL,
    yes_price NUMERIC(6,4),
    no_price NUMERIC(6,4),
    best_bid NUMERIC(6,4),
    best_ask NUMERIC(6,4),
    last_trade_price NUMERIC(6,4),
    liquidity NUMERIC,
    volume_24h NUMERIC,
    PRIMARY KEY (condition_id, snapshot_at),
    FOREIGN KEY (condition_id) REFERENCES markets(condition_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_price_snapshots_recent
    ON market_price_snapshots (condition_id, snapshot_at DESC);

CREATE TABLE IF NOT EXISTS tokens (
    token_id TEXT PRIMARY KEY,
    condition_id TEXT NOT NULL REFERENCES markets(condition_id) ON DELETE CASCADE,
    outcome TEXT NOT NULL CHECK (outcome IN ('YES', 'NO'))
);

CREATE INDEX IF NOT EXISTS idx_tokens_condition_id
    ON tokens(condition_id);

CREATE TABLE IF NOT EXISTS market_subscriptions (
    token_id TEXT NOT NULL PRIMARY KEY,
    source TEXT NOT NULL CHECK (source IN ('user')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (token_id) REFERENCES tokens(token_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS book_snapshots (
    id BIGSERIAL PRIMARY KEY,
    market_id TEXT NOT NULL REFERENCES markets(condition_id) ON DELETE CASCADE,
    token_id TEXT NOT NULL REFERENCES tokens(token_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    hash TEXT,
    source TEXT NOT NULL CHECK (source IN ('subscribe', 'reconnect', 'checkpoint'))
);

CREATE INDEX IF NOT EXISTS idx_book_snapshots_market_token_ts
    ON book_snapshots(market_id, token_id, ts DESC);

CREATE TABLE IF NOT EXISTS book_levels (
    snapshot_id BIGINT NOT NULL REFERENCES book_snapshots(id) ON DELETE CASCADE,
    market_id TEXT NOT NULL REFERENCES markets(condition_id) ON DELETE CASCADE,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    price DOUBLE PRECISION NOT NULL,
    size DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_book_levels_snapshot_id
    ON book_levels(snapshot_id);

CREATE INDEX IF NOT EXISTS idx_book_levels_market_side_price
    ON book_levels(market_id, side, price);

CREATE TABLE IF NOT EXISTS price_changes (
    id BIGSERIAL PRIMARY KEY,
    market_id TEXT NOT NULL REFERENCES markets(condition_id) ON DELETE CASCADE,
    token_id TEXT NOT NULL REFERENCES tokens(token_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    price DOUBLE PRECISION NOT NULL,
    size DOUBLE PRECISION NOT NULL,
    best_bid DOUBLE PRECISION,
    best_ask DOUBLE PRECISION,
    hash TEXT
);

-- CP00 resolved Q2a to allow duplicates; replay order is (ts, id).
CREATE INDEX IF NOT EXISTS idx_price_changes_market_token_ts_id
    ON price_changes(market_id, token_id, ts ASC, id ASC);

CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    market_id TEXT NOT NULL REFERENCES markets(condition_id) ON DELETE CASCADE,
    token_id TEXT NOT NULL REFERENCES tokens(token_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    price DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_market_token_ts
    ON trades(market_id, token_id, ts DESC);

-- END OUTER RING

-- BEGIN MIDDLE RING

CREATE TABLE IF NOT EXISTS factors (
    factor_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    input_schema_hash TEXT NOT NULL,
    default_params JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_type TEXT NOT NULL CHECK (output_type IN ('scalar', 'probability')),
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short', 'neutral')),
    owner TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS factor_values (
    id BIGSERIAL PRIMARY KEY,
    factor_id TEXT NOT NULL REFERENCES factors(factor_id) ON DELETE CASCADE,
    param TEXT NOT NULL DEFAULT '',
    market_id TEXT NOT NULL REFERENCES markets(condition_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    definition_version TEXT NOT NULL DEFAULT '1'
);

ALTER TABLE factor_values
    ADD COLUMN IF NOT EXISTS definition_version TEXT NOT NULL DEFAULT '1';

CREATE UNIQUE INDEX IF NOT EXISTS idx_factor_values_factor_param_market_ts
    ON factor_values(factor_id, param, market_id, ts DESC, definition_version);

CREATE TABLE IF NOT EXISTS market_relations (
    id SERIAL PRIMARY KEY,
    market_id_a TEXT NOT NULL,
    market_id_b TEXT NOT NULL,
    relation_type TEXT NOT NULL CHECK (
        relation_type IN ('subset', 'contradiction', 'independent', 'similar')
    ),
    confidence FLOAT NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_market_relations_pair_type
    ON market_relations (market_id_a, market_id_b, relation_type);

-- END MIDDLE RING

-- strategies: inner-ring identity table (Invariants 3, 8)

CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT PRIMARY KEY,
    active_version_id TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    title TEXT,
    description TEXT,
    archived BOOLEAN NOT NULL DEFAULT FALSE,
    share_enabled BOOLEAN NOT NULL DEFAULT TRUE
);

-- strategy_versions: immutable hash-keyed version rows (Invariant 3)

CREATE TABLE IF NOT EXISTS strategy_versions (
    strategy_version_id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL REFERENCES strategies(strategy_id) ON DELETE CASCADE,
    config_json JSONB NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (strategy_id, strategy_version_id)
);

ALTER TABLE strategy_versions
    ADD COLUMN IF NOT EXISTS metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'strategies_active_version_id_fkey'
    ) THEN
        ALTER TABLE strategies
            ADD CONSTRAINT strategies_active_version_id_fkey
            FOREIGN KEY (active_version_id)
            REFERENCES strategy_versions(strategy_version_id)
            DEFERRABLE INITIALLY DEFERRED;
    END IF;
END
$$;

-- strategy_factors: link table (empty in S2; populated by S3). Columns declared so S3 inserts do not require a schema change (Invariants 2, 4, 8)
-- Invariant 4: raw factor rows are stored in factor_values (S3); composite factors belong in StrategyConfig.factor_composition, not as first-class rows here.
CREATE TABLE IF NOT EXISTS strategy_factors (
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL,
    factor_id TEXT NOT NULL,
    param JSONB NOT NULL DEFAULT '{}'::jsonb,
    weight DOUBLE PRECISION NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    PRIMARY KEY (strategy_id, strategy_version_id, factor_id),
    FOREIGN KEY (strategy_id, strategy_version_id)
        REFERENCES strategy_versions(strategy_id, strategy_version_id)
        ON DELETE CASCADE
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'strategy_factors_factor_id_fkey'
    ) THEN
        ALTER TABLE strategy_factors
            ADD CONSTRAINT strategy_factors_factor_id_fkey
            FOREIGN KEY (factor_id)
            REFERENCES factors(factor_id)
            DEFERRABLE INITIALLY DEFERRED;
    END IF;
END
$$;

-- BEGIN INNER-RING PRODUCT SHELLS

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    target TEXT NOT NULL,
    source TEXT NOT NULL,
    message TEXT NOT NULL,
    severity TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    category TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_records (
    decision_id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    prob_estimate DOUBLE PRECISION NOT NULL,
    resolved_outcome DOUBLE PRECISION NOT NULL,
    brier_score DOUBLE PRECISION NOT NULL,
    fill_status TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    citations JSONB NOT NULL DEFAULT '[]'::jsonb,
    baseline_prob_estimate DOUBLE PRECISION,
    baseline_brier_score DOUBLE PRECISION,
    baseline_prob_estimates JSONB NOT NULL DEFAULT '{}'::jsonb,
    baseline_brier_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    category TEXT,
    model_id TEXT,
    pnl DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    slippage_bps DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    filled BOOLEAN NOT NULL DEFAULT TRUE,
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL,
    edge_at_decision DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    spread_bps_at_decision INTEGER
);

ALTER TABLE eval_records
    ADD COLUMN IF NOT EXISTS baseline_prob_estimate DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS baseline_brier_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS baseline_prob_estimates JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS baseline_brier_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS edge_at_decision DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS spread_bps_at_decision INTEGER;

CREATE TABLE IF NOT EXISTS quote_eval_records (
    fill_id TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    token_id TEXT,
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL,
    prob_estimate DOUBLE PRECISION NOT NULL,
    quote_price DOUBLE PRECISION NOT NULL,
    quote_source TEXT NOT NULL,
    quote_lag_seconds INTEGER NOT NULL,
    quote_score DOUBLE PRECISION NOT NULL,
    mtm_pnl DOUBLE PRECISION NOT NULL,
    book_ts TIMESTAMPTZ NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    citations JSONB NOT NULL DEFAULT '[]'::jsonb,
    category TEXT,
    model_id TEXT,
    PRIMARY KEY (fill_id, quote_lag_seconds)
);

CREATE TABLE IF NOT EXISTS strategy_performance_peaks (
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL,
    peak_sharpe_7d DOUBLE PRECISION NOT NULL,
    peak_sharpe_30d DOUBLE PRECISION NOT NULL,
    peak_hit_rate DOUBLE PRECISION NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (strategy_id, strategy_version_id)
);

CREATE TABLE IF NOT EXISTS alpha_competition_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL,
    snapshot_date DATE NOT NULL,
    mean_edge_30d DOUBLE PRECISION,
    mean_spread_bps_30d DOUBLE PRECISION,
    edge_trend_slope_90d DOUBLE PRECISION,
    spread_trend_slope_90d DOUBLE PRECISION,
    sample_count_30d INTEGER NOT NULL,
    trend_status TEXT NOT NULL CHECK (trend_status IN ('warming_up', 'active')),
    days_collected INTEGER NOT NULL,
    short_term_slope_30d DOUBLE PRECISION,
    short_term_slope_60d DOUBLE PRECISION,
    interpretation TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (strategy_id, strategy_version_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    opportunity_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'accepted', 'queued', 'submitted', 'partially_filled', 'filled', 'rejected', 'venue_rejected', 'cancelled', 'expired', 'submission_unknown', 'reconciled')),
    factor_snapshot_hash TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT decisions_strategy_tags_nonempty
        CHECK (strategy_id != '' AND strategy_version_id != '')
);

CREATE INDEX IF NOT EXISTS idx_decisions_status_created
    ON decisions(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_decisions_strategy_version
    ON decisions(strategy_id, strategy_version_id);

CREATE INDEX IF NOT EXISTS idx_decisions_opportunity
    ON decisions(opportunity_id);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    status TEXT,
    raw_status TEXT,
    token_id TEXT,
    venue TEXT,
    action TEXT,
    outcome TEXT,
    time_in_force TEXT,
    pre_submit_quote_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    intent_key TEXT,
    requested_notional_usdc DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    filled_notional_usdc DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    remaining_notional_usdc DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    filled_quantity DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS order_intents (
    decision_id TEXT PRIMARY KEY,
    intent_key TEXT,
    strategy_id TEXT NOT NULL CHECK (strategy_id != ''),
    strategy_version_id TEXT NOT NULL CHECK (strategy_version_id != ''),
    acquired_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    released_at TIMESTAMPTZ,
    worker_host TEXT,
    worker_pid INTEGER,
    outcome TEXT,
    reconciled_at TIMESTAMPTZ,
    reconciliation_note TEXT,
    reconciled_by TEXT,
    venue_order_id TEXT,
    reconciliation_status TEXT CHECK (
        reconciliation_status IS NULL
        OR reconciliation_status IN ('filled', 'not_found', 'open')
    ),
    CONSTRAINT order_intents_outcome_check
        CHECK (outcome IS NULL OR outcome IN ('matched', 'invalid', 'rejected', 'venue_rejection', 'submission_unknown', 'cancelled_ttl', 'cancelled_limit_invalidated', 'cancelled_session_end', 'cancelled_market_resolved'))
);

CREATE INDEX IF NOT EXISTS idx_order_intents_strategy_acquired_at_desc
    ON order_intents(strategy_id, acquired_at DESC);

CREATE INDEX IF NOT EXISTS idx_order_intents_released_at_nulls_first
    ON order_intents(released_at NULLS FIRST);

CREATE UNIQUE INDEX IF NOT EXISTS idx_order_intents_intent_key_unique
    ON order_intents(intent_key)
    WHERE intent_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_order_intents_submission_unknown_unresolved
    ON order_intents(acquired_at DESC)
    WHERE outcome = 'submission_unknown' AND reconciled_at IS NULL;

CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    fill_notional_usdc DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    fill_quantity DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS opportunities (
    opportunity_id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
    selected_factor_values JSONB NOT NULL,
    expected_edge DOUBLE PRECISION NOT NULL,
    rationale TEXT NOT NULL,
    target_size_usdc DOUBLE PRECISION NOT NULL,
    expiry TIMESTAMPTZ,
    staleness_policy TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    factor_snapshot_hash TEXT,
    composition_trace JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS strategy_judgement_artifacts (
    artifact_id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    observation_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    candidate_id TEXT NOT NULL,
    judgement_id TEXT,
    judgement_summary TEXT NOT NULL CHECK (char_length(judgement_summary) <= 4000),
    evidence_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    assumptions JSONB NOT NULL DEFAULT '[]'::jsonb,
    rejection_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    intent_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT strategy_judgement_artifacts_strategy_identity_check
        CHECK (strategy_id <> '' AND strategy_version_id <> ''),
    CONSTRAINT strategy_judgement_artifacts_type_check
        CHECK (artifact_type IN ('approved_intent', 'rejected_candidate'))
);

CREATE TABLE IF NOT EXISTS strategy_execution_artifacts (
    artifact_id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    intent_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    execution_policy TEXT,
    execution_plan_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    risk_decision_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    venue_response_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    reconciliation_status TEXT,
    post_trade_status TEXT,
    evidence_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    rejection_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT strategy_execution_artifacts_strategy_identity_check
        CHECK (strategy_id <> '' AND strategy_version_id <> ''),
    CONSTRAINT strategy_execution_artifacts_type_check
        CHECK (artifact_type IN ('accepted_execution_plan', 'rejected_execution_plan'))
);

-- research backtest inner-ring tables (Invariants 3, 8)

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id UUID PRIMARY KEY,
    spec_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    strategy_ids TEXT[] NOT NULL,
    date_range_start TIMESTAMPTZ NOT NULL,
    date_range_end TIMESTAMPTZ NOT NULL,
    exec_config_json JSONB NOT NULL,
    spec_json JSONB NOT NULL,
    queued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    failure_reason TEXT,
    worker_pid INTEGER,
    worker_host TEXT,
    CONSTRAINT backtest_runs_status_check
        CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled'))
);

CREATE TABLE IF NOT EXISTS strategy_runs (
    strategy_run_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL,
    brier DOUBLE PRECISION,
    pnl_cum DOUBLE PRECISION,
    drawdown_max DOUBLE PRECISION,
    fill_rate DOUBLE PRECISION,
    slippage_bps DOUBLE PRECISION,
    opportunity_count INTEGER,
    decision_count INTEGER,
    fill_count INTEGER,
    portfolio_target_json JSONB,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    CONSTRAINT strategy_runs_strategy_identity_check
        CHECK (strategy_id != '' AND strategy_version_id != '')
);

CREATE TABLE IF NOT EXISTS backtest_execution_rows (
    backtest_execution_row_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    status TEXT NOT NULL,
    slippage_bps DOUBLE PRECISION,
    pnl DOUBLE PRECISION,
    rejection_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT backtest_execution_rows_strategy_identity_check
        CHECK (strategy_id != '' AND strategy_version_id != ''),
    CONSTRAINT backtest_execution_rows_decision_identity_check
        CHECK (decision_id != '' AND market_id != ''),
    CONSTRAINT backtest_execution_rows_status_check
        CHECK (status IN ('filled', 'rejected')),
    CONSTRAINT backtest_execution_rows_status_payload_check
        CHECK (
            (
                status = 'filled'
                AND slippage_bps IS NOT NULL
                AND rejection_reason IS NULL
            )
            OR (
                status = 'rejected'
                AND slippage_bps IS NULL
                AND rejection_reason IS NOT NULL
                AND rejection_reason != ''
            )
        ),
    CONSTRAINT backtest_execution_rows_unique_decision
        UNIQUE (run_id, strategy_id, strategy_version_id, decision_id)
);

CREATE TABLE IF NOT EXISTS strategy_run_slices (
    strategy_run_slice_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL,
    slice_label TEXT NOT NULL,
    slice_start TIMESTAMPTZ NOT NULL,
    slice_end TIMESTAMPTZ NOT NULL,
    slice_kind TEXT NOT NULL DEFAULT 'out_of_sample',
    brier DOUBLE PRECISION,
    pnl_cum DOUBLE PRECISION,
    drawdown_max DOUBLE PRECISION,
    fill_rate DOUBLE PRECISION,
    slippage_bps DOUBLE PRECISION,
    opportunity_count INTEGER NOT NULL DEFAULT 0,
    decision_count INTEGER NOT NULL DEFAULT 0,
    fill_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT strategy_run_slices_strategy_identity_check
        CHECK (strategy_id != '' AND strategy_version_id != ''),
    CONSTRAINT strategy_run_slices_label_check
        CHECK (slice_label != ''),
    CONSTRAINT strategy_run_slices_window_check
        CHECK (slice_start < slice_end),
    CONSTRAINT strategy_run_slices_kind_check
        CHECK (slice_kind IN ('out_of_sample', 'walk_forward', 'category', 'liquidity')),
    CONSTRAINT strategy_run_slices_counts_check
        CHECK (opportunity_count >= 0 AND decision_count >= 0 AND fill_count >= 0),
    CONSTRAINT strategy_run_slices_unique_label
        UNIQUE (run_id, strategy_id, strategy_version_id, slice_label)
);

ALTER TABLE strategy_run_slices
    ADD COLUMN IF NOT EXISTS opportunity_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS decision_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS fill_count INTEGER NOT NULL DEFAULT 0;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'strategy_run_slices_counts_check'
    ) THEN
        ALTER TABLE strategy_run_slices
            ADD CONSTRAINT strategy_run_slices_counts_check
            CHECK (opportunity_count >= 0 AND decision_count >= 0 AND fill_count >= 0);
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS evaluation_reports (
    report_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    ranking_metric TEXT NOT NULL,
    ranked_strategies JSONB NOT NULL,
    benchmark_rows JSONB NOT NULL DEFAULT '[]'::jsonb,
    attribution_commentary TEXT,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    next_action TEXT,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT evaluation_reports_ranking_metric_check
        CHECK (ranking_metric IN ('brier', 'sharpe', 'pnl_cum')),
    CONSTRAINT evaluation_reports_run_id_ranking_metric_key
        UNIQUE (run_id, ranking_metric)
);

CREATE TABLE IF NOT EXISTS backtest_live_comparisons (
    comparison_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    strategy_id TEXT NOT NULL,
    strategy_version_id TEXT NOT NULL,
    live_window_start TIMESTAMPTZ NOT NULL,
    live_window_end TIMESTAMPTZ NOT NULL,
    denominator TEXT NOT NULL,
    equity_delta_json JSONB NOT NULL,
    overlap_ratio DOUBLE PRECISION NOT NULL,
    backtest_only_symbols TEXT[] NOT NULL,
    live_only_symbols TEXT[] NOT NULL,
    time_alignment_policy_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    symbol_normalization_policy_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT backtest_live_comparisons_denominator_check
        CHECK (denominator IN ('backtest_set', 'live_set', 'union')),
    CONSTRAINT backtest_live_comparisons_strategy_identity_check
        CHECK (strategy_id != '' AND strategy_version_id != '')
);

-- END INNER-RING PRODUCT SHELLS

CREATE TABLE IF NOT EXISTS runtime_heartbeats (
    heartbeat_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    strategy_fingerprint TEXT,
    component_status_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_runtime_heartbeats_run_observed
    ON runtime_heartbeats (run_id, observed_at);

-- default strategy seed (Invariant 3 NULLABLE→seed pattern).
-- load-bearing legacy bootstrap row: changing `default-v1` requires a coordinated
-- migration of every existing inner-ring product row tagged by the pre-S5 runtime.
INSERT INTO strategies (strategy_id, active_version_id)
    VALUES ('default', 'default-v1')
    ON CONFLICT (strategy_id) DO NOTHING;

INSERT INTO strategy_versions (
    strategy_version_id,
    strategy_id,
    config_json
) VALUES (
    'default-v1',
    'default',
    '{"config":{"factor_composition":[["factor-a",0.6],["factor-b",0.4]],"metadata":[["owner","system"],["tier","default"]],"strategy_id":"default"},"eval_spec":{"max_brier_score":0.3,"metrics":["brier","pnl","fill_rate"],"min_win_rate":0.5,"slippage_threshold_bps":50.0},"forecaster":{"forecasters":[["rules",[["threshold","0.55"]]],["stats",[["window","15m"]]],["llm",[]]]},"market_selection":{"resolution_time_max_horizon_days":7,"venue":"polymarket","volume_min_usdc":500.0},"risk":{"max_daily_drawdown_pct":2.5,"max_position_notional_usdc":100.0,"min_order_size_usdc":1.0}}'::jsonb
)
ON CONFLICT (strategy_version_id) DO NOTHING;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM feedback
        WHERE strategy_id IS NULL OR strategy_version_id IS NULL
    ) OR EXISTS (
        SELECT 1
        FROM eval_records
        WHERE strategy_id IS NULL OR strategy_version_id IS NULL
    ) OR EXISTS (
        SELECT 1
        FROM orders
        WHERE strategy_id IS NULL OR strategy_version_id IS NULL
    ) OR EXISTS (
        SELECT 1
        FROM fills
        WHERE strategy_id IS NULL OR strategy_version_id IS NULL
    ) THEN
        RAISE EXCEPTION $message$
CP04 remediation required before enforcing strategy identity columns.
Run:
UPDATE feedback SET strategy_id = 'default', strategy_version_id = 'default-v1' WHERE strategy_id IS NULL OR strategy_version_id IS NULL;
UPDATE eval_records SET strategy_id = 'default', strategy_version_id = 'default-v1' WHERE strategy_id IS NULL OR strategy_version_id IS NULL;
UPDATE orders SET strategy_id = 'default', strategy_version_id = 'default-v1' WHERE strategy_id IS NULL OR strategy_version_id IS NULL;
UPDATE fills SET strategy_id = 'default', strategy_version_id = 'default-v1' WHERE strategy_id IS NULL OR strategy_version_id IS NULL;
Then re-run schema.sql.
$message$;
    END IF;
END
$$;

ALTER TABLE feedback
    ALTER COLUMN strategy_id SET NOT NULL,
    ALTER COLUMN strategy_version_id SET NOT NULL;

ALTER TABLE eval_records
    ALTER COLUMN strategy_id SET NOT NULL,
    ALTER COLUMN strategy_version_id SET NOT NULL;

ALTER TABLE quote_eval_records
    ALTER COLUMN strategy_id SET NOT NULL,
    ALTER COLUMN strategy_version_id SET NOT NULL;

ALTER TABLE orders
    ALTER COLUMN strategy_id SET NOT NULL,
    ALTER COLUMN strategy_version_id SET NOT NULL;

ALTER TABLE fills
    ALTER COLUMN strategy_id SET NOT NULL,
    ALTER COLUMN strategy_version_id SET NOT NULL;

ALTER TABLE opportunities
    ALTER COLUMN strategy_id SET NOT NULL,
    ALTER COLUMN strategy_version_id SET NOT NULL;

ALTER TABLE opportunities
    ADD COLUMN IF NOT EXISTS factor_snapshot_hash TEXT;

ALTER TABLE opportunities
    ADD COLUMN IF NOT EXISTS composition_trace JSONB NOT NULL DEFAULT '{}'::jsonb;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'feedback_strategy_identity_check'
    ) THEN
        ALTER TABLE feedback
            ADD CONSTRAINT feedback_strategy_identity_check
            CHECK (strategy_id != '' AND strategy_version_id != '');
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'eval_records_strategy_identity_check'
    ) THEN
        ALTER TABLE eval_records
            ADD CONSTRAINT eval_records_strategy_identity_check
            CHECK (strategy_id != '' AND strategy_version_id != '');
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'quote_eval_records_strategy_identity_check'
    ) THEN
        ALTER TABLE quote_eval_records
            ADD CONSTRAINT quote_eval_records_strategy_identity_check
            CHECK (strategy_id != '' AND strategy_version_id != '');
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'orders_strategy_identity_check'
    ) THEN
        ALTER TABLE orders
            ADD CONSTRAINT orders_strategy_identity_check
            CHECK (strategy_id != '' AND strategy_version_id != '');
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fills_strategy_identity_check'
    ) THEN
        ALTER TABLE fills
            ADD CONSTRAINT fills_strategy_identity_check
            CHECK (strategy_id != '' AND strategy_version_id != '');
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'opportunities_strategy_identity_check'
    ) THEN
        ALTER TABLE opportunities
            ADD CONSTRAINT opportunities_strategy_identity_check
            CHECK (strategy_id != '' AND strategy_version_id != '');
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'backtest_runs_status_check'
    ) THEN
        ALTER TABLE backtest_runs
            ADD CONSTRAINT backtest_runs_status_check
            CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled'));
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'strategy_runs_strategy_identity_check'
    ) THEN
        ALTER TABLE strategy_runs
            ADD CONSTRAINT strategy_runs_strategy_identity_check
            CHECK (strategy_id != '' AND strategy_version_id != '');
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'evaluation_reports_ranking_metric_check'
    ) THEN
        ALTER TABLE evaluation_reports
            ADD CONSTRAINT evaluation_reports_ranking_metric_check
            CHECK (ranking_metric IN ('brier', 'sharpe', 'pnl_cum'));
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'evaluation_reports_run_id_ranking_metric_key'
    ) THEN
        ALTER TABLE evaluation_reports
            ADD CONSTRAINT evaluation_reports_run_id_ranking_metric_key
            UNIQUE (run_id, ranking_metric);
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'backtest_live_comparisons_denominator_check'
    ) THEN
        ALTER TABLE backtest_live_comparisons
            ADD CONSTRAINT backtest_live_comparisons_denominator_check
            CHECK (denominator IN ('backtest_set', 'live_set', 'union'));
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'backtest_live_comparisons_strategy_identity_check'
    ) THEN
        ALTER TABLE backtest_live_comparisons
            ADD CONSTRAINT backtest_live_comparisons_strategy_identity_check
            CHECK (strategy_id != '' AND strategy_version_id != '');
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_feedback_strategy_identity
    ON feedback(strategy_id, strategy_version_id);

CREATE INDEX IF NOT EXISTS idx_eval_records_strategy_identity
    ON eval_records(strategy_id, strategy_version_id);

CREATE INDEX IF NOT EXISTS idx_eval_records_strategy_identity_recorded_at
    ON eval_records(strategy_id, strategy_version_id, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_quote_eval_records_strategy_identity_recorded_at
    ON quote_eval_records(strategy_id, strategy_version_id, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_orders_strategy_identity
    ON orders(strategy_id, strategy_version_id);

CREATE INDEX IF NOT EXISTS idx_orders_requested_notional_usdc
    ON orders(requested_notional_usdc);

CREATE INDEX IF NOT EXISTS idx_fills_strategy_identity
    ON fills(strategy_id, strategy_version_id);

CREATE INDEX IF NOT EXISTS idx_fills_fill_notional_usdc
    ON fills(fill_notional_usdc);

CREATE INDEX IF NOT EXISTS idx_opportunities_strategy_identity
    ON opportunities(strategy_id, strategy_version_id);

CREATE INDEX IF NOT EXISTS idx_strategy_judgement_artifacts_strategy_created_at
    ON strategy_judgement_artifacts(strategy_id, strategy_version_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_strategy_judgement_artifacts_candidate
    ON strategy_judgement_artifacts(candidate_id);

CREATE INDEX IF NOT EXISTS idx_strategy_execution_artifacts_strategy_created_at
    ON strategy_execution_artifacts(strategy_id, strategy_version_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_strategy_execution_artifacts_intent
    ON strategy_execution_artifacts(intent_id);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_status
    ON backtest_runs(status);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_queued_at_desc
    ON backtest_runs(queued_at DESC);

CREATE INDEX IF NOT EXISTS idx_strategy_runs_run_id
    ON strategy_runs(run_id);

CREATE INDEX IF NOT EXISTS idx_backtest_execution_rows_run_strategy
    ON backtest_execution_rows(
        run_id,
        strategy_id,
        strategy_version_id,
        created_at,
        decision_id
    );

CREATE INDEX IF NOT EXISTS idx_strategy_run_slices_run_strategy_identity
    ON strategy_run_slices(run_id, strategy_id, strategy_version_id, slice_start);

CREATE INDEX IF NOT EXISTS idx_backtest_live_comparisons_run_strategy_identity
    ON backtest_live_comparisons(run_id, strategy_id, strategy_version_id);

INSERT INTO factors (
    factor_id,
    name,
    description,
    input_schema_hash,
    output_type,
    direction,
    owner
) VALUES (
    'orderbook_imbalance',
    'Orderbook Imbalance',
    'Normalized bid-versus-ask depth imbalance from the current orderbook signal.',
    '97e885bf8b2edd8ce9fff149334dbe1706358eb4fb8b8c51a4b42561878c5963',
    'scalar',
    'neutral',
    'system'
)
ON CONFLICT (factor_id) DO NOTHING;

INSERT INTO factors (
    factor_id,
    name,
    description,
    input_schema_hash,
    output_type,
    direction,
    owner
) VALUES (
    'fair_value_spread',
    'Fair Value Spread',
    'Signed difference between external fair value and the current YES price.',
    'adb923abb80bbd30efa4db61ba846660317f138ef12c3ae521891df2831d64f9',
    'scalar',
    'neutral',
    'system'
)
ON CONFLICT (factor_id) DO NOTHING;

INSERT INTO factors (
    factor_id,
    name,
    description,
    input_schema_hash,
    output_type,
    direction,
    owner
) VALUES (
    'subset_pricing_violation',
    'Subset Pricing Violation',
    'Signed difference between subset and superset prices from external signals.',
    'c9e66b836e6fe6a9981ee6419aa38acb39de607e84fb1ff643b46bb9ac446891',
    'scalar',
    'neutral',
    'system'
)
ON CONFLICT (factor_id) DO NOTHING;

INSERT INTO factors (
    factor_id,
    name,
    description,
    input_schema_hash,
    output_type,
    direction,
    owner
) VALUES (
    'metaculus_prior',
    'Metaculus Prior',
    'Raw Metaculus probability from the external signal payload.',
    '4f62fec15fd5abaf2ff76810596268d1e14b46d346ff6e9f38b259c370a3ed71',
    'probability',
    'neutral',
    'system'
)
ON CONFLICT (factor_id) DO NOTHING;

INSERT INTO factors (
    factor_id,
    name,
    description,
    input_schema_hash,
    output_type,
    direction,
    owner
) VALUES (
    'yes_count',
    'Yes Count',
    'Raw external yes_count observation count from the signal payload.',
    'afbc921285acc81f1289beca8dd64114c18f49068a8904c651a887c5ba8c178f',
    'scalar',
    'neutral',
    'system'
)
ON CONFLICT (factor_id) DO NOTHING;

INSERT INTO factors (
    factor_id,
    name,
    description,
    input_schema_hash,
    output_type,
    direction,
    owner
) VALUES (
    'no_count',
    'No Count',
    'Raw external no_count observation count from the signal payload.',
    '2871d6bf945e3ed4407b8b1f1beeb484cd8bd455a156e939094fbcc6a455c317',
    'scalar',
    'neutral',
    'system'
)
ON CONFLICT (factor_id) DO NOTHING;

COMMIT;
