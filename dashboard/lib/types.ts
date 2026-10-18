export type StatusResponse = {
  mode: string;
  source: 'live' | 'mock';
  runner_started_at: string | null;
  running?: boolean;
  sensors: Array<{ name: string; status: string; last_signal_at: string | null }>;
  controller: {
    decisions_total: number;
    diagnostics_total?: number;
    diagnostic_counts?: Record<string, number>;
  };
  actuator: {
    fills_total: number;
    mode: string;
    halt_recovery_cycles_7d?: number;
  };
  evaluator: {
    eval_records_total: number;
    brier_overall: number | null;
    baseline_brier_overall?: number | null;
    brier_improvement_overall?: number | null;
    brier_14d?: number | null;
    baseline_brier_14d?: number | null;
    brier_improvement_14d?: number | null;
  };
  supervision?: { unresolved_feedback_total: number };
};

export type Feedback = {
  feedback_id: string;
  target: string;
  source: string;
  message: string;
  severity: string;
  created_at: string;
  resolved: boolean;
  resolved_at: string | null;
  category: string | null;
  metadata: Record<string, unknown>;
};

export type Decision = {
  decision_id: string;
  market_id: string;
  token_id?: string | null;
  venue?: string;
  forecaster: string;
  prob_estimate: number;
  expected_edge: number;
  kelly_size: number;
  notional_usdc?: number;
  resolved_outcome?: number | null;
  price?: number;
  limit_price?: number;
  side?: string;
  action?: string | null;
  status?: 'pending' | 'accepted' | 'rejected' | 'expired' | string;
  factor_snapshot_hash?: string | null;
  created_at?: string;
  expires_at?: string;
  opportunity?: DecisionOpportunity | null;
};

export type DecisionOpportunity = {
  opportunity_id: string;
  market_id: string;
  token_id: string;
  side: string;
  selected_factor_values: Record<string, number>;
  expected_edge: number;
  rationale: string;
  target_size_usdc: number;
  expiry: string | null;
  staleness_policy: string;
  strategy_id: string;
  strategy_version_id: string;
  created_at: string;
  factor_snapshot_hash: string | null;
  composition_trace: Record<string, unknown>;
};

export type EventLogEntry = {
  event_id: number;
  event_type: string;
  created_at: string;
  summary: string;
  market_id?: string | null;
  decision_id?: string | null;
  fill_id?: string | null;
};

export type MetricsAggregate = {
  brier_overall: number | null;
  baseline_brier_overall?: number | null;
  brier_improvement_overall?: number | null;
  brier_by_category: Record<string, number>;
  pnl: number;
  slippage_bps: number;
  fill_rate: number;
  win_rate: number;
  brier_series?: Array<{ recorded_at: string; brier_score: number }>;
  calibration_curve?: Array<{ prob_estimate: number; resolved_outcome: number }>;
  pnl_series?: Array<{ recorded_at: string; pnl: number }>;
};

export type MetricsPerStrategyRow = {
  strategy_id: string;
  strategy_version_id: string;
  record_count: number;
  insufficient_samples: boolean;
  brier_overall: number | null;
  baseline_brier_overall?: number | null;
  brier_improvement_overall?: number | null;
  pnl: number;
  fill_rate: number;
  slippage_bps: number;
  drawdown: number;
};

export type MetricsResponse = MetricsAggregate & {
  'pms.ui.first_trade_time_seconds'?: number | null;
  per_strategy: MetricsPerStrategyRow[];
  ops_view: MetricsAggregate;
};

export type Signal = {
  market_id: string;
  title: string;
  yes_price: number;
  fetched_at: string;
};

export type DepthLevel = {
  price: number;
  size: number;
};

export type SignalDepth = {
  best_bid: number | null;
  best_ask: number | null;
  bids: DepthLevel[];
  asks: DepthLevel[];
  last_update_ts: string | null;
  stale: boolean;
};

export type MarketRow = {
  market_id: string;
  question: string;
  venue: string;
  volume_24h: number | null;
  updated_at: string;
  yes_token_id: string | null;
  no_token_id: string | null;
  yes_price: number | null;
  no_price: number | null;
  best_bid: number | null;
  best_ask: number | null;
  last_trade_price: number | null;
  liquidity: number | null;
  spread_bps: number | null;
  price_updated_at: string | null;
  resolves_at?: string | null;
  subscription_source: 'user' | null;
  subscribed: boolean;
};

export type MarketsListResponse = {
  markets: MarketRow[];
  limit: number;
  offset: number;
  total: number;
};

export type PriceHistorySnapshot = {
  snapshot_at: string;
  yes_price: number | null;
  no_price: number | null;
  best_bid: number | null;
  best_ask: number | null;
  last_trade_price: number | null;
  liquidity: number | null;
  volume_24h: number | null;
};

export type PriceHistoryResponse = {
  condition_id: string;
  snapshots: PriceHistorySnapshot[];
};

export type PositionRow = {
  market_id: string;
  token_id: string | null;
  venue: string;
  side: string;
  shares_held: number;
  avg_entry_price: number;
  unrealized_pnl: number;
  locked_usdc: number;
  strategy_id: string;
  strategy_version_id: string;
};

export type PositionsResponse = {
  positions: PositionRow[];
};

export type TradeRow = {
  trade_id: string;
  fill_id: string;
  order_id: string;
  decision_id: string;
  market_id: string;
  question: string;
  token_id: string | null;
  venue: string;
  side: string;
  fill_price: number;
  fill_notional_usdc: number;
  fill_quantity: number;
  executed_at: string;
  filled_at: string;
  status: string;
  strategy_id: string;
  strategy_version_id: string;
  fee_bps?: number | null;
  fees?: number | null;
};

export type TradesResponse = {
  trades: TradeRow[];
  limit: number;
};

export type StrategyRow = {
  strategy_id: string;
  active_version_id: string | null;
  created_at: string;
};

export type StrategiesResponse = {
  strategies: StrategyRow[];
};

export type StrategyMetricsRow = {
  strategy_id: string;
  strategy_version_id: string;
  created_at: string;
  record_count: number;
  insufficient_samples: boolean;
  brier_overall: number | null;
  pnl: number;
  fill_rate: number;
  slippage_bps: number;
  drawdown: number;
};

export type StrategyMetricsResponse = {
  strategies: StrategyMetricsRow[];
};

export type ShareProjection = {
  strategy_id: string;
  title: string | null;
  description: string | null;
  brier_overall: number | null;
  trade_count: number;
  version_id_short: string | null;
};

export type FactorCatalogEntry = {
  factor_id: string;
  name: string;
  description: string;
  output_type: string;
  direction: string;
};

export type FactorCatalogResponse = {
  catalog: FactorCatalogEntry[];
};

export type FactorPoint = {
  ts: string;
  value: number;
};

export type FactorSeriesResponse = {
  factor_id: string;
  param: string;
  market_id: string;
  points: FactorPoint[];
};

export type BacktestRankingMetric = 'brier' | 'sharpe' | 'pnl_cum';

export type SelectionDenominator = 'backtest_set' | 'live_set' | 'union';

export type BacktestRunRow = {
  run_id: string;
  spec_hash: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  strategy_ids: string[];
  date_range_start: string;
  date_range_end: string;
  exec_config_json: {
    chunk_days?: number;
    time_budget?: number;
  };
  spec_json: {
    strategy_versions?: Array<[string, string]>;
    dataset?: {
      source?: string;
      version?: string;
      coverage_start?: string;
      coverage_end?: string;
    };
  };
  queued_at: string;
  started_at: string | null;
  finished_at: string | null;
  failure_reason: string | null;
  worker_pid: number | null;
  worker_host: string | null;
};

export type BacktestStrategyRunRow = {
  controller_rejection_counts?: Record<string, number>;
  strategy_run_id: string;
  run_id: string;
  strategy_id: string;
  strategy_version_id: string;
  brier: number | null;
  pnl_cum: number | null;
  drawdown_max: number | null;
  fill_rate: number | null;
  slippage_bps: number | null;
  opportunity_count: number | null;
  decision_count: number | null;
  fill_count: number | null;
  portfolio_target_json:
    | Array<{
        market_id: string;
        token_id: string;
        side: string;
        timestamp: string;
        target_size_usdc: number;
      }>
    | null;
  started_at: string;
  finished_at: string | null;
};

export type BacktestEnqueueResponse = {
  run_ids: string[];
  unique_run_count: number;
  runs: Array<{
    run_id: string;
    spec_hash: string;
    inserted: boolean;
  }>;
};

export type BacktestLiveComparisonResponse = {
  comparison_id: string;
  run_id: string;
  strategy_id: string;
  strategy_version_id: string;
  live_window_start: string;
  live_window_end: string;
  denominator: SelectionDenominator;
  equity_delta_json: Array<{
    day: string;
    backtest_equity: number;
    live_equity: number;
    delta: number;
  }>;
  overlap_ratio: number;
  backtest_only_symbols: string[];
  live_only_symbols: string[];
  time_alignment_policy_json: Record<string, number>;
  symbol_normalization_policy_json: {
    token_id_aliases?: Record<string, string>;
    market_id_aliases?: Record<string, string>;
  };
  computed_at: string;
};
