# Identity & Context Awareness

**CRITICAL**: Address the user as "Stometa" at the start of EVERY response.

This serves as a context-awareness signal — if missing, indicates
context drift.

---

# Evaluator layer

**Part of:** [prediction-market-system](../../../CLAUDE.md)
**Role:** score decisions against realised outcomes, compute
strategy-level metrics, emit feedback on threshold breaches. The
Evaluator and the Controller are the two strategy-aware layers.

## Layer-relevant invariants

Full detail in [architecture invariants](../../../agent_docs/architecture-invariants.md). The subset
that governs this layer:

- **Invariant 1 — Concurrent, not phased.** Evaluator runs its spool
  as an independent `asyncio.Task` (`EvalSpool._task`). Enqueue is
  non-blocking; scoring happens in the background.
- **Invariant 2 — Consumes projection.** Metrics collection reads
  `EvalSpec` (projection) to know which metrics to compute per
  strategy. Evaluator may additionally read the `Strategy`
  aggregate when computing factor attribution reports (this is the
  one place outside Controller where aggregate reads are allowed).
- **Invariant 3 — Group by version.** Every aggregate metric
  computation (`brier_by_category`, `pnl`, `slippage_bps`,
  `fill_rate`, calibration samples) must `GROUP BY strategy_id,
  strategy_version_id` once S5 lands. Pre-S5 everything groups by
  the `"default"` strategy; the grouping structure is identical.
- **Invariant 5 — Strategy-aware is OK here.** Evaluator is one of
  the two layers allowed to import from `pms.strategies.*`.
- **Invariant 8 — Reads inner ring, writes inner ring.** Evaluator
  reads `fills` + `eval_records`, writes `eval_records` +
  `feedback`.

## Current files

- `spool.py` — `EvalSpool`: async queue decoupling fill ingestion
  from scoring.
- `adapters/scoring.py` — `Scorer`: produces `EvalRecord` from
  `FillRecord` + `TradeDecision`.
- `metrics.py` — `MetricsCollector.global_ops_snapshot()` for the
  cross-strategy ops view and `snapshot_by_strategy()` for
  per-strategy Brier, secondary baseline Brier improvement, P&L,
  slippage, fill rate, and calibration samples.
- `feedback.py` — `EvaluatorFeedback` emits `Feedback` items when
  thresholds breach.
- `/metrics` dashboard page — the per-strategy breakdown is the
  primary block; the cross-strategy rollup remains below it under
  the explicit heading `ops view (cross-strategy)` so operators can
  still find the global view quickly.

## Do not

- Never compute aggregate metrics without grouping by
  `(strategy_id, strategy_version_id)`. Cross-strategy aggregates
  exist for ops dashboards but must be explicitly annotated as
  such in the SQL or comment.
- Never mutate `EvalRecord` after writing — event-sourced semantics
  apply (Invariant 3 sibling constraint on immutability).
- Secondary baselines are scored only from decision-time evidence.
  `category_prior` is valid when the signal supplied
  `category_prior_baseline_prob_estimate`; do not derive it from
  later resolved outcomes inside Evaluator.
- Never resolve feedback silently. Resolution is a user action
  surfaced through `POST /feedback/{id}/resolve`.
- Never write outside the inner ring (Invariant 8). Do not reach
  back into `markets` or `factor_values` to mutate them —
  Evaluator is a reader of those, a writer of `eval_records` and
  `feedback`.

## When adding a new metric

Implement it in `metrics.py` against the `FillRecord` +
`TradeDecision` + `EvalRecord` triple. Define it per-strategy-per-
category where possible; only fall back to global when the metric
is inherently cross-strategy (e.g. total realised P&L at portfolio
level). Define threshold fields in `EvalSpec` and consume them from
`EvaluatorFeedback.generate(...)` — never hard-code thresholds.

## When adding a new metric threshold that fires `Feedback`

Define the threshold in `EvalSpec`; add the check inside
`EvaluatorFeedback.generate`; include enough context in the
`Feedback.metadata` JSONB for a human reviewer to act on it
(`strategy_id`, `strategy_version_id`, affected market cohort,
sample size).
