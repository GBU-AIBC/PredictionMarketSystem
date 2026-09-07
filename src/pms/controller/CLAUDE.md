# Identity & Context Awareness

**CRITICAL**: Address the user as "Stometa" at the start of EVERY response.

This serves as a context-awareness signal — if missing, indicates
context drift.

---

# Controller layer

**Part of:** [prediction-market-system](../../../CLAUDE.md)
**Role:** consume `MarketSignal` + factor values, run per-strategy
forecasting / calibration / sizing, emit `TradeDecision`. The
Controller and the Evaluator are the two strategy-aware layers.

## Layer-relevant invariants

Full detail in [architecture invariants](../../../agent_docs/architecture-invariants.md). The subset
that governs this layer:

- **Invariant 1 — Concurrent, not phased.** Controller runs as an
  independent `asyncio.Task` (`runner.py:_controller_loop`). The
  only synchronous barrier is the intentional `asyncio.gather` for
  multi-forecaster fan-in inside `ControllerPipeline.on_signal`
  (documented exception; do not extend this pattern elsewhere).
- **Invariant 2 — Aggregate reader.** Controller is the only layer
  (with Evaluator) that reads the `Strategy` aggregate. It reads
  `StrategyConfig`, `ForecasterSpec`, `MarketSelectionSpec` as
  projections. Do not leak the aggregate to downstream layers;
  downstream sees only the projection it was handed.
- **Invariant 3 — Tag every decision.** Every `TradeDecision`
  emitted by Controller carries `(strategy_id,
  strategy_version_id)`. Post-S5, these fields are NOT NULL on the
  DDL; pre-S5, they are tagged with `"default"`.
- **Invariant 5 — Strategy-aware is OK here.** Controller may
  import from `pms.strategies.aggregate`, `pms.factors.*`, and
  `pms.market_selection.*`. This is the designed responsibility
  boundary.
- **Invariant 6 — MarketSelector home.** The market selection
  logic is implemented in the sibling `pms.market_selection` module.
  It reads `MarketSelectionSpec` projections from the strategy
  registry, filters the discovered universe, and feeds the result
  into `SensorSubscriptionController`.
- **Invariant 8 — Reads middle, writes inner.** Controller reads
  `factor_values` (middle ring); writes `TradeDecision` records to
  the inner ring (via Actuator for order products, via Evaluator
  for eval products).

## Current files

- `pipeline.py` — composes forecaster → calibrator → sizer → router.
  Currently single-global pipeline. **S5 makes this per-strategy.**
- `router.py` — gating (volume / liquidity thresholds), venue
  selection, stop-condition assembly. `stop_conditions` on
  `TradeDecision` is the gate-threshold list only
  (`min_volume:...`, `near_resolution_price_band:...`,
  `resolves_at:...`). Strategy / model provenance must live in
  explicit typed fields, not in `stop_conditions`.
- `forecasters/rules.py`, `forecasters/statistical.py`,
  `forecasters/llm.py` — the three forecasters.
- `baselines.py` — decision-time baseline helpers, including the
  strict historical resolution CSV loader and category-prior
  estimator used to enrich `MarketSignal.external_signal` before a
  decision is persisted.
- `calibrators/netcal.py` — isotonic calibration per model_id.
- `sizers/kelly.py` — fractional Kelly with fee-aware formula.

## Do not

- Never persist `Strategy` aggregate references in `TradeDecision`
  or any downstream entity. Emit the `strategy_id` /
  `strategy_version_id` fields (Invariant 3) and let downstream
  layers re-read if they need the full aggregate (only Evaluator
  does).
- Never extend `stop_conditions: list[str]` with new strategy
  semantics. New strategy provenance goes in dedicated typed
  fields on `TradeDecision`.
- Never write outside the inner ring (Invariant 8).
- Never compute per-market subscription lists inside sensor code.
  The flow is: `MarketSelectionSpec` projection →
  `MarketSelector` → `SensorSubscriptionController` → Sensor.
  Controller owns the strategy side of that handoff; Sensor remains
  strategy-agnostic.

## When adding a new forecaster / calibrator / sizer

Drop the implementation into the matching subdirectory. The
`ControllerPipeline` picks it up via Protocol interfaces in
`pms.core.interfaces` — no orchestrator change needed. Mandatory:
implement the relevant Protocol exactly; satisfy the forecaster /
calibrator / sizer acceptance criteria from the relevant spec.
