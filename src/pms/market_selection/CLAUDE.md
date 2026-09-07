# Identity & Context Awareness

**CRITICAL**: Address the user as "Stometa" at the start of EVERY response.

This serves as a context-awareness signal — if missing, indicates
context drift.

---

# Market selection layer

**Part of:** [prediction-market-system](../../../CLAUDE.md)
**Role:** derive active-perception market subscriptions from strategy
projections, merge strategy-level market sets, and push subscription
updates into the Sensor boundary.

## Layer-relevant invariants

Full detail in [architecture invariants](../../../agent_docs/architecture-invariants.md). The subset
that governs this layer:

- **Invariant 2 — Projection-only strategy reads.** This layer reads
  `MarketSelectionSpec` projections, not the `Strategy` aggregate.
- **Invariant 3 — Tag every selection.** Any market-selection result
  that is persisted or logged with strategy identity must carry both
  `strategy_id` and `strategy_version_id`.
- **Invariant 5 — Strategy-aware orchestration lives here, not in
  Sensor.** `MarketSelector` is the strategy-aware boundary for
  active perception; Sensor modules remain strategy-agnostic.
- **Invariant 6 — Active perception loop.** `MarketSelector` and
  `SensorSubscriptionController` are the landed implementation of the
  feedback edge from strategy projections back to Sensor
  subscriptions.
- **Invariant 8 — Outer ring only.** Market selection reads outer-ring
  market data and writes only subscription state, never strategy
  products.

## Current files

- `selector.py` — loads eligible markets for each `MarketSelectionSpec`
  and returns merged asset ids.
- `merge.py` — merge policies and conflict reporting for strategy
  market sets.
- `subscription_controller.py` — serializes subscription updates and
  forwards them to the Sensor sink.

## Import rules

- May import from `pms.core.*`, `pms.storage.market_data_store`,
  `pms.strategies.projections`, and local module helpers.
- May not import from `pms.sensor.*` or `pms.actuator.*`.
- Must not import `pms.strategies.aggregate.Strategy`; consume
  projection dataclasses only.
- Must not reach into controller internals beyond the sink protocol
  boundary.

## Do not

- Never read live subscriptions from Sensor state to derive the next
  selection.
- Never compute selection inside Sensor adapters.
- Never widen the module into a general strategy orchestrator; it is
  only the active-perception market-selection boundary.
