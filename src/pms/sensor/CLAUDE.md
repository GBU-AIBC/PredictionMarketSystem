# Identity & Context Awareness

**CRITICAL**: Address the user as "Stometa" at the start of EVERY response.

This serves as a context-awareness signal — if missing, indicates
context drift.

---

# Sensor layer

**Part of:** [prediction-market-system](../../../CLAUDE.md)
**Role:** ingest venue market data, normalize to `MarketSignal`,
write to the outer ring.

## Layer-relevant invariants

Full detail in [architecture invariants](../../../agent_docs/architecture-invariants.md). The subset
that governs this layer:

- **Invariant 1 — Concurrent, not phased.** Sensor tasks run
  concurrently with Controller / Actuator / Evaluator via
  `asyncio.Queue` fan-in (`src/pms/sensor/stream.py`). Do not
  introduce synchronous barriers between Sensor and the downstream
  queue consumer.
- **Invariant 5 — Strategy-agnostic.** Sensor modules do not import
  anything from `pms.strategies.*` or `pms.controller.*`. Sensor
  accepts a `market_ids: list[str]` subscription input and emits
  `MarketSignal` with no knowledge of which strategy consumes it.
- **Invariant 6 — Subscription sink.** The `MarketDataSensor`
  accepts subscription updates *from outside* (pushed by
  `SensorSubscriptionController`, landed in S4). It never pulls the
  subscription list itself; it receives it.
- **Invariant 7 — Two-layer structure.** There are exactly two
  sensor types per venue:
  - `MarketDiscoverySensor` — unconditional full-universe scan,
    writes `markets` / `tokens`.
  - `MarketDataSensor` — subscription-driven streaming, writes
    `book_snapshots` / `book_levels` / `price_changes` / `trades`.
- **Invariant 8 — Outer ring only.** Sensor writes only to outer-
  ring tables. No sensor code references `strategy_id`.

## Current files

- `adapters/historical.py` — JSONL/CSV replay (backtest mode).
- `adapters/market_discovery.py` — Gamma `/markets` discovery poller
  that writes `markets` / `tokens`.
- `adapters/market_data.py` — WebSocket sensor. Maintains the
  per-asset orderbook mirror, persists `book` / `price_change` /
  `last_trade_price`, and accepts subscription updates from
  outside the sensor layer.
- `stream.py` — fan-in: merges sensors into `Queue[MarketSignal]`.
- `watchdog.py` — existing but not yet wired to stream sensor
  (S1 completes the wiring).

## Do not

- Never `from pms.strategies.aggregate import Strategy` or similar —
  triggers Invariant 5 violation (fails the import-linter rule S2
  adds).
- Never `from pms.controller import …` — same reason.
- Never add `strategy_id` to a sensor write or a sensor-level data
  structure. All per-strategy filtering is downstream.
- Never conflate discovery and data in one sensor class (Invariant 7).
- Never let a sensor call decide its own subscription list at
  runtime — the list must arrive from `SensorSubscriptionController`.

## When adding a new venue

Land the venue as a **pair** of sensors (discovery + data) plus a
shared adapter module for the venue client. The pair mirrors the
two-ring structure of the outer ring writes.
