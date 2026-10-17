# Strategy Authoring Guide

**Status:** active as of 2026-05-28.
**Audience:** strategy authors (researchers / quants) who want to
ship a strategy on top of the existing PMS platform without touching
the engineering plumbing (Sensor / Actuator / runner / schema).
**Scope:** end-to-end workflow from a strategy idea to a backtested
PAPER-ready `strategy_version` row, plus the iteration loop for
parameter tuning. Channel-A (configuration-only strategies) is the
primary path covered here; Channel-B (plugin strategies with their
own observation sources and agent logic) gets pointers only.

**Related documents:**
- `agent_docs/architecture-invariants.md` — the 9 load-bearing rules
  every strategy must honor. Read at minimum Invariants 2, 3, 4, 5.
- `agent_docs/promoted-rules.md` — process rules promoted from
  retrospectives. The relevant ones for strategy authors are listed
  in §9.
- `docs/research/backtest-spec-format.md` — the canonical reference
  for the sweep YAML schema accepted by `pms-research sweep`.
- `src/pms/controller/CLAUDE.md` — controller-layer detail (relevant
  if a strategy needs a forecaster variant the four built-ins do not
  cover).

---

## How to read this document

This is a procedure manual, not a design doc. Each step lists the
exact file to create or edit, the exact command to run, and the
observable outcome that means "step done." If a step does not
produce its observable, stop and read §9 (common pitfalls) before
proceeding — do not skip ahead.

The architecture rationale lives elsewhere. This document deliberately
does not re-explain *why* `(strategy_id, strategy_version_id)` is
immutable, or *why* factor composites are not first-class — those
answers are in `architecture-invariants.md`. Stay focused on *how*.

---

## 1. Prerequisites

Before you write a single line, confirm:

1. A working dev PostgreSQL is reachable and the schema is current:

   ```bash
   docker compose up -d postgres
   export DATABASE_URL=postgres://postgres:postgres@localhost:5432/pms_test
   uv run alembic upgrade head
   ```

2. `uv sync` runs cleanly and the canonical gates pass on a fresh
   clone (promoted rule: fresh-clone baseline verification):

   ```bash
   uv sync
   uv run pytest -q
   uv run mypy src/ tests/ --strict
   uv run lint-imports
   (cd dashboard && npm ci && npm run test:ci)
   ```

   Use the current head's gate output as the source of truth instead
   of copying historical pass/skipped count snapshots.

3. You can boot the API and reach the dashboard:

   ```bash
   uv run pms-api                       # http://127.0.0.1:8000
   (cd dashboard && PMS_API_BASE_URL=http://127.0.0.1:8000 npm run dev)
   #                                    # http://127.0.0.1:3100
   ```

If any of the above fails, stop and fix the environment first. A
strategy author should never debug the platform inside their own
strategy PR — that is a category mistake that retros have flagged as
"private-helper-boundary-drift" before.

---

## 2. Which channel does my strategy belong to?

Two channels exist. Pick before writing code; switching mid-flight
costs roughly 3x.

| Question | Channel A | Channel B |
|---|---|---|
| Can the decision be expressed as "compose existing raw factors with weights / thresholds, then route through one of the 4 built-in forecasters"? | yes | no |
| Need an observation source the main Sensor layer does not provide (news, social, external LLM reasoning over a corpus you fetch yourself)? | no | yes |
| Need a custom `propose` → `judge` → `build_intents` pipeline? | no | yes |
| Time-to-first-backtest (skilled author) | ~half day | ~2–3 days |
| Files you create | 2 (one strategy builder + one install script) | ~5 (the FLB/ripple template) |
| Engineering involvement | 0 (after the platform is stable) | 1 first-time registration in `runner.py` |
| Canonical example | `src/pms/strategies/paper_multifactor.py` | `src/pms/strategies/flb/` |

If you are not sure, default to A. You can always graduate to B
later, and a working A strategy is the cheapest way to learn the
platform's vocabulary.

The rest of this document covers Channel A end-to-end. Channel B
pointers are in §7.

---

## 3. Worked example — Channel A end-to-end

The example used through §3.1–§3.11 is a hypothetical strategy
called `imbalance_with_prior_v1`: take a position on the YES side
when `orderbook_imbalance > 0.15` **and** `metaculus_prior > 0.6`,
using the `rules` forecaster, with a 90-day horizon ceiling and a
$2 max position. This is illustrative; substitute your own
parameters.

### 3.1 Check the raw factors you need already exist

Strategies compose raw factors; they do not create new ones inline
(Invariant 4). The currently registered raw factors are listed in
`src/pms/factors/definitions/__init__.py`:

| Factor ID | What it measures |
|---|---|
| `anchoring_lag_divergence` | News-trigger divergence between catalyst time and market move |
| `fair_value_spread` | Subset/complement pricing gap on related markets |
| `favorite_longshot_bias` | Distance from observed favorite-longshot curve |
| `metaculus_prior` | External Metaculus / category prior probability |
| `no_count` | Per-market NO-side resolution sample count |
| `orderbook_imbalance` | Signed `(bid_depth − ask_depth) / total_depth` |
| `subset_pricing_violation` | Arbitrage-violating subset pricing |
| `yes_count` | Per-market YES-side resolution sample count |

For `imbalance_with_prior_v1` both required factors (`orderbook_imbalance`
and `metaculus_prior`) already exist. Done. If you need a factor
that does not exist, see §8 before continuing.

### 3.2 Write `build_imbalance_with_prior_strategy()`

Create `src/pms/strategies/imbalance_with_prior.py`. The skeleton
below is the smallest valid Channel-A strategy; copy
`src/pms/strategies/paper_multifactor.py` as the reference when you
need more knobs.

```python
from __future__ import annotations

from pms.strategies.aggregate import Strategy
from pms.strategies.projections import (
    CalibrationSpec,
    EvalSpec,
    FactorCompositionStep,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)

IMBALANCE_WITH_PRIOR_STRATEGY_ID = "imbalance_with_prior_v1"
_FACTOR_FRESHNESS_S = 300.0


def build_imbalance_with_prior_strategy() -> Strategy:
    return Strategy(
        config=StrategyConfig(
            strategy_id=IMBALANCE_WITH_PRIOR_STRATEGY_ID,
            factor_composition=(
                FactorCompositionStep(
                    factor_id="orderbook_imbalance",
                    # Raw depth is scaled as a rule delta before it becomes a
                    # probability; threshold_edge is reserved for already
                    # probability-shaped edges such as fair_value_spread.
                    role="rule_delta",
                    param="",
                    weight=0.25,
                    threshold=0.15,
                    required=True,
                    freshness_sla_s=_FACTOR_FRESHNESS_S,
                ),
                FactorCompositionStep(
                    factor_id="metaculus_prior",
                    role="threshold_edge",
                    param="",
                    weight=1.0,
                    threshold=0.60,
                    required=True,
                    freshness_sla_s=_FACTOR_FRESHNESS_S,
                ),
                FactorCompositionStep(
                    factor_id="rules",
                    role="blend_weighted",
                    param="",
                    weight=1.0,
                    threshold=None,
                ),
            ),
            metadata=(
                ("owner", "<your-handle>"),
                ("purpose", "imbalance + prior demo"),
                ("price_reference", "best_ask"),
                ("live_allowed", "false"),
            ),
        ),
        risk=RiskParams(
            max_position_notional_usdc=2.0,
            max_daily_drawdown_pct=50.0,
            min_order_size_usdc=0.50,
        ),
        eval_spec=EvalSpec(
            metrics=("brier", "pnl", "fill_rate"),
            max_brier_score=0.30,
            slippage_threshold_bps=50.0,
            min_win_rate=0.45,
        ),
        forecaster=ForecasterSpec(
            forecasters=(
                ("rules", (("threshold", "0.55"),)),
            )
        ),
        market_selection=MarketSelectionSpec(
            venue="polymarket",
            resolution_time_max_horizon_days=90,
            volume_min_usdc=100.0,
        ),
        calibration=CalibrationSpec(
            enabled=True,
            shrinkage_factor=0.35,
            extreme_clamp_low=0.08,
            extreme_clamp_high=0.92,
            min_resolved_for_extreme=20,
        ),
    )
```

Field-level reference is in §5. The constructor validates every composition
role and factor id against the catalog (`pms.factors.composition`); unknown
roles, factor ids, and forbidden raw-factor role combinations fail loudly
before a version can be registered.

**Run mypy before going further.** This file ships with the strict-mypy
baseline:

```bash
uv run mypy src/pms/strategies/imbalance_with_prior.py --strict
```

### 3.3 Write the install script

Create `scripts/install_imbalance_with_prior_strategy.py`. Use
`scripts/install_paper_multi_factor_strategy.py` as the template —
replace two imports and two function names; that is the whole change.

```python
"""Register imbalance_with_prior_v1 as an active PAPER strategy."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import asyncpg

from pms.factors.definitions import REGISTERED
from pms.storage.strategy_registry import PostgresStrategyRegistry
from pms.strategies.aggregate import Strategy
from pms.strategies.imbalance_with_prior import (
    build_imbalance_with_prior_strategy,
)
from pms.strategies.projections import FactorCompositionStep, StrategyVersion

_RAW_FACTOR_ROLES = frozenset(
    {
        "weighted",
        "precedence_rank",
        "threshold_edge",
        "posterior_prior",
        "posterior_success",
        "posterior_failure",
    }
)
_REGISTERED_FACTOR_IDS = frozenset(
    factor_cls.factor_id for factor_cls in REGISTERED
)


async def install(database_url: str) -> StrategyVersion:
    pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)
    try:
        registry = PostgresStrategyRegistry(pool)
        strategy = build_imbalance_with_prior_strategy()
        version = await registry.create_version(strategy)
        await registry.populate_strategy_factors(
            strategy.config.strategy_id,
            version.strategy_version_id,
            _raw_factor_steps(strategy),
        )
        return version
    finally:
        await pool.close()


def _raw_factor_steps(strategy: Strategy) -> tuple[FactorCompositionStep, ...]:
    return tuple(
        step
        for step in strategy.config.factor_composition
        if step.role in _RAW_FACTOR_ROLES
        and step.factor_id in _REGISTERED_FACTOR_IDS
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
    )
    args = parser.parse_args(argv)
    if not args.database_url:
        print("error: DATABASE_URL is not set", file=sys.stderr)
        return 2
    version = asyncio.run(install(args.database_url))
    print(f"strategy_id: {version.strategy_id}")
    print(f"strategy_version_id: {version.strategy_version_id}")
    print(f"created_at: {version.created_at.isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### 3.4 Install to the dev DB

```bash
uv run python scripts/install_imbalance_with_prior_strategy.py
```

Expected output (the version id is a deterministic SHA-256 of the
strategy config — same inputs always produce the same id):

```
strategy_id: imbalance_with_prior_v1
strategy_version_id: <64-hex-chars>
created_at: 2026-05-28T...
```

What `create_version` did under the hood (`strategy_registry.py:78-153`):

1. Upserted a row into `strategies (strategy_id, metadata_json)`.
2. Inserted a new row into `strategy_versions (strategy_version_id,
   strategy_id, config_json, created_at)`. The version id is
   computed by `compute_strategy_version_id(...)` over the full
   immutable snapshot.
3. Updated `strategies.active_version_id` to your new version.
4. Notified any registered `StrategyChangeCallback` (the runner
   subscribes — see §3.6).

**Re-running the script with the same config is idempotent**: same
inputs → same hash → `ON CONFLICT DO NOTHING`. You only get a new
row when you change something semantically meaningful (a weight, a
threshold, a factor, a risk param). This is what makes
`(strategy_id, strategy_version_id)` a safe primary key on every
downstream record (Invariant 3).

### 3.5 Verify the registry sees you

Three quick reads:

```bash
psql "$DATABASE_URL" -c "SELECT strategy_id, active_version_id FROM strategies WHERE strategy_id='imbalance_with_prior_v1';"

psql "$DATABASE_URL" -c "SELECT strategy_version_id, created_at FROM strategy_versions WHERE strategy_id='imbalance_with_prior_v1';"

psql "$DATABASE_URL" -c "SELECT factor_id, role, weight, threshold FROM strategy_factors WHERE strategy_id='imbalance_with_prior_v1';"
```

The third query should list only the **raw** factors filtered by
`_RAW_FACTOR_ROLES` in the install script — `rules` is a forecaster
binding, not a raw factor, so it correctly does not appear in
`strategy_factors`.

### 3.6 Boot the runner and watch the strategy come alive

```bash
PMS_AUTO_START=1 uv run pms-api
```

On boot, `runner.py:1787` calls
`strategy_registry.list_active_strategies()` and builds one
`ControllerPipeline` per active strategy. While the runner is
running, the registry's `register_change_callback` hooks
(`runner.py:985-986`) cause new versions to be picked up live: a
fresh install of a strategy version triggers
`_request_reselection` (re-computes the Sensor subscription) and
`_sync_controller_runtimes` (instantiates / retires controller
pipelines to match the registry).

**Net effect: a strategy author never has to restart the API to ship
a new strategy version.** Reinstall the script, the running runner
adopts it.

### 3.7 Confirm in the dashboard

With the API up and the dashboard at `http://127.0.0.1:3100`:

| Page | What you should see |
|---|---|
| `/strategies` | `imbalance_with_prior_v1` listed with the new version id |
| `/markets` | Markets matching your `MarketSelectionSpec` filters are subscribed (`venue=polymarket`, `horizon ≤ 90d`, `volume ≥ 100 USDC`) |
| `/signals` | Live `MarketSignal` rows for the subscribed markets, with `orderbook_imbalance` and `metaculus_prior` factor values |
| `/decisions` | `TradeDecision` rows tagged with `(imbalance_with_prior_v1, <your-version-id>)` |

If `/strategies` lists the strategy but `/markets` is empty, your
`MarketSelectionSpec` filter excluded everything in the local
universe — relax `resolution_time_max_horizon_days` or
`volume_min_usdc`.

### 3.8 Write a backtest sweep YAML

For a "this strategy works on historical data" claim you want a
backtest, not just live observation. Create `sweeps/imbalance_with_prior_v1.yaml`:

```yaml
base_spec:
  strategy_versions:
    - ["imbalance_with_prior_v1", "<paste-your-version-id-here>"]
  dataset:
    source: "polymarket_paper_2026q2"
    version: "v1"
    coverage_start: "2026-04-01T00:00:00+00:00"
    coverage_end:   "2026-05-01T00:00:00+00:00"
    market_universe_filter:
      venue: "polymarket"
    data_quality_gaps: []
  execution_model:
    fee_rate: 0.04
    slippage_bps: 10.0
    latency_ms: 250.0
    staleness_ms: 120000.0
    fill_policy: "immediate_or_cancel"
    displayed_depth_fill_ratio: 0.75
    adverse_selection_bps: 5.0
    calibration_source: "static_live_estimate"
  risk_policy:
    max_position_notional_usdc: 2.0
    max_daily_drawdown_pct: 50.0
    min_order_size_usdc: 0.50
  date_range_start: "2026-04-01T00:00:00+00:00"
  date_range_end:   "2026-05-01T00:00:00+00:00"
exec_config:
  chunk_days: 7
  time_budget: 1800
parameter_grid:
  execution_model.slippage_bps: [5.0, 10.0, 15.0]
```

`calibration_source: "static_live_estimate"` is the honest default
when you have no telemetry yet — `paper_report.py` will surface a
warning until you re-derive the model with
`scripts/execution_model_from_telemetry.py`.

Full schema reference: `docs/research/backtest-spec-format.md`.

### 3.9 Run the sweep

Two terminals.

```bash
# Terminal A — start the worker (long-lived; it picks up queued runs)
uv run pms-research worker --poll-interval 1.0
```

```bash
# Terminal B — queue the sweep and block on results
uv run pms-research sweep sweeps/imbalance_with_prior_v1.yaml --wait
```

The CLI prints `run_id`s. The worker owns the actual execution; the
API and dashboard only read results.

### 3.10 Read the results

Three surfaces:

1. **Dashboard** — `/backtest` lists runs; `/backtest/<run_id>` shows
   per-strategy metrics; `/backtest/<run_id>/compare` compares
   multiple `(strategy_id, version_id)` rows on the same run.
2. **JSON API** — `GET /api/pms/research/backtest/<runId>` returns
   the structured report.
3. **PostgreSQL** — `eval_records` (grouped by `(strategy_id,
   strategy_version_id)` — Invariant 3) plus the research-specific
   tables.

What to look for:

- **Brier score** under `eval_spec.max_brier_score`. If above, the
  strategy is mis-calibrated regardless of P&L.
- **Fill rate** > ~30% on a `fill_policy: immediate_or_cancel`
  spec. Lower suggests `slippage_bps` is unrealistic or the
  factor thresholds are too tight.
- **Per-variant deltas** across the `parameter_grid`. The grid
  expansion you wrote produces three runs (one per `slippage_bps`).
  If results swing wildly across a small slippage change, your edge
  is fragile.

### 3.11 Iterate

Iteration = edit your `build_<id>()` (e.g. tighten a threshold), rerun
the install script. You get a **new** `strategy_version_id`; the old
version is preserved untouched in `strategy_versions`. Cross-version
comparison is well-defined because every downstream row carries the
version it came from (Invariant 3).

The loop:

```
edit builder → uv run mypy <file> --strict
            → uv run python scripts/install_<id>_strategy.py
            → wait for dashboard to confirm runner re-synced
            → re-queue sweep with the new version id
            → compare new version vs old in /backtest/<run>/compare
```

**Never** edit a `strategy_versions.config_json` row in-place. That
is an Invariant 3 violation and will silently corrupt historical
metric aggregations.

### 3.12 Graduating: PAPER soak → LIVE gate

A strategy that backtests well next goes through PAPER soak — same
runtime, paper actuator. Concretely:

```bash
# Keep runner up for at least 24h with the strategy installed,
# then generate an evidence artifact:
uv run python scripts/paper_report.py --require-go > paper_soak_$(date +%F).txt
```

The artifact contains gates, P&L, drawdown, Brier improvement vs
baseline, Sharpe, risk events, and calibration. A clean GO outcome is
the precondition for the engineering team to put the strategy behind
the LIVE preflight gate. See
`agent_docs/production-readiness-2026-05.md` for the LIVE-side
blockers that have to clear before the canary order.

PAPER soak is **not** the strategy author's responsibility to ship to
production; it is the artifact you hand off to the engineering side.

---

## 4. The two-file diff that defines a Channel-A strategy

A finished Channel-A strategy contributes exactly two new files to
the repo (plus tests — see §6). Concretely:

```
src/pms/strategies/<your_id>.py          # build_<your_id>_strategy()
scripts/install_<your_id>_strategy.py    # registry install entrypoint
sweeps/<your_id>_v1.yaml                 # initial backtest spec (optional, recommended)
tests/unit/strategies/test_<your_id>.py  # unit tests for the builder
```

Anything more than this is a smell. If your strategy needs runner
changes, schema changes, or sensor changes, you are either (a) writing
a Channel-B strategy in disguise, or (b) breaking the strategy /
infrastructure boundary (Invariant 5). Stop and re-read §2.

---

## 5. Reference: projection field semantics

This section is the field dictionary for the five projection types a
Channel-A strategy must populate. Skim once; come back when you need
to know what a field controls.

### 5.1 `StrategyConfig`

| Field | Required | Notes |
|---|---|---|
| `strategy_id` | yes | Must be unique and human-readable. Use `<theme>_v<n>`. Once installed, a new id means a new strategy from the registry's POV — version-bumping a `_v1` to `_v2` is fine for major semantic change. |
| `factor_composition` | yes | A tuple of `FactorCompositionStep` (see §5.2). Order matters for `precedence_rank` roles. |
| `metadata` | yes | A tuple of `(key, value)` pairs of strings. Convention keys: `owner`, `purpose`, `tier`, `price_reference`, `live_allowed`, `requires_strict_factor_gates`. Set `live_allowed=false` until the strategy has cleared PAPER soak. |

### 5.2 `FactorCompositionStep.role` values

`role` is a string discriminator that tells the controller pipeline
how to interpret the step. The accepted values are partitioned into
"raw factor roles" (consumed by `RulesForecaster` /
`StatisticalForecaster`) and "binding roles" (which forecaster /
calibrator participates).

| Role | Kind | What it does |
|---|---|---|
| `weighted` | raw | Linear contribution to a weighted score |
| `precedence_rank` | raw | Ordinal rank for `threshold_edge` evaluation: **lower** weight = **higher** priority. Steps are tried in ascending rank order and the first firing edge wins (reproduces legacy `RulesForecaster.predict` precedence) |
| `threshold_edge` | raw | Fires when `value >= threshold` (or `<= -threshold` for sign-aware factors) |
| `posterior_prior` | raw | Beta prior `α` contribution |
| `posterior_success` | raw | Beta prior `β`-success contribution |
| `posterior_failure` | raw | Beta prior `β`-failure contribution |
| `blend_weighted` | binding | Includes a forecaster (`rules` / `stats` / `llm`) in the final weighted blend |
| `rule_delta` | binding | Additive shift to the rules forecaster output |
| `runtime_probability` | binding | Forecaster output used as direct probability (typically `llm`) |

The install script's `_RAW_FACTOR_ROLES` filter (§3.3) is the
authoritative list of raw roles.

### 5.3 `ForecasterSpec`

A tuple of `(name, params)` where `name` is one of the four
forecasters wired in `src/pms/controller/factory.py:98-135`:

| Name | Mode constraint | Params accepted |
|---|---|---|
| `rules` | any | `threshold` (default `0.02`, the min-edge required to act) |
| `stats` | any | `window` (free-form, currently unused), `prior_strength` (default `2.0`) |
| `llm` | any | none today; rejects unknown params explicitly |
| `paper_canary` | PAPER only | `edge_bps`, `max_probability`, `min_price`, `max_price`, `sample_modulus`, `sample_remainder` |

Multiple forecasters can run side-by-side; the `blend_weighted` role
in `factor_composition` controls the weighted blend.

### 5.4 `MarketSelectionSpec`

| Field | Required | Notes |
|---|---|---|
| `venue` | yes | `"polymarket"` (only adapter shipped). Kalshi is reserved but not implemented. |
| `resolution_time_max_horizon_days` | yes (nullable) | Excludes markets resolving farther out. Tighter horizons reduce universe size dramatically. |
| `volume_min_usdc` | yes | Excludes thin markets. `100.0` is a reasonable starting floor. |
| `spread_max_bps` | no | Optional max bid-ask spread filter in midpoint-relative bps |
| `depth_min_usdc` | no | Optional min top-of-book depth filter |
| `liquidity_min_usdc` | no | Optional composite liquidity threshold |
| `yes_price_min` | no | Optional minimum YES price. Use this to avoid subscribing to markets the router will reject as out of price band. |
| `yes_price_max` | no | Optional maximum YES price. Use this to avoid subscribing to markets the router will reject as out of price band. |
| `accepting_orders` | no (default `true`) | Filter to markets currently accepting orders |

Tighter filters = smaller subscription = lower bandwidth = faster
backtests. Don't optimize this until your strategy works on a
permissive universe.

### 5.5 `RiskParams`

| Field | Notes |
|---|---|
| `max_position_notional_usdc` | Hard cap per market. For PAPER demos keep $1–$5. For LIVE canary `every_order` mode, $1 is the proven floor. |
| `max_daily_drawdown_pct` | Triggers the auto-halt when crossed (`runner.py` auto-halt code path). Strategy authors should set this conservatively; the platform's own envelope is independent and stricter. |
| `min_order_size_usdc` | Smaller orders are rejected at the executability gate. |

### 5.6 `EvalSpec`

| Field | Notes |
|---|---|
| `metrics` | Tuple of metric names. `("brier", "pnl", "fill_rate")` is the standard set. Adding a metric without a corresponding implementation in `pms.evaluation.metrics` will fail at eval time. |
| `max_brier_score` | Acceptance threshold. Default `0.30`. |
| `slippage_threshold_bps` | Triggers slippage-spike auto-halt. |
| `min_win_rate` | Acceptance threshold. |

### 5.7 `CalibrationSpec`

Optional. Defaults are sane for new strategies (`enabled=False`).
Once you have ≥ ~20 resolved samples, set `enabled=True` to apply
isotonic calibration via `NetcalCalibrator`. The shrinkage parameters
control how aggressively raw forecaster output is pulled toward the
empirical posterior.

---

## 6. Testing

A strategy ships with a unit test that locks in the **business
invariant**, not just the symptom (promoted rule). Tests live in
`tests/unit/strategies/test_<your_id>.py`.

Minimal coverage for a Channel-A strategy:

1. **Builder returns a valid `Strategy`.** Trivial; mostly a typo
   guard.
2. **Version id is deterministic and content-addressed.** Build
   twice, hash twice, assert equal. Then mutate one threshold,
   rebuild, assert new hash differs.
3. **`MarketSelectionSpec` excludes the markets you expect.** Drive
   the spec through `MarketSelector` (or its fake) with a fixture
   universe; assert which markets pass.
4. **Risk caps are respected.** A composed market signal that would
   produce an oversized order is sized down by `KellySizer` to fit
   `max_position_notional_usdc`. Use a small `tests/fixtures/`
   `MarketSignal` and assert the emitted order quantity.

What **not** to test:

- Don't mock the runner. The strategy works because the platform
  wires it; testing your own wiring is meaningless.
- Don't mock the database. If you need persistence behavior, use
  the `PMS_RUN_INTEGRATION=1` integration path.

Run before committing:

```bash
uv run pytest tests/unit/strategies/test_<your_id>.py -q
uv run mypy src/pms/strategies/<your_id>.py scripts/install_<your_id>_strategy.py --strict
```

---

## 7. When you actually need Channel B

You need Channel B if any of these is true:

- The strategy ingests data the main Sensor layer does not provide
  (news headlines, social posts, an external API the platform does
  not adapt).
- The strategy uses an LLM agent to reason over candidates, not just
  as a forecaster called per-signal.
- The strategy's intent is structurally different from "edge over
  market on a single token" (e.g. basket trades, paired arbitrage,
  conditional orders).

Read `src/pms/strategies/base.py` for the protocol surface. Then copy
`src/pms/strategies/flb/` as a working template — it implements the
`source / controller / agent / strategy / evaluator` quintet for a
fast-loop-back strategy backed by a warehouse source.

**Critical constraints** (Invariant 5):

- Your `StrategyModule` never calls a Sensor adapter directly.
  Plugin-local observation goes through `StrategyObservationSource`.
- Your `StrategyAgent` produces `TradeIntent` / `BasketIntent`, not
  orders. The `runtime_bridge` (`src/pms/strategies/runtime_bridge.py`)
  converts intents to `TradeDecision` and they go through
  `RiskManager` + `Actuator` like any other strategy. Bypassing this
  bridge will fail import-linter and code review.
- Your plugin must not write to outer-ring or middle-ring tables
  (Invariant 8). Strategy-specific artifacts go in
  `strategy_judgement_artifacts` and `strategy_execution_artifacts`
  (inner ring).

First registration of a new Channel-B module type requires a
one-time addition in `runner.py` (look at `_build_flb_module`,
`_build_ripple_module`, `_build_anchoring_module` for the pattern).
This is the only time engineering involvement is non-optional. After
the type is registered, every subsequent version of your
Channel-B strategy iterates exactly like Channel A — same install
script pattern, same hot-reload behavior.

---

## 8. Adding a new raw factor

Sometimes the existing eight raw factors aren't enough. Adding one
is a small, well-contained job — but it's still infrastructure, not
strategy, and it lives under Invariant 4 (raw factors only).

Steps:

1. Create `src/pms/factors/definitions/<your_factor>.py`. Pattern:
   subclass `FactorDefinition`, set `factor_id` (string, must be
   globally unique), set `required_inputs` (e.g. `("orderbook",)`),
   implement `compute(signal, outer_ring) -> FactorValueRow | None`.
   See `src/pms/factors/definitions/orderbook_imbalance.py:24-49` for
   the canonical shape.
2. Register it in `src/pms/factors/definitions/__init__.py` — add to
   the `REGISTERED` tuple and to `__all__`.
3. Write piecewise tests (promoted rule
   `domain-math-piecewise`): if your `compute()` has guards (zero
   depth → `None`, missing field → fallback, etc.), test each branch
   plus the boundaries between them.
4. Run the gate:

   ```bash
   uv run pytest tests/unit/factors/test_<your_factor>.py -q
   uv run mypy src/pms/factors/definitions/<your_factor>.py --strict
   ```

5. Only after the factor is in the registry can a strategy reference
   its `factor_id` in `FactorCompositionStep`.

Reusability test: if your factor encodes strategy-specific weights
or thresholds, it isn't raw. Move the parameter to the strategy's
`FactorCompositionStep`.

---

## 9. Common pitfalls

### Deferred infrastructure

Meta-evidence producers (F-12) and cross-market relation factor lifecycle
integration (F-13) remain deferred until they have a bounded producer,
freshness contract, and production writer. Their factor ids are therefore
excluded from the seeded/public catalog; do not author a strategy against
`subset_relation`, `contradiction_relation`, `semantic_similarity`, or
`cross_market_mispricing` yet.

Listed in rough frequency order based on retro patterns plus the
invariants most likely to bite during strategy authoring.

| Pitfall | What goes wrong | The rule that says don't |
|---|---|---|
| Putting a composite expression in `factor_composition` (e.g. `factor_id="0.3*imbalance + 0.7*prior"`) | `_RAW_FACTOR_ROLES` filter drops it from `strategy_factors`, runtime treats it as missing | Invariant 4 |
| Editing a `strategy_versions.config_json` row in-place after install | Historical eval rows tagged with the old hash become incomparable; `mean(brier)` silently averages across versions | Invariant 3 |
| Strategy code importing from `pms.sensor` or `pms.actuator` | Import-linter rejects the PR | Invariant 5 |
| Plugin strategy writing to `markets` / `book_snapshots` / `factor_values` | Schema is grep-checked; PR rejected | Invariant 8 |
| Using `live_allowed=true` in metadata before PAPER soak GO | LIVE startup gate will refuse to load the strategy; risk-team will reject | LIVE preflight + `production-readiness-2026-05.md` |
| Setting `min_order_size_usdc` below the venue minimum | Orders silently rejected at the executability gate; backtest looks fine, live looks broken | Risk envelope is independent of `RiskParams` |
| Forgetting `freshness_sla_s` on a `required=True` step | A stale factor value passes the gate, decision is made on old data | Promoted rule: behavioural fixes, not comments |
| Mocking the runner / DB in unit tests | Tests pass but ship broken integration; retro pattern `migration-isolation-test-discipline` flagged this repeatedly | Promoted rule: tests must encode the business invariant |
| Committing the install script run output (`paper_soak_*.txt`) by accident | Generated-artifact-drift; the file isn't product code | Promoted rule: generated-artifact-drift |
| Comments-as-fix when a behavioural finding appears in review | The fix must change runtime behaviour | Promoted rule: "Comments are not fixes" |

---

## 10. Further reading

| Topic | File |
|---|---|
| Architecture invariants (the 9 load-bearing rules) | `agent_docs/architecture-invariants.md` |
| Promoted process rules from retros | `agent_docs/promoted-rules.md` |
| Backtest sweep YAML reference | `docs/research/backtest-spec-format.md` |
| Controller layer detail (new forecasters / calibrators) | `src/pms/controller/CLAUDE.md` |
| Production readiness blockers (LIVE side) | `agent_docs/production-readiness-2026-05.md` |
| Sensor layer detail (factor inputs) | `src/pms/sensor/CLAUDE.md` |
| Project roadmap (sub-spec dependency DAG) | `agent_docs/project-roadmap.md` |

---

## 11. Maintenance of this guide

Update this guide when:

- A new raw factor lands → §3.1 table refresh.
- A new forecaster lands → §5.3 table refresh.
- A new `FactorCompositionStep.role` value is added → §5.2 table
  refresh.
- A new common pitfall is observed in code review or retro → §9.
- Channel-B registration becomes self-service (no `runner.py` edit
  needed) → §7 simplification.

Stale tables in this guide are a worse problem than a missing
section: a confident wrong answer wastes more strategy-author time
than an honest "not covered." If you spot a stale row, fix it or
flag it in a retro.
