# Prediction Market System (pms)

Modular prediction market trading system organised as a concurrent cybernetic
feedback web across Sensor, Controller, Actuator, Evaluator, and feedback edges.
Sensor, Controller, Actuator, and Evaluator run as concurrent asyncio tasks
with bidirectional feedback edges; the runtime is not a phased pipeline.

Target venues: Polymarket (primary). Kalshi is reserved in the venue enum but
has no adapter in v1 — see CP06's stub gate. Implemented run modes are
`backtest`, `paper`, and gated Polymarket `live`. LIVE mode remains fail-closed
unless `live_trading_enabled=true`, required Polymarket credentials validate,
and `operator_approval_mode=every_order` keeps each live order behind an
operator gate during the initial real-money phase.

## Status: Gate 2 CLOSED — Paper Soak Blocked Pending Launch Artifacts

All core PRs merged as of 2026-05-03. The system is code-complete for
H1 (FLB contrarian) strategy, but the supervised live-data paper soak is
fail-closed until required non-secret launch artifacts are staged. In
particular, `config.live-soak.yaml` does not start until
`/secure/pms/category-prior-observations.csv`,
`/secure/pms/flb-calibration.csv`, and the FLB `.provenance.json` sidecar exist
and pass schema/sample validation. Those files are not credentials; they must
be generated with the artifact commands in step 4a below. The checked-in
`docs/research/flb-deciles.csv` is an old Gamma fallback diagnostic and is not a
launch artifact. H2 anchoring-lag /
LLM-news replay remains research-only until the H1 historical data spine proves
enough coverage and measurable edge.

| Milestone | Status | PR | Description |
|-----------|--------|-----|-------------|
| Gate 1: Research | ✅ | #16 | Prediction methodology brief (H1+H2 strategy) |
| FLB Feasibility | ✅ | #45 | Data feasibility analysis + contract-level FLB analysis |
| Risk Config | ✅ | #43 | Live-soak risk envelope (drawdown halt, exposure limits) |
| Config Loading | ✅ | #44 | Paper soak config at API startup |
| FLB Data Pipeline | ✅ | #47 | Historical warehouse source for robust FLB measurement |
| LLM Forecaster | ✅ | #46 | Provider-switchable (Anthropic/OpenAI) with per-market cache |
| Strategy Relax | ✅ | #48 | Default strategy required factors relaxed for PAPER |
| **Gate 2: Edge Validation** | ⚠️ | — | Code gates closed; paper soak startup still requires secure FLB calibration artifact |

### What's Needed Before Live Trading

Six things remain between the current branch and real capital:

1. **Non-secret launch artifacts** — Generate
   `/secure/pms/category-prior-observations.csv`,
   `/secure/pms/flb-calibration.csv`, and
   `/secure/pms/flb-calibration.csv.provenance.json`. The FLB artifacts come
   from `/secure/pms/polymarket_resolved_binary.csv` with the checked-in Dune
   export SQL plus `scripts/flb_data_feasibility.py --source warehouse-csv`;
   the category-prior artifact comes from
   `scripts/export_category_prior_observations.py`. Paper-soak startup fails
   closed until these artifacts exist and pass runtime sample gates. The Dune
   API key is a credential; the export, category-prior, and calibration
   artifacts are not.
2. **Polymarket credentials** — 6 fields (private_key, api_key, api_secret,
   api_passphrase, funder_address, signature_type). For local LIVE, stage them
   in the chmod 600 `local_secret_file` outside the working tree; never put
   them in shell exports, `.env`, or config files.
3. **Confirm risk envelope** — Ratified PAPER soak defaults from
   `config.live-soak.yaml`: `max_position_per_market=$1`,
   `max_total_exposure=$50`, `max_drawdown_pct=20%`,
   `max_daily_loss_usdc=$20`, `max_open_positions=50`,
   `max_exposure_per_risk_group=$1`, `min_order_usdc=$1`,
   `slippage_threshold_bps=50`, `max_quantity_shares=500`.
4. **30-day paper soak** — Run with live Polymarket data and require the
   machine-checkable Go/No-Go report gate to pass before risking capital. See
   [Orchestration guide](#orchestration-guide) below.
5. **Active LIVE strategy** — Confirm the active Postgres strategy version has
   an explicit `metadata.live_allowed=true` opt-in; `pms-live preflight`
   rejects paper-only or unmarked strategies before writing a final go/no-go
   artifact, and LIVE startup rejects that artifact if the active strategy set
   or projection changes afterward.
6. **Operator approval rehearsal** — Run `scripts/rehearse_first_order.py`
   and keep its PASS report at `live_operator_rehearsal_report_path` so LIVE
   validation can prove the approval gate denies, matches, and consumes
   artefacts before the first real submit.

See [docs/operations/live-polymarket-runbook.md](docs/operations/live-polymarket-runbook.md)
for the full PAPER soak, credential setup, operator approval, rollback, and
emergency stop runbook. Install the optional live SDK with
`uv sync --extra live --extra llm` before starting LIVE mode or live-soak
paper mode. The true LIVE template leaves LLM disabled by default; keep it
disabled for the first real-money path unless you separately stage
`PMS_LLM__API_KEY`, accept the provider dependency, and rerun preflight.

## Orchestration Guide

### Architecture: Four Concurrent Layers

```
┌──────────────────────────────────────────────────────┐
│                    Sensor Layer                       │
│  HistoricalSensor → MarketDiscoverySensor → Stream   │
│  (pulls market data from Polymarket / Gamma API)      │
├──────────────────────────────────────────────────────┤
│                  Controller Layer                     │
│  Pipeline: features → forecaster → sizer → planner   │
│  (Beta-Binomial + LLM forecaster + Kelly sizing)      │
├──────────────────────────────────────────────────────┤
│                  Actuator Layer                       │
│  RiskManager → ExecutionPlan → OrderState            │
│  (7 auto-halt triggers, exposure limits, drawdown)    │
├──────────────────────────────────────────────────────┤
│                  Evaluator Layer                      │
│  MetricsCollector → EvalSpool → FeedbackEngine       │
│  (Brier score, hit rate, calibration, FLB edge)       │
└──────────────────────────────────────────────────────┘
        ▲                                              │
        └──────────── Feedback Edges ──────────────────┘
```

### Strategy Components

| Component | What It Does | Config Key |
|-----------|-------------|------------|
| **Beta-Binomial Forecaster** | Bayesian probability model using historical resolution data | `forecaster: beta_binomial` |
| **LLM Forecaster** | Provider-switchable (Anthropic/OpenAI) market prediction with per-market 30s TTL cache | `forecaster: llm`, `llm.provider`, `llm.base_url` |
| **FLB Signal** | Detects Favorite-Longshot Bias in Polymarket pricing | `pms.strategies.flb.FlbStrategyModule` |
| **Kelly Sizer** | Position sizing based on edge magnitude and Kelly fraction | `sizing.kelly_fraction` |
| **RiskManager** | 7 auto-halt triggers plus per-market, total, and risk-group exposure caps | `risk.max_drawdown_pct`, `risk.max_daily_loss_usdc`, `risk.max_exposure_per_risk_group` |

### Run Modes

| Mode | Purpose | Command |
|------|---------|---------|
| **backtest** | Historical simulation through the runner/API control plane | `PMS_MODE=backtest uv run pms-api`, then authenticated `POST /run/start` |
| **paper** | Live data, simulated trades, no real orders | `uv run pms-api --config config.live-soak.yaml`, then authenticated `POST /run/start` |
| **live** | Real Polymarket trading (gated) | `uv run pms-api` with `PMS_MODE=live` + credentials |

### Step-by-Step: From Zero to Paper Soak

```bash
# 1. Install dependencies for live-data paper mode, Polymarket SDK, and
#    optional LLM parity checks. H1 FLB keeps `llm.enabled: false` for the
#    launch soak unless an operator explicitly opts in and stages PMS_LLM__API_KEY.
uv sync --extra live --extra llm

# 2. Start PostgreSQL (required for market data persistence)
docker compose up -d postgres

# 3. Apply migrations
export DATABASE_URL=postgres://postgres:postgres@localhost:5432/pms_test
# Before running migrations or installing a canary strategy, verify that
# `DATABASE_URL` points at the intended Postgres server and database; another
# local Postgres on port 5432 can silently catch `localhost` traffic.
psql "$DATABASE_URL" -Atc "select current_database(), inet_server_addr(), inet_server_port(), version();"
uv run alembic upgrade head

# Optional no-credential plumbing smoke before launch artifacts are staged.
# This verifies live-data -> controller -> paper-actuator flow only; it is not
# H1 paper-soak performance evidence and cannot satisfy the final GO gate.
uv run python scripts/prepare_local_paper_soak_config.py \
  --output config.local.paper-canary.yaml \
  --paper-canary
uv run python scripts/install_paper_canary_strategy.py \
  --database-url "$DATABASE_URL" \
  --archive-default \
  --sample-modulus 1
# Start the API with config.local.paper-canary.yaml, POST /run/start from a
# second shell, save /status, /readiness, /strategies, /markets, /decisions,
# /trades, /positions, and /metrics JSON snapshots, then validate them with:
uv run python scripts/check_paper_canary_smoke.py \
  --status-json "$PAPER_CANARY_EVIDENCE_DIR/status.json" \
  --readiness-json "$PAPER_CANARY_EVIDENCE_DIR/readiness.json" \
  --strategies-json "$PAPER_CANARY_EVIDENCE_DIR/strategies.json" \
  --markets-json "$PAPER_CANARY_EVIDENCE_DIR/markets.json" \
  --decisions-json "$PAPER_CANARY_EVIDENCE_DIR/decisions.json" \
  --trades-json "$PAPER_CANARY_EVIDENCE_DIR/trades.json" \
  --positions-json "$PAPER_CANARY_EVIDENCE_DIR/positions.json" \
  --metrics-json "$PAPER_CANARY_EVIDENCE_DIR/metrics.json"

# 4. Create a private artifact directory and repo-ignored local config.
#    On macOS, root-level /secure may be unavailable; keep local PAPER
#    artifacts in a user-owned chmod 700 directory and let the helper rewrite
#    config.local.live-soak.yaml to that directory. Re-run with --overwrite
#    only after preserving any local edits.
export PMS_SECURE_DIR="${PMS_SECURE_DIR:-$HOME/.local/share/pms/secure}"
uv run python scripts/prepare_local_paper_soak_config.py \
  --secure-dir "$PMS_SECURE_DIR"
# Edit config.local.live-soak.yaml:
#   - Adjust risk.max_position_per_market (paper default: $1)
#   - Adjust risk.max_total_exposure (proposed: $50)
#   - Adjust risk.max_drawdown_pct (proposed: 20)
#   - Keep strategies.flb_calibration_path pointed at the local secure CSV below
#   - Keep controller.category_prior_observations_path pointed at the local secure CSV below

# 4a. Generate required non-secret launch artifacts outside the repo.
#     The artifact exporters reject output paths inside the working tree.
#     The Dune API key is a credential; keep it in an operator secret store
#     and load it only into this shell.
export DUNE_API_KEY="<load from operator secret store>"
uv run python scripts/export_flb_warehouse_from_dune.py \
  --output "$PMS_SECURE_DIR/polymarket_resolved_binary.csv" \
  --performance large
uv run python scripts/flb_data_feasibility.py \
  --source warehouse-csv \
  --input "$PMS_SECURE_DIR/polymarket_resolved_binary.csv" \
  --output "$PMS_SECURE_DIR/flb-feasibility.md" \
  --csv "$PMS_SECURE_DIR/flb-deciles.csv" \
  --calibration-csv "$PMS_SECURE_DIR/flb-calibration.csv" \
  --calibration-source-label warehouse-flb-v1 \
  --calibration-provenance-json \
    "$PMS_SECURE_DIR/flb-calibration.csv.provenance.json"
uv run python scripts/export_category_prior_observations.py \
  --output "$PMS_SECURE_DIR/category-prior-observations.csv" \
  --min-observations 100

# Fly/LIVE volume staging keeps the same artifact names under /secure/pms,
# for example: --calibration-csv /secure/pms/flb-calibration.csv and
# --calibration-provenance-json /secure/pms/flb-calibration.csv.provenance.json

# 4b. Fail fast before starting the API; this uses the same artifact loaders
#     as runtime startup, also verifies each configured private artifact parent,
#     and exits nonzero when required launch artifacts are missing, malformed,
#     or staged in a permissive directory.
uv run python scripts/check_paper_soak_artifacts.py \
  --config config.local.live-soak.yaml

# 5. Start the paper-soak API control plane. Keep the token private;
#    scripts/paper_report.py reads the same token from PMS_API_TOKEN.
export PMS_API_TOKEN="$(openssl rand -hex 32)"
uv run pms-api --config config.local.live-soak.yaml

# 6. In another shell, start the runner and monitor status.
#    The `pms-api` command starts the API control plane; it does not start
#    the runner until an authenticated `POST /run/start` succeeds.
curl -X POST \
  -H "Authorization: Bearer $PMS_API_TOKEN" \
  http://127.0.0.1:8000/run/start
curl -H "Authorization: Bearer $PMS_API_TOKEN" \
  http://127.0.0.1:8000/status

# Optional H1 FLB runtime smoke after launch artifacts pass startup gates.
# Capture /status, /readiness, /strategies, /markets, /decisions, /trades,
# /positions, and /metrics JSON snapshots into H1_FLB_EVIDENCE_DIR, then
# validate them.
uv run python scripts/check_h1_flb_smoke.py \
  --status-json "$H1_FLB_EVIDENCE_DIR/status.json" \
  --readiness-json "$H1_FLB_EVIDENCE_DIR/readiness.json" \
  --strategies-json "$H1_FLB_EVIDENCE_DIR/strategies.json" \
  --markets-json "$H1_FLB_EVIDENCE_DIR/markets.json" \
  --decisions-json "$H1_FLB_EVIDENCE_DIR/decisions.json" \
  --trades-json "$H1_FLB_EVIDENCE_DIR/trades.json" \
  --positions-json "$H1_FLB_EVIDENCE_DIR/positions.json" \
  --metrics-json "$H1_FLB_EVIDENCE_DIR/metrics.json" \
  --min-decisions 1 \
  --min-trades 1 \
  --min-positions 1

# The H1 FLB runtime smoke is plumbing evidence only; it does not satisfy the
# 30-day paper-soak GO gate.

# 7. Generate daily paper soak report
uv run python scripts/paper_report.py --date 2026-05-03
```

### Step-by-Step: From Paper Soak to Live

After 30 days of paper soak and a passing Go/No-Go gate:

```bash
# 0. Confirm the latest paper report passes the machine-checkable gate
sudo install -d -m 700 -o "$USER" /secure/pms
export PAPER_SOAK_REPORT_DATE="$(date -u +%F)"  # use the completed soak report date
uv run python scripts/paper_report.py \
  --date "$PAPER_SOAK_REPORT_DATE" \
  --output /secure/pms/paper-soak-go-report.md \
  --require-go

# `scripts/paper_report.py --require-go` also fetches `/readiness` and records
# a NO-GO risk event unless readiness status is `ready` and every check is
# `ready` or `disabled`.

# 1. Stage Polymarket credentials in a chmod 600 local secret file,
#    never shell exports or .env:
#    ~/.config/pms/polymarket.local-secrets.yaml

# 2. Create the ignored local LIVE runtime config from the non-secret template.
cp config.live.yaml.example config.live.yaml
#    Fill live_* operator/compliance fields and keep the committed risk envelope.
#    Do not add Polymarket credential fields to config.live.yaml.

# 3. Reconfirm the secure operator artifact directory, but leave the approval
#    JSON absent for preflight. Do not create the approval JSON before preflight.
#    This directory must be outside the repo working tree; the artifact path
#    must be a regular file, not a symlink.
sudo install -d -m 700 -o "$USER" /secure/pms

# 4. Run the read-only live preflight and write the startup gate artifact
uv run pms-live preflight \
  --config config.live.yaml \
  --output /secure/pms/credentialed-preflight.json

# 5. Start live mode. Create the approval JSON only after preview review.
uv run pms-api --config config.live.yaml

# 6. After the first approved live order fills, reconcile PMS state against
#    Polymarket and persist the post-live proof artifact. This requires the
#    configured credentialed preflight artifact to still validate.
uv run pms-live reconcile-live-order \
  --config config.live.yaml \
  --decision-id <decision-id> \
  --reconciled-by <operator-id> \
  --output /secure/pms/first-live-order-reconciliation.json
```

For Fly LIVE capital, do not repurpose the paper-soak `fly.toml`. Copy
`fly.live.toml.example` to the ignored `fly.live.toml`, replace every
`__FILL_IN_*__` value, create the `/secure` volume, stage `DATABASE_URL`,
`PMS_API_TOKEN`, `PMS_DISCORD__WEBHOOK_URL`, and all Polymarket credentials
with `fly secrets import -c fly.live.toml`, then deploy with
`fly deploy -c fly.live.toml` after the credentialed preflight artifact has
passed.

**Critical**: The system is fail-closed. If any credential is missing, risk
limits are violated, the credentialed preflight artifact is missing/invalid or
stale for the current active strategies, or an auto-halt trigger fires, the
system stops submitting orders automatically.

## Agent Strategy Boundary

Agent strategy modules may propose, judge, and explain market actions, but they
cannot submit orders, cannot override risk, and cannot override reconciliation.
Their typed output path is `TradeIntent | BasketIntent` -> `ExecutionPlan` ->
`RiskDecision` -> `OrderState` -> reconciliation -> evaluator.

Execution planning is an executability gate, not execution authority. The
planner checks quotes, depth, book freshness, minimum size, tick size, and
slippage before any planned order can reach the existing risk and actuator
path. RiskManager and ActuatorExecutor remain the only order-submission route.
Predict-Raven is an external reference pattern for plugin/runtime shape and
durable artifacts, not an architecture PMS copies wholesale.

## Layout

```
src/pms/               # Python package
  actuator/            # risk + executor + feedback adapters
  api/                 # FastAPI app + `pms-api` CLI entry
  controller/          # decision pipeline (forecaster, sizer, planner)
  core/                # frozen dataclasses, enums, Protocol interfaces
  evaluation/          # metrics collector + eval spool + feedback engine
  market_selection/    # active-perception selector + subscription controller
  sensor/              # HistoricalSensor + MarketDiscoverySensor + stream
  storage/             # JSONL stores + Postgres market-data persistence
  runner.py            # orchestrator wiring all four layers
  config.py            # PMSSettings (pydantic-settings)
dashboard/             # Next.js console (port 3100)
scripts/               # Utility scripts (paper report, migrations, strategy)
tests/                 # pytest suite (unit + integration)
docs/
  operations/          # Runbooks (live-polymarket-runbook.md)
  research/            # Research briefs, FLB analysis, backtest specs
```

## Quick start — backend + dashboard end-to-end

```bash
# 1. Start PostgreSQL for local development
docker compose up -d postgres

# 2. Install Python deps
uv sync

# 3. Point PMS at the compose-created database and apply migrations
export DATABASE_URL=postgres://postgres:postgres@localhost:5432/pms_test
# Before running migrations or installing a canary strategy, verify that
# `DATABASE_URL` points at the intended Postgres server and database; another
# local Postgres on port 5432 can silently catch `localhost` traffic.
psql "$DATABASE_URL" -Atc "select current_database(), inet_server_addr(), inet_server_port(), version();"
uv run alembic upgrade head

# Escape hatch: roll back to the pre-migration state for the current DATABASE_URL
uv run alembic downgrade base

# 4. Start the FastAPI backend (port 8000 by default)
uv run pms-api                       # → http://127.0.0.1:8000
# Optional supervised auto-start; PMS_AUTO_START=1 requires PMS_DISCORD__WEBHOOK_URL
# so startup alerts have an external operator channel.
PMS_AUTO_START=1 \
PMS_DISCORD__WEBHOOK_URL="https://discord.com/api/webhooks/..." \
uv run pms-api
# First-live paper soak: load the tight risk envelope explicitly.
# This starts the control plane; use the Runner lifecycle commands below
# to POST /run/start.
uv run pms-api --config config.live-soak.yaml

# 5. In another shell, start the dashboard (port 3100)
cd dashboard
npm install
PMS_API_BASE_URL=http://127.0.0.1:8000 npm run dev
#   → http://127.0.0.1:3100
```

schema.sql is a reference artifact, not the runtime source; apply runtime
schema changes with Alembic.

If `PMS_API_BASE_URL` is unset outside production, the dashboard falls back to
the bundled mock store (`dashboard/lib/mock-store.ts`) for pure frontend work.
In production, API routes fail closed with HTTP 503 until `PMS_API_BASE_URL`
points at a live backend, so fabricated dashboard data cannot mask a missing
runtime connection.

## Runner lifecycle via the API

```bash
export PMS_API_TOKEN="$(openssl rand -hex 32)"

curl -X POST \
  -H "Authorization: Bearer $PMS_API_TOKEN" \
  http://127.0.0.1:8000/run/start   # begin ingesting signals
curl -H "Authorization: Bearer $PMS_API_TOKEN" \
  http://127.0.0.1:8000/status      # {running: true, ...}
curl -X POST \
  -H "Authorization: Bearer $PMS_API_TOKEN" \
  http://127.0.0.1:8000/run/stop    # graceful shutdown
```

The `Overview` dashboard page includes a **Runner Controls**
panel that calls these endpoints directly.
Changing mode through `POST /config` requires a stopped runner. Changing to
`live` is guarded by the same candidate LIVE validation and credentialed
preflight artifact check as startup.

## MCP proxy for Claude Code and Codex

The optional `agent` extra installs the official MCP Python SDK and exposes a
stdio server that forwards a curated, scoped tool set to the typed PMS HTTP
API. At startup it fetches `/openapi.json`, derives each tool schema from every
mapped operation, and fails closed when an operation is missing or drifts; the
Wave 2B contribution tools are feature-detected from that document. The proxy has no trading logic and
cannot start LIVE, switch mode, accept a live decision, clear a halt, or
resolve an incident.

```bash
uv sync --extra agent
export PMS_AGENT_API_BASE_URL=http://127.0.0.1:8000
export PMS_AGENT_TOKEN='<configured agent token>'
uv run pms-mcp
```

Claude Code can launch it from `.mcp.json` (keep the token in the process
environment rather than checking it into the file):

```json
{
  "mcpServers": {
    "pms": {
      "command": "uv",
      "args": ["--directory", "/path/to/prediction-market-system", "run", "--extra", "agent", "pms-mcp"],
      "env": {
        "PMS_AGENT_API_BASE_URL": "http://127.0.0.1:8000"
      }
    }
  }
}
```

For Codex, add the same command as a local MCP server in its MCP settings and
set `PMS_AGENT_API_BASE_URL` and `PMS_AGENT_TOKEN` in the environment used by
Codex.

CI should exercise the optional SDK path explicitly:

```bash
uv sync --extra agent
uv run pytest -q tests/unit/test_agent_mcp.py
```

## Development

```bash
uv sync
uv run pytest -q                              # full default suite
docker compose up -d postgres
export PMS_TEST_DATABASE_URL=postgres://postgres:postgres@localhost:5432/pms_test
PMS_RUN_INTEGRATION=1 uv run pytest -q -m integration   # DB-backed integration + gated live-network tests
uv run mypy src/ tests/ --strict              # strict type check
uv run lint-imports                           # import-linter contracts
(cd dashboard && npm ci && npm run test:ci)   # dashboard Vitest
```

Baseline invariants enforced by CI:
- pytest default suite stays green; integration checks are gated on
  `PMS_RUN_INTEGRATION=1`, with PostgreSQL-backed checks requiring
  `PMS_TEST_DATABASE_URL`.
- mypy strict must be clean on every committed source file.
- import-linter contracts must keep architecture boundaries intact.
- dashboard Vitest must stay green.
- Research sweep and worker spec format: `docs/research/backtest-spec-format.md`

### Isolating dev state

Runtime state now lives in PostgreSQL. Use a per-shell database name so
parallel sessions do not share evaluator or feedback rows:

```bash
export DATABASE_URL=postgres://localhost/pms_dev_$(whoami)
uv run pms-api
```

Legacy `.data/*.jsonl` files are no longer part of the runtime contract.
If you need to preserve old local rows, migrate them once with:

```bash
python scripts/migrate_jsonl_to_pg.py --data-dir .data --database-url "$DATABASE_URL"
```

See `CLAUDE.md` for the active engineering rules promoted from retros.
