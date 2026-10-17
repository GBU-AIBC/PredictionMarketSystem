# Wave 3 end-to-end acceptance — 2026-10-18

Target: `/Users/stometa/dev/pms-w3`, branch `feat/w3-e2e-acceptance`, ref `c12b139` (rebased onto `feat/agent-framework` at `43d74fd`, including lane 2C). No push was performed. Local scratch files are under the ignored `.data/w3-e2e/`; `config.local.paper-canary.yaml` is ignored by `config.local*.yaml`.

The acceptance used PostgreSQL `127.0.0.1:55432`, database `pms_e2e`, and never used port 5432. `uv run alembic upgrade head` advanced the preserved data from `0025_merge_wave1_heads` through `0027_merge_wave2_heads`; the existing market/book data remained available. A fresh empty `pms_w3` database was migrated for integration checks. The API was bound only to `127.0.0.1:8766`. Test bearer values were generated in the shell and are omitted here.

## Results

| Step | Result | Evidence |
| --- | --- | --- |
| 1. Gates and fresh PostgreSQL integration | **PASS** | `3056 passed, 219 skipped`; strict mypy clean on 503 files; import-linter 9 kept/0 broken; integration `217 passed, 3 skipped` |
| 2. PAPER canary | **PASS** | Real Gamma discovery, `/readiness=ready`, 2 sensor layers, 2,058 markets, 53 decisions, 49 trades, 45 open positions; `check_paper_canary_smoke.py` all checks PASS |
| 3. Agent API and least privilege | **PARTIAL** | OpenAPI and read-only endpoints pass; evidence/factor writes return 201; risk-check returns 200 with no domain-state mutation; kill token returns 200 and killer restart returns 403. The agent token receives 200 on PAPER `/run/start`, matching the existing 2A matrix tests (`tests/unit/test_api_agent_contract.py:350-440`) but conflicting with the Wave 3 wording requiring 403. No LIVE accept was attempted because LIVE correctly cannot start without credentials. |
| 4. Evidence reaches forecaster/replay input | **PASS (test-backed)** | `uv run pytest -q tests/integration/test_agent_contributions_w2b.py::test_posted_evidence_changes_paper_llm_prompt_but_not_before_as_of tests/integration/test_agent_contributions_w2b.py::test_paper_evidence_replays_through_backtest_factory_without_provider` → `2 passed in 1.17s`; this exercises PIT evidence and no-provider replay. |
| 5. Resolution backfill and sweep | **PARTIAL** | Gamma backfill: `requested_market_count=929`, `resolved_market_count=169`. A queued worker completed a sweep, but the collected window's available strategy/data combination produced null Brier/PnL and no benchmark rows. A PAPER-only canary sweep is rejected in normal BACKTEST mode (`paper_canary forecaster is PAPER-only`); the default registry strategy has required `factor-a`/`factor-b` inputs absent from this collected market. This is an acceptance gap, not reported as a successful non-null report. |
| 6. LIVE fail-closed | **PASS** | `pms-api --config config.live.yaml.example` exits 1: `Local secret file contains placeholder Polymarket credential fields: api_key, api_passphrase, api_secret, funder_address, private_key`; `pms-live preflight --json` returns `ok:false` with the same reason. No order was sent. Agent `/config {mode:live}` and `/live/halt/clear` both return 403. |
| 7. MCP | **PASS** | Rebasing onto `feat/agent-framework` at `43d74fd`; `uv sync --extra agent` installed `mcp==1.26.0`; stdio `tools/list` returned 17 tools, and live MCP proxy calls to status/markets/selection/evidence/risk-check succeeded. |

## Commands and real outputs

### Gates

```text
$ uv run alembic heads
0027_merge_wave2_heads (w2a, w2b, w1b, w1c) (head)
$ uv run alembic upgrade head
... 0025_merge_wave1_heads -> 0026_agent_contrib_w2b
... 0025_merge_wave1_heads -> 0026_request_audit_w2a
... 0026_request_audit_w2a, 0026_agent_contrib_w2b -> 0027_merge_wave2_heads
$ uv run pytest -q
3056 passed, 219 skipped in 29.47s
$ uv run mypy src/ tests/ --strict
Success: no issues found in 503 source files
$ uv run lint-imports
Analyzed 251 files, 1611 dependencies.
Contracts: 9 kept, 0 broken.
$ PMS_RUN_INTEGRATION=1 DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/pms_w3 uv run pytest -q -m integration
217 passed, 3 skipped, 3055 deselected in 135.43s (0:02:15)
```

### PAPER canary

```text
$ uv run python scripts/prepare_local_paper_soak_config.py --paper-canary --output config.local.paper-canary.yaml --overwrite
local paper-soak config written: config.local.paper-canary.yaml
paper-canary plumbing config does not require launch artifacts
$ uv run python scripts/install_paper_canary_strategy.py --database-url <local pms_e2e DSN> --archive-default --sample-modulus 1
strategy_id: paper_canary_v1
strategy_version_id: 3fd5644e9ff9edc867ef547ed58cd223ee8a870b75b2b85c7c18478a7b5816dd
archived_default: true
sample_modulus: 1
$ POST /run/start (legacy operator token)
{"status":"started","mode":"paper","runner_started_at":"2026-10-17T23:37:29.032574+00:00"}
$ uv run python scripts/check_paper_canary_smoke.py ...
[PASS] paper_mode
[PASS] readiness
[PASS] runtime_continuity
[PASS] sensor_activity
[PASS] active_strategy: paper_canary_v1@3fd5644e...
[PASS] market_discovery: observed_markets=2058
[PASS] controller_decisions: decisions_total=53
[PASS] paper_trades: fills_total=49
[PASS] open_positions: open_positions=45
[PASS] selection_funnel: controller_emitted_total=50; traded_total=44
[PASS] first_trade_time: pms.ui.first_trade_time_seconds=0.032146
```

The first canary snapshot with the committed tight spread limit reported `controller_emitted_total=0`; widening only the ignored local canary config's paper quote freshness/spread gates (`max_book_age_ms=60000`, `max_spread_bps=10000`) allowed the real market data to exercise the paper actuator. This is test configuration, not a source change.

### Agent path

The agent token successfully called `GET /openapi.json`; the captured schema contained 47 operations and zero operations missing tags/responses. `GET /selection/explain` returned `reasons=["selected"]` for a subscribed market and `reasons=["resolution_horizon","cap"]` for a non-subscribed market. `POST /evidence` returned `201` with `submitted_by_token_id="agent"`; `POST /factors/agent_probability/observations` returned `201` with `provenance_written=true`.

`POST /decisions/{id}/risk-check` returned:

```json
{"decision_id":"exit-stop_loss-…","approved":false,"reason":"decision_filled"}
```

The request added only its own `api_request_audit` row; decision/order/fill domain counts were unchanged. `GET /approvals/pending` returned `{"approvals":[]}`.

The token matrix probes returned 403 for `/config` LIVE, `/live/reconcile-submission-unknown`, and `/live/halt/clear`. The kill token returned `{"status":"killed"}` from `/run/kill`; the same kill token then received 403 from `/run/start`.

A factor observation was visible to the PIT reader in SQL and in the factor series. No subsequent `paper_agent_probability_v1` decision appeared during the live window: controller diagnostics were dominated by `decision_net_edge_not_positive`, and the strategy's only current runtime generated zero agent decisions. The existing unit/integration coverage proves the contribution path, but this machine run does not claim the requested next-decision mutation.

### MCP

After the coordinator notified that lane 2C landed, the branch was rebased onto `feat/agent-framework` (`43d74fd`) and the optional dependency was installed:

```text
$ uv sync --extra agent
... + mcp==1.26.0 ...
$ PMS_AGENT_API_BASE_URL=http://127.0.0.1:8766 PMS_AGENT_TOKEN=<local> uv run pms-mcp
initialize: serverInfo.name=pms-mcp, protocolVersion=2025-06-18
tools/list: 17 tools (get_status, list_markets, explain_selection, list_signals, get_factor_catalog, get_factor_series, submit_evidence, submit_agent_factor, run_backtest, get_research_report, get_evaluation_report, list_evaluation_reports, explain_decision, risk_check_decision, list_pending_approvals, list_positions, kill_switch)
get_status: mode=paper, running=true, healthy=true
list_markets: total=2074, rows=1
explain_selection: available=true, reasons=[resolution_horizon]
submit_evidence: 201, submitted_by_token_id=agent
risk_check_decision: approved=false, reason=decision_filled
```

The static MCP tool list contains no LIVE start, mode switch, LIVE accept, halt-clear, or reconciliation operation. `kill_switch` is the only control tool and maps to the kill-scoped `/run/kill` endpoint.

### Resolution and research

```text
$ uv run pms-research resolution-backfill --database-url <local pms_e2e DSN> --start 2026-06-03T00:00:00Z --end 2026-10-18T00:00:00Z
{"requested_market_count":929,"resolved_market_count":169}
$ uv run pms-research sweep .data/w3-e2e/sweep-spec.yaml --database-url <local pms_e2e DSN> --wait --no-cache
... default/default-v1 ... "status":"completed"
```

The resulting `strategy_runs` row had `decision_count=0`, `fill_count=0`, `brier=NULL`, and `pnl_cum=NULL`; `evaluation_reports.benchmark_rows=[]`. The sweep spec used the collected window and is preserved at ignored `.data/w3-e2e/sweep-spec.yaml`. A canary retry explicitly failed with `paper_canary forecaster is PAPER-only` under normal BACKTEST settings. This must be resolved by providing a backtest-compatible strategy and resolved markets with replayable signals before claiming the step.

## Code and safety references

- Agent evidence is scoped to `paper_operate` or `research` and rejects LIVE writes in `src/pms/api/routes/agent_evidence.py:107-119`.
- Agent factor provenance is persisted separately and returned by `src/pms/api/routes/agent_factors.py:236-273`.
- Dynamic mode and kill authorization are covered by `tests/unit/test_api_agent_contract.py:65-112` and the generated route matrix at `tests/unit/test_api_agent_contract.py:350-440`.
- Read-only risk/explain behavior is covered by `tests/unit/test_api_agent_contract.py:639-701`.
- LIVE startup validation is enforced by `src/pms/api/__main__.py:35-76` and `src/pms/config.py:620-770`.

No source defect was safely fixed during this run. The two residual acceptance gaps are the agent decision mutation in this real snapshot window and the research report's null Brier/PnL. MCP is now verified after the lane 2C rebase. No real order, venue credential, LLM provider call, or tracked secret was used.
