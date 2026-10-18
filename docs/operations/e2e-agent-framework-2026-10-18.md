# Wave 3 end-to-end acceptance — 2026-10-18

Target: `/Users/stometa/dev/pms-w3`, branch `feat/w3-e2e-acceptance`, ref `c12b139` (rebased onto `feat/agent-framework` at `43d74fd`, including lane 2C). No push was performed. Local scratch files are under the ignored `.data/w3-e2e/`; `config.local.paper-canary.yaml` is ignored by `config.local*.yaml`.

The acceptance used PostgreSQL `127.0.0.1:55432`, database `pms_e2e`, and never used port 5432. `uv run alembic upgrade head` advanced the preserved data from `0025_merge_wave1_heads` through `0027_merge_wave2_heads`; the existing market/book data remained available. A fresh empty `pms_w3` database was migrated for integration checks. The API was bound only to `127.0.0.1:8766`. Test bearer values were generated in the shell and are omitted here.

## Results

| Step | Result | Evidence |
| --- | --- | --- |
| 1. Gates and fresh PostgreSQL integration | **PASS** | `3056 passed, 219 skipped`; strict mypy clean on 503 files; import-linter 9 kept/0 broken; integration `217 passed, 3 skipped` |
| 2. PAPER canary | **PASS** | Real Gamma discovery, `/readiness=ready`, 2 sensor layers, 2,058 markets, 53 decisions, 49 trades, 45 open positions; `check_paper_canary_smoke.py` all checks PASS |
| 3. Agent API and least privilege | **PASS by design** | A token with paper_operate is intentionally allowed to start PAPER (the 2A contract); a read+research-only token returned 403 on /run/start. OpenAPI, evidence/factor writes, risk-check no-mutation, pending approvals, LIVE/config/halt/reconciliation denials, and kill/restart behavior all pass. |
| 4. Evidence reaches forecaster/replay input | **PASS (test-backed)** | `uv run pytest -q tests/integration/test_agent_contributions_w2b.py::test_posted_evidence_changes_paper_llm_prompt_but_not_before_as_of tests/integration/test_agent_contributions_w2b.py::test_paper_evidence_replays_through_backtest_factory_without_provider` → `2 passed in 1.17s`; this exercises PIT evidence and no-provider replay. |
| 5. Resolution backfill and sweep | **PASS (scored rerun)** | Gamma backfill retained requested_market_count=929, resolved_market_count=169. A queued paper_multi_factor_v1 sweep over the sole market with replayable factors and a resolution completed with decision_count=1, fill_count=1, brier=0.999998000001, and pnl_cum=-1.00998; the evaluation report contains a non-null benchmark row and records unresolved markets outside the filtered universe. |
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

Read+research-only probe: POST /run/start with the research token returned HTTP 403 {"detail":"API token does not have the required scope."}. The token matrix probes returned 403 for `/config` LIVE, `/live/reconcile-submission-unknown`, and `/live/halt/clear`. The kill token returned `{"status":"killed"}` from `/run/kill`; the same kill token then received 403 from `/run/start`.

A factor observation was visible to the PIT reader in SQL and in the factor series. The pinned market selection returned reasons=[user_pin, selected]. With yes_price=0.905 and agent_probability=0.8, the real runner recorded decision_probability=0.9084126870, gross_edge=0.0034125024, and net_edge_after_costs=-0.0177174976 after spread=0.009955, fee=0.00665, and slippage=0.004525; decision_net_edge_not_positive was the correct blocking gate. A direct pipeline replay of the same signal with agent_probability=0.99 emitted a decision with selected_probability=0.9935689637 and net_edge_after_costs=0.0334489637 (factor snapshot hash 620f6ac8…). No provider call occurred. The live runner did not persist a paper fill for this market, so this remains a documented residual.

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

The initial default and canary attempts remain negative controls (default: decision_count=0; canary rejected as paper_canary forecaster is PAPER-only). The scored rerun used paper_multi_factor_v1@82792c5f4555b36515c2972ecfeb5ebf6cbd74c0aca1b35d6c94f716f8e39131 and the sole market with replayable factors plus a resolution.

Scored command output:
$ uv run pms-research sweep .data/w3-e2e/gapB-sweep.yaml --database-url <local pms_e2e DSN> --no-cache
{cache_hit_rate:0.0, run_ids:[ab350871-d5f0-4542-b81b-470945824531], inserted:true}
$ scored worker (BacktestRunner, evaluation_as_of=2026-10-18T04:01:00Z)
{runner_succeeded:true, status:completed, evaluation_report_id:caada8ae-7f7c-4348-b3f7-f231cbba203d}
{strategy_id:paper_multi_factor_v1, decision_count:1, fill_count:1, brier:0.999998000001, pnl_cum:-1.00998, fill_rate:1.0}

Spec: coverage 2026-10-17T18:34:00Z..2026-10-17T23:23:00Z; market_universe_filter=[0xce2c3d408a3ffe6e625885b415af3ecf65eeba64f23ffa90fc141d6dda97a386]; fee_rate=0.01; slippage_bps=0.0; latency_ms=0.0; staleness_ms=60000.0; fill_policy=immediate_or_cancel; risk max_notional=100.0, max_drawdown=50.0, min_order=1.0; parameter_grid slippage_bps=[0.0]. The full YAML is preserved at ignored .data/w3-e2e/gapB-sweep.yaml.
The evaluation_as_of cutoff is after Gamma resolved_at=2026-10-18T03:59:00Z; the normal current-time worker correctly withheld that future label. The backtest-only extreme-price allowance and execution-price guard are covered by tests/unit/test_controller_router.py::test_backtest_router_allows_resolved_price_extremes; 78 emissions were observed from 147 replay signals and 51 boundary signals were dropped safely.

## Code and safety references

- Agent evidence is scoped to `paper_operate` or `research` and rejects LIVE writes in `src/pms/api/routes/agent_evidence.py:107-119`.
- Agent factor provenance is persisted separately and returned by `src/pms/api/routes/agent_factors.py:236-273`.
- Dynamic mode and kill authorization are covered by `tests/unit/test_api_agent_contract.py:65-112` and the generated route matrix at `tests/unit/test_api_agent_contract.py:350-440`.
- Read-only risk/explain behavior is covered by `tests/unit/test_api_agent_contract.py:639-701`.
- LIVE startup validation is enforced by `src/pms/api/__main__.py:35-76` and `src/pms/config.py:620-770`.

The backtest defects fixed in this run are minimal and regression-tested: BACKTEST now permits replaying resolved boundary quotes while the pipeline drops non-domain execution prices before constructing a TradeDecision; PAPER/LIVE router price gates remain unchanged. The agent runner residual is documented with the exact real-data edge calculation above; no source bypass was added. MCP is verified after the lane 2C rebase. No real order, venue credential, LLM provider call, or tracked secret was used.

Focused regression verification after the fix: uv run pytest -q tests/unit/test_controller_router.py tests/unit/test_controller_cp02.py -> 51 passed in 0.61s; strict mypy on the touched controller/tests files -> Success: no issues found in 23 source files.
