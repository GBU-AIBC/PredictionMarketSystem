# Wave 3 end-to-end acceptance — 2026-10-18

**Acceptance correction:** the scored backtest below is historical, invalidated evidence. Follow-up review found missing YES/NO token metadata in replay, so a NO-token quote could be forecast as YES before side selection. After fixing token orientation and book reference pricing, the same preserved-data window produces **0 decisions and 0 fills**. A follow-up single-market smoke has now persisted an agent-factor decision and fill through the real PAPER runner. Wave 3 is not fully accepted: a correctly scored real-data backtest remains outstanding. See the follow-up section below.

Target: `/Users/stometa/dev/pms-w3`, branch `feat/w3-e2e-acceptance`, ref `c12b139` (rebased onto `feat/agent-framework` at `43d74fd`, including lane 2C). No push was performed. Local scratch files are under the ignored `.data/w3-e2e/`; `config.local.paper-canary.yaml` is ignored by `config.local*.yaml`.

The acceptance used PostgreSQL `127.0.0.1:55432`, database `pms_e2e`, and never used port 5432. `uv run alembic upgrade head` advanced the preserved data from `0025_merge_wave1_heads` through `0027_merge_wave2_heads`; the existing market/book data remained available. A fresh empty `pms_w3` database was migrated for integration checks. The API was bound only to `127.0.0.1:8766`. Test bearer values were generated in the shell and are omitted here.

## Results

| Step | Result | Evidence |
| --- | --- | --- |
| 1. Gates and fresh PostgreSQL integration | **PASS** | `3056 passed, 219 skipped`; strict mypy clean on 503 files; import-linter 9 kept/0 broken; integration `217 passed, 3 skipped` |
| 2. PAPER canary | **PASS** | Real Gamma discovery, `/readiness=ready`, 2 sensor layers, 2,058 markets, 53 decisions, 49 trades, 45 open positions; `check_paper_canary_smoke.py` all checks PASS |
| 3. Agent API and least privilege | **PASS by design** | A token with paper_operate is intentionally allowed to start PAPER (the 2A contract); a read+research-only token returned 403 on /run/start. OpenAPI, evidence/factor writes, risk-check no-mutation, pending approvals, LIVE/config/halt/reconciliation denials, and kill/restart behavior all pass. |
| 4. Evidence reaches forecaster/replay input | **PASS (test-backed)** | `uv run pytest -q tests/integration/test_agent_contributions_w2b.py::test_posted_evidence_changes_paper_llm_prompt_but_not_before_as_of tests/integration/test_agent_contributions_w2b.py::test_paper_evidence_replays_through_backtest_factory_without_provider` → `2 passed in 1.17s`; this exercises PIT evidence and no-provider replay. |
| 5. Resolution backfill and sweep | **NOT ACCEPTED (old score invalidated)** | Backfill retained requested_market_count=929, resolved_market_count=169. The old scored run used a misoriented NO quote. Corrected replay of the same window yields decision_count=0, fill_count=0; Brier/PnL remain null. |
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

A factor observation was visible to the PIT reader in SQL and in the factor series. The pinned market selection returned reasons=[user_pin, selected]. With yes_price=0.905 and agent_probability=0.8, the real runner recorded decision_probability=0.9084126870, gross_edge=0.0034125024, and net_edge_after_costs=-0.0177174976 after spread=0.009955, fee=0.00665, and slippage=0.004525; decision_net_edge_not_positive was the correct blocking gate. A direct pipeline replay of the same signal with agent_probability=0.99 emitted a decision with selected_probability=0.9935689637 and net_edge_after_costs=0.0334489637 (factor snapshot hash 620f6ac8…). No provider call occurred. The live runner did not persist a paper fill for this market, so that run did not close the agent-path gap. The follow-up PAPER smoke below supersedes that residual.

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

Historical scored command output (invalidated; do not use as acceptance evidence):

```text
$ uv run pms-research sweep .data/w3-e2e/gapB-sweep.yaml --database-url <local pms_e2e DSN> --no-cache
{cache_hit_rate:0.0, run_ids:[ab350871-d5f0-4542-b81b-470945824531], inserted:true}
$ scored worker (BacktestRunner, evaluation_as_of=2026-10-18T04:01:00Z)
{runner_succeeded:true, status:completed, evaluation_report_id:caada8ae-7f7c-4348-b3f7-f231cbba203d}
{strategy_id:paper_multi_factor_v1, decision_count:1, fill_count:1, brier:0.999998000001, pnl_cum:-1.00998, fill_rate:1.0}
```

Spec: coverage 2026-10-17T18:34:00Z..2026-10-17T23:23:00Z; market_universe_filter=[0xce2c3d408a3ffe6e625885b415af3ecf65eeba64f23ffa90fc141d6dda97a386]; fee_rate=0.01; slippage_bps=0.0; latency_ms=0.0; staleness_ms=60000.0; fill_policy=immediate_or_cancel; risk max_notional=100.0, max_drawdown=50.0, min_order=1.0; parameter_grid slippage_bps=[0.0]. The full YAML is preserved at ignored .data/w3-e2e/gapB-sweep.yaml.
The evaluation_as_of cutoff is after Gamma resolved_at=2026-10-18T03:59:00Z; the normal current-time worker correctly withheld that future label. The backtest-only extreme-price allowance and execution-price guard are covered by tests/unit/test_controller_router.py::test_backtest_router_allows_resolved_price_extremes; 78 emissions were observed from 147 replay signals and 51 boundary signals were dropped safely.

## Code and safety references

- Agent evidence is scoped to `paper_operate` or `research` and rejects LIVE writes in `src/pms/api/routes/agent_evidence.py:107-119`.
- Agent factor provenance is persisted separately and returned by `src/pms/api/routes/agent_factors.py:236-273`.
- Dynamic mode and kill authorization are covered by `tests/unit/test_api_agent_contract.py:65-112` and the generated route matrix at `tests/unit/test_api_agent_contract.py:350-440`.
- Read-only risk/explain behavior is covered by `tests/unit/test_api_agent_contract.py:639-701`.
- LIVE startup validation is enforced by `src/pms/api/__main__.py:35-76` and `src/pms/config.py:620-770`.

The backtest defects fixed in this run are minimal and regression-tested: BACKTEST now permits replaying resolved boundary quotes while the pipeline drops non-domain execution prices before constructing a TradeDecision; PAPER/LIVE router price gates remain unchanged. The original agent runner rejection is documented with the exact real-data edge calculation above; no source bypass was added. MCP is verified after the lane 2C rebase. No real order, venue credential, LLM provider call, or tracked secret was used.

Focused regression verification after the fix: uv run pytest -q tests/unit/test_controller_router.py tests/unit/test_controller_cp02.py -> 51 passed in 0.61s; strict mypy on the touched controller/tests files -> Success: no issues found in 23 source files.

## Follow-up correctness review

Checkout: `/Users/stometa/dev/pms-agent-framework`, branch `feat/agent-framework`, follow-up to `88197e5`. No push or real-money order was performed.

- Resolution labels must commit before their fills are scored/resolved. A failed market-label write leaves the fill discoverable on the next sweep; the failure also increments runner health counters. Source: `src/pms/evaluation/resolution.py:196` and `src/pms/runner.py:2370`. Regression: `test_resolution_persistence_failure_keeps_fill_retryable`.
- GTC orders retain their market deadline and cancel at or after it, including when subsequent events come from another market. Already filled quantity is preserved. The backtest report records the terminal reason. Source: `src/pms/research/execution.py:273` and `src/pms/research/runner.py:673`. Regressions: `test_gtc_cancels_at_resolution_without_filling_remainder` and `test_backtest_runner_records_resolution_cancellation_for_gtc_order`.
- New LLM results identify the complete provider prompt, retaining that original hash through TTL-cache reuse and durable replay. Legacy replay rows without this provenance retain the old system-only hash; their original dynamic prompt cannot be reconstructed reliably. Source: `src/pms/controller/forecasters/llm.py:304` and `src/pms/controller/forecasters/llm.py:451`. Regression: `test_llm_prompt_provenance_survives_cache_and_replay`.
- Replay book snapshots prefer the bid/ask midpoint, matching the real sensor. Without a midpoint or previous trade, the unknown reference is non-finite and rejected by the controller rather than interpreted as zero; execution can still read the original book. Replay also carries YES/NO token IDs from PostgreSQL or explicit JSONL metadata (including token identity first supplied on a later row), allowing canonicalization before forecasts. Source: `src/pms/research/replay.py:283`, `src/pms/research/replay.py:699`, and `src/pms/research/replay.py:769`. Regressions: `test_book_replay_uses_midpoint_before_last_trade`, `test_book_replay_missing_reference_is_not_a_zero_probability`, `test_no_token_replay_is_canonicalized_before_forecasting`, and the PostgreSQL replay integration suite.

The preserved `pms_e2e` database was read with `default_transaction_read_only=on`. The rerun used the original `.data/w3-e2e/gapB-sweep.yaml`, its slippage grid value `0.0`, the stored strategy version, actual `ControllerPipelineFactory`, `MarketUniverseReplayEngine`, and `BacktestRunner._run_strategy`. No synthetic factor observations or future evaluation clock were supplied. At `2026-10-18T04:08:43.035674+00:00` (after the stored resolution timestamp), output was:

```json
{"decision_count":0,"fill_count":0,"brier":null,"pnl_cum":null,"emissions":[],"rows":[]}
```

Before token metadata was restored, the witness emitted NO at `0.002` with probability `0.999999` after treating the incoming NO price `0.002` as a YES reference. The same false fill disappears after the correction. The stored resolution also has `resolved_at=2026-10-18T03:59:00Z`; the earlier current-time rerun correctly withheld that future label. The old manual `evaluation_as_of=04:01Z` result is not a current-time acceptance run.

### Factor panel orientation and version boundary

Following the replay fix, review traced the same coordinate error into `FactorService`: it consumes raw sensor quotes (`src/pms/runner.py:1437`), so NO=0.001 previously persisted `favorite_longshot_bias=-0.099` and a NO ask-only book persisted imbalance=-1. The service now converts the price to YES before computation; `OrderbookImbalance` itself orients signed depth to YES, and the controller no longer applies a second sign flip. Token IDs take precedence over the outcome marker. Regression: `tests/unit/test_factor_service.py::test_factor_service_no_quote_persists_yes_coordinate` covers both metadata forms, expects +0.099/+1, and verifies the raw signal is unchanged.

`favorite_longshot_bias`, `orderbook_imbalance`, `fair_value_spread`, and `anchoring_lag_divergence` now use definition version `2`. Existing factor readers filter by this version, preventing reuse of possibly contaminated version-1 observations. No historical row was rewritten or deleted. Strategies using these factors need a newly registered version and fresh/recomputed factor observations; historical scores must not be carried forward as validated. This is another reason the preserved backtest window is not acceptance evidence. The PAPER smoke below was repeated successfully after this version change in a fresh database.

A clean recomputation was also attempted in `pms_scored_followup_20261018`, separate from both the source data and test database. A read-only source query found only one resolved market with captured books. Copied rows: 1 market, 2 tokens, 1 resolution, 136 snapshots, 7828 levels, 0 price changes and 11 trades. `FactorService` recomputed 152 observations with the corrected code. The unmodified current `build_paper_multi_factor_strategy()` produced version `b260560a9a407a1094393c477250dad80c9c766b3a9d587e67900e15baf8a5d6`; its configured calibration and market-selection gates were preserved. At actual evaluation time `2026-10-18T04:10:37.088921+00:00`, the real backtest runner again produced 0 decisions / 0 fills / null Brier / null PnL. Raw output: `/tmp/pms-scored-followup-result.json`. A qualifying resolved-market sample remains necessary for scored acceptance; no gates or factor values were changed to manufacture a trade.

### Agent factor through the PAPER runner

The isolated database `pms_paper_followup_v2_20261018` was migrated to head `0027`. A test harness fetched a fresh public CLOB book, passed it through `MarketDataSensor._handle_book`, and queued the resulting signal into the actual `Runner` with its PostgreSQL registry, factor reader, controller, risk checks, PAPER actuator and persistence. This was a single-market transport smoke, not another discovery/WebSocket soak.

The harness used the existing `paper_agent_probability_v1` builder with only `price_reference=best_ask` added in a local immutable strategy version (`9a8ecc6e35916e684a231ad01fad7228623a5ba2d177f151ce34455ba3a149ee`). The default midpoint-limited version correctly failed liquidity checks at `0.275`; the local variant could buy at the real ask `0.28`. Risk sizing remained 1 USDC and slippage remained 50 bps. PAPER smoke freshness/spread limits were 60 seconds / 10000 bps as in the earlier canary; production configuration was not changed.

An ephemeral read+research token submitted the explicit test forecast `agent_probability=0.8` through the real ASGI API. LLM providers were disabled. Public quotes were real; the submitted forecast was an acceptance input, not a claim of predictive accuracy. The API did not authorize that token to start the runner; the harness started PAPER as the operator.

```text
$ uv run python /tmp/pms_paper_followup.py
factor_http_status: 201
research_start_status: 403
market_id: 0x7586a96520578acaaaa4ea84a2582f197f84255da1f3392a7aa300386c187b37
quote_timestamp: 2026-10-18T04:06:35.330000+00:00
quote_midpoint: 0.275
decision_id: decision-0412072603da4440bb09da17051e48e1
decision_status: filled
limit_price: 0.28
prob_estimate: 0.8166030174875702
fill_id: paper-f28343fafea948b09789217630d73533
fill_notional_usdc: 1.0
fill_quantity: 3.5714285714285716
```

SQL confirmed the decision payload and opportunity composition trace share factor snapshot `edaba7413114fbba14953f18d1a85a53e5173b5844a516d401bb52b3e2843b0b`, with `agent_probability` present and no missing factors. Local raw output is `/tmp/pms-paper-followup-result.json`. The harness stopped the runner and closed its pools in `finally`; no LIVE order or private venue credentials were used. This closes the agent-contribution plumbing gap, but this still-open market cannot supply a resolved backtest score.

### Verification

Final code verification (follow-up checkout; dashboard source was unchanged):

```text
$ env -u DATABASE_URL uv run pytest -q -p no:cacheprovider
3100 passed, 227 skipped in 32.50s
$ uv run mypy src/ tests/ --strict
Success: no issues found in 507 source files
$ uv run lint-imports
Contracts: 10 kept, 0 broken.
$ PMS_RUN_INTEGRATION=1 PMS_TEST_DATABASE_URL=<isolated pms_followup_20261018 DSN> DATABASE_URL=<same DSN> uv run pytest -q -m integration -p no:cacheprovider
221 passed, 4 skipped, 3102 deselected in 108.83s (0:01:48)
$ cd dashboard && npm ci && npm run test:ci
Test Files 26 passed (26)
Tests 65 passed (65)
```

### Review receipt

Grok completed two read-only reviews through Orca: replay investigation (`task_c782c395bc1a`, dispatch `ctx_2f4ba600c7cb`, report `/tmp/pms-grok-followup-review.md`) and final patch review (`task_559ffe7da6fd`, dispatch `ctx_affb38f64602`, report `/tmp/pms-grok-final-followup-review.md`). The first review confirmed the false NO-token acceptance fill and the missing-reference/token-metadata fixes. The final review reported: “No reproducible actionable regression in these patches.” Its final scope covered resolution retry, GTC cancellation, prompt provenance and factor orientation; it did not independently rerun the full test gates or rereview the final JSONL follow-up. The coordinator reproduced and tested the latter and executed the gates recorded above. No Opus or Sonnet reviewer was used in this follow-up.

## Ponytail integration appendix — 2026-10-18

Integrated source: `feat/agent-framework` at `6efc246e931e19b3a4eaf1a648f564d04e4b9d7a`. This round bounds per-strategy pending queues (`src/pms/runner.py:316`) and research replay chunks (`src/pms/research/replay.py:118`), uses historical outcome books in BACKTEST (`src/pms/research/runner.py:549`), makes new strategy sharing opt-in (`alembic/versions/0028_strategy_share_opt_in.py:19`), and removes the unused Rust scaffold (`06cbe23`). These are bounded-buffer changes, not evidence of bounded total memory or a completed load soak.

Migration `0029_controller_rejections` adds `strategy_runs.controller_rejection_counts_json JSONB NOT NULL DEFAULT '{}'::jsonb` after `0028`; downgrade discards that output column (`alembic/versions/0029_controller_rejections.py:8`, `:14`, `:21`). Rejections are counted separately from research inputs and persisted with strategy results (`src/pms/research/runner.py:713`, `:925`); the API exposes the decoded `controller_rejection_counts` object (`src/pms/api/research_routes.py:147`).

Supplied coordinator gate logs, before this integration-test correction:

| Check | Actual result | Local evidence |
| --- | --- | --- |
| Default pytest | 3135 passed, 227 skipped in 42.83s | `/tmp/pms-ponytail-pytest.log` |
| Strict mypy | No issues in 509 source files | `/tmp/pms-ponytail-mypy.log` |
| Import contracts | 10 kept, 0 broken | `/tmp/pms-ponytail-imports.log` |
| Dashboard Vitest | 27 files / 66 tests passed | `/tmp/pms-ponytail-dashboard.log` |
| Dashboard types | `tsc --noEmit`, no diagnostics | `/tmp/pms-ponytail-dashboard-types.log` |
| Full PostgreSQL integration | **1 failed, 224 passed, 3 skipped**, 3134 deselected in 186.26s | `/tmp/pms-ponytail-integration.log` |

The sole full-integration failure was the old exact payload assertion at `tests/integration/test_research_api_cp06b.py:178` in the supplied run, which omitted the intended new output field. The isolated correction checkout is `/Users/stometa/orca/workspaces/prediction-market-system/fix-ponytail-integration`, branch `stone16/fix-ponytail-integration`, based on the exact source ref above. The exact response assertion now covers both empty and populated persisted counts (`tests/integration/test_research_api_cp06b.py:153`). A separate test uses existing synthetic replay fixtures with the real controller and `BacktestRunner.execute`, then verifies PostgreSQL and HTTP results: two low-volume rejections, one closed-market rejection, completed status, zero opportunities/decisions/fills/execution rows, and null Brier/PnL (`tests/integration/test_research_api_cp06b.py:219`). No production source changed in this correction.

The test database `pms_ponytail_api_20261018` was newly created on the confirmed `pms-agent-fw-pg` PostgreSQL 16 container, mapped to `127.0.0.1:55432`; existing integration fixtures migrated it to `0029`. No shared/source database or host port 5432 was used. The original API failure was reproduced before editing (`/tmp/pms-ponytail-api-before.log`: 1 failed in 5.76s). Focused verification: `PMS_RUN_INTEGRATION=1 PMS_TEST_DATABASE_URL=<isolated DSN> DATABASE_URL=<same DSN> uv run pytest -q tests/integration/test_research_api_cp06b.py -p no:cacheprovider` → **4 passed in 2.73s** (`/tmp/pms-ponytail-api-focused.log`); `uv run mypy src/ tests/ --strict` → **no issues in 509 source files** (`/tmp/pms-ponytail-api-mypy.log`). The relevant group, `uv run pytest -q tests/integration/test_research_*.py tests/integration/test_schema_apply_research_cp04a.py -p no:cacheprovider`, with the same isolated environment, passed **36 tests in 19.39s** (`/tmp/pms-ponytail-api-integration.log`). The coordinator owns the subsequent full integration rerun; no corrected full-suite pass is claimed here.

The supplied real-data probe read `pms_e2e` with `default_transaction_read_only=on`, copied one resolved market into isolated `pms_ponytail_scored_20261018`, and recomputed 152 factors from 2 tokens, 1 resolution, 136 snapshots, 7828 levels, 0 price changes and 11 trades (`/tmp/pms_ponytail_scored.py:23`, `:30`, `:45`). At actual evaluation time `2026-10-18T05:59:04.990453+00:00`, the unchanged multi-factor strategy version `b260560a9a407a1094393c477250dad80c9c766b3a9d587e67900e15baf8a5d6` produced **0 decisions, 0 fills, null Brier and null PnL**, with **57 `router_gate:yes_price_out_of_band`** and **90 `calibration_clamp_rejected`** (`/tmp/pms-ponytail-scored-result.json`). That probe calls `_run_strategy` and serializes its accumulator; it does not prove a persisted `execute`/API roundtrip (`/tmp/pms_ponytail_scored.py:51`). The synthetic PostgreSQL/API test above supplies that separate persistence evidence. These diagnostics provide no evidence to loosen the economic gates; scored real-data acceptance remains outstanding.

Independent **gpt-6-astra approval is still pending**. This correction made no LIVE/order/provider calls and performed no push, PR edit, or main-branch commit.
