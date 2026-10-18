# Real-data E2E and bounded-load validation — 2026-10-18

The framework has demonstrated real public-data PAPER execution and real-data historical execution/settlement/scoring. These are separate experiments. There is no funded trade, prospective settled PAPER sample, profitability claim, or production-capacity acceptance.

Three supervised `gpt-6.1-sol` workers used Orca run `run_3f874ae7ad99`; independent `gpt-6-astra` review is recorded below. Repository: `GBU-AIBC/PredictionMarketSystem`, integration checkout `/Users/stometa/dev/pms-agent-framework`, branch `feat/agent-framework`, PR #94. Experiment base: `e68cbd216460202e89a316781c45c7626e2678cd`.

## Results and scope

| Experiment | Observable result | What it establishes |
| --- | --- | --- |
| Prospective public-data PAPER | 1,092 real signals, 316 factor rows, one decision/order/simulated fill; actual HTTP lifecycle and retrieval; no resolution yet | Agent contribution, quote ingestion, controller/risk, PAPER execution, persistence and API plumbing |
| Historical conservative control | Eight-market universe, 682 signals, 340 decisions; zero fills; Brier 0.19387063801298116, PnL null | Rejections, strict limit behavior, decision-level scoring and durable research results |
| Historical quoted-depth scenario | Same universe/signals; six decisions/fills in six markets; Brier 0.18621376095712389; modeled-net settlement PnL **-2.99201671957672 USDC** | Nonempty execution and independently recomputable settlement accounting under explicit idealized fill assumptions |
| Synthetic local load | 8,930 quotes persisted in 123.87 seconds, 100 simulated fills, Python peak RSS 88.64 MiB | Short, fixed-cardinality operation and bounded fault/restart probes; not maximum sustainable trading throughput |

The historical sample is availability-selected and retrospective, not blinded or held out. Conservative Brier weights 340 repeated decisions, not 340 independent markets. The quoted scenario has only six samples. Neither score supports a prediction-quality conclusion.

## One prospective sample

Market `0x7b0ac4dbaec820a12f16b7b3e01d5886bb08fef30a47fa9fb552854ed842c960`: “Will the price of Bitcoin be above $78,000 on October 19?” Scheduled end: `2026-10-19T00:00:00Z`.

The acceptance input was an explicitly generated constant probability **0.99**, submitted through the Agent API; it is not an LLM forecast. On a real YES book with bid/ask **0.976/0.978**, the PAPER adapter simulated buying **1.0224948875255624 shares** for **1 USDC** at **0.978**. Real displayed ask size was 55 shares. Decision `decision-3bb70e1b6afc4d32affd47ad65f144ee` and fill `paper-a1328d0e37434e86a8d724fc265c8ff1` share strategy version `40af3e2c0499597d59c410ae63e2165c9b75d156da22e1e1d9f0d4b7bcf190c6`.

The market remained unresolved: realized PnL and Brier are **null**. At the final captured bid 0.976, the separate hypothetical open-position mark was **-0.0035849897750510774 USDC after modeled entry fee**, excluding any exit fee. The HTTP position used a different Gamma mark and reported -0.0025562372188138787 USDC gross; these marks are not interchangeable.

PAPER used fallback fee coefficient 0.07 and charged 0.00154 USDC. A contemporary public fee endpoint returned coefficient 0.10, which would imply 0.0022 USDC under the same formula. This evidence therefore reports modeled costs, not venue-reconciled net returns. Venue event age at reception was 5.286 seconds; the accepted same-token book age was 10.083 ms since local reception. This does not establish exchange-event freshness below one second.

Actual loopback HTTP and public WebSocket transport were used. Default controller/risk gates remained unchanged. Earlier constant-0.5 attempts produced zero fills and are retained: four narrow-universe candidates failed spread, and the broader attempt hit spread/counterpart freshness gates. The separate 0.99 attempt was declared before capture; no winner was used for selection. The explicit sensor selection does not test the complete dynamic market-selection loop.

## Historical score and arithmetic

Eight resolved markets were found by querying 71 already-captured candidates, selected by book availability and recorded deadlines or prior PAPER exposure. All eight enter the denominator. Copied data contains 638 snapshots, 36,893 levels, 44 trades and 16 tokens; source/destination table hashes match. The fixed retrospective rule is:

`p_yes = clip(P_yes - 0.5 * YES_oriented_orderbook_imbalance, 0.01, 0.99)`.

It uses real current/past quote factors, with no outcome input. The first attempted retrospective agent contribution was correctly rejected because its actual submission provenance was newer than the historical decision; that gate was not bypassed.

Both execution scenarios use actual token-specific historical asks, finite displayed depth, IOC, 250 ms modeled latency and fee coefficient 0.07. The conservative control uses 10 bps extra slippage plus 5 bps adverse selection: all orders are unfilled because adjusted cost exceeds their fixed ask limit. The separate idealized scenario sets those two additional impacts to zero, retaining spread, depth, fees, risk and price limits. It assumes full displayed liquidity is available; it is not proof of venue execution.

| Historical market | Bought | Cost | Modeled fee | Payout | Net PnL |
| --- | --- | ---: | ---: | ---: | ---: |
| Bitcoin dip to $80,000, October 17 | YES | 1.00 | 0.06328 | 0 | -1.06328 |
| FC Dallas win, October 17 | NO | 1.00 | 0.02100 | 1.428571428571 | +0.407571428571 |
| US–Iran ceasefire through October 17 | NO | 1.00 | 0.06986 | 0 | -1.06986 |
| NYCFC–LAFC both teams score | NO | 1.00 | 0.04130 | 0 | -1.04130 |
| Houston Dynamo win, October 17 | NO | 1.00 | 0.03220 | 1.851851851852 | +0.819651851852 |
| Vancouver Whitecaps win, October 17 | NO | 1.00 | 0.04480 | 0 | -1.04480 |
| **Total** | | **6.00** | **0.27244** | **3.280423280423** | **-2.992016719577** |

For each row, `net_pnl = shares * token_outcome - cost - fee`, and `Brier = (p_yes - resolved_yes)^2`. First row: 1/0.096 = 10.416666666667 YES shares; outcome 0; fee = 1 × 0.07 × (1−0.096) = 0.06328; PnL = -1.06328. Brier = 0.44239420517312483² = 0.19571263277076087.

Final runs on `d62eae966ee98708359258b3e99c50d9c6e14737`: conservative `cb1fdec8-a004-43b1-964b-6b9cff8ebc01`; quoted-depth `0de6a56b-8818-4354-ab8b-bef8bf2cc37e`. Both execute → PostgreSQL → actual network HTTP completed. Research uses `strategy_runs`, slices and `backtest_execution_rows`; it does not populate production order/fill tables. Labels are scoring-only and were available by the recorded evaluation cutoff. Unknown resolution timestamps use actual observation time, never scheduled end time or an invented historical timestamp.

## Load and failure behavior

The real Runner, sensor persistence, factor service, controller, PAPER actuator and PostgreSQL processed synthetic inputs for 100 markets on a shared Apple M2 Max host. Source 8,930 = controller processed 8,729 + deliberately coalesced 201. Processed 8,729 = emitted 100 + rejected 8,629. SQL stored 8,930 snapshots, 17,860 levels, 17,860 factor rows and 100 fills. The final factor timestamp matched each market's latest book.

Observed throughput was 72.09 quotes/s including drain; only 0.807 fills/s. Most later inputs hit market capacity, so quote throughput is not trading capacity. Event-to-controller p50/p95/p99 was 6.676/19.575/1005.920 ms, including injected one-second stalls. Peak Python RSS was 88.64 MiB, excluding PostgreSQL and other agents. All observed queues drained and tasks/connections closed.

Twenty injected transient factor writes recovered automatically. A decision-store error instead detaches the controller and requires explicit recovery; no uncertain submission is automatically retried. Fixed behavior reports unhealthy/readiness 503, then returns healthy/200 after explicit restart. Shutdown now cancels only still-queued, not-yet-submitted actuator work: the decisive probe retained one in-flight submitted record, cancelled ten queued records and executed zero old fills on restart with zero new source events.

Queue-only BACKTEST probes processed 10k/50k/100k inputs in order; they are not PostgreSQL throughput measurements. Actuator capacity remains unbounded, and the short fixed-cardinality soak does not establish bounded total memory or unattended production readiness.

## Defects fixed during validation

| Finding | Minimal correction | Regression evidence |
| --- | --- | --- |
| Backtest loader dropped stored calibration | Forward the full calibration projection, preserving DB identity/version | Both enabled states and all parameters; real stored projection red/green |
| Scheduled `endDate` treated as label availability | Keep explicit resolution timestamps; otherwise use actual observation | Cold-before-observation excluded; future explicit labels withheld |
| Old actuator work survived stop/restart | Cancel only queued unsubmitted work; release reservations and balance queue accounting | Stop/start failure/cancellation regressions plus actual PostgreSQL witness |
| Failed strategy reported healthy | Include expected dynamic controllers in health and readiness | Held-live sensor failure → 503; restart → 200; zero/removed strategies preserved |
| Decision audit omitted fallback fee | Reuse controller cost calculation and immutable strategy fallback | Actual captured mismatch corrected; explicit zero/override and BUY/SELL tests |

## Final verification and independent review

Combined source head `97dbee522e156fc56087a50c2b742e78f38ebb46` passed the default offline suite: **3,173 passed, 229 explicitly gated skips in 36.66 seconds**; strict mypy: **511 files clean**; import-linter: **10 kept, zero broken**. Dashboard Vitest: **27 files / 66 tests passed**; TypeScript passed. Commands: `env -u DATABASE_URL -u PMS_RUN_INTEGRATION -u PMS_TEST_DATABASE_URL uv run pytest -q`, `uv run mypy src/ tests/ --strict`, `uv run lint-imports`, and dashboard `npm run test:ci` / `npx tsc --noEmit`.

This turn did not rerun local database-creating/deleting schema fixtures; dedicated real-PostgreSQL scenario probes are described above. Full integration CI and the independent Astra verdict are recorded against their exact commits in [PR #94](https://github.com/GBU-AIBC/PredictionMarketSystem/pull/94). Do not infer either from this report's offline test count.

[Machine-readable evidence](e2e-framework-validation-2026-10-18.json) includes all six fills with actual used ask levels, public resolution response fields, source hashes, both historical runs, the prospective sample and lifecycle probes. [Full-precision fill CSV](e2e-framework-fills-2026-10-18.csv) supports independent arithmetic. Coordinator Decimal recomputation gives net PnL -2.992016719576719811 and Brier 0.1862137609571238762335581398, agreeing with stored floats within 1e-12. The independent reviewer must still verify provenance and execution semantics; agreement of arithmetic alone is insufficient.

Raw artifacts and complete attempt ledgers remain under `/tmp/pms-e2e-20261018-{scored,paper,load}*`; worker reports identify exact commands and files. Dedicated local PostgreSQL databases on `127.0.0.1:55432` remain intact; source databases were read-only and host port 5432 was untouched. The original conservative/no-fill and intermediate cold-label runs remain available rather than being overwritten by the successful scenario.

Not checked: funded LIVE execution, prospective natural settlement, long-duration soak, many-strategy capacity or quiet-host maximum throughput. These limitations remain acceptance gaps even when software tests and the scoped replay/PAPER checks pass.
