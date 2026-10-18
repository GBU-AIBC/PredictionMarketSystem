'use client';

import Link from 'next/link';
import { Nav } from '@/components/Nav';
import { formatDateTime, parseStrategyIdentity } from '@/lib/backtest';
import { useLiveData } from '@/lib/useLiveData';
import type { BacktestRunRow, BacktestStrategyRunRow } from '@/lib/types';

type BacktestStrategyDetailViewProps = {
  runId: string;
  strategyId: string;
};

export function BacktestStrategyDetailView({
  runId,
  strategyId
}: BacktestStrategyDetailViewProps) {
  const runState = useLiveData<BacktestRunRow>(`/research/backtest/${runId}`, 15_000);
  const strategyState = useLiveData<BacktestStrategyRunRow[]>(
    `/research/backtest/${runId}/strategies`,
    15_000
  );
  const parsedIdentity = parseStrategyIdentity(strategyId);
  const strategyRun =
    (strategyState.data ?? []).find((row) =>
      parsedIdentity
        ? row.strategy_id === parsedIdentity.strategyId &&
          row.strategy_version_id === parsedIdentity.strategyVersionId
        : row.strategy_id === strategyId
    ) ?? null;
  const strategyLabel = strategyRun?.strategy_id ?? parsedIdentity?.strategyId ?? strategyId;

  return (
    <main className="shell">
      <Nav />
      <section className="page" data-testid="backtest-strategy-detail-view">
        <div className="hero">
          <div>
            <p className="eyebrow">Research</p>
            <h1>{strategyLabel}</h1>
            <p className="lede">Bookmarkable detail view for the strategy row inside this backtest run.</p>
          </div>
          <div className="hero-actions">
            <Link href={`/backtest/${runId}`} className="ghost-button">
              Back to run
            </Link>
          </div>
        </div>

        {strategyRun === null ? (
          <section className="card empty-state">
            <h3>Strategy row unavailable</h3>
            <p className="muted">Run metadata or strategy metrics are still loading.</p>
          </section>
        ) : (
          <>
            <div className="summary-grid backtest-summary-grid">
              <section className="card summary-card">
                <span className="muted">Version</span>
                <div className="metric compact">{strategyRun.strategy_version_id}</div>
              </section>
              <section className="card summary-card">
                <span className="muted">Brier</span>
                <div className="metric compact">{formatNumber(strategyRun.brier, 3)}</div>
              </section>
              <section className="card summary-card">
                <span className="muted">P&amp;L</span>
                <div className="metric compact">{formatNumber(strategyRun.pnl_cum, 2)}</div>
              </section>
              <section className="card summary-card">
                <span className="muted">Fill rate</span>
                <div className="metric compact">{formatRate(strategyRun.fill_rate)}</div>
              </section>
              <section className="card summary-card">
                <span className="muted">Started</span>
                <p>{formatDateTime(strategyRun.started_at)}</p>
              </section>
              <section className="card summary-card">
                <span className="muted">Finished</span>
                <p>{formatDateTime(strategyRun.finished_at)}</p>
              </section>
            </div>

            {Object.keys(strategyRun.controller_rejection_counts ?? {}).length > 0 && (
              <section className="card detail-section">
                <h3>Controller rejections</h3>
                <ul>
                  {Object.entries(strategyRun.controller_rejection_counts ?? {})
                    .sort(([codeA, countA], [codeB, countB]) => countB - countA || codeA.localeCompare(codeB))
                    .map(([code, count]) => <li key={code}>{code}: {count}</li>)}
                </ul>
              </section>
            )}

            <section className="card detail-section">
              <div className="section-heading">
                <h3>Portfolio targets</h3>
                <span className="muted">
                  {runState.data?.status ?? 'unknown'} run · {strategyRun.portfolio_target_json?.length ?? 0} rows
                </span>
              </div>
              {strategyRun.portfolio_target_json && strategyRun.portfolio_target_json.length > 0 ? (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Market</th>
                        <th>Token</th>
                        <th>Side</th>
                        <th>Timestamp</th>
                        <th>Target size</th>
                      </tr>
                    </thead>
                    <tbody>
                      {strategyRun.portfolio_target_json.map((target) => (
                        <tr key={`${target.market_id}:${target.token_id}:${target.timestamp}`}>
                          <td>{target.market_id}</td>
                          <td>{target.token_id}</td>
                          <td>{target.side}</td>
                          <td>{formatDateTime(target.timestamp)}</td>
                          <td>{target.target_size_usdc.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="muted">No portfolio targets recorded for this strategy run.</p>
              )}
            </section>
          </>
        )}
      </section>
    </main>
  );
}

function formatNumber(value: number | null, digits: number): string {
  return value === null ? 'n/a' : value.toFixed(digits);
}

function formatRate(value: number | null): string {
  return value === null ? 'n/a' : `${(value * 100).toFixed(1)}%`;
}
