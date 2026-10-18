import { render, screen, within } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import { BacktestStrategyDetailView } from '@/components/backtest/BacktestStrategyDetailView';

vi.mock('@/components/Nav', () => ({ Nav: () => <nav /> }));
vi.mock('@/lib/useLiveData', () => ({
  useLiveData: (path: string) => ({
    data: path.endsWith('/strategies') ? [{
      strategy_id: 'alpha', strategy_version_id: 'v1',
      brier: null, pnl_cum: null, fill_rate: 0,
      started_at: '2026-03-01T00:00:00Z', finished_at: '2026-03-02T00:00:00Z',
      portfolio_target_json: [],
      controller_rejection_counts: { no_positive_edge: 2, missing_required_factor: 7 }
    }] : { status: 'completed' }
  })
}));

test('completed zero-trade run shows controller rejection codes in count order', () => {
  render(<BacktestStrategyDetailView runId="run" strategyId="alpha" />);
  expect(screen.getByText('completed run · 0 rows')).toBeInTheDocument();
  const reasons = within(screen.getByRole('list')).getAllByRole('listitem');
  expect(reasons.map((item) => item.textContent)).toEqual([
    'missing_required_factor: 7', 'no_positive_edge: 2'
  ]);
  expect(screen.getByText('No portfolio targets recorded for this strategy run.')).toBeInTheDocument();
});
