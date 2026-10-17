from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
import pytest

from pms.actuator.risk import RiskManager
from pms.config import PMSSettings, RiskSettings
from pms.core.enums import RunMode
from pms.core.models import FillRecord
from pms.runner import Runner
from pms.storage.fill_store import FillStore
from pms.storage.live_emergency_audit import LiveEmergencyAuditWriter
from pms.storage.live_reconciliation import SubmissionUnknownReconciliationStore


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("PMS_RUN_INTEGRATION") != "1",
    reason="set PMS_RUN_INTEGRATION=1 to run integration tests",
)
@pytest.mark.asyncio
async def test_submission_unknown_filled_reconciles_idempotent_fill(
    tmp_path: Path,
) -> None:
    dsn = os.environ.get("PMS_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("PMS_TEST_DATABASE_URL is required")
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2)
    decision_id = "lane-1a-t8-decision"
    fill_id = f"reconciled:{decision_id}:lane-1a-venue-order"
    try:
        async with pool.acquire() as connection:
            await connection.execute(
                "CREATE TABLE IF NOT EXISTS decision_payloads ("
                "decision_id TEXT PRIMARY KEY REFERENCES decisions(decision_id) ON DELETE CASCADE, "
                "payload JSONB NOT NULL)"
            )
            await connection.execute(
                "CREATE TABLE IF NOT EXISTS fill_payloads ("
                "fill_id TEXT PRIMARY KEY REFERENCES fills(fill_id) ON DELETE CASCADE, "
                "payload JSONB NOT NULL)"
            )
            await connection.execute(
                "DELETE FROM fill_payloads WHERE fill_id = $1",
                fill_id,
            )
            await connection.execute("DELETE FROM fills WHERE fill_id = $1", fill_id)
            await connection.execute(
                "DELETE FROM fill_payloads WHERE fill_id = $1",
                "lane-1a-unresolved-trade",
            )
            await connection.execute(
                "DELETE FROM fills WHERE fill_id = $1",
                "lane-1a-unresolved-trade",
            )
            await connection.execute(
                "DELETE FROM order_intents WHERE decision_id = $1",
                decision_id,
            )
            await connection.execute(
                "DELETE FROM decision_payloads WHERE decision_id = $1",
                decision_id,
            )
            await connection.execute(
                "DELETE FROM decisions WHERE decision_id = $1",
                decision_id,
            )
            await connection.execute(
                """
                INSERT INTO decisions (
                    decision_id, opportunity_id, strategy_id, strategy_version_id,
                    status, expires_at
                ) VALUES ($1, $2, $3, $4, 'submission_unknown', now() + interval '1 day')
                """,
                decision_id,
                "lane-1a-t8-opportunity",
                "lane-1a-strategy",
                "lane-1a-v1",
            )
            await connection.execute(
                "INSERT INTO decision_payloads (decision_id, payload) VALUES ($1, $2::jsonb)",
                decision_id,
                json.dumps(
                    {
                        "market_id": "lane-1a-t8-market",
                        "token_id": "lane-1a-t8-token",
                        "venue": "polymarket",
                        "side": "BUY",
                        "action": "BUY",
                        "risk_group_id": "lane-1a-risk-group",
                    }
                ),
            )
            await connection.execute(
                """
                INSERT INTO order_intents (
                    decision_id, strategy_id, strategy_version_id, outcome
                ) VALUES ($1, $2, $3, 'submission_unknown')
                """,
                decision_id,
                "lane-1a-strategy",
                "lane-1a-v1",
            )

        store = SubmissionUnknownReconciliationStore(pool)
        filled_at = datetime.now(tz=UTC)
        assert await store.reconcile_submission_unknown(
            decision_id=decision_id,
            venue_order_id="lane-1a-venue-order",
            status="filled",
            reconciled_by="operator",
            fill_price=0.5,
            fill_notional_usdc=5.0,
            fill_quantity=10.0,
            filled_at=filled_at,
            fees=0.01,
            fee_bps=10,
        )
        fill = await FillStore(pool).get(fill_id)
        assert fill is not None
        assert fill.fill_quantity == pytest.approx(10.0)
        assert fill.risk_group_id == "lane-1a-risk-group"
        trades = await FillStore(pool).read_trades(limit=10)
        assert trades[0].slippage_bps == pytest.approx(0.0)

        live_settings = PMSSettings(
            mode=RunMode.LIVE,
            live_emergency_audit_path=str(tmp_path / "live-audit.jsonl"),
            risk=RiskSettings(max_daily_loss_usdc=20.0),
        )
        restart_runner = Runner(config=live_settings)
        restart_runner.bind_pg_pool(pool)
        await restart_runner._reconcile_risk_from_db()  # noqa: SLF001

        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO fills (
                    fill_id, order_id, market_id, ts, fill_notional_usdc,
                    fill_quantity, strategy_id, strategy_version_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                "legacy-r4-witness",
                "legacy-r4-order",
                "legacy-r4-market",
                filled_at,
                5.0,
                10.0,
                "lane-1a-strategy",
                "lane-1a-v1",
            )
        legacy_runner = Runner(config=live_settings)
        legacy_runner.bind_pg_pool(pool)
        with pytest.raises(RuntimeError, match="legacy-r4-witness"):
            await legacy_runner._reconcile_risk_from_db()  # noqa: SLF001
        async with pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM fills WHERE fill_id = $1",
                "legacy-r4-witness",
            )

        legacy_fee_fill = FillRecord(
            trade_id="legacy-fee-no-slippage",
            order_id="legacy-fee-order",
            decision_id="legacy-fee-decision",
            market_id="legacy-fee-market",
            token_id="legacy-fee-token",
            venue="polymarket",
            side="BUY",
            fill_price=0.5,
            fill_notional_usdc=5.0,
            fill_quantity=10.0,
            executed_at=filled_at,
            filled_at=filled_at,
            status="filled",
            anomaly_flags=["run_mode=live"],
            strategy_id="lane-1a-strategy",
            strategy_version_id="lane-1a-v1",
            fees=0.01,
            fee_bps=20,
        )
        legacy_store = FillStore(pool)
        await legacy_store.insert(legacy_fee_fill)
        legacy_fee_runner = Runner(config=live_settings)
        legacy_fee_runner.bind_pg_pool(pool)
        with pytest.raises(RuntimeError, match="legacy-fee-no-slippage"):
            await legacy_fee_runner._reconcile_risk_from_db()  # noqa: SLF001
        assert await legacy_store.reconcile_fee(
            legacy_fee_fill.trade_id,
            fees=None,
            slippage_bps=0.0,
            audit_pending=False,
        )
        await LiveEmergencyAuditWriter(
            Path(live_settings.live_emergency_audit_path)
        ).append_fee_reconcile(
            fill_id=legacy_fee_fill.trade_id,
            fees=None,
            fee_bps=None,
            slippage_bps=0.0,
            reconciled_by="operator",
            evidence="legacy-venue-trade-43",
        )
        legacy_recovered_runner = Runner(config=live_settings)
        legacy_recovered_runner.bind_pg_pool(pool)
        await legacy_recovered_runner._reconcile_risk_from_db()  # noqa: SLF001
        assert await legacy_store.reconcile_fee(
            legacy_fee_fill.trade_id,
            fees=0.02,
            fee_bps=30,
            slippage_bps=0.0,
            reconciled_by="operator",
            evidence="legacy-venue-trade-override",
            audit_pending=False,
        ) is False
        assert await legacy_store.reconcile_fee(
            legacy_fee_fill.trade_id,
            fees=0.02,
            fee_bps=30,
            slippage_bps=0.0,
            reconciled_by="operator",
            evidence="legacy-venue-trade-override",
            override=True,
            audit_pending=False,
        )

        unresolved = FillRecord(
            trade_id="lane-1a-unresolved-trade",
            order_id="lane-1a-unresolved-order",
            decision_id="lane-1a-unresolved-decision",
            market_id="lane-1a-unresolved-market",
            token_id="lane-1a-unresolved-token",
            venue="polymarket",
            side="BUY",
            fill_price=0.5,
            fill_notional_usdc=5.0,
            fill_quantity=10.0,
            executed_at=filled_at,
            filled_at=filled_at,
            status="filled",
            anomaly_flags=[
                "fee_unreconciled",
                "execution_slippage_bps=0.000000000",
                "run_mode=live",
            ],
            strategy_id="lane-1a-strategy",
            strategy_version_id="lane-1a-v1",
        )
        unresolved_store = FillStore(pool)
        await unresolved_store.insert(unresolved)
        refused_runner = Runner(config=live_settings)
        refused_runner.bind_pg_pool(pool)
        with pytest.raises(RuntimeError, match="reconcile-live-fee"):
            await refused_runner._reconcile_risk_from_db()  # noqa: SLF001
        assert await unresolved_store.reconcile_fee(
            unresolved.trade_id,
            fees=0.02,
            fee_bps=40,
            audit_pending=False,
        )
        await LiveEmergencyAuditWriter(
            Path(live_settings.live_emergency_audit_path)
        ).append_fee_reconcile(
            fill_id=unresolved.trade_id,
            fees=0.02,
            fee_bps=40,
            slippage_bps=None,
            reconciled_by="operator",
            evidence="venue-trade-42",
        )
        recovered_runner = Runner(config=live_settings)
        recovered_runner.bind_pg_pool(pool)
        await recovered_runner._reconcile_risk_from_db()  # noqa: SLF001
        assert await store.reconcile_submission_unknown(
            decision_id=decision_id,
            venue_order_id="lane-1a-venue-order",
            status="filled",
            reconciled_by="operator",
            fill_price=0.5,
            fill_notional_usdc=5.0,
            fill_quantity=10.0,
            filled_at=filled_at,
            fees=0.01,
            fee_bps=10,
        ) is False
        positions = await FillStore(pool).read_positions()
        assert any(
            position.market_id == "lane-1a-t8-market"
            and position.shares_held == pytest.approx(10.0)
            for position in positions
        )
    finally:
        await pool.close()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("PMS_RUN_INTEGRATION") != "1",
    reason="set PMS_RUN_INTEGRATION=1 to run integration tests",
)
@pytest.mark.asyncio
async def test_live_reconciliation_partitions_fill_provenance(
    tmp_path: Path,
) -> None:
    dsn = os.environ.get("PMS_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("PMS_TEST_DATABASE_URL is required")
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2)
    store = FillStore(pool)
    timestamp = datetime(2026, 10, 18, 1, 0, tzinfo=UTC)

    def fill(*, fill_id: str, side: str, price: float, mode: str, at: datetime) -> FillRecord:
        return FillRecord(
            trade_id=fill_id,
            order_id=f"order-{fill_id}",
            decision_id=f"decision-{fill_id}",
            market_id=f"market-{mode}",
            token_id=f"token-{mode}",
            venue="polymarket",
            side=side,
            fill_price=price,
            fill_notional_usdc=price * 100.0,
            fill_quantity=100.0,
            executed_at=at,
            filled_at=at,
            status="filled",
            anomaly_flags=[
                f"run_mode={mode}",
                "execution_slippage_bps=0.000000000",
            ],
            strategy_id="lane-1a-strategy",
            strategy_version_id="lane-1a-v1",
            fees=0.0,
            fee_bps=0,
        )

    try:
        async with pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM fill_payloads WHERE fill_id LIKE 'mode-partition-%'"
            )
            await connection.execute("DELETE FROM fills WHERE fill_id LIKE 'mode-partition-%'")
        await store.insert(
            fill(
                fill_id="mode-partition-paper-open",
                side="BUY",
                price=0.50,
                mode="paper",
                at=timestamp,
            )
        )
        await store.insert(
            fill(
                fill_id="mode-partition-paper-close",
                side="SELL",
                price=0.40,
                mode="paper",
                at=timestamp.replace(hour=2),
            )
        )
        await store.insert(
            fill(
                fill_id="mode-partition-live-open",
                side="BUY",
                price=0.50,
                mode="live",
                at=timestamp,
            )
        )
        await store.insert(
            fill(
                fill_id="mode-partition-live-close",
                side="SELL",
                price=0.40,
                mode="live",
                at=timestamp.replace(hour=2),
            )
        )

        settings = PMSSettings(
            mode=RunMode.LIVE,
            live_emergency_audit_path=str(tmp_path / "mode-audit.jsonl"),
            risk=RiskSettings(max_daily_loss_usdc=5.0),
        )
        runner = Runner(config=settings)
        runner.bind_pg_pool(pool)
        risk = RiskManager(settings.risk)
        runner.actuator_executor = type("Executor", (), {"risk": risk})()

        await runner._reconcile_portfolio_from_db()  # noqa: SLF001
        await runner._reconcile_risk_from_db()  # noqa: SLF001

        assert risk._daily_trades  # noqa: SLF001
        assert [trade.trace_id for trade in risk._daily_trades] == [  # noqa: SLF001
            "mode-partition-live-close"
        ]
        assert all(
            position.market_id != "market-paper"
            for position in runner.portfolio.open_positions
        )
    finally:
        async with pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM fill_payloads WHERE fill_id LIKE 'mode-partition-%'"
            )
            await connection.execute("DELETE FROM fills WHERE fill_id LIKE 'mode-partition-%'")
        await pool.close()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("PMS_RUN_INTEGRATION") != "1",
    reason="set PMS_RUN_INTEGRATION=1 to run integration tests",
)
@pytest.mark.asyncio
async def test_legacy_fill_provenance_backfill_restores_restarts(
    tmp_path: Path,
) -> None:
    dsn = os.environ.get("PMS_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("PMS_TEST_DATABASE_URL is required")
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2)
    fill_id = "legacy-provenance-r6-open"
    filled_at = datetime(2026, 10, 18, 1, 0, tzinfo=UTC)
    try:
        async with pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM fill_payloads WHERE fill_id = $1", fill_id
            )
            await connection.execute("DELETE FROM fills WHERE fill_id = $1", fill_id)
            await connection.execute(
                """
                INSERT INTO fills (
                    fill_id, order_id, market_id, ts, fill_notional_usdc,
                    fill_quantity, strategy_id, strategy_version_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                fill_id,
                "legacy-provenance-r6-order",
                "legacy-provenance-r6-market",
                filled_at,
                10.0,
                20.0,
                "lane-1a-strategy",
                "lane-1a-v1",
            )
            await connection.execute(
                """
                INSERT INTO fill_payloads (fill_id, payload)
                VALUES ($1, $2::jsonb)
                """,
                fill_id,
                json.dumps(
                    {
                        "trade_id": fill_id,
                        "decision_id": "legacy-provenance-r6-decision",
                        "token_id": "legacy-provenance-r6-token",
                        "venue": "polymarket",
                        "side": "BUY",
                        "fill_price": 0.5,
                        "executed_at": filled_at.isoformat(),
                        "status": "filled",
                        "anomaly_flags": ["execution_slippage_bps=0.000000000"],
                        "fees": 0.0,
                        "fee_bps": 0,
                    }
                ),
            )

        legacy_trades = await FillStore(pool).read_trades(limit=10)
        assert any(trade.fill_id == fill_id for trade in legacy_trades)
        replay_trades = await FillStore(pool).read_trades_for_replay(page_size=10)
        assert any(
            trade.fill_id == fill_id and trade.run_mode is None
            for trade in replay_trades
        )
        paper = Runner(
            config=PMSSettings(
                mode=RunMode.PAPER,
                live_emergency_audit_path=str(tmp_path / "paper-audit.jsonl"),
            )
        )
        paper.bind_pg_pool(pool)
        await paper._reconcile_portfolio_from_db()  # noqa: SLF001
        assert any(
            position.market_id == "legacy-provenance-r6-market"
            and position.shares_held == pytest.approx(20.0)
            for position in paper.portfolio.open_positions
        )

        live = Runner(
            config=PMSSettings(
                mode=RunMode.LIVE,
                live_emergency_audit_path=str(tmp_path / "live-audit.jsonl"),
            )
        )
        live.bind_pg_pool(pool)
        with pytest.raises(RuntimeError, match=fill_id):
            await live._reconcile_portfolio_from_db()  # noqa: SLF001
        async with pool.acquire() as connection:
            await connection.execute(
                """
                UPDATE fill_payloads
                SET payload = jsonb_set(
                    payload,
                    '{anomaly_flags}',
                    '["run_mode=staging", "execution_slippage_bps=0.000000000"]'::jsonb
                )
                WHERE fill_id = $1
                """,
                fill_id,
            )
        with pytest.raises(RuntimeError, match=fill_id):
            await live._reconcile_portfolio_from_db()  # noqa: SLF001

        store = FillStore(pool)
        assert await store.tag_fill_provenance(
            mode="live",
            operator="r6-operator",
            evidence="legacy-ledger-review-r6",
        ) >= 1
        assert await store.tag_fill_provenance(
            mode="live",
            operator="r6-operator",
            evidence="legacy-ledger-review-r6-repeat",
        ) == 0
        tagged = await store.get(fill_id)
        assert tagged is not None
        assert "run_mode=live" in tagged.anomaly_flags
        await live._reconcile_portfolio_from_db()  # noqa: SLF001
        assert any(
            position.market_id == "legacy-provenance-r6-market"
            for position in live.portfolio.open_positions
        )
    finally:
        async with pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM fill_payloads WHERE fill_id = $1", fill_id
            )
            await connection.execute("DELETE FROM fills WHERE fill_id = $1", fill_id)
        await pool.close()
