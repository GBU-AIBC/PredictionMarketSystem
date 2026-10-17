from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import asyncpg
import pytest

from pms.storage.fill_store import FillStore
from pms.storage.live_reconciliation import SubmissionUnknownReconciliationStore


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("PMS_RUN_INTEGRATION") != "1",
    reason="set PMS_RUN_INTEGRATION=1 to run integration tests",
)
@pytest.mark.asyncio
async def test_submission_unknown_filled_reconciles_idempotent_fill() -> None:
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
                "DELETE FROM fill_payloads WHERE fill_id = $1",
                fill_id,
            )
            await connection.execute("DELETE FROM fills WHERE fill_id = $1", fill_id)
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
