from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

import pytest
from pydantic import SecretStr

from pms.actuator.adapters.polymarket import (
    FileFirstLiveOrderGate,
    LivePreSubmitQuote,
    PolymarketActuator,
    PolymarketOrderResult,
)
from pms.actuator.executor import ActuatorExecutor
from pms.actuator.feedback import ActuatorFeedback
from pms.actuator.risk import RiskManager
from pms.config import (
    ControllerSettings,
    DiscordSettings,
    PMSSettings,
    PolymarketSettings,
    RiskSettings,
)
from pms.core.enums import OrderStatus, RunMode, Side, TimeInForce
from pms.core.models import FillRecord, OrderState, TradeDecision
from pms.runner import Runner
from pms.storage.dedup_store import InMemoryDedupStore
from pms.storage.eval_store import EvalStore
from pms.storage.feedback_store import FeedbackStore
from pms.storage.fill_store import FillStore
from pms.storage.order_store import OrderStore
from tests.support.fake_stores import InMemoryEvalStore, InMemoryFeedbackStore
from tests.support.live_paths import (
    make_live_preflight_artifact_path,
    make_live_report_paths,
    make_private_live_paths,
)


@dataclass
class RecordingOrderStore:
    inserted: list[OrderState] = field(default_factory=list)

    async def insert(self, order: OrderState) -> None:
        self.inserted.append(order)


@dataclass
class RecordingFillStore:
    inserted: list[FillRecord] = field(default_factory=list)

    async def insert(self, fill: FillRecord) -> None:
        self.inserted.append(fill)


@dataclass(frozen=True)
class AllowQuoteProvider:
    async def quote(
        self,
        order: object,
        credentials: object,
    ) -> LivePreSubmitQuote:
        del credentials
        notional = getattr(order, "notional_usdc")
        price = getattr(order, "price")
        return LivePreSubmitQuote(
            market_status="open",
            book_age_ms=20.0,
            executable_notional_usdc=float(notional),
            best_executable_price=float(price),
            spread_bps=10.0,
            quote_hash="quote-integration",
            book_ts=datetime(2026, 4, 26, tzinfo=UTC),
        )


@dataclass
class MockPolymarketClient:
    # Signals validate_live_mode_ready that this injected client does not
    # drive the real Polymarket SDK, so the py_clob_client_v2 runtime
    # dependency is not required to exercise the order/fill persistence path.
    requires_live_mode: bool = False
    submitted: list[object] = field(default_factory=list)

    async def submit_order(
        self,
        order: object,
        credentials: object,
    ) -> PolymarketOrderResult:
        del credentials
        self.submitted.append(order)
        return PolymarketOrderResult(
            order_id="pm-live-integration-order",
            status=OrderStatus.MATCHED.value,
            raw_status="matched",
            filled_notional_usdc=12.0,
            remaining_notional_usdc=0.0,
            fill_price=0.48,
            filled_quantity=25.0,
            fees=0.0,
            fee_bps=0,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_polymarket_order_uses_runner_order_and_fill_persistence_paths() -> None:
    order_store = RecordingOrderStore()
    fill_store = RecordingFillStore()
    feedback_store = InMemoryFeedbackStore()
    settings = _live_settings()
    settings.live_preflight_artifact_path = make_live_preflight_artifact_path(
        prefix="pms-live-integration-preflight-",
        settings=settings,
    )
    runner = Runner(
        config=settings,
        eval_store=cast(EvalStore, InMemoryEvalStore()),
        feedback_store=cast(FeedbackStore, feedback_store),
        order_store=cast(OrderStore, order_store),
        fill_store=cast(FillStore, fill_store),
    )
    client = MockPolymarketClient()
    approval_path = Path(cast(str, settings.polymarket.first_live_order_approval_path))
    approval_payload = _approval_payload()
    approval_path.write_text(json.dumps(approval_payload), encoding="utf-8")
    approval_sidecar_path = _sidecar_path(approval_path)
    approval_sidecar_path.write_text(
        json.dumps(_approval_sidecar_payload(approval_payload)),
        encoding="utf-8",
    )
    gate = FileFirstLiveOrderGate(
        approval_path,
        require_approver_sidecar=True,
        approval_max_age_s=settings.polymarket.operator_approval_max_age_s,
    )
    runner.actuator_executor = ActuatorExecutor(
        adapter=PolymarketActuator(
            settings,
            client=client,
            operator_gate=gate,
            quote_provider=AllowQuoteProvider(),
            live_preflight_validated=True,
        ),
        risk=RiskManager(settings.risk),
        feedback=ActuatorFeedback(cast(FeedbackStore, feedback_store)),
        dedup_store=InMemoryDedupStore(),
    )

    await runner.enqueue_accepted_decision(_decision())
    task = asyncio.create_task(runner._actuator_loop())
    await runner._decision_queue.join()
    runner._stop_event.set()
    await task

    assert len(client.submitted) == 1
    assert approval_path.exists() is False
    assert approval_sidecar_path.exists() is False
    assert [order.order_id for order in order_store.inserted] == [
        "pm-live-integration-order"
    ]
    assert [fill.order_id for fill in fill_store.inserted] == [
        "pm-live-integration-order"
    ]
    assert fill_store.inserted[0].strategy_id == "default"
    assert fill_store.inserted[0].strategy_version_id == "default-v1"


def _live_settings() -> PMSSettings:
    approval_path, audit_path = make_private_live_paths(prefix="pms-live-integration-")
    paper_report_path, rehearsal_report_path = make_live_report_paths(
        prefix="pms-live-integration-reports-"
    )
    attested_at = datetime.now(tz=UTC)
    return PMSSettings(
        mode=RunMode.LIVE,
        secret_source="fly",
        live_trading_enabled=True,
        api_token="live-api-token",
        live_emergency_audit_path=str(
            Path(approval_path).parent / "live-emergency-audit.jsonl"
        ),
        live_first_order_audit_path=audit_path,
        live_exit_criteria_ratified_by="test-operator",
        live_exit_criteria_ratified_at=attested_at,
        live_compliance_reviewed_by="test-compliance",
        live_compliance_reviewed_at=attested_at,
        live_compliance_jurisdiction="US",
        live_paper_soak_report_path=paper_report_path,
        live_operator_rehearsal_report_path=rehearsal_report_path,
        auto_migrate_default_v2=False,
        controller=ControllerSettings(time_in_force="IOC", quote_source="dual"),
        discord=DiscordSettings(
            webhook_url=SecretStr("https://discord.example/webhooks/integration/unit"),
            alert_dir=str(Path(approval_path).parent / "discord-alerts"),
        ),
        risk=RiskSettings(
            max_position_per_market=1000.0,
            max_total_exposure=10_000.0,
            max_drawdown_pct=20.0,
            max_daily_loss_usdc=20.0,
            max_open_positions=5,
            max_exposure_per_risk_group=5_000.0,
            max_quantity_shares=500.0,
        ),
        polymarket=PolymarketSettings(
            private_key="private-key",
            api_key="api-key",
            api_secret="api-secret",
            api_passphrase="passphrase",
            signature_type=1,
            funder_address="0x1111111111111111111111111111111111111111",
            operator_approval_mode="every_order",
            first_live_order_approval_path=approval_path,
        ),
    )


def _approval_payload() -> dict[str, object]:
    return {
        "approved": True,
        "max_notional_usdc": 12.0,
        "venue": "polymarket",
        "market_id": "m-live-integration",
        "token_id": "t-live-yes",
        "side": Side.BUY.value,
        "outcome": "YES",
        "limit_price": 0.48,
        "max_slippage_bps": 50,
    }


def _approval_sidecar_payload(
    approval_payload: dict[str, object],
) -> dict[str, object]:
    return {
        "approver_id": "test-operator",
        "approval_sha256": _approval_payload_hash(approval_payload),
        "ts": datetime.now(UTC).isoformat(),
    }


def _approval_payload_hash(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _sidecar_path(approval_path: Path) -> Path:
    return Path(str(approval_path) + ".meta.json")


def _decision(
    *,
    side: Literal["BUY", "SELL"] = Side.BUY.value,
) -> TradeDecision:
    return TradeDecision(
        decision_id="d-live-integration",
        market_id="m-live-integration",
        token_id="t-live-yes",
        venue="polymarket",
        side=side,
        notional_usdc=12.0,
        order_type="limit",
        max_slippage_bps=50,
        stop_conditions=["integration-test"],
        prob_estimate=0.6,
        expected_edge=0.2,
        time_in_force=TimeInForce.IOC,
        opportunity_id="op-live-integration",
        strategy_id="default",
        strategy_version_id="default-v1",
        action=side,
        limit_price=0.48,
        outcome="YES",
        risk_group_id="event:live-integration",
    )
