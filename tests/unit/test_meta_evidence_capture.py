from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import asyncpg
import pytest

from pms.config import ControllerSettings, PMSSettings
from pms.controller.outcome_tokens import OutcomeTokens
from pms.controller.pipeline import ControllerPipeline
from pms.core.enums import OrderStatus, Side, TimeInForce
from pms.core.interfaces import IForecaster
from pms.core.models import FillRecord, MarketSignal, Portfolio, TradeDecision
from pms.evaluation.adapters.scoring import Scorer
from pms.storage.decision_store import DecisionStore


class ConstantForecaster:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict(self, signal: MarketSignal) -> tuple[float, float, str]:
        del signal
        return (self.probability, 0.8, "constant")

    async def forecast(self, signal: MarketSignal) -> float:
        del signal
        return self.probability


class StaticOutcomeTokenResolver:
    async def resolve(
        self,
        *,
        market_id: str,
        signal_token_id: str | None,
    ) -> OutcomeTokens:
        del market_id
        return OutcomeTokens(yes_token_id=signal_token_id, no_token_id="token-meta-no")


class _RecordingConnection:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, tuple[object, ...], bool]] = []
        self.fetch_rows: list[dict[str, object]] = []
        self.in_transaction = False

    async def execute(self, query: str, *args: object) -> str:
        self.execute_calls.append((query, args, self.in_transaction))
        return "OK"

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        del query, args
        return list(self.fetch_rows)

    def transaction(self) -> _Transaction:
        return _Transaction(self)


class _Transaction:
    def __init__(self, connection: _RecordingConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _Transaction:
        self._connection.in_transaction = True
        return self

    async def __aexit__(self, *_: object) -> None:
        self._connection.in_transaction = False


class _Acquire:
    def __init__(self, connection: _RecordingConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _RecordingConnection:
        return self._connection

    async def __aexit__(self, *_: object) -> None:
        return None


class _Pool:
    def __init__(self, connection: _RecordingConnection) -> None:
        self._connection = connection

    def acquire(self) -> _Acquire:
        return _Acquire(self._connection)


def _signal(
    *,
    external_signal: dict[str, Any] | None = None,
    orderbook: dict[str, Any] | None = None,
) -> MarketSignal:
    return MarketSignal(
        market_id="market-meta-capture",
        token_id="token-meta-yes",
        venue="polymarket",
        title="Will meta evidence capture spread?",
        yes_price=0.4,
        volume_24h=1000.0,
        resolves_at=datetime(2026, 5, 30, tzinfo=UTC),
        orderbook=orderbook
        or {
            "bids": [{"price": 0.39, "size": 10.0}],
            "asks": [{"price": 0.41, "size": 10.0}],
        },
        external_signal=external_signal or {"fair_value": 0.7},
        fetched_at=datetime(2026, 5, 6, 0, 0, tzinfo=UTC),
        market_status="open",
    )


def _portfolio() -> Portfolio:
    return Portfolio(
        total_usdc=1000.0,
        free_usdc=1000.0,
        locked_usdc=0.0,
        open_positions=[],
    )


def _decision(
    *,
    spread_bps_at_decision: int | None = 123,
    calibration_prob_estimate: float | None = None,
) -> TradeDecision:
    return TradeDecision(
        decision_id="decision-meta-capture",
        market_id="market-meta-capture",
        token_id="token-meta-yes",
        venue="polymarket",
        side=Side.BUY.value,
        notional_usdc=25.0,
        order_type="limit",
        max_slippage_bps=50,
        stop_conditions=["unit"],
        prob_estimate=0.7,
        expected_edge=0.3,
        time_in_force=TimeInForce.GTC,
        opportunity_id="opportunity-meta-capture",
        strategy_id="default",
        strategy_version_id="default-v1",
        limit_price=0.4,
        action=Side.BUY.value,
        model_id="unit",
        spread_bps_at_decision=spread_bps_at_decision,
        calibration_prob_estimate=calibration_prob_estimate,
    )


def _fill() -> FillRecord:
    return FillRecord(
        trade_id="trade-meta-capture",
        order_id="order-meta-capture",
        decision_id="decision-meta-capture",
        market_id="market-meta-capture",
        token_id="token-meta-yes",
        venue="polymarket",
        side=Side.BUY.value,
        fill_price=0.41,
        fill_notional_usdc=25.0,
        fill_quantity=60.9756,
        executed_at=datetime(2026, 5, 6, tzinfo=UTC),
        filled_at=datetime(2026, 5, 6, tzinfo=UTC),
        status=OrderStatus.MATCHED.value,
        anomaly_flags=[],
        strategy_id="default",
        strategy_version_id="default-v1",
        resolved_outcome=1.0,
    )


@pytest.mark.asyncio
async def test_controller_captures_spread_bps_from_explicit_signal_field() -> None:
    pipeline = ControllerPipeline(
        forecasters=(cast(IForecaster, ConstantForecaster(0.7)),),
        settings=PMSSettings(controller=ControllerSettings(max_spread_bps=200.0)),
    )

    decision = await pipeline.decide(
        _signal(external_signal={"fair_value": 0.7, "spread_bps": 123}),
        portfolio=_portfolio(),
    )

    assert decision is not None
    assert decision.spread_bps_at_decision == 123


@pytest.mark.asyncio
async def test_controller_computes_spread_bps_from_orderbook_when_missing() -> None:
    pipeline = ControllerPipeline(
        forecasters=(cast(IForecaster, ConstantForecaster(0.7)),),
        settings=PMSSettings(),
    )

    decision = await pipeline.decide(_signal(), portfolio=_portfolio())

    assert decision is not None
    assert decision.spread_bps_at_decision == 500


@pytest.mark.asyncio
async def test_controller_captures_buy_no_spread_for_traded_token() -> None:
    pipeline = ControllerPipeline(
        forecasters=(cast(IForecaster, ConstantForecaster(0.3)),),
        outcome_token_resolver=StaticOutcomeTokenResolver(),
        settings=PMSSettings(),
    )

    decision = await pipeline.decide(_signal(), portfolio=_portfolio())

    assert decision is not None
    assert decision.outcome == "NO"
    assert decision.token_id == "token-meta-no"
    assert decision.spread_bps_at_decision == 333


def test_scorer_copies_decision_edge_and_spread_to_eval_record() -> None:
    record = Scorer().score(_fill(), _decision(spread_bps_at_decision=321))

    assert record.edge_at_decision == pytest.approx(0.3)
    assert record.spread_bps_at_decision == 321


@pytest.mark.asyncio
async def test_decision_store_round_trips_spread_bps_in_payload() -> None:
    connection = _RecordingConnection()
    store = DecisionStore(cast(asyncpg.Pool, _Pool(connection)))
    created_at = datetime(2026, 5, 6, tzinfo=UTC)

    await store.insert(
        _decision(spread_bps_at_decision=456),
        factor_snapshot_hash="snapshot",
        created_at=created_at,
        expires_at=created_at + timedelta(minutes=15),
    )

    payload_args = next(
        args
        for query, args, _ in connection.execute_calls
        if "INSERT INTO decision_payloads" in query
    )
    payload = json.loads(cast(str, payload_args[1]))
    assert payload["spread_bps_at_decision"] == 456

    connection.fetch_rows = [
        {
            "decision_id": "decision-meta-capture",
            "opportunity_id": "opportunity-meta-capture",
            "strategy_id": "default",
            "strategy_version_id": "default-v1",
            "status": "pending",
            "factor_snapshot_hash": "snapshot",
            "created_at": created_at,
            "updated_at": created_at,
            "expires_at": created_at + timedelta(minutes=15),
            "payload": json.dumps(
                {
                    "decision_id": "decision-meta-capture",
                    "market_id": "market-meta-capture",
                    "token_id": "token-meta-yes",
                    "venue": "polymarket",
                    "side": "BUY",
                    "notional_usdc": 25.0,
                    "order_type": "limit",
                    "max_slippage_bps": 50,
                    "stop_conditions": ["unit"],
                    "prob_estimate": 0.7,
                    "expected_edge": 0.3,
                    "time_in_force": "GTC",
                    "opportunity_id": "opportunity-meta-capture",
                    "strategy_id": "default",
                    "strategy_version_id": "default-v1",
                    "limit_price": 0.4,
                    "action": "BUY",
                    "outcome": "YES",
                    "model_id": "unit",
                    "intent_key": None,
                    "spread_bps_at_decision": 789,
                }
            ),
            "opportunity_row_id": None,
        }
    ]

    rows = await store.read_decisions(limit=1)

    assert rows[0].decision.spread_bps_at_decision == 789


@pytest.mark.asyncio
async def test_decision_store_round_trips_calibration_probability() -> None:
    connection = _RecordingConnection()
    store = DecisionStore(cast(asyncpg.Pool, _Pool(connection)))
    created_at = datetime(2026, 5, 6, tzinfo=UTC)

    await store.insert(
        _decision(calibration_prob_estimate=0.31),
        factor_snapshot_hash="snapshot",
        created_at=created_at,
        expires_at=created_at + timedelta(minutes=15),
    )
    payload_args = next(
        args
        for query, args, _ in connection.execute_calls
        if "INSERT INTO decision_payloads" in query
    )
    payload = json.loads(cast(str, payload_args[1]))
    assert payload["calibration_prob_estimate"] == pytest.approx(0.31)


@pytest.mark.asyncio
async def test_decision_store_round_trips_decision_time_evidence_payload() -> None:
    connection = _RecordingConnection()
    store = DecisionStore(cast(asyncpg.Pool, _Pool(connection)))
    created_at = datetime(2026, 5, 6, tzinfo=UTC)

    decision_evidence = {
        "book_hash": "book-hash-unit",
        "book_top_levels": {
            "bids": [{"price": 0.39, "size": 10.0}],
            "asks": [{"price": 0.41, "size": 11.0}],
        },
        "book_age_ms": 42.0,
        "quote_source": "postgres_snapshot",
        "factor_snapshot_hash": "snapshot",
    }

    await store.insert(
        _decision(spread_bps_at_decision=456),
        factor_snapshot_hash="snapshot",
        created_at=created_at,
        expires_at=created_at + timedelta(minutes=15),
        decision_evidence=decision_evidence,
    )

    payload_args = next(
        args
        for query, args, _ in connection.execute_calls
        if "INSERT INTO decision_payloads" in query
    )
    payload = json.loads(cast(str, payload_args[1]))
    assert payload["decision_evidence"] == decision_evidence

    connection.fetch_rows = [
        {
            "decision_id": "decision-meta-capture",
            "opportunity_id": "opportunity-meta-capture",
            "strategy_id": "default",
            "strategy_version_id": "default-v1",
            "status": "pending",
            "factor_snapshot_hash": "snapshot",
            "created_at": created_at,
            "updated_at": created_at,
            "expires_at": created_at + timedelta(minutes=15),
            "payload": json.dumps(payload),
            "opportunity_row_id": None,
        }
    ]

    rows = await store.read_decisions(limit=1)

    assert rows[0].decision_evidence == decision_evidence


@pytest.mark.asyncio
async def test_decision_store_tolerates_float_string_and_malformed_spread_payload() -> None:
    connection = _RecordingConnection()
    store = DecisionStore(cast(asyncpg.Pool, _Pool(connection)))
    created_at = datetime(2026, 5, 6, tzinfo=UTC)

    base_payload = {
        "decision_id": "decision-meta-capture",
        "market_id": "market-meta-capture",
        "token_id": "token-meta-yes",
        "venue": "polymarket",
        "side": "BUY",
        "notional_usdc": 25.0,
        "order_type": "limit",
        "max_slippage_bps": 50,
        "stop_conditions": ["unit"],
        "prob_estimate": 0.7,
        "expected_edge": 0.3,
        "time_in_force": "GTC",
        "opportunity_id": "opportunity-meta-capture",
        "strategy_id": "default",
        "strategy_version_id": "default-v1",
        "limit_price": 0.4,
        "action": "BUY",
        "outcome": "YES",
        "model_id": "unit",
        "intent_key": None,
    }
    connection.fetch_rows = [
        {
            "decision_id": "decision-meta-capture",
            "opportunity_id": "opportunity-meta-capture",
            "strategy_id": "default",
            "strategy_version_id": "default-v1",
            "status": "pending",
            "factor_snapshot_hash": "snapshot",
            "created_at": created_at,
            "updated_at": created_at,
            "expires_at": created_at + timedelta(minutes=15),
            "payload": json.dumps({**base_payload, "spread_bps_at_decision": "120.0"}),
            "opportunity_row_id": None,
        },
        {
            "decision_id": "decision-meta-capture-bad",
            "opportunity_id": "opportunity-meta-capture",
            "strategy_id": "default",
            "strategy_version_id": "default-v1",
            "status": "pending",
            "factor_snapshot_hash": "snapshot",
            "created_at": created_at,
            "updated_at": created_at,
            "expires_at": created_at + timedelta(minutes=15),
            "payload": json.dumps(
                {
                    **base_payload,
                    "decision_id": "decision-meta-capture-bad",
                    "spread_bps_at_decision": "bad-data",
                }
            ),
            "opportunity_row_id": None,
        },
    ]

    rows = await store.read_decisions(limit=2)

    assert rows[0].decision.spread_bps_at_decision == 120
    assert rows[1].decision.spread_bps_at_decision is None
