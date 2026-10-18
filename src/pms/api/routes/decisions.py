from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from pms.actuator.risk import RiskDecision
from pms.core.models import Opportunity, Portfolio, TradeDecision


class AcceptDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    factor_snapshot_hash: str


class OpportunityRow(BaseModel):
    opportunity_id: str
    market_id: str
    token_id: str
    side: str
    selected_factor_values: dict[str, float]
    expected_edge: float
    rationale: str
    target_size_usdc: float
    expiry: str | None
    staleness_policy: str
    strategy_id: str
    strategy_version_id: str
    created_at: str
    factor_snapshot_hash: str | None
    composition_trace: dict[str, Any]


class DecisionRow(BaseModel):
    decision_id: str
    market_id: str
    token_id: str | None
    venue: str
    side: str
    notional_usdc: float
    order_type: str
    max_slippage_bps: int
    stop_conditions: list[str]
    prob_estimate: float
    expected_edge: float
    time_in_force: str
    opportunity_id: str
    strategy_id: str
    strategy_version_id: str
    limit_price: float
    action: str | None
    outcome: str
    model_id: str | None
    status: str
    factor_snapshot_hash: str | None
    created_at: str
    updated_at: str
    expires_at: str
    forecaster: str
    kelly_size: float
    spread_bps_at_decision: int | None = None
    risk_group_id: str | None = None
    decision_evidence: dict[str, Any]
    opportunity: OpportunityRow | None = None


class AcceptDecisionResponse(BaseModel):
    decision_id: str
    status: Literal["accepted"]
    fill_id: str | None = None


_ACCEPTED_DOWNSTREAM_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "accepted",
        "queued",
        "submitted",
        "partially_filled",
        "filled",
    }
)


class StoredDecisionLike(Protocol):
    decision: TradeDecision
    status: str
    factor_snapshot_hash: str | None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    opportunity: Opportunity | None


class DecisionsReader(Protocol):
    async def read_decisions(
        self,
        *,
        limit: int,
        offset: int = 0,
        status: str | None = None,
        include_opportunity: bool = False,
        until: datetime | None = None,
    ) -> Sequence[StoredDecisionLike]: ...

    async def get_decision(
        self,
        decision_id: str,
        *,
        include_opportunity: bool = False,
    ) -> StoredDecisionLike | None: ...

    async def update_status(
        self,
        decision_id: str,
        *,
        current_status: str,
        next_status: str,
        updated_at: datetime,
    ) -> bool: ...


class RiskChecker(Protocol):
    def check(self, decision: TradeDecision, portfolio: Portfolio) -> RiskDecision: ...


class DedupStoreLike(Protocol):
    async def acquire(self, decision: TradeDecision) -> bool: ...
    async def release(self, decision_id: str, outcome: str) -> None: ...


class DecisionNotFoundError(LookupError):
    pass


class DecisionMarketChangedError(RuntimeError):
    def __init__(self, current_factor_snapshot_hash: str | None) -> None:
        super().__init__("market_changed")
        self.current_factor_snapshot_hash = current_factor_snapshot_hash


class DecisionRiskRejectedError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class DecisionEnqueueRejectedError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


async def list_decisions(
    store: DecisionsReader,
    *,
    limit: int,
    offset: int = 0,
    status: str | None = None,
    include_opportunity: bool = False,
    until: datetime | None = None,
) -> list[DecisionRow]:
    rows = await store.read_decisions(
        limit=limit,
        offset=offset,
        status=status,
        include_opportunity=include_opportunity,
        until=until,
    )
    return [_decision_row(row) for row in rows]


async def get_decision(
    store: DecisionsReader,
    *,
    decision_id: str,
    include_opportunity: bool = False,
) -> DecisionRow | None:
    row = await store.get_decision(
        decision_id,
        include_opportunity=include_opportunity,
    )
    if row is None:
        return None
    return _decision_row(row)


async def accept_decision(
    store: DecisionsReader,
    *,
    decision_id: str,
    factor_snapshot_hash: str,
    dedup_store: DedupStoreLike,
    risk: RiskChecker,
    portfolio: Portfolio,
    enqueue: Callable[[TradeDecision], Awaitable[bool | None]],
) -> AcceptDecisionResponse:
    row = await store.get_decision(decision_id, include_opportunity=False)
    if row is None:
        raise DecisionNotFoundError("Decision not found")
    if row.status in _ACCEPTED_DOWNSTREAM_STATUSES:
        return _accepted_response(row.decision.decision_id)
    if row.status != "pending":
        raise DecisionNotFoundError("Decision not found")
    if factor_snapshot_hash != (row.factor_snapshot_hash or ""):
        raise DecisionMarketChangedError(row.factor_snapshot_hash)

    acquired = await dedup_store.acquire(row.decision)
    if not acquired:
        return _accepted_response(row.decision.decision_id)

    lease_released = False
    lease_transferred = False
    try:
        risk_decision = risk.check(row.decision, portfolio)
        if not risk_decision.approved:
            await store.update_status(
                row.decision.decision_id,
                current_status="pending",
                next_status="rejected",
                updated_at=datetime.now(tz=UTC),
            )
            await dedup_store.release(row.decision.decision_id, "invalid")
            lease_released = True
            raise DecisionRiskRejectedError(risk_decision.reason)

        updated = await store.update_status(
            row.decision.decision_id,
            current_status="pending",
            next_status="accepted",
            updated_at=datetime.now(tz=UTC),
        )
        if not updated:
            refreshed = await store.get_decision(
                row.decision.decision_id,
                include_opportunity=False,
            )
            if refreshed is None or refreshed.status != "accepted":
                raise DecisionNotFoundError("Decision not found")
        enqueued = await enqueue(row.decision)
        if enqueued is False:
            refreshed = await store.get_decision(
                row.decision.decision_id,
                include_opportunity=False,
            )
            if refreshed is None or refreshed.status != "rejected":
                await store.update_status(
                    row.decision.decision_id,
                    current_status="accepted",
                    next_status="rejected",
                    updated_at=datetime.now(tz=UTC),
                )
                await dedup_store.release(row.decision.decision_id, "rejected")
            lease_released = True
            raise DecisionEnqueueRejectedError("max_open_positions_capacity")
        lease_transferred = True
        return _accepted_response(row.decision.decision_id)
    finally:
        if not lease_transferred and not lease_released:
            with suppress(Exception):
                await dedup_store.release(row.decision.decision_id, "invalid")


def _accepted_response(decision_id: str) -> AcceptDecisionResponse:
    return AcceptDecisionResponse(
        decision_id=decision_id,
        status="accepted",
        fill_id=None,
    )


def _decision_row(row: StoredDecisionLike) -> DecisionRow:
    return DecisionRow(
        decision_id=row.decision.decision_id,
        market_id=row.decision.market_id,
        token_id=row.decision.token_id,
        venue=row.decision.venue,
        side=row.decision.side,
        notional_usdc=row.decision.notional_usdc,
        order_type=row.decision.order_type,
        max_slippage_bps=row.decision.max_slippage_bps,
        stop_conditions=list(row.decision.stop_conditions),
        prob_estimate=row.decision.prob_estimate,
        expected_edge=row.decision.expected_edge,
        time_in_force=row.decision.time_in_force.value,
        opportunity_id=row.decision.opportunity_id,
        strategy_id=row.decision.strategy_id,
        strategy_version_id=row.decision.strategy_version_id,
        limit_price=row.decision.limit_price,
        action=row.decision.action,
        outcome=row.decision.outcome,
        model_id=row.decision.model_id,
        status=row.status,
        factor_snapshot_hash=row.factor_snapshot_hash,
        created_at=row.created_at.isoformat(),
        updated_at=row.updated_at.isoformat(),
        expires_at=row.expires_at.isoformat(),
        forecaster=row.decision.model_id or "rules",
        kelly_size=row.decision.notional_usdc,
        spread_bps_at_decision=row.decision.spread_bps_at_decision,
        risk_group_id=row.decision.risk_group_id,
        decision_evidence=_decision_evidence(row),
        opportunity=(
            OpportunityRow(
                opportunity_id=row.opportunity.opportunity_id,
                market_id=row.opportunity.market_id,
                token_id=row.opportunity.token_id,
                side=row.opportunity.side,
                selected_factor_values=dict(row.opportunity.selected_factor_values),
                expected_edge=row.opportunity.expected_edge,
                rationale=row.opportunity.rationale,
                target_size_usdc=row.opportunity.target_size_usdc,
                expiry=(
                    row.opportunity.expiry.isoformat()
                    if row.opportunity.expiry is not None
                    else None
                ),
                staleness_policy=row.opportunity.staleness_policy,
                strategy_id=row.opportunity.strategy_id,
                strategy_version_id=row.opportunity.strategy_version_id,
                created_at=row.opportunity.created_at.isoformat(),
                factor_snapshot_hash=row.opportunity.factor_snapshot_hash,
                composition_trace=dict(row.opportunity.composition_trace),
            )
            if row.opportunity is not None
            else None
        ),
    )


def _decision_evidence(row: StoredDecisionLike) -> dict[str, Any]:
    raw_evidence = getattr(row, "decision_evidence", {})
    if isinstance(raw_evidence, dict):
        return dict(raw_evidence)
    return {}
