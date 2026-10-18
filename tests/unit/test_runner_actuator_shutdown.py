from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime

import pytest

from pms.config import PMSSettings, RiskSettings
from pms.core.enums import RunMode
from pms.core.models import Portfolio, TradeDecision, OrderState
from pms.runner import Runner
from pms.storage.decision_store import DecisionStore
from pms.storage.dedup_store import InMemoryDedupStore
from tests.unit.test_runner_position_capacity import _decision


class MemoryDecisionStore(DecisionStore):
    def __init__(self) -> None:
        super().__init__()
        self.statuses: dict[str, str] = {}

    async def update_status(
        self, decision_id: str, *, current_status: str,
        next_status: str, updated_at: datetime,
    ) -> bool:
        if self.statuses.get(decision_id) != current_status:
            return False
        self.statuses[decision_id] = next_status
        return True


class RecordingDedupStore(InMemoryDedupStore):
    def __init__(self) -> None:
        super().__init__()
        self.released: list[tuple[str, str]] = []

    async def release(self, decision_id: str, outcome: str) -> None:
        self.released.append((decision_id, outcome))
        await super().release(decision_id, outcome)


@pytest.mark.parametrize("exit_path", ["stop", "start_failure", "start_cancellation"])
async def test_shutdown_cancels_only_unsubmitted_work_and_restart_is_fresh(
    exit_path: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryDecisionStore()
    runner = Runner(config=PMSSettings(
        mode=RunMode.BACKTEST, auto_migrate_default_v2=False,
        risk=RiskSettings(max_open_positions=2),
    ), sensors=[], decision_store=store)
    dedup = RecordingDedupStore()
    runner.actuator_executor.dedup_store = dedup
    queued = _decision()
    store.statuses[queued.decision_id] = "accepted"
    store.statuses["already-submitted"] = "submitted"
    store.statuses["unknown-at-venue"] = "submission_unknown"
    assert await dedup.acquire(queued)
    assert await runner.enqueue_accepted_decision(queued)
    old_queue = runner._decision_queue
    assert store.statuses[queued.decision_id] == "queued"

    if exit_path == "stop":
        await runner.stop()
    else:
        async def fail_start() -> None:
            if exit_path == "start_cancellation":
                raise asyncio.CancelledError
            raise RuntimeError("failed after accepting queued work")

        with monkeypatch.context() as patch:
            patch.setattr(runner, "_configure_controllers", fail_start)
            with pytest.raises(asyncio.CancelledError if exit_path == "start_cancellation" else RuntimeError):
                await runner.start()

    assert old_queue.empty()
    await asyncio.wait_for(old_queue.join(), 1)
    assert store.statuses == {
        queued.decision_id: "cancelled",
        "already-submitted": "submitted",
        "unknown-at-venue": "submission_unknown",
    }
    assert dedup.released == [(queued.decision_id, "rejected")]
    assert not runner._pending_open_position_keys_by_decision

    calls: list[str] = []

    async def unexpected_execution(
        decision: TradeDecision, portfolio: Portfolio,
        *, dedup_acquired: bool = False,
    ) -> OrderState:
        calls.append(decision.decision_id)
        raise AssertionError("old work executed after restart")

    monkeypatch.setattr(runner.actuator_executor, "execute", unexpected_execution)
    try:
        await runner.start()
        await asyncio.sleep(.06)
        assert calls == []
        # Capacity reserved by the old queue must be available again.
        fresh = replace(queued, decision_id="new-decision")
        store.statuses[fresh.decision_id] = "accepted"
        assert await runner.enqueue_accepted_decision(fresh)
    finally:
        await runner.stop()
