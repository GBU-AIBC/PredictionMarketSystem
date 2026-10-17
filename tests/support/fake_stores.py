from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from pms.core.models import EvalRecord, Feedback, Opportunity


class InMemoryEvalStore:
    def __init__(self, records: list[EvalRecord] | None = None) -> None:
        self._records = list(records or [])

    async def append(self, record: EvalRecord) -> bool:
        self._records.append(record)
        return True

    async def replace(self, record: EvalRecord) -> bool:
        for index, existing in enumerate(self._records):
            if existing.decision_id == record.decision_id:
                self._records[index] = record
                return True
        return False

    async def all(self) -> list[EvalRecord]:
        return list(self._records)

    async def all_for_strategy(
        self,
        strategy_id: str,
        strategy_version_id: str,
    ) -> list[EvalRecord]:
        return [
            record
            for record in self._records
            if record.strategy_id == strategy_id
            and record.strategy_version_id == strategy_version_id
        ]


class InMemoryFeedbackStore:
    def __init__(self, items: list[Feedback] | None = None) -> None:
        self._items = list(items or [])

    async def append(
        self,
        feedback: Feedback,
        *,
        strategy_id: str | None = None,
        strategy_version_id: str | None = None,
    ) -> None:
        del strategy_id, strategy_version_id
        self._items.append(feedback)

    async def all(self) -> list[Feedback]:
        return list(self._items)

    async def list(self, resolved: bool | None = None) -> list[Feedback]:
        if resolved is None:
            return await self.all()
        return [item for item in self._items if item.resolved is resolved]

    async def get(self, feedback_id: str) -> Feedback | None:
        for item in self._items:
            if item.feedback_id == feedback_id:
                return item
        return None

    async def resolve(self, feedback_id: str) -> None:
        for index, item in enumerate(self._items):
            if item.feedback_id == feedback_id:
                self._items[index] = replace(
                    item,
                    resolved=True,
                    resolved_at=datetime.now(tz=UTC),
                )
                return


class InMemoryOpportunityStore:
    def __init__(self, items: list[Opportunity] | None = None) -> None:
        self._items = list(items or [])

    async def insert(self, opportunity: Opportunity) -> None:
        self._items.append(opportunity)

    async def all(self) -> list[Opportunity]:
        return list(self._items)


class LegacyPathEvalStore(InMemoryEvalStore):
    def __init__(
        self,
        path: Path,
        records: list[EvalRecord] | None = None,
    ) -> None:
        super().__init__(records)
        self.path = path


class LegacyPathFeedbackStore(InMemoryFeedbackStore):
    def __init__(
        self,
        path: Path,
        items: list[Feedback] | None = None,
    ) -> None:
        super().__init__(items)
        self.path = path
