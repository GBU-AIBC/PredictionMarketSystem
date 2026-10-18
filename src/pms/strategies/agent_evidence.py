"""Point-in-time strategy projection for shared agent evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pms.storage.agent_evidence_store import AgentEvidenceStore
from pms.storage.agent_evidence_store import AgentEvidence
from pms.strategies.intents import StrategyContext, StrategyObservation


@dataclass(frozen=True, slots=True)
class AgentEvidenceObservationSource:
    """Expose only current evidence for the bounded market scope.

    A strategy passes its market scope in ``StrategyContext.metadata`` under
    ``market_ids`` (or ``market_id`` for a one-market run). Evidence rows stay
    outer-ring records and receive strategy identity only in this projection.
    """

    store: AgentEvidenceStore
    market_ids: Sequence[str] = ()

    async def read_evidence(
        self,
        market_id: str,
        *,
        as_of: datetime,
        limit: int = 8,
    ) -> Sequence[AgentEvidence]:
        configured = set(self.market_ids)
        if configured and market_id not in configured:
            return ()
        return await self.store.list_for_market(market_id, as_of=as_of, limit=limit)

    async def observe(self, context: StrategyContext) -> Sequence[StrategyObservation]:
        configured = tuple(self.market_ids)
        metadata_ids = context.metadata.get("market_ids")
        if isinstance(metadata_ids, Sequence) and not isinstance(metadata_ids, (str, bytes)):
            requested = tuple(str(market_id) for market_id in metadata_ids)
            configured = (
                tuple(market_id for market_id in requested if market_id in configured)
                if configured
                else requested
            )
        single_market = context.metadata.get("market_id")
        if isinstance(single_market, str) and single_market:
            configured = (
                (single_market,)
                if not configured or single_market in configured
                else ()
            )
        observations: list[StrategyObservation] = []
        for market_id in dict.fromkeys(configured):
            for evidence in await self.read_evidence(market_id, as_of=context.as_of):
                observations.append(
                    StrategyObservation(
                        observation_id=evidence.evidence_id,
                        strategy_id=context.strategy_id,
                        strategy_version_id=context.strategy_version_id,
                        source=evidence.source,
                        observed_at=evidence.captured_at,
                        summary=evidence.summary,
                        payload={
                            "market_id": evidence.market_id,
                            "model": evidence.model,
                            "url": evidence.url,
                            "payload": dict(evidence.payload),
                            "content_hash": evidence.content_hash,
                            "expires_at": (
                                None
                                if evidence.expires_at is None
                                else evidence.expires_at.isoformat()
                            ),
                        },
                        evidence_refs=(evidence.evidence_id,),
                    )
                )
        return tuple(observations)
