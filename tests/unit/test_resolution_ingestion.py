from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from pms.config import DatabaseSettings, PMSSettings, RiskSettings
from pms.core.enums import OrderStatus, RunMode, Side, TimeInForce
from pms.core.models import EvalRecord, FillRecord, MarketSignal, TradeDecision
from pms.evaluation.adapters.scoring import Scorer
from pms.evaluation.resolution import (
    GammaResolutionSource,
    ResolutionSweeper,
    ResolutionSweepResult,
)
from pms.evaluation.spool import EvalSpool
from pms.runner import Runner
from pms.storage.eval_store import EvalStore
from pms.storage.fill_store import FillStore
from tests.support.fake_stores import InMemoryEvalStore


FIXTURE_PATH = Path("tests/fixtures/polymarket_7day_synthetic.jsonl")


def _decision(
    *,
    decision_id: str = "d-res-1",
    market_id: str = "0xmarket-1",
    prob: float = 0.7,
    price: float = 0.4,
    strategy_id: str = "default",
    strategy_version_id: str = "default-v1",
) -> TradeDecision:
    return TradeDecision(
        decision_id=decision_id,
        market_id=market_id,
        token_id="t-yes",
        venue="polymarket",
        side=Side.BUY.value,
        limit_price=price,
        notional_usdc=price * 10.0,
        order_type="limit",
        max_slippage_bps=100,
        stop_conditions=["min_volume:100.00"],
        prob_estimate=prob,
        expected_edge=prob - price,
        time_in_force=TimeInForce.GTC,
        opportunity_id=f"op-{decision_id}",
        strategy_id=strategy_id,
        strategy_version_id=strategy_version_id,
        model_id="model-a",
    )


def _fill(
    *,
    fill_id: str = "fill-res-1",
    decision_id: str = "d-res-1",
    market_id: str = "0xmarket-1",
    resolved_outcome: float | None = None,
    strategy_id: str = "default",
    strategy_version_id: str = "default-v1",
) -> FillRecord:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    return FillRecord(
        trade_id=f"trade-{fill_id}",
        fill_id=fill_id,
        order_id=f"order-{fill_id}",
        decision_id=decision_id,
        market_id=market_id,
        token_id="t-yes",
        venue="polymarket",
        side=Side.BUY.value,
        fill_price=0.42,
        fill_notional_usdc=4.2,
        fill_quantity=10.0,
        executed_at=now,
        filled_at=now,
        status=OrderStatus.MATCHED.value,
        anomaly_flags=[],
        strategy_id=strategy_id,
        strategy_version_id=strategy_version_id,
        resolved_outcome=resolved_outcome,
    )


class FakeResolutionFillStore:
    def __init__(
        self,
        fills: Sequence[FillRecord],
        *,
        call_log: list[tuple[str, str]] | None = None,
    ) -> None:
        self.fills: dict[str, FillRecord] = {
            fill.fill_id or fill.trade_id: fill for fill in fills
        }
        self.resolve_calls: list[tuple[str, float]] = []
        self.call_log = call_log if call_log is not None else []

    async def read_unresolved_fills(self) -> list[FillRecord]:
        return [
            fill for fill in self.fills.values() if fill.resolved_outcome is None
        ]

    async def resolve_fill(self, fill_id: str, *, resolved_outcome: float) -> bool:
        self.resolve_calls.append((fill_id, resolved_outcome))
        self.call_log.append(("resolve", fill_id))
        fill = self.fills.get(fill_id)
        if fill is None or fill.resolved_outcome is not None:
            return False
        self.fills[fill_id] = replace(fill, resolved_outcome=resolved_outcome)
        return True


class AlreadyResolvedElsewhereFillStore(FakeResolutionFillStore):
    """Simulates a concurrent writer winning the resolved-outcome update."""

    async def resolve_fill(self, fill_id: str, *, resolved_outcome: float) -> bool:
        self.resolve_calls.append((fill_id, resolved_outcome))
        return False


@dataclass(frozen=True)
class _StoredDecision:
    decision: TradeDecision
    decision_evidence: Mapping[str, Any]


class FakeDecisionReader:
    def __init__(self, rows: Mapping[str, _StoredDecision]) -> None:
        self._rows = dict(rows)

    async def get_decision(self, decision_id: str) -> _StoredDecision | None:
        return self._rows.get(decision_id)


class FakeResolutionSource:
    def __init__(self, resolutions: Mapping[str, float]) -> None:
        self.resolutions = dict(resolutions)
        self.calls: list[list[str]] = []

    async def fetch_resolutions(
        self,
        condition_ids: Sequence[str],
    ) -> Mapping[str, float]:
        self.calls.append(list(condition_ids))
        return dict(self.resolutions)


class FailingResolutionStore:
    available = False

    async def upsert(self, *_args: Any, **_kwargs: Any) -> None:
        if not self.available:
            raise RuntimeError("resolution persistence unavailable")


class RecordingEvalSpool:
    def __init__(
        self,
        *,
        call_log: list[tuple[str, str]] | None = None,
    ) -> None:
        self.processed: list[
            tuple[FillRecord, TradeDecision, Mapping[str, object] | None]
        ] = []
        self.call_log = call_log if call_log is not None else []

    async def process(
        self,
        fill: FillRecord,
        decision: TradeDecision,
        *,
        decision_evidence: Mapping[str, object] | None = None,
    ) -> bool:
        self.processed.append((fill, decision, decision_evidence))
        self.call_log.append(("process", fill.fill_id or fill.trade_id))
        return True


class _RecordingFeedbackGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, metrics_by_strategy: object) -> None:
        del metrics_by_strategy
        self.calls += 1


def _sweeper(
    *,
    fill_store: FakeResolutionFillStore,
    eval_spool: RecordingEvalSpool | EvalSpool,
    source: FakeResolutionSource,
    decisions: Mapping[str, _StoredDecision],
) -> ResolutionSweeper:
    return ResolutionSweeper(
        fill_store=fill_store,
        decision_reader=FakeDecisionReader(decisions),
        eval_spool=eval_spool,
        resolution_source=source,
    )


@pytest.mark.asyncio
async def test_sweep_routes_resolved_fill_through_eval_spool_and_hooks() -> None:
    """Swept resolutions must flow through EvalSpool.process — the exact
    scoring path the live queue worker uses — so post-append hooks
    (EvaluatorFeedback today, any future post-append sink) fire for
    PAPER/LIVE eval records, which only ever arrive via the sweep. The
    spool worker task is deliberately NOT started: swept evidence must
    commit independently of the queue worker's liveness."""
    fill_store = FakeResolutionFillStore([_fill()])
    eval_store = InMemoryEvalStore()
    feedback_generator = _RecordingFeedbackGenerator()

    async def metrics_provider() -> Any:
        return {("default", "default-v1"): ("snapshot", "spec")}

    spool = EvalSpool(
        store=cast(EvalStore, eval_store),
        scorer=Scorer(),
        feedback_generator=cast(Any, feedback_generator),
        metrics_provider=cast(Any, metrics_provider),
    )
    source = FakeResolutionSource({"0xmarket-1": 1.0})
    sweeper = _sweeper(
        fill_store=fill_store,
        eval_spool=spool,
        source=source,
        decisions={
            "d-res-1": _StoredDecision(
                decision=_decision(),
                decision_evidence={
                    "mid_quote_baseline_prob_estimate": 0.45,
                    "last_trade_baseline_prob_estimate": 0.43,
                },
            )
        },
    )

    result = await sweeper.sweep_once()

    assert result == ResolutionSweepResult(
        unresolved_fills=1,
        fills_resolved=1,
        eval_records_appended=1,
        fills_failed=0,
    )
    assert fill_store.resolve_calls == [("fill-res-1", 1.0)]
    assert fill_store.fills["fill-res-1"].resolved_outcome == 1.0
    records = await eval_store.all()
    assert len(records) == 1
    record = records[0]
    assert record.decision_id == "d-res-1"
    assert record.market_id == "0xmarket-1"
    assert record.resolved_outcome == 1.0
    assert record.brier_score == pytest.approx((0.7 - 1.0) ** 2)
    # Baseline-Brier inputs are recovered from persisted decision evidence,
    # passed through enqueue exactly as the live fill path does.
    assert record.baseline_prob_estimates == {
        "market_implied": 0.4,
        "mid_quote": 0.45,
        "last_trade": 0.43,
    }
    assert record.baseline_brier_scores["mid_quote"] == pytest.approx(
        (0.45 - 1.0) ** 2
    )
    assert record.citations == ["trade-fill-res-1"]
    # The post-append hook fired: the Evaluator->Controller feedback edge
    # stays live for swept records.
    assert feedback_generator.calls == 1


@pytest.mark.asyncio
async def test_resolution_persistence_failure_keeps_fill_retryable() -> None:
    fill_store = FakeResolutionFillStore([_fill()])
    spool = RecordingEvalSpool()
    source = FakeResolutionSource({"0xmarket-1": 1.0})
    sweeper = _sweeper(
        fill_store=fill_store,
        eval_spool=spool,
        source=source,
        decisions={
            "d-res-1": _StoredDecision(
                decision=_decision(),
                decision_evidence={},
            )
        },
    )
    resolution_store = FailingResolutionStore()
    sweeper.resolution_store = resolution_store
    result = await sweeper.sweep_once()
    assert result.persistence_failures == 1
    assert result.fills_resolved == 0
    assert result.eval_records_appended == 0
    assert fill_store.fills["fill-res-1"].resolved_outcome is None
    assert spool.processed == []

    resolution_store.available = True
    retried = await sweeper.sweep_once()
    assert retried.persistence_failures == 0
    assert retried.fills_resolved == 1
    assert retried.eval_records_appended == 1
    assert source.calls == [["0xmarket-1"], ["0xmarket-1"]]
    assert fill_store.fills["fill-res-1"].resolved_outcome == 1.0


class _CallLogEvalStore(InMemoryEvalStore):
    """InMemoryEvalStore that records each committed append in a shared log."""

    def __init__(self, call_log: list[tuple[str, str]]) -> None:
        super().__init__()
        self._call_log = call_log

    async def append(self, record: EvalRecord) -> bool:
        self._call_log.append(("append", record.decision_id))
        return await super().append(record)


class _FailingEvalStore:
    def __init__(self) -> None:
        self.attempts = 0

    async def append(self, record: EvalRecord) -> bool:
        del record
        self.attempts += 1
        raise RuntimeError("transient eval store outage")


class _DuplicateEvalStore:
    def __init__(self) -> None:
        self.attempts = 0

    async def append(self, record: EvalRecord) -> bool:
        del record
        self.attempts += 1
        return False


@pytest.mark.asyncio
async def test_sweep_commits_eval_append_before_resolving_fill() -> None:
    """Ordering is the crash-safety contract: eval_store.append has
    COMMITTED (synchronously, via EvalSpool.process — no in-memory queue)
    before resolve_fill commits the irreversible NULL -> value transition.
    A crash in between leaves the fill unresolved and the next sweep
    retries it; the eval store's ON CONFLICT guard absorbs the duplicate
    append. Resolve-first would lose the record forever."""
    call_log: list[tuple[str, str]] = []
    fill_store = FakeResolutionFillStore([_fill()], call_log=call_log)
    eval_store = _CallLogEvalStore(call_log)
    spool = EvalSpool(store=cast(EvalStore, eval_store), scorer=Scorer())
    sweeper = _sweeper(
        fill_store=fill_store,
        eval_spool=spool,
        source=FakeResolutionSource({"0xmarket-1": 1.0}),
        decisions={
            "d-res-1": _StoredDecision(
                decision=_decision(),
                decision_evidence={"mid_quote_baseline_prob_estimate": 0.45},
            )
        },
    )

    await sweeper.sweep_once()

    assert call_log == [("append", "d-res-1"), ("resolve", "fill-res-1")]
    records = await eval_store.all()
    assert len(records) == 1
    assert records[0].decision_id == "d-res-1"
    assert records[0].resolved_outcome == 1.0
    # Decision evidence reached the scorer through process(), exactly as
    # the live fill path passes it.
    assert records[0].baseline_prob_estimates["mid_quote"] == 0.45


@pytest.mark.asyncio
async def test_failed_eval_append_leaves_fill_unresolved_for_retry() -> None:
    """Append-before-resolve is the durability guarantee: a store whose
    append fails must leave the fill UNRESOLVED so the next sweep retries
    it. The old enqueue-then-resolve order committed the fill transition
    while the eval record sat in an in-memory queue — one transient append
    error killed the spool worker and lost the record forever."""
    fill_store = FakeResolutionFillStore([_fill()])
    failing_store = _FailingEvalStore()
    spool = EvalSpool(store=cast(EvalStore, failing_store), scorer=Scorer())
    sweeper = _sweeper(
        fill_store=fill_store,
        eval_spool=spool,
        source=FakeResolutionSource({"0xmarket-1": 1.0}),
        decisions={
            "d-res-1": _StoredDecision(
                decision=_decision(),
                decision_evidence={},
            )
        },
    )

    result = await sweeper.sweep_once()

    assert fill_store.fills["fill-res-1"].resolved_outcome is None
    assert fill_store.resolve_calls == []
    assert failing_store.attempts == 1
    assert result.fills_resolved == 0
    assert result.eval_records_appended == 0
    assert result.fills_failed == 1


@pytest.mark.asyncio
async def test_duplicate_eval_append_does_not_increment_appended_count() -> None:
    """A retried sweep can hit the eval store's ON CONFLICT path. That is a
    durable success, but not a newly appended row, so the sweep result must
    not inflate eval_records_appended."""
    fill_store = FakeResolutionFillStore([_fill()])
    duplicate_store = _DuplicateEvalStore()
    spool = EvalSpool(store=cast(EvalStore, duplicate_store), scorer=Scorer())
    sweeper = _sweeper(
        fill_store=fill_store,
        eval_spool=spool,
        source=FakeResolutionSource({"0xmarket-1": 1.0}),
        decisions={
            "d-res-1": _StoredDecision(
                decision=_decision(),
                decision_evidence={},
            )
        },
    )

    result = await sweeper.sweep_once()

    assert duplicate_store.attempts == 1
    assert result == ResolutionSweepResult(
        unresolved_fills=1,
        fills_resolved=1,
        eval_records_appended=0,
        fills_failed=0,
    )


@pytest.mark.asyncio
async def test_sweep_leaves_fills_for_unresolved_markets_untouched() -> None:
    fill_store = FakeResolutionFillStore([_fill()])
    spool = RecordingEvalSpool()
    source = FakeResolutionSource({})
    sweeper = _sweeper(
        fill_store=fill_store,
        eval_spool=spool,
        source=source,
        decisions={
            "d-res-1": _StoredDecision(
                decision=_decision(),
                decision_evidence={},
            )
        },
    )

    result = await sweeper.sweep_once()

    assert result == ResolutionSweepResult(
        unresolved_fills=1,
        fills_resolved=0,
        eval_records_appended=0,
        fills_failed=0,
    )
    assert fill_store.resolve_calls == []
    assert fill_store.fills["fill-res-1"].resolved_outcome is None
    assert spool.processed == []
    assert source.calls == [["0xmarket-1"]]


@pytest.mark.asyncio
async def test_second_sweep_does_not_duplicate_eval_records() -> None:
    fill_store = FakeResolutionFillStore([_fill()])
    spool = RecordingEvalSpool()
    source = FakeResolutionSource({"0xmarket-1": 0.0})
    sweeper = _sweeper(
        fill_store=fill_store,
        eval_spool=spool,
        source=source,
        decisions={
            "d-res-1": _StoredDecision(
                decision=_decision(),
                decision_evidence={},
            )
        },
    )

    first = await sweeper.sweep_once()
    second = await sweeper.sweep_once()

    assert first.eval_records_appended == 1
    assert second == ResolutionSweepResult(
        unresolved_fills=0,
        fills_resolved=0,
        eval_records_appended=0,
        fills_failed=0,
    )
    assert len(spool.processed) == 1
    # The second sweep finds no unresolved fills, so Gamma is not re-polled.
    assert source.calls == [["0xmarket-1"]]


@pytest.mark.asyncio
async def test_sweep_appends_evidence_even_when_resolve_update_loses_race() -> None:
    """append-first means a lost resolve race still appends the record:
    dedup lives in the eval store's ON CONFLICT (decision_id) DO NOTHING,
    not in the fill UPDATE, so the append is safe to attempt either way."""
    fill_store = AlreadyResolvedElsewhereFillStore([_fill()])
    spool = RecordingEvalSpool()
    sweeper = _sweeper(
        fill_store=fill_store,
        eval_spool=spool,
        source=FakeResolutionSource({"0xmarket-1": 1.0}),
        decisions={
            "d-res-1": _StoredDecision(
                decision=_decision(),
                decision_evidence={},
            )
        },
    )

    result = await sweeper.sweep_once()

    assert result == ResolutionSweepResult(
        unresolved_fills=1,
        fills_resolved=0,
        eval_records_appended=1,
        fills_failed=0,
    )
    assert len(spool.processed) == 1


@pytest.mark.asyncio
async def test_sweep_resolves_fill_without_recoverable_decision_but_skips_eval() -> None:
    fill_store = FakeResolutionFillStore([_fill()])
    spool = RecordingEvalSpool()
    sweeper = _sweeper(
        fill_store=fill_store,
        eval_spool=spool,
        source=FakeResolutionSource({"0xmarket-1": 1.0}),
        decisions={},
    )

    result = await sweeper.sweep_once()

    assert result == ResolutionSweepResult(
        unresolved_fills=1,
        fills_resolved=1,
        eval_records_appended=0,
        fills_failed=0,
    )
    assert fill_store.fills["fill-res-1"].resolved_outcome == 1.0
    assert spool.processed == []


@pytest.mark.asyncio
async def test_sweep_leaves_unscorable_fill_unresolved_for_retry() -> None:
    """A deterministic scoring failure inside EvalSpool.process must NOT
    mark the fill resolved — that ordering lost the eval record forever.
    The fill stays unresolved (retried on every sweep, visible in
    unresolved counts and fills_failed), no eval record is appended for
    it, and the rest of the sweep proceeds."""
    fill_store = FakeResolutionFillStore(
        [
            _fill(fill_id="fill-bad", decision_id="d-bad", market_id="0xmarket-1"),
            _fill(fill_id="fill-good", decision_id="d-good", market_id="0xmarket-2"),
        ]
    )
    eval_store = InMemoryEvalStore()
    spool = EvalSpool(store=cast(EvalStore, eval_store), scorer=Scorer())
    sweeper = _sweeper(
        fill_store=fill_store,
        eval_spool=spool,
        source=FakeResolutionSource({"0xmarket-1": 1.0, "0xmarket-2": 1.0}),
        decisions={
            # Strategy identity mismatch makes Scorer.score raise for this fill.
            "d-bad": _StoredDecision(
                decision=_decision(
                    decision_id="d-bad",
                    strategy_id="other-strategy",
                ),
                decision_evidence={},
            ),
            "d-good": _StoredDecision(
                decision=_decision(
                    decision_id="d-good",
                    market_id="0xmarket-2",
                ),
                decision_evidence={},
            ),
        },
    )

    result = await sweeper.sweep_once()

    assert [fill_id for fill_id, _outcome in fill_store.resolve_calls] == [
        "fill-good"
    ]
    assert fill_store.fills["fill-bad"].resolved_outcome is None
    assert [
        record.decision_id for record in await eval_store.all()
    ] == ["d-good"]
    assert result.unresolved_fills == 2
    assert result.fills_resolved == 1
    assert result.eval_records_appended == 1
    assert result.fills_failed == 1

    # The unscorable fill is retried on the next sweep instead of being
    # silently dropped from the unresolved set.
    second = await sweeper.sweep_once()
    assert second.unresolved_fills == 1
    assert second.fills_resolved == 0


def _gamma_market_row(
    condition_id: str,
    *,
    closed: bool,
    outcomes: Sequence[str] = ("Yes", "No"),
    prices: Sequence[str] = ("1", "0"),
) -> dict[str, Any]:
    return {
        "conditionId": condition_id,
        "closed": closed,
        "outcomes": json.dumps(list(outcomes)),
        "outcomePrices": json.dumps(list(prices)),
    }


def _gamma_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://gamma-api.test",
    )


@pytest.mark.asyncio
async def test_gamma_resolution_source_parses_settled_closed_markets() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=[
                _gamma_market_row("0xresolved-yes", closed=True, prices=("1", "0")),
                _gamma_market_row(
                    "0xresolved-no-reversed",
                    closed=True,
                    outcomes=("No", "Yes"),
                    prices=("1", "0"),
                ),
                # Closed but not settled: near-resolution trade prices must
                # not be admitted as Brier ground truth.
                _gamma_market_row(
                    "0xclosed-unsettled",
                    closed=True,
                    prices=("0.97", "0.03"),
                ),
                _gamma_market_row("0xstill-open", closed=False, prices=("1", "0")),
                {
                    "conditionId": "0xnon-binary",
                    "closed": True,
                    "outcomes": json.dumps(["A", "B", "C"]),
                    "outcomePrices": json.dumps(["1", "0", "0"]),
                },
            ],
        )

    async with _gamma_client(handler) as client:
        source = GammaResolutionSource(http_client=client)
        resolutions = await source.fetch_resolutions(
            [
                "0xresolved-yes",
                "0xresolved-no-reversed",
                "0xclosed-unsettled",
                "0xstill-open",
                "0xnon-binary",
            ]
        )

    assert resolutions == {
        "0xresolved-yes": 1.0,
        "0xresolved-no-reversed": 0.0,
    }
    assert len(requests) == 1
    assert requests[0].url.params.get_list("condition_ids") == [
        "0xresolved-yes",
        "0xresolved-no-reversed",
        "0xclosed-unsettled",
        "0xstill-open",
        "0xnon-binary",
    ]


@pytest.mark.asyncio
async def test_gamma_resolution_source_retains_resolution_timestamp() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json=[
                {
                    **_gamma_market_row("0xresolved-at", closed=True),
                    "resolvedAt": "2026-03-01T00:00:00Z",
                }
            ],
        )

    async with _gamma_client(handler) as client:
        source = GammaResolutionSource(http_client=client)
        assert await source.fetch_resolutions(["0xresolved-at"]) == {
            "0xresolved-at": 1.0
        }
    assert source.resolved_at_by_market["0xresolved-at"] == datetime(
        2026, 3, 1, tzinfo=UTC
    )


@pytest.mark.asyncio
async def test_gamma_resolution_source_query_pins_closed_true_and_explicit_limit() -> None:
    """Pins the outbound query-string contract, not a MockTransport echo.

    Verified against live Gamma on 2026-06-10: batching closed markets'
    condition_ids WITHOUT closed=true returns 0 rows (each id alone returns
    its row), so production resolutions stay empty forever unless closed=true
    is sent. The explicit limit pins the page size to the batch instead of
    relying on the server default (measured: exactly 20) covering it.
    """
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    async with _gamma_client(handler) as client:
        source = GammaResolutionSource(http_client=client)
        await source.fetch_resolutions(["0xa", "0xb", "0xc"])

    assert len(requests) == 1
    query = requests[0].url.query.decode("ascii")
    assert "closed=true" in query
    assert "limit=3" in query
    assert query.count("condition_ids=") == 3


@pytest.mark.asyncio
async def test_gamma_resolution_source_parses_recorded_live_closed_batch() -> None:
    """Contract fixture recorded verbatim from live Gamma on 2026-06-10:

    GET /markets?condition_ids=<a>&condition_ids=<b>&closed=true&limit=2

    The same two condition_ids WITHOUT closed=true returned 0 rows, which is
    the contract gap this fixture pins. Both markets settled NO, so the real
    response shape exercises closed=True parsing, JSON-encoded outcome lists,
    and the reversed-payout (yes=0, no=1) branch.
    """
    payload = json.loads(
        Path("tests/fixtures/gamma_markets_closed_batch.json").read_text()
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    condition_ids = [
        "0x6b45806ccf9734807ac19f9d968c9a634ac7335161d15349c694aae45e901570",
        "0xed92e5fde4f1b6c6dc5bdfdb63941b08df7aa6310af29e19be95cf1c557fdce1",
    ]
    async with _gamma_client(handler) as client:
        source = GammaResolutionSource(http_client=client)
        resolutions = await source.fetch_resolutions(condition_ids)

    assert resolutions == {condition_id: 0.0 for condition_id in condition_ids}


@pytest.mark.asyncio
async def test_gamma_resolution_source_batches_condition_ids() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        rows = [
            _gamma_market_row(condition_id, closed=True)
            for condition_id in request.url.params.get_list("condition_ids")
        ]
        return httpx.Response(200, json=rows)

    async with _gamma_client(handler) as client:
        source = GammaResolutionSource(http_client=client, batch_size=2)
        resolutions = await source.fetch_resolutions(["0xa", "0xb", "0xc"])

    assert resolutions == {"0xa": 1.0, "0xb": 1.0, "0xc": 1.0}
    assert [request.url.params.get_list("condition_ids") for request in requests] == [
        ["0xa", "0xb"],
        ["0xc"],
    ]
    # Each request's explicit limit tracks its own batch size, so raising
    # batch_size above the server default page size cannot truncate rows.
    assert [request.url.params.get("limit") for request in requests] == ["2", "1"]


class _RecordingConnection:
    def __init__(self, *, execute_result: str = "INSERT 0 1") -> None:
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetch_rows: list[object] = []
        self.fetchrow_result: object | None = None
        self.execute_result = execute_result

    async def execute(self, query: str, *args: object) -> str:
        self.execute_calls.append((query, args))
        return self.execute_result

    async def fetch(self, query: str, *args: object) -> list[object]:
        self.fetch_calls.append((query, args))
        return list(self.fetch_rows)

    async def fetchrow(self, query: str, *args: object) -> object | None:
        self.fetchrow_calls.append((query, args))
        return self.fetchrow_result


class _AcquireContext:
    def __init__(self, connection: _RecordingConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _RecordingConnection:
        return self._connection

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb


class _RecordingPool:
    def __init__(self, connection: _RecordingConnection) -> None:
        self._connection = connection

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self._connection)


def _unresolved_fill_row() -> dict[str, object]:
    return {
        "fill_id": "fill-res-1",
        "order_id": "order-fill-res-1",
        "market_id": "0xmarket-1",
        "ts": datetime(2026, 6, 1, tzinfo=UTC),
        "fill_notional_usdc": 4.2,
        "fill_quantity": 10.0,
        "strategy_id": "default",
        "strategy_version_id": "default-v1",
        "payload": {
            "trade_id": "trade-fill-res-1",
            "decision_id": "d-res-1",
            "token_id": "t-yes",
            "venue": "polymarket",
            "side": "BUY",
            "fill_price": 0.42,
            "executed_at": "2026-06-01T00:00:00+00:00",
            "status": "MATCHED",
            "anomaly_flags": [],
            "resolved_outcome": None,
        },
    }


@pytest.mark.asyncio
async def test_eval_store_append_is_idempotent_on_decision_id_conflict() -> None:
    """eval_records.decision_id is a PRIMARY KEY; the conflict guard is the
    retry-safety contract for the resolution sweep (append-first ordering
    re-appends after a crash between append and resolve_fill) and keeps
    LIVE partial fills — multiple fills per decision — from raising
    UniqueViolation through the spool."""
    connection = _RecordingConnection(execute_result="INSERT 0 0")
    store = EvalStore(pool=cast(Any, _RecordingPool(connection)))

    inserted = await store.append(
        EvalRecord(
            market_id="0xmarket-1",
            decision_id="d-res-1",
            strategy_id="default",
            strategy_version_id="default-v1",
            prob_estimate=0.7,
            resolved_outcome=1.0,
            brier_score=0.09,
            fill_status=OrderStatus.MATCHED.value,
            recorded_at=datetime(2026, 6, 1, tzinfo=UTC),
            citations=["trade-fill-res-1"],
        )
    )

    (query, args) = connection.execute_calls[0]
    assert "ON CONFLICT (decision_id) DO NOTHING" in query
    assert args[0] == "d-res-1"
    assert inserted is False


@pytest.mark.asyncio
async def test_fill_store_read_unresolved_fills_filters_on_null_outcome() -> None:
    connection = _RecordingConnection()
    connection.fetch_rows = [_unresolved_fill_row()]
    store = FillStore(pool=cast(Any, _RecordingPool(connection)))

    fills = await store.read_unresolved_fills()

    assert len(fills) == 1
    assert fills[0].fill_id == "fill-res-1"
    assert fills[0].decision_id == "d-res-1"
    assert fills[0].resolved_outcome is None
    (query, args) = connection.fetch_calls[0]
    assert "payload->>'resolved_outcome' IS NULL" in query
    assert args == ()


@pytest.mark.asyncio
async def test_fill_store_resolve_fill_updates_payload_once() -> None:
    connection = _RecordingConnection()
    connection.fetchrow_result = {"fill_id": "fill-res-1"}
    store = FillStore(pool=cast(Any, _RecordingPool(connection)))

    updated = await store.resolve_fill("fill-res-1", resolved_outcome=1.0)

    assert updated is True
    (query, args) = connection.fetchrow_calls[0]
    assert "UPDATE fill_payloads" in query
    assert "jsonb_set" in query
    # The NULL guard is the idempotency contract: a second writer (or a
    # second sweep racing the first) must not re-resolve the same fill.
    assert "payload->>'resolved_outcome' IS NULL" in query
    assert args == ("fill-res-1", 1.0)


@pytest.mark.asyncio
async def test_fill_store_resolve_fill_returns_false_when_already_resolved() -> None:
    connection = _RecordingConnection()
    connection.fetchrow_result = None
    store = FillStore(pool=cast(Any, _RecordingPool(connection)))

    updated = await store.resolve_fill("fill-res-1", resolved_outcome=1.0)

    assert updated is False


class HoldingSensor:
    def __aiter__(self) -> AsyncIterator[MarketSignal]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MarketSignal]:
        while True:
            await asyncio.sleep(60.0)
            yield cast(MarketSignal, None)


def _paper_settings(**overrides: Any) -> PMSSettings:
    return PMSSettings(
        mode=RunMode.PAPER,
        auto_migrate_default_v2=False,
        risk=RiskSettings(
            max_position_per_market=1_000.0,
            max_total_exposure=10_000.0,
        ),
        **overrides,
    )


async def _wait_for(predicate: Any, *, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_paper_runner_starts_resolution_sweep_task_and_stop_clears_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner(
        config=_paper_settings(),
        historical_data_path=FIXTURE_PATH,
        sensors=[HoldingSensor()],
    )
    sweep_calls = 0

    async def fake_sweep_once() -> ResolutionSweepResult:
        nonlocal sweep_calls
        sweep_calls += 1
        return ResolutionSweepResult(
            unresolved_fills=0,
            fills_resolved=0,
            eval_records_appended=0,
            fills_failed=0,
        )

    monkeypatch.setattr(runner, "_resolution_sweep_once", fake_sweep_once)

    await runner.start()
    try:
        assert runner._resolution_sweep_task is not None
        await _wait_for(lambda: sweep_calls >= 1)
        await _wait_for(lambda: runner.resolution_sweeps_total >= 1)
    finally:
        await runner.stop()

    assert runner._resolution_sweep_task is None


@pytest.mark.asyncio
async def test_backtest_runner_does_not_start_resolution_sweep_task() -> None:
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.BACKTEST,
            auto_migrate_default_v2=False,
            database=DatabaseSettings(dsn="postgresql://localhost/pms_test_runner"),
        ),
        historical_data_path=FIXTURE_PATH,
        sensors=[HoldingSensor()],
    )

    await runner.start()
    try:
        assert runner.pg_pool is not None
        assert runner._resolution_sweep_task is None
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_resolution_sweep_loop_survives_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner(
        config=_paper_settings(resolution_poll_interval_s=0.01),
        historical_data_path=FIXTURE_PATH,
        sensors=[HoldingSensor()],
    )
    calls = 0

    async def flaky_sweep_once() -> ResolutionSweepResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("gamma transient boom")
        return ResolutionSweepResult(
            unresolved_fills=2,
            fills_resolved=1,
            eval_records_appended=1,
            fills_failed=1 if calls == 2 else 0,
            persistence_failures=1 if calls == 2 else 0,
        )

    monkeypatch.setattr(runner, "_resolution_sweep_once", flaky_sweep_once)

    loop_task = asyncio.create_task(runner._resolution_sweep_loop())
    await _wait_for(lambda: calls >= 2)
    runner._stop_event.set()
    await asyncio.wait_for(loop_task, timeout=2.0)

    assert calls >= 2
    assert runner.resolution_sweeps_total >= 1
    assert runner.resolution_fills_resolved_total >= 1
    # Failures must increment a counter so /status can distinguish "Gamma
    # down for a week" from healthy idle: whole-sweep exceptions count 1
    # each, and per-fill processing failures (fills_failed) add on top.
    assert runner.resolution_sweep_failures_total == 3


def test_resolution_poll_interval_defaults_to_five_minutes() -> None:
    assert PMSSettings(mode=RunMode.BACKTEST).resolution_poll_interval_s == 300.0
