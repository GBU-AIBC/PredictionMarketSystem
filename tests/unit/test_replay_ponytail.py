from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Any, cast

import pytest

from pms.config import ControllerSettings, PMSSettings, RiskSettings
from pms.controller.calibrators.netcal import NetcalCalibrator
from pms.controller.pipeline import ControllerPipeline
from pms.controller.router import Router
from pms.controller.sizers.kelly import KellySizer
from pms.core.enums import RunMode
from pms.research.replay import MarketUniverseReplayEngine
from pms.research.specs import BacktestExecutionConfig
from tests.unit.test_controller_cp02 import (
    BearishForecaster, VeryBullishForecaster, _best_ask_strategy, _portfolio,
)
from tests.unit.test_research_replay_cp02 import _spec


START = datetime(2026, 3, 1, tzinfo=UTC)


def _row(sequence: int, ts: datetime, token: str, ask: float = 0.41) -> dict[str, object]:
    return {
        "event_type": "book_snapshot", "market_id": "market-cancel",
        "token_id": token, "yes_token_id": "token-yes", "no_token_id": "token-no",
        "ts": ts.isoformat(), "sequence": sequence,
        "bids": [{"price": ask - 0.01, "size": 1000.0}],
        "asks": [{"price": ask, "size": 1000.0}],
    }


def _engine(path: Path, rows: list[dict[str, object]]) -> MarketUniverseReplayEngine:
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return MarketUniverseReplayEngine.from_jsonl(path)


@pytest.mark.asyncio
async def test_lookup_before_first_checkpoint_never_returns_future(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "future.jsonl", [_row(1, START + timedelta(seconds=10), "token-no")])
    _ = [s async for s in engine.stream(_spec(), BacktestExecutionConfig())]
    with pytest.raises(LookupError):
        await engine.signal_at(START, market_id="market-cancel", token_id="token-no")


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["token-no", "token-yes"])
async def test_cross_outcome_best_ask_uses_replay_clock_and_history(
    tmp_path: Path, target: str,
) -> None:
    source = "token-yes" if target == "token-no" else "token-no"
    engine = _engine(tmp_path / "direct.jsonl", [
        _row(1, START, target),
        _row(2, START + timedelta(seconds=1), source, 0.59),
        _row(3, START + timedelta(seconds=2), target, 0.80),
    ])
    signals = [s async for s in engine.stream(_spec(), BacktestExecutionConfig())]
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(),
        forecasters=[BearishForecaster() if target == "token-no" else VeryBullishForecaster()],
        calibrator=NetcalCalibrator(),
        sizer=KellySizer(risk=RiskSettings(max_position_per_market=500.0)),
        router=Router(ControllerSettings(max_spread_bps=1000.0), allow_extreme_prices=True),
        historical_book_reader=engine,
        settings=PMSSettings(mode=RunMode.BACKTEST, controller=ControllerSettings(max_book_age_ms=1500.0)),
    )
    emission = await pipeline.on_signal(signals[1], portfolio=_portfolio())
    assert emission is not None, pipeline.last_diagnostic
    assert emission[1].token_id == target
    assert emission[1].limit_price == pytest.approx(0.41)
    stale = replace(signals[1], fetched_at=START + timedelta(seconds=1.6))
    assert await pipeline.on_signal(stale, portfolio=_portfolio()) is None
    assert pipeline.last_diagnostic is not None
    assert pipeline.last_diagnostic.metadata["direct_book_failure"] == "stale"
    missing = await pipeline._read_direct_outcome_execution_signal(
        replace(signals[1], fetched_at=START - timedelta(seconds=1)), token_id=target,
        outcome="NO" if target == "token-no" else "YES",
    )
    assert missing.signal is None and missing.failure == "snapshot_missing"


@pytest.mark.asyncio
async def test_chunk_stream_is_lazy_repeatable_and_lookup_retention_is_bounded(tmp_path: Path) -> None:
    rows = [_row(day + 1, START + timedelta(days=day), "token-yes", 0.41 + day / 1000) for day in range(20)]
    engine = _engine(tmp_path / "chunks.jsonl", rows)
    chunks: list[datetime] = []
    engine.chunk_observer = lambda start, end: chunks.append(start)
    config = BacktestExecutionConfig(chunk_days=1)
    stream = engine.stream(_spec(), config)
    first = await anext(stream)
    assert chunks == [START]
    # Latency can look several chunks ahead without advancing the strategy stream.
    future = await engine.signal_at(START + timedelta(days=3), market_id=first.market_id, token_id=first.token_id)
    assert future.orderbook["asks"][0]["price"] == pytest.approx(0.413)
    assert (await anext(stream)).fetched_at == START + timedelta(days=1)
    # A query behind the lookahead cursor must still recover the old book.
    past = await engine.signal_at(START, market_id=first.market_id, token_id=first.token_id)
    assert past.orderbook == first.orderbook
    rest = [s async for s in stream]
    for day in range(20):
        await engine.signal_at(START + timedelta(days=day), market_id=first.market_id, token_id=first.token_id)
        assert sum(len(items) for items in engine._signal_history.values()) <= 2
    replay = [s async for s in engine.stream(_spec(), config)]
    assert [first.fetched_at, START + timedelta(days=1), *(s.fetched_at for s in rest)] == [s.fetched_at for s in replay]
    assert replay[0].orderbook == first.orderbook
    await engine.aclose()


@pytest.mark.asyncio
async def test_backtest_never_uses_live_direct_reader() -> None:
    from tests.unit.test_controller_cp02 import _signal

    class ForbiddenReader:
        async def read_latest_snapshot(self, *args: object) -> None:
            pytest.fail("BACKTEST fetched a current venue/store book")

    pipeline = ControllerPipeline(
        settings=PMSSettings(mode=RunMode.BACKTEST),
        direct_book_reader=cast(Any, ForbiddenReader()),
    )
    result = await pipeline._read_direct_outcome_execution_signal(
        _signal(), token_id="token-no", outcome="NO",
    )
    assert result.signal is None
    assert result.failure == "reader_unavailable"


@pytest.mark.asyncio
async def test_zero_trade_run_completes_with_persisted_rejections_without_stale_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pms.controller.diagnostics import ControllerDiagnostic
    from pms.core.models import MarketSignal, Portfolio
    from pms.research.runner import BacktestRunner, _ClaimedRun, _StrategyAccumulator
    from tests.unit.test_research_runner_cp04b import _ReplayEngine, _active_strategy, _signal, _spec as runner_spec

    class RejectingPipeline:
        last_diagnostic: ControllerDiagnostic | None = None
        calls = 0

        async def on_signal(self, signal: MarketSignal, portfolio: Portfolio | None = None) -> None:
            self.calls += 1
            if self.calls != 2:
                self.last_diagnostic = ControllerDiagnostic(
                    code="missing_required_factor", message="missing", market_id=signal.market_id,
                    strategy_id="alpha", strategy_version_id="alpha-v1", token_id=signal.token_id,
                )

    class Factory:
        def build(self, strategy: object) -> RejectingPipeline:
            return RejectingPipeline()

    persisted: list[tuple[object, ...]] = []
    completed: list[str] = []

    async def claim(self: BacktestRunner, run_id: str) -> _ClaimedRun:
        return _ClaimedRun(run_id, runner_spec(), BacktestExecutionConfig())

    async def persist(self: BacktestRunner, *, run_id: str, accumulators: list[_StrategyAccumulator]) -> None:
        persisted.extend(a.as_insert_args(run_id=run_id) for a in accumulators)

    async def complete(self: BacktestRunner, run_id: str) -> None:
        completed.append(run_id)

    async def load(keys: object) -> list[Any]:
        return [_active_strategy()]

    monkeypatch.setattr(BacktestRunner, "_claim_run", claim)
    monkeypatch.setattr(BacktestRunner, "_insert_strategy_runs_atomically", persist)
    monkeypatch.setattr(BacktestRunner, "_mark_completed", complete)
    runner = BacktestRunner(
        writable_pool=cast(Any, object()), readonly_pool=cast(Any, object()),
        replay_engine=_ReplayEngine([_signal(START + timedelta(seconds=i)) for i in range(3)]),
        controller_factory=Factory(), strategy_loader=load,
    )
    assert await runner.execute("run-zero")
    assert completed == ["run-zero"]
    assert persisted[0][10:12] == (0, 0)
    assert json.loads(cast(str, persisted[0][-1])) == {"missing_required_factor": 2}


def test_rejection_counts_are_exposed_in_existing_strategy_result_payload() -> None:
    from pms.api.research_routes import _record_to_json

    assert _record_to_json({"controller_rejection_counts": '{"no_positive_edge":3}'}) == {
        "controller_rejection_counts": {"no_positive_edge": 3},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("gtc", [False, True])
async def test_real_runner_cross_chunk_latency_gtc_and_two_strategies(
    tmp_path: Path, gtc: bool,
) -> None:
    from pms.core.enums import TimeInForce
    from pms.core.models import MarketSignal, Opportunity, Portfolio, TradeDecision
    from pms.research.runner import BacktestRunner
    from pms.research.specs import ExecutionModel
    from pms.strategies.projections import ActiveStrategy
    from tests.unit.test_research_runner_cp04b import _PortfolioRecordingPipeline, _active_strategy

    trigger_at = START + timedelta(days=1, seconds=-1)
    engine = _engine(tmp_path / "latency.jsonl", [
        _row(1, START, "token-no", 0.51),
        _row(2, trigger_at, "token-yes", 0.59),
        _row(3, START + timedelta(days=1, seconds=1), "token-no", 0.51),
        _row(4, START + timedelta(days=2), "token-no", 0.41),
    ])

    class Pipeline(_PortfolioRecordingPipeline):
        async def on_signal(
            self, signal: MarketSignal, portfolio: Portfolio | None = None,
        ) -> tuple[Opportunity, TradeDecision] | None:
            if signal.fetched_at != trigger_at:
                return None
            emission = await super().on_signal(signal, portfolio)
            assert emission is not None
            opportunity, decision = emission
            return opportunity, replace(
                decision, token_id="token-no", outcome="NO",
                limit_price=0.45 if gtc else 0.55,
                time_in_force=TimeInForce.GTC if gtc else TimeInForce.IOC,
            )

    class Factory:
        def build(self, strategy: ActiveStrategy) -> Pipeline:
            return Pipeline()

    runner = BacktestRunner(
        writable_pool=cast(Any, object()), readonly_pool=cast(Any, object()),
        controller_factory=Factory(), replay_engine=engine,
    )
    model = replace(
        ExecutionModel.polymarket_paper(), latency_ms=2000.0,
        fill_policy="good_til_cancelled" if gtc else "immediate_or_cancel",
        order_ttl_ms=3 * 86_400_000, replay_window_ms=3 * 86_400_000,
    )
    results = []
    for strategy_id in ("alpha", "beta"):
        accumulator = await runner._run_strategy(
            strategy=replace(_active_strategy(), strategy_id=strategy_id,
                config=replace(_active_strategy().config, strategy_id=strategy_id)),
            spec=replace(_spec(), execution_model=model),
            exec_config=BacktestExecutionConfig(chunk_days=1),
        )
        assert accumulator.decision_count == accumulator.fill_count == 1
        assert accumulator.execution_rows[0].status == "filled"
        results.append(accumulator.as_insert_args(run_id="test")[4:13])
    assert results[0] == results[1]


@pytest.mark.asyncio
async def test_chunk_lookup_keeps_absent_token_anchor_and_same_timestamp_order(tmp_path: Path) -> None:
    ts = START + timedelta(days=2)
    engine = _engine(tmp_path / "ordering.jsonl", [
        _row(1, START, "token-no"),
        {"event_type": "trade", "market_id": "market-cancel", "token_id": "token-yes",
         "ts": ts.isoformat(), "sequence": 2, "price": 0.7},
        _row(3, ts, "token-yes", 0.51),
        {"event_type": "level_update", "market_id": "market-cancel", "token_id": "token-yes",
         "ts": ts.isoformat(), "sequence": 4, "side": "ask", "price": 0.41, "size": 10.0},
    ])
    signals = [s async for s in engine.stream(_spec(), BacktestExecutionConfig(chunk_days=1))]
    assert [s.external_signal["raw_event_type"] for s in signals] == [
        "book", "last_trade_price", "book", "price_change",
    ]
    no = await engine.signal_at(ts, market_id="market-cancel", token_id="token-no")
    assert no.fetched_at == START
    yes = await engine.signal_at(ts, market_id="market-cancel", token_id="token-yes")
    assert yes.yes_price == pytest.approx(0.41)
    assert yes.orderbook["asks"][0]["price"] == pytest.approx(0.41)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_runner_failure_and_cancellation_close_both_replay_cursors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool,
) -> None:
    import asyncio
    import os
    from pms.core.models import MarketSignal, Portfolio
    from pms.research.runner import BacktestRunner
    from tests.unit.test_research_runner_cp04b import _active_strategy

    path = tmp_path / "handles.jsonl"
    engine = _engine(path, [_row(day + 1, START + timedelta(days=day), "token-yes") for day in range(3)])
    opened: list[int] = []
    real_open = os.open

    def recording_open(path_arg: Any, flags: int, mode: int = 0o777) -> int:
        fd = real_open(path_arg, flags, mode)
        if Path(path_arg) == path:
            opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", recording_open)
    ready = asyncio.Event()

    class Pipeline:
        async def on_signal(self, signal: MarketSignal, portfolio: Portfolio | None = None) -> None:
            await engine.signal_at(signal.fetched_at, market_id=signal.market_id, token_id=signal.token_id)
            ready.set()
            if cancel:
                await asyncio.Event().wait()
            raise RuntimeError("test pipeline failure")

    class Factory:
        def build(self, strategy: object) -> Pipeline:
            return Pipeline()

    runner = BacktestRunner(
        writable_pool=cast(Any, object()), readonly_pool=cast(Any, object()),
        controller_factory=Factory(), replay_engine=engine,
    )
    task = asyncio.create_task(runner._run_strategy(
        strategy=_active_strategy(), spec=_spec(), exec_config=BacktestExecutionConfig(chunk_days=1),
    ))
    await ready.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await task
    assert len(opened) >= 3
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert not engine._signal_history
    assert engine._lookup_chunks is None
    # Standalone and backward queries also have an explicit release boundary.
    await engine.signal_at(START + timedelta(days=2), market_id="market-cancel", token_id="token-yes")
    await engine.signal_at(START, market_id="market-cancel", token_id="token-yes")
    await engine.aclose()
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.asyncio
async def test_runtime_historical_reader_is_as_of_and_disables_venue_refresh(tmp_path: Path) -> None:
    from pms.controller.factory import ControllerPipelineFactory
    from pms.runner import _controller_direct_book_reader
    from pms.sensor.adapters.historical import HistoricalSensor
    from tests.unit.test_sensor_historical import _write_jsonl_signal

    path = tmp_path / "runtime.jsonl"
    _write_jsonl_signal(path)
    reader = HistoricalSensor(path)
    settings = PMSSettings(mode=RunMode.BACKTEST, controller=ControllerSettings(venue_book_refresh_enabled=True))
    assert _controller_direct_book_reader(settings, cast(Any, object())) is None
    factory = ControllerPipelineFactory(settings=settings, historical_book_reader=reader)
    assert factory.build(_best_ask_strategy()).historical_book_reader is reader
    with pytest.raises(LookupError):
        await reader.signal_at(START, market_id="m1", token_id="t1")
    signal = await reader.signal_at(datetime(2026, 4, 14, tzinfo=UTC), market_id="m1", token_id="t1")
    assert signal.fetched_at == datetime(2026, 4, 13, 0, 0, 1, tzinfo=UTC)


@pytest.mark.asyncio
async def test_trade_does_not_refresh_stale_selected_token_book(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "stale-trade.jsonl", [
        _row(1, START, "token-yes"),
        {"event_type": "trade", "market_id": "market-cancel", "token_id": "token-yes",
         "ts": (START + timedelta(seconds=2)).isoformat(), "sequence": 2, "price": 0.5},
    ])
    signals = [s async for s in engine.stream(_spec(), BacktestExecutionConfig())]
    assert signals[-1].external_signal["book_received_at"] == START.isoformat()
    pipeline = ControllerPipeline(
        strategy=_best_ask_strategy(), forecasters=[VeryBullishForecaster()],
        historical_book_reader=engine,
        settings=PMSSettings(mode=RunMode.BACKTEST, controller=ControllerSettings(max_book_age_ms=1500)),
    )
    assert await pipeline.on_signal(signals[-1], portfolio=_portfolio()) is None
    assert pipeline.last_diagnostic is not None
    assert pipeline.last_diagnostic.metadata["direct_book_failure"] == "stale"


@pytest.mark.asyncio
async def test_postgres_source_fetches_only_consumed_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pms.research.replay import _ChunkWindow, _MarketMetadata, _ReplayEvent, _iter_jsonl_events

    path = tmp_path / "pg-events.jsonl"
    _engine(path, [_row(day + 1, START + timedelta(days=day), "token-yes") for day in range(20)])
    events = list(_iter_jsonl_events(path, {}))
    loaded: list[datetime] = []

    async def metadata(self: object, market_universe_filter: object) -> dict[str, _MarketMetadata]:
        return {"market-cancel": _MarketMetadata("test", "polymarket", None)}

    async def chunk(self: object, *, window: _ChunkWindow, market_ids: object) -> list[_ReplayEvent]:
        loaded.append(window.start)
        return [event for event in events if window.start <= event.ts < window.end]

    monkeypatch.setattr(MarketUniverseReplayEngine, "_load_market_metadata", metadata)
    monkeypatch.setattr(MarketUniverseReplayEngine, "_load_chunk_events", chunk)
    engine = MarketUniverseReplayEngine(pool=cast(Any, object()))
    stream = engine.stream(_spec(), BacktestExecutionConfig(chunk_days=1))
    await anext(stream)
    assert loaded == [START]
    await stream.aclose()


@pytest.mark.asyncio
async def test_rejection_counts_are_in_atomic_strategy_insert(monkeypatch: pytest.MonkeyPatch) -> None:
    from pms.research.runner import BacktestRunner, _StrategyAccumulator
    from tests.unit.test_research_runner_cp04b import _InsertTrackingConnection, _InsertTrackingPool

    writes: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self: object, query: str, *args: object) -> str:
        writes.append((query, args))
        return "INSERT 1"

    monkeypatch.setattr(_InsertTrackingConnection, "execute", execute)
    runner = BacktestRunner(writable_pool=cast(Any, _InsertTrackingPool([])), readonly_pool=cast(Any, object()))
    accumulator = _StrategyAccumulator("alpha", "v1", _spec().execution_model)
    accumulator.controller_rejection_counts["no_positive_edge"] = 3
    await runner._insert_strategy_runs_atomically(run_id="test", accumulators=[accumulator])
    assert "controller_rejection_counts_json" in writes[0][0]
    assert json.loads(cast(str, writes[0][1][-1])) == {"no_positive_edge": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_side", ["bids", "asks"])
async def test_historical_direct_book_preserves_two_sided_depth_requirement(
    tmp_path: Path, missing_side: str,
) -> None:
    target = _row(1, START, "token-no")
    target[missing_side] = []
    engine = _engine(tmp_path / "missing-depth.jsonl", [target, _row(2, START, "token-yes")])
    signals = [s async for s in engine.stream(_spec(), BacktestExecutionConfig())]
    pipeline = ControllerPipeline(settings=PMSSettings(mode=RunMode.BACKTEST), historical_book_reader=engine)
    result = await pipeline._read_direct_outcome_execution_signal(signals[-1], token_id="token-no", outcome="NO")
    assert result.signal is None and result.failure == "levels_missing"
