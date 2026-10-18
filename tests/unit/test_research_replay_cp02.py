from __future__ import annotations

import asyncio
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from pms.research.specs import (
    BacktestDataset,
    BacktestExecutionConfig,
    BacktestSpec,
    ExecutionModel,
    RiskPolicy,
)


def _write_replay_jsonl(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "event_type": "trade",
                "market_id": "market-cancel",
                "price": 0.42,
                "sequence": 1,
                "ts": "2026-03-01T00:00:00Z",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _spec(*, market_id: str = "market-cancel") -> BacktestSpec:
    dataset = BacktestDataset(
        source="postgresql",
        version="outer-ring-v1",
        coverage_start=datetime(2026, 3, 1, tzinfo=UTC),
        coverage_end=datetime(2026, 3, 31, 23, 59, tzinfo=UTC),
        market_universe_filter={"venue": "polymarket", "market_ids": (market_id,)},
        data_quality_gaps=(),
    )
    return BacktestSpec(
        strategy_versions=(("alpha", "v1"),),
        dataset=dataset,
        execution_model=ExecutionModel.polymarket_paper(),
        risk_policy=RiskPolicy(
            max_position_notional_usdc=100.0,
            max_daily_drawdown_pct=2.5,
            min_order_size_usdc=1.0,
        ),
        date_range_start=datetime(2026, 3, 1, tzinfo=UTC),
        date_range_end=datetime(2026, 3, 31, 23, 59, tzinfo=UTC),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_trade", [None, 0.9])
async def test_book_replay_uses_midpoint_before_last_trade(
    tmp_path: Path, prior_trade: float | None,
) -> None:
    from pms.research.replay import MarketUniverseReplayEngine

    rows: list[dict[str, object]] = []
    if prior_trade is not None:
        rows.append({
            "event_type": "trade", "market_id": "market-cancel", "price": prior_trade,
            "sequence": 1, "ts": "2026-03-01T00:00:00Z",
        })
    rows.append({
        "event_type": "book_snapshot", "market_id": "market-cancel",
        "sequence": 2, "ts": "2026-03-01T00:00:01Z",
        "bids": [{"price": 0.4, "size": 10.0}],
        "asks": [{"price": 0.6, "size": 10.0}],
    })
    path = tmp_path / "book.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    engine = MarketUniverseReplayEngine.from_jsonl(path)
    signals = [s async for s in engine.stream(_spec(), BacktestExecutionConfig())]
    assert signals[-1].yes_price == pytest.approx(0.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("token_ids_on_first_row", [True, False])
@pytest.mark.parametrize("no_price", [0.2, 0.002])
async def test_no_token_replay_is_canonicalized_before_forecasting(
    tmp_path: Path, token_ids_on_first_row: bool, no_price: float,
) -> None:
    from pms.config import PMSSettings
    from pms.core.enums import RunMode
    from pms.core.models import MarketSignal, Portfolio
    from pms.controller.pipeline import ControllerPipeline
    from pms.research.replay import MarketUniverseReplayEngine

    class Forecaster:
        def predict(self, signal: MarketSignal) -> tuple[float, float, str]:
            assert signal.yes_price == pytest.approx(1 - no_price)
            return (1 - no_price - 0.13, 0.9, "YES probability")

        async def forecast(self, signal: MarketSignal) -> float:
            return self.predict(signal)[0]

    path = tmp_path / "no-token.jsonl"
    token_ids = {"yes_token_id": "token-yes", "no_token_id": "token-no"}
    rows = [
        {"event_type": "trade", "market_id": "market-cancel", "token_id": "token-no",
         "price": no_price, "sequence": 1, "ts": "2026-03-01T00:00:00Z",
         **(token_ids if token_ids_on_first_row else {})},
        {"event_type": "trade", "market_id": "market-cancel", "token_id": "token-no",
         "price": no_price, "sequence": 2, "ts": "2026-03-01T00:00:01Z", **token_ids},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    engine = MarketUniverseReplayEngine.from_jsonl(path)
    signals = [s async for s in engine.stream(_spec(), BacktestExecutionConfig())]
    pipeline = ControllerPipeline(
        forecasters=[Forecaster()], settings=PMSSettings(mode=RunMode.BACKTEST),
    )
    emission = await pipeline.on_signal(
        signals[0],
        portfolio=Portfolio(total_usdc=1000.0, free_usdc=1000.0, locked_usdc=0.0, open_positions=[]),
    )
    assert emission is not None, pipeline.last_diagnostic
    _, decision = emission
    assert decision.outcome == "NO"
    assert decision.token_id == "token-no"
    assert decision.limit_price == pytest.approx(no_price)
    assert decision.prob_estimate == pytest.approx(no_price + 0.13)


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_trade", [None, 0.7])
async def test_book_replay_missing_reference_is_not_a_zero_probability(
    tmp_path: Path, prior_trade: float | None,
) -> None:
    from pms.controller.router import Router
    from pms.research.replay import MarketUniverseReplayEngine

    rows: list[dict[str, object]] = []
    if prior_trade is not None:
        rows.append({
            "event_type": "trade", "market_id": "market-cancel", "price": prior_trade,
            "sequence": 1, "ts": "2026-03-01T00:00:00Z",
        })
    rows.append({
        "event_type": "book_snapshot", "market_id": "market-cancel",
        "sequence": 2, "ts": "2026-03-01T00:00:01Z",
        "bids": [], "asks": [{"price": 0.6, "size": 10.0}],
    })
    path = tmp_path / "book.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    engine = MarketUniverseReplayEngine.from_jsonl(path)
    signals = [s async for s in engine.stream(_spec(), BacktestExecutionConfig())]
    signal = signals[-1]
    if prior_trade is None:
        assert math.isnan(signal.yes_price)
        assert Router(allow_extreme_prices=True).gate_reason(signal) == "yes_price_out_of_band"
    else:
        assert signal.yes_price == prior_trade
    book = await engine.book_state_at(signal.timestamp, market_id=signal.market_id, token_id=signal.token_id)
    assert book["asks"] == [{"price": 0.6, "size": 10.0}]


class _CancellationConnection:
    def __init__(self) -> None:
        self._fetch_calls = 0

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        del query, args
        self._fetch_calls += 1
        if self._fetch_calls == 1:
            return [
                {
                    "condition_id": "market-cancel",
                    "question": "Will CP02 replay cancel cleanly?",
                    "venue": "polymarket",
                    "resolves_at": None,
                }
            ]
        raise asyncio.CancelledError


class _LeakTrackingPool:
    def __init__(self, connection: _CancellationConnection) -> None:
        self._connection = connection
        self.acquire_count = 0
        self.release_count = 0
        self.currently_acquired = 0

    async def acquire(self) -> _CancellationConnection:
        self.acquire_count += 1
        self.currently_acquired += 1
        return self._connection

    async def release(self, connection: _CancellationConnection) -> None:
        assert connection is self._connection
        self.release_count += 1
        self.currently_acquired -= 1


@pytest.mark.asyncio
async def test_market_universe_replay_engine_releases_pool_connection_on_cancellation() -> None:
    from pms.research.replay import MarketUniverseReplayEngine

    pool = _LeakTrackingPool(_CancellationConnection())
    engine = MarketUniverseReplayEngine(pool=cast(Any, pool))

    with pytest.raises(asyncio.CancelledError):
        async for _ in engine.stream(_spec(), BacktestExecutionConfig(chunk_days=7)):
            pass

    assert pool.acquire_count == pool.release_count
    assert pool.currently_acquired == 0


class _EmptyMetadataConnection:
    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        del query, args
        return []


class _EmptyMetadataPool:
    def __init__(self) -> None:
        self._connection = _EmptyMetadataConnection()

    async def acquire(self) -> _EmptyMetadataConnection:
        return self._connection

    async def release(self, connection: _EmptyMetadataConnection) -> None:
        del connection


@pytest.mark.asyncio
async def test_market_universe_replay_engine_rejects_empty_filter_match() -> None:
    from pms.research.replay import MarketUniverseReplayEngine

    engine = MarketUniverseReplayEngine(pool=cast(Any, _EmptyMetadataPool()))

    with pytest.raises(ValueError, match="matched zero markets"):
        async for _ in engine.stream(_spec(), BacktestExecutionConfig(chunk_days=7)):
            pass


def test_parse_jsonl_events_rejects_symlink_file(tmp_path: Path) -> None:
    from pms.research.replay import _parse_jsonl_events

    target_path = tmp_path / "target-replay.jsonl"
    _write_replay_jsonl(target_path)
    path = tmp_path / "replay.jsonl"
    path.symlink_to(target_path)

    with pytest.raises(ValueError, match="replay JSONL cannot be read safely"):
        _parse_jsonl_events(path)


def test_parse_jsonl_events_opens_file_with_no_follow_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pms.research.replay import _parse_jsonl_events

    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    if no_follow_flag == 0:
        pytest.skip("os.O_NOFOLLOW is unavailable on this platform")

    path = tmp_path / "replay.jsonl"
    _write_replay_jsonl(path)
    observed: list[tuple[Path, int]] = []
    real_open = os.open

    def recording_open(
        path_arg: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        observed.append((Path(os.fsdecode(os.fspath(path_arg))), flags))
        return real_open(path_arg, flags, mode)

    monkeypatch.setattr(os, "open", recording_open)

    metadata, events = _parse_jsonl_events(path)

    observed_by_path = {observed_path: flags for observed_path, flags in observed}
    assert set(metadata) == {"market-cancel"}
    assert len(events) == 1
    assert observed_by_path[path] & no_follow_flag


def test_parse_jsonl_events_rejects_hardlink_swap_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pms.research.replay import _parse_jsonl_events

    path = tmp_path / "replay.jsonl"
    _write_replay_jsonl(path)
    replacement_source = tmp_path / "replacement-replay.jsonl"
    _write_replay_jsonl(replacement_source)
    real_open = os.open
    swapped = False

    def swapping_open(
        path_arg: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        nonlocal swapped
        observed_path = Path(os.fsdecode(os.fspath(path_arg)))
        if observed_path == path and not swapped:
            swapped = True
            path.unlink()
            os.link(replacement_source, path)
        return real_open(path_arg, flags, mode)

    monkeypatch.setattr(os, "open", swapping_open)

    with pytest.raises(ValueError, match="replay JSONL cannot be read safely"):
        _parse_jsonl_events(path)

    assert swapped is True
