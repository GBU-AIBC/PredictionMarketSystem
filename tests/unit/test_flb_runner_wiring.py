from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from pms.config import (
    ControllerSettings,
    PMSSettings,
    RiskSettings,
    StrategyRuntimeSettings,
)
from pms.core.enums import RunMode, TimeInForce
from pms.core.models import TradeDecision
from pms.runner import Runner
from pms.strategies.flb import FlbAgent, FlbController, FlbStrategyModule, LiveFlbSource
from pms.strategies.flb.projection import build_h1_flb_strategy
from pms.strategies.flb.source import FlbMarketSnapshot
from pms.strategies.ripple.source import build_ripple_strategy
from pms.strategies.anchoring.source import build_anchoring_strategy
from pms.strategies.intents import TradeIntent
from pms.strategies.versioning import compute_strategy_version_id


NOW = datetime(2026, 5, 3, 15, 0, tzinfo=UTC)


class _StaticFlbMarketReader:
    def __init__(self, market: FlbMarketSnapshot | None) -> None:
        self.market = market
        self.calls: list[tuple[str, datetime]] = []

    async def latest(
        self,
        market_id: str,
        *,
        as_of: datetime,
    ) -> FlbMarketSnapshot | None:
        self.calls.append((market_id, as_of))
        return self.market


class _Row(dict[str, object]):
    def __getattr__(self, item: str) -> object:
        return self[item]


class _RecordingFlbConnection:
    def __init__(
        self,
        *,
        resolved_market_id: str | None = "market-flb-1",
        market_row: _Row | None = None,
        price_row: _Row | None = None,
    ) -> None:
        self.resolved_market_id = resolved_market_id
        self.market_row = market_row
        self.price_row = price_row
        self.fetchval_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchval(self, query: str, *args: object) -> object | None:
        self.fetchval_calls.append((query, args))
        return self.resolved_market_id

    async def fetchrow(self, query: str, *args: object) -> _Row | None:
        self.fetchrow_calls.append((query, args))
        if "FROM market_price_snapshots" in query:
            return self.price_row
        return self.market_row


class _AcquireFlbConnection:
    def __init__(self, connection: _RecordingFlbConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _RecordingFlbConnection:
        return self._connection

    async def __aexit__(self, *_: object) -> None:
        return None


class _FlbPool:
    def __init__(self, connection: _RecordingFlbConnection) -> None:
        self.connection = connection

    def acquire(self) -> _AcquireFlbConnection:
        return _AcquireFlbConnection(self.connection)


def _settings() -> PMSSettings:
    return PMSSettings(
        mode=RunMode.PAPER,
        risk=RiskSettings(max_position_per_market=5.0),
        strategies=StrategyRuntimeSettings(
            flb_entry_execution_cost_bps=0.0,
            flb_fee_rate=0.0,
        ),
    )


def _write_flb_calibration(
    path: Path,
    *,
    longshot_probability: float = 0.99,
) -> Path:
    path.write_text(
        "\n".join(
            (
                "signal_name,probability_estimate,sample_count,source_label",
                (
                    "longshot_yes_overpriced_buy_no,"
                    f"{longshot_probability},150,warehouse-flb-v1"
                ),
                "favorite_yes_underpriced_buy_yes,0.97,151,warehouse-flb-v1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    Path(f"{path}.provenance.json").write_text(
        json.dumps(
            {
                "artifact_type": "flb_calibration_provenance",
                "generated_by": "scripts/flb_data_feasibility.py",
                "source": "warehouse-csv",
                "generated_at": "2026-06-01T00:00:00+00:00",
                "warehouse_csv_sha256": sha256(
                    b"unit warehouse provenance fixture"
                ).hexdigest(),
                "warehouse_market_count": 301,
                "warehouse_longshot_count": 150,
                "warehouse_favorite_count": 151,
                "calibration_csv_sha256": sha256(path.read_bytes()).hexdigest(),
                "calibration_source_label": "warehouse-flb-v1",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_category_prior(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "market_id,category,yes_payout,no_payout,resolved_at",
                "m-1,politics,1,0,2026-01-01T00:00:00Z",
                "m-2,politics,0,1,2026-01-02T00:00:00Z",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _market(**overrides: object) -> FlbMarketSnapshot:
    data = {
        "market_id": "market-flb-1",
        "title": "Will H1 FLB production wiring emit the contrarian side?",
        "yes_token_id": "token-yes",
        "no_token_id": "token-no",
        "yes_price": 0.05,
        "observed_at": NOW,
        "yes_best_ask": 0.05,
        "no_best_ask": 0.96,
        "resolves_at": NOW + timedelta(days=5),
    }
    data.update(overrides)
    return FlbMarketSnapshot(**data)  # type: ignore[arg-type]


def _market_row(**overrides: object) -> _Row:
    data: dict[str, object] = {
        "market_id": "market-flb-1",
        "question": "Will H1 FLB production wiring emit the contrarian side?",
        "venue": "polymarket",
        "resolves_at": NOW + timedelta(days=5),
        "last_seen_at": NOW - timedelta(minutes=2),
        "yes_price": 0.03,
        "no_price": 0.97,
        "best_bid": 0.02,
        "best_ask": 0.04,
        "price_updated_at": NOW - timedelta(minutes=2),
        "closed": False,
        "accepting_orders": True,
        "status_updated_at": NOW - timedelta(days=1),
        "yes_token_id": "token-yes",
        "no_token_id": "token-no",
    }
    data.update(overrides)
    return _Row(data)


def _price_row(**overrides: object) -> _Row:
    data: dict[str, object] = {
        "snapshot_at": NOW - timedelta(minutes=10),
        "yes_price": 0.05,
        "no_price": 0.95,
        "best_bid": 0.04,
        "best_ask": 0.06,
    }
    data.update(overrides)
    return _Row(data)


def test_runner_builds_flb_strategy_module_with_live_components() -> None:
    runner = Runner(config=_settings())
    reader = _StaticFlbMarketReader(_market())

    module = runner.build_flb_strategy_module(
        strategy_id="h1_flb",
        strategy_version_id="h1-flb-v1",
        market_ids=("market-flb-1",),
        market_reader=reader,
    )

    assert isinstance(module, FlbStrategyModule)
    assert isinstance(module.source, LiveFlbSource)
    assert isinstance(module.controller, FlbController)
    assert isinstance(module.agent, FlbAgent)
    assert module.strategy_id == "h1_flb"
    assert module.strategy_version_id == "h1-flb-v1"
    assert module.source.confidence == pytest.approx(0.65)
    assert getattr(module.agent, "min_confidence") == pytest.approx(0.60)
    position_sizer_risk = getattr(module.source.position_sizer, "risk")
    assert position_sizer_risk is not None
    assert position_sizer_risk.max_position_per_market == pytest.approx(1.0)


def test_runner_flb_module_uses_versioned_time_in_force() -> None:
    settings = PMSSettings(
        mode=RunMode.PAPER,
        controller=ControllerSettings(time_in_force="IOC"),
    )
    runner = Runner(config=settings)
    strategy = build_h1_flb_strategy(time_in_force="IOC")
    version_id = compute_strategy_version_id(*strategy.snapshot())

    module = runner.build_flb_strategy_module(
        strategy_id="h1_flb",
        strategy_version_id=version_id,
        market_ids=("market-flb-1",),
        market_reader=_StaticFlbMarketReader(_market()),
    )

    assert getattr(module.source, "time_in_force").value == "IOC"


def test_runner_flb_module_rejects_mismatched_strategy_version() -> None:
    runner = Runner(config=_settings())

    with pytest.raises(ValueError, match="runtime strategy projection"):
        runner.build_flb_strategy_module(
            strategy_id="h1_flb",
            strategy_version_id="mismatched-version",
            market_ids=("market-flb-1",),
            market_reader=_StaticFlbMarketReader(_market()),
        )


def test_paper_fill_fee_uses_versioned_strategy_fee() -> None:
    strategy = build_h1_flb_strategy()
    version_id = compute_strategy_version_id(*strategy.snapshot())
    runner = Runner(
        config=PMSSettings(
            strategies=StrategyRuntimeSettings(flb_fee_rate=0.25),
        )
    )
    runner._active_strategies_by_version[("h1_flb", version_id)] = strategy.to_active(
        strategy_version_id=version_id
    )
    decision = TradeDecision(
        decision_id="decision-fee",
        market_id="market-fee",
        token_id="token-fee",
        venue="polymarket",
        side="BUY",
        limit_price=0.5,
        notional_usdc=10.0,
        order_type="limit",
        max_slippage_bps=50,
        stop_conditions=[],
        prob_estimate=0.7,
        expected_edge=0.2,
        time_in_force=TimeInForce.GTC,
        opportunity_id="opportunity-fee",
        strategy_id="h1_flb",
        strategy_version_id=version_id,
        action="BUY",
    )

    assert runner._paper_fill_fee_rate(decision) == pytest.approx(0.07)


@pytest.mark.parametrize(
    ("builder", "strategy_id"),
    (
        (build_ripple_strategy, "ripple"),
        (build_anchoring_strategy, "h2_anchoring_lag"),
    ),
)
def test_paper_fill_fee_uses_versioned_fee_for_non_h1_strategies(
    builder: Any,
    strategy_id: str,
) -> None:
    strategy = builder(strategy_id)
    version_id = compute_strategy_version_id(*strategy.snapshot())
    runner = Runner(
        config=PMSSettings(
            strategies=StrategyRuntimeSettings(flb_fee_rate=0.25),
        )
    )
    runner._active_strategies_by_version[(strategy_id, version_id)] = strategy.to_active(
        strategy_version_id=version_id
    )
    decision = TradeDecision(
        decision_id=f"decision-fee-{strategy_id}",
        market_id="market-fee",
        token_id="token-fee",
        venue="polymarket",
        side="BUY",
        limit_price=0.5,
        notional_usdc=10.0,
        order_type="limit",
        max_slippage_bps=50,
        stop_conditions=[],
        prob_estimate=0.7,
        expected_edge=0.2,
        time_in_force=TimeInForce.GTC,
        opportunity_id=f"opportunity-fee-{strategy_id}",
        strategy_id=strategy_id,
        strategy_version_id=version_id,
        action="BUY",
    )

    assert runner._paper_fill_fee_rate(decision) == pytest.approx(0.07)


def test_runner_rejects_configured_flb_calibration_without_provenance(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "flb-calibration.csv"
    model_path.write_text(
        "\n".join(
            (
                "signal_name,probability_estimate,sample_count,source_label",
                "longshot_yes_overpriced_buy_no,0.99,150,warehouse-flb-v1",
                "favorite_yes_underpriced_buy_yes,0.97,151,warehouse-flb-v1",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="FLB calibration provenance JSON"):
        Runner(
            config=PMSSettings(
                mode=RunMode.PAPER,
                risk=RiskSettings(max_position_per_market=5.0),
                strategies=StrategyRuntimeSettings(
                    flb_calibration_path=str(model_path),
                ),
            ),
        )


def test_runner_rejects_configured_flb_calibration_inside_working_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    artifact_dir = repo_dir / "private"
    artifact_dir.mkdir(mode=0o700)
    model_path = _write_flb_calibration(artifact_dir / "flb-calibration.csv")
    monkeypatch.chdir(repo_dir)

    with pytest.raises(ValueError, match="outside the working tree"):
        Runner(
            config=PMSSettings(
                mode=RunMode.PAPER,
                risk=RiskSettings(max_position_per_market=5.0),
                strategies=StrategyRuntimeSettings(
                    flb_calibration_path=str(model_path),
                ),
            ),
        )


def test_runner_rejects_configured_flb_calibration_permissive_parent(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "permissive"
    artifact_dir.mkdir()
    artifact_dir.chmod(0o755)
    model_path = _write_flb_calibration(artifact_dir / "flb-calibration.csv")

    with pytest.raises(ValueError, match="too permissive"):
        Runner(
            config=PMSSettings(
                mode=RunMode.PAPER,
                risk=RiskSettings(max_position_per_market=5.0),
                strategies=StrategyRuntimeSettings(
                    flb_calibration_path=str(model_path),
                ),
            ),
        )


def test_runner_rejects_configured_category_prior_inside_working_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    artifact_dir = repo_dir / "private"
    artifact_dir.mkdir(mode=0o700)
    category_prior_path = _write_category_prior(artifact_dir / "category-prior.csv")
    monkeypatch.chdir(repo_dir)

    with pytest.raises(ValueError, match="outside the working tree"):
        Runner(
            config=PMSSettings(
                mode=RunMode.PAPER,
                risk=RiskSettings(max_position_per_market=5.0),
                controller=ControllerSettings(
                    category_prior_observations_path=str(category_prior_path),
                    category_prior_min_global_samples=2,
                ),
            ),
        )


def test_runner_rejects_configured_category_prior_permissive_parent(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "permissive"
    artifact_dir.mkdir()
    artifact_dir.chmod(0o755)
    category_prior_path = _write_category_prior(artifact_dir / "category-prior.csv")

    with pytest.raises(ValueError, match="too permissive"):
        Runner(
            config=PMSSettings(
                mode=RunMode.PAPER,
                risk=RiskSettings(max_position_per_market=5.0),
                controller=ControllerSettings(
                    category_prior_observations_path=str(category_prior_path),
                    category_prior_min_global_samples=2,
                ),
            ),
        )


@pytest.mark.asyncio
async def test_runner_calls_flb_module_without_live_capital() -> None:
    runner = Runner(config=_settings())
    reader = _StaticFlbMarketReader(_market())

    results = await runner.run_flb_strategy_once(
        strategy_id="h1_flb",
        strategy_version_id="h1-flb-v1",
        market_ids=("market-flb-1",),
        as_of=NOW,
        market_reader=reader,
    )

    assert reader.calls == [("market-flb-1", NOW)]
    # H1's hashed economics include the 7% fee and 15 bps execution cost;
    # the gross 2c edge is therefore correctly rejected.
    assert results == ()

    assert runner.state.decisions == []
    assert runner._decision_queue.empty()  # noqa: SLF001


@pytest.mark.asyncio
async def test_runner_wires_configured_flb_calibration_model(tmp_path: Path) -> None:
    model_path = _write_flb_calibration(tmp_path / "flb-calibration.csv")
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.PAPER,
            risk=RiskSettings(max_position_per_market=5.0),
            strategies=StrategyRuntimeSettings(
                flb_calibration_path=str(model_path),
            ),
        ),
    )
    reader = _StaticFlbMarketReader(_market())

    results = await runner.run_flb_strategy_once(
        strategy_id="h1_flb",
        strategy_version_id="h1-flb-v1",
        market_ids=("market-flb-1",),
        as_of=NOW,
        market_reader=reader,
    )

    assert len(results) == 1
    intent = results[0].intents[0]
    assert isinstance(intent, TradeIntent)
    assert intent.expected_price == pytest.approx(0.99)


@pytest.mark.asyncio
async def test_runner_wires_configured_flb_entry_costs(tmp_path: Path) -> None:
    model_path = _write_flb_calibration(
        tmp_path / "flb-calibration.csv",
        longshot_probability=0.985,
    )
    runner = Runner(
        config=PMSSettings(
            mode=RunMode.PAPER,
            risk=RiskSettings(max_position_per_market=5.0),
            strategies=StrategyRuntimeSettings(
                flb_calibration_path=str(model_path),
                flb_entry_execution_cost_bps=50.0,
                flb_fee_rate=0.04,
            ),
        ),
    )
    reader = _StaticFlbMarketReader(_market())

    results = await runner.run_flb_strategy_once(
        strategy_id="h1_flb",
        strategy_version_id="h1-flb-v1",
        market_ids=("market-flb-1",),
        as_of=NOW,
        market_reader=reader,
    )

    assert len(results) == 1
    assert len(results[0].intents) == 1
    module = runner.build_flb_strategy_module(
        strategy_id="h1_flb",
        strategy_version_id="h1-flb-v1",
        market_ids=("market-flb-1",),
        market_reader=reader,
    )
    assert isinstance(module.source, LiveFlbSource)
    assert module.source.entry_execution_cost_bps == pytest.approx(15.0)
    assert module.source.fee_rate == pytest.approx(0.07)


def test_runner_flb_module_requires_explicit_markets_without_pg_pool() -> None:
    runner = Runner(config=_settings())

    with pytest.raises(ValueError, match="market_ids"):
        runner.build_flb_strategy_module(
            strategy_id="h1_flb",
            strategy_version_id="h1-flb-v1",
            market_ids=(),
            market_reader=_StaticFlbMarketReader(None),
        )


@pytest.mark.asyncio
async def test_postgres_flb_reader_uses_as_of_price_snapshot() -> None:
    runner = Runner(config=_settings())
    connection = _RecordingFlbConnection(
        market_row=_market_row(yes_price=0.03),
        price_row=_price_row(yes_price=0.08, best_bid=0.07, best_ask=0.09),
    )
    runner.bind_pg_pool(_FlbPool(connection))

    results = await runner.run_flb_strategy_once(
        strategy_id="h1_flb",
        strategy_version_id="h1-flb-v1",
        market_ids=("token-yes",),
        as_of=NOW,
    )

    assert results == ()
    assert connection.fetchval_calls[0][1] == ("token-yes",)
    assert connection.fetchrow_calls[1][1] == ("market-flb-1", NOW)
    assert "ORDER BY match_rank ASC" in connection.fetchval_calls[0][0]
    assert "snapshot_at <= $2" in connection.fetchrow_calls[1][0]


@pytest.mark.asyncio
async def test_postgres_flb_reader_rejects_future_status_change() -> None:
    runner = Runner(config=_settings())
    connection = _RecordingFlbConnection(
        market_row=_market_row(
            closed=True,
            status_updated_at=NOW + timedelta(minutes=1),
        ),
        price_row=_price_row(),
    )
    runner.bind_pg_pool(_FlbPool(connection))

    results = await runner.run_flb_strategy_once(
        strategy_id="h1_flb",
        strategy_version_id="h1-flb-v1",
        market_ids=("market-flb-1",),
        as_of=NOW,
    )

    assert results == ()
