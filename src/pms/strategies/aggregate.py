"""Strategy aggregate that owns immutable projection state."""

from __future__ import annotations

from typing import TypeVar

from pms.factors.composition import validate_composition

from .projections import (
    ActiveStrategy,
    CalibrationSpec,
    EvalSpec,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
)


ProjectionT = TypeVar("ProjectionT")


class Strategy:
    __slots__ = (
        "_config",
        "_risk",
        "_eval_spec",
        "_forecaster",
        "_market_selection",
        "_calibration",
    )

    def __init__(
        self,
        *,
        config: StrategyConfig,
        risk: RiskParams,
        eval_spec: EvalSpec,
        forecaster: ForecasterSpec,
        market_selection: MarketSelectionSpec,
        calibration: CalibrationSpec | None = None,
    ) -> None:
        self._config = self._require_projection("config", config)
        validate_composition(self._config.factor_composition)
        self._risk = self._require_projection("risk", risk)
        self._eval_spec = self._require_projection("eval_spec", eval_spec)
        self._forecaster = self._require_projection("forecaster", forecaster)
        self._market_selection = self._require_projection(
            "market_selection",
            market_selection,
        )
        self._calibration = (
            CalibrationSpec()
            if calibration is None
            else self._require_projection("calibration", calibration)
        )

    @staticmethod
    def _require_projection(
        field_name: str,
        value: ProjectionT | None,
    ) -> ProjectionT:
        if value is None:
            raise TypeError(f"{field_name} must not be None")
        return value

    @property
    def config(self) -> StrategyConfig:
        return self._config

    @property
    def risk(self) -> RiskParams:
        return self._risk

    @property
    def eval_spec(self) -> EvalSpec:
        return self._eval_spec

    @property
    def forecaster(self) -> ForecasterSpec:
        return self._forecaster

    @property
    def market_selection(self) -> MarketSelectionSpec:
        return self._market_selection

    @property
    def calibration(self) -> CalibrationSpec:
        return self._calibration

    def snapshot(
        self,
    ) -> tuple[
        StrategyConfig,
        RiskParams,
        EvalSpec,
        ForecasterSpec,
        MarketSelectionSpec,
        CalibrationSpec,
    ]:
        return (
            self._config,
            self._risk,
            self._eval_spec,
            self._forecaster,
            self._market_selection,
            self._calibration,
        )

    def to_active(self, *, strategy_version_id: str) -> ActiveStrategy:
        return ActiveStrategy(
            strategy_id=self.config.strategy_id,
            strategy_version_id=strategy_version_id,
            config=self.config,
            risk=self.risk,
            eval_spec=self.eval_spec,
            forecaster=self.forecaster,
            market_selection=self.market_selection,
            calibration=self.calibration,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Strategy):
            return NotImplemented
        return self.snapshot() == other.snapshot()

    def __hash__(self) -> int:
        return hash(self.snapshot())
