from __future__ import annotations

import asyncio
import logging
from math import isfinite
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import inspect
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, TypeVar, cast

from pms.config import PMSSettings
from pms.controller._price_utils import best_ask, spread_bps_at_decision
from pms.controller.calibrators.extreme_clamp import ExtremeProbClamp
from pms.controller.calibrators.netcal import NetcalCalibrator
from pms.controller.calibrators.shrinkage import LogitShrinkageCalibrator
from pms.controller.diagnostics import ControllerDiagnostic, DiagnosticSeverity
from pms.controller.factor_snapshot import (
    FactorKey,
    FactorSnapshotReader,
    NullFactorSnapshotReader,
    required_factor_keys,
)
from pms.controller.forecasters.llm import LLMForecaster, durable_spend_ledger_path
from pms.controller.forecasters.rules import RulesForecaster
from pms.controller.forecasters.statistical import StatisticalForecaster
from pms.controller.outcome_tokens import (
    NullOutcomeTokenResolver,
    OutcomeTokens,
    OutcomeTokenResolver,
)
from pms.controller.router import Router
from pms.controller.sizers.kelly import KellySizer
from pms.core.enums import RunMode, TimeInForce
from pms.core.interfaces import ICalibrator, IForecaster, ISizer
from pms.core.models import (
    BookLevel,
    BookSnapshot,
    MarketSignal,
    Opportunity,
    Portfolio,
    TradeDecision,
)
from pms.execution.fees import market_fee_rate_from_metadata
from pms.factors.base import EMPTY_OUTER_RING
from pms.factors.composition import apply_composition, evaluate_branch_probabilities
from pms.factors.definitions.orderbook_imbalance import OrderbookImbalance
from pms.metrics import (
    SELECTION_FUNNEL_CONTROLLER_EMITTED_TOTAL_METRIC,
    SELECTION_FUNNEL_FORECASTED_TOTAL_METRIC,
    SELECTION_FUNNEL_ROUTED_TOTAL_METRIC,
    increment_metric,
)
from pms.strategies.projections import ActiveStrategy, CalibrationContext, CalibrationSpec
from pms.strategies.economics import (
    strategy_economic_float,
    strategy_economic_int,
    strategy_economic_str,
    strategy_metadata,
    strategy_risk_settings,
)

logger = logging.getLogger(__name__)

ForecastResult = tuple[float, float, str]
OpportunityEmission = tuple[Opportunity, TradeDecision]
T = TypeVar("T")
_ORIGINAL_RULES_PREDICT = RulesForecaster.predict
_ORIGINAL_STATISTICAL_PREDICT = StatisticalForecaster.predict


@dataclass(frozen=True, slots=True)
class _SignalOutcomeContext:
    yes_token_id: str | None
    no_token_id: str | None
    signal_outcome: Literal["YES", "NO"]


class DirectBookSnapshotReader(Protocol):
    async def read_latest_snapshot(
        self,
        market_id: str,
        token_id: str,
    ) -> BookSnapshot | None: ...

    async def read_levels_for_snapshot(self, snapshot_id: int) -> list[BookLevel]: ...


@dataclass(frozen=True, slots=True)
class _DirectBookRead:
    signal: MarketSignal | None
    failure: str | None = None
    age_ms: float | None = None


async def _read_direct_fee_rate_bps(
    reader: DirectBookSnapshotReader | None,
    *,
    market_id: str,
    token_id: str,
) -> float | None:
    if reader is None:
        return None
    fee_reader = getattr(reader, "read_fee_rate_bps", None)
    if not callable(fee_reader):
        return None
    typed_reader = cast(
        Callable[[str, str], Awaitable[float | None]],
        fee_reader,
    )
    try:
        fee_rate_bps = await typed_reader(market_id, token_id)
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "direct fee-rate read failed for %s/%s (%s): %s",
            market_id,
            token_id,
            type(error).__name__,
            str(error) or "(no message)",
        )
        return None
    if (
        fee_rate_bps is None
        or not isfinite(fee_rate_bps)
        or fee_rate_bps < 0.0
        or fee_rate_bps > 10_000.0
    ):
        return None
    return fee_rate_bps


@dataclass
class ControllerPipeline:
    strategy_id: str = "default"
    strategy_version_id: str = "default-v1"
    strategy: ActiveStrategy | None = None
    factor_reader: FactorSnapshotReader = field(
        default_factory=NullFactorSnapshotReader
    )
    outcome_token_resolver: OutcomeTokenResolver = field(
        default_factory=NullOutcomeTokenResolver
    )
    direct_book_reader: DirectBookSnapshotReader | None = None
    forecasters: Sequence[IForecaster] | None = None
    calibrator: ICalibrator | None = None
    sizer: ISizer | None = None
    router: Router | None = None
    settings: PMSSettings = field(default_factory=PMSSettings)
    last_diagnostic: ControllerDiagnostic | None = field(init=False, default=None)
    last_execution_signal: MarketSignal | None = field(init=False, default=None)
    suppressed_zero_size: int = field(init=False, default=0)
    _last_decision_emitted_at: dict[tuple[str, str, str, str, str], datetime] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if self.strategy is not None:
            self.strategy_id = self.strategy.strategy_id
            self.strategy_version_id = self.strategy.strategy_version_id
        if self.forecasters is None:
            llm_settings = self.settings.llm
            if llm_settings.spend_ledger_path is None:
                llm_settings = llm_settings.model_copy(
                    update={
                        "spend_ledger_path": durable_spend_ledger_path(
                            database_dsn=self.settings.database.dsn,
                            data_path=self.settings.live_emergency_audit_path,
                            scope=self.settings.mode.value,
                        )
                    }
                )
            self.forecasters = (
                RulesForecaster(),
                StatisticalForecaster(),
                LLMForecaster(config=llm_settings),
            )
        if self.calibrator is None:
            self.calibrator = NetcalCalibrator()
        if self.sizer is None:
            self.sizer = KellySizer(
                risk=strategy_risk_settings(
                    self.strategy,
                    fallback=self.settings.risk,
                )
            )
        if self.router is None:
            self.router = Router(self.settings.controller)

    def _set_drop_diagnostic(
        self,
        signal: MarketSignal,
        *,
        code: str,
        message: str,
        severity: DiagnosticSeverity = "info",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record (and log) a ControllerDiagnostic for a signal that the
        pipeline drops without emitting a decision.

        Every operationally-meaningful early return in ``on_signal`` calls this
        so an operator watching ``state.controller_diagnostics`` can tell *why*
        order flow stopped — distinguishing 'idle' from 'every signal filtered'.
        """
        self.last_diagnostic = ControllerDiagnostic(
            code=code,
            message=message,
            market_id=signal.market_id,
            strategy_id=self.strategy_id,
            strategy_version_id=self.strategy_version_id,
            token_id=signal.token_id,
            severity=severity,
            metadata=dict(metadata or {}),
        )
        logger.info(
            "controller diagnostic %s for %s",
            code,
            signal.market_id,
            extra={"controller_diagnostic": self.last_diagnostic},
        )

    async def on_signal(
        self,
        signal: MarketSignal,
        portfolio: Portfolio | None = None,
    ) -> OpportunityEmission | None:
        self.last_diagnostic = None
        self.last_execution_signal = None
        router = _required(self.router, "router")
        if signal.token_id is None:
            _log_pipeline_funnel(signal, forecasted_count=0, emitted_count=0)
            self.last_diagnostic = ControllerDiagnostic(
                code="missing_token_id",
                message=(
                    "Signal arrived without token_id; skipping decision until "
                    "the sensor populates the outcome token."
                ),
                market_id=signal.market_id,
                strategy_id=self.strategy_id,
                strategy_version_id=self.strategy_version_id,
                token_id=None,
                severity="info",
                metadata={},
            )
            return None

        signal_token_id = signal.token_id
        raw_signal = signal
        signal_outcome = _signal_outcome_context_from_signal(
            signal,
            signal_token_id=signal_token_id,
        )
        signal = _canonical_yes_signal(signal, signal_outcome)
        signal = _with_decision_time_book_age(signal, self.settings)
        if router.gate_reason(signal) == "book_too_stale":
            signal = await self._refresh_signal_book_before_router_gate(
                signal,
                outcome=signal_outcome.signal_outcome,
            )
        # Call gate() (rather than gate_reason() twice) so the router funnel
        # log is emitted exactly once per signal. gate() returns a bool; we
        # re-derive the specific reason only on failure so we can attach it
        # to the diagnostic.
        if not router.gate(signal):
            gate_reason = router.gate_reason(signal)
            _log_pipeline_funnel(signal, forecasted_count=0, emitted_count=0)
            gate_metadata: dict[str, object] = {
                "gate_reason": gate_reason or "unknown"
            }
            if gate_reason == "book_too_stale":
                gate_metadata["book_age_ms"] = signal.external_signal.get("book_age_ms")
                gate_metadata["max_book_age_ms"] = router.controller.max_book_age_ms
            self.last_diagnostic = ControllerDiagnostic(
                code=f"router_gate:{gate_reason}",
                message=(
                    "Signal rejected by router gate: "
                    f"{(gate_reason or 'unknown').replace('_', ' ')}."
                ),
                market_id=signal.market_id,
                strategy_id=self.strategy_id,
                strategy_version_id=self.strategy_version_id,
                token_id=signal.token_id,
                severity="info",
                metadata=gate_metadata,
            )
            return None

        active_portfolio = portfolio or _default_portfolio()
        sizer = _required(self.sizer, "sizer")
        min_order_usdc = _effective_min_order_usdc(
            strategy=self.strategy,
            sizer=sizer,
            settings=self.settings,
        )
        strategy_max_slippage_bps = strategy_economic_int(
            self.strategy,
            "max_slippage_bps",
            self.settings.controller.max_slippage_bps,
        )
        risk_group_id = _risk_group_id(signal)
        group_capacity_diagnostic = _risk_group_capacity_diagnostic(
            portfolio=active_portfolio,
            settings=self.settings,
            min_order_usdc=min_order_usdc,
            risk_group_id=risk_group_id,
        )
        if group_capacity_diagnostic is not None:
            code, metadata = group_capacity_diagnostic
            self._set_drop_diagnostic(
                signal,
                code=code,
                message=(
                    "Skipping decision before forecasting because the risk-group "
                    "capacity cannot accept another minimum-size order."
                ),
                metadata=metadata,
            )
            _log_pipeline_funnel(signal, forecasted_count=0, emitted_count=0)
            return None

        forecasters = _required(self.forecasters, "forecasters")
        tasks = [
            self._predict_forecaster(forecaster, signal)
            for forecaster in forecasters
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        probabilities: list[float] = []
        confidence_weights: list[float] = []
        llm_weight_flags: list[bool] = []
        model_ids: list[str] = []
        rationales: list[str] = []
        runtime_probabilities: dict[str, float] = {}
        forecaster_names = _strategy_forecaster_names(self.strategy)
        calibrator = _required(self.calibrator, "calibrator")
        for index, (forecaster, result) in enumerate(
            zip(forecasters, results, strict=True)
        ):
            if isinstance(result, BaseException):
                logger.warning("forecaster failed: %s", result)
                continue
            if result is None:
                if isinstance(forecaster, LLMForecaster):
                    failure_reason = forecaster.last_failure_reason or "no_output"
                    self._set_drop_diagnostic(
                        signal,
                        code=f"llm_{failure_reason}",
                        message=(
                            "LLM forecaster produced no usable forecast: "
                            f"{failure_reason}."
                        ),
                        severity="info",
                    )
                continue
            model_id = _model_id(result, forecaster)
            model_ids.append(model_id)
            probabilities.append(float(result[0]))
            confidence_weight = max(0.0, min(1.0, float(result[1])))
            if isinstance(forecaster, LLMForecaster):
                # Cap LLM influence at 25%; zero confidence means zero weight.
                confidence_weight = min(confidence_weight, 0.25)
            confidence_weights.append(confidence_weight)
            llm_weight_flags.append(isinstance(forecaster, LLMForecaster))
            if result[2]:
                rationales.append(f"{model_id}:{result[2]}")
            runtime_factor_id = _runtime_factor_id(
                forecaster_names[index] if index < len(forecaster_names) else None
            )
            if runtime_factor_id is not None and not _is_placeholder_forecast(
                result,
                model_id=model_id,
            ):
                runtime_probabilities[runtime_factor_id] = float(result[0])

        has_factor_composition = bool(
            self.strategy is not None and self.strategy.config.factor_composition
        )
        if not probabilities and not has_factor_composition:
            _log_pipeline_funnel(signal, forecasted_count=0, emitted_count=0)
            self.last_diagnostic = ControllerDiagnostic(
                code="no_forecaster_output",
                message=(
                    "No forecaster produced a probability and the strategy has "
                    "no factor composition; skipping decision."
                ),
                market_id=signal.market_id,
                strategy_id=self.strategy_id,
                strategy_version_id=self.strategy_version_id,
                token_id=signal.token_id,
                severity="info",
                metadata={"forecaster_count": len(forecasters)},
            )
            return None

        non_llm_weight = sum(
            weight
            for weight, is_llm in zip(
                confidence_weights,
                llm_weight_flags,
                strict=True,
            )
            if not is_llm
        )
        if any(llm_weight_flags) and non_llm_weight <= 0.0 and not has_factor_composition:
            self._set_drop_diagnostic(
                signal,
                code="llm_only_forecast",
                message=(
                    "Skipping decision because no non-LLM forecaster has "
                    "positive confidence to bound LLM influence."
                ),
                severity="info",
            )
            _log_pipeline_funnel(
                signal,
                forecasted_count=len(probabilities),
                emitted_count=0,
            )
            return None

        prob_estimate = (
            _weighted_probability(
                probabilities,
                confidence_weights,
                llm_mask=llm_weight_flags,
            )
            if probabilities
            else signal.yes_price
        )
        factor_snapshot_hash: str | None = None
        composition_trace: dict[str, object] = {}
        signal_factor_values = _signal_factor_values(signal)
        runtime_factor_values = {
            (factor_id, ""): value
            for factor_id, value in runtime_probabilities.items()
        }
        if has_factor_composition and self.strategy is not None:
            factor_snapshot = await self.factor_reader.snapshot(
                market_id=signal.market_id,
                as_of=signal.timestamp,
                required=self.strategy.config.factor_composition,
                strategy_id=self.strategy.strategy_id,
                strategy_version_id=self.strategy.strategy_version_id,
            )
            factor_values = dict(factor_snapshot.values)
            factor_values.update(signal_factor_values)
            factor_values.update(runtime_factor_values)
            reported_missing_factors = factor_snapshot.missing_factors
            if not reported_missing_factors and not factor_snapshot.values:
                reported_missing_factors = required_factor_keys(
                    self.strategy.config.factor_composition
                )
            unresolved_missing_factors = tuple(
                key for key in reported_missing_factors if key not in factor_values
            )
            if _strict_factor_gates(self.settings) and unresolved_missing_factors:
                self.last_diagnostic = _composition_diagnostic(
                    code="missing_required_factors",
                    message="Skipping decision because required raw factors are missing.",
                    signal=signal,
                    strategy_id=self.strategy_id,
                    strategy_version_id=self.strategy_version_id,
                    factors=unresolved_missing_factors,
                )
                logger.info(
                    "controller diagnostic %s for %s",
                    self.last_diagnostic.code,
                    signal.market_id,
                    extra={"controller_diagnostic": self.last_diagnostic},
                )
                return None
            unresolved_stale_factors = tuple(
                key for key in factor_snapshot.stale_factors if key not in signal_factor_values
            )
            if _strict_factor_gates(self.settings) and unresolved_stale_factors:
                self.last_diagnostic = _composition_diagnostic(
                    code="stale_required_factors",
                    message="Skipping decision because required raw factors are stale.",
                    signal=signal,
                    strategy_id=self.strategy_id,
                    strategy_version_id=self.strategy_version_id,
                    factors=unresolved_stale_factors,
                )
                logger.info(
                    "controller diagnostic %s for %s",
                    self.last_diagnostic.code,
                    signal.market_id,
                    extra={"controller_diagnostic": self.last_diagnostic},
                )
                return None
            try:
                branch_probabilities = evaluate_branch_probabilities(
                    self.strategy.config.factor_composition,
                    factor_values,
                )
                if (
                    self.settings.mode == RunMode.BACKTEST
                    and not branch_probabilities
                    and unresolved_missing_factors
                ):
                    prob_estimate = 0.5
                else:
                    prob_estimate = apply_composition(
                        self.strategy.config.factor_composition,
                        factor_values,
                    )
                factor_snapshot_hash = factor_snapshot.snapshot_hash
                composition_trace = {
                    "selected_probability": prob_estimate,
                    "expected_edge": prob_estimate - signal.yes_price,
                    "factor_snapshot_hash": factor_snapshot.snapshot_hash,
                    "missing_factors": [
                        _factor_key_label((factor_id, param))
                        for factor_id, param in unresolved_missing_factors
                    ],
                    "branch_probabilities": branch_probabilities,
                }
                if unresolved_stale_factors:
                    composition_trace["stale_factors"] = [
                        _factor_key_label((factor_id, param))
                        for factor_id, param in unresolved_stale_factors
                    ]
                if factor_snapshot.ages_ms:
                    composition_trace["factor_ages_ms"] = {
                        _factor_key_label((factor_id, param)): age
                        for (factor_id, param), age in factor_snapshot.ages_ms.items()
                    }
            except (KeyError, ValueError) as exc:
                logger.warning("composition resolution failed: %s", exc)
                self._set_drop_diagnostic(
                    signal,
                    code="composition_resolution_failed",
                    message=(
                        "Skipping decision because factor composition could not "
                        "resolve a probability."
                    ),
                    severity="error",
                    metadata={
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                _log_pipeline_funnel(
                    signal,
                    forecasted_count=len(probabilities),
                    emitted_count=0,
                )
                return None
        else:
            factor_values = dict(signal_factor_values)
            factor_values.update(runtime_factor_values)
        calibration_prob_estimate = prob_estimate
        calibrated_estimate = _apply_pre_calibrators(
            prob_estimate,
            model_ids=model_ids,
            calibrator=calibrator,
            strategy=self.strategy,
            signal=signal,
        )
        if calibrated_estimate is None:
            self._set_drop_diagnostic(
                signal,
                code="calibration_clamp_rejected",
                message=(
                    "Skipping decision because pre-calibration rejected the "
                    "forecast (extreme probability with insufficient resolved "
                    "samples to trust it)."
                ),
                metadata={"raw_probability": prob_estimate},
            )
            _log_pipeline_funnel(
                signal,
                forecasted_count=len(probabilities),
                emitted_count=0,
            )
            return None
        prob_estimate = calibrated_estimate
        yes_probability = prob_estimate
        yes_reference_price = _yes_reference_price(signal, self.strategy)
        if signal_outcome.signal_outcome == "NO":
            yes_reference_price = signal.yes_price
        yes_edge = yes_probability - yes_reference_price
        opportunity_side: Literal["yes", "no"] = "yes"
        decision_token_id = signal_outcome.yes_token_id or signal_token_id
        decision_yes_token_id: str | None = signal_outcome.yes_token_id or signal_token_id
        decision_outcome: Literal["YES", "NO"] = "YES"
        decision_probability = yes_probability
        decision_price = yes_reference_price
        decision_edge = yes_edge
        execution_signal = signal
        if yes_edge >= 0.0 and signal_outcome.signal_outcome == "NO":
            if (
                signal_outcome.yes_token_id is None
                or signal_outcome.yes_token_id == signal_token_id
            ):
                resolved_tokens = await self.outcome_token_resolver.resolve(
                    market_id=signal.market_id,
                    signal_token_id=signal_token_id,
                )
                signal_outcome = _SignalOutcomeContext(
                    yes_token_id=resolved_tokens.yes_token_id
                    or signal_outcome.yes_token_id,
                    no_token_id=resolved_tokens.no_token_id
                    or signal_outcome.no_token_id,
                    signal_outcome=signal_outcome.signal_outcome,
                )
            if (
                signal_outcome.yes_token_id is None
                or signal_outcome.yes_token_id == signal_token_id
            ):
                self.last_diagnostic = ControllerDiagnostic(
                    code="missing_yes_token",
                    message=(
                        "Skipping bullish decision because no YES token could be "
                        "resolved for a NO-token signal."
                    ),
                    market_id=signal.market_id,
                    strategy_id=self.strategy_id,
                    strategy_version_id=self.strategy_version_id,
                    token_id=signal.token_id,
                    severity="warning",
                    metadata={
                        "signal_token_id": signal.token_id,
                        "no_token_id": signal_outcome.no_token_id,
                        "outcome": "YES",
                    },
                )
                logger.info(
                    "controller diagnostic %s for %s",
                    self.last_diagnostic.code,
                    signal.market_id,
                    extra={"controller_diagnostic": self.last_diagnostic},
                )
                return None
            decision_token_id = signal_outcome.yes_token_id
            decision_yes_token_id = signal_outcome.yes_token_id
        if yes_edge < 0.0:
            if signal_outcome.no_token_id is None:
                resolved_tokens = await self.outcome_token_resolver.resolve(
                    market_id=signal.market_id,
                    signal_token_id=signal_token_id,
                )
                signal_outcome = _signal_outcome_context(
                    signal_token_id=signal_token_id,
                    outcome_tokens=resolved_tokens,
                )
            if signal_outcome.no_token_id is None:
                self.last_diagnostic = ControllerDiagnostic(
                    code="missing_no_token",
                    message=(
                        "Skipping bearish decision because no NO token could be resolved."
                    ),
                    market_id=signal.market_id,
                    strategy_id=self.strategy_id,
                    strategy_version_id=self.strategy_version_id,
                    token_id=signal.token_id,
                    severity="warning",
                    metadata={
                        "signal_token_id": signal.token_id,
                        "yes_token_id": signal_outcome.yes_token_id,
                        "outcome": "NO",
                    },
                )
                logger.info(
                    "controller diagnostic %s for %s",
                    self.last_diagnostic.code,
                    signal.market_id,
                    extra={"controller_diagnostic": self.last_diagnostic},
                )
                return None
            decision_yes_token_id = signal_outcome.yes_token_id
            decision_token_id = signal_outcome.no_token_id
            decision_outcome = "NO"
            opportunity_side = "no"
            decision_probability = 1.0 - yes_probability
            decision_price = max(1e-6, min(1.0 - 1e-6, 1.0 - yes_reference_price))
            if signal_outcome.signal_outcome == "NO":
                decision_price = _direct_signal_reference_price(
                    raw_signal,
                    strategy=self.strategy,
                )
            decision_edge = decision_probability - decision_price
        if (
            strategy_metadata(self.strategy).get("price_reference") == "best_ask"
        ):
            direct_required = not _decision_uses_signal_orderbook(
                execution_signal,
                token_id=decision_token_id,
                outcome=decision_outcome,
            )
            direct_read = await self._read_direct_outcome_execution_signal(
                signal,
                token_id=decision_token_id,
                outcome=decision_outcome,
                price_coordinate="target_token",
            )
            if direct_read.signal is not None:
                execution_signal = direct_read.signal
                direct_price = best_ask(execution_signal)
                if direct_price is None:
                    self._set_drop_diagnostic(
                        signal,
                        code="direct_outcome_orderbook_required",
                        message=(
                            "Skipping decision because the selected outcome direct "
                            "orderbook has no executable ask."
                        ),
                        metadata={
                            "signal_token_id": signal.token_id,
                            "decision_token_id": decision_token_id,
                            "decision_outcome": decision_outcome,
                        },
                    )
                    _log_pipeline_funnel(
                        signal,
                        forecasted_count=len(probabilities),
                        emitted_count=0,
                    )
                    return None
                decision_price = direct_price
                decision_edge = decision_probability - decision_price
            elif direct_required:
                direct_metadata: dict[str, object] = {
                    "signal_token_id": signal.token_id,
                    "decision_token_id": decision_token_id,
                    "decision_outcome": decision_outcome,
                    "yes_reference_price": yes_reference_price,
                    "synthetic_decision_price": decision_price,
                }
                if direct_read.failure is not None:
                    direct_metadata["direct_book_failure"] = direct_read.failure
                if direct_read.age_ms is not None:
                    direct_metadata["direct_book_age_ms"] = direct_read.age_ms
                self._set_drop_diagnostic(
                    signal,
                    code="direct_outcome_orderbook_required",
                    message=(
                        "Skipping decision because best-ask execution pricing "
                        "requires a direct orderbook for the selected outcome token."
                    ),
                    metadata=direct_metadata,
                )
                _log_pipeline_funnel(
                    signal,
                    forecasted_count=len(probabilities),
                    emitted_count=0,
                )
                return None
        if decision_edge <= 0.0:
            self._set_drop_diagnostic(
                signal,
                code="decision_edge_not_positive",
                message=(
                    "Skipping decision because the best available edge after "
                    "side selection is not positive."
                ),
                metadata={
                    "decision_edge": decision_edge,
                    "decision_outcome": decision_outcome,
                    "decision_probability": decision_probability,
                    "decision_price": decision_price,
                },
            )
            _log_pipeline_funnel(
                signal,
                forecasted_count=len(probabilities),
                emitted_count=0,
            )
            return None
        decision_spread_bps = spread_bps_at_decision(
            execution_signal,
            token_id=decision_token_id,
            outcome=decision_outcome,
            yes_token_id=decision_yes_token_id,
        )
        if (
            decision_spread_bps is not None
            and self.settings.mode != RunMode.BACKTEST
            and decision_spread_bps > router.controller.max_spread_bps
        ):
            self._set_drop_diagnostic(
                signal,
                code="router_gate:spread_too_wide",
                message=(
                    "Skipping decision because the selected outcome book spread "
                    "exceeds the configured maximum."
                ),
                metadata={
                    "spread_bps_at_decision": decision_spread_bps,
                    "max_spread_bps": router.controller.max_spread_bps,
                    "decision_outcome": decision_outcome,
                    "decision_token_id": decision_token_id,
                },
            )
            _log_pipeline_funnel(
                signal,
                forecasted_count=len(probabilities),
                emitted_count=0,
            )
            return None
        # When decision_price is the executable best ask, decision_edge is
        # already net of crossing the spread; charging spread_bps again would
        # double-count it. The price-equality check (not metadata alone)
        # covers the best_ask strategies whose pricing fell back to
        # yes_price because no ask existed — there the spread is a real cost.
        spread_already_in_price = (
            strategy_metadata(self.strategy).get("price_reference")
            == "best_ask"
            and decision_price == best_ask(execution_signal)
        )
        decision_costs = _decision_cost_edges(
            decision_price=decision_price,
            spread_bps=decision_spread_bps,
            settings=self.settings,
            fee_rate=_strategy_fee_rate(
                self.strategy,
                execution_signal,
                self.settings,
            ),
            max_slippage_bps=strategy_max_slippage_bps,
            spread_already_in_price=spread_already_in_price,
        )
        net_edge_after_costs = decision_edge - decision_costs.total_edge
        if net_edge_after_costs <= 0.0:
            self._set_drop_diagnostic(
                signal,
                code="decision_net_edge_not_positive",
                message=(
                    "Skipping decision because configured entry costs erase "
                    "the gross edge after side selection."
                ),
                metadata={
                    "gross_edge": decision_edge,
                    "spread_bps_at_decision": decision_spread_bps,
                    "spread_edge": decision_costs.spread_edge,
                    "spread_already_in_price": spread_already_in_price,
                    "fee_rate": decision_costs.fee_rate,
                    "fee_edge": decision_costs.fee_edge,
                    "max_slippage_bps": strategy_max_slippage_bps,
                    "slippage_edge": decision_costs.slippage_edge,
                    "net_edge_after_costs": net_edge_after_costs,
                    "decision_outcome": decision_outcome,
                    "decision_probability": decision_probability,
                    "decision_price": decision_price,
                },
            )
            _log_pipeline_funnel(
                signal,
                forecasted_count=len(probabilities),
                emitted_count=0,
            )
            return None
        size = _size_with_costs(
            sizer,
            prob=decision_probability,
            market_price=decision_price,
            portfolio=active_portfolio,
            cost_edge=decision_costs.total_edge if self.strategy is not None else 0.0,
        )
        market_exposure_usdc = _market_exposure_usdc(
            active_portfolio,
            market_id=signal.market_id,
        )
        max_market_position_usdc = _max_position_per_market_usdc(
            strategy=self.strategy,
            sizer=sizer,
            settings=self.settings,
        )
        remaining_market_capacity_usdc = max(
            0.0,
            max_market_position_usdc - market_exposure_usdc,
        )
        if remaining_market_capacity_usdc < min_order_usdc and size >= min_order_usdc:
            self._set_drop_diagnostic(
                signal,
                code="market_position_capacity_below_minimum",
                message=(
                    "Skipping decision because the remaining market position "
                    "capacity is below the minimum order notional."
                ),
                metadata={
                    "size_usdc": size,
                    "min_order_usdc": min_order_usdc,
                    "market_exposure_usdc": market_exposure_usdc,
                    "max_market_position_usdc": max_market_position_usdc,
                    "remaining_market_capacity_usdc": remaining_market_capacity_usdc,
                    "decision_outcome": decision_outcome,
                },
            )
            _log_pipeline_funnel(
                signal,
                forecasted_count=len(probabilities),
                emitted_count=0,
            )
            return None
        size = min(size, remaining_market_capacity_usdc)
        remaining_risk_group_capacity_usdc = _remaining_risk_group_capacity_usdc(
            portfolio=active_portfolio,
            settings=self.settings,
            risk_group_id=risk_group_id,
        )
        if remaining_risk_group_capacity_usdc is not None:
            size = min(size, remaining_risk_group_capacity_usdc)
        executable_depth_usdc = _executable_buy_depth_usdc(
            execution_signal.orderbook,
            limit_price=decision_price,
            max_slippage_bps=strategy_max_slippage_bps,
        )
        if (
            executable_depth_usdc is not None
            and self.settings.mode != RunMode.BACKTEST
            and strategy_metadata(self.strategy).get("price_reference") == "best_ask"
            and _decision_uses_signal_orderbook(
                execution_signal,
                token_id=decision_token_id,
                outcome=decision_outcome,
            )
        ):
            if executable_depth_usdc < min_order_usdc:
                self._set_drop_diagnostic(
                    signal,
                    code="executable_depth_below_minimum",
                    message=(
                        "Skipping decision because executable book depth at "
                        "the configured limit and slippage is below the "
                        "minimum order notional."
                    ),
                    metadata={
                        "executable_depth_usdc": executable_depth_usdc,
                        "min_order_usdc": min_order_usdc,
                        "decision_outcome": decision_outcome,
                        "decision_price": decision_price,
                        "max_slippage_bps": (
                            strategy_max_slippage_bps
                        ),
                    },
                )
                _log_pipeline_funnel(
                    signal,
                    forecasted_count=len(probabilities),
                    emitted_count=0,
                )
                return None
            size = min(size, executable_depth_usdc)
        if size <= 0.0 or size < min_order_usdc:
            self.suppressed_zero_size += 1
            self._set_drop_diagnostic(
                signal,
                code="order_size_below_minimum",
                message=(
                    "Skipping decision because the sized order is below the "
                    "minimum order notional."
                ),
                metadata={
                    "size_usdc": size,
                    "min_order_usdc": min_order_usdc,
                    "decision_outcome": decision_outcome,
                },
            )
            _log_pipeline_funnel(
                signal,
                forecasted_count=len(probabilities),
                emitted_count=0,
            )
            return None
        cooldown_key = (
            self.strategy_id,
            self.strategy_version_id,
            signal.market_id,
            decision_token_id,
            decision_outcome,
        )
        if _within_decision_cooldown(
            self._last_decision_emitted_at,
            key=cooldown_key,
            current_ts=signal.timestamp,
            settings=self.settings,
        ):
            self._set_drop_diagnostic(
                signal,
                code="within_decision_cooldown",
                message=(
                    "Skipping decision because an identical decision was emitted "
                    "within the configured cooldown window."
                ),
                metadata={
                    "cooldown_s": self.settings.controller.decision_cooldown_s,
                    "decision_outcome": decision_outcome,
                },
            )
            _log_pipeline_funnel(
                signal,
                forecasted_count=len(probabilities),
                emitted_count=0,
            )
            return None
        decision_created_at = datetime.now(tz=UTC)
        final_book_age_ms = _execution_book_age_ms_at_decision_time(
            execution_signal,
            settings=self.settings,
            decision_created_at=decision_created_at,
        )
        if final_book_age_ms is not None:
            if final_book_age_ms > router.controller.max_book_age_ms:
                self._set_drop_diagnostic(
                    signal,
                    code="router_gate:book_too_stale",
                    message=(
                        "Skipping decision because the selected outcome "
                        "orderbook became too stale before emission."
                    ),
                    metadata={
                        "gate_reason": "book_too_stale",
                        "phase": "pre_emit",
                        "book_age_ms": final_book_age_ms,
                        "max_book_age_ms": router.controller.max_book_age_ms,
                        "decision_outcome": decision_outcome,
                        "decision_token_id": decision_token_id,
                    },
                )
                _log_pipeline_funnel(
                    signal,
                    forecasted_count=len(probabilities),
                    emitted_count=0,
                )
                return None
            execution_signal = replace(
                execution_signal,
                external_signal={
                    **execution_signal.external_signal,
                    "book_age_ms": final_book_age_ms,
                },
            )
        self._last_decision_emitted_at[cooldown_key] = signal.timestamp
        _log_pipeline_funnel(
            signal,
            forecasted_count=len(probabilities),
            emitted_count=1,
        )
        opportunity = Opportunity(
            opportunity_id=f"opportunity-{uuid.uuid4().hex}",
            market_id=signal.market_id,
            token_id=decision_token_id,
            side=opportunity_side,
            selected_factor_values=_selected_factor_values(factor_values),
            expected_edge=decision_edge,
            rationale=_rationale_text(rationales),
            target_size_usdc=size,
            expiry=signal.resolves_at,
            staleness_policy="market_signal_freshness",
            strategy_id=self.strategy_id,
            strategy_version_id=self.strategy_version_id,
            created_at=decision_created_at,
            factor_snapshot_hash=factor_snapshot_hash,
            composition_trace={
                **composition_trace,
                "selected_probability": decision_probability,
                "expected_edge": decision_edge,
                "yes_probability": yes_probability,
                "yes_reference_price": yes_reference_price,
                "yes_edge": yes_edge,
                "traded_outcome": decision_outcome,
                "traded_probability": decision_probability,
                "traded_price": decision_price,
                "traded_edge": decision_edge,
                "gross_edge": decision_edge,
                "spread_bps_at_decision": decision_spread_bps,
                "spread_edge": decision_costs.spread_edge,
                "spread_already_in_price": spread_already_in_price,
                "fee_rate": decision_costs.fee_rate,
                "fee_edge": decision_costs.fee_edge,
                "max_slippage_bps": strategy_max_slippage_bps,
                "slippage_edge": decision_costs.slippage_edge,
                "net_edge_after_costs": net_edge_after_costs,
            },
        )

        intent_key = _intent_key(
            strategy_id=self.strategy_id,
            strategy_version_id=self.strategy_version_id,
            market_id=signal.market_id,
            token_id=decision_token_id,
            action="BUY",
            outcome=decision_outcome,
            limit_price=decision_price,
            notional_usdc=size,
            factor_snapshot_hash=factor_snapshot_hash,
            signal_timestamp=signal.timestamp,
        )
        self.last_execution_signal = execution_signal
        return opportunity, TradeDecision(
            decision_id=f"decision-{uuid.uuid4().hex}",
            market_id=signal.market_id,
            token_id=decision_token_id,
            venue=signal.venue,
            side="BUY",
            limit_price=decision_price,
            notional_usdc=size,
            order_type="limit",
            max_slippage_bps=strategy_max_slippage_bps,
            stop_conditions=router.stop_conditions(signal),
            prob_estimate=decision_probability,
            expected_edge=decision_edge,
            time_in_force=TimeInForce(
                strategy_economic_str(
                    self.strategy,
                    "time_in_force",
                    self.settings.controller.time_in_force.upper(),
                ).upper()
            ),
            opportunity_id=opportunity.opportunity_id,
            strategy_id=self.strategy_id,
            strategy_version_id=self.strategy_version_id,
            action="BUY",
            outcome=decision_outcome,
            model_id=_decision_model_id(model_ids),
            intent_key=intent_key,
            risk_group_id=_risk_group_id(signal),
            spread_bps_at_decision=decision_spread_bps,
            calibration_prob_estimate=calibration_prob_estimate,
        )

    async def _refresh_signal_book_before_router_gate(
        self,
        signal: MarketSignal,
        *,
        outcome: Literal["YES", "NO"],
    ) -> MarketSignal:
        if (
            self.settings.mode == RunMode.BACKTEST
            or self.direct_book_reader is None
            or strategy_metadata(self.strategy).get("price_reference") != "best_ask"
        ):
            return signal
        direct_read = await self._read_direct_outcome_execution_signal(
            signal,
            token_id=signal.token_id,
            outcome=outcome,
        )
        return signal if direct_read.signal is None else direct_read.signal

    async def _read_direct_outcome_execution_signal(
        self,
        signal: MarketSignal,
        *,
        token_id: str | None,
        outcome: Literal["YES", "NO"],
        price_coordinate: Literal["canonical_yes", "target_token"] = "canonical_yes",
    ) -> _DirectBookRead:
        if token_id is None or self.direct_book_reader is None:
            failure = "missing_token_id" if token_id is None else "reader_unavailable"
            return _DirectBookRead(signal=None, failure=failure)
        try:
            snapshot = await self.direct_book_reader.read_latest_snapshot(
                signal.market_id,
                token_id,
            )
            if snapshot is None:
                return _DirectBookRead(signal=None, failure="snapshot_missing")
            snapshot_ts = _aware_utc(snapshot.ts)
            age_ms = max(
                0.0,
                (datetime.now(tz=UTC) - snapshot_ts).total_seconds() * 1000.0,
            )
            if age_ms > self.settings.controller.max_book_age_ms:
                return _DirectBookRead(signal=None, failure="stale", age_ms=age_ms)
            levels = await self.direct_book_reader.read_levels_for_snapshot(snapshot.id)
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "direct outcome book read failed for %s/%s: %s",
                signal.market_id,
                token_id,
                error,
            )
            return _DirectBookRead(signal=None, failure="read_failed")

        orderbook = _orderbook_from_levels(levels)
        if orderbook is None:
            return _DirectBookRead(signal=None, failure="levels_missing", age_ms=age_ms)
        external_signal: dict[str, Any] = {
            key: value
            for key, value in signal.external_signal.items()
            if key not in {"best_ask", "best_bid", "direct_token_price", "spread_bps"}
        }
        external_signal.update(
            {
                "signal_token_outcome": outcome,
                "book_age_ms": age_ms,
                "book_received_at": snapshot_ts.isoformat(),
                "direct_outcome_book_source": snapshot.source,
            }
        )
        fee_rate_bps = await _read_direct_fee_rate_bps(
            self.direct_book_reader,
            market_id=signal.market_id,
            token_id=token_id,
        )
        if fee_rate_bps is not None:
            external_signal["fee_rate_bps"] = fee_rate_bps
        best_bid = _best_level(orderbook, "bids")
        if best_bid is not None:
            external_signal["best_bid"] = best_bid
        direct_best_ask = _best_level(orderbook, "asks")
        if direct_best_ask is not None:
            external_signal["best_ask"] = direct_best_ask
            external_signal["direct_token_price"] = direct_best_ask
        direct_signal = replace(
            signal,
            token_id=token_id,
            yes_price=_refreshed_direct_signal_price(
                signal,
                outcome=outcome,
                direct_best_ask=direct_best_ask,
                price_coordinate=price_coordinate,
            ),
            orderbook=orderbook,
            external_signal=external_signal,
        )
        spread_bps = spread_bps_at_decision(direct_signal)
        if spread_bps is not None:
            external_signal["spread_bps"] = spread_bps
        return _DirectBookRead(
            signal=replace(direct_signal, external_signal=external_signal),
            age_ms=age_ms,
        )

    async def decide(
        self,
        signal: MarketSignal,
        portfolio: Portfolio | None = None,
    ) -> TradeDecision | None:
        emission = await self.on_signal(signal, portfolio=portfolio)
        if emission is None:
            return None
        _, decision = emission
        return decision

    async def _predict_forecaster(
        self,
        forecaster: IForecaster,
        signal: MarketSignal,
    ) -> ForecastResult | None:
        async_predict = getattr(forecaster, "apredict", None)
        if callable(async_predict) and not _predict_method_overridden(forecaster):
            try:
                return await asyncio.wait_for(
                    cast(
                        Callable[[MarketSignal], Awaitable[ForecastResult | None]],
                        async_predict,
                    )(signal),
                    timeout=self.settings.llm.timeout_s,
                )
            except TimeoutError:
                return None
        timeout_s = self.settings.llm.timeout_s
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(forecaster.predict, signal),
                timeout=timeout_s,
            )
        except TimeoutError:
            logger.info(
                "forecaster timed out; skipping late forecast",
                extra={"forecaster": type(forecaster).__name__, "market_id": signal.market_id},
            )
            return None


def _apply_pre_calibrators(
    prob_estimate: float,
    *,
    model_ids: Sequence[str],
    calibrator: ICalibrator,
    strategy: ActiveStrategy | None,
    signal: MarketSignal,
) -> float | None:
    raw_prob = prob_estimate
    spec = strategy.calibration if strategy is not None else CalibrationSpec()
    if not spec.enabled:
        return raw_prob
    context = CalibrationContext(
        resolved_sample_count=_resolved_sample_count(calibrator, model_ids),
        model_id=_decision_model_id(model_ids) or "unknown",
    )
    # Netcal is trained from the raw decision coordinate persisted with each
    # EvalRecord, so apply it to that same coordinate before presentation
    # shrinkage and extreme clamping.
    current: float | None = calibrator.calibrate(raw_prob, model_id=context.model_id)
    if current is None:
        return None
    current = LogitShrinkageCalibrator(spec).calibrate(current, context=context)
    if current is None:
        return None
    next_prob = ExtremeProbClamp(spec).calibrate(current, context=context)
    if next_prob is None:
        logger.info(
            "calibration clamp rejected forecast for %s",
            signal.market_id,
            extra={
                "event": "clamp_rejection",
                "market_id": signal.market_id,
                "raw_prob": raw_prob,
                "calibrated_prob": current,
                "clamp_action": "reject",
                "resolved_sample_count": context.resolved_sample_count,
            },
        )
        return None
    current = next_prob
    return current


def _weighted_probability(
    probabilities: Sequence[float],
    weights: Sequence[float],
    *,
    llm_mask: Sequence[bool] | None = None,
) -> float:
    if not probabilities:
        return 0.5
    effective = [max(0.0, weight) for weight in weights]
    if llm_mask is not None:
        if len(llm_mask) != len(effective):
            raise ValueError("llm_mask must match probabilities and weights")
        non_llm_total = sum(
            weight for weight, is_llm in zip(effective, llm_mask, strict=True) if not is_llm
        )
        llm_total = sum(
            weight for weight, is_llm in zip(effective, llm_mask, strict=True) if is_llm
        )
        if non_llm_total <= 0.0 and llm_total > 0.0:
            return 0.5
        max_llm_total = non_llm_total / 3.0
        if llm_total > max_llm_total > 0.0:
            scale = max_llm_total / llm_total
            effective = [
                weight * scale if is_llm else weight
                for weight, is_llm in zip(effective, llm_mask, strict=True)
            ]
    total = sum(effective)
    if total <= 0.0:
        return sum(probabilities) / len(probabilities)
    return sum(probability * weight for probability, weight in zip(probabilities, effective, strict=True)) / total


def _size_with_costs(
    sizer: ISizer,
    *,
    prob: float,
    market_price: float,
    portfolio: Portfolio,
    cost_edge: float,
) -> float:
    parameters = inspect.signature(sizer.size).parameters
    kwargs: dict[str, object] = {
        "prob": prob,
        "market_price": market_price,
        "portfolio": portfolio,
    }
    if "cost_edge" in parameters:
        kwargs["cost_edge"] = cost_edge
    size_fn = cast(Callable[..., float], sizer.size)
    return size_fn(**kwargs)


def _resolved_sample_count(calibrator: ICalibrator, model_ids: Sequence[str]) -> int:
    sample_count = getattr(calibrator, "sample_count", None)
    if not callable(sample_count) or not model_ids:
        return 0
    # EvalRecords are keyed by the decision-level model id ("ensemble" for
    # multi-forecaster strategies), so the unlock counter must read the same
    # bucket — not the per-forecaster ids.
    return int(sample_count(_decision_model_id(model_ids) or "unknown"))


def _log_pipeline_funnel(
    signal: MarketSignal,
    *,
    forecasted_count: int,
    emitted_count: int,
) -> None:
    increment_metric(SELECTION_FUNNEL_ROUTED_TOTAL_METRIC)
    increment_metric(SELECTION_FUNNEL_FORECASTED_TOTAL_METRIC, forecasted_count)
    increment_metric(SELECTION_FUNNEL_CONTROLLER_EMITTED_TOTAL_METRIC, emitted_count)
    logger.info(
        "controller pipeline funnel market_id=%s forecasted=%d emitted=%d",
        signal.market_id,
        forecasted_count,
        emitted_count,
        extra={
            "event": "funnel_pipeline",
            "market_id": signal.market_id,
            "forecasted_count": forecasted_count,
            "emitted_count": emitted_count,
        },
    )


@dataclass(frozen=True)
class _DecisionCostEdges:
    spread_edge: float
    fee_rate: float
    fee_edge: float
    slippage_edge: float

    @property
    def total_edge(self) -> float:
        return self.spread_edge + self.fee_edge + self.slippage_edge


def _decision_cost_edges(
    *,
    decision_price: float,
    spread_bps: int | None,
    settings: PMSSettings,
    fee_rate: float | None = None,
    max_slippage_bps: int | None = None,
    spread_already_in_price: bool = False,
) -> _DecisionCostEdges:
    price = Decimal(str(decision_price))
    effective_fee_rate = (
        settings.strategies.flb_fee_rate if fee_rate is None else fee_rate
    )
    fee_rate_decimal = Decimal(str(effective_fee_rate))
    fee_edge = fee_rate_decimal * (Decimal("1.0") - price)
    slippage_edge = (
        Decimal(
            settings.controller.max_slippage_bps
            if max_slippage_bps is None
            else max_slippage_bps
        )
        / Decimal("10000")
    ) * price
    spread_edge = (
        Decimal("0")
        if spread_bps is None or spread_already_in_price
        else (Decimal(spread_bps) / Decimal("10000")) * price
    )
    return _DecisionCostEdges(
        spread_edge=float(spread_edge),
        fee_rate=effective_fee_rate,
        fee_edge=float(fee_edge),
        slippage_edge=float(slippage_edge),
    )


def _strategy_fee_rate(
    strategy: ActiveStrategy | None,
    signal: MarketSignal,
    settings: PMSSettings,
) -> float:
    strategy_fallback = strategy_economic_float(
        strategy,
        "fee_rate",
        settings.strategies.flb_fee_rate,
    )
    return market_fee_rate_from_metadata(
        signal.external_signal,
        fallback_rate=strategy_fallback,
    )


def _signal_outcome_context(
    *,
    signal_token_id: str,
    outcome_tokens: OutcomeTokens,
) -> _SignalOutcomeContext:
    signal_outcome: Literal["YES", "NO"] = "YES"
    if outcome_tokens.no_token_id is not None and signal_token_id == outcome_tokens.no_token_id:
        signal_outcome = "NO"
    return _SignalOutcomeContext(
        yes_token_id=outcome_tokens.yes_token_id,
        no_token_id=outcome_tokens.no_token_id,
        signal_outcome=signal_outcome,
    )


def _signal_outcome_context_from_signal(
    signal: MarketSignal,
    *,
    signal_token_id: str,
) -> _SignalOutcomeContext:
    yes_token_id = _external_token_id(signal, "yes_token_id")
    no_token_id = _external_token_id(signal, "no_token_id")
    signal_outcome: Literal["YES", "NO"] = _signal_token_outcome(signal)
    if no_token_id is not None and signal_token_id == no_token_id:
        signal_outcome = "NO"
    elif yes_token_id is not None and signal_token_id == yes_token_id:
        signal_outcome = "YES"
    return _SignalOutcomeContext(
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        signal_outcome=signal_outcome,
    )


def _external_token_id(signal: MarketSignal, key: str) -> str | None:
    value = _external_text(signal, key)
    return value


def _canonical_yes_signal(
    signal: MarketSignal,
    outcome_context: _SignalOutcomeContext,
) -> MarketSignal:
    external_signal = {
        **signal.external_signal,
        "signal_token_outcome": outcome_context.signal_outcome,
    }
    if outcome_context.yes_token_id is not None:
        external_signal["yes_token_id"] = outcome_context.yes_token_id
    if outcome_context.no_token_id is not None:
        external_signal["no_token_id"] = outcome_context.no_token_id
    if outcome_context.signal_outcome != "NO":
        return replace(signal, external_signal=external_signal)

    external_signal["direct_token_price"] = signal.yes_price
    canonical_yes_price = _complement_probability_or_none(signal.yes_price)
    if canonical_yes_price is None:
        return replace(signal, external_signal=external_signal)
    return replace(
        signal,
        yes_price=canonical_yes_price,
        external_signal=external_signal,
    )


def _direct_signal_reference_price(
    signal: MarketSignal,
    *,
    strategy: ActiveStrategy | None,
) -> float:
    if strategy_metadata(strategy).get("price_reference") != "best_ask":
        return signal.yes_price
    executable_price = best_ask(signal)
    return signal.yes_price if executable_price is None else executable_price


def _refreshed_direct_signal_price(
    signal: MarketSignal,
    *,
    outcome: Literal["YES", "NO"],
    direct_best_ask: float | None,
    price_coordinate: Literal["canonical_yes", "target_token"],
) -> float:
    if direct_best_ask is None:
        return signal.yes_price
    if price_coordinate == "target_token":
        return direct_best_ask
    if outcome == "NO":
        canonical_yes_price = _complement_probability_or_none(direct_best_ask)
        return signal.yes_price if canonical_yes_price is None else canonical_yes_price
    return direct_best_ask


def _complement_probability_or_none(value: float) -> float | None:
    probability = _decimal_or_none(value)
    if probability is None or probability <= 0 or probability >= 1:
        return None
    return float(Decimal("1.0") - probability)


def _decision_uses_signal_orderbook(
    signal: MarketSignal,
    *,
    token_id: str | None,
    outcome: str,
) -> bool:
    if token_id is None:
        return outcome == "YES"
    return token_id == signal.token_id


def _with_decision_time_book_age(
    signal: MarketSignal,
    settings: PMSSettings,
) -> MarketSignal:
    if settings.mode == RunMode.BACKTEST:
        return signal
    if "book_received_at" not in signal.external_signal:
        return signal
    age_ms = _decision_time_book_age_ms(
        signal,
        allowed_clock_skew_ms=settings.controller.allowed_book_clock_skew_ms,
    )
    return replace(
        signal,
        external_signal={
            **signal.external_signal,
            "book_age_ms": age_ms,
        },
    )


def _execution_book_age_ms_at_decision_time(
    signal: MarketSignal,
    *,
    settings: PMSSettings,
    decision_created_at: datetime,
) -> float | None:
    if settings.mode == RunMode.BACKTEST:
        return None
    if "book_received_at" not in signal.external_signal:
        return None
    return _decision_time_book_age_ms(
        signal,
        allowed_clock_skew_ms=settings.controller.allowed_book_clock_skew_ms,
        now=decision_created_at,
    )


def _decision_time_book_age_ms(
    signal: MarketSignal,
    *,
    allowed_clock_skew_ms: float,
    now: datetime | None = None,
) -> float:
    book_received_at = _external_signal_datetime(
        signal.external_signal.get("book_received_at")
    )
    if book_received_at is None:
        return float("inf")
    age_ms = (
        _aware_utc(now or datetime.now(tz=UTC)) - _aware_utc(book_received_at)
    ).total_seconds() * 1000.0
    if age_ms < -allowed_clock_skew_ms:
        return float("inf")
    return max(0.0, age_ms)


def _external_signal_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    raw_value = value.strip()
    if not raw_value:
        return None
    if raw_value.endswith("Z"):
        raw_value = f"{raw_value[:-1]}+00:00"
    try:
        return datetime.fromisoformat(raw_value)
    except ValueError:
        return None


def _orderbook_from_levels(levels: Sequence[BookLevel]) -> dict[str, list[dict[str, float]]] | None:
    bids = [
        {"price": level.price, "size": level.size}
        for level in levels
        if level.side == "BUY" and level.price > 0.0 and level.size > 0.0
    ]
    asks = [
        {"price": level.price, "size": level.size}
        for level in levels
        if level.side == "SELL" and level.price > 0.0 and level.size > 0.0
    ]
    if not bids or not asks:
        return None
    bids.sort(key=lambda level: level["price"], reverse=True)
    asks.sort(key=lambda level: level["price"])
    return {"bids": bids, "asks": asks}


def _best_level(orderbook: Mapping[str, Any], side: Literal["bids", "asks"]) -> float | None:
    raw_levels = orderbook.get(side)
    if not isinstance(raw_levels, list):
        return None
    prices = [
        _decimal_or_none(level.get("price"))
        for level in raw_levels
        if isinstance(level, Mapping)
    ]
    valid_prices = [price for price in prices if price is not None and price > 0]
    if not valid_prices:
        return None
    selected = max(valid_prices) if side == "bids" else min(valid_prices)
    return float(selected)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _executable_buy_depth_usdc(
    orderbook: Mapping[str, Any],
    *,
    limit_price: float,
    max_slippage_bps: int,
) -> float | None:
    raw_levels = orderbook.get("asks")
    if not isinstance(raw_levels, list) or not raw_levels:
        return None
    limit = _decimal_or_none(limit_price)
    if limit is None:
        return None
    effective_limit = limit
    if max_slippage_bps > 0:
        effective_limit = limit * (
            Decimal("1") + Decimal(max_slippage_bps) / Decimal("10000")
        )
    executable = Decimal("0")
    saw_valid_level = False
    for raw in raw_levels:
        if not isinstance(raw, Mapping):
            continue
        price = _decimal_or_none(raw.get("price"))
        size = _decimal_or_none(raw.get("size"))
        if price is None or size is None or price <= 0 or size <= 0:
            continue
        saw_valid_level = True
        if price <= effective_limit:
            executable += price * size
    if not saw_valid_level:
        return None
    return float(executable)


def _decimal_or_none(value: object) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite():
        return None
    return parsed


def _effective_min_order_usdc(
    *,
    strategy: ActiveStrategy | None,
    sizer: ISizer,
    settings: PMSSettings,
) -> float:
    candidates = [settings.risk.min_order_usdc]
    strategy_min_order = None if strategy is None else strategy.risk.min_order_size_usdc
    if strategy_min_order is not None:
        candidates.append(strategy_min_order)
    sizer_risk = getattr(sizer, "risk", None)
    sizer_min_order = getattr(sizer_risk, "min_order_usdc", None)
    if isinstance(sizer_min_order, int | float):
        candidates.append(float(sizer_min_order))
    return max(candidates)


def _model_id(result: ForecastResult, forecaster: IForecaster) -> str:
    raw_model_id = getattr(result, "model_id", None)
    if isinstance(raw_model_id, str) and raw_model_id:
        prompt_hash = getattr(result, "prompt_hash", None)
        provider = getattr(result, "provider", None)
        if isinstance(prompt_hash, str) and prompt_hash:
            prefix = f"{provider}:" if isinstance(provider, str) else ""
            return f"{prefix}{raw_model_id}:prompt-{prompt_hash[:12]}"
        return raw_model_id
    return forecaster.__class__.__name__


def _decision_model_id(model_ids: Sequence[str]) -> str | None:
    unique_ids = tuple(dict.fromkeys(model_ids))
    if not unique_ids:
        return None
    if len(unique_ids) == 1:
        return unique_ids[0]
    return "ensemble"


def _selected_factor_values(factor_values: dict[FactorKey, float]) -> dict[str, float]:
    return {
        _factor_key_label(key): value
        for key, value in sorted(factor_values.items())
    }


def _composition_diagnostic(
    *,
    code: str,
    message: str,
    signal: MarketSignal,
    strategy_id: str,
    strategy_version_id: str,
    factors: Sequence[FactorKey],
) -> ControllerDiagnostic:
    return ControllerDiagnostic(
        code=code,
        message=message,
        market_id=signal.market_id,
        strategy_id=strategy_id,
        strategy_version_id=strategy_version_id,
        token_id=signal.token_id,
        severity="warning",
        metadata={
            "factors": [_factor_key_label(key) for key in factors],
        },
    )


def _is_placeholder_forecast(
    result: ForecastResult,
    *,
    model_id: str,
) -> bool:
    return model_id == "neutral" or result[2] == "pre-s5-neutral"


def _predict_method_overridden(forecaster: IForecaster) -> bool:
    if isinstance(forecaster, RulesForecaster):
        return type(forecaster).predict is not _ORIGINAL_RULES_PREDICT
    if isinstance(forecaster, StatisticalForecaster):
        return type(forecaster).predict is not _ORIGINAL_STATISTICAL_PREDICT
    return False


def _signal_factor_values(signal: MarketSignal) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {("yes_price", ""): signal.yes_price}
    row = OrderbookImbalance().compute(signal, EMPTY_OUTER_RING)
    if row is not None:
        value = row.value
        if _signal_token_outcome(signal) == "NO":
            value = -value
        values[(row.factor_id, row.param)] = value
    return values


def _signal_token_outcome(signal: MarketSignal) -> Literal["YES", "NO"]:
    raw_outcome = str(signal.external_signal.get("signal_token_outcome", "YES")).upper()
    return "NO" if raw_outcome == "NO" else "YES"


def _risk_group_id(signal: MarketSignal) -> str | None:
    explicit = _external_text(signal, "risk_group_id")
    if explicit is not None:
        return explicit
    legacy = _external_text(signal, "risk_group")
    if legacy is not None:
        return legacy
    event_id = _external_text(signal, "event_id")
    if event_id is not None:
        return f"event:{event_id}"
    category = _external_text(signal, "category")
    if category is not None:
        return f"category:{category}"
    return None


def _external_text(signal: MarketSignal, key: str) -> str | None:
    raw_value = signal.external_signal.get(key)
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    return value or None


def _strict_factor_gates(settings: PMSSettings) -> bool:
    # Strictness is a runtime safety policy owned by ControllerSettings (and
    # forced on in LIVE). Strategy metadata does not carry a second gate flag.
    return settings.mode == RunMode.LIVE or settings.controller.strict_factor_gates


def _within_decision_cooldown(
    emitted_at_by_key: dict[tuple[str, str, str, str, str], datetime],
    *,
    key: tuple[str, str, str, str, str],
    current_ts: datetime,
    settings: PMSSettings,
) -> bool:
    if settings.mode == RunMode.BACKTEST:
        return False
    cooldown_s = settings.controller.decision_cooldown_s
    if cooldown_s <= 0.0:
        return False
    previous_ts = emitted_at_by_key.get(key)
    if previous_ts is None:
        return False
    return (current_ts - previous_ts).total_seconds() < cooldown_s


def _factor_key_label(key: FactorKey) -> str:
    factor_id, param = key
    if not param:
        return factor_id
    return f"{factor_id}:{param}"


def _strategy_forecaster_names(strategy: ActiveStrategy | None) -> tuple[str, ...]:
    if strategy is None:
        return ()
    return tuple(name for name, _ in strategy.forecaster.forecasters)


def _runtime_factor_id(forecaster_name: str | None) -> str | None:
    if forecaster_name is None:
        return None
    if forecaster_name == "stats":
        return "statistical"
    if forecaster_name in {"rules", "llm"}:
        return forecaster_name
    return None


def _yes_reference_price(
    signal: MarketSignal,
    strategy: ActiveStrategy | None,
) -> float:
    if strategy_metadata(strategy).get("price_reference") != "best_ask":
        return signal.yes_price
    executable_price = best_ask(signal)
    return signal.yes_price if executable_price is None else executable_price


def _market_exposure_usdc(portfolio: Portfolio, *, market_id: str) -> float:
    return sum(
        position.locked_usdc
        for position in portfolio.open_positions
        if position.market_id == market_id
    )


def _risk_group_capacity_diagnostic(
    *,
    portfolio: Portfolio,
    settings: PMSSettings,
    min_order_usdc: float,
    risk_group_id: str | None,
) -> tuple[str, dict[str, object]] | None:
    max_group_exposure = settings.risk.max_exposure_per_risk_group
    if max_group_exposure is None:
        return None
    if risk_group_id is None:
        return (
            "missing_risk_group_id",
            {
                "max_exposure_per_risk_group": max_group_exposure,
                "min_order_usdc": min_order_usdc,
            },
        )
    remaining_group_capacity_usdc = _remaining_risk_group_capacity_usdc(
        portfolio,
        settings=settings,
        risk_group_id=risk_group_id,
    )
    if remaining_group_capacity_usdc is None:
        return None
    group_exposure_usdc = max_group_exposure - remaining_group_capacity_usdc
    if remaining_group_capacity_usdc >= min_order_usdc:
        return None
    return (
        "risk_group_capacity_below_minimum",
        {
            "risk_group_id": risk_group_id,
            "group_exposure_usdc": group_exposure_usdc,
            "max_exposure_per_risk_group": max_group_exposure,
            "remaining_group_capacity_usdc": remaining_group_capacity_usdc,
            "min_order_usdc": min_order_usdc,
        },
    )


def _remaining_risk_group_capacity_usdc(
    portfolio: Portfolio,
    *,
    settings: PMSSettings,
    risk_group_id: str | None,
) -> float | None:
    max_group_exposure = settings.risk.max_exposure_per_risk_group
    if max_group_exposure is None or risk_group_id is None:
        return None
    group_exposure_usdc = _risk_group_exposure_usdc(
        portfolio,
        risk_group_id=risk_group_id,
    )
    remaining_group_capacity_usdc = max(0.0, max_group_exposure - group_exposure_usdc)
    return remaining_group_capacity_usdc


def _risk_group_exposure_usdc(
    portfolio: Portfolio,
    *,
    risk_group_id: str,
) -> float:
    return sum(
        position.locked_usdc
        for position in portfolio.open_positions
        if position.risk_group_id == risk_group_id
    )


def _max_position_per_market_usdc(
    *,
    strategy: ActiveStrategy | None,
    sizer: ISizer,
    settings: PMSSettings,
) -> float:
    if strategy is not None:
        return strategy.risk.max_position_notional_usdc
    risk = getattr(sizer, "risk", None)
    value = getattr(risk, "max_position_per_market", None)
    if isinstance(value, bool):
        return settings.risk.max_position_per_market
    if isinstance(value, int | float):
        return float(value)
    return settings.risk.max_position_per_market


def _intent_key(
    *,
    strategy_id: str,
    strategy_version_id: str,
    market_id: str,
    token_id: str | None,
    action: str,
    outcome: str,
    limit_price: float,
    notional_usdc: float,
    factor_snapshot_hash: str | None,
    signal_timestamp: datetime,
) -> str:
    payload = {
        "strategy_id": strategy_id,
        "strategy_version_id": strategy_version_id,
        "market_id": market_id,
        "token_id": token_id,
        "action": action,
        "outcome": outcome,
        "limit_price": round(limit_price, 4),
        "notional_usdc": round(notional_usdc, 2),
        "factor_snapshot_hash": factor_snapshot_hash,
        "signal_ts_bucket": signal_timestamp.replace(microsecond=0).isoformat(),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _rationale_text(rationales: Sequence[str]) -> str:
    if rationales:
        return " | ".join(dict.fromkeys(rationales))
    return "calibrated market signal"


def _default_portfolio() -> Portfolio:
    return Portfolio(
        total_usdc=0.0,
        free_usdc=0.0,
        locked_usdc=0.0,
        open_positions=[],
    )


def _required(value: T | None, name: str) -> T:
    if value is None:
        msg = f"ControllerPipeline {name} is not initialized"
        raise RuntimeError(msg)
    return value
