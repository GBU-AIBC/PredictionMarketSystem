from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import cast
from dataclasses import dataclass, field

from pms.artifact_path_safety import (
    require_path_outside_working_tree,
    require_private_parent,
)
from pms.config import LLMSettings, PMSSettings, RiskSettings
from pms.controller.calibrators.netcal import NetcalCalibrator
from pms.controller.factor_snapshot import (
    FactorSnapshotReader,
    NullFactorSnapshotReader,
)
from pms.controller.forecasters.llm import (
    LLMForecaster,
    validate_llm_runtime_dependencies,
)
from pms.controller.forecasters.flb import FlbForecaster
from pms.controller.forecasters.paper_canary import PaperCanaryForecaster
from pms.controller.forecasters.rules import RulesForecaster
from pms.controller.forecasters.statistical import StatisticalForecaster
from pms.core.enums import RunMode
from pms.controller.outcome_tokens import (
    NullOutcomeTokenResolver,
    OutcomeTokenResolver,
)
from pms.controller.pipeline import ControllerPipeline, DirectBookSnapshotReader
from pms.controller.router import Router
from pms.controller.sizers.kelly import KellySizer
from pms.core.interfaces import IForecaster
from pms.strategies.flb.artifacts import (
    flb_calibration_provenance_path,
    require_flb_calibration_provenance_for_model,
)
from pms.strategies.flb.source import load_flb_calibration_csv
from pms.strategies.projections import ActiveStrategy, FactorCompositionStep
from pms.strategies.economics import strategy_risk_settings


@dataclass
class ControllerPipelineFactory:
    settings: PMSSettings = field(default_factory=PMSSettings)
    factor_reader: FactorSnapshotReader = field(
        default_factory=NullFactorSnapshotReader
    )
    outcome_token_resolver: OutcomeTokenResolver = field(
        default_factory=NullOutcomeTokenResolver
    )
    direct_book_reader: DirectBookSnapshotReader | None = None

    def build_many(
        self,
        strategies: Sequence[ActiveStrategy],
    ) -> dict[str, ControllerPipeline]:
        return {
            strategy.strategy_id: self.build(strategy)
            for strategy in strategies
        }

    def build(self, strategy: ActiveStrategy) -> ControllerPipeline:
        _assert_strategy_mode_allowed(strategy, mode=self.settings.mode)
        return ControllerPipeline(
            strategy=strategy,
            strategy_id=strategy.strategy_id,
            strategy_version_id=strategy.strategy_version_id,
            factor_reader=self.factor_reader,
            outcome_token_resolver=self.outcome_token_resolver,
            direct_book_reader=self.direct_book_reader,
            forecasters=self._build_forecasters(strategy),
            calibrator=NetcalCalibrator(),
            sizer=KellySizer(
                risk=strategy_risk_settings(
                    strategy,
                    fallback=self.settings.risk,
                )
            ),
            router=Router(self.settings.controller),
            settings=self.settings,
        )

    def _build_forecasters(
        self,
        strategy: ActiveStrategy,
    ) -> tuple[IForecaster, ...]:
        return tuple(
            _build_forecaster(
                name=name,
                raw_params=raw_params,
                settings=self.settings,
                mode=self.settings.mode,
                factor_reader=self.factor_reader,
                factor_composition=strategy.config.factor_composition,
                strategy_id=strategy.strategy_id,
                strategy_version_id=strategy.strategy_version_id,
            )
            for name, raw_params in strategy.forecaster.forecasters
        )


def _build_forecaster(
    *,
    name: str,
    raw_params: tuple[tuple[str, str], ...],
    settings: PMSSettings,
    mode: RunMode,
    factor_reader: FactorSnapshotReader,
    factor_composition: tuple[FactorCompositionStep, ...],
    strategy_id: str,
    strategy_version_id: str,
) -> IForecaster:
    params = dict(raw_params)
    if name == "rules":
        threshold = params.get("threshold")
        return RulesForecaster(
            factor_reader=factor_reader,
            composition=factor_composition,
            strategy_id=strategy_id,
            strategy_version_id=strategy_version_id,
            min_edge=0.02 if threshold is None else float(threshold)
        )
    if name == "stats":
        prior_strength = params.get("prior_strength")
        return StatisticalForecaster(
            factor_reader=factor_reader,
            composition=factor_composition,
            strategy_id=strategy_id,
            strategy_version_id=strategy_version_id,
            prior_strength=2.0 if prior_strength is None else float(prior_strength),
        )
    if name == "llm":
        allowed = {"model", "provider", "prompt_version", "max_tokens"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(
                "LLMForecaster does not yet accept per-strategy params "
                f"{sorted(unknown)!r}"
            )
        updates: dict[str, object] = {
            key: value for key, value in params.items() if key in allowed
        }
        if "max_tokens" in updates:
            updates["max_tokens"] = int(cast(str, updates["max_tokens"]))
        llm_payload = {**settings.llm.model_dump(), **updates}
        if llm_payload.get("spend_ledger_path") is None:
            # Keep the restart-safe ledger scoped to the configured database,
            # rather than placing a process/environment-shared file in HOME.
            # API keys are deliberately excluded from this identity.
            database_digest = sha256(settings.database.dsn.encode()).hexdigest()[:24]
            ledger_dir = Path(settings.live_emergency_audit_path).expanduser().parent
            llm_payload["spend_ledger_path"] = str(
                ledger_dir / f"llm-spend-{database_digest}.json"
            )
        llm_config = LLMSettings.model_validate(llm_payload)
        validate_llm_runtime_dependencies(llm_config)
        return LLMForecaster(config=llm_config)
    if name == "flb":
        if raw_params:
            msg = (
                "FlbForecaster does not yet accept per-strategy params: "
                f"{raw_params!r}"
            )
            raise ValueError(msg)
        raw_path = settings.strategies.flb_calibration_path
        if raw_path is None or raw_path.strip() == "":
            msg = "flb forecaster requires strategies.flb_calibration_path"
            raise ValueError(msg)
        path = Path(raw_path).expanduser()
        require_path_outside_working_tree(path, label="FLB calibration artifact")
        require_private_parent(path, label="FLB calibration artifact")
        provenance_path = flb_calibration_provenance_path(path)
        require_path_outside_working_tree(
            provenance_path,
            label="FLB calibration provenance JSON",
        )
        require_private_parent(
            provenance_path,
            label="FLB calibration provenance JSON",
        )
        calibration_model = load_flb_calibration_csv(
            path,
            min_sample_count=settings.strategies.flb_min_calibration_samples,
        )
        require_flb_calibration_provenance_for_model(
            path,
            model=calibration_model,
        )
        return FlbForecaster(
            calibration_model=calibration_model,
        )
    if name == "paper_canary":
        if mode != RunMode.PAPER:
            msg = "paper_canary forecaster is PAPER-only"
            raise ValueError(msg)
        return PaperCanaryForecaster(
            edge_bps=float(params.get("edge_bps", "1000")),
            max_probability=float(params.get("max_probability", "0.97")),
            min_price=float(params.get("min_price", "0.05")),
            max_price=float(params.get("max_price", "0.90")),
            sample_modulus=int(params.get("sample_modulus", "25")),
            sample_remainder=int(params.get("sample_remainder", "0")),
        )
    msg = f"Unsupported forecaster {name!r}"
    raise ValueError(msg)


def _assert_strategy_mode_allowed(strategy: ActiveStrategy, *, mode: RunMode) -> None:
    if mode != RunMode.LIVE:
        return
    metadata = dict(strategy.config.metadata)
    live_allowed = metadata.get("live_allowed")
    if live_allowed is not None and live_allowed.strip().lower() == "false":
        msg = f"{strategy.strategy_id} is PAPER-only"
        raise ValueError(msg)
    if live_allowed is None or live_allowed.strip().lower() != "true":
        msg = (
            f"{strategy.strategy_id} requires metadata.live_allowed=true "
            "for LIVE mode"
        )
        raise ValueError(msg)
    _assert_live_strategy_metadata_ready(strategy, metadata)
    if not strategy.calibration.enabled:
        msg = (
            f"{strategy.strategy_id} requires calibration.enabled=true "
            "for LIVE mode"
        )
        raise ValueError(msg)
    if not _live_strategy_has_non_llm_forecaster(strategy):
        msg = (
            f"{strategy.strategy_id} requires at least one non-LLM forecaster "
            "for LIVE mode"
        )
        raise ValueError(msg)


def _live_strategy_has_non_llm_forecaster(strategy: ActiveStrategy) -> bool:
    return any(name != "llm" for name, _raw_params in strategy.forecaster.forecasters)


def _assert_live_strategy_metadata_ready(
    strategy: ActiveStrategy,
    metadata: dict[str, str],
) -> None:
    for key in _REQUIRED_LIVE_STRATEGY_EVIDENCE_METADATA_KEYS:
        value = metadata.get(key)
        if value is None or value.strip() == "":
            msg = (
                f"{strategy.strategy_id} requires metadata.{key} "
                "for LIVE mode"
            )
            raise ValueError(msg)
        if _looks_like_unready_live_strategy_metadata(value):
            msg = (
                f"{strategy.strategy_id} metadata.{key} must not be "
                "placeholder/static for LIVE mode"
            )
            raise ValueError(msg)

    for key in _LIVE_STRATEGY_EVIDENCE_METADATA_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        if _looks_like_unready_live_strategy_metadata(value):
            msg = (
                f"{strategy.strategy_id} metadata.{key} must not be "
                "placeholder/static for LIVE mode"
            )
            raise ValueError(msg)


_REQUIRED_LIVE_STRATEGY_EVIDENCE_METADATA_KEYS = (
    "alpha_source",
    "edge_model_source",
    "calibration_source",
    "evidence_source",
)


_LIVE_STRATEGY_EVIDENCE_METADATA_KEYS = (
    "model_source",
    *_REQUIRED_LIVE_STRATEGY_EVIDENCE_METADATA_KEYS,
)


def _looks_like_unready_live_strategy_metadata(value: str) -> bool:
    normalized = value.strip().lower()
    unready_markers = (
        "placeholder",
        "todo",
        "fill_in",
        "__fill",
        "replace",
        "static_live_estimate",
        "static_estimate",
        "paper_soak_placeholder",
        "uncalibrated",
    )
    return any(marker in normalized for marker in unready_markers)
