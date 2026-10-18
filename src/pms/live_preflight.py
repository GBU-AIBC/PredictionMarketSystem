from __future__ import annotations

import importlib.util
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import asyncpg
from pydantic import SecretStr

from pms.controller.factory import ControllerPipelineFactory
from pms.config import (
    PMSSettings,
    live_runtime_dependency_requirements,
    validate_live_readiness_reports_for_submission,
    validate_live_mode_ready,
)
from pms.core.category_prior import load_category_prior_observations_csv
from pms.core.enums import RunMode
from pms.core.models import (
    LiveTradingDisabledError,
    Portfolio,
    Position,
    ReconciliationReport,
    Venue,
)
from pms.live_preflight_artifact import (
    is_sha256_hexdigest as _is_sha256_hexdigest,
    live_preflight_readiness_reports_fingerprint as live_preflight_readiness_reports_fingerprint,
    live_preflight_settings_fingerprint as live_preflight_settings_fingerprint,
    loads_json_rejecting_duplicate_keys as _loads_json_rejecting_duplicate_keys,
)
from pms.redaction import redact_database_error, redact_live_error_values
from pms.research.spec_codec import deserialize_execution_model
from pms.storage.schema_check import ensure_schema_current
from pms.storage.strategy_registry import (
    PostgresStrategyRegistry,
    llm_identity_for_settings,
)
from pms.strategies.flb.artifacts import (
    file_sha256_no_follow,
    flb_calibration_provenance_path,
    load_flb_calibration_provenance_json,
)
from pms.strategies.flb.source import load_flb_calibration_csv
from pms.strategies.projections import ActiveStrategy
from pms.strategies.versioning import serialize_strategy_config_json


if TYPE_CHECKING:
    from pms.actuator.adapters.polymarket import PolymarketVenueAccountReconciler


_REQUIRED_FINAL_PREFLIGHT_CHECKS: tuple[str, ...] = (
    "live_config",
    "runtime_dependencies",
    "operator_approval",
    "emergency_audit",
    "first_order_audit",
    "database_connection",
    "schema_current",
    "market_data_freshness",
    "submission_unknown",
    "live_open_orders",
    "active_strategies",
    "venue_reconciliation",
)
_LIVE_PAPER_BACKTEST_DIFF_GENERATED_BY = "scripts/paper_backtest_execution_diff.py"
_LIVE_PAPER_BACKTEST_DIFF_ARTIFACT_MODE = "paper_backtest_execution_diff"
_LIVE_EXECUTION_MODEL_GENERATED_BY = "scripts/execution_model_from_telemetry.py"
_LIVE_EXECUTION_MODEL_ARTIFACT_MODE = "telemetry_execution_model"
_MIN_LIVE_EXECUTION_MODEL_TELEMETRY_SAMPLES = 10
_REQUIRED_LIVE_PAPER_BACKTEST_DIFF_METRICS: tuple[str, ...] = (
    "paper_decision_count",
    "backtest_decision_count",
    "matched_decision_count",
    "fill_rate_delta_abs",
    "rejection_rate_delta_abs",
    "avg_slippage_bps_delta_abs",
    "total_pnl_delta_abs",
)
_LIVE_PAPER_BACKTEST_DIFF_COUNT_METRICS: tuple[str, ...] = (
    "paper_decision_count",
    "backtest_decision_count",
    "matched_decision_count",
)
_MIN_LIVE_PAPER_BACKTEST_DIFF_MATCHED_DECISIONS = 10
_LIVE_PAPER_BACKTEST_DIFF_RATE_DELTA_METRICS: tuple[str, ...] = (
    "fill_rate_delta_abs",
    "rejection_rate_delta_abs",
)
_LIVE_PAPER_BACKTEST_DIFF_THRESHOLD_METRICS: tuple[tuple[str, str], ...] = (
    ("fill_rate_delta_abs", "max_fill_rate_delta"),
    ("rejection_rate_delta_abs", "max_rejection_rate_delta"),
    ("avg_slippage_bps_delta_abs", "max_avg_slippage_bps_delta"),
    ("total_pnl_delta_abs", "max_total_pnl_delta"),
)
_LIVE_PAPER_BACKTEST_DIFF_MIN_MATCHED_DECISIONS_THRESHOLD = "min_matched_decisions"
_LIVE_PAPER_BACKTEST_DIFF_EMPTY_LIST_FIELDS: tuple[tuple[str, str], ...] = (
    ("paper_only_decision_ids", "paper-only decisions"),
    ("backtest_only_decision_ids", "backtest-only decisions"),
    ("status_mismatches", "status mismatches"),
)


def _read_bytes_no_follow(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o777)
    try:
        path_stat = os.fstat(fd)
        if not stat.S_ISREG(path_stat.st_mode):
            raise OSError(f"path is not a regular file: {path}")
        if path_stat.st_nlink != 1:
            raise OSError(f"path is not a single-link file: {path}")
        with os.fdopen(fd, "rb") as file:
            fd = -1
            return file.read()
    finally:
        if fd >= 0:
            os.close(fd)


def _read_text_no_follow(path: Path) -> str:
    return _read_bytes_no_follow(path).decode("utf-8")


@dataclass(frozen=True, slots=True)
class LivePreflightCheck:
    name: str
    ok: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class LivePreflightResult:
    checks: tuple[LivePreflightCheck, ...]
    active_strategies_fingerprint: str | None = None

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def require_check(self, name: str) -> LivePreflightCheck:
        for check in self.checks:
            if check.name == name:
                return check
        msg = f"preflight check not found: {name}"
        raise KeyError(msg)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "checks": [check.as_dict() for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class _ActiveStrategiesPreflight:
    check: LivePreflightCheck
    fingerprint: str | None


class ActiveStrategyRegistry(Protocol):
    async def list_active_strategies(self) -> Sequence[ActiveStrategy]: ...


def live_preflight_result_is_final_go_no_go_valid(
    result: LivePreflightResult,
    *,
    skip_venue: bool,
    skip_credentials: bool = False,
    database_url_override_used: bool,
) -> bool:
    if not result.ok or skip_venue or skip_credentials or database_url_override_used:
        return False
    if (
        result.active_strategies_fingerprint is None
        or result.active_strategies_fingerprint.strip() == ""
        or _looks_like_preflight_placeholder_detail(
            result.active_strategies_fingerprint
        )
        or not _is_sha256_hexdigest(result.active_strategies_fingerprint)
    ):
        return False
    check_names = [check.name for check in result.checks]
    if len(set(check_names)) != len(check_names):
        return False
    if set(check_names) != set(_REQUIRED_FINAL_PREFLIGHT_CHECKS):
        return False
    if any(
        check.detail.strip() == ""
        or _looks_like_preflight_placeholder_detail(check.detail)
        for check in result.checks
    ):
        return False
    checks_by_name = {check.name: check for check in result.checks}
    return all(
        checks_by_name.get(name, LivePreflightCheck(name, False, "")).ok
        for name in _REQUIRED_FINAL_PREFLIGHT_CHECKS
    )


async def run_live_preflight(
    settings: PMSSettings,
    *,
    pool: asyncpg.Pool | None = None,
    venue_reconciler: PolymarketVenueAccountReconciler | None = None,
    skip_venue: bool = False,
    skip_credentials: bool = False,
) -> LivePreflightResult:
    checks: list[LivePreflightCheck] = []
    credentials_ok = False
    non_venue_config_ok = False
    active_strategies_fingerprint: str | None = None

    try:
        if settings.mode != RunMode.LIVE:
            msg = f"mode must be live, got {settings.mode.value}"
            raise ValueError(msg)
        validation_settings = (
            _settings_with_diagnostic_credentials(settings)
            if skip_credentials
            else settings
        )
        validate_live_mode_ready(validation_settings)
        _validate_live_strategy_artifacts(settings)
    except Exception as exc:  # noqa: BLE001
        checks.append(
            LivePreflightCheck(
                "live_config",
                False,
                redact_live_error(str(exc), settings),
            )
        )
    else:
        non_venue_config_ok = True
        credentials_ok = not skip_credentials
        detail = (
            "LIVE config validates with diagnostic credentials; final preflight "
            "still requires real credentials"
            if skip_credentials
            else "LIVE config validates"
        )
        checks.append(LivePreflightCheck("live_config", True, detail))

    checks.append(_runtime_dependencies_check(settings))
    checks.append(_operator_approval_check(settings))
    checks.append(_emergency_audit_check(settings))
    checks.append(_first_order_audit_check(settings))

    created_pool: asyncpg.Pool | None = None
    active_pool = pool
    if active_pool is None:
        try:
            created_pool = await asyncpg.create_pool(
                dsn=settings.database.dsn,
                min_size=1,
                max_size=1,
            )
        except Exception as exc:  # noqa: BLE001
            checks.append(
                LivePreflightCheck(
                    "database_connection",
                    False,
                    _database_connection_failure_detail(exc),
                )
            )
            return LivePreflightResult(tuple(checks))
        active_pool = created_pool
        checks.append(
            LivePreflightCheck(
                "database_connection",
                True,
                "database connection established",
            )
        )
    else:
        checks.append(
            LivePreflightCheck(
                "database_connection",
                True,
                "database pool supplied by caller",
            )
        )

    try:
        checks.append(await _schema_check(settings, active_pool))
        checks.append(await _market_data_freshness_check(settings, active_pool))
        checks.append(await _submission_unknown_check(settings, active_pool))
        checks.append(await _live_open_orders_check(settings, active_pool))
        active_strategies = await _active_strategies_check(settings, active_pool)
        checks.append(active_strategies.check)
        active_strategies_fingerprint = active_strategies.fingerprint
        checks.append(
            await _venue_reconciliation_check(
                settings,
                active_pool,
                venue_reconciler=venue_reconciler,
                skip_venue=skip_venue or skip_credentials,
                credentials_ok=(
                    credentials_ok or (skip_credentials and non_venue_config_ok)
                ),
            )
        )
    finally:
        if created_pool is not None:
            await created_pool.close()

    return LivePreflightResult(
        tuple(checks),
        active_strategies_fingerprint=active_strategies_fingerprint,
    )


def _settings_with_diagnostic_credentials(settings: PMSSettings) -> PMSSettings:
    """Fill secret-shaped fields only to keep non-credential checks running."""
    polymarket = settings.polymarket.model_copy(
        update={
            "private_key": "diagnostic-polymarket-private-key",
            "api_key": "diagnostic-polymarket-api-key",
            "api_secret": "diagnostic-polymarket-api-secret",
            "api_passphrase": "diagnostic-polymarket-api-passphrase",
            "signature_type": 1,
            "funder_address": "0x1111111111111111111111111111111111111111",
        }
    )
    llm = settings.llm
    if llm.enabled and (llm.api_key is None or llm.api_key.strip() == ""):
        llm = llm.model_copy(update={"api_key": "diagnostic-llm-api-key"})
    discord = settings.discord
    if discord.webhook_url is None:
        discord = discord.model_copy(
            update={
                "webhook_url": SecretStr(
                    "https://discord.example/webhooks/diagnostic/preflight"
                )
            }
        )
    api_token = settings.api_token
    if (
        api_token is None
        or api_token.strip() == ""
        or _looks_like_preflight_placeholder_detail(api_token)
    ):
        api_token = "diagnostic-api-token"
    return settings.model_copy(
        update={
            "secret_source": "fly",
            "local_secret_file": None,
            "api_token": api_token,
            "polymarket": polymarket,
            "llm": llm,
            "discord": discord,
        }
    )


def require_live_preflight_artifact(settings: PMSSettings) -> None:
    path, artifact = _load_live_preflight_artifact(settings)
    _validate_live_preflight_artifact(settings, artifact, path=path)
    _validate_live_strategy_artifacts(settings)


async def require_live_preflight_active_strategies_artifact(
    settings: PMSSettings,
    registry: ActiveStrategyRegistry,
) -> str:
    path, artifact = _load_live_preflight_artifact(settings)
    _validate_live_preflight_artifact(settings, artifact, path=path)
    _validate_live_strategy_artifacts(settings)
    observed = _require_preflight_active_strategies_fingerprint(artifact)
    strategies = await registry.list_active_strategies()
    if not strategies:
        msg = "LIVE credentialed preflight active strategies missing in database"
        raise LiveTradingDisabledError(msg)
    expected = live_preflight_active_strategies_fingerprint(strategies)
    if observed != expected:
        msg = "LIVE credentialed preflight active strategies fingerprint mismatch"
        raise LiveTradingDisabledError(msg)
    return observed


def _load_live_preflight_artifact(
    settings: PMSSettings,
) -> tuple[Path, dict[str, object]]:
    raw_path = settings.live_preflight_artifact_path
    if raw_path is None or raw_path.strip() == "":
        msg = (
            "LIVE credentialed preflight artifact missing: "
            "live_preflight_artifact_path"
        )
        raise LiveTradingDisabledError(msg)

    path = Path(raw_path).expanduser()
    require_preflight_artifact_concrete_path(path)
    require_preflight_artifact_outside_working_tree(path)
    _require_preflight_artifact_parent_owner_writable(path)
    require_preflight_artifact_regular_file_path(path, must_exist=True)
    try:
        artifact = _loads_json_rejecting_duplicate_keys(
            _read_text_no_follow(path),
            label="LIVE credentialed preflight artifact",
        )
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"LIVE credentialed preflight artifact is unreadable: {path}"
        raise LiveTradingDisabledError(msg) from exc
    if not isinstance(artifact, dict):
        msg = "LIVE credentialed preflight artifact must be a JSON object"
        raise LiveTradingDisabledError(msg)
    return path, cast(dict[str, object], artifact)


def require_preflight_artifact_concrete_path(path: Path) -> None:
    if _looks_like_preflight_placeholder_detail(str(path)):
        msg = "LIVE credentialed preflight artifact path contains placeholder"
        raise LiveTradingDisabledError(msg)


def require_preflight_artifact_parent_owner_writable(path: Path) -> None:
    parent = path.parent
    try:
        parent_stat = parent.lstat()
    except FileNotFoundError:
        msg = f"LIVE credentialed preflight artifact parent does not exist: {parent}"
        raise LiveTradingDisabledError(msg)
    if not stat.S_ISDIR(parent_stat.st_mode):
        msg = f"LIVE credentialed preflight artifact parent is not a directory: {parent}"
        raise LiveTradingDisabledError(msg)
    mode = stat.S_IMODE(parent_stat.st_mode)
    if mode & 0o077:
        msg = (
            "LIVE credentialed preflight artifact parent "
            f"{parent} is too permissive; run `chmod 700 {parent}`."
        )
        raise LiveTradingDisabledError(msg)
    if not mode & stat.S_IWUSR:
        msg = (
            "LIVE credentialed preflight artifact parent "
            f"{parent} is not owner-writable; run `chmod 700 {parent}`."
        )
        raise LiveTradingDisabledError(msg)


def require_preflight_artifact_outside_working_tree(path: Path) -> None:
    configured_path = _absolute_path_without_symlink_resolution(path)
    resolved_path = path.expanduser().resolve(strict=False)
    working_tree = _working_tree_root(Path.cwd().resolve(strict=False))
    working_trees = [working_tree]
    for candidate in (configured_path, resolved_path):
        candidate_working_tree = _containing_working_tree_root(candidate)
        if candidate_working_tree is not None:
            working_trees.append(candidate_working_tree)

    for working_tree_candidate in dict.fromkeys(working_trees):
        if working_tree_candidate.parent == working_tree_candidate:
            continue
        for candidate in (configured_path, resolved_path):
            try:
                candidate.relative_to(working_tree_candidate)
            except ValueError:
                continue
            msg = (
                "LIVE credentialed preflight artifact must live outside "
                f"the working tree: {candidate}"
            )
            raise LiveTradingDisabledError(msg)


def require_preflight_artifact_regular_file_path(
    path: Path,
    *,
    must_exist: bool,
) -> None:
    try:
        path_stat = path.expanduser().lstat()
    except FileNotFoundError:
        if not must_exist:
            return
        msg = f"LIVE credentialed preflight artifact does not exist: {path}"
        raise LiveTradingDisabledError(msg) from None
    mode = path_stat.st_mode
    if not stat.S_ISREG(mode):
        msg = (
            "LIVE credentialed preflight artifact path is not a regular file: "
            f"{path}"
        )
        raise LiveTradingDisabledError(msg)
    if path_stat.st_nlink != 1:
        msg = (
            "LIVE credentialed preflight artifact path is not a single-link "
            f"file: {path}"
        )
        raise LiveTradingDisabledError(msg)


def _absolute_path_without_symlink_resolution(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(expanded))


def _working_tree_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def _containing_working_tree_root(path: Path) -> Path | None:
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _require_preflight_artifact_parent_owner_writable(path: Path) -> None:
    require_preflight_artifact_parent_owner_writable(path)


def _validate_live_preflight_artifact(
    settings: PMSSettings,
    artifact: Mapping[str, object],
    *,
    path: Path,
) -> None:
    if artifact.get("generated_by") != "pms-live preflight":
        msg = "LIVE credentialed preflight artifact generated_by is invalid"
        raise LiveTradingDisabledError(msg)
    _require_preflight_output_path_matches(artifact, path=path)
    _require_preflight_settings_fingerprint(artifact, settings=settings)
    _require_preflight_readiness_reports_fingerprint(artifact, settings=settings)
    generated_at = _require_preflight_generated_at(artifact)
    _require_preflight_after_readiness(settings, generated_at=generated_at)
    _require_preflight_after_readiness_reports(settings, generated_at=generated_at)
    _require_preflight_after_emergency_audit(settings, generated_at=generated_at)
    _require_preflight_fresh(settings, generated_at=generated_at)
    if artifact.get("artifact_mode") != "credentialed_preflight":
        msg = "LIVE credentialed preflight artifact_mode must be credentialed_preflight"
        raise LiveTradingDisabledError(msg)
    if artifact.get("final_go_no_go_valid") is not True:
        msg = "LIVE credentialed preflight artifact final_go_no_go_valid must be true"
        raise LiveTradingDisabledError(msg)
    if artifact.get("skip_venue") is not False:
        msg = "LIVE credentialed preflight artifact must not skip venue reconciliation"
        raise LiveTradingDisabledError(msg)
    if artifact.get("skip_credentials") is not False:
        msg = (
            "LIVE credentialed preflight artifact skip_credentials must be false; "
            "must not skip credential validation"
        )
        raise LiveTradingDisabledError(msg)
    if artifact.get("database_url_override_used") is not False:
        msg = (
            "LIVE credentialed preflight artifact must not use "
            "a database-url override"
        )
        raise LiveTradingDisabledError(msg)

    result = artifact.get("result")
    if not isinstance(result, dict) or result.get("ok") is not True:
        msg = "LIVE credentialed preflight artifact result must be ok"
        raise LiveTradingDisabledError(msg)

    checks = result.get("checks")
    if not isinstance(checks, list):
        msg = "LIVE credentialed preflight artifact checks must be a list"
        raise LiveTradingDisabledError(msg)
    malformed_artifact_checks = _malformed_preflight_check_names(checks)
    if malformed_artifact_checks:
        fields = ", ".join(malformed_artifact_checks)
        msg = f"LIVE credentialed preflight artifact malformed checks: {fields}"
        raise LiveTradingDisabledError(msg)
    duplicate_artifact_checks = _duplicate_preflight_check_names(checks)
    if duplicate_artifact_checks:
        fields = ", ".join(duplicate_artifact_checks)
        msg = f"LIVE credentialed preflight artifact duplicate checks: {fields}"
        raise LiveTradingDisabledError(msg)
    unknown_artifact_checks = _unknown_preflight_check_names(checks)
    if unknown_artifact_checks:
        fields = ", ".join(unknown_artifact_checks)
        msg = f"LIVE credentialed preflight artifact unknown checks: {fields}"
        raise LiveTradingDisabledError(msg)
    failed_artifact_checks = _failed_preflight_check_names(checks)
    if failed_artifact_checks:
        fields = ", ".join(failed_artifact_checks)
        msg = (
            "LIVE credentialed preflight artifact contains failed checks: "
            f"{fields}"
        )
        raise LiveTradingDisabledError(msg)
    checks_by_name = _preflight_checks_by_name(checks)
    missing_checks = [
        name for name in _REQUIRED_FINAL_PREFLIGHT_CHECKS if name not in checks_by_name
    ]
    if missing_checks:
        fields = ", ".join(missing_checks)
        msg = f"LIVE credentialed preflight artifact missing checks: {fields}"
        raise LiveTradingDisabledError(msg)

    failed_checks = [
        name
        for name in _REQUIRED_FINAL_PREFLIGHT_CHECKS
        if checks_by_name[name].get("ok") is not True
    ]
    if failed_checks:
        fields = ", ".join(failed_checks)
        msg = f"LIVE credentialed preflight artifact failed checks: {fields}"
        raise LiveTradingDisabledError(msg)
    _require_preflight_active_strategies_fingerprint(artifact)


def live_preflight_active_strategies_fingerprint(
    strategies: Sequence[ActiveStrategy],
) -> str:
    payload: dict[str, object] = {
        "active_strategies": sorted(
            (
                _active_strategy_fingerprint_payload(strategy)
                for strategy in strategies
            ),
            key=lambda item: (
                str(item["strategy_id"]),
                str(item["strategy_version_id"]),
            ),
        )
    }
    return _canonical_sha256(payload)


def _active_strategy_fingerprint_payload(
    strategy: ActiveStrategy,
) -> dict[str, object]:
    return {
        "strategy_id": strategy.strategy_id,
        "strategy_version_id": strategy.strategy_version_id,
        "projection": json.loads(
            serialize_strategy_config_json(
                strategy.config,
                strategy.risk,
                strategy.eval_spec,
                strategy.forecaster,
                strategy.market_selection,
            )
        ),
        "calibration": {
            "enabled": strategy.calibration.enabled,
            "extreme_clamp_high": strategy.calibration.extreme_clamp_high,
            "extreme_clamp_low": strategy.calibration.extreme_clamp_low,
            "min_resolved_for_extreme": strategy.calibration.min_resolved_for_extreme,
            "shrinkage_bias": strategy.calibration.shrinkage_bias,
            "shrinkage_factor": strategy.calibration.shrinkage_factor,
        },
    }


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _require_preflight_settings_fingerprint(
    artifact: Mapping[str, object],
    *,
    settings: PMSSettings,
) -> None:
    observed = artifact.get("settings_fingerprint")
    if not isinstance(observed, str) or observed.strip() == "":
        msg = "LIVE credentialed preflight artifact missing settings fingerprint"
        raise LiveTradingDisabledError(msg)
    expected = live_preflight_settings_fingerprint(settings)
    if observed != expected:
        msg = "LIVE credentialed preflight artifact settings fingerprint mismatch"
        raise LiveTradingDisabledError(msg)


def _require_preflight_readiness_reports_fingerprint(
    artifact: Mapping[str, object],
    *,
    settings: PMSSettings,
) -> None:
    observed = artifact.get("readiness_reports_fingerprint")
    if not isinstance(observed, str) or observed.strip() == "":
        msg = (
            "LIVE credentialed preflight artifact missing readiness reports "
            "fingerprint"
        )
        raise LiveTradingDisabledError(msg)
    expected = live_preflight_readiness_reports_fingerprint(settings)
    if observed != expected:
        msg = "LIVE credentialed preflight artifact readiness reports fingerprint mismatch"
        raise LiveTradingDisabledError(msg)


def _require_preflight_active_strategies_fingerprint(
    artifact: Mapping[str, object],
) -> str:
    observed = artifact.get("active_strategies_fingerprint")
    if not isinstance(observed, str) or observed.strip() == "":
        msg = (
            "LIVE credentialed preflight artifact missing active strategies "
            "fingerprint"
        )
        raise LiveTradingDisabledError(msg)
    if _looks_like_preflight_placeholder_detail(observed):
        msg = (
            "LIVE credentialed preflight artifact active strategies "
            "fingerprint contains placeholder"
        )
        raise LiveTradingDisabledError(msg)
    if not _is_sha256_hexdigest(observed):
        msg = (
            "LIVE credentialed preflight artifact active strategies "
            "fingerprint must be a sha256 hex digest"
        )
        raise LiveTradingDisabledError(msg)
    return observed


def _require_preflight_output_path_matches(
    artifact: Mapping[str, object],
    *,
    path: Path,
) -> None:
    output_path = artifact.get("output_path")
    if not isinstance(output_path, str) or output_path.strip() == "":
        msg = "LIVE credentialed preflight artifact missing output_path"
        raise LiveTradingDisabledError(msg)

    expected = path.resolve(strict=False)
    observed = Path(output_path).expanduser().resolve(strict=False)
    if observed != expected:
        msg = (
            "LIVE credentialed preflight artifact output_path must match "
            f"configured path: {observed} != {expected}"
        )
        raise LiveTradingDisabledError(msg)


def _require_preflight_generated_at(artifact: Mapping[str, object]) -> datetime:
    generated_at = artifact.get("generated_at")
    if not isinstance(generated_at, str) or generated_at.strip() == "":
        msg = "LIVE credentialed preflight artifact missing generated_at"
        raise LiveTradingDisabledError(msg)
    try:
        generated_at_dt = datetime.fromisoformat(
            generated_at.strip().replace("Z", "+00:00")
        )
    except ValueError as exc:
        msg = "LIVE credentialed preflight artifact generated_at is invalid"
        raise LiveTradingDisabledError(msg) from exc
    generated_at_utc = _require_timezone_aware_datetime(
        generated_at_dt,
        label="LIVE credentialed preflight artifact",
    )
    if generated_at_utc > datetime.now(tz=UTC):
        msg = "LIVE credentialed preflight artifact generated_at is in the future"
        raise LiveTradingDisabledError(msg)
    return generated_at_utc


def _require_preflight_after_readiness(
    settings: PMSSettings,
    *,
    generated_at: datetime,
) -> None:
    readiness_timestamps = {
        "live_exit_criteria_ratified_at": settings.live_exit_criteria_ratified_at,
        "live_compliance_reviewed_at": settings.live_compliance_reviewed_at,
    }
    stale_fields = [
        field_name
        for field_name, timestamp_value in readiness_timestamps.items()
        if timestamp_value is not None
        and _coerce_preflight_datetime(timestamp_value) > generated_at
    ]
    if stale_fields:
        fields = ", ".join(stale_fields)
        msg = (
            "LIVE credentialed preflight artifact generated_at predates "
            f"LIVE readiness: {fields}"
        )
        raise LiveTradingDisabledError(msg)


def _require_preflight_after_readiness_reports(
    settings: PMSSettings,
    *,
    generated_at: datetime,
) -> None:
    stale_reports = [
        label
        for label, report_generated_at in _readiness_report_generated_at_values(
            settings
        )
        if report_generated_at > generated_at
    ]
    if stale_reports:
        fields = ", ".join(stale_reports)
        msg = (
            "LIVE credentialed preflight artifact generated_at predates "
            f"readiness reports: {fields}"
        )
        raise LiveTradingDisabledError(msg)


def _require_preflight_after_emergency_audit(
    settings: PMSSettings,
    *,
    generated_at: datetime,
) -> None:
    latest_emergency_audit_at = _latest_live_emergency_audit_timestamp(
        settings.live_emergency_audit_path
    )
    if latest_emergency_audit_at is None or latest_emergency_audit_at <= generated_at:
        return

    msg = (
        "LIVE credentialed preflight artifact generated_at predates "
        "emergency audit: live_emergency_audit_path"
    )
    raise LiveTradingDisabledError(msg)


def _latest_live_emergency_audit_timestamp(raw_path: str | None) -> datetime | None:
    if raw_path is None or raw_path.strip() == "":
        return None
    path = Path(raw_path).expanduser()
    non_regular_detail = _audit_path_non_regular_detail(
        path,
        label="LIVE emergency audit path",
    )
    if non_regular_detail is not None:
        msg = (
            f"{non_regular_detail} for credentialed preflight chronology"
        )
        raise LiveTradingDisabledError(msg)
    try:
        audit_text = _read_text_no_follow(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        msg = (
            "LIVE emergency audit is unreadable for credentialed "
            f"preflight chronology: {path}"
        )
        raise LiveTradingDisabledError(msg) from exc

    latest: datetime | None = None
    for line_number, raw_line in enumerate(audit_text.splitlines(), start=1):
        line = raw_line.strip()
        if line == "":
            continue
        record = _emergency_audit_record(line, path=path, line_number=line_number)
        timestamp = _emergency_audit_record_timestamp(
            record,
            path=path,
            line_number=line_number,
        )
        if latest is None or timestamp > latest:
            latest = timestamp
    return latest


def _emergency_audit_record(
    line: str,
    *,
    path: Path,
    line_number: int,
) -> Mapping[str, object]:
    try:
        record = _loads_json_rejecting_duplicate_keys(
            line,
            label=(
                "LIVE emergency audit record invalid for credentialed "
                f"preflight chronology: {path}:{line_number}"
            ),
        )
    except json.JSONDecodeError as exc:
        msg = (
            "LIVE emergency audit record invalid for credentialed preflight "
            f"chronology: {path}:{line_number}"
        )
        raise LiveTradingDisabledError(msg) from exc
    if not isinstance(record, dict):
        msg = (
            "LIVE emergency audit record invalid for credentialed preflight "
            f"chronology: {path}:{line_number}"
        )
        raise LiveTradingDisabledError(msg)
    return cast(Mapping[str, object], record)


def _emergency_audit_record_timestamp(
    record: Mapping[str, object],
    *,
    path: Path,
    line_number: int,
) -> datetime:
    raw_timestamp = record.get("timestamp")
    if not isinstance(raw_timestamp, str) or raw_timestamp.strip() == "":
        msg = (
            "LIVE emergency audit record invalid for credentialed preflight "
            f"chronology: {path}:{line_number} missing timestamp"
        )
        raise LiveTradingDisabledError(msg)
    try:
        parsed = datetime.fromisoformat(raw_timestamp.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        msg = (
            "LIVE emergency audit record invalid for credentialed preflight "
            f"chronology: {path}:{line_number} invalid timestamp"
        )
        raise LiveTradingDisabledError(msg) from exc
    return _require_timezone_aware_datetime(
        parsed,
        label=(
            "LIVE emergency audit record invalid for credentialed preflight "
            f"chronology: {path}:{line_number}"
        ),
        field_name="timestamp",
    )


def _readiness_report_generated_at_values(
    settings: PMSSettings,
) -> tuple[tuple[str, datetime], ...]:
    return (
        (
            "live_paper_soak_report_path",
            _readiness_report_generated_at(
                settings.live_paper_soak_report_path,
                label="LIVE paper soak GO report",
            ),
        ),
        (
            "live_operator_rehearsal_report_path",
            _readiness_report_generated_at(
                settings.live_operator_rehearsal_report_path,
                label="LIVE operator rehearsal report",
            ),
        ),
        (
            "live_execution_model_path",
            _json_artifact_generated_at(
                settings.live_execution_model_path,
                label="LIVE execution-model artifact",
            ),
        ),
        (
            "live_paper_backtest_diff_path",
            _json_artifact_generated_at(
                settings.live_paper_backtest_diff_path,
                label="LIVE paper-vs-backtest execution diff artifact",
            ),
        ),
    )


def _json_artifact_generated_at(raw_path: str | None, *, label: str) -> datetime:
    if raw_path is None or raw_path.strip() == "":
        msg = f"{label} path missing for credentialed preflight chronology"
        raise LiveTradingDisabledError(msg)
    path = Path(raw_path).expanduser()
    try:
        raw_payload = _loads_json_rejecting_duplicate_keys(
            _read_text_no_follow(path),
            label=label,
        )
    except json.JSONDecodeError as exc:
        msg = f"{label} is invalid JSON for credentialed preflight chronology"
        raise LiveTradingDisabledError(msg) from exc
    except OSError as exc:
        msg = f"{label} is unreadable for credentialed preflight chronology: {path}"
        raise LiveTradingDisabledError(msg) from exc
    if not isinstance(raw_payload, dict):
        msg = f"{label} must be a JSON object for credentialed preflight chronology"
        raise LiveTradingDisabledError(msg)
    generated_at = raw_payload.get("generated_at")
    if not isinstance(generated_at, str) or generated_at.strip() == "":
        msg = f"{label} missing generated_at for credentialed preflight chronology"
        raise LiveTradingDisabledError(msg)
    try:
        parsed = datetime.fromisoformat(generated_at.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        msg = f"{label} generated_at is invalid for credentialed preflight chronology"
        raise LiveTradingDisabledError(msg) from exc
    return _require_timezone_aware_datetime(parsed, label=label)


def _readiness_report_generated_at(raw_path: str | None, *, label: str) -> datetime:
    if raw_path is None or raw_path.strip() == "":
        msg = f"{label} path missing for credentialed preflight chronology"
        raise LiveTradingDisabledError(msg)
    path = Path(raw_path).expanduser()
    try:
        report_text = _read_text_no_follow(path)
    except OSError as exc:
        msg = f"{label} is unreadable for credentialed preflight chronology: {path}"
        raise LiveTradingDisabledError(msg) from exc
    raw_generated_at = _markdown_report_provenance_value(
        report_text,
        field_name="generated_at",
        label=label,
    )
    try:
        generated_at = datetime.fromisoformat(raw_generated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        msg = f"{label} persisted provenance generated_at is invalid"
        raise LiveTradingDisabledError(msg) from exc
    return _require_timezone_aware_datetime(
        generated_at,
        label=label,
        field_name="persisted provenance generated_at",
    )


def _markdown_report_provenance_value(
    report_text: str,
    *,
    field_name: str,
    label: str,
) -> str:
    in_provenance = False
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            in_provenance = line == "## Report Provenance"
            continue
        if not in_provenance or not line.startswith("|"):
            continue
        cells = _markdown_table_cells(line)
        if len(cells) < 2 or cells[0] != field_name:
            continue
        if len(cells) != 2:
            msg = f"{label} persisted provenance malformed {field_name} row"
            raise LiveTradingDisabledError(msg)
        value = cells[1].strip()
        if value == "":
            break
        return value
    msg = f"{label} persisted provenance missing {field_name}"
    raise LiveTradingDisabledError(msg)


def _require_preflight_fresh(
    settings: PMSSettings,
    *,
    generated_at: datetime,
) -> None:
    max_age_s = settings.live_preflight_artifact_max_age_s
    age_s = (datetime.now(tz=UTC) - generated_at).total_seconds()
    if age_s <= max_age_s:
        return

    msg = (
        "LIVE credentialed preflight artifact is stale: "
        f"age {age_s:.1f}s exceeds {max_age_s:.1f}s"
    )
    raise LiveTradingDisabledError(msg)


def _coerce_preflight_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _require_timezone_aware_datetime(
    value: datetime,
    *,
    label: str,
    field_name: str = "generated_at",
) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        msg = f"{label} {field_name} must include timezone"
        raise LiveTradingDisabledError(msg)
    return value.astimezone(UTC)


def _preflight_checks_by_name(
    checks: list[object],
) -> dict[str, Mapping[str, object]]:
    checks_by_name: dict[str, Mapping[str, object]] = {}
    for check in checks:
        if not isinstance(check, dict):
            continue
        name = check.get("name")
        if isinstance(name, str):
            checks_by_name[name] = check
    return checks_by_name


def _malformed_preflight_check_names(checks: list[object]) -> list[str]:
    malformed: list[str] = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            malformed.append(f"malformed[{index}]")
            continue
        raw_name = check.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) else ""
        label = name or f"unnamed[{index}]"
        raw_detail = check.get("detail")
        if (
            name == ""
            or not isinstance(check.get("ok"), bool)
            or not isinstance(raw_detail, str)
            or raw_detail.strip() == ""
            or _looks_like_preflight_placeholder_detail(raw_detail)
        ):
            malformed.append(label)
    return malformed


def _looks_like_preflight_placeholder_detail(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "":
        return False
    placeholder_markers = (
        "fill_in",
        "__fill",
        "todo",
        "replace",
        "placeholder",
    )
    return any(marker in normalized for marker in placeholder_markers)


def _duplicate_preflight_check_names(checks: list[object]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        raw_name = check.get("name")
        if not isinstance(raw_name, str):
            continue
        name = raw_name.strip()
        if name == "":
            continue
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def _unknown_preflight_check_names(checks: list[object]) -> list[str]:
    known = set(_REQUIRED_FINAL_PREFLIGHT_CHECKS)
    unknown: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        raw_name = check.get("name")
        if not isinstance(raw_name, str):
            continue
        name = raw_name.strip()
        if name != "" and name not in known and name not in unknown:
            unknown.append(name)
    return unknown


def _failed_preflight_check_names(checks: list[object]) -> list[str]:
    failed: list[str] = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            failed.append(f"malformed[{index}]")
            continue
        raw_name = check.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) else ""
        if name == "":
            name = f"unnamed[{index}]"
        if check.get("ok") is not True:
            failed.append(name)
    return failed


def _database_connection_failure_detail(exc: Exception) -> str:
    message = redact_database_error(str(exc))
    return f"database connection failed ({type(exc).__name__}): {message}"


def redact_live_error(message: str, settings: PMSSettings) -> str:
    return redact_live_error_values(
        message,
        (
            settings.api_token,
            settings.polymarket.private_key,
            settings.polymarket.api_key,
            settings.polymarket.api_secret,
            settings.polymarket.api_passphrase,
            settings.polymarket.funder_address,
        ),
    )


def _runtime_dependencies_check(settings: PMSSettings) -> LivePreflightCheck:
    try:
        required_modules = live_runtime_dependency_requirements(settings)
    except Exception as exc:  # noqa: BLE001
        return LivePreflightCheck(
            "runtime_dependencies",
            False,
            redact_live_error(str(exc), settings),
        )

    missing = [
        f"{module_name} (install with `{install_command}`)"
        for module_name, install_command in required_modules
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        return LivePreflightCheck(
            "runtime_dependencies",
            False,
            "missing required LIVE dependency module(s): " + ", ".join(missing),
        )

    present_modules = ", ".join(module_name for module_name, _ in required_modules)
    llm_detail = (
        f"LLM provider {settings.llm.provider} enabled"
        if settings.llm.enabled
        else "LLM disabled"
    )
    return LivePreflightCheck(
        "runtime_dependencies",
        True,
        f"required LIVE dependency module(s) importable: {present_modules}; {llm_detail}",
    )


def _validate_live_strategy_artifacts(settings: PMSSettings) -> None:
    validate_live_readiness_reports_for_submission(settings)
    _validate_live_execution_model_artifact(settings)
    _validate_live_paper_backtest_diff_artifact(settings)
    _validate_live_category_prior_artifact(settings)
    _validate_live_flb_calibration_artifact(settings)


def _validate_live_execution_model_artifact(settings: PMSSettings) -> None:
    raw_path = settings.live_execution_model_path
    if raw_path is None or raw_path.strip() == "":
        msg = (
            "LIVE execution-model artifact path is required: "
            "live_execution_model_path"
        )
        raise LiveTradingDisabledError(msg)
    if _looks_like_preflight_placeholder_detail(raw_path):
        msg = "LIVE execution-model artifact path contains placeholder"
        raise LiveTradingDisabledError(msg)
    path = _require_live_strategy_artifact_path(
        raw_path,
        label="LIVE execution-model artifact",
    )
    try:
        raw_text = _read_text_no_follow(path)
    except OSError as exc:
        msg = f"LIVE execution-model artifact is unreadable: {path}"
        raise LiveTradingDisabledError(msg) from exc
    try:
        payload = _loads_json_rejecting_duplicate_keys(
            raw_text,
            label="LIVE execution-model artifact",
        )
    except json.JSONDecodeError as exc:
        msg = "LIVE execution-model artifact must be valid JSON"
        raise LiveTradingDisabledError(msg) from exc
    if not isinstance(payload, dict):
        msg = "LIVE execution-model artifact must be a JSON object"
        raise LiveTradingDisabledError(msg)
    _require_live_execution_model_provenance(
        payload,
        max_age_s=settings.live_readiness_report_max_age_s,
    )
    _require_strategy_artifact_strategy_evidence(
        payload,
        artifact_label="LIVE execution-model artifact",
        expected_labels=_paper_soak_report_strategy_labels(settings),
    )
    _require_live_execution_model_input_hash(payload)
    try:
        execution_model = deserialize_execution_model(payload)
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"LIVE execution-model artifact invalid: {exc}"
        raise LiveTradingDisabledError(msg) from exc
    if execution_model.calibration_source != "telemetry_calibrated":
        msg = "LIVE execution-model artifact must be telemetry_calibrated"
        raise LiveTradingDisabledError(msg)
    if not math.isfinite(execution_model.staleness_ms):
        msg = "LIVE execution-model artifact staleness_ms must be finite"
        raise LiveTradingDisabledError(msg)
    if execution_model.adverse_selection_bps <= 0.0:
        msg = (
            "LIVE execution-model artifact must include positive "
            "adverse_selection_bps"
        )
        raise LiveTradingDisabledError(msg)
    _require_live_execution_model_telemetry_sample_contract(payload)


def _require_live_execution_model_provenance(
    payload: Mapping[str, object],
    *,
    max_age_s: float,
) -> None:
    if payload.get("generated_by") != _LIVE_EXECUTION_MODEL_GENERATED_BY:
        msg = "LIVE execution-model artifact generated_by is invalid"
        raise LiveTradingDisabledError(msg)
    if payload.get("artifact_mode") != _LIVE_EXECUTION_MODEL_ARTIFACT_MODE:
        msg = "LIVE execution-model artifact artifact_mode is invalid"
        raise LiveTradingDisabledError(msg)
    _require_json_artifact_generated_at(
        payload,
        label="LIVE execution-model artifact",
        max_age_s=max_age_s,
    )


def _require_live_execution_model_telemetry_sample_contract(
    payload: Mapping[str, object],
) -> None:
    required_fields = (
        "min_samples",
        "telemetry_sample_count",
        "adverse_selection_sample_count",
        "require_adverse_selection",
    )
    missing = [field for field in required_fields if field not in payload]
    if missing:
        msg = (
            "LIVE execution-model artifact missing telemetry sample contract: "
            f"{', '.join(missing)}"
        )
        raise LiveTradingDisabledError(msg)
    if payload["require_adverse_selection"] is not True:
        msg = "LIVE execution-model artifact require_adverse_selection must be true"
        raise LiveTradingDisabledError(msg)
    min_samples = _require_live_execution_model_positive_integer(
        payload,
        field_name="min_samples",
    )
    if min_samples < _MIN_LIVE_EXECUTION_MODEL_TELEMETRY_SAMPLES:
        msg = (
            "LIVE execution-model artifact min_samples must be at least "
            f"{_MIN_LIVE_EXECUTION_MODEL_TELEMETRY_SAMPLES}"
        )
        raise LiveTradingDisabledError(msg)
    telemetry_sample_count = _require_live_execution_model_positive_integer(
        payload,
        field_name="telemetry_sample_count",
    )
    if telemetry_sample_count < min_samples:
        msg = (
            "LIVE execution-model artifact telemetry_sample_count must be at least "
            "min_samples"
        )
        raise LiveTradingDisabledError(msg)
    adverse_selection_sample_count = _require_live_execution_model_positive_integer(
        payload,
        field_name="adverse_selection_sample_count",
    )
    if adverse_selection_sample_count < min_samples:
        msg = (
            "LIVE execution-model artifact adverse_selection_sample_count must be "
            "at least min_samples"
        )
        raise LiveTradingDisabledError(msg)


def _require_live_execution_model_positive_integer(
    payload: Mapping[str, object],
    *,
    field_name: str,
) -> int:
    raw_value = payload[field_name]
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        msg = (
            "LIVE execution-model artifact telemetry sample contract field "
            f"{field_name} must be a finite positive integer"
        )
        raise LiveTradingDisabledError(msg)
    value = float(raw_value)
    if not math.isfinite(value) or value <= 0.0 or not value.is_integer():
        msg = (
            "LIVE execution-model artifact telemetry sample contract field "
            f"{field_name} must be a finite positive integer"
        )
        raise LiveTradingDisabledError(msg)
    return int(value)


def _validate_live_paper_backtest_diff_artifact(settings: PMSSettings) -> None:
    raw_path = settings.live_paper_backtest_diff_path
    if raw_path is None or raw_path.strip() == "":
        msg = (
            "LIVE paper-vs-backtest execution diff artifact path is required: "
            "live_paper_backtest_diff_path"
        )
        raise LiveTradingDisabledError(msg)
    if _looks_like_preflight_placeholder_detail(raw_path):
        msg = "LIVE paper-vs-backtest execution diff artifact path contains placeholder"
        raise LiveTradingDisabledError(msg)
    path = _require_live_strategy_artifact_path(
        raw_path,
        label="LIVE paper-vs-backtest execution diff artifact",
    )
    try:
        raw_text = _read_text_no_follow(path)
    except OSError as exc:
        msg = (
            "LIVE paper-vs-backtest execution diff artifact is unreadable: "
            f"{path}"
        )
        raise LiveTradingDisabledError(msg) from exc
    try:
        raw_payload = _loads_json_rejecting_duplicate_keys(
            raw_text,
            label="LIVE paper-vs-backtest execution diff artifact",
        )
    except json.JSONDecodeError as exc:
        msg = "LIVE paper-vs-backtest execution diff artifact must be valid JSON"
        raise LiveTradingDisabledError(msg) from exc
    if not isinstance(raw_payload, dict):
        msg = "LIVE paper-vs-backtest execution diff artifact must be a JSON object"
        raise LiveTradingDisabledError(msg)
    payload = cast(dict[str, object], raw_payload)
    if payload.get("generated_by") != _LIVE_PAPER_BACKTEST_DIFF_GENERATED_BY:
        msg = (
            "LIVE paper-vs-backtest execution diff artifact generated_by "
            "is invalid"
        )
        raise LiveTradingDisabledError(msg)
    if payload.get("artifact_mode") != _LIVE_PAPER_BACKTEST_DIFF_ARTIFACT_MODE:
        msg = (
            "LIVE paper-vs-backtest execution diff artifact artifact_mode "
            "is invalid"
        )
        raise LiveTradingDisabledError(msg)
    _require_json_artifact_generated_at(
        payload,
        label="LIVE paper-vs-backtest execution diff artifact",
        max_age_s=settings.live_readiness_report_max_age_s,
    )
    _require_strategy_artifact_strategy_evidence(
        payload,
        artifact_label="LIVE paper-vs-backtest execution diff artifact",
        expected_labels=_paper_soak_report_strategy_labels(settings),
    )
    _require_paper_backtest_diff_input_hashes(payload)
    if payload.get("final_go_no_go_valid") is not True:
        msg = (
            "LIVE paper-vs-backtest execution diff artifact must be final GO"
        )
        raise LiveTradingDisabledError(msg)
    failures = payload.get("failures")
    if not isinstance(failures, list):
        msg = "LIVE paper-vs-backtest execution diff artifact failures must be a list"
        raise LiveTradingDisabledError(msg)
    if failures:
        msg = "LIVE paper-vs-backtest execution diff artifact must not contain failures"
        raise LiveTradingDisabledError(msg)
    _require_paper_backtest_diff_empty_lists(payload)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        msg = "LIVE paper-vs-backtest execution diff artifact must include metrics"
        raise LiveTradingDisabledError(msg)
    missing_metrics = [
        metric
        for metric in _REQUIRED_LIVE_PAPER_BACKTEST_DIFF_METRICS
        if metric not in metrics
    ]
    if missing_metrics:
        msg = (
            "LIVE paper-vs-backtest execution diff artifact missing metrics: "
            f"{', '.join(missing_metrics)}"
        )
        raise LiveTradingDisabledError(msg)
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        msg = "LIVE paper-vs-backtest execution diff artifact must include thresholds"
        raise LiveTradingDisabledError(msg)

    metric_values: dict[str, float] = {}
    for metric_name in _REQUIRED_LIVE_PAPER_BACKTEST_DIFF_METRICS:
        metric_value = _require_paper_backtest_diff_metric_number(
            metrics,
            metric_name=metric_name,
        )
        metric_values[metric_name] = metric_value
        if metric_name in _LIVE_PAPER_BACKTEST_DIFF_COUNT_METRICS:
            if metric_value <= 0.0 or not metric_value.is_integer():
                msg = (
                    "LIVE paper-vs-backtest execution diff artifact metric "
                    f"{metric_name} must be a positive integer count"
                )
                raise LiveTradingDisabledError(msg)
        elif metric_value < 0.0:
            msg = (
                "LIVE paper-vs-backtest execution diff artifact metric "
                f"{metric_name} must be nonnegative"
            )
            raise LiveTradingDisabledError(msg)
    thresholds_map = cast(Mapping[str, object], thresholds)
    min_matched_decisions = (
        _require_paper_backtest_diff_min_matched_decisions_threshold(thresholds_map)
    )
    _require_paper_backtest_diff_metric_consistency(
        metric_values,
        min_matched_decisions=min_matched_decisions,
    )
    _require_paper_backtest_diff_thresholds(
        metric_values,
        thresholds=thresholds_map,
    )


def _require_paper_backtest_diff_metric_number(
    metrics: Mapping[str, object],
    *,
    metric_name: str,
) -> float:
    raw_value = metrics[metric_name]
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        msg = (
            "LIVE paper-vs-backtest execution diff artifact metric "
            f"{metric_name} must be a finite number"
        )
        raise LiveTradingDisabledError(msg)
    value = float(raw_value)
    if not math.isfinite(value):
        msg = (
            "LIVE paper-vs-backtest execution diff artifact metric "
            f"{metric_name} must be a finite number"
        )
        raise LiveTradingDisabledError(msg)
    return value


def _require_paper_backtest_diff_input_hashes(payload: Mapping[str, object]) -> None:
    raw_value = payload.get("input_csv_sha256")
    if not isinstance(raw_value, dict):
        msg = (
            "LIVE paper-vs-backtest execution diff artifact "
            "input_csv_sha256 is required"
        )
        raise LiveTradingDisabledError(msg)
    for field_name in ("paper", "backtest"):
        digest = raw_value.get(field_name)
        if not isinstance(digest, str) or not _is_sha256_hexdigest(digest):
            msg = (
                "LIVE paper-vs-backtest execution diff artifact "
                f"input_csv_sha256.{field_name} must be a sha256 hex digest"
            )
            raise LiveTradingDisabledError(msg)


def _require_live_execution_model_input_hash(payload: Mapping[str, object]) -> None:
    digest = payload.get("input_csv_sha256")
    if not isinstance(digest, str) or not _is_sha256_hexdigest(digest):
        msg = (
            "LIVE execution-model artifact input_csv_sha256 "
            "must be a sha256 hex digest"
        )
        raise LiveTradingDisabledError(msg)


def _require_strategy_artifact_strategy_evidence(
    payload: Mapping[str, object],
    *,
    artifact_label: str,
    expected_labels: Sequence[str],
) -> None:
    raw_value = payload.get("strategy_evidence")
    if not isinstance(raw_value, str) or raw_value.strip() == "":
        msg = f"{artifact_label} strategy_evidence is required"
        raise LiveTradingDisabledError(msg)
    observed_labels = _strategy_evidence_labels(raw_value, artifact_label=artifact_label)
    expected = set(expected_labels)
    observed = set(observed_labels)
    if observed != expected:
        msg = (
            f"{artifact_label} strategy_evidence must match active strategies from "
            "paper-soak GO report: "
            f"expected={', '.join(sorted(expected))}; "
            f"observed={', '.join(sorted(observed))}"
        )
        raise LiveTradingDisabledError(msg)


def _strategy_evidence_labels(
    raw_value: str,
    *,
    artifact_label: str,
) -> tuple[str, ...]:
    labels = tuple(
        label.strip()
        for label in raw_value.split(",")
        if label.strip() != ""
    )
    if (
        not labels
        or len(set(labels)) != len(labels)
        or any(
            label.lower() == "unknown"
            or "@" not in label
            or _looks_like_preflight_placeholder_detail(label)
            for label in labels
        )
    ):
        msg = (
            f"{artifact_label} strategy_evidence must contain concrete "
            "strategy_id@strategy_version_id"
        )
        raise LiveTradingDisabledError(msg)
    return labels


def _require_json_artifact_generated_at(
    payload: Mapping[str, object],
    *,
    label: str,
    max_age_s: float,
) -> datetime:
    raw_generated_at = payload.get("generated_at")
    if not isinstance(raw_generated_at, str) or raw_generated_at.strip() == "":
        msg = f"{label} missing generated_at"
        raise LiveTradingDisabledError(msg)
    try:
        generated_at = datetime.fromisoformat(
            raw_generated_at.strip().replace("Z", "+00:00")
        )
    except ValueError as exc:
        msg = f"{label} generated_at is invalid"
        raise LiveTradingDisabledError(msg) from exc
    generated_at = _require_timezone_aware_datetime(generated_at, label=label)
    now = datetime.now(tz=UTC)
    if generated_at > now:
        msg = f"{label} generated_at is in the future"
        raise LiveTradingDisabledError(msg)
    age_s = (now - generated_at).total_seconds()
    if age_s > max_age_s:
        msg = f"{label} is stale: age {age_s:.1f}s exceeds {max_age_s:.1f}s"
        raise LiveTradingDisabledError(msg)
    return generated_at


def _require_paper_backtest_diff_metric_consistency(
    metrics: Mapping[str, float],
    *,
    min_matched_decisions: int,
) -> None:
    matched_count = metrics["matched_decision_count"]
    paper_count = metrics["paper_decision_count"]
    backtest_count = metrics["backtest_decision_count"]
    if matched_count > paper_count:
        msg = (
            "LIVE paper-vs-backtest execution diff artifact metric "
            "matched_decision_count cannot exceed paper_decision_count"
        )
        raise LiveTradingDisabledError(msg)
    if matched_count > backtest_count:
        msg = (
            "LIVE paper-vs-backtest execution diff artifact metric "
            "matched_decision_count cannot exceed backtest_decision_count"
        )
        raise LiveTradingDisabledError(msg)
    if matched_count < _MIN_LIVE_PAPER_BACKTEST_DIFF_MATCHED_DECISIONS:
        msg = (
            "LIVE paper-vs-backtest execution diff artifact metric "
            "matched_decision_count must be at least "
            f"{_MIN_LIVE_PAPER_BACKTEST_DIFF_MATCHED_DECISIONS}"
        )
        raise LiveTradingDisabledError(msg)
    if matched_count < min_matched_decisions:
        msg = (
            "LIVE paper-vs-backtest execution diff artifact metric "
            "matched_decision_count must be at least min_matched_decisions"
        )
        raise LiveTradingDisabledError(msg)
    if matched_count != paper_count:
        msg = (
            "LIVE paper-vs-backtest execution diff artifact metric "
            "matched_decision_count must equal paper_decision_count when "
            "paper_only_decision_ids is empty"
        )
        raise LiveTradingDisabledError(msg)
    if matched_count != backtest_count:
        msg = (
            "LIVE paper-vs-backtest execution diff artifact metric "
            "matched_decision_count must equal backtest_decision_count when "
            "backtest_only_decision_ids is empty"
        )
        raise LiveTradingDisabledError(msg)
    for metric_name in _LIVE_PAPER_BACKTEST_DIFF_RATE_DELTA_METRICS:
        if metrics[metric_name] > 1.0:
            msg = (
                "LIVE paper-vs-backtest execution diff artifact metric "
                f"{metric_name} must be between 0 and 1"
            )
            raise LiveTradingDisabledError(msg)


def _require_paper_backtest_diff_thresholds(
    metrics: Mapping[str, float],
    *,
    thresholds: Mapping[str, object],
) -> None:
    for metric_name, threshold_name in _LIVE_PAPER_BACKTEST_DIFF_THRESHOLD_METRICS:
        if threshold_name not in thresholds:
            msg = (
                "LIVE paper-vs-backtest execution diff artifact missing "
                f"threshold: {threshold_name}"
            )
            raise LiveTradingDisabledError(msg)
        threshold = _require_paper_backtest_diff_threshold_number(
            thresholds,
            threshold_name=threshold_name,
        )
        if metrics[metric_name] > threshold:
            msg = (
                "LIVE paper-vs-backtest execution diff artifact metric "
                f"{metric_name} exceeds {threshold_name}"
            )
            raise LiveTradingDisabledError(msg)


def _require_paper_backtest_diff_min_matched_decisions_threshold(
    thresholds: Mapping[str, object],
) -> int:
    threshold_name = _LIVE_PAPER_BACKTEST_DIFF_MIN_MATCHED_DECISIONS_THRESHOLD
    if threshold_name not in thresholds:
        msg = (
            "LIVE paper-vs-backtest execution diff artifact missing "
            f"threshold: {threshold_name}"
        )
        raise LiveTradingDisabledError(msg)
    raw_value = thresholds[threshold_name]
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        msg = (
            "LIVE paper-vs-backtest execution diff artifact threshold "
            f"{threshold_name} must be a finite positive integer count"
        )
        raise LiveTradingDisabledError(msg)
    value = float(raw_value)
    if not math.isfinite(value) or value <= 0.0 or not value.is_integer():
        msg = (
            "LIVE paper-vs-backtest execution diff artifact threshold "
            f"{threshold_name} must be a finite positive integer count"
        )
        raise LiveTradingDisabledError(msg)
    threshold = int(value)
    if threshold < _MIN_LIVE_PAPER_BACKTEST_DIFF_MATCHED_DECISIONS:
        msg = (
            "LIVE paper-vs-backtest execution diff artifact threshold "
            f"{threshold_name} must be at least "
            f"{_MIN_LIVE_PAPER_BACKTEST_DIFF_MATCHED_DECISIONS}"
        )
        raise LiveTradingDisabledError(msg)
    return threshold


def _require_paper_backtest_diff_threshold_number(
    thresholds: Mapping[str, object],
    *,
    threshold_name: str,
) -> float:
    raw_value = thresholds[threshold_name]
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        msg = (
            "LIVE paper-vs-backtest execution diff artifact threshold "
            f"{threshold_name} must be a finite nonnegative number"
        )
        raise LiveTradingDisabledError(msg)
    value = float(raw_value)
    if not math.isfinite(value) or value < 0.0:
        msg = (
            "LIVE paper-vs-backtest execution diff artifact threshold "
            f"{threshold_name} must be a finite nonnegative number"
        )
        raise LiveTradingDisabledError(msg)
    return value


def _require_paper_backtest_diff_empty_lists(payload: Mapping[str, object]) -> None:
    for field_name, label in _LIVE_PAPER_BACKTEST_DIFF_EMPTY_LIST_FIELDS:
        value = payload.get(field_name)
        if not isinstance(value, list):
            msg = (
                "LIVE paper-vs-backtest execution diff artifact field "
                f"{field_name} must be a list"
            )
            raise LiveTradingDisabledError(msg)
        if value:
            msg = (
                "LIVE paper-vs-backtest execution diff artifact contains "
                f"{label}"
            )
            raise LiveTradingDisabledError(msg)


def _validate_live_category_prior_artifact(settings: PMSSettings) -> None:
    raw_prior_path = settings.controller.category_prior_observations_path
    if raw_prior_path is None or raw_prior_path.strip() == "":
        msg = (
            "LIVE category-prior artifact path is required: "
            "controller.category_prior_observations_path"
        )
        raise LiveTradingDisabledError(msg)
    if _looks_like_preflight_placeholder_detail(raw_prior_path):
        msg = (
            "LIVE category-prior artifact path contains placeholder: "
            "controller.category_prior_observations_path"
        )
        raise LiveTradingDisabledError(msg)
    prior_path = _require_live_strategy_artifact_path(
        raw_prior_path,
        label="LIVE category-prior artifact",
    )
    try:
        loaded = load_category_prior_observations_csv(prior_path)
    except ValueError as exc:
        msg = f"LIVE category-prior artifact invalid: {exc}"
        raise LiveTradingDisabledError(msg) from exc
    if len(loaded.observations) < settings.controller.category_prior_min_global_samples:
        msg = (
            "LIVE category-prior artifact has too few observations: "
            f"{len(loaded.observations)} < "
            "controller.category_prior_min_global_samples="
            f"{settings.controller.category_prior_min_global_samples}"
        )
        raise LiveTradingDisabledError(msg)


def _validate_live_flb_calibration_artifact(settings: PMSSettings) -> None:
    raw_flb_path = settings.strategies.flb_calibration_path
    if raw_flb_path is None or raw_flb_path.strip() == "":
        msg = (
            "LIVE FLB calibration artifact path is required: "
            "strategies.flb_calibration_path"
        )
        raise LiveTradingDisabledError(msg)
    if _looks_like_preflight_placeholder_detail(raw_flb_path):
        msg = (
            "LIVE FLB calibration artifact path contains placeholder: "
            "strategies.flb_calibration_path"
        )
        raise LiveTradingDisabledError(msg)
    flb_path = _require_live_strategy_artifact_path(
        raw_flb_path,
        label="LIVE FLB calibration artifact",
    )
    try:
        model = load_flb_calibration_csv(
            flb_path,
            min_sample_count=settings.strategies.flb_min_calibration_samples,
        )
        calibration_sha256 = file_sha256_no_follow(
            flb_path,
            label="LIVE FLB calibration artifact",
        )
    except ValueError as exc:
        msg = f"LIVE FLB calibration artifact invalid: {exc}"
        raise LiveTradingDisabledError(msg) from exc
    provenance_path = _require_live_strategy_artifact_path(
        str(flb_calibration_provenance_path(flb_path)),
        label="LIVE FLB calibration provenance JSON",
    )
    try:
        load_flb_calibration_provenance_json(
            provenance_path,
            calibration_csv_sha256=calibration_sha256,
            source_labels=tuple(row.source_label for row in model.calibrations),
            signal_sample_counts={
                row.signal_name: row.sample_count for row in model.calibrations
            },
            min_sample_count=model.min_sample_count,
        )
    except ValueError as exc:
        msg = f"LIVE FLB calibration provenance JSON invalid: {exc}"
        raise LiveTradingDisabledError(msg) from exc


def _require_live_strategy_artifact_path(raw_path: str, *, label: str) -> Path:
    path = Path(raw_path).expanduser()
    outside_working_tree_detail = _outside_working_tree_detail(path, label=label)
    if outside_working_tree_detail is not None:
        raise LiveTradingDisabledError(outside_working_tree_detail)
    _require_live_strategy_artifact_parent_owner_writable(path, label=label)
    _require_live_strategy_artifact_regular_file(path, label=label)
    return path


def _require_live_strategy_artifact_parent_owner_writable(
    path: Path,
    *,
    label: str,
) -> None:
    parent = path.parent
    try:
        parent_stat = parent.lstat()
    except FileNotFoundError as exc:
        msg = f"{label} parent directory does not exist: {parent}"
        raise LiveTradingDisabledError(msg) from exc
    if not stat.S_ISDIR(parent_stat.st_mode):
        msg = f"{label} parent path is not a directory: {parent}"
        raise LiveTradingDisabledError(msg)
    mode = stat.S_IMODE(parent_stat.st_mode)
    if mode & 0o077:
        msg = f"{label} parent directory {parent} is too permissive; run chmod 700"
        raise LiveTradingDisabledError(msg)
    if not mode & stat.S_IWUSR:
        msg = f"{label} parent directory {parent} is not owner-writable; run chmod 700"
        raise LiveTradingDisabledError(msg)


def _require_live_strategy_artifact_regular_file(
    path: Path,
    *,
    label: str,
) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError as exc:
        msg = f"{label} does not exist: {path}"
        raise LiveTradingDisabledError(msg) from exc
    if not stat.S_ISREG(path_stat.st_mode):
        msg = f"{label} is not a regular file: {path}"
        raise LiveTradingDisabledError(msg)
    if path_stat.st_nlink != 1:
        msg = f"{label} is not a single-link file: {path}"
        raise LiveTradingDisabledError(msg)


def _operator_approval_check(settings: PMSSettings) -> LivePreflightCheck:
    approval_path = settings.polymarket.first_live_order_approval_path
    if approval_path is None or approval_path.strip() == "":
        return LivePreflightCheck(
            "operator_approval",
            False,
            "polymarket.first_live_order_approval_path is required",
        )

    path = Path(approval_path).expanduser()
    outside_working_tree_detail = _outside_working_tree_detail(
        path,
        label="approval path",
    )
    if outside_working_tree_detail is not None:
        return LivePreflightCheck(
            "operator_approval",
            False,
            outside_working_tree_detail,
        )
    parent = path.parent
    try:
        parent_stat = parent.lstat()
    except FileNotFoundError:
        return LivePreflightCheck(
            "operator_approval",
            False,
            f"approval parent directory does not exist: {parent}",
        )
    if not stat.S_ISDIR(parent_stat.st_mode):
        return LivePreflightCheck(
            "operator_approval",
            False,
            f"approval parent path is not a directory: {parent}",
        )

    mode = stat.S_IMODE(parent_stat.st_mode)
    if mode & 0o077:
        return LivePreflightCheck(
            "operator_approval",
            False,
            f"approval parent directory {parent} is too permissive; run chmod 700",
        )
    if not mode & stat.S_IWUSR:
        return LivePreflightCheck(
            "operator_approval",
            False,
            f"approval parent directory {parent} is not owner-writable; run chmod 700",
        )
    try:
        approval_path_mode = path.lstat().st_mode
    except FileNotFoundError:
        approval_path_mode = None
    if approval_path_mode is not None and not stat.S_ISREG(approval_path_mode):
        return LivePreflightCheck(
            "operator_approval",
            False,
            f"approval path is not a regular file: {path}",
        )
    if approval_path_mode is not None:
        return LivePreflightCheck(
            "operator_approval",
            False,
            (
                f"stale approval file already exists at {path}; remove it "
                "before final preflight and create approval only after preview review"
            ),
        )
    stale_sidecar_detail = _operator_approval_sidecar_detail(path)
    if stale_sidecar_detail is not None:
        return LivePreflightCheck(
            "operator_approval",
            False,
            stale_sidecar_detail,
        )

    return LivePreflightCheck(
        "operator_approval",
        True,
        (
            f"path={path} mode={settings.polymarket.operator_approval_mode}; "
            "approval JSON is expected to be created only after preview review"
        ),
    )


def _operator_approval_sidecar_detail(path: Path) -> str | None:
    sidecar_path = Path(str(path) + ".meta.json")
    try:
        sidecar_stat = sidecar_path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(sidecar_stat.st_mode):
        return f"approval sidecar path is not a regular file: {sidecar_path}"
    if sidecar_stat.st_nlink != 1:
        return f"approval sidecar path is not a single-link file: {sidecar_path}"
    return (
        f"stale approval sidecar already exists at {sidecar_path}; remove it "
        "before final preflight and create approval only after preview review"
    )


def _emergency_audit_check(settings: PMSSettings) -> LivePreflightCheck:
    raw_path = settings.live_emergency_audit_path
    if raw_path.strip() == "":
        return LivePreflightCheck(
            "emergency_audit",
            False,
            "live_emergency_audit_path is required",
        )

    path = Path(raw_path).expanduser()
    outside_working_tree_detail = _outside_working_tree_detail(
        path,
        label="emergency audit path",
    )
    if outside_working_tree_detail is not None:
        return LivePreflightCheck(
            "emergency_audit",
            False,
            outside_working_tree_detail,
        )
    parent = path.parent
    try:
        parent_stat = parent.lstat()
    except FileNotFoundError:
        return LivePreflightCheck(
            "emergency_audit",
            False,
            f"emergency audit parent directory does not exist: {parent}",
        )
    if not stat.S_ISDIR(parent_stat.st_mode):
        return LivePreflightCheck(
            "emergency_audit",
            False,
            f"emergency audit parent path is not a directory: {parent}",
        )

    mode = stat.S_IMODE(parent_stat.st_mode)
    if mode & 0o077:
        return LivePreflightCheck(
            "emergency_audit",
            False,
            (
                f"emergency audit parent directory {parent} is too "
                "permissive; run chmod 700"
            ),
        )
    if not mode & stat.S_IWUSR:
        return LivePreflightCheck(
            "emergency_audit",
            False,
            (
                f"emergency audit parent directory {parent} is not "
                "owner-writable; run chmod 700"
            ),
        )
    non_regular_detail = _audit_path_non_regular_detail(
        path,
        label="emergency audit path",
    )
    if non_regular_detail is not None:
        return LivePreflightCheck(
            "emergency_audit",
            False,
            non_regular_detail,
        )

    try:
        _latest_live_emergency_audit_timestamp(str(path))
    except LiveTradingDisabledError as exc:
        return LivePreflightCheck(
            "emergency_audit",
            False,
            str(exc),
        )

    return LivePreflightCheck(
        "emergency_audit",
        True,
        f"path={path}",
    )


def _first_order_audit_check(settings: PMSSettings) -> LivePreflightCheck:
    raw_path = settings.live_first_order_audit_path
    if raw_path.strip() == "":
        return LivePreflightCheck(
            "first_order_audit",
            False,
            "live_first_order_audit_path is required",
        )

    path = Path(raw_path).expanduser()
    emergency_path = Path(settings.live_emergency_audit_path).expanduser()
    if path == emergency_path:
        return LivePreflightCheck(
            "first_order_audit",
            False,
            "first-order audit path must be distinct from live_emergency_audit_path",
        )

    outside_working_tree_detail = _outside_working_tree_detail(
        path,
        label="first-order audit path",
    )
    if outside_working_tree_detail is not None:
        return LivePreflightCheck(
            "first_order_audit",
            False,
            outside_working_tree_detail,
        )
    parent = path.parent
    try:
        parent_stat = parent.lstat()
    except FileNotFoundError:
        return LivePreflightCheck(
            "first_order_audit",
            False,
            f"first-order audit parent directory does not exist: {parent}",
        )
    if not stat.S_ISDIR(parent_stat.st_mode):
        return LivePreflightCheck(
            "first_order_audit",
            False,
            f"first-order audit parent path is not a directory: {parent}",
        )

    mode = stat.S_IMODE(parent_stat.st_mode)
    if mode & 0o077:
        return LivePreflightCheck(
            "first_order_audit",
            False,
            (
                f"first-order audit parent directory {parent} is too "
                "permissive; run chmod 700"
            ),
        )
    if not mode & stat.S_IWUSR:
        return LivePreflightCheck(
            "first_order_audit",
            False,
            (
                f"first-order audit parent directory {parent} is not "
                "owner-writable; run chmod 700"
            ),
        )
    non_regular_detail = _audit_path_non_regular_detail(
        path,
        label="first-order audit path",
    )
    if non_regular_detail is not None:
        return LivePreflightCheck(
            "first_order_audit",
            False,
            non_regular_detail,
        )

    return LivePreflightCheck(
        "first_order_audit",
        True,
        f"path={path}; distinct from live_emergency_audit_path",
    )


def _audit_path_non_regular_detail(path: Path, *, label: str) -> str | None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return None
    mode = path_stat.st_mode
    if stat.S_ISREG(mode):
        if path_stat.st_nlink != 1:
            return f"{label} is not a single-link file: {path}"
        return None
    return f"{label} is not a regular file: {path}"


def _outside_working_tree_detail(path: Path, *, label: str) -> str | None:
    configured_path = _absolute_path_without_symlink_resolution(path)
    resolved_path = path.expanduser().resolve(strict=False)
    working_tree = _working_tree_root(Path.cwd().resolve(strict=False))
    working_trees = [working_tree]
    for candidate in (configured_path, resolved_path):
        candidate_working_tree = _containing_working_tree_root(candidate)
        if candidate_working_tree is not None:
            working_trees.append(candidate_working_tree)

    for working_tree_candidate in dict.fromkeys(working_trees):
        if working_tree_candidate.parent == working_tree_candidate:
            continue
        for candidate in (configured_path, resolved_path):
            try:
                candidate.relative_to(working_tree_candidate)
            except ValueError:
                continue
            return f"{label} must live outside the working tree: {candidate}"
    return None


async def _schema_check(
    settings: PMSSettings,
    pool: asyncpg.Pool,
) -> LivePreflightCheck:
    try:
        await ensure_schema_current(pool)
    except Exception as exc:  # noqa: BLE001
        return LivePreflightCheck(
            "schema_current",
            False,
            redact_live_error(str(exc), settings),
        )
    return LivePreflightCheck(
        "schema_current",
        True,
        "database schema is at Alembic head",
    )


async def _market_data_freshness_check(
    settings: PMSSettings,
    pool: asyncpg.Pool,
) -> LivePreflightCheck:
    try:
        latest_age_s = await _latest_book_snapshot_age_s(pool)
        latest_usable_age_s = await _latest_usable_book_snapshot_age_s(pool)
    except Exception as exc:  # noqa: BLE001
        return LivePreflightCheck(
            "market_data_freshness",
            False,
            redact_live_error(str(exc), settings),
        )

    if latest_age_s is None:
        return LivePreflightCheck(
            "market_data_freshness",
            False,
            "no book_snapshots rows found; live market-data ingestion is not proven",
        )

    max_age_s = settings.dashboard.stale_snapshot_threshold_s
    if latest_age_s > max_age_s:
        return LivePreflightCheck(
            "market_data_freshness",
            False,
            (
                f"latest book snapshot age {latest_age_s:.1f}s exceeds "
                f"{max_age_s:.1f}s stale threshold"
            ),
        )

    if latest_usable_age_s is None:
        return LivePreflightCheck(
            "market_data_freshness",
            False,
            (
                "no two-sided book snapshot with positive BUY and SELL depth "
                "found; live market-data depth is not proven"
            ),
        )

    if latest_usable_age_s > max_age_s:
        return LivePreflightCheck(
            "market_data_freshness",
            False,
            (
                f"latest usable book snapshot age {latest_usable_age_s:.1f}s "
                f"exceeds {max_age_s:.1f}s stale threshold"
            ),
        )

    try:
        missing_subscribed_usable_token_count = (
            await _fresh_usable_launch_token_missing_count(
                pool,
                max_age_s=max_age_s,
            )
        )
    except Exception as exc:  # noqa: BLE001
        return LivePreflightCheck(
            "market_data_freshness",
            False,
            redact_live_error(str(exc), settings),
        )
    if missing_subscribed_usable_token_count > 0:
        noun = (
            "token"
            if missing_subscribed_usable_token_count == 1
            else "tokens"
        )
        return LivePreflightCheck(
            "market_data_freshness",
            False,
            (
                f"{missing_subscribed_usable_token_count} launch {noun} "
                "lack fresh usable book depth; live launch subscription "
                "and strategy freshness is not proven"
            ),
        )

    if settings.risk.max_exposure_per_risk_group is not None:
        try:
            missing_risk_metadata_count = (
                await _fresh_usable_book_market_missing_risk_metadata_count(
                    pool,
                    max_age_s=max_age_s,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return LivePreflightCheck(
                "market_data_freshness",
                False,
                redact_live_error(str(exc), settings),
            )
        if missing_risk_metadata_count > 0:
            noun = (
                "market"
                if missing_risk_metadata_count == 1
                else "markets"
            )
            return LivePreflightCheck(
                "market_data_freshness",
                False,
                (
                    f"{missing_risk_metadata_count} fresh usable {noun} lack "
                    "markets.risk_group_id; live risk-group caps would reject "
                    "their decisions"
                ),
            )

    return LivePreflightCheck(
        "market_data_freshness",
        True,
        (
            f"latest book snapshot age {latest_age_s:.1f}s within "
            f"{max_age_s:.1f}s stale threshold; latest usable book snapshot "
            f"age {latest_usable_age_s:.1f}s within {max_age_s:.1f}s stale "
            "threshold"
        ),
    )


async def ensure_live_market_data_freshness(
    settings: PMSSettings,
    pool: asyncpg.Pool,
) -> None:
    check = await _market_data_freshness_check(settings, pool)
    if check.ok:
        return
    msg = f"LIVE market data freshness check failed: {check.detail}"
    raise RuntimeError(msg)


async def _latest_book_snapshot_age_s(pool: asyncpg.Pool) -> float | None:
    async with pool.acquire() as connection:
        value = await connection.fetchval(
            """
            SELECT EXTRACT(EPOCH FROM (NOW() - MAX(ts)))::double precision
            FROM book_snapshots
            WHERE ts IS NOT NULL
            """
        )
    if value is None:
        return None
    return max(0.0, float(value))


async def _latest_usable_book_snapshot_age_s(pool: asyncpg.Pool) -> float | None:
    async with pool.acquire() as connection:
        value = await connection.fetchval(
            """
            WITH usable_book_snapshots AS (
                SELECT book_snapshots.id, book_snapshots.ts
                FROM book_snapshots
                JOIN book_levels AS bid_levels
                    ON bid_levels.snapshot_id = book_snapshots.id
                   AND bid_levels.side = 'BUY'
                   AND bid_levels.size > 0.0
                   AND bid_levels.price > 0.0
                JOIN book_levels AS ask_levels
                    ON ask_levels.snapshot_id = book_snapshots.id
                   AND ask_levels.side = 'SELL'
                   AND ask_levels.size > 0.0
                   AND ask_levels.price > 0.0
                WHERE book_snapshots.ts IS NOT NULL
                GROUP BY book_snapshots.id, book_snapshots.ts
            )
            SELECT EXTRACT(EPOCH FROM (NOW() - MAX(ts)))::double precision
            FROM usable_book_snapshots
            """
        )
    if value is None:
        return None
    return max(0.0, float(value))


_LIVE_LAUNCH_TOKENS_CTE = """
manual_tokens AS (
    SELECT token_id
    FROM market_subscriptions
    WHERE source = 'user'
),
active_strategy_specs AS (
    SELECT
        versions.config_json->'market_selection' AS market_selection,
        CASE
            WHEN versions.config_json->'market_selection' ? 'spread_max_bps'
                THEN NULLIF(
                    versions.config_json->'market_selection'->>'spread_max_bps',
                    ''
                )::double precision
            ELSE 100.0
        END AS spread_max_bps,
        CASE
            WHEN versions.config_json->'market_selection' ? 'depth_min_usdc'
                THEN NULLIF(
                    versions.config_json->'market_selection'->>'depth_min_usdc',
                    ''
                )::double precision
            ELSE 250.0
        END AS depth_min_usdc,
        COALESCE(
            (
                versions.config_json->'market_selection'->>'accepting_orders'
            )::boolean,
            true
        ) AS accepting_orders
    FROM strategies
    JOIN strategy_versions AS versions
        ON versions.strategy_version_id = strategies.active_version_id
    WHERE strategies.active_version_id IS NOT NULL
      AND strategies.archived IS NOT TRUE
),
active_strategy_tokens AS (
    SELECT tokens.token_id
    FROM active_strategy_specs
    JOIN markets
        ON markets.venue = COALESCE(
            active_strategy_specs.market_selection->>'venue',
            'polymarket'
        )
    JOIN tokens
        ON tokens.condition_id = markets.condition_id
    WHERE tokens.token_id IS NOT NULL
      AND tokens.outcome IN ('YES', 'NO')
      AND (
          COALESCE(
              NULLIF(
                  active_strategy_specs.market_selection->>'volume_min_usdc',
                  ''
              )::double precision,
              0.0
          ) <= 0.0
          OR (
              markets.volume_24h IS NOT NULL
              AND markets.volume_24h >= COALESCE(
                  NULLIF(
                      active_strategy_specs.market_selection->>'volume_min_usdc',
                      ''
                  )::double precision,
                  0.0
              )
          )
      )
      AND (
          (
              NULLIF(
                  active_strategy_specs.market_selection
                      ->>'resolution_time_max_horizon_days',
                  ''
              ) IS NULL
              AND (
                  markets.resolves_at IS NULL
                  OR markets.resolves_at > NOW()
              )
          )
          OR (
              NULLIF(
                  active_strategy_specs.market_selection
                      ->>'resolution_time_max_horizon_days',
                  ''
              ) IS NOT NULL
              AND markets.resolves_at IS NOT NULL
              AND markets.resolves_at > NOW()
              AND markets.resolves_at <= (
                  NOW() + (
                      NULLIF(
                          active_strategy_specs.market_selection
                              ->>'resolution_time_max_horizon_days',
                          ''
                      )::integer * INTERVAL '1 day'
                  )
              )
          )
      )
      AND (
          active_strategy_specs.accepting_orders IS NOT TRUE
          OR markets.accepting_orders IS NOT FALSE
      )
      AND (
          NULLIF(
              active_strategy_specs.market_selection->>'liquidity_min_usdc',
              ''
          ) IS NULL
          OR (
              markets.liquidity IS NOT NULL
              AND markets.liquidity >= NULLIF(
                  active_strategy_specs.market_selection->>'liquidity_min_usdc',
                  ''
              )::double precision
          )
      )
      AND (
          active_strategy_specs.spread_max_bps IS NULL
          OR (
              markets.spread_bps IS NOT NULL
              AND markets.spread_bps <= active_strategy_specs.spread_max_bps
          )
      )
      AND (
          active_strategy_specs.depth_min_usdc IS NULL
          OR (
              markets.liquidity IS NOT NULL
              AND markets.liquidity >= active_strategy_specs.depth_min_usdc
          )
      )
      AND (
          NULLIF(
              active_strategy_specs.market_selection->>'yes_price_min',
              ''
          ) IS NULL
          OR (
              markets.yes_price IS NOT NULL
              AND markets.yes_price >= NULLIF(
                  active_strategy_specs.market_selection->>'yes_price_min',
                  ''
              )::double precision
          )
      )
      AND (
          NULLIF(
              active_strategy_specs.market_selection->>'yes_price_max',
              ''
          ) IS NULL
          OR (
              markets.yes_price IS NOT NULL
              AND markets.yes_price <= NULLIF(
                  active_strategy_specs.market_selection->>'yes_price_max',
                  ''
              )::double precision
          )
      )
),
launch_tokens AS (
    SELECT token_id FROM manual_tokens
    UNION
    SELECT token_id FROM active_strategy_tokens
)
"""


async def _fresh_usable_launch_token_missing_count(
    pool: asyncpg.Pool,
    *,
    max_age_s: float,
) -> int:
    async with pool.acquire() as connection:
        value = await connection.fetchval(
            f"""
            WITH {_LIVE_LAUNCH_TOKENS_CTE},
            latest_usable_launch_tokens AS (
                SELECT book_snapshots.token_id, MAX(book_snapshots.ts) AS latest_ts
                FROM book_snapshots
                JOIN launch_tokens
                    ON launch_tokens.token_id = book_snapshots.token_id
                JOIN book_levels AS bid_levels
                    ON bid_levels.snapshot_id = book_snapshots.id
                   AND bid_levels.side = 'BUY'
                   AND bid_levels.size > 0.0
                   AND bid_levels.price > 0.0
                JOIN book_levels AS ask_levels
                    ON ask_levels.snapshot_id = book_snapshots.id
                   AND ask_levels.side = 'SELL'
                   AND ask_levels.size > 0.0
                   AND ask_levels.price > 0.0
                WHERE book_snapshots.ts IS NOT NULL
                  AND book_snapshots.ts >= (
                      NOW() - ($1::double precision * INTERVAL '1 second')
                  )
                GROUP BY book_snapshots.token_id
            ),
            missing_launch_usable_tokens AS (
                SELECT launch_tokens.token_id
                FROM launch_tokens
                LEFT JOIN latest_usable_launch_tokens
                    ON latest_usable_launch_tokens.token_id = launch_tokens.token_id
                WHERE latest_usable_launch_tokens.token_id IS NULL
            )
            SELECT COUNT(*)::bigint
            FROM missing_launch_usable_tokens
            """,
            max_age_s,
        )
    return int(value or 0)


async def _fresh_usable_book_market_missing_risk_metadata_count(
    pool: asyncpg.Pool,
    *,
    max_age_s: float,
) -> int:
    async with pool.acquire() as connection:
        value = await connection.fetchval(
            f"""
            WITH {_LIVE_LAUNCH_TOKENS_CTE},
            fresh_usable_launch_markets AS (
                SELECT DISTINCT book_snapshots.market_id
                FROM book_snapshots
                JOIN launch_tokens
                    ON launch_tokens.token_id = book_snapshots.token_id
                JOIN book_levels AS bid_levels
                    ON bid_levels.snapshot_id = book_snapshots.id
                   AND bid_levels.side = 'BUY'
                   AND bid_levels.size > 0.0
                   AND bid_levels.price > 0.0
                JOIN book_levels AS ask_levels
                    ON ask_levels.snapshot_id = book_snapshots.id
                   AND ask_levels.side = 'SELL'
                   AND ask_levels.size > 0.0
                   AND ask_levels.price > 0.0
                WHERE book_snapshots.ts IS NOT NULL
                  AND book_snapshots.ts >= (
                      NOW() - ($1::double precision * INTERVAL '1 second')
                  )
            ),
            missing_market_risk_metadata AS (
                SELECT fresh_usable_launch_markets.market_id
                FROM fresh_usable_launch_markets
                LEFT JOIN markets
                    ON markets.condition_id = fresh_usable_launch_markets.market_id
                WHERE markets.condition_id IS NULL
                   OR (
                      markets.risk_group_id IS NULL
                      OR btrim(markets.risk_group_id) = ''
                  )
            )
            SELECT COUNT(*)::bigint
            FROM missing_market_risk_metadata
            """,
            max_age_s,
        )
    return int(value or 0)


async def _submission_unknown_check(
    settings: PMSSettings,
    pool: asyncpg.Pool,
) -> LivePreflightCheck:
    try:
        count = await _unresolved_submission_unknown_count(pool)
    except Exception as exc:  # noqa: BLE001
        return LivePreflightCheck(
            "submission_unknown",
            False,
            redact_live_error(str(exc), settings),
        )
    if count > 0:
        return LivePreflightCheck(
            "submission_unknown",
            False,
            (
                f"{count} unresolved submission_unknown incident(s); run "
                "pms-live reconcile-submission-unknown after venue reconciliation"
            ),
        )
    return LivePreflightCheck(
        "submission_unknown",
        True,
        "no unresolved submission_unknown incidents",
    )


async def _unresolved_submission_unknown_count(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as connection:
        count = await connection.fetchval(
            """
            SELECT COUNT(*)
            FROM order_intents
            WHERE outcome = 'submission_unknown'
              AND reconciled_at IS NULL
            """
        )
    return int(count or 0)


async def _live_open_orders_check(
    settings: PMSSettings,
    pool: asyncpg.Pool,
) -> LivePreflightCheck:
    try:
        count = await _persisted_live_open_order_count(pool)
    except Exception as exc:  # noqa: BLE001
        return LivePreflightCheck(
            "live_open_orders",
            False,
            redact_live_error(str(exc), settings),
        )
    if count > 0:
        return LivePreflightCheck(
            "live_open_orders",
            False,
            (
                f"{count} persisted live open order(s); PMS has no durable "
                "live open-order ledger yet, so cancel/reconcile venue and DB "
                "before LIVE restart"
            ),
        )
    return LivePreflightCheck(
        "live_open_orders",
        True,
        "no persisted live open orders",
    )


async def _persisted_live_open_order_count(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as connection:
        count = await connection.fetchval(
            """
            SELECT COUNT(*)
            FROM orders
            WHERE venue = 'polymarket'
              AND LOWER(COALESCE(status, '')) IN ('live', 'open', 'unmatched', 'partial')
              AND remaining_notional_usdc > 1e-9
            """
        )
    return int(count or 0)


async def _active_strategies_check(
    settings: PMSSettings,
    pool: asyncpg.Pool,
) -> _ActiveStrategiesPreflight:
    registry = PostgresStrategyRegistry(
        pool,
        llm_identity=llm_identity_for_settings(settings),
    )
    try:
        strategies = await registry.list_active_strategies()
    except Exception as exc:  # noqa: BLE001
        detail = redact_live_error(str(exc), settings)
        return _ActiveStrategiesPreflight(
            LivePreflightCheck(
                "active_strategies",
                False,
                f"active strategy validation failed ({type(exc).__name__}): {detail}",
            ),
            None,
        )
    if not strategies:
        return _ActiveStrategiesPreflight(
            LivePreflightCheck(
                "active_strategies",
                False,
                "no active strategy versions configured",
            ),
            None,
        )
    try:
        ControllerPipelineFactory(settings=settings).build_many(strategies)
        _require_paper_soak_report_covers_active_strategies(settings, strategies)
    except Exception as exc:  # noqa: BLE001
        detail = redact_live_error(str(exc), settings)
        return _ActiveStrategiesPreflight(
            LivePreflightCheck(
                "active_strategies",
                False,
                f"active strategy validation failed ({type(exc).__name__}): {detail}",
            ),
            None,
        )
    fingerprint = live_preflight_active_strategies_fingerprint(strategies)
    strategy_labels = ", ".join(
        f"{strategy.strategy_id}@{strategy.strategy_version_id}"
        for strategy in strategies
    )
    return _ActiveStrategiesPreflight(
        LivePreflightCheck(
            "active_strategies",
            True,
            (
                f"{len(strategies)} active strategy version(s) validate for "
                f"LIVE: {strategy_labels}"
            ),
        ),
        fingerprint,
    )


def _require_paper_soak_report_covers_active_strategies(
    settings: PMSSettings,
    strategies: Sequence[ActiveStrategy],
) -> None:
    expected_labels = {
        f"{strategy.strategy_id}@{strategy.strategy_version_id}"
        for strategy in strategies
    }
    observed_labels = set(_paper_soak_report_strategy_labels(settings))
    missing_labels = sorted(expected_labels - observed_labels)
    if not missing_labels:
        return

    fields = ", ".join(missing_labels)
    msg = (
        "LIVE paper-soak GO report strategy mismatch; missing active "
        f"strategy version(s): {fields}"
    )
    raise LiveTradingDisabledError(msg)


def _paper_soak_report_strategy_labels(settings: PMSSettings) -> tuple[str, ...]:
    raw_path = settings.live_paper_soak_report_path
    if raw_path is None or raw_path.strip() == "":
        msg = "LIVE paper-soak GO report missing for active strategy binding"
        raise LiveTradingDisabledError(msg)
    path = Path(raw_path).expanduser()
    try:
        report_text = _read_text_no_follow(path)
    except OSError as exc:
        msg = f"LIVE paper-soak GO report is unreadable for active strategy binding: {path}"
        raise LiveTradingDisabledError(msg) from exc
    strategy_label = _markdown_summary_strategy_value(report_text)
    if (
        strategy_label.strip() == ""
        or strategy_label.strip().lower() == "unknown"
        or _looks_like_preflight_placeholder_detail(strategy_label)
    ):
        msg = "LIVE paper-soak GO report missing concrete strategy evidence"
        raise LiveTradingDisabledError(msg)
    labels = tuple(
        label.strip()
        for label in strategy_label.split(",")
        if label.strip() != ""
    )
    if not labels:
        msg = "LIVE paper-soak GO report missing concrete strategy evidence"
        raise LiveTradingDisabledError(msg)
    return labels


def _markdown_summary_strategy_value(report_text: str) -> str:
    in_summary = False
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            in_summary = line == "## Summary"
            continue
        if not in_summary or not line.startswith("|"):
            continue
        cells = _markdown_table_cells(line)
        if len(cells) < 2 or cells[0] != "Strategy":
            continue
        if len(cells) != 3:
            msg = "LIVE paper-soak GO report malformed Summary Strategy row"
            raise LiveTradingDisabledError(msg)
        return cells[1].strip()
    msg = "LIVE paper-soak GO report missing Summary Strategy row"
    raise LiveTradingDisabledError(msg)


def _markdown_table_cells(line: str) -> list[str]:
    cells: list[str] = []
    current: list[str] = []
    consecutive_backslashes = 0
    for character in line.strip():
        if character == "|" and consecutive_backslashes % 2 == 0:
            cells.append("".join(current))
            current = []
        else:
            current.append(character)

        if character == "\\":
            consecutive_backslashes += 1
        else:
            consecutive_backslashes = 0
    cells.append("".join(current))

    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return [_unescape_markdown_table_cell(cell.strip()) for cell in cells]


def _unescape_markdown_table_cell(value: str) -> str:
    chars: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if (
            character == "\\"
            and index + 1 < len(value)
            and value[index + 1] == "|"
        ):
            chars.append("|")
            index += 2
            continue
        chars.append(character)
        index += 1
    return "".join(chars)


async def _venue_reconciliation_check(
    settings: PMSSettings,
    pool: asyncpg.Pool,
    *,
    venue_reconciler: PolymarketVenueAccountReconciler | None,
    skip_venue: bool,
    credentials_ok: bool,
) -> LivePreflightCheck:
    if not credentials_ok:
        return LivePreflightCheck(
            "venue_reconciliation",
            False,
            "skipped because LIVE config validation failed",
        )
    if skip_venue:
        return LivePreflightCheck(
            "venue_reconciliation",
            False,
            (
                "skipped by operator flag; incomplete preflight is not valid "
                "for final live go/no-go"
            ),
        )

    credentials = validate_live_mode_ready(settings)
    if venue_reconciler is None:
        from pms.actuator.adapters.polymarket import PolymarketVenueAccountReconciler

        reconciler = PolymarketVenueAccountReconciler()
    else:
        reconciler = venue_reconciler
    try:
        snapshot = await reconciler.snapshot(credentials)
        portfolio = await _portfolio_from_persisted_positions(
            pool,
            total_budget_usdc=settings.risk.max_total_exposure,
        )
        report: ReconciliationReport = await reconciler.compare(portfolio, snapshot)
    except Exception as exc:  # noqa: BLE001
        return LivePreflightCheck(
            "venue_reconciliation",
            False,
            redact_live_error(str(exc), settings),
        )

    if not report.ok:
        return LivePreflightCheck(
            "venue_reconciliation",
            False,
            redact_live_error("; ".join(report.mismatches), settings)
            or "unknown mismatch",
        )
    return LivePreflightCheck(
        "venue_reconciliation",
        True,
        "venue account snapshot reconciles with persisted positions and open orders",
    )


async def _portfolio_from_persisted_positions(
    pool: asyncpg.Pool,
    *,
    total_budget_usdc: float,
) -> Portfolio:
    positions = await _read_persisted_positions(pool)
    locked_usdc = sum(position.locked_usdc for position in positions)
    free_usdc = max(0.0, total_budget_usdc - locked_usdc)
    return Portfolio(
        total_usdc=total_budget_usdc,
        free_usdc=free_usdc,
        locked_usdc=locked_usdc,
        open_positions=list(positions),
    )


async def _read_persisted_positions(pool: asyncpg.Pool) -> tuple[Position, ...]:
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT
                fills.market_id,
                fill_payloads.payload->>'token_id' AS token_id,
                COALESCE(fill_payloads.payload->>'venue', 'polymarket') AS venue,
                COALESCE(fill_payloads.payload->>'side', 'BUY') AS side,
                fill_payloads.payload->>'risk_group_id' AS risk_group_id,
                SUM(fills.fill_quantity)::double precision AS shares_held,
                CASE
                    WHEN SUM(fills.fill_quantity) = 0 THEN 0.0
                    ELSE (
                        SUM(fills.fill_notional_usdc) / SUM(fills.fill_quantity)
                    )::double precision
                END AS avg_entry_price,
                SUM(fills.fill_notional_usdc)::double precision AS locked_usdc
            FROM fills
            INNER JOIN fill_payloads
                ON fill_payloads.fill_id = fills.fill_id
            GROUP BY
                fills.market_id,
                fill_payloads.payload->>'token_id',
                fill_payloads.payload->>'venue',
                fill_payloads.payload->>'side',
                fill_payloads.payload->>'risk_group_id'
            HAVING SUM(fills.fill_quantity) > 0
            """
        )
    return tuple(_position_from_row(row) for row in rows)


def _position_from_row(row: Mapping[str, object]) -> Position:
    return Position(
        market_id=str(row["market_id"]),
        token_id=_optional_str(row["token_id"]),
        venue=cast(Venue, str(row["venue"])),
        side=str(row["side"]),
        shares_held=_required_float(row["shares_held"], field_name="shares_held"),
        avg_entry_price=_required_float(
            row["avg_entry_price"],
            field_name="avg_entry_price",
        ),
        unrealized_pnl=0.0,
        locked_usdc=_required_float(row["locked_usdc"], field_name="locked_usdc"),
        risk_group_id=_optional_str(row.get("risk_group_id")),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _required_float(value: object, *, field_name: str) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    msg = f"position field {field_name} is not numeric: {value!r}"
    raise TypeError(msg)
