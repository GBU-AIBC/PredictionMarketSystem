from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import NoReturn, cast
from uuid import uuid4

import asyncpg
import yaml

from pms.actuator.adapters.polymarket import PolymarketVenueAccountReconciler
from pms.config import (
    PMSSettings,
    load_settings,
    safe_load_yaml_no_duplicate_keys,
    validate_live_mode_ready,
    yaml_load_error_message,
)
from pms.core.enums import Side
from pms.core.models import Portfolio, Position, ReconciliationReport
from pms.live_preflight import (
    LivePreflightCheck,
    LivePreflightResult,
    live_preflight_readiness_reports_fingerprint,
    live_preflight_result_is_final_go_no_go_valid,
    live_preflight_settings_fingerprint,
    redact_live_error,
    require_live_preflight_artifact,
    require_preflight_artifact_concrete_path,
    require_preflight_artifact_outside_working_tree,
    require_preflight_artifact_parent_owner_writable,
    require_preflight_artifact_regular_file_path,
    run_live_preflight,
)
from pms.live_preflight_artifact import loads_json_rejecting_duplicate_keys
from pms.redaction import redact_database_error
from pms.storage.fill_store import FillStore
from pms.storage.live_emergency_audit import (
    EmergencyRestartMode,
    LiveEmergencyAuditWriter,
)
from pms.storage.live_reconciliation import (
    LiveOrderReconciliationRecord,
    LiveOrderReconciliationStore,
    SubmissionUnknownReconciliationStore,
    SubmissionUnknownResolutionStatus,
    normalize_submission_unknown_decision_id,
    normalize_submission_unknown_reconciled_by,
    normalize_submission_unknown_venue_order_id,
)
from pms.storage.schema_check import ensure_schema_current


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pms-live")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight",
        help="Run read-only LIVE readiness checks before starting real trading.",
    )
    preflight.add_argument("--config")
    preflight.add_argument("--database-url")
    preflight.add_argument(
        "--skip-venue",
        action="store_true",
        help="Skip the read-only venue account snapshot; not valid for final go/no-go.",
    )
    preflight.add_argument(
        "--skip-credentials",
        action="store_true",
        help=(
            "Use diagnostic placeholder secrets so non-credential checks can run; "
            "not valid for final go/no-go."
        ),
    )
    preflight.add_argument("--json", action="store_true")
    preflight.add_argument(
        "--output",
        help=(
            "Write a persisted JSON preflight artifact with provenance. "
            "The artifact records whether it is valid for final go/no-go."
        ),
    )

    reconcile = subparsers.add_parser(
        "reconcile-submission-unknown",
        help="Resolve a submission_unknown live incident after venue reconciliation.",
    )
    reconcile.add_argument("--config")
    reconcile.add_argument("--decision-id", required=True)
    reconcile.add_argument("--venue-order-id")
    reconcile.add_argument(
        "--status",
        required=True,
        choices=("filled", "not_found", "open"),
    )
    reconcile.add_argument("--reconciled-by", required=True)
    reconcile.add_argument("--note")
    reconcile.add_argument("--fill-price", type=float)
    reconcile.add_argument("--fill-notional-usdc", type=float)
    reconcile.add_argument("--fill-quantity", type=float)
    reconcile.add_argument("--filled-at")
    reconcile.add_argument("--fees", type=float)
    reconcile.add_argument("--fee-bps", type=int)
    reconcile.add_argument("--database-url")
    fee_reconcile = subparsers.add_parser(
        "reconcile-live-fee",
        help="Record audited venue fee evidence for an unresolved LIVE fill.",
    )
    fee_reconcile.add_argument("--config")
    fee_reconcile.add_argument("--fill-id", required=True)
    fee_reconcile.add_argument("--fees", required=True, type=float)
    fee_reconcile.add_argument("--fee-bps", type=int)
    fee_reconcile.add_argument("--slippage-bps", type=float)
    fee_reconcile.add_argument("--reconciled-by", required=True)
    fee_reconcile.add_argument("--evidence", required=True)
    fee_reconcile.add_argument("--database-url")
    reconcile_live_order = subparsers.add_parser(
        "reconcile-live-order",
        help=(
            "Write a post-live order reconciliation artifact after a real "
            "Polymarket fill."
        ),
    )
    reconcile_live_order.add_argument("--config")
    reconcile_live_order.add_argument("--decision-id", required=True)
    reconcile_live_order.add_argument("--reconciled-by", required=True)
    reconcile_live_order.add_argument("--output", required=True)
    reconcile_live_order.add_argument("--database-url")
    emergency_stop = subparsers.add_parser(
        "record-emergency-stop",
        help="Append a manual emergency-stop completion record to the LIVE audit log.",
    )
    emergency_stop.add_argument("--config")
    emergency_stop.add_argument("--stopped-by", required=True)
    emergency_stop.add_argument("--reason", required=True)
    emergency_stop.add_argument("--runner-stopped", action="store_true", required=True)
    emergency_stop.add_argument(
        "--credentials-rotated",
        action="store_true",
        required=True,
    )
    emergency_stop.add_argument(
        "--runtime-secrets-removed",
        action="store_true",
        required=True,
    )
    emergency_stop.add_argument(
        "--venue-open-orders-reviewed",
        action="store_true",
        required=True,
    )
    emergency_stop.add_argument(
        "--database-reconciled",
        action="store_true",
        required=True,
    )
    emergency_stop.add_argument(
        "--restart-mode",
        required=True,
        choices=("paper", "backtest"),
        help="Mode used until exposure and open orders are reconciled.",
    )
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    if args.command == "preflight":
        try:
            settings = _load_cli_settings(
                cast(str | None, args.config),
                skip_local_secret_file=cast(bool, args.skip_credentials),
            )
        except Exception as exc:  # noqa: BLE001
            result = LivePreflightResult(
                (
                    LivePreflightCheck(
                        "config_load",
                        False,
                        redact_database_error(str(exc)),
                    ),
                )
            )
            _print_preflight_result(result, as_json=cast(bool, args.json))
            return 1
        if args.database_url:
            settings = settings.model_copy(
                update={
                    "database": settings.database.model_copy(
                        update={"dsn": cast(str, args.database_url)}
                    )
                }
            )
        result = await run_live_preflight(
            settings,
            skip_venue=cast(bool, args.skip_venue),
            skip_credentials=cast(bool, args.skip_credentials),
        )
        artifact_final_go_no_go_valid: bool | None = None
        if args.output:
            try:
                artifact_final_go_no_go_valid = _write_preflight_artifact(
                    result,
                    settings=settings,
                    output_path=Path(cast(str, args.output)),
                    config_path=_effective_cli_config_path(
                        cast(str | None, args.config)
                    ),
                    skip_venue=cast(bool, args.skip_venue),
                    skip_credentials=cast(bool, args.skip_credentials),
                    database_url_override_used=bool(args.database_url),
                )
            except Exception as exc:  # noqa: BLE001
                artifact_final_go_no_go_valid = False
                result = _append_preflight_artifact_write_failure(result, exc)
        _print_preflight_result(result, as_json=cast(bool, args.json))
        if artifact_final_go_no_go_valid is False:
            return 1
        if args.skip_venue or args.skip_credentials:
            return 1
        return 0 if result.ok else 1

    if args.command == "reconcile-submission-unknown":
        status = cast(SubmissionUnknownResolutionStatus, args.status)
        try:
            decision_id = normalize_submission_unknown_decision_id(
                cast(str, args.decision_id)
            )
            venue_order_id = normalize_submission_unknown_venue_order_id(
                status=status,
                venue_order_id=cast(str | None, args.venue_order_id),
            )
            reconciled_by = normalize_submission_unknown_reconciled_by(
                cast(str, args.reconciled_by)
            )
            filled_at = (
                datetime.fromisoformat(
                    cast(str, args.filled_at).replace("Z", "+00:00")
                )
                if args.filled_at
                else None
            )
        except ValueError as exc:
            _print_reconcile_failure(args, exc)
            return 1
        try:
            settings = _load_cli_settings(cast(str | None, args.config))
        except Exception as exc:  # noqa: BLE001
            _print_reconcile_failure(args, exc)
            return 1
        database_url = args.database_url or settings.database.dsn
        try:
            pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=1)
        except Exception as exc:  # noqa: BLE001
            _print_reconcile_failure(args, exc, settings=settings)
            return 1
        try:
            try:
                await ensure_schema_current(pool)
                reconcile_kwargs: dict[str, object] = {
                    "decision_id": decision_id,
                    "venue_order_id": venue_order_id,
                    "status": status,
                    "reconciled_by": reconciled_by,
                    "note": cast(str | None, args.note),
                }
                if any(
                    value is not None
                    for value in (
                        args.fill_price,
                        args.fill_notional_usdc,
                        args.fill_quantity,
                        args.filled_at,
                        args.fees,
                        args.fee_bps,
                    )
                ):
                    reconcile_kwargs.update(
                        {
                            "fill_price": cast(float | None, args.fill_price),
                            "fill_notional_usdc": cast(
                                float | None,
                                args.fill_notional_usdc,
                            ),
                            "fill_quantity": cast(float | None, args.fill_quantity),
                            "filled_at": filled_at,
                            "fees": cast(float | None, args.fees),
                            "fee_bps": cast(int | None, args.fee_bps),
                        }
                    )
                updated = await SubmissionUnknownReconciliationStore(
                    pool
                ).reconcile_submission_unknown(**reconcile_kwargs)  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                _print_reconcile_failure(args, exc, settings=settings)
                return 1
        finally:
            await pool.close()
        print(
            json.dumps(
                {
                    "updated": updated,
                    "decision_id": decision_id,
                    "status": args.status,
                    **(
                        {}
                        if updated
                        else {
                            "error": _submission_unknown_no_update_error(decision_id),
                        }
                    ),
                },
                sort_keys=True,
            )
        )
        return 0 if updated else 1
    if args.command == "reconcile-live-fee":
        try:
            fill_id = _normalize_fee_reconcile_fill_id(cast(str, args.fill_id))
            reconciled_by = normalize_submission_unknown_reconciled_by(
                cast(str, args.reconciled_by)
            )
            evidence = _normalize_fee_reconcile_evidence(cast(str, args.evidence))
            fees = float(args.fees)
            if not math.isfinite(fees) or fees < 0.0:
                raise ValueError("fees must be finite and non-negative")
            fee_bps = cast(int | None, args.fee_bps)
            if fee_bps is not None and (isinstance(fee_bps, bool) or fee_bps < 0):
                raise ValueError("fee_bps must be non-negative")
            slippage_bps = cast(float | None, args.slippage_bps)
            if slippage_bps is not None and not math.isfinite(slippage_bps):
                raise ValueError("slippage_bps must be finite")
        except (TypeError, ValueError) as exc:
            _print_fee_reconcile_failure(args, exc)
            return 1
        try:
            settings = _load_cli_settings(cast(str | None, args.config))
        except Exception as exc:  # noqa: BLE001
            _print_fee_reconcile_failure(args, exc)
            return 1
        database_url = args.database_url or settings.database.dsn
        try:
            pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=1)
        except Exception as exc:  # noqa: BLE001
            _print_fee_reconcile_failure(args, exc, settings=settings)
            return 1
        updated = False
        try:
            try:
                await ensure_schema_current(pool)
                updated = await FillStore(pool).reconcile_fee(
                    fill_id,
                    fees=fees,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                )
                if updated:
                    await LiveEmergencyAuditWriter(
                        Path(settings.live_emergency_audit_path)
                    ).append_fee_reconcile(
                        fill_id=fill_id,
                        fees=fees,
                        fee_bps=fee_bps,
                        slippage_bps=slippage_bps,
                        reconciled_by=reconciled_by,
                        evidence=evidence,
                    )
            except Exception as exc:  # noqa: BLE001
                _print_fee_reconcile_failure(args, exc, settings=settings)
                return 1
        finally:
            await pool.close()
        if not updated:
            _print_fee_reconcile_failure(
                args,
                RuntimeError(
                    "fill was not found or already has reconciled fee telemetry"
                ),
                settings=settings,
            )
            return 1
        print(
            json.dumps(
                {
                    "updated": True,
                    "fill_id": fill_id,
                    "fees": fees,
                    "fee_bps": fee_bps,
                    "reconciled_by": reconciled_by,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "reconcile-live-order":
        try:
            decision_id = normalize_submission_unknown_decision_id(
                cast(str, args.decision_id)
            )
            reconciled_by = normalize_submission_unknown_reconciled_by(
                cast(str, args.reconciled_by)
            )
        except ValueError as exc:
            _print_live_order_reconcile_failure(args, exc)
            return 1
        try:
            settings = _load_cli_settings(cast(str | None, args.config))
        except Exception as exc:  # noqa: BLE001
            _print_live_order_reconcile_failure(args, exc)
            return 1
        database_url = args.database_url or settings.database.dsn
        try:
            credentials = validate_live_mode_ready(settings)
        except Exception as exc:  # noqa: BLE001
            _print_live_order_reconcile_failure(args, exc, settings=settings)
            return 1
        database_url_override_used = bool(args.database_url)
        credentialed_preflight_artifact: dict[str, object] | None = None
        if not database_url_override_used:
            try:
                credentialed_preflight_artifact = (
                    _credentialed_preflight_artifact_reference(settings)
                )
            except Exception as exc:  # noqa: BLE001
                _print_live_order_reconcile_failure(args, exc, settings=settings)
                return 1
        try:
            pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=1)
        except Exception as exc:  # noqa: BLE001
            _print_live_order_reconcile_failure(args, exc, settings=settings)
            return 1
        try:
            try:
                await ensure_schema_current(pool)
                record = await LiveOrderReconciliationStore(
                    pool
                ).load_live_order_record(decision_id=decision_id)
                if record is None:
                    msg = (
                        "live order reconciliation evidence not found: "
                        f"{decision_id}"
                    )
                    raise RuntimeError(msg)
                if not database_url_override_used:
                    credentialed_preflight_artifact = (
                        _credentialed_preflight_artifact_reference(
                            settings,
                            submitted_at=record.submitted_at,
                        )
                    )
                portfolio = await _portfolio_from_fill_store(
                    pool,
                    total_budget_usdc=settings.risk.max_total_exposure,
                )
                reconciler = PolymarketVenueAccountReconciler()
                venue_snapshot = await reconciler.snapshot(credentials)
                report = await reconciler.compare(portfolio, venue_snapshot)
                if not report.ok:
                    details = "; ".join(report.mismatches)
                    msg = f"live order venue reconciliation mismatch: {details}"
                    raise RuntimeError(msg)
                final_post_live_valid = not database_url_override_used
                output_path = Path(cast(str, args.output))
                _write_live_order_reconciliation_artifact(
                    record,
                    report=report,
                    settings=settings,
                    output_path=output_path,
                    config_path=_effective_cli_config_path(
                        cast(str | None, args.config)
                    ),
                    reconciled_by=reconciled_by,
                    database_url_override_used=database_url_override_used,
                    portfolio=portfolio,
                    final_post_live_valid=final_post_live_valid,
                    credentialed_preflight_artifact=credentialed_preflight_artifact,
                )
            except Exception as exc:  # noqa: BLE001
                _print_live_order_reconcile_failure(args, exc, settings=settings)
                return 1
        finally:
            await pool.close()
        print(
            json.dumps(
                {
                    "reconciled": True,
                    "decision_id": decision_id,
                    "artifact_mode": (
                        "post_live_order_reconciliation"
                        if final_post_live_valid
                        else "incomplete_post_live_order_reconciliation"
                    ),
                    "final_post_live_valid": final_post_live_valid,
                    "output_path": _artifact_output_path(output_path),
                },
                sort_keys=True,
            )
        )
        return 0 if final_post_live_valid else 1
    if args.command == "record-emergency-stop":
        try:
            audit_path = _load_emergency_stop_audit_path(cast(str | None, args.config))
        except Exception as exc:  # noqa: BLE001
            _print_emergency_stop_failure(args, exc)
            return 1
        try:
            writer = LiveEmergencyAuditWriter(audit_path)
            await writer.append_manual_stop(
                stopped_by=cast(str, args.stopped_by),
                reason=cast(str, args.reason),
                runner_stopped=cast(bool, args.runner_stopped),
                credentials_rotated=cast(bool, args.credentials_rotated),
                runtime_secrets_removed=cast(bool, args.runtime_secrets_removed),
                venue_open_orders_reviewed=cast(
                    bool,
                    args.venue_open_orders_reviewed,
                ),
                database_reconciled=cast(bool, args.database_reconciled),
                restart_mode=cast(EmergencyRestartMode, args.restart_mode),
            )
        except Exception as exc:  # noqa: BLE001
            _print_emergency_stop_failure(args, exc)
            return 1
        print(
            json.dumps(
                {
                    "recorded": True,
                    "event": "manual_emergency_stop",
                    "path": _artifact_output_path(audit_path),
                },
                sort_keys=True,
            )
        )
        return 0
    _unreachable(args.command)


def _load_cli_settings(
    config_path: str | None,
    *,
    skip_local_secret_file: bool = False,
) -> PMSSettings:
    with _diagnostic_env_for_skipped_credentials(skip_local_secret_file):
        if config_path is None:
            return load_settings(load_local_secret_file=not skip_local_secret_file)
        return PMSSettings.load(
            config_path,
            load_local_secret_file=not skip_local_secret_file,
        )


@contextmanager
def _diagnostic_env_for_skipped_credentials(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    defaults = {
        "PMS_LLM__API_KEY": "diagnostic-llm-api-key",
        "PMS_LLM__PROVIDER": "anthropic",
    }
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in defaults}
    try:
        for key, default_value in defaults.items():
            current = os.environ.get(key)
            if current is None or current.strip() == "":
                os.environ[key] = default_value
        yield
    finally:
        for key, previous_value in previous.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


def _effective_cli_config_path(config_path: str | None) -> str:
    if config_path is not None:
        return config_path
    return os.environ.get("PMS_CONFIG_PATH") or "config.yaml"


def _load_emergency_stop_audit_path(config_path: str | None) -> Path:
    configured_path = _effective_cli_config_path(config_path)
    config_data: dict[str, object] = {}
    path = Path(configured_path)
    if path.exists():
        try:
            loaded = safe_load_yaml_no_duplicate_keys(
                _read_bytes_no_follow(path).decode("utf-8")
            )
        except OSError:
            msg = f"Config file cannot be read safely: {path}"
            raise ValueError(msg) from None
        except yaml.YAMLError as exc:
            msg = yaml_load_error_message("Config file is not valid YAML", path, exc)
            raise ValueError(msg) from None
        if loaded is not None:
            if not isinstance(loaded, dict):
                msg = f"Expected mapping in config file {path}"
                raise ValueError(msg)
            if not all(isinstance(key, str) for key in loaded):
                msg = f"Expected string keys in config file {path}"
                raise ValueError(msg)
            config_data = cast(dict[str, object], loaded)
    raw_path = config_data.get("live_emergency_audit_path")
    if raw_path is None:
        raw_path = os.environ.get("PMS_LIVE_EMERGENCY_AUDIT_PATH")
    if not isinstance(raw_path, str) or raw_path.strip() == "":
        msg = (
            "live_emergency_audit_path is required for record-emergency-stop; "
            "pass --config or set PMS_LIVE_EMERGENCY_AUDIT_PATH"
        )
        raise ValueError(msg)
    if _looks_like_live_artifact_placeholder(raw_path):
        msg = "live_emergency_audit_path must not contain a placeholder"
        raise ValueError(msg)
    audit_path = Path(raw_path).expanduser()
    _require_emergency_stop_audit_path_outside_working_tree(audit_path)
    _require_emergency_stop_audit_path_distinct_from_config_file(audit_path, path)
    _require_emergency_stop_audit_path_distinct_from_configured_live_inputs(
        audit_path,
        config_data,
    )
    return audit_path


def _require_emergency_stop_audit_path_distinct_from_config_file(
    audit_path: Path,
    config_path: Path,
) -> None:
    if not _path_identities_overlap(
        _path_identities(audit_path),
        _path_identities(config_path),
    ):
        return
    msg = f"live_emergency_audit_path must be distinct from config file: {audit_path}"
    raise ValueError(msg)


def _require_emergency_stop_audit_path_distinct_from_configured_live_inputs(
    path: Path,
    config_data: Mapping[str, object],
) -> None:
    path_identities = _path_identities(path)
    for label, raw_path in _configured_live_input_paths_for_emergency_stop(
        config_data
    ):
        if raw_path is None or raw_path.strip() == "":
            continue
        if not _path_identities_overlap(
            path_identities,
            _path_identities(Path(raw_path)),
        ):
            continue
        msg = f"live_emergency_audit_path must be distinct from {label}: {path}"
        raise ValueError(msg)


def _configured_live_input_paths_for_emergency_stop(
    config_data: Mapping[str, object],
) -> tuple[tuple[str, str | None], ...]:
    approval_path = _configured_nested_str(
        config_data,
        "polymarket",
        "first_live_order_approval_path",
        env_name="PMS_POLYMARKET__FIRST_LIVE_ORDER_APPROVAL_PATH",
    )
    paths: list[tuple[str, str | None]] = [
        (
            "LIVE credentialed preflight artifact",
            _configured_str(
                config_data,
                "live_preflight_artifact_path",
                env_name="PMS_LIVE_PREFLIGHT_ARTIFACT_PATH",
            ),
        ),
        (
            "LIVE first-order audit path",
            _configured_str(
                config_data,
                "live_first_order_audit_path",
                env_name="PMS_LIVE_FIRST_ORDER_AUDIT_PATH",
            ),
        ),
        (
            "LIVE operator approval path",
            approval_path,
        ),
        (
            "LIVE local secret file",
            _configured_str(
                config_data,
                "local_secret_file",
                env_name="PMS_LOCAL_SECRET_FILE",
            ),
        ),
        (
            "LIVE paper soak GO report",
            _configured_str(
                config_data,
                "live_paper_soak_report_path",
                env_name="PMS_LIVE_PAPER_SOAK_REPORT_PATH",
            ),
        ),
        (
            "LIVE operator rehearsal report",
            _configured_str(
                config_data,
                "live_operator_rehearsal_report_path",
                env_name="PMS_LIVE_OPERATOR_REHEARSAL_REPORT_PATH",
            ),
        ),
        (
            "LIVE execution-model artifact",
            _configured_str(
                config_data,
                "live_execution_model_path",
                env_name="PMS_LIVE_EXECUTION_MODEL_PATH",
            ),
        ),
        (
            "LIVE paper-vs-backtest execution diff artifact",
            _configured_str(
                config_data,
                "live_paper_backtest_diff_path",
                env_name="PMS_LIVE_PAPER_BACKTEST_DIFF_PATH",
            ),
        ),
        (
            "LIVE category-prior artifact",
            _configured_nested_str(
                config_data,
                "controller",
                "category_prior_observations_path",
                env_name="PMS_CONTROLLER__CATEGORY_PRIOR_OBSERVATIONS_PATH",
            ),
        ),
        (
            "LIVE FLB calibration artifact",
            _configured_nested_str(
                config_data,
                "strategies",
                "flb_calibration_path",
                env_name="PMS_STRATEGIES__FLB_CALIBRATION_PATH",
            ),
        ),
        (
            "LIVE discord alert directory",
            _configured_nested_str(
                config_data,
                "discord",
                "alert_dir",
                env_name="PMS_DISCORD__ALERT_DIR",
            ),
        ),
    ]
    if approval_path is not None and approval_path.strip() != "":
        paths.append(
            ("LIVE operator approval sidecar path", f"{approval_path}.meta.json")
        )
    return tuple(paths)


def _configured_str(
    config_data: Mapping[str, object],
    key: str,
    *,
    env_name: str | None = None,
) -> str | None:
    value = config_data.get(key)
    if isinstance(value, str):
        return value
    if env_name is None:
        return None
    env_value = os.environ.get(env_name)
    if env_value is None:
        return None
    return env_value


def _configured_nested_str(
    config_data: Mapping[str, object],
    section: str,
    key: str,
    *,
    env_name: str | None = None,
) -> str | None:
    section_data = config_data.get(section)
    if isinstance(section_data, Mapping):
        value = section_data.get(key)
        if isinstance(value, str):
            return value
    if env_name is None:
        return None
    env_value = os.environ.get(env_name)
    if env_value is None:
        return None
    return env_value


def _require_emergency_stop_audit_path_outside_working_tree(path: Path) -> None:
    path_identities = _path_identities(path)
    working_tree = _working_tree_root(Path.cwd().resolve(strict=False))
    working_trees = [working_tree]
    for candidate in path_identities:
        containing_working_tree = _containing_working_tree_root(candidate)
        if containing_working_tree is not None:
            working_trees.append(containing_working_tree)

    for working_tree_candidate in dict.fromkeys(working_trees):
        if working_tree_candidate.parent == working_tree_candidate:
            continue
        for candidate in path_identities:
            try:
                candidate.relative_to(working_tree_candidate)
            except ValueError:
                continue
            msg = (
                "live_emergency_audit_path must live outside the working tree: "
                f"{candidate}"
            )
            raise ValueError(msg)


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


def _normalize_fee_reconcile_fill_id(fill_id: str) -> str:
    normalized = fill_id.strip()
    if normalized == "":
        raise ValueError("fill_id is required")
    if any(marker in normalized.lower() for marker in ("placeholder", "todo", "fill_in", "<", ">")):
        raise ValueError("fill_id must not contain a placeholder")
    return normalized


def _normalize_fee_reconcile_evidence(evidence: str) -> str:
    normalized = evidence.strip()
    if normalized == "":
        raise ValueError("evidence is required")
    if any(character in normalized for character in ("\n", "\r")):
        raise ValueError("evidence must not contain newlines")
    return normalized


def _print_fee_reconcile_failure(
    args: argparse.Namespace,
    exc: Exception,
    *,
    settings: PMSSettings | None = None,
) -> None:
    error = (
        redact_database_error(str(exc))
        if settings is None
        else redact_live_error(str(exc), settings)
    )
    print(
        json.dumps(
            {
                "updated": False,
                "fill_id": cast(str, args.fill_id),
                "error": error,
            },
            sort_keys=True,
        )
    )


def _print_reconcile_failure(
    args: argparse.Namespace,
    exc: Exception,
    *,
    settings: PMSSettings | None = None,
) -> None:
    error = (
        redact_database_error(str(exc))
        if settings is None
        else redact_live_error(str(exc), settings)
    )
    print(
        json.dumps(
            {
                "updated": False,
                "decision_id": cast(str, args.decision_id),
                "status": cast(str, args.status),
                "error": error,
            },
            sort_keys=True,
        )
    )


def _submission_unknown_no_update_error(decision_id: str) -> str:
    return (
        "submission_unknown incident was not updated; verify the decision is still "
        "submission_unknown and has not already been reconciled: "
        f"{decision_id}"
    )


def _print_live_order_reconcile_failure(
    args: argparse.Namespace,
    exc: Exception,
    *,
    settings: PMSSettings | None = None,
) -> None:
    error = (
        redact_database_error(str(exc))
        if settings is None
        else redact_live_error(str(exc), settings)
    )
    print(
        json.dumps(
            {
                "reconciled": False,
                "decision_id": cast(str, args.decision_id),
                "error": error,
            },
            sort_keys=True,
        )
    )


def _print_emergency_stop_failure(
    args: argparse.Namespace,
    exc: Exception,
    *,
    settings: PMSSettings | None = None,
) -> None:
    del args
    error = (
        redact_database_error(str(exc))
        if settings is None
        else redact_live_error(str(exc), settings)
    )
    print(
        json.dumps(
            {
                "recorded": False,
                "event": "manual_emergency_stop",
                "error": error,
            },
            sort_keys=True,
        )
    )


def _print_preflight_result(
    result: LivePreflightResult,
    *,
    as_json: bool,
) -> None:
    if as_json:
        print(json.dumps(result.as_dict(), sort_keys=True))
        return
    status = "PASS" if result.ok else "FAIL"
    print(f"live preflight: {status}")
    for check in result.checks:
        check_status = "PASS" if check.ok else "FAIL"
        print(f"- {check.name}: {check_status} — {check.detail}")


def _append_preflight_artifact_write_failure(
    result: LivePreflightResult,
    exc: Exception,
) -> LivePreflightResult:
    return LivePreflightResult(
        (
            *result.checks,
            LivePreflightCheck(
                "artifact_write",
                False,
                redact_database_error(str(exc)),
            ),
        ),
        active_strategies_fingerprint=result.active_strategies_fingerprint,
    )


def _write_preflight_artifact(
    result: LivePreflightResult,
    *,
    settings: PMSSettings,
    output_path: Path,
    config_path: str | None,
    skip_venue: bool,
    skip_credentials: bool = False,
    database_url_override_used: bool,
) -> bool:
    final_go_no_go_valid = live_preflight_result_is_final_go_no_go_valid(
        result,
        skip_venue=skip_venue,
        skip_credentials=skip_credentials,
        database_url_override_used=database_url_override_used,
    )
    require_preflight_artifact_concrete_path(output_path)
    require_preflight_artifact_outside_working_tree(output_path)
    _require_output_distinct_from_config_file(output_path, config_path)
    _require_preflight_output_distinct_from_live_inputs(
        settings,
        output_path,
        include_preflight_artifact=not final_go_no_go_valid,
    )
    if final_go_no_go_valid:
        _require_final_preflight_output_matches_configured_path(settings, output_path)
        require_preflight_artifact_parent_owner_writable(output_path)
        require_preflight_artifact_regular_file_path(output_path, must_exist=False)
    artifact = {
        "generated_by": "pms-live preflight",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "artifact_mode": (
            "credentialed_preflight"
            if final_go_no_go_valid
            else "incomplete_preflight"
        ),
        "final_go_no_go_valid": final_go_no_go_valid,
        "skip_venue": skip_venue,
        "skip_credentials": skip_credentials,
        "config_path": config_path,
        "database_url_override_used": database_url_override_used,
        "settings_fingerprint": live_preflight_settings_fingerprint(settings),
        "readiness_reports_fingerprint": (
            live_preflight_readiness_reports_fingerprint(settings)
            if final_go_no_go_valid
            else None
        ),
        "active_strategies_fingerprint": result.active_strategies_fingerprint,
        "output_path": _artifact_output_path(output_path),
        "result": result.as_dict(),
    }
    _require_strict_json_artifact(artifact, path="$")
    _prepare_preflight_artifact_parent(output_path)
    _write_text_no_follow(
        output_path,
        json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )
    return final_go_no_go_valid


def _require_strict_json_artifact(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        msg = f"{path} must be finite for live artifact JSON"
        raise ValueError(msg)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"{path} contains a non-string JSON object key"
                raise TypeError(msg)
            child_path = key if path == "$" else f"{path}.{key}"
            _require_strict_json_artifact(item, path=child_path)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_strict_json_artifact(item, path=f"{path}[{index}]")
        return
    msg = f"{path} is not JSON-serializable for live artifact JSON"
    raise TypeError(msg)


def _require_output_distinct_from_config_file(
    output_path: Path,
    config_path: str | None,
) -> None:
    if config_path is None or config_path.strip() == "":
        return
    path = Path(config_path)
    if not _path_identities_overlap(
        _path_identities(output_path),
        _path_identities(path),
    ):
        return
    msg = f"output path must be distinct from config file: {output_path}"
    raise ValueError(msg)


def _require_final_preflight_output_matches_configured_path(
    settings: PMSSettings,
    output_path: Path,
) -> None:
    configured = settings.live_preflight_artifact_path
    if configured is None or configured.strip() == "":
        msg = (
            "final preflight output requires configured "
            "live_preflight_artifact_path"
        )
        raise ValueError(msg)
    configured_path = Path(configured).expanduser().resolve(strict=False)
    observed_path = output_path.expanduser().resolve(strict=False)
    if observed_path == configured_path:
        return
    msg = (
        "final preflight output path must match configured "
        f"live_preflight_artifact_path: {observed_path} != {configured_path}"
    )
    raise ValueError(msg)


def _require_preflight_output_distinct_from_live_inputs(
    settings: PMSSettings,
    output_path: Path,
    *,
    include_preflight_artifact: bool,
) -> None:
    _require_output_distinct_from_live_inputs(
        settings,
        output_path,
        output_label="preflight output path",
        include_preflight_artifact=include_preflight_artifact,
    )


def _require_live_order_output_distinct_from_live_inputs(
    settings: PMSSettings,
    output_path: Path,
) -> None:
    _require_output_distinct_from_live_inputs(
        settings,
        output_path,
        output_label="live order reconciliation output path",
        include_preflight_artifact=True,
    )


def _require_output_distinct_from_live_inputs(
    settings: PMSSettings,
    output_path: Path,
    *,
    output_label: str,
    include_preflight_artifact: bool,
) -> None:
    output_identities = _path_identities(output_path)

    for label, raw_path in _protected_live_input_paths(
        settings,
        include_preflight_artifact=include_preflight_artifact,
    ):
        if raw_path is None or raw_path.strip() == "":
            continue
        if not _path_identities_overlap(
            output_identities,
            _path_identities(Path(raw_path)),
        ):
            continue
        msg = f"{output_label} must be distinct from {label}: {output_path}"
        raise ValueError(msg)


def _protected_live_input_paths(
    settings: PMSSettings,
    *,
    include_preflight_artifact: bool,
) -> tuple[tuple[str, str | None], ...]:
    approval_path = settings.polymarket.first_live_order_approval_path
    candidates: list[tuple[str, str | None]] = [
        ("LIVE paper soak GO report", settings.live_paper_soak_report_path),
        (
            "LIVE operator rehearsal report",
            settings.live_operator_rehearsal_report_path,
        ),
        ("LIVE execution-model artifact", settings.live_execution_model_path),
        (
            "LIVE paper-vs-backtest execution diff artifact",
            settings.live_paper_backtest_diff_path,
        ),
        (
            "LIVE category-prior artifact",
            settings.controller.category_prior_observations_path,
        ),
        ("LIVE FLB calibration artifact", settings.strategies.flb_calibration_path),
        ("LIVE local secret file", settings.local_secret_file),
        ("LIVE operator approval path", approval_path),
        ("LIVE first-order audit path", settings.live_first_order_audit_path),
        ("LIVE emergency audit path", settings.live_emergency_audit_path),
        ("LIVE discord alert directory", settings.discord.alert_dir),
    ]
    if include_preflight_artifact:
        candidates.append(
            (
                "LIVE credentialed preflight artifact",
                settings.live_preflight_artifact_path,
            )
        )
    if approval_path is not None and approval_path.strip() != "":
        candidates.append(
            ("LIVE operator approval sidecar path", f"{approval_path}.meta.json")
        )
    return tuple(candidates)


def _path_identities(path: Path) -> frozenset[Path]:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return frozenset(
        (
            Path(os.path.abspath(expanded)),
            expanded.resolve(strict=False),
        )
    )


def _path_identities_overlap(left: frozenset[Path], right: frozenset[Path]) -> bool:
    return any(
        _paths_overlap(left_path, right_path)
        for left_path in left
        for right_path in right
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        right.relative_to(left)
    except ValueError:
        pass
    else:
        return True
    try:
        left.relative_to(right)
    except ValueError:
        return False
    return True


async def _portfolio_from_fill_store(
    pool: asyncpg.Pool,
    *,
    total_budget_usdc: float,
) -> Portfolio:
    positions = await FillStore(pool).read_positions()
    open_positions = [position for position in positions if position.shares_held > 0.0]
    locked_usdc = sum(position.locked_usdc for position in open_positions)
    return Portfolio(
        total_usdc=total_budget_usdc,
        free_usdc=max(0.0, total_budget_usdc - locked_usdc),
        locked_usdc=locked_usdc,
        open_positions=open_positions,
    )


def _write_live_order_reconciliation_artifact(
    record: LiveOrderReconciliationRecord,
    *,
    report: ReconciliationReport,
    settings: PMSSettings,
    output_path: Path,
    config_path: str | None,
    reconciled_by: str,
    database_url_override_used: bool,
    portfolio: Portfolio,
    final_post_live_valid: bool,
    credentialed_preflight_artifact: Mapping[str, object] | None,
) -> None:
    reconciled_by = normalize_submission_unknown_reconciled_by(reconciled_by)
    _require_preflight_reference_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        settings=settings,
        credentialed_preflight_artifact=credentialed_preflight_artifact,
    )
    _require_core_identity_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_order_intent_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_venue_reconciliation_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        report=report,
    )
    _require_canonical_database_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        database_url_override_used=database_url_override_used,
    )
    _require_closed_order_evidence_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_polymarket_venue_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_filled_order_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_filled_decision_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_filled_fill_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_ioc_or_fok_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_requested_notional_risk_cap_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
        settings=settings,
    )
    _require_filled_notional_domain_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_remaining_notional_domain_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_notional_accounting_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_fill_price_domain_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_fill_arithmetic_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_order_fill_consistency_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_live_order_chronology_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_pre_submit_quote_source_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_pre_submit_quote_hash_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_pre_submit_quote_fingerprint_for_final_post_live_artifact(
        final_post_live_valid=final_post_live_valid,
        record=record,
    )
    _require_finite_portfolio_for_post_live_artifact(portfolio)
    require_preflight_artifact_concrete_path(output_path)
    require_preflight_artifact_outside_working_tree(output_path)
    _require_output_distinct_from_config_file(output_path, config_path)
    _require_live_order_output_distinct_from_live_inputs(settings, output_path)
    require_preflight_artifact_parent_owner_writable(output_path)
    require_preflight_artifact_regular_file_path(output_path, must_exist=False)
    artifact_mode = (
        "post_live_order_reconciliation"
        if final_post_live_valid
        else "incomplete_post_live_order_reconciliation"
    )
    artifact = {
        "generated_by": "pms-live reconcile-live-order",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "artifact_mode": artifact_mode,
        "final_post_live_valid": final_post_live_valid,
        "config_path": config_path,
        "database_url_override_used": database_url_override_used,
        "settings_fingerprint": live_preflight_settings_fingerprint(settings),
        "credentialed_preflight_artifact": credentialed_preflight_artifact,
        "output_path": _artifact_output_path(output_path),
        "reconciled_by": reconciled_by,
        "portfolio": _portfolio_artifact_payload(portfolio),
        "venue_reconciliation": {
            "ok": report.ok,
            "mismatches": list(report.mismatches),
        },
        **record.as_artifact_payload(),
    }
    _write_text_no_follow(
        output_path,
        json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def _require_finite_portfolio_for_post_live_artifact(portfolio: Portfolio) -> None:
    _require_finite_artifact_float("portfolio.total_usdc", portfolio.total_usdc)
    _require_finite_artifact_float("portfolio.free_usdc", portfolio.free_usdc)
    _require_finite_artifact_float("portfolio.locked_usdc", portfolio.locked_usdc)
    for index, position in enumerate(portfolio.open_positions):
        prefix = f"portfolio.open_positions[{index}]"
        _require_finite_artifact_float(
            f"{prefix}.shares_held",
            position.shares_held,
        )
        _require_finite_artifact_float(
            f"{prefix}.avg_entry_price",
            position.avg_entry_price,
        )
        _require_finite_artifact_float(
            f"{prefix}.locked_usdc",
            position.locked_usdc,
        )


def _require_finite_artifact_float(field_name: str, value: float) -> None:
    if math.isfinite(value):
        return
    msg = f"{field_name} must be finite for post-live reconciliation artifact"
    raise ValueError(msg)


def _require_venue_reconciliation_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    report: ReconciliationReport,
) -> None:
    if not final_post_live_valid:
        return
    if report.ok and not report.mismatches:
        return
    msg = "venue reconciliation must pass for final post-live artifact"
    raise ValueError(msg)


def _require_core_identity_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    for field_name, value in (
        ("decision_id", record.decision_id),
        ("order_id", record.order_id),
        ("market_id", record.market_id),
        ("token_id", record.token_id),
        ("strategy_id", record.strategy_id),
        ("strategy_version_id", record.strategy_version_id),
        ("fill_id", record.fill_id),
    ):
        if value.strip() != "" and not _looks_like_live_artifact_placeholder(value):
            continue
        msg = f"{field_name} must be concrete for final post-live artifact"
        raise ValueError(msg)


def _require_order_intent_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    action = record.action
    if (
        action is None
        or action.strip() == ""
        or _looks_like_live_artifact_placeholder(action)
    ):
        msg = "action must be concrete for final post-live artifact"
        raise ValueError(msg)
    if action.strip().upper() not in {Side.BUY.value, Side.SELL.value}:
        msg = "action must be BUY or SELL for final post-live artifact"
        raise ValueError(msg)

    outcome = record.outcome
    if (
        outcome is None
        or outcome.strip() == ""
        or _looks_like_live_artifact_placeholder(outcome)
    ):
        msg = "outcome must be concrete for final post-live artifact"
        raise ValueError(msg)


def _require_canonical_database_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    database_url_override_used: bool,
) -> None:
    if not final_post_live_valid or not database_url_override_used:
        return
    msg = "database-url override is not valid for final post-live artifact"
    raise ValueError(msg)


def _require_closed_order_evidence_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    open_statuses = {"live", "unmatched", "partial"}
    if (
        record.order_status.strip().lower() in open_statuses
        and record.remaining_notional_usdc > 1e-9
    ):
        msg = "open order evidence is not valid for final post-live artifact"
        raise ValueError(msg)


def _require_polymarket_venue_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    if record.venue.strip().lower() == "polymarket":
        return
    msg = "venue must be polymarket for final post-live artifact"
    raise ValueError(msg)


def _require_filled_order_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    if record.order_status.strip().lower() in {"matched", "filled"}:
        return
    msg = "order status must be matched or filled for final post-live artifact"
    raise ValueError(msg)


def _require_filled_decision_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    if record.decision_status.strip().lower() == "filled":
        return
    msg = "decision status must be filled for final post-live artifact"
    raise ValueError(msg)


def _require_filled_fill_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    if record.fill_status.strip().lower() in {"matched", "filled"}:
        return
    msg = "fill status must be matched or filled for final post-live artifact"
    raise ValueError(msg)


def _require_ioc_or_fok_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    if record.time_in_force.strip().upper() in {"IOC", "FOK"}:
        return
    msg = "time_in_force must be IOC/FOK for final post-live artifact"
    raise ValueError(msg)


def _require_requested_notional_risk_cap_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
    settings: PMSSettings,
) -> None:
    if not final_post_live_valid:
        return
    if record.requested_notional_usdc < settings.risk.min_order_usdc:
        msg = (
            "requested_notional_usdc is below risk.min_order_usdc for final "
            "post-live artifact"
        )
        raise ValueError(msg)
    if record.requested_notional_usdc <= settings.risk.max_position_per_market:
        return
    msg = (
        "requested_notional_usdc exceeds risk.max_position_per_market for final "
        "post-live artifact"
    )
    raise ValueError(msg)


def _require_filled_notional_domain_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    if record.filled_notional_usdc > 0.0:
        return
    msg = "filled_notional_usdc must be > 0.0 for final post-live artifact"
    raise ValueError(msg)


def _require_remaining_notional_domain_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    if record.remaining_notional_usdc >= 0.0:
        return
    msg = "remaining_notional_usdc must be >= 0.0 for final post-live artifact"
    raise ValueError(msg)


def _require_notional_accounting_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    observed_total = record.filled_notional_usdc + record.remaining_notional_usdc
    if abs(observed_total - record.requested_notional_usdc) <= 1e-9:
        return
    msg = "notional accounting must balance for final post-live artifact"
    raise ValueError(msg)


def _require_fill_price_domain_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    if 0.0 < record.fill_price < 1.0:
        return
    msg = "fill_price must satisfy 0.0 < fill_price < 1.0 for final post-live artifact"
    raise ValueError(msg)


def _require_fill_arithmetic_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    implied_notional = record.filled_quantity * record.fill_price
    if abs(implied_notional - record.filled_notional_usdc) <= 1e-9:
        return
    msg = "fill arithmetic must match for final post-live artifact"
    raise ValueError(msg)


def _require_order_fill_consistency_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    if (
        abs(record.fill_notional_usdc - record.filled_notional_usdc) <= 1e-9
        and abs(record.fill_quantity - record.filled_quantity) <= 1e-9
    ):
        return
    msg = "fill record must match order fill totals for final post-live artifact"
    raise ValueError(msg)


def _require_live_order_chronology_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    submitted_at = _live_order_artifact_timestamp_utc(record.submitted_at)
    last_updated_at = _live_order_artifact_timestamp_utc(record.last_updated_at)
    filled_at = _live_order_artifact_timestamp_utc(record.filled_at)
    if submitted_at <= last_updated_at and submitted_at <= filled_at:
        return
    msg = "live order chronology is invalid for final post-live artifact"
    raise ValueError(msg)


def _require_pre_submit_quote_source_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    source = record.pre_submit_quote_source
    if source is not None and source.strip() in {"dual", "venue_direct"}:
        return
    msg = "pre-submit quote source must be dual or venue_direct for final post-live artifact"
    raise ValueError(msg)


def _require_pre_submit_quote_hash_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    if record.pre_submit_quote_hash.strip() != "":
        return
    msg = "pre-submit quote hash is required for final post-live artifact"
    raise ValueError(msg)


def _require_pre_submit_quote_fingerprint_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    record: LiveOrderReconciliationRecord,
) -> None:
    if not final_post_live_valid:
        return
    if _is_sha256_hexdigest(record.pre_submit_quote_fingerprint):
        return
    msg = "pre-submit quote fingerprint must be a sha256 hex digest for final post-live artifact"
    raise ValueError(msg)


def _live_order_artifact_timestamp_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _require_preflight_reference_for_final_post_live_artifact(
    *,
    final_post_live_valid: bool,
    settings: PMSSettings,
    credentialed_preflight_artifact: Mapping[str, object] | None,
) -> None:
    if not final_post_live_valid:
        return
    if credentialed_preflight_artifact is None:
        msg = "credentialed_preflight_artifact is required for final post-live artifact"
        raise ValueError(msg)
    for field_name in ("path", "sha256"):
        value = credentialed_preflight_artifact.get(field_name)
        if not isinstance(value, str) or value.strip() == "":
            msg = (
                "credentialed_preflight_artifact must include non-empty "
                f"{field_name}"
            )
            raise ValueError(msg)
    reference_path = credentialed_preflight_artifact["path"]
    assert isinstance(reference_path, str)
    if _looks_like_live_artifact_placeholder(reference_path):
        msg = "credentialed_preflight_artifact path contains placeholder"
        raise ValueError(msg)
    reference_sha = credentialed_preflight_artifact["sha256"]
    assert isinstance(reference_sha, str)
    if not _is_sha256_hexdigest(reference_sha):
        msg = "credentialed_preflight_artifact sha256 must be a sha256 hex digest"
        raise ValueError(msg)
    _require_preflight_reference_matches_current_artifact(
        credentialed_preflight_artifact,
        settings=settings,
    )
    for field_name in ("generated_at", "artifact_mode"):
        value = credentialed_preflight_artifact.get(field_name)
        if not isinstance(value, str) or value.strip() == "":
            msg = (
                "credentialed_preflight_artifact must include non-empty "
                f"{field_name}"
            )
            raise ValueError(msg)
    if credentialed_preflight_artifact.get("final_go_no_go_valid") is not True:
        msg = (
            "credentialed_preflight_artifact must include "
            "final_go_no_go_valid=true"
        )
        raise ValueError(msg)


def _require_preflight_reference_matches_current_artifact(
    credentialed_preflight_artifact: Mapping[str, object],
    *,
    settings: PMSSettings,
) -> None:
    raw_reference_path = credentialed_preflight_artifact["path"]
    if not isinstance(raw_reference_path, str):
        msg = "credentialed_preflight_artifact must include non-empty path"
        raise ValueError(msg)
    reference_path = Path(raw_reference_path).expanduser()
    if not reference_path.is_absolute():
        msg = "credentialed_preflight_artifact path must be absolute"
        raise ValueError(msg)

    configured_path = settings.live_preflight_artifact_path
    if configured_path is None or configured_path.strip() == "":
        msg = (
            "credentialed_preflight_artifact path must match configured "
            "live_preflight_artifact_path"
        )
        raise ValueError(msg)
    else:
        configured = Path(configured_path).expanduser()
        if reference_path.resolve(strict=False) != configured.resolve(strict=False):
            msg = (
                "credentialed_preflight_artifact path must match configured "
                "live_preflight_artifact_path"
            )
            raise ValueError(msg)

    try:
        require_live_preflight_artifact(settings)
    except RuntimeError as exc:
        msg = (
            "credentialed_preflight_artifact must reference a valid LIVE "
            "credentialed preflight artifact"
        )
        raise ValueError(msg) from exc

    try:
        content = _read_bytes_no_follow(reference_path)
    except OSError as exc:
        msg = "credentialed_preflight_artifact path is unreadable"
        raise ValueError(msg) from exc
    reference_sha = credentialed_preflight_artifact["sha256"]
    if not isinstance(reference_sha, str):
        msg = "credentialed_preflight_artifact must include non-empty sha256"
        raise ValueError(msg)
    if sha256(content).hexdigest() != reference_sha:
        msg = (
            "credentialed_preflight_artifact sha256 does not match referenced "
            "preflight artifact"
        )
        raise ValueError(msg)
    current_reference = _credentialed_preflight_reference_from_content(
        content,
        path=reference_path,
    )
    for field_name in ("generated_at", "artifact_mode", "final_go_no_go_valid"):
        if credentialed_preflight_artifact.get(field_name) != current_reference[field_name]:
            msg = (
                "credentialed_preflight_artifact metadata does not match "
                f"referenced preflight artifact: {field_name}"
            )
            raise ValueError(msg)


def _looks_like_live_artifact_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "":
        return False
    placeholder_markers = (
        "fill_in",
        "__fill",
        "<",
        ">",
        "todo",
        "replace",
        "placeholder",
    )
    return any(marker in normalized for marker in placeholder_markers)


def _is_sha256_hexdigest(value: str) -> bool:
    candidate = value.strip()
    return len(candidate) == 64 and all(
        character in "0123456789abcdef" for character in candidate
    )


def _artifact_output_path(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def _portfolio_artifact_payload(portfolio: Portfolio) -> dict[str, object]:
    return {
        "total_usdc": portfolio.total_usdc,
        "free_usdc": portfolio.free_usdc,
        "locked_usdc": portfolio.locked_usdc,
        "open_positions_count": len(portfolio.open_positions),
        "open_positions": [
            _position_artifact_payload(position)
            for position in portfolio.open_positions
        ],
    }


def _position_artifact_payload(position: Position) -> dict[str, object]:
    return {
        "market_id": position.market_id,
        "token_id": position.token_id,
        "venue": position.venue,
        "side": position.side,
        "shares_held": position.shares_held,
        "avg_entry_price": position.avg_entry_price,
        "locked_usdc": position.locked_usdc,
        "risk_group_id": position.risk_group_id,
    }


def _credentialed_preflight_artifact_reference(
    settings: PMSSettings,
    *,
    submitted_at: datetime | None = None,
) -> dict[str, object]:
    require_live_preflight_artifact(settings)
    raw_path = settings.live_preflight_artifact_path
    if raw_path is None or raw_path.strip() == "":
        msg = "LIVE credentialed preflight artifact missing"
        raise RuntimeError(msg)
    path = Path(raw_path).expanduser()
    try:
        content = _read_bytes_no_follow(path)
    except OSError as exc:
        msg = f"LIVE credentialed preflight artifact is unreadable: {path}"
        raise RuntimeError(msg) from exc
    if submitted_at is not None:
        _require_preflight_artifact_before_live_order(
            content,
            submitted_at=submitted_at,
        )
    return _credentialed_preflight_reference_from_content(content, path=path)


def _credentialed_preflight_reference_from_content(
    content: bytes,
    *,
    path: Path,
) -> dict[str, object]:
    artifact = _credentialed_preflight_artifact_payload(content)
    return {
        "path": str(path.resolve(strict=False)),
        "sha256": sha256(content).hexdigest(),
        "generated_at": _required_preflight_reference_string(
            artifact,
            field_name="generated_at",
        ),
        "artifact_mode": _required_preflight_reference_string(
            artifact,
            field_name="artifact_mode",
        ),
        "final_go_no_go_valid": artifact.get("final_go_no_go_valid") is True,
    }


def _credentialed_preflight_artifact_payload(content: bytes) -> Mapping[str, object]:
    try:
        artifact = loads_json_rejecting_duplicate_keys(
            content.decode("utf-8"),
            label="LIVE credentialed preflight artifact",
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        msg = "LIVE credentialed preflight artifact is unreadable"
        raise RuntimeError(msg) from exc
    if not isinstance(artifact, dict):
        msg = "LIVE credentialed preflight artifact must be a JSON object"
        raise RuntimeError(msg)
    return cast(Mapping[str, object], artifact)


def _required_preflight_reference_string(
    artifact: Mapping[str, object],
    *,
    field_name: str,
) -> str:
    raw_value = artifact.get(field_name)
    if not isinstance(raw_value, str) or raw_value.strip() == "":
        msg = f"LIVE credentialed preflight artifact missing {field_name}"
        raise RuntimeError(msg)
    return raw_value


def _require_preflight_artifact_before_live_order(
    content: bytes,
    *,
    submitted_at: datetime,
) -> None:
    artifact = _credentialed_preflight_artifact_payload(content)
    raw_generated_at = _required_preflight_reference_string(
        artifact,
        field_name="generated_at",
    )
    try:
        generated_at = datetime.fromisoformat(
            raw_generated_at.strip().replace("Z", "+00:00")
        )
    except ValueError as exc:
        msg = "LIVE credentialed preflight artifact generated_at is invalid"
        raise RuntimeError(msg) from exc
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)
    if submitted_at.tzinfo is None:
        submitted_at = submitted_at.replace(tzinfo=UTC)
    if generated_at.astimezone(UTC) <= submitted_at.astimezone(UTC):
        return
    msg = (
        "LIVE credentialed preflight artifact generated_at postdates live order "
        "submission"
    )
    raise RuntimeError(msg)


def _prepare_preflight_artifact_parent(path: Path) -> None:
    parent = path.parent
    try:
        mode = parent.lstat().st_mode
    except FileNotFoundError:
        parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.chmod(parent, 0o700)
        return
    if not stat.S_ISDIR(mode):
        raise OSError(f"preflight artifact parent path is not a directory: {parent}")
    permissions = stat.S_IMODE(mode)
    if permissions & 0o077:
        raise OSError(
            f"preflight artifact parent directory {parent} is too permissive; "
            f"run `chmod 700 {parent}`."
        )
    if not permissions & stat.S_IWUSR:
        raise OSError(
            f"preflight artifact parent directory {parent} is not owner-writable; "
            f"run `chmod 700 {parent}`."
        )


def _write_text_no_follow(path: Path, content: str) -> None:
    _require_regular_file_or_absent(path)
    fd, temp_path = _open_artifact_temp_file(path)
    published = False
    try:
        os.fchmod(fd, 0o600)
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            fd = -1
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        _require_regular_file_or_absent(path)
        os.replace(temp_path, path)
        published = True
        _fsync_parent_directory(path)
    finally:
        if fd >= 0:
            os.close(fd)
        if not published:
            _unlink_regular_single_link_file_if_present(temp_path)


def _open_artifact_temp_file(path: Path) -> tuple[int, Path]:
    _require_regular_file_or_absent(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for _ in range(16):
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            fd = os.open(temp_path, flags, 0o600)
        except FileExistsError:
            continue
        try:
            path_stat = os.fstat(fd)
            if not stat.S_ISREG(path_stat.st_mode):
                raise OSError(
                    "preflight artifact output path is not a regular file: "
                    f"{temp_path}"
                )
            if path_stat.st_nlink != 1:
                raise OSError(
                    "preflight artifact output path is not a single-link file: "
                    f"{temp_path}"
                )
            os.fchmod(fd, 0o600)
        except BaseException:
            os.close(fd)
            _unlink_regular_single_link_file_if_present(temp_path)
            raise
        return fd, temp_path
    raise FileExistsError(f"could not create temporary live artifact for {path}")


def _unlink_regular_single_link_file_if_present(path: Path) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
        return
    path.unlink()


def _fsync_parent_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path.parent, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        return
    finally:
        os.close(fd)


def _read_bytes_no_follow(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
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


def _require_regular_file_or_absent(path: Path) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return
    mode = path_stat.st_mode
    if not stat.S_ISREG(mode):
        raise OSError(f"preflight artifact output path is not a regular file: {path}")
    if path_stat.st_nlink != 1:
        raise OSError(
            f"preflight artifact output path is not a single-link file: {path}"
        )


def _unreachable(command: object) -> NoReturn:
    raise RuntimeError(f"unsupported pms-live command: {command!r}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
