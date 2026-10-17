from __future__ import annotations

import asyncio
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pms.core.models import OrderState, TradeDecision
from pms.actuator.risk import HaltState


EmergencyRestartMode = Literal["paper", "backtest"]


@dataclass(frozen=True, slots=True)
class LiveEmergencyAuditWriter:
    path: Path

    async def append(
        self,
        *,
        phase: str,
        decision: TradeDecision,
        order_state: OrderState | None,
        error: BaseException,
        error_detail: str | None = None,
    ) -> None:
        record = _audit_record(
            phase=phase,
            decision=decision,
            order_state=order_state,
            error=error,
            error_detail=error_detail,
        )
        await asyncio.to_thread(_append_jsonl, self.path, record)

    async def append_fee_reconcile(
        self,
        *,
        fill_id: str,
        fees: float | None,
        fee_bps: int | None,
        slippage_bps: float | None,
        reconciled_by: str,
        evidence: str,
        override: bool = False,
    ) -> None:
        if not fill_id.strip():
            raise ValueError("fill_id is required")
        if fees is not None and (not math.isfinite(fees) or fees < 0.0):
            raise ValueError("fees must be finite and non-negative")
        if fee_bps is not None and (
            isinstance(fee_bps, bool)
            or not isinstance(fee_bps, int)
            or fee_bps < 0
        ):
            raise ValueError("fee_bps must be a non-negative integer")
        if slippage_bps is not None and not math.isfinite(slippage_bps):
            raise ValueError("slippage_bps must be finite")
        record = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "phase": "live_fee_reconcile",
            "event": "fee_overridden" if override else "fee_reconciled",
            "fill_id": fill_id.strip(),
            "fees": fees,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "reconciled_by": _require_operator_id(reconciled_by, "reconciled_by"),
            "evidence": _require_operator_text(evidence, "evidence"),
        }
        await asyncio.to_thread(_append_jsonl, self.path, record)

    async def append_provenance_tag(
        self,
        *,
        mode: str,
        tagged_count: int,
        operator: str,
        evidence: str,
    ) -> None:
        if mode not in {"paper", "live"}:
            raise ValueError("mode must be paper or live")
        if tagged_count < 0:
            raise ValueError("tagged_count must be non-negative")
        record = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "phase": "live_fill_provenance_tag",
            "event": "fill_provenance_tagged",
            "mode": mode,
            "tagged_count": tagged_count,
            "operator": _require_operator_id(operator, "operator"),
            "evidence": _require_operator_text(evidence, "evidence"),
        }
        await asyncio.to_thread(_append_jsonl, self.path, record)

    async def append_manual_stop(
        self,
        *,
        stopped_by: str,
        reason: str,
        runner_stopped: bool,
        credentials_rotated: bool,
        runtime_secrets_removed: bool,
        venue_open_orders_reviewed: bool,
        database_reconciled: bool,
        restart_mode: EmergencyRestartMode,
    ) -> None:
        record = _manual_stop_record(
            stopped_by=stopped_by,
            reason=reason,
            runner_stopped=runner_stopped,
            credentials_rotated=credentials_rotated,
            runtime_secrets_removed=runtime_secrets_removed,
            venue_open_orders_reviewed=venue_open_orders_reviewed,
            database_reconciled=database_reconciled,
            restart_mode=restart_mode,
        )
        await asyncio.to_thread(_append_jsonl, self.path, record)

    async def append_halt(self, state: HaltState) -> None:
        """Persist the active breaker so a restart cannot silently resume."""
        await asyncio.to_thread(
            _append_jsonl,
            self.path,
            {
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "phase": "risk_halt",
                "event": "risk_halt",
                "reason": state.reason,
                "trigger_kind": state.trigger_kind,
                "triggered_at": state.triggered_at.isoformat(),
            },
        )

    async def append_halt_clear(self, *, cleared_by: str) -> None:
        await asyncio.to_thread(
            _append_jsonl,
            self.path,
            {
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "phase": "risk_halt_clear",
                "event": "risk_halt_clear",
                "cleared_by": _require_operator_id(cleared_by, "cleared_by"),
            },
        )


def read_active_halt(path: Path) -> HaltState | None:
    """Read the last durable halt/clear pair without trusting malformed lines."""
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    active: HaltState | None = None
    for line in raw_lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        phase = payload.get("phase")
        if phase == "risk_halt_clear":
            active = None
            continue
        if phase != "risk_halt":
            continue
        try:
            triggered_at = datetime.fromisoformat(
                str(payload["triggered_at"]).replace("Z", "+00:00")
            )
            if triggered_at.tzinfo is None:
                triggered_at = triggered_at.replace(tzinfo=UTC)
            trigger_kind = payload["trigger_kind"]
            if trigger_kind not in {
                "consecutive_losses",
                "slippage_spike",
                "credential_failure",
                "order_without_fill",
                "rate_limit_exceeded",
                "drawdown_circuit_breaker",
                "daily_loss_limit",
            }:
                continue
            active = HaltState(
                halted=True,
                reason=str(payload["reason"]),
                triggered_at=triggered_at,
                trigger_kind=trigger_kind,
            )
        except (KeyError, TypeError, ValueError):
            continue
    return active


def read_last_halt_clear(path: Path) -> datetime | None:
    """Return the latest durable operator clear timestamp, if any."""
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return None
    latest: datetime | None = None
    for line in raw_lines:
        try:
            payload = json.loads(line)
            if not isinstance(payload, Mapping) or payload.get("phase") != "risk_halt_clear":
                continue
            value = datetime.fromisoformat(
                str(payload["timestamp"]).replace("Z", "+00:00")
            )
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            if latest is None or value > latest:
                latest = value
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return latest


def _manual_stop_record(
    *,
    stopped_by: str,
    reason: str,
    runner_stopped: bool,
    credentials_rotated: bool,
    runtime_secrets_removed: bool,
    venue_open_orders_reviewed: bool,
    database_reconciled: bool,
    restart_mode: EmergencyRestartMode,
) -> dict[str, Any]:
    stopped_by = _require_operator_id(stopped_by, "stopped_by")
    reason = _require_operator_text(reason, "reason")
    _require_true(runner_stopped, "runner_stopped")
    _require_true(credentials_rotated, "credentials_rotated")
    _require_true(runtime_secrets_removed, "runtime_secrets_removed")
    _require_true(venue_open_orders_reviewed, "venue_open_orders_reviewed")
    _require_true(database_reconciled, "database_reconciled")
    if restart_mode not in ("paper", "backtest"):
        msg = "restart_mode must be paper or backtest"
        raise ValueError(msg)
    return {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "phase": "manual_emergency_stop",
        "event": "manual_emergency_stop",
        "stopped_by": stopped_by,
        "reason": reason,
        "runner_stopped": runner_stopped,
        "credentials_rotated": credentials_rotated,
        "runtime_secrets_removed": runtime_secrets_removed,
        "venue_open_orders_reviewed": venue_open_orders_reviewed,
        "database_reconciled": database_reconciled,
        "restart_mode": restart_mode,
    }


def _require_operator_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if normalized == "":
        msg = f"{field_name} is required"
        raise ValueError(msg)
    if _looks_like_placeholder(normalized):
        msg = f"{field_name} must not contain a placeholder"
        raise ValueError(msg)
    return normalized


def _require_operator_id(value: str, field_name: str) -> str:
    normalized = _require_operator_text(value, field_name)
    if any(character in normalized for character in ("|", "\n", "\r")):
        msg = f"{field_name} must not contain delimiters or newlines"
        raise ValueError(msg)
    return normalized


def _require_true(value: bool, field_name: str) -> None:
    if value is True:
        return
    msg = f"{field_name} must be confirmed"
    raise ValueError(msg)


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return (
        lowered == ""
        or "fill_in" in lowered
        or "placeholder" in lowered
        or "todo" in lowered
        or "replace" in lowered
        or lowered.startswith("<")
        or lowered.endswith(">")
    )


def _audit_record(
    *,
    phase: str,
    decision: TradeDecision,
    order_state: OrderState | None,
    error: BaseException,
    error_detail: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "phase": phase,
        "decision_id": decision.decision_id,
        "intent_key": decision.intent_key,
        "market_id": decision.market_id,
        "token_id": decision.token_id,
        "venue": decision.venue,
        "strategy_id": decision.strategy_id,
        "strategy_version_id": decision.strategy_version_id,
        "order_id": None if order_state is None else order_state.order_id,
        "raw_status": None if order_state is None else order_state.raw_status,
        "status": None if order_state is None else order_state.status,
        "pre_submit_quote": (
            {} if order_state is None else dict(order_state.pre_submit_quote)
        ),
        "requested_notional_usdc": (
            decision.notional_usdc
            if order_state is None
            else order_state.requested_notional_usdc
        ),
        "filled_notional_usdc": (
            None if order_state is None else order_state.filled_notional_usdc
        ),
        "remaining_notional_usdc": (
            None if order_state is None else order_state.remaining_notional_usdc
        ),
        "error_type": type(error).__name__,
        "error": str(error) if error_detail is None else error_detail,
    }


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    _prepare_private_parent_directory(path)
    _require_strict_json_value(record, path="$")
    line = (
        json.dumps(
            record,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    _append_line_no_follow(path, line)


def _require_strict_json_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        msg = f"{path} must be finite for emergency audit JSONL"
        raise ValueError(msg)
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_label = str(key)
            child_path = key_label if path == "$" else f"{path}.{key_label}"
            _require_strict_json_value(item, path=child_path)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_strict_json_value(item, path=f"{path}[{index}]")
        return
    msg = f"{path} is not JSON-serializable for emergency audit JSONL"
    raise TypeError(msg)


def _append_line_no_follow(path: Path, line: str) -> None:
    _require_regular_file_or_absent(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        _require_open_regular_single_link_file(fd, path)
        os.fchmod(fd, 0o600)
        _write_all(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_parent_directory(path)


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise OSError("emergency audit write made no progress")
        offset += written


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


def _require_open_regular_single_link_file(fd: int, path: Path) -> None:
    path_stat = os.fstat(fd)
    if not stat.S_ISREG(path_stat.st_mode):
        raise OSError(f"emergency audit path is not a regular file: {path}")
    if path_stat.st_nlink != 1:
        raise OSError(f"emergency audit path is not a single-link file: {path}")


def _prepare_private_parent_directory(path: Path) -> None:
    parent = path.parent
    try:
        parent_stat = parent.lstat()
    except FileNotFoundError:
        parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.chmod(parent, 0o700)
        return
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise OSError(f"emergency audit parent path is not a directory: {parent}")
    mode = stat.S_IMODE(parent_stat.st_mode)
    if mode & 0o077:
        raise OSError(
            f"emergency audit parent directory {parent} is too permissive; "
            f"run `chmod 700 {parent}`."
        )
    if not mode & stat.S_IWUSR:
        raise OSError(
            f"emergency audit parent directory {parent} is not owner-writable; "
            f"run `chmod 700 {parent}`."
        )


def _require_regular_file_or_absent(path: Path) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return
    mode = path_stat.st_mode
    if not stat.S_ISREG(mode):
        raise OSError(f"emergency audit path is not a regular file: {path}")
    if path_stat.st_nlink != 1:
        raise OSError(f"emergency audit path is not a single-link file: {path}")
