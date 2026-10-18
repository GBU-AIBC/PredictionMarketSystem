from __future__ import annotations

import asyncio
from typing import Any

from pms.runner import Runner
from pms.supervision import ALIVE_WORKER_STATES


def health_payload(*, shutting_down: bool) -> tuple[int, dict[str, Any]]:
    if shutting_down:
        return 200, {"status": "shutting_down"}
    return 200, {"status": "ok"}


def readiness_payload(
    runner: Runner,
    *,
    halt_subscriber_task: asyncio.Task[None] | None,
    eod_scheduler_task: asyncio.Task[None] | None,
    shutting_down: bool = False,
    forced_running: bool = False,
) -> tuple[int, dict[str, Any]]:
    if shutting_down:
        return 503, {"status": "shutting_down", "checks": {}}
    worker_status, worker_detail = _worker_readiness(
        runner,
        forced_running=forced_running,
    )
    checks = {
        "sensors": _sensor_readiness(runner, forced_running=forced_running),
        "event_loop": _event_loop_readiness(runner, forced_running=forced_running),
        "workers": worker_status,
        "halt_subscriber": _task_readiness(halt_subscriber_task),
        "eod_scheduler": _task_readiness(eod_scheduler_task),
    }
    ready_values = {"ready", "disabled"}
    status = "ready" if all(value in ready_values for value in checks.values()) else "not_ready"
    code = 200 if status == "ready" else 503
    return code, {"status": status, "checks": checks, "workers": worker_detail}


def _sensor_readiness(runner: Runner, *, forced_running: bool) -> str:
    if forced_running:
        return "ready"
    if not runner.active_sensors:
        return "not_started"
    if any(not task.done() for task in runner.sensor_stream.tasks):
        return "ready"
    return "not_started"


def _event_loop_readiness(runner: Runner, *, forced_running: bool) -> str:
    if forced_running:
        return "ready"
    if any(not task.done() for task in runner.tasks):
        return "ready"
    return "not_started"


def _worker_readiness(
    runner: Runner,
    *,
    forced_running: bool,
) -> tuple[str, dict[str, Any]]:
    """Ready iff every spawned required worker (dispatcher, actuator,
    factor service when PG runtime, heartbeat writer in PAPER/LIVE) is
    `running` or `restarting`. The detail names dead workers so process
    supervisors and alerting can act on the 503."""
    if forced_running:
        return "ready", {"required": {}, "dead": []}
    snapshot = runner.worker_health_snapshot()
    required = runner.required_worker_names()
    if not required:
        return "not_started", {"required": {}, "dead": []}
    states = {
        name: snapshot[name].state if name in snapshot else "not_started"
        for name in required
    }
    dead = [
        name
        for name, state in states.items()
        if state not in ALIVE_WORKER_STATES
    ]
    detail = {"required": states, "dead": dead}
    if dead:
        return "not_ready", detail
    return "ready", detail


def _task_readiness(task: asyncio.Task[None] | None) -> str:
    if task is None:
        return "disabled"
    if task.done():
        return "stopped"
    return "ready"
