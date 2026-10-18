from __future__ import annotations

import asyncio
import csv
import json
import os
import stat
from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO, cast

from pms.core.enums import Venue
from pms.core.exceptions import KalshiStubError  # noqa: F401
from pms.core.models import MarketSignal, Venue as VenueValue
from pms.core.venue_support import kalshi_stub_error, normalize_venue


@dataclass(frozen=True)
class HistoricalSensor:
    data_path: Path
    replay_speed: float = 0.0

    def __post_init__(self) -> None:
        if self.replay_speed < 0.0:
            msg = "replay_speed must be non-negative"
            raise ValueError(msg)

    def __aiter__(self) -> AsyncIterator[MarketSignal]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MarketSignal]:
        previous: datetime | None = None
        for signal in self._load_signals():
            if previous is not None and self.replay_speed > 0.0:
                delay_seconds = (
                    signal.fetched_at - previous
                ).total_seconds() / self.replay_speed
                if delay_seconds > 0.0:
                    await asyncio.sleep(delay_seconds)
            yield signal
            previous = signal.fetched_at

    async def signal_at(
        self, ts: datetime, *, market_id: str, token_id: str | None,
    ) -> MarketSignal:
        latest: MarketSignal | None = None
        for row in _read_rows(self.data_path):
            signal = _row_to_signal(row)
            if signal.market_id == market_id and signal.token_id == token_id and signal.fetched_at <= ts:
                if latest is None or signal.fetched_at >= latest.fetched_at:
                    latest = signal
        if latest is None:
            raise LookupError(f"no historical book as of {ts} for {market_id!r}/{token_id!r}")
        return latest

    def _load_signals(self) -> list[MarketSignal]:
        rows = [_row_to_signal(row) for row in _read_rows(self.data_path)]
        return sorted(rows, key=lambda signal: signal.fetched_at)


def _read_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".csv":
        with _open_text_no_follow(path, newline="") as csv_file:
            yield from csv.DictReader(csv_file)
        return

    if path.suffix in {".jsonl", ".ndjson"}:
        with _open_text_no_follow(path) as jsonl_file:
            for line in jsonl_file:
                if line.strip():
                    loaded = json.loads(line)
                    if not isinstance(loaded, dict):
                        msg = f"Expected JSON object row in {path}"
                        raise ValueError(msg)
                    yield loaded
        return

    msg = f"Unsupported historical data file type: {path.suffix}"
    raise ValueError(msg)


@contextmanager
def _open_text_no_follow(
    path: Path,
    *,
    newline: str | None = None,
) -> Iterator[TextIO]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        fd = os.open(path, flags)
        path_stat = os.fstat(fd)
        if not stat.S_ISREG(path_stat.st_mode):
            raise OSError("not a regular file")
        if path_stat.st_nlink != 1:
            raise OSError("multiple hardlinks")
        with os.fdopen(fd, "r", encoding="utf-8", newline=newline) as file:
            fd = -1
            yield file
    except OSError as exc:
        msg = f"historical data cannot be read safely: {path}"
        raise ValueError(msg) from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _row_to_signal(row: dict[str, Any]) -> MarketSignal:
    fetched_at = _required_datetime(row, "fetched_at")
    venue = normalize_venue(
        row.get("venue"),
        context="HistoricalSensor._row_to_signal",
    )
    if venue == Venue.KALSHI.value:
        raise kalshi_stub_error("HistoricalSensor._row_to_signal")
    return MarketSignal(
        market_id=str(row["market_id"]),
        token_id=_optional_str(row.get("token_id")),
        venue=cast(VenueValue, venue),
        title=str(row["title"]),
        yes_price=float(row["yes_price"]),
        volume_24h=_optional_float(row.get("volume_24h")),
        resolves_at=_optional_datetime(row.get("resolves_at")),
        orderbook=_mapping(row["orderbook"], field_name="orderbook"),
        external_signal=_mapping(row["external_signal"], field_name="external_signal"),
        fetched_at=fetched_at,
        market_status=str(row["market_status"]),
    )


def _mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, str):
        loaded = json.loads(value)
    else:
        loaded = value
    if not isinstance(loaded, dict):
        msg = f"{field_name} must be a mapping"
        raise ValueError(msg)
    return loaded


def _optional_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(cast(str | int | float, value))


def _optional_datetime(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    return _parse_datetime(str(value))


def _required_datetime(row: dict[str, Any], field_name: str) -> datetime:
    return _parse_datetime(str(row[field_name]))


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
