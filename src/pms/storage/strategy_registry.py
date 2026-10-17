from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
import logging
import math
from typing import cast

import asyncpg

from pms.strategies.aggregate import Strategy
from pms.strategies.projections import (
    ActiveStrategy,
    CalibrationSpec,
    DEFAULT_MAX_BRIER_SCORE,
    DEFAULT_MIN_WIN_RATE,
    DEFAULT_SLIPPAGE_THRESHOLD_BPS,
    EvalSpec,
    FactorCompositionStep,
    ForecasterSpec,
    MarketSelectionSpec,
    RiskParams,
    StrategyConfig,
    StrategyRow,
    StrategyVersion,
)
from pms.strategies.versioning import (
    compute_legacy_strategy_version_id,
    compute_strategy_version_id,
    serialize_strategy_config_json,
)


logger = logging.getLogger(__name__)

StrategyChangeCallback = Callable[[], Awaitable[None]]

# The load-bearing legacy bootstrap version id seeded by schema.sql
# (`INSERT INTO strategy_versions ... 'default-v1'`). It predates
# content-addressed version ids, so its label is not the config hash. It is
# exempt from the content-address check below; `runner._ensure_default_v2_version`
# uses this same sentinel to detect and migrate the legacy default to v2.
_LEGACY_BOOTSTRAP_STRATEGY_VERSION_ID = "default-v1"


class PostgresStrategyRegistry:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._strategy_change_callbacks: list[StrategyChangeCallback] = []

    def register_change_callback(self, fn: StrategyChangeCallback) -> None:
        if fn not in self._strategy_change_callbacks:
            self._strategy_change_callbacks.append(fn)

    def unregister_change_callback(self, fn: StrategyChangeCallback) -> None:
        if fn in self._strategy_change_callbacks:
            self._strategy_change_callbacks.remove(fn)

    async def create_strategy(
        self,
        strategy_id: str,
        metadata: dict[str, str] | None = None,
    ) -> None:
        query = """
        INSERT INTO strategies (strategy_id, metadata_json)
        VALUES ($1, $2::jsonb)
        ON CONFLICT (strategy_id) DO NOTHING
        """
        metadata_json = json.dumps(
            metadata or {},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        async with self._pool.acquire() as connection:
            await connection.execute(query, strategy_id, metadata_json)

    async def create_version(
        self,
        strategy: Strategy,
        *,
        activate: bool = True,
    ) -> StrategyVersion:
        """Register a strategy snapshot and optionally make that version active.

        Re-registering an existing version is idempotent for the
        `strategy_versions` row. By default this still re-points
        `strategies.active_version_id` to the requested version; installers can
        pass `activate=False` to populate dependent rows before activation.
        """
        strategy_id = strategy.config.strategy_id
        strategy_version_id = compute_strategy_version_id(*strategy.snapshot())
        config_json = _strategy_to_config_json(strategy)
        metadata_json = json.dumps(
            dict(strategy.config.metadata),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        insert_strategy_query = """
        INSERT INTO strategies (strategy_id, metadata_json)
        VALUES ($1, $2::jsonb)
        ON CONFLICT (strategy_id) DO NOTHING
        """
        insert_version_query = """
        INSERT INTO strategy_versions (
            strategy_version_id,
            strategy_id,
            config_json,
            created_at
        ) VALUES ($1, $2, $3::jsonb, clock_timestamp())
        ON CONFLICT (strategy_version_id) DO NOTHING
        RETURNING created_at
        """
        select_version_query = """
        SELECT created_at
        FROM strategy_versions
        WHERE strategy_id = $1 AND strategy_version_id = $2
        """
        update_strategy_query = """
        UPDATE strategies
        SET active_version_id = $2,
            archived = FALSE
        WHERE strategy_id = $1
        """
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    insert_strategy_query,
                    strategy_id,
                    metadata_json,
                )
                created_at = await connection.fetchval(
                    insert_version_query,
                    strategy_version_id,
                    strategy_id,
                    config_json,
                )
                if created_at is None:
                    created_at = await connection.fetchval(
                        select_version_query,
                        strategy_id,
                        strategy_version_id,
                    )
                if not isinstance(created_at, datetime):
                    msg = "strategy_versions.created_at did not return a timestamp"
                    raise TypeError(msg)
                if activate:
                    await connection.execute(
                        update_strategy_query,
                        strategy_id,
                        strategy_version_id,
                    )
        if activate:
            await self._notify_strategy_change()
        return StrategyVersion(
            strategy_id=strategy_id,
            strategy_version_id=strategy_version_id,
            created_at=_ensure_utc(created_at),
        )

    async def set_version_metadata(
        self,
        strategy_id: str,
        strategy_version_id: str,
        metadata: dict[str, object],
    ) -> None:
        metadata_json = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        query = """
        UPDATE strategy_versions
        SET metadata_json = metadata_json || $3::jsonb
        WHERE strategy_id = $1 AND strategy_version_id = $2
        """
        async with self._pool.acquire() as connection:
            status = await connection.execute(
                query,
                strategy_id,
                strategy_version_id,
                metadata_json,
            )
        updated = _command_tag_row_count(status)
        if updated != 1:
            msg = (
                "No strategy_versions row matched "
                f"strategy_id={strategy_id!r}, strategy_version_id={strategy_version_id!r}"
            )
            raise LookupError(msg)

    async def get_by_id(self, strategy_id: str) -> Strategy | None:
        query = """
        SELECT versions.config_json
        FROM strategies
        LEFT JOIN strategy_versions AS versions
            ON versions.strategy_version_id = strategies.active_version_id
        WHERE strategies.strategy_id = $1
        """
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(query, strategy_id)
        if row is None or row["config_json"] is None:
            return None
        return _strategy_from_config_json(row["config_json"])

    async def list_strategies(self) -> list[StrategyRow]:
        query = """
        SELECT strategy_id, active_version_id, created_at
        FROM strategies
        ORDER BY strategy_id ASC
        """
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(query)
        return [
            StrategyRow(
                strategy_id=row["strategy_id"],
                active_version_id=row["active_version_id"],
                created_at=_ensure_utc(row["created_at"]),
            )
            for row in rows
        ]

    async def list_active_strategy_rows(self) -> list[StrategyRow]:
        query = """
        SELECT strategy_id, active_version_id, created_at
        FROM strategies
        WHERE active_version_id IS NOT NULL
          AND archived IS NOT TRUE
        ORDER BY strategy_id ASC
        """
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(query)
        return [
            StrategyRow(
                strategy_id=row["strategy_id"],
                active_version_id=row["active_version_id"],
                created_at=_ensure_utc(row["created_at"]),
            )
            for row in rows
        ]

    async def list_versions(self, strategy_id: str) -> list[StrategyVersion]:
        query = """
        SELECT strategy_id, strategy_version_id, created_at
        FROM strategy_versions
        WHERE strategy_id = $1
        ORDER BY created_at ASC, strategy_version_id ASC
        """
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(query, strategy_id)
        return [
            StrategyVersion(
                strategy_id=row["strategy_id"],
                strategy_version_id=row["strategy_version_id"],
                created_at=_ensure_utc(row["created_at"]),
            )
            for row in rows
        ]

    async def list_market_selections(
        self,
    ) -> list[tuple[str, str, MarketSelectionSpec]]:
        query = """
        SELECT
            strategies.strategy_id,
            strategies.active_version_id AS strategy_version_id,
            versions.config_json
        FROM strategies
        JOIN strategy_versions AS versions
            ON versions.strategy_id = strategies.strategy_id
           AND versions.strategy_version_id = strategies.active_version_id
        WHERE strategies.active_version_id IS NOT NULL
          AND strategies.archived IS NOT TRUE
        ORDER BY strategies.strategy_id ASC
        """
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(query)
        selections: list[tuple[str, str, MarketSelectionSpec]] = []
        for row in rows:
            strategy_id = _json_string(row["strategy_id"], "strategy_id")
            strategy_version_id = row["strategy_version_id"]
            if not isinstance(strategy_version_id, str):
                msg = "strategies.active_version_id did not return a string"
                raise TypeError(msg)
            strategy = _strategy_from_versioned_config_json(
                row["config_json"],
                strategy_id=strategy_id,
                strategy_version_id=strategy_version_id,
            )
            selections.append(
                (
                    strategy_id,
                    strategy_version_id,
                    strategy.market_selection,
                )
            )
        return selections

    async def list_active_strategies(self) -> list[ActiveStrategy]:
        query = """
        SELECT
            strategies.strategy_id,
            strategies.active_version_id AS strategy_version_id,
            versions.config_json
        FROM strategies
        JOIN strategy_versions AS versions
            ON versions.strategy_id = strategies.strategy_id
           AND versions.strategy_version_id = strategies.active_version_id
        WHERE strategies.active_version_id IS NOT NULL
          AND strategies.archived IS NOT TRUE
        ORDER BY strategies.strategy_id ASC
        """
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(query)
        active_strategies: list[ActiveStrategy] = []
        for row in rows:
            strategy_id = _json_string(row["strategy_id"], "strategy_id")
            strategy_version_id = row["strategy_version_id"]
            if not isinstance(strategy_version_id, str):
                msg = "strategies.active_version_id did not return a string"
                raise TypeError(msg)
            strategy = _strategy_from_versioned_config_json(
                row["config_json"],
                strategy_id=strategy_id,
                strategy_version_id=strategy_version_id,
            )
            calibration = _calibration_from_config_json(row["config_json"])
            active_strategies.append(
                ActiveStrategy(
                    strategy_id=strategy_id,
                    strategy_version_id=strategy_version_id,
                    config=strategy.config,
                    risk=strategy.risk,
                    eval_spec=strategy.eval_spec,
                    forecaster=strategy.forecaster,
                    market_selection=strategy.market_selection,
                    calibration=calibration,
                )
            )
        return active_strategies

    async def set_active(
        self,
        strategy_id: str,
        strategy_version_id: str,
    ) -> None:
        select_version_query = """
        SELECT strategy_id, strategy_version_id, config_json
        FROM strategy_versions
        WHERE strategy_id = $1 AND strategy_version_id = $2
        """
        update_strategy_query = """
        UPDATE strategies
        SET active_version_id = $2,
            archived = FALSE
        WHERE strategy_id = $1
          AND EXISTS (
              SELECT 1
              FROM strategy_versions
              WHERE strategy_versions.strategy_id = $1
                AND strategy_versions.strategy_version_id = $2
          )
        """
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                select_version_query,
                strategy_id,
                strategy_version_id,
            )
            if row is None:
                msg = (
                    "No strategy_versions row matched activation target "
                    f"strategy_id={strategy_id!r}, "
                    f"strategy_version_id={strategy_version_id!r}"
                )
                raise LookupError(msg)
            row_strategy_id = _json_string(
                row["strategy_id"],
                "strategy_versions.strategy_id",
            )
            row_strategy_version_id = _json_string(
                row["strategy_version_id"],
                "strategy_versions.strategy_version_id",
            )
            _strategy_from_versioned_config_json(
                row["config_json"],
                strategy_id=row_strategy_id,
                strategy_version_id=row_strategy_version_id,
            )
            status = await connection.execute(
                update_strategy_query,
                strategy_id,
                strategy_version_id,
            )
        updated = _command_tag_row_count(status)
        if updated != 1:
            msg = (
                "No strategy_versions row matched activation target "
                f"strategy_id={strategy_id!r}, strategy_version_id={strategy_version_id!r}"
            )
            raise LookupError(msg)
        await self._notify_strategy_change()

    async def archive_strategy(self, strategy_id: str) -> None:
        query = """
        UPDATE strategies
        SET archived = TRUE,
            active_version_id = NULL
        WHERE strategy_id = $1
        """
        async with self._pool.acquire() as connection:
            status = await connection.execute(query, strategy_id)
        updated = _command_tag_row_count(status)
        if updated == 0:
            msg = f"No strategies row matched archive target strategy_id={strategy_id!r}"
            raise LookupError(msg)
        await self._notify_strategy_change()

    async def populate_strategy_factors(
        self,
        strategy_id: str,
        strategy_version_id: str,
        steps: Sequence[FactorCompositionStep],
    ) -> None:
        query = """
        INSERT INTO strategy_factors (
            strategy_id,
            strategy_version_id,
            factor_id,
            param,
            weight,
            direction
        ) VALUES ($1, $2, $3, $4::jsonb, $5, $6)
        ON CONFLICT (strategy_id, strategy_version_id, factor_id) DO NOTHING
        """
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                for step in steps:
                    await connection.execute(
                        query,
                        strategy_id,
                        strategy_version_id,
                        step.factor_id,
                        _strategy_factor_param_json(step.param),
                        step.weight,
                        "long",
                    )

    async def _notify_strategy_change(self) -> None:
        if not self._strategy_change_callbacks:
            return
        for callback in tuple(self._strategy_change_callbacks):
            try:
                await callback()
            except Exception as error:  # noqa: BLE001
                logger.warning("strategy change callback failed: %s", error)


def _ensure_utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        msg = f"expected datetime, got {type(value).__name__}"
        raise TypeError(msg)
    return value.astimezone(UTC)


def _strategy_factor_param_json(param: str) -> str:
    return json.dumps(param, separators=(",", ":"), ensure_ascii=True)


def _strategy_from_config_json(raw_value: object) -> Strategy:
    payload = _load_json_object(raw_value)
    config_payload = _json_object(payload["config"], "config")
    risk_payload = _json_object(payload["risk"], "risk")
    eval_spec_payload = _json_object(payload["eval_spec"], "eval_spec")
    forecaster_payload = _json_object(payload["forecaster"], "forecaster")
    market_selection_payload = _json_object(
        payload["market_selection"],
        "market_selection",
    )
    return Strategy(
        config=StrategyConfig(
            strategy_id=_json_string(config_payload["strategy_id"], "config.strategy_id"),
            factor_composition=_json_factor_composition(config_payload["factor_composition"]),
            metadata=tuple(
                (
                    _json_string(item[0], "config.metadata.key"),
                    _json_string(item[1], "config.metadata.value"),
                )
                for item in _json_pairs(config_payload["metadata"])
            ),
        ),
        risk=RiskParams(
            max_position_notional_usdc=_json_float(
                risk_payload["max_position_notional_usdc"],
                "risk.max_position_notional_usdc",
            ),
            max_daily_drawdown_pct=_json_float(
                risk_payload["max_daily_drawdown_pct"],
                "risk.max_daily_drawdown_pct",
            ),
            min_order_size_usdc=_json_float(
                risk_payload["min_order_size_usdc"],
                "risk.min_order_size_usdc",
            ),
        ),
        eval_spec=EvalSpec(
            metrics=tuple(_json_string_list(eval_spec_payload["metrics"], "eval_spec.metrics")),
            max_brier_score=_json_float(
                eval_spec_payload.get("max_brier_score", DEFAULT_MAX_BRIER_SCORE),
                "eval_spec.max_brier_score",
            ),
            slippage_threshold_bps=_json_float(
                eval_spec_payload.get(
                    "slippage_threshold_bps",
                    DEFAULT_SLIPPAGE_THRESHOLD_BPS,
                ),
                "eval_spec.slippage_threshold_bps",
            ),
            min_win_rate=_json_float(
                eval_spec_payload.get("min_win_rate", DEFAULT_MIN_WIN_RATE),
                "eval_spec.min_win_rate",
            ),
            min_brier_samples=_json_int(
                eval_spec_payload.get("min_brier_samples", 20),
                "eval_spec.min_brier_samples",
            ),
            min_slippage_samples=_json_int(
                eval_spec_payload.get("min_slippage_samples", 20),
                "eval_spec.min_slippage_samples",
            ),
            min_win_rate_samples=_json_int(
                eval_spec_payload.get("min_win_rate_samples", 20),
                "eval_spec.min_win_rate_samples",
            ),
        ),
        forecaster=ForecasterSpec(
            forecasters=tuple(
                (
                    _json_string(item[0], "forecaster.forecasters.name"),
                    tuple(
                        (
                            _json_string(param[0], "forecaster.forecasters.param.key"),
                            _json_string(param[1], "forecaster.forecasters.param.value"),
                        )
                        for param in _json_pairs(item[1])
                    ),
                )
                for item in _json_pairs(forecaster_payload["forecasters"])
            )
        ),
        market_selection=_market_selection_from_payload(market_selection_payload),
        calibration=_calibration_from_config_json(payload),
    )


def _strategy_from_versioned_config_json(
    raw_value: object,
    *,
    strategy_id: str,
    strategy_version_id: str,
) -> Strategy:
    payload = _load_json_object(raw_value)
    strategy = _strategy_from_config_json(raw_value)
    if strategy.config.strategy_id != strategy_id:
        msg = (
            "strategy_id does not match strategy config payload: "
            f"row={strategy_id!r}, payload={strategy.config.strategy_id!r}"
        )
        raise ValueError(msg)
    expected_version_id = compute_strategy_version_id(*strategy.snapshot())
    if (
        strategy_version_id != _LEGACY_BOOTSTRAP_STRATEGY_VERSION_ID
        and expected_version_id != strategy_version_id
    ):
        # Rows written before factor-definition versions were part of the
        # payload have no `factor_definition_versions` key. Preserve their
        # historical identity while requiring the new hash for new rows.
        payload_has_factor_versions = "factor_definition_versions" in payload
        if (
            not payload_has_factor_versions
            and compute_legacy_strategy_version_id(*strategy.snapshot())
            == strategy_version_id
        ):
            logger.warning(
                "legacy strategy version accepted with factor definition drift",
                extra={
                    "event": "factor_definition_version_drift",
                    "strategy_id": strategy_id,
                    "strategy_version_id": strategy_version_id,
                    "expected_version_id": expected_version_id,
                },
            )
            return strategy
        msg = (
            "strategy_version_id does not match strategy config payload: "
            f"row={strategy_version_id!r}, expected={expected_version_id!r}"
        )
        raise ValueError(msg)
    return strategy


def _market_selection_from_config_json(raw_value: object) -> MarketSelectionSpec:
    payload = _load_json_object(raw_value)
    market_selection_payload = _json_object(
        payload["market_selection"],
        "market_selection",
    )
    return _market_selection_from_payload(market_selection_payload)


def _market_selection_from_payload(
    market_selection_payload: dict[str, object],
) -> MarketSelectionSpec:
    return MarketSelectionSpec(
        venue=_json_string(
            market_selection_payload["venue"],
            "market_selection.venue",
        ),
        resolution_time_max_horizon_days=_json_optional_int(
            market_selection_payload["resolution_time_max_horizon_days"],
            "market_selection.resolution_time_max_horizon_days",
        ),
        volume_min_usdc=_json_float(
            market_selection_payload["volume_min_usdc"],
            "market_selection.volume_min_usdc",
        ),
        spread_max_bps=_json_optional_float(
            _default_when_missing(market_selection_payload, "spread_max_bps", 100.0),
            "market_selection.spread_max_bps",
        ),
        depth_min_usdc=_json_optional_float(
            _default_when_missing(market_selection_payload, "depth_min_usdc", 250.0),
            "market_selection.depth_min_usdc",
        ),
        liquidity_min_usdc=_json_optional_float(
            market_selection_payload.get("liquidity_min_usdc"),
            "market_selection.liquidity_min_usdc",
        ),
        yes_price_min=_json_optional_float(
            market_selection_payload.get("yes_price_min"),
            "market_selection.yes_price_min",
        ),
        yes_price_max=_json_optional_float(
            market_selection_payload.get("yes_price_max"),
            "market_selection.yes_price_max",
        ),
        accepting_orders=_json_optional_bool(
            market_selection_payload.get("accepting_orders"),
            "market_selection.accepting_orders",
            default=True,
        ),
    )


def _default_when_missing(
    payload: dict[str, object],
    key: str,
    default: object,
) -> object:
    if key not in payload:
        return default
    return payload[key]


def _calibration_from_config_json(raw_value: object) -> CalibrationSpec:
    payload = _load_json_object(raw_value)
    if "calibration" not in payload:
        return CalibrationSpec()
    calibration_payload = _json_object(payload["calibration"], "calibration")
    min_resolved = _json_optional_int(
        calibration_payload.get(
            "min_resolved",
            calibration_payload.get("min_resolved_for_extreme", 500),
        ),
        "calibration.min_resolved",
    )
    return CalibrationSpec(
        enabled=_json_optional_bool(
            calibration_payload.get("enabled"),
            "calibration.enabled",
            default=True,
        ),
        shrinkage_factor=_json_float(
            calibration_payload.get("shrinkage_factor", 0.35),
            "calibration.shrinkage_factor",
        ),
        shrinkage_bias=_json_float(
            calibration_payload.get("bias", calibration_payload.get("shrinkage_bias", 0.0)),
            "calibration.bias",
        ),
        extreme_clamp_low=_json_float(
            calibration_payload.get("extreme_clamp_low", 0.08),
            "calibration.extreme_clamp_low",
        ),
        extreme_clamp_high=_json_float(
            calibration_payload.get("extreme_clamp_high", 0.92),
            "calibration.extreme_clamp_high",
        ),
        min_resolved_for_extreme=500 if min_resolved is None else min_resolved,
    )


def _strategy_to_config_json(strategy: Strategy) -> str:
    return serialize_strategy_config_json(*strategy.snapshot())


def _load_json_object(raw_value: object) -> dict[str, object]:
    if isinstance(raw_value, str):
        loaded = json.loads(raw_value)
    else:
        loaded = raw_value
    if not isinstance(loaded, dict):
        msg = "strategy config payload must be a JSON object"
        raise TypeError(msg)
    return cast(dict[str, object], loaded)


def _json_object(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"{field_name} must decode to a JSON object"
        raise TypeError(msg)
    return cast(dict[str, object], value)


def _json_pairs(value: object) -> list[list[object]]:
    if not isinstance(value, list):
        msg = "expected JSON array of pairs"
        raise TypeError(msg)
    pairs = cast(list[object], value)
    decoded_pairs: list[list[object]] = []
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            msg = "expected JSON array pair"
            raise TypeError(msg)
        decoded_pairs.append(cast(list[object], pair))
    return decoded_pairs


def _json_factor_composition(value: object) -> tuple[FactorCompositionStep, ...]:
    if not isinstance(value, list):
        msg = "config.factor_composition must decode to a JSON array"
        raise TypeError(msg)
    items = cast(list[object], value)
    if not items:
        return ()
    if all(isinstance(item, list) for item in items):
        return tuple(
            FactorCompositionStep(
                factor_id=_json_string(item[0], "config.factor_composition.key"),
                role="weighted",
                param="",
                weight=_json_float(item[1], "config.factor_composition"),
                threshold=None,
            )
            for item in _json_pairs(value)
        )

    steps: list[FactorCompositionStep] = []
    for raw_step in items:
        step_payload = _json_object(raw_step, "config.factor_composition.step")
        steps.append(
            FactorCompositionStep(
                factor_id=_json_string(
                    _json_required_field(
                        step_payload,
                        "factor_id",
                        "config.factor_composition.step.factor_id",
                    ),
                    "config.factor_composition.step.factor_id",
                ),
                role=_json_string(
                    _json_required_field(
                        step_payload,
                        "role",
                        "config.factor_composition.step.role",
                    ),
                    "config.factor_composition.step.role",
                ),
                param=_json_string(
                    _json_required_field(
                        step_payload,
                        "param",
                        "config.factor_composition.step.param",
                    ),
                    "config.factor_composition.step.param",
                ),
                weight=_json_float(
                    _json_required_field(
                        step_payload,
                        "weight",
                        "config.factor_composition.step.weight",
                    ),
                    "config.factor_composition.step.weight",
                ),
                threshold=_json_optional_float(
                    step_payload.get("threshold"),
                    "config.factor_composition.step.threshold",
                ),
                required=_json_optional_bool(
                    step_payload.get("required"),
                    "config.factor_composition.step.required",
                    default=True,
                ),
                freshness_sla_s=_json_optional_float(
                    step_payload.get("freshness_sla_s"),
                    "config.factor_composition.step.freshness_sla_s",
                ),
                allow_neutral_fallback=_json_optional_bool(
                    step_payload.get("allow_neutral_fallback"),
                    "config.factor_composition.step.allow_neutral_fallback",
                    default=False,
                ),
                enabled=_json_optional_bool(
                    step_payload.get("enabled"),
                    "config.factor_composition.step.enabled",
                    default=True,
                ),
            )
        )
    return tuple(steps)


def _json_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        msg = f"{field_name} must decode to a string"
        raise TypeError(msg)
    return value


def _command_tag_row_count(status: str) -> int | None:
    try:
        return int(status.rsplit(" ", 1)[-1])
    except (IndexError, ValueError):
        return None


def _json_required_field(
    payload: dict[str, object],
    key: str,
    field_name: str,
) -> object:
    if key not in payload:
        msg = f"{field_name} is required"
        raise TypeError(msg)
    return payload[key]


def _json_string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list):
        msg = f"{field_name} must decode to a JSON array"
        raise TypeError(msg)
    decoded: list[str] = []
    for item in value:
        decoded.append(_json_string(item, field_name))
    return decoded


def _json_optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{field_name} must decode to an int or null"
        raise TypeError(msg)
    return value


def _json_optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    return _json_float(value, field_name)


def _json_optional_bool(value: object, field_name: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        msg = f"{field_name} must decode to a boolean or null"
        raise TypeError(msg)
    return value


def _json_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        msg = f"{field_name} must decode to a float-compatible value"
        raise TypeError(msg)
    decoded = float(value)
    if not math.isfinite(decoded):
        msg = f"{field_name} must decode to a finite float"
        raise ValueError(msg)
    return decoded


def _json_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value
