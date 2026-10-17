from __future__ import annotations

import getpass
import importlib.util
import math
import os
import re
import stat
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pms.core.enums import RunMode
from pms.core.models import LiveTradingDisabledError, VenueCredentials


class MissingPolymarketCredentialsError(LiveTradingDisabledError):
    """Raised when LIVE mode lacks required Polymarket credentials."""


SecretSource = Literal["fly", "local_file"]
OperatorApprovalMode = Literal["first_order", "every_order"]
DiscoveryOrder = Literal["volume24hr", "volumeNum", "liquidityNum", "createdAt"]


_LIVE_PAPER_REPORT_GENERATOR = "scripts/paper_report.py"
_LIVE_OPERATOR_REHEARSAL_REPORT_GENERATOR = "scripts/rehearse_first_order.py"
_MAX_LIVE_OPERATOR_APPROVAL_AGE_S = 5 * 60
_MAX_LIVE_PREFLIGHT_ARTIFACT_AGE_S = 60 * 60
_MAX_LIVE_READINESS_REPORT_AGE_S = 7 * 24 * 60 * 60
_MAX_LIVE_LLM_DAILY_COST_TO_MIN_ORDER_RATIO = 0.05
_LOOPBACK_API_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_SHA256_DIGESTS = frozenset(
    character * 64 for character in "0123456789abcdef"
)
_REQUIRED_LIVE_PAPER_SOAK_GATE_CHECKS: tuple[str, ...] = (
    "soak_days",
    "decisions_accepted",
    "fills",
    "distinct_markets",
    "distinct_risk_groups",
    "max_market_fill_share",
    "max_risk_group_fill_share",
    "fill_rate",
    "average_slippage_bps",
    "todays_pnl",
    "cumulative_pnl",
    "max_drawdown_pct",
    "open_positions",
    "total_exposure",
    "brier_score",
    "brier_improvement",
    "hit_rate",
    "average_edge_bps",
    "average_net_edge_bps",
    "sharpe_ratio",
    "readiness",
    "strategy_evidence",
    "unresolved_incidents",
    "risk_events",
)
_PAPER_ONLY_STRATEGY_IDS = frozenset(
    {
        "paper_canary_v1",
        "paper_multi_factor_v1",
    }
)
_REQUIRED_LIVE_PAPER_SOAK_BASELINE_SOURCES: tuple[str, ...] = (
    "market_implied",
    "mid_quote",
    "category_prior",
)
_REQUIRED_LIVE_OPERATOR_REHEARSAL_GATE_CHECKS: tuple[str, ...] = (
    "approval_denied",
    "approval_matched",
    "approval_consumed",
    "strict_sidecar_provenance",
    "fresh_approval_required",
    "unexpected_events",
    "operator_id",
)
_EXPECTED_LIVE_OPERATOR_REHEARSAL_EVENTS: tuple[str, ...] = (
    "approval_denied",
    "approval_matched",
    "approval_consumed",
    "approval_denied",
)
_POLYMARKET_CREDENTIAL_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "private_key",
        "api_key",
        "api_secret",
        "api_passphrase",
        "signature_type",
        "funder_address",
    }
)
_PATH_PLACEHOLDER_WORD_RE = re.compile(
    r"(?<![a-z0-9_])(?:todo|replace|placeholder)(?![a-z0-9_])"
)
_TEXT_PLACEHOLDER_WORD_RE = re.compile(
    r"(?<![a-z0-9])(?:todo|replace|placeholder)(?![a-z0-9])"
)
_STRONG_PLACEHOLDER_MARKERS: tuple[str, ...] = ("fill_in", "__fill")
_POLYMARKET_TEXT_CREDENTIAL_CONFIG_FIELDS: frozenset[str] = (
    _POLYMARKET_CREDENTIAL_CONFIG_FIELDS - {"signature_type"}
)
_POLYMARKET_SIGNATURE_TYPES: frozenset[int] = frozenset({0, 1, 2, 3})
_GAMMA_KEYSET_MAX_PAGE_LIMIT = 100


def _read_text_no_follow(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o777)
    try:
        path_stat = os.fstat(fd)
        if not stat.S_ISREG(path_stat.st_mode):
            raise OSError(f"path is not a regular file: {path}")
        if path_stat.st_nlink != 1:
            raise OSError(f"path is not a single-link file: {path}")
        with os.fdopen(fd, "r", encoding="utf-8") as file:
            fd = -1
            return file.read()
    finally:
        if fd >= 0:
            os.close(fd)


class _NoDuplicateSafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping_rejecting_duplicate_keys(
    loader: _NoDuplicateSafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
        if key in mapping:
            msg = f"duplicate YAML key: {key}"
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                msg,
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(  # type: ignore[no-untyped-call]
            value_node,
            deep=deep,
        )
    return mapping


_NoDuplicateSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_rejecting_duplicate_keys,
)


def safe_load_yaml_no_duplicate_keys(text: str) -> object:
    return yaml.load(text, Loader=_NoDuplicateSafeLoader)


def yaml_load_error_message(prefix: str, path: Path, exc: yaml.YAMLError) -> str:
    detail = str(exc)
    if "duplicate YAML key:" in detail:
        return f"{prefix}: {path}: {detail}"
    return f"{prefix}: {path}"


class PolymarketSettings(BaseModel):
    host: str = "https://clob.polymarket.com"
    websocket_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/"
    private_key: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    api_passphrase: str | None = None
    signature_type: int | None = None
    funder_address: str | None = None
    chain_id: int = 137
    first_live_order_approval_path: str | None = None
    operator_approval_mode: OperatorApprovalMode = "first_order"
    operator_approval_max_age_s: float = Field(default=5 * 60, gt=0.0)

    def credentials(self) -> VenueCredentials:
        return VenueCredentials(
            venue="polymarket",
            host=self.host,
            private_key=self.private_key,
            api_key=self.api_key,
            api_secret=self.api_secret,
            api_passphrase=self.api_passphrase,
            signature_type=self.signature_type,
            funder_address=self.funder_address,
            chain_id=self.chain_id,
        )


class LLMSettings(BaseModel):
    enabled: bool = False
    provider: Literal["anthropic", "openai"] | None = None
    api_key: str | None = None
    base_url: str | None = None
    model: str = "claude-sonnet-4-6"
    prompt_version: str = "v1"
    timeout_s: float = Field(default=5.0, gt=0)
    cache_ttl_s: float = Field(default=30.0, ge=0)
    max_tokens: int = Field(default=256, gt=0)
    max_daily_llm_cost_usdc: float | None = Field(default=5.0, gt=0)
    max_retries: int = Field(default=0, ge=0, le=2)
    input_price_usdc_per_million_tokens: float = Field(default=0.0, ge=0.0)
    output_price_usdc_per_million_tokens: float = Field(default=0.0, ge=0.0)
    spend_ledger_path: str | None = None

    @model_validator(mode="after")
    def _validate_when_enabled(self) -> Self:
        if not self.enabled:
            return self
        if self.provider is None:
            raise ValueError("provider is required when LLM is enabled")
        if not self.api_key:
            raise ValueError("api_key is required when LLM is enabled")
        return self


class DiscordSettings(BaseModel):
    webhook_url: SecretStr | None = None
    alert_dir: str = ".alerts"

    @field_validator("webhook_url", mode="before")
    @classmethod
    def _validate_webhook_url(cls, value: object) -> object:
        normalized = normalize_webhook_url(value)
        if normalized is None:
            return None
        if not normalized.startswith("https://"):
            raise ValueError("webhook_url must be an HTTPS URL")
        return normalized

    @field_validator("alert_dir", mode="before")
    @classmethod
    def _validate_alert_dir(cls, value: object) -> object:
        if value is None:
            raise ValueError("alert_dir is required")
        normalized = str(value).strip()
        if normalized == "":
            raise ValueError("alert_dir must not be blank")
        if _path_looks_like_placeholder(normalized):
            raise ValueError("alert_dir must not contain a placeholder")
        return normalized

    def require_webhook_url(self) -> SecretStr:
        if self.webhook_url is None:
            raise ValidationError.from_exception_data(
                "DiscordSettings",
                [
                    {
                        "type": "value_error",
                        "loc": ("webhook_url",),
                        "input": None,
                        "ctx": {"error": ValueError("webhook_url is required")},
                    }
                ],
            )
        return self.webhook_url


def normalize_webhook_url(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        raw_value = value.get_secret_value()
    elif isinstance(value, str):
        raw_value = value
    else:
        return str(value)
    stripped = raw_value.strip()
    return stripped or None


class ControllerSettings(BaseModel):
    min_volume: float = 0.0
    max_slippage_bps: int = 50
    time_in_force: str = "GTC"
    decision_cooldown_s: float = Field(default=60.0, ge=0.0)
    max_book_age_ms: float = 1_000.0
    allowed_book_clock_skew_ms: float = 250.0
    max_spread_bps: float = 100.0
    strict_factor_gates: bool = True
    quote_source: Literal["postgres_snapshot", "venue_direct", "dual"] = (
        "postgres_snapshot"
    )
    venue_book_refresh_enabled: bool = False
    venue_book_refresh_timeout_s: float = Field(default=5.0, gt=0.0)
    direct_quote_min_notional_usdc: float | None = 100.0
    dual_quote_max_price_delta_bps: float = 25.0
    category_prior_observations_path: str | None = None
    category_prior_min_category_samples: int = Field(default=20, ge=1)
    category_prior_min_global_samples: int = Field(default=100, ge=1)
    category_prior_smoothing_alpha: float = Field(default=1.0, gt=0.0)
    category_prior_smoothing_beta: float = Field(default=1.0, gt=0.0)


class RiskSettings(BaseModel):
    max_position_per_market: float = 100.0
    max_total_exposure: float = 1000.0
    max_drawdown_pct: float | None = None
    max_daily_loss_usdc: float | None = Field(default=None, gt=0.0)
    max_open_positions: int | None = None
    max_exposure_per_risk_group: float | None = Field(default=None, gt=0.0)
    min_order_usdc: float = 1.0
    slippage_threshold_bps: float = 50.0
    # Maximum share/contract count per order. Catches the low-price-token
    # blow-up case: at limit_price=0.001 and notional=$10, quantity=10000
    # shares — well beyond a small-test risk envelope. None disables.
    max_quantity_shares: float | None = None


class PositionExitSettings(BaseModel):
    enabled: bool = False
    stop_loss_pct: float | None = Field(default=None, gt=0.0)
    profit_take_pct: float | None = Field(default=None, gt=0.0)
    max_holding_days: int | None = Field(default=None, gt=0)
    reentry_cooldown_s: float = Field(default=0.0, ge=0.0)


class SensorSettings(BaseModel):
    poll_interval_s: float = 5.0
    max_reconnect_interval_s: float = 60.0
    market_data_ws_max_size_bytes: int = Field(default=8 * 1024 * 1024, ge=1)
    max_subscription_asset_ids: int | None = Field(default=100, ge=1)
    discovery_page_limit: int = Field(
        default=_GAMMA_KEYSET_MAX_PAGE_LIMIT,
        ge=1,
        le=500,
    )
    discovery_max_pages: int = Field(default=1, ge=1)
    discovery_pagination_mode: Literal["keyset", "offset"] = "keyset"
    discovery_order: DiscoveryOrder | None = None
    discovery_ascending: bool = False
    discovery_http_timeout_s: float = Field(default=10.0, gt=0.0)
    discovery_http_pool_timeout_s: float = Field(default=10.0, gt=0.0)
    discovery_http_max_connections: int = Field(default=10, ge=1)
    discovery_http_max_keepalive_connections: int = Field(default=5, ge=0)
    discovery_http_keepalive_expiry_s: float = Field(default=120.0, ge=0.0)
    persist_discovery_price_snapshots: bool = False
    persist_price_changes: bool = False

    @model_validator(mode="after")
    def _validate_keyset_page_limit(self) -> Self:
        if (
            self.discovery_pagination_mode == "keyset"
            and self.discovery_page_limit > _GAMMA_KEYSET_MAX_PAGE_LIMIT
        ):
            raise ValueError(
                "sensor.discovery_page_limit for keyset discovery_pagination_mode "
                f"must be <= {_GAMMA_KEYSET_MAX_PAGE_LIMIT}"
            )
        return self


class DashboardSettings(BaseModel):
    stale_snapshot_threshold_s: float = 300.0


class StrategyRuntimeSettings(BaseModel):
    flb_calibration_path: str | None = None
    flb_min_calibration_samples: int = Field(default=100, ge=1)
    flb_entry_execution_cost_bps: float = Field(
        default=15.0,
        ge=0.0,
        allow_inf_nan=False,
    )
    flb_fee_rate: float = Field(
        default=0.07,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )


def _default_database_dsn() -> str:
    override = os.environ.get("DATABASE_URL")
    if override:
        return override
    return f"postgresql://localhost/pms_dev_{getpass.getuser()}"


class DatabaseSettings(BaseModel):
    dsn: str = Field(default_factory=_default_database_dsn)
    pool_min_size: int = 2
    pool_max_size: int = 10
    expired_decision_retention_s: float = Field(default=24 * 60 * 60, ge=0.0)


class PMSSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PMS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: RunMode = RunMode.BACKTEST
    secret_source: SecretSource | None = None
    local_secret_file: str | None = None
    live_trading_enabled: bool = False
    agent_strategy_runtime_enabled: bool = False
    auto_migrate_default_v2: bool = True
    paper_soak_strategy_id: Literal["paper_multi_factor_v1", "h1_flb"] | None = None
    paper_soak_archive_default: bool = False
    enforce_schema_check: bool | None = None
    factor_cadence_s: float = 1.0
    resolution_poll_interval_s: float = Field(default=300.0, gt=0.0)
    api_host: str = "127.0.0.1"
    api_token: str | None = None
    live_account_reconciliation_required: bool = True
    live_emergency_audit_path: str = ".data/live-emergency-audit.jsonl"
    live_first_order_audit_path: str = ".data/first-order-audit.jsonl"
    live_paper_soak_report_path: str | None = None
    live_operator_rehearsal_report_path: str | None = None
    live_execution_model_path: str | None = None
    live_paper_backtest_diff_path: str | None = None
    live_preflight_artifact_path: str | None = None
    live_preflight_artifact_max_age_s: float = Field(default=60 * 60, gt=0.0)
    live_readiness_report_max_age_s: float = Field(default=7 * 24 * 60 * 60, gt=0.0)
    live_exit_criteria_ratified_by: str | None = None
    live_exit_criteria_ratified_at: datetime | None = None
    live_compliance_reviewed_by: str | None = None
    live_compliance_reviewed_at: datetime | None = None
    live_compliance_jurisdiction: str | None = None
    regime_volatility_threshold: float = Field(default=0.15, ge=0.0)
    regime_drift_threshold: float = Field(default=0.02, ge=0.0)
    regime_min_resolved_samples: int = Field(default=5, ge=0)
    decay_min_resolved_samples: int = Field(default=10, ge=0)
    polymarket: PolymarketSettings = Field(default_factory=PolymarketSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    discord: DiscordSettings = Field(default_factory=DiscordSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    position_exit: PositionExitSettings = Field(default_factory=PositionExitSettings)
    sensor: SensorSettings = Field(default_factory=SensorSettings)
    controller: ControllerSettings = Field(default_factory=ControllerSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    strategies: StrategyRuntimeSettings = Field(default_factory=StrategyRuntimeSettings)

    @classmethod
    def load(
        cls,
        config_path: str | Path | None = "config.yaml",
        *,
        load_local_secret_file: bool = True,
    ) -> Self:
        config_data: dict[str, Any] = {}
        path: Path | None = None

        if config_path is not None:
            path = Path(config_path)
            if path.exists():
                try:
                    loaded = safe_load_yaml_no_duplicate_keys(
                        _read_text_no_follow(path)
                    )
                except OSError as exc:
                    msg = f"Config file cannot be read safely: {path}"
                    raise ValueError(msg) from exc
                except yaml.YAMLError as exc:
                    msg = yaml_load_error_message(
                        "Config file is not valid YAML",
                        path,
                        exc,
                    )
                    raise ValueError(msg) from None
                if loaded is not None:
                    if not isinstance(loaded, dict):
                        msg = f"Expected mapping in config file {path}"
                        raise ValueError(msg)
                    config_data = loaded

        _reject_inline_polymarket_credentials(config_data)
        _reject_inline_llm_api_key(config_data)
        if load_local_secret_file:
            config_data = _merge_local_secret_file(config_data)
        settings = cls(**config_data)
        if path is not None:
            _require_live_config_file_distinct_from_protected_paths(
                settings,
                config_path=path,
            )
        return settings


def load_settings(
    config_path: str | Path | None = None,
    *,
    load_local_secret_file: bool = True,
) -> PMSSettings:
    configured_path = config_path
    if configured_path is None:
        configured_path = os.environ.get("PMS_CONFIG_PATH") or "config.yaml"
    return PMSSettings.load(
        configured_path,
        load_local_secret_file=load_local_secret_file,
    )


def validate_live_mode_ready(
    settings: PMSSettings,
    *,
    allow_pending_operator_approval: bool = False,
    require_live_mode: bool = True,
) -> VenueCredentials:
    if not settings.live_trading_enabled:
        msg = "Live trading is disabled. Set live_trading_enabled=true in config."
        raise LiveTradingDisabledError(msg)
    if require_live_mode and settings.mode != RunMode.LIVE:
        msg = (
            "Live trading requires mode=live. Set PMS_MODE=live before "
            f"real-money startup; got mode={settings.mode.value!r}."
        )
        raise LiveTradingDisabledError(msg)

    credentials = settings.polymarket.credentials()
    missing = _missing_polymarket_fields(credentials)
    if missing:
        fields = ", ".join(missing)
        msg = f"Missing Polymarket credential fields: {fields}"
        raise MissingPolymarketCredentialsError(msg)
    placeholder_fields = _placeholder_polymarket_fields(credentials)
    if placeholder_fields:
        fields = ", ".join(placeholder_fields)
        msg = f"Placeholder Polymarket credential fields: {fields}"
        raise MissingPolymarketCredentialsError(msg)
    if credentials.signature_type not in _POLYMARKET_SIGNATURE_TYPES:
        expected_signature_types = ", ".join(
            str(signature_type)
            for signature_type in sorted(_POLYMARKET_SIGNATURE_TYPES)
        )
        msg = (
            "Invalid Polymarket signature_type: "
            f"{credentials.signature_type}; expected one of {expected_signature_types}"
        )
        raise MissingPolymarketCredentialsError(msg)
    if not _is_evm_address(credentials.funder_address):
        msg = (
            "Invalid Polymarket funder_address: expected 0x-prefixed "
            "40 hex characters"
        )
        raise MissingPolymarketCredentialsError(msg)
    if settings.mode == RunMode.LIVE and settings.secret_source not in {"fly", "local_file"}:
        msg = (
            "LIVE mode requires Polymarket credentials to come from an approved "
            "secret source. Set PMS_SECRET_SOURCE=local_file for temporary "
            "local live trading, or PMS_SECRET_SOURCE=fly for Fly deployment."
        )
        raise LiveTradingDisabledError(msg)
    if (
        settings.mode == RunMode.LIVE
        and settings.secret_source == "fly"
        and settings.local_secret_file is not None
        and settings.local_secret_file.strip() != ""
    ):
        msg = (
            "LIVE secret_source=fly must not set local_secret_file; remove "
            "PMS_LOCAL_SECRET_FILE or switch PMS_SECRET_SOURCE=local_file."
        )
        raise LiveTradingDisabledError(msg)
    if settings.mode == RunMode.LIVE and settings.secret_source == "local_file":
        if settings.local_secret_file is None or settings.local_secret_file.strip() == "":
            msg = "LIVE local_file secret source requires PMS_LOCAL_SECRET_FILE."
            raise LiveTradingDisabledError(msg)
        if _path_looks_like_placeholder(settings.local_secret_file):
            msg = "LIVE local secret file path contains placeholder"
            raise LiveTradingDisabledError(msg)
        try:
            secret_path = Path(settings.local_secret_file).expanduser()
            _require_local_secret_file_outside_working_tree(secret_path)
            _require_private_local_secret_file(secret_path)
            _require_private_local_secret_parent(secret_path)
        except ValueError as exc:
            raise LiveTradingDisabledError(str(exc)) from exc
    live_time_in_force = settings.controller.time_in_force.upper()
    if live_time_in_force == "GTC":
        msg = (
            "LIVE GTC disabled until an open-order ledger reserves "
            "resting order exposure"
        )
        raise LiveTradingDisabledError(msg)
    if settings.mode == RunMode.LIVE and live_time_in_force not in {"IOC", "FOK"}:
        msg = (
            "LIVE time_in_force must be IOC or FOK during the initial "
            f"real-money phase; got {settings.controller.time_in_force!r}"
        )
        raise LiveTradingDisabledError(msg)
    if (
        settings.mode == RunMode.LIVE
        and settings.controller.quote_source == "postgres_snapshot"
    ):
        msg = (
            "LIVE quote_source must be venue_direct or dual so pre-submit "
            "validation uses fresh venue evidence"
        )
        raise LiveTradingDisabledError(msg)
    if settings.mode == RunMode.LIVE and not settings.controller.strict_factor_gates:
        msg = (
            "LIVE strict_factor_gates must be true so required raw-factor "
            "evidence cannot be relaxed"
        )
        raise LiveTradingDisabledError(msg)
    if settings.mode == RunMode.LIVE and settings.agent_strategy_runtime_enabled:
        msg = (
            "LIVE agent strategy runtime is disabled until agent-intent "
            "semantics, basket execution, and real-money audit evidence are "
            "production-ready"
        )
        raise LiveTradingDisabledError(msg)
    if (
        settings.mode == RunMode.LIVE
        and settings.polymarket.operator_approval_mode != "every_order"
    ):
        msg = (
            "LIVE mode requires polymarket.operator_approval_mode=every_order "
            "for the initial real-money phase"
        )
        raise LiveTradingDisabledError(msg)
    if settings.mode == RunMode.LIVE:
        _require_live_api_control_plane_auth(settings)
        _require_live_preflight_artifact_path(settings)
        _require_live_alert_fallback_dir(settings)
        _require_live_risk_envelope(settings)
        _require_live_operator_approval_path(
            settings,
            allow_pending_operator_approval=allow_pending_operator_approval,
        )
        _require_distinct_live_launch_control_paths(settings)
        if require_live_mode:
            # The Polymarket SDK is a *runtime* dependency of the real
            # PolymarketSDKClient only. Callers that inject a non-SDK client
            # (test doubles, replay harnesses) pass require_live_mode=False to
            # signal the real venue SDK path will not execute — requiring
            # py_clob_client_v2 then would be wrong. Every production caller
            # (runner.start, live preflight, live CLI, /config) uses the
            # default require_live_mode=True, and live preflight independently
            # re-checks the import via find_spec, so the real-money path stays
            # fully gated.
            _require_live_polymarket_runtime_dependency()
        _require_live_llm_runtime_dependency(settings)
    if settings.mode == RunMode.LIVE and not settings.live_account_reconciliation_required:
        msg = (
            "LIVE account reconciliation must be required before autonomous "
            "live trading can start"
        )
        raise LiveTradingDisabledError(msg)
    if settings.mode == RunMode.LIVE:
        _require_live_freshness_windows(settings)
        missing_readiness = _missing_live_operator_readiness_fields(settings)
        if missing_readiness:
            fields = ", ".join(missing_readiness)
            msg = f"LIVE operator readiness attestation missing: {fields}"
            raise LiveTradingDisabledError(msg)
        invalid_readiness = _invalid_live_operator_readiness_fields(settings)
        if invalid_readiness:
            fields = ", ".join(invalid_readiness)
            msg = f"LIVE operator readiness attestation invalid: {fields}"
            raise LiveTradingDisabledError(msg)
        _require_distinct_live_audit_paths(settings)
        paper_report_generated_at = _require_live_paper_soak_go_report(settings)
        rehearsal_report_generated_at = _require_live_operator_rehearsal_report(
            settings
        )
        _require_live_operator_readiness_after_evidence(
            settings,
            evidence_generated_at=(
                paper_report_generated_at,
                rehearsal_report_generated_at,
            ),
        )
    return credentials


def validate_live_readiness_reports_for_submission(settings: PMSSettings) -> None:
    _require_live_paper_soak_go_report(settings)
    _require_live_operator_rehearsal_report(settings)


def _require_live_api_control_plane_auth(settings: PMSSettings) -> None:
    api_token = settings.api_token
    if api_token is not None and _looks_like_placeholder(api_token):
        msg = "LIVE api_token contains placeholder"
        raise LiveTradingDisabledError(msg)

    if api_token is not None and api_token.strip() != "":
        return
    msg = (
        "LIVE mode requires PMS_API_TOKEN before real-money startup, including "
        "loopback control-plane binds"
    )
    raise LiveTradingDisabledError(msg)


def _require_live_preflight_artifact_path(settings: PMSSettings) -> None:
    raw_path = settings.live_preflight_artifact_path
    if raw_path is None or raw_path.strip() == "":
        msg = (
            "LIVE credentialed preflight artifact path missing: "
            "live_preflight_artifact_path"
        )
        raise LiveTradingDisabledError(msg)
    if _path_looks_like_placeholder(raw_path):
        msg = "LIVE credentialed preflight artifact path contains placeholder"
        raise LiveTradingDisabledError(msg)

    path = Path(raw_path).expanduser()
    _require_live_path_outside_working_tree(
        path,
        label="LIVE credentialed preflight artifact path",
    )
    _require_live_path_parent_owner_writable(
        path,
        label="LIVE credentialed preflight artifact parent directory",
    )
    _require_live_artifact_regular_file_or_absent(
        path,
        label="LIVE credentialed preflight artifact path",
    )
    _require_live_preflight_artifact_path_distinct_from_local_secret(
        settings,
        path,
    )


def _require_live_preflight_artifact_path_distinct_from_local_secret(
    settings: PMSSettings,
    path: Path,
) -> None:
    raw_secret_path = settings.local_secret_file
    if raw_secret_path is None or raw_secret_path.strip() == "":
        return

    preflight_path = _absolute_path_without_symlink_resolution(path)
    secret_path = _absolute_path_without_symlink_resolution(Path(raw_secret_path))
    if not _paths_overlap(preflight_path, secret_path):
        return
    msg = (
        "LIVE credentialed preflight artifact path must be distinct from "
        f"local secret file: {preflight_path}"
    )
    raise LiveTradingDisabledError(msg)


def _require_live_alert_fallback_dir(settings: PMSSettings) -> None:
    if normalize_webhook_url(settings.discord.webhook_url) is None:
        msg = (
            "LIVE discord.webhook_url is required for real-money operator "
            "alerting; set PMS_DISCORD__WEBHOOK_URL before startup"
        )
        raise LiveTradingDisabledError(msg)
    raw_path = settings.discord.alert_dir
    if _path_looks_like_placeholder(raw_path):
        msg = "LIVE discord.alert_dir contains placeholder"
        raise LiveTradingDisabledError(msg)
    path = Path(raw_path).expanduser()
    _require_live_alert_dir_distinct_from_launch_control_paths(settings, path)
    _require_live_path_outside_working_tree(path, label="LIVE discord.alert_dir")
    _require_live_path_parent_owner_writable(path, label="LIVE discord.alert_dir parent")
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(path_stat.st_mode):
        msg = f"LIVE discord.alert_dir is not a directory: {path}"
        raise LiveTradingDisabledError(msg)
    mode = stat.S_IMODE(path_stat.st_mode)
    if mode & 0o077:
        msg = f"LIVE discord.alert_dir {path} is too permissive; run `chmod 700 {path}`."
        raise LiveTradingDisabledError(msg)
    if not mode & stat.S_IWUSR:
        msg = f"LIVE discord.alert_dir is not owner-writable; run `chmod 700 {path}`."
        raise LiveTradingDisabledError(msg)


def _require_live_alert_dir_distinct_from_launch_control_paths(
    settings: PMSSettings,
    path: Path,
) -> None:
    alert_path = _absolute_path_without_symlink_resolution(path)
    approval_path = settings.polymarket.first_live_order_approval_path
    candidates: list[tuple[str, str | None]] = [
        ("operator approval path", approval_path),
        ("first-order audit path", settings.live_first_order_audit_path),
        ("emergency audit path", settings.live_emergency_audit_path),
        ("preflight artifact path", settings.live_preflight_artifact_path),
        ("local secret file", settings.local_secret_file),
    ]
    if approval_path is not None and approval_path.strip() != "":
        candidates.append(
            ("operator approval sidecar path", f"{approval_path}.meta.json")
        )

    for label, raw_candidate in candidates:
        if raw_candidate is None or raw_candidate.strip() == "":
            continue
        candidate_path = _absolute_path_without_symlink_resolution(Path(raw_candidate))
        if not _paths_overlap(alert_path, candidate_path):
            continue
        msg = f"LIVE discord.alert_dir must be distinct from {label}: {alert_path}"
        raise LiveTradingDisabledError(msg)


def _require_live_risk_envelope(settings: PMSSettings) -> None:
    invalid: list[str] = []
    required_positive_fields = {
        "risk.max_position_per_market": settings.risk.max_position_per_market,
        "risk.max_total_exposure": settings.risk.max_total_exposure,
        "risk.max_drawdown_pct": settings.risk.max_drawdown_pct,
        "risk.max_daily_loss_usdc": settings.risk.max_daily_loss_usdc,
        "risk.max_exposure_per_risk_group": (
            settings.risk.max_exposure_per_risk_group
        ),
        "risk.max_quantity_shares": settings.risk.max_quantity_shares,
        "risk.min_order_usdc": settings.risk.min_order_usdc,
    }
    for field_name, value in required_positive_fields.items():
        if value is None:
            invalid.append(f"{field_name} is required")
        elif not _is_finite_positive(value):
            invalid.append(f"{field_name} must be finite and > 0")

    if settings.risk.max_open_positions is None:
        invalid.append("risk.max_open_positions is required")
    elif settings.risk.max_open_positions <= 0:
        invalid.append("risk.max_open_positions must be > 0")

    if (
        _is_finite_positive(settings.risk.min_order_usdc)
        and _is_finite_positive(settings.risk.max_position_per_market)
        and settings.risk.min_order_usdc > settings.risk.max_position_per_market
    ):
        invalid.append(
            "risk.min_order_usdc must be <= risk.max_position_per_market"
        )

    if (
        _is_finite_positive(settings.risk.max_position_per_market)
        and _is_finite_positive(settings.risk.max_total_exposure)
        and settings.risk.max_position_per_market > settings.risk.max_total_exposure
    ):
        invalid.append(
            "risk.max_position_per_market must be <= risk.max_total_exposure"
        )

    if (
        settings.risk.max_exposure_per_risk_group is not None
        and _is_finite_positive(settings.risk.max_exposure_per_risk_group)
        and _is_finite_positive(settings.risk.min_order_usdc)
        and settings.risk.max_exposure_per_risk_group < settings.risk.min_order_usdc
    ):
        invalid.append(
            "risk.max_exposure_per_risk_group must be >= risk.min_order_usdc"
        )

    if (
        settings.risk.max_exposure_per_risk_group is not None
        and _is_finite_positive(settings.risk.max_exposure_per_risk_group)
        and _is_finite_positive(settings.risk.max_total_exposure)
        and settings.risk.max_exposure_per_risk_group
        > settings.risk.max_total_exposure
    ):
        invalid.append(
            "risk.max_exposure_per_risk_group must be <= risk.max_total_exposure"
        )

    if not math.isfinite(settings.risk.slippage_threshold_bps):
        invalid.append("risk.slippage_threshold_bps must be finite")
    elif settings.risk.slippage_threshold_bps < 0.0:
        invalid.append("risk.slippage_threshold_bps must be >= 0")

    if settings.llm.enabled:
        llm_daily_cost = settings.llm.max_daily_llm_cost_usdc
        max_daily_loss = settings.risk.max_daily_loss_usdc
        if llm_daily_cost is None:
            invalid.append(
                "llm.max_daily_llm_cost_usdc is required when LLM is enabled"
            )
        elif not _is_finite_positive(llm_daily_cost):
            invalid.append("llm.max_daily_llm_cost_usdc must be finite and > 0")
        elif (
            max_daily_loss is not None
            and _is_finite_positive(max_daily_loss)
            and llm_daily_cost > max_daily_loss
        ):
            invalid.append(
                "llm.max_daily_llm_cost_usdc must be <= risk.max_daily_loss_usdc"
            )
        if (
            llm_daily_cost is not None
            and _is_finite_positive(llm_daily_cost)
            and _is_finite_positive(settings.risk.min_order_usdc)
            and llm_daily_cost
            > settings.risk.min_order_usdc
            * _MAX_LIVE_LLM_DAILY_COST_TO_MIN_ORDER_RATIO
        ):
            invalid.append(
                "llm.max_daily_llm_cost_usdc must be <= 5% of risk.min_order_usdc"
            )

    if invalid:
        fields = ", ".join(invalid)
        msg = f"LIVE risk envelope invalid: {fields}"
        raise LiveTradingDisabledError(msg)


def _is_finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _require_live_freshness_windows(settings: PMSSettings) -> None:
    if (
        settings.polymarket.operator_approval_max_age_s
        > _MAX_LIVE_OPERATOR_APPROVAL_AGE_S
    ):
        msg = (
            "LIVE operator approval freshness window exceeds "
            f"maximum {_MAX_LIVE_OPERATOR_APPROVAL_AGE_S:.1f}s"
        )
        raise LiveTradingDisabledError(msg)
    if (
        settings.live_preflight_artifact_max_age_s
        > _MAX_LIVE_PREFLIGHT_ARTIFACT_AGE_S
    ):
        msg = (
            "LIVE credentialed preflight artifact freshness window exceeds "
            f"maximum {_MAX_LIVE_PREFLIGHT_ARTIFACT_AGE_S:.1f}s"
        )
        raise LiveTradingDisabledError(msg)
    if (
        settings.live_readiness_report_max_age_s
        > _MAX_LIVE_READINESS_REPORT_AGE_S
    ):
        msg = (
            "LIVE readiness report freshness window exceeds "
            f"maximum {_MAX_LIVE_READINESS_REPORT_AGE_S:.1f}s"
        )
        raise LiveTradingDisabledError(msg)


def _missing_live_operator_readiness_fields(settings: PMSSettings) -> list[str]:
    missing: list[str] = []
    text_fields = {
        "live_exit_criteria_ratified_by": settings.live_exit_criteria_ratified_by,
        "live_compliance_reviewed_by": settings.live_compliance_reviewed_by,
        "live_compliance_jurisdiction": settings.live_compliance_jurisdiction,
    }
    for field_name, value in text_fields.items():
        if value is None or value.strip() == "":
            missing.append(field_name)
    if settings.live_exit_criteria_ratified_at is None:
        missing.append("live_exit_criteria_ratified_at")
    if settings.live_compliance_reviewed_at is None:
        missing.append("live_compliance_reviewed_at")
    return missing


def _invalid_live_operator_readiness_fields(settings: PMSSettings) -> list[str]:
    invalid: list[str] = []
    text_fields = {
        "live_exit_criteria_ratified_by": settings.live_exit_criteria_ratified_by,
        "live_compliance_reviewed_by": settings.live_compliance_reviewed_by,
        "live_compliance_jurisdiction": settings.live_compliance_jurisdiction,
    }
    for field_name, text_value in text_fields.items():
        if text_value is not None and _looks_like_placeholder(text_value):
            invalid.append(f"{field_name} contains placeholder")

    timestamp_fields = {
        "live_exit_criteria_ratified_at": settings.live_exit_criteria_ratified_at,
        "live_compliance_reviewed_at": settings.live_compliance_reviewed_at,
    }
    now = datetime.now(tz=UTC)
    for field_name, timestamp_value in timestamp_fields.items():
        if (
            timestamp_value is not None
            and _coerce_aware_datetime(timestamp_value) > now
        ):
            invalid.append(f"{field_name} is in the future")
    return invalid


def _require_live_operator_readiness_after_evidence(
    settings: PMSSettings,
    *,
    evidence_generated_at: tuple[datetime, ...],
) -> None:
    latest_evidence_at = max(evidence_generated_at)
    invalid: list[str] = []
    timestamp_fields = {
        "live_exit_criteria_ratified_at": settings.live_exit_criteria_ratified_at,
        "live_compliance_reviewed_at": settings.live_compliance_reviewed_at,
    }
    for field_name, timestamp_value in timestamp_fields.items():
        if timestamp_value is None:
            continue
        if _coerce_aware_datetime(timestamp_value) < latest_evidence_at:
            invalid.append(f"{field_name} predates LIVE evidence")

    if invalid:
        fields = ", ".join(invalid)
        msg = f"LIVE operator readiness attestation invalid: {fields}"
        raise LiveTradingDisabledError(msg)


def _looks_like_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "":
        return False
    return (
        _contains_strong_placeholder_marker(normalized)
        or "<" in normalized
        or ">" in normalized
        or _TEXT_PLACEHOLDER_WORD_RE.search(normalized) is not None
    )


def _path_looks_like_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "":
        return False
    return (
        _contains_strong_placeholder_marker(normalized)
        or "<" in normalized
        or ">" in normalized
        or _PATH_PLACEHOLDER_WORD_RE.search(normalized) is not None
    )


def _contains_strong_placeholder_marker(normalized: str) -> bool:
    return any(marker in normalized for marker in _STRONG_PLACEHOLDER_MARKERS)


def _looks_like_report_placeholder_detail(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "":
        return False
    return (
        _contains_strong_placeholder_marker(normalized)
        or _TEXT_PLACEHOLDER_WORD_RE.search(normalized) is not None
    )


def _coerce_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _require_live_operator_approval_path(
    settings: PMSSettings,
    *,
    allow_pending_operator_approval: bool = False,
) -> None:
    raw_path = settings.polymarket.first_live_order_approval_path
    if raw_path is None or raw_path.strip() == "":
        msg = (
            "LIVE operator approval path missing: "
            "polymarket.first_live_order_approval_path"
        )
        raise LiveTradingDisabledError(msg)
    if _path_looks_like_placeholder(raw_path):
        msg = "LIVE operator approval path contains placeholder"
        raise LiveTradingDisabledError(msg)
    path = Path(raw_path).expanduser()
    _require_live_path_outside_working_tree(
        path,
        label="LIVE operator approval path",
    )
    _require_live_path_parent_owner_writable(
        path,
        label="LIVE operator approval parent directory",
    )
    try:
        approval_path_stat = path.lstat()
    except FileNotFoundError:
        if not allow_pending_operator_approval:
            _require_no_live_operator_approval_sidecar(path)
        return
    if not stat.S_ISREG(approval_path_stat.st_mode):
        msg = f"LIVE operator approval path is not a regular file: {path}"
        raise LiveTradingDisabledError(msg)
    if approval_path_stat.st_nlink != 1:
        msg = f"LIVE operator approval path is not a single-link file: {path}"
        raise LiveTradingDisabledError(msg)
    if allow_pending_operator_approval:
        _require_pending_live_operator_approval_sidecar_safe(path)
        return
    msg = (
        f"LIVE stale approval file already exists at {path}; remove it "
        "before startup and create approval only after preview review"
    )
    raise LiveTradingDisabledError(msg)


def _require_pending_live_operator_approval_sidecar_safe(path: Path) -> None:
    sidecar_path = Path(str(path) + ".meta.json")
    try:
        sidecar_stat = sidecar_path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(sidecar_stat.st_mode):
        msg = f"LIVE operator approval sidecar path is not a regular file: {sidecar_path}"
        raise LiveTradingDisabledError(msg)
    if sidecar_stat.st_nlink != 1:
        msg = (
            "LIVE operator approval sidecar path is not a single-link file: "
            f"{sidecar_path}"
        )
        raise LiveTradingDisabledError(msg)


def _require_no_live_operator_approval_sidecar(path: Path) -> None:
    sidecar_path = Path(str(path) + ".meta.json")
    try:
        sidecar_stat = sidecar_path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(sidecar_stat.st_mode):
        msg = f"LIVE operator approval sidecar path is not a regular file: {sidecar_path}"
        raise LiveTradingDisabledError(msg)
    if sidecar_stat.st_nlink != 1:
        msg = (
            "LIVE operator approval sidecar path is not a single-link file: "
            f"{sidecar_path}"
        )
        raise LiveTradingDisabledError(msg)
    msg = (
        f"LIVE stale approval sidecar already exists at {sidecar_path}; remove it "
        "before startup and create approval only after preview review"
    )
    raise LiveTradingDisabledError(msg)


def _require_distinct_live_launch_control_paths(settings: PMSSettings) -> None:
    approval_path = settings.polymarket.first_live_order_approval_path
    paths: list[tuple[str, Path]] = []

    def add_path(label: str, raw_path: str | None) -> None:
        if raw_path is None or raw_path.strip() == "":
            return
        paths.append(
            (label, _absolute_path_without_symlink_resolution(Path(raw_path)))
        )

    add_path("operator approval path", approval_path)
    if approval_path is not None and approval_path.strip() != "":
        add_path("operator approval sidecar path", f"{approval_path}.meta.json")
    add_path("first-order audit path", settings.live_first_order_audit_path)
    add_path("emergency audit path", settings.live_emergency_audit_path)
    add_path("preflight artifact path", settings.live_preflight_artifact_path)
    add_path("local secret file", settings.local_secret_file)

    for index, (left_label, left_path) in enumerate(paths):
        for right_label, right_path in paths[index + 1 :]:
            if left_path != right_path:
                continue
            msg = (
                f"LIVE {left_label} must be distinct from "
                f"{right_label}: {left_path}"
            )
            raise LiveTradingDisabledError(msg)


def _require_live_config_file_distinct_from_protected_paths(
    settings: PMSSettings,
    *,
    config_path: Path,
) -> None:
    if settings.mode != RunMode.LIVE:
        return

    config_file_path = _absolute_path_without_symlink_resolution(config_path)
    approval_path = settings.polymarket.first_live_order_approval_path
    paths: list[tuple[str, str | None]] = [
        ("operator approval path", approval_path),
        ("first-order audit path", settings.live_first_order_audit_path),
        ("emergency audit path", settings.live_emergency_audit_path),
        ("preflight artifact path", settings.live_preflight_artifact_path),
        ("local secret file", settings.local_secret_file),
        ("paper soak GO report", settings.live_paper_soak_report_path),
        (
            "operator rehearsal report",
            settings.live_operator_rehearsal_report_path,
        ),
        ("execution-model artifact", settings.live_execution_model_path),
        (
            "paper-vs-backtest execution diff artifact",
            settings.live_paper_backtest_diff_path,
        ),
        (
            "category-prior artifact",
            settings.controller.category_prior_observations_path,
        ),
        ("FLB calibration artifact", settings.strategies.flb_calibration_path),
        ("discord alert directory", settings.discord.alert_dir),
    ]
    if approval_path is not None and approval_path.strip() != "":
        paths.append(("operator approval sidecar path", f"{approval_path}.meta.json"))

    for label, raw_path in paths:
        if raw_path is None or raw_path.strip() == "":
            continue
        candidate_path = _absolute_path_without_symlink_resolution(Path(raw_path))
        if not _paths_overlap(candidate_path, config_file_path):
            continue
        msg = f"LIVE config file must be distinct from {label}: {config_file_path}"
        raise ValueError(msg)


def _require_distinct_live_audit_paths(settings: PMSSettings) -> None:
    first_order_path = settings.live_first_order_audit_path
    if first_order_path.strip() == "":
        msg = "LIVE first-order audit path missing: live_first_order_audit_path"
        raise LiveTradingDisabledError(msg)
    if settings.live_emergency_audit_path.strip() == "":
        msg = "LIVE emergency audit path missing: live_emergency_audit_path"
        raise LiveTradingDisabledError(msg)
    if _path_looks_like_placeholder(first_order_path):
        msg = "LIVE first-order audit path contains placeholder"
        raise LiveTradingDisabledError(msg)
    if _path_looks_like_placeholder(settings.live_emergency_audit_path):
        msg = "LIVE emergency audit path contains placeholder"
        raise LiveTradingDisabledError(msg)

    emergency_path = Path(settings.live_emergency_audit_path).expanduser()
    approval_audit_path = Path(first_order_path).expanduser()
    _require_live_path_outside_working_tree(
        approval_audit_path,
        label="LIVE first-order audit path",
    )
    _require_live_path_outside_working_tree(
        emergency_path,
        label="LIVE emergency audit path",
    )
    _require_live_path_parent_owner_writable(
        approval_audit_path,
        label="LIVE first-order audit parent directory",
    )
    _require_live_path_parent_owner_writable(
        emergency_path,
        label="LIVE emergency audit parent directory",
    )
    _require_live_artifact_regular_file_or_absent(
        emergency_path,
        label="LIVE emergency audit path",
    )
    _require_live_artifact_regular_file_or_absent(
        approval_audit_path,
        label="LIVE first-order audit path",
    )
    if emergency_path == approval_audit_path:
        msg = (
            "LIVE first-order audit path must be distinct from "
            "live_emergency_audit_path"
        )
        raise LiveTradingDisabledError(msg)


def _require_live_path_parent_owner_writable(path: Path, *, label: str) -> None:
    parent = path.parent
    try:
        parent_stat = parent.lstat()
    except FileNotFoundError:
        msg = f"{label} does not exist: {parent}"
        raise LiveTradingDisabledError(msg)
    if not stat.S_ISDIR(parent_stat.st_mode):
        msg = f"{label} is not a directory: {parent}"
        raise LiveTradingDisabledError(msg)
    mode = stat.S_IMODE(parent_stat.st_mode)
    if mode & 0o077:
        msg = f"{label} {parent} is too permissive; run `chmod 700 {parent}`."
        raise LiveTradingDisabledError(msg)
    if not mode & stat.S_IWUSR:
        msg = f"{label} is not owner-writable; run `chmod 700 {parent}`."
        raise LiveTradingDisabledError(msg)


def _require_live_path_outside_working_tree(path: Path, *, label: str) -> None:
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
            msg = f"{label} must live outside the working tree: {candidate}"
            raise LiveTradingDisabledError(msg)


def _require_live_artifact_regular_file_or_absent(path: Path, *, label: str) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return
    mode = path_stat.st_mode
    if not stat.S_ISREG(mode):
        msg = f"{label} is not a regular file: {path}"
        raise LiveTradingDisabledError(msg)
    if path_stat.st_nlink != 1:
        msg = f"{label} is not a single-link file: {path}"
        raise LiveTradingDisabledError(msg)


def live_llm_runtime_dependency_requirement(
    settings: PMSSettings,
) -> tuple[str, str] | None:
    if not settings.llm.enabled:
        return None
    if settings.llm.provider == "anthropic":
        return ("anthropic", "uv sync --extra llm")
    if settings.llm.provider == "openai":
        return ("openai", "uv sync --extra llm")
    msg = "LIVE LLM dependency missing: llm.provider is required when LLM is enabled"
    raise LiveTradingDisabledError(msg)


def live_runtime_dependency_requirements(
    settings: PMSSettings,
) -> tuple[tuple[str, str], ...]:
    requirements: list[tuple[str, str]] = [
        _live_polymarket_runtime_dependency_requirement()
    ]
    llm_requirement = live_llm_runtime_dependency_requirement(settings)
    if llm_requirement is not None:
        requirements.append(llm_requirement)
    return tuple(requirements)


def _live_polymarket_runtime_dependency_requirement() -> tuple[str, str]:
    return ("py_clob_client_v2", "uv sync --extra live")


def _require_live_polymarket_runtime_dependency() -> None:
    module_name, install_command = _live_polymarket_runtime_dependency_requirement()
    if importlib.util.find_spec(module_name) is not None:
        return
    msg = (
        f"LIVE Polymarket dependency missing: module {module_name!r} is not "
        f"importable. Install with `{install_command}` before enabling LIVE mode."
    )
    raise LiveTradingDisabledError(msg)


def _require_live_llm_runtime_dependency(settings: PMSSettings) -> None:
    requirement = live_llm_runtime_dependency_requirement(settings)
    if requirement is None:
        return
    module_name, install_command = requirement
    if importlib.util.find_spec(module_name) is not None:
        return
    msg = (
        f"LIVE LLM dependency missing: module {module_name!r} is not importable. "
        f"Install with `{install_command}` before enabling LIVE mode."
    )
    raise LiveTradingDisabledError(msg)


def _require_live_paper_soak_go_report(settings: PMSSettings) -> datetime:
    raw_path = settings.live_paper_soak_report_path
    if raw_path is None or raw_path.strip() == "":
        msg = "LIVE paper soak GO report missing: live_paper_soak_report_path"
        raise LiveTradingDisabledError(msg)
    if _path_looks_like_placeholder(raw_path):
        msg = "LIVE paper soak GO report path contains placeholder"
        raise LiveTradingDisabledError(msg)

    path = Path(raw_path).expanduser()
    _reject_live_test_fixture_report_path(
        path,
        label="LIVE paper soak GO report",
    )
    _require_live_path_outside_working_tree(
        path,
        label="LIVE paper soak GO report",
    )
    _require_live_report_parent_owner_writable(
        path,
        label="LIVE paper soak GO report parent directory",
    )
    _require_live_report_regular_file(path, label="LIVE paper soak GO report")
    try:
        report_text = _read_text_no_follow(path)
    except OSError as exc:
        msg = f"LIVE paper soak GO report is unreadable: {path}"
        raise LiveTradingDisabledError(msg) from exc

    if "## Go/No-Go Gate" not in report_text:
        msg = "LIVE paper soak GO report is missing Go/No-Go Gate section"
        raise LiveTradingDisabledError(msg)
    decision = _markdown_section_decision(report_text, heading="## Go/No-Go Gate")
    if decision != "GO" or "| FAIL |" in report_text:
        msg = (
            "LIVE paper soak GO report must contain a Go/No-Go GO decision "
            "with no failed gate rows"
        )
        raise LiveTradingDisabledError(msg)
    _require_markdown_gate_rows_all_pass(
        report_text,
        heading="## Go/No-Go Gate",
        label="LIVE paper soak GO report",
    )
    _require_markdown_gate_rows_unique(
        report_text,
        heading="## Go/No-Go Gate",
        label="LIVE paper soak GO report",
    )
    _require_live_paper_soak_gate_rows(report_text, risk=settings.risk)
    _require_live_paper_soak_baseline_evidence(report_text)
    report_date = _require_live_paper_soak_report_date_not_future(report_text)
    generated_at = _require_live_paper_soak_persisted_provenance(
        report_text,
        path=path,
        max_age_s=settings.live_readiness_report_max_age_s,
    )
    _require_live_report_generated_at_covers_report_date(
        generated_at=generated_at,
        report_date=report_date,
        label="LIVE paper soak GO report",
    )
    return generated_at


def _require_live_paper_soak_gate_rows(
    report_text: str,
    *,
    risk: RiskSettings,
) -> None:
    rows = _paper_soak_gate_rows(report_text)
    missing_checks = [
        check
        for check in _REQUIRED_LIVE_PAPER_SOAK_GATE_CHECKS
        if check not in rows
    ]
    if missing_checks:
        fields = ", ".join(missing_checks)
        msg = f"LIVE paper soak GO report missing required gate checks: {fields}"
        raise LiveTradingDisabledError(msg)

    non_passing_checks = [
        f"{check}={rows[check][0]}"
        for check in _REQUIRED_LIVE_PAPER_SOAK_GATE_CHECKS
        if rows[check][0] != "PASS"
    ]
    if non_passing_checks:
        fields = ", ".join(non_passing_checks)
        msg = (
            "LIVE paper soak GO report required gate checks must all PASS: "
            f"{fields}"
        )
        raise LiveTradingDisabledError(msg)
    _require_live_paper_soak_gate_details(rows, risk=risk)
    _require_live_paper_soak_strategy_evidence_matches_summary(
        report_text,
        rows=rows,
    )


def _require_live_paper_soak_report_date_not_future(report_text: str) -> date:
    report_date = _dated_report_title_date(
        report_text,
        prefix="# Paper Daily Report - ",
        label="LIVE paper soak GO report",
    )
    if report_date > datetime.now(tz=UTC).date():
        msg = (
            "LIVE paper soak GO report date is in the future: "
            f"{report_date.isoformat()}"
        )
        raise LiveTradingDisabledError(msg)
    return report_date


def _require_live_report_generated_at_covers_report_date(
    *,
    generated_at: datetime,
    report_date: date,
    label: str,
) -> None:
    if generated_at.date() >= report_date:
        return
    msg = (
        f"{label} generated_at predates report date: "
        f"{generated_at.isoformat()} < {report_date.isoformat()}"
    )
    raise LiveTradingDisabledError(msg)


def _dated_report_title_date(
    report_text: str,
    *,
    prefix: str,
    label: str,
) -> date:
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line.startswith(prefix):
            continue
        raw_date = line.removeprefix(prefix).strip()
        try:
            return date.fromisoformat(raw_date)
        except ValueError as exc:
            msg = f"{label} date is invalid"
            raise LiveTradingDisabledError(msg) from exc
    msg = f"{label} missing report date"
    raise LiveTradingDisabledError(msg)


def _paper_soak_gate_statuses(report_text: str) -> dict[str, str]:
    return dict(
        _markdown_gate_status_rows(report_text, heading="## Go/No-Go Gate")
    )


def _paper_soak_gate_rows(report_text: str) -> dict[str, tuple[str, str]]:
    return {
        check_name: (status, detail)
        for check_name, status, detail in _markdown_gate_rows(
            report_text,
            heading="## Go/No-Go Gate",
        )
    }


def _require_live_paper_soak_gate_details(
    rows: dict[str, tuple[str, str]],
    *,
    risk: RiskSettings,
) -> None:
    invalid: list[str] = []
    for check_name in _REQUIRED_LIVE_PAPER_SOAK_GATE_CHECKS:
        detail = rows[check_name][1].strip()
        if detail == "":
            invalid.append(check_name)
        elif _looks_like_report_placeholder_detail(detail):
            invalid.append(f"{check_name} contains placeholder")
    invalid.extend(_live_paper_soak_gate_detail_semantic_errors(rows, risk=risk))
    if invalid:
        fields = ", ".join(invalid)
        msg = f"LIVE paper soak GO report invalid PASS details: {fields}"
        raise LiveTradingDisabledError(msg)


def _live_paper_soak_gate_detail_semantic_errors(
    rows: dict[str, tuple[str, str]],
    *,
    risk: RiskSettings,
) -> list[str]:
    invalid: list[str] = []
    for check_name, minimum, strict in (
        ("soak_days", 30.0, False),
        ("decisions_accepted", 30.0, False),
        ("fills", 50.0, False),
        ("distinct_markets", 3.0, False),
        ("distinct_risk_groups", 3.0, False),
        ("fill_rate", 0.0, True),
        ("cumulative_pnl", 0.0, True),
        ("brier_improvement", 0.0, True),
        ("hit_rate", 0.45, True),
        ("average_edge_bps", 5.0, True),
        ("average_net_edge_bps", 0.0, True),
        ("sharpe_ratio", 0.0, True),
    ):
        value, error = _paper_soak_numeric_gate_detail_value(rows, check_name)
        if error is not None:
            invalid.append(error)
        elif strict and value <= minimum:
            invalid.append(f"{check_name} detail below LIVE threshold")
        elif not strict and value < minimum:
            invalid.append(f"{check_name} detail below LIVE threshold")

    for check_name, maximum in (
        ("average_slippage_bps", 50.0),
        ("max_market_fill_share", 0.60),
        ("max_risk_group_fill_share", 0.60),
        ("brier_score", 0.20),
    ):
        value, error = _paper_soak_numeric_gate_detail_value(rows, check_name)
        if error is not None:
            invalid.append(error)
        elif value > maximum:
            invalid.append(f"{check_name} detail exceeds LIVE threshold")

    if risk.max_daily_loss_usdc is not None:
        value, error = _paper_soak_numeric_gate_detail_value(rows, "todays_pnl")
        if error is not None:
            invalid.append(error)
        elif value < -risk.max_daily_loss_usdc:
            invalid.append("todays_pnl detail below LIVE threshold")

    if risk.max_drawdown_pct is not None:
        value, error = _paper_soak_numeric_gate_detail_value(rows, "max_drawdown_pct")
        if error is not None:
            invalid.append(error)
        elif value > risk.max_drawdown_pct:
            invalid.append("max_drawdown_pct detail exceeds LIVE threshold")

    if risk.max_open_positions is not None:
        value, error = _paper_soak_numeric_gate_detail_value(rows, "open_positions")
        if error is not None:
            invalid.append(error)
        elif value > float(risk.max_open_positions):
            invalid.append("open_positions detail exceeds LIVE threshold")

    value, error = _paper_soak_numeric_gate_detail_value(rows, "total_exposure")
    if error is not None:
        invalid.append(error)
    elif value > risk.max_total_exposure:
        invalid.append("total_exposure detail exceeds LIVE threshold")

    for check_name in ("unresolved_incidents", "risk_events"):
        value, error = _paper_soak_leading_count_detail_value(rows, check_name)
        if error is not None:
            invalid.append(error)
        elif value != 0:
            invalid.append(f"{check_name} detail must be zero")

    return invalid


def _paper_soak_numeric_gate_detail_value(
    rows: dict[str, tuple[str, str]],
    check_name: str,
) -> tuple[float, str | None]:
    detail = rows[check_name][1].strip()
    parts = detail.split()
    if len(parts) != 3:
        return 0.0, f"{check_name} detail is not numeric gate evidence"
    raw_value, operator, raw_threshold = parts
    try:
        value = float(raw_value)
        threshold = float(raw_threshold)
    except ValueError:
        return 0.0, f"{check_name} detail is not numeric gate evidence"
    if not math.isfinite(value) or not math.isfinite(threshold):
        return 0.0, f"{check_name} detail is not finite gate evidence"
    if not _numeric_gate_relation_holds(value, operator, threshold):
        return 0.0, f"{check_name} detail contradicts PASS"
    return value, None


def _paper_soak_leading_count_detail_value(
    rows: dict[str, tuple[str, str]],
    check_name: str,
) -> tuple[int, str | None]:
    detail = rows[check_name][1].strip()
    parts = detail.split()
    if not parts:
        return 0, f"{check_name} detail is missing count evidence"
    try:
        value = int(parts[0])
    except ValueError:
        return 0, f"{check_name} detail is not count evidence"
    return value, None


def _numeric_gate_relation_holds(
    value: float,
    operator: str,
    threshold: float,
) -> bool:
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    return False


def _require_live_paper_soak_strategy_evidence_matches_summary(
    report_text: str,
    *,
    rows: dict[str, tuple[str, str]],
) -> None:
    summary_strategy = _markdown_summary_table_value(
        report_text,
        field_name="Strategy",
        label="LIVE paper soak GO report",
    )
    strategy_evidence = rows["strategy_evidence"][1].strip()
    evidence_labels = _strategy_label_set(strategy_evidence)
    summary_labels = _strategy_label_set(summary_strategy)
    if evidence_labels != summary_labels:
        msg = (
            "LIVE paper soak GO report strategy_evidence must match "
            "Summary Strategy"
        )
        raise LiveTradingDisabledError(msg)

    invalid_labels = sorted(
        label
        for label in summary_labels
        if (
            label.lower() == "unknown"
            or "@" not in label
            or _looks_like_report_placeholder_detail(label)
        )
    )
    if invalid_labels:
        fields = ", ".join(invalid_labels)
        msg = (
            "LIVE paper soak GO report strategy_evidence must contain "
            f"concrete strategy_id@strategy_version_id: {fields}"
        )
        raise LiveTradingDisabledError(msg)

    paper_only_labels = sorted(
        label for label in summary_labels if _is_paper_only_strategy_label(label)
    )
    if paper_only_labels:
        fields = ", ".join(paper_only_labels)
        msg = (
            "LIVE paper soak GO report paper-only strategy cannot be "
            f"final GO evidence: {fields}"
        )
        raise LiveTradingDisabledError(msg)


def _is_paper_only_strategy_label(label: str) -> bool:
    strategy_id, _, _strategy_version_id = label.partition("@")
    return strategy_id.strip().lower() in _PAPER_ONLY_STRATEGY_IDS


def _require_live_paper_soak_baseline_evidence(report_text: str) -> None:
    coverage_by_source = _require_live_paper_soak_baseline_coverage(report_text)
    _require_live_paper_soak_secondary_brier(
        report_text,
        coverage_by_source=coverage_by_source,
    )


def _require_live_paper_soak_baseline_coverage(
    report_text: str,
) -> dict[str, tuple[int, int]]:
    rows = _markdown_table_rows(
        report_text,
        heading="## Baseline Evidence Coverage",
        label="LIVE paper soak GO report",
    )
    if not rows:
        msg = (
            "LIVE paper soak GO report missing Baseline Evidence Coverage "
            "section"
        )
        raise LiveTradingDisabledError(msg)
    _require_valid_paper_soak_baseline_sources(rows)
    _require_unique_paper_soak_baseline_rows(
        rows,
        label="duplicate baseline coverage rows",
    )
    rows_by_source = {row[0]: row for row in rows}
    missing_sources = [
        source
        for source in _REQUIRED_LIVE_PAPER_SOAK_BASELINE_SOURCES
        if source not in rows_by_source
    ]
    if missing_sources:
        fields = ", ".join(missing_sources)
        msg = (
            "LIVE paper soak GO report missing baseline coverage rows: "
            f"{fields}"
        )
        raise LiveTradingDisabledError(msg)
    coverage_by_source: dict[str, tuple[int, int]] = {}
    invalid: list[str] = []
    for source, row in rows_by_source.items():
        _source, decisions, coverage = row
        covered, total = _paper_soak_coverage_value(decisions, source=source)
        coverage_by_source[source] = (covered, total)
        if total <= 0:
            invalid.append(f"{source} coverage has no decision evidence")
        elif not _paper_soak_coverage_percentage_matches_counts(
            coverage,
            covered=covered,
            total=total,
        ):
            invalid.append(
                f"{source} coverage percentage contradicts count evidence"
            )
    required_totals = {
        coverage_by_source[source][1]
        for source in _REQUIRED_LIVE_PAPER_SOAK_BASELINE_SOURCES
        if source in coverage_by_source and coverage_by_source[source][1] > 0
    }
    reported_decision_total = (
        next(iter(required_totals)) if len(required_totals) == 1 else None
    )
    if len(required_totals) > 1:
        invalid.append("required baseline coverage totals disagree")
    if reported_decision_total is not None:
        for source, (_covered, total) in coverage_by_source.items():
            if total != reported_decision_total:
                invalid.append(
                    f"{source} coverage total differs from reported decision set"
                )
    for source in _REQUIRED_LIVE_PAPER_SOAK_BASELINE_SOURCES:
        _source, decisions, coverage = rows_by_source[source]
        covered, total = coverage_by_source[source]
        if covered != total:
            invalid.append(f"{source} coverage is incomplete")
    if invalid:
        fields = ", ".join(invalid)
        msg = f"LIVE paper soak GO report invalid baseline coverage: {fields}"
        raise LiveTradingDisabledError(msg)
    return coverage_by_source


def _require_unique_paper_soak_baseline_rows(
    rows: list[tuple[str, str, str]],
    *,
    label: str,
) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for source, _value, _detail in rows:
        if source in seen and source not in duplicates:
            duplicates.append(source)
        seen.add(source)
    if duplicates:
        fields = ", ".join(duplicates)
        msg = f"LIVE paper soak GO report {label}: {fields}"
        raise LiveTradingDisabledError(msg)


def _require_valid_paper_soak_baseline_sources(
    rows: list[tuple[str, str, str]],
) -> None:
    invalid: list[str] = []
    for source, _value, _detail in rows:
        if source.strip() == "":
            invalid.append("unnamed")
        elif _looks_like_placeholder(source):
            invalid.append(source)
        elif not _is_paper_soak_baseline_source_label(source):
            invalid.append(source)
    if invalid:
        fields = ", ".join(invalid)
        msg = f"LIVE paper soak GO report invalid baseline source label: {fields}"
        raise LiveTradingDisabledError(msg)


def _is_paper_soak_baseline_source_label(value: str) -> bool:
    if value[0] < "a" or value[0] > "z" or value[-1] == "_":
        return False
    return all(
        character == "_" or "a" <= character <= "z" or "0" <= character <= "9"
        for character in value
    )


def _paper_soak_coverage_value(value: str, *, source: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split("/")]
    if len(parts) != 2:
        msg = (
            "LIVE paper soak GO report invalid baseline coverage: "
            f"{source} coverage is not count evidence"
        )
        raise LiveTradingDisabledError(msg)
    try:
        covered = int(parts[0])
        total = int(parts[1])
    except ValueError as exc:
        msg = (
            "LIVE paper soak GO report invalid baseline coverage: "
            f"{source} coverage is not count evidence"
        )
        raise LiveTradingDisabledError(msg) from exc
    if covered < 0 or total < 0 or covered > total:
        msg = (
            "LIVE paper soak GO report invalid baseline coverage: "
            f"{source} coverage count is impossible"
        )
        raise LiveTradingDisabledError(msg)
    return covered, total


def _paper_soak_coverage_percentage_matches_counts(
    value: str,
    *,
    covered: int,
    total: int,
) -> bool:
    if not value.endswith("%"):
        return False
    try:
        percent = float(value[:-1].strip())
    except ValueError:
        return False
    if not math.isfinite(percent):
        return False
    expected = (covered / total) * 100.0
    return math.isclose(percent, expected, rel_tol=0.0, abs_tol=0.05)


def _require_live_paper_soak_secondary_brier(
    report_text: str,
    *,
    coverage_by_source: dict[str, tuple[int, int]],
) -> None:
    rows = _markdown_table_rows(
        report_text,
        heading="## Secondary Baseline Brier",
        label="LIVE paper soak GO report",
    )
    if not rows:
        msg = "LIVE paper soak GO report missing Secondary Baseline Brier section"
        raise LiveTradingDisabledError(msg)
    _require_valid_paper_soak_baseline_sources(rows)
    _require_unique_paper_soak_baseline_rows(
        rows,
        label="duplicate secondary baseline rows",
    )
    rows_by_source = {row[0]: row for row in rows}
    missing_sources = [
        source
        for source in _REQUIRED_LIVE_PAPER_SOAK_BASELINE_SOURCES
        if source not in rows_by_source
    ]
    if missing_sources:
        fields = ", ".join(missing_sources)
        msg = (
            "LIVE paper soak GO report missing secondary baseline rows: "
            f"{fields}"
        )
        raise LiveTradingDisabledError(msg)
    missing_scored_sources = [
        source
        for source, (covered, _total) in coverage_by_source.items()
        if covered > 0 and source not in rows_by_source
    ]
    if missing_scored_sources:
        fields = ", ".join(missing_scored_sources)
        msg = (
            "LIVE paper soak GO report coverage evidence without "
            f"secondary baseline rows: {fields}"
        )
        raise LiveTradingDisabledError(msg)
    missing_coverage_sources = [
        source for source in rows_by_source if source not in coverage_by_source
    ]
    if missing_coverage_sources:
        fields = ", ".join(missing_coverage_sources)
        msg = (
            "LIVE paper soak GO report secondary baseline rows without "
            f"coverage evidence: {fields}"
        )
        raise LiveTradingDisabledError(msg)
    invalid: list[str] = []
    for source, row in rows_by_source.items():
        _source, baseline_brier, improvement = row
        covered, _total = coverage_by_source[source]
        if covered <= 0:
            invalid.append(f"{source} coverage has no decision evidence")
        error = _paper_soak_secondary_brier_row_error(
            baseline_brier,
            improvement,
        )
        if error is not None:
            invalid.append(f"{source} {error}")
    if invalid:
        fields = ", ".join(invalid)
        msg = f"LIVE paper soak GO report invalid secondary baseline Brier: {fields}"
        raise LiveTradingDisabledError(msg)


def _paper_soak_secondary_brier_row_error(
    baseline_brier: str,
    improvement: str,
) -> str | None:
    try:
        baseline_value = float(baseline_brier)
        improvement_value = float(improvement)
    except ValueError:
        return "baseline Brier or improvement is not numeric"
    if not math.isfinite(baseline_value):
        return "baseline Brier is not finite"
    if not math.isfinite(improvement_value):
        return "improvement is not finite"
    if baseline_value < 0.0 or baseline_value > 1.0:
        return "baseline Brier must be between 0 and 1"
    if improvement_value <= 0.0:
        return "improvement must be positive"
    if improvement_value > baseline_value:
        return "improvement exceeds baseline Brier"
    return None


def _strategy_label_set(value: str) -> set[str]:
    return {label.strip() for label in value.split(",") if label.strip() != ""}


def _markdown_summary_table_value(
    report_text: str,
    *,
    field_name: str,
    label: str,
) -> str:
    in_summary = False
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            in_summary = line == "## Summary"
            continue
        if not in_summary or not line.startswith("|"):
            continue
        cells = _markdown_table_cells(line)
        if len(cells) < 2 or cells[0] != field_name:
            continue
        if len(cells) != 3:
            msg = f"{label} malformed Summary {field_name} row"
            raise LiveTradingDisabledError(msg)
        value = cells[1].strip()
        if value != "":
            return value
        break
    msg = f"{label} missing Summary {field_name} row"
    raise LiveTradingDisabledError(msg)


def _markdown_gate_status_rows(
    report_text: str,
    *,
    heading: str,
) -> list[tuple[str, str]]:
    return [
        (check_name, status)
        for check_name, status, _detail in _markdown_gate_rows(
            report_text,
            heading=heading,
        )
    ]


def _markdown_gate_rows(
    report_text: str,
    *,
    heading: str,
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    in_gate = False
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            in_gate = line == heading
            continue
        if not in_gate or not line.startswith("|"):
            continue

        cells = _markdown_table_cells(line)
        if len(cells) < 2:
            continue
        check_name = cells[0]
        if check_name in {"Check", "---"}:
            continue
        if len(cells) != 3:
            msg = f"{heading} malformed gate row: {check_name or 'unnamed'}"
            raise LiveTradingDisabledError(msg)
        status = cells[1].upper()
        detail = cells[2] if len(cells) >= 3 else ""
        rows.append((check_name, status, detail))
    return rows


def _markdown_table_rows(
    report_text: str,
    *,
    heading: str,
    label: str,
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    in_section = False
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            in_section = line == heading
            continue
        if not in_section or not line.startswith("|"):
            continue

        cells = _markdown_table_cells(line)
        if len(cells) < 2:
            continue
        first_cell = cells[0]
        if first_cell in {"Baseline", "---"}:
            continue
        if len(cells) != 3:
            msg = f"{label} malformed {heading} row: {first_cell or 'unnamed'}"
            raise LiveTradingDisabledError(msg)
        rows.append((first_cell, cells[1], cells[2]))
    return rows


def _require_markdown_gate_rows_all_pass(
    report_text: str,
    *,
    heading: str,
    label: str,
) -> None:
    non_passing_rows = [
        f"{check_name}={status or 'missing'}"
        for check_name, status in _markdown_gate_status_rows(
            report_text,
            heading=heading,
        )
        if status != "PASS"
    ]
    if non_passing_rows:
        fields = ", ".join(non_passing_rows)
        msg = f"{label} contains non-PASS gate rows: {fields}"
        raise LiveTradingDisabledError(msg)


def _require_markdown_gate_rows_unique(
    report_text: str,
    *,
    heading: str,
    label: str,
) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for check_name, _status, _detail in _markdown_gate_rows(
        report_text,
        heading=heading,
    ):
        if check_name in seen and check_name not in duplicates:
            duplicates.append(check_name)
        seen.add(check_name)
    if duplicates:
        fields = ", ".join(duplicates)
        msg = f"{label} duplicate gate rows: {fields}"
        raise LiveTradingDisabledError(msg)


def _require_live_paper_soak_persisted_provenance(
    report_text: str,
    *,
    path: Path,
    max_age_s: float,
) -> datetime:
    provenance = _markdown_report_provenance(report_text)
    if not provenance:
        msg = "LIVE paper soak GO report missing persisted report provenance"
        raise LiveTradingDisabledError(msg)

    generated_by = provenance.get("generated_by")
    if generated_by != _LIVE_PAPER_REPORT_GENERATOR:
        msg = (
            "LIVE paper soak GO report must be generated by "
            f"{_LIVE_PAPER_REPORT_GENERATOR}"
        )
        raise LiveTradingDisabledError(msg)

    generated_at = _require_report_generated_at(
        provenance,
        label="LIVE paper soak GO report",
    )
    _require_report_fresh(
        generated_at,
        label="LIVE paper soak GO report",
        max_age_s=max_age_s,
    )

    artifact_mode = provenance.get("artifact_mode")
    if artifact_mode != "persisted":
        msg = (
            "LIVE paper soak GO report must be a persisted report artifact; "
            f"artifact_mode={artifact_mode or 'missing'}"
        )
        raise LiveTradingDisabledError(msg)

    output_path = provenance.get("output_path", "").strip()
    if output_path == "" or output_path == "stdout":
        msg = "LIVE paper soak GO report persisted provenance missing output_path"
        raise LiveTradingDisabledError(msg)
    _require_report_output_path_matches(
        path=path,
        output_path=output_path,
        label="LIVE paper soak GO report",
    )
    _require_paper_report_input_snapshot_sha256(provenance)
    return generated_at


def _require_paper_report_input_snapshot_sha256(provenance: dict[str, str]) -> None:
    digest = provenance.get("input_snapshot_sha256", "").strip()
    if digest == "":
        msg = (
            "LIVE paper soak GO report persisted provenance missing "
            "input_snapshot_sha256"
        )
        raise LiveTradingDisabledError(msg)
    if _SHA256_HEX_PATTERN.fullmatch(digest) is None:
        msg = (
            "LIVE paper soak GO report persisted provenance "
            "input_snapshot_sha256 must be a sha256 hex digest"
        )
        raise LiveTradingDisabledError(msg)
    if digest in _PLACEHOLDER_SHA256_DIGESTS:
        msg = (
            "LIVE paper soak GO report persisted provenance "
            "input_snapshot_sha256 must not be a placeholder hash"
        )
        raise LiveTradingDisabledError(msg)


def _require_report_generated_at(
    provenance: dict[str, str],
    *,
    label: str,
) -> datetime:
    raw_generated_at = provenance.get("generated_at", "").strip()
    if raw_generated_at == "":
        msg = f"{label} persisted provenance missing generated_at"
        raise LiveTradingDisabledError(msg)
    try:
        generated_at = datetime.fromisoformat(
            raw_generated_at.replace("Z", "+00:00")
        )
    except ValueError as exc:
        msg = f"{label} persisted provenance generated_at is invalid"
        raise LiveTradingDisabledError(msg) from exc

    aware_generated_at = _require_aware_datetime(
        generated_at,
        label=label,
        field_name="persisted provenance generated_at",
    )
    if aware_generated_at > datetime.now(tz=UTC):
        msg = f"{label} persisted provenance generated_at is in the future"
        raise LiveTradingDisabledError(msg)
    return aware_generated_at


def _require_aware_datetime(
    value: datetime,
    *,
    label: str,
    field_name: str,
) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        msg = f"{label} {field_name} must include timezone"
        raise LiveTradingDisabledError(msg)
    return value.astimezone(UTC)


def _require_report_fresh(
    generated_at: datetime,
    *,
    label: str,
    max_age_s: float,
) -> None:
    age_s = (datetime.now(tz=UTC) - generated_at).total_seconds()
    if age_s <= max_age_s:
        return

    msg = f"{label} is stale: age {age_s:.1f}s exceeds {max_age_s:.1f}s"
    raise LiveTradingDisabledError(msg)


def _markdown_report_provenance(report_text: str) -> dict[str, str]:
    provenance: dict[str, str] = {}
    seen_fields: set[str] = set()
    in_provenance = False
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            in_provenance = line == "## Report Provenance"
            continue
        if not in_provenance or not line.startswith("|"):
            continue

        cells = _markdown_table_cells(line)
        if len(cells) < 2:
            continue
        field_name = cells[0]
        value = cells[1]
        if field_name in {"Field", "---"}:
            continue
        if len(cells) != 2:
            msg = f"LIVE report malformed provenance row: {field_name or 'unnamed'}"
            raise LiveTradingDisabledError(msg)
        if field_name in seen_fields:
            msg = f"LIVE report duplicate provenance field: {field_name}"
            raise LiveTradingDisabledError(msg)
        seen_fields.add(field_name)
        provenance[field_name] = value
    return provenance


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


def _require_live_operator_rehearsal_report(settings: PMSSettings) -> datetime:
    raw_path = settings.live_operator_rehearsal_report_path
    if raw_path is None or raw_path.strip() == "":
        msg = (
            "LIVE operator rehearsal report missing: "
            "live_operator_rehearsal_report_path"
        )
        raise LiveTradingDisabledError(msg)
    if _path_looks_like_placeholder(raw_path):
        msg = "LIVE operator rehearsal report path contains placeholder"
        raise LiveTradingDisabledError(msg)

    path = Path(raw_path).expanduser()
    _reject_live_test_fixture_report_path(
        path,
        label="LIVE operator rehearsal report",
    )
    _require_live_path_outside_working_tree(
        path,
        label="LIVE operator rehearsal report",
    )
    _require_live_report_parent_owner_writable(
        path,
        label="LIVE operator rehearsal report parent directory",
    )
    _require_live_report_regular_file(path, label="LIVE operator rehearsal report")
    try:
        report_text = _read_text_no_follow(path)
    except OSError as exc:
        msg = f"LIVE operator rehearsal report is unreadable: {path}"
        raise LiveTradingDisabledError(msg) from exc

    if "## Operator Approval Rehearsal" not in report_text:
        msg = "LIVE operator rehearsal report is missing rehearsal section"
        raise LiveTradingDisabledError(msg)
    decision = _markdown_section_decision(
        report_text,
        heading="## Operator Approval Rehearsal",
    )
    if decision != "PASS" or "| FAIL |" in report_text:
        msg = (
            "LIVE operator rehearsal report must contain a PASS decision "
            "with no failed gate rows"
        )
        raise LiveTradingDisabledError(msg)
    _require_markdown_gate_rows_all_pass(
        report_text,
        heading="## Operator Approval Rehearsal",
        label="LIVE operator rehearsal report",
    )
    _require_markdown_gate_rows_unique(
        report_text,
        heading="## Operator Approval Rehearsal",
        label="LIVE operator rehearsal report",
    )

    rows = _operator_rehearsal_gate_rows(report_text)
    missing_checks = [
        check
        for check in _REQUIRED_LIVE_OPERATOR_REHEARSAL_GATE_CHECKS
        if check not in rows
    ]
    if missing_checks:
        fields = ", ".join(missing_checks)
        msg = f"LIVE operator rehearsal report missing PASS checks: {fields}"
        raise LiveTradingDisabledError(msg)
    failed_checks = [
        f"{check}={rows[check][0]}"
        for check in _REQUIRED_LIVE_OPERATOR_REHEARSAL_GATE_CHECKS
        if rows[check][0] != "PASS"
    ]
    if failed_checks:
        fields = ", ".join(failed_checks)
        msg = f"LIVE operator rehearsal report required checks must PASS: {fields}"
        raise LiveTradingDisabledError(msg)
    _require_live_operator_rehearsal_gate_details(rows)
    report_date = _require_live_operator_rehearsal_report_date_not_future(
        report_text
    )
    generated_at = _require_live_operator_rehearsal_persisted_provenance(
        report_text,
        path=path,
        max_age_s=settings.live_readiness_report_max_age_s,
    )
    _require_live_report_generated_at_covers_report_date(
        generated_at=generated_at,
        report_date=report_date,
        label="LIVE operator rehearsal report",
    )
    return generated_at


def _operator_rehearsal_statuses(report_text: str) -> dict[str, str]:
    return dict(
        _markdown_gate_status_rows(
            report_text,
            heading="## Operator Approval Rehearsal",
        )
    )


def _operator_rehearsal_gate_rows(
    report_text: str,
) -> dict[str, tuple[str, str]]:
    return {
        check_name: (status, detail)
        for check_name, status, detail in _markdown_gate_rows(
            report_text,
            heading="## Operator Approval Rehearsal",
        )
    }


def _require_live_operator_rehearsal_gate_details(
    rows: dict[str, tuple[str, str]],
) -> None:
    invalid: list[str] = []
    for check_name in _REQUIRED_LIVE_OPERATOR_REHEARSAL_GATE_CHECKS:
        detail = rows[check_name][1].strip()
        if detail == "":
            invalid.append(f"{check_name}=missing detail")
        elif _looks_like_report_placeholder_detail(detail):
            invalid.append(f"{check_name} contains placeholder")

    operator_id = rows["operator_id"][1].strip()
    if operator_id != "" and _looks_like_placeholder(operator_id):
        invalid.append("operator_id contains placeholder")

    invalid.extend(_operator_rehearsal_required_detail_errors(rows))

    unexpected_events_detail = rows["unexpected_events"][1].strip()
    expected_unexpected_events_detail = (
        f"events={list(_EXPECTED_LIVE_OPERATOR_REHEARSAL_EVENTS)}"
    )
    if unexpected_events_detail != expected_unexpected_events_detail:
        invalid.append("unexpected_events does not match expected audit sequence")

    if invalid:
        fields = ", ".join(invalid)
        msg = f"LIVE operator rehearsal report invalid PASS details: {fields}"
        raise LiveTradingDisabledError(msg)


def _operator_rehearsal_required_detail_errors(
    rows: dict[str, tuple[str, str]],
) -> list[str]:
    invalid: list[str] = []
    detail = rows["approval_denied"][1].lower()
    if "before approval file existed" not in detail:
        invalid.append(
            "approval_denied detail does not prove gate denial before approval file"
        )

    detail = rows["approval_matched"][1].lower()
    if "approval json matched preview" not in detail:
        invalid.append("approval_matched detail does not prove preview match")

    detail = rows["approval_consumed"][1].lower()
    if "approval json and sidecar were unlinked" not in detail:
        invalid.append("approval_consumed detail does not prove cleanup")

    detail = rows["strict_sidecar_provenance"][1].lower()
    required_sidecar_terms = ("sidecar", "approver_id", "timestamp", "approval hash")
    if any(term not in detail for term in required_sidecar_terms):
        invalid.append(
            "strict_sidecar_provenance detail does not prove sidecar provenance"
        )

    detail = rows["fresh_approval_required"][1].lower()
    required_fresh_terms = ("every-order", "denied", "approval consume")
    if any(term not in detail for term in required_fresh_terms):
        invalid.append(
            "fresh_approval_required detail does not prove approval reuse denial"
        )
    return invalid


def _require_live_operator_rehearsal_report_date_not_future(
    report_text: str,
) -> date:
    report_date = _dated_report_title_date(
        report_text,
        prefix="# Operator Approval Rehearsal - ",
        label="LIVE operator rehearsal report",
    )
    if report_date > datetime.now(tz=UTC).date():
        msg = (
            "LIVE operator rehearsal report date is in the future: "
            f"{report_date.isoformat()}"
        )
        raise LiveTradingDisabledError(msg)
    return report_date


def _markdown_section_decision(report_text: str, *, heading: str) -> str | None:
    in_section = False
    decisions: list[str] = []
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            in_section = line == heading
            continue
        if not in_section or not line.startswith("**Decision:**"):
            continue
        decisions.append(line.removeprefix("**Decision:**").strip().strip("*").upper())
    if len(decisions) > 1:
        msg = f"{heading} contains multiple decision rows"
        raise LiveTradingDisabledError(msg)
    if not decisions:
        return None
    return decisions[0]


def _require_live_operator_rehearsal_persisted_provenance(
    report_text: str,
    *,
    path: Path,
    max_age_s: float,
) -> datetime:
    provenance = _markdown_report_provenance(report_text)
    if not provenance:
        msg = "LIVE operator rehearsal report missing persisted report provenance"
        raise LiveTradingDisabledError(msg)

    generated_by = provenance.get("generated_by")
    if generated_by != _LIVE_OPERATOR_REHEARSAL_REPORT_GENERATOR:
        msg = (
            "LIVE operator rehearsal report must be generated by "
            f"{_LIVE_OPERATOR_REHEARSAL_REPORT_GENERATOR}"
        )
        raise LiveTradingDisabledError(msg)

    generated_at = _require_report_generated_at(
        provenance,
        label="LIVE operator rehearsal report",
    )
    _require_report_fresh(
        generated_at,
        label="LIVE operator rehearsal report",
        max_age_s=max_age_s,
    )

    artifact_mode = provenance.get("artifact_mode")
    if artifact_mode != "persisted":
        msg = (
            "LIVE operator rehearsal report must be a persisted report artifact; "
            f"artifact_mode={artifact_mode or 'missing'}"
        )
        raise LiveTradingDisabledError(msg)

    output_path = provenance.get("output_path", "").strip()
    if output_path == "" or output_path == "stdout":
        msg = "LIVE operator rehearsal report persisted provenance missing output_path"
        raise LiveTradingDisabledError(msg)
    _require_report_output_path_matches(
        path=path,
        output_path=output_path,
        label="LIVE operator rehearsal report",
    )
    return generated_at


def _require_report_output_path_matches(
    *,
    path: Path,
    output_path: str,
    label: str,
) -> None:
    expected = path.expanduser().resolve(strict=False)
    observed = Path(output_path).expanduser().resolve(strict=False)
    if observed != expected:
        msg = (
            f"{label} provenance output_path must match configured path: "
            f"{observed} != {expected}"
        )
        raise LiveTradingDisabledError(msg)


def _require_live_report_parent_owner_writable(path: Path, *, label: str) -> None:
    parent = path.parent
    try:
        parent_stat = parent.lstat()
    except FileNotFoundError:
        msg = f"{label} does not exist: {parent}"
        raise LiveTradingDisabledError(msg)
    if not stat.S_ISDIR(parent_stat.st_mode):
        msg = f"{label} is not a directory: {parent}"
        raise LiveTradingDisabledError(msg)
    mode = stat.S_IMODE(parent_stat.st_mode)
    if mode & 0o077:
        msg = f"{label} {parent} is too permissive; run `chmod 700 {parent}`."
        raise LiveTradingDisabledError(msg)
    if not mode & stat.S_IWUSR:
        msg = f"{label} is not owner-writable; run `chmod 700 {parent}`."
        raise LiveTradingDisabledError(msg)


def _require_live_report_regular_file(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        msg = f"{label} does not exist: {path}"
        raise LiveTradingDisabledError(msg) from exc
    if not stat.S_ISREG(mode):
        msg = f"{label} is not a regular file: {path}"
        raise LiveTradingDisabledError(msg)
    if path.lstat().st_nlink != 1:
        msg = f"{label} is not a single-link file: {path}"
        raise LiveTradingDisabledError(msg)


def _reject_live_test_fixture_report_path(path: Path, *, label: str) -> None:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "tests" and parts[index + 1] == "fixtures":
            msg = f"{label} must not use test fixture path: {path}"
            raise LiveTradingDisabledError(msg)


def _missing_polymarket_fields(credentials: VenueCredentials) -> list[str]:
    missing: list[str] = []
    required_text_fields = {
        "private_key": credentials.private_key,
        "api_key": credentials.api_key,
        "api_secret": credentials.api_secret,
        "api_passphrase": credentials.api_passphrase,
        "funder_address": credentials.funder_address,
    }
    for field_name, value in required_text_fields.items():
        if value is None or value.strip() == "":
            missing.append(field_name)
    if credentials.signature_type is None:
        missing.append("signature_type")
    return missing


def _placeholder_polymarket_fields(credentials: VenueCredentials) -> list[str]:
    placeholder_fields: list[str] = []
    required_text_fields = {
        "private_key": credentials.private_key,
        "api_key": credentials.api_key,
        "api_secret": credentials.api_secret,
        "api_passphrase": credentials.api_passphrase,
        "funder_address": credentials.funder_address,
    }
    for field_name, value in required_text_fields.items():
        if value is not None and _looks_like_placeholder(value):
            placeholder_fields.append(field_name)
    return placeholder_fields


def _is_evm_address(value: str | None) -> bool:
    if value is None:
        return False
    candidate = value.strip()
    if len(candidate) != 42 or not candidate.startswith("0x"):
        return False
    return all(character in "0123456789abcdefABCDEF" for character in candidate[2:])


def _reject_inline_polymarket_credentials(config_data: dict[str, Any]) -> None:
    raw_polymarket = config_data.get("polymarket")
    if not isinstance(raw_polymarket, dict):
        return

    inline_fields = sorted(
        field_name
        for field_name in _POLYMARKET_CREDENTIAL_CONFIG_FIELDS
        if field_name in raw_polymarket and raw_polymarket[field_name] is not None
    )
    if not inline_fields:
        return

    fields = ", ".join(inline_fields)
    msg = (
        "Polymarket credential fields must not be set in config files: "
        f"{fields}. Use local_secret_file or the production secret manager."
    )
    raise ValueError(msg)


def _reject_inline_llm_api_key(config_data: dict[str, Any]) -> None:
    raw_llm = config_data.get("llm")
    if not isinstance(raw_llm, dict):
        return
    if "api_key" not in raw_llm:
        return

    msg = (
        "LLM api_key must not be set in config files. "
        "Use PMS_LLM__API_KEY or the production secret manager."
    )
    raise ValueError(msg)


def _merge_local_secret_file(config_data: dict[str, Any]) -> dict[str, Any]:
    secret_source = _configured_string(
        config_data,
        field_name="secret_source",
        env_name="PMS_SECRET_SOURCE",
    )
    if secret_source != "local_file":
        return config_data

    raw_path = _configured_string(
        config_data,
        field_name="local_secret_file",
        env_name="PMS_LOCAL_SECRET_FILE",
    )
    if raw_path is None or raw_path.strip() == "":
        return config_data

    secret_path = Path(raw_path).expanduser()
    local_secrets = _load_local_polymarket_secret_file(secret_path)
    merged = dict(config_data)
    existing_polymarket = merged.get("polymarket") or {}
    if not isinstance(existing_polymarket, dict):
        msg = "Expected polymarket config mapping before applying local secret file"
        raise ValueError(msg)

    merged["secret_source"] = "local_file"
    merged["local_secret_file"] = str(secret_path)
    merged["polymarket"] = {**existing_polymarket, **local_secrets}
    return merged


def _configured_string(
    config_data: dict[str, Any],
    *,
    field_name: str,
    env_name: str,
) -> str | None:
    raw_value = config_data.get(field_name)
    if isinstance(raw_value, str):
        return raw_value
    env_value = os.environ.get(env_name)
    if env_value is None:
        return None
    return env_value


def _load_local_polymarket_secret_file(path: Path) -> dict[str, object]:
    _require_local_secret_file_outside_working_tree(path)
    _require_private_local_secret_file(path)
    _require_private_local_secret_parent(path)
    try:
        loaded = safe_load_yaml_no_duplicate_keys(_read_text_no_follow(path))
    except OSError as exc:
        msg = f"Local secret file is unreadable: {path}"
        raise ValueError(msg) from exc
    except yaml.YAMLError as exc:
        msg = yaml_load_error_message(
            "Local secret file is not valid YAML",
            path,
            exc,
        )
        raise ValueError(msg) from None
    if not isinstance(loaded, dict):
        msg = f"Expected mapping in local secret file {path}"
        raise ValueError(msg)

    raw_polymarket = loaded.get("polymarket")
    if not isinstance(raw_polymarket, dict):
        msg = f"Expected polymarket mapping in local secret file {path}"
        raise ValueError(msg)

    allowed_fields = {
        "private_key",
        "api_key",
        "api_secret",
        "api_passphrase",
        "signature_type",
        "funder_address",
    }
    secrets = {
        field_name: value
        for field_name, value in raw_polymarket.items()
        if field_name in allowed_fields
    }
    invalid_fields = _invalid_local_polymarket_secret_field_types(secrets)
    if invalid_fields:
        fields = ", ".join(invalid_fields)
        msg = (
            "Local secret file contains invalid Polymarket credential field "
            f"types: {fields}"
        )
        raise ValueError(msg)
    placeholder_fields = _placeholder_local_polymarket_secret_fields(secrets)
    if placeholder_fields:
        fields = ", ".join(placeholder_fields)
        msg = (
            "Local secret file contains placeholder Polymarket credential "
            f"fields: {fields}"
        )
        raise ValueError(msg)
    return secrets


def _placeholder_local_polymarket_secret_fields(
    secrets: dict[str, object],
) -> list[str]:
    placeholder_fields: list[str] = []
    for field_name in sorted(_POLYMARKET_TEXT_CREDENTIAL_CONFIG_FIELDS):
        value = secrets.get(field_name)
        if isinstance(value, str) and _looks_like_placeholder(value):
            placeholder_fields.append(field_name)
    return placeholder_fields


def _invalid_local_polymarket_secret_field_types(
    secrets: dict[str, object],
) -> list[str]:
    invalid: list[str] = []
    for field_name in sorted(_POLYMARKET_TEXT_CREDENTIAL_CONFIG_FIELDS):
        value = secrets.get(field_name)
        if value is not None and not isinstance(value, str):
            invalid.append(field_name)

    signature_type = secrets.get("signature_type")
    if (
        signature_type is not None
        and (isinstance(signature_type, bool) or not isinstance(signature_type, int))
    ):
        invalid.append("signature_type")
    return invalid


def _require_private_local_secret_file(path: Path) -> None:
    if not path.exists():
        msg = f"Local secret file does not exist: {path}"
        raise ValueError(msg)
    path_stat = path.lstat()
    path_mode = path_stat.st_mode
    if not stat.S_ISREG(path_mode):
        msg = f"Local secret path is not a regular file: {path}"
        raise ValueError(msg)
    if path_stat.st_nlink != 1:
        msg = f"Local secret path is not a single-link file: {path}"
        raise ValueError(msg)
    mode = stat.S_IMODE(path_mode)
    if mode & 0o077:
        msg = f"Local secret file {path} is too permissive; run `chmod 600 {path}`."
        raise ValueError(msg)


def _require_private_local_secret_parent(path: Path) -> None:
    parent = path.expanduser().parent
    try:
        parent_stat = parent.lstat()
    except FileNotFoundError:
        msg = f"Local secret file parent directory does not exist: {parent}"
        raise ValueError(msg)
    if not stat.S_ISDIR(parent_stat.st_mode):
        msg = f"Local secret file parent path is not a directory: {parent}"
        raise ValueError(msg)
    mode = stat.S_IMODE(parent_stat.st_mode)
    if mode & 0o077:
        msg = (
            f"Local secret file parent directory {parent} is too permissive; "
            f"run `chmod 700 {parent}`."
        )
        raise ValueError(msg)
    if not mode & stat.S_IWUSR:
        msg = (
            f"Local secret file parent directory {parent} is not owner-writable; "
            f"run `chmod 700 {parent}`."
        )
        raise ValueError(msg)


def _require_local_secret_file_outside_working_tree(path: Path) -> None:
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
            msg = f"Local secret file must live outside the working tree: {candidate}"
            raise ValueError(msg)


def _absolute_path_without_symlink_resolution(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(expanded))


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
