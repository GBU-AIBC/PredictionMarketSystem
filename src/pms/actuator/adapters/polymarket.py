from __future__ import annotations

import asyncio
import importlib
import json
import logging
import math
import os
import stat
from hashlib import sha256
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Final, Literal, Protocol, cast
from uuid import uuid4

from pms.config import PMSSettings, validate_live_mode_ready
from pms.core.enums import OrderStatus, RunMode
from pms.core.models import (
    BookLevel,
    BookSnapshot,
    LiveTradingDisabledError,
    Market,
    OrderState,
    Position,
    Portfolio,
    ReconciliationReport,
    TradeDecision,
    VenueAccountSnapshot,
    VenueCredentials,
)
from pms.core.validation import validate_finite, validate_positive, validate_price
from pms.live_preflight_artifact import (
    is_sha256_hexdigest as _is_sha256_hexdigest,
    latest_live_emergency_audit_timestamp,
    live_preflight_readiness_report_generated_at_values,
    live_preflight_readiness_reports_fingerprint,
    live_preflight_settings_fingerprint,
    loads_json_rejecting_duplicate_keys,
    validate_live_strategy_artifacts_for_submission,
)
from pms.redaction import redact_live_error_values


logger = logging.getLogger(__name__)


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


def _read_text_no_follow(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
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


class OperatorApprovalRequiredError(LiveTradingDisabledError):
    """Raised when a required live-order operator approval is absent."""


class PolymarketVenueRejectionError(LiveTradingDisabledError):
    """Raised when the venue returned a definitive order rejection."""


class PolymarketSubmissionUnknownError(RuntimeError):
    """Raised when a Polymarket order submission timed out — the order may
    have reached the venue. Operators must reconcile before retrying.
    """

    order_state: OrderState | None

    def __init__(
        self,
        message: str,
        *,
        order_state: OrderState | None = None,
    ) -> None:
        super().__init__(message)
        self.order_state = order_state


@dataclass(frozen=True)
class LiveOrderPreview:
    max_notional_usdc: float
    venue: str
    market_id: str
    token_id: str | None
    side: str
    limit_price: float
    max_slippage_bps: int
    outcome: str = "YES"
    market_slug: str | None = None
    question: str | None = None


@dataclass(frozen=True)
class PolymarketOrderRequest:
    market_id: str
    token_id: str
    side: str
    price: float
    size: float
    notional_usdc: float
    estimated_quantity: float
    order_type: str
    time_in_force: str
    max_slippage_bps: int


@dataclass(frozen=True)
class PolymarketOrderResult:
    order_id: str
    status: str
    raw_status: str
    filled_notional_usdc: float
    remaining_notional_usdc: float
    fill_price: float | None
    filled_quantity: float
    fees: float | None = None
    fee_bps: int | None = None
    transaction_ref: str | None = None


@dataclass(frozen=True)
class LivePreSubmitQuote:
    market_status: str
    book_age_ms: float
    executable_notional_usdc: float
    best_executable_price: float
    spread_bps: float
    quote_hash: str
    book_ts: datetime
    source: Literal["postgres_snapshot", "venue_direct", "dual"] = "postgres_snapshot"


@dataclass(frozen=True)
class LiveVenueBook:
    market_id: str
    token_id: str
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    book_ts: datetime
    quote_hash: str
    market_status: str = "open"


class PolymarketClient(Protocol):
    async def submit_order(
        self,
        order: PolymarketOrderRequest,
        credentials: VenueCredentials,
    ) -> PolymarketOrderResult: ...


def _client_requires_live_mode(client: PolymarketClient) -> bool:
    return getattr(client, "requires_live_mode", True) is not False


class LiveQuoteProvider(Protocol):
    async def quote(
        self,
        order: PolymarketOrderRequest,
        credentials: VenueCredentials,
    ) -> LivePreSubmitQuote: ...


class LiveOrderBookClient(Protocol):
    async def read_order_book(
        self,
        order: PolymarketOrderRequest,
        credentials: VenueCredentials,
    ) -> LiveVenueBook: ...


class LiveVenueAccountClient(Protocol):
    async def read_account_snapshot(
        self,
        credentials: VenueCredentials,
    ) -> VenueAccountSnapshot: ...


class BookQuoteStore(Protocol):
    async def read_market(self, market_id: str) -> Market | None: ...

    async def read_latest_snapshot(
        self,
        market_id: str,
        token_id: str,
    ) -> BookSnapshot | None: ...

    async def read_levels_for_snapshot(self, snapshot_id: int) -> list[BookLevel]: ...


@dataclass(frozen=True)
class MissingLiveQuoteProvider:
    async def quote(
        self,
        order: PolymarketOrderRequest,
        credentials: VenueCredentials,
    ) -> LivePreSubmitQuote:
        del order, credentials
        msg = (
            "Polymarket pre-submit quote guard is not configured; "
            "LIVE submit requires a fresh venue book check"
        )
        raise LiveTradingDisabledError(msg)


@dataclass(frozen=True)
class PolymarketBookQuoteProvider:
    store: BookQuoteStore
    clock: Callable[[], datetime] = field(default_factory=lambda: _utc_now)
    allowed_clock_skew_ms: float = 250.0

    async def quote(
        self,
        order: PolymarketOrderRequest,
        credentials: VenueCredentials,
    ) -> LivePreSubmitQuote:
        del credentials
        now = self.clock()
        market = await self.store.read_market(order.market_id)
        if market is None:
            msg = f"Polymarket market data missing for {order.market_id}"
            raise LiveTradingDisabledError(msg)
        snapshot = await self.store.read_latest_snapshot(order.market_id, order.token_id)
        if snapshot is None:
            msg = f"Polymarket book snapshot missing for token {order.token_id}"
            raise LiveTradingDisabledError(msg)
        _raise_if_future_book_ts(
            snapshot.ts,
            now=now,
            allowed_clock_skew_ms=self.allowed_clock_skew_ms,
        )
        levels = await self.store.read_levels_for_snapshot(snapshot.id)
        return _quote_from_levels(
            order=order,
            market_status=_market_status(market, now),
            bid_levels=[level for level in levels if level.side == "BUY"],
            ask_levels=[level for level in levels if level.side == "SELL"],
            quote_hash=snapshot.hash or f"snapshot:{snapshot.id}",
            book_ts=snapshot.ts,
            now=now,
            source="postgres_snapshot",
        )


@dataclass(frozen=True)
class PolymarketDirectQuoteProvider:
    book_client: LiveOrderBookClient
    clock: Callable[[], datetime] = field(default_factory=lambda: _utc_now)
    allowed_clock_skew_ms: float = 250.0

    async def quote(
        self,
        order: PolymarketOrderRequest,
        credentials: VenueCredentials,
    ) -> LivePreSubmitQuote:
        now = self.clock()
        book = await self.book_client.read_order_book(order, credentials)
        _raise_if_future_book_ts(
            book.book_ts,
            now=now,
            allowed_clock_skew_ms=self.allowed_clock_skew_ms,
        )
        return _quote_from_levels(
            order=order,
            market_status=book.market_status,
            bid_levels=book.bids,
            ask_levels=book.asks,
            quote_hash=book.quote_hash,
            book_ts=book.book_ts,
            now=now,
            source="venue_direct",
        )


@dataclass(frozen=True)
class PolymarketRoutingQuoteProvider:
    snapshot_provider: LiveQuoteProvider
    direct_provider: LiveQuoteProvider
    settings: PMSSettings

    async def quote(
        self,
        order: PolymarketOrderRequest,
        credentials: VenueCredentials,
    ) -> LivePreSubmitQuote:
        return await self.quote_for_order(
            order,
            credentials,
            first_live_order=False,
        )

    async def quote_for_order(
        self,
        order: PolymarketOrderRequest,
        credentials: VenueCredentials,
        *,
        first_live_order: bool,
    ) -> LivePreSubmitQuote:
        del first_live_order
        source = self.settings.controller.quote_source
        if source == "venue_direct" or _requires_direct_quote(order, self.settings):
            return await self.direct_provider.quote(order, credentials)
        if source == "dual":
            snapshot_quote = await self.snapshot_provider.quote(order, credentials)
            direct_quote = await self.direct_provider.quote(order, credentials)
            _validate_dual_quote_match(
                snapshot_quote,
                direct_quote,
                self.settings,
            )
            return replace(direct_quote, source="dual")
        return await self.snapshot_provider.quote(order, credentials)


class FirstLiveOrderGate(Protocol):
    async def approve_first_order(self, preview: LiveOrderPreview) -> bool: ...

    async def consume(self, preview: LiveOrderPreview) -> None: ...


@dataclass(frozen=True)
class DenyFirstLiveOrderGate:
    async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
        del preview
        return False

    async def consume(self, preview: LiveOrderPreview) -> None:
        del preview


class FirstOrderAuditWriter(Protocol):
    """Persistence sink for first-live-order operator events.

    Implementations record one event per call. Failures must NOT raise into
    the trading hot path — the actuator wraps each call in try/except to
    keep audit-write degradation independent of order submission. See
    `runner.py:1319-1320` for the same pattern on emergency-audit writes.
    """

    async def record_event(
        self,
        *,
        event: str,
        preview: LiveOrderPreview,
        approver_id: str | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class _NoopFirstOrderAuditWriter:
    """Default no-op audit writer used when no first-order audit sink is
    configured. Keeps offline tests, paper mode, and dev runs from
    requiring a writable audit path on disk."""

    async def record_event(
        self,
        *,
        event: str,
        preview: LiveOrderPreview,
        approver_id: str | None = None,
    ) -> None:
        del event, preview, approver_id


@dataclass
class FirstLiveOrderApprovalState:
    approved: bool = False
    consume_failed: bool = False


@dataclass(frozen=True)
class FileFirstLiveOrderGate:
    path: Path
    require_approver_sidecar: bool = False
    approval_max_age_s: float | None = None
    _approved_sidecar_fingerprint: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    async def approve_first_order(self, preview: LiveOrderPreview) -> bool:
        self._set_approved_sidecar_fingerprint(None)
        if not self._path_is_regular_file(self.path):
            return False
        try:
            payload = loads_json_rejecting_duplicate_keys(
                _read_text_no_follow(self.path),
                label="LIVE first-order approval artifact",
            )
        except (OSError, json.JSONDecodeError, LiveTradingDisabledError):
            return False
        if not isinstance(payload, dict):
            return False
        approval_payload = cast(dict[str, object], payload)
        if not _approval_payload_matches(approval_payload, preview):
            return False
        if not self.require_approver_sidecar:
            return True
        sidecar_payload = self._valid_approval_sidecar(approval_payload)
        if sidecar_payload is None:
            return False
        self._set_approved_sidecar_fingerprint(
            _canonical_json_fingerprint(sidecar_payload)
        )
        return True

    async def consume(self, preview: LiveOrderPreview) -> None:
        # Atomically consume the approval artefact so it cannot be replayed
        # by a future restart or a concurrent `approve_first_order` call.
        # Both the approval JSON and its sidecar metadata file share a
        # single authorization lifetime; unlink both so a stale sidecar
        # cannot misattribute a future authorization to the previous
        # approver.
        del preview
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        try:
            self._sidecar_path().unlink()
        except FileNotFoundError:
            pass

    def read_approver_id(self) -> str | None:
        payload = self._read_sidecar_payload()
        if payload is None:
            return None
        approver_id = self._sidecar_approver_id(payload)
        if approver_id is None:
            return None
        if self.require_approver_sidecar and not self._sidecar_timestamp_fresh(
            payload
        ):
            return None
        if (
            self.require_approver_sidecar
            and self._approved_sidecar_fingerprint
            != _canonical_json_fingerprint(payload)
        ):
            return None
        return approver_id

    def _sidecar_path(self) -> Path:
        return Path(str(self.path) + ".meta.json")

    def _approval_sidecar_valid(self, approval_payload: Mapping[str, object]) -> bool:
        return self._valid_approval_sidecar(approval_payload) is not None

    def _valid_approval_sidecar(
        self,
        approval_payload: Mapping[str, object],
    ) -> dict[str, object] | None:
        payload = self._read_sidecar_payload()
        if payload is None:
            return None
        if self._sidecar_approver_id(payload) is None:
            return None
        if not self._sidecar_timestamp_fresh(payload):
            return None
        if not self._sidecar_hash_matches_approval(payload, approval_payload):
            return None
        return payload

    def _read_sidecar_payload(self) -> dict[str, object] | None:
        sidecar = self._sidecar_path()
        if not self._path_is_regular_file(sidecar):
            return None
        try:
            payload = loads_json_rejecting_duplicate_keys(
                _read_text_no_follow(sidecar),
                label="LIVE first-order approval sidecar",
            )
        except (OSError, json.JSONDecodeError, LiveTradingDisabledError):
            return None
        if not isinstance(payload, dict):
            return None
        return cast(dict[str, object], payload)

    @staticmethod
    def _path_is_regular_file(path: Path) -> bool:
        try:
            path_stat = path.lstat()
        except FileNotFoundError:
            return False
        return (
            stat.S_ISREG(path_stat.st_mode)
            and path_stat.st_nlink == 1
            and FileFirstLiveOrderGate._parent_is_private(path)
        )

    @staticmethod
    def _parent_is_private(path: Path) -> bool:
        parent = path.parent
        try:
            parent_stat = parent.lstat()
        except FileNotFoundError:
            return False
        mode = stat.S_IMODE(parent_stat.st_mode)
        return (
            stat.S_ISDIR(parent_stat.st_mode)
            and mode & 0o077 == 0
            and bool(mode & stat.S_IWUSR)
        )

    def _sidecar_approver_id(self, payload: Mapping[str, object]) -> str | None:
        approver_id = payload.get("approver_id")
        if not isinstance(approver_id, str):
            return None
        normalized = approver_id.strip()
        if normalized == "" or _looks_like_placeholder(normalized):
            return None
        return normalized

    def _sidecar_timestamp_fresh(self, payload: Mapping[str, object]) -> bool:
        raw_ts = payload.get("ts")
        if not isinstance(raw_ts, str) or raw_ts.strip() == "":
            return False
        try:
            ts = datetime.fromisoformat(raw_ts.strip().replace("Z", "+00:00"))
        except ValueError:
            return False
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        ts = ts.astimezone(UTC)
        now = datetime.now(tz=UTC)
        if ts > now:
            return False
        if self.approval_max_age_s is None:
            return True
        return (now - ts).total_seconds() <= self.approval_max_age_s

    def _sidecar_hash_matches_approval(
        self,
        sidecar_payload: Mapping[str, object],
        approval_payload: Mapping[str, object],
    ) -> bool:
        expected_hash = sidecar_payload.get("approval_sha256")
        if not isinstance(expected_hash, str) or expected_hash.strip() == "":
            return False
        return expected_hash == _approval_payload_hash(approval_payload)

    def _set_approved_sidecar_fingerprint(self, fingerprint: str | None) -> None:
        object.__setattr__(self, "_approved_sidecar_fingerprint", fingerprint)


@dataclass(frozen=True)
class PolymarketSDKClient:
    async def submit_order(
        self,
        order: PolymarketOrderRequest,
        credentials: VenueCredentials,
    ) -> PolymarketOrderResult:
        return await asyncio.to_thread(self._submit_order_sync, order, credentials)

    async def read_order_book(
        self,
        order: PolymarketOrderRequest,
        credentials: VenueCredentials,
    ) -> LiveVenueBook:
        return await asyncio.to_thread(self._read_order_book_sync, order, credentials)

    async def read_account_snapshot(
        self,
        credentials: VenueCredentials,
    ) -> VenueAccountSnapshot:
        return await asyncio.to_thread(self._read_account_snapshot_sync, credentials)

    def _submit_order_sync(
        self,
        order: PolymarketOrderRequest,
        credentials: VenueCredentials,
    ) -> PolymarketOrderResult:
        try:
            sdk = importlib.import_module("py_clob_client_v2")
        except ModuleNotFoundError:
            msg = (
                "Polymarket live SDK is not installed. Install the live extra "
                "with `uv sync --extra live` before enabling LIVE mode."
            )
            raise LiveTradingDisabledError(msg) from None

        try:
            client = _build_sdk_client(sdk, credentials)
            response = _post_sdk_order(sdk, client, order)
        except (asyncio.CancelledError, KeyboardInterrupt):
            # Cooperative cancellation — caller decides what to do. The
            # underlying request may still be in flight on Polymarket; the
            # caller is responsible for reconciliation if it suppresses this.
            raise
        except (asyncio.TimeoutError, TimeoutError) as exc:
            # Bare timeout — the venue never responded. Surface as
            # submission_unknown so operators reconcile before retrying.
            redacted = _redacted_exception_message(exc, credentials)
            logger.warning(
                "Polymarket live order submission timed out (%s): %s",
                type(exc).__name__,
                redacted,
            )
            msg = (
                "Polymarket live order submission timed out; order status is "
                "unknown — reconcile with venue before retrying"
            )
            raise PolymarketSubmissionUnknownError(msg) from None
        except Exception as exc:  # noqa: BLE001
            redacted = _redacted_exception_message(exc, credentials)
            # `py_clob_client_v2.exceptions.PolyApiException` wraps httpx
            # request errors (timeouts, connection drops) with `resp=None`,
            # and HTTP-level venue errors with a populated httpx Response.
            # The bare `except (TimeoutError, ...)` block above misses
            # these wrapped transport failures — without this branch a
            # real timeout still routes to `LiveTradingDisabledError` =
            # venue_rejection, defeating the f2 contract that timeouts
            # must surface as submission_unknown. Duck-typed (class name
            # + attribute) so polymarket.py keeps no hard dependency on
            # the SDK's exception class.
            if _is_sdk_transport_failure(exc):
                logger.warning(
                    "Polymarket live order submission transport failure (%s): %s; "
                    "treating as submission_unknown",
                    type(exc).__name__,
                    redacted,
                )
                msg = (
                    "Polymarket live order submission transport failure; "
                    "order status is unknown — reconcile with venue before retrying"
                )
                raise PolymarketSubmissionUnknownError(msg) from None
            logger.warning(
                "Polymarket live order submission failed (%s): %s",
                type(exc).__name__,
                redacted,
            )
            msg = (
                "Polymarket live order submission failed "
                f"({type(exc).__name__}); venue error redacted"
            )
            raise PolymarketVenueRejectionError(msg) from None

        return _order_result_from_sdk_response(order, response)

    def _read_order_book_sync(
        self,
        order: PolymarketOrderRequest,
        credentials: VenueCredentials,
    ) -> LiveVenueBook:
        try:
            sdk = importlib.import_module("py_clob_client_v2")
        except ModuleNotFoundError:
            msg = (
                "Polymarket live SDK is not installed. Install the live extra "
                "with `uv sync --extra live` before enabling LIVE mode."
            )
            raise LiveTradingDisabledError(msg) from None

        try:
            client = _build_sdk_client(sdk, credentials)
            response = _get_sdk_order_book(client, order.token_id)
        except Exception as exc:  # noqa: BLE001
            redacted = _redacted_exception_message(exc, credentials)
            logger.warning(
                "Polymarket live order book fetch failed (%s): %s",
                type(exc).__name__,
                redacted,
            )
            msg = (
                "Polymarket live order book fetch failed "
                f"({type(exc).__name__}); venue error redacted"
            )
            raise LiveTradingDisabledError(msg) from None

        return _venue_book_from_sdk_response(
            response,
            market_id=order.market_id,
            token_id=order.token_id,
        )

    def _read_account_snapshot_sync(
        self,
        credentials: VenueCredentials,
    ) -> VenueAccountSnapshot:
        try:
            sdk = importlib.import_module("py_clob_client_v2")
        except ModuleNotFoundError:
            msg = (
                "Polymarket live SDK is not installed. Install the live extra "
                "with `uv sync --extra live` before enabling LIVE mode."
            )
            raise LiveTradingDisabledError(msg) from None

        try:
            client = _build_sdk_client(sdk, credentials)
            open_orders = _order_states_from_open_orders(
                _get_sdk_open_orders(client),
                credentials=credentials,
            )
            positions = _positions_from_sdk_positions(
                _get_sdk_positions(client, credentials),
                credentials=credentials,
            )
            balances = _get_sdk_balances(client, sdk, credentials)
        except Exception as exc:  # noqa: BLE001
            redacted = _redacted_exception_message(exc, credentials)
            logger.warning(
                "Polymarket live account snapshot failed (%s): %s",
                type(exc).__name__,
                redacted,
            )
            msg = (
                "Polymarket live account snapshot failed "
                f"({type(exc).__name__}); venue error redacted"
            )
            raise LiveTradingDisabledError(msg) from None

        return VenueAccountSnapshot(
            balances=balances,
            open_orders=tuple(open_orders),
            positions=tuple(positions),
        )


@dataclass(frozen=True)
class PolymarketVenueAccountReconciler:
    client: LiveVenueAccountClient = field(default_factory=PolymarketSDKClient)
    position_tolerance_shares: float = 1e-6
    notional_tolerance_usdc: float = 1e-4
    cash_tolerance_usdc: float = 1e-4

    async def snapshot(self, credentials: VenueCredentials) -> VenueAccountSnapshot:
        return await self.client.read_account_snapshot(credentials)

    async def compare(
        self,
        db_portfolio: Portfolio,
        venue_snapshot: VenueAccountSnapshot,
    ) -> ReconciliationReport:
        mismatches: list[str] = []
        if venue_snapshot.open_orders:
            mismatches.append(
                f"venue has {len(venue_snapshot.open_orders)} open orders; "
                "PMS has no durable live open-order ledger yet"
            )
        cash_balance = _venue_cash_balance(venue_snapshot.balances)
        if cash_balance is None:
            mismatches.append(
                "venue pUSD balance missing; cannot prove LIVE cash budget"
            )
        elif not math.isfinite(cash_balance):
            mismatches.append(
                "venue pUSD balance invalid; cannot prove LIVE cash budget"
            )
        elif cash_balance + self.cash_tolerance_usdc < db_portfolio.free_usdc:
            mismatches.append(
                "venue pUSD balance below PMS free cash: "
                f"venue={cash_balance:.8f} DB={db_portfolio.free_usdc:.8f}"
            )
        collateral_allowance = _venue_pusd_allowance(venue_snapshot.balances)
        if cash_balance is not None and math.isfinite(cash_balance):
            if collateral_allowance is None:
                mismatches.append(
                    "venue pUSD allowance missing; cannot prove LIVE buy capacity"
                )
            elif not math.isfinite(collateral_allowance):
                mismatches.append(
                    "venue pUSD allowance invalid; cannot prove LIVE buy capacity"
                )
            elif (
                collateral_allowance + self.cash_tolerance_usdc
                < db_portfolio.free_usdc
            ):
                mismatches.append(
                    "venue pUSD allowance below PMS free cash: "
                    f"venue={collateral_allowance:.8f} "
                    f"DB={db_portfolio.free_usdc:.8f}"
                )
        mismatches.extend(
            _compare_positions(
                db_portfolio.open_positions,
                venue_snapshot.positions,
                share_tolerance=self.position_tolerance_shares,
                notional_tolerance=self.notional_tolerance_usdc,
            )
        )
        return ReconciliationReport(ok=not mismatches, mismatches=tuple(mismatches))


@dataclass(frozen=True)
class MissingPolymarketClient:
    async def submit_order(
        self,
        order: PolymarketOrderRequest,
        credentials: VenueCredentials,
    ) -> PolymarketOrderResult:
        del order, credentials
        msg = (
            "Polymarket live client is not configured. Inject a PolymarketClient "
            "or install and wire a venue client before enabling LIVE mode."
        )
        raise LiveTradingDisabledError(msg)


@dataclass(frozen=True)
class PolymarketActuator:
    settings: PMSSettings = field(default_factory=PMSSettings)
    client: PolymarketClient = field(default_factory=MissingPolymarketClient)
    operator_gate: FirstLiveOrderGate = field(default_factory=DenyFirstLiveOrderGate)
    quote_provider: LiveQuoteProvider = field(default_factory=MissingLiveQuoteProvider)
    audit_writer: FirstOrderAuditWriter = field(
        default_factory=_NoopFirstOrderAuditWriter,
    )
    live_preflight_validated: bool = False
    live_preflight_active_strategies_fingerprint: str | None = None
    _approval_state: FirstLiveOrderApprovalState = field(
        default_factory=FirstLiveOrderApprovalState,
        init=False,
        repr=False,
        compare=False,
    )
    _approval_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    async def execute(
        self,
        decision: TradeDecision,
        portfolio: Portfolio | None = None,
    ) -> OrderState:
        if not self.settings.live_trading_enabled:
            raise LiveTradingDisabledError("Polymarket live trading is disabled")
        if self.settings.mode != RunMode.LIVE and _client_requires_live_mode(
            self.client
        ):
            msg = (
                "Polymarket live submission requires mode=live; "
                f"got mode={self.settings.mode.value!r}"
            )
            raise LiveTradingDisabledError(msg)
        del portfolio
        credentials = validate_live_mode_ready(
            self.settings,
            allow_pending_operator_approval=True,
            require_live_mode=_client_requires_live_mode(self.client),
        )
        request = _order_request(decision)
        self._require_live_preflight_artifact()
        self._require_strict_operator_gate_for_true_live()
        self._raise_if_operator_approval_blocked()

        # Fast path: in first-order mode, once the first order has been
        # confirmed by the venue, subsequent submits do not serialize on
        # `_approval_lock`. In every-order mode this remains closed.
        if not self._operator_approval_required():
            try:
                return await self._submit_with_quote_guard(
                    decision=decision,
                    request=request,
                    credentials=credentials,
                )
            except asyncio.CancelledError as exc:
                raise PolymarketSubmissionUnknownError(
                    "Polymarket live submission was cancelled; order status is "
                    "unknown — reconcile with venue before retrying",
                    order_state=_submission_unknown_order_state(decision),
                ) from exc

        # Slow path: hold `_approval_lock` across the *entire* approval +
        # submit + commit window. The previous implementation released
        # the lock before `submit_order()` and only flipped
        # `_approval_state.approved` after the venue replied — leaving a
        # window where a second concurrent task could re-read the same
        # approval file (which `consume()` had not yet unlinked) and
        # authorize a parallel first-order submit. Holding the lock
        # through submit serializes only the first-order path while
        # preserving "consume-on-success" semantics: if `submit_order`
        # raises, the lock releases without flipping the flag and the
        # next caller will re-prompt the gate.
        async with self._approval_lock:
            # Double-check after acquiring — in first-order mode another
            # task may have just opened the fast path while we were waiting.
            if not self._operator_approval_required():
                return await self._submit_with_quote_guard(
                    decision=decision,
                    request=request,
                    credentials=credentials,
                )

            preview = await self._live_order_preview(decision)
            approved = await self.operator_gate.approve_first_order(preview)
            # Capture approver_id from the operator gate once, before
            # consume() can unlink any sidecar. Threading the same
            # value through matched/denied/consumed events guarantees
            # the audit log answers "who authorized this" consistently
            # across all three records for one authorization act.
            approver_id = self._read_approver_id()
            if (
                approved
                and self._operator_gate_requires_approver_id()
                and approver_id is None
            ):
                await self._emit_audit(
                    event="approval_denied",
                    preview=preview,
                    approver_id=None,
                )
                raise OperatorApprovalRequiredError(
                    self._operator_approval_error_message(preview)
                )
            if not approved:
                await self._emit_audit(
                    event="approval_denied",
                    preview=preview,
                    approver_id=approver_id,
                )
                raise OperatorApprovalRequiredError(
                    self._operator_approval_error_message(preview)
                )

            await self._emit_audit(
                event="approval_matched",
                preview=preview,
                approver_id=approver_id,
            )

            # Submit while still holding the lock. If this raises, the
            # `async with` releases the lock and `_approval_state.approved`
            # stays False — the next caller will see no commit and
            # re-prompt the gate (correct consume-on-success behavior).
            try:
                order_state = await self._submit_with_quote_guard(
                    decision=decision,
                    request=request,
                    credentials=credentials,
                )
            except PolymarketSubmissionUnknownError as error:
                self._approval_state.approved = True
                if error.order_state is None:
                    error.order_state = _submission_unknown_order_state(decision)
                try:
                    await asyncio.shield(self.operator_gate.consume(preview))
                except Exception as consume_error:  # noqa: BLE001
                    logger.error(
                        "operator approval consume failed after unknown submit: %s",
                        consume_error,
                    )
                    if self.settings.polymarket.operator_approval_mode == "every_order":
                        self._approval_state.consume_failed = True
                await self._emit_audit(
                    event="approval_consumed",
                    preview=preview,
                    approver_id=approver_id,
                )
                raise
            except asyncio.CancelledError as exc:
                # Cancellation does not prove the worker thread stopped. The
                # approval is consumed before surfacing an unknown incident so
                # a restart cannot replay the same authorization blindly.
                self._approval_state.approved = True
                try:
                    await asyncio.shield(self.operator_gate.consume(preview))
                except Exception as consume_error:  # noqa: BLE001
                    logger.error(
                        "operator approval consume failed after cancelled submit: %s",
                        consume_error,
                    )
                    if self.settings.polymarket.operator_approval_mode == "every_order":
                        self._approval_state.consume_failed = True
                await self._emit_audit(
                    event="approval_consumed",
                    preview=preview,
                    approver_id=approver_id,
                )
                raise PolymarketSubmissionUnknownError(
                    "Polymarket live submission was cancelled; order status is "
                    "unknown — reconcile with venue before retrying",
                    order_state=_submission_unknown_order_state(decision),
                ) from exc

            # First venue submission succeeded: lock in first-order
            # completion and consume the operator-side approval artefact
            # so it cannot be replayed by a future restart.
            #
            # `_approval_state.approved=True` MUST stay set even if
            # consume() fails — flipping it back would cause the next
            # in-process decision to re-prompt the gate and double-submit
            # against the still-present approval file. The cleanup-
            # failure handling below is forensic, not transactional.
            self._approval_state.approved = True
            try:
                await asyncio.shield(self.operator_gate.consume(preview))
            except Exception as exc:  # noqa: BLE001
                detail = _redacted_settings_exception_message(exc, self.settings)
                # Record the truth: cleanup failed, the artefact may
                # still be on disk, a future restart could replay it.
                # ERROR-level so an external alerter can page the
                # operator for manual cleanup.
                logger.error(
                    "operator approval gate consume failed: %s. Approval "
                    "artefact may remain on disk and could replay on "
                    "restart; manual cleanup required.",
                    detail,
                )
                if self.settings.polymarket.operator_approval_mode == "every_order":
                    self._approval_state.consume_failed = True
                await self._emit_audit(
                    event="approval_consume_failed",
                    preview=preview,
                    approver_id=approver_id,
                )
            except asyncio.CancelledError as exc:
                if self.settings.polymarket.operator_approval_mode == "every_order":
                    self._approval_state.consume_failed = True
                await self._emit_audit(
                    event="approval_consume_failed",
                    preview=preview,
                    approver_id=approver_id,
                )
                raise PolymarketSubmissionUnknownError(
                    "Polymarket live submission bookkeeping was cancelled; order "
                    "status is unknown — reconcile with venue before retrying",
                    order_state=order_state,
                ) from exc
            else:
                try:
                    await self._emit_audit(
                        event="approval_consumed",
                        preview=preview,
                        approver_id=approver_id,
                    )
                except asyncio.CancelledError as exc:
                    raise PolymarketSubmissionUnknownError(
                        "Polymarket live submission bookkeeping was cancelled; "
                        "order status is unknown — reconcile with venue before retrying",
                        order_state=order_state,
                    ) from exc

        return order_state

    async def _emit_audit(
        self,
        *,
        event: str,
        preview: LiveOrderPreview,
        approver_id: str | None,
    ) -> None:
        # Audit emission is fire-and-forget from the trading hot path: a
        # writer outage must never block or roll back an order. Mirrors the
        # precedent at runner.py:1319-1320 for emergency-audit append.
        try:
            await self.audit_writer.record_event(
                event=event,
                preview=preview,
                approver_id=approver_id,
            )
        except Exception as exc:  # noqa: BLE001
            detail = _redacted_settings_exception_message(exc, self.settings)
            logger.warning(
                "first-order audit write failed: event=%s err=%s",
                event,
                detail,
            )

    def _read_approver_id(self) -> str | None:
        # Duck-typed: only FileFirstLiveOrderGate exposes
        # `read_approver_id` today. Other gates (Recording, Deny,
        # Blocking, *CountingFile) fall through to None — they have
        # no sidecar concept. Same getattr pattern used at
        # _submit_with_quote_guard to keep optional gate features off
        # the FirstLiveOrderGate Protocol surface.
        reader = getattr(self.operator_gate, "read_approver_id", None)
        if not callable(reader):
            return None
        try:
            result = reader()
        except Exception as exc:  # noqa: BLE001
            detail = _redacted_settings_exception_message(exc, self.settings)
            logger.warning(
                "first-order operator gate read_approver_id failed: %s",
                detail,
            )
            return None
        return result if isinstance(result, str) else None

    def _operator_gate_requires_approver_id(self) -> bool:
        return bool(getattr(self.operator_gate, "require_approver_sidecar", False))

    def _first_order_approved(self) -> bool:
        return self._approval_state.approved

    def _operator_approval_required(self) -> bool:
        if self.settings.polymarket.operator_approval_mode == "every_order":
            return True
        return not self._first_order_approved()

    def _require_strict_operator_gate_for_true_live(self) -> None:
        if self.settings.mode != RunMode.LIVE:
            return
        if self.settings.polymarket.operator_approval_mode != "every_order":
            return
        if (
            isinstance(self.operator_gate, FileFirstLiveOrderGate)
            and self.operator_gate.require_approver_sidecar
        ):
            return
        msg = (
            "Polymarket LIVE every-order approval requires a strict sidecar "
            "operator gate; configure FileFirstLiveOrderGate with "
            "require_approver_sidecar=True."
        )
        raise LiveTradingDisabledError(msg)

    def _raise_if_operator_approval_blocked(self) -> None:
        if (
            self.settings.polymarket.operator_approval_mode == "every_order"
            and self._approval_state.consume_failed
        ):
            msg = (
                "Every-order operator approval blocked: approval consume failed "
                "after a prior submit. Stop the runner, remove the stale approval "
                "artefact, reconcile venue state, and restart."
            )
            raise LiveTradingDisabledError(msg)

    def _operator_approval_error_message(self, preview: LiveOrderPreview) -> str:
        subject = (
            "Every Polymarket live order"
            if self.settings.polymarket.operator_approval_mode == "every_order"
            else "First Polymarket live order"
        )
        return (
            f"{subject} requires operator approval: "
            f"venue={preview.venue} market={preview.market_id} "
            f"token={preview.token_id} side={preview.side} "
            f"outcome={preview.outcome} "
            f"max_notional_usdc={preview.max_notional_usdc} "
            f"limit_price={preview.limit_price} "
            f"max_slippage_bps={preview.max_slippage_bps}"
        )

    async def _live_order_preview(self, decision: TradeDecision) -> LiveOrderPreview:
        market_slug, question = await self._preview_market_metadata(decision.market_id)
        return LiveOrderPreview(
            max_notional_usdc=decision.notional_usdc,
            venue=decision.venue,
            market_id=decision.market_id,
            token_id=decision.token_id,
            side=decision.side,
            limit_price=decision.limit_price,
            max_slippage_bps=decision.max_slippage_bps,
            outcome=decision.outcome,
            market_slug=market_slug,
            question=question,
        )

    async def _preview_market_metadata(
        self,
        market_id: str,
    ) -> tuple[str | None, str | None]:
        quote_provider = self.quote_provider
        if isinstance(quote_provider, PolymarketRoutingQuoteProvider):
            quote_provider = quote_provider.snapshot_provider
        if not isinstance(quote_provider, PolymarketBookQuoteProvider):
            return None, None
        market = await quote_provider.store.read_market(market_id)
        if market is None:
            return None, None
        return market.slug, market.question

    async def _submit_with_quote_guard(
        self,
        *,
        decision: TradeDecision,
        request: PolymarketOrderRequest,
        credentials: VenueCredentials,
    ) -> OrderState:
        quote_for_order = getattr(self.quote_provider, "quote_for_order", None)
        if callable(quote_for_order):
            quote = await quote_for_order(
                request,
                credentials,
                first_live_order=not self._first_order_approved(),
            )
        else:
            quote = await self.quote_provider.quote(request, credentials)
        _validate_pre_submit_quote(request, quote, self.settings)
        result = await self.client.submit_order(request, credentials)
        order_state = _order_state_from_result(decision, result, quote=quote)
        _raise_if_unexpected_resting_non_gtc(
            settings=self.settings,
            request=request,
            order_state=order_state,
        )
        return order_state

    def _require_live_preflight_artifact(self) -> None:
        if self.settings.mode != RunMode.LIVE:
            return
        raw_path = self.settings.live_preflight_artifact_path
        if raw_path is None or raw_path.strip() == "":
            msg = (
                "LIVE credentialed preflight artifact missing: "
                "live_preflight_artifact_path"
            )
            raise LiveTradingDisabledError(msg)
        if _looks_like_placeholder(raw_path):
            msg = "LIVE credentialed preflight artifact path contains placeholder"
            raise LiveTradingDisabledError(msg)
        path = Path(raw_path).expanduser()
        _require_live_preflight_artifact_outside_working_tree(path)
        _require_live_preflight_artifact_parent_owner_writable(path)
        try:
            path_stat = path.lstat()
        except FileNotFoundError as exc:
            msg = f"LIVE credentialed preflight artifact does not exist: {path}"
            raise LiveTradingDisabledError(msg) from exc
        if not stat.S_ISREG(path_stat.st_mode):
            msg = (
                "LIVE credentialed preflight artifact path is not a regular "
                f"file: {path}"
            )
            raise LiveTradingDisabledError(msg)
        if path_stat.st_nlink != 1:
            msg = (
                "LIVE credentialed preflight artifact path is not a single-link "
                f"file: {path}"
            )
            raise LiveTradingDisabledError(msg)
        try:
            artifact = loads_json_rejecting_duplicate_keys(
                _read_text_no_follow(path),
                label="LIVE credentialed preflight artifact",
            )
        except (OSError, json.JSONDecodeError) as exc:
            msg = f"LIVE credentialed preflight artifact is unreadable: {path}"
            raise LiveTradingDisabledError(msg) from exc
        if not isinstance(artifact, dict):
            msg = "LIVE credentialed preflight artifact must be a JSON object"
            raise LiveTradingDisabledError(msg)
        _require_final_live_preflight_artifact_shape(
            cast(dict[str, object], artifact),
            path=path,
            settings=self.settings,
        )
        self._require_validated_active_strategies_fingerprint(
            cast(dict[str, object], artifact)
        )
        validate_live_strategy_artifacts_for_submission(self.settings)

    def _require_validated_active_strategies_fingerprint(
        self,
        artifact: Mapping[str, object],
    ) -> None:
        if not self.live_preflight_validated:
            return
        expected = self.live_preflight_active_strategies_fingerprint
        if expected is None:
            return
        observed = _require_preflight_fingerprint_field(
            artifact,
            "active_strategies_fingerprint",
        )
        if observed == expected:
            return
        msg = "LIVE credentialed preflight active strategies fingerprint mismatch"
        raise LiveTradingDisabledError(msg)


def _require_live_preflight_artifact_outside_working_tree(path: Path) -> None:
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


def _require_live_preflight_artifact_parent_owner_writable(path: Path) -> None:
    parent = path.parent
    try:
        parent_stat = parent.lstat()
    except FileNotFoundError as exc:
        msg = f"LIVE credentialed preflight artifact parent does not exist: {parent}"
        raise LiveTradingDisabledError(msg) from exc
    if not stat.S_ISDIR(parent_stat.st_mode):
        msg = (
            "LIVE credentialed preflight artifact parent is not a directory: "
            f"{parent}"
        )
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


def _require_final_live_preflight_artifact_shape(
    artifact: Mapping[str, object],
    *,
    path: Path,
    settings: PMSSettings,
) -> None:
    if artifact.get("generated_by") != "pms-live preflight":
        msg = "LIVE credentialed preflight artifact generated_by is invalid"
        raise LiveTradingDisabledError(msg)
    if artifact.get("artifact_mode") != "credentialed_preflight":
        msg = (
            "LIVE credentialed preflight artifact_mode must be "
            "credentialed_preflight"
        )
        raise LiveTradingDisabledError(msg)
    if artifact.get("final_go_no_go_valid") is not True:
        msg = "LIVE credentialed preflight artifact final_go_no_go_valid must be true"
        raise LiveTradingDisabledError(msg)
    if artifact.get("skip_venue") is not False:
        msg = "LIVE credentialed preflight artifact must not skip venue reconciliation"
        raise LiveTradingDisabledError(msg)
    if artifact.get("database_url_override_used") is not False:
        msg = (
            "LIVE credentialed preflight artifact must not use "
            "database URL override"
        )
        raise LiveTradingDisabledError(msg)
    _require_preflight_artifact_output_path(artifact, path=path)
    _require_preflight_settings_fingerprint(artifact, settings=settings)
    _require_preflight_readiness_reports_fingerprint(artifact, settings=settings)
    _require_preflight_fingerprint_field(artifact, "active_strategies_fingerprint")
    generated_at = _require_fresh_live_preflight_artifact_timestamp(
        artifact,
        settings,
    )
    _require_preflight_after_readiness_reports(settings, generated_at=generated_at)
    _require_preflight_after_readiness(settings, generated_at=generated_at)
    _require_preflight_after_emergency_audit(settings, generated_at=generated_at)
    _require_final_live_preflight_result_shape(artifact)


def _require_preflight_artifact_output_path(
    artifact: Mapping[str, object],
    *,
    path: Path,
) -> None:
    raw_output_path = artifact.get("output_path")
    if not isinstance(raw_output_path, str) or raw_output_path.strip() == "":
        msg = "LIVE credentialed preflight artifact missing output_path"
        raise LiveTradingDisabledError(msg)
    if _looks_like_placeholder(raw_output_path):
        msg = "LIVE credentialed preflight artifact output_path contains placeholder"
        raise LiveTradingDisabledError(msg)
    if Path(raw_output_path).expanduser() != path:
        msg = (
            "LIVE credentialed preflight artifact output_path must match "
            "live_preflight_artifact_path"
        )
        raise LiveTradingDisabledError(msg)


def _require_preflight_fingerprint_field(
    artifact: Mapping[str, object],
    field_name: str,
) -> str:
    raw_value = artifact.get(field_name)
    if not isinstance(raw_value, str) or raw_value.strip() == "":
        msg = f"LIVE credentialed preflight artifact missing {field_name}"
        raise LiveTradingDisabledError(msg)
    if _looks_like_placeholder(raw_value) or not _is_sha256_hexdigest(raw_value):
        msg = f"LIVE credentialed preflight artifact {field_name} is invalid"
        raise LiveTradingDisabledError(msg)
    return raw_value


def _require_preflight_settings_fingerprint(
    artifact: Mapping[str, object],
    *,
    settings: PMSSettings,
) -> None:
    observed = _require_preflight_fingerprint_field(artifact, "settings_fingerprint")
    expected = live_preflight_settings_fingerprint(settings)
    if observed != expected:
        msg = "LIVE credentialed preflight artifact settings fingerprint mismatch"
        raise LiveTradingDisabledError(msg)


def _require_preflight_readiness_reports_fingerprint(
    artifact: Mapping[str, object],
    *,
    settings: PMSSettings,
) -> None:
    observed = _require_preflight_fingerprint_field(
        artifact,
        "readiness_reports_fingerprint",
    )
    expected = live_preflight_readiness_reports_fingerprint(settings)
    if observed != expected:
        msg = (
            "LIVE credentialed preflight artifact readiness reports "
            "fingerprint mismatch"
        )
        raise LiveTradingDisabledError(msg)


def _require_final_live_preflight_result_shape(
    artifact: Mapping[str, object],
) -> None:
    raw_result = artifact.get("result")
    if not isinstance(raw_result, Mapping):
        msg = "LIVE credentialed preflight artifact result must be a JSON object"
        raise LiveTradingDisabledError(msg)
    if raw_result.get("ok") is not True:
        msg = "LIVE credentialed preflight artifact result must be ok"
        raise LiveTradingDisabledError(msg)
    raw_checks = raw_result.get("checks")
    if not isinstance(raw_checks, list):
        msg = "LIVE credentialed preflight artifact checks must be a list"
        raise LiveTradingDisabledError(msg)
    observed_names: list[str] = []
    failed_names: list[str] = []
    malformed_names: list[str] = []
    for index, raw_check in enumerate(raw_checks):
        if not isinstance(raw_check, Mapping):
            malformed_names.append(str(index))
            continue
        raw_name = raw_check.get("name")
        raw_detail = raw_check.get("detail")
        if (
            not isinstance(raw_name, str)
            or raw_name.strip() == ""
            or not isinstance(raw_detail, str)
            or raw_detail.strip() == ""
            or _looks_like_placeholder(raw_detail)
        ):
            malformed_names.append(str(index))
            continue
        name = raw_name.strip()
        observed_names.append(name)
        if raw_check.get("ok") is not True:
            failed_names.append(name)
    if malformed_names:
        fields = ", ".join(malformed_names)
        msg = f"LIVE credentialed preflight artifact malformed checks: {fields}"
        raise LiveTradingDisabledError(msg)
    duplicate_names = sorted(
        name for name in set(observed_names) if observed_names.count(name) > 1
    )
    if duplicate_names:
        fields = ", ".join(duplicate_names)
        msg = f"LIVE credentialed preflight artifact duplicate checks: {fields}"
        raise LiveTradingDisabledError(msg)
    required_names = set(_REQUIRED_FINAL_PREFLIGHT_CHECKS)
    unknown_names = sorted(set(observed_names) - required_names)
    if unknown_names:
        fields = ", ".join(unknown_names)
        msg = f"LIVE credentialed preflight artifact unknown checks: {fields}"
        raise LiveTradingDisabledError(msg)
    missing_names = sorted(required_names - set(observed_names))
    if missing_names:
        fields = ", ".join(missing_names)
        msg = f"LIVE credentialed preflight artifact missing checks: {fields}"
        raise LiveTradingDisabledError(msg)
    if failed_names:
        fields = ", ".join(failed_names)
        msg = f"LIVE credentialed preflight artifact failed checks: {fields}"
        raise LiveTradingDisabledError(msg)


def _require_fresh_live_preflight_artifact_timestamp(
    artifact: Mapping[str, object],
    settings: PMSSettings,
) -> datetime:
    raw_generated_at = artifact.get("generated_at")
    if not isinstance(raw_generated_at, str) or raw_generated_at.strip() == "":
        msg = "LIVE credentialed preflight artifact missing generated_at"
        raise LiveTradingDisabledError(msg)
    try:
        generated_at = datetime.fromisoformat(
            raw_generated_at.strip().replace("Z", "+00:00")
        )
    except ValueError as exc:
        msg = "LIVE credentialed preflight artifact generated_at is invalid"
        raise LiveTradingDisabledError(msg) from exc
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        msg = "LIVE credentialed preflight artifact generated_at must include timezone"
        raise LiveTradingDisabledError(msg)
    generated_at = generated_at.astimezone(UTC)
    now = datetime.now(tz=UTC)
    if generated_at > now:
        msg = "LIVE credentialed preflight artifact generated_at is in the future"
        raise LiveTradingDisabledError(msg)
    age_s = (now - generated_at).total_seconds()
    if age_s <= settings.live_preflight_artifact_max_age_s:
        return generated_at
    msg = (
        "LIVE credentialed preflight artifact is stale: "
        f"age {age_s:.1f}s exceeds {settings.live_preflight_artifact_max_age_s:.1f}s"
    )
    raise LiveTradingDisabledError(msg)


def _require_preflight_after_emergency_audit(
    settings: PMSSettings,
    *,
    generated_at: datetime,
) -> None:
    latest_emergency_audit_at = latest_live_emergency_audit_timestamp(
        settings.live_emergency_audit_path
    )
    if latest_emergency_audit_at is None or latest_emergency_audit_at <= generated_at:
        return
    msg = (
        "LIVE credentialed preflight artifact generated_at predates "
        "emergency audit: live_emergency_audit_path"
    )
    raise LiveTradingDisabledError(msg)


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
    if not stale_fields:
        return
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
        for label, report_generated_at in (
            live_preflight_readiness_report_generated_at_values(settings)
        )
        if report_generated_at > generated_at
    ]
    if not stale_reports:
        return
    fields = ", ".join(stale_reports)
    msg = (
        "LIVE credentialed preflight artifact generated_at predates "
        f"readiness reports: {fields}"
    )
    raise LiveTradingDisabledError(msg)


def _coerce_preflight_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _order_request(decision: TradeDecision) -> PolymarketOrderRequest:
    if decision.time_in_force.value not in {"IOC", "FOK"}:
        msg = (
            "LIVE order time_in_force must be IOC or FOK until PMS has a "
            "durable open-order ledger"
        )
        raise LiveTradingDisabledError(msg)
    if decision.token_id is None:
        msg = "Polymarket live execution requires decision.token_id"
        raise LiveTradingDisabledError(msg)
    try:
        notional = validate_positive(
            decision.notional_usdc,
            field_name="PolymarketOrderRequest.notional_usdc",
        )
        price = validate_price(
            decision.limit_price,
            field_name="PolymarketOrderRequest.price",
        )
        max_slippage_bps = int(
            validate_finite(
                decision.max_slippage_bps,
                field_name="PolymarketOrderRequest.max_slippage_bps",
            )
        )
    except ValueError as exc:
        raise LiveTradingDisabledError(
            "Polymarket order request contains invalid or non-finite values"
        ) from exc
    estimated_quantity = _decision_quantity(decision, notional=notional, price=price)
    return PolymarketOrderRequest(
        market_id=decision.market_id,
        token_id=decision.token_id,
        side=decision.action if decision.action is not None else decision.side,
        price=price,
        size=_sdk_order_size(decision, estimated_quantity=estimated_quantity),
        notional_usdc=notional,
        estimated_quantity=estimated_quantity,
        order_type=decision.order_type,
        time_in_force=decision.time_in_force.value,
        max_slippage_bps=max_slippage_bps,
    )


def _order_state_from_result(
    decision: TradeDecision,
    result: PolymarketOrderResult,
    *,
    quote: LivePreSubmitQuote | None = None,
) -> OrderState:
    _validate_order_result_accounting(decision, result)
    now = datetime.now(tz=UTC)
    status = _canonical_polymarket_order_status(result.status or OrderStatus.LIVE.value)
    return OrderState(
        order_id=_concrete_result_order_id(result),
        decision_id=decision.decision_id,
        status=status,
        market_id=decision.market_id,
        token_id=decision.token_id,
        venue=decision.venue,
        requested_notional_usdc=decision.notional_usdc,
        filled_notional_usdc=result.filled_notional_usdc,
        remaining_notional_usdc=result.remaining_notional_usdc,
        fill_price=result.fill_price,
        submitted_at=now,
        last_updated_at=now,
        raw_status=result.raw_status,
        strategy_id=decision.strategy_id,
        strategy_version_id=decision.strategy_version_id,
        filled_quantity=result.filled_quantity,
        pre_submit_quote={
            **({} if quote is None else _quote_payload(quote)),
            **({} if result.fees is None else {"fees": result.fees}),
            **({} if result.fee_bps is None else {"fee_bps": result.fee_bps}),
            **(
                {}
                if result.transaction_ref is None
                else {"transaction_ref": result.transaction_ref}
            ),
        },
        action=decision.action,
        outcome=decision.outcome,
        time_in_force=decision.time_in_force.value,
        intent_key=decision.intent_key,
        risk_group_id=decision.risk_group_id,
    )


def _submission_unknown_order_state(decision: TradeDecision) -> OrderState:
    now = datetime.now(tz=UTC)
    return OrderState(
        order_id=f"submission-unknown-{uuid4().hex}",
        decision_id=decision.decision_id,
        status=OrderStatus.INVALID.value,
        market_id=decision.market_id,
        token_id=decision.token_id,
        venue=decision.venue,
        requested_notional_usdc=decision.notional_usdc,
        filled_notional_usdc=0.0,
        remaining_notional_usdc=decision.notional_usdc,
        fill_price=None,
        submitted_at=now,
        last_updated_at=now,
        raw_status="submission_unknown",
        strategy_id=decision.strategy_id,
        strategy_version_id=decision.strategy_version_id,
        filled_quantity=0.0,
        action=decision.action,
        outcome=decision.outcome,
        time_in_force=decision.time_in_force.value,
        intent_key=decision.intent_key,
        risk_group_id=decision.risk_group_id,
    )


def _concrete_result_order_id(result: PolymarketOrderResult) -> str:
    order_id = result.order_id.strip()
    if order_id != "" and not _looks_like_placeholder(order_id):
        return order_id
    msg = (
        "Polymarket live order result missing concrete venue order id; "
        "order status is unknown — reconcile with venue before retrying"
    )
    raise PolymarketSubmissionUnknownError(msg)


def _validate_order_result_accounting(
    decision: TradeDecision,
    result: PolymarketOrderResult,
) -> None:
    requested_notional = decision.notional_usdc
    filled_notional = result.filled_notional_usdc
    remaining_notional = result.remaining_notional_usdc
    filled_quantity = result.filled_quantity
    fill_price = result.fill_price

    _require_finite_result_number(
        filled_notional,
        field_name="filled_notional_usdc",
    )
    _require_finite_result_number(
        remaining_notional,
        field_name="remaining_notional_usdc",
    )
    _require_finite_result_number(
        filled_quantity,
        field_name="filled_quantity",
    )
    if fill_price is not None:
        _require_finite_result_number(fill_price, field_name="fill_price")

    if filled_notional < 0.0:
        _raise_unknown_malformed_result("filled_notional_usdc must be >= 0.0")
    if filled_notional > requested_notional + _NOTIONAL_OVERFILL_TOLERANCE:
        _raise_unknown_malformed_result(
            "filled_notional_usdc exceeds requested notional"
        )
    if remaining_notional < 0.0:
        _raise_unknown_malformed_result("remaining_notional_usdc must be >= 0.0")
    if filled_quantity < 0.0:
        _raise_unknown_malformed_result("filled_quantity must be >= 0.0")
    if fill_price is not None and not (
        _PROBABILITY_PRICE_MIN < fill_price <= _PROBABILITY_PRICE_MAX
    ):
        _raise_unknown_malformed_result("fill_price must satisfy 0.0 < price <= 1.0")

    if filled_notional > 0.0:
        if fill_price is None:
            _raise_unknown_malformed_result(
                "fill_price is required when filled_notional_usdc > 0.0"
            )
        if filled_quantity <= 0.0:
            _raise_unknown_malformed_result(
                "filled_quantity must be > 0.0 when filled_notional_usdc > 0.0"
            )
        assert fill_price is not None
        if not math.isclose(
            filled_notional,
            filled_quantity * fill_price,
            rel_tol=0.01,
            abs_tol=0.01,
        ):
            _raise_unknown_malformed_result(
                "fill accounting mismatch: filled_notional_usdc must equal "
                "filled_quantity * fill_price"
            )
    elif filled_quantity > 0.0:
        _raise_unknown_malformed_result(
            "filled_quantity must be 0.0 when filled_notional_usdc is 0.0"
        )

    if not math.isclose(
        filled_notional + remaining_notional,
        requested_notional,
        rel_tol=0.01,
        abs_tol=0.01,
    ):
        _raise_unknown_malformed_result(
            "notional accounting mismatch: filled_notional_usdc plus "
            "remaining_notional_usdc must equal requested notional"
        )


def _require_finite_result_number(value: float, *, field_name: str) -> None:
    if math.isfinite(value):
        return
    _raise_unknown_malformed_result(f"{field_name} must be finite")


def _raise_unknown_malformed_result(detail: str) -> None:
    msg = (
        "Polymarket live order result has malformed accounting: "
        f"{detail}; order status is unknown — reconcile with venue before retrying"
    )
    raise PolymarketSubmissionUnknownError(msg)


def _raise_if_unexpected_resting_non_gtc(
    *,
    settings: PMSSettings,
    request: PolymarketOrderRequest,
    order_state: OrderState,
) -> None:
    if (
        settings.mode == RunMode.LIVE
        and request.time_in_force != "GTC"
        and order_state.status == OrderStatus.LIVE.value
        and order_state.remaining_notional_usdc > 1e-9
    ):
        msg = "Non-GTC live order appears resting; requires venue reconciliation"
        raise PolymarketSubmissionUnknownError(msg, order_state=order_state)


def _validate_pre_submit_quote(
    request: PolymarketOrderRequest,
    quote: LivePreSubmitQuote,
    settings: PMSSettings,
) -> None:
    _validate_pre_submit_quote_shape(quote)
    if quote.market_status.lower() != "open":
        msg = f"Polymarket market is not open at submit: {quote.market_status}"
        raise LiveTradingDisabledError(msg)
    if quote.book_age_ms > settings.controller.max_book_age_ms:
        msg = (
            "Polymarket book is stale at submit: "
            f"{quote.book_age_ms:.0f}ms > {settings.controller.max_book_age_ms:.0f}ms"
        )
        raise LiveTradingDisabledError(msg)
    if quote.executable_notional_usdc + 1e-9 < request.notional_usdc:
        msg = (
            "Polymarket executable depth is below requested notional at submit: "
            f"{quote.executable_notional_usdc:.2f} < {request.notional_usdc:.2f}"
        )
        raise LiveTradingDisabledError(msg)
    if request.side == "BUY" and quote.best_executable_price > request.price:
        msg = (
            "Polymarket best executable price exceeds limit at submit: "
            f"{quote.best_executable_price:.4f} > {request.price:.4f}"
        )
        raise LiveTradingDisabledError(msg)
    if request.side == "SELL" and quote.best_executable_price < request.price:
        msg = (
            "Polymarket best executable price is below limit at submit: "
            f"{quote.best_executable_price:.4f} < {request.price:.4f}"
        )
        raise LiveTradingDisabledError(msg)
    if quote.spread_bps > settings.controller.max_spread_bps:
        msg = (
            "Polymarket spread exceeds pre-submit guard: "
            f"{quote.spread_bps:.1f}bps > {settings.controller.max_spread_bps:.1f}bps"
        )
        raise LiveTradingDisabledError(msg)


def _validate_pre_submit_quote_shape(quote: LivePreSubmitQuote) -> None:
    if not math.isfinite(quote.book_age_ms) or quote.book_age_ms < 0.0:
        msg = "Polymarket pre-submit quote book_age_ms is invalid"
        raise LiveTradingDisabledError(msg)
    if (
        not math.isfinite(quote.executable_notional_usdc)
        or quote.executable_notional_usdc < 0.0
    ):
        msg = "Polymarket pre-submit quote executable_notional_usdc is invalid"
        raise LiveTradingDisabledError(msg)
    if not (
        math.isfinite(quote.best_executable_price)
        and _PROBABILITY_PRICE_MIN < quote.best_executable_price <= _PROBABILITY_PRICE_MAX
    ):
        msg = "Polymarket pre-submit quote best_executable_price is invalid"
        raise LiveTradingDisabledError(msg)
    if not math.isfinite(quote.spread_bps) or quote.spread_bps < 0.0:
        msg = "Polymarket pre-submit quote spread_bps is invalid"
        raise LiveTradingDisabledError(msg)
    if quote.quote_hash.strip() == "" or _looks_like_placeholder(quote.quote_hash):
        msg = "Polymarket pre-submit quote quote_hash is invalid"
        raise LiveTradingDisabledError(msg)


def _quote_payload(quote: LivePreSubmitQuote) -> dict[str, object]:
    return {
        "market_status": quote.market_status,
        "book_age_ms": quote.book_age_ms,
        "executable_notional_usdc": quote.executable_notional_usdc,
        "best_executable_price": quote.best_executable_price,
        "spread_bps": quote.spread_bps,
        "quote_hash": quote.quote_hash,
        "book_ts": quote.book_ts.isoformat(),
        "source": quote.source,
    }


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _market_status(market: Market, now: datetime) -> str:
    if market.closed is True:
        return "closed"
    if market.active is False:
        return "inactive"
    if market.accepting_orders is False:
        return "not_accepting_orders"
    if market.resolves_at is not None and market.resolves_at <= now:
        return "closed"
    return "open"


def _quote_from_levels(
    *,
    order: PolymarketOrderRequest,
    market_status: str,
    bid_levels: Sequence[BookLevel],
    ask_levels: Sequence[BookLevel],
    quote_hash: str,
    book_ts: datetime,
    now: datetime,
    source: Literal["postgres_snapshot", "venue_direct", "dual"],
) -> LivePreSubmitQuote:
    sorted_bids = sorted(
        bid_levels,
        key=lambda level: level.price,
        reverse=True,
    )
    sorted_asks = sorted(ask_levels, key=lambda level: level.price)
    executable_levels = (
        [level for level in sorted_asks if level.price <= order.price]
        if order.side == "BUY"
        else [level for level in sorted_bids if level.price >= order.price]
    )
    best_executable_price = (
        executable_levels[0].price if executable_levels else order.price
    )
    executable_notional_usdc = sum(
        max(0.0, level.price) * max(0.0, level.size)
        for level in executable_levels
    )
    best_bid = sorted_bids[0].price if sorted_bids else None
    best_ask = sorted_asks[0].price if sorted_asks else None
    return LivePreSubmitQuote(
        market_status=market_status,
        book_age_ms=max(0.0, (now - book_ts).total_seconds() * 1000.0),
        executable_notional_usdc=executable_notional_usdc,
        best_executable_price=best_executable_price,
        spread_bps=_spread_bps(best_bid=best_bid, best_ask=best_ask),
        quote_hash=quote_hash,
        book_ts=book_ts,
        source=source,
    )


def _raise_if_future_book_ts(
    book_ts: datetime,
    *,
    now: datetime,
    allowed_clock_skew_ms: float,
) -> None:
    skew_ms = (book_ts - now).total_seconds() * 1000.0
    if skew_ms <= allowed_clock_skew_ms:
        return
    msg = (
        "Polymarket book snapshot timestamp is in the future; "
        f"refusing live submit ({skew_ms:.0f}ms > {allowed_clock_skew_ms:.0f}ms)"
    )
    raise LiveTradingDisabledError(msg)


def _requires_direct_quote(
    order: PolymarketOrderRequest,
    settings: PMSSettings,
) -> bool:
    threshold = settings.controller.direct_quote_min_notional_usdc
    return threshold is not None and order.notional_usdc >= threshold


def _validate_dual_quote_match(
    snapshot_quote: LivePreSubmitQuote,
    direct_quote: LivePreSubmitQuote,
    settings: PMSSettings,
) -> None:
    delta_bps = _price_delta_bps(
        snapshot_quote.best_executable_price,
        direct_quote.best_executable_price,
    )
    if delta_bps > settings.controller.dual_quote_max_price_delta_bps:
        msg = (
            "Polymarket dual quote mismatch: best executable price delta "
            f"{delta_bps:.1f}bps > "
            f"{settings.controller.dual_quote_max_price_delta_bps:.1f}bps"
        )
        raise LiveTradingDisabledError(msg)


def _price_delta_bps(first: float, second: float) -> float:
    midpoint = (abs(first) + abs(second)) / 2.0
    if midpoint <= 0.0:
        return math.inf if first != second else 0.0
    return (abs(first - second) / midpoint) * 10_000.0


def _spread_bps(*, best_bid: float | None, best_ask: float | None) -> float:
    if best_bid is None or best_ask is None:
        return math.inf
    midpoint = (best_bid + best_ask) / 2.0
    if midpoint <= 0.0:
        return math.inf
    return ((best_ask - best_bid) / midpoint) * 10_000.0


def _decision_quantity(
    decision: TradeDecision,
    *,
    notional: float | None = None,
    price: float | None = None,
) -> float:
    normalized_notional = validate_positive(
        notional if notional is not None else decision.notional_usdc,
        field_name="PolymarketOrderRequest.notional_usdc",
    )
    normalized_price = validate_price(
        price if price is not None else decision.limit_price,
        field_name="PolymarketOrderRequest.price",
    )
    quantity = normalized_notional / normalized_price
    return validate_positive(quantity, field_name="PolymarketOrderRequest.size")


def _sdk_order_size(
    decision: TradeDecision,
    *,
    estimated_quantity: float,
) -> float:
    if decision.order_type.lower() == "market" and decision.side == "BUY":
        return decision.notional_usdc
    return estimated_quantity


def _approval_payload_matches(
    payload: dict[str, object],
    preview: LiveOrderPreview,
) -> bool:
    if payload.get("approved") is not True:
        return False
    if payload.get("venue") != preview.venue:
        return False
    if payload.get("market_id") != preview.market_id:
        return False
    if payload.get("token_id") != preview.token_id:
        return False
    if payload.get("side") != preview.side:
        return False
    if payload.get("outcome") != preview.outcome:
        return False
    if payload.get("max_slippage_bps") != preview.max_slippage_bps:
        return False
    approved_notional = _numeric_approval_value(payload.get("max_notional_usdc"))
    if approved_notional is None:
        return False
    notional_tolerance = _approval_tolerance(approved_notional)
    if preview.side.upper() == "BUY":
        if not math.isclose(
            preview.max_notional_usdc,
            approved_notional,
            rel_tol=0.0,
            abs_tol=min(0.0005, notional_tolerance),
        ):
            return False
    elif preview.max_notional_usdc > approved_notional + notional_tolerance:
        return False
    approved_price = _numeric_approval_value(payload.get("limit_price"))
    if approved_price is None:
        return False
    price_tolerance = _approval_tolerance(approved_price)
    if preview.side.upper() == "BUY":
        if preview.limit_price > approved_price + price_tolerance:
            return False
    elif preview.limit_price < approved_price - price_tolerance:
        return False
    return True


def _numeric_approval_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _approval_tolerance(expected: float) -> float:
    # One Polymarket cent tick plus a small representation tolerance. The
    # approval remains a one-sided ceiling/floor and cannot authorize a wider
    # move or resize than this bounded mark window.
    return max(0.0005, min(0.01, abs(expected) * 0.01))


def _approval_payload_hash(payload: Mapping[str, object]) -> str:
    return _canonical_json_fingerprint(payload)


def _looks_like_placeholder(value: str) -> bool:
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


def _canonical_json_fingerprint(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _build_sdk_client(sdk: object, credentials: VenueCredentials) -> object:
    api_creds = getattr(sdk, "ApiCreds")(
        api_key=_required_secret(credentials.api_key, "api_key"),
        api_secret=_required_secret(credentials.api_secret, "api_secret"),
        api_passphrase=_required_secret(
            credentials.api_passphrase,
            "api_passphrase",
        ),
    )
    return getattr(sdk, "ClobClient")(
        host=credentials.host,
        chain_id=credentials.chain_id,
        key=_required_secret(credentials.private_key, "private_key"),
        creds=api_creds,
        signature_type=_sdk_signature_type(sdk, credentials.signature_type),
        funder=credentials.funder_address,
    )


def _post_sdk_order(
    sdk: object,
    client: object,
    order: PolymarketOrderRequest,
) -> object:
    order_type = _sdk_order_type(getattr(sdk, "OrderType"), order)
    options = getattr(sdk, "PartialCreateOrderOptions")()
    if order.order_type.lower() == "market":
        order_args = getattr(sdk, "MarketOrderArgs")(
            token_id=order.token_id,
            amount=order.size,
            side=_sdk_side(getattr(sdk, "Side"), order.side),
            price=order.price,
            order_type=order_type,
        )
        return getattr(client, "create_and_post_market_order")(
            order_args=order_args,
            options=options,
            order_type=order_type,
        )

    order_args = getattr(sdk, "OrderArgs")(
        token_id=order.token_id,
        price=order.price,
        side=_sdk_side(getattr(sdk, "Side"), order.side),
        size=order.size,
    )
    return getattr(client, "create_and_post_order")(
        order_args=order_args,
        options=options,
        order_type=order_type,
    )


def _get_sdk_order_book(client: object, token_id: str) -> object:
    for method_name in ("get_order_book", "get_orderbook", "get_book"):
        method = getattr(client, method_name, None)
        if not callable(method):
            continue
        try:
            return method(token_id)
        except TypeError:
            return method(token_id=token_id)
    msg = "Polymarket SDK client does not expose an order-book fetch method"
    raise LiveTradingDisabledError(msg)


def _get_sdk_open_orders(client: object) -> object:
    for method_name in ("get_orders", "get_open_orders", "getOpenOrders"):
        method = getattr(client, method_name, None)
        if callable(method):
            return method()
    return ()


def _get_sdk_positions(client: object, credentials: VenueCredentials) -> object:
    for method_name in ("get_positions", "get_current_positions", "getPositions"):
        method = getattr(client, method_name, None)
        if not callable(method):
            continue
        try:
            return method(user=credentials.funder_address)
        except TypeError:
            try:
                return method(credentials.funder_address)
            except TypeError:
                return method()
    return ()


def _get_sdk_balances(
    client: object,
    sdk: object,
    credentials: VenueCredentials,
) -> dict[str, float]:
    _sync_sdk_balance_allowance(client, sdk, credentials)
    for response in _sdk_balance_responses(client, sdk, credentials):
        balances = _collateral_balances_from_response(response)
        if balances:
            return balances
    return {}


def _sync_sdk_balance_allowance(
    client: object,
    sdk: object,
    credentials: VenueCredentials,
) -> None:
    for method_name in ("update_balance_allowance", "updateBalanceAllowance"):
        method = getattr(client, method_name, None)
        if callable(method):
            _call_sdk_collateral_balance_allowance(method, sdk, credentials)
            return
    msg = "Polymarket SDK client does not expose balance allowance sync"
    raise LiveTradingDisabledError(msg)


def _sdk_balance_responses(
    client: object,
    sdk: object,
    credentials: VenueCredentials,
) -> list[object]:
    responses: list[object] = []
    balance_allowance = getattr(client, "get_balance_allowance", None)
    if callable(balance_allowance):
        responses.append(
            _call_sdk_collateral_balance_allowance(
                balance_allowance,
                sdk,
                credentials,
            )
        )
    for method_name in ("get_balance", "get_balances", "getBalance", "getBalances"):
        method = getattr(client, method_name, None)
        if callable(method):
            try:
                responses.append(method())
            except TypeError:
                pass
    return responses


def _call_sdk_collateral_balance_allowance(
    method: Callable[..., object],
    sdk: object,
    credentials: VenueCredentials,
) -> object:
    params_cls = getattr(sdk, "BalanceAllowanceParams", None)
    asset_type_cls = getattr(sdk, "AssetType", None)
    asset_type = getattr(asset_type_cls, "COLLATERAL", "COLLATERAL")
    params_kwargs: dict[str, object] = {"asset_type": asset_type}
    if credentials.signature_type == 3:
        params_kwargs["signature_type"] = _sdk_signature_type(
            sdk,
            credentials.signature_type,
        )
    if callable(params_cls):
        try:
            params = params_cls(**params_kwargs)
            return method(params=params)
        except TypeError:
            try:
                params = params_cls(**params_kwargs)
                return method(params)
            except TypeError:
                pass
    try:
        return method(**params_kwargs)
    except TypeError:
        return method()


def _sdk_signature_type(sdk: object, signature_type: int | None) -> object:
    if signature_type is None:
        return None
    signature_type_cls = getattr(sdk, "SignatureTypeV2", None)
    enum_names_by_signature_type = {
        0: ("EOA",),
        1: ("POLY_PROXY",),
        2: ("GNOSIS_SAFE",),
        3: ("POLY_1271",),
    }
    for enum_name in enum_names_by_signature_type.get(signature_type, ()):
        value = getattr(signature_type_cls, enum_name, None)
        if value is not None:
            return value
    return signature_type


def _collateral_balances_from_response(response: object) -> dict[str, float]:
    direct = _coerce_float_or_none(response)
    if direct is not None:
        return {"PUSD": direct}
    if isinstance(response, Sequence) and not isinstance(response, (str, bytes)):
        for item in response:
            asset = _response_value(item, "asset", "asset_type", "currency", "token")
            balances = _collateral_balance_payload(
                item,
                asset_name=None if asset is None else str(asset),
            )
            if balances:
                return balances
        return {}
    return _collateral_balance_payload(response, asset_name=None)


def _collateral_balance_payload(
    response: object,
    *,
    asset_name: str | None,
) -> dict[str, float]:
    normalized_asset = "" if asset_name is None else asset_name.strip().upper()
    if normalized_asset not in {"", "PUSD", "USDC", "COLLATERAL"}:
        return {}

    balance = _coerce_float_or_none(
        _response_value(
            response,
            "balance",
            "available",
            "available_balance",
            "cash",
            "pusd",
            "usdc",
            "collateral",
        )
    )
    allowance = _coerce_float_or_none(
        _response_value(
            response,
            "allowance",
            "available_allowance",
            "collateral_allowance",
            "pusd_allowance",
            "approved",
            "approval",
        )
    )
    balances: dict[str, float] = {}
    if balance is not None:
        balances["USDC" if normalized_asset == "USDC" else "PUSD"] = balance
    if allowance is not None:
        balances["PUSD_ALLOWANCE"] = allowance
    return balances


def _sdk_order_type(order_type_cls: object, order: PolymarketOrderRequest) -> object:
    # Polymarket SDK enum: GTC (rest), FAK (fill-and-kill = IOC), FOK (all-or-nothing),
    # GTD (good-till-date). PMS TimeInForce maps GTC→GTC, IOC→FAK, FOK→FOK
    # symmetrically across market and limit orders. Without explicit IOC/FOK
    # mapping for limit orders, an IOC limit silently rests in the book as
    # GTC — a real behaviour change.
    if order.time_in_force == "IOC":
        return getattr(order_type_cls, "FAK")
    if order.time_in_force == "FOK":
        return getattr(order_type_cls, "FOK")
    if order.order_type.lower() == "market":
        # Default for market orders without an explicit TIF is FOK
        # (all-or-nothing) — the SDK's own MarketOrderArgs default.
        return getattr(order_type_cls, "FOK")
    return getattr(order_type_cls, "GTC")


def _sdk_side(side_cls: object, side: str) -> object:
    return getattr(side_cls, side)


def _order_result_from_sdk_response(
    order: PolymarketOrderRequest,
    response: object,
) -> PolymarketOrderResult:
    if _sdk_response_success_is_false(response) or _response_value(
        response,
        "errorMsg",
        "error_msg",
    ):
        msg = "Polymarket live order rejected by venue; venue error redacted"
        raise LiveTradingDisabledError(msg)

    raw_status = str(
        _response_value(response, "status") or OrderStatus.LIVE.value
    ).strip().lower()
    status = _canonical_polymarket_order_status(raw_status)

    # Parse explicit partial-fill data from the SDK response. Polymarket
    # can return positive `filled_notional` with status != MATCHED (e.g.
    # an IOC limit that filled half its size and cancelled the rest, or
    # a GTC limit with a partial match still resting in the book). Field
    # names vary across SDK versions; we accept any of the documented
    # aliases and fall back to the status-based heuristic when none are
    # present. Pre-fix, only MATCHED produced fill data — every partial
    # fill was silently dropped.
    # Track raw key presence and coerced value separately. A
    # `_coerce_float_or_none` of "nan"/"inf"/null returns None, but the
    # field IS still in the venue response — we must NOT treat that as
    # "no explicit data" (which would trigger the matched-fallback
    # full-fill synthesis). Reject unparseable values up front.
    notional_field_present = _response_field_present(
        response, "filled_notional_usdc", "filledNotional", "filled_amount"
    )
    raw_notional = _response_value(
        response, "filled_notional_usdc", "filledNotional", "filled_amount"
    )
    explicit_filled_notional = _coerce_float_or_none(raw_notional)
    if notional_field_present and explicit_filled_notional is None:
        msg = (
            "Polymarket reported unparseable filled_notional "
            "(non-finite or null); refusing to persist suspect fill"
        )
        raise LiveTradingDisabledError(msg)

    # `fill_count` is the venue-side contract count alias documented at
    # `docs/research/schema-spec.md:284,307` — order-response contract
    # count used for order-state reconciliation. Including it here
    # closes the bypass codex called out in Round 7 finding f14: a
    # response with `fill_count: "nan"` was previously not seen by the
    # raw-presence check and therefore could route through the matched
    # full-fill synthesis path. NOTE: `price` is intentionally NOT
    # aliased to fill_price — in `py_clob_client_v2.OrderArgs`/
    # `MarketOrderArgs`, `price` is the *request* price, not the fill
    # price. Aliasing it would conflate request and execution data.
    quantity_field_present = _response_field_present(
        response, "filled_quantity", "filledQuantity", "filled_size", "filled", "fill_count"
    )
    raw_quantity = _response_value(
        response, "filled_quantity", "filledQuantity", "filled_size", "filled", "fill_count"
    )
    explicit_filled_quantity = _coerce_float_or_none(raw_quantity)
    if quantity_field_present and explicit_filled_quantity is None:
        msg = (
            "Polymarket reported unparseable filled_quantity "
            "(non-finite or null); refusing to persist suspect fill"
        )
        raise LiveTradingDisabledError(msg)

    price_field_present = _response_field_present(
        response, "fill_price", "fillPrice", "average_price", "avg_price"
    )
    raw_price = _response_value(
        response, "fill_price", "fillPrice", "average_price", "avg_price"
    )
    explicit_fill_price = _coerce_float_or_none(raw_price)
    if price_field_present and explicit_fill_price is None:
        msg = (
            "Polymarket reported unparseable fill_price "
            "(non-finite or null); refusing to persist suspect fill"
        )
        raise LiveTradingDisabledError(msg)

    filled_notional_usdc: float
    filled_quantity: float
    fill_price: float | None

    # Validate every non-None explicit fill field BEFORE branching. A
    # malformed venue response must not be able to corrupt accounting
    # via the matched fallback (which previously trusted explicit values
    # without checking) — a status="matched" response with negative
    # filled_quantity or fill_price=1.5 used to slip through.
    _validate_explicit_fill_fields(
        order=order,
        notional=explicit_filled_notional,
        quantity=explicit_filled_quantity,
        price=explicit_fill_price,
    )

    # Use *raw* presence here, not coerced. A field that arrived but
    # failed to coerce was already rejected above; this flag is for
    # legitimately-set but zero-valued fields (e.g. resting limit with
    # filled_notional=0).
    explicit_field_present = (
        notional_field_present or quantity_field_present or price_field_present
    )

    if explicit_filled_notional is not None and explicit_filled_notional > 0.0:
        # Partial-or-full fill with explicit data from the venue.
        # A positive filled_notional MUST imply positive filled_quantity —
        # a fill of $4 worth of zero shares is contradictory and would
        # corrupt portfolio share accounting (`_fill_from_order` would
        # persist a row with shares=0). Resolve in two passes: first
        # determine fill_price (from explicit or fall back to limit),
        # then derive quantity from notional/price when the venue
        # omitted it; reject explicit-zero quantity with positive
        # notional outright.
        filled_notional_usdc = explicit_filled_notional
        if explicit_fill_price is not None:
            fill_price = explicit_fill_price
        elif (
            explicit_filled_quantity is not None
            and explicit_filled_quantity > 0.0
        ):
            implied_price = filled_notional_usdc / explicit_filled_quantity
            if not (
                _PROBABILITY_PRICE_MIN < implied_price <= _PROBABILITY_PRICE_MAX
            ):
                msg = (
                    "Polymarket implied fill_price (notional/quantity) "
                    "outside (0, 1] range; refusing to persist suspect fill"
                )
                raise LiveTradingDisabledError(msg)
            fill_price = implied_price
        else:
            # Neither explicit price nor quantity available — fall back
            # to the limit price (best estimate). Range-valid by the
            # `TradeDecision.__post_init__` invariant `0 < limit_price < 1`.
            fill_price = order.price
        if explicit_filled_quantity is not None:
            if explicit_filled_quantity == 0.0:
                msg = (
                    "Polymarket reported filled_notional > 0 with explicit "
                    "filled_quantity == 0; refusing to persist suspect fill"
                )
                raise LiveTradingDisabledError(msg)
            filled_quantity = explicit_filled_quantity
        elif fill_price > 0.0:
            # Derive quantity from notional/price so share accounting
            # remains consistent across all venue response shapes.
            filled_quantity = filled_notional_usdc / fill_price
        else:
            # Defensive: should be unreachable given the price-resolution
            # logic above, but raise rather than persist a bad fill.
            msg = (
                "Polymarket fill price resolved to zero; cannot derive "
                "filled_quantity for positive notional"
            )
            raise LiveTradingDisabledError(msg)
    elif status == OrderStatus.MATCHED.value and not explicit_field_present:
        # Backwards-compat: status=matched WITH NO EXPLICIT FILL DATA
        # implies full fill at limit price. The full-fill heuristic must
        # only fire when the venue offered no fill fields at all —
        # otherwise a partial response (e.g. status=matched with explicit
        # filled_notional=0, or with explicit_quantity=0 + price set)
        # would silently synthesize a fake $10 / 0-share fill.
        filled_notional_usdc = order.notional_usdc
        filled_quantity = order.estimated_quantity
        fill_price = order.price
    elif status == OrderStatus.MATCHED.value:
        # Status=matched but the explicit fields disagree with a full
        # fill (zero notional, or partial fields without positive
        # notional). The partial-fill branch above only fires for
        # `explicit_filled_notional > 0`, so reaching here means the
        # response is contradictory. Refuse to fabricate a fill.
        msg = (
            "Polymarket reported status=matched with inconsistent "
            "explicit fill fields (zero notional or partial data); "
            "refusing to persist contradictory fill"
        )
        raise LiveTradingDisabledError(msg)
    else:
        # Live / pending / cancelled with no positive fill — pass
        # through any explicit zeros from the venue without synthesis.
        filled_notional_usdc = 0.0
        filled_quantity = (
            explicit_filled_quantity if explicit_filled_quantity is not None else 0.0
        )
        fill_price = explicit_fill_price

    remaining_notional_usdc = max(0.0, order.notional_usdc - filled_notional_usdc)

    order_id = _sdk_response_order_id(response)
    return PolymarketOrderResult(
        order_id=order_id,
        status=status,
        raw_status=raw_status,
        filled_notional_usdc=filled_notional_usdc,
        remaining_notional_usdc=remaining_notional_usdc,
        fill_price=fill_price,
        filled_quantity=filled_quantity,
        fees=_response_fee(response),
        fee_bps=_response_fee_bps(response),
        transaction_ref=_response_transaction_ref(response),
    )


def _canonical_polymarket_order_status(status: str) -> str:
    normalized = status.strip().lower()
    if normalized == "open":
        return OrderStatus.LIVE.value
    if normalized == "canceled":
        return OrderStatus.CANCELLED.value
    return normalized


def _sdk_response_order_id(response: object) -> str:
    raw_order_id = _response_value(response, "orderID", "order_id", "id")
    order_id = str(raw_order_id).strip() if raw_order_id is not None else ""
    if order_id != "" and not _looks_like_placeholder(order_id):
        return order_id
    msg = (
        "Polymarket live order response missing concrete venue order id; "
        "order status is unknown — reconcile with venue before retrying"
    )
    raise PolymarketSubmissionUnknownError(msg)


def _response_fee(response: object) -> float | None:
    raw = _response_value(response, "fees", "fee", "fee_usdc", "feeAmount")
    if raw is None:
        return None
    try:
        value = validate_finite(raw, field_name="venue fee")
    except ValueError as exc:
        raise PolymarketSubmissionUnknownError(
            "Polymarket live order response has invalid fee; order status is "
            "unknown — reconcile with venue before retrying"
        ) from exc
    if value < 0.0:
        raise PolymarketSubmissionUnknownError(
            "Polymarket live order response has negative fee; order status is "
            "unknown — reconcile with venue before retrying"
        )
    return value


def _response_fee_bps(response: object) -> int | None:
    raw = _response_value(response, "fee_bps", "feeBps")
    if raw is None:
        return None
    try:
        value = validate_finite(raw, field_name="venue fee_bps")
    except ValueError as exc:
        raise PolymarketSubmissionUnknownError(
            "Polymarket live order response has invalid fee_bps; order status is "
            "unknown — reconcile with venue before retrying"
        ) from exc
    if value < 0.0:
        raise PolymarketSubmissionUnknownError(
            "Polymarket live order response has negative fee_bps; order status is "
            "unknown — reconcile with venue before retrying"
        )
    return int(value)


def _response_transaction_ref(response: object) -> str | None:
    raw = _response_value(response, "transaction_ref", "transactionHash", "tx_hash")
    if raw is None:
        return None
    normalized = str(raw).strip()
    return normalized or None


def _sdk_response_success_is_false(response: object) -> bool:
    if not _response_field_present(response, "success"):
        return False
    success = _coerce_bool_or_none(_response_value(response, "success"))
    if success is None:
        msg = (
            "Polymarket live order response has unparseable success flag; "
            "venue error redacted"
        )
        raise LiveTradingDisabledError(msg)
    return success is False


def _venue_book_from_sdk_response(
    response: object,
    *,
    market_id: str,
    token_id: str,
) -> LiveVenueBook:
    book_ts = _require_venue_book_timestamp(
        _response_value(response, "timestamp", "ts", "created_at", "createdAt")
    )
    raw_hash = _response_value(response, "hash", "book_hash", "bookHash")
    quote_hash = str(raw_hash or f"venue:{token_id}:{book_ts.isoformat()}")
    market_status = _venue_market_status_from_response(response)
    return LiveVenueBook(
        market_id=market_id,
        token_id=token_id,
        bids=tuple(
            _book_levels_from_response_side(
                _response_value(response, "bids", "buy"),
                market_id=market_id,
                side="BUY",
            )
        ),
        asks=tuple(
            _book_levels_from_response_side(
                _response_value(response, "asks", "sell"),
                market_id=market_id,
                side="SELL",
            )
        ),
        book_ts=book_ts,
        quote_hash=quote_hash,
        market_status=market_status,
    )


def _book_levels_from_response_side(
    raw_levels: object,
    *,
    market_id: str,
    side: Literal["BUY", "SELL"],
) -> list[BookLevel]:
    if raw_levels is None:
        return []
    if not isinstance(raw_levels, Sequence) or isinstance(raw_levels, (str, bytes)):
        msg = "Polymarket order book levels must be a sequence"
        raise LiveTradingDisabledError(msg)
    levels: list[BookLevel] = []
    for raw_level in raw_levels:
        price = _coerce_float_or_none(_response_value(raw_level, "price", "p"))
        size = _coerce_float_or_none(_response_value(raw_level, "size", "s"))
        if price is None or size is None:
            msg = "Polymarket order book level has unparseable price or size"
            raise LiveTradingDisabledError(msg)
        if not (_PROBABILITY_PRICE_MIN < price <= _PROBABILITY_PRICE_MAX):
            msg = "Polymarket order book level price is outside (0, 1]"
            raise LiveTradingDisabledError(msg)
        if size <= 0.0:
            msg = "Polymarket order book level size must be positive"
            raise LiveTradingDisabledError(msg)
        levels.append(
            BookLevel(
                snapshot_id=0,
                market_id=market_id,
                side=side,
                price=price,
                size=size,
            )
        )
    return levels


def _venue_market_status_from_response(response: object) -> str:
    closed = _venue_bool_flag(response, ("closed", "is_closed"), label="closed")
    if closed is True:
        return "closed"
    active = _venue_bool_flag(response, ("active", "is_active"), label="active")
    if active is False:
        return "inactive"
    accepting_orders = _venue_bool_flag(
        response,
        ("accepting_orders", "acceptingOrders"),
        label="accepting_orders",
    )
    if accepting_orders is False:
        return "not_accepting_orders"
    raw_status = _response_value(response, "market_status", "status")
    if raw_status is None:
        return "open"
    normalized = str(raw_status).strip().lower()
    return normalized or "open"


def _venue_bool_flag(
    response: object,
    keys: tuple[str, ...],
    *,
    label: str,
) -> bool | None:
    if not _response_field_present(response, *keys):
        return None
    parsed = _coerce_bool_or_none(_response_value(response, *keys))
    if parsed is None:
        msg = f"Polymarket order book status flag {label} is invalid"
        raise LiveTradingDisabledError(msg)
    return parsed


def _coerce_bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    return None


def _require_venue_book_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        raw = float(value)
        if not math.isfinite(raw):
            msg = "Polymarket order book timestamp is invalid"
            raise LiveTradingDisabledError(msg)
        if raw > 10_000_000_000:
            raw = raw / 1000.0
        return datetime.fromtimestamp(raw, tz=UTC)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            msg = "Polymarket order book timestamp is invalid"
            raise LiveTradingDisabledError(msg)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    msg = "Polymarket order book timestamp is missing"
    raise LiveTradingDisabledError(msg)


def _order_states_from_open_orders(
    raw_orders: object,
    *,
    credentials: VenueCredentials,
) -> list[OrderState]:
    raw_order_rows = _account_collection_sequence(
        raw_orders,
        label="open orders",
        wrapper_keys=("orders", "open_orders", "openOrders", "data", "results"),
        item_identity_keys=("order_id", "id", "orderID"),
    )
    states: list[OrderState] = []
    for raw_order in raw_order_rows:
        now = _utc_now()
        order_id = str(_response_value(raw_order, "order_id", "id", "orderID") or "").strip()
        if order_id == "" or _looks_like_placeholder(order_id):
            msg = "Polymarket venue open order missing concrete order id"
            raise LiveTradingDisabledError(msg)
        market_id = str(
            _response_value(raw_order, "market_id", "condition_id", "market") or ""
        )
        token_id_value = _response_value(raw_order, "token_id", "asset_id", "assetId")
        raw_remaining = _response_value(
            raw_order,
            "remaining_notional_usdc",
            "remaining",
            "size",
        )
        remaining = _coerce_float_or_none(raw_remaining)
        if remaining is None:
            msg = "Polymarket venue open order has invalid remaining notional"
            raise LiveTradingDisabledError(msg)
        if remaining < 0.0:
            msg = "Polymarket venue open order has negative remaining notional"
            raise LiveTradingDisabledError(msg)
        price = _coerce_float_or_none(_response_value(raw_order, "price"))
        states.append(
            OrderState(
                order_id=order_id,
                decision_id=f"venue-open-{order_id}",
                status=OrderStatus.LIVE.value,
                market_id=market_id or "unknown",
                token_id=None if token_id_value is None else str(token_id_value),
                venue=credentials.venue,
                requested_notional_usdc=remaining,
                filled_notional_usdc=0.0,
                remaining_notional_usdc=remaining,
                fill_price=price,
                submitted_at=now,
                last_updated_at=now,
                raw_status=str(_response_value(raw_order, "status") or "open"),
                strategy_id="venue",
                strategy_version_id="venue",
            )
        )
    return states


def _positions_from_sdk_positions(
    raw_positions: object,
    *,
    credentials: VenueCredentials,
) -> list[Position]:
    raw_position_rows = _account_collection_sequence(
        raw_positions,
        label="positions",
        wrapper_keys=("positions", "data", "results"),
        item_identity_keys=("shares", "size", "quantity", "balance"),
    )
    positions: list[Position] = []
    for raw_position in raw_position_rows:
        raw_shares = _response_value(raw_position, "shares", "size", "quantity", "balance")
        shares = _coerce_float_or_none(raw_shares)
        if shares is None:
            msg = "Polymarket venue position has invalid shares"
            raise LiveTradingDisabledError(msg)
        if shares < 0.0:
            msg = "Polymarket venue position has negative shares"
            raise LiveTradingDisabledError(msg)
        if shares == 0.0:
            continue
        market_id = str(
            _response_value(raw_position, "market_id", "condition_id", "market") or ""
        )
        token_id_value = _response_value(raw_position, "token_id", "asset_id", "assetId")
        avg_price = _optional_position_price(
            raw_position,
            ("avg_entry_price", "avgPrice", "price"),
            label="avg_entry_price",
        )
        current_price = _optional_position_price(
            raw_position,
            ("current_price", "curPrice"),
            label="current_price",
        )
        entry_price = avg_price if avg_price is not None else current_price
        if entry_price is None:
            msg = "Polymarket venue position missing price basis"
            raise LiveTradingDisabledError(msg)
        positions.append(
            Position(
                market_id=market_id or "unknown",
                token_id=None if token_id_value is None else str(token_id_value),
                venue=credentials.venue,
                side=str(_response_value(raw_position, "side", "outcome") or "BUY"),
                shares_held=shares,
                avg_entry_price=entry_price,
                unrealized_pnl=0.0,
                locked_usdc=shares * entry_price,
            )
        )
    return positions


def _optional_position_price(
    raw_position: object,
    keys: tuple[str, ...],
    *,
    label: str,
) -> float | None:
    if not _response_field_present(raw_position, *keys):
        return None
    price = _coerce_float_or_none(_response_value(raw_position, *keys))
    if price is None or not (_PROBABILITY_PRICE_MIN < price <= _PROBABILITY_PRICE_MAX):
        msg = f"Polymarket venue position has invalid {label}"
        raise LiveTradingDisabledError(msg)
    return price


def _account_collection_sequence(
    raw_collection: object,
    *,
    label: str,
    wrapper_keys: tuple[str, ...],
    item_identity_keys: tuple[str, ...],
) -> Sequence[object]:
    if raw_collection is None:
        return ()
    collection = raw_collection
    for _ in range(3):
        if not isinstance(collection, Mapping):
            break
        mapping = cast(Mapping[str, object], collection)
        for key in wrapper_keys:
            if key in mapping:
                collection = mapping[key]
                break
        else:
            if any(key in mapping for key in item_identity_keys):
                return (mapping,)
            msg = f"Polymarket account {label} response has unsupported object shape"
            raise LiveTradingDisabledError(msg)
    if isinstance(collection, Sequence) and not isinstance(collection, (str, bytes)):
        return collection
    msg = f"Polymarket account {label} response must be a sequence"
    raise LiveTradingDisabledError(msg)


def _compare_positions(
    db_positions: Sequence[Position],
    venue_positions: Sequence[Position],
    *,
    share_tolerance: float,
    notional_tolerance: float,
) -> list[str]:
    mismatches: list[str] = []
    db_by_key = _aggregate_positions_by_contract(db_positions)
    venue_by_key = _aggregate_positions_by_contract(venue_positions)
    for key in sorted(set(db_by_key) | set(venue_by_key)):
        db_position = db_by_key.get(key)
        venue_position = venue_by_key.get(key)
        if db_position is None:
            mismatches.append(f"venue position missing from DB: {key}")
            continue
        if venue_position is None:
            mismatches.append(f"DB position missing from venue: {key}")
            continue
        if abs(db_position.shares_held - venue_position.shares_held) > share_tolerance:
            mismatches.append(
                "position shares mismatch "
                f"{key}: DB={db_position.shares_held:.8f} "
                f"venue={venue_position.shares_held:.8f}"
            )
            mismatches.extend(
                _strategy_position_residuals(
                    db_positions,
                    key,
                    venue_position.shares_held,
                    share_tolerance,
                )
            )
        if abs(db_position.locked_usdc - venue_position.locked_usdc) > notional_tolerance:
            mismatches.append(
                "position notional mismatch "
                f"{key}: DB={db_position.locked_usdc:.8f} "
                f"venue={venue_position.locked_usdc:.8f}"
            )
            mismatches.extend(
                _strategy_position_residuals(
                    db_positions,
                    key,
                    venue_position.shares_held,
                    share_tolerance,
                )
            )
    return mismatches


def _strategy_position_residuals(
    db_positions: Sequence[Position],
    key: tuple[str, str | None, str],
    venue_shares: float,
    tolerance: float,
) -> list[str]:
    rows = [position for position in db_positions if _position_key(position) == key]
    total = sum(position.shares_held for position in rows)
    if len(rows) <= 1 or total <= tolerance:
        return []
    return [
        "strategy position residual "
        f"{position.strategy_id}@{position.strategy_version_id}: "
        f"allocated={venue_shares * position.shares_held / total:.8f} "
        f"db={position.shares_held:.8f}"
        for position in rows
    ]


def _aggregate_positions_by_contract(
    positions: Sequence[Position],
) -> dict[tuple[str, str | None, str], Position]:
    """Aggregate strategy-tagged rows before comparing venue contracts.

    The venue account has no strategy dimension, while the inner-ring ledger
    intentionally does. Summing by contract preserves every strategy's
    exposure in the DB comparison instead of letting the last row overwrite
    earlier versions.
    """
    grouped: dict[tuple[str, str | None, str], Position] = {}
    for position in positions:
        key = _position_key(position)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = position
            continue
        shares = existing.shares_held + position.shares_held
        locked = existing.locked_usdc + position.locked_usdc
        avg_price = (
            0.0
            if shares <= 1e-12
            else (
                existing.avg_entry_price * existing.shares_held
                + position.avg_entry_price * position.shares_held
            )
            / shares
        )
        grouped[key] = replace(
            existing,
            shares_held=shares,
            locked_usdc=locked,
            avg_entry_price=avg_price,
            unrealized_pnl=existing.unrealized_pnl + position.unrealized_pnl,
        )
    return grouped


def _venue_cash_balance(balances: Mapping[str, float]) -> float | None:
    for key, value in balances.items():
        if key.upper() in {"PUSD", "USDC", "COLLATERAL"}:
            return value
    return None


def _venue_pusd_allowance(balances: Mapping[str, float]) -> float | None:
    for key, value in balances.items():
        if key.upper() in {
            "PUSD_ALLOWANCE",
            "COLLATERAL_ALLOWANCE",
            "ALLOWANCE",
        }:
            return value
    return None


def _position_key(position: Position) -> tuple[str, str | None, str]:
    return position.market_id, position.token_id, position.venue


def _is_sdk_transport_failure(exc: BaseException) -> bool:
    """Detect SDK exceptions that represent a transport-level failure
    (no HTTP response from the venue) — distinct from the venue
    rejecting the order with an HTTP error response.

    `py_clob_client_v2.exceptions.PolyApiException` takes `resp` in
    its constructor but does NOT retain it as an attribute — the
    instance stores `status_code` (extracted from `resp.status_code`,
    or None if `resp` was None) plus `error_msg`. So the right
    transport-failure signal is `status_code is None`. Verified against
    real SDK 1.0.0:
        PolyApiException(resp=None).__dict__ ==
            {'status_code': None, 'error_msg': ...}
        PolyApiException(resp=<httpx_response>).__dict__ ==
            {'status_code': 400, 'error_msg': ...}

    Duck-typed (class name + attribute) so this module does not take a
    hard import dependency on the SDK exception class.
    """
    if type(exc).__name__ != "PolyApiException":
        return False
    status_code = getattr(exc, "status_code", _MISSING_SENTINEL)
    return status_code is None


_MISSING_SENTINEL: Final[object] = object()


def _validate_explicit_fill_fields(
    *,
    order: PolymarketOrderRequest,
    notional: float | None,
    quantity: float | None,
    price: float | None,
) -> None:
    """Reject malformed venue-supplied fill fields before any branch
    consumes them. Runs ahead of the matched/partial/no-fill branches so
    a `status=matched` response with `filled_quantity=-8.0` (for
    example) cannot be persisted via the backwards-compat fallback.
    """
    if notional is not None:
        if notional < 0.0:
            msg = (
                "Polymarket reported negative filled_notional; refusing "
                "to persist suspect fill"
            )
            raise LiveTradingDisabledError(msg)
        if notional > order.notional_usdc + _NOTIONAL_OVERFILL_TOLERANCE:
            msg = (
                "Polymarket reported filled_notional exceeds requested "
                "notional; refusing to persist suspect fill"
            )
            raise LiveTradingDisabledError(msg)
    if quantity is not None and quantity < 0.0:
        msg = (
            "Polymarket reported negative filled_quantity; refusing to "
            "persist suspect fill"
        )
        raise LiveTradingDisabledError(msg)
    if price is not None and not (
        _PROBABILITY_PRICE_MIN < price <= _PROBABILITY_PRICE_MAX
    ):
        msg = (
            "Polymarket reported fill_price outside (0, 1] range; "
            "refusing to persist suspect fill"
        )
        raise LiveTradingDisabledError(msg)
    # Cross-field consistency: when the venue surfaces all three of
    # `filled_notional_usdc`, `filled_quantity`, and `fill_price` with
    # positive values, the identity `notional == quantity * price` must
    # hold within rounding tolerance. Without this check a malformed
    # triple like (notional=4, quantity=100, price=0.5) — where the
    # true notional should be 50 — passes individual range validation
    # and is silently persisted, corrupting share accounting.
    #
    # `math.isclose` with rel_tol=1% and abs_tol=$0.01 accommodates
    # legitimate venue rounding (Polymarket CLOB matches at discrete
    # prices) while catching the kind of order-of-magnitude divergence
    # the example above exhibits.
    if (
        notional is not None
        and notional > 0.0
        and quantity is not None
        and quantity > 0.0
        and price is not None
        and price > 0.0
        and not math.isclose(notional, quantity * price, rel_tol=0.01, abs_tol=0.01)
    ):
        expected = quantity * price
        msg = (
            "Polymarket reported inconsistent fill triple "
            f"(notional={notional}, quantity={quantity}, price={price}; "
            f"expected notional≈{expected:.4f}); refusing to persist "
            "suspect fill"
        )
        raise LiveTradingDisabledError(msg)


def _coerce_float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        try:
            f = float(value)
        except ValueError:
            return None
    else:
        return None
    # Reject NaN / +/- infinity. Persisting either as `filled_notional`
    # / `filled_quantity` / `fill_price` would silently corrupt the
    # portfolio and downstream metrics. A malformed venue response that
    # parses to a non-finite number must be treated as "no fill data".
    if not math.isfinite(f):
        return None
    return f


_PROBABILITY_PRICE_MIN: Final[float] = 0.0
_PROBABILITY_PRICE_MAX: Final[float] = 1.0
_NOTIONAL_OVERFILL_TOLERANCE: Final[float] = 1e-6


def _response_value(response: object, *keys: str) -> object | None:
    if isinstance(response, Mapping):
        mapping = cast(Mapping[str, object], response)
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None
    for key in keys:
        value = getattr(response, key, None)
        if value is not None:
            return cast(object, value)
    return None


def _response_field_present(response: object, *keys: str) -> bool:
    """Distinct from `_response_value`: returns True iff the venue
    actually surfaced one of the keys (regardless of whether the value
    parses). Used to detect malformed-but-present fields, which must be
    rejected before the matched-fallback can synthesize a full fill.
    """
    if isinstance(response, Mapping):
        mapping = cast(Mapping[str, object], response)
        return any(key in mapping for key in keys)
    return any(hasattr(response, key) for key in keys)


def _required_secret(value: str | None, field_name: str) -> str:
    if value is None or value.strip() == "":
        msg = f"Missing Polymarket credential fields: {field_name}"
        raise LiveTradingDisabledError(msg)
    return value


def _redacted_exception_message(
    error: Exception,
    credentials: VenueCredentials,
) -> str:
    return redact_live_error_values(
        str(error),
        (
            credentials.private_key,
            credentials.api_key,
            credentials.api_secret,
            credentials.api_passphrase,
            credentials.funder_address,
        ),
    )


def _redacted_settings_exception_message(
    error: Exception,
    settings: PMSSettings,
) -> str:
    return redact_live_error_values(
        str(error),
        (
            settings.polymarket.private_key,
            settings.polymarket.api_key,
            settings.polymarket.api_secret,
            settings.polymarket.api_passphrase,
            settings.polymarket.funder_address,
        ),
    )
