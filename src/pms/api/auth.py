from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from collections.abc import Callable
from typing import NoReturn

from fastapi import HTTPException, Request, status

from pms.config import PMSSettings


logger = logging.getLogger(__name__)
AUTH_FAILURE_DETAIL = "Missing or invalid API token."
FORBIDDEN_SCOPE_DETAIL = "API token does not have the required scope."
REDACTED_BEARER_TOKEN = "Bearer [REDACTED]"
SCOPES = frozenset({"read", "research", "paper_operate", "kill", "live_admin"})


@dataclass(frozen=True)
class AuthPrincipal:
    token_id: str
    scopes: frozenset[str]


def _redact_authorization_header(value: str | None) -> str | None:
    if value is None:
        return None

    scheme, _, _token = value.partition(" ")
    if scheme.lower() == "bearer":
        return REDACTED_BEARER_TOKEN
    return "[REDACTED]"


def _settings_from_request(request: Request) -> PMSSettings:
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, PMSSettings):
        return settings

    msg = "Application settings are not configured on app.state.settings"
    raise RuntimeError(msg)


def _raise_unauthorized(request: Request) -> NoReturn:
    logger.warning(
        "Rejected API request due to missing or invalid bearer token: method=%s path=%s authorization=%s",
        request.method,
        request.url.path,
        _redact_authorization_header(request.headers.get("Authorization")),
    )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=AUTH_FAILURE_DETAIL,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _raise_forbidden() -> NoReturn:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=FORBIDDEN_SCOPE_DETAIL)


def _principal_for_request(request: Request) -> AuthPrincipal | None:
    settings = _settings_from_request(request)
    authorization = request.headers.get("Authorization")
    if not settings.api_token and not settings.api_tokens:
        # Preserve open local-development mode for non-sensitive routes while
        # refusing to turn an arbitrary bearer value into an operator token.
        # LIVE and kill dependencies convert this missing principal to 401.
        return None
    if authorization is None:
        if settings.api_token or settings.api_tokens:
            _raise_unauthorized(request)
        return None

    scheme, _, presented_token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        _raise_unauthorized(request)

    stripped_token = presented_token.strip()
    if not stripped_token:
        _raise_unauthorized(request)

    if settings.api_token and secrets.compare_digest(stripped_token, settings.api_token):
        return AuthPrincipal("legacy", frozenset(SCOPES))
    for configured in settings.api_tokens:
        try:
            expected = configured.resolved_secret()
        except ValueError:
            # Missing environment secrets fail closed as an invalid token.
            continue
        if secrets.compare_digest(stripped_token, expected):
            return AuthPrincipal(configured.token_id, frozenset(configured.scopes))
    _raise_unauthorized(request)


def authenticate(request: Request) -> AuthPrincipal | None:
    principal = _principal_for_request(request)
    request.state.auth_principal = principal
    request.state.token_id = None if principal is None else principal.token_id
    return principal


def require_api_token(request: Request) -> None:
    """Compatibility dependency: authenticate when a token is configured."""
    authenticate(request)


def require_scope(scope: str) -> Callable[[Request], None]:
    """Return a FastAPI dependency enforcing one API scope."""
    if scope not in SCOPES:
        raise ValueError(f"unknown API scope: {scope}")

    def dependency(request: Request) -> None:
        request.state.auth_scope = scope
        principal = authenticate(request)
        # Unconfigured local development remains open for non-LIVE routes.
        if principal is None:
            # Kill and LIVE control are always authenticated, even when the
            # local API has no configured bearer secret.  These scopes can
            # increase or terminate real-world risk and must never inherit
            # the development open mode.
            if scope in {"kill", "live_admin"}:
                _raise_unauthorized(request)
            return
        if scope not in principal.scopes:
            _raise_forbidden()

    dependency.__name__ = f"require_scope_{scope}"
    return dependency


def require_any_scope(*scopes: str) -> Callable[[Request], None]:
    """Return a dependency accepting any one of ``scopes``."""
    if not scopes or any(scope not in SCOPES for scope in scopes):
        raise ValueError("require_any_scope received an unknown or empty scope list")

    def dependency(request: Request) -> None:
        request.state.auth_scope = ",".join(scopes)
        principal = authenticate(request)
        if principal is None:
            return
        if not principal.scopes.intersection(scopes):
            _raise_forbidden()

    dependency.__name__ = "require_any_scope_" + "_".join(scopes)
    return dependency


def principal(request: Request) -> AuthPrincipal | None:
    """Read the authenticated principal established by an auth dependency."""
    return getattr(request.state, "auth_principal", None)
