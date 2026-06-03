"""Active-broker credential routing.

Single source of truth for "which MT5 account does the bot log into".

Phase 5/B model — generic broker profiles:
  Any number of brokers can be stored in `.env` via the
  `BROKER_<NAME>_LOGIN/PASSWORD/SERVER/PATH/LABEL` convention; the
  `ACTIVE_BROKER=<NAME>` key selects which one is used. The interactive
  prompt in `config.credential_prompt` writes profiles in this format.

Legacy compatibility (Phase 4 .env style):
  Older `.env` files used hardcoded prefixes — `FTMO_LOGIN` etc. for the
  FTMO terminal, `MT5_LOGIN` etc. for the RoboForex / The5%ers terminal.
  Those are still honoured as a fallback so an unmigrated .env continues
  to work; the prompt will rewrite them into the new format on first run.

  ACTIVE_BROKER=<NAME>     → reads BROKER_<NAME>_LOGIN/PASSWORD/SERVER/PATH
  ACTIVE_BROKER=FTMO       → ALSO recognised; falls back to FTMO_LOGIN ...
  ACTIVE_BROKER=ROBOFOREX  → ALSO recognised; falls back to MT5_LOGIN ...
  (unset)                  → first complete profile in .env, then legacy.

If the active profile is incomplete and no fallback works,
`BrokerCredentialsMissing` is raised — caller should invoke the
interactive prompt.

`MT5_PATH` is shared — only one MT5 terminal binary lives on the host.

Hinglish: bot kis account me login karega yeh ek hi jagah pe tay hota
hai. .env me `BROKER_THE5ERS_*` jaise blocks save karke `ACTIVE_BROKER=THE5ERS`
set karo. Purane `FTMO_*` / `MT5_*` style bhi chalega.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional


# Legacy hardcoded broker labels — still recognised as fallback names so
# Phase-4 `.env` files keep working.
_LEGACY_BROKERS: tuple[str, ...] = ("FTMO", "ROBOFOREX")


class BrokerCredentialsMissing(RuntimeError):
    """Raised when required env vars for a broker are absent/blank."""


@dataclass(frozen=True)
class BrokerCredentials:
    broker: str
    login: int
    password: str
    server: str
    path: str  # may be empty if MT5_PATH not set


def _opt_env(key: str) -> Optional[str]:
    v = os.environ.get(key)
    if v is None:
        return None
    s = v.strip()
    return s if s else None


def active_broker_name() -> str:
    """Resolve ACTIVE_BROKER. Returns the raw uppercase profile name.

    No hardcoded enum any more — any value is accepted; the caller is
    responsible for confirming a matching profile exists. Empty / unset
    resolves to the first legacy broker we can find creds for, so a
    Phase-4 `.env` without `ACTIVE_BROKER` still boots.
    """
    raw = (os.environ.get("ACTIVE_BROKER") or "").strip().upper()
    if raw:
        return raw
    # No ACTIVE_BROKER → infer from whichever legacy creds are present.
    for legacy in _LEGACY_BROKERS:
        try:
            _credentials_for(legacy)
            return legacy
        except BrokerCredentialsMissing:
            continue
    # Last resort — return a sentinel; downstream will trigger the prompt.
    return "ROBOFOREX"


def _credentials_for_profile(name: str) -> BrokerCredentials:
    """Read `BROKER_<NAME>_*` profile keys. Raises if incomplete."""
    pfx = f"BROKER_{name.upper()}_"
    login = _opt_env(pfx + "LOGIN")
    password = _opt_env(pfx + "PASSWORD")
    server = _opt_env(pfx + "SERVER")
    path = _opt_env(pfx + "PATH") or _opt_env("MT5_PATH") or ""
    if not (login and password and server):
        missing = [
            k for k, v in (
                ("LOGIN", login), ("PASSWORD", password), ("SERVER", server),
            ) if not v
        ]
        raise BrokerCredentialsMissing(
            f"BROKER_{name}: missing {','.join(missing)} env var(s)"
        )
    try:
        login_int = int(login)
    except ValueError as exc:
        raise BrokerCredentialsMissing(
            f"BROKER_{name}: LOGIN must be integer, got {login!r}"
        ) from exc
    return BrokerCredentials(
        broker=name.upper(), login=login_int, password=password,
        server=server, path=path,
    )


def _credentials_for_legacy(broker: str) -> BrokerCredentials:
    """Legacy Phase-4 prefix groups (`FTMO_*`, `MT5_*`). Kept for back-compat."""
    if broker == "FTMO":
        login = _opt_env("FTMO_LOGIN")
        password = _opt_env("FTMO_PASSWORD")
        server = _opt_env("FTMO_SERVER")
        path = _opt_env("FTMO_PATH") or _opt_env("MT5_PATH") or ""
    elif broker == "ROBOFOREX":
        login = _opt_env("MT5_LOGIN")
        password = _opt_env("MT5_PASSWORD")
        server = _opt_env("MT5_SERVER")
        path = _opt_env("ROBOFOREX_PATH") or _opt_env("MT5_PATH") or ""
    else:
        raise BrokerCredentialsMissing(
            f"No legacy creds path for {broker!r}"
        )
    if not (login and password and server):
        raise BrokerCredentialsMissing(
            f"Legacy {broker}: incomplete credentials"
        )
    try:
        login_int = int(login)
    except ValueError as exc:
        raise BrokerCredentialsMissing(
            f"Legacy {broker}: LOGIN must be integer, got {login!r}"
        ) from exc
    return BrokerCredentials(
        broker=broker, login=login_int, password=password,
        server=server, path=path,
    )


def _credentials_for(broker: str) -> BrokerCredentials:
    """Resolve a broker name → credentials.

    Lookup order:
      1. New `BROKER_<NAME>_*` profile (any user-defined name).
      2. Legacy `FTMO_*` / `MT5_*` fallback if NAME ∈ {FTMO, ROBOFOREX}.
    """
    try:
        return _credentials_for_profile(broker)
    except BrokerCredentialsMissing as primary_exc:
        if broker.upper() in _LEGACY_BROKERS:
            try:
                return _credentials_for_legacy(broker.upper())
            except BrokerCredentialsMissing:
                pass
        raise primary_exc


def get_active_credentials() -> BrokerCredentials:
    """Resolve and return the active broker's credentials.

    Falls back to any other complete profile (new format) and then to the
    legacy known brokers (FTMO / ROBOFOREX). If nothing is configured,
    `BrokerCredentialsMissing` propagates and the CLI should invoke
    `config.credential_prompt.ensure_active_credentials(reset=True)`.
    """
    primary = active_broker_name()
    try:
        return _credentials_for(primary)
    except BrokerCredentialsMissing as exc:
        from utils.logger import logger
        logger.warning(f"broker_config: {exc} — trying fallback profiles")
        # Try every other profile we can detect from env.
        for k in sorted(os.environ):
            if not k.startswith("BROKER_") or not k.endswith("_LOGIN"):
                continue
            name = k[len("BROKER_"):-len("_LOGIN")]
            if name.upper() == primary.upper():
                continue
            try:
                return _credentials_for(name)
            except BrokerCredentialsMissing:
                continue
        # Last resort: try legacy buckets we haven't tried yet.
        for legacy in _LEGACY_BROKERS:
            if legacy == primary.upper():
                continue
            try:
                return _credentials_for(legacy)
            except BrokerCredentialsMissing:
                continue
        raise


def get_credentials_for(broker: str) -> BrokerCredentials:
    """Public test/preflight hook to force a specific broker name."""
    return _credentials_for(broker.upper())
