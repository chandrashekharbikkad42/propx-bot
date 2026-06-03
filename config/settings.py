"""Centralized config loader."""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Optional
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

BrokerType = Literal["IC_MARKETS", "ROBOFOREX"]
AccountType = Literal["STANDARD", "PROCENT"]
ExecutionMode = Literal["PAPER", "REAL"]
# Phase 8B — prop firm types. Auto-detection in Phase 8D; for now the user
# pins the active prop firm in .env so the rules engine knows which caps to load.
PropFirmType = Literal[
    "FTMO_1STEP", "FTMO_2STEP", "THE5ERS_BOOTCAMP", "THE5ERS_HRP", "NONE"
]

_VALID_BROKERS: tuple[str, ...] = ("IC_MARKETS", "ROBOFOREX")
_VALID_ACCOUNTS: tuple[str, ...] = ("STANDARD", "PROCENT")
_VALID_EXECUTION_MODES: tuple[str, ...] = ("PAPER", "REAL")
_VALID_PROP_FIRMS: tuple[str, ...] = (
    "FTMO_1STEP", "FTMO_2STEP", "THE5ERS_BOOTCAMP", "THE5ERS_HRP", "NONE",
)

# Phase 8B — default forex universe (28 majors + minors). Bot scans these on
# every 1H bar close. Auto-detection from broker symbols is a Phase 9 feature;
# until then we hardcode to keep 8B decoupled from a live MT5 connection.
DEFAULT_FOREX_PAIRS: tuple[str, ...] = (
    # Majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    # EUR crosses
    "EURJPY", "EURGBP", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
    # GBP crosses
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
    # AUD / NZD / CAD / CHF crosses
    "AUDJPY", "AUDCHF", "AUDNZD", "AUDCAD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY",
)

# Phase 8B — IST trading window defaults. London open (12:30 IST) through NY close.
DEFAULT_IST_WINDOW_START = "12:30"
DEFAULT_IST_WINDOW_END = "22:30"
DEFAULT_TIMEZONE = "Asia/Kolkata"

# Per-session spread ceilings (in POINTS). Phase 7: a single rolling p90 cap
# does not generalize across sessions — Asian needs to stay tight, NY needs
# more headroom. Detectors take min(p90 * mult, session_cap).
SESSION_SPREAD_CAPS: Mapping[str, float] = MappingProxyType({
    "ASIAN": 12.0,
    "LONDON": 10.0,
    "NY": 25.0,
    "LONDON_NY_OVERLAP": 15.0,
    "OFF": 50.0,
})

_VALID_SESSIONS: frozenset[str] = frozenset(SESSION_SPREAD_CAPS.keys())


def _session_caps_from_env() -> Mapping[str, float]:
    raw = os.getenv("SESSION_SPREAD_CAPS_JSON", "").strip()
    if not raw:
        return SESSION_SPREAD_CAPS
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid SESSION_SPREAD_CAPS_JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("SESSION_SPREAD_CAPS_JSON must decode to an object")
    merged: dict[str, float] = dict(SESSION_SPREAD_CAPS)
    for k, v in parsed.items():
        key = str(k).upper()
        if key not in _VALID_SESSIONS:
            raise RuntimeError(
                f"Unknown session key {key!r} in SESSION_SPREAD_CAPS_JSON"
            )
        merged[key] = float(v)
    return MappingProxyType(merged)


def _required(key: str) -> str:
    val = os.getenv(key)
    if not val or not val.strip():
        raise RuntimeError(f"Missing required env var: {key}")
    return val.strip()


def _int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val else default


def _float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val else default


def _enum(key: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = (os.getenv(key) or default).strip().upper()
    if raw not in allowed:
        raise RuntimeError(
            f"Invalid {key}={raw!r}. Allowed: {allowed}"
        )
    return raw


def _bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _opt(key: str) -> Optional[str]:
    val = os.getenv(key)
    if val is None:
        return None
    val = val.strip()
    return val or None


@dataclass(frozen=True)
class Settings:
    mt5_login: int
    mt5_password: str
    mt5_server: str
    mt5_path: str
    mt5_timeout_ms: int
    mt5_retry_attempts: int
    mt5_retry_delay_sec: int

    symbol: str

    broker_type: BrokerType
    account_type: AccountType

    simulated_starting_capital: float
    risk_per_trade_pct: float

    project_root: Path
    log_dir: Path
    data_dir: Path

    log_level: str

    session_spread_caps: Mapping[str, float] = field(
        default_factory=lambda: SESSION_SPREAD_CAPS
    )

    # Phase 8 — production deployment
    execution_mode: ExecutionMode = "PAPER"
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    auto_restart_enabled: bool = True
    max_restart_attempts: int = 5
    restart_backoff_sec: int = 30

    # Phase 8B — Griff prop-firm pivot. All optional with safe defaults so
    # the existing tick-microstructure code path stays untouched.
    prop_firm_type: PropFirmType = "FTMO_1STEP"
    forex_pairs: tuple[str, ...] = field(default_factory=lambda: DEFAULT_FOREX_PAIRS)
    ist_window_start: str = DEFAULT_IST_WINDOW_START  # "HH:MM"
    ist_window_end: str = DEFAULT_IST_WINDOW_END      # "HH:MM"
    timezone: str = DEFAULT_TIMEZONE
    auto_detect_pairs: bool = False
    # bars_dir for 1H parquet bars (separate from tick data_dir)
    bars_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "bars")


def _forex_pairs_from_env() -> tuple[str, ...]:
    """Parse FOREX_PAIRS env var (comma-separated). Falls back to defaults.

    Bhai: kabhi extra pairs add karne ho ya kam karne ho, env me set kar do.
    Example: FOREX_PAIRS=EURUSD,GBPUSD,XAUUSD
    """
    raw = os.getenv("FOREX_PAIRS", "").strip()
    if not raw:
        return DEFAULT_FOREX_PAIRS
    pairs = tuple(p.strip().upper() for p in raw.split(",") if p.strip())
    if not pairs:
        return DEFAULT_FOREX_PAIRS
    return pairs


def _validate_hhmm(label: str, value: str) -> str:
    """Validate HH:MM format. Returns the value if OK; raises otherwise."""
    parts = value.split(":")
    if len(parts) != 2:
        raise RuntimeError(f"Invalid {label}={value!r} — expected HH:MM")
    try:
        h, m = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise RuntimeError(f"Invalid {label}={value!r}: {exc}") from exc
    if not (0 <= h < 24 and 0 <= m < 60):
        raise RuntimeError(f"Invalid {label}={value!r} — out of range")
    return f"{h:02d}:{m:02d}"


def _build() -> Settings:
    log_dir = PROJECT_ROOT / os.getenv("LOG_DIR", "logs")
    data_dir = PROJECT_ROOT / "data" / "ticks"
    bars_dir = PROJECT_ROOT / "data" / "bars"
    log_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    bars_dir.mkdir(parents=True, exist_ok=True)

    # Phase 9 — credentials routed via ACTIVE_BROKER (FTMO / ROBOFOREX).
    # `get_active_credentials()` falls back gracefully if the primary
    # broker's env vars are incomplete, preserving the legacy MT5_* flow.
    from config.broker_config import get_active_credentials
    creds = get_active_credentials()

    return Settings(
        mt5_login=creds.login,
        mt5_password=creds.password,
        mt5_server=creds.server,
        mt5_path=creds.path or _required("MT5_PATH"),
        mt5_timeout_ms=_int("MT5_TIMEOUT_MS", 10000),
        mt5_retry_attempts=_int("MT5_RETRY_ATTEMPTS", 3),
        mt5_retry_delay_sec=_int("MT5_RETRY_DELAY_SEC", 2),
        symbol=os.getenv("SYMBOL", "XAUUSD").strip(),
        broker_type=_enum("BROKER_TYPE", "IC_MARKETS", _VALID_BROKERS),  # type: ignore[arg-type]
        account_type=_enum("ACCOUNT_TYPE", "STANDARD", _VALID_ACCOUNTS),  # type: ignore[arg-type]
        simulated_starting_capital=_float("SIMULATED_STARTING_CAPITAL", 10000.0),
        risk_per_trade_pct=_float("RISK_PER_TRADE_PCT", 0.5),
        project_root=PROJECT_ROOT,
        log_dir=log_dir,
        data_dir=data_dir,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        session_spread_caps=_session_caps_from_env(),
        execution_mode=_enum("EXECUTION_MODE", "PAPER", _VALID_EXECUTION_MODES),  # type: ignore[arg-type]
        telegram_bot_token=_opt("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_opt("TELEGRAM_CHAT_ID"),
        auto_restart_enabled=_bool("AUTO_RESTART_ENABLED", True),
        max_restart_attempts=_int("MAX_RESTART_ATTEMPTS", 5),
        restart_backoff_sec=_int("RESTART_BACKOFF_SEC", 30),
        # Phase 8B — Griff / prop-firm settings
        prop_firm_type=_enum("PROP_FIRM_TYPE", "FTMO_1STEP", _VALID_PROP_FIRMS),  # type: ignore[arg-type]
        forex_pairs=_forex_pairs_from_env(),
        ist_window_start=_validate_hhmm(
            "IST_WINDOW_START",
            os.getenv("IST_WINDOW_START", DEFAULT_IST_WINDOW_START).strip(),
        ),
        ist_window_end=_validate_hhmm(
            "IST_WINDOW_END",
            os.getenv("IST_WINDOW_END", DEFAULT_IST_WINDOW_END).strip(),
        ),
        timezone=os.getenv("TIMEZONE", DEFAULT_TIMEZONE).strip(),
        auto_detect_pairs=_bool("AUTO_DETECT_PAIRS", False),
        bars_dir=bars_dir,
    )


settings = _build()