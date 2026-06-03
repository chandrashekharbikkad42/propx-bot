"""ICT Silver Bullet — strategy constants (standalone candidate test).

Single source of truth for `docs/SILVER_BULLET_SPEC.md`. Isolated from
`config/multi_setup_config.py` so changes here do not leak into the
proven Asian Sweep V5 / propX Multi-Setup bundles.

Reuses pip / contract / spread / SL-buffer maps from `multi_setup_config`
where the data is purely broker-spec (avoids duplicating broker truth),
and adds the 2 missing pairs (CADCHF, XAGUSD) so the full 30-symbol
shortlist is covered.

Hinglish: SB ke saare numbers yahin. Detector me hard-coded value mat
rakhna — yahaan se aaye.
"""

from __future__ import annotations
from types import MappingProxyType
from typing import Mapping, Tuple

from config.multi_setup_config import (
    CONTRACT_SIZE as _MS_CONTRACT_SIZE,
    MAX_SPREAD_PIPS as _MS_MAX_SPREAD,
    PIP_SIZE as _MS_PIP_SIZE,
    SL_BUFFER_PIPS as _MS_SL_BUF,
)

# ─────────────────────────────────────────────────────────────────────────────
# Pair universe — 30 symbols (multi_setup 28 + CADCHF + XAGUSD)
# ─────────────────────────────────────────────────────────────────────────────
PAIRS: tuple[str, ...] = (
    # USD majors (7)
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
    # EUR crosses (6)
    "EURJPY", "EURGBP", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
    # GBP crosses (5)
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
    # AUD crosses (4)
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    # NZD crosses (3)
    "NZDJPY", "NZDCHF", "NZDCAD",
    # JPY / CHF crosses (3) — CADCHF added vs. multi_setup_config
    "CADJPY", "CHFJPY", "CADCHF",
    # Metals (2) — XAGUSD added
    "XAUUSD", "XAGUSD",
)

# ─────────────────────────────────────────────────────────────────────────────
# Timeframes
# ─────────────────────────────────────────────────────────────────────────────
HTF: str = "1H"
LTF: str = "15M"

# ─────────────────────────────────────────────────────────────────────────────
# Silver Bullet ET windows  (id, start_hour_et_inclusive, end_hour_et_exclusive)
# ─────────────────────────────────────────────────────────────────────────────
WINDOWS: Tuple[Tuple[str, int, int], ...] = (
    ("LO", 3, 4),    # London Open    — 03:00–04:00 ET
    ("AM", 10, 11),  # NY AM          — 10:00–11:00 ET
    ("PM", 14, 15),  # NY PM          — 14:00–15:00 ET
)

# ─────────────────────────────────────────────────────────────────────────────
# Sweep parameters (spec §2.2)
# ─────────────────────────────────────────────────────────────────────────────
SWEEP_LOOKBACK_BARS: int = 20                 # 15M bars before current
SWEEP_REF_BARS: int = 12                      # LTF-only fallback reference
SWEEP_PENETRATION_PIPS_FLOOR: float = 1.0
SWEEP_PENETRATION_ATR_MULT: float = 0.10
SWEEP_MAX_PENETRATION_ATR_MULT: float = 4.0

# ─────────────────────────────────────────────────────────────────────────────
# FVG parameters (spec §2.3)
# ─────────────────────────────────────────────────────────────────────────────
FVG_MIN_PIPS_DEFAULT: float = 2.0
FVG_MIN_PIPS_METAL: float = 3.0
FVG_GRADE_A_PIPS: float = 4.0
FVG_VALIDITY_BARS: int = 6                    # 90 min

# ─────────────────────────────────────────────────────────────────────────────
# Risk envelope (spec §0 / §2.5)
# ─────────────────────────────────────────────────────────────────────────────
RISK_PCT: float = 0.5
TP1_R: float = 1.5
TP2_R: float = 2.5
MIN_RISK_PIPS: float = 5.0
SL_BUFFER_PIPS_METAL: float = 5.0             # for XAUUSD / XAGUSD; FX uses MS map

# Trade lifecycle (mirror multi_setup_config — backtest engine reuses these)
PARTIAL_FRACTION: float = 0.50
BE_SHIFT_R: float = 1.0
TRAIL_STEP_R: float = 0.30
TIME_STOP_HOURS: int = 48

# Backtest invariants
SLIPPAGE_MARKET_FRAC_OF_MAX_SPREAD: float = 0.5
COMMISSION_PER_LOT_ROUNDTURN_USD: float = 7.0
ATR_LEN: int = 14
L_SWING: int = 3                              # mirror multi_setup_config

# Per-window cap for standalone test
MAX_TRADES_PER_WINDOW_PER_PAIR: int = 1

# ─────────────────────────────────────────────────────────────────────────────
# Per-pair broker maps — start with multi_setup, then add CADCHF + XAGUSD
# ─────────────────────────────────────────────────────────────────────────────
def _augment_map(base: Mapping, extras: Mapping) -> Mapping:
    merged = dict(base)
    merged.update(extras)
    return MappingProxyType(merged)


# CADCHF: 5-digit FX, pip = 0.0001, lot = 100,000 (same as other crosses)
# XAGUSD: 3-digit, pip = 0.01, contract = 5000 oz (standard FTMO silver)
PIP_SIZE: Mapping[str, float] = _augment_map(
    _MS_PIP_SIZE, {"CADCHF": 0.0001, "XAGUSD": 0.01}
)
CONTRACT_SIZE: Mapping[str, float] = _augment_map(
    _MS_CONTRACT_SIZE, {"CADCHF": 100000.0, "XAGUSD": 5000.0}
)
MAX_SPREAD_PIPS: Mapping[str, float] = _augment_map(
    _MS_MAX_SPREAD, {"CADCHF": 4.0, "XAGUSD": 8.0}
)
SL_BUFFER_PIPS: Mapping[str, float] = _augment_map(
    _MS_SL_BUF, {"CADCHF": 3.0, "XAGUSD": 5.0}
)


def pip_size_for(symbol: str) -> float:
    return float(PIP_SIZE[symbol])


def max_spread_pips_for(symbol: str) -> float:
    return float(MAX_SPREAD_PIPS[symbol])


def sl_buffer_pips_for(symbol: str) -> float:
    return float(SL_BUFFER_PIPS[symbol])


def contract_size_for(symbol: str) -> float:
    return float(CONTRACT_SIZE[symbol])


def fvg_min_pips_for(symbol: str) -> float:
    if symbol in ("XAUUSD", "XAGUSD"):
        return FVG_MIN_PIPS_METAL
    return FVG_MIN_PIPS_DEFAULT
