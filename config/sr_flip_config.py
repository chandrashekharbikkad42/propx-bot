"""S/R Flip — strategy constants (standalone candidate test).

Single source of truth for `docs/SR_FLIP_SPEC.md`. Isolated from
`config/multi_setup_config.py` so changes here do not leak into the proven
Asian Sweep V5 / propX Multi-Setup bundles.

Reuses pip / contract / spread / SL-buffer maps from `multi_setup_config`
where the data is purely broker-spec, and adds the 2 missing pairs
(CADCHF, XAGUSD) so the full 30-symbol shortlist is covered — same
pattern as `silver_bullet_config.py`.
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
# Pair universe (30) — same as silver_bullet_config
# ─────────────────────────────────────────────────────────────────────────────
PAIRS: Tuple[str, ...] = (
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
    "EURJPY", "EURGBP", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CHFJPY", "CADCHF",
    "XAUUSD", "XAGUSD",
)

HTF: str = "1H"
LTF: str = "15M"

# ─────────────────────────────────────────────────────────────────────────────
# Level discovery
# ─────────────────────────────────────────────────────────────────────────────
LEVEL_CLUSTER_TOLERANCE_FLOOR: float = 5.0     # pips
LEVEL_CLUSTER_TOLERANCE_ATR_MULT: float = 0.30  # × ATR_HTF_pips
MIN_LEVEL_TOUCHES: int = 2
LEVEL_MAX_AGE_HTF_BARS: int = 200              # ~8 days of 1H

# ─────────────────────────────────────────────────────────────────────────────
# Break detection
# ─────────────────────────────────────────────────────────────────────────────
BREAK_MARGIN_PIPS_FLOOR: float = 3.0
BREAK_MARGIN_ATR_MULT: float = 0.20             # × ATR_HTF_pips
MAX_BREAK_AGE_HTF_BARS: int = 48                # 2 days
REENTRY_BLOCK_PIPS: float = 5.0

# ─────────────────────────────────────────────────────────────────────────────
# Retest
# ─────────────────────────────────────────────────────────────────────────────
RETEST_TOL_PIPS_FLOOR: float = 2.0
RETEST_TOL_ATR_MULT: float = 0.15               # × ATR_LTF_pips
RETEST_LOOKBACK_LTF_BARS: int = 48              # 12h debounce

# ─────────────────────────────────────────────────────────────────────────────
# Risk
# ─────────────────────────────────────────────────────────────────────────────
RISK_PCT: float = 0.5
TP1_R: float = 1.5
TP2_R: float = 2.5
MIN_RISK_PIPS: float = 5.0

# Trade lifecycle (mirror silver_bullet_config / multi_setup_config)
PARTIAL_FRACTION: float = 0.50
BE_SHIFT_R: float = 1.0
TRAIL_STEP_R: float = 0.30
TIME_STOP_HOURS: int = 48

# Backtest invariants
SLIPPAGE_MARKET_FRAC_OF_MAX_SPREAD: float = 0.5
COMMISSION_PER_LOT_ROUNDTURN_USD: float = 7.0
ATR_LEN: int = 14
L_SWING: int = 3

# Per-(pair, day) cap for standalone test
MAX_TRADES_PER_DAY_PER_PAIR: int = 2

# ─────────────────────────────────────────────────────────────────────────────
# Per-pair broker maps — augment multi_setup with CADCHF + XAGUSD
# ─────────────────────────────────────────────────────────────────────────────
def _augment_map(base: Mapping, extras: Mapping) -> Mapping:
    merged = dict(base)
    merged.update(extras)
    return MappingProxyType(merged)


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
