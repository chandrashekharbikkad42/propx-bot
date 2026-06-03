"""Asian Range London Sweep V5 — strategy constants.

Single source of truth for the V5 strategy ported from
`multi_pair_backtest.py` (verified 1-yr backtest: PF 2.27, 239 trades).

Strategy summary (V5):
  1. Mark Asian range from previous day 19:30 UTC → current day 00:30 UTC
     (= 01:00 IST → 06:00 IST).
  2. Detect sweep + close-back in two windows:
       LONDON sweep window  06:00–10:30 UTC  (= 11:30–16:00 IST)
       NY     sweep window  12:00–15:30 UTC  (= 17:30–21:00 IST)
     LONG  = sweep Asian Low + close above; bullish/neutral HTF; both sessions.
     SHORT = sweep Asian High + close below; bearish HTF; LONDON ONLY (V5 rule).
  3. Entry = swept level ± broker spread. SL = wick ± sl_pts buffer.
     TP1 = 1R (partial 50%), then SL → BE. TP2 = 2.5R. Trail 0.3R after TP1.
  4. Force-close at 16:00 UTC (= 21:30 IST).
  5. Skip Monday. Weak months (Nov/Dec/Jan) → 0.3% risk.
  6. Daily DD circuit at 3% from day-start.

Hinglish: Asian range nikalo (prev 19:30 → today 00:30 UTC), London ya NY me
sweep + close-back dekho, reversal me ghuso. SHORT sirf London me, LONG dono
session me. Max 2 trade/day, 1 per direction.
"""

from __future__ import annotations
from types import MappingProxyType
from typing import Mapping

# ─────────────────────────────────────────────────────────────────────────────
# Pair universe — verified 8 pairs from multi_pair_backtest.py v5
# ─────────────────────────────────────────────────────────────────────────────
PAIRS: tuple[str, ...] = (
    "XAUUSD", "GBPUSD", "AUDUSD", "EURUSD",
    "USDCAD", "USDCHF", "AUDCHF", "AUDNZD", "NZDUSD", "EURNZD", "GBPCAD", "GBPAUD", "HK50.cash", "GER40.cash",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session windows — IST anchors (display) + UTC HMS (computation)
# ─────────────────────────────────────────────────────────────────────────────
ASIAN_START_IST: str = "01:00"   # = 19:30 UTC previous day
ASIAN_END_IST:   str = "06:00"   # = 00:30 UTC current day
LONDON_SWEEP_IST_START: str = "11:30"  # = 06:00 UTC
LONDON_SWEEP_IST_END:   str = "16:00"  # = 10:30 UTC
NY_SWEEP_IST_START:     str = "17:30"  # = 12:00 UTC
NY_SWEEP_IST_END:       str = "21:00"  # = 15:30 UTC
NY_END_IST:             str = "21:30"  # = 16:00 UTC — forced close

# UTC numerics — used directly by the detector to bucket 1H bars.
ASIAN_START_UTC_H, ASIAN_START_UTC_M = 19, 30    # previous day
ASIAN_END_UTC_H,   ASIAN_END_UTC_M   = 0, 30     # current day
LONDON_SWEEP_UTC_H_START = 6
LONDON_SWEEP_UTC_H_END   = 10                    # bars 06..10 inclusive
NY_SWEEP_UTC_H_START     = 12
NY_SWEEP_UTC_H_END       = 15                    # bars 12..15 inclusive
SESSION_FORCE_CLOSE_UTC_H = 16                   # EOD flatten

# ─────────────────────────────────────────────────────────────────────────────
# Trade-management constants — frozen exactly as backtested
# ─────────────────────────────────────────────────────────────────────────────
MAX_TRADES_PER_DAY: int = 2
PARTIAL_CLOSE_FRACTION: float = 0.50
RR_TP1: float = 1.0
RR_TP2: float = 2.5
TRAILING_STEP_R: float = 0.30        # trail SL by 0.3 × risk after TP1
MAX_DAILY_DD_PCT: float = 3.0        # circuit-break at 3% intraday loss
SKIP_MONDAY: bool = True

# ─────────────────────────────────────────────────────────────────────────────
# V5 hard safety caps — last-line-of-defence on the LIVE sizing path
# (back-tested logic is untouched; these only short-circuit pathological
# inputs that the verified backtest never produced).
# ─────────────────────────────────────────────────────────────────────────────
# (1) Reject any signal whose SL is closer than this floor — closes the
# "tiny SL ⇒ massive lots" hole on broken/degenerate signals. 1 pip = 10
# broker points (MT5 convention) so this applies uniformly: XAUUSD
# (point=0.01 → 5 pips = $0.50), 5-digit FX (point=0.00001 → 5 pips =
# 0.0005), indices (point=0.01 → 5 pips = 0.50 index points).
MIN_SL_DISTANCE_PIPS: float = 5.0

# (2) Absolute USD ceiling on per-trade risk, applied AFTER risk-% sizing.
# Scales lots DOWN regardless of equity / risk_pct — no single trade can
# leak through into a prop-account-blowing loss.
MAX_RISK_USD_PER_TRADE: float = 150.0

# (3) Total-account drawdown kill switch — measured from the bot's
# observed equity high-water mark (NOT starting_equity, so a profitable
# stretch does not lift the ceiling). Set below the prop firm's 10% rule
# for safety margin. Daily DD (3%) is the intraday guard; this is the
# total-account guard.
MAX_TOTAL_DD_PCT: float = 8.0

# ─────────────────────────────────────────────────────────────────────────────
# Risk per trade — per-pair override + weak-month dampener
# ─────────────────────────────────────────────────────────────────────────────
RISK_PCT: Mapping[str, float] = MappingProxyType({
    "XAUUSD": 0.5,
    "default": 0.8,
})
WEAK_MONTH_RISK_PCT: float = 0.3
WEAK_MONTHS: tuple[int, ...] = (11, 12, 1)  # Nov / Dec / Jan

# News blackout window (± minutes around high-impact events)
NEWS_BLACKOUT_MIN: int = 2

# ─────────────────────────────────────────────────────────────────────────────
# Per-pair broker config — exact values from FTMO Demo symbols_info.csv
# (replicated from multi_pair_backtest.SYMBOLS, do not edit casually)
#
# Fields:
#   point          : MT5 broker point (= smallest price increment)
#   contract_size  : contract / lot multiplier
#   lot_max        : broker cap on lot per order
#   spread_pts     : typical spread in broker points (entry offset)
#   sl_pts         : SL buffer beyond sweep wick, in broker points
#   min_range_pts  : Asian range minimum (filter junk days)
#   max_range_pts  : Asian range maximum (filter event days)
#   quality        : Scanner ranking score 1–10 (higher wins same-day slot)
#   category       : Metal / Major / Cross — for logging only
#   jpy            : profit-currency-is-JPY flag → /150 conversion
#   risk_override  : per-pair risk override (None = use RISK_PCT[default])
# ─────────────────────────────────────────────────────────────────────────────
PAIR_CONFIG: Mapping[str, Mapping[str, object]] = MappingProxyType({
    "XAUUSD": MappingProxyType({
        "point": 0.01, "contract_size": 100.0, "lot_max": 50.0,
        "spread_pts": 45, "sl_pts": 70,
        "min_range_pts": 100, "max_range_pts": 3000,
        "quality": 10, "category": "Metal", "jpy": False,
        "risk_override": 0.5,
    }),
    "EURUSD": MappingProxyType({
        "point": 0.00001, "contract_size": 100000.0, "lot_max": 50.0,
        "spread_pts": 4, "sl_pts": 80,
        "min_range_pts": 200, "max_range_pts": 2000,
        "quality": 9, "category": "Major", "jpy": False,
        "risk_override": None,
    }),
    "AUDUSD": MappingProxyType({
        "point": 0.00001, "contract_size": 100000.0, "lot_max": 50.0,
        "spread_pts": 3, "sl_pts": 80,
        "min_range_pts": 150, "max_range_pts": 1800,
        "quality": 9, "category": "Major", "jpy": False,
        "risk_override": None,
    }),
    "GBPUSD": MappingProxyType({
        "point": 0.00001, "contract_size": 100000.0, "lot_max": 50.0,
        "spread_pts": 8, "sl_pts": 100,
        "min_range_pts": 200, "max_range_pts": 2500,
        "quality": 8, "category": "Major", "jpy": False,
        "risk_override": None,
    }),
    "USDCAD": MappingProxyType({
        "point": 0.00001, "contract_size": 100000.0, "lot_max": 50.0,
        "spread_pts": 5, "sl_pts": 80,
        "min_range_pts": 150, "max_range_pts": 2000,
        "quality": 7, "category": "Major", "jpy": False,
        "risk_override": None,
    }),
    "USDCHF": MappingProxyType({
        "point": 0.00001, "contract_size": 100000.0, "lot_max": 50.0,
        "spread_pts": 6, "sl_pts": 80,
        "min_range_pts": 150, "max_range_pts": 2000,
        "quality": 7, "category": "Major", "jpy": False,
        "risk_override": None,
    }),
    "AUDCHF": MappingProxyType({
        "point": 0.00001, "contract_size": 100000.0, "lot_max": 50.0,
        "spread_pts": 8, "sl_pts": 80,
        "min_range_pts": 150, "max_range_pts": 1800,
        "quality": 5, "category": "Cross", "jpy": False,
        "risk_override": None,
    }),
    "NZDUSD": MappingProxyType({
        "point": 0.00001, "contract_size": 100000.0, "lot_max": 50.0,
        "spread_pts": 7, "sl_pts": 80,
        "min_range_pts": 150, "max_range_pts": 1800,
        "quality": 7, "category": "Major", "jpy": False,
        "risk_override": None,
    }),
    "EURNZD": MappingProxyType({
        "point": 0.00001, "contract_size": 100000.0, "lot_max": 50.0,
        "spread_pts": 12, "sl_pts": 80,
        "min_range_pts": 150, "max_range_pts": 1800,
        "quality": 8, "category": "Cross", "jpy": False,
        "risk_override": None,
    }),
    "GBPCAD": MappingProxyType({
        "point": 0.00001, "contract_size": 100000.0, "lot_max": 50.0,
        "spread_pts": 12, "sl_pts": 80,
        "min_range_pts": 150, "max_range_pts": 1800,
        "quality": 8, "category": "Cross", "jpy": False,
        "risk_override": None,
    }),
    "GBPAUD": MappingProxyType({
        "point": 0.00001, "contract_size": 100000.0, "lot_max": 50.0,
        "spread_pts": 12, "sl_pts": 80,
        "min_range_pts": 150, "max_range_pts": 1800,
        "quality": 8, "category": "Cross", "jpy": False,
        "risk_override": None,
    }),
    "HK50.cash": MappingProxyType({
        "point": 0.01, "contract_size": 1.0, "lot_max": 50.0,
        "spread_pts": 50, "sl_pts": 2000,
        "min_range_pts": 100, "max_range_pts": 30000,
        "quality": 9, "category": "Index", "jpy": False,
        "risk_override": None,
    }),
    "GER40.cash": MappingProxyType({
        "point": 0.01, "contract_size": 1.0, "lot_max": 50.0,
        "spread_pts": 30, "sl_pts": 2000,
        "min_range_pts": 100, "max_range_pts": 30000,
        "quality": 8, "category": "Index", "jpy": False,
        "risk_override": None,
    }),
    "AUDNZD": MappingProxyType({
        "point": 0.00001, "contract_size": 100000.0, "lot_max": 50.0,
        "spread_pts": 12, "sl_pts": 80,
        "min_range_pts": 150, "max_range_pts": 1800,
        "quality": 4, "category": "Cross", "jpy": False,
        "risk_override": None,
    }),
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def point_for(symbol: str) -> float:
    """Broker point for a symbol. Raises KeyError for unknown pairs."""
    return float(PAIR_CONFIG[symbol]["point"])  # type: ignore[arg-type]


def risk_pct_for(symbol: str, *, month: int | None = None) -> float:
    """Resolve effective risk % for a symbol.

    Priority: weak-month dampener > per-pair override > RISK_PCT[symbol]
    > RISK_PCT["default"].
    """
    if month is not None and month in WEAK_MONTHS:
        return WEAK_MONTH_RISK_PCT
    cfg = PAIR_CONFIG.get(symbol)
    if cfg is not None:
        override = cfg.get("risk_override")
        if override is not None:
            return float(override)  # type: ignore[arg-type]
    if symbol in RISK_PCT:
        return float(RISK_PCT[symbol])
    return float(RISK_PCT["default"])


def quality_for(symbol: str) -> int:
    """Scanner ranking score. Unknown pair → 0 (will be deprioritised)."""
    cfg = PAIR_CONFIG.get(symbol)
    if cfg is None:
        return 0
    return int(cfg["quality"])  # type: ignore[arg-type]

