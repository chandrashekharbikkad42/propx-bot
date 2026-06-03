"""propX Multi-Setup — strategy constants (single source of truth).

Mirrors `docs/MULTI_SETUP_SPEC.md` Appendix A v1.0. Any spec change must
land here too. The four detectors and the confluence resolver import
exclusively from this module — no magic numbers inside detector code.

Sibling: `config/asian_sweep_config.py` is the analogous file for the V5
Asian Sweep strategy. The two strategies share infra (broker, compliance,
risk caps) but maintain independent constants.

Hinglish: yeh file Multi-Setup ka brain hai. Spec ke saare numbers yahin
hain. Detector me hard-coded number kabhi mat likhna — sab yahaan se aaye.
"""

from __future__ import annotations
from types import MappingProxyType
from typing import Mapping


# ─────────────────────────────────────────────────────────────────────────────
# Pair universe — 28 FX/metal pairs confirmed tradeable on FTMO-Demo
# (Phase 1 verification, 2026-05-26).
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
    # JPY crosses (2)
    "CADJPY", "CHFJPY",
    # Metal (1)
    "XAUUSD",
)

# ─────────────────────────────────────────────────────────────────────────────
# Timeframes (spec §0)
# ─────────────────────────────────────────────────────────────────────────────
HTF: str = "1H"
LTF: str = "15M"

# ─────────────────────────────────────────────────────────────────────────────
# Risk envelope (spec §0)
# ─────────────────────────────────────────────────────────────────────────────
RISK_PCT: float = 0.5
MAX_TRADES_PER_DAY: int = 2
MAX_CONCURRENT: int = 2
TP1_R: float = 1.5
TP2_R: float = 2.5
PARTIAL_FRACTION: float = 0.50
BE_SHIFT_R: float = 1.0
TRAIL_STEP_R: float = 0.30
TIME_STOP_HOURS: int = 48
FRIDAY_FLATTEN_UTC: str = "23:55"
DAILY_DD_PCT: float = 3.0          # reuse existing 3% intraday DD kill-switch
NEWS_BLACKOUT_MIN: int = 2          # ±2 min around high-impact news

# ─────────────────────────────────────────────────────────────────────────────
# Swing / structure (spec §1.2 – §1.3)
# ─────────────────────────────────────────────────────────────────────────────
L_SWING: int = 3                    # fractal lookback (left + right) for swings

# ─────────────────────────────────────────────────────────────────────────────
# Impulsive move — 15M LTF (spec §1.4)
# ─────────────────────────────────────────────────────────────────────────────
N_IMP: int = 3                      # consecutive same-direction LTF closes
IMP_MIN_PIPS_BASE: float = 15.0     # base pip threshold (× adaptive ATR factor)
IMP_ATR_FACTOR_MIN: float = 0.5     # clamp lower bound
IMP_ATR_FACTOR_MAX: float = 2.0     # clamp upper bound
IMP_BAR_REVERSAL_MAX_FRAC: float = 0.30   # any single bar reversal ≤ 30% of its range
IMP_VOL_MULT: float = 1.2           # run volume ≥ 1.2× median over last 50 bars
IMP_VOL_LOOKBACK_BARS: int = 50     # median volume window

# ─────────────────────────────────────────────────────────────────────────────
# ATR (spec §1.5)
# ─────────────────────────────────────────────────────────────────────────────
ATR_LEN: int = 14                   # Wilder ATR(14), closed bars only

# ─────────────────────────────────────────────────────────────────────────────
# Rejection candle (spec §1.8)
# ─────────────────────────────────────────────────────────────────────────────
PIN_BODY_MAX_FRAC: float = 0.33
PIN_WICK_MIN_BODY_MULT: float = 2.0
PIN_WICK_MIN_RANGE_FRAC: float = 0.55
ENGULF_BODY_MIN_MULT: float = 1.0
ENGULF_BODY_MIN_RANGE_FRAC: float = 0.40
REJECT_VOL_MIN_MULT: float = 1.1
REJECT_VOL_LOOKBACK_BARS: int = 20

# Pin body color — strict by default (close > open for bullish pin). Spec §1.8
# notes an OPTIONAL relax for doji body; toggle here.
PIN_REQUIRE_BODY_COLOR: bool = True

# ─────────────────────────────────────────────────────────────────────────────
# Setup #1 — Liquidity Sweep + Reversal (spec §2)
# ─────────────────────────────────────────────────────────────────────────────
SWEEP_RECLAIM_BARS: int = 3
SWEEP_LIMIT_EXPIRY_BARS: int = 4
SWEEP_PENETRATION_PIPS_FLOOR: float = 1.0         # adaptive: max(floor, 0.10×ATR_LTF)
SWEEP_PENETRATION_ATR_MULT: float = 0.10
SWEEP_MAX_PENETRATION_ATR_MULT: float = 4.0
SWEEP_ENTRY_MODE: str = "LIMIT"                   # "LIMIT" (default, retest) | "MARKET"

# ─────────────────────────────────────────────────────────────────────────────
# Setup #2 — Order Block (spec §3)
# ─────────────────────────────────────────────────────────────────────────────
N_IMP_HTF: int = 2                                # 1H impulse = 2 strong same-direction bars
OB_IMP_MIN_PIPS_FLOOR: float = 20.0               # adaptive: max(floor, 1.5×ATR_HTF)
OB_IMP_ATR_MULT: float = 1.5
OB_IMP_CLEAR_FRAC: float = 0.5                    # impulse must clear OB by ≥ 0.5×min displacement
OB_MAX_AGE_BARS_HTF: int = 200                    # ~8 trading days
OB_RETEST_BARS_LTF: int = 3
OB_ENTRY_MODE: str = "TOP_OF_OB"                  # "TOP_OF_OB" (default) | "MID_OB"

# ─────────────────────────────────────────────────────────────────────────────
# Setup #3 — BoS + Retest (spec §4)
# ─────────────────────────────────────────────────────────────────────────────
BOS_BUFFER_PIPS_FLOOR: float = 2.0                # adaptive: max(floor, 0.10×ATR_HTF)
BOS_BUFFER_ATR_MULT: float = 0.10
BOS_RETEST_TOLERANCE_FLOOR: float = 3.0           # adaptive: max(floor, 0.15×ATR_HTF)
BOS_RETEST_TOLERANCE_ATR_MULT: float = 0.15
BOS_MAX_AGE_BARS_HTF: int = 100
BOS_RETEST_BARS_LTF: int = 4
BOS_ENTRY_MODE: str = "MARKET"                    # "MARKET" (default) | "LIMIT"
BOS_NEWS_SPIKE_WINDOW_MIN: int = 10               # discard BoS within ±10min of news

# ─────────────────────────────────────────────────────────────────────────────
# Setup #4 — S/R Rejection (spec §5)
# ─────────────────────────────────────────────────────────────────────────────
LEVEL_MIN_TOUCHES: int = 3
LEVEL_CLUSTER_TOLERANCE_FLOOR: float = 3.0        # adaptive: max(floor, 0.20×ATR_HTF)
LEVEL_CLUSTER_TOLERANCE_ATR_MULT: float = 0.20
LEVEL_LOOKBACK_BARS_HTF: int = 200
LEVEL_MIN_GAP_BARS: int = 5
LEVEL_BREAK_PIPS_FLOOR: float = 5.0               # adaptive: max(floor, 0.30×ATR_HTF)
LEVEL_BREAK_ATR_MULT: float = 0.30
LEVEL_NEARBY_TOL_MULT: float = 3.0                # cleanliness — no competing level within ±tol×3
SR_REJECT_BARS_LTF: int = 3
SR_WICK_BREAK_FRAC: float = 1.5                   # allow wick break up to tol×1.5 below support

# ─────────────────────────────────────────────────────────────────────────────
# Confluence (spec §6)
# ─────────────────────────────────────────────────────────────────────────────
CONFLUENCE_PRICE_TOL_PIPS: float = 5.0
CONFLUENCE_BAR_TOL_LTF: int = 3
# Risk stays at RISK_PCT — confluence is a confidence boost, not a sizing boost.
CONFLUENCE_CONFIDENCE_BOOST: float = 0.15         # added to max(child confidences), clamped to 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Setup rank (spec §0 — used when >2 setups qualify same day)
# ─────────────────────────────────────────────────────────────────────────────
SETUP_RANK: Mapping[str, int] = MappingProxyType({
    "BOS_RETEST":   4,
    "ORDER_BLOCK":  3,
    "LIQ_SWEEP":    2,
    "SR_REJECTION": 1,
})

# ─────────────────────────────────────────────────────────────────────────────
# Per-pair spread guard (spec §1.6) — pips
# ─────────────────────────────────────────────────────────────────────────────
MAX_SPREAD_PIPS: Mapping[str, float] = MappingProxyType({
    # Majors (2.0)
    **{p: 2.0 for p in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
                        "NZDUSD", "USDCAD", "USDCHF")},
    # EUR / GBP crosses (3.0)
    **{p: 3.0 for p in ("EURJPY", "EURGBP", "EURCHF",
                        "GBPJPY", "GBPCHF")},
    # AUD / NZD / CAD crosses (4.0)
    **{p: 4.0 for p in ("AUDJPY", "AUDCAD", "AUDCHF", "AUDNZD",
                        "NZDJPY", "NZDCHF", "NZDCAD",
                        "CADJPY", "CHFJPY")},
    # Exotic crosses (5.0)
    **{p: 5.0 for p in ("EURAUD", "EURNZD", "EURCAD",
                        "GBPAUD", "GBPNZD", "GBPCAD")},
    # Metal
    "XAUUSD": 5.0,
})

# ─────────────────────────────────────────────────────────────────────────────
# Per-pair SL buffer beyond structural level (spec §1.7) — pips
# ─────────────────────────────────────────────────────────────────────────────
SL_BUFFER_PIPS: Mapping[str, float] = MappingProxyType({
    # Majors
    **{p: 2.0 for p in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
                        "NZDUSD", "USDCAD", "USDCHF")},
    # All crosses
    **{p: 3.0 for p in (
        "EURJPY", "EURGBP", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
        "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
        "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
        "NZDJPY", "NZDCHF", "NZDCAD",
        "CADJPY", "CHFJPY",
    )},
    # Metal
    "XAUUSD": 5.0,
})

# Entry slippage absorption buffer — default 0; live engine may inject
# 0.5 × current spread (spec §1.7).
ENTRY_BUFFER_PIPS_DEFAULT: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Per-pair pip size (spec §1.1)
# ─────────────────────────────────────────────────────────────────────────────
# Pip is the strategy unit; broker "point" is the price increment. For the
# 28-pair universe at FTMO-Demo:
#   5-digit FX (EURUSD etc.): point=0.00001 → 1 pip = 10 points = 0.0001
#   3-digit JPY pairs:        point=0.001   → 1 pip = 10 points = 0.01
#   XAUUSD (2-digit):         point=0.01    → 1 pip = 10 points = 0.10
PIP_SIZE: Mapping[str, float] = MappingProxyType({
    # 3-digit JPY pairs
    **{p: 0.01 for p in ("USDJPY", "EURJPY", "GBPJPY", "AUDJPY",
                         "NZDJPY", "CADJPY", "CHFJPY")},
    # XAUUSD (2-digit)
    "XAUUSD": 0.10,
    # 5-digit FX (everything else)
    **{p: 0.0001 for p in (
        "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
        "EURGBP", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
        "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
        "AUDCHF", "AUDCAD", "AUDNZD",
        "NZDCHF", "NZDCAD",
    )},
})

# ─────────────────────────────────────────────────────────────────────────────
# Per-pair contract / lot size (broker spec) — used by risk sizing
# ─────────────────────────────────────────────────────────────────────────────
CONTRACT_SIZE: Mapping[str, float] = MappingProxyType({
    **{p: 100000.0 for p in (
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
        "EURJPY", "EURGBP", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
        "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
        "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
        "NZDJPY", "NZDCHF", "NZDCAD",
        "CADJPY", "CHFJPY",
    )},
    "XAUUSD": 100.0,
})

# ─────────────────────────────────────────────────────────────────────────────
# Backtest invariants (spec §8)
# ─────────────────────────────────────────────────────────────────────────────
SLIPPAGE_MARKET_FRAC_OF_MAX_SPREAD: float = 0.5    # 0.5 × MAX_SPREAD_PIPS on market orders
COMMISSION_PER_LOT_ROUNDTURN_USD: float = 7.0      # FTMO default
BACKTEST_MAX_DAILY_DD_PCT: float = 4.0             # invariant guard (spec §8.5)

# Minimum risk distance — defensive floor (mirror Asian Sweep _MIN_RISK_PT_MULT
# idea). 5 pips minimum on every setup per spec §2.4 invariant.
MIN_RISK_PIPS: float = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────
def pip_size_for(symbol: str) -> float:
    """Return pip size for a symbol. Raises KeyError on unknown pairs (loud
    failure is preferable to silent miscalc here — strategy correctness
    depends on the right pip)."""
    return float(PIP_SIZE[symbol])


def max_spread_pips_for(symbol: str) -> float:
    """Spread cap above which the detector must SKIP."""
    return float(MAX_SPREAD_PIPS[symbol])


def sl_buffer_pips_for(symbol: str) -> float:
    """Buffer beyond structural level (swing wick, OB extreme, etc.)."""
    return float(SL_BUFFER_PIPS[symbol])


def contract_size_for(symbol: str) -> float:
    return float(CONTRACT_SIZE[symbol])


def setup_rank_for(setup_name: str) -> int:
    """Higher = better tie-breaker for daily slot selection. 0 for unknown."""
    return int(SETUP_RANK.get(setup_name, 0))
