"""NY Gold Sweep — config constants (Appendix A of NY_GOLD_SWEEP_SPEC.md v1.1).

Every value here is the spec's default. Do not change without updating the
spec first; the spec is the source of truth.

§0.2 pip convention (XAUUSD):
    1 pip = 100 broker points = 1.00 of price (= $1.00 per oz)
    point = 0.01 of price
"""

from __future__ import annotations

# ─── Symbol & broker constants ──────────────────────────────────────────────
SYMBOL: str = "XAUUSD"

PIP_SIZE: float = 1.00          # 1 pip = 1.00 of price (§0.2)
POINT_SIZE: float = 0.01        # 1 broker point = 0.01 of price
POINTS_PER_PIP: int = 100       # = PIP_SIZE / POINT_SIZE

CONTRACT_SIZE: float = 100.0    # 1 lot = 100 oz XAUUSD
USD_PER_PIP_PER_LOT: float = 100.0   # 1 pip × 1 lot = $100 (= $1/oz × 100oz)

LOT_MIN: float = 0.01
LOT_STEP: float = 0.01
LOT_MAX: float = 50.0

# ─── §0 risk envelope ───────────────────────────────────────────────────────
RISK_PCT: float = 0.50          # % of running balance per trade
MIN_RISK_PIPS: float = 0.30     # stop-distance floor (pips)

INITIAL_BALANCE: float = 100_000.0   # The5%ers $100k account

# ─── §1 NY session window (UTC) ─────────────────────────────────────────────
NY_SESSION_START_UTC: tuple[int, int, int] = (12, 0, 0)   # 12:00:00 inclusive
NY_SESSION_END_UTC:   tuple[int, int, int] = (17, 0, 0)   # 17:00:00 exclusive

# Skip the first 1M bar of session (open_time = 12:00) and block new entries
# in the final 5M (open_time >= 16:55).
SESSION_SKIP_FIRST_MIN: int = 1
SESSION_NO_NEW_ENTRY_LAST_MIN: int = 5

# ─── §2 15M bias / level discovery ──────────────────────────────────────────
L_SWING_15M: int = 2
LEVEL_LOOKBACK_HOURS: int = 8                # → 32 × 15M bars
LEVEL_INVALIDATE_PIPS: float = 0.50          # freshness penetration cap

# ─── §3 5M zone (proximity gate) ────────────────────────────────────────────
ZONE_PROXIMITY_PIPS: float = 1.50
ZONE_MAX_DWELL_MIN: int = 25                 # → 5 × 5M bars

# ─── §4 1M sweep + reversal ─────────────────────────────────────────────────
SWEEP_MIN_PENETRATION_PIPS: float = 0.10
SWEEP_MAX_PENETRATION_PIPS: float = 0.80     # floor; actual = max(floor, ATR_mult × ATR_5M)
SWEEP_MAX_ATR_MULT: float = 0.40
SWEEP_REJECT_TOLERANCE_PIPS: float = 0.10

REVERSAL_MAX_WAIT_BARS: int = 3
ENGULF_MIN_BODY_PIPS: float = 0.20
PIN_WICK_BODY_RATIO: float = 2.0
PIN_MIN_WICK_PIPS: float = 0.30

# ATR window for §4.1 adaptive penetration cap
ATR_5M_PERIOD: int = 14

# ─── §5 entry / SL / TP ─────────────────────────────────────────────────────
SL_BUFFER_PIPS: float = 0.20

TP_MODE: str = "C"               # "A" fixed RR | "B" opposing | "C" hybrid
TP_RR: float = 1.50              # fixed RR multiple (Mode A & fallback in C)
TP_RR_MAX: float = 3.00          # cap for Mode C
MIN_RR_FOR_OPPOSING: float = 1.00
OPPOSING_BUFFER_PIPS: float = 0.20
OPPOSING_MAX_DISTANCE_PIPS: float = 8.00

# ─── §6 lifecycle (static SL/TP only) ───────────────────────────────────────
TIME_STOP_MIN: int = 45
SESSION_FLATTEN_UTC: tuple[int, int, int] = NY_SESSION_END_UTC

# ─── §7 The5%ers compliance gates ───────────────────────────────────────────
MIN_HOLD_SEC: int = 60
NEWS_BLACKOUT_BEFORE_MIN: int = 5
NEWS_BLACKOUT_AFTER_MIN: int = 5
NEWS_BLACKOUT_WINDOW_MIN: int = 5   # symmetric → passed to is_news_blackout
DAILY_DD_HALT_PCT: float = 5.0
TOTAL_DD_HALT_PCT: float = 10.0
MAX_TRADES_PER_DAY: int = 3
COOLDOWN_AFTER_LOSS_MIN: int = 30
COOLDOWN_AFTER_TWO_LOSSES_MIN: int = 60

# ─── §8 cost model ──────────────────────────────────────────────────────────
SLIPPAGE_PIPS: float = 0.20
DEFAULT_SPREAD_PIPS: float = 0.45            # fallback when bar.spread_mean is NaN
COMMISSION_USD_PER_LOT_ROUND_TURN: float = 7.00

# ─── §10 grading ────────────────────────────────────────────────────────────
GRADE_A_MIN_TOUCHES: int = 2
GRADE_A_PENETRATION_LO: float = 0.20
GRADE_A_PENETRATION_HI: float = 0.60

# Touch detection: a "prior touch" = a 15M bar's high/low within this band of
# the level price. Spec §10 does not numerically define touch tolerance; we
# pick a tight value in line with §2.3 invalidate.
TOUCH_TOLERANCE_PIPS: float = 0.50

# ─── Bar timeframe durations (ms) ───────────────────────────────────────────
BAR_MS: dict[str, int] = {
    "1M": 60_000,
    "5M": 5 * 60_000,
    "15M": 15 * 60_000,
    "1H": 60 * 60_000,
}
