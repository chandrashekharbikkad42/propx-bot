"""Shared helpers for the 4 Griff pattern detectors (Phase 8C-Patterns).

What lives here:
  - `GRIFF_PAIRS`         : the 6 currency pairs Griff trades.
  - `INITIAL_SL_PIPS`     : per-pair fixed pip stops used by Continuation,
                            Combo, and Reversal patterns. Flag uses structural
                            (Flag-Low/High) SL instead and ignores this table.
  - `pip_size(pair)`      : 0.01 for JPY pairs, 0.0001 otherwise.
  - `synthesize_tp(...)`  : placeholder TP at 1:2 R. See "TP NOTE" below.
  - `body / range_ / ...` : tiny OHLC math helpers reused across detectors.

TP NOTE — important for reviewers:
  The Griff strategy as captured in this prompt specifies entry + initial SL
  only; there is NO take-profit target — exits are governed by `risk/trailing_sl.py`
  (swing-based trail + spread-hour protection). However, the existing
  `PatternSignal` contract (strategy/patterns/base.py) REQUIRES a positive
  `tp` value and enforces BUY: sl<entry<tp / SELL: tp<entry<sl ordering.
  To honour the "don't redefine Signal" rule from the phase brief, every
  detector synthesises a placeholder TP at 1:2 R (`entry ± 2 × |entry-sl|`).
  This is a SENTINEL, not a trade target — the trailing-SL module owns the
  actual exit. Flagged in docs/GRIFF_PATTERN_AMBIGUITIES.md (item #1) for
  human review.

Hinglish: TP nahi hota Griff me — trailing SL hi exit deta hai. Par Signal
dataclass me TP mandatory hai, isliye 1:2 R ka placeholder thok ke contract
satisfy karte hain. Trailing SL real boss hai.
"""

from __future__ import annotations
from types import MappingProxyType
from typing import Mapping

from data.bar_aggregator import Bar
from strategy.patterns.base import Direction


GRIFF_PAIRS: tuple[str, ...] = (
    "AUDJPY", "AUDUSD", "EURUSD", "EURJPY", "GBPUSD", "NZDJPY",
)


# Per-pair INITIAL_SL pip stops (Continuation, Combo, Reversal). Flag is
# structural so it ignores this table.
INITIAL_SL_PIPS: Mapping[str, float] = MappingProxyType({
    "AUDJPY": 10.0,
    "AUDUSD": 10.0,
    "EURJPY": 10.0,
    "EURUSD": 12.0,
    "GBPUSD": 10.0,
    "NZDJPY": 12.0,
})

# Entry offset used by patterns that place pending orders beyond a level.
PENDING_ENTRY_OFFSET_PIPS: float = 2.0

# Placeholder TP multiple of risk. See module docstring "TP NOTE".
_TP_R_MULTIPLE: float = 2.0

# Reversal hard exclusion list. See reversal.py.
REVERSAL_EXCLUDED_PAIRS: frozenset[str] = frozenset({"GBPUSD"})

# Number of bars used as the "average body" lookback for the
# excessively-large-entry-candle check in Flag detection.
AVG_BODY_LOOKBACK: int = 10

# Threshold for "excessively large" entry candle — body > N × avg body of
# the last AVG_BODY_LOOKBACK bars. Flagged: spec said "excessively large",
# default chosen per phase brief.
EXCESSIVE_BODY_MULT: float = 2.0

# Continuation pullback thresholds (per phase brief).
CONT_PULLBACK_BODY_PCT_MAX: float = 0.40   # pullback body < 40% of impulse body
CONT_PULLBACK_WICK_PCT_MIN: float = 0.60   # rejection wick > 60% of pullback range


def pip_size(pair: str) -> float:
    """Standard forex pip — 0.01 for JPY pairs, 0.0001 otherwise.

    Mirrors `risk.trailing_sl.pip_size`. Duplicated (one line) to avoid
    creating an import cycle between strategy and risk packages.
    """
    return 0.01 if "JPY" in pair.upper() else 0.0001


def body(bar: Bar) -> float:
    """Absolute candle body size (open→close, sign-stripped)."""
    return abs(bar.close - bar.open)


def range_(bar: Bar) -> float:
    """Full candle range (high - low). Always >= body."""
    return bar.high - bar.low


def upper_wick(bar: Bar) -> float:
    """Distance from body top to bar high."""
    return bar.high - max(bar.open, bar.close)


def lower_wick(bar: Bar) -> float:
    """Distance from body bottom to bar low."""
    return min(bar.open, bar.close) - bar.low


def is_bullish(bar: Bar) -> bool:
    return bar.close > bar.open


def is_bearish(bar: Bar) -> bool:
    return bar.close < bar.open


def avg_body(bars, n: int = AVG_BODY_LOOKBACK) -> float:
    """Mean body of the last n bars. Empty → 0.0 (caller treats as no-gate)."""
    tail = list(bars)[-n:]
    if not tail:
        return 0.0
    return sum(body(b) for b in tail) / len(tail)


def synthesize_tp(entry: float, sl: float, direction: Direction,
                  r_multiple: float = _TP_R_MULTIPLE) -> float:
    """Placeholder TP at `r_multiple` × |entry-sl| from entry, in the trade's
    favourable direction. See module docstring "TP NOTE".
    """
    risk = abs(entry - sl)
    if direction == Direction.BUY:
        return entry + r_multiple * risk
    return entry - r_multiple * risk
