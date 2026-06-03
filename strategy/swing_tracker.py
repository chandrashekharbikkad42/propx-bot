"""Swing-high / swing-low detector + wick-break checks (Phase 8C / Griff).

A "swing high" is a 1-bar fractal — a bar whose `high` is STRICTLY greater
than the bar immediately before AND the bar immediately after. Symmetric for
swing low. We confirm a swing one bar LATE: the right neighbor must arrive
before we can stamp the middle bar.

A "wick break" of a swing high is any bar whose `high` pierces the swing —
even a 1-point wick counts, the close is irrelevant. Same for low. This
matches Griff's trail trigger: structure breaks the instant price reaches
through it, not when it closes through.

`update(pair, bar)` returns a 4-key dict:
  new_swing_high: Optional[float] — set if THIS bar confirms the previous
                                    bar as a swing high (mid of [-3,-2,-1]).
  new_swing_low:  Optional[float] — same for lows.
  broke_high: bool — bar.high > last confirmed swing high (if any).
  broke_low:  bool — bar.low  < last confirmed swing low  (if any).

State is kept per pair — EURUSD ka swing GBPJPY ko affect nahi karta.

Hinglish: yeh module 1H bars dekh ke "yeh bar swing tha" detect karta hai.
Wick-break ko break maante hain — candle close ka wait nahi karte.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from data.bar_aggregator import Bar


@dataclass
class _PairState:
    """Per-pair rolling buffer + confirmed-swing history."""
    bars: Deque[Bar] = field(default_factory=lambda: deque(maxlen=3))
    highs: List[float] = field(default_factory=list)
    lows: List[float] = field(default_factory=list)


class SwingTracker:
    """Multi-pair swing detector. Strict-inequality fractal — equal highs
    don't form a swing (otherwise consolidations would spam swings).
    """

    def __init__(self) -> None:
        self._state: Dict[str, _PairState] = {}

    # -------------------------------------------------------------- main API

    def update(self, pair: str, bar: Bar) -> dict:
        st = self._state.setdefault(pair, _PairState())

        # 1) Wick-break check BEFORE appending — the break event is "this bar
        #    pierced prior structure". last_h / last_l are the most recently
        #    CONFIRMED swing levels (None until 3 bars have arrived).
        last_h = st.highs[-1] if st.highs else None
        last_l = st.lows[-1] if st.lows else None
        broke_high = last_h is not None and bar.high > last_h
        broke_low = last_l is not None and bar.low < last_l

        # 2) Slide the bar into the 3-deep window.
        st.bars.append(bar)

        # 3) Confirm a swing on the MIDDLE of the latest triplet.
        new_swing_high: Optional[float] = None
        new_swing_low: Optional[float] = None
        if len(st.bars) == 3:
            left, mid, right = st.bars[0], st.bars[1], st.bars[2]
            if mid.high > left.high and mid.high > right.high:
                st.highs.append(mid.high)
                new_swing_high = mid.high
            if mid.low < left.low and mid.low < right.low:
                st.lows.append(mid.low)
                new_swing_low = mid.low

        return {
            "new_swing_high": new_swing_high,
            "new_swing_low": new_swing_low,
            "broke_high": broke_high,
            "broke_low": broke_low,
        }

    # --------------------------------------------------------------- getters

    def get_last_swing_high(self, pair: str) -> Optional[float]:
        st = self._state.get(pair)
        if not st or not st.highs:
            return None
        return st.highs[-1]

    def get_last_swing_low(self, pair: str) -> Optional[float]:
        st = self._state.get(pair)
        if not st or not st.lows:
            return None
        return st.lows[-1]

    # ------------------------------------- pure helpers (instance-free OK)

    @staticmethod
    def is_break_of_high(bar: Bar, swing_high: float) -> bool:
        return bar.high > swing_high

    @staticmethod
    def is_break_of_low(bar: Bar, swing_low: float) -> bool:
        return bar.low < swing_low
