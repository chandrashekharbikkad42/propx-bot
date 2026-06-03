"""Setup #1 — Liquidity Sweep + Reversal (propX Multi-Setup spec §2).

Logic:
  1. Identify last confirmed HTF (1H) swing low / high.
  2. On LTF (15M), look for a bar that sweeps the swung level
     (low < swing_low - PENETRATION) within a recent window.
  3. Within SWEEP_RECLAIM_BARS = 3 LTF bars, price closes back inside
     (close > swing_low) — call that bar B_reclaim.
  4. B_reclaim itself is a rejection candle, OR the next LTF bar is.
  5. Entry = swung level (limit retest); SL = sweep wick ± buffer;
     TP ladder 1.5R / 2.5R.

Counter-trend filter (§2.5):
  Long sweep requires swept level is a swing LOW (always true here by
  construction). Short sweep requires swung level is a swing HIGH.
  We discard if HTF trend is BEARISH and the level was a swing HIGH (long
  setup at HH inside a downtrend is continuation, not reversal) — but a
  long sweep at a swing LOW is permitted in any trend per spec.

PatternSignal mapping:
  pattern_name = "LIQ_SWEEP"
  tp           = TP2 (1.5R / 2.5R ladder; TP1 encoded in confluences_met)
  confluences_met:
    - "long_sweep_low" or "short_sweep_high"
    - f"trend_{BULLISH|BEARISH|RANGE}"
    - f"penetration_{pips:.1f}"
    - f"reclaim_+{n}" where n = bars between sweep and reclaim
    - f"tp1_{price:.5f}"

Hinglish: HTF swing wahaan se LTF me jhaadu chala, fir wapas andar aaya,
rejection candle bani — entry retest pe, SL wick ke peeche, TP ladder.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple

from data.bar_aggregator import Bar
from strategy.patterns.base import (
    Direction, Grade, MarketContext, PatternDetector, PatternSignal,
)
from strategy.patterns._multi_setup_common import (
    SwingKind, Trend,
    adaptive_imp_threshold_pips,
    atr_in_pips, classify_trend, compute_atr,
    find_swings, is_rejection_bearish, is_rejection_bullish,
    last_swing, pips_to_price, price_to_pips,
)
from config.multi_setup_config import (
    ATR_LEN,
    ENTRY_BUFFER_PIPS_DEFAULT,
    MIN_RISK_PIPS,
    SWEEP_MAX_PENETRATION_ATR_MULT,
    SWEEP_PENETRATION_ATR_MULT, SWEEP_PENETRATION_PIPS_FLOOR,
    SWEEP_RECLAIM_BARS,
    TP1_R, TP2_R,
    pip_size_for, sl_buffer_pips_for,
)


# Grade A when trend supports the reversal (i.e. RANGE or trend-against-sweep
# direction, since sweep is a reversal setup the desirable HTF state is the
# OPPOSITE of trade direction or RANGE).
_GRADE_A_CONFIDENCE: float = 0.85
_GRADE_B_CONFIDENCE: float = 0.70


class LiquiditySweepDetector(PatternDetector):
    """Spec §2 — sweep + close-back + rejection. One signal per call max."""

    name: str = "LIQ_SWEEP"
    # We need ~5 LTF bars minimum to form a sweep+reclaim+rejection sequence
    # and a tail to compute ATR adaptively. Enforce conservatively.
    min_bars_required: int = ATR_LEN + 5
    timeframe: str = "15M"

    def detect(
        self, bars: Sequence[Bar], context: MarketContext
    ) -> Optional[PatternSignal]:
        ltf_bars = bars
        if len(ltf_bars) < self.min_bars_required:
            return None
        htf_bars = context.htf_bars or ()
        if len(htf_bars) < ATR_LEN + 2 * 3 + 1:  # need swings + ATR
            return None

        symbol = context.symbol

        # ATR — closed bars only (the caller is expected to feed closed bars).
        atr_ltf_price = compute_atr(ltf_bars, ATR_LEN)
        atr_htf_price = compute_atr(htf_bars, ATR_LEN)
        if atr_ltf_price is None:
            return None
        atr_ltf_pips = atr_in_pips(atr_ltf_price, symbol)

        # HTF swings + trend.
        swings = find_swings(htf_bars)
        if not swings:
            return None
        trend = classify_trend(swings)

        # Try long first (sweep of swing low), then short.
        sig = self._try_long(ltf_bars, swings, trend, symbol, atr_ltf_pips)
        if sig is not None:
            return sig
        sig = self._try_short(ltf_bars, swings, trend, symbol, atr_ltf_pips)
        return sig

    # --------------------------------------------------------------- internals

    def _try_long(
        self,
        ltf_bars: Sequence[Bar],
        swings,
        trend: Trend,
        symbol: str,
        atr_ltf_pips: float,
    ) -> Optional[PatternSignal]:
        swing_low = last_swing(swings, SwingKind.LOW)
        if swing_low is None:
            return None
        P = swing_low.price

        last_idx = len(ltf_bars) - 1
        # Confirming bar must be a bullish rejection candle.
        if not is_rejection_bullish(ltf_bars, last_idx):
            return None

        pen_pips = max(SWEEP_PENETRATION_PIPS_FLOOR,
                       SWEEP_PENETRATION_ATR_MULT * atr_ltf_pips)
        max_pen_price = pips_to_price(
            SWEEP_MAX_PENETRATION_ATR_MULT * atr_ltf_pips, symbol,
        )
        pen_price = pips_to_price(pen_pips, symbol)
        if pen_price <= 0 or max_pen_price <= 0:
            return None

        found = _find_sweep_reclaim_long(
            ltf_bars, P, pen_price, max_pen_price, last_idx,
        )
        if found is None:
            return None
        i_sweep, i_reclaim = found

        # Construct trade.
        sweep_low = ltf_bars[i_sweep].low
        pen_actual_pips = price_to_pips(P - sweep_low, symbol)

        entry = P + pips_to_price(ENTRY_BUFFER_PIPS_DEFAULT, symbol)
        sl = sweep_low - pips_to_price(sl_buffer_pips_for(symbol), symbol)
        risk_price = entry - sl
        risk_pips = price_to_pips(risk_price, symbol)
        if risk_pips < MIN_RISK_PIPS:
            return None
        tp1 = entry + TP1_R * risk_price
        tp2 = entry + TP2_R * risk_price
        if not (sl < entry < tp2):
            return None

        # Grade — A when trend isn't strongly against the reversal (sweep of
        # a low in a downtrend is a low-quality counter-trend trade; allowed
        # but graded B).
        grade = Grade.A if trend != Trend.BEARISH else Grade.B
        confidence = _GRADE_A_CONFIDENCE if grade == Grade.A else _GRADE_B_CONFIDENCE

        return PatternSignal(
            pattern_name=self.name,
            symbol=symbol,
            direction=Direction.BUY,
            entry=entry, sl=sl, tp=tp2,
            confidence=confidence, grade=grade,
            confluences_met=(
                "long_sweep_low",
                f"trend_{trend.value}",
                f"penetration_{pen_actual_pips:.1f}p",
                f"reclaim_+{i_reclaim - i_sweep}",
                f"tp1_{tp1:.5f}",
            ),
            bar_time_msc=ltf_bars[last_idx].time_msc,
        )

    def _try_short(
        self,
        ltf_bars: Sequence[Bar],
        swings,
        trend: Trend,
        symbol: str,
        atr_ltf_pips: float,
    ) -> Optional[PatternSignal]:
        swing_high = last_swing(swings, SwingKind.HIGH)
        if swing_high is None:
            return None
        P = swing_high.price

        last_idx = len(ltf_bars) - 1
        if not is_rejection_bearish(ltf_bars, last_idx):
            return None

        pen_pips = max(SWEEP_PENETRATION_PIPS_FLOOR,
                       SWEEP_PENETRATION_ATR_MULT * atr_ltf_pips)
        max_pen_price = pips_to_price(
            SWEEP_MAX_PENETRATION_ATR_MULT * atr_ltf_pips, symbol,
        )
        pen_price = pips_to_price(pen_pips, symbol)
        if pen_price <= 0 or max_pen_price <= 0:
            return None

        found = _find_sweep_reclaim_short(
            ltf_bars, P, pen_price, max_pen_price, last_idx,
        )
        if found is None:
            return None
        i_sweep, i_reclaim = found

        sweep_high = ltf_bars[i_sweep].high
        pen_actual_pips = price_to_pips(sweep_high - P, symbol)

        entry = P - pips_to_price(ENTRY_BUFFER_PIPS_DEFAULT, symbol)
        sl = sweep_high + pips_to_price(sl_buffer_pips_for(symbol), symbol)
        risk_price = sl - entry
        risk_pips = price_to_pips(risk_price, symbol)
        if risk_pips < MIN_RISK_PIPS:
            return None
        tp1 = entry - TP1_R * risk_price
        tp2 = entry - TP2_R * risk_price
        if not (tp2 < entry < sl):
            return None

        grade = Grade.A if trend != Trend.BULLISH else Grade.B
        confidence = _GRADE_A_CONFIDENCE if grade == Grade.A else _GRADE_B_CONFIDENCE

        return PatternSignal(
            pattern_name=self.name,
            symbol=symbol,
            direction=Direction.SELL,
            entry=entry, sl=sl, tp=tp2,
            confidence=confidence, grade=grade,
            confluences_met=(
                "short_sweep_high",
                f"trend_{trend.value}",
                f"penetration_{pen_actual_pips:.1f}p",
                f"reclaim_+{i_reclaim - i_sweep}",
                f"tp1_{tp1:.5f}",
            ),
            bar_time_msc=ltf_bars[last_idx].time_msc,
        )


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------

def _find_sweep_reclaim_long(
    ltf_bars: Sequence[Bar],
    P: float,
    pen_price: float,
    max_pen_price: float,
    last_idx: int,
) -> Optional[Tuple[int, int]]:
    """Return (i_sweep, i_reclaim) consistent with spec §2.1, or None.

    Two valid cases:
      A. B_reclaim == bars[-1] (rejection at last bar IS the reclaim).
         → last_idx.close > P, and some bar in
           [last_idx - SWEEP_RECLAIM_BARS + 1 .. last_idx] is the sweep.
      B. B_reclaim == bars[-2] (rejection at last bar is the bar AFTER reclaim).
         → bars[-2].close > P, and some bar in
           [last_idx - SWEEP_RECLAIM_BARS .. last_idx - 1] is the sweep.

    For each case we take the EARLIEST sweep that satisfies penetration limits.
    """
    # Case A
    if ltf_bars[last_idx].close > P:
        lo = max(0, last_idx - SWEEP_RECLAIM_BARS + 1)
        for i in range(lo, last_idx + 1):
            pen_actual = P - ltf_bars[i].low
            if pen_actual > pen_price and pen_actual <= max_pen_price:
                return i, last_idx
    # Case B
    if last_idx >= 1 and ltf_bars[last_idx - 1].close > P:
        lo = max(0, last_idx - SWEEP_RECLAIM_BARS)
        for i in range(lo, last_idx):
            pen_actual = P - ltf_bars[i].low
            if pen_actual > pen_price and pen_actual <= max_pen_price:
                return i, last_idx - 1
    return None


def _find_sweep_reclaim_short(
    ltf_bars: Sequence[Bar],
    P: float,
    pen_price: float,
    max_pen_price: float,
    last_idx: int,
) -> Optional[Tuple[int, int]]:
    # Case A
    if ltf_bars[last_idx].close < P:
        lo = max(0, last_idx - SWEEP_RECLAIM_BARS + 1)
        for i in range(lo, last_idx + 1):
            pen_actual = ltf_bars[i].high - P
            if pen_actual > pen_price and pen_actual <= max_pen_price:
                return i, last_idx
    # Case B
    if last_idx >= 1 and ltf_bars[last_idx - 1].close < P:
        lo = max(0, last_idx - SWEEP_RECLAIM_BARS)
        for i in range(lo, last_idx):
            pen_actual = ltf_bars[i].high - P
            if pen_actual > pen_price and pen_actual <= max_pen_price:
                return i, last_idx - 1
    return None
