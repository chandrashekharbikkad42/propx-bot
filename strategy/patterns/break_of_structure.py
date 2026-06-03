"""Setup #3 — Break of Structure + Retest (propX Multi-Setup spec §4).

Logic (bullish BoS):
  1. Walk HTF (1H) — find the most recent confirmed swing high `SH` at
     price `P_BoS`.
  2. A subsequent 1H bar `B_break` closes > P_BoS + BOS_BUFFER_PIPS
     (adaptive to ATR1H).
  3. The trend BEFORE `B_break` must be BEARISH or RANGE (this break
     CHANGES structure; continuation breaks are handled by Setup #2 OB).
  4. Once `B_break` fires, the broken level becomes a retest zone:
     [P_BoS - tol, P_BoS + tol]. Zone expires after BOS_MAX_AGE_BARS_HTF
     (~100h) or after a 1H close back below P_BoS - buffer.
  5. On LTF (15M), price re-enters zone from above
     (low ≤ P_BoS + tol). Within BOS_RETEST_BARS_LTF (=4) LTF bars,
     a bullish rejection candle (§1.8) prints with low ≥ P_BoS - tol.
  6. Entry = market on rejection close (default).
     SL = min(P_BoS - tol, low of last 3 LTF bars) - buffer.
     TP ladder 1.5R / 2.5R.

PatternSignal mapping:
  pattern_name = "BOS_RETEST"
  tp           = TP2
  confluences_met:
    - "bullish_bos" or "bearish_bos"
    - f"prev_trend_{trend.value}"  (BEARISH or RANGE)
    - f"bos_age_{n}h"
    - f"break_displacement_{pips:.1f}p"
    - f"tp1_{price:.5f}"

Hinglish: structure flip ka setup. Pehle range/bearish tha, fir HH ka break
hua, fir wapas broken level pe retest pe rejection candle → entry.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from data.bar_aggregator import Bar
from strategy.patterns.base import (
    Direction, Grade, MarketContext, PatternDetector, PatternSignal,
)
from strategy.patterns._multi_setup_common import (
    SwingKind, Trend,
    atr_in_pips, classify_trend, compute_atr,
    find_swings,
    is_rejection_bearish, is_rejection_bullish,
    pips_to_price, price_to_pips,
)
from config.multi_setup_config import (
    ATR_LEN,
    BOS_BUFFER_ATR_MULT, BOS_BUFFER_PIPS_FLOOR,
    BOS_MAX_AGE_BARS_HTF,
    BOS_RETEST_BARS_LTF,
    BOS_RETEST_TOLERANCE_ATR_MULT, BOS_RETEST_TOLERANCE_FLOOR,
    MIN_RISK_PIPS,
    TP1_R, TP2_R,
    sl_buffer_pips_for,
)


_GRADE_A_CONFIDENCE: float = 0.85
_GRADE_B_CONFIDENCE: float = 0.70


@dataclass(frozen=True)
class _BoSEvent:
    direction: int                   # +1 bullish (break of high), -1 bearish
    level_price: float               # P_BoS (the swing high/low broken)
    break_idx: int                   # HTF index of B_break
    break_time_msc: int
    prev_trend: Trend                # trend BEFORE B_break (must be BEARISH or RANGE)
    age_htf_bars: int
    displacement_pips: float         # |B_break.close - P_BoS|


class BreakOfStructureDetector(PatternDetector):
    """Spec §4. Looks for the latest valid (unmitigated, unexpired) BoS event
    and confirms via LTF retest + rejection."""

    name: str = "BOS_RETEST"
    min_bars_required: int = ATR_LEN + 2
    timeframe: str = "15M"

    def detect(
        self, bars: Sequence[Bar], context: MarketContext
    ) -> Optional[PatternSignal]:
        ltf_bars = bars
        if len(ltf_bars) < self.min_bars_required:
            return None
        htf_bars = context.htf_bars or ()
        if len(htf_bars) < ATR_LEN + 2 * 3 + 2:
            return None

        symbol = context.symbol

        atr_htf_price = compute_atr(htf_bars, ATR_LEN)
        if atr_htf_price is None:
            return None
        atr_htf_pips = atr_in_pips(atr_htf_price, symbol)

        buffer_pips = max(BOS_BUFFER_PIPS_FLOOR, BOS_BUFFER_ATR_MULT * atr_htf_pips)
        tol_pips = max(BOS_RETEST_TOLERANCE_FLOOR,
                       BOS_RETEST_TOLERANCE_ATR_MULT * atr_htf_pips)
        buffer_price = pips_to_price(buffer_pips, symbol)
        tol_price = pips_to_price(tol_pips, symbol)

        event = _find_latest_bos_event(htf_bars, symbol, buffer_price)
        if event is None:
            return None

        last_idx = len(ltf_bars) - 1
        if event.direction == +1:
            return self._try_bullish_retest(
                ltf_bars, event, tol_price, symbol, last_idx,
            )
        else:
            return self._try_bearish_retest(
                ltf_bars, event, tol_price, symbol, last_idx,
            )

    # --------------------------------------------------------------- internals

    def _try_bullish_retest(
        self,
        ltf_bars: Sequence[Bar],
        event: _BoSEvent,
        tol_price: float,
        symbol: str,
        last_idx: int,
    ) -> Optional[PatternSignal]:
        if not is_rejection_bullish(ltf_bars, last_idx):
            return None
        zone_low = event.level_price - tol_price
        zone_high = event.level_price + tol_price

        # First touch into the zone, within BOS_RETEST_BARS_LTF, AFTER break time.
        lo = max(0, last_idx - BOS_RETEST_BARS_LTF + 1)
        first_touch = -1
        for i in range(lo, last_idx + 1):
            if ltf_bars[i].time_msc <= event.break_time_msc:
                continue
            if ltf_bars[i].low <= zone_high:
                first_touch = i
                break
        if first_touch < 0:
            return None

        # Spec §4.2 — invalidate if any LTF bar within the retest window
        # closes below zone_low.
        for i in range(first_touch, last_idx + 1):
            if ltf_bars[i].close < zone_low:
                return None

        # SL = min(zone_low, low of last 3 LTF bars) - buffer.
        last3_low = min(b.low for b in ltf_bars[max(0, last_idx - 2): last_idx + 1])
        sl_base = min(zone_low, last3_low)
        sl = sl_base - pips_to_price(sl_buffer_pips_for(symbol), symbol)

        # Entry — market on rejection close (default).
        entry = ltf_bars[last_idx].close

        risk_price = entry - sl
        risk_pips = price_to_pips(risk_price, symbol)
        if risk_pips < MIN_RISK_PIPS:
            return None
        tp1 = entry + TP1_R * risk_price
        tp2 = entry + TP2_R * risk_price
        if not (sl < entry < tp2):
            return None

        # Grade — BoS is itself a structure change; A when the break came from
        # a clean BEARISH trend (real flip), B when from RANGE.
        grade = Grade.A if event.prev_trend == Trend.BEARISH else Grade.B
        confidence = _GRADE_A_CONFIDENCE if grade == Grade.A else _GRADE_B_CONFIDENCE

        return PatternSignal(
            pattern_name=self.name,
            symbol=symbol,
            direction=Direction.BUY,
            entry=entry, sl=sl, tp=tp2,
            confidence=confidence, grade=grade,
            confluences_met=(
                "bullish_bos",
                f"prev_trend_{event.prev_trend.value}",
                f"bos_age_{event.age_htf_bars}h",
                f"break_displacement_{event.displacement_pips:.1f}p",
                f"tp1_{tp1:.5f}",
            ),
            bar_time_msc=ltf_bars[last_idx].time_msc,
        )

    def _try_bearish_retest(
        self,
        ltf_bars: Sequence[Bar],
        event: _BoSEvent,
        tol_price: float,
        symbol: str,
        last_idx: int,
    ) -> Optional[PatternSignal]:
        if not is_rejection_bearish(ltf_bars, last_idx):
            return None
        zone_low = event.level_price - tol_price
        zone_high = event.level_price + tol_price

        lo = max(0, last_idx - BOS_RETEST_BARS_LTF + 1)
        first_touch = -1
        for i in range(lo, last_idx + 1):
            if ltf_bars[i].time_msc <= event.break_time_msc:
                continue
            if ltf_bars[i].high >= zone_low:
                first_touch = i
                break
        if first_touch < 0:
            return None

        for i in range(first_touch, last_idx + 1):
            if ltf_bars[i].close > zone_high:
                return None

        last3_high = max(b.high for b in ltf_bars[max(0, last_idx - 2): last_idx + 1])
        sl_base = max(zone_high, last3_high)
        sl = sl_base + pips_to_price(sl_buffer_pips_for(symbol), symbol)

        entry = ltf_bars[last_idx].close

        risk_price = sl - entry
        risk_pips = price_to_pips(risk_price, symbol)
        if risk_pips < MIN_RISK_PIPS:
            return None
        tp1 = entry - TP1_R * risk_price
        tp2 = entry - TP2_R * risk_price
        if not (tp2 < entry < sl):
            return None

        grade = Grade.A if event.prev_trend == Trend.BULLISH else Grade.B
        confidence = _GRADE_A_CONFIDENCE if grade == Grade.A else _GRADE_B_CONFIDENCE

        return PatternSignal(
            pattern_name=self.name,
            symbol=symbol,
            direction=Direction.SELL,
            entry=entry, sl=sl, tp=tp2,
            confidence=confidence, grade=grade,
            confluences_met=(
                "bearish_bos",
                f"prev_trend_{event.prev_trend.value}",
                f"bos_age_{event.age_htf_bars}h",
                f"break_displacement_{event.displacement_pips:.1f}p",
                f"tp1_{tp1:.5f}",
            ),
            bar_time_msc=ltf_bars[last_idx].time_msc,
        )


# ---------------------------------------------------------------------------
# BoS finder
# ---------------------------------------------------------------------------

def _find_latest_bos_event(
    htf_bars: Sequence[Bar],
    symbol: str,
    buffer_price: float,
) -> Optional[_BoSEvent]:
    """Find the most recent valid (unmitigated, unexpired) BoS.

    Bullish BoS:
      - Pick a confirmed swing high SH.
      - A later 1H bar `B_break` closes > SH + buffer.
      - Trend computed from swings up to (and including) the swing immediately
        before SH is BEARISH or RANGE.
      - Not yet mitigated: no 1H close < SH - buffer after the break.
      - Age ≤ BOS_MAX_AGE_BARS_HTF since break.

    Bearish BoS: mirror.

    Scans from newest backwards; returns the FIRST matching event.
    """
    n = len(htf_bars)
    if n < 2:
        return None
    latest_complete = n - 1
    # We'll scan candidate B_break indices from newest backwards.
    min_break_idx = max(0, latest_complete - BOS_MAX_AGE_BARS_HTF)

    swings = find_swings(htf_bars)
    if not swings:
        return None

    for break_idx in range(latest_complete, min_break_idx - 1, -1):
        b = htf_bars[break_idx]
        # Swings strictly before this bar (a swing must be confirmed before
        # the break candle exists in our walk; we already have all swings
        # because find_swings filtered in-progress swings out).
        swings_before = [s for s in swings if s.index < break_idx]
        if not swings_before:
            continue

        # Try bullish break: pick most recent swing high before this bar.
        sh_candidate = None
        for s in reversed(swings_before):
            if s.kind == SwingKind.HIGH:
                sh_candidate = s
                break
        if sh_candidate is not None:
            P = sh_candidate.price
            if b.close > P + buffer_price:
                # Trend before this break: classify on swings strictly before sh_candidate
                # (we want the structure state PRIOR to the break swing forming).
                pre_swings = [s for s in swings_before if s.index <= sh_candidate.index]
                prev_trend = classify_trend(pre_swings)
                if prev_trend in (Trend.BEARISH, Trend.RANGE):
                    # Mitigation: any later bar closes < P - buffer?
                    mitigated = False
                    for j in range(break_idx + 1, latest_complete + 1):
                        if htf_bars[j].close < P - buffer_price:
                            mitigated = True
                            break
                    if not mitigated:
                        disp_pips = price_to_pips(b.close - P, symbol)
                        return _BoSEvent(
                            direction=+1, level_price=P,
                            break_idx=break_idx, break_time_msc=b.time_msc,
                            prev_trend=prev_trend,
                            age_htf_bars=latest_complete - break_idx,
                            displacement_pips=disp_pips,
                        )

        # Try bearish break: most recent swing low before this bar.
        sl_candidate = None
        for s in reversed(swings_before):
            if s.kind == SwingKind.LOW:
                sl_candidate = s
                break
        if sl_candidate is not None:
            P = sl_candidate.price
            if b.close < P - buffer_price:
                pre_swings = [s for s in swings_before if s.index <= sl_candidate.index]
                prev_trend = classify_trend(pre_swings)
                if prev_trend in (Trend.BULLISH, Trend.RANGE):
                    mitigated = False
                    for j in range(break_idx + 1, latest_complete + 1):
                        if htf_bars[j].close > P + buffer_price:
                            mitigated = True
                            break
                    if not mitigated:
                        disp_pips = price_to_pips(P - b.close, symbol)
                        return _BoSEvent(
                            direction=-1, level_price=P,
                            break_idx=break_idx, break_time_msc=b.time_msc,
                            prev_trend=prev_trend,
                            age_htf_bars=latest_complete - break_idx,
                            displacement_pips=disp_pips,
                        )

    return None
