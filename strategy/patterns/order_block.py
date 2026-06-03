"""Setup #2 — Order Block (propX Multi-Setup spec §3).

Logic (bullish OB example; bearish = mirror):
  1. Walk HTF (1H) backwards looking for the most recent bullish impulse:
     - A bearish bar `B_OB` immediately followed by N_IMP_HTF (=2) bullish
       closes whose total displacement ≥ OB_IMP_MIN_PIPS (adaptive to ATR1H).
     - The impulse must clear B_OB.high by ≥ OB_IMP_CLEAR_FRAC × min displacement.
  2. OB lives for OB_MAX_AGE_BARS_HTF (~200 1H bars, ~8 trading days), or
     until a 1H close BELOW B_OB.low (mitigation by full breach).
  3. On LTF (15M), price re-enters the OB zone (low ≤ B_OB.high).
  4. Within OB_RETEST_BARS_LTF (=3) LTF bars of the touch, a bullish
     rejection candle (§1.8) confirms.
  5. Entry = top of OB (limit BUY at B_OB.high); SL = B_OB.low - buffer;
     TP ladder 1.5R / 2.5R.

Trend filter (§3.6): HTF trend BEARISH → SKIP bullish OB (counter-trend);
                     HTF trend BULLISH → SKIP bearish OB.

PatternSignal mapping:
  pattern_name = "ORDER_BLOCK"
  tp           = TP2 (1.5R / 2.5R ladder; TP1 in confluences_met)
  confluences_met:
    - "bullish_ob" or "bearish_ob"
    - f"trend_{...}"
    - f"ob_age_{n}h"
    - f"ob_imp_{pips:.1f}p"
    - f"tp1_{price:.5f}"

Hinglish: HTF pe impulse ke aage ka opposing candle dhundo. Wahi OB hai.
Price wapas aaye → retest pe rejection candle ho → limit entry zone ke top pe.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple

from data.bar_aggregator import Bar
from strategy.patterns.base import (
    Direction, Grade, MarketContext, PatternDetector, PatternSignal,
)
from strategy.patterns._multi_setup_common import (
    Trend,
    atr_in_pips, classify_trend, compute_atr,
    detect_impulse_htf, find_swings, htf_index_at_or_before,
    is_bearish, is_bullish, is_rejection_bearish, is_rejection_bullish,
    pips_to_price, price_to_pips,
)
from config.multi_setup_config import (
    ATR_LEN,
    MIN_RISK_PIPS,
    N_IMP_HTF,
    OB_IMP_CLEAR_FRAC, OB_IMP_MIN_PIPS_FLOOR,
    OB_MAX_AGE_BARS_HTF, OB_RETEST_BARS_LTF,
    TP1_R, TP2_R,
    sl_buffer_pips_for,
)


_GRADE_A_CONFIDENCE: float = 0.85
_GRADE_B_CONFIDENCE: float = 0.70


class OrderBlockDetector(PatternDetector):
    """Spec §3. Walks HTF for fresh OB, gates LTF retest + rejection."""

    name: str = "ORDER_BLOCK"
    min_bars_required: int = ATR_LEN + 2  # LTF; HTF checked separately
    timeframe: str = "15M"

    def detect(
        self, bars: Sequence[Bar], context: MarketContext
    ) -> Optional[PatternSignal]:
        ltf_bars = bars
        if len(ltf_bars) < self.min_bars_required:
            return None
        htf_bars = context.htf_bars or ()
        if len(htf_bars) < ATR_LEN + N_IMP_HTF + 1:
            return None
        symbol = context.symbol

        atr_htf_price = compute_atr(htf_bars, ATR_LEN)
        if atr_htf_price is None:
            return None
        atr_htf_pips = atr_in_pips(atr_htf_price, symbol)

        # HTF trend gate (using full swing structure on HTF).
        swings = find_swings(htf_bars)
        trend = classify_trend(swings)

        # Find most recent bullish + bearish OBs.
        bull_ob = _find_latest_bullish_ob(htf_bars, symbol, atr_htf_pips)
        bear_ob = _find_latest_bearish_ob(htf_bars, symbol, atr_htf_pips)

        last_idx = len(ltf_bars) - 1

        # Long bullish OB retest.
        if bull_ob is not None and trend != Trend.BEARISH:
            sig = self._try_bullish_retest(
                ltf_bars, bull_ob, trend, symbol, last_idx,
            )
            if sig is not None:
                return sig

        if bear_ob is not None and trend != Trend.BULLISH:
            sig = self._try_bearish_retest(
                ltf_bars, bear_ob, trend, symbol, last_idx,
            )
            if sig is not None:
                return sig

        return None

    # --------------------------------------------------------------- internals

    def _try_bullish_retest(
        self,
        ltf_bars: Sequence[Bar],
        ob: "_OrderBlock",
        trend: Trend,
        symbol: str,
        last_idx: int,
    ) -> Optional[PatternSignal]:
        # Find the LTF index when price first re-entered the OB zone in the
        # recent retest window. We accept that bars[-1] is the rejection bar
        # AND that the retest entry happened within OB_RETEST_BARS_LTF bars.
        if not is_rejection_bullish(ltf_bars, last_idx):
            return None

        # Locate first touch in [last_idx - OB_RETEST_BARS_LTF + 1, last_idx].
        lo = max(0, last_idx - OB_RETEST_BARS_LTF + 1)
        first_touch = -1
        for i in range(lo, last_idx + 1):
            if ltf_bars[i].low <= ob.high and ltf_bars[i].time_msc > ob.created_time_msc:
                first_touch = i
                break
        if first_touch < 0:
            return None

        # Spec §3.6 — skip if the retest is a strong bearish bar that engulfs OB top.
        rb = ltf_bars[first_touch]
        if is_bearish(rb) and rb.close < ob.low:
            return None

        # Trade construction.
        entry = ob.high  # top of OB
        sl = ob.low - pips_to_price(sl_buffer_pips_for(symbol), symbol)
        risk_price = entry - sl
        risk_pips = price_to_pips(risk_price, symbol)
        if risk_pips < MIN_RISK_PIPS:
            return None
        tp1 = entry + TP1_R * risk_price
        tp2 = entry + TP2_R * risk_price
        if not (sl < entry < tp2):
            return None

        grade = Grade.A if trend == Trend.BULLISH else Grade.B
        confidence = _GRADE_A_CONFIDENCE if grade == Grade.A else _GRADE_B_CONFIDENCE

        return PatternSignal(
            pattern_name=self.name,
            symbol=symbol,
            direction=Direction.BUY,
            entry=entry, sl=sl, tp=tp2,
            confidence=confidence, grade=grade,
            confluences_met=(
                "bullish_ob",
                f"trend_{trend.value}",
                f"ob_age_{ob.age_htf_bars}h",
                f"ob_imp_{ob.impulse_pips:.1f}p",
                f"tp1_{tp1:.5f}",
            ),
            bar_time_msc=ltf_bars[last_idx].time_msc,
        )

    def _try_bearish_retest(
        self,
        ltf_bars: Sequence[Bar],
        ob: "_OrderBlock",
        trend: Trend,
        symbol: str,
        last_idx: int,
    ) -> Optional[PatternSignal]:
        if not is_rejection_bearish(ltf_bars, last_idx):
            return None

        lo = max(0, last_idx - OB_RETEST_BARS_LTF + 1)
        first_touch = -1
        for i in range(lo, last_idx + 1):
            if ltf_bars[i].high >= ob.low and ltf_bars[i].time_msc > ob.created_time_msc:
                first_touch = i
                break
        if first_touch < 0:
            return None

        rb = ltf_bars[first_touch]
        if is_bullish(rb) and rb.close > ob.high:
            return None

        entry = ob.low  # bottom of bearish OB
        sl = ob.high + pips_to_price(sl_buffer_pips_for(symbol), symbol)
        risk_price = sl - entry
        risk_pips = price_to_pips(risk_price, symbol)
        if risk_pips < MIN_RISK_PIPS:
            return None
        tp1 = entry - TP1_R * risk_price
        tp2 = entry - TP2_R * risk_price
        if not (tp2 < entry < sl):
            return None

        grade = Grade.A if trend == Trend.BEARISH else Grade.B
        confidence = _GRADE_A_CONFIDENCE if grade == Grade.A else _GRADE_B_CONFIDENCE

        return PatternSignal(
            pattern_name=self.name,
            symbol=symbol,
            direction=Direction.SELL,
            entry=entry, sl=sl, tp=tp2,
            confidence=confidence, grade=grade,
            confluences_met=(
                "bearish_ob",
                f"trend_{trend.value}",
                f"ob_age_{ob.age_htf_bars}h",
                f"ob_imp_{ob.impulse_pips:.1f}p",
                f"tp1_{tp1:.5f}",
            ),
            bar_time_msc=ltf_bars[last_idx].time_msc,
        )


# ---------------------------------------------------------------------------
# Internal OB record + finders
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass(frozen=True)
class _OrderBlock:
    low: float
    high: float
    created_idx: int            # HTF bar index of B_OB
    created_time_msc: int
    impulse_pips: float
    age_htf_bars: int           # how many HTF bars since OB formation


def _find_latest_bullish_ob(
    htf_bars: Sequence[Bar],
    symbol: str,
    atr_htf_pips: float,
) -> Optional[_OrderBlock]:
    """Walk HTF from newest backwards. For each potential B_OB (bearish bar
    followed by N_IMP_HTF bullish bars forming a valid impulse), return the
    most recent UNMITIGATED OB.

    Mitigation: any 1H close BELOW B_OB.low after creation invalidates the OB.
    Age cap: OB_MAX_AGE_BARS_HTF (default 200) — older OBs are stale.
    """
    n = len(htf_bars)
    # Scan from newest OB candidate backwards. B_OB candidate index i requires
    # i + N_IMP_HTF < n (room for the impulse confirm bars).
    latest_complete = n - 1  # newest closed HTF bar index
    # We need the impulse bars to be fully completed too.
    max_b_ob_idx = latest_complete - N_IMP_HTF
    min_b_ob_idx = max(0, latest_complete - OB_MAX_AGE_BARS_HTF)
    for i in range(max_b_ob_idx, min_b_ob_idx - 1, -1):
        b_ob = htf_bars[i]
        if not is_bearish(b_ob):
            continue
        imp = detect_impulse_htf(htf_bars, i + 1, symbol, atr_htf_pips, N_IMP_HTF)
        if not imp.is_impulsive or imp.direction != +1:
            continue
        # Clearance: impulse end close - B_OB.high >= 0.5 × threshold (in pips)
        impulse_last = htf_bars[i + N_IMP_HTF]
        clear_price = impulse_last.close - b_ob.high
        clear_pips = price_to_pips(clear_price, symbol)
        if clear_pips < OB_IMP_CLEAR_FRAC * imp.threshold_pips:
            continue
        # Mitigation check: any HTF bar after OB closes BELOW B_OB.low?
        mitigated = False
        for j in range(i + 1, latest_complete + 1):
            if htf_bars[j].close < b_ob.low:
                mitigated = True
                break
        if mitigated:
            continue
        return _OrderBlock(
            low=b_ob.low, high=b_ob.high,
            created_idx=i, created_time_msc=b_ob.time_msc,
            impulse_pips=imp.displacement_pips,
            age_htf_bars=latest_complete - i,
        )
    return None


def _find_latest_bearish_ob(
    htf_bars: Sequence[Bar],
    symbol: str,
    atr_htf_pips: float,
) -> Optional[_OrderBlock]:
    n = len(htf_bars)
    latest_complete = n - 1
    max_b_ob_idx = latest_complete - N_IMP_HTF
    min_b_ob_idx = max(0, latest_complete - OB_MAX_AGE_BARS_HTF)
    for i in range(max_b_ob_idx, min_b_ob_idx - 1, -1):
        b_ob = htf_bars[i]
        if not is_bullish(b_ob):
            continue
        imp = detect_impulse_htf(htf_bars, i + 1, symbol, atr_htf_pips, N_IMP_HTF)
        if not imp.is_impulsive or imp.direction != -1:
            continue
        impulse_last = htf_bars[i + N_IMP_HTF]
        clear_price = b_ob.low - impulse_last.close
        clear_pips = price_to_pips(clear_price, symbol)
        if clear_pips < OB_IMP_CLEAR_FRAC * imp.threshold_pips:
            continue
        mitigated = False
        for j in range(i + 1, latest_complete + 1):
            if htf_bars[j].close > b_ob.high:
                mitigated = True
                break
        if mitigated:
            continue
        return _OrderBlock(
            low=b_ob.low, high=b_ob.high,
            created_idx=i, created_time_msc=b_ob.time_msc,
            impulse_pips=imp.displacement_pips,
            age_htf_bars=latest_complete - i,
        )
    return None
