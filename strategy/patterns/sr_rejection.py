"""Setup #4 — Support/Resistance Rejection (propX Multi-Setup spec §5).

Logic (long at support):
  1. Find clean horizontal S/R levels on HTF (1H) — clusters of ≥ 3 wick
     touches over the last 200 bars (~8 trading days), each separated by
     ≥ LEVEL_MIN_GAP_BARS bars, with no competing level within ±tol×3.
  2. On LTF (15M), a bar `B_test` tags a support level
     (low ≤ P_sup + tol). Within SR_REJECT_BARS_LTF (=3) LTF bars,
     a bullish rejection candle prints with
     low ≥ P_sup - tol × SR_WICK_BREAK_FRAC.
  3. Entry = market on rejection close.
     SL = min(low of B_test .. confirming bar) - buffer.
     TP ladder 1.5R / 2.5R, with TP2 CAPPED at next opposing level.

Trend filter (§5.7):
  HTF BEARISH at resistance → SKIP (don't fight trend at trend-aligned barrier).
  HTF BULLISH at support    → A grade (trend bounce).
  HTF BEARISH at support    → B grade (counter-trend range bounce; allowed).

PatternSignal mapping:
  pattern_name = "SR_REJECTION"
  tp           = TP2 (capped per §5.6)
  confluences_met:
    - "support_bounce" or "resistance_rejection"
    - f"trend_{...}"
    - f"touches_{n}"
    - f"level_{price:.5f}"
    - f"tp1_{price:.5f}"

Hinglish: clean horizontal level pe rejection candle → bounce/rejection trade.
TP next opposing level se zyada nahi jaata.
"""

from __future__ import annotations
from typing import List, Optional, Sequence

from data.bar_aggregator import Bar
from strategy.patterns.base import (
    Direction, Grade, MarketContext, PatternDetector, PatternSignal,
)
from strategy.patterns._multi_setup_common import (
    SRLevel, Trend,
    atr_in_pips, classify_trend, compute_atr,
    find_sr_levels, find_swings,
    is_rejection_bearish, is_rejection_bullish,
    pips_to_price, price_to_pips,
)
from config.multi_setup_config import (
    ATR_LEN,
    LEVEL_CLUSTER_TOLERANCE_ATR_MULT, LEVEL_CLUSTER_TOLERANCE_FLOOR,
    MIN_RISK_PIPS,
    SR_REJECT_BARS_LTF, SR_WICK_BREAK_FRAC,
    TP1_R, TP2_R,
    sl_buffer_pips_for,
)


_GRADE_A_CONFIDENCE: float = 0.85
_GRADE_B_CONFIDENCE: float = 0.70


class SRRejectionDetector(PatternDetector):
    """Spec §5. Clean horizontal S/R + rejection candle."""

    name: str = "SR_REJECTION"
    min_bars_required: int = ATR_LEN + 2
    timeframe: str = "15M"

    def detect(
        self, bars: Sequence[Bar], context: MarketContext
    ) -> Optional[PatternSignal]:
        ltf_bars = bars
        if len(ltf_bars) < self.min_bars_required:
            return None
        htf_bars = context.htf_bars or ()
        if len(htf_bars) < ATR_LEN + 2 * 3 + 1:
            return None

        symbol = context.symbol

        atr_htf_price = compute_atr(htf_bars, ATR_LEN)
        if atr_htf_price is None:
            return None
        atr_htf_pips = atr_in_pips(atr_htf_price, symbol)

        # Cluster tolerance in price.
        tol_pips = max(LEVEL_CLUSTER_TOLERANCE_FLOOR,
                       LEVEL_CLUSTER_TOLERANCE_ATR_MULT * atr_htf_pips)
        tol_price = pips_to_price(tol_pips, symbol)
        if tol_price <= 0:
            return None

        levels = find_sr_levels(htf_bars, symbol, atr_htf_pips)
        if not levels:
            return None

        swings = find_swings(htf_bars)
        trend = classify_trend(swings)
        last_idx = len(ltf_bars) - 1

        # Try long bounces first (most recent support that just got tagged),
        # then short rejections.
        sig = self._try_long_bounce(
            ltf_bars, levels, trend, symbol, tol_price, last_idx,
        )
        if sig is not None:
            return sig
        sig = self._try_short_rejection(
            ltf_bars, levels, trend, symbol, tol_price, last_idx,
        )
        return sig

    # --------------------------------------------------------------- internals

    def _try_long_bounce(
        self,
        ltf_bars: Sequence[Bar],
        levels: Sequence[SRLevel],
        trend: Trend,
        symbol: str,
        tol_price: float,
        last_idx: int,
    ) -> Optional[PatternSignal]:
        if not is_rejection_bullish(ltf_bars, last_idx):
            return None

        cur_close = ltf_bars[last_idx].close
        supports = [L for L in levels if L.is_support and L.price < cur_close]
        if not supports:
            return None

        # Pick the highest support BELOW current close as the most likely tag.
        supports.sort(key=lambda L: L.price, reverse=True)
        chosen: Optional[SRLevel] = None
        i_test = -1
        confirm_low = None
        for L in supports:
            # Did the last SR_REJECT_BARS_LTF bars tag this level (low ≤ price+tol)?
            lo_idx = max(0, last_idx - SR_REJECT_BARS_LTF + 1)
            tag_idx = -1
            for i in range(lo_idx, last_idx + 1):
                if ltf_bars[i].low <= L.price + tol_price:
                    tag_idx = i
                    break
            if tag_idx < 0:
                continue
            # Reject if a body close pierced below the wick-break floor.
            wick_floor = L.price - tol_price * SR_WICK_BREAK_FRAC
            if any(b.close < wick_floor for b in ltf_bars[tag_idx: last_idx + 1]):
                continue
            chosen = L
            i_test = tag_idx
            confirm_low = min(b.low for b in ltf_bars[tag_idx: last_idx + 1])
            break

        if chosen is None:
            return None

        # SL = confirm_low - buffer
        sl = confirm_low - pips_to_price(sl_buffer_pips_for(symbol), symbol)
        entry = ltf_bars[last_idx].close
        risk_price = entry - sl
        risk_pips = price_to_pips(risk_price, symbol)
        if risk_pips < MIN_RISK_PIPS:
            return None
        tp1 = entry + TP1_R * risk_price
        tp2 = entry + TP2_R * risk_price

        # Cap TP2 at next opposing resistance - tol (spec §5.6).
        next_res = [L for L in levels if L.is_resistance and L.price > entry]
        if next_res:
            next_res.sort(key=lambda L: L.price)
            cap = next_res[0].price - tol_price
            if cap <= entry:
                # No room — SKIP.
                return None
            if tp2 > cap:
                tp2 = cap
            if tp1 > cap:
                return None  # TP1 beyond cap → skip per §5.6
        if not (sl < entry < tp2):
            return None

        # Trend: BULLISH/RANGE at support → A; BEARISH at support → B.
        grade = Grade.A if trend in (Trend.BULLISH, Trend.RANGE) else Grade.B
        confidence = _GRADE_A_CONFIDENCE if grade == Grade.A else _GRADE_B_CONFIDENCE

        return PatternSignal(
            pattern_name=self.name,
            symbol=symbol,
            direction=Direction.BUY,
            entry=entry, sl=sl, tp=tp2,
            confidence=confidence, grade=grade,
            confluences_met=(
                "support_bounce",
                f"trend_{trend.value}",
                f"touches_{chosen.touches}",
                f"level_{chosen.price:.5f}",
                f"tp1_{tp1:.5f}",
            ),
            bar_time_msc=ltf_bars[last_idx].time_msc,
        )

    def _try_short_rejection(
        self,
        ltf_bars: Sequence[Bar],
        levels: Sequence[SRLevel],
        trend: Trend,
        symbol: str,
        tol_price: float,
        last_idx: int,
    ) -> Optional[PatternSignal]:
        if not is_rejection_bearish(ltf_bars, last_idx):
            return None

        # Spec §5.7 — HTF BEARISH at resistance → SKIP (handled by other setups).
        if trend == Trend.BEARISH:
            return None

        cur_close = ltf_bars[last_idx].close
        resistances = [L for L in levels if L.is_resistance and L.price > cur_close]
        if not resistances:
            return None
        resistances.sort(key=lambda L: L.price)
        chosen: Optional[SRLevel] = None
        i_test = -1
        confirm_high = None
        for L in resistances:
            lo_idx = max(0, last_idx - SR_REJECT_BARS_LTF + 1)
            tag_idx = -1
            for i in range(lo_idx, last_idx + 1):
                if ltf_bars[i].high >= L.price - tol_price:
                    tag_idx = i
                    break
            if tag_idx < 0:
                continue
            wick_ceiling = L.price + tol_price * SR_WICK_BREAK_FRAC
            if any(b.close > wick_ceiling for b in ltf_bars[tag_idx: last_idx + 1]):
                continue
            chosen = L
            i_test = tag_idx
            confirm_high = max(b.high for b in ltf_bars[tag_idx: last_idx + 1])
            break

        if chosen is None:
            return None

        sl = confirm_high + pips_to_price(sl_buffer_pips_for(symbol), symbol)
        entry = ltf_bars[last_idx].close
        risk_price = sl - entry
        risk_pips = price_to_pips(risk_price, symbol)
        if risk_pips < MIN_RISK_PIPS:
            return None
        tp1 = entry - TP1_R * risk_price
        tp2 = entry - TP2_R * risk_price

        next_sup = [L for L in levels if L.is_support and L.price < entry]
        if next_sup:
            next_sup.sort(key=lambda L: L.price, reverse=True)
            cap = next_sup[0].price + tol_price
            if cap >= entry:
                return None
            if tp2 < cap:
                tp2 = cap
            if tp1 < cap:
                return None
        if not (tp2 < entry < sl):
            return None

        grade = Grade.A if trend == Trend.RANGE else Grade.B
        confidence = _GRADE_A_CONFIDENCE if grade == Grade.A else _GRADE_B_CONFIDENCE

        return PatternSignal(
            pattern_name=self.name,
            symbol=symbol,
            direction=Direction.SELL,
            entry=entry, sl=sl, tp=tp2,
            confidence=confidence, grade=grade,
            confluences_met=(
                "resistance_rejection",
                f"trend_{trend.value}",
                f"touches_{chosen.touches}",
                f"level_{chosen.price:.5f}",
                f"tp1_{tp1:.5f}",
            ),
            bar_time_msc=ltf_bars[last_idx].time_msc,
        )
