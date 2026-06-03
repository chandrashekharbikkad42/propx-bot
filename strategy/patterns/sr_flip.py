"""S/R Flip detector (docs/SR_FLIP_SPEC.md v1.0).

Pure price-action detector for support↔resistance flips:
  1. HTF cluster of swing highs (or lows) ≥ MIN_LEVEL_TOUCHES → "level".
  2. A clean HTF break of the level by ≥ break-margin.
  3. No HTF close re-crosses the level since the break (failed-flip filter).
  4. Current LTF bar retests + rejects the level in the flip direction.

PatternSignal mapping:
  pattern_name      = "SR_FLIP"
  tp                = TP2 (1.5R / 2.5R ladder)
  confluences_met:
    - "flip_long" / "flip_short"
    - "level_<touches>t"
    - "break_<bar_age>h"
    - "rejection_<pin|engulf>"
    - "tp1_<price>"
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from data.bar_aggregator import Bar
from strategy.patterns.base import (
    Direction, Grade, MarketContext, PatternDetector, PatternSignal,
)
from strategy.patterns._multi_setup_common import (
    SwingKind, compute_atr, find_swings,
    is_bullish_engulf, is_bullish_pin,
    is_bearish_engulf, is_bearish_pin,
    Swing,
)
from config.sr_flip_config import (
    ATR_LEN,
    BREAK_MARGIN_ATR_MULT, BREAK_MARGIN_PIPS_FLOOR,
    LEVEL_CLUSTER_TOLERANCE_ATR_MULT, LEVEL_CLUSTER_TOLERANCE_FLOOR,
    LEVEL_MAX_AGE_HTF_BARS, MAX_BREAK_AGE_HTF_BARS,
    MIN_LEVEL_TOUCHES, MIN_RISK_PIPS,
    REENTRY_BLOCK_PIPS,
    RETEST_LOOKBACK_LTF_BARS,
    RETEST_TOL_ATR_MULT, RETEST_TOL_PIPS_FLOOR,
    TP1_R, TP2_R,
    pip_size_for, sl_buffer_pips_for,
)


_GRADE_A_CONFIDENCE: float = 0.85
_GRADE_B_CONFIDENCE: float = 0.70


# Local pip helpers — use sr_flip_config.pip_size_for (which includes
# CADCHF + XAGUSD). The shared helpers in _multi_setup_common reach into
# multi_setup_config and would KeyError on those 2 symbols.
def pips_to_price(pips: float, symbol: str) -> float:
    return pips * pip_size_for(symbol)


def price_to_pips(price_delta: float, symbol: str) -> float:
    return price_delta / pip_size_for(symbol)


def atr_in_pips(atr_price: float, symbol: str) -> float:
    return atr_price / pip_size_for(symbol)


# ─────────────────────────────────────────────────────────────────────────────
# Level discovery — cluster HTF swing highs/lows by price
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Level:
    price: float
    touches: int
    last_swing_idx: int     # HTF bar index of most recent contributing swing


def _cluster_swings(
    swings: Sequence[Swing],
    tol_price: float,
    min_touches: int,
) -> List[_Level]:
    """Greedy-cluster swings by price. Two prices are in the same cluster if
    they are within `2 × tol_price` of the cluster anchor. Returns clusters of
    size ≥ min_touches.
    """
    if not swings:
        return []
    sorted_by_price = sorted(swings, key=lambda s: s.price)
    clusters: List[List[Swing]] = [[sorted_by_price[0]]]
    for s in sorted_by_price[1:]:
        anchor = clusters[-1][0].price
        if abs(s.price - anchor) <= 2.0 * tol_price:
            clusters[-1].append(s)
        else:
            clusters.append([s])

    out: List[_Level] = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        prices = sorted(c.price for c in cluster)
        median_price = prices[len(prices) // 2]
        last_idx = max(c.index for c in cluster)
        out.append(_Level(
            price=median_price, touches=len(cluster), last_swing_idx=last_idx,
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Break + retest search
# ─────────────────────────────────────────────────────────────────────────────

def _find_break_above(
    htf: Sequence[Bar], level_price: float, break_margin_price: float,
    reentry_block_price: float, last_swing_idx: int,
) -> Optional[int]:
    """Earliest HTF bar index `b > last_swing_idx` with close > L + margin,
    and no subsequent HTF close < L - reentry_block (failed-flip filter).
    Also enforces `len(htf) - b <= MAX_BREAK_AGE_HTF_BARS`.
    """
    n = len(htf)
    start = max(last_swing_idx + 1, n - MAX_BREAK_AGE_HTF_BARS)
    break_idx = -1
    for i in range(start, n):
        if htf[i].close > level_price + break_margin_price:
            break_idx = i
            break
    if break_idx < 0:
        return None
    for j in range(break_idx + 1, n):
        if htf[j].close < level_price - reentry_block_price:
            return None
    return break_idx


def _find_break_below(
    htf: Sequence[Bar], level_price: float, break_margin_price: float,
    reentry_block_price: float, last_swing_idx: int,
) -> Optional[int]:
    n = len(htf)
    start = max(last_swing_idx + 1, n - MAX_BREAK_AGE_HTF_BARS)
    break_idx = -1
    for i in range(start, n):
        if htf[i].close < level_price - break_margin_price:
            break_idx = i
            break
    if break_idx < 0:
        return None
    for j in range(break_idx + 1, n):
        if htf[j].close > level_price + reentry_block_price:
            return None
    return break_idx


def _prior_retest_within_debounce_long(
    ltf: Sequence[Bar], last_idx: int, level_price: float, retest_tol_price: float,
) -> bool:
    lo = max(0, last_idx - RETEST_LOOKBACK_LTF_BARS)
    for i in range(lo, last_idx):
        b = ltf[i]
        if b.low <= level_price + retest_tol_price and b.close > level_price:
            return True
    return False


def _prior_retest_within_debounce_short(
    ltf: Sequence[Bar], last_idx: int, level_price: float, retest_tol_price: float,
) -> bool:
    lo = max(0, last_idx - RETEST_LOOKBACK_LTF_BARS)
    for i in range(lo, last_idx):
        b = ltf[i]
        if b.high >= level_price - retest_tol_price and b.close < level_price:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class SRFlipDetector(PatternDetector):
    """S/R Flip. One signal per call max. MTF (1H + 15M)."""

    name: str = "SR_FLIP"
    timeframe: str = "15M"
    min_bars_required: int = ATR_LEN + RETEST_LOOKBACK_LTF_BARS + 5

    def detect(
        self, bars: Sequence[Bar], context: MarketContext
    ) -> Optional[PatternSignal]:
        ltf = bars
        if len(ltf) < self.min_bars_required:
            return None
        htf = context.htf_bars or ()
        if len(htf) < 30:
            return None

        last_idx = len(ltf) - 1

        atr_ltf_price = compute_atr(ltf, ATR_LEN)
        if atr_ltf_price is None or atr_ltf_price <= 0:
            return None
        atr_ltf_pips = atr_in_pips(atr_ltf_price, context.symbol)

        atr_htf_price = compute_atr(htf, ATR_LEN)
        if atr_htf_price is None or atr_htf_price <= 0:
            return None
        atr_htf_pips = atr_in_pips(atr_htf_price, context.symbol)

        # 1. HTF swings + cluster by direction.
        swings = find_swings(htf)
        if not swings:
            return None
        cluster_tol_pips = max(
            LEVEL_CLUSTER_TOLERANCE_FLOOR,
            LEVEL_CLUSTER_TOLERANCE_ATR_MULT * atr_htf_pips,
        )
        cluster_tol_price = pips_to_price(cluster_tol_pips, context.symbol)

        highs = [s for s in swings if s.kind == SwingKind.HIGH]
        lows = [s for s in swings if s.kind == SwingKind.LOW]
        resistance_levels = _cluster_swings(highs, cluster_tol_price, MIN_LEVEL_TOUCHES)
        support_levels = _cluster_swings(lows, cluster_tol_price, MIN_LEVEL_TOUCHES)

        # Drop stale levels.
        htf_n = len(htf)
        resistance_levels = [
            L for L in resistance_levels
            if htf_n - L.last_swing_idx <= LEVEL_MAX_AGE_HTF_BARS
        ]
        support_levels = [
            L for L in support_levels
            if htf_n - L.last_swing_idx <= LEVEL_MAX_AGE_HTF_BARS
        ]

        if not resistance_levels and not support_levels:
            return None

        # Sort by recency (newest last_swing first).
        resistance_levels.sort(key=lambda L: L.last_swing_idx, reverse=True)
        support_levels.sort(key=lambda L: L.last_swing_idx, reverse=True)

        # 2. Try long flip (resistance broken upward, retested as support).
        sig = self._try_long_flip(
            ltf, htf, last_idx, resistance_levels,
            atr_htf_pips, atr_ltf_pips, context.symbol,
        )
        if sig is not None:
            return sig

        # 3. Try short flip.
        return self._try_short_flip(
            ltf, htf, last_idx, support_levels,
            atr_htf_pips, atr_ltf_pips, context.symbol,
        )

    # ----------------------------------------------------------------- internals

    def _try_long_flip(
        self,
        ltf: Sequence[Bar],
        htf: Sequence[Bar],
        last_idx: int,
        levels: List[_Level],
        atr_htf_pips: float,
        atr_ltf_pips: float,
        symbol: str,
    ) -> Optional[PatternSignal]:
        break_margin_pips = max(
            BREAK_MARGIN_PIPS_FLOOR,
            BREAK_MARGIN_ATR_MULT * atr_htf_pips,
        )
        break_margin_price = pips_to_price(break_margin_pips, symbol)
        reentry_block_price = pips_to_price(REENTRY_BLOCK_PIPS, symbol)
        retest_tol_pips = max(
            RETEST_TOL_PIPS_FLOOR,
            RETEST_TOL_ATR_MULT * atr_ltf_pips,
        )
        retest_tol_price = pips_to_price(retest_tol_pips, symbol)
        sl_buf_price = pips_to_price(sl_buffer_pips_for(symbol), symbol)

        cur = ltf[last_idx]

        for level in levels:
            # Retest condition first (cheap) — current bar must have wicked
            # to or below the level and closed back above.
            if cur.low > level.price + retest_tol_price:
                continue
            if cur.close <= level.price:
                continue

            # Break detection on HTF.
            break_idx = _find_break_above(
                htf, level.price, break_margin_price,
                reentry_block_price, level.last_swing_idx,
            )
            if break_idx is None:
                continue

            # Debounce: don't double-trade if a prior LTF bar already did
            # this retest in the lookback window.
            if _prior_retest_within_debounce_long(
                ltf, last_idx, level.price, retest_tol_price,
            ):
                continue

            # Rejection on the current LTF bar.
            prev = ltf[last_idx - 1] if last_idx >= 1 else None
            pin = is_bullish_pin(cur)
            engulf = prev is not None and is_bullish_engulf(prev, cur)
            if not (pin or engulf):
                continue

            # Trade construction.
            entry = cur.close
            sl = min(cur.low, level.price) - sl_buf_price
            risk_price = entry - sl
            risk_pips = price_to_pips(risk_price, symbol)
            if risk_pips < MIN_RISK_PIPS:
                continue
            tp1 = entry + TP1_R * risk_price
            tp2 = entry + TP2_R * risk_price
            if not (sl < entry < tp2):
                continue

            # Grade A: ≥3 touches AND engulfing rejection.
            is_grade_a = (level.touches >= 3 and engulf)
            grade = Grade.A if is_grade_a else Grade.B
            confidence = _GRADE_A_CONFIDENCE if is_grade_a else _GRADE_B_CONFIDENCE
            rejection_tag = "engulf" if engulf else "pin"
            break_age_h = len(htf) - 1 - break_idx

            return PatternSignal(
                pattern_name=self.name,
                symbol=symbol,
                direction=Direction.BUY,
                entry=entry, sl=sl, tp=tp2,
                confidence=confidence, grade=grade,
                confluences_met=(
                    "flip_long",
                    f"level_{level.touches}t",
                    f"break_{break_age_h}h",
                    f"rejection_{rejection_tag}",
                    f"tp1_{tp1:.5f}",
                ),
                bar_time_msc=cur.time_msc,
            )
        return None

    def _try_short_flip(
        self,
        ltf: Sequence[Bar],
        htf: Sequence[Bar],
        last_idx: int,
        levels: List[_Level],
        atr_htf_pips: float,
        atr_ltf_pips: float,
        symbol: str,
    ) -> Optional[PatternSignal]:
        break_margin_pips = max(
            BREAK_MARGIN_PIPS_FLOOR,
            BREAK_MARGIN_ATR_MULT * atr_htf_pips,
        )
        break_margin_price = pips_to_price(break_margin_pips, symbol)
        reentry_block_price = pips_to_price(REENTRY_BLOCK_PIPS, symbol)
        retest_tol_pips = max(
            RETEST_TOL_PIPS_FLOOR,
            RETEST_TOL_ATR_MULT * atr_ltf_pips,
        )
        retest_tol_price = pips_to_price(retest_tol_pips, symbol)
        sl_buf_price = pips_to_price(sl_buffer_pips_for(symbol), symbol)

        cur = ltf[last_idx]

        for level in levels:
            if cur.high < level.price - retest_tol_price:
                continue
            if cur.close >= level.price:
                continue

            break_idx = _find_break_below(
                htf, level.price, break_margin_price,
                reentry_block_price, level.last_swing_idx,
            )
            if break_idx is None:
                continue

            if _prior_retest_within_debounce_short(
                ltf, last_idx, level.price, retest_tol_price,
            ):
                continue

            prev = ltf[last_idx - 1] if last_idx >= 1 else None
            pin = is_bearish_pin(cur)
            engulf = prev is not None and is_bearish_engulf(prev, cur)
            if not (pin or engulf):
                continue

            entry = cur.close
            sl = max(cur.high, level.price) + sl_buf_price
            risk_price = sl - entry
            risk_pips = price_to_pips(risk_price, symbol)
            if risk_pips < MIN_RISK_PIPS:
                continue
            tp1 = entry - TP1_R * risk_price
            tp2 = entry - TP2_R * risk_price
            if not (tp2 < entry < sl):
                continue

            is_grade_a = (level.touches >= 3 and engulf)
            grade = Grade.A if is_grade_a else Grade.B
            confidence = _GRADE_A_CONFIDENCE if is_grade_a else _GRADE_B_CONFIDENCE
            rejection_tag = "engulf" if engulf else "pin"
            break_age_h = len(htf) - 1 - break_idx

            return PatternSignal(
                pattern_name=self.name,
                symbol=symbol,
                direction=Direction.SELL,
                entry=entry, sl=sl, tp=tp2,
                confidence=confidence, grade=grade,
                confluences_met=(
                    "flip_short",
                    f"level_{level.touches}t",
                    f"break_{break_age_h}h",
                    f"rejection_{rejection_tag}",
                    f"tp1_{tp1:.5f}",
                ),
                bar_time_msc=cur.time_msc,
            )
        return None
