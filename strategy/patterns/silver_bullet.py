"""ICT Silver Bullet detector (docs/SILVER_BULLET_SPEC.md v1.0).

Pure price-action + time detector. Three ET windows
(03–04 LO, 10–11 NY AM, 14–15 NY PM) — DST-aware via zoneinfo. Emits a
PatternSignal when:
  1. Current 15M bar opens inside one of the 3 ET windows.
  2. A liquidity sweep of a recent low/high happened within the last
     SWEEP_LOOKBACK_BARS, then was rejected (close back inside).
  3. A 3-candle FVG formed after the sweep, large enough to trade.
  4. Price has retraced back into the FVG on the trigger bar.

PatternSignal mapping:
  pattern_name      = "SILVER_BULLET"
  tp                = TP2 (1.5R / 2.5R ladder)
  confluences_met:
    - "sb_LO" / "sb_AM" / "sb_PM"
    - "long_sweep" / "short_sweep"
    - "fvg_<size_pips>p"
    - "sweep_ref_htf" / "sweep_ref_ltf"
    - "tp1_<price>"

Hinglish: SB ek time-based setup hai. Pehle teen windows me se kisi ek ka
intezaar. Phir liquidity sweep hua, FVG bana, price wapas FVG me aaya —
limit-style entry FVG ke top pe (long) ya bottom pe (short).
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover — Python 3.9+ has zoneinfo
    from backports.zoneinfo import ZoneInfo  # type: ignore

from data.bar_aggregator import Bar
from strategy.patterns.base import (
    Direction, Grade, MarketContext, PatternDetector, PatternSignal,
)
from strategy.patterns._multi_setup_common import (
    SwingKind, compute_atr, find_swings, last_swing,
)
from config.silver_bullet_config import (
    ATR_LEN,
    FVG_GRADE_A_PIPS, FVG_VALIDITY_BARS,
    MIN_RISK_PIPS,
    SWEEP_LOOKBACK_BARS, SWEEP_MAX_PENETRATION_ATR_MULT,
    SWEEP_PENETRATION_ATR_MULT, SWEEP_PENETRATION_PIPS_FLOOR,
    SWEEP_REF_BARS,
    TP1_R, TP2_R,
    WINDOWS,
    fvg_min_pips_for, pip_size_for, sl_buffer_pips_for,
)


# Local pip helpers — use silver_bullet_config.pip_size_for (which includes
# CADCHF + XAGUSD) instead of _multi_setup_common's helpers (which reach into
# multi_setup_config and KeyError on the 2 extra pairs).
def pips_to_price(pips: float, symbol: str) -> float:
    return pips * pip_size_for(symbol)


def price_to_pips(price_delta: float, symbol: str) -> float:
    return price_delta / pip_size_for(symbol)


def atr_in_pips(atr_price: float, symbol: str) -> float:
    return atr_price / pip_size_for(symbol)


_ET = ZoneInfo("America/New_York")
_GRADE_A_CONFIDENCE: float = 0.85
_GRADE_B_CONFIDENCE: float = 0.70


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — window membership + FVG dataclass
# ─────────────────────────────────────────────────────────────────────────────

def _et_hour(time_msc: int) -> int:
    dt = datetime.fromtimestamp(time_msc / 1000.0, tz=timezone.utc).astimezone(_ET)
    return dt.hour


def window_id_for(time_msc: int) -> Optional[str]:
    """Return the SB window id ('LO'/'AM'/'PM') for a bar open time, or None."""
    h = _et_hour(time_msc)
    for wid, start, end in WINDOWS:
        if start <= h < end:
            return wid
    return None


@dataclass(frozen=True)
class _FVG:
    """3-candle imbalance. (low, high) bounds the gap."""
    a_idx: int          # bar A (left side)
    c_idx: int          # bar C (right side)
    low: float          # A.high (long FVG) / C.high (short FVG)
    high: float         # C.low  (long FVG) / A.low  (short FVG)
    size_pips: float
    is_bullish: bool    # True = upward gap (long setup)


@dataclass(frozen=True)
class _Sweep:
    """Confirmed liquidity sweep + reclaim."""
    sweep_idx: int
    reclaim_idx: int
    swept_price: float
    sweep_extreme: float    # S.low (long) / S.high (short)
    ref_kind: str           # "htf" / "ltf"


# ─────────────────────────────────────────────────────────────────────────────
# Sweep search
# ─────────────────────────────────────────────────────────────────────────────

def _find_long_sweep(
    bars: Sequence[Bar],
    last_idx: int,
    htf_swing_low: Optional[float],
    atr_ltf_pips: float,
    symbol: str,
) -> Optional[_Sweep]:
    """Find the most recent rejected sweep of a low within SWEEP_LOOKBACK_BARS.

    Reference price (the "level" being swept):
      - HTF (1H) swing low if available (preferred); else
      - lowest low of the SWEEP_REF_BARS 15M bars *before* the sweep candidate.

    The sweep is REJECTED if some bar between the sweep and last_idx (inclusive)
    closes strictly back above the reference price.
    """
    pen_pips = max(SWEEP_PENETRATION_PIPS_FLOOR, SWEEP_PENETRATION_ATR_MULT * atr_ltf_pips)
    pen_price = pips_to_price(pen_pips, symbol)
    max_pen_price = pips_to_price(SWEEP_MAX_PENETRATION_ATR_MULT * atr_ltf_pips, symbol)
    if pen_price <= 0 or max_pen_price <= 0:
        return None

    lo = max(0, last_idx - SWEEP_LOOKBACK_BARS + 1)
    # Walk from most-recent backwards — return earliest match for stability.
    best: Optional[_Sweep] = None
    for i in range(last_idx, lo - 1, -1):
        # Build reference price candidates.
        ref_candidates: list[Tuple[float, str]] = []
        if htf_swing_low is not None:
            ref_candidates.append((htf_swing_low, "htf"))
        # LTF lookback low — lowest low of [i - SWEEP_REF_BARS .. i - 1]
        ltf_lo_start = max(0, i - SWEEP_REF_BARS)
        if ltf_lo_start < i:
            ltf_ref = min(b.low for b in bars[ltf_lo_start:i])
            ref_candidates.append((ltf_ref, "ltf"))

        for ref_price, ref_kind in ref_candidates:
            pen_actual = ref_price - bars[i].low
            if not (pen_price < pen_actual <= max_pen_price):
                continue
            # Reclaim — any later bar closes back above ref_price.
            reclaim_idx = -1
            for j in range(i, last_idx + 1):
                if bars[j].close > ref_price:
                    reclaim_idx = j
                    break
            if reclaim_idx < 0:
                continue
            best = _Sweep(
                sweep_idx=i, reclaim_idx=reclaim_idx,
                swept_price=ref_price, sweep_extreme=bars[i].low,
                ref_kind=ref_kind,
            )
            break
        if best is not None:
            break
    return best


def _find_short_sweep(
    bars: Sequence[Bar],
    last_idx: int,
    htf_swing_high: Optional[float],
    atr_ltf_pips: float,
    symbol: str,
) -> Optional[_Sweep]:
    pen_pips = max(SWEEP_PENETRATION_PIPS_FLOOR, SWEEP_PENETRATION_ATR_MULT * atr_ltf_pips)
    pen_price = pips_to_price(pen_pips, symbol)
    max_pen_price = pips_to_price(SWEEP_MAX_PENETRATION_ATR_MULT * atr_ltf_pips, symbol)
    if pen_price <= 0 or max_pen_price <= 0:
        return None

    lo = max(0, last_idx - SWEEP_LOOKBACK_BARS + 1)
    best: Optional[_Sweep] = None
    for i in range(last_idx, lo - 1, -1):
        ref_candidates: list[Tuple[float, str]] = []
        if htf_swing_high is not None:
            ref_candidates.append((htf_swing_high, "htf"))
        ltf_lo_start = max(0, i - SWEEP_REF_BARS)
        if ltf_lo_start < i:
            ltf_ref = max(b.high for b in bars[ltf_lo_start:i])
            ref_candidates.append((ltf_ref, "ltf"))

        for ref_price, ref_kind in ref_candidates:
            pen_actual = bars[i].high - ref_price
            if not (pen_price < pen_actual <= max_pen_price):
                continue
            reclaim_idx = -1
            for j in range(i, last_idx + 1):
                if bars[j].close < ref_price:
                    reclaim_idx = j
                    break
            if reclaim_idx < 0:
                continue
            best = _Sweep(
                sweep_idx=i, reclaim_idx=reclaim_idx,
                swept_price=ref_price, sweep_extreme=bars[i].high,
                ref_kind=ref_kind,
            )
            break
        if best is not None:
            break
    return best


# ─────────────────────────────────────────────────────────────────────────────
# FVG search — 3-candle imbalance, after a given index
# ─────────────────────────────────────────────────────────────────────────────

def _find_bullish_fvg_after(
    bars: Sequence[Bar],
    after_idx: int,
    last_idx: int,
    symbol: str,
) -> Optional[_FVG]:
    """Most recent bullish FVG (A.high < C.low) with A.index > after_idx and
    C.index <= last_idx and C.index >= last_idx - FVG_VALIDITY_BARS.

    Returns None if none qualify. Picks the LATEST qualifying FVG (most recent).
    """
    min_pips = fvg_min_pips_for(symbol)
    pip = pips_to_price(1.0, symbol)
    if pip <= 0:
        return None

    earliest_c = max(after_idx + 3, last_idx - FVG_VALIDITY_BARS)
    # Walk C from latest backward to find a fresh FVG.
    for c_idx in range(last_idx, earliest_c - 1, -1):
        a_idx = c_idx - 2
        if a_idx <= after_idx:
            break
        bar_a = bars[a_idx]
        bar_b = bars[a_idx + 1]
        bar_c = bars[c_idx]
        gap = bar_c.low - bar_a.high
        if gap <= 0:
            continue
        size_pips = gap / pip
        if size_pips < min_pips:
            continue
        # bar B must be bullish or doji (close >= open).
        if bar_b.close < bar_b.open:
            continue
        return _FVG(
            a_idx=a_idx, c_idx=c_idx,
            low=bar_a.high, high=bar_c.low,
            size_pips=size_pips, is_bullish=True,
        )
    return None


def _find_bearish_fvg_after(
    bars: Sequence[Bar],
    after_idx: int,
    last_idx: int,
    symbol: str,
) -> Optional[_FVG]:
    """Mirror of bullish: A.low > C.high. Bar B must be bearish or doji."""
    min_pips = fvg_min_pips_for(symbol)
    pip = pips_to_price(1.0, symbol)
    if pip <= 0:
        return None

    earliest_c = max(after_idx + 3, last_idx - FVG_VALIDITY_BARS)
    for c_idx in range(last_idx, earliest_c - 1, -1):
        a_idx = c_idx - 2
        if a_idx <= after_idx:
            break
        bar_a = bars[a_idx]
        bar_b = bars[a_idx + 1]
        bar_c = bars[c_idx]
        gap = bar_a.low - bar_c.high
        if gap <= 0:
            continue
        size_pips = gap / pip
        if size_pips < min_pips:
            continue
        if bar_b.close > bar_b.open:
            continue
        return _FVG(
            a_idx=a_idx, c_idx=c_idx,
            low=bar_c.high, high=bar_a.low,
            size_pips=size_pips, is_bullish=False,
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class SilverBulletDetector(PatternDetector):
    """ICT Silver Bullet. One signal per call max. MTF (1H + 15M)."""

    name: str = "SILVER_BULLET"
    timeframe: str = "15M"
    # Need ATR window + lookback room for sweep + FVG.
    min_bars_required: int = ATR_LEN + SWEEP_LOOKBACK_BARS + 3

    def detect(
        self, bars: Sequence[Bar], context: MarketContext
    ) -> Optional[PatternSignal]:
        ltf = bars
        if len(ltf) < self.min_bars_required:
            return None

        # 1. Window check on the most recent bar.
        last_idx = len(ltf) - 1
        wid = window_id_for(ltf[last_idx].time_msc)
        if wid is None:
            return None

        # 2. ATR on LTF.
        atr_price = compute_atr(ltf, ATR_LEN)
        if atr_price is None or atr_price <= 0:
            return None
        atr_pips = atr_in_pips(atr_price, context.symbol)

        # 3. HTF swing extremes (for preferred sweep reference).
        htf = context.htf_bars or ()
        htf_swing_low: Optional[float] = None
        htf_swing_high: Optional[float] = None
        if len(htf) >= 7:
            swings = find_swings(htf)
            sl_obj = last_swing(swings, SwingKind.LOW)
            sh_obj = last_swing(swings, SwingKind.HIGH)
            if sl_obj is not None:
                htf_swing_low = sl_obj.price
            if sh_obj is not None:
                htf_swing_high = sh_obj.price

        # 4. Try long first, then short.
        sig = self._try_long(ltf, last_idx, htf_swing_low, atr_pips, context.symbol, wid)
        if sig is not None:
            return sig
        return self._try_short(ltf, last_idx, htf_swing_high, atr_pips, context.symbol, wid)

    # ----------------------------------------------------------------- internals

    def _try_long(
        self,
        ltf: Sequence[Bar],
        last_idx: int,
        htf_swing_low: Optional[float],
        atr_pips: float,
        symbol: str,
        wid: str,
    ) -> Optional[PatternSignal]:
        sweep = _find_long_sweep(ltf, last_idx, htf_swing_low, atr_pips, symbol)
        if sweep is None:
            return None
        fvg = _find_bullish_fvg_after(ltf, sweep.reclaim_idx, last_idx, symbol)
        if fvg is None:
            return None

        # Entry trigger — current bar has retraced down into the FVG.
        cur = ltf[last_idx]
        if cur.low > fvg.high:
            return None

        # Trade construction. Entry = top of FVG (limit-conceptual; backtest
        # fills market with slippage; same convention as multi_setup BoS/OB).
        entry = fvg.high
        sl = min(sweep.sweep_extreme, fvg.low) - pips_to_price(
            sl_buffer_pips_for(symbol), symbol
        )
        risk_price = entry - sl
        risk_pips = price_to_pips(risk_price, symbol)
        if risk_pips < MIN_RISK_PIPS:
            return None
        tp1 = entry + TP1_R * risk_price
        tp2 = entry + TP2_R * risk_price
        if not (sl < entry < tp2):
            return None

        # Grade — A if HTF-referenced sweep AND FVG size ≥ FVG_GRADE_A_PIPS.
        is_grade_a = (sweep.ref_kind == "htf" and fvg.size_pips >= FVG_GRADE_A_PIPS)
        grade = Grade.A if is_grade_a else Grade.B
        confidence = _GRADE_A_CONFIDENCE if is_grade_a else _GRADE_B_CONFIDENCE

        return PatternSignal(
            pattern_name=self.name,
            symbol=symbol,
            direction=Direction.BUY,
            entry=entry, sl=sl, tp=tp2,
            confidence=confidence, grade=grade,
            confluences_met=(
                f"sb_{wid}",
                "long_sweep",
                f"fvg_{fvg.size_pips:.1f}p",
                f"sweep_ref_{sweep.ref_kind}",
                f"tp1_{tp1:.5f}",
            ),
            bar_time_msc=cur.time_msc,
        )

    def _try_short(
        self,
        ltf: Sequence[Bar],
        last_idx: int,
        htf_swing_high: Optional[float],
        atr_pips: float,
        symbol: str,
        wid: str,
    ) -> Optional[PatternSignal]:
        sweep = _find_short_sweep(ltf, last_idx, htf_swing_high, atr_pips, symbol)
        if sweep is None:
            return None
        fvg = _find_bearish_fvg_after(ltf, sweep.reclaim_idx, last_idx, symbol)
        if fvg is None:
            return None

        cur = ltf[last_idx]
        if cur.high < fvg.low:
            return None

        entry = fvg.low
        sl = max(sweep.sweep_extreme, fvg.high) + pips_to_price(
            sl_buffer_pips_for(symbol), symbol
        )
        risk_price = sl - entry
        risk_pips = price_to_pips(risk_price, symbol)
        if risk_pips < MIN_RISK_PIPS:
            return None
        tp1 = entry - TP1_R * risk_price
        tp2 = entry - TP2_R * risk_price
        if not (tp2 < entry < sl):
            return None

        is_grade_a = (sweep.ref_kind == "htf" and fvg.size_pips >= FVG_GRADE_A_PIPS)
        grade = Grade.A if is_grade_a else Grade.B
        confidence = _GRADE_A_CONFIDENCE if is_grade_a else _GRADE_B_CONFIDENCE

        return PatternSignal(
            pattern_name=self.name,
            symbol=symbol,
            direction=Direction.SELL,
            entry=entry, sl=sl, tp=tp2,
            confidence=confidence, grade=grade,
            confluences_met=(
                f"sb_{wid}",
                "short_sweep",
                f"fvg_{fvg.size_pips:.1f}p",
                f"sweep_ref_{sweep.ref_kind}",
                f"tp1_{tp1:.5f}",
            ),
            bar_time_msc=cur.time_msc,
        )
