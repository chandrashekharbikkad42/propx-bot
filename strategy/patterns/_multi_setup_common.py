"""Shared primitives for the 4 propX Multi-Setup detectors.

Single source for every structural definition the spec references:
  - pip / OHLC math
  - swing-high/low detection (fractal, L_SWING window)
  - structure labelling (HH/HL/LH/LL → BULLISH/BEARISH/RANGE)
  - Wilder ATR(14)
  - impulsive-move detector (LTF and HTF flavours)
  - rejection candle classifier (pin / engulfing + volume gate)
  - S/R level finder (clustered touches over lookback window)

Detectors MUST import from here — never reimplement a swing or a pin-bar
test inside detector code. If a definition needs to evolve, change it ONCE
here so all four setups stay aligned.

Hinglish: ek hi jagah primitives — swings, structure, ATR, pins, S/R levels.
Detector me sirf orchestration likhna, math yahin hai.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from data.bar_aggregator import Bar
from config.multi_setup_config import (
    ATR_LEN,
    ENGULF_BODY_MIN_MULT, ENGULF_BODY_MIN_RANGE_FRAC,
    IMP_ATR_FACTOR_MAX, IMP_ATR_FACTOR_MIN,
    IMP_BAR_REVERSAL_MAX_FRAC, IMP_MIN_PIPS_BASE,
    IMP_VOL_LOOKBACK_BARS, IMP_VOL_MULT,
    L_SWING,
    LEVEL_CLUSTER_TOLERANCE_ATR_MULT, LEVEL_CLUSTER_TOLERANCE_FLOOR,
    LEVEL_LOOKBACK_BARS_HTF, LEVEL_MIN_GAP_BARS, LEVEL_MIN_TOUCHES,
    LEVEL_NEARBY_TOL_MULT,
    N_IMP, N_IMP_HTF,
    OB_IMP_ATR_MULT, OB_IMP_MIN_PIPS_FLOOR,
    PIN_BODY_MAX_FRAC, PIN_REQUIRE_BODY_COLOR,
    PIN_WICK_MIN_BODY_MULT, PIN_WICK_MIN_RANGE_FRAC,
    REJECT_VOL_LOOKBACK_BARS, REJECT_VOL_MIN_MULT,
    pip_size_for,
)


# ─────────────────────────────────────────────────────────────────────────────
# OHLC math
# ─────────────────────────────────────────────────────────────────────────────

def body(bar: Bar) -> float:
    return abs(bar.close - bar.open)


def range_(bar: Bar) -> float:
    return bar.high - bar.low


def upper_wick(bar: Bar) -> float:
    return bar.high - max(bar.open, bar.close)


def lower_wick(bar: Bar) -> float:
    return min(bar.open, bar.close) - bar.low


def is_bullish(bar: Bar) -> bool:
    return bar.close > bar.open


def is_bearish(bar: Bar) -> bool:
    return bar.close < bar.open


def price_to_pips(price_delta: float, symbol: str) -> float:
    """Convert an absolute price delta to pips for `symbol`."""
    return price_delta / pip_size_for(symbol)


def pips_to_price(pips: float, symbol: str) -> float:
    """Convert pips to absolute price delta for `symbol`."""
    return pips * pip_size_for(symbol)


# ─────────────────────────────────────────────────────────────────────────────
# Swing detection (spec §1.2) — fractal, L_SWING lookback on both sides
# ─────────────────────────────────────────────────────────────────────────────

class SwingKind(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


@dataclass(frozen=True)
class Swing:
    """One confirmed swing point. `index` is the bar index in the input list.

    A swing at `index = i` is confirmed only when `i + L_SWING < len(bars)` —
    i.e. we have enough bars on the right side for the fractal test. Detectors
    using `find_swings` will therefore never see an in-progress swing.
    """
    index: int
    time_msc: int
    price: float
    kind: SwingKind


def find_swings(
    bars: Sequence[Bar], l_swing: int = L_SWING
) -> List[Swing]:
    """Return all confirmed swings (highs + lows) in `bars`, time-ordered.

    Spec §1.2:
      - swing high at `i`: `high[i] > high[i±k]` for all k ∈ [1, l_swing]
      - swing low at  `i`: `low[i]  < low[i±k]`  for all k ∈ [1, l_swing]
      - in-progress swings (right edge insufficient) are NOT returned

    Edge handling:
      - Strict inequality (> / <) — equal highs/lows do NOT form a swing.
        Avoids double-counting flat tops / bottoms.
      - First `l_swing` bars and last `l_swing` bars cannot be swings.
    """
    if l_swing < 1:
        raise ValueError(f"l_swing must be >= 1, got {l_swing}")
    n = len(bars)
    out: List[Swing] = []
    for i in range(l_swing, n - l_swing):
        b = bars[i]
        # Check left side
        is_high = True
        is_low = True
        for k in range(1, l_swing + 1):
            if not (b.high > bars[i - k].high and b.high > bars[i + k].high):
                is_high = False
            if not (b.low < bars[i - k].low and b.low < bars[i + k].low):
                is_low = False
            if not is_high and not is_low:
                break
        if is_high:
            out.append(Swing(
                index=i, time_msc=b.time_msc, price=b.high, kind=SwingKind.HIGH,
            ))
        if is_low:
            out.append(Swing(
                index=i, time_msc=b.time_msc, price=b.low, kind=SwingKind.LOW,
            ))
    out.sort(key=lambda s: (s.index, 0 if s.kind == SwingKind.HIGH else 1))
    return out


def last_swing(
    swings: Sequence[Swing], kind: SwingKind
) -> Optional[Swing]:
    """Most recent confirmed swing of the given kind, or None."""
    for s in reversed(swings):
        if s.kind == kind:
            return s
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Structure / trend (spec §1.3) — HH/HL/LH/LL on swing sequence
# ─────────────────────────────────────────────────────────────────────────────

class Trend(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGE = "RANGE"


def classify_trend(swings: Sequence[Swing]) -> Trend:
    """Spec §1.3 trend state from the last two swing pairs.

    BULLISH ↔ last two highs are HH (last high > prior high) AND
              last two lows  are HL (last low  > prior low).
    BEARISH ↔ last two highs are LH AND last two lows are LL.
    Otherwise → RANGE.

    Returns RANGE if fewer than 2 highs and 2 lows exist.
    """
    highs = [s for s in swings if s.kind == SwingKind.HIGH]
    lows = [s for s in swings if s.kind == SwingKind.LOW]
    if len(highs) < 2 or len(lows) < 2:
        return Trend.RANGE
    hh = highs[-1].price > highs[-2].price
    lh = highs[-1].price < highs[-2].price
    hl = lows[-1].price > lows[-2].price
    ll = lows[-1].price < lows[-2].price
    if hh and hl:
        return Trend.BULLISH
    if lh and ll:
        return Trend.BEARISH
    return Trend.RANGE


# ─────────────────────────────────────────────────────────────────────────────
# ATR (spec §1.5) — Wilder ATR(14), closed bars only
# ─────────────────────────────────────────────────────────────────────────────

def true_range(prev_close: float, bar: Bar) -> float:
    return max(
        bar.high - bar.low,
        abs(bar.high - prev_close),
        abs(bar.low - prev_close),
    )


def compute_atr(bars: Sequence[Bar], length: int = ATR_LEN) -> Optional[float]:
    """Wilder ATR(length). Returns None if fewer than `length + 1` bars.

    Closed bars only — the caller must ensure the in-progress bar is excluded.
    """
    if len(bars) < length + 1:
        return None
    trs: List[float] = []
    for i in range(1, len(bars)):
        trs.append(true_range(bars[i - 1].close, bars[i]))
    # Seed with simple average of first `length` TR values.
    atr = sum(trs[:length]) / length
    # Wilder smoothing for the remainder.
    for tr in trs[length:]:
        atr = (atr * (length - 1) + tr) / length
    return atr


def atr_in_pips(atr_price: float, symbol: str) -> float:
    return atr_price / pip_size_for(symbol)


# ─────────────────────────────────────────────────────────────────────────────
# Impulsive move (spec §1.4)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ImpulseResult:
    is_impulsive: bool
    direction: int           # +1 bullish, -1 bearish, 0 none
    displacement_pips: float
    threshold_pips: float
    reason: str = ""         # populated when is_impulsive=False


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])


def detect_impulse_ltf(
    bars: Sequence[Bar],
    end_idx: int,
    symbol: str,
    atr_value_price: Optional[float],
    n_imp: int = N_IMP,
) -> ImpulseResult:
    """Spec §1.4 — check if bars[end_idx - n_imp + 1 .. end_idx] form an impulse.

    `atr_value_price` is the LTF ATR in price units; pass None for "no
    adaptive factor" (uses base threshold). Caller computes ATR over closed
    bars excluding the in-progress one.
    """
    if end_idx + 1 < n_imp:
        return ImpulseResult(False, 0, 0.0, 0.0, "insufficient bars")
    window = bars[end_idx - n_imp + 1: end_idx + 1]

    # Direction by close sequence.
    if all(is_bullish(b) for b in window):
        direction = +1
    elif all(is_bearish(b) for b in window):
        direction = -1
    else:
        return ImpulseResult(False, 0, 0.0, 0.0, "mixed bar colours")

    # Displacement = last close - first open (signed); we report absolute.
    displacement_price = abs(window[-1].close - window[0].open)
    displacement_pips = price_to_pips(displacement_price, symbol)

    # Adaptive threshold.
    base = IMP_MIN_PIPS_BASE
    if atr_value_price is None:
        threshold = base
    else:
        # Need a baseline median to derive factor — caller does that in
        # `adaptive_imp_threshold_pips` for correctness; here we fall back to
        # the base value when ATR is provided without a baseline.
        threshold = base

    if displacement_pips < threshold:
        return ImpulseResult(
            False, direction, displacement_pips, threshold,
            "displacement below threshold",
        )

    # No single bar reverses by > 30 % of its own range.
    for b in window:
        bar_range = range_(b)
        if bar_range <= 0:
            continue
        # Reversal = wick against the move direction.
        if direction == +1:
            counter = max(b.open - b.low, 0.0)  # downside wick from open
            # use the larger of (open-low) and (close-low) penalty? Spec says
            # "single bar reverses by > 30 % of that bar's range" — interpret
            # as: counter-trend body+wick within bar > 30% of range. We use
            # the lower-wick from min(open,close).
            counter = lower_wick(b)
        else:
            counter = upper_wick(b)
        if counter > IMP_BAR_REVERSAL_MAX_FRAC * bar_range:
            return ImpulseResult(
                False, direction, displacement_pips, threshold,
                "single-bar reversal exceeds 30% of range",
            )

    # Volume gate.
    if len(bars) >= IMP_VOL_LOOKBACK_BARS:
        recent_vols = [b.volume for b in bars[max(0, end_idx - IMP_VOL_LOOKBACK_BARS): end_idx + 1]]
        med = _median(recent_vols)
        if med > 0:
            run_vol_avg = sum(b.volume for b in window) / n_imp
            if run_vol_avg < IMP_VOL_MULT * med:
                return ImpulseResult(
                    False, direction, displacement_pips, threshold,
                    "volume below 1.2x median",
                )

    return ImpulseResult(True, direction, displacement_pips, threshold)


def adaptive_imp_threshold_pips(
    ltf_bars: Sequence[Bar],
    end_idx: int,
    symbol: str,
) -> float:
    """Spec §1.4 adaptive threshold:
       base * clamp(ATR14_now / median(ATR14 over last 100 bars), [0.5, 2.0])

    Returns IMP_MIN_PIPS_BASE if not enough history to compute the factor.
    """
    base = IMP_MIN_PIPS_BASE
    if end_idx < ATR_LEN + 100:
        return base
    # Current ATR ending at end_idx (closed bars only).
    atr_now = compute_atr(ltf_bars[: end_idx + 1], ATR_LEN)
    if atr_now is None:
        return base
    # Rolling ATR values over the last 100 bars ending at end_idx.
    atrs: List[float] = []
    for j in range(end_idx - 99, end_idx + 1):
        v = compute_atr(ltf_bars[: j + 1], ATR_LEN)
        if v is not None:
            atrs.append(v)
    if not atrs:
        return base
    baseline = _median(atrs)
    if baseline <= 0:
        return base
    factor = atr_now / baseline
    factor = max(IMP_ATR_FACTOR_MIN, min(IMP_ATR_FACTOR_MAX, factor))
    return base * factor


def detect_impulse_htf(
    htf_bars: Sequence[Bar],
    start_idx: int,
    symbol: str,
    atr_htf_pips: Optional[float],
    n_imp: int = N_IMP_HTF,
) -> ImpulseResult:
    """Spec §3.1 — HTF impulse starting at `start_idx` (inclusive), covering
    `n_imp` consecutive same-direction bullish/bearish closes.

    Threshold = max(OB_IMP_MIN_PIPS_FLOOR, OB_IMP_ATR_MULT × ATR_HTF_pips).
    """
    end_idx = start_idx + n_imp - 1
    if end_idx >= len(htf_bars):
        return ImpulseResult(False, 0, 0.0, 0.0, "insufficient HTF bars")
    window = htf_bars[start_idx: end_idx + 1]
    if all(is_bullish(b) for b in window):
        direction = +1
    elif all(is_bearish(b) for b in window):
        direction = -1
    else:
        return ImpulseResult(False, 0, 0.0, 0.0, "mixed bar colours")

    displacement_price = abs(window[-1].close - window[0].open)
    displacement_pips = price_to_pips(displacement_price, symbol)

    if atr_htf_pips is None:
        threshold = OB_IMP_MIN_PIPS_FLOOR
    else:
        threshold = max(OB_IMP_MIN_PIPS_FLOOR, OB_IMP_ATR_MULT * atr_htf_pips)

    if displacement_pips < threshold:
        return ImpulseResult(
            False, direction, displacement_pips, threshold,
            "HTF displacement below threshold",
        )
    return ImpulseResult(True, direction, displacement_pips, threshold)


# ─────────────────────────────────────────────────────────────────────────────
# Rejection candle (spec §1.8)
# ─────────────────────────────────────────────────────────────────────────────

def _volume_ok(
    bars: Sequence[Bar], idx: int,
    lookback: int = REJECT_VOL_LOOKBACK_BARS,
    mult: float = REJECT_VOL_MIN_MULT,
) -> bool:
    """Spec §1.8(c) — current tick volume ≥ mult × mean of last `lookback` bars.

    Returns True if there's insufficient history to compute the mean (we don't
    block detection on cold-start; the swing/level rules already gate that).
    """
    if idx <= 0:
        return True
    start = max(0, idx - lookback)
    window = bars[start: idx]
    if not window:
        return True
    mean_vol = sum(b.volume for b in window) / len(window)
    if mean_vol <= 0:
        return True
    return bars[idx].volume >= mult * mean_vol


def is_bullish_pin(bar: Bar) -> bool:
    """Spec §1.8(a) bullish pin: body ≤ 33% range, lower wick ≥ 2× body and
    ≥ 55% range. Pin color: bullish body unless PIN_REQUIRE_BODY_COLOR=False.
    """
    rng = range_(bar)
    if rng <= 0:
        return False
    b = body(bar)
    lw = lower_wick(bar)
    if b > PIN_BODY_MAX_FRAC * rng:
        return False
    if lw < PIN_WICK_MIN_BODY_MULT * b and b > 0:
        return False
    if lw < PIN_WICK_MIN_RANGE_FRAC * rng:
        return False
    if PIN_REQUIRE_BODY_COLOR and not is_bullish(bar) and b > 0:
        return False
    return True


def is_bearish_pin(bar: Bar) -> bool:
    rng = range_(bar)
    if rng <= 0:
        return False
    b = body(bar)
    uw = upper_wick(bar)
    if b > PIN_BODY_MAX_FRAC * rng:
        return False
    if uw < PIN_WICK_MIN_BODY_MULT * b and b > 0:
        return False
    if uw < PIN_WICK_MIN_RANGE_FRAC * rng:
        return False
    if PIN_REQUIRE_BODY_COLOR and not is_bearish(bar) and b > 0:
        return False
    return True


def is_bullish_engulf(prev: Bar, cur: Bar) -> bool:
    """Spec §1.8(b) bullish engulfing — prev bearish; cur opens at/below prev
    close and closes at/above prev open; cur body ≥ prev body; cur body
    ≥ 40 % of cur range."""
    if not is_bearish(prev):
        return False
    if not (cur.open <= prev.close and cur.close >= prev.open):
        return False
    cur_body = cur.close - cur.open
    if cur_body <= 0:
        return False
    prev_body = abs(prev.close - prev.open)
    if cur_body < ENGULF_BODY_MIN_MULT * prev_body:
        return False
    cur_range = range_(cur)
    if cur_range <= 0 or cur_body < ENGULF_BODY_MIN_RANGE_FRAC * cur_range:
        return False
    return True


def is_bearish_engulf(prev: Bar, cur: Bar) -> bool:
    if not is_bullish(prev):
        return False
    if not (cur.open >= prev.close and cur.close <= prev.open):
        return False
    cur_body = cur.open - cur.close
    if cur_body <= 0:
        return False
    prev_body = abs(prev.close - prev.open)
    if cur_body < ENGULF_BODY_MIN_MULT * prev_body:
        return False
    cur_range = range_(cur)
    if cur_range <= 0 or cur_body < ENGULF_BODY_MIN_RANGE_FRAC * cur_range:
        return False
    return True


def is_rejection_bullish(bars: Sequence[Bar], idx: int) -> bool:
    """Combined rejection check at `bars[idx]`, bullish direction.

    Passes if either pin or engulfing pattern matches AND volume ≥ 1.1×
    mean of last REJECT_VOL_LOOKBACK_BARS bars.
    """
    if idx < 0 or idx >= len(bars):
        return False
    cur = bars[idx]
    pin = is_bullish_pin(cur)
    engulf = idx >= 1 and is_bullish_engulf(bars[idx - 1], cur)
    if not (pin or engulf):
        return False
    return _volume_ok(bars, idx)


def is_rejection_bearish(bars: Sequence[Bar], idx: int) -> bool:
    if idx < 0 or idx >= len(bars):
        return False
    cur = bars[idx]
    pin = is_bearish_pin(cur)
    engulf = idx >= 1 and is_bearish_engulf(bars[idx - 1], cur)
    if not (pin or engulf):
        return False
    return _volume_ok(bars, idx)


# ─────────────────────────────────────────────────────────────────────────────
# S/R level finder (spec §5.1)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SRLevel:
    price: float
    touches: int
    is_support: bool
    is_resistance: bool
    last_touch_idx: int
    last_touch_time_msc: int


def find_sr_levels(
    htf_bars: Sequence[Bar],
    symbol: str,
    atr_htf_pips: Optional[float],
    lookback_bars: int = LEVEL_LOOKBACK_BARS_HTF,
    min_touches: int = LEVEL_MIN_TOUCHES,
    min_gap_bars: int = LEVEL_MIN_GAP_BARS,
) -> List[SRLevel]:
    """Spec §5.1 — find clustered horizontal S/R levels in the last
    `lookback_bars` of HTF data.

    Algorithm:
      1. Take all wicks (high and low) in the window.
      2. Greedy cluster: sort wicks by price; group prices within
         `cluster_tolerance_price` of an emerging cluster centre.
      3. For each cluster, count touches that are ≥ `min_gap_bars` apart
         (filter price-hugging).
      4. Keep clusters with ≥ `min_touches` qualifying touches.
      5. Label each cluster as support / resistance based on the direction
         of bar approach (low-touching with close above → support;
         high-touching with close below → resistance).

    Returns a list of SRLevel sorted by `last_touch_idx` descending (newest first).
    """
    n = len(htf_bars)
    if n == 0:
        return []
    start = max(0, n - lookback_bars)
    window = htf_bars[start:]
    # Cluster tolerance in price units.
    tol_pips_floor = LEVEL_CLUSTER_TOLERANCE_FLOOR
    if atr_htf_pips is None:
        tol_pips = tol_pips_floor
    else:
        tol_pips = max(tol_pips_floor, LEVEL_CLUSTER_TOLERANCE_ATR_MULT * atr_htf_pips)
    tol_price = pips_to_price(tol_pips, symbol)
    if tol_price <= 0:
        return []

    # Collect (price, bar_index_in_full_history, kind ∈ {"high","low"})
    touches: List[Tuple[float, int, str]] = []
    for j, b in enumerate(window):
        touches.append((b.high, start + j, "high"))
        touches.append((b.low, start + j, "low"))

    touches.sort(key=lambda t: t[0])

    # Greedy cluster on sorted prices.
    clusters: List[List[Tuple[float, int, str]]] = []
    for p, idx, kind in touches:
        if clusters and abs(p - clusters[-1][0][0]) <= tol_price * 2:
            # within 2× tol of cluster anchor (allows widening)
            clusters[-1].append((p, idx, kind))
        else:
            clusters.append([(p, idx, kind)])

    out: List[SRLevel] = []
    for cluster in clusters:
        if not cluster:
            continue
        # Cluster centre = median price of touches.
        prices = [t[0] for t in cluster]
        centre = _median(prices)
        # Sort cluster touches by bar index; collapse same-bar touches; enforce min_gap.
        cluster.sort(key=lambda t: t[1])
        kept: List[Tuple[float, int, str]] = []
        last_idx = -10 ** 9
        for t in cluster:
            if t[1] - last_idx >= min_gap_bars:
                kept.append(t)
                last_idx = t[1]
        if len(kept) < min_touches:
            continue
        # Support if ≥ 2 of kept touches were 'low' approaches with close above centre.
        # Resistance if ≥ 2 were 'high' approaches with close below centre.
        n_support = 0
        n_resistance = 0
        for p, idx, kind in kept:
            bar = htf_bars[idx]
            if kind == "low" and bar.close > centre:
                n_support += 1
            elif kind == "high" and bar.close < centre:
                n_resistance += 1
        is_support = n_support >= 2
        is_resistance = n_resistance >= 2
        if not (is_support or is_resistance):
            continue
        last_touch = kept[-1]
        out.append(SRLevel(
            price=centre,
            touches=len(kept),
            is_support=is_support,
            is_resistance=is_resistance,
            last_touch_idx=last_touch[1],
            last_touch_time_msc=htf_bars[last_touch[1]].time_msc,
        ))

    # Cleanliness filter (spec §5.2) — drop levels with another level within
    # ±tol×LEVEL_NEARBY_TOL_MULT pips.
    nearby_band = tol_price * LEVEL_NEARBY_TOL_MULT
    clean: List[SRLevel] = []
    for L in out:
        crowded = False
        for M in out:
            if M is L:
                continue
            if abs(L.price - M.price) <= nearby_band:
                crowded = True
                break
        if not crowded:
            clean.append(L)

    clean.sort(key=lambda L: L.last_touch_idx, reverse=True)
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# Utility — get index of HTF bar matching a given time
# ─────────────────────────────────────────────────────────────────────────────

def htf_index_at_or_before(htf_bars: Sequence[Bar], time_msc: int) -> int:
    """Return the largest HTF bar index whose `time_msc <= time_msc`, or -1."""
    lo, hi = 0, len(htf_bars) - 1
    ans = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if htf_bars[mid].time_msc <= time_msc:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans
