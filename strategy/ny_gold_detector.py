"""NY Gold Sweep — detector (NY_GOLD_SWEEP_SPEC.md v1.1 §2–§5, §10).

Stateful detector instantiated once per backtest run / live session.
Call `.detect(t, current_idx, v1, v5, v15)` per closed 1M bar inside the
NY session — backtest is responsible for session filtering (§1) and
all §7 compliance gates (news / cooldown / DD / cap / concurrency).

Detector responsibilities (this file):
  §2 find confirmed 15M swings → fresh-level set, freshness check, burn
  §3 5M zone proximity + dwell debounce (burn on stale)
  §4 1M sweep candidate + reversal trigger (engulf or pin) + concurrency lock
  §5 SL placement, TP price (Mode A / B / C), risk-floor reject
  §10 grade A/B classification

Returns at most one `NYGoldSignal` per call; the trigger time is the
close of the 1M bar at `current_idx` (== `t`). Entry is `R+1 open` —
the backtest fills it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import ny_gold_sweep_config as cfg
from strategy.ny_gold_data import BarFrame, utc_date_key


# ─── value objects ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Level:
    price: float
    swing_time: int           # open-time of the swing bar (ms)
    direction: str            # "low" (long target) | "high" (short target)
    confirmation_time: int    # close-time of the second confirming bar
    touches: int              # bars in lookback within TOUCH_TOLERANCE of price


@dataclass(frozen=True)
class NYGoldSignal:
    """Per §11. Mirrors the spec contract; backtest consumes this verbatim."""
    pattern_name: str          # "NY_GOLD_SWEEP"
    symbol: str                # "XAUUSD"
    direction: str             # "BUY" | "SELL"
    sl: float                  # SL price level (mid)
    tp: float                  # TP price level (mid)
    grade: str                 # "A" | "B"
    confidence: float
    bar_time_msc: int          # R's close timestamp (trigger)
    fill_time_msc: int         # R+1's open timestamp (entry)
    r_idx: int                 # R's index in v1 at trigger time
    s_idx: int                 # S's index in v1 at trigger time
    level_price: float
    level_swing_time: int      # for burned_levels keying
    sweep_depth_pips: float
    reversal_kind: str         # "engulf" | "pin"
    tp_mode: str               # active TP mode that produced `tp`
    confluences: tuple[str, ...]


# ─── helpers ────────────────────────────────────────────────────────────────
def _atr_5m_pips(v5: BarFrame, period: int = cfg.ATR_5M_PERIOD) -> float:
    """Wilder-ish ATR over last `period` 5M bars, returned in pips.

    Uses simple mean of true range (matches the spec which doesn't specify
    Wilder vs SMA; SMA is faster and per-bar stable). Returns 0.0 if not
    enough bars — caller falls back to the SWEEP_MAX_PENETRATION_PIPS floor.
    """
    n = len(v5)
    if n < period + 1:
        return 0.0
    h = v5.high[-(period + 1):]
    l = v5.low[-(period + 1):]
    c = v5.close[-(period + 1):]
    tr = np.maximum.reduce([
        h[1:] - l[1:],
        np.abs(h[1:] - c[:-1]),
        np.abs(l[1:] - c[:-1]),
    ])
    return float(tr.mean()) / cfg.PIP_SIZE


def _is_bullish_engulf(v1: BarFrame, i: int, min_body_price: float) -> bool:
    """§4.2 long: bullish engulfing of a prior bearish bar."""
    if i < 1:
        return False
    o, c = v1.open[i], v1.close[i]
    if c <= o:
        return False
    if (c - o) < min_body_price:
        return False
    po, pc = v1.open[i - 1], v1.close[i - 1]
    if pc >= po:   # prev must be bearish
        return False
    if c < po:     # body must engulf prev open
        return False
    if o > pc:     # current open must be <= prev close
        return False
    return True


def _is_bullish_pin(
    v1: BarFrame, i: int, wick_ratio: float, min_wick_price: float
) -> bool:
    """§4.2 long: bullish pin — long lower wick + close in upper third."""
    o, h, l, c = v1.open[i], v1.high[i], v1.low[i], v1.close[i]
    if c <= o:
        return False
    body = c - o
    lower_wick = o - l
    if body <= 0:
        return False
    if lower_wick < wick_ratio * body:
        return False
    if lower_wick < min_wick_price:
        return False
    rng = h - l
    if rng <= 0:
        return False
    upper_third = l + (rng * 2.0 / 3.0)
    return c >= upper_third


def _is_bearish_engulf(v1: BarFrame, i: int, min_body_price: float) -> bool:
    """§4.4 short: bearish engulfing of a prior bullish bar."""
    if i < 1:
        return False
    o, c = v1.open[i], v1.close[i]
    if c >= o:
        return False
    if (o - c) < min_body_price:
        return False
    po, pc = v1.open[i - 1], v1.close[i - 1]
    if pc <= po:   # prev must be bullish
        return False
    if c > po:     # current close must engulf prev open
        return False
    if o < pc:     # current open must be >= prev close
        return False
    return True


def _is_bearish_pin(
    v1: BarFrame, i: int, wick_ratio: float, min_wick_price: float
) -> bool:
    """§4.4 short: bearish pin — long upper wick + close in lower third."""
    o, h, l, c = v1.open[i], v1.high[i], v1.low[i], v1.close[i]
    if c >= o:
        return False
    body = o - c
    upper_wick = h - o
    if body <= 0:
        return False
    if upper_wick < wick_ratio * body:
        return False
    if upper_wick < min_wick_price:
        return False
    rng = h - l
    if rng <= 0:
        return False
    lower_third = l + (rng * 1.0 / 3.0)
    return c <= lower_third


# ─── detector class ─────────────────────────────────────────────────────────
class NYGoldSweepDetector:
    """Stateful per-session detector. Reset state across UTC days."""

    def __init__(self) -> None:
        self._current_day: int = -1
        # Burned levels persist for the rest of the UTC day. Key = (price, swing_time).
        self._burned: set[tuple[float, int]] = set()
        # First-time-zone-active per level (for §3.3 dwell debounce). Key as above.
        self._zone_first_seen: dict[tuple[float, int], int] = {}

    # ─── public API ──────────────────────────────────────────────────
    def detect(
        self,
        t: int,
        current_idx: int,
        v1: BarFrame,
        v5: BarFrame,
        v15: BarFrame,
    ) -> Optional[NYGoldSignal]:
        # Per-day state reset
        day = utc_date_key(t)
        if day != self._current_day:
            self._current_day = day
            self._burned.clear()
            self._zone_first_seen.clear()

        # §2 — fresh levels in 15M lookback
        levels = self._find_fresh_levels(t, v1, v15)
        if not levels:
            return None

        # §3 — zone proximity + dwell
        zoned = self._filter_zone(t, v5, levels)
        if not zoned:
            return None

        # Sort candidates by nearest-to-price first (§4 spec doesn't define
        # ordering; nearest is the natural pick for a sweep that's
        # actually-happening NOW).
        cur_close = float(v1.close[current_idx])
        zoned.sort(key=lambda L: abs(cur_close - L.price))

        # §4 — try sweep+reversal trigger
        atr5_pips = _atr_5m_pips(v5)
        for L in zoned:
            sig = self._try_trigger(t, current_idx, v1, v5, v15, L, atr5_pips)
            if sig is not None:
                # §4.3 / §2.3 burn on use
                self._burned.add((L.price, L.swing_time))
                return sig
        return None

    # ─── §2 levels ───────────────────────────────────────────────────
    def _find_fresh_levels(
        self, t: int, v1: BarFrame, v15: BarFrame
    ) -> list[Level]:
        n = len(v15)
        Lw = cfg.L_SWING_15M
        if n < 2 * Lw + 1:
            return []

        lookback_ms = cfg.LEVEL_LOOKBACK_HOURS * 3600 * 1000
        start_t = t - lookback_ms

        # Candidate confirmed swings: idx in [Lw, n-1-Lw].
        # Restrict search to last ~80 bars for speed (covers 20h of 15M).
        search_start = max(Lw, n - 80)
        search_end = n - Lw  # exclusive

        highs = v15.high
        lows = v15.low
        times = v15.time_msc

        levels: list[Level] = []
        for i in range(search_start, search_end):
            ti = int(times[i])
            if ti < start_t:
                continue
            # Swing low: lows[i] strictly less than all 2 bars each side
            is_low = (
                lows[i] < lows[i - 2] and lows[i] < lows[i - 1]
                and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]
            )
            if is_low:
                conf_time = int(times[i + Lw]) + v15.bar_ms
                price = float(lows[i])
                key = (price, ti)
                if key in self._burned:
                    continue
                touches = self._count_touches(v15, "low", price, start_t, t)
                if self._is_level_fresh(v1, price, "low", conf_time, t):
                    levels.append(Level(
                        price=price, swing_time=ti, direction="low",
                        confirmation_time=conf_time, touches=touches,
                    ))
                else:
                    # Stale (already penetrated). Burn so we don't re-test it.
                    self._burned.add(key)

            is_high = (
                highs[i] > highs[i - 2] and highs[i] > highs[i - 1]
                and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]
            )
            if is_high:
                conf_time = int(times[i + Lw]) + v15.bar_ms
                price = float(highs[i])
                key = (price, ti)
                if key in self._burned:
                    continue
                touches = self._count_touches(v15, "high", price, start_t, t)
                if self._is_level_fresh(v1, price, "high", conf_time, t):
                    levels.append(Level(
                        price=price, swing_time=ti, direction="high",
                        confirmation_time=conf_time, touches=touches,
                    ))
                else:
                    self._burned.add(key)

        return levels

    @staticmethod
    def _count_touches(
        v15: BarFrame, direction: str, price: float, start_t: int, end_t: int
    ) -> int:
        """§10 grade A: count 15M bars whose low/high is within tolerance."""
        tol = cfg.TOUCH_TOLERANCE_PIPS * cfg.PIP_SIZE
        arr = v15.low if direction == "low" else v15.high
        times = v15.time_msc
        # Vectorised filter
        mask = (times >= start_t) & (times <= end_t) & (np.abs(arr - price) <= tol)
        return int(mask.sum())

    @staticmethod
    def _is_level_fresh(
        v1: BarFrame, price: float, direction: str,
        confirmation_time: int, t: int,
    ) -> bool:
        """§2.3 freshness. Vectorised scan of 1M bars in [conf_time, t]."""
        invalidate = cfg.LEVEL_INVALIDATE_PIPS * cfg.PIP_SIZE
        times = v1.time_msc
        # Bars whose open_time >= confirmation_time AND open_time <= t
        # (close_time of any bar in our slice is <= t by construction)
        mask = (times >= confirmation_time) & (times <= t)
        if not mask.any():
            return True
        if direction == "low":
            return not np.any(v1.low[mask] < (price - invalidate))
        else:
            return not np.any(v1.high[mask] > (price + invalidate))

    # ─── §3 zone ─────────────────────────────────────────────────────
    def _filter_zone(
        self, t: int, v5: BarFrame, levels: list[Level]
    ) -> list[Level]:
        if len(v5) == 0:
            return []
        last_high = float(v5.high[-1])
        last_low = float(v5.low[-1])
        proximity = cfg.ZONE_PROXIMITY_PIPS * cfg.PIP_SIZE
        dwell_ms = cfg.ZONE_MAX_DWELL_MIN * 60 * 1000

        out: list[Level] = []
        for L in levels:
            key = (L.price, L.swing_time)
            if L.direction == "low":
                in_zone = last_low <= (L.price + proximity)
            else:
                in_zone = last_high >= (L.price - proximity)

            if not in_zone:
                # Reset dwell tracker if we exited the zone
                self._zone_first_seen.pop(key, None)
                continue

            # Track first-zone-active time
            first = self._zone_first_seen.get(key)
            if first is None:
                self._zone_first_seen[key] = t
                first = t

            if (t - first) > dwell_ms:
                # §3.3 stale: burn and skip
                self._burned.add(key)
                self._zone_first_seen.pop(key, None)
                continue

            out.append(L)
        return out

    # ─── §4 sweep + reversal ─────────────────────────────────────────
    def _try_trigger(
        self,
        t: int,
        cur_idx: int,
        v1: BarFrame,
        v5: BarFrame,
        v15: BarFrame,
        L: Level,
        atr5_pips: float,
    ) -> Optional[NYGoldSignal]:
        pip = cfg.PIP_SIZE
        pen_max_pips = max(cfg.SWEEP_MAX_PENETRATION_PIPS, cfg.SWEEP_MAX_ATR_MULT * atr5_pips)
        pen_max = pen_max_pips * pip
        pen_min = cfg.SWEEP_MIN_PENETRATION_PIPS * pip
        reject_tol = cfg.SWEEP_REJECT_TOLERANCE_PIPS * pip
        engulf_body = cfg.ENGULF_MIN_BODY_PIPS * pip
        pin_wick_min = cfg.PIN_MIN_WICK_PIPS * pip

        # Iterate wait_offset = 0..REVERSAL_MAX_WAIT_BARS (0 = same-bar short-circuit)
        for wait in range(0, cfg.REVERSAL_MAX_WAIT_BARS + 1):
            s_idx = cur_idx - wait
            if s_idx < 0:
                continue
            # S must not predate the level's confirmation (sweeps before
            # the swing was confirmed are meaningless)
            if int(v1.time_msc[s_idx]) < L.confirmation_time:
                continue

            sig = self._eval_pair(
                s_idx, cur_idx, v1, L, pen_max, pen_min, reject_tol,
                engulf_body, pin_wick_min,
            )
            if sig is not None:
                # Augment signal with TP / SL / grade
                return self._finalise_signal(
                    t, cur_idx, sig, v1, v15, L, atr5_pips, pen_max_pips,
                )
        return None

    def _eval_pair(
        self, s_idx: int, r_idx: int, v1: BarFrame, L: Level,
        pen_max: float, pen_min: float, reject_tol: float,
        engulf_body: float, pin_wick_min: float,
    ) -> Optional[dict]:
        """Evaluate one (S, R) pair. Returns a partial result dict on match."""
        if L.direction == "low":
            S_low = float(v1.low[s_idx])
            S_close = float(v1.close[s_idx])
            # §4.1.1 penetration: L - pen_max <= S.low < L - pen_min
            if not ((L.price - pen_max) <= S_low < (L.price - pen_min)):
                return None
            # §4.1.2 rejection: S.close > L - reject_tol
            if S_close <= (L.price - reject_tol):
                return None

            # Intermediate wait bars (S+1 .. R-1): no deeper wick
            # (§4.2 — wait bar wicking deeper than pen_max abort + burn)
            for k in range(s_idx + 1, r_idx):
                if float(v1.low[k]) < (L.price - pen_max):
                    # Burn the level
                    self._burned.add((L.price, L.swing_time))
                    return None

            # R checks
            R_low = float(v1.low[r_idx])
            R_close = float(v1.close[r_idx])
            if R_low < (L.price - pen_max):
                self._burned.add((L.price, L.swing_time))
                return None
            if R_close <= L.price:
                return None

            # Engulf or pin (and §4.1.3 short-circuit when r_idx == s_idx)
            is_engulf = _is_bullish_engulf(v1, r_idx, engulf_body)
            is_pin = _is_bullish_pin(
                v1, r_idx, cfg.PIN_WICK_BODY_RATIO, pin_wick_min,
            )
            if not (is_engulf or is_pin):
                return None

            return {
                "direction": "BUY",
                "s_idx": s_idx, "r_idx": r_idx,
                "sweep_low_or_high": S_low,
                "sweep_depth_pips": (L.price - S_low) / cfg.PIP_SIZE,
                "reversal_kind": "engulf" if is_engulf else "pin",
            }

        else:  # direction == "high"
            S_high = float(v1.high[s_idx])
            S_close = float(v1.close[s_idx])
            if not ((L.price + pen_min) < S_high <= (L.price + pen_max)):
                return None
            if S_close >= (L.price + reject_tol):
                return None

            for k in range(s_idx + 1, r_idx):
                if float(v1.high[k]) > (L.price + pen_max):
                    self._burned.add((L.price, L.swing_time))
                    return None

            R_high = float(v1.high[r_idx])
            R_close = float(v1.close[r_idx])
            if R_high > (L.price + pen_max):
                self._burned.add((L.price, L.swing_time))
                return None
            if R_close >= L.price:
                return None

            is_engulf = _is_bearish_engulf(v1, r_idx, engulf_body)
            is_pin = _is_bearish_pin(
                v1, r_idx, cfg.PIN_WICK_BODY_RATIO, pin_wick_min,
            )
            if not (is_engulf or is_pin):
                return None

            return {
                "direction": "SELL",
                "s_idx": s_idx, "r_idx": r_idx,
                "sweep_low_or_high": S_high,
                "sweep_depth_pips": (S_high - L.price) / cfg.PIP_SIZE,
                "reversal_kind": "engulf" if is_engulf else "pin",
            }

    # ─── §5 finalise (SL, TP, grade) ────────────────────────────────
    def _finalise_signal(
        self, t: int, cur_idx: int, partial: dict, v1: BarFrame,
        v15: BarFrame, L: Level, atr5_pips: float, pen_max_pips: float,
    ) -> Optional[NYGoldSignal]:
        pip = cfg.PIP_SIZE
        sl_buf = cfg.SL_BUFFER_PIPS * pip
        direction = partial["direction"]
        s_idx = partial["s_idx"]

        # §5.4 SL beyond sweep wick
        if direction == "BUY":
            sl = float(v1.low[s_idx]) - sl_buf
        else:
            sl = float(v1.high[s_idx]) + sl_buf

        # Provisional entry mid_open for risk-floor check.
        # The R+1 bar is not visible at decision time — we approximate entry
        # using R.close (current bar close) for the risk-floor check.
        # Real backtest fill uses R+1.open + ask/bid adjust; if the next-bar
        # entry pushes risk below MIN_RISK_PIPS, the backtest will reject.
        ref_entry = float(v1.close[cur_idx])
        if direction == "BUY":
            risk_pips_provisional = (ref_entry - sl) / pip
        else:
            risk_pips_provisional = (sl - ref_entry) / pip
        if risk_pips_provisional < cfg.MIN_RISK_PIPS:
            return None
        risk_price = abs(ref_entry - sl)

        # §5.5 TP per active mode
        tp_a = self._tp_mode_a(ref_entry, sl, direction)
        tp_b = self._tp_mode_b(ref_entry, direction, v15, t)
        tp_mode_used = cfg.TP_MODE
        if tp_mode_used == "A":
            tp = tp_a
        elif tp_mode_used == "B":
            tp = tp_b if tp_b is not None else tp_a
        else:  # C hybrid
            tp = self._tp_mode_c(ref_entry, sl, direction, tp_b)

        # §10 grade
        sweep_depth = partial["sweep_depth_pips"]
        is_grade_a = (
            L.touches >= cfg.GRADE_A_MIN_TOUCHES
            and partial["reversal_kind"] == "engulf"
            and cfg.GRADE_A_PENETRATION_LO <= sweep_depth <= cfg.GRADE_A_PENETRATION_HI
        )
        grade = "A" if is_grade_a else "B"
        confidence = 0.85 if is_grade_a else 0.70

        # Sanity: TP must be on the correct side of entry
        if direction == "BUY":
            if tp <= ref_entry:
                return None
        else:
            if tp >= ref_entry:
                return None

        bar_ms_1m = v1.bar_ms
        fill_time = int(v1.time_msc[cur_idx]) + bar_ms_1m  # = t + 60s = R+1.open_time

        return NYGoldSignal(
            pattern_name="NY_GOLD_SWEEP",
            symbol=cfg.SYMBOL,
            direction=direction,
            sl=sl,
            tp=tp,
            grade=grade,
            confidence=confidence,
            bar_time_msc=t,
            fill_time_msc=fill_time,
            r_idx=cur_idx,
            s_idx=s_idx,
            level_price=L.price,
            level_swing_time=L.swing_time,
            sweep_depth_pips=sweep_depth,
            reversal_kind=partial["reversal_kind"],
            tp_mode=tp_mode_used,
            confluences=(
                f"ny_sweep_{'long' if direction == 'BUY' else 'short'}",
                f"sweep_depth_{sweep_depth:.2f}p",
                f"rev_{partial['reversal_kind']}",
                f"level_{L.touches}touches",
                f"tp_mode_{tp_mode_used}",
            ),
        )

    @staticmethod
    def _tp_mode_a(entry: float, sl: float, direction: str) -> float:
        risk = abs(entry - sl)
        if direction == "BUY":
            return entry + cfg.TP_RR * risk
        return entry - cfg.TP_RR * risk

    @staticmethod
    def _tp_mode_b(
        entry: float, direction: str, v15: BarFrame, t: int,
    ) -> Optional[float]:
        """Opposing-level TP: nearest fresh opposite swing within reach."""
        pip = cfg.PIP_SIZE
        max_dist = cfg.OPPOSING_MAX_DISTANCE_PIPS * pip
        buf = cfg.OPPOSING_BUFFER_PIPS * pip
        Lw = cfg.L_SWING_15M
        n = len(v15)
        if n < 2 * Lw + 1:
            return None

        lookback_ms = cfg.LEVEL_LOOKBACK_HOURS * 3600 * 1000
        start_t = t - lookback_ms
        search_start = max(Lw, n - 80)
        search_end = n - Lw

        best: Optional[float] = None
        for i in range(search_start, search_end):
            ti = int(v15.time_msc[i])
            if ti < start_t:
                continue
            if direction == "BUY":
                # Need a swing high above entry, within max_dist
                is_swing = (
                    v15.high[i] > v15.high[i - 2] and v15.high[i] > v15.high[i - 1]
                    and v15.high[i] > v15.high[i + 1] and v15.high[i] > v15.high[i + 2]
                )
                if not is_swing:
                    continue
                price = float(v15.high[i])
                if price <= entry:
                    continue
                dist = price - entry
                if dist > max_dist:
                    continue
                cand = price - buf
                if cand <= entry:
                    continue
                if best is None or cand < best:
                    best = cand
            else:
                is_swing = (
                    v15.low[i] < v15.low[i - 2] and v15.low[i] < v15.low[i - 1]
                    and v15.low[i] < v15.low[i + 1] and v15.low[i] < v15.low[i + 2]
                )
                if not is_swing:
                    continue
                price = float(v15.low[i])
                if price >= entry:
                    continue
                dist = entry - price
                if dist > max_dist:
                    continue
                cand = price + buf
                if cand >= entry:
                    continue
                if best is None or cand > best:
                    best = cand
        return best

    @staticmethod
    def _tp_mode_c(
        entry: float, sl: float, direction: str, tp_b: Optional[float]
    ) -> float:
        """Hybrid: prefer Mode B if RR >= MIN_RR_FOR_OPPOSING, capped at TP_RR_MAX."""
        risk = abs(entry - sl)
        tp_a = (entry + cfg.TP_RR * risk) if direction == "BUY" else (entry - cfg.TP_RR * risk)
        if tp_b is None:
            return tp_a
        if direction == "BUY":
            rr_b = (tp_b - entry) / risk
            if rr_b >= cfg.MIN_RR_FOR_OPPOSING:
                cap = entry + cfg.TP_RR_MAX * risk
                return min(tp_b, cap)
            return tp_a
        else:
            rr_b = (entry - tp_b) / risk
            if rr_b >= cfg.MIN_RR_FOR_OPPOSING:
                cap = entry - cfg.TP_RR_MAX * risk
                return max(tp_b, cap)
            return tp_a
