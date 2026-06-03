"""Asian Range London Sweep V5 — pattern detector.

Direct port of `scan_signals` from `multi_pair_backtest.py` (verified 1-yr
backtest: PF 2.27, 239 trades). Decision logic and constants are pinned to
that backtest; any deviation is a bug.

Strategy (V5 rules — read this before changing anything):

  1. Mark Asian range from previous day 19:30 UTC → current day 00:30 UTC
     (= 01:00 → 06:00 IST). Need ≥ 2 bars to be valid.
  2. Compute HTF bias on H1 close EMA200 frozen at trading-day start
     (prev day 23:00 UTC):
        close > ema * 1.001  → bullish
        close < ema * 0.999  → bearish
        else                  → neutral
     If we have < 200 bars of bias history → neutral fallback.
  3. Two sweep windows:
        LONDON  06:00–10:30 UTC  (bars h=6..10)   → LONG + SHORT
        NY      12:00–15:30 UTC  (bars h=12..15)  → LONG only (V5)
     SHORT is LONDON-ONLY by design — V5 disabled NY shorts.
  4. LONG  trigger : bias ∈ {bullish, neutral} AND bar.low  < AL AND bar.close > AL
     SHORT trigger : bias == bearish           AND bar.high > AH AND bar.close < AH
  5. Entry = swept level ± broker spread (LONG: AL+spread, SHORT: AH−spread).
     SL    = wick ± sl_pts buffer       (LONG: low−sl_pts, SHORT: high+sl_pts).
     TP1   = entry ± 1.0R               (partial 50%, then SL→BE).
     TP2   = entry ± 2.5R               (the PatternSignal.tp slot).
  6. Range filter — reject if Asian range < min_range_pts or > max_range_pts
     (junk-day / event-day guard, per PAIR_CONFIG).
  7. Minimum risk guard — if |entry−SL| < 3 × point, reject (degenerate).

Grade mapping (Scanner drops Grade.C silently):
    quality 9–10 → Grade.A
    quality 4–8  → Grade.B
    confidence    = quality / 10.0          # tie-breaker after grade

PatternSignal mapping:
    pattern_name      = "ASIAN_SWEEP"
    direction         = BUY | SELL
    entry, sl         = computed above
    tp                = TP2 (signal contract has one tp slot)
    confluences_met   = ("asian_sweep_low" | "asian_sweep_high",
                         session,            # "LONDON" or "NY"
                         f"bias_{bias}",
                         f"q{quality}",
                         f"tp1_{tp1:.5f}")   # TP1 encoded for the exit module

Hinglish: Asian range nikalo, London/NY bar pe sweep + close-back dekho,
bias check karo (SHORT sirf London me bearish bias pe), entry/SL/TP1/TP2
calc karo, PatternSignal return. Exit module TP1 confluence se uthata hai.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence, Tuple

from data.bar_aggregator import Bar
from strategy.patterns.base import (
    Direction,
    Grade,
    MarketContext,
    PatternDetector,
    PatternSignal,
)
from config.asian_sweep_config import (
    ASIAN_START_UTC_H, ASIAN_START_UTC_M,
    ASIAN_END_UTC_H, ASIAN_END_UTC_M,
    LONDON_SWEEP_UTC_H_START, LONDON_SWEEP_UTC_H_END,
    NY_SWEEP_UTC_H_START, NY_SWEEP_UTC_H_END,
    PAIR_CONFIG,
    RR_TP1, RR_TP2,
    SKIP_MONDAY,
    quality_for,
)


UTC = timezone.utc

# Bias thresholds — frozen at backtest values (multi_pair_backtest.get_bias).
_BIAS_BULL_MULT: float = 1.001
_BIAS_BEAR_MULT: float = 0.999
_BIAS_EMA_SPAN: int = 200
_BIAS_MIN_BARS: int = 200      # below this → neutral fallback

# Minimum risk distance — wick + sl_pts buffer occasionally collapses below
# a few points on quiet bars. Backtest guards with risk < pt * 3.
_MIN_RISK_PT_MULT: int = 3

# Grade cutoff — quality >= this is Grade.A, otherwise Grade.B. Grade.C
# never emitted (would be dropped by the scanner anyway).
_GRADE_A_QUALITY_CUTOFF: int = 9


class AsianSweepDetector(PatternDetector):
    """V5 detector — emits one signal at most per call."""

    name: str = "ASIAN_SWEEP"
    # 5 Asian bars + headroom; bias falls back to neutral when bars<200.
    # Live script seeds 250+ bars so EMA200 normally has its full window.
    min_bars_required: int = 30
    timeframe: str = "1H"

    # ------------------------------------------------------------------ detect

    def detect(
        self, bars: Sequence[Bar], context: MarketContext
    ) -> Optional[PatternSignal]:
        if len(bars) < self.min_bars_required:
            return None

        current = bars[-1]
        cur_dt = datetime.fromtimestamp(current.time_msc / 1000.0, tz=UTC)

        # SKIP_MONDAY: weekday() == 0
        if SKIP_MONDAY and cur_dt.weekday() == 0:
            return None

        # ---- session window (V5 SHORT = LONDON only) ----
        h = cur_dt.hour
        if LONDON_SWEEP_UTC_H_START <= h <= LONDON_SWEEP_UTC_H_END:
            session = "LONDON"
            allow_short = True
        elif NY_SWEEP_UTC_H_START <= h <= NY_SWEEP_UTC_H_END:
            session = "NY"
            allow_short = False
        else:
            return None

        # ---- pair config ----
        symbol = context.symbol
        cfg = PAIR_CONFIG.get(symbol)
        if cfg is None:
            return None
        pt = float(cfg["point"])         # type: ignore[arg-type]
        sl_pts = float(cfg["sl_pts"])    # type: ignore[arg-type]
        spread_pts = float(cfg["spread_pts"])  # type: ignore[arg-type]
        min_range_pts = float(cfg["min_range_pts"])  # type: ignore[arg-type]
        max_range_pts = float(cfg["max_range_pts"])  # type: ignore[arg-type]
        quality = quality_for(symbol)

        # ---- Asian range (prev 19:30 → today 00:30 UTC) ----
        ah, al = _compute_asian_range(bars, cur_dt)
        if ah is None or al is None or ah <= al:
            return None

        rng_pts = round((ah - al) / pt)
        if rng_pts < min_range_pts or rng_pts > max_range_pts:
            return None

        # ---- HTF bias (EMA200 frozen at prev day 23:00 close) ----
        bias = _compute_bias(bars, cur_dt)

        # ---- SHORT (LONDON only, bearish bias, sweep high + close back) ----
        if (
            allow_short
            and bias == "bearish"
            and current.high > ah
            and current.close < ah
        ):
            entry = ah - spread_pts * pt
            sl = current.high + sl_pts * pt
            risk = abs(sl - entry)
            if risk < pt * _MIN_RISK_PT_MULT:
                return None
            tp1 = entry - risk * RR_TP1
            tp2 = entry - risk * RR_TP2
            # Honour PatternSignal invariant: SELL needs tp < entry < sl.
            if not (tp2 < entry < sl):
                return None
            return _build_signal(
                symbol=symbol,
                direction=Direction.SELL,
                entry=entry, sl=sl, tp2=tp2, tp1=tp1,
                quality=quality, session=session, bias=bias,
                bar_time_msc=current.time_msc,
                sweep_tag="asian_sweep_high",
            )

        # ---- LONG (LONDON + NY, bullish/neutral, sweep low + close back) ----
        if (
            bias in ("bullish", "neutral")
            and current.low < al
            and current.close > al
        ):
            entry = al + spread_pts * pt
            sl = current.low - sl_pts * pt
            risk = abs(entry - sl)
            if risk < pt * _MIN_RISK_PT_MULT:
                return None
            tp1 = entry + risk * RR_TP1
            tp2 = entry + risk * RR_TP2
            if not (sl < entry < tp2):
                return None
            return _build_signal(
                symbol=symbol,
                direction=Direction.BUY,
                entry=entry, sl=sl, tp2=tp2, tp1=tp1,
                quality=quality, session=session, bias=bias,
                bar_time_msc=current.time_msc,
                sweep_tag="asian_sweep_low",
            )

        return None


# ---------------------------------------------------------------------------
# Module-private helpers — pure, testable independently of the detector class.
# ---------------------------------------------------------------------------

def _compute_asian_range(
    bars: Sequence[Bar], cur_dt: datetime
) -> Tuple[Optional[float], Optional[float]]:
    """Return (asian_high, asian_low) over [prev_day 19:30, today 00:30) UTC.

    Mirrors `multi_pair_backtest.get_asian_range` exactly:
      - inclusive on the lower bound, exclusive on the upper bound;
      - operates on bar OPEN time;
      - requires ≥ 2 bars in the window, else returns (None, None).
    """
    cur_date = cur_dt.date()
    prev_date = cur_date - timedelta(days=1)
    start = datetime(
        prev_date.year, prev_date.month, prev_date.day,
        ASIAN_START_UTC_H, ASIAN_START_UTC_M, tzinfo=UTC,
    )
    end = datetime(
        cur_date.year, cur_date.month, cur_date.day,
        ASIAN_END_UTC_H, ASIAN_END_UTC_M, tzinfo=UTC,
    )
    start_msc = int(start.timestamp() * 1000)
    end_msc = int(end.timestamp() * 1000)

    highs: list[float] = []
    lows: list[float] = []
    for bar in bars:
        if start_msc <= bar.time_msc < end_msc:
            highs.append(bar.high)
            lows.append(bar.low)
    if len(highs) < 2:
        return None, None
    return max(highs), min(lows)


def _compute_bias(bars: Sequence[Bar], cur_dt: datetime) -> str:
    """Compute HTF bias from H1 closes up to (and including) prev day 23:00.

    Mirrors `multi_pair_backtest.get_bias`:
      ema200 = closes.ewm(span=200).mean()  at the cutoff
      cl     = last close at the cutoff
      cl > ema * 1.001 → bullish
      cl < ema * 0.999 → bearish
      else (or < 200 bars available) → neutral
    """
    cur_date = cur_dt.date()
    midnight = datetime(
        cur_date.year, cur_date.month, cur_date.day, 0, 0, tzinfo=UTC,
    )
    cutoff = midnight - timedelta(hours=1)
    cutoff_msc = int(cutoff.timestamp() * 1000)

    closes: list[float] = [b.close for b in bars if b.time_msc <= cutoff_msc]
    if len(closes) < _BIAS_MIN_BARS:
        return "neutral"

    ema = _ema(closes, _BIAS_EMA_SPAN)
    last = closes[-1]
    if last > ema * _BIAS_BULL_MULT:
        return "bullish"
    if last < ema * _BIAS_BEAR_MULT:
        return "bearish"
    return "neutral"


def _ema(values: Sequence[float], span: int) -> float:
    """Standard pandas-style EMA (`adjust=False`) seeded from the first value.

    Equivalent to `pd.Series(values).ewm(span=span, adjust=False).mean().iloc[-1]`.
    """
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1)
    ema = values[0]
    for v in values[1:]:
        ema = alpha * v + (1.0 - alpha) * ema
    return ema


def _build_signal(
    *,
    symbol: str,
    direction: Direction,
    entry: float,
    sl: float,
    tp2: float,
    tp1: float,
    quality: int,
    session: str,
    bias: str,
    bar_time_msc: int,
    sweep_tag: str,
) -> PatternSignal:
    """Construct the PatternSignal. TP2 fills the `tp` slot; TP1 rides in
    `confluences_met` for the exit module to pick up.
    """
    grade = Grade.A if quality >= _GRADE_A_QUALITY_CUTOFF else Grade.B
    confidence = max(0.0, min(1.0, quality / 10.0))
    return PatternSignal(
        pattern_name="ASIAN_SWEEP",
        symbol=symbol,
        direction=direction,
        entry=entry,
        sl=sl,
        tp=tp2,
        confidence=confidence,
        grade=grade,
        confluences_met=(
            sweep_tag,
            session,
            f"bias_{bias}",
            f"q{quality}",
            f"tp1_{tp1:.5f}",
        ),
        bar_time_msc=bar_time_msc,
    )
