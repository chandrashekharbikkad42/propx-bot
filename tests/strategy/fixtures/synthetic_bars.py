"""Synthetic bar generators for AsianSweepDetector tests.

These helpers build deterministic sequences of `Bar` objects that match the
V5 detector's expectations:

  - 1H bars aligned to the hour (time_msc = hour boundary UTC, ms)
  - Asian range  = prev day [19:30 UTC, 00:30 UTC next day)  → 5 bars (20..00)
  - LONDON sweep = current day 06..10 UTC
  - NY     sweep = current day 12..15 UTC
  - HTF bias     = EMA200 on closes <= prev day 23:00 UTC
    fewer than 200 closes → neutral (used as the default in most tests)

Everything is plain Python; no MT5 / pandas dependency at test time.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from config.asian_sweep_config import PAIR_CONFIG
from data.bar_aggregator import Bar


UTC = timezone.utc


def baseline_low(pair: str) -> float:
    """Realistic Asian-low baseline matched to a pair's price scale.

    The detector builds SL = trigger.low - sl_pts * pt. Indices carry a huge
    buffer (sl_pts=2000, pt=0.01 → 20 index points) so an FX-style ~1.10
    baseline drives SL below zero and trips
    PatternSignal.__post_init__ ("entry, sl, tp must all be positive").
    Anchoring each instrument class to its real price scale keeps the synthetic
    Asian levels and the derived SL/TP strictly positive. XAUUSD (100.0) and
    5-digit FX (1.10000) baselines are preserved exactly so existing
    assertions are unaffected; only index instruments get a realistic scale.
    """
    if pair == "XAUUSD":
        return 100.0
    cfg = PAIR_CONFIG.get(pair)
    if cfg is not None and cfg.get("category") == "Index":
        return 20_000.0
    return 1.10000


def hour_msc(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    """UTC datetime → ms."""
    return int(
        datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000
    )


def make_bar(
    *,
    symbol: str = "EURUSD",
    time_msc: int,
    open: float = 1.0,
    high: Optional[float] = None,
    low: Optional[float] = None,
    close: float = 1.0,
    volume: int = 1,
    spread_mean: float = 0.0,
) -> Bar:
    """Construct a Bar with sensible defaults.

    If high/low are not given they are derived from open/close so the bar
    is internally consistent (high = max, low = min).
    """
    if high is None:
        high = max(open, close)
    if low is None:
        low = min(open, close)
    return Bar(
        symbol=symbol,
        time_msc=time_msc,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
        spread_mean=spread_mean,
    )


def build_scenario(
    *,
    symbol: str,
    year: int,
    month: int,
    day: int,
    asian_high: float,
    asian_low: float,
    trigger_hour: int,
    trigger_high: float,
    trigger_low: float,
    trigger_close: float,
    trigger_open: Optional[float] = None,
    history_bars: int = 50,
    bias: str = "neutral",
    asian_high_at_hour: int = 22,
    asian_low_at_hour: int = 21,
) -> List[Bar]:
    """Build a minimal scenario for the detector.

    Layout (all timestamps UTC, aligned to the hour):
      - Seed history: bars BEFORE prev_day 20:00 UTC. Their price determines
        the EMA200; the Asian range and trigger bars are unaffected.
        For bias != "neutral" we emit >= 200 seed bars at `seed_price` so
        that EMA settles there. Then the cutoff bar at prev_day 23:00
        sits at `base` and pushes `last_close` past the bias threshold.
      - Asian window: prev_day 20:00, 21:00, 22:00, 23:00 + cur_day 00:00.
        All five bars are at `base` except the H/L hours, which are
        stamped with `asian_high` / `asian_low` respectively.
      - Filler bars from 01:00 cur_day up to `trigger_hour`-1, at `base`.
      - Trigger bar at cur_day `trigger_hour`:00 with the requested OHLC.

    `bias`:
      - "neutral"  → < 200 closes before the cutoff (default).
      - "bullish"  → seed at `base * 0.99`; cutoff close at `base` pushes
        last close above EMA * 1.001.
      - "bearish"  → seed at `base * 1.01`; cutoff close at `base` pushes
        last close below EMA * 0.999.
    """
    cur_date = datetime(year, month, day, 0, 0, tzinfo=UTC).date()
    prev_date = cur_date - timedelta(days=1)

    base = (asian_high + asian_low) / 2.0

    bars: List[Bar] = []

    # ----- seed bars before Asian window -----
    # They sit strictly before prev_day 20:00 UTC (which is when the Asian
    # window starts emitting bars). The Asian-window filter is
    # [prev_day 19:30, cur_day 00:30); a bar opened at 19:00 prev_day has
    # time_msc < 19:30 so it is excluded.
    if bias == "bullish":
        seed_price = base * 0.99
    elif bias == "bearish":
        seed_price = base * 1.01
    else:
        seed_price = base

    n_seed = 230 if bias in ("bullish", "bearish") else max(history_bars, 30)

    # Anchor: the seed bars END at prev_day 19:00 (the bar just before the
    # Asian window). We go backwards `n_seed` hours from there.
    seed_anchor = datetime(prev_date.year, prev_date.month, prev_date.day,
                           19, 0, tzinfo=UTC)
    for i in range(n_seed, 0, -1):
        dt = seed_anchor - timedelta(hours=i)
        bars.append(
            make_bar(
                symbol=symbol,
                time_msc=int(dt.timestamp() * 1000),
                open=seed_price, high=seed_price,
                low=seed_price, close=seed_price,
            )
        )
    # The anchor itself (prev_day 19:00) — still seed; excluded from Asian
    # range because 19:00 < 19:30.
    bars.append(
        make_bar(
            symbol=symbol,
            time_msc=int(seed_anchor.timestamp() * 1000),
            open=seed_price, high=seed_price,
            low=seed_price, close=seed_price,
        )
    )

    # ----- Asian window: 5 bars -----
    asian_hours = [
        (prev_date, 20),
        (prev_date, 21),
        (prev_date, 22),
        (prev_date, 23),
        (cur_date, 0),
    ]
    for d, h in asian_hours:
        dt = datetime(d.year, d.month, d.day, h, 0, tzinfo=UTC)
        hi = base
        lo = base
        cl_ = base
        if d == prev_date and h == asian_high_at_hour:
            hi = asian_high
            cl_ = (asian_high + base) / 2.0
        if d == prev_date and h == asian_low_at_hour:
            lo = asian_low
            cl_ = (asian_low + base) / 2.0
        bars.append(
            make_bar(
                symbol=symbol,
                time_msc=int(dt.timestamp() * 1000),
                open=base,
                high=max(hi, base, cl_),
                low=min(lo, base, cl_),
                close=cl_,
            )
        )

    # ----- filler bars from 01:00 cur_day up to (but not including) trigger_hour -----
    for h in range(1, trigger_hour):
        dt = datetime(cur_date.year, cur_date.month, cur_date.day, h, 0,
                      tzinfo=UTC)
        bars.append(
            make_bar(
                symbol=symbol,
                time_msc=int(dt.timestamp() * 1000),
                open=base, high=base, low=base, close=base,
            )
        )

    # ----- trigger bar -----
    tdt = datetime(cur_date.year, cur_date.month, cur_date.day,
                   trigger_hour, 0, tzinfo=UTC)
    op = trigger_open if trigger_open is not None else base
    bars.append(
        make_bar(
            symbol=symbol,
            time_msc=int(tdt.timestamp() * 1000),
            open=op,
            high=max(trigger_high, op, trigger_close),
            low=min(trigger_low, op, trigger_close),
            close=trigger_close,
        )
    )

    return bars


def long_sweep_bars(
    *,
    symbol: str,
    pt: float,
    asian_low: float = 1.10300,
    asian_high: float = 1.10500,
    trigger_hour: int = 8,
    wick_below_pts: float = 50.0,
    close_above_pts: float = 10.0,
    bias: str = "neutral",
    year: int = 2026, month: int = 4, day: int = 15,
) -> List[Bar]:
    """Bars producing a LONG sweep (wick under AL, close above AL)."""
    trigger_low = asian_low - wick_below_pts * pt
    trigger_close = asian_low + close_above_pts * pt
    trigger_high = max(trigger_close, asian_low + 1 * pt)
    return build_scenario(
        symbol=symbol,
        year=year, month=month, day=day,
        asian_high=asian_high,
        asian_low=asian_low,
        trigger_hour=trigger_hour,
        trigger_high=trigger_high,
        trigger_low=trigger_low,
        trigger_close=trigger_close,
        bias=bias,
    )


def short_sweep_bars(
    *,
    symbol: str,
    pt: float,
    asian_low: float = 1.10300,
    asian_high: float = 1.10500,
    trigger_hour: int = 8,
    wick_above_pts: float = 50.0,
    close_below_pts: float = 10.0,
    year: int = 2026, month: int = 4, day: int = 15,
) -> List[Bar]:
    """Bars producing a SHORT sweep (wick over AH, close below AH).

    Forces bearish bias since the detector requires it for shorts.
    """
    trigger_high = asian_high + wick_above_pts * pt
    trigger_close = asian_high - close_below_pts * pt
    trigger_low = min(trigger_close, asian_high - 1 * pt)
    return build_scenario(
        symbol=symbol,
        year=year, month=month, day=day,
        asian_high=asian_high,
        asian_low=asian_low,
        trigger_hour=trigger_hour,
        trigger_high=trigger_high,
        trigger_low=trigger_low,
        trigger_close=trigger_close,
        bias="bearish",
    )
