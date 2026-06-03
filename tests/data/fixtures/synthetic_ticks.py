"""Synthetic tick generators for data-pipeline tests.

Pure / deterministic helpers (no I/O). The functions return either a
plain `list[Tick]` (consumed by BarAggregator / MultiPairFeed) or a
structured numpy array shaped like MT5's `copy_ticks_from` output
(consumed by TickCollector → connector mock).

Conventions used across the suite:
  - Bid/ask spread defaults to 0.0002 (≈ 2 pips on a 5-digit major).
  - Mid price = (bid + ask) / 2 — matches BarAggregator's internal mid.
  - `time_msc` is UTC epoch milliseconds; tests build it from explicit
    `datetime(..., tzinfo=timezone.utc)` calls to keep DST traps away.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from data.tick_collector import Tick


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def utc_ms(year: int, month: int, day: int,
           hour: int = 0, minute: int = 0, sec: int = 0, ms: int = 0) -> int:
    """Build a UTC ms timestamp from civil date parts."""
    dt = datetime(year, month, day, hour, minute, sec, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000) + ms


def add_ms(base_msc: int, *, hours: int = 0, minutes: int = 0,
           seconds: int = 0, ms: int = 0) -> int:
    return base_msc + (((hours * 60) + minutes) * 60 + seconds) * 1000 + ms


# ---------------------------------------------------------------------------
# Single tick builder
# ---------------------------------------------------------------------------

def make_tick(time_msc: int, bid: float,
              ask: Optional[float] = None,
              last: Optional[float] = None,
              volume: int = 1, volume_real: float = 1.0,
              flags: int = 0, spread: float = 0.0002) -> Tick:
    """Build a single Tick with sensible defaults for tests."""
    ask_v = ask if ask is not None else bid + spread
    last_v = last if last is not None else bid
    return Tick(
        time_msc=time_msc,
        bid=bid,
        ask=ask_v,
        last=last_v,
        volume=volume,
        volume_real=volume_real,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Stream generators (return list[Tick])
# ---------------------------------------------------------------------------

def random_walk_ticks(n: int, start_msc: int, *,
                      start_price: float = 1.10,
                      step_ms: int = 1000,
                      sigma: float = 0.0001,
                      spread: float = 0.0002,
                      seed: int = 42) -> List[Tick]:
    """Geometric-ish random walk. Deterministic per seed."""
    rng = np.random.default_rng(seed)
    moves = rng.normal(loc=0.0, scale=sigma, size=n)
    price = start_price
    out: List[Tick] = []
    for i, m in enumerate(moves):
        price = max(1e-6, price + m)
        out.append(make_tick(start_msc + i * step_ms, price, spread=spread))
    return out


def trend_ticks(n: int, start_msc: int, *,
                start_price: float = 1.10, slope: float = 1e-5,
                step_ms: int = 1000,
                spread: float = 0.0002) -> List[Tick]:
    """Linear trend. Useful when assertions need a monotone series."""
    return [
        make_tick(start_msc + i * step_ms, start_price + i * slope, spread=spread)
        for i in range(n)
    ]


def sideways_ticks(n: int, start_msc: int, *,
                   center: float = 1.10, amplitude: float = 0.0005,
                   period: int = 10, step_ms: int = 1000,
                   spread: float = 0.0002) -> List[Tick]:
    """Sinusoidal oscillation around `center`."""
    out: List[Tick] = []
    for i in range(n):
        price = center + amplitude * np.sin(2 * np.pi * i / max(1, period))
        out.append(make_tick(start_msc + i * step_ms, float(price), spread=spread))
    return out


def hour_filling_ticks(open_msc_hour: int, *,
                       n_per_hour: int = 10,
                       n_hours: int = 1,
                       start_price: float = 1.10,
                       spread: float = 0.0002) -> List[Tick]:
    """Evenly-spaced ticks inside `n_hours` consecutive H1 bars.

    Useful when a test wants K closed bars: feed n_hours+1 worth of ticks
    and the last tick crosses into the (n_hours+1)-th window, closing the
    n_hours-th bar.
    """
    out: List[Tick] = []
    one_hour_ms = 60 * 60 * 1000
    interval = max(1, one_hour_ms // max(1, n_per_hour))
    for h in range(n_hours):
        for k in range(n_per_hour):
            time_msc = open_msc_hour + h * one_hour_ms + k * interval
            out.append(make_tick(time_msc, start_price + h * 0.001 + k * 1e-5,
                                 spread=spread))
    return out


def duplicate_ts_ticks(time_msc: int, count: int = 5,
                       start_price: float = 1.10,
                       spread: float = 0.0002) -> List[Tick]:
    """Burst of ticks sharing the same time_msc (used to test dedup paths)."""
    return [
        make_tick(time_msc, start_price + i * 1e-5, spread=spread)
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# MT5-shaped numpy arrays (for TickCollector via connector mock)
# ---------------------------------------------------------------------------

# Mirrors MT5 copy_ticks_from() structured dtype. We use float64 for price
# fields and uint64 for volume — same as a real terminal returns.
MT5_TICK_DTYPE = np.dtype([
    ("time", "i8"),
    ("bid", "f8"),
    ("ask", "f8"),
    ("last", "f8"),
    ("volume", "u8"),
    ("time_msc", "i8"),
    ("flags", "u4"),
    ("volume_real", "f8"),
])


# Mirrors MT5 copy_rates_from_pos() structured dtype.
MT5_RATES_DTYPE = np.dtype([
    ("time", "i8"),
    ("open", "f8"),
    ("high", "f8"),
    ("low", "f8"),
    ("close", "f8"),
    ("tick_volume", "i8"),
    ("spread", "i4"),
    ("real_volume", "i8"),
])


# Some MT5 builds omit 'spread' — keep a dtype without it for tests of the
# capture-utils fallback branch.
MT5_RATES_DTYPE_NO_SPREAD = np.dtype([
    ("time", "i8"),
    ("open", "f8"),
    ("high", "f8"),
    ("low", "f8"),
    ("close", "f8"),
    ("tick_volume", "i8"),
    ("real_volume", "i8"),
])


def mt5_tick_array(rows: Sequence[Tuple]) -> np.ndarray:
    """Build a MT5-shaped tick ndarray.

    Each row: (time_s, bid, ask, last, volume, time_msc, flags, volume_real).
    """
    arr = np.zeros(len(rows), dtype=MT5_TICK_DTYPE)
    for i, r in enumerate(rows):
        arr[i] = r
    return arr


def ticks_to_mt5_array(ticks: Iterable[Tick]) -> np.ndarray:
    """Convert a `list[Tick]` to the MT5 ndarray shape."""
    ticks = list(ticks)
    arr = np.zeros(len(ticks), dtype=MT5_TICK_DTYPE)
    for i, t in enumerate(ticks):
        arr[i] = (
            t.time_msc // 1000, t.bid, t.ask, t.last,
            t.volume, t.time_msc, t.flags, t.volume_real,
        )
    return arr


def mt5_rates_array(rows: Sequence[Tuple], with_spread: bool = True) -> np.ndarray:
    """Build a MT5-shaped rates ndarray.

    Row schema with_spread=True:
      (time_s, open, high, low, close, tick_vol, spread, real_vol)
    Without spread:
      (time_s, open, high, low, close, tick_vol, real_vol)
    """
    dtype = MT5_RATES_DTYPE if with_spread else MT5_RATES_DTYPE_NO_SPREAD
    arr = np.zeros(len(rows), dtype=dtype)
    for i, r in enumerate(rows):
        arr[i] = r
    return arr


def consecutive_h1_rates(start_hour_utc: datetime, count: int, *,
                         start_price: float = 1.10,
                         step: float = 0.0001,
                         tick_volume: int = 100,
                         spread: int = 2) -> np.ndarray:
    """Convenience: `count` consecutive H1 bars starting at `start_hour_utc`."""
    rows = []
    for i in range(count):
        bar_open = start_hour_utc + timedelta(hours=i)
        epoch = int(bar_open.timestamp())
        open_p = start_price + i * step
        rows.append((
            epoch,
            open_p,
            open_p + 0.0005,
            open_p - 0.0005,
            open_p + step,
            tick_volume + i,
            spread,
            0,
        ))
    return mt5_rates_array(rows)


# ---------------------------------------------------------------------------
# Symbol universes (used for parametrization in multi-pair tests)
# ---------------------------------------------------------------------------

# 8 pairs requested by the spec. Spread baseline (in price units) per pair —
# JPY pairs use 3-digit pricing, so spread is 0.01-ish vs 0.0002 for majors.
EIGHT_PAIRS: Tuple[str, ...] = (
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "XAUUSD",
)


def base_price_for(symbol: str) -> float:
    """A realistic-ish starting mid for each pair (rough 2026 levels)."""
    return {
        "EURUSD": 1.10,
        "GBPUSD": 1.25,
        "USDJPY": 150.0,
        "USDCHF": 0.90,
        "AUDUSD": 0.66,
        "NZDUSD": 0.60,
        "USDCAD": 1.35,
        "XAUUSD": 2300.0,
    }.get(symbol, 1.0)


def spread_for(symbol: str) -> float:
    return {
        "USDJPY": 0.02,
        "XAUUSD": 0.20,
    }.get(symbol, 0.0002)
