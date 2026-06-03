"""Tick → 1H OHLCV bar aggregator + parquet bar reader.

Two paths, one module:
  - LIVE  : BarAggregator.on_tick(tick) → optional Bar (emitted when an hour boundary crosses)
  - BACKTEST: read_bars_parquet(symbol, timeframe) → DataFrame of bars

The aggregator is per-symbol. For multi-symbol live use, see
`data.multi_pair_feed.MultiPairFeed`, which composes one aggregator per pair.

Bar timing convention (CRITICAL — used everywhere downstream):
  - `time_msc` = bar OPEN timestamp (UTC), aligned to the hour boundary.
  - Bar covers the half-open interval [time_msc, time_msc + timeframe).
  - A tick at exactly `time_msc + timeframe` belongs to the NEXT bar.
  - First tick after construction opens the bar; we do NOT back-fill from
    midnight — caller decides how much history to seed.

Hinglish note: 1H bar open time = ghante ki seedhi shuruwat (00:00, 01:00 ...).
Hour boundary cross = naya bar emit, purana bar close + return ho jaata hai.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import settings
from data.tick_collector import Tick


# Parquet schema for stored 1H bars. Frozen — never infer at runtime.
BAR_SCHEMA: pa.Schema = pa.schema([
    ("time_msc", pa.int64()),   # bar open, UTC milliseconds, aligned to hour
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.int64()),     # tick count (real volume from MT5 if available)
    ("spread_mean", pa.float64()),
])


@dataclass(frozen=True)
class Bar:
    """One OHLCV bar. Frozen value object."""
    symbol: str
    time_msc: int   # bar OPEN (UTC ms)
    open: float
    high: float
    low: float
    close: float
    volume: int
    spread_mean: float = 0.0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def range_pts(self) -> float:
        """High–low range as a raw price delta. Caller divides by point if needed."""
        return self.high - self.low


def floor_to_timeframe_ms(time_msc: int, timeframe_minutes: int) -> int:
    """Snap a UTC ms timestamp DOWN to the nearest timeframe boundary.

    1H example: floor_to_timeframe_ms(t, 60) returns the open time of the
    hour containing t. 4H example: floor to 00:00, 04:00, 08:00 ... UTC.
    """
    if timeframe_minutes <= 0:
        raise ValueError(f"timeframe_minutes must be > 0, got {timeframe_minutes}")
    period_ms = timeframe_minutes * 60 * 1000
    return (time_msc // period_ms) * period_ms


class BarAggregator:
    """Stateful per-symbol OHLCV aggregator.

    Feed ticks via on_tick(). When a tick crosses into the next bar's window,
    the current bar is CLOSED and returned; the new bar opens from that tick.
    No bar is emitted until the second window's first tick — i.e. you need
    AT LEAST one tick in the next window to see the bar before it close.
    Call flush() at end-of-stream to drain the in-progress bar.

    The aggregator never modifies a Bar after it's emitted (Bar is frozen).
    """

    def __init__(self, symbol: str, timeframe_minutes: int = 60) -> None:
        if timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes must be > 0")
        self._symbol = symbol
        self._tf_min = timeframe_minutes
        self._period_ms = timeframe_minutes * 60 * 1000

        # In-progress bar state. None until the first tick arrives.
        self._bar_open_msc: Optional[int] = None
        self._open: float = 0.0
        self._high: float = 0.0
        self._low: float = 0.0
        self._close: float = 0.0
        self._tick_count: int = 0
        self._spread_sum: float = 0.0

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def timeframe_minutes(self) -> int:
        return self._tf_min

    @property
    def has_open_bar(self) -> bool:
        return self._bar_open_msc is not None

    @property
    def current_bar_open_msc(self) -> Optional[int]:
        return self._bar_open_msc

    def on_tick(self, tick: Tick) -> Optional[Bar]:
        """Ingest one tick. Return a closed Bar if this tick crossed a boundary.

        Mid price = (bid + ask) / 2 — same convention as the strategy layer.
        Returns None when the tick belongs to the current (still-open) bar.
        """
        bucket = floor_to_timeframe_ms(tick.time_msc, self._tf_min)
        mid = (tick.bid + tick.ask) * 0.5
        spread = tick.ask - tick.bid

        # First tick: open the first bar.
        if self._bar_open_msc is None:
            self._open_new_bar(bucket, mid, spread)
            return None

        # Same bar: update high/low/close + accumulate.
        if bucket == self._bar_open_msc:
            if mid > self._high:
                self._high = mid
            if mid < self._low:
                self._low = mid
            self._close = mid
            self._tick_count += 1
            self._spread_sum += spread
            return None

        # New bar: close the prior and open the next.
        # Note: a single tick can skip multiple empty windows (illiquid pairs,
        # weekend roll). We emit only the CLOSED bar; intervening empty bars
        # are silently dropped — caller can detect gaps via time_msc deltas.
        closed = self._build_closed_bar()
        self._open_new_bar(bucket, mid, spread)
        return closed

    def flush(self) -> Optional[Bar]:
        """End-of-stream drain. Returns the in-progress bar if one is open."""
        if self._bar_open_msc is None:
            return None
        bar = self._build_closed_bar()
        self._bar_open_msc = None
        self._tick_count = 0
        self._spread_sum = 0.0
        return bar

    # ----------------------------------------------------------------- helpers

    def _open_new_bar(self, bucket: int, mid: float, spread: float) -> None:
        self._bar_open_msc = bucket
        self._open = mid
        self._high = mid
        self._low = mid
        self._close = mid
        self._tick_count = 1
        self._spread_sum = spread

    def _build_closed_bar(self) -> Bar:
        assert self._bar_open_msc is not None
        spread_mean = (
            self._spread_sum / self._tick_count if self._tick_count > 0 else 0.0
        )
        return Bar(
            symbol=self._symbol,
            time_msc=self._bar_open_msc,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._tick_count,
            spread_mean=spread_mean,
        )


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------

def bars_path(symbol: str, timeframe: str = "1H", bars_dir: Optional[Path] = None) -> Path:
    """Conventional path for stored bars. Mirrors `symbol={X}_{timeframe}.parquet`."""
    base = bars_dir if bars_dir is not None else settings.bars_dir
    return base / f"{symbol}_{timeframe}.parquet"


def write_bars_parquet(
    bars: list[Bar],
    symbol: str,
    timeframe: str = "1H",
    bars_dir: Optional[Path] = None,
    compression: str = "snappy",
) -> Path:
    """Write a list of bars to parquet. Overwrites the file if it exists.

    `symbol` is taken from the argument, not bars[0].symbol, so an empty list
    still produces a deterministic path. Caller should ensure all bars belong
    to the same symbol.
    """
    path = bars_path(symbol, timeframe, bars_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "time_msc": pa.array([b.time_msc for b in bars], type=pa.int64()),
            "open": pa.array([b.open for b in bars], type=pa.float64()),
            "high": pa.array([b.high for b in bars], type=pa.float64()),
            "low": pa.array([b.low for b in bars], type=pa.float64()),
            "close": pa.array([b.close for b in bars], type=pa.float64()),
            "volume": pa.array([b.volume for b in bars], type=pa.int64()),
            "spread_mean": pa.array([b.spread_mean for b in bars], type=pa.float64()),
        },
        schema=BAR_SCHEMA,
    )
    pq.write_table(table, path, compression=compression)
    return path


def read_bars_parquet(
    symbol: str,
    timeframe: str = "1H",
    bars_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Read stored bars to a DataFrame, sorted ascending by time_msc.

    DataFrame columns: time_msc, open, high, low, close, volume, spread_mean.
    Caller can `pd.to_datetime(df.time_msc, unit="ms", utc=True)` to get
    timezone-aware timestamps.
    """
    path = bars_path(symbol, timeframe, bars_dir)
    if not path.exists():
        raise FileNotFoundError(f"Bars not found: {path}")
    table = pq.read_table(path)
    df = table.to_pandas()
    df = df.sort_values("time_msc").reset_index(drop=True)
    return df


def check_bar_integrity(df: pd.DataFrame, timeframe_minutes: int = 60) -> dict:
    """Quick integrity scan over a bar DataFrame. Pure function — returns a dict.

    Flags:
      - monotonic            : strictly increasing time_msc.
      - aligned              : all time_msc on the timeframe boundary.
      - missing_count        : how many expected bars are absent in the range.
      - ohlc_consistent      : low <= min(open, close) and high >= max(open, close).
      - rows                 : total bar count.
    """
    if df.empty:
        return {
            "rows": 0, "monotonic": True, "aligned": True,
            "missing_count": 0, "ohlc_consistent": True,
        }

    times = df["time_msc"].to_numpy()
    period_ms = timeframe_minutes * 60 * 1000

    monotonic = bool((times[1:] > times[:-1]).all()) if len(times) > 1 else True
    aligned = bool((times % period_ms == 0).all())

    # Expected count given the range, vs actual.
    span = times[-1] - times[0]
    expected = span // period_ms + 1
    missing = int(expected - len(times))

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    oc_min = o.copy()
    oc_min = (oc_min < c).astype(float) * oc_min + (oc_min >= c).astype(float) * c
    oc_max = o.copy()
    oc_max = (oc_max > c).astype(float) * oc_max + (oc_max <= c).astype(float) * c
    ohlc_consistent = bool((l <= oc_min + 1e-9).all() and (h >= oc_max - 1e-9).all())

    return {
        "rows": int(len(times)),
        "monotonic": monotonic,
        "aligned": aligned,
        "missing_count": missing,
        "ohlc_consistent": ohlc_consistent,
    }
