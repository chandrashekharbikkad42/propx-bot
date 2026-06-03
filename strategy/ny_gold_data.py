"""NY Gold Sweep — look-ahead-safe data layer (NY_GOLD_SWEEP_SPEC.md §0.1 / §9).

Loads XAUUSD parquet bars and exposes ONLY past-visible slices:

  ngd = NYGoldData.load()
  for t, bar_1m in ngd.iter_decisions(start_ms, end_ms):
      v1 = ngd.get_visible("1M",  t)
      v5 = ngd.get_visible("5M",  t)
      vH = ngd.get_visible("15M", t)
      ngd.assert_no_lookahead(t, v1, v5, vH)
      # ... detector reads ONLY these slices.

Time semantics
--------------
- `time_msc` in the parquet = bar OPEN time (broker convention, MT5).
- `close_time = time_msc + BAR_MS[tf]`.
- Decision time `t` is the close timestamp of the trigger 1M bar.

Visibility
----------
At decision time `t`, a bar of timeframe TF is **visible** iff its close_time
is `<= t`. For 1M, the trigger bar itself (close_time == t) is visible. For
5M / 15M / 1H, a bar currently forming (open_time <= t < close_time) is
NOT visible — exactly the rule that killed prior phases.

No DataFrame returned by `get_visible` may contain bars beyond t. The
`assert_no_lookahead` helper verifies this on every call (cheap O(1) check
since slices are tail-trimmed using bisect).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from config.ny_gold_sweep_config import BAR_MS, SYMBOL


_TIMEFRAMES: tuple[str, ...] = ("1M", "5M", "15M", "1H")

_PARQUET_DIR = Path(__file__).resolve().parents[1] / "data" / "bars"


class LookaheadError(AssertionError):
    """Raised by assert_no_lookahead when any slice contains future bars."""


@dataclass(frozen=True)
class BarFrame:
    """One timeframe's bar data, columnar for hot-loop access.

    All arrays are aligned and parallel — index i refers to the same bar
    across `time_msc`, `open`, `high`, etc. `close_time = time_msc + bar_ms`.

    `close_time_arr` is precomputed and sorted, used by `get_visible` for
    bisect-based tail-trim.
    """
    tf: str
    bar_ms: int
    time_msc: np.ndarray     # int64 open time
    close_time: np.ndarray   # int64 close time (= time_msc + bar_ms)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    spread_pts: np.ndarray   # broker points; divide by 100 for pips (§0.2)

    def __len__(self) -> int:
        return int(self.time_msc.shape[0])

    def visible_count(self, t: int) -> int:
        """Number of bars with close_time <= t — uses sorted bisect_right.

        Returns the count, NOT a slice — callers slice with this index to
        avoid materializing a new DataFrame on every decision.
        """
        return bisect.bisect_right(self.close_time, t)

    def slice(self, t: int) -> "BarFrame":
        """Return a new BarFrame containing only bars with close_time <= t."""
        n = self.visible_count(t)
        if n == len(self):
            return self
        return BarFrame(
            tf=self.tf,
            bar_ms=self.bar_ms,
            time_msc=self.time_msc[:n],
            close_time=self.close_time[:n],
            open=self.open[:n],
            high=self.high[:n],
            low=self.low[:n],
            close=self.close[:n],
            volume=self.volume[:n],
            spread_pts=self.spread_pts[:n],
        )


@dataclass(frozen=True)
class NYGoldData:
    """Container of all four XAUUSD timeframes."""
    one_m: BarFrame
    five_m: BarFrame
    fifteen_m: BarFrame
    one_h: BarFrame

    @classmethod
    def load(cls, symbol: str = SYMBOL, parquet_dir: Optional[Path] = None) -> "NYGoldData":
        d = parquet_dir or _PARQUET_DIR
        frames: dict[str, BarFrame] = {}
        for tf in _TIMEFRAMES:
            path = d / f"{symbol}_{tf}.parquet"
            df = pd.read_parquet(path)
            if not df["time_msc"].is_monotonic_increasing:
                df = df.sort_values("time_msc", kind="stable").reset_index(drop=True)
            bar_ms = BAR_MS[tf]
            time_msc = df["time_msc"].to_numpy(dtype=np.int64)
            close_time = time_msc + bar_ms
            frames[tf] = BarFrame(
                tf=tf,
                bar_ms=bar_ms,
                time_msc=time_msc,
                close_time=close_time,
                open=df["open"].to_numpy(dtype=np.float64),
                high=df["high"].to_numpy(dtype=np.float64),
                low=df["low"].to_numpy(dtype=np.float64),
                close=df["close"].to_numpy(dtype=np.float64),
                volume=df["volume"].to_numpy(dtype=np.int64),
                spread_pts=df["spread_mean"].to_numpy(dtype=np.float64),
            )
        return cls(
            one_m=frames["1M"],
            five_m=frames["5M"],
            fifteen_m=frames["15M"],
            one_h=frames["1H"],
        )

    # ─── visibility API ─────────────────────────────────────────────────
    def get_visible(self, tf: str, t: int) -> BarFrame:
        """Return the slice of TF bars with close_time <= t (§0.1 / §9)."""
        if tf == "1M":
            return self.one_m.slice(t)
        if tf == "5M":
            return self.five_m.slice(t)
        if tf == "15M":
            return self.fifteen_m.slice(t)
        if tf == "1H":
            return self.one_h.slice(t)
        raise ValueError(f"unknown timeframe: {tf}")

    def assert_no_lookahead(
        self,
        t: int,
        v1: Optional[BarFrame] = None,
        v5: Optional[BarFrame] = None,
        v15: Optional[BarFrame] = None,
        v1h: Optional[BarFrame] = None,
    ) -> None:
        """Verify each provided slice has no bar with close_time > t.

        Called every decision in the backtest main loop. O(1) per slice
        (just inspects the last close_time). Raises LookaheadError on
        violation — fails LOUD, no silent drift.
        """
        for frame, label in (
            (v1, "1M"), (v5, "5M"), (v15, "15M"), (v1h, "1H"),
        ):
            if frame is None or len(frame) == 0:
                continue
            last_ct = int(frame.close_time[-1])
            if last_ct > t:
                raise LookaheadError(
                    f"{label} slice contains bar with close_time={last_ct} > t={t} "
                    f"(delta={last_ct - t} ms)"
                )

    # ─── decision iterator ──────────────────────────────────────────────
    def iter_decisions(
        self,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> Iterator[Tuple[int, int]]:
        """Yield `(t, idx)` for every closed 1M bar in [start_ms, end_ms].

        `t` = close_time of the bar (= the legal "now" for any detector
        decision at this bar's close).
        `idx` = positional index into the 1M arrays of THIS bar (= the
        trigger candidate S).

        Filtering by NY session window is the CALLER's responsibility —
        this iterator simply walks the closed 1M series. Backtest narrows
        further; detector double-checks.
        """
        ct = self.one_m.close_time
        lo = 0 if start_ms is None else bisect.bisect_left(ct, int(start_ms))
        hi = len(self.one_m) if end_ms is None else bisect.bisect_right(ct, int(end_ms))
        for i in range(lo, hi):
            yield int(ct[i]), i


# ─── helpers ────────────────────────────────────────────────────────────────
def utc_hms(time_msc: int) -> Tuple[int, int, int]:
    """Decompose epoch-ms to UTC (h, m, s). Pure stdlib for hot-loop safety."""
    sec = time_msc // 1000
    h = (sec // 3600) % 24
    m = (sec // 60) % 60
    s = sec % 60
    return int(h), int(m), int(s)


def utc_date_key(time_msc: int) -> int:
    """UTC calendar day key (epoch days). Used by daily counter / DD reset."""
    return int(time_msc // 86_400_000)


def in_session(
    time_msc_open: int,
    start_hms: Tuple[int, int, int],
    end_hms: Tuple[int, int, int],
) -> bool:
    """True if the bar's OPEN time falls in [start, end) UTC.

    Per §1: open=12:00 in, open=17:00 out, open=16:59 in.
    """
    h, m, s = utc_hms(time_msc_open)
    cur = h * 3600 + m * 60 + s
    lo = start_hms[0] * 3600 + start_hms[1] * 60 + start_hms[2]
    hi = end_hms[0] * 3600 + end_hms[1] * 60 + end_hms[2]
    return lo <= cur < hi


def is_weekend(time_msc: int) -> bool:
    """True if the UTC weekday is Saturday or Sunday (epoch day mod 7)."""
    # 1970-01-01 was a Thursday → epoch_day % 7: 0=Thu,1=Fri,2=Sat,3=Sun,4=Mon,5=Tue,6=Wed
    return (utc_date_key(time_msc) % 7) in (2, 3)
