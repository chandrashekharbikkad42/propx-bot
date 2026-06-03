"""Pure helpers for the historical-bar capture pipeline.

Kept separate from `scripts/capture_historical_bars.py` so the conversion
logic (MT5 rates ndarray → list[Bar]) is testable without a live MT5
connection.
"""

from __future__ import annotations
from typing import Sequence

import numpy as np

from data.bar_aggregator import Bar


def mt5_rates_to_bars(rates: np.ndarray, symbol: str) -> list[Bar]:
    """Convert an MT5 `copy_rates_*` ndarray to `list[Bar]`.

    MT5 rates fields used:
      - time         : epoch seconds (UTC)
      - open / high / low / close
      - tick_volume  : tick count for the bar
      - spread       : POINTS at bar close (we mirror this into spread_mean)
      - real_volume  : optional, ignored — we use tick_volume

    Returns bars in input order (MT5 returns ascending by time).
    """
    if rates is None or len(rates) == 0:
        return []
    bars: list[Bar] = []
    for row in rates:
        time_msc = int(row["time"]) * 1000
        spread_raw = float(row["spread"]) if "spread" in rates.dtype.names else 0.0
        bars.append(
            Bar(
                symbol=symbol,
                time_msc=time_msc,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["tick_volume"]),
                spread_mean=spread_raw,
            )
        )
    return bars


def bars_summary(bars: Sequence[Bar]) -> dict:
    """Quick summary stats for a captured bar set."""
    if not bars:
        return {"count": 0, "first_msc": 0, "last_msc": 0, "span_days": 0.0}
    first = bars[0].time_msc
    last = bars[-1].time_msc
    span_days = (last - first) / (1000.0 * 60 * 60 * 24)
    return {
        "count": len(bars),
        "first_msc": first,
        "last_msc": last,
        "span_days": span_days,
    }
