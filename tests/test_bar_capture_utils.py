"""Phase 8B — bar capture utility tests (no live MT5).

Tests the pure conversion logic: MT5 rates ndarray → list[Bar] and the
bars_summary helper. Live MT5 fetch is exercised manually by the user via
`scripts/capture_historical_bars.py`.
"""

from __future__ import annotations
from datetime import datetime, timezone

import numpy as np

from data.bar_aggregator import Bar
from data.bar_capture_utils import bars_summary, mt5_rates_to_bars


def _mt5_rates_dtype() -> np.dtype:
    """Mirrors MT5 copy_rates_range structured array dtype."""
    return np.dtype([
        ("time", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("tick_volume", "i8"),
        ("spread", "i4"),
        ("real_volume", "i8"),
    ])


def _make_rates(rows: list[tuple]) -> np.ndarray:
    """rows: list of (time_s, open, high, low, close, tick_vol, spread, real_vol)."""
    arr = np.zeros(len(rows), dtype=_mt5_rates_dtype())
    for i, r in enumerate(rows):
        arr[i] = r
    return arr


class TestMt5RatesToBars:
    def test_empty_input_returns_empty_list(self):
        assert mt5_rates_to_bars(np.empty(0, dtype=_mt5_rates_dtype()), "EURUSD") == []

    def test_none_input_returns_empty(self):
        assert mt5_rates_to_bars(None, "EURUSD") == []  # type: ignore[arg-type]

    def test_single_row_conversion(self):
        t = int(datetime(2026, 5, 17, 7, 0, tzinfo=timezone.utc).timestamp())
        rates = _make_rates([(t, 1.10, 1.11, 1.09, 1.105, 50, 12, 50)])
        bars = mt5_rates_to_bars(rates, "EURUSD")
        assert len(bars) == 1
        b = bars[0]
        assert isinstance(b, Bar)
        assert b.symbol == "EURUSD"
        assert b.time_msc == t * 1000
        assert b.open == 1.10
        assert b.high == 1.11
        assert b.low == 1.09
        assert b.close == 1.105
        assert b.volume == 50
        assert b.spread_mean == 12.0  # mirrored from MT5 'spread' field

    def test_multiple_rows_preserve_order(self):
        t0 = int(datetime(2026, 5, 17, 7, 0, tzinfo=timezone.utc).timestamp())
        rates = _make_rates([
            (t0 + i * 3600, 1.10 + i * 0.001, 1.11, 1.09, 1.10 + i * 0.001, 50, 10, 50)
            for i in range(5)
        ])
        bars = mt5_rates_to_bars(rates, "EURUSD")
        assert len(bars) == 5
        # Should be ascending by time_msc (MT5 returns ascending).
        times = [b.time_msc for b in bars]
        assert times == sorted(times)


class TestBarsSummary:
    def test_empty(self):
        s = bars_summary([])
        assert s == {"count": 0, "first_msc": 0, "last_msc": 0, "span_days": 0.0}

    def test_single_bar_zero_span(self):
        b = Bar("EURUSD", 1_000_000, 1.0, 1.0, 1.0, 1.0, 1)
        s = bars_summary([b])
        assert s["count"] == 1
        assert s["span_days"] == 0.0

    def test_span_two_days(self):
        day_ms = 24 * 60 * 60 * 1000
        bars = [
            Bar("EURUSD", 0, 1.0, 1.0, 1.0, 1.0, 1),
            Bar("EURUSD", 2 * day_ms, 1.0, 1.0, 1.0, 1.0, 1),
        ]
        s = bars_summary(bars)
        assert s["count"] == 2
        assert s["span_days"] == 2.0
