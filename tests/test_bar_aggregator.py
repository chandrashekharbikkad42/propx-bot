"""Phase 8B — BarAggregator + parquet I/O.

Covers:
  - Hour-boundary detection
  - OHLC correctness (open=first, high=max, low=min, close=last)
  - Bar emission on boundary cross
  - flush() drains in-progress bar
  - Multi-bar tick stream
  - Skipped windows (illiquid gap)
  - Parquet round-trip
  - Integrity checker
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from data.bar_aggregator import (
    BAR_SCHEMA,
    Bar,
    BarAggregator,
    bars_path,
    check_bar_integrity,
    floor_to_timeframe_ms,
    read_bars_parquet,
    write_bars_parquet,
)
from data.tick_collector import Tick


def _utc_ms(year: int, month: int, day: int, hour: int, minute: int, sec: int = 0) -> int:
    return int(
        datetime(year, month, day, hour, minute, sec, tzinfo=timezone.utc).timestamp() * 1000
    )


def _mk_tick(time_msc: int, bid: float, ask: Optional[float] = None) -> Tick:
    ask_v = ask if ask is not None else bid + 0.0002
    return Tick(
        time_msc=time_msc,
        bid=bid,
        ask=ask_v,
        last=bid,
        volume=1,
        volume_real=1.0,
        flags=0,
    )


class TestFloorToTimeframe:
    def test_floors_to_hour_start(self):
        # 2026-05-17 07:43:21 UTC → 07:00
        t = _utc_ms(2026, 5, 17, 7, 43, 21)
        floored = floor_to_timeframe_ms(t, 60)
        assert floored == _utc_ms(2026, 5, 17, 7, 0)

    def test_already_on_boundary(self):
        t = _utc_ms(2026, 5, 17, 12, 0)
        assert floor_to_timeframe_ms(t, 60) == t

    def test_4h_boundary(self):
        # 09:00 UTC → 08:00 (4H buckets: 00,04,08,12,16,20)
        t = _utc_ms(2026, 5, 17, 9, 0)
        assert floor_to_timeframe_ms(t, 240) == _utc_ms(2026, 5, 17, 8, 0)

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            floor_to_timeframe_ms(0, 0)
        with pytest.raises(ValueError):
            floor_to_timeframe_ms(0, -5)


class TestBarAggregatorBasics:
    def test_first_tick_opens_bar_no_emit(self):
        agg = BarAggregator("EURUSD", 60)
        out = agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 7, 5), 1.1000))
        assert out is None
        assert agg.has_open_bar is True
        assert agg.current_bar_open_msc == _utc_ms(2026, 5, 17, 7, 0)

    def test_same_bar_ticks_no_emit(self):
        agg = BarAggregator("EURUSD", 60)
        agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 7, 0), 1.1000))
        out = agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 7, 30), 1.1010))
        out2 = agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 7, 59), 1.1005))
        assert out is None and out2 is None
        assert agg.has_open_bar is True

    def test_boundary_cross_emits_prior_bar(self):
        agg = BarAggregator("EURUSD", 60)
        agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 7, 0), 1.1000))
        agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 7, 30), 1.1020))
        agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 7, 45), 1.0990))
        # Cross into the 08:00 bar.
        bar = agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 8, 5), 1.1005))
        assert isinstance(bar, Bar)
        assert bar.symbol == "EURUSD"
        assert bar.time_msc == _utc_ms(2026, 5, 17, 7, 0)
        # Mids of (1.1000+0.0002)/2 style — but our mock uses ask=bid+0.0002
        # so mid == bid+0.0001.
        assert bar.open == pytest.approx(1.1001)
        assert bar.high == pytest.approx(1.1021)
        assert bar.low == pytest.approx(1.0991)
        assert bar.close == pytest.approx(1.0991)  # last tick of prior bar
        assert bar.volume == 3

    def test_flush_drains_open_bar(self):
        agg = BarAggregator("GBPUSD", 60)
        agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 9, 0), 1.2500))
        agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 9, 30), 1.2510))
        bar = agg.flush()
        assert isinstance(bar, Bar)
        assert bar.volume == 2
        assert bar.time_msc == _utc_ms(2026, 5, 17, 9, 0)
        # After flush, no bar pending.
        assert agg.flush() is None

    def test_flush_on_empty_returns_none(self):
        agg = BarAggregator("EURUSD", 60)
        assert agg.flush() is None

    def test_invalid_timeframe(self):
        with pytest.raises(ValueError):
            BarAggregator("X", 0)


class TestBarAggregatorOhlcCorrectness:
    def test_high_is_max_low_is_min(self):
        agg = BarAggregator("EURUSD", 60)
        prices = [1.1000, 1.1050, 1.0990, 1.1030, 1.1010]
        for i, p in enumerate(prices):
            agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 7, i * 5), p))
        bar = agg.flush()
        assert bar is not None
        # mids = price + 0.0001
        assert bar.open == pytest.approx(prices[0] + 0.0001)
        assert bar.close == pytest.approx(prices[-1] + 0.0001)
        assert bar.high == pytest.approx(max(prices) + 0.0001)
        assert bar.low == pytest.approx(min(prices) + 0.0001)
        assert bar.volume == len(prices)

    def test_spread_mean(self):
        agg = BarAggregator("EURUSD", 60)
        agg.on_tick(Tick(_utc_ms(2026, 5, 17, 7, 0), 1.0, 1.0002, 1.0, 1, 1.0, 0))
        agg.on_tick(Tick(_utc_ms(2026, 5, 17, 7, 30), 1.0, 1.0004, 1.0, 1, 1.0, 0))
        bar = agg.flush()
        assert bar is not None
        assert bar.spread_mean == pytest.approx(0.0003)


class TestBarAggregatorSkippedWindows:
    def test_gap_skips_silently_emits_prior_only(self):
        # Tick at 07:00, then nothing until 12:00 — five empty windows.
        # We emit only the closed 07:00 bar; 08–11 are dropped.
        agg = BarAggregator("EURUSD", 60)
        agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 7, 0), 1.1000))
        bar = agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 12, 5), 1.1100))
        assert isinstance(bar, Bar)
        assert bar.time_msc == _utc_ms(2026, 5, 17, 7, 0)
        assert agg.current_bar_open_msc == _utc_ms(2026, 5, 17, 12, 0)


class TestBarAggregatorMultiBarStream:
    def test_three_hour_stream(self):
        agg = BarAggregator("EURUSD", 60)
        bars: list[Bar] = []
        # Bar 07: ticks at :00, :30
        for t in [
            _mk_tick(_utc_ms(2026, 5, 17, 7, 0), 1.10),
            _mk_tick(_utc_ms(2026, 5, 17, 7, 30), 1.11),
        ]:
            out = agg.on_tick(t)
            if out: bars.append(out)
        # Cross into bar 08 — emit 07.
        out = agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 8, 0), 1.12))
        if out: bars.append(out)
        out = agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 8, 45), 1.115))
        if out: bars.append(out)
        # Cross into bar 09 — emit 08.
        out = agg.on_tick(_mk_tick(_utc_ms(2026, 5, 17, 9, 5), 1.118))
        if out: bars.append(out)
        # Drain.
        final = agg.flush()
        if final: bars.append(final)

        assert len(bars) == 3
        assert bars[0].time_msc == _utc_ms(2026, 5, 17, 7, 0)
        assert bars[1].time_msc == _utc_ms(2026, 5, 17, 8, 0)
        assert bars[2].time_msc == _utc_ms(2026, 5, 17, 9, 0)


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------

class TestParquetRoundTrip:
    def test_write_and_read_preserves_data(self, tmp_path: Path):
        bars = [
            Bar("EURUSD", _utc_ms(2026, 5, 17, 7, 0), 1.10, 1.11, 1.09, 1.105, 100, 0.0002),
            Bar("EURUSD", _utc_ms(2026, 5, 17, 8, 0), 1.105, 1.115, 1.10, 1.11, 80, 0.0003),
        ]
        path = write_bars_parquet(bars, "EURUSD", "1H", bars_dir=tmp_path)
        assert path.exists()
        df = read_bars_parquet("EURUSD", "1H", bars_dir=tmp_path)
        assert len(df) == 2
        assert list(df.columns) == [
            "time_msc", "open", "high", "low", "close", "volume", "spread_mean"
        ]
        assert df.iloc[0]["open"] == pytest.approx(1.10)
        assert df.iloc[1]["close"] == pytest.approx(1.11)

    def test_read_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_bars_parquet("NEVER", "1H", bars_dir=tmp_path)

    def test_path_convention(self, tmp_path: Path):
        p = bars_path("EURUSD", "1H", bars_dir=tmp_path)
        assert p.name == "EURUSD_1H.parquet"


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

class TestCheckBarIntegrity:
    def _df(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_empty(self):
        r = check_bar_integrity(self._df([]))
        assert r["rows"] == 0 and r["monotonic"] and r["aligned"]

    def test_clean_three_consecutive_bars(self):
        rows = []
        for h in (7, 8, 9):
            ms = _utc_ms(2026, 5, 17, h, 0)
            rows.append({"time_msc": ms, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 10, "spread_mean": 0.0001})
        r = check_bar_integrity(self._df(rows))
        assert r["rows"] == 3
        assert r["monotonic"] is True
        assert r["aligned"] is True
        assert r["missing_count"] == 0
        assert r["ohlc_consistent"] is True

    def test_detects_gap(self):
        # 07, 08, then jump to 11 — missing 09 and 10.
        rows = []
        for h in (7, 8, 11):
            ms = _utc_ms(2026, 5, 17, h, 0)
            rows.append({"time_msc": ms, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 10, "spread_mean": 0.0})
        r = check_bar_integrity(self._df(rows))
        assert r["missing_count"] == 2

    def test_detects_misalignment(self):
        # Bar starts at 07:30 — not on a 1H boundary.
        rows = [
            {"time_msc": _utc_ms(2026, 5, 17, 7, 30), "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1, "spread_mean": 0.0},
        ]
        r = check_bar_integrity(self._df(rows))
        assert r["aligned"] is False

    def test_detects_ohlc_inconsistency(self):
        # low > open → inconsistent
        rows = [
            {"time_msc": _utc_ms(2026, 5, 17, 7, 0), "open": 1.0, "high": 1.1, "low": 1.05, "close": 0.95, "volume": 1, "spread_mean": 0.0},
        ]
        r = check_bar_integrity(self._df(rows))
        assert r["ohlc_consistent"] is False


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------

class TestBarSchema:
    def test_fields_in_expected_order(self):
        names = [f.name for f in BAR_SCHEMA]
        assert names == [
            "time_msc", "open", "high", "low", "close", "volume", "spread_mean"
        ]

    def test_types(self):
        types = {f.name: str(f.type) for f in BAR_SCHEMA}
        assert types["time_msc"] == "int64"
        assert types["open"] == "double"
        assert types["volume"] == "int64"
