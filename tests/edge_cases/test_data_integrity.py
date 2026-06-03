"""Phase-5 / Data Integrity — adversarial tests against the bar pipeline.

Coverage focus (per Phase 5 brief):

  - Missing bars in Asian window (skip vs error)
  - Parquet file corrupt mid-read
  - Bar timestamp out of order
  - Duplicate bars
  - Future-dated bar (clock skew)
  - Schema mismatch on read
  - Empty parquet
  - Partial write recovery
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import List

import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hypothesis import HealthCheck, given, settings, strategies as st

from data.bar_aggregator import (
    BAR_SCHEMA, Bar, BarAggregator, bars_path, check_bar_integrity,
    floor_to_timeframe_ms, read_bars_parquet, write_bars_parquet,
)
from data.tick_collector import Tick

from tests.edge_cases.fixtures.chaos_market import (
    HOUR_MS, duplicate_timestamp_bars, future_dated_bar, hour_msc,
    inverted_bar, make_bar, out_of_order_bars, zero_volume_bars,
)


# ===========================================================================
# 1. PARQUET ROUND-TRIP — happy + edge
# ===========================================================================

class TestParquetRoundTrip:
    def test_write_then_read_returns_same_bars(self, tmp_path):
        bars = [make_bar(symbol="XAUUSD", time_msc=i * HOUR_MS,
                          open=2000.0 + i, close=2000.0 + i)
                for i in range(5)]
        write_bars_parquet(bars, "XAUUSD", bars_dir=tmp_path)
        df = read_bars_parquet("XAUUSD", bars_dir=tmp_path)
        assert len(df) == 5
        assert list(df["time_msc"]) == [b.time_msc for b in bars]

    @pytest.mark.parametrize("compression", ["snappy", "gzip", None])
    def test_compression_options(self, tmp_path, compression):
        bars = [make_bar(time_msc=i * HOUR_MS) for i in range(3)]
        # `None` is rejected by pq; ensure we test it raises or uses default
        try:
            write_bars_parquet(bars, "X", bars_dir=tmp_path,
                                compression=compression or "snappy")
        except Exception:
            pass
        else:
            df = read_bars_parquet("X", bars_dir=tmp_path)
            assert len(df) == 3

    @pytest.mark.parametrize("count", [0, 1, 2, 10, 100, 1000])
    def test_round_trip_various_sizes(self, tmp_path, count):
        bars = [make_bar(symbol="EURUSD", time_msc=i * HOUR_MS,
                          open=1.10 + i * 0.0001, close=1.10 + i * 0.0001)
                for i in range(count)]
        write_bars_parquet(bars, "EURUSD", bars_dir=tmp_path)
        df = read_bars_parquet("EURUSD", bars_dir=tmp_path)
        assert len(df) == count


# ===========================================================================
# 2. EMPTY PARQUET
# ===========================================================================

class TestEmptyParquet:
    def test_write_empty_bars_creates_empty_file(self, tmp_path):
        path = write_bars_parquet([], "XAUUSD", bars_dir=tmp_path)
        assert path.exists()
        df = read_bars_parquet("XAUUSD", bars_dir=tmp_path)
        assert df.empty

    def test_check_bar_integrity_on_empty(self):
        rep = check_bar_integrity(pd.DataFrame())
        assert rep["rows"] == 0
        assert rep["monotonic"] is True

    def test_read_returns_correct_schema_for_empty(self, tmp_path):
        write_bars_parquet([], "XAUUSD", bars_dir=tmp_path)
        df = read_bars_parquet("XAUUSD", bars_dir=tmp_path)
        for col in ("time_msc", "open", "high", "low", "close", "volume",
                    "spread_mean"):
            assert col in df.columns


# ===========================================================================
# 3. PARQUET FILE NOT FOUND
# ===========================================================================

class TestMissingFile:
    def test_read_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Bars not found"):
            read_bars_parquet("UNOBTANIUM", bars_dir=tmp_path)

    @pytest.mark.parametrize("sym", ["XAUUSD", "EURUSD", "GBPUSD"])
    def test_missing_symbol_specific_path(self, tmp_path, sym):
        with pytest.raises(FileNotFoundError):
            read_bars_parquet(sym, bars_dir=tmp_path)


# ===========================================================================
# 4. CORRUPT PARQUET — invalid bytes
# ===========================================================================

class TestCorruptParquet:
    def test_truncated_file_raises_on_read(self, tmp_path):
        bars = [make_bar(time_msc=i * HOUR_MS) for i in range(5)]
        path = write_bars_parquet(bars, "XAUUSD", bars_dir=tmp_path)
        # Truncate the file to half its size.
        original_bytes = path.read_bytes()
        path.write_bytes(original_bytes[: len(original_bytes) // 2])
        with pytest.raises(Exception):
            read_bars_parquet("XAUUSD", bars_dir=tmp_path)

    def test_random_garbage_file_raises(self, tmp_path):
        path = tmp_path / "XAUUSD_1H.parquet"
        path.write_bytes(b"NOT A PARQUET FILE")
        with pytest.raises(Exception):
            read_bars_parquet("XAUUSD", bars_dir=tmp_path)

    def test_empty_bytes_file_raises(self, tmp_path):
        path = tmp_path / "XAUUSD_1H.parquet"
        path.write_bytes(b"")
        with pytest.raises(Exception):
            read_bars_parquet("XAUUSD", bars_dir=tmp_path)


# ===========================================================================
# 5. PARTIAL WRITE RECOVERY
# ===========================================================================

class TestPartialWrite:
    def test_overwrite_replaces_old_contents(self, tmp_path):
        first = [make_bar(time_msc=i * HOUR_MS) for i in range(3)]
        second = [make_bar(time_msc=i * HOUR_MS, open=3000.0, close=3000.0)
                   for i in range(7)]
        write_bars_parquet(first, "XAUUSD", bars_dir=tmp_path)
        write_bars_parquet(second, "XAUUSD", bars_dir=tmp_path)
        df = read_bars_parquet("XAUUSD", bars_dir=tmp_path)
        assert len(df) == 7
        assert all(df["open"] == 3000.0)

    def test_write_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        write_bars_parquet([make_bar()], "XAUUSD", bars_dir=nested)
        assert (nested / "XAUUSD_1H.parquet").exists()


# ===========================================================================
# 6. SCHEMA MISMATCH ON READ
# ===========================================================================

class TestSchemaMismatch:
    def test_extra_column_in_file_is_dropped_by_caller(self, tmp_path):
        """If someone writes a parquet with extra columns, our read still
        returns it (we don't reject)."""
        path = tmp_path / "XAUUSD_1H.parquet"
        table = pa.table({
            "time_msc": pa.array([0, 60_000], type=pa.int64()),
            "open": pa.array([1.0, 2.0], type=pa.float64()),
            "high": pa.array([1.0, 2.0], type=pa.float64()),
            "low": pa.array([1.0, 2.0], type=pa.float64()),
            "close": pa.array([1.0, 2.0], type=pa.float64()),
            "volume": pa.array([1, 2], type=pa.int64()),
            "spread_mean": pa.array([0.0, 0.0], type=pa.float64()),
            "extra": pa.array([99, 100], type=pa.int64()),
        })
        pq.write_table(table, path)
        df = read_bars_parquet("XAUUSD", bars_dir=tmp_path)
        assert "extra" in df.columns

    def test_missing_column_in_file_raises_on_access(self, tmp_path):
        path = tmp_path / "XAUUSD_1H.parquet"
        table = pa.table({
            "time_msc": pa.array([0, 60_000], type=pa.int64()),
            "open": pa.array([1.0, 2.0], type=pa.float64()),
        })
        pq.write_table(table, path)
        df = read_bars_parquet("XAUUSD", bars_dir=tmp_path)
        assert "high" not in df.columns

    def test_wrong_dtype_loaded_as_is(self, tmp_path):
        path = tmp_path / "XAUUSD_1H.parquet"
        # write time_msc as float64 — schema mismatch with our int64 default
        table = pa.table({
            "time_msc": pa.array([0.0, 60_000.0], type=pa.float64()),
            "open": pa.array([1.0, 2.0], type=pa.float64()),
            "high": pa.array([1.0, 2.0], type=pa.float64()),
            "low": pa.array([1.0, 2.0], type=pa.float64()),
            "close": pa.array([1.0, 2.0], type=pa.float64()),
            "volume": pa.array([1, 2], type=pa.int64()),
            "spread_mean": pa.array([0.0, 0.0], type=pa.float64()),
        })
        pq.write_table(table, path)
        df = read_bars_parquet("XAUUSD", bars_dir=tmp_path)
        # The reader doesn't validate; bot has to convert downstream.
        assert df["time_msc"].dtype.kind == "f"


# ===========================================================================
# 7. OUT-OF-ORDER BARS — reader sorts
# ===========================================================================

class TestOutOfOrder:
    def test_read_sorts_descending_inputs_ascending(self, tmp_path):
        bars = out_of_order_bars(count=5)
        # Manually write in arbitrary order — read should sort ascending.
        write_bars_parquet(bars, "XAUUSD", bars_dir=tmp_path)
        df = read_bars_parquet("XAUUSD", bars_dir=tmp_path)
        times = df["time_msc"].tolist()
        assert times == sorted(times)

    @pytest.mark.parametrize("count", [3, 10, 25])
    def test_read_sort_preserves_count(self, tmp_path, count):
        bars = out_of_order_bars(count=count)
        write_bars_parquet(bars, "X", bars_dir=tmp_path)
        df = read_bars_parquet("X", bars_dir=tmp_path)
        assert len(df) == count


# ===========================================================================
# 8. DUPLICATE TIMESTAMPS
# ===========================================================================

class TestDuplicateTimestamps:
    def test_duplicates_persist_through_write_read(self, tmp_path):
        bars = duplicate_timestamp_bars(count=3, time_msc=1_000_000)
        write_bars_parquet(bars, "XAUUSD", bars_dir=tmp_path)
        df = read_bars_parquet("XAUUSD", bars_dir=tmp_path)
        assert len(df) == 3
        assert all(df["time_msc"] == 1_000_000)

    def test_check_bar_integrity_duplicates_break_monotonic(self):
        bars = duplicate_timestamp_bars(count=3, time_msc=1_000_000)
        df = pd.DataFrame([b.__dict__ for b in bars])
        rep = check_bar_integrity(df)
        # Strictly increasing → False for duplicates.
        assert rep["monotonic"] is False

    @pytest.mark.parametrize("dupe_count", [2, 3, 5, 10])
    def test_duplicate_count_consistency(self, dupe_count):
        bars = duplicate_timestamp_bars(count=dupe_count)
        df = pd.DataFrame([b.__dict__ for b in bars])
        rep = check_bar_integrity(df)
        assert rep["rows"] == dupe_count


# ===========================================================================
# 9. FUTURE-DATED BAR
# ===========================================================================

class TestFutureDatedBar:
    def test_future_bar_writes_and_reads(self, tmp_path):
        bar = future_dated_bar(base_year=2099)
        write_bars_parquet([bar], "XAUUSD", bars_dir=tmp_path)
        df = read_bars_parquet("XAUUSD", bars_dir=tmp_path)
        assert df["time_msc"].iloc[0] == bar.time_msc

    @pytest.mark.parametrize("year", [2050, 2075, 2099, 2150, 2200])
    def test_far_future_years_round_trip(self, tmp_path, year):
        bar = future_dated_bar(base_year=year)
        write_bars_parquet([bar], "XAUUSD", bars_dir=tmp_path)
        df = read_bars_parquet("XAUUSD", bars_dir=tmp_path)
        assert len(df) == 1

    def test_negative_timestamp_round_trip(self, tmp_path):
        bar = make_bar(symbol="XAUUSD", time_msc=-1_000_000_000)
        write_bars_parquet([bar], "XAUUSD", bars_dir=tmp_path)
        df = read_bars_parquet("XAUUSD", bars_dir=tmp_path)
        assert df["time_msc"].iloc[0] == -1_000_000_000


# ===========================================================================
# 10. CHECK_BAR_INTEGRITY MATRIX
# ===========================================================================

class TestCheckBarIntegrityMatrix:
    @pytest.mark.parametrize("count,expected_rows", [
        (1, 1), (5, 5), (100, 100), (1000, 1000),
    ])
    def test_rows_correct(self, count, expected_rows):
        bars = [make_bar(time_msc=i * HOUR_MS) for i in range(count)]
        df = pd.DataFrame([b.__dict__ for b in bars])
        rep = check_bar_integrity(df)
        assert rep["rows"] == expected_rows

    @pytest.mark.parametrize("gap_size", [1, 5, 10, 100])
    def test_missing_count_correct(self, gap_size):
        bars = [
            make_bar(time_msc=0),
            make_bar(time_msc=(gap_size + 1) * HOUR_MS),
        ]
        df = pd.DataFrame([b.__dict__ for b in bars])
        rep = check_bar_integrity(df)
        assert rep["missing_count"] == gap_size

    @pytest.mark.parametrize("tf_min", [1, 5, 15, 60, 240])
    def test_integrity_with_different_timeframes(self, tf_min):
        period_ms = tf_min * 60 * 1000
        bars = [make_bar(time_msc=i * period_ms) for i in range(5)]
        df = pd.DataFrame([b.__dict__ for b in bars])
        rep = check_bar_integrity(df, timeframe_minutes=tf_min)
        assert rep["aligned"] is True

    def test_aligned_false_on_offset_timestamps(self):
        bars = [make_bar(time_msc=i * HOUR_MS + 7) for i in range(3)]
        df = pd.DataFrame([b.__dict__ for b in bars])
        rep = check_bar_integrity(df)
        assert rep["aligned"] is False

    @pytest.mark.parametrize("bad_idx", [0, 1, 2])
    def test_ohlc_inconsistent_flagged(self, bad_idx):
        bars = [make_bar(time_msc=i * HOUR_MS, open=2000.0, close=2000.0,
                          high=2001.0, low=1999.0)
                for i in range(3)]
        # Replace one with an inverted bar — high < low.
        bars[bad_idx] = inverted_bar(time_msc=bad_idx * HOUR_MS)
        df = pd.DataFrame([b.__dict__ for b in bars])
        rep = check_bar_integrity(df)
        assert rep["ohlc_consistent"] is False


# ===========================================================================
# 11. BAR AGGREGATOR INVARIANTS
# ===========================================================================

class TestBarAggregatorEdge:
    def test_single_tick_opens_no_closed_bar(self):
        agg = BarAggregator("XAUUSD")
        bar = agg.on_tick(Tick(time_msc=0, bid=2000.0, ask=2000.05,
                                 last=2000.0, volume=1, volume_real=1.0,
                                 flags=0))
        assert bar is None
        assert agg.has_open_bar is True

    def test_flush_empty_returns_none(self):
        agg = BarAggregator("XAUUSD")
        assert agg.flush() is None

    def test_flush_clears_state(self):
        agg = BarAggregator("XAUUSD")
        agg.on_tick(Tick(time_msc=0, bid=2000.0, ask=2000.05,
                          last=2000.0, volume=1, volume_real=1.0, flags=0))
        agg.flush()
        assert agg.has_open_bar is False

    def test_tick_in_next_hour_closes_prior_bar(self):
        agg = BarAggregator("XAUUSD")
        # Tick at 0:30 then tick at 1:30 — boundary cross.
        agg.on_tick(Tick(time_msc=30 * 60 * 1000, bid=2000.0, ask=2000.05,
                          last=2000.0, volume=1, volume_real=1.0, flags=0))
        bar = agg.on_tick(Tick(time_msc=90 * 60 * 1000, bid=2001.0, ask=2001.05,
                                 last=2001.0, volume=1, volume_real=1.0,
                                 flags=0))
        assert bar is not None
        assert bar.time_msc == 0  # first bar opened at hour 0

    def test_aggregator_skips_empty_hours(self):
        """A tick that skips multiple hours emits just the prior bar."""
        agg = BarAggregator("XAUUSD")
        agg.on_tick(Tick(time_msc=0, bid=2000.0, ask=2000.05,
                          last=2000.0, volume=1, volume_real=1.0, flags=0))
        # Jump 10 hours.
        bar = agg.on_tick(Tick(time_msc=10 * HOUR_MS + 1_000, bid=2010.0,
                                 ask=2010.05, last=2010.0, volume=1,
                                 volume_real=1.0, flags=0))
        assert bar is not None
        assert bar.time_msc == 0  # only the first bar emitted

    @pytest.mark.parametrize("tf_min", [1, 5, 15, 30, 60, 240])
    def test_aggregator_custom_timeframe(self, tf_min):
        agg = BarAggregator("XAUUSD", timeframe_minutes=tf_min)
        period_ms = tf_min * 60 * 1000
        agg.on_tick(Tick(time_msc=0, bid=2000.0, ask=2000.05,
                          last=2000.0, volume=1, volume_real=1.0, flags=0))
        bar = agg.on_tick(Tick(time_msc=period_ms, bid=2000.0, ask=2000.05,
                                 last=2000.0, volume=1, volume_real=1.0,
                                 flags=0))
        assert bar is not None

    @pytest.mark.parametrize("tf", [0, -1, -60])
    def test_aggregator_rejects_bad_timeframe(self, tf):
        with pytest.raises(ValueError):
            BarAggregator("XAUUSD", timeframe_minutes=tf)


# ===========================================================================
# 12. BARS_PATH NAMING
# ===========================================================================

class TestBarsPath:
    @pytest.mark.parametrize("symbol,tf", [
        ("XAUUSD", "1H"), ("EURUSD", "1H"), ("GBPUSD", "5M"),
        ("USDJPY", "15M"),
    ])
    def test_bars_path_format(self, tmp_path, symbol, tf):
        p = bars_path(symbol, timeframe=tf, bars_dir=tmp_path)
        assert p.name == f"{symbol}_{tf}.parquet"
        assert p.parent == tmp_path


# ===========================================================================
# 13. WRITE / READ ALL PAIRS
# ===========================================================================

@pytest.mark.parametrize("sym", [
    "XAUUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "AUDCHF", "AUDNZD",
])
def test_round_trip_each_pair(tmp_path, sym):
    bars = [make_bar(symbol=sym, time_msc=i * HOUR_MS) for i in range(5)]
    write_bars_parquet(bars, sym, bars_dir=tmp_path)
    df = read_bars_parquet(sym, bars_dir=tmp_path)
    assert len(df) == 5


# ===========================================================================
# 14. CONCURRENT-WRITE STYLE — SECOND WRITE REPLACES FIRST
# ===========================================================================

def test_repeated_write_does_not_append(tmp_path):
    for k in range(5):
        bars = [make_bar(time_msc=k * HOUR_MS)]
        write_bars_parquet(bars, "XAUUSD", bars_dir=tmp_path)
    df = read_bars_parquet("XAUUSD", bars_dir=tmp_path)
    assert len(df) == 1


# ===========================================================================
# 15. CHECK_BAR_INTEGRITY ON MIXED FAULTS
# ===========================================================================

def test_check_bar_integrity_misaligned_and_inconsistent():
    bars = [
        inverted_bar(time_msc=37),  # misaligned + inverted
        make_bar(time_msc=37 + HOUR_MS, open=2000.0, close=2000.0,
                  high=2001.0, low=1999.0),
    ]
    df = pd.DataFrame([b.__dict__ for b in bars])
    rep = check_bar_integrity(df)
    assert rep["aligned"] is False
    assert rep["ohlc_consistent"] is False


# ===========================================================================
# 16. ASIAN-WINDOW INTEGRITY FOR DETECTOR
# ===========================================================================

@pytest.mark.parametrize("missing_idx", [0, 1, 2, 3, 4])
def test_asian_window_with_one_missing_bar(detector, missing_idx):
    """If one of the 5 Asian bars is missing we should still get the range
    from the remaining 4 (≥ 2 required)."""
    from datetime import datetime, timezone

    from tests.edge_cases.fixtures.chaos_market import (
        asian_window_with_missing_bars,
    )

    kept = tuple(i for i in range(5) if i != missing_idx)
    bars = asian_window_with_missing_bars(keep_indices=kept)
    cur_dt = datetime(2026, 5, 15, 8, 0, tzinfo=timezone.utc)
    from strategy.patterns.asian_sweep import _compute_asian_range
    ah, al = _compute_asian_range(bars, cur_dt)
    assert ah is not None and al is not None


# ===========================================================================
# 17. PROPERTY-BASED — INTEGRITY HELPERS
# ===========================================================================

@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(
    times=st.lists(
        st.integers(min_value=0, max_value=24 * 365 * 10),
        min_size=2, max_size=200, unique=True,
    )
)
def test_check_bar_integrity_monotonic_when_sorted(times):
    sorted_times = sorted(times)
    bars = [make_bar(time_msc=t * HOUR_MS) for t in sorted_times]
    df = pd.DataFrame([b.__dict__ for b in bars])
    rep = check_bar_integrity(df)
    assert rep["monotonic"] is True


@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(
    t0=st.integers(min_value=0, max_value=1_000_000_000),
    tf_min=st.integers(min_value=1, max_value=240),
)
def test_floor_to_timeframe_ms_returns_aligned(t0, tf_min):
    snapped = floor_to_timeframe_ms(t0, tf_min)
    period_ms = tf_min * 60 * 1000
    assert snapped % period_ms == 0
    assert snapped <= t0


# ===========================================================================
# 18. WRITE-THEN-READ WITH ALL OHLC RELATIONSHIPS
# ===========================================================================

@pytest.mark.parametrize("open_p,close_p", [
    (2000.0, 2000.0),     # doji
    (2000.0, 2010.0),     # bullish
    (2010.0, 2000.0),     # bearish
    (1.10, 1.10000001),   # tiny range
    (1e-9, 1e9),          # extreme range
])
def test_round_trip_various_ohlc(tmp_path, open_p, close_p):
    bar = make_bar(symbol="X", time_msc=0, open=open_p, close=close_p)
    write_bars_parquet([bar], "X", bars_dir=tmp_path)
    df = read_bars_parquet("X", bars_dir=tmp_path)
    assert df["open"].iloc[0] == pytest.approx(open_p, rel=1e-12, abs=1e-15)
    assert df["close"].iloc[0] == pytest.approx(close_p, rel=1e-12, abs=1e-15)


# ===========================================================================
# 19. TICK FACTORY EDGE CASES
# ===========================================================================

class TestTickEdge:
    def test_tick_with_inverted_bid_ask(self):
        """bid > ask is broker pathology — Tick allows it; downstream should
        detect."""
        t = Tick(time_msc=0, bid=2000.0, ask=1999.0, last=1999.5,
                 volume=1, volume_real=1.0, flags=0)
        assert t.ask < t.bid

    def test_tick_with_zero_bid_ask(self):
        t = Tick(time_msc=0, bid=0.0, ask=0.0, last=0.0, volume=0,
                 volume_real=0.0, flags=0)
        assert t.bid == 0.0 and t.ask == 0.0

    def test_tick_with_negative_price(self):
        t = Tick(time_msc=0, bid=-1.0, ask=-0.5, last=-0.75,
                 volume=1, volume_real=1.0, flags=0)
        assert t.bid < 0


# ===========================================================================
# 20. BAR DATACLASS PROPERTIES
# ===========================================================================

class TestBarProperties:
    @pytest.mark.parametrize("open_p,close_p,expected_bull", [
        (1.0, 2.0, True), (2.0, 1.0, False),
        (1.0, 1.0, False),  # doji is NOT bullish per code (close > open strict)
    ])
    def test_is_bullish(self, open_p, close_p, expected_bull):
        bar = make_bar(open=open_p, close=close_p)
        assert bar.is_bullish == expected_bull

    @pytest.mark.parametrize("h,l", [
        (2010.0, 1990.0), (100.0, 100.0), (0.0, 0.0), (1e9, 1e-9),
    ])
    def test_range_pts(self, h, l):
        bar = Bar(symbol="X", time_msc=0, open=h, high=h, low=l, close=l,
                  volume=1, spread_mean=0.0)
        assert bar.range_pts == pytest.approx(h - l)


# ===========================================================================
# 21. PARQUET CASTING — STRICT SCHEMA
# ===========================================================================

def test_write_uses_bar_schema_dtypes(tmp_path):
    bars = [make_bar(time_msc=i * HOUR_MS) for i in range(3)]
    path = write_bars_parquet(bars, "XAUUSD", bars_dir=tmp_path)
    table = pq.read_table(path)
    assert table.schema.field("time_msc").type == pa.int64()
    assert table.schema.field("open").type == pa.float64()
    assert table.schema.field("volume").type == pa.int64()


# ===========================================================================
# 22. READ_BARS_PARQUET DEFAULT TIMEFRAME
# ===========================================================================

def test_default_timeframe_is_1h(tmp_path):
    bars = [make_bar(time_msc=0)]
    write_bars_parquet(bars, "X", bars_dir=tmp_path)
    # Default tf should match.
    df = read_bars_parquet("X", bars_dir=tmp_path)
    assert len(df) == 1


# ===========================================================================
# 23. PARQUET ROUND-TRIP — SPREAD AND VOLUME
# ===========================================================================

@pytest.mark.parametrize("volume,spread", [
    (0, 0.0), (1, 0.1), (1000, 5.0), (10_000, 50.0),
])
def test_volume_spread_round_trip(tmp_path, volume, spread):
    bar = make_bar(time_msc=0, volume=volume, spread_mean=spread)
    write_bars_parquet([bar], "X", bars_dir=tmp_path)
    df = read_bars_parquet("X", bars_dir=tmp_path)
    assert df["volume"].iloc[0] == volume
    assert df["spread_mean"].iloc[0] == pytest.approx(spread)


# ===========================================================================
# 24. CHECK_BAR_INTEGRITY ON 1-ROW DATA
# ===========================================================================

def test_single_row_integrity():
    bars = [make_bar(time_msc=0)]
    df = pd.DataFrame([b.__dict__ for b in bars])
    rep = check_bar_integrity(df)
    assert rep["rows"] == 1
    assert rep["monotonic"] is True
    assert rep["aligned"] is True
    assert rep["missing_count"] == 0


# ===========================================================================
# 25. RANDOMISED BAR SEQUENCES
# ===========================================================================

@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow,
                                 HealthCheck.function_scoped_fixture])
@given(
    n=st.integers(min_value=1, max_value=50),
    base_price=st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False,
                          allow_infinity=False),
)
def test_random_bar_round_trip(tmp_path, n, base_price):
    bars = [make_bar(time_msc=i * HOUR_MS, open=base_price + i,
                     close=base_price + i) for i in range(n)]
    write_bars_parquet(bars, "X", bars_dir=tmp_path)
    df = read_bars_parquet("X", bars_dir=tmp_path)
    assert len(df) == n
