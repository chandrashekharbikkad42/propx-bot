"""BarAggregator + Bar + parquet I/O — exhaustive unit tests.

This complements the top-level tests/test_bar_aggregator.py with broader
parametrization (8 pairs × scenarios) and hypothesis-driven OHLC property
checks.

Targets:
  - Bar dataclass (frozen, fields, helpers)
  - floor_to_timeframe_ms across boundaries, timeframes, invalid inputs
  - BarAggregator: open/close, OHLC correctness, gap skipping, multi-bar streams
  - Per-pair scenario sweep (8 pairs × multiple scenarios)
  - bars_path + write_bars_parquet + read_bars_parquet round-trip
  - check_bar_integrity edge cases (DST, weekend, gaps, alignment)
  - BAR_SCHEMA shape
  - Hypothesis property tests on OHLC invariants
"""

from __future__ import annotations
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings as hyp_settings, strategies as st, HealthCheck

from data.bar_aggregator import (
    BAR_SCHEMA, Bar, BarAggregator,
    bars_path, check_bar_integrity, floor_to_timeframe_ms,
    read_bars_parquet, write_bars_parquet,
)
from data.tick_collector import Tick
from tests.data.fixtures.synthetic_ticks import (
    EIGHT_PAIRS, base_price_for, spread_for,
    hour_filling_ticks, make_tick, random_walk_ticks, trend_ticks,
    sideways_ticks, utc_ms,
)


# ===========================================================================
# A. Bar dataclass
# ===========================================================================


class TestBarDataclass:

    def test_construction(self):
        b = Bar("EURUSD", 1000, 1.1, 1.2, 1.0, 1.15, 50, 0.0001)
        assert b.symbol == "EURUSD"
        assert b.time_msc == 1000
        assert b.open == 1.1
        assert b.high == 1.2
        assert b.low == 1.0
        assert b.close == 1.15
        assert b.volume == 50
        assert b.spread_mean == 0.0001

    def test_default_spread_mean(self):
        b = Bar("X", 0, 1.0, 1.0, 1.0, 1.0, 1)
        assert b.spread_mean == 0.0

    def test_frozen(self):
        b = Bar("X", 0, 1.0, 1.0, 1.0, 1.0, 1)
        import dataclasses
        with pytest.raises(dataclasses.FrozenInstanceError):
            b.open = 2.0        # type: ignore[misc]

    @pytest.mark.parametrize("open_,close,expected", [
        (1.0, 1.1, True),
        (1.0, 1.0, False),
        (1.1, 1.0, False),
        (1.0, 0.5, False),
        (0.5, 1.5, True),
    ])
    def test_is_bullish(self, open_, close, expected):
        b = Bar("X", 0, open_, max(open_, close), min(open_, close), close, 1)
        assert b.is_bullish is expected

    @pytest.mark.parametrize("high,low,expected_range", [
        (1.2, 1.0, 0.2),
        (1.0, 1.0, 0.0),
        (2300.5, 2299.0, 1.5),
        (150.05, 149.95, 0.10),
    ])
    def test_range_pts(self, high, low, expected_range):
        b = Bar("X", 0, low, high, low, high, 1)
        assert b.range_pts == pytest.approx(expected_range)

    def test_equality_by_value(self):
        a = Bar("EURUSD", 1000, 1.0, 1.1, 0.9, 1.05, 10, 0.0001)
        b = Bar("EURUSD", 1000, 1.0, 1.1, 0.9, 1.05, 10, 0.0001)
        assert a == b

    def test_symbol_difference(self):
        a = Bar("EURUSD", 1000, 1.0, 1.1, 0.9, 1.05, 10)
        b = Bar("GBPUSD", 1000, 1.0, 1.1, 0.9, 1.05, 10)
        assert a != b

    def test_hashable(self):
        b = Bar("X", 0, 1.0, 1.0, 1.0, 1.0, 1)
        assert len({b, b}) == 1

    @pytest.mark.parametrize("sym", EIGHT_PAIRS)
    def test_symbol_field_for_each_pair(self, sym):
        b = Bar(sym, 0, 1.0, 1.0, 1.0, 1.0, 1)
        assert b.symbol == sym


# ===========================================================================
# B. floor_to_timeframe_ms
# ===========================================================================


class TestFloorToTimeframe:

    @pytest.mark.parametrize("h,m,s,want_h", [
        (0, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 59, 59, 0),
        (1, 0, 0, 1), (1, 30, 0, 1),
        (12, 0, 0, 12), (12, 45, 30, 12),
        (23, 59, 59, 23),
    ])
    def test_60min_hour_floor(self, h, m, s, want_h):
        t = utc_ms(2026, 5, 18, h, m, s)
        want = utc_ms(2026, 5, 18, want_h, 0, 0)
        assert floor_to_timeframe_ms(t, 60) == want

    @pytest.mark.parametrize("h,want_4h", [
        (0, 0), (1, 0), (3, 0), (4, 4), (7, 4), (8, 8),
        (11, 8), (12, 12), (15, 12), (16, 16), (19, 16),
        (20, 20), (23, 20),
    ])
    def test_4h_floor(self, h, want_4h):
        t = utc_ms(2026, 5, 18, h, 30)
        want = utc_ms(2026, 5, 18, want_4h, 0)
        assert floor_to_timeframe_ms(t, 240) == want

    @pytest.mark.parametrize("tf_min", [1, 5, 15, 30, 60, 120, 240, 1440])
    def test_idempotent_on_boundary(self, tf_min):
        # A value already on the boundary should map to itself.
        t = (utc_ms(2026, 5, 18, 0, 0) // (tf_min * 60 * 1000)) * (tf_min * 60 * 1000)
        assert floor_to_timeframe_ms(t, tf_min) == t

    @pytest.mark.parametrize("bad", [0, -1, -60, -10**6])
    def test_rejects_non_positive(self, bad):
        with pytest.raises(ValueError):
            floor_to_timeframe_ms(0, bad)

    def test_floor_is_monotone(self):
        a = floor_to_timeframe_ms(utc_ms(2026, 5, 18, 7, 0), 60)
        b = floor_to_timeframe_ms(utc_ms(2026, 5, 18, 8, 0), 60)
        c = floor_to_timeframe_ms(utc_ms(2026, 5, 18, 9, 0), 60)
        assert a < b < c

    @given(st.integers(min_value=0, max_value=10**13),
           st.sampled_from([1, 5, 15, 30, 60, 240, 1440]))
    @hyp_settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_floor_property(self, t, tf):
        out = floor_to_timeframe_ms(t, tf)
        period_ms = tf * 60 * 1000
        assert out % period_ms == 0
        assert out <= t < out + period_ms


# ===========================================================================
# C. BAR_SCHEMA shape
# ===========================================================================


class TestBarSchemaShape:

    def test_field_order(self):
        assert [f.name for f in BAR_SCHEMA] == [
            "time_msc", "open", "high", "low", "close", "volume", "spread_mean",
        ]

    @pytest.mark.parametrize("name,kind", [
        ("time_msc", "int64"),
        ("open", "double"),
        ("high", "double"),
        ("low", "double"),
        ("close", "double"),
        ("volume", "int64"),
        ("spread_mean", "double"),
    ])
    def test_field_types(self, name, kind):
        types = {f.name: str(f.type) for f in BAR_SCHEMA}
        assert types[name] == kind

    def test_seven_fields(self):
        assert len(BAR_SCHEMA) == 7


# ===========================================================================
# D. BarAggregator — basic lifecycle
# ===========================================================================


class TestAggregatorBasics:

    @pytest.mark.parametrize("bad", [0, -1, -60])
    def test_invalid_timeframe(self, bad):
        with pytest.raises(ValueError):
            BarAggregator("X", bad)

    @pytest.mark.parametrize("tf", [1, 5, 15, 30, 60, 240])
    def test_timeframe_stored(self, tf):
        a = BarAggregator("X", tf)
        assert a.timeframe_minutes == tf

    def test_symbol_stored(self):
        a = BarAggregator("EURUSD", 60)
        assert a.symbol == "EURUSD"

    def test_has_no_bar_initially(self):
        a = BarAggregator("X", 60)
        assert a.has_open_bar is False
        assert a.current_bar_open_msc is None

    def test_first_tick_opens_bar_no_emit(self):
        a = BarAggregator("X", 60)
        assert a.on_tick(make_tick(utc_ms(2026, 5, 18, 10, 5), 1.10)) is None
        assert a.has_open_bar is True
        assert a.current_bar_open_msc == utc_ms(2026, 5, 18, 10, 0)

    @pytest.mark.parametrize("minute", [0, 1, 15, 30, 45, 59])
    def test_first_tick_aligns_to_hour(self, minute):
        a = BarAggregator("X", 60)
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 10, minute), 1.10))
        assert a.current_bar_open_msc == utc_ms(2026, 5, 18, 10, 0)

    def test_flush_empty_returns_none(self):
        assert BarAggregator("X", 60).flush() is None

    def test_flush_clears_state(self):
        a = BarAggregator("X", 60)
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        a.flush()
        assert a.has_open_bar is False
        assert a.current_bar_open_msc is None
        assert a.flush() is None


# ===========================================================================
# E. OHLC correctness with controlled inputs
# ===========================================================================


def _agg_drain(symbol, ticks, tf=60) -> List[Bar]:
    a = BarAggregator(symbol, tf)
    bars: List[Bar] = []
    for t in ticks:
        out = a.on_tick(t)
        if out is not None:
            bars.append(out)
    tail = a.flush()
    if tail is not None:
        bars.append(tail)
    return bars


class TestOHLCCorrectness:

    def test_single_tick_bar_open_eq_close_eq_high_eq_low(self):
        a = BarAggregator("X", 60)
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 10, 30), 1.10))
        b = a.flush()
        assert b is not None
        assert b.open == b.high == b.low == b.close

    def test_two_ticks_close_eq_last(self):
        a = BarAggregator("X", 60)
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 10, 30), 1.20))
        b = a.flush()
        assert b is not None
        mid = (1.20 + (1.20 + 0.0002)) / 2.0
        assert b.close == pytest.approx(mid)

    @pytest.mark.parametrize("prices", [
        [1.0, 1.1, 1.05],
        [1.0, 0.9, 0.95],
        [1.0, 2.0, 1.5, 0.5],
        [1.10, 1.10, 1.10],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 5.0, 0.5, 10.0, 0.1],
    ])
    def test_high_is_max_low_is_min(self, prices):
        bars = _agg_drain("X", [
            make_tick(utc_ms(2026, 5, 18, 10, i), p)
            for i, p in enumerate(prices)
        ])
        b = bars[0]
        # bid+spread*0.5 = mid
        mids = [p + 0.0001 for p in prices]
        assert b.high == pytest.approx(max(mids))
        assert b.low == pytest.approx(min(mids))
        assert b.open == pytest.approx(mids[0])
        assert b.close == pytest.approx(mids[-1])

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_volume_equals_tick_count(self, n):
        ticks = [make_tick(utc_ms(2026, 5, 18, 10, 0) + i * 1000, 1.10 + i * 1e-5)
                 for i in range(n)]
        bars = _agg_drain("X", ticks)
        assert bars[0].volume == n

    @pytest.mark.parametrize("spread", [0.0, 0.0001, 0.0002, 0.001, 0.01, 1.0])
    def test_spread_mean_single_tick(self, spread):
        a = BarAggregator("X", 60)
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 10, 0), 1.0, spread=spread))
        b = a.flush()
        assert b is not None
        assert b.spread_mean == pytest.approx(spread)

    def test_spread_mean_average(self):
        a = BarAggregator("X", 60)
        # Spreads 0.0001 and 0.0003 → mean 0.0002
        a.on_tick(Tick(utc_ms(2026, 5, 18, 10, 0), 1.0, 1.0001, 1.0, 1, 1.0, 0))
        a.on_tick(Tick(utc_ms(2026, 5, 18, 10, 30), 1.0, 1.0003, 1.0, 1, 1.0, 0))
        b = a.flush()
        assert b is not None
        assert b.spread_mean == pytest.approx(0.0002)


# ===========================================================================
# F. Boundary cross / multi-bar streams
# ===========================================================================


class TestBoundaryCross:

    def test_emit_on_cross(self):
        a = BarAggregator("X", 60)
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 10, 30), 1.11))
        out = a.on_tick(make_tick(utc_ms(2026, 5, 18, 11, 0), 1.12))
        assert isinstance(out, Bar)
        assert out.time_msc == utc_ms(2026, 5, 18, 10, 0)

    def test_emit_on_exact_minute_boundary(self):
        # Tick at exactly the next hour belongs to the NEXT bar.
        a = BarAggregator("X", 60)
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        out = a.on_tick(make_tick(utc_ms(2026, 5, 18, 11, 0, 0), 1.12))
        assert isinstance(out, Bar)
        assert out.time_msc == utc_ms(2026, 5, 18, 10, 0)
        assert a.current_bar_open_msc == utc_ms(2026, 5, 18, 11, 0)

    @pytest.mark.parametrize("n_hours", [1, 2, 3, 5, 10, 24])
    def test_n_hour_stream_emits_n_bars(self, n_hours):
        # Fill n_hours+1 hours of ticks → emit n_hours bars (last open).
        ticks = hour_filling_ticks(
            utc_ms(2026, 5, 18, 10, 0), n_per_hour=10, n_hours=n_hours + 1,
        )
        bars = _agg_drain("X", ticks)
        assert len(bars) == n_hours + 1   # +1 from flush of the final hour
        # Bars must be in ascending time order
        times = [b.time_msc for b in bars]
        assert times == sorted(times)

    def test_skipped_window_does_not_emit_intermediate(self):
        a = BarAggregator("X", 60)
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        # Jump 5 hours forward.
        out = a.on_tick(make_tick(utc_ms(2026, 5, 18, 15, 5), 1.20))
        assert isinstance(out, Bar)
        assert out.time_msc == utc_ms(2026, 5, 18, 10, 0)
        # The new bar bucket is 15:00.
        assert a.current_bar_open_msc == utc_ms(2026, 5, 18, 15, 0)

    @pytest.mark.parametrize("gap_hours", [1, 2, 5, 10, 24, 72])
    def test_arbitrary_gap_sizes(self, gap_hours):
        a = BarAggregator("X", 60)
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 0, 0), 1.10))
        bar = a.on_tick(make_tick(utc_ms(2026, 5, 18, 0, 0) + gap_hours * 3600 * 1000, 1.20))
        assert isinstance(bar, Bar)
        assert bar.time_msc == utc_ms(2026, 5, 18, 0, 0)


# ===========================================================================
# G. Eight-pair × scenarios sweep (≈ 8 × 40 = 320 tests)
# ===========================================================================


# Scenarios are pure functions that produce ticks. Add new ones here and
# every pair gets coverage automatically.

def _scn_single(pair_base, t0):
    return [make_tick(t0 + 30 * 60 * 1000, pair_base)]


def _scn_two(pair_base, t0):
    return [
        make_tick(t0, pair_base),
        make_tick(t0 + 30 * 60 * 1000, pair_base * 1.001),
    ]


def _scn_bull(pair_base, t0):
    return [
        make_tick(t0 + i * 5 * 60 * 1000, pair_base * (1 + i * 1e-4))
        for i in range(12)
    ]


def _scn_bear(pair_base, t0):
    return [
        make_tick(t0 + i * 5 * 60 * 1000, pair_base * (1 - i * 1e-4))
        for i in range(12)
    ]


def _scn_doji(pair_base, t0):
    return [
        make_tick(t0, pair_base),
        make_tick(t0 + 30 * 60 * 1000, pair_base * 1.0005),
        make_tick(t0 + 50 * 60 * 1000, pair_base * 0.9995),
        make_tick(t0 + 59 * 60 * 1000, pair_base),
    ]


SCENARIOS = {
    "single": _scn_single,
    "two": _scn_two,
    "bull": _scn_bull,
    "bear": _scn_bear,
    "doji": _scn_doji,
}


@pytest.mark.parametrize("pair", EIGHT_PAIRS)
@pytest.mark.parametrize("scn_name", list(SCENARIOS.keys()))
class TestPairScenarioSweep:

    def _bars(self, pair, scn_name):
        scn = SCENARIOS[scn_name]
        base = base_price_for(pair)
        ticks = scn(base, utc_ms(2026, 5, 18, 10, 0))
        # All scenarios fit inside ONE hour — flush to retrieve.
        return _agg_drain(pair, ticks)

    def test_emits_exactly_one_bar(self, pair, scn_name):
        bars = self._bars(pair, scn_name)
        assert len(bars) == 1

    def test_symbol_set_correctly(self, pair, scn_name):
        bars = self._bars(pair, scn_name)
        assert bars[0].symbol == pair

    def test_volume_positive(self, pair, scn_name):
        bars = self._bars(pair, scn_name)
        assert bars[0].volume >= 1

    def test_high_ge_low(self, pair, scn_name):
        bars = self._bars(pair, scn_name)
        assert bars[0].high >= bars[0].low

    def test_high_ge_open_and_close(self, pair, scn_name):
        b = self._bars(pair, scn_name)[0]
        assert b.high >= max(b.open, b.close)

    def test_low_le_open_and_close(self, pair, scn_name):
        b = self._bars(pair, scn_name)[0]
        assert b.low <= min(b.open, b.close)

    def test_time_msc_aligned_to_hour(self, pair, scn_name):
        b = self._bars(pair, scn_name)[0]
        assert b.time_msc % (60 * 60 * 1000) == 0

    def test_spread_mean_nonneg(self, pair, scn_name):
        b = self._bars(pair, scn_name)[0]
        assert b.spread_mean >= 0.0


# ===========================================================================
# H. Hypothesis property tests on OHLC math
# ===========================================================================


_HSET = hyp_settings(max_examples=40, deadline=None,
                     suppress_health_check=[HealthCheck.too_slow,
                                            HealthCheck.function_scoped_fixture])


class TestHypothesisInvariants:

    @_HSET
    @given(prices=st.lists(st.floats(min_value=0.5, max_value=2.0,
                                     allow_nan=False, allow_infinity=False),
                           min_size=1, max_size=50))
    def test_high_is_max_mid(self, prices):
        spread = 0.0002
        ticks = [
            make_tick(utc_ms(2026, 5, 18, 10, 0) + i * 1000, p, spread=spread)
            for i, p in enumerate(prices)
        ]
        b = _agg_drain("X", ticks)[0]
        mids = [p + spread / 2 for p in prices]
        assert b.high == pytest.approx(max(mids))
        assert b.low == pytest.approx(min(mids))

    @_HSET
    @given(prices=st.lists(st.floats(min_value=0.5, max_value=2.0,
                                     allow_nan=False, allow_infinity=False),
                           min_size=1, max_size=50))
    def test_open_eq_first_close_eq_last(self, prices):
        ticks = [make_tick(utc_ms(2026, 5, 18, 10, 0) + i * 1000, p)
                 for i, p in enumerate(prices)]
        b = _agg_drain("X", ticks)[0]
        mids = [p + 0.0001 for p in prices]
        assert b.open == pytest.approx(mids[0])
        assert b.close == pytest.approx(mids[-1])

    @_HSET
    @given(n=st.integers(min_value=1, max_value=200))
    def test_volume_equals_n(self, n):
        ticks = [make_tick(utc_ms(2026, 5, 18, 10, 0) + i * 100, 1.10 + i * 1e-7)
                 for i in range(n)]
        b = _agg_drain("X", ticks)[0]
        assert b.volume == n

    @_HSET
    @given(spreads=st.lists(st.floats(min_value=0.0, max_value=0.01,
                                      allow_nan=False, allow_infinity=False),
                            min_size=1, max_size=30))
    def test_spread_mean_equals_arithmetic_mean(self, spreads):
        ticks = []
        for i, s in enumerate(spreads):
            ticks.append(Tick(utc_ms(2026, 5, 18, 10, 0) + i * 1000,
                              1.0, 1.0 + s, 1.0, 1, 1.0, 0))
        b = _agg_drain("X", ticks)[0]
        assert b.spread_mean == pytest.approx(sum(spreads) / len(spreads), abs=1e-9)


# ===========================================================================
# I. Parquet I/O — bars_path, write_bars_parquet, read_bars_parquet
# ===========================================================================


class TestBarsPath:

    def test_default_path_pattern(self, patch_bars_dir: Path):
        p = bars_path("EURUSD", "1H")
        assert p.parent == patch_bars_dir
        assert p.name == "EURUSD_1H.parquet"

    def test_override_dir(self, tmp_path):
        p = bars_path("EURUSD", "1H", bars_dir=tmp_path)
        assert p == tmp_path / "EURUSD_1H.parquet"

    @pytest.mark.parametrize("tf", ["1H", "4H", "1D", "5M", "15M", "30M"])
    def test_timeframe_in_filename(self, tf, patch_bars_dir):
        assert bars_path("EURUSD", tf).name == f"EURUSD_{tf}.parquet"

    @pytest.mark.parametrize("sym", EIGHT_PAIRS)
    def test_symbol_in_filename(self, sym, patch_bars_dir):
        assert bars_path(sym, "1H").name == f"{sym}_1H.parquet"


class TestWriteReadParquet:

    def _sample_bars(self) -> List[Bar]:
        t0 = utc_ms(2026, 5, 18, 10, 0)
        return [
            Bar("EURUSD", t0 + i * 3600 * 1000,
                1.10 + i * 0.001, 1.11 + i * 0.001,
                1.09 + i * 0.001, 1.105 + i * 0.001,
                100 + i, 0.0002 + i * 1e-5)
            for i in range(5)
        ]

    def test_round_trip_preserves_count(self, tmp_path):
        bars = self._sample_bars()
        write_bars_parquet(bars, "EURUSD", "1H", bars_dir=tmp_path)
        df = read_bars_parquet("EURUSD", "1H", bars_dir=tmp_path)
        assert len(df) == len(bars)

    def test_round_trip_columns(self, tmp_path):
        bars = self._sample_bars()
        write_bars_parquet(bars, "EURUSD", "1H", bars_dir=tmp_path)
        df = read_bars_parquet("EURUSD", "1H", bars_dir=tmp_path)
        assert list(df.columns) == [
            "time_msc", "open", "high", "low", "close", "volume", "spread_mean",
        ]

    @pytest.mark.parametrize("field", ["open", "high", "low", "close", "spread_mean"])
    def test_round_trip_float_field(self, field, tmp_path):
        bars = self._sample_bars()
        write_bars_parquet(bars, "EURUSD", "1H", bars_dir=tmp_path)
        df = read_bars_parquet("EURUSD", "1H", bars_dir=tmp_path)
        for i, b in enumerate(bars):
            assert df.iloc[i][field] == pytest.approx(getattr(b, field))

    def test_round_trip_time_msc(self, tmp_path):
        bars = self._sample_bars()
        write_bars_parquet(bars, "EURUSD", "1H", bars_dir=tmp_path)
        df = read_bars_parquet("EURUSD", "1H", bars_dir=tmp_path)
        assert list(df["time_msc"]) == [b.time_msc for b in bars]

    def test_round_trip_volume(self, tmp_path):
        bars = self._sample_bars()
        write_bars_parquet(bars, "EURUSD", "1H", bars_dir=tmp_path)
        df = read_bars_parquet("EURUSD", "1H", bars_dir=tmp_path)
        assert list(df["volume"]) == [b.volume for b in bars]

    def test_write_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "deep" / "subdir"
        write_bars_parquet(self._sample_bars(), "EURUSD", "1H", bars_dir=nested)
        assert (nested / "EURUSD_1H.parquet").exists()

    def test_write_overwrites_existing(self, tmp_path):
        write_bars_parquet(self._sample_bars(), "EURUSD", "1H", bars_dir=tmp_path)
        # Now write fewer bars — should fully overwrite.
        write_bars_parquet([self._sample_bars()[0]], "EURUSD", "1H", bars_dir=tmp_path)
        df = read_bars_parquet("EURUSD", "1H", bars_dir=tmp_path)
        assert len(df) == 1

    def test_read_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_bars_parquet("NEVER", "1H", bars_dir=tmp_path)

    def test_empty_bars_writes_empty_file(self, tmp_path):
        p = write_bars_parquet([], "EURUSD", "1H", bars_dir=tmp_path)
        assert p.exists()
        df = read_bars_parquet("EURUSD", "1H", bars_dir=tmp_path)
        assert len(df) == 0

    def test_read_sorts_ascending(self, tmp_path):
        t0 = utc_ms(2026, 5, 18, 10, 0)
        # Write out-of-order.
        bars = [
            Bar("X", t0 + 2 * 3600_000, 1.0, 1.1, 0.9, 1.05, 1),
            Bar("X", t0, 1.0, 1.1, 0.9, 1.05, 1),
            Bar("X", t0 + 1 * 3600_000, 1.0, 1.1, 0.9, 1.05, 1),
        ]
        write_bars_parquet(bars, "X", "1H", bars_dir=tmp_path)
        df = read_bars_parquet("X", "1H", bars_dir=tmp_path)
        # read_bars_parquet sorts ascending.
        assert list(df["time_msc"]) == sorted(b.time_msc for b in bars)

    @pytest.mark.parametrize("compression", ["snappy", "gzip", "zstd"])
    def test_compression_codecs(self, compression, tmp_path):
        write_bars_parquet(self._sample_bars(), "EURUSD", "1H",
                           bars_dir=tmp_path, compression=compression)
        df = read_bars_parquet("EURUSD", "1H", bars_dir=tmp_path)
        assert len(df) == 5

    def test_schema_on_disk(self, tmp_path):
        write_bars_parquet(self._sample_bars(), "EURUSD", "1H", bars_dir=tmp_path)
        p = tmp_path / "EURUSD_1H.parquet"
        on_disk = pq.ParquetFile(p).schema_arrow
        assert on_disk.equals(BAR_SCHEMA)


# ===========================================================================
# J. check_bar_integrity
# ===========================================================================


def _df_from_bars(bars: List[Bar]) -> pd.DataFrame:
    return pd.DataFrame({
        "time_msc": [b.time_msc for b in bars],
        "open": [b.open for b in bars],
        "high": [b.high for b in bars],
        "low": [b.low for b in bars],
        "close": [b.close for b in bars],
        "volume": [b.volume for b in bars],
        "spread_mean": [b.spread_mean for b in bars],
    })


class TestCheckBarIntegrity:

    def test_empty_df(self):
        r = check_bar_integrity(pd.DataFrame())
        assert r["rows"] == 0
        assert r["monotonic"] is True
        assert r["aligned"] is True
        assert r["missing_count"] == 0
        assert r["ohlc_consistent"] is True

    @pytest.mark.parametrize("n", [1, 2, 5, 24, 100])
    def test_clean_n_consecutive(self, n):
        t0 = utc_ms(2026, 5, 18, 0, 0)
        bars = [
            Bar("X", t0 + h * 3600_000, 1.0, 1.1, 0.9, 1.05, 10, 0.0001)
            for h in range(n)
        ]
        r = check_bar_integrity(_df_from_bars(bars))
        assert r["rows"] == n
        assert r["monotonic"] is True
        assert r["aligned"] is True
        assert r["missing_count"] == 0
        assert r["ohlc_consistent"] is True

    @pytest.mark.parametrize("gap_size", [1, 2, 5, 10, 23])
    def test_detect_gap(self, gap_size):
        t0 = utc_ms(2026, 5, 18, 0, 0)
        bars = [Bar("X", t0, 1.0, 1.1, 0.9, 1.05, 10),
                Bar("X", t0 + (gap_size + 1) * 3600_000, 1.0, 1.1, 0.9, 1.05, 10)]
        r = check_bar_integrity(_df_from_bars(bars))
        assert r["missing_count"] == gap_size

    @pytest.mark.parametrize("misalign_min", [1, 5, 30, 59])
    def test_detect_misalignment(self, misalign_min):
        t = utc_ms(2026, 5, 18, 10, misalign_min)
        bars = [Bar("X", t, 1.0, 1.0, 1.0, 1.0, 1)]
        r = check_bar_integrity(_df_from_bars(bars))
        assert r["aligned"] is False

    def test_detect_non_monotonic(self):
        t0 = utc_ms(2026, 5, 18, 10, 0)
        df = pd.DataFrame({
            "time_msc": [t0, t0 + 3600_000, t0],
            "open": [1.0, 1.0, 1.0],
            "high": [1.1, 1.1, 1.1],
            "low": [0.9, 0.9, 0.9],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
            "spread_mean": [0.0, 0.0, 0.0],
        })
        r = check_bar_integrity(df)
        assert r["monotonic"] is False

    def test_detect_ohlc_inconsistent_low_above_open(self):
        t = utc_ms(2026, 5, 18, 10, 0)
        df = _df_from_bars([Bar("X", t, 1.0, 1.1, 1.05, 0.95, 1)])
        r = check_bar_integrity(df)
        assert r["ohlc_consistent"] is False

    def test_detect_ohlc_inconsistent_high_below_close(self):
        t = utc_ms(2026, 5, 18, 10, 0)
        df = _df_from_bars([Bar("X", t, 1.0, 1.05, 0.9, 1.1, 1)])
        r = check_bar_integrity(df)
        assert r["ohlc_consistent"] is False

    @pytest.mark.parametrize("tf", [15, 30, 60, 240])
    def test_different_timeframes(self, tf):
        t0 = utc_ms(2026, 5, 18, 0, 0)
        step = tf * 60 * 1000
        bars = [Bar("X", t0 + i * step, 1.0, 1.1, 0.9, 1.05, 1) for i in range(5)]
        r = check_bar_integrity(_df_from_bars(bars), timeframe_minutes=tf)
        assert r["aligned"] is True
        assert r["missing_count"] == 0


# ===========================================================================
# K. DST / weekend gap behaviour (UTC bars don't shift, but tests verify it)
# ===========================================================================


class TestDstWeekendBehaviour:

    def test_dst_spring_forward_no_skip_in_utc(self):
        # 2026-03-08 US DST. UTC bars should still be consecutive — Bar.time_msc
        # is UTC, so no skip is expected.
        t0 = utc_ms(2026, 3, 8, 6, 0)        # 1am EST = 6am UTC
        bars = [Bar("X", t0 + h * 3600_000, 1.0, 1.1, 0.9, 1.05, 1) for h in range(6)]
        r = check_bar_integrity(_df_from_bars(bars))
        assert r["missing_count"] == 0
        assert r["aligned"] is True

    def test_weekend_gap_detected(self):
        # Friday 21:00 UTC → Sunday 22:00 UTC = 49 hours gap → 48 missing bars.
        fri = utc_ms(2026, 5, 22, 21, 0)
        sun = utc_ms(2026, 5, 24, 22, 0)
        bars = [Bar("X", fri, 1.0, 1.1, 0.9, 1.05, 1),
                Bar("X", sun, 1.0, 1.1, 0.9, 1.05, 1)]
        r = check_bar_integrity(_df_from_bars(bars))
        assert r["missing_count"] == 48


# ===========================================================================
# L. Random-walk integration — aggregator → parquet → integrity
# ===========================================================================


class TestRandomWalkRoundTrip:

    def test_random_walk_three_hours(self, tmp_path):
        ticks = (
            hour_filling_ticks(utc_ms(2026, 5, 18, 10, 0), n_per_hour=50, n_hours=4)
        )
        bars = _agg_drain("EURUSD", ticks)
        # Drop the in-progress flush bar to keep the integrity clean across hours.
        bars_clean = bars[:-1]
        write_bars_parquet(bars_clean, "EURUSD", "1H", bars_dir=tmp_path)
        df = read_bars_parquet("EURUSD", "1H", bars_dir=tmp_path)
        r = check_bar_integrity(df)
        assert r["aligned"] is True
        assert r["monotonic"] is True


# ===========================================================================
# M. State introspection between bars
# ===========================================================================


class TestState:

    def test_current_bar_open_msc_advances(self):
        a = BarAggregator("X", 60)
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        assert a.current_bar_open_msc == utc_ms(2026, 5, 18, 10, 0)
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 11, 0), 1.11))
        assert a.current_bar_open_msc == utc_ms(2026, 5, 18, 11, 0)

    def test_has_open_bar_toggles_with_flush(self):
        a = BarAggregator("X", 60)
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        assert a.has_open_bar is True
        a.flush()
        assert a.has_open_bar is False

    def test_flush_after_emit_returns_in_progress(self):
        a = BarAggregator("X", 60)
        a.on_tick(make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        emitted = a.on_tick(make_tick(utc_ms(2026, 5, 18, 11, 0), 1.20))
        flushed = a.flush()
        assert emitted is not None and emitted.time_msc == utc_ms(2026, 5, 18, 10, 0)
        assert flushed is not None and flushed.time_msc == utc_ms(2026, 5, 18, 11, 0)


# ===========================================================================
# N. Timeframe parametrization
# ===========================================================================


@pytest.mark.parametrize("tf", [1, 5, 15, 30, 60, 120, 240])
def test_aggregator_works_for_timeframe(tf):
    a = BarAggregator("X", tf)
    period_ms = tf * 60 * 1000
    t0 = (utc_ms(2026, 5, 18, 10, 0) // period_ms) * period_ms
    a.on_tick(make_tick(t0, 1.10))
    a.on_tick(make_tick(t0 + period_ms // 2, 1.11))
    # Cross into next bar.
    out = a.on_tick(make_tick(t0 + period_ms + 1, 1.12))
    assert isinstance(out, Bar)
    assert out.time_msc == t0
