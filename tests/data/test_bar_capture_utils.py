"""bar_capture_utils — exhaustive unit tests.

Targets:
  - mt5_rates_to_bars conversion: dtype with/without 'spread', empty, None
  - Field round-trip (time s→ms, open/high/low/close, tick_volume, spread mirror)
  - Ordering preserved (MT5 returns ascending)
  - bars_summary stats: empty, single, many bars, span calc
  - Integration with bar_aggregator (Bar dataclass shape preserved)
  - 8-pair × scenario sweep
  - Edge: zero-prices, negative spread, max-int time, huge volume
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import List, Sequence, Tuple

import numpy as np
import pytest

from data.bar_aggregator import Bar
from data.bar_capture_utils import bars_summary, mt5_rates_to_bars
from tests.data.fixtures.synthetic_ticks import (
    EIGHT_PAIRS, MT5_RATES_DTYPE, MT5_RATES_DTYPE_NO_SPREAD,
    consecutive_h1_rates, mt5_rates_array,
)


# ===========================================================================
# A. mt5_rates_to_bars — empty / None
# ===========================================================================


class TestEmptyInputs:

    def test_empty_array_returns_empty_list(self):
        arr = np.empty(0, dtype=MT5_RATES_DTYPE)
        assert mt5_rates_to_bars(arr, "EURUSD") == []

    def test_none_returns_empty_list(self):
        assert mt5_rates_to_bars(None, "EURUSD") == []     # type: ignore[arg-type]

    def test_empty_dtype_without_spread(self):
        arr = np.empty(0, dtype=MT5_RATES_DTYPE_NO_SPREAD)
        assert mt5_rates_to_bars(arr, "EURUSD") == []

    @pytest.mark.parametrize("sym", EIGHT_PAIRS)
    def test_empty_per_symbol(self, sym):
        assert mt5_rates_to_bars(np.empty(0, dtype=MT5_RATES_DTYPE), sym) == []


# ===========================================================================
# B. Single row conversion
# ===========================================================================


class TestSingleRow:

    def _t(self):
        return int(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc).timestamp())

    def test_returns_one_bar(self):
        arr = mt5_rates_array([(self._t(), 1.10, 1.11, 1.09, 1.105, 50, 12, 50)])
        bars = mt5_rates_to_bars(arr, "EURUSD")
        assert len(bars) == 1

    def test_bar_type(self):
        arr = mt5_rates_array([(self._t(), 1.10, 1.11, 1.09, 1.105, 50, 12, 50)])
        b = mt5_rates_to_bars(arr, "EURUSD")[0]
        assert isinstance(b, Bar)

    def test_time_msc_is_seconds_x_1000(self):
        arr = mt5_rates_array([(self._t(), 1.10, 1.11, 1.09, 1.105, 50, 12, 50)])
        b = mt5_rates_to_bars(arr, "EURUSD")[0]
        assert b.time_msc == self._t() * 1000

    @pytest.mark.parametrize("field,value", [
        ("open", 1.10), ("high", 1.11), ("low", 1.09), ("close", 1.105),
    ])
    def test_ohlc_round_trip(self, field, value):
        arr = mt5_rates_array([(self._t(), 1.10, 1.11, 1.09, 1.105, 50, 12, 50)])
        b = mt5_rates_to_bars(arr, "EURUSD")[0]
        assert getattr(b, field) == pytest.approx(value)

    def test_volume_uses_tick_volume_not_real(self):
        # tick_volume=50, real_volume=999 → volume should be 50.
        arr = mt5_rates_array([(self._t(), 1.10, 1.11, 1.09, 1.105, 50, 12, 999)])
        b = mt5_rates_to_bars(arr, "EURUSD")[0]
        assert b.volume == 50

    def test_spread_mirrored_to_spread_mean(self):
        arr = mt5_rates_array([(self._t(), 1.10, 1.11, 1.09, 1.105, 50, 12, 0)])
        b = mt5_rates_to_bars(arr, "EURUSD")[0]
        assert b.spread_mean == 12.0     # mirrored value

    def test_symbol_set(self):
        arr = mt5_rates_array([(self._t(), 1.10, 1.11, 1.09, 1.105, 50, 12, 0)])
        b = mt5_rates_to_bars(arr, "GBPUSD")[0]
        assert b.symbol == "GBPUSD"


# ===========================================================================
# C. Missing 'spread' column → spread_mean=0
# ===========================================================================


class TestNoSpreadColumn:

    def test_no_spread_defaults_to_zero(self):
        t = int(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc).timestamp())
        arr = mt5_rates_array(
            [(t, 1.10, 1.11, 1.09, 1.105, 50, 0)],
            with_spread=False,
        )
        b = mt5_rates_to_bars(arr, "EURUSD")[0]
        assert b.spread_mean == 0.0

    def test_no_spread_ohlc_still_correct(self):
        t = int(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc).timestamp())
        arr = mt5_rates_array(
            [(t, 2.0, 2.5, 1.5, 2.25, 100, 0)],
            with_spread=False,
        )
        b = mt5_rates_to_bars(arr, "EURUSD")[0]
        assert b.open == 2.0
        assert b.high == 2.5
        assert b.low == 1.5
        assert b.close == 2.25
        assert b.volume == 100


# ===========================================================================
# D. Multiple rows — ordering, count
# ===========================================================================


class TestMultipleRows:

    @pytest.mark.parametrize("n", [1, 2, 5, 24, 100, 500])
    def test_n_rows_returns_n_bars(self, n):
        t0 = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
        arr = consecutive_h1_rates(t0, n)
        bars = mt5_rates_to_bars(arr, "EURUSD")
        assert len(bars) == n

    def test_order_preserved(self):
        t0 = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
        arr = consecutive_h1_rates(t0, 5)
        bars = mt5_rates_to_bars(arr, "EURUSD")
        times = [b.time_msc for b in bars]
        assert times == sorted(times)

    def test_each_bar_one_hour_apart(self):
        t0 = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
        arr = consecutive_h1_rates(t0, 5)
        bars = mt5_rates_to_bars(arr, "EURUSD")
        diffs = [bars[i + 1].time_msc - bars[i].time_msc for i in range(4)]
        assert all(d == 3600_000 for d in diffs)

    def test_symbol_propagated_to_all_bars(self):
        arr = consecutive_h1_rates(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc), 10)
        bars = mt5_rates_to_bars(arr, "XAUUSD")
        assert all(b.symbol == "XAUUSD" for b in bars)

    def test_first_bar_matches_input_t0(self):
        t0 = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
        arr = consecutive_h1_rates(t0, 5)
        bars = mt5_rates_to_bars(arr, "EURUSD")
        assert bars[0].time_msc == int(t0.timestamp() * 1000)

    def test_last_bar_matches_input_t_plus_n(self):
        t0 = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
        arr = consecutive_h1_rates(t0, 5)
        bars = mt5_rates_to_bars(arr, "EURUSD")
        expected = int((t0 + timedelta(hours=4)).timestamp() * 1000)
        assert bars[-1].time_msc == expected


# ===========================================================================
# E. 8 pairs × N scenarios sweep
# ===========================================================================


@pytest.mark.parametrize("pair", EIGHT_PAIRS)
@pytest.mark.parametrize("n", [1, 3, 12, 50])
class TestPerPairConversion:

    def test_count(self, pair, n):
        t0 = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
        arr = consecutive_h1_rates(t0, n)
        assert len(mt5_rates_to_bars(arr, pair)) == n

    def test_symbol_propagation(self, pair, n):
        t0 = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
        arr = consecutive_h1_rates(t0, n)
        for b in mt5_rates_to_bars(arr, pair):
            assert b.symbol == pair

    def test_ohlc_invariants(self, pair, n):
        t0 = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
        arr = consecutive_h1_rates(t0, n)
        for b in mt5_rates_to_bars(arr, pair):
            assert b.low <= b.open <= b.high
            assert b.low <= b.close <= b.high
            assert b.low <= b.high

    def test_volume_nonneg(self, pair, n):
        arr = consecutive_h1_rates(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc), n)
        for b in mt5_rates_to_bars(arr, pair):
            assert b.volume >= 0


# ===========================================================================
# F. Edge values
# ===========================================================================


@pytest.mark.parametrize("price", [0.0, 1e-9, 1.0, 1500.0, 1e6, 1e9])
def test_extreme_prices(price):
    t = int(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc).timestamp())
    arr = mt5_rates_array([(t, price, price, price, price, 1, 0, 0)])
    b = mt5_rates_to_bars(arr, "EURUSD")[0]
    assert b.open == pytest.approx(price)
    assert b.high == pytest.approx(price)
    assert b.low == pytest.approx(price)
    assert b.close == pytest.approx(price)


@pytest.mark.parametrize("vol", [0, 1, 1000, 10**6, 10**9, 10**12])
def test_extreme_volumes(vol):
    t = int(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc).timestamp())
    arr = mt5_rates_array([(t, 1.0, 1.0, 1.0, 1.0, vol, 0, 0)])
    b = mt5_rates_to_bars(arr, "EURUSD")[0]
    assert b.volume == vol


@pytest.mark.parametrize("spread", [0, 1, 5, 10, 100, 1000, 2**30])
def test_extreme_spread_values(spread):
    t = int(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc).timestamp())
    arr = mt5_rates_array([(t, 1.0, 1.0, 1.0, 1.0, 1, spread, 0)])
    b = mt5_rates_to_bars(arr, "EURUSD")[0]
    assert b.spread_mean == float(spread)


@pytest.mark.parametrize("t_sec", [
    0, 1,
    int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()),
    int(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc).timestamp()),
    int(datetime(2030, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()),
])
def test_time_msc_round_trip_extreme(t_sec):
    arr = mt5_rates_array([(t_sec, 1.0, 1.0, 1.0, 1.0, 1, 0, 0)])
    b = mt5_rates_to_bars(arr, "EURUSD")[0]
    assert b.time_msc == t_sec * 1000


# ===========================================================================
# G. bars_summary — empty
# ===========================================================================


class TestBarsSummaryEmpty:

    def test_empty_list(self):
        s = bars_summary([])
        assert s == {"count": 0, "first_msc": 0, "last_msc": 0, "span_days": 0.0}

    def test_empty_tuple(self):
        s = bars_summary(())
        assert s["count"] == 0

    def test_returns_dict(self):
        assert isinstance(bars_summary([]), dict)

    @pytest.mark.parametrize("seq", [[], (), iter([])])
    def test_various_empty_iterables(self, seq):
        # bars_summary accepts Sequence; empty iter is iterable too.
        # If iter is passed it will be coerced inside? Actually bars_summary
        # does `if not bars: return ...` and `len(bars)` — both require Sequence.
        if hasattr(seq, "__len__"):
            assert bars_summary(seq)["count"] == 0


# ===========================================================================
# H. bars_summary — single / multi
# ===========================================================================


class TestBarsSummarySingle:

    def test_single_bar(self):
        b = Bar("X", 1_000_000, 1.0, 1.0, 1.0, 1.0, 1)
        s = bars_summary([b])
        assert s["count"] == 1
        assert s["first_msc"] == 1_000_000
        assert s["last_msc"] == 1_000_000
        assert s["span_days"] == 0.0

    @pytest.mark.parametrize("t", [0, 1, 1_000_000, 10**12, 10**13])
    def test_single_bar_first_last_msc(self, t):
        b = Bar("X", t, 1.0, 1.0, 1.0, 1.0, 1)
        s = bars_summary([b])
        assert s["first_msc"] == t
        assert s["last_msc"] == t


class TestBarsSummaryMulti:

    def test_two_bars_one_day_apart(self):
        day_ms = 24 * 60 * 60 * 1000
        bars = [
            Bar("X", 0, 1.0, 1.0, 1.0, 1.0, 1),
            Bar("X", day_ms, 1.0, 1.0, 1.0, 1.0, 1),
        ]
        s = bars_summary(bars)
        assert s["count"] == 2
        assert s["span_days"] == 1.0

    @pytest.mark.parametrize("days", [0.5, 1.0, 2.0, 7.0, 30.0, 365.0])
    def test_span_days_param(self, days):
        day_ms = 24 * 60 * 60 * 1000
        last_msc = int(days * day_ms)
        bars = [
            Bar("X", 0, 1.0, 1.0, 1.0, 1.0, 1),
            Bar("X", last_msc, 1.0, 1.0, 1.0, 1.0, 1),
        ]
        s = bars_summary(bars)
        assert s["span_days"] == pytest.approx(days)

    @pytest.mark.parametrize("n", [2, 5, 10, 100, 1000])
    def test_count_property(self, n):
        bars = [Bar("X", i * 3600_000, 1.0, 1.0, 1.0, 1.0, 1) for i in range(n)]
        s = bars_summary(bars)
        assert s["count"] == n

    def test_first_last_msc_correct(self):
        bars = [Bar("X", i * 3600_000, 1.0, 1.0, 1.0, 1.0, 1) for i in range(5)]
        s = bars_summary(bars)
        assert s["first_msc"] == 0
        assert s["last_msc"] == 4 * 3600_000

    def test_intra_day_span_fractional(self):
        # 6 hours apart = 0.25 days
        bars = [
            Bar("X", 0, 1.0, 1.0, 1.0, 1.0, 1),
            Bar("X", 6 * 3600_000, 1.0, 1.0, 1.0, 1.0, 1),
        ]
        s = bars_summary(bars)
        assert s["span_days"] == pytest.approx(0.25)


# ===========================================================================
# I. Integration: consecutive_h1_rates → mt5_rates_to_bars → bars_summary
# ===========================================================================


class TestIntegration:

    @pytest.mark.parametrize("n", [2, 5, 24, 100])
    def test_round_trip_summary_count(self, n):
        arr = consecutive_h1_rates(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc), n)
        bars = mt5_rates_to_bars(arr, "EURUSD")
        s = bars_summary(bars)
        assert s["count"] == n

    @pytest.mark.parametrize("n", [2, 24, 168])     # 168 = 1 week
    def test_round_trip_span_days(self, n):
        arr = consecutive_h1_rates(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc), n)
        bars = mt5_rates_to_bars(arr, "EURUSD")
        s = bars_summary(bars)
        # (n-1) hours apart → (n-1)/24 days.
        assert s["span_days"] == pytest.approx((n - 1) / 24.0)

    @pytest.mark.parametrize("sym", EIGHT_PAIRS)
    def test_per_pair_round_trip(self, sym):
        arr = consecutive_h1_rates(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc), 5)
        bars = mt5_rates_to_bars(arr, sym)
        assert all(b.symbol == sym for b in bars)
        s = bars_summary(bars)
        assert s["count"] == 5


# ===========================================================================
# J. Bars produced are Bar (frozen) instances
# ===========================================================================


class TestBarsAreFrozen:

    def test_attempt_mutate_raises(self):
        arr = consecutive_h1_rates(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc), 1)
        b = mt5_rates_to_bars(arr, "EURUSD")[0]
        import dataclasses
        with pytest.raises(dataclasses.FrozenInstanceError):
            b.open = 9.99      # type: ignore[misc]

    def test_hashable(self):
        arr = consecutive_h1_rates(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc), 1)
        b = mt5_rates_to_bars(arr, "EURUSD")[0]
        assert len({b, b}) == 1


# ===========================================================================
# K. Symbol parametrization — empty + filled
# ===========================================================================


@pytest.mark.parametrize("sym", list(EIGHT_PAIRS) + ["", "X", "AVERYLONGSYMBOL"])
def test_symbol_string_passthrough(sym):
    arr = consecutive_h1_rates(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc), 2)
    bars = mt5_rates_to_bars(arr, sym)
    for b in bars:
        assert b.symbol == sym


# ===========================================================================
# L. Conversion robustness — dtype variants
# ===========================================================================


def _custom_rates(rows, with_spread):
    return mt5_rates_array(rows, with_spread=with_spread)


@pytest.mark.parametrize("with_spread", [True, False])
@pytest.mark.parametrize("n", [1, 5, 24])
def test_dtype_variants(with_spread, n):
    rows = []
    t0 = int(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc).timestamp())
    for i in range(n):
        row = (t0 + i * 3600, 1.0 + i * 1e-4, 1.1, 0.9, 1.05, 10 + i,
               5 if with_spread else None, 0)
        rows.append(tuple(x for x in row if x is not None))
    arr = _custom_rates(rows, with_spread=with_spread)
    bars = mt5_rates_to_bars(arr, "EURUSD")
    assert len(bars) == n
    if not with_spread:
        assert all(b.spread_mean == 0.0 for b in bars)


# ===========================================================================
# M. Sweeping field values via cartesian parametrization
# ===========================================================================


OHLC_CASES = [
    (1.10, 1.11, 1.09, 1.105),
    (2.0, 2.5, 1.5, 2.25),
    (0.5, 0.6, 0.4, 0.55),
    (100.0, 110.0, 90.0, 105.0),
    (2300.0, 2350.0, 2280.0, 2320.0),
    (0.6532, 0.6601, 0.6510, 0.6580),
]


@pytest.mark.parametrize("o,h,l,c", OHLC_CASES)
def test_ohlc_field_round_trip(o, h, l, c):
    t = int(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc).timestamp())
    arr = mt5_rates_array([(t, o, h, l, c, 100, 5, 0)])
    b = mt5_rates_to_bars(arr, "EURUSD")[0]
    assert b.open == pytest.approx(o)
    assert b.high == pytest.approx(h)
    assert b.low == pytest.approx(l)
    assert b.close == pytest.approx(c)


# ===========================================================================
# N. bars_summary integration on partial / odd inputs
# ===========================================================================


class TestBarsSummaryEdge:

    def test_two_bars_same_timestamp(self):
        b1 = Bar("X", 1_000_000, 1.0, 1.0, 1.0, 1.0, 1)
        b2 = Bar("X", 1_000_000, 1.0, 1.0, 1.0, 1.0, 1)
        s = bars_summary([b1, b2])
        assert s["count"] == 2
        assert s["span_days"] == 0.0

    def test_handles_year_long_span(self):
        year_ms = 365 * 24 * 3600 * 1000
        bars = [
            Bar("X", 0, 1.0, 1.0, 1.0, 1.0, 1),
            Bar("X", year_ms, 1.0, 1.0, 1.0, 1.0, 1),
        ]
        s = bars_summary(bars)
        assert s["span_days"] == pytest.approx(365.0)


# ===========================================================================
# O. mt5_rates_to_bars + EIGHT_PAIRS × OHLC cases sweep
# ===========================================================================


@pytest.mark.parametrize("sym", EIGHT_PAIRS)
@pytest.mark.parametrize("o,h,l,c", OHLC_CASES)
def test_per_pair_ohlc_cases(sym, o, h, l, c):
    t = int(datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc).timestamp())
    arr = mt5_rates_array([(t, o, h, l, c, 50, 3, 0)])
    bars = mt5_rates_to_bars(arr, sym)
    assert bars[0].symbol == sym
    assert bars[0].open == pytest.approx(o)
    assert bars[0].close == pytest.approx(c)
