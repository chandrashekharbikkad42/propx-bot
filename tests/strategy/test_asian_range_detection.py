"""Asian range computation + min/max-range filter tests.

Covers `_compute_asian_range` directly plus the detector's range-filter
behaviour. Window = prev_day [19:30, 00:30) UTC, inclusive lower /
exclusive upper, on bar OPEN time. ≥ 2 bars required.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone

import pytest

from config.asian_sweep_config import PAIR_CONFIG, PAIRS
from strategy.patterns.asian_sweep import (
    AsianSweepDetector,
    _compute_asian_range,
)
from strategy.patterns.base import MarketContext

from tests.strategy.fixtures.synthetic_bars import (
    baseline_low, build_scenario, hour_msc, long_sweep_bars, make_bar,
)

UTC = timezone.utc
ALL_PAIRS = list(PAIRS)


def _cur_dt(year=2026, month=4, day=15, hour=8):
    return datetime(year, month, day, hour, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# 1. Window inclusion / exclusion
# ---------------------------------------------------------------------------

class TestAsianWindowBoundaries:
    def test_bar_at_1900_prev_excluded(self):
        """19:00 prev day < 19:30 → excluded."""
        cur_dt = _cur_dt()
        prev = cur_dt.date() - timedelta(days=1)
        bars = [
            make_bar(time_msc=hour_msc(prev.year, prev.month, prev.day, 19),
                     high=1.5, low=1.0, close=1.2),
            make_bar(time_msc=hour_msc(prev.year, prev.month, prev.day, 20),
                     high=1.1, low=1.05, close=1.08),
            make_bar(time_msc=hour_msc(prev.year, prev.month, prev.day, 21),
                     high=1.1, low=1.05, close=1.08),
        ]
        ah, al = _compute_asian_range(bars, cur_dt)
        # 19:00 bar (h=1.5, l=1.0) must NOT contribute.
        assert ah == 1.1
        assert al == 1.05

    def test_bar_at_2000_prev_included(self):
        cur_dt = _cur_dt()
        prev = cur_dt.date() - timedelta(days=1)
        bars = [
            make_bar(time_msc=hour_msc(prev.year, prev.month, prev.day, 20),
                     high=2.0, low=1.5, close=1.7),
            make_bar(time_msc=hour_msc(prev.year, prev.month, prev.day, 21),
                     high=1.6, low=1.55, close=1.58),
        ]
        ah, al = _compute_asian_range(bars, cur_dt)
        assert ah == 2.0

    def test_bar_at_0030_cur_excluded(self):
        """Bar at 00:30 has no valid time_msc (bars align to hour), but a
        00:00 bar with time_msc == 00:00 is included; 01:00 is excluded."""
        cur_dt = _cur_dt()
        prev = cur_dt.date() - timedelta(days=1)
        cur = cur_dt.date()
        bars = [
            make_bar(time_msc=hour_msc(prev.year, prev.month, prev.day, 23),
                     high=1.10, low=1.05, close=1.07),
            make_bar(time_msc=hour_msc(cur.year, cur.month, cur.day, 0),
                     high=1.20, low=1.08, close=1.15),
            make_bar(time_msc=hour_msc(cur.year, cur.month, cur.day, 1),
                     high=5.0, low=0.5, close=2.0),  # outside window
        ]
        ah, al = _compute_asian_range(bars, cur_dt)
        # 01:00 bar's wide range must NOT contribute.
        assert ah == 1.20
        assert al == 1.05

    def test_bar_at_2330_prev_included(self):
        """A hypothetical 23:30 bar (if it existed) would be included
        because 23:30 ≥ 19:30 and < 00:30 next day."""
        cur_dt = _cur_dt()
        prev = cur_dt.date() - timedelta(days=1)
        # 23:30 prev day timestamp.
        t_2330 = int(datetime(prev.year, prev.month, prev.day, 23, 30,
                              tzinfo=UTC).timestamp() * 1000)
        bars = [
            make_bar(time_msc=hour_msc(prev.year, prev.month, prev.day, 20),
                     high=1.10, low=1.05, close=1.08),
            make_bar(time_msc=t_2330,
                     high=2.50, low=2.40, close=2.45),
        ]
        ah, al = _compute_asian_range(bars, cur_dt)
        assert ah == 2.50
        assert al == 1.05


class TestAsianMinBars:
    def test_less_than_2_bars_returns_none(self):
        cur_dt = _cur_dt()
        prev = cur_dt.date() - timedelta(days=1)
        bars = [
            make_bar(time_msc=hour_msc(prev.year, prev.month, prev.day, 20),
                     high=1.10, low=1.05, close=1.08),
        ]
        ah, al = _compute_asian_range(bars, cur_dt)
        assert ah is None
        assert al is None

    def test_empty_bars_returns_none(self):
        ah, al = _compute_asian_range([], _cur_dt())
        assert ah is None and al is None

    def test_no_in_window_bars_returns_none(self):
        cur_dt = _cur_dt()
        prev = cur_dt.date() - timedelta(days=1)
        bars = [
            make_bar(time_msc=hour_msc(prev.year, prev.month, prev.day, 18),
                     high=1.10, low=1.05, close=1.08),
            make_bar(time_msc=hour_msc(prev.year, prev.month, prev.day, 19),
                     high=1.10, low=1.05, close=1.08),
        ]
        ah, al = _compute_asian_range(bars, cur_dt)
        assert ah is None

    def test_exactly_two_bars_succeeds(self):
        cur_dt = _cur_dt()
        prev = cur_dt.date() - timedelta(days=1)
        bars = [
            make_bar(time_msc=hour_msc(prev.year, prev.month, prev.day, 20),
                     high=1.10, low=1.00, close=1.05),
            make_bar(time_msc=hour_msc(prev.year, prev.month, prev.day, 21),
                     high=1.20, low=1.05, close=1.15),
        ]
        ah, al = _compute_asian_range(bars, cur_dt)
        assert ah == 1.20
        assert al == 1.00


# ---------------------------------------------------------------------------
# 2. Range computation correctness (max-of-highs, min-of-lows)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("highs,lows,expected_h,expected_l", [
    ([1.10, 1.12, 1.15, 1.11, 1.09], [1.05, 1.04, 1.06, 1.03, 1.02], 1.15, 1.02),
    ([1.0, 1.0, 1.0, 1.0, 1.0], [0.9, 0.9, 0.9, 0.9, 0.9], 1.0, 0.9),
    ([2.0, 1.0, 1.5, 1.8, 1.6], [0.5, 0.3, 0.4, 0.45, 0.55], 2.0, 0.3),
    ([1.20, 1.21, 1.19, 1.18, 1.17], [1.10, 1.09, 1.08, 1.07, 1.06], 1.21, 1.06),
    ([1.50, 1.51, 1.49, 1.52, 1.50], [1.40, 1.39, 1.41, 1.38, 1.42], 1.52, 1.38),
])
class TestRangeComputation:
    def test_max_high(self, highs, lows, expected_h, expected_l):
        cur_dt = _cur_dt()
        prev = cur_dt.date() - timedelta(days=1)
        cur = cur_dt.date()
        hours = [(prev, 20), (prev, 21), (prev, 22), (prev, 23), (cur, 0)]
        bars = [
            make_bar(time_msc=hour_msc(d.year, d.month, d.day, h),
                     high=hi, low=lo, close=(hi + lo) / 2)
            for (d, h), hi, lo in zip(hours, highs, lows)
        ]
        ah, _ = _compute_asian_range(bars, cur_dt)
        assert ah == expected_h

    def test_min_low(self, highs, lows, expected_h, expected_l):
        cur_dt = _cur_dt()
        prev = cur_dt.date() - timedelta(days=1)
        cur = cur_dt.date()
        hours = [(prev, 20), (prev, 21), (prev, 22), (prev, 23), (cur, 0)]
        bars = [
            make_bar(time_msc=hour_msc(d.year, d.month, d.day, h),
                     high=hi, low=lo, close=(hi + lo) / 2)
            for (d, h), hi, lo in zip(hours, highs, lows)
        ]
        _, al = _compute_asian_range(bars, cur_dt)
        assert al == expected_l


# ---------------------------------------------------------------------------
# 3. Range filter — accept when min < rng < max
# ---------------------------------------------------------------------------

# For each pair, build a long-sweep scenario at a range exactly equal to
# `min_range_pts + slack` and verify the detector emits a signal.

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestRangeFilterAcceptance:
    def test_emits_when_range_above_min(self, pair, detector):
        cfg = PAIR_CONFIG[pair]
        pt = float(cfg["point"])
        min_r = float(cfg["min_range_pts"])
        range_pts = min_r + 50.0
        asian_low = baseline_low(pair)
        asian_high = asian_low + range_pts * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=asian_low, asian_high=asian_high,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is not None, f"{pair} should accept range above min_range_pts"

    def test_emits_when_range_well_below_max(self, pair, detector):
        cfg = PAIR_CONFIG[pair]
        pt = float(cfg["point"])
        min_r = float(cfg["min_range_pts"])
        max_r = float(cfg["max_range_pts"])
        range_pts = (min_r + max_r) / 2.0
        asian_low = baseline_low(pair)
        asian_high = asian_low + range_pts * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=asian_low, asian_high=asian_high,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is not None


# ---------------------------------------------------------------------------
# 4. Range filter — reject when below min or above max
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestRangeFilterRejection:
    def test_rejects_below_min(self, pair, detector):
        cfg = PAIR_CONFIG[pair]
        pt = float(cfg["point"])
        min_r = float(cfg["min_range_pts"])
        range_pts = max(1.0, min_r - 50.0)
        asian_low = baseline_low(pair)
        asian_high = asian_low + range_pts * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=asian_low, asian_high=asian_high,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is None, f"{pair} should reject range below min"

    def test_rejects_above_max(self, pair, detector):
        cfg = PAIR_CONFIG[pair]
        pt = float(cfg["point"])
        max_r = float(cfg["max_range_pts"])
        range_pts = max_r + 50.0
        asian_low = baseline_low(pair)
        asian_high = asian_low + range_pts * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=asian_low, asian_high=asian_high,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is None, f"{pair} should reject range above max"


# ---------------------------------------------------------------------------
# 5. Range filter — exact boundary behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestRangeFilterBoundaries:
    def test_exact_min_range_accepted(self, pair, detector):
        # Production rejects when `rng_pts < min_range_pts`. Equality is OK.
        cfg = PAIR_CONFIG[pair]
        pt = float(cfg["point"])
        min_r = float(cfg["min_range_pts"])
        asian_low = baseline_low(pair)
        asian_high = asian_low + min_r * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=asian_low, asian_high=asian_high,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is not None

    def test_exact_max_range_accepted(self, pair, detector):
        # Production rejects when `rng_pts > max_range_pts`. Equality is OK.
        cfg = PAIR_CONFIG[pair]
        pt = float(cfg["point"])
        max_r = float(cfg["max_range_pts"])
        asian_low = baseline_low(pair)
        asian_high = asian_low + max_r * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=asian_low, asian_high=asian_high,
            wick_below_pts=10.0, close_above_pts=5.0,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is not None

    def test_one_pt_below_min_rejected(self, pair, detector):
        cfg = PAIR_CONFIG[pair]
        pt = float(cfg["point"])
        min_r = float(cfg["min_range_pts"])
        asian_low = baseline_low(pair)
        # min - 1 pt: rng_pts rounded should land at min-1 → reject
        asian_high = asian_low + (min_r - 1) * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=asian_low, asian_high=asian_high,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None

    def test_one_pt_above_max_rejected(self, pair, detector):
        cfg = PAIR_CONFIG[pair]
        pt = float(cfg["point"])
        max_r = float(cfg["max_range_pts"])
        asian_low = baseline_low(pair)
        asian_high = asian_low + (max_r + 1) * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=asian_low, asian_high=asian_high,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None


# ---------------------------------------------------------------------------
# 6. Edge: range == 0 (flat market)
# ---------------------------------------------------------------------------

class TestZeroRange:
    def test_zero_range_returns_none(self, detector):
        # When all asian bars share the same H == L, ah <= al filter kicks
        # in (production guards `if ah <= al: return None`).
        cur_dt = _cur_dt()
        prev = cur_dt.date() - timedelta(days=1)
        cur = cur_dt.date()
        hours = [(prev, 20), (prev, 21), (prev, 22), (prev, 23), (cur, 0)]
        bars = [
            make_bar(symbol="EURUSD",
                     time_msc=hour_msc(d.year, d.month, d.day, h),
                     open=1.10, high=1.10, low=1.10, close=1.10)
            for d, h in hours
        ]
        # need a trigger bar too so detector reaches range computation.
        bars.append(make_bar(symbol="EURUSD",
                             time_msc=hour_msc(cur.year, cur.month, cur.day, 8),
                             open=1.10, high=1.10, low=1.10, close=1.10))
        # also need >= 30 bars total → pad more flat history before window.
        for i in range(40, 0, -1):
            dt = datetime(prev.year, prev.month, prev.day, 19, 0,
                          tzinfo=UTC) - timedelta(hours=i)
            bars.insert(0, make_bar(
                symbol="EURUSD",
                time_msc=int(dt.timestamp() * 1000),
                open=1.10, high=1.10, low=1.10, close=1.10,
            ))
        ctx = MarketContext(symbol="EURUSD", current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None


# ---------------------------------------------------------------------------
# 7. Edge: ah <= al guard
# ---------------------------------------------------------------------------

class TestInvertedRange:
    def test_when_compute_returns_inverted_detector_rejects(self, detector):
        # We can't naturally make max < min, but we can test the explicit
        # detector guard by mocking _compute_asian_range to return ah==al.
        from strategy.patterns import asian_sweep as mod
        original = mod._compute_asian_range
        try:
            mod._compute_asian_range = lambda bars, dt: (1.10, 1.10)
            bars = long_sweep_bars(symbol="EURUSD", pt=0.00001)
            ctx = MarketContext(symbol="EURUSD",
                                current_time_msc=bars[-1].time_msc)
            assert detector.detect(bars, ctx) is None
        finally:
            mod._compute_asian_range = original

    def test_when_compute_returns_none_detector_rejects(self, detector):
        from strategy.patterns import asian_sweep as mod
        original = mod._compute_asian_range
        try:
            mod._compute_asian_range = lambda bars, dt: (None, None)
            bars = long_sweep_bars(symbol="EURUSD", pt=0.00001)
            ctx = MarketContext(symbol="EURUSD",
                                current_time_msc=bars[-1].time_msc)
            assert detector.detect(bars, ctx) is None
        finally:
            mod._compute_asian_range = original


# ---------------------------------------------------------------------------
# 8. Asian range constructed by the build_scenario helper (per pair)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("ah_hour,al_hour", [
    (20, 21), (21, 20), (22, 21), (23, 20), (22, 23),
])
class TestHelperPlacesRangeBars:
    def test_helper_places_high_and_low(self, pair, ah_hour, al_hour, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        # Choose a range comfortably above min and below max.
        cfg = PAIR_CONFIG[pair]
        range_pts = (float(cfg["min_range_pts"])
                     + float(cfg["max_range_pts"])) / 2.0
        asian_low = baseline_low(pair)
        asian_high = asian_low + range_pts * pt
        from tests.strategy.fixtures.synthetic_bars import build_scenario
        bars = build_scenario(
            symbol=pair, year=2026, month=4, day=15,
            asian_high=asian_high, asian_low=asian_low,
            trigger_hour=8,
            trigger_high=asian_low + 5 * pt,
            trigger_low=asian_low - 50 * pt,
            trigger_close=asian_low + 5 * pt,
            asian_high_at_hour=ah_hour,
            asian_low_at_hour=al_hour,
        )
        cur_dt = _cur_dt()
        ah, al = _compute_asian_range(bars, cur_dt)
        assert ah == pytest.approx(asian_high, rel=1e-9, abs=pt / 2)
        assert al == pytest.approx(asian_low, rel=1e-9, abs=pt / 2)


# ---------------------------------------------------------------------------
# 9. Five asian bars are all included
# ---------------------------------------------------------------------------

class TestFiveAsianBarsContribute:
    def test_high_appearing_in_each_of_five_hours(self, detector):
        """If we stamp the high on each of the 5 asian hours in turn, the
        detector should pick it up."""
        cur_dt = _cur_dt()
        prev = cur_dt.date() - timedelta(days=1)
        cur = cur_dt.date()
        slots = [(prev, 20), (prev, 21), (prev, 22), (prev, 23), (cur, 0)]
        for hi_hour_d, hi_hour_h in slots:
            bars = []
            for d, h in slots:
                hi = 1.20 if (d, h) == (hi_hour_d, hi_hour_h) else 1.10
                bars.append(make_bar(symbol="EURUSD",
                                     time_msc=hour_msc(d.year, d.month, d.day, h),
                                     high=hi, low=1.05, close=1.08))
            ah, al = _compute_asian_range(bars, cur_dt)
            assert ah == 1.20, (
                f"Failed when high stamped at {hi_hour_d} h={hi_hour_h}"
            )
            assert al == 1.05


# ---------------------------------------------------------------------------
# 10. Per-pair × min_range slack matrix
# ---------------------------------------------------------------------------

SLACK_PTS = [-1, 0, 1, 50, 100, 250, 500]


@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("slack", SLACK_PTS)
class TestPerPairMinRangeMatrix:
    def test_min_range_slack(self, pair, slack, detector):
        cfg = PAIR_CONFIG[pair]
        pt = float(cfg["point"])
        min_r = float(cfg["min_range_pts"])
        max_r = float(cfg["max_range_pts"])
        range_pts = min_r + slack
        if range_pts > max_r:
            pytest.skip("slack would push past max")
        asian_low = baseline_low(pair)
        asian_high = asian_low + range_pts * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=asian_low, asian_high=asian_high,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        if slack < 0:
            assert sig is None
        else:
            assert sig is not None


# ---------------------------------------------------------------------------
# 11. Per-pair × max_range slack matrix
# ---------------------------------------------------------------------------

MAX_SLACK_PTS = [-500, -100, -50, -1, 0, 1, 50, 100]


@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("slack", MAX_SLACK_PTS)
class TestPerPairMaxRangeMatrix:
    def test_max_range_slack(self, pair, slack, detector):
        cfg = PAIR_CONFIG[pair]
        pt = float(cfg["point"])
        min_r = float(cfg["min_range_pts"])
        max_r = float(cfg["max_range_pts"])
        range_pts = max_r + slack
        if range_pts < min_r:
            pytest.skip("slack would push below min")
        asian_low = baseline_low(pair)
        asian_high = asian_low + range_pts * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=asian_low, asian_high=asian_high,
            wick_below_pts=10.0, close_above_pts=5.0,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        if slack > 0:
            assert sig is None
        else:
            assert sig is not None


# ---------------------------------------------------------------------------
# 12. Range with weekend gap
# ---------------------------------------------------------------------------

class TestWeekendGap:
    def test_weekend_gap_handled(self, detector):
        # Saturday/Sunday have no bars. Trading day = Monday → but
        # SKIP_MONDAY is True. So pick Tuesday after a weekend. The
        # detector should still find the prev-day (Monday) asian bars.
        # 2026-04-14 is a Tuesday.
        bars = long_sweep_bars(
            symbol="EURUSD", pt=0.00001,
            year=2026, month=4, day=14,
        )
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is not None


# ---------------------------------------------------------------------------
# 13. Bars on multiple unrelated dates don't contaminate
# ---------------------------------------------------------------------------

class TestBarsAcrossDates:
    def test_old_bars_ignored(self, detector):
        """Bars from a different week shouldn't contaminate the asian range."""
        cur_dt = _cur_dt()
        # Build a normal scenario.
        bars = long_sweep_bars(symbol="EURUSD", pt=0.00001)
        # Prepend a noisy bar from 30 days earlier.
        old_dt = cur_dt - timedelta(days=30, hours=23)
        bars.insert(0, make_bar(
            symbol="EURUSD",
            time_msc=int(old_dt.timestamp() * 1000),
            open=999.0, high=1000.0, low=998.0, close=999.5,
        ))
        ah, al = _compute_asian_range(bars, cur_dt)
        assert ah is not None
        assert ah < 100  # not contaminated by the 999 bar
