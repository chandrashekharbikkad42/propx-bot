"""NY-session continuation tests for AsianSweepDetector.

V5 rule: NY = LONG only. The detector flips `allow_short=False` when the
hour is in [NY_SWEEP_UTC_H_START, NY_SWEEP_UTC_H_END]. Even a textbook
SHORT setup (bearish bias + sweep AH + close back) must NOT emit during NY.
"""

from __future__ import annotations

import pytest

from config.asian_sweep_config import (
    NY_SWEEP_UTC_H_END, NY_SWEEP_UTC_H_START, PAIR_CONFIG, PAIRS,
)
from strategy.patterns.asian_sweep import AsianSweepDetector
from strategy.patterns.base import Direction, MarketContext

from tests.strategy.fixtures.synthetic_bars import (
    build_scenario, long_sweep_bars, short_sweep_bars,
)


ALL_PAIRS = list(PAIRS)
NY_HOURS = list(range(NY_SWEEP_UTC_H_START, NY_SWEEP_UTC_H_END + 1))  # 12..15


def _baseline_low(pair: str) -> float:
    return 100.0 if pair == "XAUUSD" else 1.10000


def _baseline_range_pts(pair: str) -> float:
    cfg = PAIR_CONFIG[pair]
    return (float(cfg["min_range_pts"])
            + float(cfg["max_range_pts"])) / 2.0


# ---------------------------------------------------------------------------
# 1. NY window: LONG fires
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("hour", NY_HOURS)
class TestNyLongPerHour:
    def test_long_emits(self, pair, hour, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
            trigger_hour=hour,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is not None
        assert sig.direction == Direction.BUY

    def test_session_tag_is_ny(self, pair, hour, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
            trigger_hour=hour,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert "NY" in sig.confluences_met

    def test_not_tagged_london(self, pair, hour, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
            trigger_hour=hour,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert "LONDON" not in sig.confluences_met


# ---------------------------------------------------------------------------
# 2. NY window: SHORT must be suppressed (V5 rule)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("hour", NY_HOURS)
class TestNyShortSuppression:
    def test_short_setup_rejected_in_ny(self, pair, hour, detector):
        """Even a textbook SHORT setup must NOT emit during NY."""
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = short_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
            trigger_hour=hour,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is None, f"{pair} NY h={hour}: SHORT must be suppressed"


# ---------------------------------------------------------------------------
# 3. Boundary hours adjacent to NY window
# ---------------------------------------------------------------------------

class TestNyWindowBoundaries:
    @pytest.mark.parametrize("hour", [11, 16, 17])
    def test_hour_outside_ny_window(self, hour, detector):
        bars = long_sweep_bars(symbol="EURUSD", pt=0.00001,
                               trigger_hour=hour)
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        # Hours 11, 16, 17 are NOT in any session window.
        assert detector.detect(bars, ctx) is None

    def test_first_ny_hour(self, detector):
        bars = long_sweep_bars(symbol="EURUSD", pt=0.00001,
                               trigger_hour=NY_SWEEP_UTC_H_START)
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is not None

    def test_last_ny_hour(self, detector):
        bars = long_sweep_bars(symbol="EURUSD", pt=0.00001,
                               trigger_hour=NY_SWEEP_UTC_H_END)
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is not None


# ---------------------------------------------------------------------------
# 4. NY long with bullish + neutral bias
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("bias", ["bullish", "neutral"])
@pytest.mark.parametrize("hour", NY_HOURS)
def test_ny_long_bias_matrix(pair, bias, hour, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    bars = long_sweep_bars(
        symbol=pair, pt=pt,
        asian_low=_baseline_low(pair),
        asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        trigger_hour=hour, bias=bias,
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert sig is not None
    assert sig.direction == Direction.BUY


# ---------------------------------------------------------------------------
# 5. NY long with bearish bias blocked
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("hour", NY_HOURS)
def test_ny_long_bearish_bias_blocked(pair, hour, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    al = _baseline_low(pair)
    ah = al + _baseline_range_pts(pair) * pt
    # Build a LONG-style setup with bearish bias — should be blocked.
    bars = build_scenario(
        symbol=pair, year=2026, month=4, day=15,
        asian_high=ah, asian_low=al,
        trigger_hour=hour,
        trigger_high=al + 20 * pt,
        trigger_low=al - 50 * pt,
        trigger_close=al + 10 * pt,
        bias="bearish",
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert sig is None


# ---------------------------------------------------------------------------
# 6. NY wick depth (LONG)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("wick_pts", [5, 20, 80, 200])
@pytest.mark.parametrize("hour", NY_HOURS)
def test_ny_long_wick_depths(pair, wick_pts, hour, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    bars = long_sweep_bars(
        symbol=pair, pt=pt,
        asian_low=_baseline_low(pair),
        asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        trigger_hour=hour, wick_below_pts=wick_pts,
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert sig is not None


# ---------------------------------------------------------------------------
# 7. NY close-back depth (LONG)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("close_pts", [1, 5, 10, 30])
@pytest.mark.parametrize("hour", NY_HOURS)
def test_ny_long_close_back_depth(pair, close_pts, hour, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    bars = long_sweep_bars(
        symbol=pair, pt=pt,
        asian_low=_baseline_low(pair),
        asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        trigger_hour=hour, close_above_pts=float(close_pts),
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert sig is not None


# ---------------------------------------------------------------------------
# 8. NY long wick-only (no close-back) rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("hour", NY_HOURS)
def test_ny_long_wick_only_rejected(pair, hour, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    al = _baseline_low(pair)
    ah = al + _baseline_range_pts(pair) * pt
    bars = build_scenario(
        symbol=pair, year=2026, month=4, day=15,
        asian_high=ah, asian_low=al,
        trigger_hour=hour,
        trigger_high=al,
        trigger_low=al - 50 * pt,
        trigger_close=al - 10 * pt,  # below AL, no close-back
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    assert detector.detect(bars, ctx) is None


# ---------------------------------------------------------------------------
# 9. NY between-window hour (11) is excluded
# ---------------------------------------------------------------------------

class TestBetweenLondonAndNy:
    def test_hour_11_no_signal(self, detector):
        bars = long_sweep_bars(symbol="EURUSD", pt=0.00001, trigger_hour=11)
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None

    def test_hour_16_no_signal(self, detector):
        bars = long_sweep_bars(symbol="EURUSD", pt=0.00001, trigger_hour=16)
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None
