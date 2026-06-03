"""London-session sweep tests for AsianSweepDetector.

LONDON window  : bars at 06,07,08,09,10 UTC (h start..h end inclusive)
LONG  trigger  : bias ∈ {bullish, neutral} AND bar.low  < AL AND bar.close > AL
SHORT trigger  : bias == bearish           AND bar.high > AH AND bar.close < AH

Both directions are valid in LONDON (vs NY which is LONG only).
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone

import pytest

from config.asian_sweep_config import (
    LONDON_SWEEP_UTC_H_END, LONDON_SWEEP_UTC_H_START, PAIR_CONFIG, PAIRS,
)
from strategy.patterns.asian_sweep import AsianSweepDetector
from strategy.patterns.base import Direction, Grade, MarketContext

from tests.strategy.fixtures.synthetic_bars import (
    baseline_low, build_scenario, long_sweep_bars, make_bar,
    short_sweep_bars, hour_msc,
)

UTC = timezone.utc
ALL_PAIRS = list(PAIRS)
LONDON_HOURS = list(range(LONDON_SWEEP_UTC_H_START,
                          LONDON_SWEEP_UTC_H_END + 1))   # 6,7,8,9,10


def _baseline_low(pair: str) -> float:
    return baseline_low(pair)


def _baseline_range_pts(pair: str) -> float:
    cfg = PAIR_CONFIG[pair]
    return (float(cfg["min_range_pts"])
            + float(cfg["max_range_pts"])) / 2.0


# ---------------------------------------------------------------------------
# 1. London window inclusion — per-hour LONG smoke
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("hour", LONDON_HOURS)
class TestLondonLongPerHour:
    def test_emits_long(self, pair, hour, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al, asian_high=ah,
            trigger_hour=hour,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is not None
        assert sig.direction == Direction.BUY

    def test_signal_session_tag_is_london(self, pair, hour, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al, asian_high=ah,
            trigger_hour=hour,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert "LONDON" in sig.confluences_met

    def test_sweep_tag_is_low(self, pair, hour, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al, asian_high=ah,
            trigger_hour=hour,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert "asian_sweep_low" in sig.confluences_met


# ---------------------------------------------------------------------------
# 2. London window inclusion — per-hour SHORT smoke
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("hour", LONDON_HOURS)
class TestLondonShortPerHour:
    def test_emits_short(self, pair, hour, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al, asian_high=ah,
            trigger_hour=hour,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is not None
        assert sig.direction == Direction.SELL

    def test_short_session_tag(self, pair, hour, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al, asian_high=ah,
            trigger_hour=hour,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert "LONDON" in sig.confluences_met

    def test_short_sweep_tag_is_high(self, pair, hour, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al, asian_high=ah,
            trigger_hour=hour,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert "asian_sweep_high" in sig.confluences_met


# ---------------------------------------------------------------------------
# 3. Bias gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestBiasGate:
    def test_short_requires_bearish_bias(self, pair, detector):
        """A short setup with NEUTRAL bias must NOT emit."""
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = build_scenario(
            symbol=pair, year=2026, month=4, day=15,
            asian_high=ah, asian_low=al,
            trigger_hour=8,
            trigger_high=ah + 50 * pt,
            trigger_low=ah - 10 * pt,
            trigger_close=ah - 10 * pt,
            bias="neutral",
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        # Neutral bias might still trigger LONG if low < al; here trigger
        # has low above al so no LONG either.
        assert sig is None

    def test_short_blocked_by_bullish_bias(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = build_scenario(
            symbol=pair, year=2026, month=4, day=15,
            asian_high=ah, asian_low=al,
            trigger_hour=8,
            trigger_high=ah + 50 * pt,
            trigger_low=ah - 10 * pt,
            trigger_close=ah - 10 * pt,
            bias="bullish",
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is None

    def test_long_allowed_with_neutral(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al, asian_high=ah, bias="neutral",
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is not None

    def test_long_allowed_with_bullish(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al, asian_high=ah, bias="bullish",
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is not None
        assert "bias_bullish" in sig.confluences_met

    def test_long_blocked_by_bearish_bias(self, pair, detector):
        """A long setup with BEARISH bias must NOT emit."""
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = build_scenario(
            symbol=pair, year=2026, month=4, day=15,
            asian_high=ah, asian_low=al,
            trigger_hour=8,
            trigger_high=al + 10 * pt,
            trigger_low=al - 50 * pt,
            trigger_close=al + 10 * pt,
            bias="bearish",
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is None


# ---------------------------------------------------------------------------
# 4. Sweep close-back required
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestCloseBack:
    def test_wick_only_long_no_signal(self, pair, detector):
        """Low pierces AL but close stays below AL → no LONG."""
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = build_scenario(
            symbol=pair, year=2026, month=4, day=15,
            asian_high=ah, asian_low=al,
            trigger_hour=8,
            trigger_high=al,        # touches AL but no break
            trigger_low=al - 50 * pt,
            trigger_close=al - 20 * pt,  # closes BELOW AL → wick only
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None

    def test_wick_only_short_no_signal(self, pair, detector):
        """High pierces AH but close stays above AH → no SHORT."""
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = build_scenario(
            symbol=pair, year=2026, month=4, day=15,
            asian_high=ah, asian_low=al,
            trigger_hour=8,
            trigger_high=ah + 50 * pt,
            trigger_low=ah,
            trigger_close=ah + 20 * pt,  # closes ABOVE AH → wick only
            bias="bearish",
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None

    def test_long_close_exactly_at_al_no_signal(self, pair, detector):
        """`close > AL` is strict; equal to AL does NOT trigger."""
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = build_scenario(
            symbol=pair, year=2026, month=4, day=15,
            asian_high=ah, asian_low=al,
            trigger_hour=8,
            trigger_high=al,
            trigger_low=al - 50 * pt,
            trigger_close=al,        # exactly equal
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None

    def test_short_close_exactly_at_ah_no_signal(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = build_scenario(
            symbol=pair, year=2026, month=4, day=15,
            asian_high=ah, asian_low=al,
            trigger_hour=8,
            trigger_high=ah + 50 * pt,
            trigger_low=ah,
            trigger_close=ah,
            bias="bearish",
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None

    def test_long_low_exactly_at_al_no_signal(self, pair, detector):
        """`low < AL` is strict — `low == AL` is not a sweep."""
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = build_scenario(
            symbol=pair, year=2026, month=4, day=15,
            asian_high=ah, asian_low=al,
            trigger_hour=8,
            trigger_high=al + 20 * pt,
            trigger_low=al,
            trigger_close=al + 20 * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None


# ---------------------------------------------------------------------------
# 5. Outside the window
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hour", [0, 1, 2, 3, 4, 5, 11, 17, 18, 22])
class TestOutsideWindow:
    def test_no_signal_outside_london_or_ny(self, hour, detector):
        bars = long_sweep_bars(
            symbol="EURUSD", pt=0.00001, trigger_hour=hour,
        )
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None


# ---------------------------------------------------------------------------
# 6. Wick depth variations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("wick_pts", [1, 5, 20, 50, 100, 250])
class TestWickDepth:
    def test_long_wick_depth(self, pair, wick_pts, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al, asian_high=ah,
            wick_below_pts=wick_pts, close_above_pts=5.0,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        # Deep wicks always trigger.
        assert sig is not None

    def test_short_wick_depth(self, pair, wick_pts, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al, asian_high=ah,
            wick_above_pts=wick_pts, close_below_pts=5.0,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is not None


# ---------------------------------------------------------------------------
# 7. Close-back depth variations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("close_back_pts", [1, 5, 10, 20, 50])
class TestCloseBackDepth:
    def test_long_close_back_strength(self, pair, close_back_pts, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al, asian_high=ah,
            wick_below_pts=30.0, close_above_pts=float(close_back_pts),
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is not None

    def test_short_close_back_strength(self, pair, close_back_pts, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al, asian_high=ah,
            wick_above_pts=30.0, close_below_pts=float(close_back_pts),
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is not None


# ---------------------------------------------------------------------------
# 8. Simultaneous high + low pierce — LONG wins because LONG-block comes
#    AFTER SHORT in code (so SHORT can pre-empt under bearish bias only).
#    Under neutral bias the SHORT block is skipped, so LONG triggers.
# ---------------------------------------------------------------------------

class TestSimultaneousHighLowPierce:
    def test_pierces_both_neutral_emits_long(self, detector):
        # Both AH and AL pierced; close above AL → LONG fires.
        pair = "EURUSD"
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = build_scenario(
            symbol=pair, year=2026, month=4, day=15,
            asian_high=ah, asian_low=al,
            trigger_hour=8,
            trigger_high=ah + 30 * pt,
            trigger_low=al - 30 * pt,
            trigger_close=al + 10 * pt,
            bias="neutral",
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is not None
        assert sig.direction == Direction.BUY

    def test_pierces_both_bearish_emits_short(self, detector):
        # Both pierced, close below AH → SHORT block runs first.
        pair = "EURUSD"
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = build_scenario(
            symbol=pair, year=2026, month=4, day=15,
            asian_high=ah, asian_low=al,
            trigger_hour=8,
            trigger_high=ah + 30 * pt,
            trigger_low=al - 30 * pt,
            trigger_close=ah - 10 * pt,  # below AH, above AL
            bias="bearish",
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig is not None
        assert sig.direction == Direction.SELL


# ---------------------------------------------------------------------------
# 9. Skip Monday
# ---------------------------------------------------------------------------

class TestSkipMonday:
    # 2026-04-13 is a Monday.
    def test_no_signal_on_monday(self, detector):
        bars = long_sweep_bars(
            symbol="EURUSD", pt=0.00001,
            year=2026, month=4, day=13,
        )
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None

    @pytest.mark.parametrize("day", [14, 15, 16, 17])  # Tue-Fri
    def test_signal_on_other_weekdays(self, day, detector):
        bars = long_sweep_bars(
            symbol="EURUSD", pt=0.00001,
            year=2026, month=4, day=day,
        )
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is not None


# ---------------------------------------------------------------------------
# 10. Unknown symbol
# ---------------------------------------------------------------------------

class TestUnknownSymbol:
    def test_unknown_symbol_returns_none(self, detector):
        bars = long_sweep_bars(
            symbol="ZZZZZZ", pt=0.00001,
        )
        ctx = MarketContext(symbol="ZZZZZZ",
                            current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None


# ---------------------------------------------------------------------------
# 11. Min bars guard
# ---------------------------------------------------------------------------

class TestMinBarsGuard:
    def test_zero_bars(self, detector):
        ctx = MarketContext(symbol="EURUSD", current_time_msc=0)
        assert detector.detect([], ctx) is None

    @pytest.mark.parametrize("n", [1, 5, 10, 20, 29])
    def test_fewer_than_min_required(self, n, detector):
        base_ms = hour_msc(2026, 4, 14, 0)
        bars = [
            make_bar(symbol="EURUSD", time_msc=base_ms + h * 3600 * 1000)
            for h in range(n)
        ]
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None

    def test_min_bars_required_value(self):
        assert AsianSweepDetector.min_bars_required == 30


# ---------------------------------------------------------------------------
# 12. Sweep tag and confluence content
# ---------------------------------------------------------------------------

class TestConfluenceContent:
    def test_long_confluences_have_all_5_tags(self, detector):
        bars = long_sweep_bars(symbol="EURUSD", pt=0.00001)
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert len(sig.confluences_met) == 5

    def test_short_confluences_have_all_5_tags(self, detector):
        bars = short_sweep_bars(symbol="EURUSD", pt=0.00001)
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert len(sig.confluences_met) == 5

    def test_q_tag_present_long(self, detector):
        bars = long_sweep_bars(symbol="EURUSD", pt=0.00001)
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert any(t.startswith("q") for t in sig.confluences_met)

    def test_tp1_tag_present_long(self, detector):
        bars = long_sweep_bars(symbol="EURUSD", pt=0.00001)
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        tp1_tags = [t for t in sig.confluences_met if t.startswith("tp1_")]
        assert len(tp1_tags) == 1

    def test_bias_tag_format(self, detector):
        bars = long_sweep_bars(symbol="EURUSD", pt=0.00001)
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        bias_tags = [t for t in sig.confluences_met if t.startswith("bias_")]
        assert len(bias_tags) == 1


# ---------------------------------------------------------------------------
# 13. Pattern + bar_time_msc + symbol fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestSignalFields:
    def test_pattern_name(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.pattern_name == "ASIAN_SWEEP"

    def test_symbol_matches_context(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.symbol == pair

    def test_bar_time_msc_is_trigger_bar(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.bar_time_msc == bars[-1].time_msc


# ---------------------------------------------------------------------------
# 14. Bias tag content matches actual bias
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bias_setting,expected_tag", [
    ("neutral", "bias_neutral"),
    ("bullish", "bias_bullish"),
])
class TestBiasTagLong:
    def test_long_bias_tag(self, bias_setting, expected_tag, detector):
        pair = "EURUSD"
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al, asian_high=ah, bias=bias_setting,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert expected_tag in sig.confluences_met


class TestBiasTagShort:
    def test_short_bias_tag(self, detector):
        bars = short_sweep_bars(symbol="EURUSD", pt=0.00001)
        ctx = MarketContext(symbol="EURUSD",
                            current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert "bias_bearish" in sig.confluences_met


# ---------------------------------------------------------------------------
# 15. Detector returns None when given bars but ctx symbol mismatches
# ---------------------------------------------------------------------------

class TestContextSymbolMismatch:
    def test_context_drives_symbol_resolution(self, detector):
        # Even if bars carry symbol='EURUSD', the detector uses
        # context.symbol to look up PAIR_CONFIG. An unknown context
        # symbol → None.
        bars = long_sweep_bars(symbol="EURUSD", pt=0.00001)
        ctx = MarketContext(symbol="UNKNOWN",
                            current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None


# ---------------------------------------------------------------------------
# 16. Confidence and grade for LONDON LONG (per pair)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestLondonSignalGrade:
    def test_confidence_in_range(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert 0.0 <= sig.confidence <= 1.0

    def test_confidence_matches_quality_div_10(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        q = PAIR_CONFIG[pair]["quality"]
        assert sig.confidence == pytest.approx(q / 10.0)

    def test_grade_a_or_b(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.grade in {Grade.A, Grade.B}


# ---------------------------------------------------------------------------
# 17. Bias-mode close-back combos
# ---------------------------------------------------------------------------

CLOSE_BACK_MATRIX = [
    # (pair, bias, expected_dir_or_none)
    ("EURUSD", "bullish", Direction.BUY),
    ("EURUSD", "neutral", Direction.BUY),
    ("EURUSD", "bearish", None),
    ("XAUUSD", "bullish", Direction.BUY),
    ("XAUUSD", "neutral", Direction.BUY),
    ("XAUUSD", "bearish", None),
    ("GBPUSD", "neutral", Direction.BUY),
    ("AUDUSD", "neutral", Direction.BUY),
    ("USDCAD", "neutral", Direction.BUY),
    ("USDCHF", "neutral", Direction.BUY),
    ("AUDCHF", "neutral", Direction.BUY),
    ("AUDNZD", "neutral", Direction.BUY),
]


@pytest.mark.parametrize("pair,bias,expected", CLOSE_BACK_MATRIX)
def test_long_bias_matrix(pair, bias, expected, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    bars = long_sweep_bars(
        symbol=pair, pt=pt,
        asian_low=_baseline_low(pair),
        asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        bias=bias,
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    if expected is None:
        assert sig is None
    else:
        assert sig is not None
        assert sig.direction == expected


# ---------------------------------------------------------------------------
# 18. Permutation: per-pair × London hour × bias for LONG
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("hour", LONDON_HOURS)
@pytest.mark.parametrize("bias", ["bullish", "neutral"])
def test_long_full_matrix(pair, hour, bias, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    al = _baseline_low(pair)
    ah = al + _baseline_range_pts(pair) * pt
    bars = long_sweep_bars(
        symbol=pair, pt=pt,
        asian_low=al, asian_high=ah,
        trigger_hour=hour, bias=bias,
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert sig is not None
    assert sig.direction == Direction.BUY


# ---------------------------------------------------------------------------
# 19. Permutation: per-pair × London hour for SHORT (bearish only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("hour", LONDON_HOURS)
def test_short_full_matrix(pair, hour, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    al = _baseline_low(pair)
    ah = al + _baseline_range_pts(pair) * pt
    bars = short_sweep_bars(
        symbol=pair, pt=pt,
        asian_low=al, asian_high=ah,
        trigger_hour=hour,
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert sig is not None
    assert sig.direction == Direction.SELL


# ---------------------------------------------------------------------------
# 20. London-edge-hour 11 is OUTSIDE the window
# ---------------------------------------------------------------------------

def test_hour_11_outside_london(detector):
    bars = long_sweep_bars(symbol="EURUSD", pt=0.00001, trigger_hour=11)
    ctx = MarketContext(symbol="EURUSD",
                        current_time_msc=bars[-1].time_msc)
    assert detector.detect(bars, ctx) is None


def test_hour_5_outside_london(detector):
    bars = long_sweep_bars(symbol="EURUSD", pt=0.00001, trigger_hour=5)
    ctx = MarketContext(symbol="EURUSD",
                        current_time_msc=bars[-1].time_msc)
    assert detector.detect(bars, ctx) is None
