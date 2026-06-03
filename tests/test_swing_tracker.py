"""Phase 8C — SwingTracker tests."""

from __future__ import annotations
from typing import Optional

import pytest

from data.bar_aggregator import Bar
from strategy.swing_tracker import SwingTracker


HOUR_MS = 3_600_000


def _bar(
    symbol: str,
    idx: int,
    *,
    hi: float,
    lo: float,
    o: Optional[float] = None,
    c: Optional[float] = None,
) -> Bar:
    """Construct a bar with explicit hi/lo; open/close default to mid."""
    o = o if o is not None else (hi + lo) / 2
    c = c if c is not None else (hi + lo) / 2
    return Bar(
        symbol=symbol,
        time_msc=idx * HOUR_MS,
        open=o,
        high=hi,
        low=lo,
        close=c,
        volume=1,
        spread_mean=0.0,
    )


@pytest.fixture
def st():
    return SwingTracker()


class TestInitialState:
    def test_no_swings_before_any_update(self, st):
        assert st.get_last_swing_high("EURUSD") is None
        assert st.get_last_swing_low("EURUSD") is None

    def test_first_bar_returns_no_swings(self, st):
        r = st.update("EURUSD", _bar("EURUSD", 0, hi=1.10, lo=1.09))
        assert r["new_swing_high"] is None
        assert r["new_swing_low"] is None
        assert r["broke_high"] is False
        assert r["broke_low"] is False

    def test_two_bars_no_swings_yet(self, st):
        st.update("EURUSD", _bar("EURUSD", 0, hi=1.10, lo=1.09))
        r = st.update("EURUSD", _bar("EURUSD", 1, hi=1.11, lo=1.10))
        assert r["new_swing_high"] is None
        assert r["new_swing_low"] is None


class TestSwingFormation:
    def test_three_bars_form_swing_high(self, st):
        st.update("EURUSD", _bar("EURUSD", 0, hi=1.10, lo=1.09))
        st.update("EURUSD", _bar("EURUSD", 1, hi=1.12, lo=1.11))  # peak
        r = st.update("EURUSD", _bar("EURUSD", 2, hi=1.11, lo=1.10))
        assert r["new_swing_high"] == pytest.approx(1.12)
        assert r["new_swing_low"] is None
        assert st.get_last_swing_high("EURUSD") == pytest.approx(1.12)

    def test_three_bars_form_swing_low(self, st):
        st.update("EURUSD", _bar("EURUSD", 0, hi=1.11, lo=1.10))
        st.update("EURUSD", _bar("EURUSD", 1, hi=1.10, lo=1.08))  # trough
        r = st.update("EURUSD", _bar("EURUSD", 2, hi=1.11, lo=1.10))
        assert r["new_swing_low"] == pytest.approx(1.08)
        assert r["new_swing_high"] is None
        assert st.get_last_swing_low("EURUSD") == pytest.approx(1.08)

    def test_no_swing_on_monotonic_rising_price(self, st):
        for i in range(5):
            r = st.update(
                "EURUSD",
                _bar("EURUSD", i, hi=1.10 + i * 0.01, lo=1.09 + i * 0.01),
            )
            assert r["new_swing_high"] is None
            assert r["new_swing_low"] is None
        assert st.get_last_swing_high("EURUSD") is None

    def test_no_swing_on_monotonic_falling_price(self, st):
        for i in range(5):
            r = st.update(
                "EURUSD",
                _bar("EURUSD", i, hi=1.15 - i * 0.01, lo=1.14 - i * 0.01),
            )
            assert r["new_swing_high"] is None
            assert r["new_swing_low"] is None


class TestWickBreak:
    def test_wick_break_of_high_close_below_still_counts(self, st):
        # Confirm a swing high at 1.12.
        st.update("EURUSD", _bar("EURUSD", 0, hi=1.10, lo=1.09))
        st.update("EURUSD", _bar("EURUSD", 1, hi=1.12, lo=1.11))
        st.update("EURUSD", _bar("EURUSD", 2, hi=1.11, lo=1.10))
        # Bar 3 wicks above 1.12 with close back at 1.10.
        r = st.update(
            "EURUSD",
            _bar("EURUSD", 3, hi=1.121, lo=1.10, o=1.105, c=1.10),
        )
        assert r["broke_high"] is True
        assert r["broke_low"] is False

    def test_wick_break_of_low_close_above_still_counts(self, st):
        st.update("EURUSD", _bar("EURUSD", 0, hi=1.11, lo=1.10))
        st.update("EURUSD", _bar("EURUSD", 1, hi=1.10, lo=1.08))
        st.update("EURUSD", _bar("EURUSD", 2, hi=1.11, lo=1.10))
        r = st.update(
            "EURUSD",
            _bar("EURUSD", 3, hi=1.11, lo=1.079, o=1.105, c=1.105),
        )
        assert r["broke_low"] is True
        assert r["broke_high"] is False

    def test_no_break_before_any_swing_confirmed(self, st):
        # No swings yet — even a huge bar can't "break" what doesn't exist.
        r = st.update("EURUSD", _bar("EURUSD", 0, hi=99.0, lo=0.1))
        assert r["broke_high"] is False
        assert r["broke_low"] is False

    def test_no_break_when_bar_stays_inside_prior_high(self, st):
        st.update("EURUSD", _bar("EURUSD", 0, hi=1.10, lo=1.09))
        st.update("EURUSD", _bar("EURUSD", 1, hi=1.12, lo=1.11))
        st.update("EURUSD", _bar("EURUSD", 2, hi=1.11, lo=1.10))
        # Bar 3 high 1.115 < swing high 1.12 → no break.
        r = st.update("EURUSD", _bar("EURUSD", 3, hi=1.115, lo=1.105))
        assert r["broke_high"] is False


class TestSequentialSwings:
    def test_multiple_swings_in_sequence(self, st):
        # 5-bar trajectory: low — high — low — higher_high — pullback.
        st.update("EURUSD", _bar("EURUSD", 0, hi=1.10, lo=1.09))
        st.update("EURUSD", _bar("EURUSD", 1, hi=1.15, lo=1.11))
        st.update("EURUSD", _bar("EURUSD", 2, hi=1.12, lo=1.05))
        st.update("EURUSD", _bar("EURUSD", 3, hi=1.18, lo=1.10))
        st.update("EURUSD", _bar("EURUSD", 4, hi=1.14, lo=1.11))
        # Bar 1 confirmed as swing high (1.15) on update of bar 2.
        # Bar 2 confirmed as swing low (1.05) on update of bar 3.
        # Bar 3 confirmed as higher swing high (1.18) on update of bar 4.
        assert st.get_last_swing_high("EURUSD") == pytest.approx(1.18)
        assert st.get_last_swing_low("EURUSD") == pytest.approx(1.05)

    def test_state_persists_across_update_calls(self, st):
        st.update("EURUSD", _bar("EURUSD", 0, hi=1.10, lo=1.09))
        st.update("EURUSD", _bar("EURUSD", 1, hi=1.12, lo=1.11))
        st.update("EURUSD", _bar("EURUSD", 2, hi=1.11, lo=1.10))
        # After 3 more bars that form NO new swings, the original 1.12 stays.
        for i in range(3, 6):
            st.update("EURUSD", _bar("EURUSD", i, hi=1.105, lo=1.095))
        assert st.get_last_swing_high("EURUSD") == pytest.approx(1.12)


class TestPerPairIsolation:
    def test_other_pair_unaffected(self, st):
        st.update("EURUSD", _bar("EURUSD", 0, hi=1.10, lo=1.09))
        st.update("EURUSD", _bar("EURUSD", 1, hi=1.12, lo=1.11))
        st.update("EURUSD", _bar("EURUSD", 2, hi=1.11, lo=1.10))
        assert st.get_last_swing_high("EURUSD") == pytest.approx(1.12)
        assert st.get_last_swing_high("GBPUSD") is None

    def test_two_pairs_independent_state(self, st):
        for i, (hi, lo) in enumerate([(1.10, 1.09), (1.12, 1.11), (1.11, 1.10)]):
            st.update("EURUSD", _bar("EURUSD", i, hi=hi, lo=lo))
        for i, (hi, lo) in enumerate(
            [(180.0, 179.0), (185.0, 183.0), (183.0, 181.0)]
        ):
            st.update("GBPJPY", _bar("GBPJPY", i, hi=hi, lo=lo))
        assert st.get_last_swing_high("EURUSD") == pytest.approx(1.12)
        assert st.get_last_swing_high("GBPJPY") == pytest.approx(185.0)


class TestTieCases:
    def test_equal_highs_does_not_form_swing(self, st):
        # mid.high == right.high → strict-greater fails → no swing.
        st.update("EURUSD", _bar("EURUSD", 0, hi=1.10, lo=1.09))
        st.update("EURUSD", _bar("EURUSD", 1, hi=1.12, lo=1.11))
        r = st.update("EURUSD", _bar("EURUSD", 2, hi=1.12, lo=1.10))
        assert r["new_swing_high"] is None
        assert st.get_last_swing_high("EURUSD") is None

    def test_equal_lows_does_not_form_swing(self, st):
        st.update("EURUSD", _bar("EURUSD", 0, hi=1.11, lo=1.10))
        st.update("EURUSD", _bar("EURUSD", 1, hi=1.10, lo=1.08))
        r = st.update("EURUSD", _bar("EURUSD", 2, hi=1.11, lo=1.08))
        assert r["new_swing_low"] is None


class TestStaticHelpers:
    def test_is_break_of_high(self):
        b_break = _bar("EURUSD", 0, hi=1.121, lo=1.10)
        b_no = _bar("EURUSD", 0, hi=1.11, lo=1.10)
        assert SwingTracker.is_break_of_high(b_break, 1.12) is True
        assert SwingTracker.is_break_of_high(b_no, 1.12) is False

    def test_is_break_of_low(self):
        b_break = _bar("EURUSD", 0, hi=1.10, lo=1.079)
        b_no = _bar("EURUSD", 0, hi=1.10, lo=1.085)
        assert SwingTracker.is_break_of_low(b_break, 1.08) is True
        assert SwingTracker.is_break_of_low(b_no, 1.08) is False
