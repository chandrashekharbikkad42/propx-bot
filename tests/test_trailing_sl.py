"""Phase 8C — TrailingStopLoss tests (swing trail + spread-hour protection)."""

from __future__ import annotations
from datetime import datetime, timezone, timedelta

import pytest

from config.griff_config import GriffConfig
from data.bar_aggregator import Bar
from execution.order import Side
from execution.position import Position, PositionState
from risk.trailing_sl import TrailingStopLoss, pip_size
from strategy.swing_tracker import SwingTracker


HOUR_MS = 3_600_000

# 15:00 UTC — well outside the [20:45, 22:00) spread window.
NEUTRAL_TIME = datetime(2026, 5, 17, 15, 0, tzinfo=timezone.utc)


def _bar(symbol, idx, *, hi, lo, o=None, c=None):
    o = o if o is not None else (hi + lo) / 2
    c = c if c is not None else (hi + lo) / 2
    return Bar(
        symbol=symbol, time_msc=idx * HOUR_MS,
        open=o, high=hi, low=lo, close=c, volume=1, spread_mean=0.0,
    )


def _long_pos(entry=1.1000, sl=1.0950, tp=1.1100, pid="p1"):
    return Position(
        position_id=pid, side=Side.BUY, lots=0.1,
        entry_price=entry, entry_time_msc=0,
        sl_price=sl, tp_price=tp, max_hold_until_msc=10 ** 18,
        state=PositionState.OPEN,
    )


def _short_pos(entry=1.1000, sl=1.1050, tp=1.0900, pid="p1"):
    return Position(
        position_id=pid, side=Side.SELL, lots=0.1,
        entry_price=entry, entry_time_msc=0,
        sl_price=sl, tp_price=tp, max_hold_until_msc=10 ** 18,
        state=PositionState.OPEN,
    )


def _with_sl(position: Position, new_sl: float) -> Position:
    """Return a fresh Position with sl_price replaced. Position is frozen."""
    return Position(
        position_id=position.position_id, side=position.side, lots=position.lots,
        entry_price=position.entry_price, entry_time_msc=position.entry_time_msc,
        sl_price=new_sl, tp_price=position.tp_price,
        max_hold_until_msc=position.max_hold_until_msc, state=position.state,
        signal_type=position.signal_type, session=position.session,
    )


def _seed_swing_low(st, pair, swing_lo, flanking_hi=1.10, flanking_lo=1.099,
                    base_hi=1.10):
    """Feed 3 bars that confirm a swing low at `swing_lo`."""
    st.update(pair, _bar(pair, 0, hi=flanking_hi, lo=flanking_lo))
    st.update(pair, _bar(pair, 1, hi=base_hi, lo=swing_lo))
    st.update(pair, _bar(pair, 2, hi=flanking_hi, lo=flanking_lo))


def _seed_swing_high(st, pair, swing_hi, flanking_hi=1.10, flanking_lo=1.099,
                     base_lo=1.099):
    """Feed 3 bars that confirm a swing high at `swing_hi`."""
    st.update(pair, _bar(pair, 0, hi=flanking_hi, lo=flanking_lo))
    st.update(pair, _bar(pair, 1, hi=swing_hi, lo=base_lo))
    st.update(pair, _bar(pair, 2, hi=flanking_hi, lo=flanking_lo))


@pytest.fixture
def st():
    return SwingTracker()


@pytest.fixture
def trail(st):
    return TrailingStopLoss(st)


# ---------------------------------------------------------------------------
# pip_size util
# ---------------------------------------------------------------------------

class TestPipSize:
    def test_jpy_pair_pip_is_0_01(self):
        assert pip_size("EURJPY") == 0.01
        assert pip_size("USDJPY") == 0.01
        assert pip_size("AUDJPY") == 0.01

    def test_non_jpy_pair_pip_is_0_0001(self):
        assert pip_size("EURUSD") == 0.0001
        assert pip_size("GBPUSD") == 0.0001
        assert pip_size("AUDUSD") == 0.0001


# ---------------------------------------------------------------------------
# Structural trail
# ---------------------------------------------------------------------------

class TestStructuralTrail:
    def test_long_trail_moves_up_on_higher_swing_low(self, st, trail):
        _seed_swing_low(st, "EURUSD", swing_lo=1.0980)
        pos = _long_pos(entry=1.1000, sl=1.0950)
        last_bar = _bar("EURUSD", 2, hi=1.10, lo=1.099)
        new_sl = trail.update(pos, last_bar, NEUTRAL_TIME)
        # target = 1.0980 - 0.0002 = 1.0978
        assert new_sl == pytest.approx(1.0978)

    def test_short_trail_moves_down_on_lower_swing_high(self, st, trail):
        _seed_swing_high(st, "EURUSD", swing_hi=1.1020)
        pos = _short_pos(entry=1.1000, sl=1.1050)
        last_bar = _bar("EURUSD", 2, hi=1.10, lo=1.099)
        new_sl = trail.update(pos, last_bar, NEUTRAL_TIME)
        # target = 1.1020 + 0.0002 = 1.1022
        assert new_sl == pytest.approx(1.1022)

    def test_no_change_when_swing_unchanged(self, st, trail):
        _seed_swing_low(st, "EURUSD", swing_lo=1.0980)
        pos = _long_pos(entry=1.1000, sl=1.0950)
        last_bar = _bar("EURUSD", 2, hi=1.10, lo=1.099)
        first = trail.update(pos, last_bar, NEUTRAL_TIME)
        pos2 = _with_sl(pos, first)
        second = trail.update(pos2, last_bar, NEUTRAL_TIME)
        assert second is None

    def test_long_trail_never_moves_down_on_lower_swing(self, st, trail):
        # First swing low at 1.0980 → trail to 1.0978.
        _seed_swing_low(st, "EURUSD", swing_lo=1.0980)
        pos = _long_pos(entry=1.1000, sl=1.0950)
        last_bar = _bar("EURUSD", 2, hi=1.10, lo=1.099)
        trail.update(pos, last_bar, NEUTRAL_TIME)
        # Now feed a LOWER swing low at 1.0900 (price retraced).
        st.update("EURUSD", _bar("EURUSD", 3, hi=1.10, lo=1.099))
        st.update("EURUSD", _bar("EURUSD", 4, hi=1.099, lo=1.0900))  # new trough
        st.update("EURUSD", _bar("EURUSD", 5, hi=1.10, lo=1.099))
        # Tracker now reports last swing low = 1.0900.
        assert st.get_last_swing_low("EURUSD") == pytest.approx(1.0900)
        # But trail should NOT drop SL back down.
        pos2 = _with_sl(pos, 1.0978)
        result = trail.update(pos2, _bar("EURUSD", 5, hi=1.10, lo=1.099), NEUTRAL_TIME)
        assert result is None

    def test_short_trail_never_moves_up_on_higher_swing(self, st, trail):
        _seed_swing_high(st, "EURUSD", swing_hi=1.1020)
        pos = _short_pos(entry=1.1000, sl=1.1050)
        last_bar = _bar("EURUSD", 2, hi=1.10, lo=1.099)
        trail.update(pos, last_bar, NEUTRAL_TIME)
        st.update("EURUSD", _bar("EURUSD", 3, hi=1.10, lo=1.099))
        st.update("EURUSD", _bar("EURUSD", 4, hi=1.1100, lo=1.099))  # higher peak
        st.update("EURUSD", _bar("EURUSD", 5, hi=1.10, lo=1.099))
        assert st.get_last_swing_high("EURUSD") == pytest.approx(1.1100)
        pos2 = _with_sl(pos, 1.1022)
        result = trail.update(pos2, _bar("EURUSD", 5, hi=1.10, lo=1.099), NEUTRAL_TIME)
        assert result is None

    def test_trail_respects_two_pip_offset(self, st, trail):
        # Custom offset → verify it's applied.
        cfg = GriffConfig(trail_offset_pips=5.0)
        trail = TrailingStopLoss(st, cfg)
        _seed_swing_low(st, "EURUSD", swing_lo=1.0980)
        pos = _long_pos(entry=1.1000, sl=1.0950)
        new_sl = trail.update(
            pos, _bar("EURUSD", 2, hi=1.10, lo=1.099), NEUTRAL_TIME
        )
        # 5-pip offset = 0.0005 → target = 1.0980 - 0.0005 = 1.0975
        assert new_sl == pytest.approx(1.0975)

    def test_no_anchor_no_change(self, st, trail):
        # SwingTracker has zero swings yet.
        pos = _long_pos(entry=1.1000, sl=1.0950)
        result = trail.update(
            pos, _bar("EURUSD", 0, hi=1.10, lo=1.099), NEUTRAL_TIME
        )
        assert result is None

    def test_wick_break_then_higher_low_triggers_trail(self, st, trail):
        # Setup: prior swing high 1.1020, then a swing low at 1.0960.
        st.update("EURUSD", _bar("EURUSD", 0, hi=1.10, lo=1.099))
        st.update("EURUSD", _bar("EURUSD", 1, hi=1.1020, lo=1.0980))  # peak 1.1020
        st.update("EURUSD", _bar("EURUSD", 2, hi=1.10, lo=1.0960))    # trough 1.0960
        st.update("EURUSD", _bar("EURUSD", 3, hi=1.10, lo=1.099))
        # Bar 4 wicks ABOVE 1.1020 (break of swing high).
        wick_bar = _bar("EURUSD", 4, hi=1.1025, lo=1.099)
        st.update("EURUSD", wick_bar)
        # Now a NEW higher swing low at 1.0985 forms (above the prior 1.0960).
        st.update("EURUSD", _bar("EURUSD", 5, hi=1.10, lo=1.099))
        st.update("EURUSD", _bar("EURUSD", 6, hi=1.10, lo=1.0985))    # new trough
        last_bar = _bar("EURUSD", 7, hi=1.10, lo=1.099)
        st.update("EURUSD", last_bar)
        # Tracker should now report last swing low at 1.0985 (most recent).
        assert st.get_last_swing_low("EURUSD") == pytest.approx(1.0985)
        pos = _long_pos(entry=1.1010, sl=1.0950)
        new_sl = trail.update(pos, last_bar, NEUTRAL_TIME)
        # target = 1.0985 - 0.0002 = 1.0983
        assert new_sl == pytest.approx(1.0983)

    def test_multi_position_isolation(self, st, trail):
        _seed_swing_low(st, "EURUSD", swing_lo=1.0980)
        pos_a = _long_pos(entry=1.1000, sl=1.0950, pid="A")
        pos_b = _long_pos(entry=1.1000, sl=1.0940, pid="B")
        last_bar = _bar("EURUSD", 2, hi=1.10, lo=1.099)
        sl_a = trail.update(pos_a, last_bar, NEUTRAL_TIME)
        sl_b = trail.update(pos_b, last_bar, NEUTRAL_TIME)
        # Both trail to the same structural level (same swing data, separate state).
        assert sl_a == pytest.approx(1.0978)
        assert sl_b == pytest.approx(1.0978)
        # Once both positions are caught up, no further change for either.
        pos_a2 = _with_sl(pos_a, 1.0978)
        pos_b2 = _with_sl(pos_b, 1.0978)
        assert trail.update(pos_a2, last_bar, NEUTRAL_TIME) is None
        assert trail.update(pos_b2, last_bar, NEUTRAL_TIME) is None


# ---------------------------------------------------------------------------
# Spread-hour protection
# ---------------------------------------------------------------------------

# Rollover 21:00, window [20:45, 22:00).
IN_WINDOW_TIME = datetime(2026, 5, 17, 20, 50, tzinfo=timezone.utc)
JUST_BEFORE_WINDOW = datetime(2026, 5, 17, 20, 44, tzinfo=timezone.utc)
EXACT_WINDOW_START = datetime(2026, 5, 17, 20, 45, tzinfo=timezone.utc)
EXACT_WINDOW_END = datetime(2026, 5, 17, 22, 0, tzinfo=timezone.utc)
AFTER_WINDOW = datetime(2026, 5, 17, 22, 1, tzinfo=timezone.utc)


class TestSpreadProtection:
    def test_in_window_widens_long_sl(self, st, trail):
        _seed_swing_low(st, "EURUSD", swing_lo=1.0980)
        pos = _long_pos(entry=1.1000, sl=1.0950)
        last_bar = _bar("EURUSD", 2, hi=1.10, lo=1.099)
        new_sl = trail.update(pos, last_bar, IN_WINDOW_TIME)
        # structural = 1.0978; widen for EURUSD = 40 pips = 0.0040
        # effective = 1.0978 - 0.0040 = 1.0938
        assert new_sl == pytest.approx(1.0938)

    def test_in_window_widens_short_sl(self, st, trail):
        _seed_swing_high(st, "EURUSD", swing_hi=1.1020)
        pos = _short_pos(entry=1.1000, sl=1.1050)
        last_bar = _bar("EURUSD", 2, hi=1.10, lo=1.099)
        new_sl = trail.update(pos, last_bar, IN_WINDOW_TIME)
        # structural = 1.1022; widen = 0.0040 → 1.1062
        assert new_sl == pytest.approx(1.1062)

    def test_window_reverts_after_end(self, st, trail):
        _seed_swing_low(st, "EURUSD", swing_lo=1.0980)
        pos = _long_pos(entry=1.1000, sl=1.0950)
        last_bar = _bar("EURUSD", 2, hi=1.10, lo=1.099)
        # In-window: SL widens to 1.0938.
        widened = trail.update(pos, last_bar, IN_WINDOW_TIME)
        assert widened == pytest.approx(1.0938)
        # After window: revert to structural 1.0978.
        pos2 = _with_sl(pos, widened)
        reverted = trail.update(pos2, last_bar, AFTER_WINDOW)
        assert reverted == pytest.approx(1.0978)

    def test_in_profit_long_skips_widening(self, st, trail):
        # Construct a long position where structural SL is at/above entry —
        # i.e., already locked-in profit. Need flanking lows ABOVE swing_lo.
        st.update("EURUSD", _bar("EURUSD", 0, hi=1.1015, lo=1.1010))
        st.update("EURUSD", _bar("EURUSD", 1, hi=1.1015, lo=1.1005))  # mid trough
        st.update("EURUSD", _bar("EURUSD", 2, hi=1.1015, lo=1.1010))
        assert st.get_last_swing_low("EURUSD") == pytest.approx(1.1005)
        pos = _long_pos(entry=1.1000, sl=1.0900)
        last_bar = _bar("EURUSD", 2, hi=1.1015, lo=1.1010)
        new_sl = trail.update(pos, last_bar, IN_WINDOW_TIME)
        # target = 1.1005 - 0.0002 = 1.1003 > entry 1.1000 → in profit, no widen.
        assert new_sl == pytest.approx(1.1003)

    def test_in_profit_short_skips_widening(self, st, trail):
        # Mirror — flanking highs BELOW swing_hi.
        st.update("EURUSD", _bar("EURUSD", 0, hi=1.0990, lo=1.0985))
        st.update("EURUSD", _bar("EURUSD", 1, hi=1.0995, lo=1.0985))  # mid peak
        st.update("EURUSD", _bar("EURUSD", 2, hi=1.0990, lo=1.0985))
        assert st.get_last_swing_high("EURUSD") == pytest.approx(1.0995)
        pos = _short_pos(entry=1.1000, sl=1.1100)
        last_bar = _bar("EURUSD", 2, hi=1.0990, lo=1.0985)
        new_sl = trail.update(pos, last_bar, IN_WINDOW_TIME)
        # target = 1.0995 + 0.0002 = 1.0997 < entry 1.1000 → in profit, no widen.
        assert new_sl == pytest.approx(1.0997)

    def test_window_start_boundary_inclusive(self, st, trail):
        _seed_swing_low(st, "EURUSD", swing_lo=1.0980)
        pos = _long_pos(entry=1.1000, sl=1.0950)
        last_bar = _bar("EURUSD", 2, hi=1.10, lo=1.099)
        new_sl = trail.update(pos, last_bar, EXACT_WINDOW_START)
        # Widening should be active exactly at the start.
        assert new_sl == pytest.approx(1.0938)

    def test_window_end_boundary_exclusive(self, st, trail):
        _seed_swing_low(st, "EURUSD", swing_lo=1.0980)
        pos = _long_pos(entry=1.1000, sl=1.0950)
        last_bar = _bar("EURUSD", 2, hi=1.10, lo=1.099)
        new_sl = trail.update(pos, last_bar, EXACT_WINDOW_END)
        # Exactly at end → out of window → no widening, structural only.
        assert new_sl == pytest.approx(1.0978)

    def test_just_before_window_no_widening(self, st, trail):
        _seed_swing_low(st, "EURUSD", swing_lo=1.0980)
        pos = _long_pos(entry=1.1000, sl=1.0950)
        last_bar = _bar("EURUSD", 2, hi=1.10, lo=1.099)
        new_sl = trail.update(pos, last_bar, JUST_BEFORE_WINDOW)
        # 20:44 UTC — out of window.
        assert new_sl == pytest.approx(1.0978)


# ---------------------------------------------------------------------------
# Explicit hooks
# ---------------------------------------------------------------------------

class TestExplicitHooks:
    def test_apply_spread_protection_explicit_long(self, st, trail):
        _seed_swing_low(st, "EURUSD", swing_lo=1.0980)
        pos = _long_pos(entry=1.1000, sl=1.0950)
        last_bar = _bar("EURUSD", 2, hi=1.10, lo=1.099)
        trail.update(pos, last_bar, NEUTRAL_TIME)  # seed state, structural=1.0978
        widened = trail.apply_spread_protection(
            _with_sl(pos, 1.0978), datetime(2026, 5, 17, 21, 0, tzinfo=timezone.utc)
        )
        assert widened == pytest.approx(1.0938)

    def test_revert_spread_protection_explicit(self, st, trail):
        _seed_swing_low(st, "EURUSD", swing_lo=1.0980)
        pos = _long_pos(entry=1.1000, sl=1.0950)
        last_bar = _bar("EURUSD", 2, hi=1.10, lo=1.099)
        trail.update(pos, last_bar, IN_WINDOW_TIME)  # widens, sets spread_active
        reverted = trail.revert_spread_protection(_with_sl(pos, 1.0938))
        assert reverted == pytest.approx(1.0978)

    def test_revert_when_no_protection_active_returns_current_sl(self, st, trail):
        pos = _long_pos(entry=1.1000, sl=1.0950)
        result = trail.revert_spread_protection(pos)
        # No prior state — should return position.sl_price unchanged.
        assert result == pytest.approx(1.0950)

    def test_apply_without_prior_update_raises(self, trail):
        # Position never seen by update() → no pair cached.
        pos = _long_pos()
        with pytest.raises(KeyError):
            trail.apply_spread_protection(
                pos, datetime(2026, 5, 17, 21, 0, tzinfo=timezone.utc)
            )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

class TestCalcWidenPips:
    def test_known_pair_returns_mapped_value(self, trail):
        assert trail._calc_widen_pips("EURUSD") == 40
        assert trail._calc_widen_pips("NZDJPY") == 60

    def test_unknown_pair_returns_default(self, trail):
        assert trail._calc_widen_pips("USDCAD") == 50  # default
