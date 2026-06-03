"""risk.circuit_breakers.CircuitBreakers — daily DD + loss-streak + session gate."""

from __future__ import annotations
from datetime import datetime, timezone

import pytest

from execution.position import CloseReason, Position, PositionState
from execution.order import Side
from risk.circuit_breakers import (
    CircuitBreakerState, CircuitBreakers,
)
from utils.session import SessionLabel


UTC = timezone.utc


def msc(year, month, day, hour, minute=0):
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC)
               .timestamp() * 1000)


def _pos(*, pnl_usd: float = 0.0, exit_msc: int | None = None,
         entry_msc: int = msc(2026, 5, 15, 10, 0)) -> Position:
    return Position(
        position_id="P", side=Side.BUY, lots=0.1,
        entry_price=1.10000, entry_time_msc=entry_msc,
        sl_price=1.09990, tp_price=1.10025,
        max_hold_until_msc=entry_msc + 3600 * 1000,
        state=PositionState.CLOSED,
        exit_price=1.10010, exit_time_msc=exit_msc,
        close_reason=CloseReason.SL_HIT if pnl_usd < 0 else CloseReason.TP_HIT,
        pnl_pts=10.0, pnl_usd=pnl_usd,
    )


# Time helpers — pick LONDON (UTC 08–12) for the "in session" cases.
LONDON_MSC = msc(2026, 5, 15, 8, 0)         # LONDON
OVERLAP_MSC = msc(2026, 5, 15, 13, 0)       # LONDON_NY_OVERLAP
NY_MSC = msc(2026, 5, 15, 17, 0)            # NY (h=17 → not in OFF range)
# Actually session_for: h<8=ASIAN, 8-11=LONDON, 12-15=OVERLAP, 16-20=NY, 21+=OFF.
NY_MSC = msc(2026, 5, 15, 18, 0)
OFF_MSC = msc(2026, 5, 15, 22, 0)
ASIAN_MSC = msc(2026, 5, 15, 3, 0)


# ===========================================================================
# 1. State default values
# ===========================================================================

class TestState:
    def test_defaults(self):
        s = CircuitBreakerState()
        assert s.daily_pnl_usd == 0.0
        assert s.daily_starting_equity == 0.0
        assert s.consecutive_losses == 0
        assert s.streak_pause_until_msc == 0
        assert s.daily_cap_hit is False
        assert s.current_day_utc is None


# ===========================================================================
# 2. Constructor parameter capture
# ===========================================================================

class TestConstructor:
    def test_default_daily_cap(self):
        b = CircuitBreakers()
        assert b._daily_cap_pct == 0.02

    def test_default_streak(self):
        b = CircuitBreakers()
        assert b._streak_threshold == 3

    def test_default_pause_ms(self):
        b = CircuitBreakers()
        assert b._pause_ms == 30 * 60 * 1000

    @pytest.mark.parametrize("pct", [0.01, 0.02, 0.03, 0.05])
    def test_custom_daily_cap_stored(self, pct):
        b = CircuitBreakers(daily_cap_pct=pct)
        assert b._daily_cap_pct == pct

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
    def test_custom_streak_threshold(self, n):
        b = CircuitBreakers(streak_threshold=n)
        assert b._streak_threshold == n

    @pytest.mark.parametrize("m", [1, 5, 15, 30, 60])
    def test_custom_pause_minutes(self, m):
        b = CircuitBreakers(pause_minutes=m)
        assert b._pause_ms == m * 60 * 1000


# ===========================================================================
# 3. Day rollover initialises state
# ===========================================================================

class TestDayRollover:
    def test_first_call_rolls_day(self):
        b = CircuitBreakers()
        b.can_trade(LONDON_MSC, 10_000.0)
        assert b.state.current_day_utc == "2026-05-15"
        assert b.state.daily_starting_equity == 10_000.0

    def test_same_day_does_not_reset(self):
        b = CircuitBreakers()
        b.can_trade(LONDON_MSC, 10_000.0)
        b.state.daily_pnl_usd = -50.0
        b.can_trade(LONDON_MSC + 1000, 9_950.0)  # same day
        assert b.state.daily_pnl_usd == -50.0

    def test_new_day_resets_pnl(self):
        b = CircuitBreakers()
        b.can_trade(LONDON_MSC, 10_000.0)
        b.state.daily_pnl_usd = -50.0
        b.state.consecutive_losses = 2
        b.can_trade(LONDON_MSC + 24 * 3600 * 1000, 9_950.0)
        assert b.state.daily_pnl_usd == 0.0
        assert b.state.consecutive_losses == 0

    def test_new_day_resets_streak_pause(self):
        b = CircuitBreakers()
        b.can_trade(LONDON_MSC, 10_000.0)
        b.state.streak_pause_until_msc = LONDON_MSC + 60 * 60 * 1000
        b.can_trade(LONDON_MSC + 24 * 3600 * 1000, 9_950.0)
        assert b.state.streak_pause_until_msc == 0

    def test_new_day_resets_daily_cap_hit(self):
        b = CircuitBreakers()
        b.can_trade(LONDON_MSC, 10_000.0)
        b.state.daily_cap_hit = True
        b.can_trade(LONDON_MSC + 24 * 3600 * 1000, 9_950.0)
        assert b.state.daily_cap_hit is False


# ===========================================================================
# 4. can_trade — session filter
# ===========================================================================

class TestSessionFilter:
    def test_london_allowed(self):
        b = CircuitBreakers()
        ok, reason = b.can_trade(LONDON_MSC, 10_000.0)
        assert ok is True
        assert reason == "ok"

    def test_overlap_allowed(self):
        b = CircuitBreakers()
        ok, _ = b.can_trade(OVERLAP_MSC, 10_000.0)
        assert ok is True

    def test_ny_allowed(self):
        b = CircuitBreakers()
        ok, _ = b.can_trade(NY_MSC, 10_000.0)
        assert ok is True

    def test_asian_blocked(self):
        b = CircuitBreakers()
        ok, reason = b.can_trade(ASIAN_MSC, 10_000.0)
        assert ok is False
        assert reason.startswith("session_blocked:")
        assert SessionLabel.ASIAN.value in reason

    def test_off_blocked(self):
        b = CircuitBreakers()
        ok, reason = b.can_trade(OFF_MSC, 10_000.0)
        assert ok is False
        assert "OFF" in reason


# ===========================================================================
# 5. record_trade_close — winning + losing
# ===========================================================================

class TestRecordTradeClose:
    def test_winning_trade_accumulates_pnl(self):
        b = CircuitBreakers()
        b.can_trade(LONDON_MSC, 10_000.0)
        b.record_trade_close(_pos(pnl_usd=10.0, exit_msc=LONDON_MSC + 60_000))
        assert b.state.daily_pnl_usd == 10.0

    def test_losing_trade_increments_streak(self):
        b = CircuitBreakers()
        b.can_trade(LONDON_MSC, 10_000.0)
        b.record_trade_close(_pos(pnl_usd=-10.0, exit_msc=LONDON_MSC + 60_000))
        assert b.state.consecutive_losses == 1

    def test_winning_trade_resets_streak(self):
        b = CircuitBreakers()
        b.can_trade(LONDON_MSC, 10_000.0)
        b.record_trade_close(_pos(pnl_usd=-10.0, exit_msc=LONDON_MSC + 60_000))
        b.record_trade_close(_pos(pnl_usd=-10.0, exit_msc=LONDON_MSC + 120_000))
        assert b.state.consecutive_losses == 2
        b.record_trade_close(_pos(pnl_usd=15.0, exit_msc=LONDON_MSC + 180_000))
        assert b.state.consecutive_losses == 0

    def test_break_even_resets_streak(self):
        """pnl == 0 is NOT a loss; streak resets."""
        b = CircuitBreakers()
        b.can_trade(LONDON_MSC, 10_000.0)
        b.record_trade_close(_pos(pnl_usd=-10.0, exit_msc=LONDON_MSC + 60_000))
        b.record_trade_close(_pos(pnl_usd=0.0, exit_msc=LONDON_MSC + 120_000))
        assert b.state.consecutive_losses == 0

    def test_pnl_none_treated_as_zero(self):
        b = CircuitBreakers()
        b.can_trade(LONDON_MSC, 10_000.0)
        pos = Position(
            position_id="P", side=Side.BUY, lots=0.1,
            entry_price=1.10000, entry_time_msc=LONDON_MSC,
            sl_price=1.09990, tp_price=1.10025,
            max_hold_until_msc=LONDON_MSC + 3600 * 1000,
            state=PositionState.CLOSED,
            pnl_usd=None,
        )
        b.record_trade_close(pos)
        assert b.state.daily_pnl_usd == 0.0


# ===========================================================================
# 6. Loss-streak pause
# ===========================================================================

class TestLossStreakPause:
    def test_n_losses_trigger_pause(self):
        b = CircuitBreakers(streak_threshold=3, pause_minutes=30)
        b.can_trade(LONDON_MSC, 10_000.0)
        for i in range(3):
            b.record_trade_close(
                _pos(pnl_usd=-10.0, exit_msc=LONDON_MSC + (i + 1) * 60_000),
            )
        assert b.state.streak_pause_until_msc > 0

    def test_pause_anchored_to_last_close(self):
        b = CircuitBreakers(streak_threshold=3, pause_minutes=30)
        b.can_trade(LONDON_MSC, 10_000.0)
        last_exit = LONDON_MSC + 5 * 60_000
        for i in range(3):
            b.record_trade_close(
                _pos(pnl_usd=-10.0, exit_msc=LONDON_MSC + (i + 1) * 60_000),
            )
        # The 3rd loss closes at LONDON_MSC + 3*60_000.
        expected = (LONDON_MSC + 3 * 60_000) + 30 * 60 * 1000
        assert b.state.streak_pause_until_msc == expected

    def test_two_losses_dont_pause(self):
        b = CircuitBreakers(streak_threshold=3, pause_minutes=30)
        b.can_trade(LONDON_MSC, 10_000.0)
        for i in range(2):
            b.record_trade_close(
                _pos(pnl_usd=-10.0, exit_msc=LONDON_MSC + (i + 1) * 60_000),
            )
        assert b.state.streak_pause_until_msc == 0

    def test_during_pause_block(self):
        b = CircuitBreakers(streak_threshold=2, pause_minutes=10)
        b.can_trade(LONDON_MSC, 10_000.0)
        for i in range(2):
            b.record_trade_close(
                _pos(pnl_usd=-10.0, exit_msc=LONDON_MSC + (i + 1) * 60_000),
            )
        ok, reason = b.can_trade(LONDON_MSC + 3 * 60_000, 10_000.0)
        assert ok is False
        assert reason == "loss_streak_pause"

    def test_after_pause_resume(self):
        b = CircuitBreakers(streak_threshold=2, pause_minutes=10)
        b.can_trade(LONDON_MSC, 10_000.0)
        for i in range(2):
            b.record_trade_close(
                _pos(pnl_usd=-10.0, exit_msc=LONDON_MSC + (i + 1) * 60_000),
            )
        # 11 minutes after last close → pause is over.
        ok, _ = b.can_trade(
            LONDON_MSC + 2 * 60_000 + 11 * 60_000, 10_000.0,
        )
        assert ok is True

    @pytest.mark.parametrize("threshold,losses,expect_pause", [
        (3, 2, False), (3, 3, True), (3, 4, True),
        (5, 3, False), (5, 4, False), (5, 5, True),
        (2, 1, False), (2, 2, True),
    ])
    def test_streak_threshold_matrix(self, threshold, losses, expect_pause):
        b = CircuitBreakers(streak_threshold=threshold, pause_minutes=10)
        b.can_trade(LONDON_MSC, 10_000.0)
        for i in range(losses):
            b.record_trade_close(
                _pos(pnl_usd=-10.0, exit_msc=LONDON_MSC + (i + 1) * 60_000),
            )
        if expect_pause:
            assert b.state.streak_pause_until_msc > 0
        else:
            assert b.state.streak_pause_until_msc == 0


# ===========================================================================
# 7. Daily DD cap
# ===========================================================================

class TestDailyDdCap:
    def test_not_hit_under_cap(self):
        b = CircuitBreakers(daily_cap_pct=0.02)
        b.can_trade(LONDON_MSC, 10_000.0)
        b.record_trade_close(_pos(pnl_usd=-50.0,
                                  exit_msc=LONDON_MSC + 60_000))
        ok, _ = b.can_trade(LONDON_MSC + 120_000, 9_950.0)
        assert ok is True

    def test_hit_at_cap(self):
        b = CircuitBreakers(daily_cap_pct=0.02)
        b.can_trade(LONDON_MSC, 10_000.0)
        # cap = $200 (2% of $10K). One -$200 loss.
        b.record_trade_close(_pos(pnl_usd=-200.0,
                                  exit_msc=LONDON_MSC + 60_000))
        assert b.state.daily_cap_hit is True

    def test_can_trade_blocks_after_cap(self):
        b = CircuitBreakers(daily_cap_pct=0.02)
        b.can_trade(LONDON_MSC, 10_000.0)
        b.record_trade_close(_pos(pnl_usd=-300.0,
                                  exit_msc=LONDON_MSC + 60_000))
        ok, reason = b.can_trade(LONDON_MSC + 120_000, 9_700.0)
        assert ok is False
        assert reason == "daily_cap_hit"

    @pytest.mark.parametrize("cap,loss,hit", [
        (0.01, -50.0, False),
        (0.01, -100.0, True),
        (0.02, -150.0, False),
        (0.02, -200.0, True),
        (0.05, -400.0, False),
        (0.05, -500.0, True),
    ])
    def test_per_cap_matrix(self, cap, loss, hit):
        b = CircuitBreakers(daily_cap_pct=cap)
        b.can_trade(LONDON_MSC, 10_000.0)
        b.record_trade_close(_pos(pnl_usd=loss,
                                  exit_msc=LONDON_MSC + 60_000))
        assert b.state.daily_cap_hit is hit

    def test_cap_with_zero_starting_equity(self):
        b = CircuitBreakers()
        b.can_trade(LONDON_MSC, 0.0)
        # Cap = $0 → check `cap_usd > 0 and daily_pnl_usd <= -cap_usd`
        # → cap_usd == 0 → check skipped.
        b.record_trade_close(_pos(pnl_usd=-100.0,
                                  exit_msc=LONDON_MSC + 60_000))
        assert b.state.daily_cap_hit is False


# ===========================================================================
# 8. Order of evaluation — daily cap > streak pause > session
# ===========================================================================

class TestEvaluationOrder:
    def test_daily_cap_short_circuits_streak(self):
        b = CircuitBreakers(daily_cap_pct=0.02, streak_threshold=2,
                            pause_minutes=10)
        b.can_trade(LONDON_MSC, 10_000.0)
        b.record_trade_close(_pos(pnl_usd=-150.0,
                                  exit_msc=LONDON_MSC + 60_000))
        b.record_trade_close(_pos(pnl_usd=-60.0,
                                  exit_msc=LONDON_MSC + 120_000))
        # Both daily cap AND streak pause should be active.
        ok, reason = b.can_trade(LONDON_MSC + 180_000, 9_790.0)
        assert ok is False
        assert reason == "daily_cap_hit"

    def test_streak_short_circuits_session(self):
        b = CircuitBreakers(streak_threshold=2, pause_minutes=10)
        b.can_trade(LONDON_MSC, 10_000.0)
        for i in range(2):
            b.record_trade_close(
                _pos(pnl_usd=-10.0, exit_msc=LONDON_MSC + (i + 1) * 60_000),
            )
        # In OFF session, AND streak pause active.
        ok, reason = b.can_trade(LONDON_MSC + 3 * 60_000, 10_000.0)
        assert ok is False
        assert reason == "loss_streak_pause"


# ===========================================================================
# 9. Per-hour smoke (all 24 UTC hours)
# ===========================================================================

EXPECTED_BY_HOUR = {
    h: (h >= 7 and h < 21)
    for h in range(24)
}


@pytest.mark.parametrize("hour", list(range(24)))
def test_session_per_hour(hour):
    b = CircuitBreakers()
    ok, _ = b.can_trade(msc(2026, 5, 15, hour), 10_000.0)
    assert ok is EXPECTED_BY_HOUR[hour]


# ===========================================================================
# 10. Multi-day sequence
# ===========================================================================

class TestMultiDay:
    def test_streak_resets_across_days(self):
        b = CircuitBreakers(streak_threshold=3, pause_minutes=10)
        b.can_trade(LONDON_MSC, 10_000.0)
        b.record_trade_close(_pos(pnl_usd=-10.0,
                                  exit_msc=LONDON_MSC + 60_000))
        b.record_trade_close(_pos(pnl_usd=-10.0,
                                  exit_msc=LONDON_MSC + 120_000))
        assert b.state.consecutive_losses == 2
        # Next day rollover.
        next_day = LONDON_MSC + 24 * 3600 * 1000
        b.can_trade(next_day, 9_980.0)
        assert b.state.consecutive_losses == 0

    def test_daily_cap_resets_across_days(self):
        b = CircuitBreakers(daily_cap_pct=0.01)
        b.can_trade(LONDON_MSC, 10_000.0)
        b.record_trade_close(_pos(pnl_usd=-200.0,
                                  exit_msc=LONDON_MSC + 60_000))
        assert b.state.daily_cap_hit is True
        b.can_trade(LONDON_MSC + 24 * 3600 * 1000, 9_800.0)
        assert b.state.daily_cap_hit is False

    def test_daily_pnl_resets_across_days(self):
        b = CircuitBreakers()
        b.can_trade(LONDON_MSC, 10_000.0)
        b.record_trade_close(_pos(pnl_usd=50.0,
                                  exit_msc=LONDON_MSC + 60_000))
        b.can_trade(LONDON_MSC + 24 * 3600 * 1000, 10_050.0)
        assert b.state.daily_pnl_usd == 0.0


# ===========================================================================
# 11. exit_time_msc fallback to entry_time_msc when None
# ===========================================================================

def test_streak_pause_uses_entry_when_exit_none():
    b = CircuitBreakers(streak_threshold=1, pause_minutes=5)
    b.can_trade(LONDON_MSC, 10_000.0)
    pos = _pos(pnl_usd=-10.0, exit_msc=None,
               entry_msc=LONDON_MSC + 60_000)
    b.record_trade_close(pos)
    expected = (LONDON_MSC + 60_000) + 5 * 60 * 1000
    assert b.state.streak_pause_until_msc == expected


# ===========================================================================
# 12. ACTIVE_SESSIONS set
# ===========================================================================

def test_active_sessions_constant():
    from risk.circuit_breakers import _ACTIVE_SESSIONS
    assert _ACTIVE_SESSIONS == {
        SessionLabel.LONDON,
        SessionLabel.LONDON_NY_OVERLAP,
        SessionLabel.NY,
    }


# ===========================================================================
# 13. Accumulating pnl across trades
# ===========================================================================

class TestPnlAccumulation:
    def test_three_trades_sum(self):
        b = CircuitBreakers()
        b.can_trade(LONDON_MSC, 10_000.0)
        b.record_trade_close(_pos(pnl_usd=-50.0,
                                  exit_msc=LONDON_MSC + 60_000))
        b.record_trade_close(_pos(pnl_usd=100.0,
                                  exit_msc=LONDON_MSC + 120_000))
        b.record_trade_close(_pos(pnl_usd=-25.0,
                                  exit_msc=LONDON_MSC + 180_000))
        assert b.state.daily_pnl_usd == pytest.approx(25.0)

    @pytest.mark.parametrize("pnls,expected", [
        ([10.0, 20.0, -5.0], 25.0),
        ([-10.0, -20.0, 5.0], -25.0),
        ([0.0, 0.0, 0.0], 0.0),
        ([1.0] * 10, 10.0),
        ([-1.0] * 10, -10.0),
    ])
    def test_pnl_sums(self, pnls, expected):
        b = CircuitBreakers()
        b.can_trade(LONDON_MSC, 10_000.0)
        for i, p in enumerate(pnls):
            b.record_trade_close(_pos(pnl_usd=p,
                                      exit_msc=LONDON_MSC + (i + 1) * 60_000))
        assert b.state.daily_pnl_usd == pytest.approx(expected)


# ===========================================================================
# 14. Pause-minutes variations
# ===========================================================================

@pytest.mark.parametrize("pause_min", [1, 5, 10, 30, 60, 120])
def test_pause_duration(pause_min):
    b = CircuitBreakers(streak_threshold=1, pause_minutes=pause_min)
    b.can_trade(LONDON_MSC, 10_000.0)
    b.record_trade_close(_pos(pnl_usd=-10.0,
                              exit_msc=LONDON_MSC + 60_000))
    expected = (LONDON_MSC + 60_000) + pause_min * 60 * 1000
    assert b.state.streak_pause_until_msc == expected
