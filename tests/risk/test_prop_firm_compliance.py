"""ComplianceEngine.can_trade — the 7 kill-switches isolated + combined.

Order of evaluation (matches production):
   0. emergency_stop latch
   1. outside_ist_window
   2. daily_loss_near_cap
   3. total_loss_near_cap
   4. daily_trade_cap_reached
   5. news_blackout
   6. sl_exceeds_remaining_daily_room
   7. exceeds_leverage_cap
"""

from __future__ import annotations
from datetime import datetime, timezone

import pytest

from data.news_calendar import StaticNewsCalendar, NewsEvent
from risk.prop_firm.compliance import (
    AccountState, ComplianceEngine, SAFETY_MARGIN_PCT,
)
from risk.prop_firm.rules import get_rules
from strategy.patterns.base import Direction, Grade, PatternSignal

from tests.risk.fixtures.account_states import make_account


UTC = timezone.utc


def msc(year, month, day, hour, minute=0):
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC)
               .timestamp() * 1000)


# A safely-inside-the-window time (Friday 2026-05-15, 14:00 IST = 08:30 UTC).
INSIDE_MSC = msc(2026, 5, 15, 8, 30)
# Outside the window: 03:00 UTC = 08:30 IST → before 12:30.
OUTSIDE_BEFORE_MSC = msc(2026, 5, 15, 3, 0)
# Outside the window: 18:00 UTC = 23:30 IST → after 22:30.
OUTSIDE_AFTER_MSC = msc(2026, 5, 15, 18, 0)


def _signal(
    *, symbol="EURUSD", entry=1.10000, risk_pts=10.0,
    direction=Direction.BUY,
) -> PatternSignal:
    pt = 0.00001 if symbol != "XAUUSD" else 0.01
    risk = risk_pts * pt
    if direction == Direction.BUY:
        sl, tp = entry - risk, entry + risk * 2.5
    else:
        sl, tp = entry + risk, entry - risk * 2.5
    tp1 = entry + risk if direction == Direction.BUY else entry - risk
    return PatternSignal(
        pattern_name="ASIAN_SWEEP", symbol=symbol, direction=direction,
        entry=entry, sl=sl, tp=tp, confidence=0.9, grade=Grade.A,
        confluences_met=("asian_sweep_low", "LONDON", "bias_neutral", "q9",
                         f"tp1_{tp1:.5f}"),
        bar_time_msc=0,
    )


@pytest.fixture
def rules():
    return get_rules("ftmo_2step_challenge")


@pytest.fixture
def funded_rules():
    return get_rules("ftmo_2step_funded")


@pytest.fixture
def engine(rules):
    return ComplianceEngine(
        rules=rules,
        max_trades_per_day=2,
        ist_window_start="12:30",
        ist_window_end="22:30",
        news_calendar=StaticNewsCalendar([]),
        safety_margin_pct=SAFETY_MARGIN_PCT,
    )


# ===========================================================================
# 0. Constructor invariants
# ===========================================================================

class TestEngineConstructor:
    def test_constructor_safety_margin_zero_raises(self, rules):
        with pytest.raises(ValueError):
            ComplianceEngine(rules=rules, safety_margin_pct=0.0)

    def test_constructor_safety_margin_above_1_raises(self, rules):
        with pytest.raises(ValueError):
            ComplianceEngine(rules=rules, safety_margin_pct=1.1)

    def test_constructor_safety_margin_negative_raises(self, rules):
        with pytest.raises(ValueError):
            ComplianceEngine(rules=rules, safety_margin_pct=-0.1)

    def test_constructor_max_trades_zero_raises(self, rules):
        with pytest.raises(ValueError):
            ComplianceEngine(rules=rules, max_trades_per_day=0)

    def test_constructor_max_trades_negative_raises(self, rules):
        with pytest.raises(ValueError):
            ComplianceEngine(rules=rules, max_trades_per_day=-1)

    def test_default_news_calendar_static(self, rules):
        eng = ComplianceEngine(rules=rules)
        assert isinstance(eng._news, StaticNewsCalendar)

    def test_rules_property(self, engine, rules):
        assert engine.rules is rules

    def test_emergency_default_false(self, engine):
        assert engine.emergency_stopped is False
        assert engine.emergency_reason is None

    def test_default_safety_margin_constant(self):
        assert SAFETY_MARGIN_PCT == 0.80


# ===========================================================================
# 1. emergency_stop latch
# ===========================================================================

class TestEmergencyStop:
    def test_blocks_after_emergency(self, engine):
        engine.emergency_stop("test")
        ok, reason = engine.can_trade(
            _signal(), INSIDE_MSC, make_account(), lots=0.01,
        )
        assert ok is False
        assert reason.startswith("emergency_stop:")

    def test_reason_propagated(self, engine):
        engine.emergency_stop("manual_reason")
        _, reason = engine.can_trade(
            _signal(), INSIDE_MSC, make_account(), lots=0.01,
        )
        assert "manual_reason" in reason

    def test_clear_emergency_re_enables(self, engine):
        engine.emergency_stop("x")
        engine.clear_emergency()
        ok, reason = engine.can_trade(
            _signal(), INSIDE_MSC, make_account(), lots=0.01,
        )
        assert ok is True
        assert reason == "ok"

    def test_clear_resets_reason(self, engine):
        engine.emergency_stop("x")
        engine.clear_emergency()
        assert engine.emergency_reason is None
        assert engine.emergency_stopped is False

    def test_emergency_no_reason_still_blocks(self, engine):
        engine.emergency_stop("")
        ok, _ = engine.can_trade(
            _signal(), INSIDE_MSC, make_account(), lots=0.01,
        )
        assert ok is False


# ===========================================================================
# 2. IST window kill-switch
# ===========================================================================

class TestIstWindowSwitch:
    def test_inside_window_allowed(self, engine):
        ok, _ = engine.can_trade(
            _signal(), INSIDE_MSC, make_account(), lots=0.01,
        )
        assert ok is True

    def test_outside_window_before_blocked(self, engine):
        ok, reason = engine.can_trade(
            _signal(), OUTSIDE_BEFORE_MSC, make_account(), lots=0.01,
        )
        assert ok is False
        assert reason == "outside_ist_window"

    def test_outside_window_after_blocked(self, engine):
        ok, reason = engine.can_trade(
            _signal(), OUTSIDE_AFTER_MSC, make_account(), lots=0.01,
        )
        assert ok is False
        assert reason == "outside_ist_window"

    @pytest.mark.parametrize("hour_utc,minute_utc,expected_ok", [
        (3, 0, False),     # 08:30 IST — outside
        (6, 59, False),    # 12:29 IST — just outside
        (7, 0, True),      # 12:30 IST — inclusive lower
        (8, 0, True),      # 13:30 IST
        (12, 30, True),    # 18:00 IST
        (16, 59, True),    # 22:29 IST
        (17, 0, False),    # 22:30 IST — exclusive upper
        (18, 0, False),    # 23:30 IST — outside
        (23, 30, False),   # 05:00 IST (next day, wraps) — outside
    ])
    def test_ist_window_boundary(self, engine, hour_utc, minute_utc,
                                 expected_ok):
        t = msc(2026, 5, 15, hour_utc, minute_utc)
        ok, _ = engine.can_trade(_signal(), t, make_account(), lots=0.01)
        assert ok is expected_ok

    def test_custom_window_open_24h_effective(self, rules):
        # Use a window 00:00→23:59 — almost always open.
        eng = ComplianceEngine(
            rules=rules, max_trades_per_day=2,
            ist_window_start="00:00", ist_window_end="23:59",
            news_calendar=StaticNewsCalendar([]),
        )
        ok, _ = eng.can_trade(
            _signal(), msc(2026, 5, 15, 18, 0), make_account(), lots=0.01,
        )
        assert ok is True

    def test_custom_window_morning_only(self, rules):
        eng = ComplianceEngine(
            rules=rules, max_trades_per_day=2,
            ist_window_start="08:00", ist_window_end="10:00",
            news_calendar=StaticNewsCalendar([]),
        )
        # 14:00 IST = 08:30 UTC — outside the 08:00-10:00 IST window.
        ok, reason = eng.can_trade(
            _signal(), INSIDE_MSC, make_account(), lots=0.01,
        )
        assert ok is False
        assert reason == "outside_ist_window"


# ===========================================================================
# 3. Daily loss cap
# ===========================================================================

class TestDailyLossCap:
    """FTMO: max_daily_loss_pct = 5 %. Engine halts at 80 % of cap
    → -4 % on starting equity."""

    def test_zero_loss_allowed(self, engine):
        acct = make_account(daily_pnl_usd=0.0)
        ok, _ = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is True

    def test_small_loss_allowed(self, engine):
        # daily_start_equity = 10000, cap = 5% = $500.
        # 80% margin → halt at -$400. Loss of $100 → allowed.
        acct = make_account(daily_pnl_usd=-100.0)
        ok, _ = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is True

    def test_at_safety_margin_blocked(self, engine):
        acct = make_account(daily_pnl_usd=-400.0)  # = 80% of $500 cap
        ok, reason = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is False
        assert reason == "daily_loss_near_cap"

    def test_at_full_cap_blocked(self, engine):
        acct = make_account(daily_pnl_usd=-500.0)
        ok, reason = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is False
        assert reason == "daily_loss_near_cap"

    def test_above_cap_still_blocked(self, engine):
        acct = make_account(daily_pnl_usd=-1_000.0)
        ok, reason = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is False
        assert reason == "daily_loss_near_cap"

    def test_one_cent_under_margin_allowed(self, engine):
        acct = make_account(daily_pnl_usd=-399.99)
        # Use a tiny worst-case so step-6 doesn't pre-empt step-2.
        ok, _ = engine.can_trade(
            _signal(risk_pts=1.0), INSIDE_MSC, acct, lots=1e-6,
        )
        assert ok is True

    def test_one_cent_above_margin_blocked(self, engine):
        acct = make_account(daily_pnl_usd=-400.01)
        ok, reason = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is False
        assert reason == "daily_loss_near_cap"

    @pytest.mark.parametrize("daily_start,loss,expected_ok", [
        (10_000.0, -100.0, True),
        (10_000.0, -399.0, True),
        (10_000.0, -400.0, False),
        (50_000.0, -1_999.0, True),
        (50_000.0, -2_000.0, False),
        (100_000.0, -3_999.99, True),
        (100_000.0, -4_000.0, False),
    ])
    def test_per_starting_equity(self, engine, daily_start, loss, expected_ok):
        acct = make_account(
            equity=daily_start + loss,
            starting_equity=daily_start,
            daily_start_equity=daily_start,
            daily_pnl_usd=loss,
        )
        # Tiny lots so the SL-room check (step 6) doesn't pre-empt step 2.
        ok, _ = engine.can_trade(
            _signal(risk_pts=1.0), INSIDE_MSC, acct, lots=1e-6,
        )
        assert ok is expected_ok

    def test_positive_pnl_not_blocked(self, engine):
        acct = make_account(daily_pnl_usd=1_000.0)
        ok, _ = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is True

    def test_safety_margin_50pct(self, rules):
        eng = ComplianceEngine(
            rules=rules, max_trades_per_day=2,
            ist_window_start="00:00", ist_window_end="23:59",
            news_calendar=StaticNewsCalendar([]),
            safety_margin_pct=0.50,
        )
        # 50% of $500 = $250 halt.
        acct_ok = make_account(daily_pnl_usd=-249.0)
        ok, _ = eng.can_trade(
            _signal(), INSIDE_MSC, acct_ok, lots=0.01,
        )
        assert ok is True
        acct_blk = make_account(daily_pnl_usd=-251.0)
        ok, reason = eng.can_trade(
            _signal(), INSIDE_MSC, acct_blk, lots=0.01,
        )
        assert ok is False
        assert reason == "daily_loss_near_cap"


# ===========================================================================
# 4. Total loss cap
# ===========================================================================

class TestTotalLossCap:
    """FTMO 2-Step: max_total_loss_pct = 10 %, halt at 80% → -8 %."""

    def test_no_drawdown_allowed(self, engine):
        acct = make_account(equity=10_000.0, starting_equity=10_000.0)
        ok, _ = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is True

    def test_4pct_drawdown_allowed(self, engine):
        acct = make_account(equity=9_600.0, starting_equity=10_000.0)
        ok, _ = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is True

    def test_at_safety_margin_blocked(self, engine):
        # 80% of 10% = 8% of $10K = $800 drawdown.
        acct = make_account(equity=9_200.0, starting_equity=10_000.0)
        ok, reason = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is False
        assert reason == "total_loss_near_cap"

    def test_above_total_cap_blocked(self, engine):
        acct = make_account(equity=8_500.0, starting_equity=10_000.0)
        ok, reason = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is False
        assert reason == "total_loss_near_cap"

    def test_one_cent_under_margin(self, engine):
        acct = make_account(equity=9_200.01, starting_equity=10_000.0)
        ok, _ = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is True

    def test_one_cent_over_margin(self, engine):
        acct = make_account(equity=9_199.99, starting_equity=10_000.0)
        ok, reason = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is False
        assert reason == "total_loss_near_cap"

    @pytest.mark.parametrize("equity,starting,expected_ok", [
        (10_000.0, 10_000.0, True),
        (9_500.0, 10_000.0, True),
        (9_200.0, 10_000.0, False),
        (8_000.0, 10_000.0, False),
        (50_000.0, 50_000.0, True),
        (46_000.0, 50_000.0, False),
        (100_000.0, 100_000.0, True),
        (92_000.0, 100_000.0, False),
    ])
    def test_per_starting_equity(self, engine, equity, starting, expected_ok):
        acct = make_account(equity=equity, starting_equity=starting,
                            daily_start_equity=equity)  # no daily DD
        ok, _ = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is expected_ok


# ===========================================================================
# 5. Daily trade-cap kill-switch (max_trades_per_day)
# ===========================================================================

class TestDailyTradeCap:
    def test_zero_trades_allowed(self, engine):
        acct = make_account(trades_today=0)
        ok, _ = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is True

    def test_one_trade_allowed(self, engine):
        acct = make_account(trades_today=1)
        ok, _ = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is True

    def test_two_trades_blocked(self, engine):
        acct = make_account(trades_today=2)
        ok, reason = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is False
        assert reason == "daily_trade_cap_reached"

    def test_three_trades_blocked(self, engine):
        acct = make_account(trades_today=3)
        ok, reason = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is False
        assert reason == "daily_trade_cap_reached"

    @pytest.mark.parametrize("cap,trades,expected_ok", [
        (1, 0, True),
        (1, 1, False),
        (2, 0, True),
        (2, 1, True),
        (2, 2, False),
        (5, 4, True),
        (5, 5, False),
        (10, 9, True),
        (10, 10, False),
    ])
    def test_per_cap(self, rules, cap, trades, expected_ok):
        eng = ComplianceEngine(
            rules=rules, max_trades_per_day=cap,
            ist_window_start="00:00", ist_window_end="23:59",
            news_calendar=StaticNewsCalendar([]),
        )
        acct = make_account(trades_today=trades)
        ok, _ = eng.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is expected_ok


# ===========================================================================
# 6. News blackout
# ===========================================================================

class TestNewsBlackout:
    def test_blocked_inside_news_window(self, rules):
        evt = NewsEvent(time_msc=INSIDE_MSC, currency="USD", title="CPI")
        eng = ComplianceEngine(
            rules=rules, max_trades_per_day=2,
            ist_window_start="00:00", ist_window_end="23:59",
            news_calendar=StaticNewsCalendar([evt]),
        )
        ok, reason = eng.can_trade(
            _signal(symbol="EURUSD"), INSIDE_MSC, make_account(), lots=0.01,
        )
        assert ok is False
        assert reason == "news_blackout"

    def test_allowed_outside_blackout(self, rules):
        # Event 1 hour away from check time
        evt = NewsEvent(time_msc=INSIDE_MSC + 60 * 60 * 1000,
                        currency="USD", title="CPI")
        eng = ComplianceEngine(
            rules=rules, max_trades_per_day=2,
            ist_window_start="00:00", ist_window_end="23:59",
            news_calendar=StaticNewsCalendar([evt]),
        )
        ok, _ = eng.can_trade(
            _signal(symbol="EURUSD"), INSIDE_MSC, make_account(), lots=0.01,
        )
        assert ok is True

    def test_unrelated_currency_not_blocked(self, rules):
        # JPY event, EURUSD signal → currency JPY not in EURUSD → not blocked.
        evt = NewsEvent(time_msc=INSIDE_MSC, currency="JPY", title="BoJ")
        eng = ComplianceEngine(
            rules=rules, max_trades_per_day=2,
            ist_window_start="00:00", ist_window_end="23:59",
            news_calendar=StaticNewsCalendar([evt]),
        )
        ok, _ = eng.can_trade(
            _signal(symbol="EURUSD"), INSIDE_MSC, make_account(), lots=0.01,
        )
        assert ok is True

    @pytest.mark.parametrize("delta_sec,expected_block", [
        (-120, True),     # exactly -2 min
        (-119, True),
        (-121, False),
        (0, True),        # at event time
        (60, True),
        (119, True),
        (120, True),
        (121, False),
        (3600, False),
    ])
    def test_blackout_boundary(self, rules, delta_sec, expected_block):
        # Use 2 min default blackout (both rules.news_blackout_minutes = 2).
        evt_time = INSIDE_MSC + 1_000_000   # offset so we are not at exact midnight noise
        evt = NewsEvent(time_msc=evt_time, currency="USD", title="CPI")
        eng = ComplianceEngine(
            rules=rules, max_trades_per_day=2,
            ist_window_start="00:00", ist_window_end="23:59",
            news_calendar=StaticNewsCalendar([evt]),
        )
        check_time = evt_time + delta_sec * 1000
        ok, _ = eng.can_trade(
            _signal(symbol="EURUSD"), check_time, make_account(), lots=0.01,
        )
        assert (not ok) is expected_block


# ===========================================================================
# 7. SL exceeds remaining daily room
# ===========================================================================

class TestSlExceedsRemainingRoom:
    def test_blocks_huge_sl_distance(self, engine):
        # Signal with very large SL distance vs lots → worst loss > daily room.
        # remaining_daily_room = 80% of $500 + (-0) = $400.
        # worst = risk_distance * 100000 * lots.
        # For risk_distance = 0.01 (100 pts) × 100000 × 1.0 lot = $1000 > $400.
        sig = _signal(symbol="EURUSD", risk_pts=1000.0)
        ok, reason = engine.can_trade(
            sig, INSIDE_MSC, make_account(), lots=1.0,
        )
        assert ok is False
        assert reason == "sl_exceeds_remaining_daily_room"

    def test_allows_small_sl(self, engine):
        # risk_pts=10 → 0.0001 × 100000 × 0.01 = $0.10 worst — tiny.
        ok, _ = engine.can_trade(
            _signal(risk_pts=10.0), INSIDE_MSC, make_account(), lots=0.01,
        )
        assert ok is True

    def test_remaining_room_shrinks_with_prior_loss(self, engine):
        """If we have -$300 PnL today, remaining_room = $400 - $300 = $100."""
        # worst = 0.0001 × 100000 × 1.0 = $10 — still fits inside $100.
        sig = _signal(symbol="EURUSD", risk_pts=10.0)
        acct = make_account(daily_pnl_usd=-300.0)
        ok, _ = engine.can_trade(sig, INSIDE_MSC, acct, lots=1.0)
        assert ok is True

    def test_remaining_room_negative_then_block(self, engine):
        # daily_pnl already past +safety margin: cap kicks in earlier
        # in the chain so we never reach SL-room check.
        sig = _signal(symbol="EURUSD", risk_pts=10.0)
        acct = make_account(daily_pnl_usd=-450.0)  # > 80% cap → blocked by step 2
        ok, reason = engine.can_trade(sig, INSIDE_MSC, acct, lots=1.0)
        assert ok is False
        assert reason == "daily_loss_near_cap"


# ===========================================================================
# 8. Leverage cap
# ===========================================================================

class TestLeverageCap:
    def test_normal_lots_under_leverage(self, engine):
        # FTMO leverage_forex = 100. EURUSD entry 1.10000 × 100000 × 0.01 lot
        # = $1100 notional. equity=$10000 → leverage = 0.11 → fine.
        ok, _ = engine.can_trade(
            _signal(), INSIDE_MSC, make_account(), lots=0.01,
        )
        assert ok is True

    def test_huge_lots_exceeds_leverage(self, engine):
        # 100 lots × 100k × 1.10 = 11M notional, equity $10k → leverage 1100×
        # Far over 100× cap.
        # Use small enough SL so SL-room check passes... but a huge SL distance
        # would block at step 6 first. We need to make SL tiny so step 7 fires.
        # Tiny SL: risk_pts=1 → worst loss = 1e-5*1e5*100 = $100, well within
        # remaining room of $400.
        sig = _signal(symbol="EURUSD", risk_pts=1.0)
        ok, reason = engine.can_trade(
            sig, INSIDE_MSC, make_account(), lots=100.0,
        )
        assert ok is False
        # Could be either room or leverage; the SL room check comes first
        # but with risk=0.00001 × 100000 × 100 = $100 < $400 → room OK.
        assert reason == "exceeds_leverage_cap"

    def test_zero_equity_skips_leverage_check(self, rules):
        """With equity == 0, the leverage check is skipped — but step 3
        (total loss cap) already fires before that."""
        eng = ComplianceEngine(
            rules=rules, max_trades_per_day=2,
            ist_window_start="00:00", ist_window_end="23:59",
            news_calendar=StaticNewsCalendar([]),
        )
        acct = make_account(equity=0.0, starting_equity=10_000.0)
        ok, reason = eng.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is False
        # Step 3 catches this first.
        assert reason == "total_loss_near_cap"


# ===========================================================================
# 9. Order of evaluation — first-failing reason
# ===========================================================================

class TestEvaluationOrder:
    def test_outside_window_short_circuits_loss_cap(self, engine):
        acct = make_account(daily_pnl_usd=-1_000.0)
        ok, reason = engine.can_trade(
            _signal(), OUTSIDE_BEFORE_MSC, acct, lots=0.01,
        )
        assert ok is False
        assert reason == "outside_ist_window"

    def test_loss_cap_short_circuits_trade_count(self, engine):
        acct = make_account(daily_pnl_usd=-1_000.0, trades_today=5)
        ok, reason = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is False
        assert reason == "daily_loss_near_cap"

    def test_total_loss_short_circuits_trade_count(self, engine):
        acct = make_account(equity=8_000.0, starting_equity=10_000.0,
                            trades_today=5)
        ok, reason = engine.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is False
        assert reason == "total_loss_near_cap"

    def test_trade_count_short_circuits_news(self, rules):
        evt = NewsEvent(time_msc=INSIDE_MSC, currency="USD", title="CPI")
        eng = ComplianceEngine(
            rules=rules, max_trades_per_day=2,
            ist_window_start="00:00", ist_window_end="23:59",
            news_calendar=StaticNewsCalendar([evt]),
        )
        acct = make_account(trades_today=2)
        ok, reason = eng.can_trade(_signal(), INSIDE_MSC, acct, lots=0.01)
        assert ok is False
        assert reason == "daily_trade_cap_reached"

    def test_news_short_circuits_sl_room(self, rules):
        evt = NewsEvent(time_msc=INSIDE_MSC, currency="USD", title="CPI")
        eng = ComplianceEngine(
            rules=rules, max_trades_per_day=2,
            ist_window_start="00:00", ist_window_end="23:59",
            news_calendar=StaticNewsCalendar([evt]),
        )
        sig = _signal(symbol="EURUSD", risk_pts=10_000.0)
        ok, reason = eng.can_trade(
            sig, INSIDE_MSC, make_account(), lots=1.0,
        )
        assert ok is False
        assert reason == "news_blackout"

    def test_sl_room_short_circuits_leverage(self, engine):
        # huge SL ⇒ room check fires before leverage.
        sig = _signal(symbol="EURUSD", risk_pts=10_000.0)
        ok, reason = engine.can_trade(
            sig, INSIDE_MSC, make_account(), lots=10.0,
        )
        assert ok is False
        assert reason == "sl_exceeds_remaining_daily_room"


# ===========================================================================
# 10. Per-firm permutation matrix (a fast sweep across rules)
# ===========================================================================

ALL_RULE_KEYS = [
    "ftmo_2step_challenge", "ftmo_2step_verification", "ftmo_2step_funded",
    "ftmo_1step_challenge", "ftmo_1step_funded",
    "the5ers_bootcamp_step1", "the5ers_bootcamp_step2",
    "the5ers_bootcamp_step3", "the5ers_bootcamp_funded",
    "the5ers_hyper_growth_step1", "the5ers_hyper_growth_funded",
    "the5ers_high_stakes_step1", "the5ers_high_stakes_step2",
    "the5ers_high_stakes_funded",
]


@pytest.mark.parametrize("key", ALL_RULE_KEYS)
def test_fresh_account_allowed_for_every_firm(key):
    eng = ComplianceEngine(
        rules=get_rules(key), max_trades_per_day=2,
        ist_window_start="00:00", ist_window_end="23:59",
        news_calendar=StaticNewsCalendar([]),
    )
    ok, _ = eng.can_trade(
        _signal(symbol="EURUSD", risk_pts=10.0),
        INSIDE_MSC, make_account(), lots=0.01,
    )
    assert ok is True


@pytest.mark.parametrize("key", ALL_RULE_KEYS)
def test_emergency_stop_blocks_every_firm(key):
    eng = ComplianceEngine(rules=get_rules(key), max_trades_per_day=2)
    eng.emergency_stop("x")
    ok, _ = eng.can_trade(_signal(), INSIDE_MSC, make_account(), lots=0.01)
    assert ok is False


@pytest.mark.parametrize("key", ALL_RULE_KEYS)
def test_outside_window_blocks_every_firm(key):
    eng = ComplianceEngine(
        rules=get_rules(key), max_trades_per_day=2,
        ist_window_start="12:30", ist_window_end="22:30",
        news_calendar=StaticNewsCalendar([]),
    )
    ok, reason = eng.can_trade(
        _signal(), OUTSIDE_BEFORE_MSC, make_account(), lots=0.01,
    )
    assert ok is False
    assert reason == "outside_ist_window"


@pytest.mark.parametrize("key", ALL_RULE_KEYS)
def test_trade_cap_blocks_every_firm(key):
    eng = ComplianceEngine(
        rules=get_rules(key), max_trades_per_day=2,
        ist_window_start="00:00", ist_window_end="23:59",
        news_calendar=StaticNewsCalendar([]),
    )
    ok, reason = eng.can_trade(
        _signal(), INSIDE_MSC,
        make_account(trades_today=2), lots=0.01,
    )
    assert ok is False
    assert reason == "daily_trade_cap_reached"


# ===========================================================================
# 11. Status report (no-mutation, snapshot-only)
# ===========================================================================

class TestStatusReport:
    def test_keys_present(self, engine):
        rpt = engine.get_status_report(make_account(), INSIDE_MSC)
        for k in [
            "firm", "equity", "starting_equity", "daily_pnl_usd",
            "daily_cap_usd", "daily_used_pct", "total_loss_usd",
            "total_cap_usd", "trades_today", "max_trades_per_day",
            "in_ist_window", "emergency_stopped", "emergency_reason",
        ]:
            assert k in rpt

    def test_firm_name(self, engine):
        rpt = engine.get_status_report(make_account(), INSIDE_MSC)
        assert rpt["firm"] == "FTMO 2-Step Challenge"

    def test_in_window_true(self, engine):
        rpt = engine.get_status_report(make_account(), INSIDE_MSC)
        assert rpt["in_ist_window"] is True

    def test_outside_window_false(self, engine):
        rpt = engine.get_status_report(make_account(), OUTSIDE_BEFORE_MSC)
        assert rpt["in_ist_window"] is False

    def test_emergency_state_in_report(self, engine):
        engine.emergency_stop("manual")
        rpt = engine.get_status_report(make_account(), INSIDE_MSC)
        assert rpt["emergency_stopped"] is True
        assert rpt["emergency_reason"] == "manual"

    def test_daily_pnl_in_report(self, engine):
        rpt = engine.get_status_report(
            make_account(daily_pnl_usd=-50.0), INSIDE_MSC,
        )
        assert rpt["daily_pnl_usd"] == -50.0

    def test_daily_cap_zero_safe_division(self, engine):
        # If daily_start_equity = 0, the cap is 0; report must not div-zero.
        acct = make_account(
            equity=0.0, starting_equity=0.0, daily_start_equity=0.0,
        )
        rpt = engine.get_status_report(acct, INSIDE_MSC)
        assert rpt["daily_cap_usd"] == 0.0
        assert rpt["daily_used_pct"] == 0.0

    def test_trades_today_field(self, engine):
        rpt = engine.get_status_report(
            make_account(trades_today=1), INSIDE_MSC,
        )
        assert rpt["trades_today"] == 1
        assert rpt["max_trades_per_day"] == 2


# ===========================================================================
# 12. AccountState dataclass invariants
# ===========================================================================

class TestAccountState:
    def test_frozen(self):
        acct = make_account()
        with pytest.raises(Exception):
            acct.equity = 9_000.0  # type: ignore[misc]

    def test_default_open_position_count(self):
        acct = AccountState(
            equity=10_000.0, starting_equity=10_000.0,
            daily_start_equity=10_000.0, daily_pnl_usd=0.0, trades_today=0,
        )
        assert acct.open_position_count == 0


# ===========================================================================
# 13. Contract size argument variations
# ===========================================================================

class TestContractSizeArg:
    @pytest.mark.parametrize("ct", [1_000.0, 10_000.0, 100_000.0])
    def test_smaller_contract_size_leverage_easier(self, engine, ct):
        sig = _signal(risk_pts=1.0)
        # Notional = 1.10000 × ct × 1 lot. equity=$10K.
        # For ct=1000: notional=$1100 → leverage 0.11 → OK.
        # For ct=100000: notional=$110000 → leverage 11 → still OK (FTMO=100x).
        ok, _ = engine.can_trade(
            sig, INSIDE_MSC, make_account(), lots=1.0, contract_size=ct,
        )
        assert ok is True

    def test_huge_contract_size_blows_leverage(self, engine):
        sig = _signal(risk_pts=1.0)
        # 1.10000 × 10_000_000 × 1 = 11M notional / 10K equity = 1100× — over 100×.
        ok, reason = engine.can_trade(
            sig, INSIDE_MSC, make_account(), lots=1.0,
            contract_size=10_000_000.0,
        )
        assert ok is False
        assert reason == "exceeds_leverage_cap"


# ===========================================================================
# 14. Custom safety margin
# ===========================================================================

@pytest.mark.parametrize("margin,trip_at", [
    (1.0, -500.0),   # full cap
    (0.80, -400.0),  # default
    (0.50, -250.0),
    (0.20, -100.0),
])
def test_custom_safety_margin_trips_at(rules, margin, trip_at):
    eng = ComplianceEngine(
        rules=rules, max_trades_per_day=2,
        ist_window_start="00:00", ist_window_end="23:59",
        news_calendar=StaticNewsCalendar([]),
        safety_margin_pct=margin,
    )
    tiny_sig = _signal(risk_pts=1.0)
    # trip_at is the threshold (a NEGATIVE PnL). 1 cent shallower
    # (i.e., closer to zero) → allowed. 1 cent deeper → blocked.
    acct = make_account(daily_pnl_usd=trip_at + 0.01)
    ok, _ = eng.can_trade(tiny_sig, INSIDE_MSC, acct, lots=1e-6)
    assert ok is True
    acct = make_account(daily_pnl_usd=trip_at - 0.01)
    ok, reason = eng.can_trade(tiny_sig, INSIDE_MSC, acct, lots=1e-6)
    assert ok is False
    assert reason == "daily_loss_near_cap"


# ===========================================================================
# 15. Direction-agnostic checks (BUY vs SELL)
# ===========================================================================

@pytest.mark.parametrize("direction", [Direction.BUY, Direction.SELL])
class TestDirectionAgnostic:
    def test_fresh_account(self, engine, direction):
        ok, _ = engine.can_trade(
            _signal(direction=direction), INSIDE_MSC, make_account(),
            lots=0.01,
        )
        assert ok is True

    def test_loss_cap(self, engine, direction):
        acct = make_account(daily_pnl_usd=-1_000.0)
        ok, _ = engine.can_trade(
            _signal(direction=direction), INSIDE_MSC, acct, lots=0.01,
        )
        assert ok is False

    def test_trade_cap(self, engine, direction):
        acct = make_account(trades_today=2)
        ok, _ = engine.can_trade(
            _signal(direction=direction), INSIDE_MSC, acct, lots=0.01,
        )
        assert ok is False


# ===========================================================================
# 16. Per-symbol smoke (news + leverage)
# ===========================================================================

@pytest.mark.parametrize("symbol", [
    "EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF",
    "AUDCHF", "AUDNZD", "XAUUSD",
])
class TestPerSymbol:
    def test_fresh_account_passes(self, engine, symbol):
        # XAUUSD has entry≠1.1; pick a reasonable price.
        entry = 100.0 if symbol == "XAUUSD" else 1.10000
        sig = _signal(symbol=symbol, entry=entry, risk_pts=10.0)
        ok, _ = engine.can_trade(
            sig, INSIDE_MSC, make_account(), lots=0.01,
        )
        assert ok is True

    def test_usd_event_blocks_usd_pair(self, rules, symbol):
        evt = NewsEvent(time_msc=INSIDE_MSC, currency="USD", title="CPI")
        eng = ComplianceEngine(
            rules=rules, max_trades_per_day=2,
            ist_window_start="00:00", ist_window_end="23:59",
            news_calendar=StaticNewsCalendar([evt]),
        )
        entry = 100.0 if symbol == "XAUUSD" else 1.10000
        sig = _signal(symbol=symbol, entry=entry, risk_pts=10.0)
        ok, reason = eng.can_trade(sig, INSIDE_MSC, make_account(), lots=0.01)
        if "USD" in symbol:
            assert ok is False
            assert reason == "news_blackout"
        else:
            assert ok is True

    def test_eur_event_blocks_only_eur(self, rules, symbol):
        evt = NewsEvent(time_msc=INSIDE_MSC, currency="EUR", title="ECB")
        eng = ComplianceEngine(
            rules=rules, max_trades_per_day=2,
            ist_window_start="00:00", ist_window_end="23:59",
            news_calendar=StaticNewsCalendar([evt]),
        )
        entry = 100.0 if symbol == "XAUUSD" else 1.10000
        sig = _signal(symbol=symbol, entry=entry, risk_pts=10.0)
        ok, _ = eng.can_trade(sig, INSIDE_MSC, make_account(), lots=0.01)
        if "EUR" in symbol:
            assert ok is False
        else:
            assert ok is True


# ===========================================================================
# 17. Big sweep matrix — many account configurations × rule firms
# ===========================================================================

ACCOUNT_CONFIGS = [
    dict(daily_pnl_usd=0.0, trades_today=0,
         equity=10_000.0, starting_equity=10_000.0,
         daily_start_equity=10_000.0),
    dict(daily_pnl_usd=-50.0, trades_today=0,
         equity=9_950.0, starting_equity=10_000.0,
         daily_start_equity=10_000.0),
    dict(daily_pnl_usd=-100.0, trades_today=1,
         equity=9_900.0, starting_equity=10_000.0,
         daily_start_equity=10_000.0),
    dict(daily_pnl_usd=200.0, trades_today=1,
         equity=10_200.0, starting_equity=10_000.0,
         daily_start_equity=10_000.0),
]


@pytest.mark.parametrize("cfg", ACCOUNT_CONFIGS)
@pytest.mark.parametrize("key", ALL_RULE_KEYS)
def test_safe_account_passes_all_firms(cfg, key):
    eng = ComplianceEngine(
        rules=get_rules(key), max_trades_per_day=2,
        ist_window_start="00:00", ist_window_end="23:59",
        news_calendar=StaticNewsCalendar([]),
    )
    sig = _signal(symbol="EURUSD", risk_pts=10.0)
    ok, _ = eng.can_trade(sig, INSIDE_MSC, make_account(**cfg), lots=0.01)
    assert ok is True
