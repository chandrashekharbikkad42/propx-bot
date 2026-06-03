"""Phase 8C — ComplianceEngine tests.

Covers all 7 individual checks, combinations, emergency stop, and the
status reporter.
"""

from __future__ import annotations
from datetime import datetime, timezone

import pytest

from data.news_calendar import NewsEvent, StaticNewsCalendar
from risk.prop_firm.compliance import (
    AccountState,
    ComplianceEngine,
    SAFETY_MARGIN_PCT,
)
from risk.prop_firm.rules import get_rules
from strategy.patterns.base import Direction, Grade, PatternSignal


def _utc_ms(year: int, month: int, day: int, hour: int, minute: int) -> int:
    return int(
        datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp() * 1000
    )


def _sig(symbol="EURUSD", entry=1.10, sl=1.099, tp=1.105, dir=Direction.BUY) -> PatternSignal:
    return PatternSignal(
        pattern_name="P", symbol=symbol, direction=dir,
        entry=entry, sl=sl, tp=tp, confidence=0.7, grade=Grade.A,
        confluences_met=("c1",), bar_time_msc=0,
    )


def _state(
    equity: float = 10_000.0,
    starting: float = 10_000.0,
    daily_start: float = 10_000.0,
    daily_pnl: float = 0.0,
    trades_today: int = 0,
) -> AccountState:
    return AccountState(
        equity=equity, starting_equity=starting, daily_start_equity=daily_start,
        daily_pnl_usd=daily_pnl, trades_today=trades_today,
    )


# Fix a reference time inside the IST window (12:00 UTC == 17:30 IST).
TIME_IN_WINDOW = _utc_ms(2026, 5, 18, 12, 0)        # Monday, in window
TIME_OUTSIDE_WINDOW = _utc_ms(2026, 5, 18, 20, 0)   # Monday, 01:30 IST next day


@pytest.fixture
def engine():
    rules = get_rules("ftmo_2step_challenge")
    return ComplianceEngine(rules)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

class TestIstWindow:
    def test_inside_window_passes(self, engine):
        ok, reason = engine.can_trade(_sig(), TIME_IN_WINDOW, _state())
        assert ok, reason

    def test_outside_window_blocked(self, engine):
        ok, reason = engine.can_trade(_sig(), TIME_OUTSIDE_WINDOW, _state())
        assert not ok and reason == "outside_ist_window"


class TestDailyLossCap:
    def test_close_to_cap_blocked(self, engine):
        # FTMO 2-Step: 5% daily on $10k = $500. 80% margin = $400.
        # Loss of $401 blocks.
        ok, reason = engine.can_trade(
            _sig(), TIME_IN_WINDOW, _state(daily_pnl=-401.0)
        )
        assert not ok and reason == "daily_loss_near_cap"

    def test_below_margin_passes(self, engine):
        ok, reason = engine.can_trade(
            _sig(), TIME_IN_WINDOW, _state(daily_pnl=-100.0)
        )
        assert ok, reason


class TestTotalLossCap:
    def test_total_loss_near_cap_blocked(self, engine):
        # FTMO 2-Step: 10% total = $1000. 80% margin = $800.
        # Equity at $9100 (lost $900) → blocked.
        ok, reason = engine.can_trade(
            _sig(), TIME_IN_WINDOW, _state(equity=9100.0)
        )
        assert not ok and reason == "total_loss_near_cap"


class TestTradeCap:
    def test_at_cap_blocked(self, engine):
        ok, reason = engine.can_trade(
            _sig(), TIME_IN_WINDOW, _state(trades_today=2)
        )
        assert not ok and reason == "daily_trade_cap_reached"

    def test_one_trade_done_passes(self, engine):
        ok, _ = engine.can_trade(_sig(), TIME_IN_WINDOW, _state(trades_today=1))
        assert ok

    def test_custom_max_trades(self):
        rules = get_rules("ftmo_2step_challenge")
        eng = ComplianceEngine(rules, max_trades_per_day=1)
        ok, reason = eng.can_trade(_sig(), TIME_IN_WINDOW, _state(trades_today=1))
        assert not ok and reason == "daily_trade_cap_reached"


class TestNewsBlackout:
    def test_during_blackout_blocked(self):
        rules = get_rules("ftmo_2step_challenge")
        cal = StaticNewsCalendar([NewsEvent(TIME_IN_WINDOW, "USD", "NFP")])
        eng = ComplianceEngine(rules, news_calendar=cal)
        ok, reason = eng.can_trade(_sig("EURUSD"), TIME_IN_WINDOW, _state())
        assert not ok and reason == "news_blackout"

    def test_event_for_other_currency_does_not_block(self):
        rules = get_rules("ftmo_2step_challenge")
        cal = StaticNewsCalendar([NewsEvent(TIME_IN_WINDOW, "GBP", "BoE")])
        eng = ComplianceEngine(rules, news_calendar=cal)
        # EURUSD has no GBP — should pass.
        ok, _ = eng.can_trade(_sig("EURUSD"), TIME_IN_WINDOW, _state())
        assert ok


class TestSlExceedsRoom:
    def test_huge_sl_blocked(self, engine):
        # SL 100 pips on EURUSD with 1 lot ≈ $1000 loss. Daily room ~ $400.
        big_sl_sig = _sig(entry=1.10, sl=1.09, tp=1.13)  # 100-pip SL
        ok, reason = engine.can_trade(big_sl_sig, TIME_IN_WINDOW, _state(), lots=1.0)
        assert not ok and reason == "sl_exceeds_remaining_daily_room"

    def test_small_sl_passes(self, engine):
        # 10-pip SL on 1 lot = $100 — within $400 daily room.
        small_sl = _sig(entry=1.10, sl=1.099, tp=1.103)
        ok, _ = engine.can_trade(small_sl, TIME_IN_WINDOW, _state(), lots=1.0)
        assert ok


class TestLeverageCap:
    def test_too_much_leverage_blocked(self, engine):
        # Pick a tiny SL so the SL-room check passes, then a lot size that
        # blows the leverage cap. FTMO leverage_forex=100, equity $10k →
        # max notional $1M. EURUSD entry 1.10 × 100k × 10 lots = $1.1M >
        # cap. The SL room of $400 - (1.10001 * 100k * 10 * 0.00001 * ...)
        # is large enough not to trip the SL gate.
        tiny_sl_sig = PatternSignal(
            pattern_name="P", symbol="EURUSD", direction=Direction.BUY,
            entry=1.10, sl=1.099999, tp=1.100001, confidence=0.7, grade=Grade.A,
            confluences_met=("c1",), bar_time_msc=0,
        )
        # With 10 lots, worst loss = 0.000001 * 100k * 10 = $1. Tiny — passes SL gate.
        # Notional = 1.10 * 100k * 10 = $1.1M / $10k equity = 110× leverage > 100 → blocked.
        ok, reason = engine.can_trade(
            tiny_sl_sig, TIME_IN_WINDOW, _state(), lots=10.0
        )
        assert not ok and reason == "exceeds_leverage_cap"

    def test_within_leverage_passes(self, engine):
        # 0.05 lots → 1.10 × 100k × 0.05 = $5500 notional / $10k = 0.55 leverage. OK.
        small_sig = _sig(entry=1.10, sl=1.099, tp=1.103)
        ok, _ = engine.can_trade(small_sig, TIME_IN_WINDOW, _state(), lots=0.05)
        assert ok


# ---------------------------------------------------------------------------
# Combined behaviour
# ---------------------------------------------------------------------------

class TestEmergencyStop:
    def test_emergency_blocks_everything(self, engine):
        engine.emergency_stop("manual_intervention")
        ok, reason = engine.can_trade(_sig(), TIME_IN_WINDOW, _state())
        assert not ok
        assert reason.startswith("emergency_stop")

    def test_clear_emergency_restores(self, engine):
        engine.emergency_stop("test")
        engine.clear_emergency()
        ok, _ = engine.can_trade(_sig(), TIME_IN_WINDOW, _state())
        assert ok
        assert engine.emergency_stopped is False


class TestStatusReport:
    def test_report_keys(self, engine):
        report = engine.get_status_report(_state(daily_pnl=-50.0), TIME_IN_WINDOW)
        required = {
            "firm", "equity", "starting_equity",
            "daily_pnl_usd", "daily_cap_usd", "daily_used_pct",
            "total_loss_usd", "total_cap_usd",
            "trades_today", "max_trades_per_day",
            "in_ist_window", "emergency_stopped", "emergency_reason",
        }
        assert required <= set(report.keys())

    def test_in_window_flag(self, engine):
        r1 = engine.get_status_report(_state(), TIME_IN_WINDOW)
        r2 = engine.get_status_report(_state(), TIME_OUTSIDE_WINDOW)
        assert r1["in_ist_window"] is True
        assert r2["in_ist_window"] is False

    def test_daily_used_pct(self, engine):
        # FTMO 2-Step: $500 daily cap. Used $100 → 20%.
        r = engine.get_status_report(_state(daily_pnl=-100.0), TIME_IN_WINDOW)
        assert r["daily_used_pct"] == pytest.approx(20.0)


class TestConstructorValidation:
    def test_zero_safety_margin_rejected(self):
        with pytest.raises(ValueError):
            ComplianceEngine(get_rules("ftmo_2step_challenge"), safety_margin_pct=0)

    def test_safety_margin_above_one_rejected(self):
        with pytest.raises(ValueError):
            ComplianceEngine(get_rules("ftmo_2step_challenge"), safety_margin_pct=1.5)

    def test_zero_max_trades_rejected(self):
        with pytest.raises(ValueError):
            ComplianceEngine(get_rules("ftmo_2step_challenge"), max_trades_per_day=0)


class TestSafetyMarginConstant:
    def test_default_is_eighty_percent(self):
        assert SAFETY_MARGIN_PCT == 0.80
