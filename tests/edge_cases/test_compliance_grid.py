"""Phase-5 / Compliance Grid — parameter-grid extensions for the compliance
engine. Each test asserts one gate's outcome on a generated point; the grid
multiplies through every (firm, equity, pnl, trades, symbol) combination
that matters.
"""

from __future__ import annotations
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from data.news_calendar import NewsEvent, StaticNewsCalendar
from risk.house_money import HouseMoneyManager
from risk.prop_firm.compliance import AccountState, ComplianceEngine
from risk.prop_firm.rules import (
    PropFirmRules, RULES_DB, get_rules, list_rule_keys,
)
from strategy.patterns.base import (
    Direction, Grade, MarketContext, PatternSignal,
)
from tests.risk.fixtures.account_states import make_account


UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")


def _ist_ms(year, month, day, hour, minute=0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=IST)
               .astimezone(UTC).timestamp() * 1000)


_INSIDE = _ist_ms(2026, 5, 14, 14)
_OUTSIDE = _ist_ms(2026, 5, 14, 6)


# ===========================================================================
# 1. EACH (firm, equity) BASE CASE
# ===========================================================================

@pytest.mark.parametrize("key", list_rule_keys())
@pytest.mark.parametrize("equity", [1_000.0, 10_000.0, 100_000.0])
def test_firm_x_equity_grid(key, equity, signal_factory):
    rules = get_rules(key)
    ce = ComplianceEngine(rules=rules,
                          news_calendar=StaticNewsCalendar([]),
                          safety_margin_pct=0.80)
    acct = make_account(equity=equity, starting_equity=equity,
                          daily_start_equity=equity)
    sig = signal_factory()
    ok, reason = ce.can_trade(sig, _INSIDE, acct)
    assert isinstance(ok, bool)


# ===========================================================================
# 2. PnL × SAFETY-MARGIN GRID
# ===========================================================================

@pytest.mark.parametrize("safety", [0.5, 0.6, 0.7, 0.8, 0.9])
@pytest.mark.parametrize("loss_ratio", [0.0, 0.25, 0.50, 0.75, 0.99, 1.0])
def test_pnl_x_safety_grid(safety, loss_ratio, signal_factory):
    """daily cap = $500. With safety s, threshold = $500 * s.
    PnL at threshold * loss_ratio of margin should NOT trip if < 1.0."""
    rules = get_rules("ftmo_2step_challenge")
    ce = ComplianceEngine(rules=rules,
                          news_calendar=StaticNewsCalendar([]),
                          safety_margin_pct=safety)
    threshold = 500.0 * safety
    pnl = -threshold * loss_ratio
    acct = make_account(daily_pnl_usd=pnl)
    ok, reason = ce.can_trade(signal_factory(), _INSIDE, acct)
    if loss_ratio < 1.0:
        # Just under the margin → daily-loss gate doesn't trip.
        assert reason != "daily_loss_near_cap"
    else:
        # At or past margin → tripped.
        assert reason == "daily_loss_near_cap"


# ===========================================================================
# 3. TOTAL DD × CURRENT EQUITY GRID
# ===========================================================================

@pytest.mark.parametrize("safety", [0.5, 0.8, 1.0])
@pytest.mark.parametrize("equity_loss", [
    0.0, 100.0, 500.0, 799.0, 800.0, 801.0, 1000.0, 5000.0,
])
def test_total_dd_grid(safety, equity_loss, signal_factory):
    rules = get_rules("ftmo_2step_challenge")
    ce = ComplianceEngine(rules=rules,
                          news_calendar=StaticNewsCalendar([]),
                          safety_margin_pct=safety)
    acct = make_account(equity=10_000.0 - equity_loss)
    sig = signal_factory()
    ok, reason = ce.can_trade(sig, _INSIDE, acct)
    threshold = 1000.0 * safety
    if equity_loss >= threshold:
        assert ok is False
        assert reason == "total_loss_near_cap"
    else:
        # Below threshold — total gate doesn't fire.
        assert reason != "total_loss_near_cap"


# ===========================================================================
# 4. MAX-TRADES GRID
# ===========================================================================

@pytest.mark.parametrize("cap", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 5, 10])
def test_max_trades_grid(compliance_factory, signal_factory,
                          account_factory, cap, count):
    ce = compliance_factory(max_trades_per_day=cap)
    acct = account_factory(trades_today=count)
    sig = signal_factory()
    ok, reason = ce.can_trade(sig, _INSIDE, acct)
    if count >= cap:
        assert ok is False
        assert reason == "daily_trade_cap_reached"


# ===========================================================================
# 5. NEWS WINDOW × PAIR GRID
# ===========================================================================

@pytest.mark.parametrize("currency", ["USD", "EUR", "GBP", "JPY", "CAD",
                                       "CHF", "AUD", "NZD"])
@pytest.mark.parametrize("pair", [
    "XAUUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "AUDCHF", "AUDNZD",
])
def test_news_currency_x_pair_grid(currency, pair):
    ev = NewsEvent(time_msc=_INSIDE, currency=currency,
                   title="X", impact="HIGH")
    cal = StaticNewsCalendar([ev])
    expected = currency in pair
    assert cal.is_blackout(pair, _INSIDE) is expected


# ===========================================================================
# 6. NEWS WINDOW MINUTE × OFFSET GRID
# ===========================================================================

@pytest.mark.parametrize("window_min", [0, 1, 2, 5, 10, 30, 60])
@pytest.mark.parametrize("offset_sec", [0, 30, 60, 120, 300, 600, 3600])
def test_news_window_x_offset_grid(window_min, offset_sec):
    cal = StaticNewsCalendar([
        NewsEvent(time_msc=_INSIDE, currency="USD", title="X", impact="HIGH"),
    ])
    msc = _INSIDE + offset_sec * 1000
    expected = offset_sec <= window_min * 60
    assert cal.is_news_blackout("EURUSD", msc,
                                  window_min=window_min) is expected


# ===========================================================================
# 7. HOUSE-MONEY GRID
# ===========================================================================

@pytest.mark.parametrize("grade", [Grade.A, Grade.B])
@pytest.mark.parametrize("equity", [10_000.0, 50_000.0, 100_000.0])
@pytest.mark.parametrize("pnl", [-1000.0, -100.0, 0.0, 100.0, 1000.0, 10_000.0])
def test_house_money_grid(grade, equity, pnl):
    hm = HouseMoneyManager()
    alloc = hm.calc_trade_risk(grade, equity=equity,
                                 todays_pnl_usd=pnl,
                                 trade_number_today=2)
    if pnl > 0:
        assert alloc.mode == "HOUSE_MONEY"
    else:
        assert alloc.mode == "DEFENSIVE"


# ===========================================================================
# 8. HOUSE-MONEY TRADE 1 — STANDARD
# ===========================================================================

@pytest.mark.parametrize("grade", [Grade.A, Grade.B])
@pytest.mark.parametrize("equity", [1000.0, 10_000.0, 100_000.0])
@pytest.mark.parametrize("pnl", [-100.0, 0.0, 100.0])
def test_house_money_trade1_grid(grade, equity, pnl):
    hm = HouseMoneyManager()
    alloc = hm.calc_trade_risk(grade, equity=equity,
                                 todays_pnl_usd=pnl,
                                 trade_number_today=1)
    assert alloc.mode == "STANDARD"


# ===========================================================================
# 9. CAN_TRADE — COMBINED GATE MATRIX
# ===========================================================================

@pytest.mark.parametrize("ist_inside", [True, False])
@pytest.mark.parametrize("pnl", [0.0, -200.0, -500.0])
@pytest.mark.parametrize("equity", [10_000.0, 9_500.0, 9_000.0])
@pytest.mark.parametrize("trades", [0, 1, 2])
def test_combined_gates_matrix(compliance_factory, signal_factory,
                                 ist_inside, pnl, equity, trades):
    ce = compliance_factory(max_trades_per_day=2)
    msc = _INSIDE if ist_inside else _OUTSIDE
    acct = make_account(equity=equity, starting_equity=10_000.0,
                          daily_start_equity=10_000.0,
                          daily_pnl_usd=pnl, trades_today=trades)
    sig = signal_factory()
    ok, reason = ce.can_trade(sig, msc, acct)
    assert isinstance(ok, bool)
    assert isinstance(reason, str)


# ===========================================================================
# 10. IST WINDOW EVERY HALF-HOUR
# ===========================================================================

@pytest.mark.parametrize("hour", list(range(24)))
@pytest.mark.parametrize("minute", [0, 30])
def test_ist_window_every_half_hour(compliance_factory, signal_factory,
                                      fresh_account, hour, minute):
    ce = compliance_factory()
    msc = _ist_ms(2026, 5, 14, hour, minute)
    sig = signal_factory()
    ok, reason = ce.can_trade(sig, msc, fresh_account)
    minutes = hour * 60 + minute
    inside_window = 12 * 60 + 30 <= minutes < 22 * 60 + 30
    if not inside_window:
        assert reason == "outside_ist_window"


# ===========================================================================
# 11. RULES_DB COMPLETENESS
# ===========================================================================

@pytest.mark.parametrize("key", list_rule_keys())
def test_rules_db_each_key_returns_rule(key):
    assert get_rules(key) is not None


@pytest.mark.parametrize("key", list_rule_keys())
def test_rules_have_metals_leverage(key):
    r = get_rules(key)
    assert r.leverage_metals > 0
    assert r.leverage_forex >= r.leverage_metals


# ===========================================================================
# 12. SIGNAL × PAIR GRID
# ===========================================================================

@pytest.mark.parametrize("symbol", [
    "XAUUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "AUDCHF", "AUDNZD",
])
@pytest.mark.parametrize("direction", [Direction.BUY, Direction.SELL])
def test_compliance_each_pair_each_direction(compliance_factory,
                                              signal_factory, fresh_account,
                                              symbol, direction):
    ce = compliance_factory()
    sig = signal_factory(symbol=symbol, direction=direction)
    ok, reason = ce.can_trade(sig, _INSIDE, fresh_account)
    assert isinstance(ok, bool)


# ===========================================================================
# 13. EMERGENCY STOP + GATE PRIORITY GRID
# ===========================================================================

@pytest.mark.parametrize("ist_inside", [True, False])
@pytest.mark.parametrize("trades", [0, 2, 5])
def test_emergency_overrides_all_gates(compliance_factory, signal_factory,
                                         account_factory, ist_inside, trades):
    ce = compliance_factory()
    ce.emergency_stop("kill")
    acct = account_factory(trades_today=trades)
    msc = _INSIDE if ist_inside else _OUTSIDE
    ok, reason = ce.can_trade(signal_factory(), msc, acct)
    assert ok is False
    assert reason.startswith("emergency_stop")


# ===========================================================================
# 14. SAFETY MARGIN INVARIANTS
# ===========================================================================

@pytest.mark.parametrize("safety", [0.01, 0.5, 0.8, 0.99, 1.0])
def test_safety_margin_accepts_valid_range(compliance_factory, safety):
    ce = compliance_factory(safety_margin_pct=safety)
    assert ce is not None


@pytest.mark.parametrize("bad_safety", [-0.1, 0.0, 1.01, 2.0])
def test_safety_margin_rejects_invalid(compliance_factory, bad_safety):
    with pytest.raises(ValueError):
        compliance_factory(safety_margin_pct=bad_safety)


# ===========================================================================
# 15. MAX-TRADES VALIDATION
# ===========================================================================

@pytest.mark.parametrize("bad", [0, -1, -100])
def test_max_trades_rejects_invalid(compliance_factory, bad):
    with pytest.raises(ValueError):
        compliance_factory(max_trades_per_day=bad)


# ===========================================================================
# 16. HOUSE-MONEY CAP INVARIANT
# ===========================================================================

@pytest.mark.parametrize("equity", [1_000.0, 10_000.0, 100_000.0])
@pytest.mark.parametrize("win_size", [1.0, 10.0, 100.0, 1000.0, 10_000.0])
def test_house_money_capped_at_2x_base(equity, win_size):
    hm = HouseMoneyManager()
    alloc = hm.calc_trade_risk(Grade.A, equity=equity,
                                 todays_pnl_usd=win_size,
                                 trade_number_today=2)
    base = 1.0  # DEFAULT_BASE_RISK_PCT[A]
    assert alloc.final_risk_pct <= base * 2.0 + 1e-9


# ===========================================================================
# 17. DEFENSIVE FLOOR INVARIANT
# ===========================================================================

@pytest.mark.parametrize("loss", [-1.0, -10.0, -100.0, -1000.0])
@pytest.mark.parametrize("equity", [1_000.0, 10_000.0])
def test_defensive_equal_to_half_base(loss, equity):
    hm = HouseMoneyManager()
    alloc = hm.calc_trade_risk(Grade.A, equity=equity,
                                 todays_pnl_usd=loss,
                                 trade_number_today=2)
    base = 1.0
    assert alloc.final_risk_pct == pytest.approx(base * 0.5)


# ===========================================================================
# 18. STATUS REPORT GRID
# ===========================================================================

@pytest.mark.parametrize("equity,pnl,trades", [
    (10_000.0, 0.0, 0),
    (10_000.0, -100.0, 1),
    (10_000.0, -500.0, 2),
    (9_500.0, -200.0, 1),
    (15_000.0, 5_000.0, 2),
])
def test_status_report_basic(compliance_factory, equity, pnl, trades):
    ce = compliance_factory()
    acct = make_account(equity=equity, starting_equity=10_000.0,
                          daily_start_equity=10_000.0,
                          daily_pnl_usd=pnl, trades_today=trades)
    rep = ce.get_status_report(acct, _INSIDE)
    assert rep["trades_today"] == trades
    assert rep["daily_pnl_usd"] == pnl


# ===========================================================================
# 19. AccountState IMMUTABILITY
# ===========================================================================

def test_account_state_replace_via_dataclass():
    from dataclasses import replace
    a = make_account()
    b = replace(a, equity=5000.0)
    assert b.equity == 5000.0
    assert a.equity == 10_000.0


# ===========================================================================
# 20. CAN_TRADE MUST RETURN (bool, str)
# ===========================================================================

@pytest.mark.parametrize("eq,pnl,trades", [
    (10_000.0, 0.0, 0),
    (10_000.0, -100.0, 1),
    (10_000.0, -500.0, 2),
    (9_000.0, -300.0, 1),
    (8_000.0, -1000.0, 0),
])
def test_can_trade_return_shape(compliance_factory, signal_factory,
                                  eq, pnl, trades):
    ce = compliance_factory()
    acct = make_account(equity=eq, starting_equity=10_000.0,
                          daily_start_equity=10_000.0,
                          daily_pnl_usd=pnl, trades_today=trades)
    ok, reason = ce.can_trade(signal_factory(), _INSIDE, acct)
    assert isinstance(ok, bool)
    assert isinstance(reason, str)
    assert len(reason) > 0


# ===========================================================================
# 21. NEWS BLACKOUT — IMPACT MATRIX
# ===========================================================================

@pytest.mark.parametrize("impact,blocks", [
    ("HIGH", True), ("MEDIUM", False), ("LOW", False),
])
def test_news_blackout_impact_matrix(impact, blocks):
    ev = NewsEvent(time_msc=_INSIDE, currency="USD",
                   title="X", impact=impact)
    cal = StaticNewsCalendar([ev])
    assert cal.is_blackout("EURUSD", _INSIDE) is blocks


# ===========================================================================
# 22. NEWS CALENDAR EMPTY × VARIOUS TIMES
# ===========================================================================

@pytest.mark.parametrize("offset", [0, 100, 1_000_000, -1_000_000])
def test_empty_calendar_never_blocks(offset):
    cal = StaticNewsCalendar([])
    assert cal.is_blackout("EURUSD", _INSIDE + offset) is False
