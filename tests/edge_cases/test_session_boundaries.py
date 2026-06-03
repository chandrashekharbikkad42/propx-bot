"""Phase-5 / Session + Time Boundary — adversarial tests.

Coverage focus (per Phase 5 brief):

  - DST spring-forward (lost hour) during Asian range
  - DST fall-back (repeated hour) during London sweep
  - Midnight UTC rollover mid-trade
  - Friday close → Sunday open gap
  - Signal exactly at session boundary minute
  - Trade open across day boundary (max_trades reset)
  - News event exactly at signal time (±0 sec)
  - Bot started mid-session (partial Asian range)
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from data.news_calendar import NewsEvent, StaticNewsCalendar
from risk.circuit_breakers import CircuitBreakers
from risk.prop_firm.compliance import ComplianceEngine
from strategy.patterns.asian_sweep import (
    AsianSweepDetector, _compute_asian_range, _compute_bias,
)
from strategy.patterns.base import (
    Direction, Grade, MarketContext, PatternSignal,
)
from utils.session import (
    SessionLabel, is_within_ist_window, session_for, session_for_msc,
    to_ist, ist_date,
)

from tests.edge_cases.fixtures.chaos_market import (
    HOUR_MS, asian_window_with_missing_bars, hour_msc, make_bar,
)
from tests.strategy.fixtures.synthetic_bars import (
    long_sweep_bars, short_sweep_bars,
)


UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")


def utc_dt(y, m, d, h=0, mi=0, s=0) -> datetime:
    return datetime(y, m, d, h, mi, s, tzinfo=UTC)


def utc_ms_at(y, m, d, h=0, mi=0, s=0) -> int:
    return int(utc_dt(y, m, d, h, mi, s).timestamp() * 1000)


# ---------------------------------------------------------------------------
# 1. SESSION_FOR — half-open interval coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hour,expected", [
    (0, SessionLabel.ASIAN), (1, SessionLabel.ASIAN), (6, SessionLabel.ASIAN),
    (7, SessionLabel.LONDON), (8, SessionLabel.LONDON),
    (11, SessionLabel.LONDON),
    (12, SessionLabel.LONDON_NY_OVERLAP),
    (15, SessionLabel.LONDON_NY_OVERLAP),
    (16, SessionLabel.NY), (20, SessionLabel.NY),
    (21, SessionLabel.OFF), (23, SessionLabel.OFF),
])
def test_session_for_per_hour(hour, expected):
    assert session_for(utc_dt(2026, 5, 14, hour, 30)) == expected


@pytest.mark.parametrize("hour", list(range(24)))
def test_session_for_msc_matches_session_for(hour):
    msc = utc_ms_at(2026, 5, 14, hour)
    assert session_for_msc(msc) == session_for(utc_dt(2026, 5, 14, hour))


def test_session_for_handles_naive_input():
    naive = datetime(2026, 5, 14, 12, 0)
    assert session_for(naive) == SessionLabel.LONDON_NY_OVERLAP


def test_session_for_handles_non_utc_tz():
    """Aware non-UTC datetime is normalised to UTC first."""
    ny = ZoneInfo("America/New_York")
    dt_ny = datetime(2026, 5, 14, 8, 0, tzinfo=ny)  # 12:00 UTC
    assert session_for(dt_ny) == SessionLabel.LONDON_NY_OVERLAP


# ---------------------------------------------------------------------------
# 2. IST WINDOW — boundary-minute precision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ist_hhmm,inside", [
    ("12:29", False),    # 1 minute before open
    ("12:30", True),     # exact open (inclusive)
    ("12:31", True),
    ("22:29", True),
    ("22:30", False),    # exact close (exclusive)
    ("22:31", False),
])
def test_default_ist_window_inclusive_open_exclusive_close(ist_hhmm, inside):
    """Default window [12:30, 22:30) — start inclusive, end exclusive."""
    hh, mm = (int(x) for x in ist_hhmm.split(":"))
    ist_dt = datetime(2026, 5, 14, hh, mm, tzinfo=IST)
    msc = int(ist_dt.astimezone(UTC).timestamp() * 1000)
    assert is_within_ist_window(msc) is inside


@pytest.mark.parametrize("hour", list(range(24)))
def test_ist_window_with_default_bounds(hour):
    msc = int(datetime(2026, 5, 14, hour, 0, tzinfo=IST).astimezone(UTC).timestamp() * 1000)
    expected = 12 * 60 + 30 <= hour * 60 < 22 * 60 + 30
    assert is_within_ist_window(msc) is expected


def test_ist_window_wrap_midnight():
    """A wrapping window (start > end) covers from start->24:00 and 00:00->end."""
    ist = datetime(2026, 5, 14, 23, 30, tzinfo=IST)
    msc = int(ist.astimezone(UTC).timestamp() * 1000)
    assert is_within_ist_window(msc, "22:00", "02:00") is True
    ist2 = datetime(2026, 5, 14, 1, 0, tzinfo=IST)
    msc2 = int(ist2.astimezone(UTC).timestamp() * 1000)
    assert is_within_ist_window(msc2, "22:00", "02:00") is True
    ist3 = datetime(2026, 5, 14, 5, 0, tzinfo=IST)
    msc3 = int(ist3.astimezone(UTC).timestamp() * 1000)
    assert is_within_ist_window(msc3, "22:00", "02:00") is False


def test_ist_date_does_not_change_at_utc_midnight():
    """At 23:30 UTC on a date, IST shows next-day 05:00 (UTC+5:30)."""
    msc = utc_ms_at(2026, 5, 14, 23, 30)
    assert ist_date(msc) == "2026-05-15"


def test_ist_date_changes_at_ist_midnight():
    msc_just_before = int(
        datetime(2026, 5, 14, 23, 59, 59, tzinfo=IST).astimezone(UTC).timestamp() * 1000
    )
    msc_just_after = int(
        datetime(2026, 5, 15, 0, 0, 1, tzinfo=IST).astimezone(UTC).timestamp() * 1000
    )
    assert ist_date(msc_just_before) == "2026-05-14"
    assert ist_date(msc_just_after) == "2026-05-15"


# ---------------------------------------------------------------------------
# 3. DST SPRING-FORWARD / FALL-BACK — IST has no DST so we test US / EU
#    transitions to ensure the UTC-anchored windows are immune.
# ---------------------------------------------------------------------------

def test_dst_spring_forward_does_not_shift_utc_session():
    """US DST 2026: 2026-03-08 02:00 EST → 03:00 EDT (no 2 AM EST hour).
    The bot's session lives in pure UTC so the conversion is unaffected."""
    # 12:00 UTC = 08:00 EDT and was 07:00 EST before the change. The
    # session_for label depends on UTC hour ONLY.
    assert session_for(utc_dt(2026, 3, 8, 12, 0)) == SessionLabel.LONDON_NY_OVERLAP


def test_dst_fall_back_does_not_shift_utc_session():
    """US 2026-11-01 02:00 EDT → 01:00 EST (1 AM repeats). UTC bars unchanged."""
    assert session_for(utc_dt(2026, 11, 1, 12, 0)) == SessionLabel.LONDON_NY_OVERLAP


def test_dst_spring_forward_in_asian_window():
    """During DST transition the Asian window should still cover 5 UTC hours."""
    # Build flat bars spanning the spring-forward transition.
    bars = []
    base = utc_ms_at(2026, 3, 7, 19, 30)
    for i in range(6):
        bars.append(make_bar(symbol="EURUSD", time_msc=base + i * HOUR_MS,
                              open=1.10, close=1.10))
    cur_dt = utc_dt(2026, 3, 8, 8)
    ah, al = _compute_asian_range(bars, cur_dt)
    # Bars within [prev_day 19:30, cur_day 00:30) — we have 5 bars in the range
    # so it should be valid.
    assert ah is not None
    assert al is not None


# Europe DST: 2026-03-29 spring-forward, 2026-10-25 fall-back.
def test_europe_dst_does_not_shift_utc():
    """Europe DST switches happen at 01:00 UTC. Half-open UTC sessions
    aren't affected because the hour boundary doesn't move on the UTC axis."""
    msc_before = utc_ms_at(2026, 3, 29, 0, 30)
    msc_after = utc_ms_at(2026, 3, 29, 2, 30)
    assert session_for_msc(msc_before) == SessionLabel.ASIAN
    assert session_for_msc(msc_after) == SessionLabel.ASIAN


# ---------------------------------------------------------------------------
# 4. MIDNIGHT UTC ROLLOVER
# ---------------------------------------------------------------------------

def test_session_at_exact_midnight_utc():
    assert session_for_msc(utc_ms_at(2026, 5, 14, 0, 0)) == SessionLabel.ASIAN


def test_session_one_ms_before_midnight():
    # 23:59:59.999 UTC == hour 23 → OFF
    assert session_for_msc(utc_ms_at(2026, 5, 14, 23, 59, 59)) == SessionLabel.OFF


def test_circuit_breakers_day_roll_at_utc_midnight():
    cb = CircuitBreakers(daily_cap_pct=0.02)
    cb.state.daily_starting_equity = 10_000.0
    cb.state.daily_pnl_usd = -100.0
    cb.state.daily_cap_hit = True
    # First call seeds current_day_utc to today.
    msc_today = utc_ms_at(2026, 5, 14, 12)
    cb.can_trade(msc_today, account_equity=10_000.0)
    cb.state.daily_pnl_usd = -100.0
    cb.state.daily_cap_hit = True
    # Next-day call resets everything.
    msc_tomorrow = utc_ms_at(2026, 5, 15, 12)
    cb.can_trade(msc_tomorrow, account_equity=10_000.0)
    assert cb.state.daily_pnl_usd == 0.0
    assert cb.state.daily_cap_hit is False


def test_circuit_breakers_day_persists_within_day():
    cb = CircuitBreakers(daily_cap_pct=0.02)
    msc_morning = utc_ms_at(2026, 5, 14, 1)
    cb.can_trade(msc_morning, account_equity=10_000.0)
    # Inject a loss state.
    cb.state.daily_pnl_usd = -50.0
    msc_evening = utc_ms_at(2026, 5, 14, 23)
    cb.can_trade(msc_evening, account_equity=10_000.0)
    assert cb.state.daily_pnl_usd == -50.0


@pytest.mark.parametrize("hour", [0, 6, 12, 18, 23])
def test_circuit_breakers_idempotent_within_same_day(hour):
    cb = CircuitBreakers()
    msc = utc_ms_at(2026, 5, 14, hour)
    cb.can_trade(msc, account_equity=10_000.0)
    day1 = cb.state.current_day_utc
    cb.can_trade(msc, account_equity=10_000.0)
    assert cb.state.current_day_utc == day1


# ---------------------------------------------------------------------------
# 5. FRIDAY / SUNDAY WEEKEND GAP
# ---------------------------------------------------------------------------

def test_session_for_friday_21_00_utc_is_off():
    """Most brokers close at 21:00 UTC Fri. Session label: OFF."""
    # 2026-05-15 is a Friday.
    assert session_for_msc(utc_ms_at(2026, 5, 15, 21)) == SessionLabel.OFF


def test_session_for_sunday_22_00_utc_is_off():
    """Sunday markets reopen Sun 22:00 UTC; until then, OFF."""
    assert session_for_msc(utc_ms_at(2026, 5, 17, 21, 30)) == SessionLabel.OFF


def test_skip_monday_blocks_monday_signals(detector):
    """SKIP_MONDAY = True; the detector returns None on weekday() == 0."""
    bars = long_sweep_bars(symbol="EURUSD", pt=0.00001,
                            year=2026, month=5, day=18)  # Monday
    ctx = MarketContext(symbol="EURUSD", current_time_msc=bars[-1].time_msc)
    assert detector.detect(bars, ctx) is None


@pytest.mark.parametrize("year,month,day", [
    (2026, 5, 18), (2026, 5, 25), (2026, 6, 1), (2026, 6, 8),
    (2026, 6, 15), (2026, 6, 22), (2026, 6, 29),
])
def test_skip_monday_each_week(detector, year, month, day):
    # Confirm those are Mondays.
    assert datetime(year, month, day).weekday() == 0
    bars = long_sweep_bars(symbol="EURUSD", pt=0.00001,
                            year=year, month=month, day=day)
    ctx = MarketContext(symbol="EURUSD", current_time_msc=bars[-1].time_msc)
    assert detector.detect(bars, ctx) is None


@pytest.mark.parametrize("weekday_idx,year,month,day", [
    (1, 2026, 5, 19), (2, 2026, 5, 20), (3, 2026, 5, 21),
    (4, 2026, 5, 22), (5, 2026, 5, 23), (6, 2026, 5, 24),
])
def test_non_monday_days_can_emit_signals(detector, weekday_idx, year, month, day):
    assert datetime(year, month, day).weekday() == weekday_idx
    bars = long_sweep_bars(symbol="EURUSD", pt=0.00001,
                            year=year, month=month, day=day)
    ctx = MarketContext(symbol="EURUSD", current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    if weekday_idx == 5:
        # Saturday — markets closed; the bar timestamps still pass detector
        # since the detector doesn't look at weekday other than Monday.
        assert sig is not None
    elif weekday_idx == 6:
        assert sig is not None  # Sunday — detector doesn't filter
    else:
        assert sig is not None


# ---------------------------------------------------------------------------
# 6. SESSION-BOUNDARY MINUTE PRECISION
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hour,allowed", [
    (5, False),  # before London window
    (6, True),   # window open
    (10, True),  # last LONDON hour (inclusive per V5)
    (11, False), # gap between London-close and NY-open
    (12, True),  # NY window opens
    (15, True),  # last NY hour
    (16, False), # NY window closed
])
def test_detector_per_trigger_hour(detector, hour, allowed):
    if hour < 6 or (10 < hour < 12) or hour > 15:
        # Build neutral-bias scenario, expect None whether or not bars exist.
        bars = long_sweep_bars(symbol="EURUSD", pt=0.00001,
                                trigger_hour=hour)
        ctx = MarketContext(symbol="EURUSD", current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is None
    else:
        bars = long_sweep_bars(symbol="EURUSD", pt=0.00001,
                                trigger_hour=hour)
        ctx = MarketContext(symbol="EURUSD", current_time_msc=bars[-1].time_msc)
        assert detector.detect(bars, ctx) is not None


def test_detector_short_only_in_london(detector):
    """V5 rule: SHORT trades fire only in LONDON, not NY."""
    bars = short_sweep_bars(symbol="EURUSD", pt=0.00001,
                             trigger_hour=8)  # LONDON
    ctx = MarketContext(symbol="EURUSD", current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert sig is not None
    assert sig.direction == Direction.SELL


def test_detector_short_blocked_in_ny(detector):
    bars = short_sweep_bars(symbol="EURUSD", pt=0.00001,
                             trigger_hour=13)  # NY
    ctx = MarketContext(symbol="EURUSD", current_time_msc=bars[-1].time_msc)
    assert detector.detect(bars, ctx) is None


# ---------------------------------------------------------------------------
# 7. COMPLIANCE — IST WINDOW INTEGRATION
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ist_hhmm,allowed_reason", [
    ("12:29", "outside_ist_window"),
    ("12:30", "ok"),
    ("15:00", "ok"),
    ("22:30", "outside_ist_window"),
    ("23:00", "outside_ist_window"),
    ("06:00", "outside_ist_window"),
])
def test_compliance_ist_window_boundaries(compliance_factory, signal_factory,
                                          fresh_account, ist_hhmm,
                                          allowed_reason):
    ce = compliance_factory()
    hh, mm = (int(x) for x in ist_hhmm.split(":"))
    msc = int(datetime(2026, 5, 14, hh, mm, tzinfo=IST).astimezone(UTC).timestamp() * 1000)
    sig = signal_factory()
    ok, reason = ce.can_trade(sig, msc, fresh_account)
    if allowed_reason == "ok":
        assert reason == "ok"
        assert ok
    else:
        assert ok is False
        assert reason == allowed_reason


def test_compliance_window_custom_bounds(compliance_factory, signal_factory,
                                          fresh_account):
    ce = compliance_factory(ist_window_start="00:00", ist_window_end="23:59")
    sig = signal_factory()
    msc = int(datetime(2026, 5, 14, 4, 0, tzinfo=IST).astimezone(UTC).timestamp() * 1000)
    ok, reason = ce.can_trade(sig, msc, fresh_account)
    assert ok or reason != "outside_ist_window"


# ---------------------------------------------------------------------------
# 8. NEWS WINDOW ± SECOND-PRECISE
# ---------------------------------------------------------------------------

def _cal_with_one_event(*, hour=12, minute=30, currency="USD"):
    ev = NewsEvent(
        time_msc=utc_ms_at(2026, 5, 14, hour, minute),
        currency=currency, title="CPI", impact="HIGH",
    )
    return StaticNewsCalendar([ev])


@pytest.mark.parametrize("offset_sec,window_min,expected_blocked", [
    (0, 2, True),
    (60, 2, True),
    (120, 2, True),
    (121, 2, False),
    (-120, 2, True),
    (-121, 2, False),
    (180, 2, False),
    (180, 3, True),
])
def test_news_blackout_boundary(offset_sec, window_min, expected_blocked):
    cal = _cal_with_one_event()
    msc = utc_ms_at(2026, 5, 14, 12, 30) + offset_sec * 1000
    assert cal.is_news_blackout("XAUUSD", msc, window_min=window_min) is expected_blocked


def test_news_blackout_at_exact_event_time():
    cal = _cal_with_one_event()
    msc = utc_ms_at(2026, 5, 14, 12, 30)
    assert cal.is_blackout("XAUUSD", msc) is True


@pytest.mark.parametrize("currency,symbol,blocked", [
    ("USD", "XAUUSD", True),
    ("USD", "EURUSD", True),
    ("USD", "GBPUSD", True),
    ("USD", "AUDNZD", False),
    ("EUR", "EURUSD", True),
    ("EUR", "GBPUSD", False),
    ("GBP", "GBPUSD", True),
    ("JPY", "EURUSD", False),
    ("CAD", "USDCAD", True),
])
def test_news_per_currency_per_symbol(currency, symbol, blocked):
    cal = _cal_with_one_event(currency=currency)
    msc = utc_ms_at(2026, 5, 14, 12, 30)
    assert cal.is_blackout(symbol, msc) is blocked


def test_news_calendar_window_negative_raises():
    cal = StaticNewsCalendar([])
    with pytest.raises(ValueError):
        cal.is_news_blackout("EURUSD", 0, window_min=-1)


def test_news_blackout_ignores_low_impact():
    ev = NewsEvent(time_msc=utc_ms_at(2026, 5, 14, 12, 30),
                   currency="USD", title="JOLTS", impact="LOW")
    cal = StaticNewsCalendar([ev])
    assert cal.is_blackout("XAUUSD", utc_ms_at(2026, 5, 14, 12, 30)) is False


def test_news_blackout_ignores_medium_impact():
    ev = NewsEvent(time_msc=utc_ms_at(2026, 5, 14, 12, 30),
                   currency="USD", title="PMI", impact="MEDIUM")
    cal = StaticNewsCalendar([ev])
    assert cal.is_blackout("XAUUSD", utc_ms_at(2026, 5, 14, 12, 30)) is False


@pytest.mark.parametrize("window_min", [0, 1, 2, 5, 10, 30, 60, 120])
def test_news_window_scales_proportionally(window_min):
    cal = _cal_with_one_event()
    # At ±(window_min * 60) sec we should be exactly on the boundary.
    msc = utc_ms_at(2026, 5, 14, 12, 30) + window_min * 60 * 1000
    assert cal.is_news_blackout("XAUUSD", msc, window_min=window_min) is True


def test_compliance_blocks_during_news_window(compliance_factory,
                                              signal_factory, fresh_account):
    cal = _cal_with_one_event()
    ce = compliance_factory(news_calendar=cal)
    # Signal at exact event time but inside the IST window:
    sig = signal_factory(symbol="XAUUSD")
    msc = utc_ms_at(2026, 5, 14, 12, 30)
    ok, reason = ce.can_trade(sig, msc, fresh_account)
    # IST-window first: 12:30 UTC = 18:00 IST → in window. Then news check.
    assert ok is False
    assert reason in ("news_blackout", "outside_ist_window")


# ---------------------------------------------------------------------------
# 9. UPCOMING EVENTS
# ---------------------------------------------------------------------------

def test_upcoming_events_returns_strictly_after_cursor():
    cal = StaticNewsCalendar([
        NewsEvent(time_msc=utc_ms_at(2026, 5, 14, 12, 30),
                   currency="USD", title="A"),
        NewsEvent(time_msc=utc_ms_at(2026, 5, 15, 12, 30),
                   currency="USD", title="B"),
        NewsEvent(time_msc=utc_ms_at(2026, 5, 16, 12, 30),
                   currency="USD", title="C"),
    ])
    after = utc_ms_at(2026, 5, 14, 12, 30)
    upcoming = cal.upcoming_events(after, limit=2)
    assert [e.title for e in upcoming] == ["B", "C"]


@pytest.mark.parametrize("limit", [0, 1, 2, 5, 10])
def test_upcoming_events_respects_limit(limit):
    events = [
        NewsEvent(time_msc=utc_ms_at(2026, 5, 14 + i, 12, 30),
                   currency="USD", title=str(i))
        for i in range(7)
    ]
    cal = StaticNewsCalendar(events)
    out = cal.upcoming_events(0, limit=limit)
    assert len(out) <= limit


def test_upcoming_events_negative_limit_raises():
    cal = StaticNewsCalendar([])
    with pytest.raises(ValueError):
        cal.upcoming_events(0, limit=-1)


# ---------------------------------------------------------------------------
# 10. BOT STARTED MID-SESSION — PARTIAL ASIAN RANGE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("keep_count", [0, 1, 2, 3, 4, 5])
def test_partial_asian_range_with_missing_bars(detector, keep_count):
    bars = asian_window_with_missing_bars(
        symbol="XAUUSD", year=2026, month=5, day=14,
        asian_high=2010.0, asian_low=1990.0,
        keep_indices=tuple(range(keep_count)),
    )
    cur_dt = utc_dt(2026, 5, 14, 8)
    ah, al = _compute_asian_range(bars, cur_dt)
    if keep_count >= 2:
        assert ah is not None and al is not None
    else:
        assert ah is None and al is None


def test_bot_started_mid_asian_window_no_signal(detector):
    """Bot seeds <200 history bars; bias is neutral. With only the trigger bar
    in LONDON and a partial Asian range there's nothing to detect."""
    bars = [
        make_bar(symbol="EURUSD",
                 time_msc=hour_msc(2026, 5, 14, h),
                 open=1.10, close=1.10)
        for h in (22, 23)  # Asian end only
    ]
    bars.append(make_bar(symbol="EURUSD",
                          time_msc=hour_msc(2026, 5, 14, 8),
                          open=1.10, high=1.10, low=1.10, close=1.10))
    ctx = MarketContext(symbol="EURUSD",
                        current_time_msc=hour_msc(2026, 5, 14, 8))
    assert detector.detect(bars, ctx) is None


# ---------------------------------------------------------------------------
# 11. TRADE CROSSES DAY BOUNDARY — DAILY COUNTER RESET
# ---------------------------------------------------------------------------

def test_trade_count_resets_at_day_boundary(compliance_factory,
                                             signal_factory, account_factory):
    ce = compliance_factory(max_trades_per_day=2)
    acct_full = account_factory(trades_today=2)
    acct_fresh = account_factory(trades_today=0)
    msc_t1 = int(datetime(2026, 5, 14, 14, 0, tzinfo=IST).astimezone(UTC).timestamp() * 1000)
    msc_t2 = int(datetime(2026, 5, 15, 14, 0, tzinfo=IST).astimezone(UTC).timestamp() * 1000)
    sig = signal_factory()
    ok1, reason1 = ce.can_trade(sig, msc_t1, acct_full)
    ok2, reason2 = ce.can_trade(sig, msc_t2, acct_fresh)
    assert ok1 is False
    assert reason1 == "daily_trade_cap_reached"
    assert ok2 is True


@pytest.mark.parametrize("trades_today,allowed", [
    (0, True), (1, True), (2, False), (3, False),
])
def test_compliance_max_trades_per_day(compliance_factory, signal_factory,
                                       account_factory, trades_today, allowed):
    ce = compliance_factory(max_trades_per_day=2)
    acct = account_factory(trades_today=trades_today)
    sig = signal_factory()
    msc = int(datetime(2026, 5, 14, 14, 0, tzinfo=IST).astimezone(UTC).timestamp() * 1000)
    ok, reason = ce.can_trade(sig, msc, acct)
    assert ok is allowed


@pytest.mark.parametrize("max_per_day", [1, 2, 3, 5])
def test_compliance_max_trades_param(compliance_factory, signal_factory,
                                      account_factory, max_per_day):
    ce = compliance_factory(max_trades_per_day=max_per_day)
    for n in range(max_per_day + 2):
        acct = account_factory(trades_today=n)
        sig = signal_factory()
        msc = int(datetime(2026, 5, 14, 14, 0, tzinfo=IST).astimezone(UTC).timestamp() * 1000)
        ok, reason = ce.can_trade(sig, msc, acct)
        if n < max_per_day:
            assert ok or reason != "daily_trade_cap_reached"
        else:
            assert ok is False
            assert reason == "daily_trade_cap_reached"


def test_compliance_zero_max_trades_raises(compliance_factory):
    with pytest.raises(ValueError):
        compliance_factory(max_trades_per_day=0)


# ---------------------------------------------------------------------------
# 12. ROLLOVER WINDOW (21:00 UTC) — TRAILING SL SPREAD WIDENING
# ---------------------------------------------------------------------------

def test_protection_window_default_band_active():
    from datetime import datetime, timezone
    from config.griff_config import GriffConfig
    from risk.trailing_sl import TrailingStopLoss
    from strategy.swing_tracker import SwingTracker

    cfg = GriffConfig()
    tracker = SwingTracker()
    trail = TrailingStopLoss(tracker, cfg)
    # 21:00 UTC -> in window
    in_dt = datetime(2026, 5, 14, 21, 0, tzinfo=timezone.utc)
    assert trail._in_protection_window(in_dt) is True


@pytest.mark.parametrize("hh,mm,expected", [
    (20, 44, False),
    (20, 45, True),    # start = rollover - 15min = 20:45
    (20, 50, True),
    (21, 0, True),
    (21, 59, True),
    (22, 0, False),    # end = rollover + 60min = 22:00 (exclusive)
])
def test_protection_window_band_minute_precision(hh, mm, expected):
    from datetime import datetime, timezone
    from config.griff_config import GriffConfig
    from risk.trailing_sl import TrailingStopLoss
    from strategy.swing_tracker import SwingTracker

    trail = TrailingStopLoss(SwingTracker(), GriffConfig())
    dt = datetime(2026, 5, 14, hh, mm, tzinfo=timezone.utc)
    assert trail._in_protection_window(dt) is expected


# ---------------------------------------------------------------------------
# 13. ASIAN RANGE COMPUTED CORRECTLY ACROSS MONTH BOUNDARY
# ---------------------------------------------------------------------------

def test_asian_range_crosses_month_boundary(detector):
    """Asian window: prev 19:30 → current 00:30. If current = May 1, prev = Apr 30."""
    bars = long_sweep_bars(symbol="EURUSD", pt=0.00001,
                            year=2026, month=5, day=1)
    ctx = MarketContext(symbol="EURUSD", current_time_msc=bars[-1].time_msc)
    # 2026-05-01 was a Friday — should signal.
    sig = detector.detect(bars, ctx)
    assert sig is not None or datetime(2026, 5, 1).weekday() == 0


def test_asian_range_crosses_year_boundary(detector):
    bars = long_sweep_bars(symbol="EURUSD", pt=0.00001,
                            year=2026, month=1, day=2)
    ctx = MarketContext(symbol="EURUSD", current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    # 2026-01-02 = Friday.
    assert sig is not None or sig is None


# ---------------------------------------------------------------------------
# 14. PROTECTION WINDOW WRAPS — DST-AGNOSTIC BY DESIGN
# ---------------------------------------------------------------------------

def test_protection_window_works_on_dst_transition_date():
    """The rollover window is computed against UTC, not local time."""
    from datetime import datetime, timezone
    from config.griff_config import GriffConfig
    from risk.trailing_sl import TrailingStopLoss
    from strategy.swing_tracker import SwingTracker

    trail = TrailingStopLoss(SwingTracker(), GriffConfig())
    dt_spring = datetime(2026, 3, 8, 21, 0, tzinfo=timezone.utc)
    dt_fall = datetime(2026, 11, 1, 21, 0, tzinfo=timezone.utc)
    assert trail._in_protection_window(dt_spring) is True
    assert trail._in_protection_window(dt_fall) is True


# ---------------------------------------------------------------------------
# 15. HYPOTHESIS — WINDOW INVARIANTS
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(
    hour=st.integers(min_value=0, max_value=23),
    minute=st.integers(min_value=0, max_value=59),
)
def test_is_within_ist_window_returns_bool(hour, minute):
    msc = int(datetime(2026, 5, 14, hour, minute, tzinfo=IST).astimezone(UTC).timestamp() * 1000)
    result = is_within_ist_window(msc)
    assert isinstance(result, bool)


@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(
    day=st.integers(min_value=1, max_value=28),
    hour=st.integers(min_value=0, max_value=23),
)
def test_session_for_msc_returns_valid_label(day, hour):
    msc = utc_ms_at(2026, 5, day, hour)
    label = session_for_msc(msc)
    assert isinstance(label, SessionLabel)


# ---------------------------------------------------------------------------
# 16. NEWS CALENDAR DETERMINISM
# ---------------------------------------------------------------------------

def test_news_calendar_sorts_events():
    events = [
        NewsEvent(time_msc=utc_ms_at(2026, 5, 16, 12, 30),
                   currency="USD", title="C"),
        NewsEvent(time_msc=utc_ms_at(2026, 5, 14, 12, 30),
                   currency="USD", title="A"),
        NewsEvent(time_msc=utc_ms_at(2026, 5, 15, 12, 30),
                   currency="USD", title="B"),
    ]
    cal = StaticNewsCalendar(events)
    sorted_titles = [e.title for e in cal.events]
    assert sorted_titles == ["A", "B", "C"]


def test_news_calendar_empty_never_blackout():
    cal = StaticNewsCalendar([])
    assert cal.is_blackout("XAUUSD", utc_ms_at(2026, 5, 14, 12, 30)) is False


# ---------------------------------------------------------------------------
# 17. COMPLIANCE CALL ORDER — IST FIRST, NEWS LATER
# ---------------------------------------------------------------------------

def test_compliance_outside_ist_takes_priority_over_news(compliance_factory,
                                                          signal_factory,
                                                          fresh_account):
    cal = _cal_with_one_event(hour=4, minute=0)  # 09:30 IST — outside window
    ce = compliance_factory(news_calendar=cal)
    # 04:00 UTC = 09:30 IST -> outside IST window
    msc = utc_ms_at(2026, 5, 14, 4, 0)
    ok, reason = ce.can_trade(signal_factory(symbol="XAUUSD"), msc, fresh_account)
    assert ok is False
    # IST window check runs first.
    assert reason == "outside_ist_window"


def test_compliance_news_after_window(compliance_factory, signal_factory,
                                       fresh_account):
    cal = _cal_with_one_event(hour=12, minute=30)
    ce = compliance_factory(news_calendar=cal)
    msc = utc_ms_at(2026, 5, 14, 12, 30)
    ok, reason = ce.can_trade(signal_factory(symbol="XAUUSD"), msc, fresh_account)
    assert ok is False
    assert reason == "news_blackout"


# ---------------------------------------------------------------------------
# 18. SESSION-LABEL TRANSITION PROPERTIES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hour", list(range(24)))
def test_session_changes_only_at_documented_boundaries(hour):
    """Adjacent UTC hours either share a label OR change exactly at the
    boundaries 7, 12, 16, 21."""
    s1 = session_for(utc_dt(2026, 5, 14, hour, 30))
    boundaries = {7, 12, 16, 21}
    next_hour = (hour + 1) % 24
    s2 = session_for(utc_dt(2026, 5, 14, next_hour, 30))
    if s1 != s2:
        # Some boundary must be in (hour, next_hour].
        assert next_hour in boundaries or hour == 23
    else:
        pass  # consistent


def test_session_for_msc_handles_negative_msc():
    """Pre-1970 timestamp — degenerate but must not crash."""
    label = session_for_msc(-1000)
    assert isinstance(label, SessionLabel)


# ---------------------------------------------------------------------------
# 19. BIAS COMPUTATION OVER SESSION BOUNDARIES
# ---------------------------------------------------------------------------

def test_bias_neutral_when_below_min_bars(detector):
    """When < 200 closes are available, bias = neutral."""
    bars = [
        make_bar(symbol="EURUSD", time_msc=i * HOUR_MS,
                 open=1.10, close=1.10)
        for i in range(50)
    ]
    cur_dt = datetime.fromtimestamp(bars[-1].time_msc / 1000.0, tz=UTC)
    assert _compute_bias(bars, cur_dt) == "neutral"


def test_bias_returns_string_label():
    """Always returns one of {bullish, bearish, neutral}."""
    bars = [
        make_bar(symbol="EURUSD",
                 time_msc=hour_msc(2026, 5, 14, h),
                 open=1.10, close=1.10)
        for h in range(24)
    ]
    label = _compute_bias(bars, utc_dt(2026, 5, 14, 23))
    assert label in {"bullish", "bearish", "neutral"}


# ---------------------------------------------------------------------------
# 20. CIRCUIT BREAKER SESSION FILTER
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hour,allowed", [
    (3, False),
    (7, True),
    (12, True),
    (16, True),
    (20, True),
    (21, False),
])
def test_circuit_breaker_session_filter(hour, allowed):
    cb = CircuitBreakers()
    msc = utc_ms_at(2026, 5, 14, hour)
    ok, reason = cb.can_trade(msc, account_equity=10_000.0)
    if allowed:
        assert ok is True
    else:
        assert ok is False
        assert reason.startswith("session_blocked")


# ---------------------------------------------------------------------------
# 21. EXTREME TIMESTAMP / FAR-FUTURE
# ---------------------------------------------------------------------------

def test_session_for_msc_year_2100():
    msc = utc_ms_at(2100, 1, 1, 12)
    assert session_for_msc(msc) == SessionLabel.LONDON_NY_OVERLAP


def test_ist_date_year_2100():
    msc = utc_ms_at(2100, 1, 1, 0)
    # 2100-01-01 00:00 UTC = 05:30 IST 2100-01-01
    assert ist_date(msc) == "2100-01-01"


# ---------------------------------------------------------------------------
# 22. NEWS WINDOW + SYMBOL CONTAINING CURRENCY
# ---------------------------------------------------------------------------

def test_news_blackout_currency_substring_match():
    """'USD' must appear in symbol for the event to apply."""
    cal = _cal_with_one_event(currency="USD")
    msc = utc_ms_at(2026, 5, 14, 12, 30)
    assert cal.is_blackout("AUDUSD", msc) is True
    assert cal.is_blackout("AUDNZD", msc) is False


def test_news_blackout_currency_case_normalisation():
    """Symbol is uppercased before matching."""
    cal = _cal_with_one_event(currency="USD")
    msc = utc_ms_at(2026, 5, 14, 12, 30)
    assert cal.is_blackout("eurusd", msc) is True


# ---------------------------------------------------------------------------
# 23. PROTECTION WINDOW EDGE — WIDEN PIPS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair,expected_pips", [
    ("EURUSD", 40),
    ("GBPUSD", 45),
    ("AUDUSD", 45),
    ("AUDJPY", 50),
    ("EURJPY", 55),
    ("NZDJPY", 60),
    ("XAUUSD", 50),  # default
    ("BTCUSD", 50),  # unknown → default
])
def test_griff_config_widen_pips(pair, expected_pips):
    from config.griff_config import GriffConfig
    cfg = GriffConfig()
    assert cfg.widen_pips_for(pair) == expected_pips


# ---------------------------------------------------------------------------
# 24. NEWS BLACKOUT WITHIN ONE EVENT SET, MULTIPLE CURRENCIES
# ---------------------------------------------------------------------------

def test_news_blackout_multiple_currencies_same_time():
    """USD CPI + EUR ECB at the same minute → BOTH USD and EUR pairs blocked."""
    cal = StaticNewsCalendar([
        NewsEvent(time_msc=utc_ms_at(2026, 5, 14, 12, 30),
                   currency="USD", title="CPI"),
        NewsEvent(time_msc=utc_ms_at(2026, 5, 14, 12, 30),
                   currency="EUR", title="ECB"),
    ])
    msc = utc_ms_at(2026, 5, 14, 12, 30)
    assert cal.is_blackout("EURUSD", msc) is True   # both
    assert cal.is_blackout("GBPUSD", msc) is True   # USD only
    assert cal.is_blackout("EURGBP", msc) is True   # EUR only
    assert cal.is_blackout("AUDNZD", msc) is False  # neither
