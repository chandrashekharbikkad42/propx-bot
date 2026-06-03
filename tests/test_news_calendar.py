"""Phase 8B — StaticNewsCalendar tests.

Covers blackout window logic, currency matching, ordering, and edge cases.
"""

from __future__ import annotations
from datetime import datetime, timezone

import pytest

from data.news_calendar import (
    DEFAULT_CALENDAR,
    DEFAULT_HIGH_IMPACT_EVENTS,
    NewsEvent,
    StaticNewsCalendar,
)


def _utc_ms(year: int, month: int, day: int, hour: int, minute: int) -> int:
    return int(
        datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp() * 1000
    )


class TestStaticCalendarBlackoutTiming:
    def test_exact_event_time_is_blackout(self):
        evt = NewsEvent(_utc_ms(2026, 5, 17, 12, 30), "USD", "NFP")
        cal = StaticNewsCalendar([evt])
        assert cal.is_news_blackout("EURUSD", _utc_ms(2026, 5, 17, 12, 30)) is True

    def test_within_window_before(self):
        evt = NewsEvent(_utc_ms(2026, 5, 17, 12, 30), "USD", "NFP")
        cal = StaticNewsCalendar([evt])
        # 1 min before — IN window
        assert cal.is_news_blackout("EURUSD", _utc_ms(2026, 5, 17, 12, 29)) is True

    def test_within_window_after(self):
        evt = NewsEvent(_utc_ms(2026, 5, 17, 12, 30), "USD", "NFP")
        cal = StaticNewsCalendar([evt])
        assert cal.is_news_blackout("EURUSD", _utc_ms(2026, 5, 17, 12, 32)) is True

    def test_exactly_at_window_edge_is_blackout(self):
        evt = NewsEvent(_utc_ms(2026, 5, 17, 12, 30), "USD", "NFP")
        cal = StaticNewsCalendar([evt])
        # ±2 min boundary inclusive
        assert cal.is_news_blackout("EURUSD", _utc_ms(2026, 5, 17, 12, 28)) is True
        assert cal.is_news_blackout("EURUSD", _utc_ms(2026, 5, 17, 12, 32)) is True

    def test_just_outside_window(self):
        evt = NewsEvent(_utc_ms(2026, 5, 17, 12, 30), "USD", "NFP")
        cal = StaticNewsCalendar([evt])
        # 3 min before — outside
        assert cal.is_news_blackout("EURUSD", _utc_ms(2026, 5, 17, 12, 27)) is False
        # 3 min after — outside
        assert cal.is_news_blackout("EURUSD", _utc_ms(2026, 5, 17, 12, 33)) is False

    def test_custom_window(self):
        evt = NewsEvent(_utc_ms(2026, 5, 17, 12, 30), "USD", "NFP")
        cal = StaticNewsCalendar([evt])
        # 5-min window: 12:34 is IN
        assert cal.is_news_blackout(
            "EURUSD", _utc_ms(2026, 5, 17, 12, 34), window_min=5
        ) is True

    def test_window_zero_means_only_exact_match(self):
        evt = NewsEvent(_utc_ms(2026, 5, 17, 12, 30), "USD", "NFP")
        cal = StaticNewsCalendar([evt])
        assert cal.is_news_blackout("EURUSD", _utc_ms(2026, 5, 17, 12, 30), 0) is True
        assert cal.is_news_blackout("EURUSD", _utc_ms(2026, 5, 17, 12, 31), 0) is False

    def test_rejects_negative_window(self):
        cal = StaticNewsCalendar([])
        with pytest.raises(ValueError):
            cal.is_news_blackout("EURUSD", _utc_ms(2026, 5, 17, 12, 30), -1)


class TestStaticCalendarCurrencyMatch:
    def test_usd_event_affects_usd_pairs(self):
        evt = NewsEvent(_utc_ms(2026, 5, 17, 12, 30), "USD", "NFP")
        cal = StaticNewsCalendar([evt])
        t = _utc_ms(2026, 5, 17, 12, 30)
        for pair in ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "XAUUSD"):
            assert cal.is_news_blackout(pair, t) is True

    def test_usd_event_does_not_affect_pure_crosses(self):
        evt = NewsEvent(_utc_ms(2026, 5, 17, 12, 30), "USD", "NFP")
        cal = StaticNewsCalendar([evt])
        t = _utc_ms(2026, 5, 17, 12, 30)
        # EURJPY has no USD — safe.
        assert cal.is_news_blackout("EURJPY", t) is False
        assert cal.is_news_blackout("GBPJPY", t) is False

    def test_eur_event_only_affects_eur_pairs(self):
        evt = NewsEvent(_utc_ms(2026, 5, 17, 9, 0), "EUR", "ECB Rate")
        cal = StaticNewsCalendar([evt])
        t = _utc_ms(2026, 5, 17, 9, 0)
        assert cal.is_news_blackout("EURUSD", t) is True
        assert cal.is_news_blackout("EURJPY", t) is True
        assert cal.is_news_blackout("GBPUSD", t) is False
        assert cal.is_news_blackout("USDJPY", t) is False

    def test_case_insensitive_symbol(self):
        evt = NewsEvent(_utc_ms(2026, 5, 17, 12, 30), "USD", "NFP")
        cal = StaticNewsCalendar([evt])
        assert cal.is_news_blackout("eurusd", _utc_ms(2026, 5, 17, 12, 30)) is True


class TestStaticCalendarImpactFilter:
    def test_low_impact_event_does_not_block(self):
        evt = NewsEvent(_utc_ms(2026, 5, 17, 12, 30), "USD", "Trivia", impact="LOW")
        cal = StaticNewsCalendar([evt])
        assert cal.is_news_blackout("EURUSD", _utc_ms(2026, 5, 17, 12, 30)) is False

    def test_medium_impact_event_does_not_block_by_default(self):
        evt = NewsEvent(_utc_ms(2026, 5, 17, 12, 30), "USD", "PMI", impact="MEDIUM")
        cal = StaticNewsCalendar([evt])
        assert cal.is_news_blackout("EURUSD", _utc_ms(2026, 5, 17, 12, 30)) is False


class TestUpcomingEvents:
    def test_returns_in_chronological_order(self):
        cal = StaticNewsCalendar([
            NewsEvent(_utc_ms(2026, 6, 5, 12, 30), "USD", "NFP"),
            NewsEvent(_utc_ms(2026, 5, 17, 12, 30), "USD", "CPI"),
            NewsEvent(_utc_ms(2026, 5, 28, 18, 0), "USD", "FOMC"),
        ])
        upcoming = cal.upcoming_events(_utc_ms(2026, 5, 1, 0, 0), limit=5)
        titles = [e.title for e in upcoming]
        assert titles == ["CPI", "FOMC", "NFP"]

    def test_strict_after_filter(self):
        evt_time = _utc_ms(2026, 5, 17, 12, 30)
        cal = StaticNewsCalendar([NewsEvent(evt_time, "USD", "NFP")])
        # after = exact time → strict, so NFP excluded
        assert cal.upcoming_events(evt_time, limit=5) == []
        # after = 1 ms before → included
        upcoming = cal.upcoming_events(evt_time - 1, limit=5)
        assert len(upcoming) == 1

    def test_limit_caps_result(self):
        events = [
            NewsEvent(_utc_ms(2026, 5, 17 + i, 12, 30), "USD", f"E{i}")
            for i in range(5)
        ]
        cal = StaticNewsCalendar(events)
        upcoming = cal.upcoming_events(_utc_ms(2026, 5, 1, 0, 0), limit=2)
        assert len(upcoming) == 2

    def test_limit_zero_returns_empty(self):
        cal = StaticNewsCalendar(DEFAULT_HIGH_IMPACT_EVENTS)
        assert cal.upcoming_events(0, limit=0) == []

    def test_rejects_negative_limit(self):
        cal = StaticNewsCalendar([])
        with pytest.raises(ValueError):
            cal.upcoming_events(0, limit=-1)


class TestDefaultCalendar:
    def test_default_singleton_has_events(self):
        assert len(DEFAULT_CALENDAR.events) > 0  # type: ignore[attr-defined]

    def test_default_events_are_sorted(self):
        events = DEFAULT_CALENDAR.events  # type: ignore[attr-defined]
        times = [e.time_msc for e in events]
        assert times == sorted(times)


class TestEmptyCalendar:
    def test_empty_calendar_never_blacks_out(self):
        cal = StaticNewsCalendar([])
        assert cal.is_news_blackout("EURUSD", _utc_ms(2026, 5, 17, 12, 30)) is False
        assert cal.upcoming_events(0) == []
