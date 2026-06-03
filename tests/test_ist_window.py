"""Phase 8B — IST window helpers.

Window semantics: half-open [start, end). 12:30–22:30 IST is the Griff trading
window — London open through NY close + overlap.

IST = UTC + 5:30 (no DST). 12:30 IST == 07:00 UTC. 22:30 IST == 17:00 UTC.
"""

from __future__ import annotations
from datetime import datetime, timezone

from utils.session import IST, ist_date, is_within_ist_window, to_ist


def _utc_ms(year: int, month: int, day: int, hour: int, minute: int) -> int:
    return int(
        datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp() * 1000
    )


class TestToIst:
    def test_returns_ist_aware_datetime(self):
        # 2026-05-17 07:00 UTC == 12:30 IST
        ms = _utc_ms(2026, 5, 17, 7, 0)
        dt = to_ist(ms)
        assert dt.tzinfo == IST
        assert dt.hour == 12
        assert dt.minute == 30

    def test_midnight_ist_is_1830_prev_utc(self):
        # 2026-05-17 18:30 UTC == 2026-05-18 00:00 IST
        ms = _utc_ms(2026, 5, 17, 18, 30)
        dt = to_ist(ms)
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.day == 18


class TestIstDate:
    def test_rollover_at_midnight_ist(self):
        # 18:29 UTC == 23:59 IST → still 2026-05-17
        assert ist_date(_utc_ms(2026, 5, 17, 18, 29)) == "2026-05-17"
        # 18:30 UTC == 00:00 IST next day → 2026-05-18
        assert ist_date(_utc_ms(2026, 5, 17, 18, 30)) == "2026-05-18"

    def test_format_is_yyyy_mm_dd(self):
        s = ist_date(_utc_ms(2026, 1, 5, 6, 0))
        assert len(s) == 10
        assert s == "2026-01-05"


class TestIsWithinIstWindow:
    # Default Griff window: 12:30–22:30 IST (07:00–17:00 UTC).

    def test_start_boundary_inclusive(self):
        assert is_within_ist_window(_utc_ms(2026, 5, 17, 7, 0)) is True  # 12:30 IST

    def test_end_boundary_exclusive(self):
        # 17:00 UTC == 22:30 IST — half-open, so this is OUT.
        assert is_within_ist_window(_utc_ms(2026, 5, 17, 17, 0)) is False

    def test_one_minute_before_end_is_in(self):
        assert is_within_ist_window(_utc_ms(2026, 5, 17, 16, 59)) is True

    def test_before_window(self):
        # 06:00 UTC == 11:30 IST
        assert is_within_ist_window(_utc_ms(2026, 5, 17, 6, 0)) is False

    def test_after_window(self):
        # 18:00 UTC == 23:30 IST
        assert is_within_ist_window(_utc_ms(2026, 5, 17, 18, 0)) is False

    def test_mid_window(self):
        # 12:00 UTC == 17:30 IST — middle of overlap
        assert is_within_ist_window(_utc_ms(2026, 5, 17, 12, 0)) is True

    def test_custom_window(self):
        # 09:00–17:00 IST == 03:30–11:30 UTC
        assert is_within_ist_window(
            _utc_ms(2026, 5, 17, 4, 0), "09:00", "17:00"
        ) is True
        assert is_within_ist_window(
            _utc_ms(2026, 5, 17, 2, 0), "09:00", "17:00"
        ) is False

    def test_wrap_midnight_window(self):
        # Window 22:00–06:00 IST wraps midnight.
        # 23:00 IST = 17:30 UTC → IN
        assert is_within_ist_window(
            _utc_ms(2026, 5, 17, 17, 30), "22:00", "06:00"
        ) is True
        # 04:00 IST = 22:30 UTC prev day → IN
        assert is_within_ist_window(
            _utc_ms(2026, 5, 16, 22, 30), "22:00", "06:00"
        ) is True
        # 10:00 IST = 04:30 UTC → OUT
        assert is_within_ist_window(
            _utc_ms(2026, 5, 17, 4, 30), "22:00", "06:00"
        ) is False
