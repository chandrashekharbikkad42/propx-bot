"""Phase 8D-Live — DailyTracker tests."""

from __future__ import annotations
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from monitoring.daily_tracker import DailyTracker, ist_trade_day


def _ms(year: int, month: int, day: int, hour: int = 12, minute: int = 0) -> int:
    dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# ============================================================================
# IST trade-day computation
# ============================================================================

class TestIstTradeDay(unittest.TestCase):
    def test_utc_noon_maps_to_same_calendar_day(self):
        # 2026-05-17 12:00 UTC → IST 17:30 → 2026-05-17.
        self.assertEqual(ist_trade_day(_ms(2026, 5, 17, 12, 0)), "2026-05-17")

    def test_utc_after_1830_maps_to_next_day(self):
        # 2026-05-17 19:00 UTC → IST 00:30 next day → 2026-05-18.
        self.assertEqual(ist_trade_day(_ms(2026, 5, 17, 19, 0)), "2026-05-18")

    def test_utc_at_1830_boundary(self):
        # Exactly at boundary belongs to the new day.
        self.assertEqual(ist_trade_day(_ms(2026, 5, 17, 18, 30)), "2026-05-18")


# ============================================================================
# Trade count
# ============================================================================

class TestTradeCount(unittest.TestCase):
    def test_starts_at_zero(self):
        t = DailyTracker(starting_equity=10_000, now_ms=_ms(2026, 5, 17, 12))
        self.assertEqual(t.trade_count, 0)

    def test_record_open_bumps_counter(self):
        t = DailyTracker(starting_equity=10_000, now_ms=_ms(2026, 5, 17, 12))
        t.record_trade_open(now_ms=_ms(2026, 5, 17, 13))
        t.record_trade_open(now_ms=_ms(2026, 5, 17, 14))
        self.assertEqual(t.trade_count, 2)


# ============================================================================
# Equity + drawdown
# ============================================================================

class TestEquityAndDrawdown(unittest.TestCase):
    def test_peak_equity_tracks_max(self):
        t = DailyTracker(starting_equity=10_000, now_ms=_ms(2026, 5, 17, 12))
        t.update_equity(10_500, now_ms=_ms(2026, 5, 17, 13))
        t.update_equity(10_200, now_ms=_ms(2026, 5, 17, 14))
        self.assertEqual(t.state.peak_equity, 10_500)

    def test_drawdown_from_peak(self):
        t = DailyTracker(starting_equity=10_000, now_ms=_ms(2026, 5, 17, 12))
        t.update_equity(10_500, now_ms=_ms(2026, 5, 17, 13))
        t.update_equity(10_200, now_ms=_ms(2026, 5, 17, 14))
        # max drawdown = peak (10500) - trough (10200) = 300
        self.assertEqual(t.max_dd_today, 300)

    def test_drawdown_persistent_after_recovery(self):
        t = DailyTracker(starting_equity=10_000, now_ms=_ms(2026, 5, 17, 12))
        t.update_equity(10_500, now_ms=_ms(2026, 5, 17, 13))
        t.update_equity(10_200, now_ms=_ms(2026, 5, 17, 14))
        # Even if equity recovers, max_dd_today stays at 300.
        t.update_equity(10_600, now_ms=_ms(2026, 5, 17, 15))
        self.assertEqual(t.max_dd_today, 300)


# ============================================================================
# Closed PnL accumulation
# ============================================================================

class TestClosedPnl(unittest.TestCase):
    def test_record_trade_closed_sums(self):
        t = DailyTracker(starting_equity=10_000, now_ms=_ms(2026, 5, 17, 12))
        t.record_trade_closed(50.0, now_ms=_ms(2026, 5, 17, 13))
        t.record_trade_closed(-20.0, now_ms=_ms(2026, 5, 17, 14))
        self.assertEqual(t.state.closed_pnl, 30.0)


# ============================================================================
# Day roll at IST midnight
# ============================================================================

class TestDayRoll(unittest.TestCase):
    def test_state_resets_when_ist_day_advances(self):
        t = DailyTracker(starting_equity=10_000, now_ms=_ms(2026, 5, 17, 12))
        t.record_trade_open(now_ms=_ms(2026, 5, 17, 13))
        t.record_trade_closed(75.0, now_ms=_ms(2026, 5, 17, 14))
        # Advance past 18:30 UTC → next IST day.
        t.update_equity(10_500, now_ms=_ms(2026, 5, 17, 19))
        self.assertEqual(t.trade_day, "2026-05-18")
        self.assertEqual(t.trade_count, 0)
        self.assertEqual(t.state.closed_pnl, 0.0)


# ============================================================================
# Persistence
# ============================================================================

class TestPersistence(unittest.TestCase):
    def test_persist_and_reload_same_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.parquet"
            t = DailyTracker(starting_equity=10_000, persist_path=path,
                             now_ms=_ms(2026, 5, 17, 12))
            t.record_trade_open(now_ms=_ms(2026, 5, 17, 13))
            t.record_trade_closed(40.0, now_ms=_ms(2026, 5, 17, 14))
            t.persist()

            # Fresh tracker on same UTC day should pick up persisted state.
            t2 = DailyTracker(starting_equity=10_000, persist_path=path,
                              now_ms=_ms(2026, 5, 17, 15))
            self.assertEqual(t2.trade_count, 1)
            self.assertEqual(t2.state.closed_pnl, 40.0)

    def test_persisted_state_ignored_after_day_roll(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.parquet"
            t = DailyTracker(starting_equity=10_000, persist_path=path,
                             now_ms=_ms(2026, 5, 17, 12))
            t.record_trade_open(now_ms=_ms(2026, 5, 17, 13))
            t.persist()

            # Reload on the NEXT IST day → state ignored, fresh tracker.
            t2 = DailyTracker(starting_equity=10_000, persist_path=path,
                              now_ms=_ms(2026, 5, 17, 19))
            self.assertEqual(t2.trade_count, 0)
            self.assertEqual(t2.trade_day, "2026-05-18")
