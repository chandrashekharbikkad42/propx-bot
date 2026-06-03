"""Phase 9 — HourlyReporter tests.

Covers:
  - HourlyStats accumulation + reset semantics
  - Format includes today (trades, P/L, DD) + last hour (bars, signals,
    compliance verdicts, open positions)
  - IST time stamp in the header
  - Idle-hour throttle (00:00–12:00 IST + nothing happening → short msg)
  - Full message when something happened during idle window
  - send() routes through TelegramNotifier and returns its result
  - Disabled notifier → no-op (returns False)
  - next_top_of_hour_ms() picks the correct wall-clock target
"""

from __future__ import annotations
import asyncio
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from alerts.telegram_notifier import TelegramNotifier
from execution.order_router import GriffOpenPosition
from execution.position_manager import GriffPositionManager
from monitoring.daily_tracker import DailyTracker
from monitoring.hourly_reporter import (
    HourlyReporter,
    HourlyStats,
    next_top_of_hour_ms,
)
from strategy.patterns.base import Direction


# ----------------------------------------------------------------- fixtures

def _utc_msc(y: int, mo: int, d: int, h: int, mi: int = 0, s: int = 0) -> int:
    return int(datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).timestamp() * 1000)


# 2026-05-18 03:30 UTC = 09:00 IST (within 00:00–12:00 IST silent window)
SILENT_HOUR_MSC = _utc_msc(2026, 5, 18, 3, 30)

# 2026-05-18 13:00 UTC = 18:30 IST (active window)
ACTIVE_HOUR_MSC = _utc_msc(2026, 5, 18, 13, 0)


def _notifier_mock(enabled: bool = True) -> TelegramNotifier:
    n = TelegramNotifier(
        token="x" if enabled else None,
        chat_id="y" if enabled else None,
    )
    # Mimic real behavior — return True only when enabled.
    n.send = AsyncMock(return_value=enabled)
    return n


def _position(symbol: str = "EURUSD") -> GriffOpenPosition:
    return GriffOpenPosition(
        position_id=f"p-{symbol}",
        mt5_ticket=1,
        symbol=symbol,
        side=Direction.BUY,
        lots=0.1,
        entry_price=1.10,
        sl_price=1.099,
        tp_price=1.102,
        opened_msc=0,
        signal_id="s",
        pattern_name="FLAG",
    )


class _StubRouter:
    """Minimal stub for GriffPositionManager's router dependency."""

    def __init__(self) -> None:
        self.dry_run = True

    async def modify_sl(self, *_, **__) -> None:  # pragma: no cover
        return None

    async def cancel_pending(self, *_, **__) -> None:  # pragma: no cover
        return None


def _position_mgr_with(positions=()) -> GriffPositionManager:
    from risk.trailing_sl import TrailingStopLoss
    from strategy.swing_tracker import SwingTracker

    swing = SwingTracker()
    trail = TrailingStopLoss(swing)
    pm = GriffPositionManager(_StubRouter(), swing, trail)
    for p in positions:
        pm.register_position(p)
    return pm


def _daily_tracker(
    starting: float = 10_000.0,
    *,
    now_ms: int = ACTIVE_HOUR_MSC,
    trades: int = 0,
    closed_pnl: float = 0.0,
    dd: float = 0.0,
) -> DailyTracker:
    dt = DailyTracker(starting_equity=starting, now_ms=now_ms)
    if dd > 0:
        # Simulate a drawdown by raising peak then dropping equity.
        dt.update_equity(starting, now_ms=now_ms)
        dt.update_equity(starting - dd, now_ms=now_ms)
    for _ in range(trades):
        dt.record_trade_open(now_ms=now_ms)
    if closed_pnl != 0.0:
        dt.record_trade_closed(closed_pnl, now_ms=now_ms)
    return dt


# ====================================================== HourlyStats unit


class TestHourlyStats(unittest.TestCase):
    def test_record_bar_tracks_unique_pairs(self):
        s = HourlyStats()
        s.record_bar("EURUSD")
        s.record_bar("EURUSD")
        s.record_bar("AUDJPY")
        self.assertEqual(s.bars_received, 3)
        self.assertEqual(s.pairs_with_bars, {"EURUSD", "AUDJPY"})

    def test_record_compliance_split(self):
        s = HourlyStats()
        s.record_compliance(passed=True)
        s.record_compliance(passed=False)
        s.record_compliance(passed=False)
        self.assertEqual(s.compliance_passed, 1)
        self.assertEqual(s.compliance_blocked, 2)

    def test_reset_clears_everything(self):
        s = HourlyStats()
        s.record_bar("X")
        s.record_signal()
        s.record_compliance(passed=True)
        s.reset()
        self.assertEqual(s.bars_received, 0)
        self.assertEqual(s.pairs_with_bars, set())
        self.assertEqual(s.signals_detected, 0)
        self.assertEqual(s.compliance_passed, 0)
        self.assertEqual(s.compliance_blocked, 0)

    def test_is_idle_true_when_nothing_happened(self):
        s = HourlyStats()
        self.assertTrue(s.is_idle())

    def test_is_idle_false_after_signal(self):
        s = HourlyStats()
        s.record_signal()
        self.assertFalse(s.is_idle())


# ============================================== HourlyReporter format


class TestFormatFull(unittest.TestCase):
    def setUp(self):
        self.notifier = _notifier_mock()
        self.stats = HourlyStats()
        self.daily = _daily_tracker(
            trades=2, closed_pnl=50.0, dd=30.0,
        )
        self.pm = _position_mgr_with([_position("EURUSD")])

    def _reporter(self, daily_loss_cap_pct: float = 5.0) -> HourlyReporter:
        return HourlyReporter(
            notifier=self.notifier,
            daily=self.daily,
            position_mgr=self.pm,
            stats=self.stats,
            num_pairs=6,
            daily_loss_cap_pct=daily_loss_cap_pct,
        )

    def test_header_has_ist_time(self):
        msg = self._reporter().format(ACTIVE_HOUR_MSC)
        # 13:00 UTC = 18:30 IST
        self.assertIn("18:30", msg)
        self.assertIn("IST", msg)
        self.assertIn("propX Bot Status", msg)

    def test_today_section_has_trades_and_pnl(self):
        msg = self._reporter().format(ACTIVE_HOUR_MSC)
        self.assertIn("Today", msg)
        self.assertIn("Trades: 2", msg)
        # Closed PnL = 50.0 on 10_000 starting → 0.5%
        self.assertIn("$50.00", msg)
        self.assertIn("0.50%", msg)

    def test_today_section_has_dd_with_cap(self):
        msg = self._reporter(daily_loss_cap_pct=5.0).format(ACTIVE_HOUR_MSC)
        # max_dd_today = $30 on $10_000 → 0.30%; cap shown as 5.0%
        self.assertIn("DD: 0.30%", msg)
        self.assertIn("5.0%", msg)

    def test_last_hour_section(self):
        self.stats.record_bar("EURUSD")
        self.stats.record_bar("AUDJPY")
        self.stats.record_signal()
        self.stats.record_compliance(passed=True)
        self.stats.record_compliance(passed=False)
        msg = self._reporter().format(ACTIVE_HOUR_MSC)
        self.assertIn("Last hour", msg)
        self.assertIn("Bars received: 2/6", msg)
        self.assertIn("Signals detected: 1", msg)
        self.assertIn("pass=1", msg)
        self.assertIn("blocked=1", msg)
        self.assertIn("Open positions: 1", msg)

    def test_healthy_when_no_issues(self):
        self.stats.record_bar("EURUSD")
        msg = self._reporter().format(ACTIVE_HOUR_MSC)
        # Healthy marker present in normal state.
        self.assertIn("Healthy", msg)


class TestFormatIdle(unittest.TestCase):
    def test_silent_hour_with_nothing_happening_is_abbreviated(self):
        notifier = _notifier_mock()
        daily = _daily_tracker(trades=0, closed_pnl=0.0, now_ms=SILENT_HOUR_MSC)
        pm = _position_mgr_with([])
        stats = HourlyStats()
        r = HourlyReporter(
            notifier=notifier, daily=daily, position_mgr=pm,
            stats=stats, num_pairs=6, daily_loss_cap_pct=5.0,
        )
        msg = r.format(SILENT_HOUR_MSC)
        # Abbreviated marker; short message; no detailed sections.
        self.assertIn("idle", msg.lower())
        self.assertNotIn("Last hour", msg)
        # IST 09:00 must still be present in the header.
        self.assertIn("09:00", msg)

    def test_silent_hour_with_open_position_is_full(self):
        notifier = _notifier_mock()
        daily = _daily_tracker(trades=0, now_ms=SILENT_HOUR_MSC)
        pm = _position_mgr_with([_position("EURUSD")])  # something happening
        stats = HourlyStats()
        r = HourlyReporter(
            notifier=notifier, daily=daily, position_mgr=pm,
            stats=stats, num_pairs=6, daily_loss_cap_pct=5.0,
        )
        msg = r.format(SILENT_HOUR_MSC)
        # Full body — Last-hour section present, no "idle" tag.
        self.assertIn("Last hour", msg)


# ============================================== HourlyReporter.send


class TestSend(unittest.TestCase):
    def test_send_calls_notifier(self):
        n = _notifier_mock()
        daily = _daily_tracker()
        pm = _position_mgr_with([])
        stats = HourlyStats()
        r = HourlyReporter(
            notifier=n, daily=daily, position_mgr=pm,
            stats=stats, num_pairs=6, daily_loss_cap_pct=5.0,
        )
        ok = asyncio.run(r.send(ACTIVE_HOUR_MSC))
        self.assertTrue(ok)
        n.send.assert_awaited_once()

    def test_send_resets_stats(self):
        n = _notifier_mock()
        daily = _daily_tracker()
        pm = _position_mgr_with([])
        stats = HourlyStats()
        stats.record_bar("EURUSD")
        stats.record_signal()
        r = HourlyReporter(
            notifier=n, daily=daily, position_mgr=pm,
            stats=stats, num_pairs=6, daily_loss_cap_pct=5.0,
        )
        asyncio.run(r.send(ACTIVE_HOUR_MSC))
        self.assertEqual(stats.bars_received, 0)
        self.assertEqual(stats.signals_detected, 0)

    def test_disabled_notifier_returns_false(self):
        n = _notifier_mock(enabled=False)
        daily = _daily_tracker()
        pm = _position_mgr_with([])
        stats = HourlyStats()
        r = HourlyReporter(
            notifier=n, daily=daily, position_mgr=pm,
            stats=stats, num_pairs=6, daily_loss_cap_pct=5.0,
        )
        ok = asyncio.run(r.send(ACTIVE_HOUR_MSC))
        self.assertFalse(ok)


# ============================================== scheduling helper


class TestNextTopOfHour(unittest.TestCase):
    def test_xx_30_rounds_to_xx_plus_one(self):
        msc = _utc_msc(2026, 5, 18, 18, 30, 0)
        nxt = next_top_of_hour_ms(msc)
        self.assertEqual(nxt - msc, 30 * 60 * 1000)

    def test_exact_hour_advances_to_next_hour(self):
        msc = _utc_msc(2026, 5, 18, 0, 0, 0)
        nxt = next_top_of_hour_ms(msc)
        self.assertEqual(nxt - msc, 60 * 60 * 1000)


if __name__ == "__main__":
    unittest.main()
