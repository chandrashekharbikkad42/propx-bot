"""Phase 8D-Live — Griff dashboard HTTP tests using aiohttp TestClient."""

from __future__ import annotations
import unittest

from aiohttp.test_utils import AioHTTPTestCase

from execution.order_router import GriffOpenPosition, GriffOrderRouter, GriffPendingOrder
from execution.position_manager import GriffPositionManager
from monitoring.daily_tracker import DailyTracker
from monitoring.dashboard import GriffDashboard
from risk.trailing_sl import TrailingStopLoss
from strategy.patterns.base import Direction, Grade, PatternSignal
from strategy.swing_tracker import SwingTracker


def _pm() -> GriffPositionManager:
    router = GriffOrderRouter(dry_run=True)
    st = SwingTracker()
    return GriffPositionManager(router, st, TrailingStopLoss(st))


def _pos() -> GriffOpenPosition:
    return GriffOpenPosition(
        position_id="p1", mt5_ticket=42, symbol="EURUSD", side=Direction.BUY,
        lots=0.1, entry_price=1.1, sl_price=1.099, tp_price=1.102,
        opened_msc=0, signal_id="s", pattern_name="FLAG",
    )


def _pending() -> GriffPendingOrder:
    return GriffPendingOrder(
        order_id="o1", mt5_ticket=43, symbol="AUDJPY", side=Direction.SELL,
        lots=0.1, pending_price=95.00, sl_price=95.10, tp_price=94.80,
        expiry_msc=99 * 3_600_000, signal_id="s", pattern_name="CONTINUATION",
        is_limit=False,
    )


def _signal() -> PatternSignal:
    return PatternSignal(
        pattern_name="FLAG", symbol="EURUSD", direction=Direction.BUY,
        entry=1.1, sl=1.099, tp=1.102, confidence=0.7, grade=Grade.A,
        confluences_met=("a",), bar_time_msc=0,
    )


class _BaseDashCase(AioHTTPTestCase):
    async def get_application(self):
        self.pm = _pm()
        self.pm.register_position(_pos())
        self.pm.register_pending(_pending())
        self.daily = DailyTracker(starting_equity=10_000, now_ms=0)
        self.dash = GriffDashboard(
            self.pm, self.daily,
            signals_provider=lambda: [_signal()],
            health_provider=lambda: {"ok": True, "mt5_connected": False,
                                      "last_bar_ms": 1},
        )
        return self.dash.build_app()


class TestSnapshotEndpoint(_BaseDashCase):
    async def test_snapshot_returns_all_sections(self):
        async with self.client.request("GET", "/") as resp:
            data = await resp.json()
        self.assertEqual(resp.status, 200)
        self.assertIn("positions", data)
        self.assertIn("pendings", data)
        self.assertIn("daily", data)
        self.assertIn("signals", data)
        self.assertIn("health", data)


class TestPositionsEndpoint(_BaseDashCase):
    async def test_positions_lists_registered(self):
        async with self.client.request("GET", "/positions") as resp:
            data = await resp.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["symbol"], "EURUSD")
        self.assertEqual(data[0]["pattern_name"], "FLAG")


class TestPendingsEndpoint(_BaseDashCase):
    async def test_pendings_lists_registered(self):
        async with self.client.request("GET", "/pendings") as resp:
            data = await resp.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["symbol"], "AUDJPY")
        self.assertFalse(data[0]["is_limit"])


class TestDailyEndpoint(_BaseDashCase):
    async def test_daily_keys(self):
        async with self.client.request("GET", "/daily") as resp:
            data = await resp.json()
        for k in ("trade_day", "peak_equity", "closed_pnl",
                  "floating_pnl", "trade_count", "max_dd_today"):
            self.assertIn(k, data)


class TestSignalsEndpoint(_BaseDashCase):
    async def test_signals_serialised(self):
        async with self.client.request("GET", "/signals") as resp:
            data = await resp.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["pattern_name"], "FLAG")
        self.assertEqual(data[0]["grade"], "A")


class TestHealthEndpoint(_BaseDashCase):
    async def test_health_reports_callable_result(self):
        async with self.client.request("GET", "/health") as resp:
            data = await resp.json()
        self.assertTrue(data["ok"])
        self.assertFalse(data["mt5_connected"])
