"""Phase 8D-Live — GriffPositionManager tests."""

from __future__ import annotations
import asyncio
import unittest

from data.bar_aggregator import Bar
from execution.order_router import (
    GriffOpenPosition,
    GriffOrderRouter,
    GriffPendingOrder,
)
from execution.position_manager import GriffPositionManager
from risk.trailing_sl import TrailingStopLoss
from strategy.patterns.base import Direction
from strategy.swing_tracker import SwingTracker


HOUR_MS = 3_600_000


def _bar(idx: int, h: float, l: float, sym: str = "EURUSD",
         o: float = None, c: float = None) -> Bar:
    o = o if o is not None else (h + l) / 2
    c = c if c is not None else (h + l) / 2
    return Bar(symbol=sym, time_msc=idx * HOUR_MS, open=o, high=h, low=l,
               close=c, volume=1, spread_mean=0.0)


def _make_mgr() -> tuple[GriffPositionManager, GriffOrderRouter, SwingTracker]:
    router = GriffOrderRouter(dry_run=True)
    st = SwingTracker()
    trail = TrailingStopLoss(st)
    mgr = GriffPositionManager(router, st, trail)
    return mgr, router, st


def _pos(symbol: str = "EURUSD", side: Direction = Direction.BUY,
         entry: float = 1.1000, sl: float = 1.0980) -> GriffOpenPosition:
    return GriffOpenPosition(
        position_id="pos1", mt5_ticket=42, symbol=symbol, side=side,
        lots=0.1, entry_price=entry, sl_price=sl, tp_price=1.1040,
        opened_msc=0, signal_id="x", pattern_name="FLAG",
    )


def _pending(symbol: str = "EURUSD", expiry_msc: int = 100 * HOUR_MS) -> GriffPendingOrder:
    return GriffPendingOrder(
        order_id="o1", mt5_ticket=99, symbol=symbol, side=Direction.BUY,
        lots=0.1, pending_price=1.1010, sl_price=1.0990, tp_price=1.1040,
        expiry_msc=expiry_msc, signal_id="x", pattern_name="CONTINUATION",
        is_limit=False,
    )


# ============================================================================
# Registration
# ============================================================================

class TestRegistration(unittest.TestCase):
    def test_register_position_added_to_open(self):
        mgr, _, _ = _make_mgr()
        mgr.register_position(_pos())
        self.assertEqual(len(mgr.open_positions), 1)
        self.assertEqual(mgr.positions_for("EURUSD")[0].position_id, "pos1")

    def test_register_pending_added(self):
        mgr, _, _ = _make_mgr()
        mgr.register_pending(_pending())
        self.assertEqual(len(mgr.pending_orders), 1)
        self.assertEqual(mgr.pendings_for("EURUSD")[0].order_id, "o1")

    def test_filter_by_pair(self):
        mgr, _, _ = _make_mgr()
        mgr.register_position(_pos(symbol="EURUSD"))
        mgr.register_position(GriffOpenPosition(
            position_id="pos2", mt5_ticket=43, symbol="AUDJPY",
            side=Direction.BUY, lots=0.1, entry_price=95.0,
            sl_price=94.8, tp_price=95.4, opened_msc=0,
            signal_id="y", pattern_name="FLAG",
        ))
        self.assertEqual(len(mgr.positions_for("EURUSD")), 1)
        self.assertEqual(len(mgr.positions_for("AUDJPY")), 1)


# ============================================================================
# Pending → fill promotion
# ============================================================================

class TestPendingFill(unittest.TestCase):
    def test_on_filled_promotes_to_position(self):
        mgr, _, _ = _make_mgr()
        mgr.register_pending(_pending())
        pos = mgr.on_pending_filled(
            "o1", fill_price=1.1010, mt5_position_ticket=500,
            fill_msc=10 * HOUR_MS,
        )
        self.assertIsNotNone(pos)
        self.assertEqual(pos.mt5_ticket, 500)
        self.assertEqual(pos.entry_price, 1.1010)
        # Pending removed, position added.
        self.assertEqual(len(mgr.pending_orders), 0)
        self.assertEqual(len(mgr.open_positions), 1)

    def test_on_filled_unknown_order_id_returns_none(self):
        mgr, _, _ = _make_mgr()
        pos = mgr.on_pending_filled(
            "ghost", fill_price=1.1, mt5_position_ticket=1, fill_msc=0)
        self.assertIsNone(pos)


# ============================================================================
# Maintain — trailing SL update path
# ============================================================================

class TestMaintainTrailing(unittest.TestCase):
    def test_swing_tracker_consumes_bar(self):
        mgr, _, st = _make_mgr()
        asyncio.run(mgr.maintain("EURUSD", _bar(0, 1.10, 1.09), now_msc=0))
        # SwingTracker has 1 bar in its deque (no confirmed swing yet).
        self.assertIsNone(st.get_last_swing_high("EURUSD"))

    def test_trailing_sl_raise_called_on_higher_swing_low(self):
        mgr, _, st = _make_mgr()
        # Seed SwingTracker so the trail has a swing low to anchor on.
        st.update("EURUSD", _bar(0, 1.1010, 1.0995))
        st.update("EURUSD", _bar(1, 1.1005, 1.0985))   # mid candidate
        st.update("EURUSD", _bar(2, 1.1020, 1.0990))   # confirms swing low @ 1.0985
        mgr.register_position(_pos(entry=1.1000, sl=1.0970))
        # Tight trail offset → new SL = 1.0985 - 2pip*0.0001 = 1.09848.
        rep = asyncio.run(mgr.maintain(
            "EURUSD", _bar(3, 1.1025, 1.1000), now_msc=3 * HOUR_MS,
        ))
        # At least one SL update should be reported.
        self.assertEqual(len(rep.sl_updates), 1)
        updated = mgr.open_positions[0]
        self.assertGreater(updated.sl_price, 1.0970)


# ============================================================================
# SL hit detection
# ============================================================================

class TestSlHit(unittest.TestCase):
    def test_long_sl_hit_when_bar_low_at_or_below(self):
        mgr, _, _ = _make_mgr()
        mgr.register_position(_pos(entry=1.10, sl=1.099))
        # Bar low = 1.0985 < 1.099 → hit.
        rep = asyncio.run(mgr.maintain(
            "EURUSD", _bar(1, 1.1005, 1.0985), now_msc=HOUR_MS,
        ))
        self.assertEqual(rep.closed_positions, ("pos1",))
        self.assertEqual(len(mgr.open_positions), 0)

    def test_short_sl_hit_when_bar_high_at_or_above(self):
        mgr, _, _ = _make_mgr()
        mgr.register_position(_pos(side=Direction.SELL, entry=1.10, sl=1.101))
        rep = asyncio.run(mgr.maintain(
            "EURUSD", _bar(1, 1.1015, 1.0995), now_msc=HOUR_MS,
        ))
        self.assertEqual(rep.closed_positions, ("pos1",))

    def test_sl_not_hit_when_bar_inside_range(self):
        mgr, _, _ = _make_mgr()
        mgr.register_position(_pos(entry=1.10, sl=1.095))
        rep = asyncio.run(mgr.maintain(
            "EURUSD", _bar(1, 1.1005, 1.0980), now_msc=HOUR_MS,
        ))
        self.assertEqual(rep.closed_positions, ())
        self.assertEqual(len(mgr.open_positions), 1)


# ============================================================================
# Pending expiry — hybrid bot-side leg
# ============================================================================

class TestPendingExpiry(unittest.TestCase):
    def test_pending_cancelled_after_expiry_msc(self):
        mgr, _, _ = _make_mgr()
        # Expiry at hour 1.
        mgr.register_pending(_pending(expiry_msc=1 * HOUR_MS))
        # Maintain at hour 2 → past expiry → cancel.
        rep = asyncio.run(mgr.maintain(
            "EURUSD", _bar(2, 1.10, 1.09), now_msc=2 * HOUR_MS,
        ))
        self.assertEqual(rep.cancelled_pendings, ("o1",))
        self.assertEqual(len(mgr.pending_orders), 0)

    def test_pending_kept_before_expiry(self):
        mgr, _, _ = _make_mgr()
        mgr.register_pending(_pending(expiry_msc=10 * HOUR_MS))
        rep = asyncio.run(mgr.maintain(
            "EURUSD", _bar(1, 1.10, 1.09), now_msc=HOUR_MS,
        ))
        self.assertEqual(rep.cancelled_pendings, ())
        self.assertEqual(len(mgr.pending_orders), 1)


# ============================================================================
# Multi-pair isolation
# ============================================================================

class TestMultiPair(unittest.TestCase):
    def test_maintain_one_pair_does_not_affect_others(self):
        mgr, _, _ = _make_mgr()
        mgr.register_position(_pos(symbol="EURUSD", entry=1.10, sl=1.099))
        mgr.register_position(GriffOpenPosition(
            position_id="audp", mt5_ticket=43, symbol="AUDJPY",
            side=Direction.BUY, lots=0.1, entry_price=95.0,
            sl_price=94.9, tp_price=95.4, opened_msc=0,
            signal_id="y", pattern_name="FLAG",
        ))
        # SL-hit a EURUSD bar; AUDJPY position untouched.
        rep = asyncio.run(mgr.maintain(
            "EURUSD", _bar(1, 1.1005, 1.0985), now_msc=HOUR_MS,
        ))
        self.assertEqual(rep.closed_positions, ("pos1",))
        # AUDJPY position still there.
        self.assertEqual(len(mgr.positions_for("AUDJPY")), 1)


# ============================================================================
# Manual forget
# ============================================================================

class TestForget(unittest.TestCase):
    def test_forget_position_removes(self):
        mgr, _, _ = _make_mgr()
        mgr.register_position(_pos())
        removed = mgr.forget_position("pos1")
        self.assertIsNotNone(removed)
        self.assertEqual(len(mgr.open_positions), 0)

    def test_forget_unknown_returns_none(self):
        mgr, _, _ = _make_mgr()
        self.assertIsNone(mgr.forget_position("nope"))
