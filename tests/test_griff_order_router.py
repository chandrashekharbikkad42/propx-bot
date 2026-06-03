"""Phase 8D-Live — GriffOrderRouter tests. MT5 is fully mocked."""

from __future__ import annotations
import asyncio
import unittest
from unittest.mock import patch, MagicMock

import MetaTrader5 as mt5

from execution.order_router import (
    GriffOrderError,
    GriffOrderRouter,
    GriffOpenPosition,
    GriffPendingOrder,
    _TRANSIENT_RETCODES,
)
from strategy.patterns.base import Direction, Grade, PatternSignal


NOW_MSC = 1_778_976_000_000
EXPIRY_MSC = NOW_MSC + 3_600_000  # +1 hour


def _sig(pattern: str = "FLAG", direction: Direction = Direction.BUY,
         symbol: str = "EURUSD", entry: float = 1.1000, sl: float = 1.0980,
         tp: float = 1.1040) -> PatternSignal:
    if direction == Direction.SELL:
        # Swap to satisfy SELL: tp<entry<sl ordering.
        sl, tp = 1.1020, 1.0960
    return PatternSignal(
        pattern_name=pattern, symbol=symbol, direction=direction,
        entry=entry, sl=sl, tp=tp, confidence=0.8, grade=Grade.A,
        confluences_met=("a",), bar_time_msc=NOW_MSC,
    )


def _result(retcode=mt5.TRADE_RETCODE_DONE, price=1.10010, order=42, deal=42):
    r = MagicMock()
    r.retcode, r.price, r.order, r.deal, r.comment = retcode, price, order, deal, ""
    return r


async def _noop(*_, **__):
    return None


# ============================================================================
# DRY_RUN mode
# ============================================================================

class TestDryRunMode(unittest.TestCase):
    def test_dry_run_market_returns_synthetic_position(self):
        r = GriffOrderRouter(dry_run=True)
        pos = asyncio.run(r.place_market(
            _sig(), lots=0.1, ask=1.1001, bid=1.1000, now_msc=NOW_MSC,
        ))
        self.assertEqual(pos.mt5_ticket, -1)
        self.assertEqual(pos.entry_price, 1.1001)
        self.assertEqual(pos.side, Direction.BUY)

    def test_dry_run_pending_stop_returns_synthetic_order(self):
        r = GriffOrderRouter(dry_run=True)
        ord_ = asyncio.run(r.place_pending_stop(
            _sig("CONTINUATION"), lots=0.1,
            expiry_msc=EXPIRY_MSC, now_msc=NOW_MSC,
        ))
        self.assertEqual(ord_.mt5_ticket, -1)
        self.assertFalse(ord_.is_limit)
        self.assertEqual(ord_.expiry_msc, EXPIRY_MSC)

    def test_dry_run_pending_limit_marks_is_limit(self):
        r = GriffOrderRouter(dry_run=True)
        ord_ = asyncio.run(r.place_pending_limit(
            _sig("COMBO"), lots=0.1,
            expiry_msc=EXPIRY_MSC, now_msc=NOW_MSC,
        ))
        self.assertTrue(ord_.is_limit)

    def test_dry_run_never_calls_mt5_order_send(self):
        r = GriffOrderRouter(dry_run=True)
        with patch("execution.order_router.mt5.order_send") as m:
            asyncio.run(r.place_market(
                _sig(), lots=0.1, ask=1.1, bid=1.1, now_msc=NOW_MSC))
            asyncio.run(r.place_pending_stop(
                _sig(), lots=0.1, expiry_msc=EXPIRY_MSC, now_msc=NOW_MSC))
        m.assert_not_called()


# ============================================================================
# Market orders (real path, mocked MT5)
# ============================================================================

class TestMarketOrders(unittest.TestCase):
    def test_buy_uses_ask_price(self):
        r = GriffOrderRouter(dry_run=False)
        captured = []
        def _cap(req):
            captured.append(req)
            return _result(price=req["price"])
        with patch("execution.order_router.mt5.order_send", side_effect=_cap):
            asyncio.run(r.place_market(
                _sig(direction=Direction.BUY), lots=0.1,
                ask=1.1005, bid=1.1003, now_msc=NOW_MSC))
        self.assertEqual(captured[0]["price"], 1.1005)
        self.assertEqual(captured[0]["type"], mt5.ORDER_TYPE_BUY)

    def test_sell_uses_bid_price(self):
        r = GriffOrderRouter(dry_run=False)
        captured = []
        def _cap(req):
            captured.append(req)
            return _result(price=req["price"])
        with patch("execution.order_router.mt5.order_send", side_effect=_cap):
            asyncio.run(r.place_market(
                _sig(direction=Direction.SELL), lots=0.1,
                ask=1.1005, bid=1.1003, now_msc=NOW_MSC))
        self.assertEqual(captured[0]["price"], 1.1003)
        self.assertEqual(captured[0]["type"], mt5.ORDER_TYPE_SELL)

    def test_uses_griff_magic_number(self):
        r = GriffOrderRouter(dry_run=False)
        captured = []
        with patch("execution.order_router.mt5.order_send",
                   side_effect=lambda req: captured.append(req) or _result()):
            asyncio.run(r.place_market(
                _sig(), lots=0.1, ask=1.1, bid=1.1, now_msc=NOW_MSC))
        # Distinct from xau_hft magic (786543).
        self.assertEqual(captured[0]["magic"], 786544)


# ============================================================================
# Pending orders + hybrid expiry
# ============================================================================

class TestPendingOrders(unittest.TestCase):
    def test_pending_stop_buy_uses_buy_stop_type(self):
        r = GriffOrderRouter(dry_run=False)
        captured = []
        with patch("execution.order_router.mt5.order_send",
                   side_effect=lambda req: captured.append(req) or _result()):
            asyncio.run(r.place_pending_stop(
                _sig(direction=Direction.BUY), lots=0.1,
                expiry_msc=EXPIRY_MSC, now_msc=NOW_MSC))
        self.assertEqual(captured[0]["type"], mt5.ORDER_TYPE_BUY_STOP)
        self.assertEqual(captured[0]["action"], mt5.TRADE_ACTION_PENDING)

    def test_pending_stop_sell_uses_sell_stop_type(self):
        r = GriffOrderRouter(dry_run=False)
        captured = []
        with patch("execution.order_router.mt5.order_send",
                   side_effect=lambda req: captured.append(req) or _result()):
            asyncio.run(r.place_pending_stop(
                _sig(direction=Direction.SELL), lots=0.1,
                expiry_msc=EXPIRY_MSC, now_msc=NOW_MSC))
        self.assertEqual(captured[0]["type"], mt5.ORDER_TYPE_SELL_STOP)

    def test_pending_limit_buy_uses_buy_limit_type(self):
        r = GriffOrderRouter(dry_run=False)
        captured = []
        with patch("execution.order_router.mt5.order_send",
                   side_effect=lambda req: captured.append(req) or _result()):
            asyncio.run(r.place_pending_limit(
                _sig(direction=Direction.BUY, pattern="COMBO"), lots=0.1,
                expiry_msc=EXPIRY_MSC, now_msc=NOW_MSC))
        self.assertEqual(captured[0]["type"], mt5.ORDER_TYPE_BUY_LIMIT)

    def test_hybrid_expiry_sets_broker_side_expiration_in_seconds(self):
        r = GriffOrderRouter(dry_run=False)
        captured = []
        with patch("execution.order_router.mt5.order_send",
                   side_effect=lambda req: captured.append(req) or _result()):
            asyncio.run(r.place_pending_stop(
                _sig("CONTINUATION"), lots=0.1,
                expiry_msc=EXPIRY_MSC, now_msc=NOW_MSC))
        # MT5 wants epoch SECONDS not ms.
        self.assertEqual(captured[0]["type_time"], mt5.ORDER_TIME_SPECIFIED)
        self.assertEqual(captured[0]["expiration"], EXPIRY_MSC // 1000)


# ============================================================================
# Cancel pending (bot-side leg of hybrid expiry)
# ============================================================================

class TestCancelPending(unittest.TestCase):
    def test_dry_cancel_is_noop_returns_true(self):
        r = GriffOrderRouter(dry_run=True)
        ord_ = asyncio.run(r.place_pending_stop(
            _sig(), lots=0.1, expiry_msc=EXPIRY_MSC, now_msc=NOW_MSC))
        with patch("execution.order_router.mt5.order_send") as m:
            ok = asyncio.run(r.cancel_pending(ord_))
        self.assertTrue(ok)
        m.assert_not_called()

    def test_real_cancel_sends_remove_action(self):
        r = GriffOrderRouter(dry_run=False)
        ord_ = GriffPendingOrder(
            order_id="o1", mt5_ticket=42, symbol="EURUSD", side=Direction.BUY,
            lots=0.1, pending_price=1.1, sl_price=1.099, tp_price=1.102,
            expiry_msc=EXPIRY_MSC, signal_id="x", pattern_name="CONTINUATION",
            is_limit=False,
        )
        captured = []
        with patch("execution.order_router.mt5.order_send",
                   side_effect=lambda req: captured.append(req) or _result()):
            ok = asyncio.run(r.cancel_pending(ord_))
        self.assertTrue(ok)
        self.assertEqual(captured[0]["action"], mt5.TRADE_ACTION_REMOVE)
        self.assertEqual(captured[0]["order"], 42)

    def test_cancel_already_gone_returns_true(self):
        r = GriffOrderRouter(dry_run=False)
        ord_ = GriffPendingOrder(
            order_id="o2", mt5_ticket=99, symbol="EURUSD", side=Direction.BUY,
            lots=0.1, pending_price=1.1, sl_price=1.099, tp_price=1.102,
            expiry_msc=EXPIRY_MSC, signal_id="x", pattern_name="C", is_limit=False,
        )
        with patch("execution.order_router.mt5.order_send",
                   return_value=_result(retcode=10027)):  # already removed
            ok = asyncio.run(r.cancel_pending(ord_))
        self.assertTrue(ok)


# ============================================================================
# Retry / error handling
# ============================================================================

class TestRetryAndErrors(unittest.TestCase):
    def test_transient_then_success(self):
        r = GriffOrderRouter(dry_run=False)
        transient = next(iter(_TRANSIENT_RETCODES))
        seq = [_result(retcode=transient), _result()]
        with patch("execution.order_router.mt5.order_send", side_effect=seq), \
             patch("execution.order_router.asyncio.sleep", new=_noop):
            pos = asyncio.run(r.place_market(
                _sig(), lots=0.1, ask=1.1, bid=1.1, now_msc=NOW_MSC))
        self.assertIsInstance(pos, GriffOpenPosition)

    def test_permanent_reject_raises(self):
        r = GriffOrderRouter(dry_run=False)
        with patch("execution.order_router.mt5.order_send",
                   return_value=_result(retcode=10013)):  # INVALID
            with self.assertRaises(GriffOrderError):
                asyncio.run(r.place_market(
                    _sig(), lots=0.1, ask=1.1, bid=1.1, now_msc=NOW_MSC))

    def test_retries_exhausted_raises(self):
        r = GriffOrderRouter(dry_run=False)
        transient = next(iter(_TRANSIENT_RETCODES))
        with patch("execution.order_router.mt5.order_send",
                   return_value=_result(retcode=transient)), \
             patch("execution.order_router.asyncio.sleep", new=_noop):
            with self.assertRaises(GriffOrderError):
                asyncio.run(r.place_market(
                    _sig(), lots=0.1, ask=1.1, bid=1.1, now_msc=NOW_MSC))


# ============================================================================
# Close + modify SL
# ============================================================================

class TestCloseAndModify(unittest.TestCase):
    def _pos(self, side=Direction.BUY) -> GriffOpenPosition:
        return GriffOpenPosition(
            position_id="p1", mt5_ticket=7, symbol="EURUSD", side=side,
            lots=0.1, entry_price=1.1, sl_price=1.099, tp_price=1.102,
            opened_msc=NOW_MSC, signal_id="x", pattern_name="FLAG",
        )

    def test_close_buy_uses_bid(self):
        r = GriffOrderRouter(dry_run=False)
        captured = []
        with patch("execution.order_router.mt5.order_send",
                   side_effect=lambda req: captured.append(req) or _result(price=req["price"])):
            exit_px = asyncio.run(r.close_position(
                self._pos(Direction.BUY), bid=1.0998, ask=1.1000, now_msc=NOW_MSC))
        self.assertEqual(exit_px, 1.0998)
        self.assertEqual(captured[0]["type"], mt5.ORDER_TYPE_SELL)

    def test_modify_sl_sends_sltp_action(self):
        r = GriffOrderRouter(dry_run=False)
        captured = []
        with patch("execution.order_router.mt5.order_send",
                   side_effect=lambda req: captured.append(req) or _result()):
            ok = asyncio.run(r.modify_sl(self._pos(), new_sl=1.0995))
        self.assertTrue(ok)
        self.assertEqual(captured[0]["action"], mt5.TRADE_ACTION_SLTP)
        self.assertEqual(captured[0]["sl"], 1.0995)

    def test_dry_close_does_not_call_mt5(self):
        r = GriffOrderRouter(dry_run=True)
        with patch("execution.order_router.mt5.order_send") as m:
            px = asyncio.run(r.close_position(
                self._pos(), bid=1.0998, ask=1.1000, now_msc=NOW_MSC))
        self.assertEqual(px, 1.0998)
        m.assert_not_called()
