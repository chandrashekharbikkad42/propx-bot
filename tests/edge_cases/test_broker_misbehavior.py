"""Phase-5 / Broker Misbehavior — adversarial MT5 stub tests.

Coverage focus (per Phase 5 brief):

  - Order rejected (retry logic)
  - Partial fill (50% filled, rest pending)
  - Requote loop (price moved 3x)
  - Position closed by broker (margin call) — bot detects
  - SL/TP modification rejected
  - Disconnect during order send (unknown state)
  - Reconnect with orphan position
  - Duplicate order (idempotency)
  - Slippage beyond tolerance
"""

from __future__ import annotations
import asyncio
from typing import Iterable

import pytest

from execution.order_router import (
    GriffOrderError, GriffOrderRouter, _ticket_from_result,
)
from execution.position_manager import GriffPositionManager
from execution.live_broker import LiveBroker, LiveBrokerError, MAX_RETRIES
from execution.order import OrderIntent, Side, SignalType
from execution.position import CloseReason, Position, PositionState
from strategy.patterns.base import Direction, Grade, PatternSignal
from strategy.swing_tracker import SwingTracker
from risk.trailing_sl import TrailingStopLoss
from utils.session import SessionLabel

from tests.edge_cases.fixtures.broker_failures import (
    inject_disconnect, inject_invalid_stops, inject_market_closed,
    inject_none_result, inject_order_not_found_on_cancel,
    inject_partial_fill, inject_permanent_no_money, inject_price_off,
    inject_reject_then_success, inject_requote_loop, inject_slippage,
    inject_zero_ticket_done, inject_deal_ticket_only,
    queue_retcode_sequence,
)
from tests.execution.fixtures.mock_mt5 import (
    DealInfo, MockMT5, OrderSendResult, PositionInfo,
    TRADE_RETCODE_DONE, TRADE_RETCODE_REQUOTE, TRADE_RETCODE_REJECT,
    TRADE_RETCODE_PRICE_OFF, TRADE_RETCODE_MARKET_CLOSED,
    TRADE_RETCODE_CONNECTION, TRADE_RETCODE_INVALID_STOPS,
    TRADE_RETCODE_NO_MONEY, TRADE_RETCODE_ORDER_NOT_FOUND,
)
from tests.execution.fixtures.mock_orders import (
    make_intent, make_signal, make_signal_sell,
)
from tests.execution.fixtures.mock_positions import (
    make_griff_open, make_griff_pending, make_position, make_tick,
)


def run(coro):
    return asyncio.run(coro)


# ===========================================================================
# 1. ORDER REJECT — RETRY LOGIC (Router + LiveBroker)
# ===========================================================================

class TestRetryOnTransient:
    @pytest.mark.parametrize("transient", sorted({
        TRADE_RETCODE_REQUOTE, TRADE_RETCODE_REJECT,
        TRADE_RETCODE_PRICE_OFF, TRADE_RETCODE_MARKET_CLOSED,
        TRADE_RETCODE_CONNECTION,
    }))
    def test_router_retries_each_transient_retcode(self, patch_router_mt5,
                                                    transient):
        m = patch_router_mt5
        m.retcode_queue.append(OrderSendResult(retcode=transient))
        m.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_DONE,
                                                order=42, price=2000.0))
        router = GriffOrderRouter(dry_run=False)
        sig = make_signal(direction=Direction.BUY, entry=2000.0,
                          sl=1995.0, tp=2010.0)
        pos = run(router.place_market(sig, 0.10, ask=2000.0, bid=1999.0,
                                       now_msc=0))
        assert pos.mt5_ticket == 42
        assert len(m.sent_requests) == 2

    @pytest.mark.parametrize("n", [1, 2])
    def test_router_succeeds_after_n_rejects(self, patch_router_mt5, n):
        m = patch_router_mt5
        inject_reject_then_success(m, n_rejects=n)
        router = GriffOrderRouter(dry_run=False)
        sig = make_signal(entry=2000.0, sl=1995.0, tp=2010.0)
        pos = run(router.place_market(sig, 0.10, ask=2000.0, bid=1999.0,
                                       now_msc=0))
        assert pos.mt5_ticket > 0
        assert len(m.sent_requests) == n + 1

    def test_router_exhausts_retries_on_persistent_reject(self,
                                                            patch_router_mt5):
        m = patch_router_mt5
        for _ in range(MAX_RETRIES):
            m.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_REJECT))
        router = GriffOrderRouter(dry_run=False)
        sig = make_signal(entry=2000.0, sl=1995.0, tp=2010.0)
        with pytest.raises(GriffOrderError, match="exhausted retries"):
            run(router.place_market(sig, 0.10, ask=2000.0, bid=1999.0,
                                     now_msc=0))


class TestPermanentReject:
    @pytest.mark.parametrize("retcode", [
        TRADE_RETCODE_NO_MONEY, TRADE_RETCODE_INVALID_STOPS,
        9999,  # unknown
    ])
    def test_router_raises_on_non_transient(self, patch_router_mt5, retcode):
        m = patch_router_mt5
        m.retcode_queue.append(OrderSendResult(retcode=retcode))
        router = GriffOrderRouter(dry_run=False)
        sig = make_signal(entry=2000.0, sl=1995.0, tp=2010.0)
        with pytest.raises(GriffOrderError, match="permanent reject"):
            run(router.place_market(sig, 0.10, ask=2000.0, bid=1999.0,
                                     now_msc=0))


class TestLiveBrokerRetry:
    @pytest.mark.parametrize("retcode", sorted({
        TRADE_RETCODE_REQUOTE, TRADE_RETCODE_REJECT,
        TRADE_RETCODE_PRICE_OFF, TRADE_RETCODE_MARKET_CLOSED,
        TRADE_RETCODE_CONNECTION,
    }))
    def test_live_broker_retries_transient(self, patch_live_broker_mt5,
                                            retcode):
        m = patch_live_broker_mt5
        m.retcode_queue.append(OrderSendResult(retcode=retcode))
        m.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_DONE,
                                                order=99, price=2000.0))
        b = LiveBroker("XAUUSD")
        i = make_intent(side=Side.BUY)
        t = make_tick(bid=1999.5, ask=2000.0)
        pos = run(b.fill_market_order(i, t))
        assert pos.entry_price == 2000.0

    @pytest.mark.parametrize("retcode", [
        TRADE_RETCODE_NO_MONEY, TRADE_RETCODE_INVALID_STOPS,
    ])
    def test_live_broker_non_transient_raises(self, patch_live_broker_mt5,
                                                retcode):
        m = patch_live_broker_mt5
        m.retcode_queue.append(OrderSendResult(retcode=retcode))
        b = LiveBroker("XAUUSD")
        with pytest.raises(LiveBrokerError, match="rejected retcode"):
            run(b.fill_market_order(make_intent(), make_tick()))

    def test_live_broker_exhausts_on_persistent_transient(self,
                                                          patch_live_broker_mt5):
        m = patch_live_broker_mt5
        for _ in range(MAX_RETRIES):
            m.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_REQUOTE))
        b = LiveBroker("XAUUSD")
        with pytest.raises(LiveBrokerError, match="after .* attempts"):
            run(b.fill_market_order(make_intent(), make_tick()))

    def test_live_broker_handles_none_result_then_recovers(self,
                                                            patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.set_last_error(1, "disconnect")
        # first call returns None (via monkey-patch), then queue handles 2nd call
        inject_none_result(m)
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=33, price=2000.0)
        b = LiveBroker("XAUUSD")
        pos = run(b.fill_market_order(make_intent(), make_tick()))
        assert pos.entry_price == 2000.0


# ===========================================================================
# 2. REQUOTE LOOP — exceeds MAX_RETRIES
# ===========================================================================

class TestRequoteLoop:
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_router_fails_after_n_requotes(self, patch_router_mt5, n):
        m = patch_router_mt5
        inject_requote_loop(m, n=n)
        router = GriffOrderRouter(dry_run=False)
        sig = make_signal()
        with pytest.raises(GriffOrderError):
            run(router.place_market(sig, 0.10, ask=2000.0, bid=1999.0,
                                     now_msc=0))

    def test_router_requote_then_success(self, patch_router_mt5):
        m = patch_router_mt5
        m.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_REQUOTE))
        m.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_REQUOTE))
        m.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_DONE,
                                                order=5, price=2000.0))
        router = GriffOrderRouter(dry_run=False)
        pos = run(router.place_market(
            make_signal(entry=2000.0, sl=1995.0, tp=2010.0),
            0.1, ask=2000.0, bid=1999.0, now_msc=0,
        ))
        assert pos.mt5_ticket == 5

    @pytest.mark.parametrize("n_retries", [1, 2])
    def test_requote_count_matches_send_count(self, patch_router_mt5, n_retries):
        m = patch_router_mt5
        inject_reject_then_success(m, n_rejects=n_retries)
        router = GriffOrderRouter(dry_run=False)
        run(router.place_market(make_signal(), 0.10,
                                 ask=2000.0, bid=1999.0, now_msc=0))
        assert len(m.sent_requests) == n_retries + 1


# ===========================================================================
# 3. PARTIAL FILL — currently silent slippage
# ===========================================================================

class TestPartialFill:
    @pytest.mark.parametrize("req,filled,bar_msc", [
        (1.0, 0.5,  1_700_000_000_000),
        (0.5, 0.1,  1_700_000_001_000),
        (10.0, 5.0, 1_700_000_002_000),
        (0.10, 0.05, 1_700_000_003_000),
    ])
    def test_router_records_actual_filled_volume(self, patch_router_mt5,
                                                   req, filled, bar_msc):
        """After Phase-5 bug-fix: router stores result.volume (actually filled)
        — partial fills are surfaced into position bookkeeping rather than
        silently mis-bookkept."""
        m = patch_router_mt5
        inject_partial_fill(m, requested_volume=req, filled_volume=filled)
        router = GriffOrderRouter(dry_run=False)
        sig = make_signal(entry=2000.0, sl=1995.0, tp=2010.0,
                          bar_time_msc=bar_msc)
        pos = run(router.place_market(sig, req, ask=2000.0, bid=1999.0,
                                       now_msc=0))
        assert pos.lots == filled

    def test_router_should_record_actual_filled_volume(self, patch_router_mt5):
        m = patch_router_mt5
        inject_partial_fill(m, requested_volume=1.0, filled_volume=0.5)
        router = GriffOrderRouter(dry_run=False)
        sig = make_signal(entry=2000.0, sl=1995.0, tp=2010.0)
        pos = run(router.place_market(sig, 1.0, ask=2000.0, bid=1999.0,
                                       now_msc=0))
        assert pos.lots == 0.5  # post-fix correct behaviour


# ===========================================================================
# 4. SLIPPAGE BEYOND TOLERANCE
# ===========================================================================

class TestSlippage:
    @pytest.mark.parametrize("req_price,fill_price", [
        (2000.0, 2010.0),
        (2000.0, 1990.0),
        (2000.0, 2050.0),
    ])
    def test_router_accepts_any_fill_price(self, patch_router_mt5,
                                            req_price, fill_price):
        """Router does NOT enforce a slippage budget; it returns whatever
        price MT5 reports. The `deviation` field passed to MT5 is the only
        upstream guard, and the mock doesn't honour it."""
        m = patch_router_mt5
        inject_slippage(m, req_price, fill_price)
        router = GriffOrderRouter(dry_run=False)
        sig = make_signal(entry=req_price, sl=req_price - 5.0,
                          tp=req_price + 10.0)
        pos = run(router.place_market(sig, 0.1, ask=req_price, bid=req_price - 0.5,
                                       now_msc=0))
        assert pos.entry_price == fill_price

    def test_live_broker_records_actual_fill_price(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=7, price=2050.0)
        b = LiveBroker("XAUUSD")
        i = make_intent(intended_price=2000.0)
        t = make_tick(bid=1999.5, ask=2000.0)
        pos = run(b.fill_market_order(i, t))
        assert pos.entry_price == 2050.0


# ===========================================================================
# 5. SL / TP MODIFICATION REJECTED
# ===========================================================================

class TestModifySLRejection:
    def test_modify_sl_returns_false_on_reject(self, patch_router_mt5):
        m = patch_router_mt5
        m.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_REJECT))
        router = GriffOrderRouter(dry_run=False)
        pos = make_griff_open(symbol="XAUUSD", mt5_ticket=42,
                               sl_price=1995.0, tp_price=2010.0)
        ok = run(router.modify_sl(pos, new_sl=1996.0))
        assert ok is False

    @pytest.mark.parametrize("retcode", [
        TRADE_RETCODE_REJECT, TRADE_RETCODE_INVALID_STOPS,
        TRADE_RETCODE_REQUOTE, TRADE_RETCODE_PRICE_OFF,
        TRADE_RETCODE_CONNECTION,
    ])
    def test_modify_sl_false_for_all_non_done(self, patch_router_mt5, retcode):
        m = patch_router_mt5
        m.retcode_queue.append(OrderSendResult(retcode=retcode))
        router = GriffOrderRouter(dry_run=False)
        pos = make_griff_open()
        ok = run(router.modify_sl(pos, new_sl=1996.0))
        assert ok is False

    def test_modify_sl_done_returns_true(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1, price=0.0)
        router = GriffOrderRouter(dry_run=False)
        ok = run(router.modify_sl(make_griff_open(), new_sl=1996.0))
        assert ok is True


# ===========================================================================
# 6. CANCEL PENDING — order already gone
# ===========================================================================

class TestCancelPending:
    def test_cancel_returns_true_on_done(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE)
        router = GriffOrderRouter(dry_run=False)
        ok = run(router.cancel_pending(make_griff_pending()))
        assert ok is True

    def test_cancel_returns_true_when_order_not_found(self, patch_router_mt5):
        m = patch_router_mt5
        inject_order_not_found_on_cancel(m)
        router = GriffOrderRouter(dry_run=False)
        ok = run(router.cancel_pending(make_griff_pending()))
        assert ok is True

    @pytest.mark.parametrize("retcode", [
        TRADE_RETCODE_REJECT, TRADE_RETCODE_CONNECTION,
        TRADE_RETCODE_INVALID_STOPS,
    ])
    def test_cancel_returns_false_on_real_failure(self, patch_router_mt5,
                                                    retcode):
        m = patch_router_mt5
        m.retcode_queue.append(OrderSendResult(retcode=retcode))
        router = GriffOrderRouter(dry_run=False)
        ok = run(router.cancel_pending(make_griff_pending()))
        assert ok is False

    def test_dry_run_cancel_is_noop(self):
        router = GriffOrderRouter(dry_run=True)
        ok = run(router.cancel_pending(make_griff_pending()))
        assert ok is True


# ===========================================================================
# 7. DISCONNECT MID-ORDER
# ===========================================================================

class TestDisconnect:
    def test_single_disconnect_retried(self, patch_router_mt5):
        m = patch_router_mt5
        inject_disconnect(m)
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=11, price=2000.0)
        router = GriffOrderRouter(dry_run=False)
        pos = run(router.place_market(make_signal(), 0.1,
                                       ask=2000.0, bid=1999.0, now_msc=0))
        assert pos.mt5_ticket == 11

    def test_persistent_disconnect_fails(self, patch_router_mt5):
        m = patch_router_mt5
        for _ in range(MAX_RETRIES):
            inject_disconnect(m)
        router = GriffOrderRouter(dry_run=False)
        with pytest.raises(GriffOrderError):
            run(router.place_market(make_signal(), 0.1,
                                     ask=2000.0, bid=1999.0, now_msc=0))


# ===========================================================================
# 8. POSITION VANISHED — broker closed it (margin call / SL hit at MT5)
# ===========================================================================

class TestPositionVanished:
    def test_live_broker_reconciles_via_deal_history(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        b = LiveBroker("XAUUSD")
        # Open first — note max_hold_until_msc set to FUTURE so time-exit doesn't fire.
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=77, price=2000.0)
        intent = make_intent(side=Side.BUY, sl_price=1995.0, tp_price=2010.0,
                             max_hold_until_msc=2_000_000_000_000)
        pos = run(b.fill_market_order(intent,
                                       make_tick(bid=1999.5, ask=2000.0,
                                                  time_msc=1_700_000_000_000)))
        # Broker closes at SL — positions_get returns empty, deal history has SL price.
        m.positions = []
        m.deals = [DealInfo(price=1995.0, profit=-5.0, time_msc=0)]
        next_tick = make_tick(bid=1995.0, ask=1995.5,
                              time_msc=1_700_000_001_000)
        closed = run(b.check_position_exit(pos, next_tick))
        assert closed is not None
        assert closed.state == PositionState.CLOSED
        assert closed.close_reason == CloseReason.SL_HIT

    def test_live_broker_reconcile_tp_when_price_at_tp(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        b = LiveBroker("XAUUSD")
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=88, price=2000.0)
        intent = make_intent(side=Side.BUY, sl_price=1995.0, tp_price=2010.0,
                             max_hold_until_msc=2_000_000_000_000)
        pos = run(b.fill_market_order(intent,
                                       make_tick(bid=1999.5, ask=2000.0,
                                                  time_msc=1_700_000_000_000)))
        m.positions = []
        m.deals = [DealInfo(price=2010.0, profit=10.0, time_msc=0)]
        closed = run(b.check_position_exit(pos,
                                            make_tick(bid=2010.0, ask=2010.5,
                                                       time_msc=1_700_000_001_000)))
        assert closed.close_reason == CloseReason.TP_HIT

    def test_live_broker_reconcile_manual_when_outside_sl_tp_band(self,
                                                                    patch_live_broker_mt5):
        m = patch_live_broker_mt5
        b = LiveBroker("XAUUSD")
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=99, price=2000.0)
        intent = make_intent(side=Side.BUY, sl_price=1995.0, tp_price=2010.0,
                             max_hold_until_msc=2_000_000_000_000)
        pos = run(b.fill_market_order(intent,
                                       make_tick(bid=1999.5, ask=2000.0,
                                                  time_msc=1_700_000_000_000)))
        m.positions = []
        m.deals = [DealInfo(price=2001.0, profit=1.0, time_msc=0)]
        closed = run(b.check_position_exit(pos,
                                            make_tick(bid=2001.0, ask=2001.5,
                                                       time_msc=1_700_000_001_000)))
        assert closed.close_reason == CloseReason.MANUAL

    def test_live_broker_synthetic_close_when_no_deal_history(self,
                                                                patch_live_broker_mt5):
        m = patch_live_broker_mt5
        b = LiveBroker("XAUUSD")
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=10, price=2000.0)
        intent = make_intent(side=Side.BUY, sl_price=1995.0, tp_price=2010.0,
                             max_hold_until_msc=2_000_000_000_000)
        pos = run(b.fill_market_order(intent,
                                       make_tick(bid=1999.5, ask=2000.0,
                                                  time_msc=1_700_000_000_000)))
        m.positions = []
        m.deals = []
        next_tick = make_tick(bid=1998.0, ask=1998.5,
                              time_msc=1_700_000_001_000)
        closed = run(b.check_position_exit(pos, next_tick))
        assert closed is not None
        assert closed.close_reason == CloseReason.MANUAL


# ===========================================================================
# 9. POSITION MANAGER — bot-side SL-hit detection
# ===========================================================================

class TestPositionMgrBookkeeping:
    @pytest.fixture
    def pm(self, patch_router_mt5):
        router = GriffOrderRouter(dry_run=True)
        tracker = SwingTracker()
        trail = TrailingStopLoss(tracker)
        return GriffPositionManager(router, tracker, trail)

    def test_register_and_forget_position(self, pm):
        p = make_griff_open()
        pm.register_position(p)
        assert p in pm.open_positions
        pm.forget_position(p.position_id)
        assert p not in pm.open_positions

    def test_register_and_forget_pending(self, pm):
        o = make_griff_pending()
        pm.register_pending(o)
        assert o in pm.pending_orders
        pm.forget_pending(o.order_id)
        assert o not in pm.pending_orders

    def test_positions_for_filters_by_pair(self, pm):
        a = make_griff_open(symbol="EURUSD")
        b = make_griff_open(symbol="XAUUSD")
        pm.register_position(a)
        pm.register_position(b)
        assert pm.positions_for("EURUSD") == (a,)
        assert pm.positions_for("XAUUSD") == (b,)

    def test_pendings_for_filters_by_pair(self, pm):
        a = make_griff_pending(symbol="EURUSD")
        b = make_griff_pending(symbol="XAUUSD")
        pm.register_pending(a)
        pm.register_pending(b)
        assert pm.pendings_for("EURUSD") == (a,)
        assert pm.pendings_for("XAUUSD") == (b,)


# ===========================================================================
# 10. ORPHAN POSITION ON RECONNECT
# ===========================================================================

class TestOrphanPositionRecovery:
    def test_positions_get_returns_unknown_position(self):
        """If MT5 reports a position the bot never registered, MockMT5 lets
        us simulate it; this is the surface the orchestrator should poll."""
        m = MockMT5()
        m.positions = [PositionInfo(ticket=123, symbol="XAUUSD",
                                     volume=0.1, price_open=2000.0)]
        result = m.positions_get()
        assert any(p.ticket == 123 for p in result)

    def test_positions_get_by_ticket(self):
        m = MockMT5()
        m.positions = [
            PositionInfo(ticket=1, symbol="EURUSD", volume=0.1, price_open=1.10),
            PositionInfo(ticket=2, symbol="XAUUSD", volume=0.05, price_open=2000.0),
        ]
        assert m.positions_get(ticket=1)[0].symbol == "EURUSD"
        assert m.positions_get(ticket=2)[0].symbol == "XAUUSD"

    def test_positions_get_by_symbol_returns_only_matching(self):
        m = MockMT5()
        m.positions = [
            PositionInfo(ticket=1, symbol="EURUSD", volume=0.1, price_open=1.1),
            PositionInfo(ticket=2, symbol="XAUUSD", volume=0.05, price_open=2000.0),
        ]
        eur = m.positions_get(symbol="EURUSD")
        assert len(eur) == 1
        assert eur[0].ticket == 1


# ===========================================================================
# 11. DUPLICATE ORDER (idempotency)
# ===========================================================================

class TestDuplicateOrder:
    def test_two_distinct_signals_produce_two_positions(self,
                                                          patch_router_mt5):
        """Distinct signals (different bar_time_msc) bypass the dedup window
        and both reach MT5 — only IDENTICAL submissions are rejected."""
        m = patch_router_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1, price=2000.0)
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=2, price=2000.0)
        router = GriffOrderRouter(dry_run=False)
        sig1 = make_signal(bar_time_msc=1_700_000_000_000)
        sig2 = make_signal(bar_time_msc=1_700_000_060_000)  # +60s
        p1 = run(router.place_market(sig1, 0.1, ask=2000.0, bid=1999.0, now_msc=0))
        p2 = run(router.place_market(sig2, 0.1, ask=2000.0, bid=1999.0, now_msc=0))
        assert p1.mt5_ticket == 1
        assert p2.mt5_ticket == 2
        assert p1.position_id != p2.position_id

    def test_duplicate_signal_should_be_deduped(self, patch_router_mt5):
        """Post-fix: an identical second submission is rejected with a
        duplicate error before reaching MT5."""
        m = patch_router_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1, price=2000.0)
        router = GriffOrderRouter(dry_run=False)
        sig = make_signal(bar_time_msc=1_700_000_000_000)
        run(router.place_market(sig, 0.1, ask=2000.0, bid=1999.0, now_msc=0))
        # Second submission of the exact same signal: should be rejected.
        with pytest.raises(GriffOrderError, match="duplicate"):
            run(router.place_market(sig, 0.1, ask=2000.0, bid=1999.0,
                                     now_msc=0))


# ===========================================================================
# 12. PENDING ORDER PATH
# ===========================================================================

class TestPendingOrders:
    @pytest.mark.parametrize("direction,is_limit", [
        (Direction.BUY, False),
        (Direction.SELL, False),
        (Direction.BUY, True),
        (Direction.SELL, True),
    ])
    def test_place_pending_routes_correctly(self, patch_router_mt5,
                                              direction, is_limit):
        from tests.execution.fixtures.mock_mt5 import (
            ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_BUY_STOP,
            ORDER_TYPE_SELL_LIMIT, ORDER_TYPE_SELL_STOP,
        )
        m = patch_router_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=123, price=2000.0)
        router = GriffOrderRouter(dry_run=False)
        sig = make_signal(direction=direction,
                          entry=2000.0,
                          sl=1995.0 if direction == Direction.BUY else 2005.0,
                          tp=2010.0 if direction == Direction.BUY else 1990.0)
        placer = router.place_pending_limit if is_limit else router.place_pending_stop
        order = run(placer(sig, 0.1, expiry_msc=1_700_003_600_000, now_msc=0))
        assert order.mt5_ticket == 123
        # Check the request type matches.
        req = m.sent_requests[-1]
        if is_limit and direction == Direction.BUY:
            assert req["type"] == ORDER_TYPE_BUY_LIMIT
        elif is_limit and direction == Direction.SELL:
            assert req["type"] == ORDER_TYPE_SELL_LIMIT
        elif not is_limit and direction == Direction.BUY:
            assert req["type"] == ORDER_TYPE_BUY_STOP
        else:
            assert req["type"] == ORDER_TYPE_SELL_STOP

    def test_pending_expiry_is_seconds_not_ms(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1, price=2000.0)
        router = GriffOrderRouter(dry_run=False)
        sig = make_signal()
        run(router.place_pending_stop(sig, 0.1,
                                        expiry_msc=2_000_000_000_000, now_msc=0))
        req = m.sent_requests[-1]
        # MT5 wants seconds.
        assert req["expiration"] == 2_000_000_000


# ===========================================================================
# 13. TICKET EXTRACTION
# ===========================================================================

class TestTicketFromResult:
    def test_ticket_from_order_field(self):
        r = OrderSendResult(retcode=TRADE_RETCODE_DONE, order=12345)
        assert _ticket_from_result(r) == 12345

    def test_ticket_from_deal_field(self):
        r = OrderSendResult(retcode=TRADE_RETCODE_DONE, deal=98765)
        assert _ticket_from_result(r) == 98765

    def test_prefers_order_over_deal_when_both_set(self):
        r = OrderSendResult(retcode=TRADE_RETCODE_DONE, order=1, deal=2)
        assert _ticket_from_result(r) == 1

    def test_ticket_zero_when_neither_set(self):
        r = OrderSendResult(retcode=TRADE_RETCODE_DONE, order=0, deal=0)
        assert _ticket_from_result(r) == 0


# ===========================================================================
# 14. CLOSE POSITION — ROUTER
# ===========================================================================

class TestClosePosition:
    @pytest.mark.parametrize("side,bid,ask,expected_price", [
        (Direction.BUY, 1999.5, 2000.0, 1999.5),  # close BUY @ bid
        (Direction.SELL, 1999.5, 2000.0, 2000.0), # close SELL @ ask
    ])
    def test_close_dry_run_uses_correct_quote(self, side, bid, ask,
                                                expected_price):
        router = GriffOrderRouter(dry_run=True)
        pos = make_griff_open(side=side)
        price = run(router.close_position(pos, bid=bid, ask=ask, now_msc=0))
        assert price == expected_price

    def test_close_live_retries_transient(self, patch_router_mt5):
        m = patch_router_mt5
        m.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_REQUOTE))
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1, price=2001.0)
        router = GriffOrderRouter(dry_run=False)
        price = run(router.close_position(make_griff_open(),
                                            bid=2000.5, ask=2001.0, now_msc=0))
        assert price == 2001.0


# ===========================================================================
# 15. FORCE-CLOSE FROM LIVE BROKER
# ===========================================================================

class TestForceClose:
    def test_force_close_records_eod(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        b = LiveBroker("XAUUSD")
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1, price=2000.0)
        intent = make_intent(side=Side.BUY, sl_price=1995.0, tp_price=2010.0)
        pos = run(b.fill_market_order(intent, make_tick(bid=1999.5, ask=2000.0)))
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1, price=2002.0)
        closed = run(b.force_close(pos, make_tick(bid=2002.0, ask=2002.5)))
        assert closed.close_reason == CloseReason.EOD

    @pytest.mark.parametrize("reason", [
        CloseReason.EOD, CloseReason.MANUAL, CloseReason.TIME_EXIT,
    ])
    def test_force_close_custom_reason(self, patch_live_broker_mt5, reason):
        m = patch_live_broker_mt5
        b = LiveBroker("XAUUSD")
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1, price=2000.0)
        intent = make_intent(side=Side.BUY, sl_price=1995.0, tp_price=2010.0)
        pos = run(b.fill_market_order(intent, make_tick(bid=1999.5, ask=2000.0)))
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1, price=2002.0)
        closed = run(b.force_close(pos, make_tick(bid=2002.0, ask=2002.5),
                                     reason))
        assert closed.close_reason == reason


# ===========================================================================
# 16. SYMBOL_INFO / TICK_INFO PATHOLOGIES
# ===========================================================================

class TestMt5SymbolInfo:
    def test_symbol_info_none_when_disabled(self):
        m = MockMT5()
        m.symbol_info_obj = None
        assert m.symbol_info("XAUUSD") is None

    def test_symbol_info_tick_none(self):
        m = MockMT5()
        m.tick_obj = None
        assert m.symbol_info_tick("XAUUSD") is None

    def test_account_info_returns_default(self):
        m = MockMT5()
        info = m.account_info()
        assert info.balance == 100_000.0
        assert info.equity == 100_000.0

    def test_terminal_info_default(self):
        m = MockMT5()
        info = m.terminal_info()
        assert info.connected is True


# ===========================================================================
# 17. PARAMETRIC RETCODE MATRIX — every router public method
# ===========================================================================

@pytest.mark.parametrize("retcode,is_transient", [
    (TRADE_RETCODE_REQUOTE, True),
    (TRADE_RETCODE_REJECT, True),
    (TRADE_RETCODE_PRICE_OFF, True),
    (TRADE_RETCODE_MARKET_CLOSED, True),
    (TRADE_RETCODE_CONNECTION, True),
    (TRADE_RETCODE_INVALID_STOPS, False),
    (TRADE_RETCODE_NO_MONEY, False),
])
def test_router_retcode_matrix(patch_router_mt5, retcode, is_transient):
    m = patch_router_mt5
    for _ in range(MAX_RETRIES):
        m.retcode_queue.append(OrderSendResult(retcode=retcode))
    router = GriffOrderRouter(dry_run=False)
    sig = make_signal()
    with pytest.raises(GriffOrderError):
        run(router.place_market(sig, 0.1, ask=2000.0, bid=1999.0, now_msc=0))
    # Transient → all 3 attempts; permanent → 1 attempt.
    expected_calls = MAX_RETRIES if is_transient else 1
    assert len(m.sent_requests) == expected_calls


# ===========================================================================
# 18. ZERO-TICKET RESPONSE
# ===========================================================================

class TestZeroTicketResponse:
    def test_router_handles_done_with_zero_ticket(self, patch_router_mt5):
        """MockMT5 auto-assigns ticket when DONE arrives with order=deal=0;
        this verifies the behaviour and that downstream code receives the
        auto-assigned ticket."""
        m = patch_router_mt5
        inject_zero_ticket_done(m)
        router = GriffOrderRouter(dry_run=False)
        pos = run(router.place_market(make_signal(), 0.1,
                                        ask=2000.0, bid=1999.0, now_msc=0))
        # MockMT5 auto-replaces 0 with next_ticket().
        assert pos.mt5_ticket > 0

    def test_router_handles_deal_only_response(self, patch_router_mt5):
        m = patch_router_mt5
        inject_deal_ticket_only(m, deal_ticket=55555)
        router = GriffOrderRouter(dry_run=False)
        pos = run(router.place_market(make_signal(), 0.1,
                                        ask=2000.0, bid=1999.0, now_msc=0))
        assert pos.mt5_ticket == 55555


# ===========================================================================
# 19. DRY-RUN COVERAGE
# ===========================================================================

class TestDryRun:
    def test_dry_market_returns_synthetic_ticket(self):
        router = GriffOrderRouter(dry_run=True)
        pos = run(router.place_market(make_signal(), 0.1,
                                        ask=2000.0, bid=1999.0, now_msc=0))
        assert pos.mt5_ticket == -1

    def test_dry_pending_stop_returns_synthetic_ticket(self):
        router = GriffOrderRouter(dry_run=True)
        order = run(router.place_pending_stop(make_signal(), 0.1,
                                                expiry_msc=1_000, now_msc=0))
        assert order.mt5_ticket == -1
        assert order.is_limit is False

    def test_dry_pending_limit_returns_synthetic_ticket(self):
        router = GriffOrderRouter(dry_run=True)
        order = run(router.place_pending_limit(make_signal(), 0.1,
                                                expiry_msc=1_000, now_msc=0))
        assert order.is_limit is True

    def test_dry_modify_sl_returns_true(self):
        router = GriffOrderRouter(dry_run=True)
        ok = run(router.modify_sl(make_griff_open(), new_sl=1996.0))
        assert ok is True


# ===========================================================================
# 20. NO-MONEY / MARGIN CALL
# ===========================================================================

class TestNoMoney:
    def test_no_money_fails_immediately(self, patch_router_mt5):
        m = patch_router_mt5
        inject_permanent_no_money(m)
        router = GriffOrderRouter(dry_run=False)
        with pytest.raises(GriffOrderError, match="permanent reject"):
            run(router.place_market(make_signal(), 0.1,
                                     ask=2000.0, bid=1999.0, now_msc=0))
        assert len(m.sent_requests) == 1


# ===========================================================================
# 21. INVALID-STOPS REJECT
# ===========================================================================

class TestInvalidStops:
    def test_invalid_stops_no_retry(self, patch_router_mt5):
        m = patch_router_mt5
        inject_invalid_stops(m)
        router = GriffOrderRouter(dry_run=False)
        with pytest.raises(GriffOrderError):
            run(router.place_market(make_signal(), 0.1,
                                     ask=2000.0, bid=1999.0, now_msc=0))
        assert len(m.sent_requests) == 1


# ===========================================================================
# 22. MARKET CLOSED — transient retry
# ===========================================================================

class TestMarketClosed:
    @pytest.mark.parametrize("n", [1, 2])
    def test_market_closed_recovers(self, patch_router_mt5, n):
        m = patch_router_mt5
        inject_market_closed(m, n=n)
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1, price=2000.0)
        router = GriffOrderRouter(dry_run=False)
        pos = run(router.place_market(make_signal(), 0.1,
                                        ask=2000.0, bid=1999.0, now_msc=0))
        assert pos.mt5_ticket == 1

    def test_market_closed_exhausts(self, patch_router_mt5):
        m = patch_router_mt5
        inject_market_closed(m, n=MAX_RETRIES)
        router = GriffOrderRouter(dry_run=False)
        with pytest.raises(GriffOrderError):
            run(router.place_market(make_signal(), 0.1,
                                     ask=2000.0, bid=1999.0, now_msc=0))


# ===========================================================================
# 23. PRICE OFF — transient
# ===========================================================================

class TestPriceOff:
    def test_price_off_retries(self, patch_router_mt5):
        m = patch_router_mt5
        inject_price_off(m)
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1, price=2000.0)
        router = GriffOrderRouter(dry_run=False)
        pos = run(router.place_market(make_signal(), 0.1,
                                        ask=2000.0, bid=1999.0, now_msc=0))
        assert pos.mt5_ticket == 1


# ===========================================================================
# 24. REQUEST PAYLOAD ASSERTIONS
# ===========================================================================

class TestRequestPayload:
    def test_market_request_contains_required_fields(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1, price=2000.0)
        router = GriffOrderRouter(dry_run=False)
        sig = make_signal(entry=2000.0, sl=1995.0, tp=2010.0)
        run(router.place_market(sig, 0.5,
                                  ask=2000.0, bid=1999.0, now_msc=0))
        req = m.sent_requests[0]
        for key in ("action", "symbol", "volume", "type", "price",
                    "sl", "tp", "deviation", "magic", "comment"):
            assert key in req

    def test_market_request_volume_matches(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1, price=2000.0)
        router = GriffOrderRouter(dry_run=False)
        run(router.place_market(make_signal(), 0.37,
                                  ask=2000.0, bid=1999.0, now_msc=0))
        assert m.sent_requests[0]["volume"] == 0.37


# ===========================================================================
# 25. PAPER BROKER EDGE
# ===========================================================================

from execution.broker_simulator import PaperBroker


class TestPaperBrokerEdge:
    def test_paper_buy_includes_slip(self):
        b = PaperBroker(slippage_pct=0.5, contract_size=100)
        intent = make_intent(side=Side.BUY, intended_price=2000.0,
                             sl_price=1995.0, tp_price=2010.0)
        tick = make_tick(bid=1999.5, ask=2000.0)
        pos = b.fill_market_order(intent, tick)
        # ask + 0.5 * spread (= 0.5)
        assert pos.entry_price == pytest.approx(2000.0 + 0.25, abs=1e-9)

    def test_paper_sell_includes_slip(self):
        b = PaperBroker(slippage_pct=0.5, contract_size=100)
        intent = make_intent(side=Side.SELL, intended_price=2000.0,
                             sl_price=2005.0, tp_price=1990.0)
        tick = make_tick(bid=1999.5, ask=2000.0)
        pos = b.fill_market_order(intent, tick)
        # bid - 0.5 * spread
        assert pos.entry_price == pytest.approx(1999.5 - 0.25, abs=1e-9)

    @pytest.mark.parametrize("slip_pct", [0.0, 0.25, 0.5, 1.0])
    def test_paper_slippage_pct(self, slip_pct):
        b = PaperBroker(slippage_pct=slip_pct, contract_size=100)
        intent = make_intent(side=Side.BUY)
        tick = make_tick(bid=1999.0, ask=2000.0)
        pos = b.fill_market_order(intent, tick)
        assert pos.entry_price == pytest.approx(2000.0 + slip_pct, abs=1e-9)

    def test_paper_time_exit_takes_priority(self):
        b = PaperBroker()
        intent = make_intent(max_hold_until_msc=100)
        pos = b.fill_market_order(intent, make_tick(time_msc=50))
        # Force exit by passing a tick AT/PAST max_hold_until_msc
        exit_pos = b.check_position_exit(pos, make_tick(time_msc=101))
        assert exit_pos is not None
        assert exit_pos.close_reason == CloseReason.TIME_EXIT

    def test_paper_buy_sl_hit(self):
        b = PaperBroker()
        intent = make_intent(side=Side.BUY,
                             intended_price=2000.0,
                             sl_price=1990.0, tp_price=2010.0,
                             max_hold_until_msc=10_000)
        pos = b.fill_market_order(intent, make_tick(bid=1999.5, ask=2000.0,
                                                      time_msc=0))
        sl_tick = make_tick(bid=1989.0, ask=1989.5, time_msc=100)
        out = b.check_position_exit(pos, sl_tick)
        assert out.close_reason == CloseReason.SL_HIT

    def test_paper_buy_tp_hit(self):
        b = PaperBroker()
        intent = make_intent(side=Side.BUY,
                             intended_price=2000.0,
                             sl_price=1990.0, tp_price=2010.0,
                             max_hold_until_msc=10_000)
        pos = b.fill_market_order(intent, make_tick(bid=1999.5, ask=2000.0,
                                                      time_msc=0))
        tp_tick = make_tick(bid=2010.5, ask=2011.0, time_msc=100)
        out = b.check_position_exit(pos, tp_tick)
        assert out.close_reason == CloseReason.TP_HIT

    def test_paper_pnl_sign_long(self):
        b = PaperBroker()
        intent = make_intent(side=Side.BUY, intended_price=2000.0,
                             sl_price=1990.0, tp_price=2010.0,
                             max_hold_until_msc=10_000)
        pos = b.fill_market_order(intent, make_tick(bid=1999.5, ask=2000.0,
                                                      time_msc=0))
        tp_tick = make_tick(bid=2010.5, ask=2011.0, time_msc=100)
        out = b.check_position_exit(pos, tp_tick)
        assert out.pnl_usd > 0

    def test_paper_pnl_sign_short(self):
        b = PaperBroker()
        intent = make_intent(side=Side.SELL, intended_price=2000.0,
                             sl_price=2010.0, tp_price=1990.0,
                             max_hold_until_msc=10_000)
        pos = b.fill_market_order(intent, make_tick(bid=1999.5, ask=2000.0,
                                                      time_msc=0))
        tp_tick = make_tick(bid=1988.0, ask=1988.5, time_msc=100)
        out = b.check_position_exit(pos, tp_tick)
        assert out.pnl_usd > 0
