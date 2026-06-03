"""LiveBroker — MT5 order routing with retries and reconciliation.

MT5 is monkey-patched module-side (execution.live_broker.mt5). All tests run
through MockMT5 with controlled retcodes and deal history.
"""

from __future__ import annotations
import asyncio

import pytest

from execution.live_broker import (
    DEVIATION_POINTS, MAGIC, MAX_RETRIES, POINT_VALUE,
    LiveBroker, LiveBrokerError, _TRANSIENT_RETCODES,
)
from execution.order import OrderIntent, Side, SignalType
from execution.position import (
    CloseReason, Position, PositionState,
)
from utils.session import SessionLabel

from tests.execution.fixtures.mock_mt5 import (
    MockMT5, OrderSendResult, DealInfo, PositionInfo,
    TRADE_RETCODE_DONE, TRADE_RETCODE_REQUOTE, TRADE_RETCODE_REJECT,
    TRADE_RETCODE_PRICE_OFF, TRADE_RETCODE_MARKET_CLOSED,
    TRADE_RETCODE_CONNECTION, TRADE_RETCODE_INVALID_STOPS,
    TRADE_RETCODE_NO_MONEY,
)
from tests.execution.fixtures.mock_orders import make_intent
from tests.execution.fixtures.mock_positions import make_position, make_tick


def run(coro):
    return asyncio.run(coro)


# ===========================================================================
# 1. Construction
# ===========================================================================

class TestConstructor:
    def test_default_symbol(self):
        b = LiveBroker(symbol="XAUUSD")
        assert b._symbol == "XAUUSD"
        assert b._deviation == DEVIATION_POINTS
        assert b._contract_size == 100

    @pytest.mark.parametrize("sym", ["XAUUSD", "EURUSD", "USDJPY",
                                      "GBPUSD"])
    def test_custom_symbol(self, sym):
        assert LiveBroker(symbol=sym)._symbol == sym

    def test_custom_slippage(self):
        b = LiveBroker("XAUUSD", slippage_pct=0.25)
        assert b._slippage_pct == 0.25

    def test_custom_deviation(self):
        b = LiveBroker("XAUUSD", deviation_points=42)
        assert b._deviation == 42

    def test_constants(self):
        assert MAX_RETRIES == 3
        assert MAGIC == 786543
        assert POINT_VALUE == 0.01


# ===========================================================================
# 2. Happy-path fill_market_order
# ===========================================================================

class TestFillMarketOrder:
    def test_buy_success(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=42, price=1.10010)

        b = LiveBroker("XAUUSD")
        i = make_intent(side=Side.BUY, lots=0.5)
        t = make_tick(bid=1.10000, ask=1.10010)

        pos = run(b.fill_market_order(i, t))
        assert pos.state == PositionState.OPEN
        assert pos.entry_price == 1.10010
        assert pos.lots == 0.5
        assert b._ticket_by_id[pos.position_id] == 42

    def test_sell_success(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=7, price=1.10000)

        b = LiveBroker("XAUUSD")
        i = make_intent(side=Side.SELL, sl_price=1.10100, tp_price=1.09900,
                        intended_price=1.10000)
        t = make_tick(bid=1.10000, ask=1.10010)

        pos = run(b.fill_market_order(i, t))
        assert pos.side == Side.SELL
        assert pos.entry_price == 1.10000

    def test_request_payload(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        b = LiveBroker("XAUUSD", deviation_points=20)
        i = make_intent(side=Side.BUY, lots=0.10,
                        sl_price=1.09000, tp_price=1.12000)
        run(b.fill_market_order(i, make_tick(bid=1.0, ask=1.1)))

        assert len(m.sent_requests) == 1
        req = m.sent_requests[0]
        assert req["action"] == m.TRADE_ACTION_DEAL
        assert req["symbol"] == "XAUUSD"
        assert req["volume"] == 0.10
        assert req["type"] == m.ORDER_TYPE_BUY
        assert req["price"] == 1.1
        assert req["sl"] == 1.09
        assert req["tp"] == 1.12
        assert req["deviation"] == 20
        assert req["magic"] == MAGIC

    def test_sell_request_uses_bid_and_sell_type(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        b = LiveBroker("EURUSD")
        i = make_intent(side=Side.SELL,
                        sl_price=1.10100, tp_price=1.09900,
                        intended_price=1.10000)
        run(b.fill_market_order(i, make_tick(bid=1.10000, ask=1.10010)))
        req = m.sent_requests[0]
        assert req["price"] == 1.10000
        assert req["type"] == m.ORDER_TYPE_SELL

    def test_position_fields_propagate(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        b = LiveBroker("XAUUSD")
        i = make_intent(max_hold_until_msc=999,
                        signal_type=SignalType.MOMENTUM,
                        session=SessionLabel.NY)
        pos = run(b.fill_market_order(i, make_tick(time_msc=42)))
        assert pos.entry_time_msc == 42
        assert pos.max_hold_until_msc == 999
        assert pos.signal_type == "MOMENTUM"
        assert pos.session == "NY"


# ===========================================================================
# 3. Retry logic — transient retcodes
# ===========================================================================

class TestRetry:
    @pytest.mark.parametrize("retcode", sorted(_TRANSIENT_RETCODES))
    def test_transient_retried_then_succeeds(self, patch_live_broker_mt5,
                                              retcode, monkeypatch):
        # Patch sleep so retries don't slow tests down.
        import execution.live_broker as lb
        async def _no_sleep(*_a, **_k):
            return None
        monkeypatch.setattr(lb.asyncio, "sleep", _no_sleep)
        m = patch_live_broker_mt5
        m.queue_retcodes(retcode, retcode, TRADE_RETCODE_DONE)

        b = LiveBroker("XAUUSD")
        pos = run(b.fill_market_order(make_intent(),
                                       make_tick()))
        assert pos.state == PositionState.OPEN
        # 3 attempts total
        assert len(m.sent_requests) == 3

    def test_exhausts_retries_then_raises(self, patch_live_broker_mt5,
                                          monkeypatch):
        import execution.live_broker as lb
        async def _no_sleep(*_a, **_k):
            return None
        monkeypatch.setattr(lb.asyncio, "sleep", _no_sleep)
        m = patch_live_broker_mt5
        m.queue_retcodes(*([TRADE_RETCODE_REQUOTE] * MAX_RETRIES))

        b = LiveBroker("XAUUSD")
        with pytest.raises(LiveBrokerError, match="failed after"):
            run(b.fill_market_order(make_intent(), make_tick()))
        assert len(m.sent_requests) == MAX_RETRIES

    @pytest.mark.parametrize("retcode", [
        TRADE_RETCODE_INVALID_STOPS,
        TRADE_RETCODE_NO_MONEY,
        99999,  # unknown permanent rejection
    ])
    def test_non_transient_raises_immediately(self, patch_live_broker_mt5,
                                              retcode):
        m = patch_live_broker_mt5
        m.queue_retcodes(retcode)
        b = LiveBroker("XAUUSD")
        with pytest.raises(LiveBrokerError, match="rejected"):
            run(b.fill_market_order(make_intent(), make_tick()))
        assert len(m.sent_requests) == 1

    def test_order_send_returns_none(self, patch_live_broker_mt5, monkeypatch):
        """MT5 sometimes returns None on transport errors."""
        import execution.live_broker as lb
        async def _no_sleep(*_a, **_k):
            return None
        monkeypatch.setattr(lb.asyncio, "sleep", _no_sleep)
        m = patch_live_broker_mt5
        # Override order_send to always return None
        m.order_send = lambda req: None  # type: ignore[method-assign]
        b = LiveBroker("XAUUSD")
        with pytest.raises(LiveBrokerError, match="failed after"):
            run(b.fill_market_order(make_intent(), make_tick()))


# ===========================================================================
# 4. check_position_exit — time exit
# ===========================================================================

class TestCheckExit:
    def test_returns_none_for_closed_position(self, patch_live_broker_mt5):
        b = LiveBroker("XAUUSD")
        p = make_position(state=PositionState.CLOSED)
        assert run(b.check_position_exit(p, make_tick())) is None

    def test_returns_none_when_unknown_ticket(self, patch_live_broker_mt5):
        b = LiveBroker("XAUUSD")
        p = make_position(max_hold_until_msc=10**13)
        # Never opened via this broker → no ticket cached
        assert run(b.check_position_exit(p, make_tick())) is None

    def test_time_exit_triggers_close(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.queue_retcodes(TRADE_RETCODE_DONE)  # for the open
        m.queue_retcodes(TRADE_RETCODE_DONE)  # for the close

        b = LiveBroker("XAUUSD")
        i = make_intent(max_hold_until_msc=500)
        pos = run(b.fill_market_order(i, make_tick(time_msc=0)))
        # Advance past max_hold
        out = run(b.check_position_exit(pos, make_tick(time_msc=1000)))
        assert out is not None
        assert out.close_reason == CloseReason.TIME_EXIT

    def test_position_still_open_returns_none(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.queue_retcodes(TRADE_RETCODE_DONE)
        b = LiveBroker("XAUUSD")
        i = make_intent(max_hold_until_msc=10**13)
        pos = run(b.fill_market_order(i, make_tick(time_msc=0)))
        # Inject a PositionInfo so positions_get returns non-empty
        m.positions.append(PositionInfo(ticket=b._ticket_by_id[pos.position_id]))
        out = run(b.check_position_exit(pos, make_tick(time_msc=1000)))
        assert out is None


# ===========================================================================
# 5. reconcile_broker_close
# ===========================================================================

class TestReconciliation:
    def test_position_vanished_with_deal_history_sl(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.queue_retcodes(TRADE_RETCODE_DONE)
        b = LiveBroker("XAUUSD")
        i = make_intent(side=Side.BUY, sl_price=0.99, tp_price=1.20,
                        max_hold_until_msc=10**13)
        pos = run(b.fill_market_order(i, make_tick(bid=1.0, ask=1.0,
                                                    time_msc=0)))
        # No positions left → broker closed; last deal at 0.98 → SL_HIT
        m.deals.append(DealInfo(price=0.98))
        out = run(b.check_position_exit(pos, make_tick(time_msc=1000,
                                                        bid=0.98, ask=0.98)))
        assert out is not None
        assert out.close_reason == CloseReason.SL_HIT
        assert out.exit_price == 0.98

    def test_position_vanished_with_deal_history_tp(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.queue_retcodes(TRADE_RETCODE_DONE)
        b = LiveBroker("XAUUSD")
        i = make_intent(side=Side.BUY, sl_price=0.99, tp_price=1.20,
                        max_hold_until_msc=10**13)
        pos = run(b.fill_market_order(i, make_tick(bid=1.0, ask=1.0)))
        m.deals.append(DealInfo(price=1.20))
        out = run(b.check_position_exit(pos, make_tick(time_msc=1000,
                                                        bid=1.20, ask=1.20)))
        assert out.close_reason == CloseReason.TP_HIT

    def test_position_vanished_no_history_returns_manual(self,
                                                         patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.queue_retcodes(TRADE_RETCODE_DONE)
        b = LiveBroker("XAUUSD")
        i = make_intent(side=Side.BUY, max_hold_until_msc=10**13)
        pos = run(b.fill_market_order(i, make_tick(bid=1.0, ask=1.0)))
        # deals is empty
        out = run(b.check_position_exit(pos, make_tick(time_msc=1000,
                                                        bid=1.05, ask=1.05)))
        assert out.close_reason == CloseReason.MANUAL

    def test_sell_position_vanished_sl(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.queue_retcodes(TRADE_RETCODE_DONE)
        b = LiveBroker("XAUUSD")
        i = make_intent(side=Side.SELL, sl_price=1.10, tp_price=0.90,
                        intended_price=1.0, max_hold_until_msc=10**13)
        pos = run(b.fill_market_order(i, make_tick(bid=1.0, ask=1.0)))
        m.deals.append(DealInfo(price=1.10))
        out = run(b.check_position_exit(pos, make_tick(time_msc=1000,
                                                        bid=1.10, ask=1.10)))
        assert out.close_reason == CloseReason.SL_HIT

    def test_sell_position_vanished_tp(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.queue_retcodes(TRADE_RETCODE_DONE)
        b = LiveBroker("XAUUSD")
        i = make_intent(side=Side.SELL, sl_price=1.10, tp_price=0.90,
                        intended_price=1.0, max_hold_until_msc=10**13)
        pos = run(b.fill_market_order(i, make_tick(bid=1.0, ask=1.0)))
        m.deals.append(DealInfo(price=0.90))
        out = run(b.check_position_exit(pos, make_tick(time_msc=1000,
                                                        bid=0.90, ask=0.90)))
        assert out.close_reason == CloseReason.TP_HIT


# ===========================================================================
# 6. force_close
# ===========================================================================

class TestForceClose:
    def test_basic(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.queue_retcodes(TRADE_RETCODE_DONE, TRADE_RETCODE_DONE)
        b = LiveBroker("XAUUSD")
        i = make_intent(max_hold_until_msc=10**13)
        pos = run(b.fill_market_order(i, make_tick(bid=1.0, ask=1.0)))
        out = run(b.force_close(pos, make_tick(bid=1.1, ask=1.1)))
        assert out.state == PositionState.CLOSED
        assert out.close_reason == CloseReason.EOD

    @pytest.mark.parametrize("reason", list(CloseReason))
    def test_any_reason(self, patch_live_broker_mt5, reason):
        m = patch_live_broker_mt5
        m.queue_retcodes(TRADE_RETCODE_DONE, TRADE_RETCODE_DONE)
        b = LiveBroker("XAUUSD")
        i = make_intent(max_hold_until_msc=10**13)
        pos = run(b.fill_market_order(i, make_tick()))
        out = run(b.force_close(pos, make_tick(), reason))
        assert out.close_reason == reason

    def test_unknown_position_raises(self, patch_live_broker_mt5):
        b = LiveBroker("XAUUSD")
        p = make_position(max_hold_until_msc=10**13)
        with pytest.raises(LiveBrokerError, match="unknown position_id"):
            run(b.force_close(p, make_tick()))

    def test_close_request_payload(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.queue_retcodes(TRADE_RETCODE_DONE, TRADE_RETCODE_DONE)
        b = LiveBroker("XAUUSD")
        i = make_intent(side=Side.BUY, max_hold_until_msc=10**13)
        pos = run(b.fill_market_order(i, make_tick(bid=1.0, ask=1.0)))
        run(b.force_close(pos, make_tick(bid=1.10, ask=1.11)))
        # Last request was the close
        req = m.sent_requests[-1]
        assert req["action"] == m.TRADE_ACTION_DEAL
        assert req["type"] == m.ORDER_TYPE_SELL  # closing a BUY → sell
        assert req["price"] == 1.10  # bid for BUY-close

    def test_ticket_removed_after_close(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.queue_retcodes(TRADE_RETCODE_DONE, TRADE_RETCODE_DONE)
        b = LiveBroker("XAUUSD")
        i = make_intent(max_hold_until_msc=10**13)
        pos = run(b.fill_market_order(i, make_tick()))
        assert pos.position_id in b._ticket_by_id
        run(b.force_close(pos, make_tick()))
        assert pos.position_id not in b._ticket_by_id


# ===========================================================================
# 7. PnL math (proxied through _build_closed)
# ===========================================================================

class TestPnL:
    @pytest.mark.parametrize("side,entry,exit_,lots,expected_pts", [
        (Side.BUY,  1.0, 1.10, 1.0,  10.0),
        (Side.BUY,  1.0, 0.90, 1.0, -10.0),
        (Side.SELL, 1.0, 0.90, 1.0,  10.0),
        (Side.SELL, 1.0, 1.10, 1.0, -10.0),
    ])
    def test_pnl_signs(self, patch_live_broker_mt5,
                       side, entry, exit_, lots, expected_pts):
        m = patch_live_broker_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1, price=entry)
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=2, price=exit_)
        b = LiveBroker("XAUUSD")
        if side == Side.BUY:
            i = make_intent(side=side, lots=lots,
                            sl_price=entry - 0.5,
                            tp_price=entry + 0.5, intended_price=entry,
                            max_hold_until_msc=10**13)
        else:
            i = make_intent(side=side, lots=lots,
                            sl_price=entry + 0.5,
                            tp_price=entry - 0.5, intended_price=entry,
                            max_hold_until_msc=10**13)
        pos = run(b.fill_market_order(i, make_tick(bid=entry, ask=entry)))
        out = run(b.force_close(pos, make_tick(bid=exit_, ask=exit_)))
        # Don't compare floats exactly — pnl uses POINT_VALUE arithmetic
        assert out.pnl_pts == pytest.approx(expected_pts * 0.01 / POINT_VALUE,
                                             abs=1e-6)
        # USD = price_diff * lots * 100
        expected_usd = (exit_ - entry if side == Side.BUY
                        else entry - exit_) * lots * 100
        assert out.pnl_usd == pytest.approx(expected_usd, abs=1e-6)


# ===========================================================================
# 8. POINT_VALUE / instance attribute
# ===========================================================================

class TestPointValue:
    def test_instance_attr_present(self):
        b = LiveBroker("XAUUSD")
        assert b.POINT_VALUE == 0.01


# ===========================================================================
# 9. Edge cases on construction
# ===========================================================================

class TestConstructorEdge:
    @pytest.mark.parametrize("slip", [0.0, 0.25, 0.5, 1.0])
    def test_any_slippage(self, slip):
        assert LiveBroker("X", slippage_pct=slip)._slippage_pct == slip

    @pytest.mark.parametrize("dev", [1, 5, 20, 100])
    def test_any_deviation(self, dev):
        assert LiveBroker("X", deviation_points=dev)._deviation == dev

    @pytest.mark.parametrize("cs", [1, 100, 100_000])
    def test_any_contract_size(self, cs):
        assert LiveBroker("X", contract_size=cs)._contract_size == cs


# ===========================================================================
# 10. Concurrent fills — ticket map grows
# ===========================================================================

class TestTicketMap:
    def test_two_fills_two_tickets(self, patch_live_broker_mt5):
        m = patch_live_broker_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=1)
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=2)
        b = LiveBroker("XAUUSD")
        i1 = make_intent(max_hold_until_msc=10**13)
        i2 = make_intent(max_hold_until_msc=10**13)
        p1 = run(b.fill_market_order(i1, make_tick()))
        p2 = run(b.fill_market_order(i2, make_tick()))
        assert b._ticket_by_id[p1.position_id] == 1
        assert b._ticket_by_id[p2.position_id] == 2
        assert p1.position_id != p2.position_id


# ===========================================================================
# 11. Latency: order_send taking long shouldn't break the broker
# ===========================================================================

class TestLatency:
    def test_slow_send_completes_eventually(self, patch_live_broker_mt5):
        # We don't simulate sleep here; just confirm one request succeeds.
        m = patch_live_broker_mt5
        m.queue_retcodes(TRADE_RETCODE_DONE)
        b = LiveBroker("XAUUSD")
        pos = run(b.fill_market_order(make_intent(), make_tick()))
        assert pos.state == PositionState.OPEN
