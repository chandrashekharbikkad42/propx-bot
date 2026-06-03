"""PaperBroker — fill, exit, force_close, slippage, PnL.

The PaperBroker is stateless. Each test constructs a fresh broker and asserts
on the returned Position (frozen value object). Slippage model:
  - entry BUY  → fill = ask + spread * slippage_pct
  - entry SELL → fill = bid - spread * slippage_pct
  - exit BUY   → fill = bid - spread * slippage_pct  (we sell to close)
  - exit SELL  → fill = ask + spread * slippage_pct  (we buy to close)
PnL conventions:
  - pnl_price (BUY)  = exit - entry
  - pnl_price (SELL) = entry - exit
  - pnl_pts          = pnl_price / POINT_VALUE  (POINT_VALUE = 0.01)
  - pnl_usd          = pnl_price * lots * contract_size  (default 100)
"""

from __future__ import annotations
import math
import pytest
from hypothesis import given, settings, strategies as st

from execution.broker_simulator import PaperBroker
from execution.order import OrderIntent, Side, SignalType
from execution.position import (
    CloseReason, Position, PositionState,
)
from utils.session import SessionLabel

from tests.execution.fixtures.mock_orders import make_intent
from tests.execution.fixtures.mock_positions import make_position, make_tick


# ===========================================================================
# 1. Construction / defaults
# ===========================================================================

class TestConstructor:
    def test_default_slippage(self):
        b = PaperBroker()
        assert b._slippage_pct == 0.5
        assert b._contract_size == 100
        assert PaperBroker.POINT_VALUE == 0.01

    @pytest.mark.parametrize("slip", [0.0, 0.25, 0.5, 1.0, 2.0])
    def test_custom_slippage(self, slip):
        b = PaperBroker(slippage_pct=slip)
        assert b._slippage_pct == slip

    @pytest.mark.parametrize("cs", [1, 100, 100_000])
    def test_custom_contract_size(self, cs):
        b = PaperBroker(contract_size=cs)
        assert b._contract_size == cs


# ===========================================================================
# 2. fill_market_order — entry price + slippage
# ===========================================================================

class TestFillEntry:
    def test_buy_fill_above_ask_by_half_spread(self):
        b = PaperBroker()
        # spread = 10 pts (0.10); slip = 5 pts (0.05)
        t = make_tick(bid=1.10000, ask=1.10010)
        i = make_intent(side=Side.BUY)
        pos = b.fill_market_order(i, t)
        assert pos.entry_price == pytest.approx(1.10010 + 0.5 * (1.10010 - 1.10000))

    def test_sell_fill_below_bid_by_half_spread(self):
        b = PaperBroker()
        t = make_tick(bid=1.10000, ask=1.10010)
        i = make_intent(side=Side.SELL, sl_price=1.10100, tp_price=1.09900,
                        intended_price=1.10000)
        pos = b.fill_market_order(i, t)
        assert pos.entry_price == pytest.approx(1.10000 - 0.5 * (1.10010 - 1.10000))

    def test_zero_slippage(self):
        b = PaperBroker(slippage_pct=0.0)
        t = make_tick(bid=1.10000, ask=1.10010)
        i = make_intent(side=Side.BUY)
        pos = b.fill_market_order(i, t)
        assert pos.entry_price == 1.10010

    def test_full_spread_slippage(self):
        b = PaperBroker(slippage_pct=1.0)
        t = make_tick(bid=1.10000, ask=1.10010)
        i = make_intent(side=Side.BUY)
        pos = b.fill_market_order(i, t)
        assert pos.entry_price == pytest.approx(1.10020)

    def test_zero_spread(self):
        b = PaperBroker()
        t = make_tick(bid=1.10000, ask=1.10000)
        i = make_intent(side=Side.BUY)
        pos = b.fill_market_order(i, t)
        assert pos.entry_price == 1.10000

    @pytest.mark.parametrize("ask,bid,slip,side,expected", [
        (1.10010, 1.10000, 0.5, Side.BUY,  1.10015),
        (1.10010, 1.10000, 0.5, Side.SELL, 0.5 * (1.10000 - 0.5 * 0.0001) + 0.5 * 1.09995),
        (2.00000, 1.99980, 0.5, Side.BUY,  2.00010),
        (2.00000, 1.99980, 0.5, Side.SELL, 1.99970),
        (2.00000, 1.99980, 1.0, Side.BUY,  2.00020),
        (2.00000, 1.99980, 0.0, Side.SELL, 1.99980),
    ])
    def test_fill_price_matrix(self, ask, bid, slip, side, expected):
        b = PaperBroker(slippage_pct=slip)
        t = make_tick(bid=bid, ask=ask)
        if side == Side.BUY:
            i = make_intent(side=side, intended_price=ask,
                            sl_price=bid - 0.001, tp_price=ask + 0.001)
        else:
            i = make_intent(side=side, intended_price=bid,
                            sl_price=ask + 0.001, tp_price=bid - 0.001)
        pos = b.fill_market_order(i, t)
        assert pos.entry_price == pytest.approx(expected, abs=1e-9)

    def test_position_state_open(self):
        b = PaperBroker()
        i = make_intent()
        pos = b.fill_market_order(i, make_tick())
        assert pos.state == PositionState.OPEN

    def test_position_id_unique_across_calls(self):
        b = PaperBroker()
        i = make_intent()
        a = b.fill_market_order(i, make_tick())
        c = b.fill_market_order(i, make_tick())
        assert a.position_id != c.position_id

    def test_lots_pass_through(self):
        b = PaperBroker()
        i = make_intent(lots=0.42)
        pos = b.fill_market_order(i, make_tick())
        assert pos.lots == 0.42

    def test_max_hold_pass_through(self):
        b = PaperBroker()
        i = make_intent(max_hold_until_msc=42)
        pos = b.fill_market_order(i, make_tick())
        assert pos.max_hold_until_msc == 42

    def test_session_pass_through(self):
        b = PaperBroker()
        i = make_intent(session=SessionLabel.NY)
        pos = b.fill_market_order(i, make_tick())
        assert pos.session == "NY"

    def test_signal_type_pass_through(self):
        b = PaperBroker()
        i = make_intent(signal_type=SignalType.MOMENTUM)
        pos = b.fill_market_order(i, make_tick())
        assert pos.signal_type == "MOMENTUM"


# ===========================================================================
# 3. fill_market_order — sl_pts / tp_pts anchor (Phase 7B)
# ===========================================================================

class TestSlTpAnchor:
    def test_zero_pts_falls_back_to_intent_prices(self):
        b = PaperBroker()
        i = make_intent(sl_price=1.09000, tp_price=1.12000)
        pos = b.fill_market_order(i, make_tick(bid=1.10000, ask=1.10010))
        assert pos.sl_price == 1.09000
        assert pos.tp_price == 1.12000

    def test_buy_anchor_to_fill(self):
        b = PaperBroker()
        # sl_pts=100 → 100 * 0.01 = 1.0 below fill
        # tp_pts=200 → 200 * 0.01 = 2.0 above fill
        i = make_intent(side=Side.BUY, sl_pts=100, tp_pts=200)
        t = make_tick(bid=1.0, ask=1.1)
        pos = b.fill_market_order(i, t)
        expected_fill = 1.1 + 0.5 * 0.1
        assert pos.entry_price == pytest.approx(expected_fill)
        assert pos.sl_price == pytest.approx(expected_fill - 100 * 0.01)
        assert pos.tp_price == pytest.approx(expected_fill + 200 * 0.01)

    def test_sell_anchor_to_fill(self):
        b = PaperBroker()
        i = make_intent(
            side=Side.SELL, sl_pts=100, tp_pts=200,
            sl_price=1.10100, tp_price=1.09900, intended_price=1.10000,
        )
        t = make_tick(bid=1.0, ask=1.1)
        pos = b.fill_market_order(i, t)
        expected_fill = 1.0 - 0.5 * 0.1
        assert pos.entry_price == pytest.approx(expected_fill)
        assert pos.sl_price == pytest.approx(expected_fill + 100 * 0.01)
        assert pos.tp_price == pytest.approx(expected_fill - 200 * 0.01)

    def test_one_pts_zero_uses_intent(self):
        b = PaperBroker()
        # Only sl_pts >0; tp_pts=0 → falls back.
        i = make_intent(sl_pts=100, tp_pts=0,
                        sl_price=1.09000, tp_price=1.12000)
        pos = b.fill_market_order(i, make_tick())
        assert pos.sl_price == 1.09000
        assert pos.tp_price == 1.12000

    @pytest.mark.parametrize("sl_pts,tp_pts", [
        (100, 200), (50, 100), (1, 1), (1000, 2500),
    ])
    def test_buy_pts_matrix(self, sl_pts, tp_pts):
        b = PaperBroker()
        i = make_intent(side=Side.BUY, sl_pts=sl_pts, tp_pts=tp_pts)
        t = make_tick(bid=1.0, ask=1.1)
        pos = b.fill_market_order(i, t)
        fill = 1.1 + 0.5 * 0.1
        assert pos.sl_price == pytest.approx(fill - sl_pts * 0.01)
        assert pos.tp_price == pytest.approx(fill + tp_pts * 0.01)


# ===========================================================================
# 4. check_position_exit — time exit
# ===========================================================================

class TestTimeExit:
    def test_time_exit_at_max_hold(self):
        b = PaperBroker()
        p = make_position(max_hold_until_msc=1000)
        t = make_tick(time_msc=1000, bid=1.0, ask=1.1)
        result = b.check_position_exit(p, t)
        assert result is not None
        assert result.close_reason == CloseReason.TIME_EXIT
        assert result.state == PositionState.CLOSED

    def test_time_exit_after_max_hold(self):
        b = PaperBroker()
        p = make_position(max_hold_until_msc=1000)
        t = make_tick(time_msc=2000, bid=1.0, ask=1.1)
        result = b.check_position_exit(p, t)
        assert result is not None
        assert result.close_reason == CloseReason.TIME_EXIT

    def test_no_time_exit_before_max_hold(self):
        b = PaperBroker()
        p = make_position(max_hold_until_msc=1000,
                          sl_price=0.5, tp_price=2.0)
        t = make_tick(time_msc=999, bid=1.0, ask=1.05)
        result = b.check_position_exit(p, t)
        assert result is None

    def test_time_exit_precedes_sl(self):
        b = PaperBroker()
        # Both time-exit AND SL hit possible; time should win.
        p = make_position(max_hold_until_msc=1000,
                          sl_price=2.0, tp_price=0.5,
                          side=Side.SELL,
                          entry_price=1.0)
        t = make_tick(time_msc=1000, bid=1.99, ask=2.00)
        result = b.check_position_exit(p, t)
        assert result.close_reason == CloseReason.TIME_EXIT


# ===========================================================================
# 5. check_position_exit — BUY SL/TP
# ===========================================================================

class TestBuyExit:
    def test_buy_sl_hit_when_bid_crosses_below(self):
        b = PaperBroker()
        p = make_position(side=Side.BUY, entry_price=1.10000,
                          sl_price=1.09800, tp_price=1.10400,
                          max_hold_until_msc=10**13)
        t = make_tick(bid=1.09800, ask=1.09810)
        out = b.check_position_exit(p, t)
        assert out is not None
        assert out.close_reason == CloseReason.SL_HIT

    def test_buy_sl_not_hit_when_bid_above(self):
        b = PaperBroker()
        p = make_position(side=Side.BUY, sl_price=1.09800,
                          tp_price=1.10400,
                          max_hold_until_msc=10**13)
        t = make_tick(bid=1.09801, ask=1.09811)
        assert b.check_position_exit(p, t) is None

    def test_buy_tp_hit_when_bid_reaches(self):
        b = PaperBroker()
        p = make_position(side=Side.BUY, sl_price=1.09800,
                          tp_price=1.10400,
                          max_hold_until_msc=10**13)
        t = make_tick(bid=1.10400, ask=1.10410)
        out = b.check_position_exit(p, t)
        assert out.close_reason == CloseReason.TP_HIT

    def test_buy_tp_not_hit_below_threshold(self):
        b = PaperBroker()
        p = make_position(side=Side.BUY, sl_price=1.09800,
                          tp_price=1.10400,
                          max_hold_until_msc=10**13)
        t = make_tick(bid=1.10399, ask=1.10409)
        assert b.check_position_exit(p, t) is None


# ===========================================================================
# 6. check_position_exit — SELL SL/TP
# ===========================================================================

class TestSellExit:
    def test_sell_sl_hit_when_ask_crosses_above(self):
        b = PaperBroker()
        p = make_position(side=Side.SELL, entry_price=1.10000,
                          sl_price=1.10200, tp_price=1.09600,
                          max_hold_until_msc=10**13)
        t = make_tick(bid=1.10200, ask=1.10210)
        out = b.check_position_exit(p, t)
        assert out.close_reason == CloseReason.SL_HIT

    def test_sell_sl_not_hit_when_ask_below(self):
        b = PaperBroker()
        p = make_position(side=Side.SELL, sl_price=1.10200,
                          tp_price=1.09600, entry_price=1.10000,
                          max_hold_until_msc=10**13)
        t = make_tick(bid=1.10100, ask=1.10199)
        assert b.check_position_exit(p, t) is None

    def test_sell_tp_hit_when_ask_at_or_below(self):
        b = PaperBroker()
        p = make_position(side=Side.SELL, sl_price=1.10200,
                          tp_price=1.09600, entry_price=1.10000,
                          max_hold_until_msc=10**13)
        t = make_tick(bid=1.09500, ask=1.09600)
        out = b.check_position_exit(p, t)
        assert out.close_reason == CloseReason.TP_HIT

    def test_sell_tp_not_hit_when_ask_above(self):
        b = PaperBroker()
        p = make_position(side=Side.SELL, sl_price=1.10200,
                          tp_price=1.09600, entry_price=1.10000,
                          max_hold_until_msc=10**13)
        t = make_tick(bid=1.09600, ask=1.09601)
        assert b.check_position_exit(p, t) is None


# ===========================================================================
# 7. check_position_exit — closed position is a no-op
# ===========================================================================

class TestClosedNoOp:
    @pytest.mark.parametrize("state", [PositionState.CLOSED])
    def test_closed_returns_none(self, state):
        b = PaperBroker()
        p = make_position(state=state, max_hold_until_msc=10**13)
        t = make_tick(bid=1.0, ask=1.0)
        assert b.check_position_exit(p, t) is None


# ===========================================================================
# 8. force_close
# ===========================================================================

class TestForceClose:
    def test_force_close_default_reason_eod(self):
        b = PaperBroker()
        p = make_position(max_hold_until_msc=10**13)
        t = make_tick()
        out = b.force_close(p, t)
        assert out.close_reason == CloseReason.EOD

    @pytest.mark.parametrize("reason", list(CloseReason))
    def test_force_close_any_reason(self, reason):
        b = PaperBroker()
        p = make_position(max_hold_until_msc=10**13)
        t = make_tick()
        out = b.force_close(p, t, reason)
        assert out.close_reason == reason

    def test_force_close_state_is_closed(self):
        b = PaperBroker()
        p = make_position(max_hold_until_msc=10**13)
        out = b.force_close(p, make_tick())
        assert out.state == PositionState.CLOSED

    def test_force_close_exit_price_buy_bid_side(self):
        b = PaperBroker()
        p = make_position(side=Side.BUY)
        t = make_tick(bid=1.10000, ask=1.10010)
        out = b.force_close(p, t)
        assert out.exit_price == pytest.approx(1.10000 - 0.5 * 0.0001)

    def test_force_close_exit_price_sell_ask_side(self):
        b = PaperBroker()
        p = make_position(side=Side.SELL, sl_price=1.10200, tp_price=1.09600)
        t = make_tick(bid=1.10000, ask=1.10010)
        out = b.force_close(p, t)
        assert out.exit_price == pytest.approx(1.10010 + 0.5 * 0.0001)


# ===========================================================================
# 9. PnL math — BUY
# ===========================================================================

class TestPnLBuy:
    def test_buy_profit(self):
        b = PaperBroker(slippage_pct=0.0)
        # Enter at ask=1.10, exit at bid=1.20 → +0.10 price
        p = make_position(side=Side.BUY, entry_price=1.10,
                          sl_price=1.05, tp_price=1.25, lots=1.0)
        out = b.force_close(p, make_tick(bid=1.20, ask=1.20))
        assert out.exit_price == 1.20
        assert out.pnl_pts == pytest.approx(0.10 / 0.01)
        assert out.pnl_usd == pytest.approx(0.10 * 1.0 * 100)

    def test_buy_loss(self):
        b = PaperBroker(slippage_pct=0.0)
        p = make_position(side=Side.BUY, entry_price=1.20,
                          sl_price=1.05, tp_price=1.25, lots=0.5)
        out = b.force_close(p, make_tick(bid=1.10, ask=1.10))
        assert out.exit_price == 1.10
        assert out.pnl_pts == pytest.approx(-0.10 / 0.01)
        assert out.pnl_usd == pytest.approx(-0.10 * 0.5 * 100)

    @pytest.mark.parametrize("lots", [0.01, 0.10, 0.50, 1.0, 5.0])
    def test_buy_pnl_scales_with_lots(self, lots):
        b = PaperBroker(slippage_pct=0.0)
        p = make_position(side=Side.BUY, entry_price=1.0,
                          sl_price=0.5, tp_price=2.0, lots=lots)
        out = b.force_close(p, make_tick(bid=1.10, ask=1.10))
        assert out.pnl_usd == pytest.approx(0.10 * lots * 100)


# ===========================================================================
# 10. PnL math — SELL
# ===========================================================================

class TestPnLSell:
    def test_sell_profit(self):
        b = PaperBroker(slippage_pct=0.0)
        p = make_position(side=Side.SELL, entry_price=1.20,
                          sl_price=1.30, tp_price=1.10, lots=1.0)
        out = b.force_close(p, make_tick(bid=1.10, ask=1.10))
        assert out.pnl_pts == pytest.approx(0.10 / 0.01)
        assert out.pnl_usd == pytest.approx(0.10 * 1.0 * 100)

    def test_sell_loss(self):
        b = PaperBroker(slippage_pct=0.0)
        p = make_position(side=Side.SELL, entry_price=1.10,
                          sl_price=1.20, tp_price=1.00, lots=1.0)
        out = b.force_close(p, make_tick(bid=1.20, ask=1.20))
        assert out.pnl_pts == pytest.approx(-0.10 / 0.01)
        assert out.pnl_usd == pytest.approx(-0.10 * 1.0 * 100)


# ===========================================================================
# 11. contract_size variation
# ===========================================================================

class TestContractSize:
    @pytest.mark.parametrize("cs", [1, 10, 100, 100_000])
    def test_pnl_usd_scales_with_contract_size(self, cs):
        b = PaperBroker(slippage_pct=0.0, contract_size=cs)
        p = make_position(side=Side.BUY, entry_price=1.0,
                          sl_price=0.5, tp_price=2.0, lots=1.0)
        out = b.force_close(p, make_tick(bid=1.1, ask=1.1))
        assert out.pnl_usd == pytest.approx(0.10 * 1.0 * cs)


# ===========================================================================
# 12. End-to-end round-trip
# ===========================================================================

class TestRoundTrip:
    def test_buy_open_then_force_close_full_cycle(self):
        b = PaperBroker(slippage_pct=0.5, contract_size=100)
        i = make_intent(side=Side.BUY, lots=1.0)
        t_open = make_tick(bid=1.10000, ask=1.10010, time_msc=0)
        p = b.fill_market_order(i, t_open)
        assert p.state == PositionState.OPEN

        t_close = make_tick(bid=1.10100, ask=1.10110, time_msc=100)
        out = b.force_close(p, t_close, reason=CloseReason.MANUAL)
        assert out.state == PositionState.CLOSED
        assert out.close_reason == CloseReason.MANUAL
        assert out.entry_price == p.entry_price
        # exit at bid - slip
        expected_exit = 1.10100 - 0.5 * (1.10110 - 1.10100)
        assert out.exit_price == pytest.approx(expected_exit)

    def test_sell_open_then_sl_hit_full_cycle(self):
        b = PaperBroker(slippage_pct=0.0)
        i = make_intent(side=Side.SELL, sl_pts=100, tp_pts=200,
                        sl_price=1.10200, tp_price=1.09900,
                        intended_price=1.10000,
                        max_hold_until_msc=10**13)
        t = make_tick(bid=1.10000, ask=1.10010)
        p = b.fill_market_order(i, t)
        # Move market against us:
        t_sl = make_tick(bid=p.sl_price - 0.0001, ask=p.sl_price)
        out = b.check_position_exit(p, t_sl)
        assert out is not None
        assert out.close_reason == CloseReason.SL_HIT


# ===========================================================================
# 13. Property-based — slippage symmetry & PnL invariants
# ===========================================================================

@settings(max_examples=200, deadline=None)
@given(
    spread_pts=st.integers(min_value=0, max_value=200),
    slip_pct=st.floats(min_value=0.0, max_value=1.0,
                       allow_nan=False, allow_infinity=False),
)
def test_buy_fill_above_ask_property(spread_pts, slip_pct):
    bid = 1.0
    ask = bid + spread_pts * 0.01
    b = PaperBroker(slippage_pct=slip_pct)
    i = make_intent(side=Side.BUY, sl_price=0.5, tp_price=2.5,
                    intended_price=ask)
    pos = b.fill_market_order(i, make_tick(bid=bid, ask=ask))
    assert pos.entry_price >= ask - 1e-9


@settings(max_examples=200, deadline=None)
@given(
    spread_pts=st.integers(min_value=0, max_value=200),
    slip_pct=st.floats(min_value=0.0, max_value=1.0,
                       allow_nan=False, allow_infinity=False),
)
def test_sell_fill_below_bid_property(spread_pts, slip_pct):
    bid = 2.0
    ask = bid + spread_pts * 0.01
    b = PaperBroker(slippage_pct=slip_pct)
    i = make_intent(side=Side.SELL, sl_price=ask + 0.1, tp_price=bid - 0.1,
                    intended_price=bid)
    pos = b.fill_market_order(i, make_tick(bid=bid, ask=ask))
    assert pos.entry_price <= bid + 1e-9


@settings(max_examples=100, deadline=None)
@given(
    entry=st.floats(min_value=0.1, max_value=10.0,
                    allow_nan=False, allow_infinity=False),
    exit_=st.floats(min_value=0.1, max_value=10.0,
                    allow_nan=False, allow_infinity=False),
    lots=st.floats(min_value=0.01, max_value=10.0,
                   allow_nan=False, allow_infinity=False),
)
def test_pnl_sign_consistent_with_direction(entry, exit_, lots):
    b = PaperBroker(slippage_pct=0.0)
    # BUY
    p = make_position(side=Side.BUY, entry_price=entry,
                      sl_price=min(entry, exit_) - 0.1,
                      tp_price=max(entry, exit_) + 0.1,
                      lots=lots)
    out = b.force_close(p, make_tick(bid=exit_, ask=exit_))
    if exit_ > entry:
        assert out.pnl_usd >= 0
    elif exit_ < entry:
        assert out.pnl_usd <= 0
    else:
        assert out.pnl_usd == 0


# ===========================================================================
# 14. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_extreme_wide_spread(self):
        b = PaperBroker(slippage_pct=0.5)
        t = make_tick(bid=1.0, ask=2.0)  # 100% spread
        i = make_intent(side=Side.BUY, intended_price=2.0,
                        sl_price=0.5, tp_price=3.0)
        pos = b.fill_market_order(i, t)
        # ask + 0.5 spread
        assert pos.entry_price == pytest.approx(2.0 + 0.5)

    def test_negative_pnl_pts(self):
        b = PaperBroker(slippage_pct=0.0)
        p = make_position(side=Side.BUY, entry_price=1.20)
        out = b.force_close(p, make_tick(bid=1.10, ask=1.10))
        assert out.pnl_pts < 0

    def test_zero_lots_yields_zero_pnl_usd(self):
        b = PaperBroker(slippage_pct=0.0)
        p = make_position(lots=0.0)
        out = b.force_close(p, make_tick(bid=1.10, ask=1.10))
        assert out.pnl_usd == 0.0

    def test_exit_msc_set_from_tick(self):
        b = PaperBroker()
        p = make_position()
        out = b.force_close(p, make_tick(time_msc=99999))
        assert out.exit_time_msc == 99999

    def test_close_preserves_position_id(self):
        b = PaperBroker()
        p = make_position(position_id="keep-me")
        out = b.force_close(p, make_tick())
        assert out.position_id == "keep-me"

    def test_close_preserves_entry_data(self):
        b = PaperBroker()
        p = make_position(entry_price=1.5, entry_time_msc=42)
        out = b.force_close(p, make_tick())
        assert out.entry_price == 1.5
        assert out.entry_time_msc == 42
