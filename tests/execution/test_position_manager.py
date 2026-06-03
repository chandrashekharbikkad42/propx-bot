"""GriffPositionManager — track positions + pending orders, per-bar maintain.

Heavy use of `unittest.mock.AsyncMock` to stub the router (since real
modify_sl / cancel_pending would call MT5). Trailing-SL is given a
controllable stub that returns scripted new-SL values.
"""

from __future__ import annotations
import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from data.bar_aggregator import Bar
from execution.order_router import (
    GriffOpenPosition, GriffPendingOrder,
)
from execution.position_manager import (
    GriffPositionManager, MaintenanceReport,
    _direction_to_side, _legacy_position, _replace_sl, _sl_hit,
)
from execution.order import Side
from execution.position import Position, PositionState
from strategy.patterns.base import Direction

from tests.execution.fixtures.mock_positions import (
    make_griff_open, make_griff_pending,
)


def run(coro):
    return asyncio.run(coro)


def make_bar(symbol="EURUSD", high=1.10100, low=1.09900,
             open_=1.10000, close=1.10050, volume=10,
             time_msc=1_700_000_000_000) -> Bar:
    return Bar(
        symbol=symbol,
        time_msc=time_msc,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        spread_mean=0.0001,
    )


def _stub_router():
    router = MagicMock()
    router.modify_sl = AsyncMock(return_value=True)
    router.cancel_pending = AsyncMock(return_value=True)
    return router


def _stub_tracker():
    tracker = MagicMock()
    tracker.update = MagicMock(return_value={})
    return tracker


def _stub_trail(new_sl=None):
    trail = MagicMock()
    trail.update = MagicMock(return_value=new_sl)
    return trail


@pytest.fixture
def pm_factory():
    def _make(new_sl=None, modify_returns=True, cancel_returns=True):
        router = _stub_router()
        router.modify_sl = AsyncMock(return_value=modify_returns)
        router.cancel_pending = AsyncMock(return_value=cancel_returns)
        tracker = _stub_tracker()
        trail = _stub_trail(new_sl)
        pm = GriffPositionManager(router, tracker, trail)
        return pm, router, tracker, trail
    return _make


# ===========================================================================
# 1. Free helpers
# ===========================================================================

class TestDirectionToSide:
    def test_buy(self):
        assert _direction_to_side(Direction.BUY) == Side.BUY

    def test_sell(self):
        assert _direction_to_side(Direction.SELL) == Side.SELL


class TestLegacyPosition:
    def test_returns_open_position(self):
        p = make_griff_open()
        out = _legacy_position(p)
        assert isinstance(out, Position)
        assert out.state == PositionState.OPEN
        assert out.entry_price == p.entry_price
        assert out.lots == p.lots

    def test_side_mapped(self):
        a = _legacy_position(make_griff_open(side=Direction.BUY))
        b = _legacy_position(make_griff_open(side=Direction.SELL))
        assert a.side == Side.BUY
        assert b.side == Side.SELL

    def test_signal_type_pass_through(self):
        out = _legacy_position(make_griff_open(pattern_name="REVERSAL"))
        assert out.signal_type == "REVERSAL"

    def test_max_hold_zero(self):
        out = _legacy_position(make_griff_open())
        assert out.max_hold_until_msc == 0


class TestReplaceSl:
    def test_sl_updated(self):
        p = make_griff_open(sl_price=1.0)
        out = _replace_sl(p, new_sl=0.95)
        assert out.sl_price == 0.95

    def test_other_fields_preserved(self):
        p = make_griff_open(entry_price=1.5, lots=0.10)
        out = _replace_sl(p, 0.95)
        assert out.entry_price == 1.5
        assert out.lots == 0.10
        assert out.position_id == p.position_id


class TestSlHit:
    def test_buy_low_below_sl(self):
        p = make_griff_open(side=Direction.BUY, sl_price=1.10000)
        bar = make_bar(low=1.09900)
        assert _sl_hit(p, bar)

    def test_buy_low_above_sl(self):
        p = make_griff_open(side=Direction.BUY, sl_price=1.10000)
        bar = make_bar(low=1.10100)
        assert not _sl_hit(p, bar)

    def test_buy_low_equal_sl_is_hit(self):
        p = make_griff_open(side=Direction.BUY, sl_price=1.10000)
        bar = make_bar(low=1.10000)
        assert _sl_hit(p, bar)

    def test_sell_high_above_sl(self):
        p = make_griff_open(side=Direction.SELL, sl_price=1.10000)
        bar = make_bar(high=1.10100)
        assert _sl_hit(p, bar)

    def test_sell_high_below_sl(self):
        p = make_griff_open(side=Direction.SELL, sl_price=1.10000)
        bar = make_bar(high=1.09900)
        assert not _sl_hit(p, bar)


# ===========================================================================
# 2. Construction & accessors
# ===========================================================================

class TestConstructor:
    def test_empty_state(self, pm_factory):
        pm, *_ = pm_factory()
        assert pm.open_positions == ()
        assert pm.pending_orders == ()

    def test_positions_for_empty(self, pm_factory):
        pm, *_ = pm_factory()
        assert pm.positions_for("EURUSD") == ()

    def test_pendings_for_empty(self, pm_factory):
        pm, *_ = pm_factory()
        assert pm.pendings_for("EURUSD") == ()


# ===========================================================================
# 3. register_position / register_pending
# ===========================================================================

class TestRegister:
    def test_register_position_adds(self, pm_factory):
        pm, *_ = pm_factory()
        p = make_griff_open()
        pm.register_position(p)
        assert p in pm.open_positions

    def test_register_pending_adds(self, pm_factory):
        pm, *_ = pm_factory()
        o = make_griff_pending()
        pm.register_pending(o)
        assert o in pm.pending_orders

    def test_two_positions_distinct(self, pm_factory):
        pm, *_ = pm_factory()
        a = make_griff_open()
        b = make_griff_open()
        pm.register_position(a)
        pm.register_position(b)
        assert len(pm.open_positions) == 2

    def test_positions_for_filters_by_symbol(self, pm_factory):
        pm, *_ = pm_factory()
        pm.register_position(make_griff_open(symbol="EURUSD"))
        pm.register_position(make_griff_open(symbol="USDJPY"))
        assert len(pm.positions_for("EURUSD")) == 1
        assert len(pm.positions_for("USDJPY")) == 1
        assert len(pm.positions_for("GBPUSD")) == 0


# ===========================================================================
# 4. on_pending_filled — promotes pending to position
# ===========================================================================

class TestOnPendingFilled:
    def test_promotes(self, pm_factory):
        pm, *_ = pm_factory()
        order = make_griff_pending()
        pm.register_pending(order)
        out = pm.on_pending_filled(
            order.order_id, fill_price=1.10005,
            mt5_position_ticket=42, fill_msc=123,
        )
        assert out is not None
        assert out.mt5_ticket == 42
        assert out.entry_price == 1.10005
        assert out.opened_msc == 123

    def test_removes_pending(self, pm_factory):
        pm, *_ = pm_factory()
        order = make_griff_pending()
        pm.register_pending(order)
        pm.on_pending_filled(order.order_id, fill_price=1.0,
                              mt5_position_ticket=1, fill_msc=0)
        assert order not in pm.pending_orders

    def test_unknown_order_returns_none(self, pm_factory):
        pm, *_ = pm_factory()
        out = pm.on_pending_filled("nope", fill_price=1.0,
                                    mt5_position_ticket=1, fill_msc=0)
        assert out is None

    def test_promoted_position_in_open_list(self, pm_factory):
        pm, *_ = pm_factory()
        order = make_griff_pending()
        pm.register_pending(order)
        promoted = pm.on_pending_filled(
            order.order_id, fill_price=1.10005,
            mt5_position_ticket=42, fill_msc=123,
        )
        assert promoted in pm.open_positions

    def test_promotion_preserves_sl_tp(self, pm_factory):
        pm, *_ = pm_factory()
        order = make_griff_pending(sl_price=1.09000, tp_price=1.11000)
        pm.register_pending(order)
        out = pm.on_pending_filled(order.order_id, fill_price=1.10005,
                                    mt5_position_ticket=1, fill_msc=0)
        assert out.sl_price == 1.09000
        assert out.tp_price == 1.11000


# ===========================================================================
# 5. maintain — basic happy path with no trail change
# ===========================================================================

class TestMaintainBasic:
    def test_empty_returns_empty_report(self, pm_factory):
        pm, *_ = pm_factory()
        report = run(pm.maintain("EURUSD", make_bar(), now_msc=0))
        assert isinstance(report, MaintenanceReport)
        assert report.pair == "EURUSD"
        assert report.sl_updates == ()
        assert report.closed_positions == ()
        assert report.cancelled_pendings == ()

    def test_swing_tracker_updated_before_trail(self, pm_factory):
        pm, _, tracker, _ = pm_factory()
        pm.register_position(make_griff_open(symbol="EURUSD"))
        bar = make_bar(symbol="EURUSD")
        run(pm.maintain("EURUSD", bar, now_msc=0))
        tracker.update.assert_called_once_with("EURUSD", bar)

    def test_position_unchanged_when_trail_returns_none(self, pm_factory):
        pm, router, _, _ = pm_factory(new_sl=None)
        p = make_griff_open(symbol="EURUSD", sl_price=1.09000)
        pm.register_position(p)
        bar = make_bar(symbol="EURUSD", low=1.10000, high=1.10500)
        report = run(pm.maintain("EURUSD", bar, now_msc=0))
        assert report.sl_updates == ()
        router.modify_sl.assert_not_called()

    def test_position_skipped_when_pair_mismatch(self, pm_factory):
        pm, _, tracker, _ = pm_factory()
        pm.register_position(make_griff_open(symbol="EURUSD"))
        bar = make_bar(symbol="USDJPY")
        report = run(pm.maintain("USDJPY", bar, now_msc=0))
        # Position not touched; tracker only got the USDJPY bar
        assert report.sl_updates == ()


# ===========================================================================
# 6. maintain — trail update path
# ===========================================================================

class TestMaintainTrail:
    def test_trail_returns_new_sl_router_called(self, pm_factory):
        pm, router, _, _ = pm_factory(new_sl=1.09500)
        p = make_griff_open(symbol="EURUSD", sl_price=1.09000)
        pm.register_position(p)
        bar = make_bar(symbol="EURUSD", low=1.09700, high=1.10500)
        report = run(pm.maintain("EURUSD", bar, now_msc=0))
        router.modify_sl.assert_called_once()
        assert len(report.sl_updates) == 1
        assert report.sl_updates[0][1] == 1.09500

    def test_position_replaced_with_new_sl(self, pm_factory):
        pm, *_ = pm_factory(new_sl=1.09500)
        p = make_griff_open(symbol="EURUSD", sl_price=1.09000)
        pm.register_position(p)
        bar = make_bar(symbol="EURUSD", low=1.09700, high=1.10500)
        run(pm.maintain("EURUSD", bar, now_msc=0))
        # The stored position should now have sl=1.09500
        stored = pm.positions_for("EURUSD")[0]
        assert stored.sl_price == 1.09500

    def test_router_failure_does_not_replace(self, pm_factory):
        pm, *_ = pm_factory(new_sl=1.09500, modify_returns=False)
        p = make_griff_open(symbol="EURUSD", sl_price=1.09000)
        pm.register_position(p)
        bar = make_bar(symbol="EURUSD", low=1.09700, high=1.10500)
        report = run(pm.maintain("EURUSD", bar, now_msc=0))
        assert report.sl_updates == ()
        stored = pm.positions_for("EURUSD")[0]
        assert stored.sl_price == 1.09000  # unchanged


# ===========================================================================
# 7. maintain — SL hit detection on bar
# ===========================================================================

class TestMaintainSlHit:
    def test_buy_sl_hit_removed_from_map(self, pm_factory):
        pm, *_ = pm_factory()
        p = make_griff_open(symbol="EURUSD", side=Direction.BUY,
                            sl_price=1.10000)
        pm.register_position(p)
        bar = make_bar(symbol="EURUSD", low=1.09900, high=1.10500)
        report = run(pm.maintain("EURUSD", bar, now_msc=0))
        assert p.position_id in report.closed_positions
        assert pm.positions_for("EURUSD") == ()

    def test_sell_sl_hit_removed_from_map(self, pm_factory):
        pm, *_ = pm_factory()
        p = make_griff_open(symbol="EURUSD", side=Direction.SELL,
                            sl_price=1.09000)
        pm.register_position(p)
        bar = make_bar(symbol="EURUSD", low=1.08500, high=1.09100)
        report = run(pm.maintain("EURUSD", bar, now_msc=0))
        assert p.position_id in report.closed_positions

    def test_no_sl_hit_keeps_position(self, pm_factory):
        pm, *_ = pm_factory()
        p = make_griff_open(symbol="EURUSD", side=Direction.BUY,
                            sl_price=1.09000)
        pm.register_position(p)
        bar = make_bar(symbol="EURUSD", low=1.09500, high=1.10500)
        report = run(pm.maintain("EURUSD", bar, now_msc=0))
        assert report.closed_positions == ()
        assert len(pm.positions_for("EURUSD")) == 1


# ===========================================================================
# 8. maintain — pending expiry cancellation
# ===========================================================================

class TestMaintainPendingExpiry:
    def test_expired_pending_cancelled(self, pm_factory):
        pm, router, *_ = pm_factory()
        o = make_griff_pending(symbol="EURUSD", expiry_msc=1000)
        pm.register_pending(o)
        report = run(pm.maintain("EURUSD", make_bar(symbol="EURUSD"),
                                  now_msc=2000))
        router.cancel_pending.assert_called_once()
        assert o.order_id in report.cancelled_pendings
        assert pm.pending_orders == ()

    def test_not_yet_expired_kept(self, pm_factory):
        pm, router, *_ = pm_factory()
        o = make_griff_pending(symbol="EURUSD", expiry_msc=10_000)
        pm.register_pending(o)
        report = run(pm.maintain("EURUSD", make_bar(symbol="EURUSD"),
                                  now_msc=1000))
        router.cancel_pending.assert_not_called()
        assert report.cancelled_pendings == ()
        assert len(pm.pending_orders) == 1

    def test_cancel_failure_keeps_pending(self, pm_factory):
        pm, router, *_ = pm_factory(cancel_returns=False)
        o = make_griff_pending(symbol="EURUSD", expiry_msc=1000)
        pm.register_pending(o)
        report = run(pm.maintain("EURUSD", make_bar(symbol="EURUSD"),
                                  now_msc=2000))
        assert report.cancelled_pendings == ()
        assert len(pm.pending_orders) == 1

    def test_pending_pair_mismatch_skipped(self, pm_factory):
        pm, router, *_ = pm_factory()
        o = make_griff_pending(symbol="USDJPY", expiry_msc=1000)
        pm.register_pending(o)
        run(pm.maintain("EURUSD", make_bar(symbol="EURUSD"), now_msc=2000))
        router.cancel_pending.assert_not_called()
        assert len(pm.pending_orders) == 1


# ===========================================================================
# 9. forget_position / forget_pending
# ===========================================================================

class TestForget:
    def test_forget_position(self, pm_factory):
        pm, *_ = pm_factory()
        p = make_griff_open()
        pm.register_position(p)
        out = pm.forget_position(p.position_id)
        assert out == p
        assert pm.open_positions == ()

    def test_forget_position_unknown_returns_none(self, pm_factory):
        pm, *_ = pm_factory()
        assert pm.forget_position("nope") is None

    def test_forget_pending(self, pm_factory):
        pm, *_ = pm_factory()
        o = make_griff_pending()
        pm.register_pending(o)
        out = pm.forget_pending(o.order_id)
        assert out == o
        assert pm.pending_orders == ()

    def test_forget_pending_unknown_returns_none(self, pm_factory):
        pm, *_ = pm_factory()
        assert pm.forget_pending("nope") is None


# ===========================================================================
# 10. MaintenanceReport
# ===========================================================================

class TestMaintenanceReport:
    def test_frozen(self):
        rep = MaintenanceReport(
            pair="X", bar_close_msc=0,
            sl_updates=(), closed_positions=(), cancelled_pendings=(),
        )
        import dataclasses
        with pytest.raises(dataclasses.FrozenInstanceError):
            rep.pair = "Y"  # type: ignore[misc]

    def test_pair_pass_through(self, pm_factory):
        pm, *_ = pm_factory()
        rep = run(pm.maintain("EURUSD", make_bar(symbol="EURUSD"),
                                now_msc=0))
        assert rep.pair == "EURUSD"

    def test_bar_close_msc_set(self, pm_factory):
        pm, *_ = pm_factory()
        bar = make_bar(symbol="EURUSD", time_msc=99)
        rep = run(pm.maintain("EURUSD", bar, now_msc=0))
        assert rep.bar_close_msc == 99


# ===========================================================================
# 11. Multi-position / multi-pair scenarios
# ===========================================================================

class TestMultiPosition:
    def test_two_positions_one_sl_hit(self, pm_factory):
        pm, *_ = pm_factory()
        a = make_griff_open(symbol="EURUSD", side=Direction.BUY,
                            sl_price=1.10000)
        b = make_griff_open(symbol="EURUSD", side=Direction.BUY,
                            sl_price=1.05000)
        pm.register_position(a)
        pm.register_position(b)
        bar = make_bar(symbol="EURUSD", low=1.09900, high=1.10500)
        rep = run(pm.maintain("EURUSD", bar, now_msc=0))
        assert a.position_id in rep.closed_positions
        assert b.position_id not in rep.closed_positions

    def test_multi_pair_only_target_pair_processed(self, pm_factory):
        pm, *_ = pm_factory()
        pm.register_position(make_griff_open(symbol="EURUSD",
                                              sl_price=1.10000))
        pm.register_position(make_griff_open(symbol="USDJPY",
                                              sl_price=140.00))
        bar = make_bar(symbol="EURUSD", low=1.09900, high=1.10500)
        rep = run(pm.maintain("EURUSD", bar, now_msc=0))
        # Only EURUSD was hit
        assert len(rep.closed_positions) == 1
        assert len(pm.positions_for("USDJPY")) == 1


# ===========================================================================
# 12. Combined scenarios
# ===========================================================================

class TestCombined:
    def test_position_sl_hit_and_pending_expired_same_bar(self, pm_factory):
        pm, *_ = pm_factory()
        p = make_griff_open(symbol="EURUSD", side=Direction.BUY,
                            sl_price=1.10000)
        o = make_griff_pending(symbol="EURUSD", expiry_msc=1000)
        pm.register_position(p)
        pm.register_pending(o)
        bar = make_bar(symbol="EURUSD", low=1.09900, high=1.10500)
        rep = run(pm.maintain("EURUSD", bar, now_msc=2000))
        assert p.position_id in rep.closed_positions
        assert o.order_id in rep.cancelled_pendings


# ===========================================================================
# 13. Edge cases
# ===========================================================================

class TestEdge:
    def test_register_position_logs_correctly(self, pm_factory):
        pm, *_ = pm_factory()
        # Just confirm no exception with arbitrary IDs
        pm.register_position(make_griff_open(position_id="x" * 64))
        assert len(pm.open_positions) == 1

    def test_open_positions_returns_tuple(self, pm_factory):
        pm, *_ = pm_factory()
        pm.register_position(make_griff_open())
        assert isinstance(pm.open_positions, tuple)

    def test_pending_orders_returns_tuple(self, pm_factory):
        pm, *_ = pm_factory()
        pm.register_pending(make_griff_pending())
        assert isinstance(pm.pending_orders, tuple)
