"""Order / Position / Side / CloseReason value objects."""

from __future__ import annotations

import dataclasses
import pytest

from execution.order import OrderIntent, Side, SignalType
from execution.position import Position, PositionState, CloseReason
from execution.order_router import (
    GriffOpenPosition, GriffPendingOrder,
)
from strategy.patterns.base import Direction
from utils.session import SessionLabel

from tests.execution.fixtures.mock_orders import make_intent
from tests.execution.fixtures.mock_positions import (
    make_position, make_griff_open, make_griff_pending,
)


# ===========================================================================
# 1. Side enum
# ===========================================================================

class TestSideEnum:
    def test_buy_value(self):
        assert Side.BUY == "BUY"
        assert Side.BUY.value == "BUY"

    def test_sell_value(self):
        assert Side.SELL == "SELL"
        assert Side.SELL.value == "SELL"

    def test_str_subclass(self):
        # Enum is a str subclass
        assert isinstance(Side.BUY, str)

    def test_distinct(self):
        assert Side.BUY != Side.SELL

    @pytest.mark.parametrize("name,val", [("BUY", "BUY"), ("SELL", "SELL")])
    def test_lookup_by_name(self, name, val):
        assert Side[name].value == val

    def test_value_round_trip(self):
        assert Side("BUY") is Side.BUY
        assert Side("SELL") is Side.SELL


# ===========================================================================
# 2. SignalType enum
# ===========================================================================

class TestSignalType:
    @pytest.mark.parametrize("kind", ["SWEEP", "MOMENTUM", "REJECTION"])
    def test_values_present(self, kind):
        assert SignalType(kind).value == kind

    def test_distinct(self):
        types = {SignalType.SWEEP, SignalType.MOMENTUM, SignalType.REJECTION}
        assert len(types) == 3

    def test_str_subclass(self):
        assert isinstance(SignalType.SWEEP, str)


# ===========================================================================
# 3. OrderIntent
# ===========================================================================

class TestOrderIntent:
    def test_construct(self, intent):
        assert intent.side == Side.BUY
        assert intent.lots == 0.10
        assert intent.signal_type == SignalType.SWEEP

    def test_frozen(self, intent):
        with pytest.raises(dataclasses.FrozenInstanceError):
            intent.lots = 1.0  # type: ignore[misc]

    def test_default_sl_pts_zero(self):
        i = make_intent()
        assert i.sl_pts == 0.0
        assert i.tp_pts == 0.0

    def test_explicit_sl_pts(self):
        i = make_intent(sl_pts=200, tp_pts=400)
        assert i.sl_pts == 200
        assert i.tp_pts == 400

    @pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
    def test_both_sides(self, side):
        # Adjust SL/TP so they remain on the same side of intended_price.
        if side == Side.BUY:
            i = make_intent(side=side, sl_price=1.0, tp_price=2.0,
                            intended_price=1.5)
        else:
            i = make_intent(side=side, sl_price=2.0, tp_price=1.0,
                            intended_price=1.5)
        assert i.side == side

    @pytest.mark.parametrize("sess", list(SessionLabel))
    def test_all_session_labels_supported(self, sess):
        i = make_intent(session=sess)
        assert i.session == sess

    def test_equality_by_value(self):
        a = make_intent(signal_id="s1")
        b = make_intent(signal_id="s1")
        assert a == b

    def test_inequality_when_lots_differ(self):
        a = make_intent(lots=0.10)
        b = make_intent(lots=0.20)
        assert a != b

    def test_hashable(self):
        # Frozen dataclasses are hashable; we can put intents in a set
        a = make_intent()
        b = make_intent()
        s = {a, b}
        assert len(s) == 1

    def test_signal_type_serializes(self):
        i = make_intent(signal_type=SignalType.MOMENTUM)
        assert i.signal_type.value == "MOMENTUM"

    @pytest.mark.parametrize("lots", [0.01, 0.10, 1.0, 10.0])
    def test_lots_pass_through(self, lots):
        assert make_intent(lots=lots).lots == lots


# ===========================================================================
# 4. PositionState / CloseReason enums
# ===========================================================================

class TestPositionState:
    def test_values(self):
        assert PositionState.OPEN == "OPEN"
        assert PositionState.CLOSED == "CLOSED"

    def test_distinct(self):
        assert PositionState.OPEN != PositionState.CLOSED


class TestCloseReason:
    @pytest.mark.parametrize("name", [
        "TP_HIT", "SL_HIT", "TIME_EXIT", "MANUAL", "EOD",
    ])
    def test_values(self, name):
        assert CloseReason(name).value == name

    def test_distinct(self):
        reasons = {CloseReason.TP_HIT, CloseReason.SL_HIT,
                   CloseReason.TIME_EXIT, CloseReason.MANUAL,
                   CloseReason.EOD}
        assert len(reasons) == 5


# ===========================================================================
# 5. Position dataclass
# ===========================================================================

class TestPosition:
    def test_construct(self, open_position):
        assert open_position.side == Side.BUY
        assert open_position.state == PositionState.OPEN
        assert open_position.exit_price is None
        assert open_position.close_reason is None

    def test_frozen(self, open_position):
        with pytest.raises(dataclasses.FrozenInstanceError):
            open_position.lots = 1.0  # type: ignore[misc]

    def test_default_optional_fields_none(self):
        p = make_position()
        assert p.exit_price is None
        assert p.exit_time_msc is None
        assert p.pnl_pts is None
        assert p.pnl_usd is None

    def test_closed_position_has_exit_data(self):
        p = make_position(
            state=PositionState.CLOSED,
            exit_price=1.10500,
            exit_time_msc=1_700_000_500_000,
            close_reason=CloseReason.TP_HIT,
            pnl_pts=500,
            pnl_usd=50.0,
        )
        assert p.state == PositionState.CLOSED
        assert p.close_reason == CloseReason.TP_HIT
        assert p.pnl_usd == 50.0

    @pytest.mark.parametrize("reason", list(CloseReason))
    def test_any_close_reason(self, reason):
        p = make_position(state=PositionState.CLOSED, close_reason=reason)
        assert p.close_reason == reason

    @pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
    def test_either_side(self, side):
        p = make_position(side=side)
        assert p.side == side

    def test_equality_by_value(self):
        a = make_position()
        b = make_position()
        assert a == b

    def test_hashable(self):
        a = make_position()
        s = {a}
        assert a in s

    @pytest.mark.parametrize("session", ["LONDON", "ASIAN", "NY",
                                          "LONDON_NY_OVERLAP", None])
    def test_session_optional(self, session):
        p = make_position(session=session)
        assert p.session == session


# ===========================================================================
# 6. GriffOpenPosition
# ===========================================================================

class TestGriffOpenPosition:
    def test_construct(self):
        p = make_griff_open()
        assert p.side == Direction.BUY
        assert p.mt5_ticket == 12345

    def test_frozen(self):
        p = make_griff_open()
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.lots = 1.0  # type: ignore[misc]

    @pytest.mark.parametrize("side", [Direction.BUY, Direction.SELL])
    def test_both_directions(self, side):
        p = make_griff_open(side=side)
        assert p.side == side

    def test_default_pattern(self):
        p = make_griff_open()
        assert p.pattern_name == "ASIAN_SWEEP"

    @pytest.mark.parametrize("pname", [
        "FLAG", "ASIAN_SWEEP", "CONTINUATION", "REVERSAL", "COMBO",
    ])
    def test_any_pattern_accepted(self, pname):
        p = make_griff_open(pattern_name=pname)
        assert p.pattern_name == pname

    def test_dry_run_ticket_minus_one(self):
        p = make_griff_open(mt5_ticket=-1)
        assert p.mt5_ticket == -1

    def test_position_id_present(self):
        p = make_griff_open()
        assert isinstance(p.position_id, str) and len(p.position_id) > 0

    @pytest.mark.parametrize("lots", [0.01, 0.1, 1.0])
    def test_lot_sizes(self, lots):
        assert make_griff_open(lots=lots).lots == lots


# ===========================================================================
# 7. GriffPendingOrder
# ===========================================================================

class TestGriffPendingOrder:
    def test_default_is_stop(self):
        o = make_griff_pending()
        assert o.is_limit is False

    def test_explicit_limit(self):
        o = make_griff_pending(is_limit=True)
        assert o.is_limit is True

    def test_frozen(self):
        o = make_griff_pending()
        with pytest.raises(dataclasses.FrozenInstanceError):
            o.lots = 1.0  # type: ignore[misc]

    @pytest.mark.parametrize("side", [Direction.BUY, Direction.SELL])
    def test_both_directions(self, side):
        o = make_griff_pending(side=side)
        assert o.side == side

    def test_expiry_msc_preserved(self):
        o = make_griff_pending(expiry_msc=1_800_000_000_000)
        assert o.expiry_msc == 1_800_000_000_000

    @pytest.mark.parametrize("price", [0.5, 1.0, 1.10, 2000.0, 100000.0])
    def test_pending_price_pass_through(self, price):
        o = make_griff_pending(pending_price=price)
        assert o.pending_price == price


# ===========================================================================
# 8. Field type matrix
# ===========================================================================

@pytest.mark.parametrize("field,expected_type", [
    ("position_id", str),
    ("mt5_ticket", int),
    ("symbol", str),
    ("lots", float),
    ("entry_price", float),
    ("sl_price", float),
    ("tp_price", float),
    ("opened_msc", int),
    ("signal_id", str),
    ("pattern_name", str),
])
def test_griff_open_field_types(field, expected_type):
    p = make_griff_open()
    assert isinstance(getattr(p, field), expected_type)


@pytest.mark.parametrize("field,expected_type", [
    ("order_id", str),
    ("mt5_ticket", int),
    ("symbol", str),
    ("lots", float),
    ("pending_price", float),
    ("sl_price", float),
    ("tp_price", float),
    ("expiry_msc", int),
    ("signal_id", str),
    ("pattern_name", str),
    ("is_limit", bool),
])
def test_griff_pending_field_types(field, expected_type):
    o = make_griff_pending()
    assert isinstance(getattr(o, field), expected_type)


@pytest.mark.parametrize("field", [
    "position_id", "side", "lots", "entry_price",
    "entry_time_msc", "sl_price", "tp_price",
    "max_hold_until_msc", "state",
])
def test_position_required_fields_present(field):
    p = make_position()
    assert getattr(p, field) is not None


# ===========================================================================
# 9. Negative path: positional misuse on frozen dataclasses
# ===========================================================================

class TestImmutability:
    @pytest.mark.parametrize("attr,val", [
        ("position_id", "new"),
        ("entry_price", 2.0),
        ("sl_price", 0.5),
    ])
    def test_position_cannot_reassign(self, attr, val):
        p = make_position()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(p, attr, val)

    @pytest.mark.parametrize("attr,val", [
        ("symbol", "USDJPY"),
        ("lots", 99.0),
        ("mt5_ticket", -42),
    ])
    def test_griff_open_cannot_reassign(self, attr, val):
        p = make_griff_open()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(p, attr, val)


# ===========================================================================
# 10. Direction enum smoke (re-tested here because OrderIntent doesn't use it
#     but Griff types do — ensure mock consistency)
# ===========================================================================

class TestDirection:
    def test_values(self):
        assert Direction.BUY == "BUY"
        assert Direction.SELL == "SELL"

    def test_distinct(self):
        assert Direction.BUY != Direction.SELL

    @pytest.mark.parametrize("val", ["BUY", "SELL"])
    def test_round_trip(self, val):
        assert Direction(val).value == val


# ===========================================================================
# 11. dataclass-level invariants
# ===========================================================================

class TestInvariants:
    def test_position_id_unique_when_random(self):
        a = make_griff_open()
        b = make_griff_open()
        assert a.position_id != b.position_id

    def test_pending_order_id_unique(self):
        a = make_griff_pending()
        b = make_griff_pending()
        assert a.order_id != b.order_id

    def test_position_open_has_no_exit_price(self):
        p = make_position(state=PositionState.OPEN)
        assert p.exit_price is None

    def test_position_closed_with_exit_price_passes(self):
        p = make_position(
            state=PositionState.CLOSED,
            exit_price=1.10100,
            exit_time_msc=1_700_000_001_000,
            close_reason=CloseReason.SL_HIT,
            pnl_pts=-100,
            pnl_usd=-10.0,
        )
        assert p.state == PositionState.CLOSED


# ===========================================================================
# 12. OrderIntent — covers every (Side, SignalType, SessionLabel) combo
# ===========================================================================

@pytest.mark.parametrize("sess", list(SessionLabel))
@pytest.mark.parametrize("kind", list(SignalType))
@pytest.mark.parametrize("side", list(Side))
def test_intent_combinations(side, kind, sess):
    if side == Side.BUY:
        i = make_intent(side=side, signal_type=kind, session=sess,
                        sl_price=1.0, tp_price=2.0, intended_price=1.5)
    else:
        i = make_intent(side=side, signal_type=kind, session=sess,
                        sl_price=2.0, tp_price=1.0, intended_price=1.5)
    assert i.side == side
    assert i.signal_type == kind
    assert i.session == sess
