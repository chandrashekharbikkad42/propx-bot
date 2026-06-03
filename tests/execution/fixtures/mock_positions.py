"""Factory helpers for Position, Tick, GriffOpenPosition, GriffPendingOrder."""

from __future__ import annotations
import uuid

from data.tick_collector import Tick
from execution.order_router import (
    GriffOpenPosition,
    GriffPendingOrder,
)
from execution.order import Side
from execution.position import Position, PositionState, CloseReason
from strategy.patterns.base import Direction


def make_tick(
    *,
    bid: float = 1.10000,
    ask: float = 1.10010,
    last: float = 1.10005,
    volume: int = 1,
    volume_real: float = 1.0,
    flags: int = 0,
    time_msc: int = 1_700_000_000_000,
) -> Tick:
    return Tick(
        time_msc=time_msc,
        bid=bid,
        ask=ask,
        last=last,
        volume=volume,
        volume_real=volume_real,
        flags=flags,
    )


def make_position(
    *,
    position_id: str = "pos-1",
    side: Side = Side.BUY,
    lots: float = 0.10,
    entry_price: float = 1.10000,
    entry_time_msc: int = 1_700_000_000_000,
    sl_price: float = 1.09800,
    tp_price: float = 1.10400,
    max_hold_until_msc: int = 1_700_001_000_000,
    state: PositionState = PositionState.OPEN,
    signal_type: str = "SWEEP",
    session: str = "LONDON",
    exit_price=None,
    exit_time_msc=None,
    close_reason=None,
    pnl_pts=None,
    pnl_usd=None,
) -> Position:
    return Position(
        position_id=position_id,
        side=side,
        lots=lots,
        entry_price=entry_price,
        entry_time_msc=entry_time_msc,
        sl_price=sl_price,
        tp_price=tp_price,
        max_hold_until_msc=max_hold_until_msc,
        state=state,
        signal_type=signal_type,
        session=session,
        exit_price=exit_price,
        exit_time_msc=exit_time_msc,
        close_reason=close_reason,
        pnl_pts=pnl_pts,
        pnl_usd=pnl_usd,
    )


def make_griff_open(
    *,
    position_id: str = "",
    mt5_ticket: int = 12345,
    symbol: str = "EURUSD",
    side: Direction = Direction.BUY,
    lots: float = 0.10,
    entry_price: float = 1.10000,
    sl_price: float = 1.09800,
    tp_price: float = 1.10400,
    opened_msc: int = 1_700_000_000_000,
    signal_id: str = "sweep:abcd1234",
    pattern_name: str = "ASIAN_SWEEP",
) -> GriffOpenPosition:
    return GriffOpenPosition(
        position_id=position_id or uuid.uuid4().hex,
        mt5_ticket=mt5_ticket,
        symbol=symbol,
        side=side,
        lots=lots,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
        opened_msc=opened_msc,
        signal_id=signal_id,
        pattern_name=pattern_name,
    )


def make_griff_pending(
    *,
    order_id: str = "",
    mt5_ticket: int = 99999,
    symbol: str = "EURUSD",
    side: Direction = Direction.BUY,
    lots: float = 0.10,
    pending_price: float = 1.10000,
    sl_price: float = 1.09800,
    tp_price: float = 1.10400,
    expiry_msc: int = 1_700_003_600_000,
    signal_id: str = "sweep:abcd1234",
    pattern_name: str = "CONTINUATION",
    is_limit: bool = False,
) -> GriffPendingOrder:
    return GriffPendingOrder(
        order_id=order_id or uuid.uuid4().hex,
        mt5_ticket=mt5_ticket,
        symbol=symbol,
        side=side,
        lots=lots,
        pending_price=pending_price,
        sl_price=sl_price,
        tp_price=tp_price,
        expiry_msc=expiry_msc,
        signal_id=signal_id,
        pattern_name=pattern_name,
        is_limit=is_limit,
    )
