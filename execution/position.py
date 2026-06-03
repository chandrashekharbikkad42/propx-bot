"""Position value object. Frozen — broker constructs new Positions for state transitions."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from execution.order import Side


class PositionState(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class CloseReason(str, Enum):
    TP_HIT = "TP_HIT"
    SL_HIT = "SL_HIT"
    TIME_EXIT = "TIME_EXIT"
    MANUAL = "MANUAL"
    EOD = "EOD"


@dataclass(frozen=True)
class Position:
    position_id: str
    side: Side
    lots: float
    entry_price: float
    entry_time_msc: int
    sl_price: float
    tp_price: float
    max_hold_until_msc: int
    state: PositionState
    signal_type: Optional[str] = None
    session: Optional[str] = None
    exit_price: Optional[float] = None
    exit_time_msc: Optional[int] = None
    close_reason: Optional[CloseReason] = None
    pnl_pts: Optional[float] = None
    pnl_usd: Optional[float] = None
