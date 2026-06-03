"""Order intent value object. Consumed by PaperBroker / LiveBroker.

Phase 5 cleanup note: `SignalType` used to live in `strategy/signals/base.py`,
but that module was deleted along with the legacy tick-microstructure
strategy. The enum is inlined here because `OrderIntent` is the only
load-bearing place where the tag is still consumed.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from utils.session import SessionLabel


class SignalType(str, Enum):
    """Origin tag for an OrderIntent. Carried for analytics / logging only;
    no behavioural branch in the live V5 path depends on the value.
    """
    SWEEP = "SWEEP"
    MOMENTUM = "MOMENTUM"
    REJECTION = "REJECTION"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class OrderIntent:
    signal_id: str
    side: Side
    lots: float
    intended_price: float
    sl_price: float
    tp_price: float
    max_hold_until_msc: int
    signal_type: SignalType
    session: SessionLabel
    # Phase 7B — distances in POINTS used by brokers that can anchor the
    # final SL/TP to the actual fill price (post-slippage). PaperBroker uses
    # these; LiveBroker still submits sl_price/tp_price as the pre-fill
    # anchor since MT5 needs absolute prices in the request payload.
    sl_pts: float = 0.0
    tp_pts: float = 0.0
