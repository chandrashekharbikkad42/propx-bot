"""Factory helpers for OrderIntent + PatternSignal."""

from __future__ import annotations
from typing import Optional

from execution.order import OrderIntent, Side, SignalType
from strategy.patterns.base import Direction, Grade, PatternSignal
from utils.session import SessionLabel


def make_intent(
    *,
    signal_id: str = "sig-1",
    side: Side = Side.BUY,
    lots: float = 0.10,
    intended_price: float = 1.10000,
    sl_price: float = 1.09000,
    tp_price: float = 1.12000,
    max_hold_until_msc: int = 0,
    signal_type: SignalType = SignalType.SWEEP,
    session: SessionLabel = SessionLabel.LONDON,
    sl_pts: float = 0.0,
    tp_pts: float = 0.0,
) -> OrderIntent:
    return OrderIntent(
        signal_id=signal_id,
        side=side,
        lots=lots,
        intended_price=intended_price,
        sl_price=sl_price,
        tp_price=tp_price,
        max_hold_until_msc=max_hold_until_msc,
        signal_type=signal_type,
        session=session,
        sl_pts=sl_pts,
        tp_pts=tp_pts,
    )


def make_signal(
    *,
    pattern_name: str = "ASIAN_SWEEP",
    symbol: str = "EURUSD",
    direction: Direction = Direction.BUY,
    entry: float = 1.10000,
    sl: float = 1.09800,
    tp: float = 1.10400,
    confidence: float = 0.85,
    grade: Grade = Grade.A,
    confluences_met: tuple = ("htf_bullish",),
    bar_time_msc: int = 1_700_000_000_000,
) -> PatternSignal:
    return PatternSignal(
        pattern_name=pattern_name,
        symbol=symbol,
        direction=direction,
        entry=entry,
        sl=sl,
        tp=tp,
        confidence=confidence,
        grade=grade,
        confluences_met=confluences_met,
        bar_time_msc=bar_time_msc,
    )


def make_signal_sell(
    *,
    pattern_name: str = "ASIAN_SWEEP",
    symbol: str = "EURUSD",
    entry: float = 1.10000,
    sl: float = 1.10200,
    tp: float = 1.09600,
    grade: Grade = Grade.A,
    confidence: float = 0.85,
    bar_time_msc: int = 1_700_000_000_000,
) -> PatternSignal:
    return PatternSignal(
        pattern_name=pattern_name,
        symbol=symbol,
        direction=Direction.SELL,
        entry=entry,
        sl=sl,
        tp=tp,
        confidence=confidence,
        grade=grade,
        confluences_met=("htf_bearish",),
        bar_time_msc=bar_time_msc,
    )
