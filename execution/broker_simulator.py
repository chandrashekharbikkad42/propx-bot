"""Paper broker. Realistic fill + exit simulation against live tick stream.

Slippage model: 50% of the current spread paid both entering AND exiting,
debited from the more-adverse side of the touch (BUY entry pays above ask;
SELL entry pays below bid; mirror on exits). All prices flow through the
contract_size multiplier — for XAUUSD that's 100 oz / lot.

The broker holds NO state — it inspects a Position and the latest Tick and
returns either a refreshed CLOSED Position or None. The RiskEngine owns the
list of open positions.
"""

from __future__ import annotations
import uuid
from typing import Optional

from data.tick_collector import Tick
from execution.order import OrderIntent, Side
from execution.position import CloseReason, Position, PositionState


class PaperBroker:
    POINT_VALUE = 0.01  # XAUUSD: 1 pt = $0.01

    def __init__(self, slippage_pct: float = 0.5, contract_size: int = 100) -> None:
        self._slippage_pct = slippage_pct
        self._contract_size = contract_size

    # ----------------------------------------------------------------- entry

    def fill_market_order(self, intent: OrderIntent, current_tick: Tick) -> Position:
        spread = current_tick.ask - current_tick.bid
        slip = spread * self._slippage_pct
        if intent.side == Side.BUY:
            fill_price = current_tick.ask + slip
        else:
            fill_price = current_tick.bid - slip

        # Phase 7B — anchor SL/TP to the actual fill price (post-slippage)
        # rather than the intent's `intended_price` (the pre-slip touch).
        # When `intent.sl_pts`/`tp_pts` are zero (legacy callers / direct
        # tests) we fall back to the intent's absolute prices so existing
        # tests stay valid without per-call rewiring.
        if intent.sl_pts > 0 and intent.tp_pts > 0:
            if intent.side == Side.BUY:
                sl_price = fill_price - intent.sl_pts * self.POINT_VALUE
                tp_price = fill_price + intent.tp_pts * self.POINT_VALUE
            else:
                sl_price = fill_price + intent.sl_pts * self.POINT_VALUE
                tp_price = fill_price - intent.tp_pts * self.POINT_VALUE
        else:
            sl_price = intent.sl_price
            tp_price = intent.tp_price

        return Position(
            position_id=uuid.uuid4().hex,
            side=intent.side,
            lots=intent.lots,
            entry_price=fill_price,
            entry_time_msc=current_tick.time_msc,
            sl_price=sl_price,
            tp_price=tp_price,
            max_hold_until_msc=intent.max_hold_until_msc,
            state=PositionState.OPEN,
            signal_type=intent.signal_type.value,
            session=intent.session.value,
        )

    # ------------------------------------------------------------------ exit

    def check_position_exit(
        self, position: Position, current_tick: Tick
    ) -> Optional[Position]:
        if position.state != PositionState.OPEN:
            return None

        # 1. Time exit takes precedence — if the bar has elapsed we close
        #    at the mid touch regardless of price level.
        if current_tick.time_msc >= position.max_hold_until_msc:
            exit_price = self._exit_fill(position.side, current_tick)
            return self._close(position, exit_price, current_tick.time_msc, CloseReason.TIME_EXIT)

        # 2. SL / TP — evaluate against the side-appropriate touch.
        #    BUY exits at the bid (we sell to close), so SL/TP triggers
        #    when bid <= SL or bid >= TP. SELL exits at the ask.
        if position.side == Side.BUY:
            if current_tick.bid <= position.sl_price:
                exit_price = self._exit_fill(position.side, current_tick)
                return self._close(position, exit_price, current_tick.time_msc, CloseReason.SL_HIT)
            if current_tick.bid >= position.tp_price:
                exit_price = self._exit_fill(position.side, current_tick)
                return self._close(position, exit_price, current_tick.time_msc, CloseReason.TP_HIT)
        else:
            if current_tick.ask >= position.sl_price:
                exit_price = self._exit_fill(position.side, current_tick)
                return self._close(position, exit_price, current_tick.time_msc, CloseReason.SL_HIT)
            if current_tick.ask <= position.tp_price:
                exit_price = self._exit_fill(position.side, current_tick)
                return self._close(position, exit_price, current_tick.time_msc, CloseReason.TP_HIT)

        return None

    def force_close(
        self, position: Position, current_tick: Tick, reason: CloseReason = CloseReason.EOD
    ) -> Position:
        exit_price = self._exit_fill(position.side, current_tick)
        return self._close(position, exit_price, current_tick.time_msc, reason)

    # --------------------------------------------------------------- helpers

    def _exit_fill(self, side: Side, tick: Tick) -> float:
        spread = tick.ask - tick.bid
        slip = spread * self._slippage_pct
        # On exit BUY -> we sell at bid (minus slip); SELL -> we buy at ask (plus slip).
        if side == Side.BUY:
            return tick.bid - slip
        return tick.ask + slip

    def _close(
        self, position: Position, exit_price: float, exit_msc: int, reason: CloseReason
    ) -> Position:
        if position.side == Side.BUY:
            pnl_price = exit_price - position.entry_price
        else:
            pnl_price = position.entry_price - exit_price

        pnl_pts = pnl_price / self.POINT_VALUE
        pnl_usd = pnl_price * position.lots * self._contract_size

        return Position(
            position_id=position.position_id,
            side=position.side,
            lots=position.lots,
            entry_price=position.entry_price,
            entry_time_msc=position.entry_time_msc,
            sl_price=position.sl_price,
            tp_price=position.tp_price,
            max_hold_until_msc=position.max_hold_until_msc,
            state=PositionState.CLOSED,
            signal_type=position.signal_type,
            session=position.session,
            exit_price=exit_price,
            exit_time_msc=exit_msc,
            close_reason=reason,
            pnl_pts=pnl_pts,
            pnl_usd=pnl_usd,
        )
