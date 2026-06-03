"""GriffPositionManager — track Griff-strategy positions and pending orders.

Owns the live state map:
  - `_positions` : position_id → GriffOpenPosition (currently in market)
  - `_pendings`  : order_id    → GriffPendingOrder (waiting to fill / expire)

Per-bar maintenance (`maintain()` is called once per closed 1H bar per pair):
  1. Drive `SwingTracker.update(pair, bar)` so trail anchors stay fresh.
  2. For each open position on that pair, ask `TrailingStopLoss` for a new
     SL; if changed, call `order_router.modify_sl(position, new_sl)` and
     replace the cached position with the updated copy.
  3. Detect bot-side SL hit: if the bar's high/low crossed the position SL,
     mark it closed (real broker will have done the same — this is
     defensive bookkeeping).
  4. For each pending order, if `now_msc >= expiry_msc` fire the bot-side
     cancel (the hybrid expiry's bot leg).

Spread-hour widening is applied by the trailing-SL module internally
(it consults the bar timestamp) — manager just passes through.

This module does NOT poll MT5 for fills. The live engine is responsible
for invoking `on_pending_filled(order_id, fill_price, mt5_position_ticket,
fill_msc)` when it detects (via MT5 polling) that a pending got filled —
that promotes the pending into an open position.

Hinglish: pending orders aur open positions ka bookkeeper. Har bar pe
maintenance — trailing SL update, SL hit detection, expiry cancel. Fill
detection bot ka kaam nahi; live engine MT5 query karke bata deta hai.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from data.bar_aggregator import Bar
from execution.order_router import (
    GriffOpenPosition,
    GriffOrderRouter,
    GriffPendingOrder,
)
from execution.order import Side
from execution.position import Position, PositionState
from risk.trailing_sl import TrailingStopLoss
from strategy.patterns.base import Direction
from strategy.swing_tracker import SwingTracker
from utils.logger import logger


def _direction_to_side(d: Direction) -> Side:
    return Side.BUY if d == Direction.BUY else Side.SELL


def _legacy_position(p: GriffOpenPosition) -> Position:
    """Adapter so TrailingStopLoss (which expects execution.position.Position)
    can consume Griff positions without TrailingStopLoss being modified.
    `max_hold_until_msc` is unused by trailing_sl so we pass 0.
    """
    return Position(
        position_id=p.position_id,
        side=_direction_to_side(p.side),
        lots=p.lots,
        entry_price=p.entry_price,
        entry_time_msc=p.opened_msc,
        sl_price=p.sl_price,
        tp_price=p.tp_price,
        max_hold_until_msc=0,
        state=PositionState.OPEN,
        signal_type=p.pattern_name,
        session=None,
    )


@dataclass(frozen=True)
class MaintenanceReport:
    """Result of a single maintain() call. Frozen — pure observation."""
    pair: str
    bar_close_msc: int
    sl_updates: Tuple[Tuple[str, float], ...]   # (position_id, new_sl)
    closed_positions: Tuple[str, ...]            # position_id list
    cancelled_pendings: Tuple[str, ...]          # order_id list


class GriffPositionManager:
    def __init__(
        self,
        router: GriffOrderRouter,
        swing_tracker: SwingTracker,
        trailing_sl: TrailingStopLoss,
    ) -> None:
        self._router = router
        self._tracker = swing_tracker
        self._trail = trailing_sl
        self._positions: Dict[str, GriffOpenPosition] = {}
        self._pendings: Dict[str, GriffPendingOrder] = {}

    # ----------------------------------------------------------- registration

    def register_position(self, p: GriffOpenPosition) -> None:
        self._positions[p.position_id] = p
        logger.info(
            f"PROPX position registered id={p.position_id[:8]} {p.symbol} "
            f"{p.side.value} entry={p.entry_price} sl={p.sl_price}"
        )

    def register_pending(self, o: GriffPendingOrder) -> None:
        self._pendings[o.order_id] = o
        logger.info(
            f"PROPX pending registered id={o.order_id[:8]} {o.symbol} "
            f"{o.side.value} @{o.pending_price} exp={o.expiry_msc}"
        )

    def on_pending_filled(
        self, order_id: str, *, fill_price: float, mt5_position_ticket: int,
        fill_msc: int,
    ) -> Optional[GriffOpenPosition]:
        """Promote a pending order into an open position. Called by the live
        engine when MT5 polling shows the pending became a position."""
        order = self._pendings.pop(order_id, None)
        if order is None:
            logger.warning(f"on_pending_filled: unknown order_id {order_id}")
            return None
        pos = GriffOpenPosition(
            position_id=uuid.uuid4().hex,
            mt5_ticket=mt5_position_ticket,
            symbol=order.symbol,
            side=order.side,
            lots=order.lots,
            entry_price=fill_price,
            sl_price=order.sl_price,
            tp_price=order.tp_price,
            opened_msc=fill_msc,
            signal_id=order.signal_id,
            pattern_name=order.pattern_name,
        )
        self._positions[pos.position_id] = pos
        logger.info(
            f"PROPX pending → position id={pos.position_id[:8]} "
            f"{pos.symbol} fill={fill_price}"
        )
        return pos

    # ------------------------------------------------------------- accessors

    @property
    def open_positions(self) -> Tuple[GriffOpenPosition, ...]:
        return tuple(self._positions.values())

    @property
    def pending_orders(self) -> Tuple[GriffPendingOrder, ...]:
        return tuple(self._pendings.values())

    def positions_for(self, pair: str) -> Tuple[GriffOpenPosition, ...]:
        return tuple(p for p in self._positions.values() if p.symbol == pair)

    def pendings_for(self, pair: str) -> Tuple[GriffPendingOrder, ...]:
        return tuple(o for o in self._pendings.values() if o.symbol == pair)

    # -------------------------------------------------------- maintain (per-bar)

    async def maintain(
        self, pair: str, bar: Bar, *, now_msc: int,
    ) -> MaintenanceReport:
        """Per-bar maintenance for ONE pair.

        Caller must invoke this for each (pair, latest_closed_bar) tuple
        after the bar closes. Returns a frozen report enumerating SL
        updates, SL-hit closes, and pending-order cancellations triggered
        by this bar.
        """
        # 1) Swing tracker gets the new bar BEFORE trail does — its swing
        #    history is what the trail anchors to.
        self._tracker.update(pair, bar)
        now_dt = datetime.fromtimestamp(now_msc / 1000.0, tz=timezone.utc)

        sl_updates: list[tuple[str, float]] = []
        closed: list[str] = []
        cancelled: list[str] = []

        # 2) Iterate positions on this pair.
        for pos in list(self._positions.values()):
            if pos.symbol != pair:
                continue
            legacy = _legacy_position(pos)
            new_sl = self._trail.update(legacy, bar, now_dt)
            if new_sl is not None:
                ok = await self._router.modify_sl(pos, new_sl)
                if ok:
                    updated = _replace_sl(pos, new_sl)
                    self._positions[pos.position_id] = updated
                    sl_updates.append((pos.position_id, new_sl))
                    pos = updated
                else:
                    logger.warning(
                        f"PROPX modify_sl failed pos={pos.position_id[:8]}"
                    )
            # SL hit detection on the bar that just closed.
            if _sl_hit(pos, bar):
                # The broker has SL set MT5-side so it already closed; we just
                # remove from our map and emit the event.
                del self._positions[pos.position_id]
                closed.append(pos.position_id)
                logger.info(
                    f"PROPX SL hit pos={pos.position_id[:8]} "
                    f"bar=[{bar.low}, {bar.high}] sl={pos.sl_price}"
                )

        # 3) Expire pending orders past their expiry (hybrid leg: bot side).
        for order in list(self._pendings.values()):
            if order.symbol != pair:
                continue
            if now_msc >= order.expiry_msc:
                ok = await self._router.cancel_pending(order)
                if ok:
                    self._pendings.pop(order.order_id, None)
                    cancelled.append(order.order_id)

        return MaintenanceReport(
            pair=pair, bar_close_msc=bar.time_msc,
            sl_updates=tuple(sl_updates),
            closed_positions=tuple(closed),
            cancelled_pendings=tuple(cancelled),
        )

    # ------------------------------------------------------- manual removal

    def forget_position(self, position_id: str) -> Optional[GriffOpenPosition]:
        return self._positions.pop(position_id, None)

    def forget_pending(self, order_id: str) -> Optional[GriffPendingOrder]:
        return self._pendings.pop(order_id, None)


def _replace_sl(p: GriffOpenPosition, new_sl: float) -> GriffOpenPosition:
    """Return a frozen copy of `p` with the SL replaced."""
    return GriffOpenPosition(
        position_id=p.position_id, mt5_ticket=p.mt5_ticket, symbol=p.symbol,
        side=p.side, lots=p.lots, entry_price=p.entry_price,
        sl_price=new_sl, tp_price=p.tp_price,
        opened_msc=p.opened_msc, signal_id=p.signal_id,
        pattern_name=p.pattern_name,
    )


def _sl_hit(pos: GriffOpenPosition, bar: Bar) -> bool:
    if pos.side == Direction.BUY:
        return bar.low <= pos.sl_price
    return bar.high >= pos.sl_price
