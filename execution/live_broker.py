"""Live MT5 broker. Real order placement via MetaTrader5 API.

Mirrors `PaperBroker`'s interface (fill_market_order / check_position_exit /
force_close) so RiskEngine + bot.py work unchanged when EXECUTION_MODE=REAL.
All synchronous MT5 calls run inside `asyncio.to_thread` so the event loop
never blocks on a slow round-trip.

Retry policy: transient `RES_S_OK == 10009` is the only happy path; on retcodes
that look transient (requote, off-quotes, timeout, market-closed-momentarily)
we retry up to MAX_RETRIES with exponential backoff. Anything else surfaces.
"""

from __future__ import annotations
import asyncio
import time
import uuid
from typing import Optional

import MetaTrader5 as mt5

from data.tick_collector import Tick
from execution.order import OrderIntent, Side
from execution.position import CloseReason, Position, PositionState
from utils.logger import logger


MAX_RETRIES = 3
RETRY_BACKOFF_SEC = (0.5, 1.5, 3.0)
DEVIATION_POINTS = 20
MAGIC = 786543
COMMENT = "xau_hft"
POINT_VALUE = 0.01
CONTRACT_SIZE = 100

# MT5 retcodes considered transient — others propagate as errors.
_TRANSIENT_RETCODES = frozenset({
    10004,  # TRADE_RETCODE_REQUOTE
    10006,  # TRADE_RETCODE_REJECT
    10021,  # TRADE_RETCODE_PRICE_OFF
    10018,  # TRADE_RETCODE_MARKET_CLOSED
    10031,  # TRADE_RETCODE_CONNECTION
})


class LiveBrokerError(RuntimeError):
    pass


class LiveBroker:
    POINT_VALUE = POINT_VALUE

    def __init__(
        self,
        symbol: str,
        slippage_pct: float = 0.5,
        contract_size: int = CONTRACT_SIZE,
        deviation_points: int = DEVIATION_POINTS,
    ) -> None:
        self._symbol = symbol
        self._slippage_pct = slippage_pct
        self._contract_size = contract_size
        self._deviation = deviation_points
        # position_id -> mt5 ticket
        self._ticket_by_id: dict[str, int] = {}

    # ----------------------------------------------------------------- entry

    async def fill_market_order(
        self, intent: OrderIntent, current_tick: Tick
    ) -> Position:
        order_type = mt5.ORDER_TYPE_BUY if intent.side == Side.BUY else mt5.ORDER_TYPE_SELL
        price = current_tick.ask if intent.side == Side.BUY else current_tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self._symbol,
            "volume": float(intent.lots),
            "type": order_type,
            "price": price,
            "sl": float(intent.sl_price),
            "tp": float(intent.tp_price),
            "deviation": self._deviation,
            "magic": MAGIC,
            "comment": COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = await self._send_with_retry(request, "open")
        ticket = int(getattr(result, "order", 0)) or int(getattr(result, "deal", 0))
        fill_price = float(getattr(result, "price", price))

        position_id = uuid.uuid4().hex
        self._ticket_by_id[position_id] = ticket

        logger.info(
            f"LIVE order opened ticket={ticket} side={intent.side.value} "
            f"lots={intent.lots:g} price={fill_price:.2f}"
        )

        return Position(
            position_id=position_id,
            side=intent.side,
            lots=intent.lots,
            entry_price=fill_price,
            entry_time_msc=current_tick.time_msc,
            sl_price=intent.sl_price,
            tp_price=intent.tp_price,
            max_hold_until_msc=intent.max_hold_until_msc,
            state=PositionState.OPEN,
            signal_type=intent.signal_type.value,
            session=intent.session.value,
        )

    # ------------------------------------------------------------------ exit

    async def check_position_exit(
        self, position: Position, current_tick: Tick
    ) -> Optional[Position]:
        if position.state != PositionState.OPEN:
            return None

        ticket = self._ticket_by_id.get(position.position_id)
        if ticket is None:
            return None

        # Time-based exit takes precedence.
        if current_tick.time_msc >= position.max_hold_until_msc:
            return await self._close_position(
                position, current_tick, CloseReason.TIME_EXIT
            )

        # Poll MT5 for position status. If it has disappeared from the live
        # list, the broker closed it (SL/TP). Pull the deal history to find
        # the actual exit price + reason.
        positions = await asyncio.to_thread(mt5.positions_get, ticket=ticket)
        if positions:
            return None  # still open

        return await self._reconcile_broker_close(position, current_tick)

    async def force_close(
        self,
        position: Position,
        current_tick: Tick,
        reason: CloseReason = CloseReason.EOD,
    ) -> Position:
        return await self._close_position(position, current_tick, reason)

    # --------------------------------------------------------------- helpers

    async def _close_position(
        self, position: Position, current_tick: Tick, reason: CloseReason
    ) -> Position:
        ticket = self._ticket_by_id.get(position.position_id)
        if ticket is None:
            raise LiveBrokerError(f"unknown position_id {position.position_id}")

        close_type = (
            mt5.ORDER_TYPE_SELL if position.side == Side.BUY else mt5.ORDER_TYPE_BUY
        )
        price = current_tick.bid if position.side == Side.BUY else current_tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self._symbol,
            "volume": float(position.lots),
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": self._deviation,
            "magic": MAGIC,
            "comment": f"{COMMENT}:{reason.value}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = await self._send_with_retry(request, "close")
        exit_price = float(getattr(result, "price", price))
        return self._build_closed(position, exit_price, current_tick.time_msc, reason)

    async def _reconcile_broker_close(
        self, position: Position, current_tick: Tick
    ) -> Position:
        """Position vanished from MT5 — broker closed it (SL/TP). Reconcile."""
        ticket = self._ticket_by_id.get(position.position_id)
        deals = await asyncio.to_thread(
            mt5.history_deals_get, position=ticket
        )
        if not deals:
            # No history yet — fall back to a synthetic close at current touch.
            exit_price = (
                current_tick.bid if position.side == Side.BUY else current_tick.ask
            )
            return self._build_closed(
                position, exit_price, current_tick.time_msc, CloseReason.MANUAL
            )

        last_deal = deals[-1]
        exit_price = float(getattr(last_deal, "price", current_tick.bid))

        # Guess reason from touch.
        if position.side == Side.BUY:
            reason = (
                CloseReason.SL_HIT if exit_price <= position.sl_price
                else CloseReason.TP_HIT if exit_price >= position.tp_price
                else CloseReason.MANUAL
            )
        else:
            reason = (
                CloseReason.SL_HIT if exit_price >= position.sl_price
                else CloseReason.TP_HIT if exit_price <= position.tp_price
                else CloseReason.MANUAL
            )

        return self._build_closed(position, exit_price, current_tick.time_msc, reason)

    def _build_closed(
        self,
        position: Position,
        exit_price: float,
        exit_msc: int,
        reason: CloseReason,
    ) -> Position:
        if position.side == Side.BUY:
            pnl_price = exit_price - position.entry_price
        else:
            pnl_price = position.entry_price - exit_price
        pnl_pts = pnl_price / POINT_VALUE
        pnl_usd = pnl_price * position.lots * self._contract_size

        self._ticket_by_id.pop(position.position_id, None)

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

    async def _send_with_retry(self, request: dict, label: str):
        last_retcode = None
        last_comment = None
        for attempt in range(MAX_RETRIES):
            result = await asyncio.to_thread(mt5.order_send, request)
            if result is None:
                err = mt5.last_error()
                logger.warning(f"MT5 {label} order_send None (attempt {attempt+1}): {err}")
                last_retcode = -1
                last_comment = str(err)
            else:
                retcode = int(getattr(result, "retcode", -1))
                last_retcode = retcode
                last_comment = getattr(result, "comment", "")
                if retcode == mt5.TRADE_RETCODE_DONE:
                    return result
                if retcode not in _TRANSIENT_RETCODES:
                    raise LiveBrokerError(
                        f"MT5 {label} rejected retcode={retcode} comment={last_comment!r}"
                    )
                logger.warning(
                    f"MT5 {label} transient retcode={retcode} "
                    f"comment={last_comment!r} (attempt {attempt+1})"
                )
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BACKOFF_SEC[attempt])

        raise LiveBrokerError(
            f"MT5 {label} failed after {MAX_RETRIES} attempts "
            f"retcode={last_retcode} comment={last_comment!r}"
        )
