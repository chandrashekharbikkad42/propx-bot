"""Multi-symbol order router for the Griff live engine (Phase 8D-Live).

Lives ALONGSIDE the existing `execution/live_broker.py` (single-symbol XAUUSD
scalping) — does NOT extend it. Two reasons:
  - existing OrderIntent is shaped for microstructure (max_hold_until_msc,
    SignalType) and doesn't fit pattern-detector output cleanly.
  - existing LiveBroker is single-symbol; Griff scans 6 pairs concurrently.

Responsibilities:
  - Place market orders for FLAG signals (entry at bar close).
  - Place pending STOP orders for CONTINUATION / REVERSAL signals
    (Buy Stop above pullback / Sell Stop below).
  - Place pending LIMIT orders for COMBO signals (price must retrace to
    the limit level).
  - HYBRID order expiry: every pending order is submitted with
    `ORDER_TIME_SPECIFIED` so MT5 auto-cancels at `expiry_msc`, AND the
    bot's live engine calls `cancel_pending(ticket)` at the next bar close
    as a belt-and-braces guarantee.
  - Retry on transient MT5 errors (same retcode list as LiveBroker).
  - DRY_RUN mode: returns a synthetic Position/PendingOrder with
    `ticket=-1` and writes a structured log line, never touches MT5.

Position bookkeeping (open positions + pending orders) lives in
`GriffPositionManager` — this module is order-issuing only.

Hinglish: yeh router signals ko MT5 calls me convert karta hai. Pending
orders hybrid expiry use karte hain — broker-side bhi expire hota hai
aur bot bhi next bar pe cancel call deta hai. DRY_RUN me bas log.
"""

from __future__ import annotations
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional

try:  # MT5 may not be importable on CI / non-Windows; tests patch it.
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None  # type: ignore[assignment]

from strategy.patterns.base import Direction, PatternSignal
from utils.logger import logger


MAGIC = 786544  # distinct from xau_hft (786543) so analytics can split
COMMENT = "propX"
DEFAULT_DEVIATION_POINTS = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = (0.5, 1.5, 3.0)
# Idempotency window: if the same (symbol, side, bar_time_msc) tuple is
# submitted twice inside this many ms, the second submission raises a
# duplicate error rather than reaching MT5.
DEDUP_WINDOW_MS = 60_000

# MT5 retcodes considered transient — others propagate as errors.
_TRANSIENT_RETCODES = frozenset({
    10004,  # TRADE_RETCODE_REQUOTE
    10006,  # TRADE_RETCODE_REJECT
    10021,  # TRADE_RETCODE_PRICE_OFF
    10018,  # TRADE_RETCODE_MARKET_CLOSED
    10031,  # TRADE_RETCODE_CONNECTION
})


class GriffOrderError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GriffOpenPosition:
    """A live or paper-open position. Frozen — produced on fill."""
    position_id: str
    mt5_ticket: int
    symbol: str
    side: Direction
    lots: float
    entry_price: float
    sl_price: float
    tp_price: float
    opened_msc: int
    signal_id: str
    pattern_name: str


@dataclass(frozen=True)
class GriffPendingOrder:
    """A pending stop / limit order, awaiting trigger."""
    order_id: str
    mt5_ticket: int
    symbol: str
    side: Direction
    lots: float
    pending_price: float
    sl_price: float
    tp_price: float
    expiry_msc: int          # hybrid: also passed to MT5 as ORDER_TIME_SPECIFIED
    signal_id: str
    pattern_name: str
    is_limit: bool           # True for Combo LIMIT; False for STOP


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class GriffOrderRouter:
    """Issue Griff orders via MT5. Multi-symbol, supports market + pending."""

    def __init__(
        self,
        *,
        dry_run: bool = True,
        deviation_points: int = DEFAULT_DEVIATION_POINTS,
        magic: int = MAGIC,
    ) -> None:
        self._dry_run = dry_run
        self._deviation = deviation_points
        self._magic = magic
        self._recent_submissions: dict[tuple, int] = {}

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    # ----------------------------------------------------- public order verbs

    async def place_market(
        self,
        signal: PatternSignal,
        lots: float,
        *,
        ask: float,
        bid: float,
        now_msc: int,
    ) -> GriffOpenPosition:
        """Open a market position for a FLAG signal."""
        if mt5 is None and not self._dry_run:  # pragma: no cover
            raise GriffOrderError("MetaTrader5 not importable but dry_run=False")
        side = signal.direction
        price = ask if side == Direction.BUY else bid

        self._check_and_record_submission(signal, now_msc)

        if self._dry_run:
            return self._dry_market(signal, lots, price, now_msc)

        order_type = mt5.ORDER_TYPE_BUY if side == Direction.BUY else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": float(lots),
            "type": order_type,
            "price": float(price),
            "sl": float(signal.sl),
            "tp": float(signal.tp),
            "deviation": self._deviation,
            "magic": self._magic,
            "comment": f"{COMMENT}:{signal.pattern_name}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = await self._send_with_retry(request, label="market")
        ticket = _ticket_from_result(result)
        fill_price = float(getattr(result, "price", price) or 0.0)
        # Some brokers return result.price == 0.0 on a market DEAL — the actual
        # fill price only materialises on the resulting position, not on the
        # send result. Recover it from positions_get so downstream PnL / SL
        # math is anchored to the real entry instead of a bogus 0.0. Falls back
        # to the requested quote if MT5 has nothing usable for us.
        if fill_price <= 0.0:
            resolved = await self._fill_price_from_position(ticket, signal.symbol)
            fill_price = resolved if resolved > 0.0 else float(price)
            logger.warning(
                f"PROPX MARKET result.price=0.0 {signal.symbol} {side.value}; "
                f"recovered entry={fill_price} (ticket={ticket})"
            )
        # Bug-fix: record the actual filled volume (not the requested lots) so
        # downstream position bookkeeping reflects the broker's true holding.
        # Partial fills (volume < requested) were silently mis-bookkept before.
        filled_volume = float(getattr(result, "volume", 0.0) or 0.0) or float(lots)
        position_id = uuid.uuid4().hex
        logger.info(
            f"PROPX MARKET fill ticket={ticket} {signal.symbol} {side.value} "
            f"lots={filled_volume:g} (req={lots:g}) @{fill_price}"
        )
        return GriffOpenPosition(
            position_id=position_id, mt5_ticket=ticket, symbol=signal.symbol,
            side=side, lots=filled_volume, entry_price=fill_price,
            sl_price=signal.sl, tp_price=signal.tp,
            opened_msc=now_msc, signal_id=signal.pattern_name + ":" + position_id[:8],
            pattern_name=signal.pattern_name,
        )

    async def _fill_price_from_position(self, ticket: int, symbol: str) -> float:
        """Resolve a market fill price from MT5's open positions.

        Used when `order_send` returns result.price == 0.0. Tries the exact
        position ticket first, then any open position on the symbol. Returns
        0.0 if MT5 is unavailable or has no usable price_open so the caller
        can fall back to the requested quote.
        """
        if mt5 is None:  # pragma: no cover — dry_run never reaches here
            return 0.0
        try:
            positions = await asyncio.to_thread(mt5.positions_get, ticket=ticket)
            if not positions:
                positions = await asyncio.to_thread(
                    mt5.positions_get, symbol=symbol,
                )
        except Exception as exc:  # noqa: BLE001 — must not break trading loop
            logger.warning(f"positions_get failed resolving fill price: {exc}")
            return 0.0
        for p in positions or ():
            price_open = float(getattr(p, "price_open", 0.0) or 0.0)
            if price_open > 0.0:
                return price_open
        return 0.0

    async def place_pending_stop(
        self,
        signal: PatternSignal,
        lots: float,
        *,
        expiry_msc: int,
        now_msc: int,
    ) -> GriffPendingOrder:
        """Buy Stop (BUY) or Sell Stop (SELL) at `signal.entry`. Used for
        Continuation and Reversal."""
        return await self._place_pending(
            signal, lots, expiry_msc=expiry_msc, now_msc=now_msc, is_limit=False,
        )

    async def place_pending_limit(
        self,
        signal: PatternSignal,
        lots: float,
        *,
        expiry_msc: int,
        now_msc: int,
    ) -> GriffPendingOrder:
        """Buy Limit (BUY) or Sell Limit (SELL) at `signal.entry`. Used for
        Combo where price must retrace to a level."""
        return await self._place_pending(
            signal, lots, expiry_msc=expiry_msc, now_msc=now_msc, is_limit=True,
        )

    async def cancel_pending(self, order: GriffPendingOrder) -> bool:
        """Bot-side leg of the hybrid expiry. Returns True if cancelled or
        already gone, False on hard MT5 failure (caller decides whether to
        treat that as fatal)."""
        if self._dry_run:
            logger.info(f"PROPX DRY cancel ticket={order.mt5_ticket}")
            return True

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(order.mt5_ticket),
        }
        result = await asyncio.to_thread(mt5.order_send, request)
        retcode = int(getattr(result, "retcode", -1))
        if retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"PROPX pending cancelled ticket={order.mt5_ticket}")
            return True
        # "Order not found" (10027) means MT5 already removed it — also OK.
        if retcode == 10027:
            return True
        logger.warning(
            f"PROPX pending cancel failed ticket={order.mt5_ticket} retcode={retcode}"
        )
        return False

    async def close_position(
        self, position: GriffOpenPosition, *, bid: float, ask: float, now_msc: int,
    ) -> float:
        """Market-close an open position. Returns the exit price."""
        if self._dry_run:
            price = bid if position.side == Direction.BUY else ask
            logger.info(
                f"PROPX DRY close ticket={position.mt5_ticket} @{price}"
            )
            return price

        close_type = (
            mt5.ORDER_TYPE_SELL if position.side == Direction.BUY else mt5.ORDER_TYPE_BUY
        )
        price = bid if position.side == Direction.BUY else ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": float(position.lots),
            "type": close_type,
            "position": int(position.mt5_ticket),
            "price": float(price),
            "deviation": self._deviation,
            "magic": self._magic,
            "comment": f"{COMMENT}:close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = await self._send_with_retry(request, label="close")
        return float(getattr(result, "price", price))

    async def modify_sl(
        self, position: GriffOpenPosition, new_sl: float,
    ) -> bool:
        """Adjust an open position's SL (used by the trailing-SL loop).
        Returns True on success or no-op (already at that SL)."""
        if self._dry_run:
            logger.info(
                f"PROPX DRY modify_sl ticket={position.mt5_ticket} new_sl={new_sl}"
            )
            return True

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(position.mt5_ticket),
            "sl": float(new_sl),
            "tp": float(position.tp_price),
            "magic": self._magic,
        }
        result = await asyncio.to_thread(mt5.order_send, request)
        retcode = int(getattr(result, "retcode", -1))
        return retcode == mt5.TRADE_RETCODE_DONE

    # ------------------------------------------------------------ internals

    def _check_and_record_submission(
        self, signal: PatternSignal, now_msc: int,
    ) -> None:
        """Idempotency guard. Raises GriffOrderError if the same signal
        (symbol, side, bar_time_msc) was submitted within DEDUP_WINDOW_MS.
        Reason: a single signal that races the scan loop (e.g. retry after
        timeout) could otherwise produce two MT5 orders for the same intent.
        """
        key = (
            signal.pattern_name, signal.symbol, signal.direction,
            int(signal.bar_time_msc),
        )
        prev = self._recent_submissions.get(key)
        if prev is not None and (now_msc - prev) < DEDUP_WINDOW_MS:
            raise GriffOrderError(
                f"duplicate submission rejected {signal.symbol} "
                f"{signal.direction.value} bar={signal.bar_time_msc}"
            )
        self._recent_submissions[key] = now_msc
        # Bounded cleanup so the dict can't grow unbounded over a long session.
        if len(self._recent_submissions) > 1024:
            cutoff = now_msc - DEDUP_WINDOW_MS
            self._recent_submissions = {
                k: v for k, v in self._recent_submissions.items() if v >= cutoff
            }

    async def _place_pending(
        self,
        signal: PatternSignal,
        lots: float,
        *,
        expiry_msc: int,
        now_msc: int,
        is_limit: bool,
    ) -> GriffPendingOrder:
        side = signal.direction
        if self._dry_run:
            return self._dry_pending(signal, lots, expiry_msc, now_msc, is_limit)

        if is_limit:
            order_type = (
                mt5.ORDER_TYPE_BUY_LIMIT if side == Direction.BUY
                else mt5.ORDER_TYPE_SELL_LIMIT
            )
        else:
            order_type = (
                mt5.ORDER_TYPE_BUY_STOP if side == Direction.BUY
                else mt5.ORDER_TYPE_SELL_STOP
            )

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": signal.symbol,
            "volume": float(lots),
            "type": order_type,
            "price": float(signal.entry),
            "sl": float(signal.sl),
            "tp": float(signal.tp),
            "deviation": self._deviation,
            "magic": self._magic,
            "comment": f"{COMMENT}:{signal.pattern_name}",
            # Hybrid expiry leg #1: broker-side ORDER_TIME_SPECIFIED.
            "type_time": mt5.ORDER_TIME_SPECIFIED,
            "expiration": int(expiry_msc / 1000),  # MT5 uses seconds, not ms.
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        result = await self._send_with_retry(request, label="pending")
        ticket = _ticket_from_result(result)
        order_id = uuid.uuid4().hex
        logger.info(
            f"PROPX PENDING placed ticket={ticket} {signal.symbol} {side.value} "
            f"{'LIMIT' if is_limit else 'STOP'} @{signal.entry} exp_msc={expiry_msc}"
        )
        return GriffPendingOrder(
            order_id=order_id, mt5_ticket=ticket, symbol=signal.symbol,
            side=side, lots=lots, pending_price=signal.entry,
            sl_price=signal.sl, tp_price=signal.tp,
            expiry_msc=expiry_msc,
            signal_id=signal.pattern_name + ":" + order_id[:8],
            pattern_name=signal.pattern_name, is_limit=is_limit,
        )

    async def _send_with_retry(self, request: dict, *, label: str):
        last_result = None
        for attempt in range(MAX_RETRIES):
            result = await asyncio.to_thread(mt5.order_send, request)
            retcode = int(getattr(result, "retcode", -1))
            if retcode == mt5.TRADE_RETCODE_DONE:
                return result
            last_result = result
            if retcode not in _TRANSIENT_RETCODES:
                raise GriffOrderError(
                    f"{label} permanent reject retcode={retcode} "
                    f"comment={getattr(result, 'comment', '')}"
                )
            await asyncio.sleep(RETRY_BACKOFF_SEC[attempt])
        raise GriffOrderError(
            f"{label} exhausted retries; last retcode="
            f"{int(getattr(last_result, 'retcode', -1))}"
        )

    # ----- DRY_RUN synthesisers (no MT5 calls)

    def _dry_market(
        self, signal: PatternSignal, lots: float, price: float, now_msc: int,
    ) -> GriffOpenPosition:
        position_id = uuid.uuid4().hex
        logger.info(
            f"PROPX DRY market {signal.symbol} {signal.direction.value} "
            f"lots={lots:g} @{price}"
        )
        return GriffOpenPosition(
            position_id=position_id, mt5_ticket=-1, symbol=signal.symbol,
            side=signal.direction, lots=lots, entry_price=price,
            sl_price=signal.sl, tp_price=signal.tp,
            opened_msc=now_msc,
            signal_id=signal.pattern_name + ":" + position_id[:8],
            pattern_name=signal.pattern_name,
        )

    def _dry_pending(
        self, signal: PatternSignal, lots: float, expiry_msc: int,
        now_msc: int, is_limit: bool,
    ) -> GriffPendingOrder:
        order_id = uuid.uuid4().hex
        kind = "LIMIT" if is_limit else "STOP"
        logger.info(
            f"PROPX DRY pending {kind} {signal.symbol} {signal.direction.value} "
            f"lots={lots:g} @{signal.entry} exp_msc={expiry_msc}"
        )
        return GriffPendingOrder(
            order_id=order_id, mt5_ticket=-1, symbol=signal.symbol,
            side=signal.direction, lots=lots, pending_price=signal.entry,
            sl_price=signal.sl, tp_price=signal.tp,
            expiry_msc=expiry_msc,
            signal_id=signal.pattern_name + ":" + order_id[:8],
            pattern_name=signal.pattern_name, is_limit=is_limit,
        )


def _ticket_from_result(result) -> int:
    """MT5 puts the new ticket in `.order` for pending and `.deal` for fills.
    Try both before giving up."""
    return int(getattr(result, "order", 0)) or int(getattr(result, "deal", 0))
