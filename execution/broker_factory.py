"""Broker selection based on `settings.execution_mode`.

Both brokers are exposed through an `AsyncBroker` protocol so callers (bot.py
strategy loop) don't have to branch on mode. PaperBroker's sync methods are
wrapped here in trivial async shims.
"""

from __future__ import annotations
from typing import Optional, Protocol

from config.settings import Settings
from data.tick_collector import Tick
from execution.broker_simulator import PaperBroker
from execution.order import OrderIntent
from execution.position import CloseReason, Position
from utils.logger import logger


class AsyncBroker(Protocol):
    async def fill_market_order(
        self, intent: OrderIntent, current_tick: Tick
    ) -> Position: ...

    async def check_position_exit(
        self, position: Position, current_tick: Tick
    ) -> Optional[Position]: ...

    async def force_close(
        self,
        position: Position,
        current_tick: Tick,
        reason: CloseReason = CloseReason.EOD,
    ) -> Position: ...


class _PaperBrokerAsyncShim:
    """Wraps the sync PaperBroker so it conforms to AsyncBroker."""

    def __init__(self, paper: PaperBroker) -> None:
        self._paper = paper

    async def fill_market_order(
        self, intent: OrderIntent, current_tick: Tick
    ) -> Position:
        return self._paper.fill_market_order(intent, current_tick)

    async def check_position_exit(
        self, position: Position, current_tick: Tick
    ) -> Optional[Position]:
        return self._paper.check_position_exit(position, current_tick)

    async def force_close(
        self,
        position: Position,
        current_tick: Tick,
        reason: CloseReason = CloseReason.EOD,
    ) -> Position:
        return self._paper.force_close(position, current_tick, reason)


def get_broker(settings: Settings) -> AsyncBroker:
    """Return the configured broker.

    PAPER (default): wrapped PaperBroker — safe, no MT5 orders.
    REAL: LiveBroker — places real MT5 market orders. Requires the MT5
    terminal to be connected before any call.
    """
    mode = settings.execution_mode
    if mode == "PAPER":
        logger.info("Broker: PaperBroker (PAPER mode)")
        return _PaperBrokerAsyncShim(PaperBroker())
    if mode == "REAL":
        # Late import — keeps mt5 import out of the paper-only path.
        from execution.live_broker import LiveBroker
        logger.warning(
            f"Broker: LiveBroker (REAL mode) symbol={settings.symbol} — "
            f"orders will be sent to {settings.broker_type}"
        )
        return LiveBroker(symbol=settings.symbol)
    raise ValueError(f"Unknown execution_mode: {mode!r}")
