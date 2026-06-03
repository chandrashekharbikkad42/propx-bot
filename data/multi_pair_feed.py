"""Multi-pair bar feed. Manages one BarAggregator per symbol; emits bar-close events.

The feed is source-agnostic: callers drive it by calling `on_tick(symbol, tick)`
for ticks arriving from MT5, a replay engine, or test fixtures. The feed does
NOT own a tick producer — that decouples it from MT5 import paths and lets
live + backtest + tests share the same code.

Bar-close events:
  - One BarCloseEvent emitted per pair, per closed 1H bar.
  - Emitted in the order their boundaries are crossed (i.e., by the driving
    tick stream order; we do not reorder).
  - A single `on_tick` call returns 0 or 1 events for THAT pair only — but
    callers may receive batched events via `flush_all()` at end-of-stream.

Gap tolerance:
  - If a pair receives no ticks for several hours, no bars are emitted for
    those windows (silent skip). Downstream scanner inspects bar deltas to
    detect gaps.
  - If a pair is unknown to the feed, `on_tick` raises KeyError — caller
    should call `register(symbol)` first or pass `register_missing=True`.

Hinglish: ek pair pe ticks aate jaate hain, jab ghante ki seema cross hoti hai
tab bar close ho jaata hai aur ek event milta hai. Scanner us event pe pattern
match karega — but woh Phase 8C ka kaam hai.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional

from data.bar_aggregator import Bar, BarAggregator
from data.tick_collector import Tick


@dataclass(frozen=True)
class BarCloseEvent:
    """A 1H bar just closed for a specific symbol."""
    symbol: str
    bar: Bar


class MultiPairFeed:
    """Per-symbol aggregator manager. Emits BarCloseEvents on hour boundaries."""

    def __init__(
        self,
        symbols: Iterable[str],
        timeframe_minutes: int = 60,
        register_missing: bool = False,
    ) -> None:
        self._tf_min = timeframe_minutes
        self._register_missing = register_missing
        self._aggregators: dict[str, BarAggregator] = {}
        for s in symbols:
            self.register(s)
        self._bars_emitted: int = 0
        self._ticks_processed: int = 0
        self._unknown_dropped: int = 0

    @property
    def timeframe_minutes(self) -> int:
        return self._tf_min

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self._aggregators.keys())

    @property
    def bars_emitted(self) -> int:
        return self._bars_emitted

    @property
    def ticks_processed(self) -> int:
        return self._ticks_processed

    @property
    def unknown_dropped(self) -> int:
        """Ticks for symbols not registered. Only > 0 when register_missing=False."""
        return self._unknown_dropped

    # ----------------------------------------------------------------- mgmt

    def register(self, symbol: str) -> None:
        """Add a symbol. No-op if already registered."""
        if symbol not in self._aggregators:
            self._aggregators[symbol] = BarAggregator(symbol, self._tf_min)

    def is_registered(self, symbol: str) -> bool:
        return symbol in self._aggregators

    # ---------------------------------------------------------------- ingest

    def on_tick(self, symbol: str, tick: Tick) -> Optional[BarCloseEvent]:
        """Feed one tick. Return a BarCloseEvent if this tick crossed a boundary."""
        agg = self._aggregators.get(symbol)
        if agg is None:
            if self._register_missing:
                self.register(symbol)
                agg = self._aggregators[symbol]
            else:
                self._unknown_dropped += 1
                return None
        self._ticks_processed += 1
        closed = agg.on_tick(tick)
        if closed is None:
            return None
        self._bars_emitted += 1
        return BarCloseEvent(symbol=symbol, bar=closed)

    def flush_all(self) -> list[BarCloseEvent]:
        """Drain every in-progress bar at end-of-stream.

        Order is insertion-order of registration — caller can sort by
        `event.bar.time_msc` if a chronological view is preferred.
        """
        events: list[BarCloseEvent] = []
        for sym, agg in self._aggregators.items():
            bar = agg.flush()
            if bar is not None:
                self._bars_emitted += 1
                events.append(BarCloseEvent(symbol=sym, bar=bar))
        return events
