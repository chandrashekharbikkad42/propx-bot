"""Phase 8B — MultiPairFeed tests.

Covers:
  - Multiple registered pairs, independent aggregators
  - Bar-close event emission only on boundary cross
  - Unknown symbol dropped (default) vs auto-registered
  - flush_all drains all pairs
  - Counters track ticks + bars
"""

from __future__ import annotations
from datetime import datetime, timezone

import pytest

from data.multi_pair_feed import BarCloseEvent, MultiPairFeed
from data.tick_collector import Tick


def _utc_ms(year: int, month: int, day: int, hour: int, minute: int) -> int:
    return int(
        datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp() * 1000
    )


def _tick(time_msc: int, bid: float = 1.10) -> Tick:
    return Tick(time_msc, bid, bid + 0.0002, bid, 1, 1.0, 0)


class TestRegistration:
    def test_initial_symbols(self):
        f = MultiPairFeed(["EURUSD", "GBPUSD"])
        assert f.is_registered("EURUSD")
        assert f.is_registered("GBPUSD")
        assert not f.is_registered("USDJPY")
        assert set(f.symbols) == {"EURUSD", "GBPUSD"}

    def test_register_after_init(self):
        f = MultiPairFeed(["EURUSD"])
        f.register("USDJPY")
        assert f.is_registered("USDJPY")

    def test_register_is_idempotent(self):
        f = MultiPairFeed(["EURUSD"])
        f.register("EURUSD")
        # No exception, still one entry.
        assert f.symbols.count("EURUSD") == 1


class TestOnTickBasic:
    def test_first_tick_no_event(self):
        f = MultiPairFeed(["EURUSD"])
        evt = f.on_tick("EURUSD", _tick(_utc_ms(2026, 5, 17, 7, 0), 1.10))
        assert evt is None
        assert f.ticks_processed == 1
        assert f.bars_emitted == 0

    def test_boundary_crossing_emits_event(self):
        f = MultiPairFeed(["EURUSD"])
        f.on_tick("EURUSD", _tick(_utc_ms(2026, 5, 17, 7, 0), 1.10))
        f.on_tick("EURUSD", _tick(_utc_ms(2026, 5, 17, 7, 30), 1.11))
        evt = f.on_tick("EURUSD", _tick(_utc_ms(2026, 5, 17, 8, 5), 1.115))
        assert isinstance(evt, BarCloseEvent)
        assert evt.symbol == "EURUSD"
        assert evt.bar.time_msc == _utc_ms(2026, 5, 17, 7, 0)
        assert f.bars_emitted == 1
        assert f.ticks_processed == 3

    def test_independent_pair_state(self):
        """One pair crossing a boundary must not affect another pair."""
        f = MultiPairFeed(["EURUSD", "GBPUSD"])
        f.on_tick("EURUSD", _tick(_utc_ms(2026, 5, 17, 7, 0), 1.10))
        f.on_tick("GBPUSD", _tick(_utc_ms(2026, 5, 17, 7, 0), 1.25))
        # Cross only EURUSD into 08:00.
        evt = f.on_tick("EURUSD", _tick(_utc_ms(2026, 5, 17, 8, 5), 1.105))
        assert evt is not None and evt.symbol == "EURUSD"
        # GBPUSD still open at 07:00.
        evt2 = f.on_tick("GBPUSD", _tick(_utc_ms(2026, 5, 17, 7, 45), 1.252))
        assert evt2 is None


class TestUnknownSymbol:
    def test_unknown_default_drops(self):
        f = MultiPairFeed(["EURUSD"])
        evt = f.on_tick("UNKNOWN", _tick(_utc_ms(2026, 5, 17, 7, 0)))
        assert evt is None
        assert f.unknown_dropped == 1
        assert f.ticks_processed == 0  # not registered, not counted

    def test_register_missing_auto_registers(self):
        f = MultiPairFeed(["EURUSD"], register_missing=True)
        evt = f.on_tick("USDJPY", _tick(_utc_ms(2026, 5, 17, 7, 0)))
        assert evt is None  # first tick, no bar yet
        assert f.is_registered("USDJPY")
        assert f.unknown_dropped == 0


class TestFlushAll:
    def test_flush_drains_each_pair(self):
        f = MultiPairFeed(["EURUSD", "GBPUSD"])
        f.on_tick("EURUSD", _tick(_utc_ms(2026, 5, 17, 7, 0), 1.10))
        f.on_tick("GBPUSD", _tick(_utc_ms(2026, 5, 17, 7, 0), 1.25))
        events = f.flush_all()
        assert len(events) == 2
        syms = {e.symbol for e in events}
        assert syms == {"EURUSD", "GBPUSD"}
        # Flush twice is safe; second time yields nothing.
        assert f.flush_all() == []

    def test_flush_skips_pair_with_no_data(self):
        f = MultiPairFeed(["EURUSD", "GBPUSD"])
        f.on_tick("EURUSD", _tick(_utc_ms(2026, 5, 17, 7, 0), 1.10))
        events = f.flush_all()
        assert len(events) == 1
        assert events[0].symbol == "EURUSD"


class TestMultiBarStream:
    def test_three_pairs_alternating_ticks(self):
        f = MultiPairFeed(["EURUSD", "GBPUSD", "USDJPY"])
        events: list[BarCloseEvent] = []

        for h in (7, 8, 9):
            for sym, base in (("EURUSD", 1.10), ("GBPUSD", 1.25), ("USDJPY", 150.0)):
                for m in (0, 30):
                    e = f.on_tick(sym, _tick(_utc_ms(2026, 5, 17, h, m), base))
                    if e is not None:
                        events.append(e)
        # 7,8 bars closed for each (3) × 2 = 6 bars; 09:xx bars still open.
        assert len([e for e in events if e.symbol == "EURUSD"]) == 2
        assert len([e for e in events if e.symbol == "GBPUSD"]) == 2
        assert len([e for e in events if e.symbol == "USDJPY"]) == 2

        # Flush remaining three.
        tail = f.flush_all()
        assert len(tail) == 3
        assert {t.symbol for t in tail} == {"EURUSD", "GBPUSD", "USDJPY"}
