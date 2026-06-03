"""MultiPairFeed — exhaustive unit tests.

Targets:
  - BarCloseEvent dataclass (frozen, fields)
  - Registration semantics (init list, register, idempotent, is_registered)
  - on_tick routing per symbol, counters, event payload
  - Unknown symbol drop vs register_missing auto-add
  - flush_all order + counter side-effects
  - Concurrent multi-pair updates (independent aggregator state)
  - Aggregator exception isolation
  - 8 pairs × scenarios
  - Mid-run register
"""

from __future__ import annotations
import dataclasses
from typing import List

import pytest

from data.bar_aggregator import Bar, BarAggregator
from data.multi_pair_feed import BarCloseEvent, MultiPairFeed
from data.tick_collector import Tick
from tests.data.fixtures.synthetic_ticks import (
    EIGHT_PAIRS, base_price_for, spread_for,
    hour_filling_ticks, make_tick, utc_ms,
)


# ===========================================================================
# A. BarCloseEvent dataclass
# ===========================================================================


class TestBarCloseEvent:

    def test_construction(self):
        b = Bar("EURUSD", 1000, 1.0, 1.0, 1.0, 1.0, 1)
        e = BarCloseEvent("EURUSD", b)
        assert e.symbol == "EURUSD"
        assert e.bar is b

    def test_frozen(self):
        b = Bar("X", 1000, 1.0, 1.0, 1.0, 1.0, 1)
        e = BarCloseEvent("X", b)
        with pytest.raises(dataclasses.FrozenInstanceError):
            e.symbol = "Y"        # type: ignore[misc]

    def test_equality(self):
        b = Bar("EURUSD", 1000, 1.0, 1.0, 1.0, 1.0, 1)
        assert BarCloseEvent("EURUSD", b) == BarCloseEvent("EURUSD", b)

    def test_different_bar_unequal(self):
        b1 = Bar("EURUSD", 1000, 1.0, 1.0, 1.0, 1.0, 1)
        b2 = Bar("EURUSD", 2000, 1.0, 1.0, 1.0, 1.0, 1)
        assert BarCloseEvent("EURUSD", b1) != BarCloseEvent("EURUSD", b2)

    def test_hashable(self):
        b = Bar("X", 0, 1.0, 1.0, 1.0, 1.0, 1)
        e = BarCloseEvent("X", b)
        assert len({e, e}) == 1

    @pytest.mark.parametrize("sym", EIGHT_PAIRS)
    def test_each_symbol(self, sym):
        b = Bar(sym, 0, 1.0, 1.0, 1.0, 1.0, 1)
        e = BarCloseEvent(sym, b)
        assert e.symbol == sym


# ===========================================================================
# B. Constructor / registration
# ===========================================================================


class TestConstruction:

    def test_default_timeframe_60(self):
        f = MultiPairFeed(["X"])
        assert f.timeframe_minutes == 60

    @pytest.mark.parametrize("tf", [1, 5, 15, 30, 60, 240])
    def test_timeframe_param(self, tf):
        f = MultiPairFeed(["X"], timeframe_minutes=tf)
        assert f.timeframe_minutes == tf

    def test_empty_symbols_allowed(self):
        f = MultiPairFeed([])
        assert f.symbols == tuple()

    @pytest.mark.parametrize("syms", [
        ["EURUSD"],
        ["EURUSD", "GBPUSD"],
        list(EIGHT_PAIRS),
        ["A", "B", "C", "D", "E"],
    ])
    def test_initial_symbols(self, syms):
        f = MultiPairFeed(syms)
        assert set(f.symbols) == set(syms)
        for s in syms:
            assert f.is_registered(s)

    def test_initial_counters_zero(self):
        f = MultiPairFeed(["X"])
        assert f.ticks_processed == 0
        assert f.bars_emitted == 0
        assert f.unknown_dropped == 0

    def test_register_after_init(self):
        f = MultiPairFeed(["A"])
        f.register("B")
        assert f.is_registered("B")
        assert "B" in f.symbols

    def test_register_idempotent(self):
        f = MultiPairFeed(["X"])
        before = len(f.symbols)
        f.register("X")
        f.register("X")
        assert len(f.symbols) == before

    def test_register_keeps_aggregator_state(self):
        f = MultiPairFeed(["X"])
        f.on_tick("X", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        f.register("X")             # Re-register should NOT reset aggregator
        # Aggregator still has its open bar.
        evt = f.on_tick("X", make_tick(utc_ms(2026, 5, 18, 11, 0), 1.11))
        assert evt is not None      # boundary cross emitted

    @pytest.mark.parametrize("sym", EIGHT_PAIRS)
    def test_each_pair_can_register(self, sym):
        f = MultiPairFeed([sym])
        assert f.is_registered(sym)

    def test_is_registered_false_for_unknown(self):
        f = MultiPairFeed(["A", "B"])
        assert not f.is_registered("C")
        assert not f.is_registered("")

    def test_symbols_is_tuple_immutable(self):
        f = MultiPairFeed(["A", "B"])
        s = f.symbols
        assert isinstance(s, tuple)


# ===========================================================================
# C. on_tick — basic routing
# ===========================================================================


class TestOnTickRouting:

    def test_first_tick_no_event(self):
        f = MultiPairFeed(["X"])
        assert f.on_tick("X", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10)) is None

    def test_ticks_processed_increments(self):
        f = MultiPairFeed(["X"])
        f.on_tick("X", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        f.on_tick("X", make_tick(utc_ms(2026, 5, 18, 10, 5), 1.11))
        assert f.ticks_processed == 2

    def test_bars_emitted_increments_only_on_close(self):
        f = MultiPairFeed(["X"])
        f.on_tick("X", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        assert f.bars_emitted == 0
        f.on_tick("X", make_tick(utc_ms(2026, 5, 18, 11, 5), 1.11))
        assert f.bars_emitted == 1

    def test_event_payload(self):
        f = MultiPairFeed(["X"])
        f.on_tick("X", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        evt = f.on_tick("X", make_tick(utc_ms(2026, 5, 18, 11, 0), 1.11))
        assert isinstance(evt, BarCloseEvent)
        assert evt.symbol == "X"
        assert evt.bar.symbol == "X"
        assert evt.bar.time_msc == utc_ms(2026, 5, 18, 10, 0)

    def test_two_symbol_routing_independent(self):
        f = MultiPairFeed(["A", "B"])
        f.on_tick("A", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        f.on_tick("B", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.20))
        # Cross only A.
        evt = f.on_tick("A", make_tick(utc_ms(2026, 5, 18, 11, 5), 1.11))
        assert evt is not None and evt.symbol == "A"
        # B still in its bar.
        assert f.on_tick("B", make_tick(utc_ms(2026, 5, 18, 10, 30), 1.21)) is None

    @pytest.mark.parametrize("n_ticks", [1, 5, 10, 100, 1000])
    def test_ticks_processed_accumulates(self, n_ticks):
        f = MultiPairFeed(["X"])
        for i in range(n_ticks):
            f.on_tick("X", make_tick(utc_ms(2026, 5, 18, 10, 0) + i, 1.10))
        assert f.ticks_processed == n_ticks


# ===========================================================================
# D. Unknown symbol handling
# ===========================================================================


class TestUnknownSymbol:

    def test_default_drops_unknown(self):
        f = MultiPairFeed(["A"])
        evt = f.on_tick("Z", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        assert evt is None
        assert f.unknown_dropped == 1
        assert f.ticks_processed == 0      # NOT counted as processed

    def test_unknown_does_not_register(self):
        f = MultiPairFeed(["A"])
        f.on_tick("Z", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        assert not f.is_registered("Z")

    @pytest.mark.parametrize("n", [1, 5, 10, 100])
    def test_unknown_drop_count(self, n):
        f = MultiPairFeed(["A"])
        for _ in range(n):
            f.on_tick("Z", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        assert f.unknown_dropped == n

    def test_register_missing_auto_registers(self):
        f = MultiPairFeed(["A"], register_missing=True)
        evt = f.on_tick("Z", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        assert evt is None                  # first tick → no close
        assert f.is_registered("Z")
        assert f.unknown_dropped == 0
        assert f.ticks_processed == 1

    @pytest.mark.parametrize("sym", EIGHT_PAIRS)
    def test_register_missing_each_pair(self, sym):
        f = MultiPairFeed([], register_missing=True)
        f.on_tick(sym, make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        assert f.is_registered(sym)


# ===========================================================================
# E. flush_all
# ===========================================================================


class TestFlushAll:

    def test_flush_empty_pairs(self):
        f = MultiPairFeed(["A", "B"])
        # No ticks → flush_all returns []
        assert f.flush_all() == []

    def test_flush_drains_open_bars(self):
        f = MultiPairFeed(["A", "B"])
        f.on_tick("A", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        f.on_tick("B", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.20))
        events = f.flush_all()
        assert len(events) == 2
        assert {e.symbol for e in events} == {"A", "B"}

    def test_flush_twice_is_safe(self):
        f = MultiPairFeed(["A"])
        f.on_tick("A", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        first = f.flush_all()
        second = f.flush_all()
        assert len(first) == 1
        assert second == []

    def test_flush_skips_idle_pair(self):
        f = MultiPairFeed(["A", "B"])
        f.on_tick("A", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        events = f.flush_all()
        assert len(events) == 1
        assert events[0].symbol == "A"

    def test_flush_updates_bars_emitted(self):
        f = MultiPairFeed(["A", "B"])
        f.on_tick("A", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        f.on_tick("B", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.20))
        f.flush_all()
        assert f.bars_emitted == 2

    def test_flush_order_matches_registration(self):
        f = MultiPairFeed(["C", "A", "B"])
        for s in ("C", "A", "B"):
            f.on_tick(s, make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        events = f.flush_all()
        # dict insertion order = registration order
        assert [e.symbol for e in events] == ["C", "A", "B"]

    @pytest.mark.parametrize("syms", [
        ["A"], ["A", "B"], ["A", "B", "C"], list(EIGHT_PAIRS),
    ])
    def test_flush_n_pairs(self, syms):
        f = MultiPairFeed(syms)
        for s in syms:
            f.on_tick(s, make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        events = f.flush_all()
        assert len(events) == len(syms)


# ===========================================================================
# F. Concurrent multi-pair updates (independent state)
# ===========================================================================


class TestIndependentState:

    def test_pair_a_boundary_does_not_emit_b(self):
        f = MultiPairFeed(["A", "B"])
        f.on_tick("A", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        f.on_tick("B", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.20))
        # Cross only A.
        evt = f.on_tick("A", make_tick(utc_ms(2026, 5, 18, 11, 0), 1.15))
        assert evt is not None and evt.symbol == "A"
        # B still open with its single 10:00 tick.
        assert f.on_tick("B", make_tick(utc_ms(2026, 5, 18, 10, 15), 1.21)) is None

    def test_interleaved_streams(self):
        f = MultiPairFeed(["A", "B"])
        events: List[BarCloseEvent] = []
        for h in (7, 8, 9):
            for sym in ("A", "B"):
                e = f.on_tick(sym, make_tick(utc_ms(2026, 5, 18, h, 0), 1.10))
                if e is not None:
                    events.append(e)
        # Two boundary crossings per pair (7→8 and 8→9) → 4 events.
        assert len(events) == 4
        assert sum(1 for e in events if e.symbol == "A") == 2
        assert sum(1 for e in events if e.symbol == "B") == 2

    @pytest.mark.parametrize("n_pairs", [2, 3, 4, 5, 8])
    def test_n_pairs_parallel(self, n_pairs):
        pairs = EIGHT_PAIRS[:n_pairs]
        f = MultiPairFeed(pairs)
        for s in pairs:
            f.on_tick(s, make_tick(utc_ms(2026, 5, 18, 10, 0), base_price_for(s)))
        events = f.flush_all()
        assert len(events) == n_pairs
        assert {e.symbol for e in events} == set(pairs)


# ===========================================================================
# G. Aggregator exception isolation
# ===========================================================================


class TestAggregatorExceptionIsolation:

    def test_per_pair_aggregator_distinct(self):
        f = MultiPairFeed(["A", "B"])
        agg_a = f._aggregators["A"]
        agg_b = f._aggregators["B"]
        assert agg_a is not agg_b
        assert isinstance(agg_a, BarAggregator)

    def test_pair_exception_does_not_corrupt_others(self):
        # Inject a broken aggregator for "A"; "B" must keep working.
        f = MultiPairFeed(["A", "B"])

        class _BrokenAgg:
            def on_tick(self, tick):
                raise RuntimeError("broken")

        f._aggregators["A"] = _BrokenAgg()   # type: ignore[assignment]
        with pytest.raises(RuntimeError):
            f.on_tick("A", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        # B still functional
        assert f.on_tick("B", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.20)) is None


# ===========================================================================
# H. 8 pairs × scenarios sweep (~ 8 × 5 = 40 tests)
# ===========================================================================


def _scn_one(base, t0):
    return [make_tick(t0 + 30 * 60 * 1000, base)]


def _scn_cross(base, t0):
    return [
        make_tick(t0, base),
        make_tick(t0 + 30 * 60 * 1000, base * 1.001),
        make_tick(t0 + 60 * 60 * 1000 + 5_000, base * 1.002),   # crosses
    ]


def _scn_multi_hour(base, t0):
    return hour_filling_ticks(t0, n_per_hour=10, n_hours=4)


def _scn_gap(base, t0):
    return [
        make_tick(t0, base),
        make_tick(t0 + 5 * 60 * 60 * 1000, base * 1.005),  # 5-hour jump
    ]


def _scn_burst(base, t0):
    return [make_tick(t0 + i * 100, base) for i in range(50)]


SCENARIOS = {
    "one_tick": _scn_one,
    "cross": _scn_cross,
    "multi_hour": _scn_multi_hour,
    "gap": _scn_gap,
    "burst": _scn_burst,
}


@pytest.mark.parametrize("pair", EIGHT_PAIRS)
@pytest.mark.parametrize("scn_name", list(SCENARIOS.keys()))
class TestPairScenarioSweep:

    def _feed_run(self, pair, scn_name):
        base = base_price_for(pair)
        ticks = SCENARIOS[scn_name](base, utc_ms(2026, 5, 18, 10, 0))
        f = MultiPairFeed([pair])
        events: List[BarCloseEvent] = []
        for t in ticks:
            e = f.on_tick(pair, t)
            if e is not None:
                events.append(e)
        events.extend(f.flush_all())
        return f, events

    def test_no_unknown_drops(self, pair, scn_name):
        f, _ = self._feed_run(pair, scn_name)
        assert f.unknown_dropped == 0

    def test_ticks_processed_matches_input(self, pair, scn_name):
        base = base_price_for(pair)
        ticks = SCENARIOS[scn_name](base, utc_ms(2026, 5, 18, 10, 0))
        f, _ = self._feed_run(pair, scn_name)
        assert f.ticks_processed == len(ticks)

    def test_every_event_is_for_this_pair(self, pair, scn_name):
        f, events = self._feed_run(pair, scn_name)
        assert all(e.symbol == pair for e in events)
        for e in events:
            assert e.bar.symbol == pair

    def test_event_bars_in_chronological_order(self, pair, scn_name):
        f, events = self._feed_run(pair, scn_name)
        times = [e.bar.time_msc for e in events]
        assert times == sorted(times)

    def test_bars_emitted_matches_event_count(self, pair, scn_name):
        f, events = self._feed_run(pair, scn_name)
        assert f.bars_emitted == len(events)


# ===========================================================================
# I. Mid-run register
# ===========================================================================


class TestMidRunRegister:

    def test_register_after_some_ticks(self):
        f = MultiPairFeed(["A"])
        f.on_tick("A", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        # Now register B.
        f.register("B")
        evt = f.on_tick("B", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.20))
        assert evt is None
        assert f.is_registered("B")
        # B aggregator independent of A's state.
        assert f.bars_emitted == 0
        f.flush_all()
        assert f.bars_emitted == 2

    def test_register_existing_does_not_reset(self):
        f = MultiPairFeed(["A"])
        f.on_tick("A", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        agg_before = f._aggregators["A"]
        f.register("A")
        assert f._aggregators["A"] is agg_before

    def test_register_after_unknown_drop_then_resends(self):
        f = MultiPairFeed(["A"])
        f.on_tick("B", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.20))
        assert f.unknown_dropped == 1
        # Now register B and feed again — counter no longer rises.
        f.register("B")
        f.on_tick("B", make_tick(utc_ms(2026, 5, 18, 10, 0), 1.20))
        assert f.unknown_dropped == 1
        assert f.ticks_processed == 1


# ===========================================================================
# J. Timeframe sweep — feed uses tf for every aggregator
# ===========================================================================


@pytest.mark.parametrize("tf", [1, 5, 15, 30, 60, 120, 240])
def test_feed_propagates_timeframe_to_aggregators(tf):
    f = MultiPairFeed(["A", "B"], timeframe_minutes=tf)
    for s in ("A", "B"):
        assert f._aggregators[s].timeframe_minutes == tf


@pytest.mark.parametrize("tf", [15, 30, 60, 240])
def test_feed_emits_on_tf_boundary(tf):
    period_ms = tf * 60 * 1000
    t0 = (utc_ms(2026, 5, 18, 0, 0) // period_ms) * period_ms
    f = MultiPairFeed(["A"], timeframe_minutes=tf)
    f.on_tick("A", make_tick(t0, 1.10))
    # Cross the boundary.
    evt = f.on_tick("A", make_tick(t0 + period_ms + 1, 1.11))
    assert evt is not None
    assert evt.bar.time_msc == t0


# ===========================================================================
# K. Mass counter accounting (~ sanity across N ticks)
# ===========================================================================


@pytest.mark.parametrize("n_ticks", [10, 100, 1000, 5000])
def test_mass_tick_counter_accounting(n_ticks):
    f = MultiPairFeed(["A"])
    for i in range(n_ticks):
        f.on_tick("A", make_tick(utc_ms(2026, 5, 18, 10, 0) + i, 1.10))
    assert f.ticks_processed == n_ticks
    assert f.unknown_dropped == 0


@pytest.mark.parametrize("n_pairs", [1, 2, 4, 8])
@pytest.mark.parametrize("n_ticks_per_pair", [1, 5, 20])
def test_balanced_multi_pair_load(n_pairs, n_ticks_per_pair):
    pairs = EIGHT_PAIRS[:n_pairs]
    f = MultiPairFeed(pairs)
    for p in pairs:
        for i in range(n_ticks_per_pair):
            f.on_tick(p, make_tick(utc_ms(2026, 5, 18, 10, 0) + i, base_price_for(p)))
    assert f.ticks_processed == n_pairs * n_ticks_per_pair


# ===========================================================================
# L. Property checks on the bar returned through the event
# ===========================================================================


class TestEventBarConsistency:

    @pytest.mark.parametrize("pair", EIGHT_PAIRS)
    def test_event_bar_ohlc_consistent(self, pair):
        f = MultiPairFeed([pair])
        base = base_price_for(pair)
        # Build several ticks inside one bar then cross.
        for i in range(10):
            f.on_tick(pair, make_tick(
                utc_ms(2026, 5, 18, 10, i),
                base + i * 1e-4,
            ))
        evt = f.on_tick(pair, make_tick(utc_ms(2026, 5, 18, 11, 0), base + 0.001))
        assert evt is not None
        b = evt.bar
        assert b.high >= max(b.open, b.close)
        assert b.low <= min(b.open, b.close)
        assert b.volume >= 1
        assert b.spread_mean >= 0


# ===========================================================================
# M. Long stream → many events
# ===========================================================================


@pytest.mark.parametrize("n_hours", [2, 5, 10, 24])
def test_long_stream_emits_n_minus_one_then_flush(n_hours):
    # n_hours of ticks → n_hours-1 closed + 1 flushed = n_hours events.
    ticks = hour_filling_ticks(utc_ms(2026, 5, 18, 10, 0), n_per_hour=5, n_hours=n_hours)
    f = MultiPairFeed(["EURUSD"])
    events: List[BarCloseEvent] = []
    for t in ticks:
        e = f.on_tick("EURUSD", t)
        if e is not None:
            events.append(e)
    events.extend(f.flush_all())
    assert len(events) == n_hours


# ===========================================================================
# N. Edge: feed with single registered pair receives tick stream of others
# ===========================================================================


class TestOnlyRegisteredAccepted:

    def test_only_one_registered_drops_others(self):
        f = MultiPairFeed(["EURUSD"])           # only EURUSD registered
        for s in EIGHT_PAIRS:
            f.on_tick(s, make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
        # 1 processed (EURUSD), 7 dropped (the rest).
        assert f.ticks_processed == 1
        assert f.unknown_dropped == sum(1 for s in EIGHT_PAIRS if s != "EURUSD")


# ===========================================================================
# O. Counter monotonicity invariants (Hypothesis-light)
# ===========================================================================


@pytest.mark.parametrize("seed", list(range(5)))
def test_counters_only_increase(seed):
    import random
    rng = random.Random(seed)
    pairs = ["A", "B", "C"]
    f = MultiPairFeed(pairs)
    last_proc = last_emit = last_drop = 0
    for _ in range(200):
        sym = rng.choice(pairs + ["Z", "Y"])
        f.on_tick(sym, make_tick(
            utc_ms(2026, 5, 18, 10, 0) + rng.randint(0, 7200) * 1000,
            1.10 + rng.random() * 0.01,
        ))
        assert f.ticks_processed >= last_proc
        assert f.bars_emitted >= last_emit
        assert f.unknown_dropped >= last_drop
        last_proc, last_emit, last_drop = (
            f.ticks_processed, f.bars_emitted, f.unknown_dropped,
        )
