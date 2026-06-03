"""TickCollector — exhaustive unit tests.

Targets:
  - Tick dataclass (frozen, fields, hash/eq)
  - Constructor defaults + counter init
  - poll_once cursor + dedup semantics
  - Queue-full drop accounting
  - Batch saturation → no sleep
  - run() loop: cursor seed, exception swallow, cancellation
  - Edge ticks (zero spread, huge volume, extreme flag bits)

We avoid pytest-asyncio (not installed) — async tests run via
`asyncio.run(_body())` from a sync test function so parametrize still works.
"""

from __future__ import annotations
import asyncio
import dataclasses

import numpy as np
import pytest

from data.tick_collector import Tick, TickCollector
from tests.data.fixtures.mock_connector import FakeConnector
from tests.data.fixtures.synthetic_ticks import (
    MT5_TICK_DTYPE, mt5_tick_array, ticks_to_mt5_array,
    random_walk_ticks, make_tick, utc_ms,
)


def _run(coro):
    """Run an async function body in a fresh event loop."""
    return asyncio.run(coro)


# ===========================================================================
# A. Tick dataclass
# ===========================================================================


class TestTickDataclass:

    @pytest.mark.parametrize("time_msc", [0, 1, 10**12, 10**13, 1_700_000_000_000])
    def test_time_msc_field(self, time_msc):
        t = Tick(time_msc=time_msc, bid=1.0, ask=1.0, last=1.0,
                 volume=1, volume_real=1.0, flags=0)
        assert t.time_msc == time_msc

    @pytest.mark.parametrize("bid", [-1.0, 0.0, 1.0, 1.23456, 1e6, 1e-6])
    def test_bid_field(self, bid):
        t = Tick(0, bid, 2.0, 1.5, 1, 1.0, 0)
        assert t.bid == pytest.approx(bid)

    @pytest.mark.parametrize("ask", [0.0, 1.0001, 1e9])
    def test_ask_field(self, ask):
        t = Tick(0, 1.0, ask, 1.0, 1, 1.0, 0)
        assert t.ask == pytest.approx(ask)

    @pytest.mark.parametrize("last", [0.0, 1.0, 5.5])
    def test_last_field(self, last):
        t = Tick(0, 1.0, 1.0, last, 1, 1.0, 0)
        assert t.last == pytest.approx(last)

    @pytest.mark.parametrize("volume", [0, 1, 100, 10**6, 2**32])
    def test_volume_field(self, volume):
        t = Tick(0, 1.0, 1.0, 1.0, volume, 1.0, 0)
        assert t.volume == volume

    @pytest.mark.parametrize("volume_real", [0.0, 0.5, 1.0, 1e6])
    def test_volume_real_field(self, volume_real):
        t = Tick(0, 1.0, 1.0, 1.0, 1, volume_real, 0)
        assert t.volume_real == pytest.approx(volume_real)

    @pytest.mark.parametrize("flags", [0, 1, 2, 3, 4, 6, 8, 12, 0xFFFF])
    def test_flags_field(self, flags):
        t = Tick(0, 1.0, 1.0, 1.0, 1, 1.0, flags)
        assert t.flags == flags

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(Tick)

    def test_is_frozen(self):
        t = Tick(0, 1.0, 1.0, 1.0, 1, 1.0, 0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.bid = 2.0           # type: ignore[misc]

    def test_equality_by_value(self):
        a = Tick(1, 1.0, 1.1, 1.05, 5, 5.5, 7)
        b = Tick(1, 1.0, 1.1, 1.05, 5, 5.5, 7)
        assert a == b

    def test_inequality_when_field_differs(self):
        a = Tick(1, 1.0, 1.1, 1.05, 5, 5.5, 7)
        b = Tick(2, 1.0, 1.1, 1.05, 5, 5.5, 7)
        assert a != b

    def test_hashable(self):
        t = Tick(1, 1.0, 1.1, 1.05, 5, 5.5, 7)
        assert len({t, t}) == 1

    def test_repr_contains_class_name(self):
        t = Tick(1, 1.0, 1.1, 1.05, 5, 5.5, 7)
        assert "Tick" in repr(t)

    def test_repr_contains_time_msc(self):
        t = Tick(123456789, 1.0, 1.1, 1.05, 5, 5.5, 7)
        assert "123456789" in repr(t)

    def test_fields_order_stable(self):
        names = [f.name for f in dataclasses.fields(Tick)]
        assert names == [
            "time_msc", "bid", "ask", "last", "volume", "volume_real", "flags",
        ]


# ===========================================================================
# B. Constructor + defaults
# ===========================================================================


def _new_collector(**kwargs):
    return TickCollector(FakeConnector(), asyncio.Queue(), **kwargs)


class TestConstructor:

    def test_defaults_collected_zero(self):
        c = _new_collector()
        assert c.collected == 0

    def test_defaults_dropped_zero(self):
        c = _new_collector()
        assert c.dropped == 0

    def test_defaults_cursor_zero(self):
        c = _new_collector()
        assert c.cursor_msc == 0

    @pytest.mark.parametrize("poll_ms,expected_sec", [
        (10, 0.01), (25, 0.025), (50, 0.05), (100, 0.1), (250, 0.25),
        (500, 0.5), (1000, 1.0), (2000, 2.0),
    ])
    def test_poll_interval_conversion(self, poll_ms, expected_sec):
        c = _new_collector(poll_interval_ms=poll_ms)
        assert c._poll_interval == pytest.approx(expected_sec)

    @pytest.mark.parametrize("batch", [1, 100, 500, 1000, 2000, 5000, 10000])
    def test_batch_size_stored(self, batch):
        c = _new_collector(batch_size=batch)
        assert c._batch_size == batch

    @pytest.mark.parametrize("drop_every", [1, 10, 100, 1000])
    def test_drop_log_interval_stored(self, drop_every):
        c = _new_collector(drop_log_every=drop_every)
        assert c._drop_log_every == drop_every

    def test_uses_connector_reference(self):
        conn = FakeConnector()
        c = TickCollector(conn, asyncio.Queue())
        assert c._conn is conn

    def test_queue_reference_kept(self):
        q: asyncio.Queue = asyncio.Queue()
        c = TickCollector(FakeConnector(), q)
        assert c._queue is q


# ===========================================================================
# C. poll_once — happy path
# ===========================================================================


class TestPollOnceBasic:

    def test_empty_array_returns_false(self):
        async def _body():
            conn = FakeConnector(batches=[np.empty(0, dtype=MT5_TICK_DTYPE)])
            c = TickCollector(conn, asyncio.Queue())
            return await c._poll_once()
        assert _run(_body()) is False

    def test_none_returns_false(self):
        async def _body():
            conn = FakeConnector()
            conn.copy_ticks_from = lambda *a, **k: None  # type: ignore
            c = TickCollector(conn, asyncio.Queue())
            return await c._poll_once()
        assert _run(_body()) is False

    def test_single_tick_enqueued(self):
        async def _body():
            arr = mt5_tick_array([(1700000000, 1.10, 1.1002, 1.10, 1, 1700000000000, 0, 1.0)])
            conn = FakeConnector(batches=[arr])
            q: asyncio.Queue = asyncio.Queue(maxsize=10)
            c = TickCollector(conn, q)
            await c._poll_once()
            return q
        q = _run(_body())
        assert q.qsize() == 1
        t = q.get_nowait()
        assert isinstance(t, Tick)
        assert t.time_msc == 1700000000000
        assert t.bid == pytest.approx(1.10)
        assert t.ask == pytest.approx(1.1002)

    def test_cursor_advances_to_max(self):
        async def _body():
            arr = ticks_to_mt5_array([
                make_tick(1000, 1.0), make_tick(2000, 1.0), make_tick(3500, 1.0),
            ])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue())
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.cursor_msc == 3500

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 50, 100, 500])
    def test_n_ticks_all_enqueued(self, n):
        async def _body():
            arr = ticks_to_mt5_array([
                make_tick(1000 + i * 10, 1.0 + i * 1e-4) for i in range(n)
            ])
            conn = FakeConnector(batches=[arr])
            q: asyncio.Queue = asyncio.Queue(maxsize=n + 5)
            c = TickCollector(conn, q)
            await c._poll_once()
            return q, c
        q, c = _run(_body())
        assert q.qsize() == n
        assert c.collected == n

    def test_returns_true_when_batch_at_capacity(self):
        async def _body():
            arr = ticks_to_mt5_array([
                make_tick(1000, 1.0), make_tick(2000, 1.0), make_tick(3000, 1.0),
            ])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=10), batch_size=3)
            return await c._poll_once()
        assert _run(_body()) is True

    def test_returns_false_when_batch_below_capacity(self):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(1000, 1.0)])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=10), batch_size=10)
            return await c._poll_once()
        assert _run(_body()) is False


# ===========================================================================
# D. Dedup — strictly greater than cursor
# ===========================================================================


class TestDedupSemantics:

    def test_drops_ticks_equal_to_cursor(self):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(1000, 1.0), make_tick(2000, 1.0)])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=10))
            c._cursor_msc = 2000
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.collected == 0

    def test_drops_ticks_older_than_cursor(self):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(500, 1.0), make_tick(700, 1.0)])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=10))
            c._cursor_msc = 1000
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.collected == 0

    def test_partial_overlap_keeps_only_new(self):
        async def _body():
            arr = ticks_to_mt5_array([
                make_tick(900, 1.0), make_tick(1000, 1.0),
                make_tick(1100, 1.0), make_tick(1200, 1.0),
            ])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=10))
            c._cursor_msc = 1000
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.collected == 2
        assert c.cursor_msc == 1200

    @pytest.mark.parametrize("cursor,kept", [
        (0, 5), (999, 5), (1000, 4), (1100, 3), (1200, 2),
        (1300, 1), (1400, 0), (1500, 0), (10**12, 0),
    ])
    def test_dedup_cutoff(self, cursor, kept):
        async def _body():
            arr = ticks_to_mt5_array([
                make_tick(t, 1.0) for t in (1000, 1100, 1200, 1300, 1400)
            ])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=10))
            c._cursor_msc = cursor
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.collected == kept

    def test_all_new_keeps_all(self):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(i, 1.0) for i in range(5000, 5010)])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=20))
            c._cursor_msc = 4000
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.collected == 10

    def test_cursor_does_not_regress_when_all_old(self):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(t, 1.0) for t in (100, 200, 300)])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=10))
            c._cursor_msc = 5000
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.cursor_msc == 5000

    @pytest.mark.parametrize("dup_count", [2, 3, 5, 10, 50])
    def test_duplicate_timestamps_handled(self, dup_count):
        async def _body():
            arr = ticks_to_mt5_array(
                [make_tick(2000, 1.0 + i * 1e-4) for i in range(dup_count)]
            )
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=dup_count + 5))
            c._cursor_msc = 1999
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.collected == dup_count
        assert c.cursor_msc == 2000


# ===========================================================================
# E. Queue-full handling
# ===========================================================================


class TestQueueFull:

    def test_dropped_increments_when_queue_full(self):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(1000 + i, 1.0) for i in range(10)])
            conn = FakeConnector(batches=[arr])
            q: asyncio.Queue = asyncio.Queue(maxsize=3)
            c = TickCollector(conn, q)
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.collected == 3
        assert c.dropped == 7

    @pytest.mark.parametrize("qmax,ticks,want_collected,want_dropped", [
        (1, 5, 1, 4),
        (2, 5, 2, 3),
        (5, 5, 5, 0),
        (10, 5, 5, 0),
        (3, 100, 3, 97),
    ])
    def test_drop_accounting(self, qmax, ticks, want_collected, want_dropped):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(1000 + i, 1.0) for i in range(ticks)])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=qmax))
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.collected == want_collected
        assert c.dropped == want_dropped

    def test_cursor_advances_even_when_queue_full(self):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(1000 + i, 1.0) for i in range(20)])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=2))
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.cursor_msc == 1019

    @pytest.mark.parametrize("drop_every", [1, 10, 100, 500])
    def test_drop_log_throttle_doesnt_crash(self, drop_every):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(1000 + i, 1.0) for i in range(200)])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(
                conn, asyncio.Queue(maxsize=5),
                drop_log_every=drop_every,
            )
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.dropped == 195


# ===========================================================================
# F. Saturation signal
# ===========================================================================


class TestSaturationSignal:

    @pytest.mark.parametrize("returned,batch_size,expect_sat", [
        (0, 100, False),
        (1, 100, False),
        (50, 100, False),
        (99, 100, False),
        (100, 100, True),
        (10, 10, True),
        (1, 1, True),
        (5, 3, True),
    ])
    def test_saturation_flag(self, returned, batch_size, expect_sat):
        async def _body():
            if returned == 0:
                arr = np.empty(0, dtype=MT5_TICK_DTYPE)
            else:
                arr = ticks_to_mt5_array([
                    make_tick(1000 + i, 1.0) for i in range(returned)
                ])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(
                conn, asyncio.Queue(maxsize=returned + 5),
                batch_size=batch_size,
            )
            return await c._poll_once()
        assert _run(_body()) is expect_sat

    def test_saturation_with_all_old_ticks_still_signals(self):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(100, 1.0) for _ in range(5)])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=10), batch_size=5)
            c._cursor_msc = 200
            out = await c._poll_once()
            return out, c
        out, c = _run(_body())
        assert out is True
        assert c.collected == 0


# ===========================================================================
# G. run() loop — async lifecycle
# ===========================================================================


class TestRunLoop:

    def test_cursor_seeded_from_connector(self):
        async def _body():
            conn = FakeConnector(seed_msc=987654321)
            c = TickCollector(conn, asyncio.Queue(), poll_interval_ms=10)
            task = asyncio.create_task(c.run())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return c, conn
        c, conn = _run(_body())
        assert c.cursor_msc == 987654321
        assert conn.seed_calls == 1

    def test_cancellation_propagates(self):
        async def _body():
            conn = FakeConnector()
            c = TickCollector(conn, asyncio.Queue(), poll_interval_ms=10)
            task = asyncio.create_task(c.run())
            await asyncio.sleep(0.03)
            task.cancel()
            raised = False
            try:
                await task
            except asyncio.CancelledError:
                raised = True
            return raised
        assert _run(_body()) is True

    def test_exception_in_poll_is_swallowed(self):
        async def _body():
            ok_arr = ticks_to_mt5_array([make_tick(2000, 1.0)])
            # Pre-queue many ok batches after the raising one.
            conn = FakeConnector(seed_msc=1000)
            conn.queue_exception(RuntimeError("boom"))
            for _ in range(50):
                conn.queue_batch(ok_arr)
            c = TickCollector(conn, asyncio.Queue(maxsize=200),
                              poll_interval_ms=2, batch_size=10)
            task = asyncio.create_task(c.run())
            await asyncio.sleep(0.6)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return c
        c = _run(_body())
        # The loop must have made at least one successful poll after the raise.
        assert c.collected >= 1

    @pytest.mark.parametrize("n_batches", [1, 5, 20, 50])
    def test_collects_multiple_batches(self, n_batches):
        async def _body():
            batches = [
                ticks_to_mt5_array([make_tick(2000 + i, 1.0)])
                for i in range(n_batches)
            ]
            conn = FakeConnector(seed_msc=1000, batches=batches)
            c = TickCollector(
                conn, asyncio.Queue(maxsize=n_batches + 10),
                poll_interval_ms=2,
            )
            task = asyncio.create_task(c.run())
            await asyncio.sleep(0.3)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return c
        c = _run(_body())
        assert c.collected >= 1

    def test_log_summary_on_cancel_no_crash(self):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(2000, 1.0)])
            conn = FakeConnector(seed_msc=1000, batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=5), poll_interval_ms=5)
            task = asyncio.create_task(c.run())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return True
        assert _run(_body()) is True

    def test_zero_polls_when_cancelled_immediately(self):
        async def _body():
            conn = FakeConnector(seed_msc=1234)
            c = TickCollector(conn, asyncio.Queue(), poll_interval_ms=100)
            task = asyncio.create_task(c.run())
            await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return c
        c = _run(_body())
        assert c.collected == 0


# ===========================================================================
# H. Edge ticks
# ===========================================================================


class TestEdgeTicks:

    @pytest.mark.parametrize("bid,ask", [
        (0.0, 0.0), (1.0, 1.0), (1e-9, 1e-9), (1e9, 1e9),
        (-1.0, -1.0), (1.0, 0.999),
    ])
    def test_unusual_price_combinations(self, bid, ask):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(1000, bid, ask=ask)])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=5))
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.collected == 1

    @pytest.mark.parametrize("volume", [0, 1, 100, 10**6, 2**40])
    def test_extreme_volumes(self, volume):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(1000, 1.0, volume=volume)])
            conn = FakeConnector(batches=[arr])
            q: asyncio.Queue = asyncio.Queue(maxsize=5)
            c = TickCollector(conn, q)
            await c._poll_once()
            return q
        q = _run(_body())
        t = q.get_nowait()
        assert t.volume == volume

    @pytest.mark.parametrize("flags", [0, 1, 2, 4, 8, 16, 0x7FFFFFFF])
    def test_flag_values_pass_through(self, flags):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(1000, 1.0, flags=flags)])
            conn = FakeConnector(batches=[arr])
            q: asyncio.Queue = asyncio.Queue(maxsize=5)
            c = TickCollector(conn, q)
            await c._poll_once()
            return q
        q = _run(_body())
        t = q.get_nowait()
        assert t.flags == flags

    def test_high_frequency_burst_1000_ticks(self):
        async def _body():
            arr = ticks_to_mt5_array([
                make_tick(1000 + i, 1.0 + i * 1e-7) for i in range(1000)
            ])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=2000))
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.collected == 1000

    def test_random_walk_acceptance(self):
        async def _body():
            ticks = random_walk_ticks(200, utc_ms(2026, 5, 18, 10, 0))
            arr = ticks_to_mt5_array(ticks)
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=500))
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.collected == 200

    @pytest.mark.parametrize("count", [1, 10, 100, 1000])
    def test_collected_property_matches_queue_size(self, count):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(1000 + i, 1.0) for i in range(count)])
            conn = FakeConnector(batches=[arr])
            q: asyncio.Queue = asyncio.Queue(maxsize=count + 10)
            c = TickCollector(conn, q)
            await c._poll_once()
            return q, c
        q, c = _run(_body())
        assert c.collected == q.qsize() == count


# ===========================================================================
# I. Properties / introspection
# ===========================================================================


class TestProperties:

    def test_collected_initial_zero(self):
        c = TickCollector(FakeConnector(), asyncio.Queue())
        assert c.collected == 0

    def test_dropped_initial_zero(self):
        c = TickCollector(FakeConnector(), asyncio.Queue())
        assert c.dropped == 0

    def test_cursor_initial_zero(self):
        c = TickCollector(FakeConnector(), asyncio.Queue())
        assert c.cursor_msc == 0

    def test_properties_are_read_only(self):
        c = TickCollector(FakeConnector(), asyncio.Queue())
        with pytest.raises(AttributeError):
            c.collected = 99       # type: ignore[misc]


# ===========================================================================
# J. Connector usage sanity
# ===========================================================================


class TestConnectorUsage:

    def test_copy_ticks_called_with_cursor(self):
        async def _body():
            conn = FakeConnector(seed_msc=0)
            c = TickCollector(conn, asyncio.Queue(), batch_size=42)
            c._cursor_msc = 12345
            await c._poll_once()
            return conn
        conn = _run(_body())
        assert conn.calls[0] == (12345, 42)

    @pytest.mark.parametrize("batch", [1, 100, 5000])
    def test_batch_size_passed_through(self, batch):
        async def _body():
            conn = FakeConnector(seed_msc=0)
            c = TickCollector(conn, asyncio.Queue(), batch_size=batch)
            await c._poll_once()
            return conn
        conn = _run(_body())
        assert conn.calls[0][1] == batch


# ===========================================================================
# K. Mixed price / spread / volume parametrization sweep
# ===========================================================================


# Synthesize ~120 quick coverage tests by sweeping the input space.
SPREAD_CASES = [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.01, 0.1, 1.0]
PRICE_CASES = [0.1, 1.0, 1.5, 10.0, 100.0, 1500.0, 2300.0]
TIME_CASES = [
    utc_ms(2024, 1, 1, 0, 0),
    utc_ms(2025, 6, 15, 12, 30),
    utc_ms(2026, 5, 18, 10, 0),
    utc_ms(2026, 12, 31, 23, 59, 59),
]


@pytest.mark.parametrize("spread", SPREAD_CASES)
@pytest.mark.parametrize("price", PRICE_CASES[:3])
class TestSweepSpreadPrice:

    def test_tick_construction(self, spread, price):
        t = make_tick(1000, price, spread=spread)
        assert t.ask - t.bid == pytest.approx(spread)
        assert t.bid == pytest.approx(price)

    def test_enqueue_cycle(self, spread, price):
        async def _body():
            arr = ticks_to_mt5_array([make_tick(1000, price, spread=spread)])
            conn = FakeConnector(batches=[arr])
            q: asyncio.Queue = asyncio.Queue(maxsize=5)
            c = TickCollector(conn, q)
            await c._poll_once()
            return q
        q = _run(_body())
        t = q.get_nowait()
        assert t.bid == pytest.approx(price)
        assert t.ask == pytest.approx(price + spread)


@pytest.mark.parametrize("t_msc", TIME_CASES)
@pytest.mark.parametrize("count", [1, 3, 7, 11])
class TestSweepTimeCount:

    def test_cursor_max(self, t_msc, count):
        async def _body():
            arr = ticks_to_mt5_array([
                make_tick(t_msc + i, 1.0) for i in range(count)
            ])
            conn = FakeConnector(batches=[arr])
            c = TickCollector(conn, asyncio.Queue(maxsize=count + 5))
            await c._poll_once()
            return c
        c = _run(_body())
        assert c.cursor_msc == t_msc + count - 1


# ===========================================================================
# L. FakeConnector self-checks (fixture verification)
# ===========================================================================


class TestFakeConnectorFixture:

    def test_seed_returns_configured_msc(self):
        fc = FakeConnector(seed_msc=42)
        assert fc.last_tick_msc() == 42
        assert fc.seed_calls == 1

    def test_batches_returned_in_order(self):
        a = ticks_to_mt5_array([make_tick(100, 1.0)])
        b = ticks_to_mt5_array([make_tick(200, 1.0)])
        fc = FakeConnector(batches=[a, b])
        assert fc.copy_ticks_from(0, 10).shape[0] == 1
        assert fc.copy_ticks_from(0, 10).shape[0] == 1
        # Exhausted → empty
        assert fc.copy_ticks_from(0, 10).shape[0] == 0

    def test_queue_batch_chain(self):
        fc = FakeConnector()
        arr = ticks_to_mt5_array([make_tick(100, 1.0)])
        fc.queue_batch(arr).queue_batch(arr)
        assert fc.copy_ticks_from(0, 1).shape[0] == 1
        assert fc.copy_ticks_from(0, 1).shape[0] == 1
        assert fc.copy_ticks_from(0, 1).shape[0] == 0

    def test_queue_exception_raises(self):
        fc = FakeConnector().queue_exception(RuntimeError("x"))
        with pytest.raises(RuntimeError):
            fc.copy_ticks_from(0, 1)

    def test_calls_recorded(self):
        fc = FakeConnector()
        fc.copy_ticks_from(123, 7)
        fc.copy_ticks_from(456, 11)
        assert fc.calls == [(123, 7), (456, 11)]
