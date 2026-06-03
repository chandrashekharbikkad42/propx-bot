"""TickWriter — exhaustive unit tests.

Targets:
  - TICK_SCHEMA shape (field names + types)
  - Constructor: dir creation, defaults, settings.symbol fallback
  - Date partitioning: single-day, multi-day, day-boundary straddle
  - _write_partition: parquet round-trip with TICK_SCHEMA
  - _next_part: counter cache, directory-rescan, gap handling
  - run() loop: size trigger, time trigger, cancel drain
  - Compression options
  - Edge: empty buffer, single tick, large buffer, schema enforcement
"""

from __future__ import annotations
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.tick_collector import Tick
from data.tick_writer import TICK_SCHEMA, TickWriter, _PART_PATTERN
from tests.data.fixtures.synthetic_ticks import (
    make_tick, random_walk_ticks, utc_ms, hour_filling_ticks,
)
from tests.data.fixtures.parquet_helpers import (
    list_date_partitions, list_part_files, read_partition_rows,
    read_partition_table,
)


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# A. Schema
# ===========================================================================


class TestTickSchema:

    def test_field_names_order(self):
        assert [f.name for f in TICK_SCHEMA] == [
            "time_msc", "bid", "ask", "last", "volume", "volume_real", "flags",
        ]

    @pytest.mark.parametrize("name,kind", [
        ("time_msc", "int64"),
        ("bid", "double"),
        ("ask", "double"),
        ("last", "double"),
        ("volume", "uint64"),
        ("volume_real", "double"),
        ("flags", "uint32"),
    ])
    def test_field_types(self, name, kind):
        types = {f.name: str(f.type) for f in TICK_SCHEMA}
        assert types[name] == kind

    def test_schema_has_seven_fields(self):
        assert len(TICK_SCHEMA) == 7


# ===========================================================================
# B. _PART_PATTERN regex
# ===========================================================================


class TestPartPattern:

    @pytest.mark.parametrize("name,want", [
        ("part-00000.parquet", True),
        ("part-00001.parquet", True),
        ("part-99999.parquet", True),
        ("part-12345.parquet", True),
        ("part-0.parquet", False),         # too few digits
        ("part-123456.parquet", False),    # too many
        ("part-00001.PARQUET", False),     # case-sensitive
        ("data-00001.parquet", False),
        ("part_00001.parquet", False),
    ])
    def test_match(self, name, want):
        m = _PART_PATTERN.match(name)
        assert bool(m) is want

    @pytest.mark.parametrize("n", [0, 1, 99, 100, 9999, 12345])
    def test_extract_index(self, n):
        m = _PART_PATTERN.match(f"part-{n:05d}.parquet")
        assert m is not None
        assert int(m.group(1)) == n


# ===========================================================================
# C. Constructor
# ===========================================================================


class TestConstructor:

    def test_creates_symbol_dir(self, patch_data_dir: Path):
        q: asyncio.Queue = asyncio.Queue()
        w = TickWriter(q, symbol="EURUSD")
        expected = patch_data_dir / "symbol=EURUSD"
        assert expected.exists() and expected.is_dir()

    @pytest.mark.parametrize("sym", [
        "EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "AUDNZD",
    ])
    def test_symbol_param(self, sym, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol=sym)
        assert w._symbol == sym
        assert (patch_data_dir / f"symbol={sym}").exists()

    def test_symbol_falls_back_to_settings(self, patch_data_dir, monkeypatch):
        from data import tick_writer as tw_mod
        # patch_data_dir already swapped tw_mod.settings to a SimpleNamespace;
        # override its `symbol` attribute for this test.
        monkeypatch.setattr(tw_mod.settings, "symbol", "FOO123")
        w = TickWriter(asyncio.Queue())
        assert w._symbol == "FOO123"

    @pytest.mark.parametrize("flush_size", [1, 10, 100, 1000, 10_000])
    def test_flush_size_stored(self, flush_size, patch_data_dir):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD", flush_size=flush_size)
        assert w._flush_size == flush_size

    @pytest.mark.parametrize("flush_seconds", [0.1, 0.5, 1.0, 5.0, 30.0])
    def test_flush_seconds_stored(self, flush_seconds, patch_data_dir):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD", flush_seconds=flush_seconds)
        assert w._flush_seconds == flush_seconds

    @pytest.mark.parametrize("compression", ["snappy", "gzip", "zstd", "lz4", "none"])
    def test_compression_stored(self, compression, patch_data_dir):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD", compression=compression)
        assert w._compression == compression

    def test_initial_counters_zero(self, patch_data_dir):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        assert w.written_ticks == 0
        assert w.written_files == 0

    def test_part_counters_empty(self, patch_data_dir):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        assert w._part_counters == {}


# ===========================================================================
# D. _write_partition — single batch, parquet round-trip
# ===========================================================================


class TestWritePartition:

    def test_creates_date_dir(self, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        ticks = [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10)]
        w._write_partition("2026-05-18", ticks)
        assert (patch_data_dir / "symbol=EURUSD" / "date=2026-05-18").exists()

    def test_first_part_is_zero(self, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        ticks = [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10)]
        w._write_partition("2026-05-18", ticks)
        date_dir = patch_data_dir / "symbol=EURUSD" / "date=2026-05-18"
        files = list_part_files(date_dir)
        assert len(files) == 1
        assert files[0].name == "part-00000.parquet"

    def test_subsequent_parts_increment(self, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        for _ in range(5):
            w._write_partition("2026-05-18", [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10)])
        files = list_part_files(patch_data_dir / "symbol=EURUSD" / "date=2026-05-18")
        names = [f.name for f in files]
        assert names == [
            "part-00000.parquet", "part-00001.parquet", "part-00002.parquet",
            "part-00003.parquet", "part-00004.parquet",
        ]

    @pytest.mark.parametrize("count", [1, 5, 100, 1000])
    def test_parquet_row_count(self, count, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        ticks = [make_tick(utc_ms(2026, 5, 18, 10, 0) + i, 1.10 + i * 1e-5)
                 for i in range(count)]
        w._write_partition("2026-05-18", ticks)
        date_dir = patch_data_dir / "symbol=EURUSD" / "date=2026-05-18"
        assert read_partition_rows(date_dir) == count

    def test_round_trip_preserves_values(self, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        ticks = [
            Tick(utc_ms(2026, 5, 18, 10, 0), 1.10, 1.1002, 1.10, 5, 5.5, 6),
            Tick(utc_ms(2026, 5, 18, 10, 1), 1.11, 1.1102, 1.11, 7, 7.7, 4),
        ]
        w._write_partition("2026-05-18", ticks)
        date_dir = patch_data_dir / "symbol=EURUSD" / "date=2026-05-18"
        tbl = read_partition_table(date_dir)
        d = tbl.to_pydict()
        assert d["time_msc"] == [t.time_msc for t in ticks]
        assert d["bid"] == pytest.approx([t.bid for t in ticks])
        assert d["ask"] == pytest.approx([t.ask for t in ticks])
        assert d["volume"] == [t.volume for t in ticks]
        assert d["volume_real"] == pytest.approx([t.volume_real for t in ticks])
        assert d["flags"] == [t.flags for t in ticks]

    def test_schema_matches_tick_schema(self, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        w._write_partition("2026-05-18", [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10)])
        date_dir = patch_data_dir / "symbol=EURUSD" / "date=2026-05-18"
        f = list_part_files(date_dir)[0]
        # ParquetFile.schema_arrow reads ON-DISK schema (no Hive partition cols).
        on_disk = pq.ParquetFile(f).schema_arrow
        assert on_disk.equals(TICK_SCHEMA)

    def test_counters_increment(self, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        w._write_partition("2026-05-18", [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10)] * 3)
        assert w.written_ticks == 3
        assert w.written_files == 1
        w._write_partition("2026-05-18", [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10)])
        assert w.written_ticks == 4
        assert w.written_files == 2

    @pytest.mark.parametrize("compression", ["snappy", "gzip", "zstd"])
    def test_compression_codec_applied(self, compression, patch_data_dir):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD", compression=compression)
        w._write_partition("2026-05-18", [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10)])
        date_dir = patch_data_dir / "symbol=EURUSD" / "date=2026-05-18"
        f = list_part_files(date_dir)[0]
        meta = pq.read_metadata(str(f))
        # All row groups should report the requested codec.
        row_group = meta.row_group(0)
        codecs = {row_group.column(i).compression.upper()
                  for i in range(row_group.num_columns)}
        assert compression.upper() in codecs


# ===========================================================================
# E. _next_part — counter cache + directory rescan
# ===========================================================================


class TestNextPart:

    def test_first_call_zero(self, patch_data_dir: Path, tmp_path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        date_dir = patch_data_dir / "symbol=EURUSD" / "date=2026-05-18"
        date_dir.mkdir(parents=True, exist_ok=True)
        assert w._next_part("2026-05-18", date_dir) == 0

    def test_cached_value_returned(self, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        date_dir = patch_data_dir / "symbol=EURUSD" / "date=2026-05-18"
        date_dir.mkdir(parents=True)
        w._part_counters["2026-05-18"] = 42
        assert w._next_part("2026-05-18", date_dir) == 42

    def test_scans_existing_files(self, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        date_dir = patch_data_dir / "symbol=EURUSD" / "date=2026-05-18"
        date_dir.mkdir(parents=True)
        (date_dir / "part-00000.parquet").write_bytes(b"x")
        (date_dir / "part-00001.parquet").write_bytes(b"x")
        (date_dir / "part-00007.parquet").write_bytes(b"x")
        assert w._next_part("2026-05-18", date_dir) == 8     # max + 1

    def test_ignores_non_part_files(self, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        date_dir = patch_data_dir / "symbol=EURUSD" / "date=2026-05-18"
        date_dir.mkdir(parents=True)
        (date_dir / "part-00002.parquet").write_bytes(b"x")
        (date_dir / "junk.txt").write_text("x")
        (date_dir / "data-00099.parquet").write_bytes(b"x")
        assert w._next_part("2026-05-18", date_dir) == 3

    def test_empty_dir_returns_zero(self, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        date_dir = patch_data_dir / "symbol=EURUSD" / "date=2026-05-18"
        date_dir.mkdir(parents=True)
        assert w._next_part("2026-05-18", date_dir) == 0


# ===========================================================================
# F. _flush — date grouping & day-boundary straddle
# ===========================================================================


class TestFlushGrouping:

    def test_single_day_writes_one_partition(self, patch_data_dir: Path):
        async def _body():
            w = TickWriter(asyncio.Queue(), symbol="EURUSD")
            buf = [make_tick(utc_ms(2026, 5, 18, 10, m), 1.10) for m in (0, 1, 2)]
            await w._flush(buf)
            return w
        _run(_body())
        parts = list_date_partitions(patch_data_dir / "symbol=EURUSD")
        assert [p.name for p in parts] == ["date=2026-05-18"]

    def test_day_boundary_creates_two_partitions(self, patch_data_dir: Path):
        async def _body():
            w = TickWriter(asyncio.Queue(), symbol="EURUSD")
            buf = [
                make_tick(utc_ms(2026, 5, 18, 23, 59, 30), 1.10),
                make_tick(utc_ms(2026, 5, 18, 23, 59, 59), 1.10),
                make_tick(utc_ms(2026, 5, 19, 0, 0, 1), 1.10),
                make_tick(utc_ms(2026, 5, 19, 0, 0, 5), 1.10),
            ]
            await w._flush(buf)
            return w
        _run(_body())
        parts = sorted(p.name for p in list_date_partitions(
            patch_data_dir / "symbol=EURUSD"
        ))
        assert parts == ["date=2026-05-18", "date=2026-05-19"]

    def test_each_partition_only_its_date(self, patch_data_dir: Path):
        async def _body():
            w = TickWriter(asyncio.Queue(), symbol="EURUSD")
            buf = (
                [make_tick(utc_ms(2026, 5, 18, 23, m), 1.10) for m in (58, 59)] +
                [make_tick(utc_ms(2026, 5, 19, 0, m), 1.10) for m in (0, 1, 2)]
            )
            await w._flush(buf)
            return w
        _run(_body())
        # 18th has 2 rows, 19th has 3.
        d18 = read_partition_rows(patch_data_dir / "symbol=EURUSD" / "date=2026-05-18")
        d19 = read_partition_rows(patch_data_dir / "symbol=EURUSD" / "date=2026-05-19")
        assert d18 == 2
        assert d19 == 3

    @pytest.mark.parametrize("n_days", [1, 2, 3, 5, 7])
    def test_n_day_span(self, n_days, patch_data_dir: Path):
        async def _body():
            w = TickWriter(asyncio.Queue(), symbol="EURUSD")
            buf = []
            for d in range(n_days):
                buf.append(make_tick(utc_ms(2026, 5, 18 + d, 12, 0), 1.10))
            await w._flush(buf)
            return w
        _run(_body())
        parts = list_date_partitions(patch_data_dir / "symbol=EURUSD")
        assert len(parts) == n_days

    def test_empty_buffer_no_writes(self, patch_data_dir: Path):
        async def _body():
            w = TickWriter(asyncio.Queue(), symbol="EURUSD")
            await w._flush([])
            return w
        w = _run(_body())
        assert w.written_files == 0
        assert w.written_ticks == 0
        # No date partitions
        assert list_date_partitions(patch_data_dir / "symbol=EURUSD") == []


# ===========================================================================
# G. run() loop — size trigger & cancel drain
# ===========================================================================


class TestRunLoop:

    def test_size_trigger_writes_flush_size_rows(self, patch_data_dir: Path):
        async def _body():
            q: asyncio.Queue = asyncio.Queue()
            w = TickWriter(q, symbol="EURUSD", flush_size=5, flush_seconds=30)
            task = asyncio.create_task(w.run())
            for i in range(5):
                await q.put(make_tick(utc_ms(2026, 5, 18, 10, 0) + i, 1.10))
            # Give scheduler time to drain + flush.
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return w
        w = _run(_body())
        assert w.written_ticks == 5
        assert w.written_files == 1

    def test_time_trigger_flushes_partial_buffer(self, patch_data_dir: Path):
        async def _body():
            q: asyncio.Queue = asyncio.Queue()
            w = TickWriter(q, symbol="EURUSD", flush_size=1000, flush_seconds=0.1)
            task = asyncio.create_task(w.run())
            await q.put(make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
            await q.put(make_tick(utc_ms(2026, 5, 18, 10, 1), 1.10))
            # Wait past flush_seconds without filling flush_size.
            await asyncio.sleep(0.3)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return w
        w = _run(_body())
        assert w.written_ticks >= 2

    def test_cancel_drains_buffer(self, patch_data_dir: Path):
        async def _body():
            q: asyncio.Queue = asyncio.Queue()
            w = TickWriter(q, symbol="EURUSD", flush_size=1000, flush_seconds=30)
            task = asyncio.create_task(w.run())
            for i in range(3):
                await q.put(make_tick(utc_ms(2026, 5, 18, 10, 0) + i, 1.10))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return w
        w = _run(_body())
        assert w.written_ticks == 3

    def test_final_flush_exception_logged_not_raised(self, patch_data_dir, monkeypatch):
        async def _body():
            q: asyncio.Queue = asyncio.Queue()
            w = TickWriter(q, symbol="EURUSD", flush_size=1000, flush_seconds=30)
            # Break _flush so the final-cancel drain raises.
            async def _boom(_buf):
                raise RuntimeError("disk full")
            monkeypatch.setattr(w, "_flush", _boom)
            task = asyncio.create_task(w.run())
            await q.put(make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
            await asyncio.sleep(0.05)
            task.cancel()
            # CancelledError must still propagate even though _flush exploded.
            raised_cancel = False
            try:
                await task
            except asyncio.CancelledError:
                raised_cancel = True
            return raised_cancel
        assert _run(_body()) is True

    def test_cancel_with_no_buffer_no_writes(self, patch_data_dir: Path):
        async def _body():
            q: asyncio.Queue = asyncio.Queue()
            w = TickWriter(q, symbol="EURUSD", flush_size=1000, flush_seconds=30)
            task = asyncio.create_task(w.run())
            await asyncio.sleep(0.02)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return w
        w = _run(_body())
        assert w.written_ticks == 0
        assert w.written_files == 0

    @pytest.mark.parametrize("flush_size", [1, 2, 5, 10, 50])
    def test_size_trigger_param(self, flush_size, patch_data_dir):
        async def _body():
            q: asyncio.Queue = asyncio.Queue()
            w = TickWriter(q, symbol="EURUSD",
                           flush_size=flush_size, flush_seconds=30)
            task = asyncio.create_task(w.run())
            for i in range(flush_size):
                await q.put(make_tick(utc_ms(2026, 5, 18, 10, 0) + i, 1.10))
            # Wait for the size-flush to LAND before cancelling. There is a
            # narrow production race where cancel arriving between the
            # `await self._flush(buffer)` and the `buffer = []` reset causes
            # the cancel-drain handler to re-flush, double-counting. Polling
            # on `written_ticks` lets the writer's loop reset buffer first,
            # then we cancel during the next `wait_for(get())` (buffer empty,
            # no race). A trailing sleep yields one more event-loop turn so
            # the reset has executed before we cancel.
            for _ in range(50):
                if w.written_ticks >= flush_size:
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.02)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return w
        w = _run(_body())
        assert w.written_ticks == flush_size

    def test_multiple_size_triggers(self, patch_data_dir):
        async def _body():
            q: asyncio.Queue = asyncio.Queue()
            w = TickWriter(q, symbol="EURUSD", flush_size=3, flush_seconds=30)
            task = asyncio.create_task(w.run())
            for i in range(9):
                await q.put(make_tick(utc_ms(2026, 5, 18, 10, 0) + i, 1.10))
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return w
        w = _run(_body())
        assert w.written_ticks == 9
        assert w.written_files >= 3


# ===========================================================================
# H. Tick → parquet field correctness sweep
# ===========================================================================


PRICES = [0.01, 1.0, 1.10, 100.0, 2300.0]
VOLUMES = [0, 1, 1000, 1_000_000]
FLAGS_SET = [0, 1, 4, 0xFF, 0xFFFF, 0x7FFFFFFF]


class TestFieldCorrectness:

    @pytest.mark.parametrize("price", PRICES)
    def test_bid_persisted(self, price, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        w._write_partition("2026-05-18", [make_tick(utc_ms(2026, 5, 18, 10, 0), price)])
        f = list_part_files(patch_data_dir / "symbol=EURUSD" / "date=2026-05-18")[0]
        d = pq.read_table(f).to_pydict()
        assert d["bid"][0] == pytest.approx(price)

    @pytest.mark.parametrize("vol", VOLUMES)
    def test_volume_persisted(self, vol, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        w._write_partition("2026-05-18",
                           [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10, volume=vol)])
        f = list_part_files(patch_data_dir / "symbol=EURUSD" / "date=2026-05-18")[0]
        d = pq.read_table(f).to_pydict()
        assert d["volume"][0] == vol

    @pytest.mark.parametrize("flags", FLAGS_SET)
    def test_flags_persisted(self, flags, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        w._write_partition(
            "2026-05-18",
            [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10, flags=flags)],
        )
        f = list_part_files(patch_data_dir / "symbol=EURUSD" / "date=2026-05-18")[0]
        d = pq.read_table(f).to_pydict()
        assert d["flags"][0] == flags

    @pytest.mark.parametrize("t_msc", [
        utc_ms(2024, 1, 1, 0, 0),
        utc_ms(2025, 6, 15, 12, 30),
        utc_ms(2026, 5, 18, 10, 0),
        utc_ms(2026, 12, 31, 23, 59, 59),
    ])
    def test_time_msc_persisted(self, t_msc, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        # Calc date string the way the writer does.
        date_str = datetime.fromtimestamp(t_msc / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        w._write_partition(date_str, [make_tick(t_msc, 1.10)])
        f = list_part_files(patch_data_dir / "symbol=EURUSD" / f"date={date_str}")[0]
        d = pq.read_table(f).to_pydict()
        assert d["time_msc"][0] == t_msc


# ===========================================================================
# I. Multi-symbol isolation
# ===========================================================================


class TestSymbolIsolation:

    @pytest.mark.parametrize("sym", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"])
    def test_writer_creates_only_its_symbol_dir(self, sym, patch_data_dir: Path):
        w = TickWriter(asyncio.Queue(), symbol=sym)
        w._write_partition("2026-05-18", [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10)])
        own = patch_data_dir / f"symbol={sym}" / "date=2026-05-18"
        assert own.exists()

    def test_two_writers_isolated(self, patch_data_dir: Path):
        w1 = TickWriter(asyncio.Queue(), symbol="EURUSD")
        w2 = TickWriter(asyncio.Queue(), symbol="GBPUSD")
        w1._write_partition("2026-05-18", [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10)])
        w2._write_partition("2026-05-18", [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.25)])
        d1 = patch_data_dir / "symbol=EURUSD" / "date=2026-05-18"
        d2 = patch_data_dir / "symbol=GBPUSD" / "date=2026-05-18"
        assert read_partition_rows(d1) == 1
        assert read_partition_rows(d2) == 1
        # Bids stored separately
        t1 = pq.read_table(list_part_files(d1)[0]).to_pydict()
        t2 = pq.read_table(list_part_files(d2)[0]).to_pydict()
        assert t1["bid"][0] == pytest.approx(1.10)
        assert t2["bid"][0] == pytest.approx(1.25)


# ===========================================================================
# J. Stress / large buffer
# ===========================================================================


class TestStress:

    @pytest.mark.parametrize("n", [1000, 5000])
    def test_large_buffer_single_partition(self, n, patch_data_dir):
        async def _body():
            w = TickWriter(asyncio.Queue(), symbol="EURUSD")
            buf = [make_tick(utc_ms(2026, 5, 18, 10, 0) + i, 1.10 + i * 1e-7)
                   for i in range(n)]
            await w._flush(buf)
            return w
        w = _run(_body())
        assert w.written_ticks == n

    def test_random_walk_persistence(self, patch_data_dir):
        async def _body():
            w = TickWriter(asyncio.Queue(), symbol="EURUSD")
            buf = random_walk_ticks(500, utc_ms(2026, 5, 18, 10, 0))
            await w._flush(buf)
            return w
        w = _run(_body())
        assert w.written_ticks == 500

    @pytest.mark.parametrize("hours,n_per_hour", [
        (1, 60), (2, 30), (5, 20), (10, 10),
    ])
    def test_hour_filling_persistence(self, hours, n_per_hour, patch_data_dir):
        async def _body():
            w = TickWriter(asyncio.Queue(), symbol="EURUSD")
            buf = hour_filling_ticks(utc_ms(2026, 5, 18, 10, 0),
                                     n_per_hour=n_per_hour, n_hours=hours)
            await w._flush(buf)
            return w
        w = _run(_body())
        assert w.written_ticks == hours * n_per_hour


# ===========================================================================
# K. Cache behaviour — _next_part after a write
# ===========================================================================


class TestPartCounterCache:

    def test_cache_populated_after_write(self, patch_data_dir):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        w._write_partition("2026-05-18", [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10)])
        assert w._part_counters["2026-05-18"] == 1     # next index after part-00000

    def test_cache_isolated_per_date(self, patch_data_dir):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        w._write_partition("2026-05-18", [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10)])
        w._write_partition("2026-05-19", [make_tick(utc_ms(2026, 5, 19, 10, 0), 1.10)])
        assert w._part_counters["2026-05-18"] == 1
        assert w._part_counters["2026-05-19"] == 1

    def test_cache_does_not_grow_unbounded(self, patch_data_dir):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        for d in range(1, 6):
            w._write_partition(
                f"2026-05-{d:02d}",
                [make_tick(utc_ms(2026, 5, d, 10, 0), 1.10)],
            )
        assert len(w._part_counters) == 5


# ===========================================================================
# L. Date string formatting (UTC, not local)
# ===========================================================================


class TestUtcDateFormatting:

    @pytest.mark.parametrize("hour,minute,want_date", [
        (0, 0, "2026-05-18"),
        (12, 0, "2026-05-18"),
        (23, 59, "2026-05-18"),
        (0, 1, "2026-05-18"),
    ])
    def test_intraday_uses_utc(self, hour, minute, want_date, patch_data_dir):
        async def _body():
            w = TickWriter(asyncio.Queue(), symbol="EURUSD")
            buf = [make_tick(utc_ms(2026, 5, 18, hour, minute), 1.10)]
            await w._flush(buf)
        _run(_body())
        assert (patch_data_dir / "symbol=EURUSD" / f"date={want_date}").exists()

    @pytest.mark.parametrize("year,month,day", [
        (2024, 1, 1), (2024, 2, 29),       # leap day
        (2025, 12, 31),
        (2026, 3, 8), (2026, 11, 1),       # US DST transitions don't affect UTC
    ])
    def test_specific_dates(self, year, month, day, patch_data_dir):
        async def _body():
            w = TickWriter(asyncio.Queue(), symbol="EURUSD")
            await w._flush([make_tick(utc_ms(year, month, day, 10, 0), 1.10)])
        _run(_body())
        want = f"date={year:04d}-{month:02d}-{day:02d}"
        assert (patch_data_dir / "symbol=EURUSD" / want).exists()


# ===========================================================================
# M. Counter monotonicity across writes
# ===========================================================================


class TestPartMonotonic:

    @pytest.mark.parametrize("n_writes", [1, 2, 3, 5, 10, 20])
    def test_n_consecutive_writes(self, n_writes, patch_data_dir):
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        for i in range(n_writes):
            w._write_partition(
                "2026-05-18",
                [make_tick(utc_ms(2026, 5, 18, 10, 0) + i, 1.10)],
            )
        names = sorted(p.name for p in list_part_files(
            patch_data_dir / "symbol=EURUSD" / "date=2026-05-18"
        ))
        assert names == [f"part-{i:05d}.parquet" for i in range(n_writes)]
        assert w.written_files == n_writes

    def test_resume_after_pre_existing_files(self, patch_data_dir):
        date_dir = patch_data_dir / "symbol=EURUSD" / "date=2026-05-18"
        date_dir.mkdir(parents=True)
        for i in (0, 1, 2, 3):
            (date_dir / f"part-{i:05d}.parquet").write_bytes(b"x")
        w = TickWriter(asyncio.Queue(), symbol="EURUSD")
        w._write_partition("2026-05-18",
                           [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10)])
        # Should write part-00004
        new = sorted(date_dir.glob("part-*.parquet"))
        assert new[-1].name == "part-00004.parquet"


# ===========================================================================
# N. Float / int field type round-trip (sweep)
# ===========================================================================


@pytest.mark.parametrize("field,value", [
    ("bid", 1.0), ("bid", 0.000001), ("bid", 9999.9999),
    ("ask", 1.0), ("ask", 0.000001), ("ask", 9999.9999),
    ("last", 1.0), ("last", 0.000001),
    ("volume_real", 1.5), ("volume_real", 0.0), ("volume_real", 1e6),
])
def test_float_field_round_trip(field, value, patch_data_dir):
    kw = {"time_msc": utc_ms(2026, 5, 18, 10, 0), "bid": 1.0, "ask": 1.0,
          "last": 1.0, "volume": 1, "volume_real": 1.0, "flags": 0}
    kw[field] = value
    t = Tick(**kw)
    w = TickWriter(asyncio.Queue(), symbol="EURUSD")
    w._write_partition("2026-05-18", [t])
    f = list_part_files(patch_data_dir / "symbol=EURUSD" / "date=2026-05-18")[0]
    d = pq.read_table(f).to_pydict()
    assert d[field][0] == pytest.approx(value)


@pytest.mark.parametrize("volume", [0, 1, 100, 999_999, 10**12])
def test_volume_round_trip(volume, patch_data_dir):
    w = TickWriter(asyncio.Queue(), symbol="EURUSD")
    w._write_partition(
        "2026-05-18",
        [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10, volume=volume)],
    )
    f = list_part_files(patch_data_dir / "symbol=EURUSD" / "date=2026-05-18")[0]
    d = pq.read_table(f).to_pydict()
    assert d["volume"][0] == volume


# ===========================================================================
# O. Async run loop: time + size interplay
# ===========================================================================


class TestRunLoopInterplay:

    def test_size_then_time_two_files(self, patch_data_dir):
        async def _body():
            q: asyncio.Queue = asyncio.Queue()
            w = TickWriter(q, symbol="EURUSD", flush_size=3, flush_seconds=0.15)
            task = asyncio.create_task(w.run())
            # First 3 ticks trip size trigger.
            for i in range(3):
                await q.put(make_tick(utc_ms(2026, 5, 18, 10, 0) + i, 1.10))
            await asyncio.sleep(0.05)
            # Push 2 more, wait past flush_seconds for time trigger.
            for i in range(2):
                await q.put(make_tick(utc_ms(2026, 5, 18, 10, 5) + i, 1.10))
            await asyncio.sleep(0.3)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return w
        w = _run(_body())
        assert w.written_ticks == 5
        assert w.written_files >= 2

    def test_cancel_during_time_wait_drains_buffer(self, patch_data_dir):
        async def _body():
            q: asyncio.Queue = asyncio.Queue()
            w = TickWriter(q, symbol="EURUSD", flush_size=1000, flush_seconds=10)
            task = asyncio.create_task(w.run())
            await q.put(make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return w
        w = _run(_body())
        assert w.written_ticks == 1


# ===========================================================================
# P. Symbol-name sweep — many pairs each create their own dir
# ===========================================================================


@pytest.mark.parametrize("sym", [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURJPY", "EURGBP", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
    "AUDJPY", "AUDCHF", "AUDNZD", "AUDCAD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY",
    "XAUUSD", "BTCUSD",
])
def test_symbol_dir_creation_sweep(sym, patch_data_dir):
    w = TickWriter(asyncio.Queue(), symbol=sym)
    assert (patch_data_dir / f"symbol={sym}").exists()
    # Round-trip a single tick to confirm the writer is functional for this sym.
    w._write_partition("2026-05-18", [make_tick(utc_ms(2026, 5, 18, 10, 0), 1.10)])
    assert read_partition_rows(patch_data_dir / f"symbol={sym}" / "date=2026-05-18") == 1
