"""End-to-end replay verification: emitted ticks must equal the on-disk recording."""

from __future__ import annotations
import asyncio
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from config.settings import settings
from data.tick_collector import Tick
from replay.replay_engine import ReplayConfig, ReplayEngine
from replay.integrity_checker import check_partition


# Partition produced by the Phase 1 live capture (and any subsequent live
# captures on the same UTC date). Test skips if absent so fresh checkouts
# don't error.
TARGET_DATE = "2026-05-12"


def _partition_dir(date: str) -> Path:
    return settings.data_dir / f"symbol={settings.symbol}" / f"date={date}"


def _snapshot_files(date: str) -> tuple[Path, ...]:
    """Freeze the file list once, so a concurrent live capture appending new
    parts cannot cause replay-vs-disk row counts to disagree."""
    return tuple(sorted(_partition_dir(date).glob("part-*.parquet")))


def _row_count_for(files: tuple[Path, ...]) -> int:
    return sum(pq.read_metadata(p).num_rows for p in files)


async def _drive_replay(
    date: str, speed: float, files: tuple[Path, ...]
) -> tuple[list[Tick], int]:
    """Run a replay, drain it, return (emitted_ticks, skipped_count)."""
    queue: asyncio.Queue[Tick] = asyncio.Queue(maxsize=100_000)
    config = ReplayConfig(
        symbol=settings.symbol, date=date, speed=speed, files=files
    )
    engine = ReplayEngine(config, queue)
    engine_task = asyncio.create_task(engine.run())

    collected: list[Tick] = []
    while True:
        if engine_task.done() and queue.empty():
            break
        try:
            tick = await asyncio.wait_for(queue.get(), timeout=0.5)
            collected.append(tick)
        except asyncio.TimeoutError:
            if engine_task.done():
                break

    if engine_task.exception():
        raise engine_task.exception()
    return collected, engine.skipped


class TestReplayMatchesRecording(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        if not _partition_dir(TARGET_DATE).exists():
            self.skipTest(f"No partition for date={TARGET_DATE}")
        self.files = _snapshot_files(TARGET_DATE)
        if not self.files:
            self.skipTest(f"Partition date={TARGET_DATE} is empty")
        self.expected_rows = _row_count_for(self.files)

    async def test_emitted_plus_skipped_equals_on_disk(self):
        # Replay applies defensive `<=` dedup, so same-msc tick co-events are
        # skipped rather than emitted. Verify no row is silently lost:
        # every on-disk row is accounted for as either emitted or skipped.
        ticks, skipped = await _drive_replay(
            TARGET_DATE, speed=0.0, files=self.files
        )
        self.assertEqual(
            len(ticks) + skipped, self.expected_rows,
            f"emitted={len(ticks)} skipped={skipped} != on-disk={self.expected_rows}",
        )

    async def test_replay_strictly_monotonic_time_msc(self):
        ticks, _ = await _drive_replay(
            TARGET_DATE, speed=0.0, files=self.files
        )
        for i in range(1, len(ticks)):
            self.assertLess(
                ticks[i - 1].time_msc, ticks[i].time_msc,
                f"non-monotonic at index {i}",
            )

    async def test_on_disk_schema_and_non_decreasing(self):
        # On-disk format guarantees: frozen schema, non-decreasing time_msc.
        # Same-msc co-events ARE permitted (sub-ms microstructure); replay's
        # dedup handles them — see test_emitted_plus_skipped_equals_on_disk.
        report = check_partition(settings.symbol, TARGET_DATE)
        self.assertTrue(report.schema_match, "on-disk schema mismatch")
        self.assertTrue(report.monotonic, "time_msc went backwards on disk")


class TestReplayDedup(unittest.IsolatedAsyncioTestCase):
    """When the engine's last_emitted cursor is pre-seeded, ticks at or before
    that cursor must be skipped (defensive dedup against duplicate/out-of-order data)."""

    def setUp(self):
        if not _partition_dir(TARGET_DATE).exists():
            self.skipTest(f"No partition for date={TARGET_DATE}")
        self.files = _snapshot_files(TARGET_DATE)
        if not self.files:
            self.skipTest(f"Partition date={TARGET_DATE} is empty")

    async def test_skips_when_cursor_already_advanced(self):
        first_table = pq.read_table(self.files[0], columns=["time_msc"])
        first_msc = int(first_table.column("time_msc")[0].as_py())

        queue: asyncio.Queue[Tick] = asyncio.Queue(maxsize=100_000)
        engine = ReplayEngine(
            ReplayConfig(
                symbol=settings.symbol, date=TARGET_DATE,
                speed=0.0, files=self.files,
            ),
            queue,
        )
        engine._last_emitted_msc = first_msc
        engine_task = asyncio.create_task(engine.run())

        consumed = 0
        while True:
            if engine_task.done() and queue.empty():
                break
            try:
                _ = await asyncio.wait_for(queue.get(), timeout=0.5)
                consumed += 1
            except asyncio.TimeoutError:
                if engine_task.done():
                    break

        if engine_task.exception():
            raise engine_task.exception()

        self.assertGreater(engine.skipped, 0, "expected at least one skip")
        self.assertEqual(consumed, engine.emitted)


if __name__ == "__main__":
    unittest.main()
