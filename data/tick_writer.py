"""Async tick consumer. Drains queue into Hive-partitioned parquet."""

from __future__ import annotations
import asyncio
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import settings
from data.tick_collector import Tick
from utils.logger import logger


TICK_SCHEMA: pa.Schema = pa.schema([
    ("time_msc", pa.int64()),
    ("bid", pa.float64()),
    ("ask", pa.float64()),
    ("last", pa.float64()),
    ("volume", pa.uint64()),
    ("volume_real", pa.float64()),
    ("flags", pa.uint32()),
])

_PART_PATTERN = re.compile(r"^part-(\d{5})\.parquet$")


class TickWriter:
    def __init__(
        self,
        queue: asyncio.Queue[Tick],
        symbol: str | None = None,
        flush_size: int = 1000,
        flush_seconds: float = 5.0,
        compression: str = "snappy",
    ) -> None:
        self._queue = queue
        self._symbol = symbol or settings.symbol
        self._flush_size = flush_size
        self._flush_seconds = flush_seconds
        self._compression = compression
        self._symbol_dir = settings.data_dir / f"symbol={self._symbol}"
        self._symbol_dir.mkdir(parents=True, exist_ok=True)
        self._part_counters: dict[str, int] = {}
        self._written_ticks: int = 0
        self._written_files: int = 0

    async def run(self) -> None:
        """Main consumer loop. Cancellable; final buffer is flushed on cancel."""
        buffer: list[Tick] = []
        last_flush = time.monotonic()
        logger.info(
            f"TickWriter started | symbol={self._symbol} "
            f"flush_size={self._flush_size} flush_seconds={self._flush_seconds} "
            f"out={self._symbol_dir}"
        )
        try:
            while True:
                remaining = max(
                    0.05, self._flush_seconds - (time.monotonic() - last_flush)
                )
                try:
                    tick = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    buffer.append(tick)
                except asyncio.TimeoutError:
                    pass

                now = time.monotonic()
                size_trigger = len(buffer) >= self._flush_size
                time_trigger = (
                    (now - last_flush) >= self._flush_seconds and buffer
                )
                if size_trigger or time_trigger:
                    await self._flush(buffer)
                    buffer = []
                    last_flush = now
        except asyncio.CancelledError:
            if buffer:
                logger.info(f"TickWriter draining final {len(buffer)} ticks")
                try:
                    await self._flush(buffer)
                except Exception as exc:
                    logger.exception(f"final flush failed: {exc}")
            logger.info(
                f"TickWriter stopped | rows={self._written_ticks} "
                f"files={self._written_files}"
            )
            raise

    async def _flush(self, buffer: list[Tick]) -> None:
        # Group by UTC date — a flush window may straddle midnight.
        grouped: dict[str, list[Tick]] = {}
        for t in buffer:
            d = datetime.fromtimestamp(
                t.time_msc / 1000.0, tz=timezone.utc
            ).strftime("%Y-%m-%d")
            grouped.setdefault(d, []).append(t)

        for date_str, ticks in grouped.items():
            await asyncio.to_thread(self._write_partition, date_str, ticks)

    def _write_partition(self, date_str: str, ticks: list[Tick]) -> None:
        date_dir = self._symbol_dir / f"date={date_str}"
        date_dir.mkdir(parents=True, exist_ok=True)
        part_n = self._next_part(date_str, date_dir)
        path = date_dir / f"part-{part_n:05d}.parquet"

        table = pa.table(
            {
                "time_msc": pa.array(
                    [t.time_msc for t in ticks], type=pa.int64()
                ),
                "bid": pa.array(
                    [t.bid for t in ticks], type=pa.float64()
                ),
                "ask": pa.array(
                    [t.ask for t in ticks], type=pa.float64()
                ),
                "last": pa.array(
                    [t.last for t in ticks], type=pa.float64()
                ),
                "volume": pa.array(
                    [t.volume for t in ticks], type=pa.uint64()
                ),
                "volume_real": pa.array(
                    [t.volume_real for t in ticks], type=pa.float64()
                ),
                "flags": pa.array(
                    [t.flags for t in ticks], type=pa.uint32()
                ),
            },
            schema=TICK_SCHEMA,
        )

        pq.write_table(table, path, compression=self._compression)
        self._part_counters[date_str] = part_n + 1
        self._written_ticks += len(ticks)
        self._written_files += 1
        logger.debug(
            f"wrote {path.name} | rows={len(ticks)} date={date_str}"
        )

    def _next_part(self, date_str: str, date_dir: Path) -> int:
        cached = self._part_counters.get(date_str)
        if cached is not None:
            return cached
        max_n = -1
        for p in date_dir.iterdir():
            m = _PART_PATTERN.match(p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
        return max_n + 1

    @property
    def written_ticks(self) -> int:
        return self._written_ticks

    @property
    def written_files(self) -> int:
        return self._written_files
