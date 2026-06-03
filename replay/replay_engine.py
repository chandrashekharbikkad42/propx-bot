"""Offline tick producer. Reads Hive-partitioned parquet, emits ticks to an asyncio.Queue.

Same producer interface as TickCollector — downstream consumers (writer, strategy)
cannot tell the difference between live and replay. NEVER imports MetaTrader5.
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

from config.settings import settings
from data.tick_collector import Tick
from data.tick_writer import TICK_SCHEMA
from utils.logger import logger


@dataclass(frozen=True)
class ReplayConfig:
    """Immutable replay configuration.

    speed semantics:
        0.0   → max-speed (no inter-tick sleep)
        1.0   → real-time (1× original wall-clock pacing)
        >1.0  → N× faster than real-time
        <1.0  → slower than real-time
    """
    symbol: str
    date: str  # YYYY-MM-DD
    speed: float = 1.0
    batch_size: int = 500
    # If provided, replay only these files (skip the partition glob). Useful
    # when something else is concurrently appending to the partition and the
    # caller wants a deterministic snapshot.
    files: Optional[tuple[Path, ...]] = None


class ReplayEngine:
    def __init__(self, config: ReplayConfig, queue: asyncio.Queue[Tick]) -> None:
        self._config = config
        self._queue = queue
        self._last_emitted_msc: int = 0
        self._emitted: int = 0
        self._skipped: int = 0

    async def run(self) -> None:
        if self._config.files is not None:
            files = list(self._config.files)
        else:
            partition_dir = (
                settings.data_dir
                / f"symbol={self._config.symbol}"
                / f"date={self._config.date}"
            )
            if not partition_dir.exists():
                raise FileNotFoundError(f"Partition not found: {partition_dir}")
            files = sorted(partition_dir.glob("part-*.parquet"))
        if not files:
            raise ValueError(f"No parquet files for replay")

        speed = self._config.speed
        mode_label = "max-speed" if speed == 0.0 else f"{speed:.2f}x"
        logger.info(
            f"ReplayEngine started | symbol={self._config.symbol} "
            f"date={self._config.date} files={len(files)} mode={mode_label}"
        )

        columns = [f.name for f in TICK_SCHEMA]
        wall_start: Optional[float] = None
        first_tick_msc: Optional[int] = None

        try:
            for file in files:
                pf = await asyncio.to_thread(pq.ParquetFile, str(file))
                batch_iter = pf.iter_batches(
                    batch_size=self._config.batch_size, columns=columns
                )
                while True:
                    batch = await asyncio.to_thread(next, batch_iter, None)
                    if batch is None:
                        break

                    time_msc = batch.column("time_msc").to_numpy()
                    bid = batch.column("bid").to_numpy()
                    ask = batch.column("ask").to_numpy()
                    last = batch.column("last").to_numpy()
                    volume = batch.column("volume").to_numpy()
                    volume_real = batch.column("volume_real").to_numpy()
                    flags = batch.column("flags").to_numpy()

                    for i in range(len(time_msc)):
                        tmsc = int(time_msc[i])

                        # Defensive dedup against duplicates / non-monotonic data.
                        if tmsc <= self._last_emitted_msc:
                            self._skipped += 1
                            continue

                        # Wall-clock pacing (skipped in max-speed mode).
                        if speed > 0.0:
                            if first_tick_msc is None:
                                first_tick_msc = tmsc
                                wall_start = time.monotonic()
                            else:
                                target_elapsed = (
                                    (tmsc - first_tick_msc) / 1000.0 / speed
                                )
                                actual_elapsed = time.monotonic() - wall_start
                                delay = target_elapsed - actual_elapsed
                                if delay > 0:
                                    await asyncio.sleep(delay)

                        tick = Tick(
                            time_msc=tmsc,
                            bid=float(bid[i]),
                            ask=float(ask[i]),
                            last=float(last[i]),
                            volume=int(volume[i]),
                            volume_real=float(volume_real[i]),
                            flags=int(flags[i]),
                        )
                        # In replay we want lossless delivery, so block if the
                        # consumer is slow rather than drop (unlike the live path).
                        await self._queue.put(tick)
                        self._emitted += 1
                        self._last_emitted_msc = tmsc

                        # Cooperative yield: in max-speed mode put() rarely
                        # blocks, so without this the inner loop would starve
                        # the event loop on large partitions.
                        if speed == 0.0 and (self._emitted & 0xFF) == 0:
                            await asyncio.sleep(0)
        except asyncio.CancelledError:
            logger.info(
                f"ReplayEngine cancelled | emitted={self._emitted} "
                f"skipped={self._skipped}"
            )
            raise

        logger.success(
            f"ReplayEngine finished | emitted={self._emitted} "
            f"skipped={self._skipped} files={len(files)}"
        )

    @property
    def emitted(self) -> int:
        return self._emitted

    @property
    def skipped(self) -> int:
        return self._skipped
