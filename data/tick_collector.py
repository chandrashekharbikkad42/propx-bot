"""Async tick producer. Polls MT5 via connector, dedups by time_msc, pushes to queue."""

from __future__ import annotations
import asyncio
from dataclasses import dataclass

from data.mt5_connector import MT5Connector
from utils.logger import logger


@dataclass(frozen=True)
class Tick:
    """Single tick. Immutable value object passed producer→consumer."""
    time_msc: int
    bid: float
    ask: float
    last: float
    volume: int
    volume_real: float
    flags: int


class TickCollector:
    def __init__(
        self,
        connector: MT5Connector,
        queue: asyncio.Queue[Tick],
        poll_interval_ms: int = 50,
        batch_size: int = 2000,
        drop_log_every: int = 100,
    ) -> None:
        self._conn = connector
        self._queue = queue
        self._poll_interval = poll_interval_ms / 1000.0
        self._batch_size = batch_size
        self._drop_log_every = drop_log_every
        self._cursor_msc: int = 0
        self._collected: int = 0
        self._dropped: int = 0

    async def run(self) -> None:
        """Main producer loop. Cancellable."""
        self._cursor_msc = await asyncio.to_thread(self._conn.last_tick_msc)
        logger.info(
            f"TickCollector started | symbol={self._conn.symbol} "
            f"cursor_msc={self._cursor_msc} poll_ms={int(self._poll_interval * 1000)} "
            f"batch={self._batch_size}"
        )
        try:
            while True:
                saturated = False
                try:
                    saturated = await self._poll_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception(f"tick poll failed: {exc}")
                    await asyncio.sleep(self._poll_interval * 4)
                    continue
                # If the batch came back full, skip the sleep and catch up.
                if not saturated:
                    await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            logger.info(
                f"TickCollector stopped | collected={self._collected} "
                f"dropped={self._dropped}"
            )
            raise

    async def _poll_once(self) -> bool:
        """Fetch one batch, enqueue new ticks. Returns True if batch was at capacity."""
        ticks = await asyncio.to_thread(
            self._conn.copy_ticks_from, self._cursor_msc, self._batch_size
        )
        if ticks is None or len(ticks) == 0:
            return False

        # Dedup: strictly newer than current cursor (drops the boundary tick already seen).
        mask = ticks["time_msc"] > self._cursor_msc
        new_ticks = ticks[mask]
        if len(new_ticks) == 0:
            return len(ticks) >= self._batch_size

        for row in new_ticks:
            t = Tick(
                time_msc=int(row["time_msc"]),
                bid=float(row["bid"]),
                ask=float(row["ask"]),
                last=float(row["last"]),
                volume=int(row["volume"]),
                volume_real=float(row["volume_real"]),
                flags=int(row["flags"]),
            )
            try:
                self._queue.put_nowait(t)
                self._collected += 1
            except asyncio.QueueFull:
                self._dropped += 1
                if self._dropped % self._drop_log_every == 1:
                    logger.warning(
                        f"queue full | dropped_total={self._dropped} "
                        f"qsize={self._queue.qsize()}/{self._queue.maxsize}"
                    )

        self._cursor_msc = int(new_ticks["time_msc"].max())
        return len(ticks) >= self._batch_size

    @property
    def collected(self) -> int:
        return self._collected

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def cursor_msc(self) -> int:
        return self._cursor_msc
