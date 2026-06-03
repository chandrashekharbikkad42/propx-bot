"""In-memory stand-in for MT5Connector used by TickCollector tests.

Only the methods TickCollector actually calls are implemented:
  - last_tick_msc()         → cursor seed at startup
  - copy_ticks_from(c, n)   → returns the next batch of mock ticks

The fake is configurable per test:
  - `seed_msc`        : value returned by last_tick_msc()
  - `batches`         : list of ndarrays returned by successive copy_ticks_from
                        calls. Defaults to a single empty batch.
  - `raise_on_batch`  : map {batch_index: Exception} — raise on those calls.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence

import numpy as np

from tests.data.fixtures.synthetic_ticks import MT5_TICK_DTYPE


class FakeConnector:
    def __init__(self, symbol: str = "EURUSD",
                 seed_msc: int = 0,
                 batches: Optional[Sequence[np.ndarray]] = None,
                 raise_on_batch: Optional[Dict[int, Exception]] = None) -> None:
        self.symbol = symbol
        self._seed_msc = int(seed_msc)
        self._batches: List[np.ndarray] = list(batches) if batches else []
        self._raise_on_batch: Dict[int, Exception] = dict(raise_on_batch or {})
        self.calls: List[tuple] = []           # (cursor, count) per copy call
        self.seed_calls: int = 0

    def last_tick_msc(self) -> int:
        self.seed_calls += 1
        return self._seed_msc

    def copy_ticks_from(self, from_msc: int, count: int) -> np.ndarray:
        idx = len(self.calls)
        self.calls.append((from_msc, count))
        exc = self._raise_on_batch.get(idx)
        if exc is not None:
            raise exc
        if idx >= len(self._batches):
            return np.empty(0, dtype=MT5_TICK_DTYPE)
        return self._batches[idx]

    # ---- builder helpers (chainable) -------------------------------

    def queue_batch(self, batch: np.ndarray) -> "FakeConnector":
        self._batches.append(batch)
        return self

    def queue_exception(self, exc: Exception) -> "FakeConnector":
        idx = len(self._batches)
        self._raise_on_batch[idx] = exc
        self._batches.append(np.empty(0, dtype=MT5_TICK_DTYPE))
        return self
