"""Phase 9 — LiveBarPoller tests.

The poller bridges MT5 H1 bars into the GriffLiveEngine scan cycle.
We mock `mt5.copy_rates_from_pos` so tests don't need a real terminal.
"""

from __future__ import annotations
import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from data.live_bar_poller import LiveBarPoller


# ----- helpers --------------------------------------------------------


_MT5_RATES_DTYPE = np.dtype([
    ("time", "i8"),
    ("open", "f8"),
    ("high", "f8"),
    ("low", "f8"),
    ("close", "f8"),
    ("tick_volume", "i8"),
    ("spread", "i4"),
    ("real_volume", "i8"),
])


def _fake_rates(start_hour_utc: datetime, count: int) -> np.ndarray:
    """Build an MT5-shaped ndarray of `count` consecutive H1 bars."""
    rows = []
    for i in range(count):
        bar_open = start_hour_utc + timedelta(hours=i)
        epoch = int(bar_open.timestamp())
        rows.append((
            epoch, 1.10 + i * 0.0001, 1.105 + i * 0.0001,
            1.099 + i * 0.0001, 1.103 + i * 0.0001,
            100 + i, 2, 0,
        ))
    return np.array(rows, dtype=_MT5_RATES_DTYPE)


def _fake_mt5(rates_by_pair: dict) -> MagicMock:
    mt5 = MagicMock()
    mt5.TIMEFRAME_H1 = 16385  # MT5 constant; arbitrary in the mock

    def _copy(symbol, timeframe, start_pos, count):
        if symbol not in rates_by_pair:
            return np.empty(0, dtype=_MT5_RATES_DTYPE)
        return rates_by_pair[symbol]

    mt5.copy_rates_from_pos = MagicMock(side_effect=_copy)
    return mt5


T0 = datetime(2026, 5, 18, 10, 0, 0, tzinfo=timezone.utc)


# ====================================================== unit


class TestPollOnce(unittest.TestCase):
    def test_returns_empty_when_mt5_has_nothing(self):
        mt5 = _fake_mt5({})
        p = LiveBarPoller(
            pairs=("EURUSD",), mt5_module=mt5, history_bars=5, poll_sec=1,
        )
        self.assertEqual(p.poll_once(), {})
        self.assertEqual(p.buffer["EURUSD"], [])

    def test_first_poll_emits_latest_close(self):
        mt5 = _fake_mt5({"EURUSD": _fake_rates(T0, 3)})
        p = LiveBarPoller(
            pairs=("EURUSD",), mt5_module=mt5, history_bars=3, poll_sec=1,
        )
        new = p.poll_once()
        self.assertIn("EURUSD", new)
        # Latest bar opens at T0 + 2h.
        expected_msc = int((T0 + timedelta(hours=2)).timestamp() * 1000)
        self.assertEqual(new["EURUSD"].time_msc, expected_msc)
        self.assertEqual(len(p.buffer["EURUSD"]), 3)

    def test_same_bars_twice_no_duplicate(self):
        mt5 = _fake_mt5({"EURUSD": _fake_rates(T0, 3)})
        p = LiveBarPoller(
            pairs=("EURUSD",), mt5_module=mt5, history_bars=3, poll_sec=1,
        )
        p.poll_once()
        self.assertEqual(p.poll_once(), {})

    def test_new_bar_after_first_emits(self):
        rates_by_pair = {"EURUSD": _fake_rates(T0, 3)}
        mt5 = _fake_mt5(rates_by_pair)
        p = LiveBarPoller(
            pairs=("EURUSD",), mt5_module=mt5, history_bars=3, poll_sec=1,
        )
        p.poll_once()
        # Now MT5 returns one more bar.
        rates_by_pair["EURUSD"] = _fake_rates(T0, 4)
        new = p.poll_once()
        expected_msc = int((T0 + timedelta(hours=3)).timestamp() * 1000)
        self.assertIn("EURUSD", new)
        self.assertEqual(new["EURUSD"].time_msc, expected_msc)

    def test_multi_pair_independent(self):
        mt5 = _fake_mt5({
            "EURUSD": _fake_rates(T0, 3),
            "AUDJPY": _fake_rates(T0, 2),
        })
        p = LiveBarPoller(
            pairs=("EURUSD", "AUDJPY"),
            mt5_module=mt5, history_bars=3, poll_sec=1,
        )
        new = p.poll_once()
        self.assertEqual(set(new.keys()), {"EURUSD", "AUDJPY"})


# ====================================================== run() loop


class TestRunLoop(unittest.IsolatedAsyncioTestCase):
    async def test_stop_exits_cleanly(self):
        mt5 = _fake_mt5({})
        p = LiveBarPoller(
            pairs=("EURUSD",), mt5_module=mt5, history_bars=5, poll_sec=0.05,
        )
        stop = asyncio.Event()
        engine = MagicMock()
        engine.process_scan_cycle = AsyncMock()
        engine.maintain_open = AsyncMock()

        task = asyncio.create_task(p.run(
            engine=engine, stop=stop,
            account_provider=lambda: MagicMock(),
            prices_provider=lambda: ({}, {}),
        ))
        await asyncio.sleep(0.1)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)
        # No bars came → process_scan_cycle was never called.
        engine.process_scan_cycle.assert_not_called()

    async def test_new_bar_invokes_engine(self):
        rates_by_pair = {"EURUSD": _fake_rates(T0, 3)}
        mt5 = _fake_mt5(rates_by_pair)
        p = LiveBarPoller(
            pairs=("EURUSD",), mt5_module=mt5, history_bars=3, poll_sec=0.05,
        )
        stop = asyncio.Event()
        engine = MagicMock()
        engine.process_scan_cycle = AsyncMock()
        engine.maintain_open = AsyncMock()

        task = asyncio.create_task(p.run(
            engine=engine, stop=stop,
            account_provider=lambda: "ACCT",
            prices_provider=lambda: ({"EURUSD": 1.1}, {"EURUSD": 1.099}),
        ))
        await asyncio.sleep(0.1)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)
        engine.process_scan_cycle.assert_awaited()
        engine.maintain_open.assert_awaited()

    async def test_swallows_poll_exception(self):
        # First poll raises; subsequent poll succeeds. Loop should not die.
        mt5 = MagicMock()
        mt5.TIMEFRAME_H1 = 16385
        ok_rates = _fake_rates(T0, 1)
        mt5.copy_rates_from_pos = MagicMock(
            side_effect=[RuntimeError("boom"), ok_rates, ok_rates, ok_rates],
        )
        p = LiveBarPoller(
            pairs=("EURUSD",), mt5_module=mt5, history_bars=1, poll_sec=0.02,
        )
        stop = asyncio.Event()
        engine = MagicMock()
        engine.process_scan_cycle = AsyncMock()
        engine.maintain_open = AsyncMock()
        task = asyncio.create_task(p.run(
            engine=engine, stop=stop,
            account_provider=lambda: "ACCT",
            prices_provider=lambda: ({}, {}),
        ))
        await asyncio.sleep(0.15)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)
        # Got past the exception and made at least one engine call.
        self.assertGreaterEqual(
            engine.process_scan_cycle.await_count, 1,
        )


if __name__ == "__main__":
    unittest.main()
