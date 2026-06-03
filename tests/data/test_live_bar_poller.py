"""LiveBarPoller — exhaustive unit tests (no live MT5).

Targets:
  - Constructor defaults, bounds clamping
  - fetch_closed_bars: ndarray → Bar list, empty / None handling
  - poll_once: per-pair newest-close detection, dedup, multi-pair
  - run(): stop event, exception swallowed, engine call sequencing
  - Buffer + last_bar_msc views (snapshot, immutability of caller-visible dict)
  - Edge: stale bar (older than last), future bar, only-one-bar history
"""

from __future__ import annotations
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from data.bar_aggregator import Bar
from data.live_bar_poller import LiveBarPoller
from tests.data.fixtures.synthetic_ticks import (
    EIGHT_PAIRS, MT5_RATES_DTYPE, consecutive_h1_rates, mt5_rates_array,
)


T0 = datetime(2026, 5, 18, 10, 0, 0, tzinfo=timezone.utc)


def _fake_mt5(rates_by_pair: Dict[str, np.ndarray] | None = None) -> MagicMock:
    m = MagicMock()
    m.TIMEFRAME_H1 = 16385
    # Bind by reference — tests can mutate the dict they passed in to model
    # MT5 returning fresh bars across polls.
    table = rates_by_pair if rates_by_pair is not None else {}

    def _copy(symbol, timeframe, start_pos, count):
        if symbol not in table:
            return np.empty(0, dtype=MT5_RATES_DTYPE)
        return table[symbol]

    m.copy_rates_from_pos = MagicMock(side_effect=_copy)
    m._table = table        # exposed for tests that want to introspect/mutate
    return m


def _engine_pair():
    eng = MagicMock()
    eng.process_scan_cycle = AsyncMock()
    eng.maintain_open = AsyncMock()
    return eng


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# A. Constructor
# ===========================================================================


class TestConstructor:

    def test_pairs_stored_as_tuple(self):
        p = LiveBarPoller(pairs=["EURUSD", "GBPUSD"], mt5_module=_fake_mt5())
        assert isinstance(p._pairs, tuple)
        assert p._pairs == ("EURUSD", "GBPUSD")

    @pytest.mark.parametrize("pairs", [
        ("EURUSD",),
        ("EURUSD", "GBPUSD"),
        ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
        EIGHT_PAIRS,
    ])
    def test_pair_lists_supported(self, pairs):
        p = LiveBarPoller(pairs=pairs, mt5_module=_fake_mt5())
        assert p._pairs == tuple(pairs)

    @pytest.mark.parametrize("hist", [1, 5, 25, 50, 100, 500])
    def test_history_bars_stored(self, hist):
        p = LiveBarPoller(pairs=("X",), mt5_module=_fake_mt5(), history_bars=hist)
        assert p._history_bars == hist

    @pytest.mark.parametrize("hist", [0, -1, -100])
    def test_history_bars_clamped_to_one(self, hist):
        p = LiveBarPoller(pairs=("X",), mt5_module=_fake_mt5(), history_bars=hist)
        assert p._history_bars == 1

    @pytest.mark.parametrize("sec", [0.01, 0.5, 1.0, 30.0, 300.0])
    def test_poll_sec_stored(self, sec):
        p = LiveBarPoller(pairs=("X",), mt5_module=_fake_mt5(), poll_sec=sec)
        assert p._poll_sec == sec

    @pytest.mark.parametrize("sec", [0.0, -1.0, -100.0])
    def test_poll_sec_clamped_floor(self, sec):
        p = LiveBarPoller(pairs=("X",), mt5_module=_fake_mt5(), poll_sec=sec)
        assert p._poll_sec == 0.01

    def test_initial_last_bar_msc_zero(self):
        p = LiveBarPoller(pairs=("A", "B"), mt5_module=_fake_mt5())
        assert p.last_bar_msc == {"A": 0, "B": 0}

    def test_initial_buffers_empty(self):
        p = LiveBarPoller(pairs=("A", "B"), mt5_module=_fake_mt5())
        assert p.buffer == {"A": [], "B": []}

    def test_history_bars_int_coerce(self):
        p = LiveBarPoller(pairs=("X",), mt5_module=_fake_mt5(), history_bars=5.7)
        assert p._history_bars == 5


# ===========================================================================
# B. fetch_closed_bars
# ===========================================================================


class TestFetchClosedBars:

    def test_empty_when_missing(self):
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=_fake_mt5({}))
        assert p.fetch_closed_bars("EURUSD") == []

    def test_returns_list_of_bars(self):
        m = _fake_mt5({"EURUSD": consecutive_h1_rates(T0, 3)})
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m)
        bars = p.fetch_closed_bars("EURUSD")
        assert len(bars) == 3
        assert all(isinstance(b, Bar) for b in bars)

    def test_skips_current_forming_bar(self):
        m = _fake_mt5({"EURUSD": consecutive_h1_rates(T0, 3)})
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m, history_bars=3)
        p.fetch_closed_bars("EURUSD")
        # start_pos=1 means "skip the currently-forming bar".
        args, kwargs = m.copy_rates_from_pos.call_args
        assert args[2] == 1                # start_pos

    def test_history_bars_request(self):
        m = _fake_mt5({"EURUSD": consecutive_h1_rates(T0, 5)})
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m, history_bars=5)
        p.fetch_closed_bars("EURUSD")
        args, _ = m.copy_rates_from_pos.call_args
        assert args[3] == 5

    def test_returns_empty_when_rates_none(self):
        m = MagicMock()
        m.TIMEFRAME_H1 = 16385
        m.copy_rates_from_pos = MagicMock(return_value=None)
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m)
        assert p.fetch_closed_bars("EURUSD") == []

    @pytest.mark.parametrize("pair", EIGHT_PAIRS)
    def test_each_pair_fetch(self, pair):
        m = _fake_mt5({pair: consecutive_h1_rates(T0, 2)})
        p = LiveBarPoller(pairs=(pair,), mt5_module=m, history_bars=2)
        bars = p.fetch_closed_bars(pair)
        assert len(bars) == 2
        assert bars[0].symbol == pair
        assert bars[1].symbol == pair


# ===========================================================================
# C. poll_once — newest-close detection
# ===========================================================================


class TestPollOnce:

    def test_no_data_yields_empty(self):
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=_fake_mt5({}))
        assert p.poll_once() == {}

    def test_first_call_emits_newest(self):
        m = _fake_mt5({"EURUSD": consecutive_h1_rates(T0, 3)})
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m, history_bars=3)
        new = p.poll_once()
        assert "EURUSD" in new
        expected = int((T0 + timedelta(hours=2)).timestamp() * 1000)
        assert new["EURUSD"].time_msc == expected

    def test_second_call_no_duplicate(self):
        m = _fake_mt5({"EURUSD": consecutive_h1_rates(T0, 3)})
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m, history_bars=3)
        p.poll_once()
        assert p.poll_once() == {}

    def test_new_bar_after_subsequent_poll(self):
        table = {"EURUSD": consecutive_h1_rates(T0, 3)}
        m = _fake_mt5(table)
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m, history_bars=3)
        p.poll_once()
        # Now MT5 returns one more bar.
        table["EURUSD"] = consecutive_h1_rates(T0, 4)
        new = p.poll_once()
        expected = int((T0 + timedelta(hours=3)).timestamp() * 1000)
        assert "EURUSD" in new
        assert new["EURUSD"].time_msc == expected

    def test_multi_pair_first_poll(self):
        m = _fake_mt5({
            "EURUSD": consecutive_h1_rates(T0, 3),
            "AUDJPY": consecutive_h1_rates(T0, 2),
        })
        p = LiveBarPoller(pairs=("EURUSD", "AUDJPY"), mt5_module=m, history_bars=3)
        new = p.poll_once()
        assert set(new.keys()) == {"EURUSD", "AUDJPY"}

    def test_only_one_pair_has_new_bar(self):
        table = {
            "EURUSD": consecutive_h1_rates(T0, 3),
            "AUDJPY": consecutive_h1_rates(T0, 3),
        }
        m = _fake_mt5(table)
        p = LiveBarPoller(pairs=("EURUSD", "AUDJPY"), mt5_module=m, history_bars=3)
        p.poll_once()
        # Only EURUSD gets a fresh bar
        table["EURUSD"] = consecutive_h1_rates(T0, 4)
        new = p.poll_once()
        assert set(new.keys()) == {"EURUSD"}

    @pytest.mark.parametrize("n_pairs", [1, 2, 3, 4, 5, 8])
    def test_n_pairs_each_gets_a_bar(self, n_pairs):
        pairs = EIGHT_PAIRS[:n_pairs]
        m = _fake_mt5({p: consecutive_h1_rates(T0, 2) for p in pairs})
        poll = LiveBarPoller(pairs=pairs, mt5_module=m, history_bars=2)
        new = poll.poll_once()
        assert set(new.keys()) == set(pairs)

    def test_buffer_replaced_on_each_poll(self):
        table = {"EURUSD": consecutive_h1_rates(T0, 3)}
        m = _fake_mt5(table)
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m, history_bars=3)
        p.poll_once()
        assert len(p.buffer["EURUSD"]) == 3
        table["EURUSD"] = consecutive_h1_rates(T0, 5)
        p.poll_once()
        assert len(p.buffer["EURUSD"]) == 5

    def test_last_bar_msc_updates(self):
        m = _fake_mt5({"EURUSD": consecutive_h1_rates(T0, 4)})
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m, history_bars=4)
        p.poll_once()
        assert p.last_bar_msc["EURUSD"] == int((T0 + timedelta(hours=3)).timestamp() * 1000)

    def test_last_bar_msc_view_is_snapshot(self):
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=_fake_mt5())
        snap = p.last_bar_msc
        snap["EURUSD"] = 999          # mutate the copy
        assert p.last_bar_msc["EURUSD"] == 0   # internal state unaffected

    def test_unknown_pair_silently_ignored(self):
        # We register only EURUSD; MT5 has data for GBPUSD too but poller
        # only loops over its registered pairs.
        m = _fake_mt5({
            "EURUSD": consecutive_h1_rates(T0, 2),
            "GBPUSD": consecutive_h1_rates(T0, 2),
        })
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m, history_bars=2)
        new = p.poll_once()
        assert set(new.keys()) == {"EURUSD"}


# ===========================================================================
# D. Edge cases: stale, future, monotonic
# ===========================================================================


class TestEdgeCases:

    def test_stale_bar_does_not_overwrite(self):
        # Step 1: see bar at T+2h. Step 2: MT5 returns OLDER bars only.
        table = {"EURUSD": consecutive_h1_rates(T0, 3)}
        m = _fake_mt5(table)
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m, history_bars=3)
        p.poll_once()
        before = p.last_bar_msc["EURUSD"]
        # Replace with rates whose newest bar is OLDER (start T-5h).
        table["EURUSD"] = consecutive_h1_rates(T0 - timedelta(hours=5), 3)
        new = p.poll_once()
        assert new == {}
        assert p.last_bar_msc["EURUSD"] == before

    def test_future_bar_accepted(self):
        # MT5 jumps forward in time (e.g., session resumed after weekend).
        m = _fake_mt5({"EURUSD": consecutive_h1_rates(T0, 3)})
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m, history_bars=3)
        p.poll_once()
        m_far = _fake_mt5({"EURUSD": consecutive_h1_rates(T0 + timedelta(hours=24), 3)})
        p._mt5 = m_far
        new = p.poll_once()
        assert "EURUSD" in new
        # Newest is T0+26h.
        assert new["EURUSD"].time_msc == int((T0 + timedelta(hours=26)).timestamp() * 1000)


# ===========================================================================
# E. run() loop — async behaviour
# ===========================================================================


class TestRunLoop:

    def test_stop_event_exits(self):
        async def _body():
            m = _fake_mt5({})
            p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m, poll_sec=0.05)
            stop = asyncio.Event()
            eng = _engine_pair()
            task = asyncio.create_task(p.run(
                engine=eng, stop=stop,
                account_provider=lambda: MagicMock(),
                prices_provider=lambda: ({}, {}),
            ))
            await asyncio.sleep(0.06)
            stop.set()
            await asyncio.wait_for(task, timeout=1.0)
            return eng
        eng = _run(_body())
        eng.process_scan_cycle.assert_not_called()

    def test_new_bar_triggers_engine(self):
        async def _body():
            m = _fake_mt5({"EURUSD": consecutive_h1_rates(T0, 3)})
            p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m,
                              history_bars=3, poll_sec=0.05)
            stop = asyncio.Event()
            eng = _engine_pair()
            task = asyncio.create_task(p.run(
                engine=eng, stop=stop,
                account_provider=lambda: "ACCT",
                prices_provider=lambda: (
                    {"EURUSD": 1.1}, {"EURUSD": 1.099},
                ),
            ))
            await asyncio.sleep(0.12)
            stop.set()
            await asyncio.wait_for(task, timeout=1.0)
            return eng
        eng = _run(_body())
        eng.process_scan_cycle.assert_awaited()
        eng.maintain_open.assert_awaited()

    def test_engine_called_with_buffer_and_now_msc(self):
        async def _body():
            m = _fake_mt5({"EURUSD": consecutive_h1_rates(T0, 3)})
            p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m,
                              history_bars=3, poll_sec=0.05)
            stop = asyncio.Event()
            eng = _engine_pair()
            task = asyncio.create_task(p.run(
                engine=eng, stop=stop,
                account_provider=lambda: "ACCT",
                prices_provider=lambda: ({"EURUSD": 1.1}, {"EURUSD": 1.099}),
            ))
            await asyncio.sleep(0.12)
            stop.set()
            await asyncio.wait_for(task, timeout=1.0)
            return eng
        eng = _run(_body())
        kwargs = eng.process_scan_cycle.call_args.kwargs
        assert "now_msc" in kwargs and isinstance(kwargs["now_msc"], int)
        assert kwargs["ask_by_pair"] == {"EURUSD": 1.1}
        assert kwargs["bid_by_pair"] == {"EURUSD": 1.099}
        assert kwargs["account"] == "ACCT"

    def test_poll_exception_swallowed(self):
        async def _body():
            m = MagicMock()
            m.TIMEFRAME_H1 = 16385
            ok = consecutive_h1_rates(T0, 1)
            m.copy_rates_from_pos = MagicMock(
                side_effect=[RuntimeError("boom"), ok, ok, ok, ok, ok, ok],
            )
            p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m,
                              history_bars=1, poll_sec=0.02)
            stop = asyncio.Event()
            eng = _engine_pair()
            task = asyncio.create_task(p.run(
                engine=eng, stop=stop,
                account_provider=lambda: "A",
                prices_provider=lambda: ({}, {}),
            ))
            await asyncio.sleep(0.15)
            stop.set()
            await asyncio.wait_for(task, timeout=1.0)
            return eng
        eng = _run(_body())
        assert eng.process_scan_cycle.await_count >= 1

    def test_engine_exception_swallowed(self):
        async def _body():
            m = _fake_mt5({"EURUSD": consecutive_h1_rates(T0, 3)})
            p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m,
                              history_bars=3, poll_sec=0.05)
            stop = asyncio.Event()
            eng = _engine_pair()
            eng.process_scan_cycle = AsyncMock(side_effect=RuntimeError("e"))
            task = asyncio.create_task(p.run(
                engine=eng, stop=stop,
                account_provider=lambda: "A",
                prices_provider=lambda: ({}, {}),
            ))
            await asyncio.sleep(0.12)
            stop.set()
            await asyncio.wait_for(task, timeout=1.0)
        # If the engine exception killed the loop, we'd hit a different error.
        _run(_body())

    def test_no_engine_call_when_no_new_bars(self):
        async def _body():
            m = _fake_mt5({})
            p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m, poll_sec=0.05)
            stop = asyncio.Event()
            eng = _engine_pair()
            task = asyncio.create_task(p.run(
                engine=eng, stop=stop,
                account_provider=lambda: "A",
                prices_provider=lambda: ({}, {}),
            ))
            await asyncio.sleep(0.2)
            stop.set()
            await asyncio.wait_for(task, timeout=1.0)
            return eng
        eng = _run(_body())
        eng.process_scan_cycle.assert_not_called()
        eng.maintain_open.assert_not_called()


# ===========================================================================
# F. Multi-pair poller scenarios (8 pairs × N scenarios)
# ===========================================================================


@pytest.mark.parametrize("pair", EIGHT_PAIRS)
class TestPerPairBehaviour:

    def test_fetch_then_poll(self, pair):
        m = _fake_mt5({pair: consecutive_h1_rates(T0, 3)})
        p = LiveBarPoller(pairs=(pair,), mt5_module=m, history_bars=3)
        new = p.poll_once()
        assert pair in new
        assert new[pair].symbol == pair

    def test_dedup_per_pair(self, pair):
        m = _fake_mt5({pair: consecutive_h1_rates(T0, 3)})
        p = LiveBarPoller(pairs=(pair,), mt5_module=m, history_bars=3)
        p.poll_once()
        assert p.poll_once() == {}

    def test_buffer_size(self, pair):
        m = _fake_mt5({pair: consecutive_h1_rates(T0, 7)})
        p = LiveBarPoller(pairs=(pair,), mt5_module=m, history_bars=7)
        p.poll_once()
        assert len(p.buffer[pair]) == 7

    def test_newest_is_last(self, pair):
        m = _fake_mt5({pair: consecutive_h1_rates(T0, 5)})
        p = LiveBarPoller(pairs=(pair,), mt5_module=m, history_bars=5)
        new = p.poll_once()
        # Newest bar matches the LAST entry in the buffer.
        assert new[pair].time_msc == p.buffer[pair][-1].time_msc

    def test_bars_are_immutable(self, pair):
        m = _fake_mt5({pair: consecutive_h1_rates(T0, 3)})
        p = LiveBarPoller(pairs=(pair,), mt5_module=m, history_bars=3)
        p.poll_once()
        b = p.buffer[pair][0]
        import dataclasses
        with pytest.raises(dataclasses.FrozenInstanceError):
            b.open = 9.99       # type: ignore[misc]


# ===========================================================================
# G. Polling cadence — history_bars × pairs sweep
# ===========================================================================


@pytest.mark.parametrize("history", [1, 3, 5, 10, 25, 50])
@pytest.mark.parametrize("n_pairs", [1, 3, 8])
def test_poll_cadence_sweep(history, n_pairs):
    pairs = EIGHT_PAIRS[:n_pairs]
    m = _fake_mt5({p: consecutive_h1_rates(T0, history) for p in pairs})
    poll = LiveBarPoller(pairs=pairs, mt5_module=m, history_bars=history)
    new = poll.poll_once()
    assert set(new.keys()) == set(pairs)
    for p in pairs:
        assert len(poll.buffer[p]) == history


# ===========================================================================
# H. View properties — buffer / last_bar_msc consistency
# ===========================================================================


class TestViews:

    def test_buffer_initially_listed_for_all_pairs(self):
        p = LiveBarPoller(pairs=EIGHT_PAIRS, mt5_module=_fake_mt5())
        assert set(p.buffer.keys()) == set(EIGHT_PAIRS)
        assert all(p.buffer[s] == [] for s in EIGHT_PAIRS)

    def test_last_bar_msc_listed_for_all_pairs(self):
        p = LiveBarPoller(pairs=EIGHT_PAIRS, mt5_module=_fake_mt5())
        assert set(p.last_bar_msc.keys()) == set(EIGHT_PAIRS)
        assert all(v == 0 for v in p.last_bar_msc.values())

    def test_last_bar_msc_after_poll(self):
        m = _fake_mt5({"EURUSD": consecutive_h1_rates(T0, 3)})
        p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m, history_bars=3)
        p.poll_once()
        assert p.last_bar_msc["EURUSD"] == int((T0 + timedelta(hours=2)).timestamp() * 1000)


# ===========================================================================
# I. Engine call ordering: scan_cycle BEFORE maintain_open
# ===========================================================================


class TestEngineCallOrder:

    def test_scan_then_maintain(self):
        order: List[str] = []

        async def scan(*a, **kw):
            order.append("scan")

        async def maint(*a, **kw):
            order.append("maint")

        async def _body():
            m = _fake_mt5({"EURUSD": consecutive_h1_rates(T0, 2)})
            p = LiveBarPoller(pairs=("EURUSD",), mt5_module=m,
                              history_bars=2, poll_sec=0.03)
            eng = MagicMock()
            eng.process_scan_cycle = scan
            eng.maintain_open = maint
            stop = asyncio.Event()
            task = asyncio.create_task(p.run(
                engine=eng, stop=stop,
                account_provider=lambda: "A",
                prices_provider=lambda: ({}, {}),
            ))
            await asyncio.sleep(0.1)
            stop.set()
            await asyncio.wait_for(task, timeout=1.0)
        _run(_body())
        # First two events should be in order.
        assert order[0] == "scan"
        assert order[1] == "maint"
