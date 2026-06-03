"""Shared fixtures for the data/* test suite."""

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Iterator

import pytest

from tests.data.fixtures.mock_connector import FakeConnector
from tests.data.fixtures.synthetic_ticks import (
    EIGHT_PAIRS,
    MT5_RATES_DTYPE,
    MT5_TICK_DTYPE,
    base_price_for,
    consecutive_h1_rates,
    duplicate_ts_ticks,
    hour_filling_ticks,
    make_tick,
    mt5_rates_array,
    mt5_tick_array,
    random_walk_ticks,
    sideways_ticks,
    spread_for,
    ticks_to_mt5_array,
    trend_ticks,
    utc_ms,
)


# ---------------------------------------------------------------------------
# Settings patching — keeps writer output away from the repo data dir
# ---------------------------------------------------------------------------

def _patch_settings_field(monkeypatch, field: str, value):
    """Settings is a frozen dataclass — patch the imported references instead.

    Modules that need redirection import `settings` at top level, so we
    monkey-patch the module-level binding in each consumer to a SimpleNamespace
    that exposes the same field. This avoids FrozenInstanceError and keeps
    the change scoped per-test.
    """
    from types import SimpleNamespace
    from config import settings as real_settings
    # Build a stub mirroring all real attributes, with `field` overridden.
    attrs = {n: getattr(real_settings, n) for n in dir(real_settings)
             if not n.startswith("_") and not callable(getattr(real_settings, n))}
    attrs[field] = value
    stub = SimpleNamespace(**attrs)
    # Wherever `settings` is imported, swap it for our stub.
    from data import tick_writer, bar_aggregator
    monkeypatch.setattr(tick_writer, "settings", stub)
    monkeypatch.setattr(bar_aggregator, "settings", stub)
    return stub


@pytest.fixture
def patch_data_dir(tmp_path: Path, monkeypatch) -> Path:
    """Redirect `settings.data_dir` (tick-writer output root) to tmp_path."""
    target = tmp_path / "ticks"
    _patch_settings_field(monkeypatch, "data_dir", target)
    return target


@pytest.fixture
def patch_bars_dir(tmp_path: Path, monkeypatch) -> Path:
    """Redirect `settings.bars_dir` (bar-aggregator parquet root) to tmp_path."""
    target = tmp_path / "bars"
    _patch_settings_field(monkeypatch, "bars_dir", target)
    return target


# ---------------------------------------------------------------------------
# Connector / queue helpers for tick-collector + tick-writer tests
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_connector():
    return FakeConnector()


@pytest.fixture
def small_queue() -> asyncio.Queue:
    return asyncio.Queue(maxsize=8)


@pytest.fixture
def medium_queue() -> asyncio.Queue:
    return asyncio.Queue(maxsize=100)


@pytest.fixture
def unbounded_queue() -> asyncio.Queue:
    return asyncio.Queue()


# ---------------------------------------------------------------------------
# Re-export tick / rate factories so tests can request them as fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def t0() -> int:
    """A clean UTC hour boundary used by most tests (2026-05-18 10:00:00Z)."""
    return utc_ms(2026, 5, 18, 10, 0)


@pytest.fixture
def eight_pairs():
    return EIGHT_PAIRS


# Convenience: expose helpers as a namespace object on the fixture surface
# so test modules can `def test_x(synth):` and reach everything.

class _SynthFacade:
    utc_ms = staticmethod(utc_ms)
    make_tick = staticmethod(make_tick)
    random_walk_ticks = staticmethod(random_walk_ticks)
    trend_ticks = staticmethod(trend_ticks)
    sideways_ticks = staticmethod(sideways_ticks)
    hour_filling_ticks = staticmethod(hour_filling_ticks)
    duplicate_ts_ticks = staticmethod(duplicate_ts_ticks)
    ticks_to_mt5_array = staticmethod(ticks_to_mt5_array)
    mt5_tick_array = staticmethod(mt5_tick_array)
    mt5_rates_array = staticmethod(mt5_rates_array)
    consecutive_h1_rates = staticmethod(consecutive_h1_rates)
    base_price_for = staticmethod(base_price_for)
    spread_for = staticmethod(spread_for)
    EIGHT_PAIRS = EIGHT_PAIRS
    MT5_TICK_DTYPE = MT5_TICK_DTYPE
    MT5_RATES_DTYPE = MT5_RATES_DTYPE


@pytest.fixture
def synth() -> _SynthFacade:
    return _SynthFacade()
