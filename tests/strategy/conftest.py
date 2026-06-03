"""Shared fixtures for AsianSweepDetector tests.

Most tests build their own bar sequences via
`tests.strategy.fixtures.synthetic_bars`; this conftest provides the
detector instance, a base MarketContext, and per-pair config snapshots.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List

import pytest

# Make repo importable when pytest is launched from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.asian_sweep_config import PAIR_CONFIG, PAIRS  # noqa: E402
from strategy.patterns.asian_sweep import AsianSweepDetector  # noqa: E402
from strategy.patterns.base import MarketContext  # noqa: E402


# ---------------------------------------------------------------------------
# Detector + context
# ---------------------------------------------------------------------------

@pytest.fixture
def detector() -> AsianSweepDetector:
    return AsianSweepDetector()


@pytest.fixture
def context_factory():
    def _make(symbol: str = "EURUSD", time_msc: int = 0) -> MarketContext:
        return MarketContext(symbol=symbol, current_time_msc=time_msc)
    return _make


# ---------------------------------------------------------------------------
# Per-pair parametrise helpers
# ---------------------------------------------------------------------------

ALL_PAIRS: List[str] = list(PAIRS)


@pytest.fixture(params=ALL_PAIRS)
def pair(request) -> str:
    return request.param


@pytest.fixture
def pair_cfg(pair):
    return PAIR_CONFIG[pair]
