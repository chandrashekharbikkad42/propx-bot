"""Shared fixtures for the Phase 6 end-to-end suite.

Most tests want the same composed runner; this module wraps the
`ScenarioRunner.build()` factory as pytest fixtures so each test gets a
fresh, isolated engine.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.asian_sweep_config import PAIR_CONFIG, PAIRS  # noqa: E402

from tests.e2e.fixtures.scenario_runner import (  # noqa: E402
    ScenarioRunner, long_sweep_bars, short_sweep_bars, hour_msc,
)


ALL_PAIRS: List[str] = list(PAIRS)


@pytest.fixture
def runner_factory():
    """Returns a callable that builds a fresh ScenarioRunner."""
    def _make(**kwargs) -> ScenarioRunner:
        return ScenarioRunner.build(**kwargs)
    return _make


@pytest.fixture
def runner() -> ScenarioRunner:
    """Default fresh runner with all pairs and FTMO 2-step rules."""
    return ScenarioRunner.build()


@pytest.fixture
def runner_5ers() -> ScenarioRunner:
    """Runner configured with The5%ers Hyper Growth Step1 rules."""
    return ScenarioRunner.build(rules_key="the5ers_hyper_growth_step1")


@pytest.fixture(params=ALL_PAIRS)
def each_pair(request):
    return request.param


@pytest.fixture
def long_bars():
    return long_sweep_bars


@pytest.fixture
def short_bars():
    return short_sweep_bars


@pytest.fixture
def msc():
    return hour_msc


@pytest.fixture
def pair_pt():
    def _pt(symbol: str) -> float:
        return float(PAIR_CONFIG[symbol]["point"])
    return _pt
