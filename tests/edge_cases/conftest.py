"""Shared fixtures for the adversarial / edge-case test suite.

Reuses the Phase 1-4 fixtures (mock_mt5, account factories, signal factory,
synthetic bar builders) and layers chaos generators + broker-failure
injectors on top so individual test files stay terse.
"""

from __future__ import annotations
import sys
from pathlib import Path

import pytest

# Make repo importable when pytest is launched from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.news_calendar import StaticNewsCalendar  # noqa: E402
from risk.prop_firm.compliance import AccountState, ComplianceEngine  # noqa: E402
from risk.prop_firm.rules import get_rules  # noqa: E402
from strategy.patterns.asian_sweep import AsianSweepDetector  # noqa: E402
from strategy.patterns.base import (  # noqa: E402
    Direction, Grade, MarketContext, PatternSignal,
)

from tests.edge_cases.fixtures import broker_failures, chaos_market  # noqa: E402,F401
from tests.execution.fixtures.mock_mt5 import MockMT5  # noqa: E402
from tests.risk.fixtures.account_states import make_account  # noqa: E402
from tests.risk.fixtures.news_events import utc_ms  # noqa: E402


# ---------------------------------------------------------------------------
# Convenience re-exports as fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector() -> AsianSweepDetector:
    return AsianSweepDetector()


@pytest.fixture
def context_factory():
    def _make(symbol: str = "EURUSD", time_msc: int = 0,
              htf_bias: str | None = None,
              spread_pts: float = 0.0) -> MarketContext:
        return MarketContext(
            symbol=symbol, current_time_msc=time_msc,
            htf_bias=htf_bias, spread_pts=spread_pts,
        )
    return _make


@pytest.fixture
def account_factory():
    return make_account


@pytest.fixture
def fresh_account():
    return make_account()


@pytest.fixture
def empty_calendar():
    return StaticNewsCalendar([])


@pytest.fixture
def ftmo_rules():
    return get_rules("ftmo_2step_challenge")


@pytest.fixture
def ftmo_funded_rules():
    return get_rules("ftmo_2step_funded")


@pytest.fixture
def compliance_factory(empty_calendar, ftmo_rules):
    def _build(
        rules=None,
        news_calendar=None,
        max_trades_per_day: int = 2,
        ist_window_start: str = "12:30",
        ist_window_end: str = "22:30",
        safety_margin_pct: float = 0.80,
    ) -> ComplianceEngine:
        return ComplianceEngine(
            rules=rules or ftmo_rules,
            max_trades_per_day=max_trades_per_day,
            ist_window_start=ist_window_start,
            ist_window_end=ist_window_end,
            news_calendar=news_calendar or empty_calendar,
            safety_margin_pct=safety_margin_pct,
        )
    return _build


@pytest.fixture
def signal_factory():
    """Build a PatternSignal honouring the 5 invariants in PatternSignal.__post_init__."""
    def _build(
        *,
        symbol: str = "EURUSD",
        direction: Direction = Direction.BUY,
        entry: float = 1.10000,
        sl: float | None = None,
        tp: float | None = None,
        confidence: float = 0.9,
        grade: Grade = Grade.A,
        confluences_met: tuple[str, ...] = (
            "asian_sweep_low", "LONDON", "bias_neutral", "q9", "tp1_1.10010",
        ),
        bar_time_msc: int = 0,
        risk_pts: float = 10.0,
        rr: float = 2.5,
    ) -> PatternSignal:
        pt_unit = 0.00001 if symbol != "XAUUSD" else 0.01
        risk = risk_pts * pt_unit
        if sl is None:
            sl = entry - risk if direction == Direction.BUY else entry + risk
        if tp is None:
            tp = entry + risk * rr if direction == Direction.BUY else entry - risk * rr
        return PatternSignal(
            pattern_name="ASIAN_SWEEP",
            symbol=symbol,
            direction=direction,
            entry=entry, sl=sl, tp=tp,
            confidence=confidence, grade=grade,
            confluences_met=confluences_met,
            bar_time_msc=bar_time_msc,
        )
    return _build


@pytest.fixture
def mock_mt5():
    return MockMT5()


@pytest.fixture
def patch_router_mt5(mock_mt5, monkeypatch):
    from execution import order_router as griff_order_router
    monkeypatch.setattr(griff_order_router, "mt5", mock_mt5)
    return mock_mt5


@pytest.fixture
def patch_live_broker_mt5(mock_mt5, monkeypatch):
    from execution import live_broker
    monkeypatch.setattr(live_broker, "mt5", mock_mt5)
    return mock_mt5


@pytest.fixture
def msc():
    return utc_ms
