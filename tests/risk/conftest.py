"""Shared fixtures for the risk + compliance test suite."""

from __future__ import annotations
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.news_calendar import StaticNewsCalendar  # noqa: E402
from risk.prop_firm.compliance import AccountState, ComplianceEngine  # noqa: E402
from risk.prop_firm.rules import RULES_DB, get_rules  # noqa: E402
from strategy.patterns.base import Direction, Grade, PatternSignal  # noqa: E402

from tests.risk.fixtures.account_states import make_account  # noqa: E402
from tests.risk.fixtures.news_events import utc_ms  # noqa: E402


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def msc():
    """Convenience: produce UTC ms from y/m/d/h/m args."""
    return utc_ms


# ---------------------------------------------------------------------------
# AccountState
# ---------------------------------------------------------------------------

@pytest.fixture
def account_factory():
    return make_account


@pytest.fixture
def fresh_account():
    return make_account()


# ---------------------------------------------------------------------------
# Prop-firm rules
# ---------------------------------------------------------------------------

@pytest.fixture
def ftmo_rules():
    return get_rules("ftmo_2step_challenge")


@pytest.fixture
def ftmo_funded_rules():
    return get_rules("ftmo_2step_funded")


@pytest.fixture
def the5ers_rules():
    return get_rules("the5ers_bootcamp_step1")


# ---------------------------------------------------------------------------
# Compliance engine
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_calendar():
    return StaticNewsCalendar([])


@pytest.fixture
def compliance_factory(empty_calendar, ftmo_rules):
    """Return a factory that builds a fresh ComplianceEngine.

    Defaults: FTMO 2-step challenge rules, empty news calendar, IST window
    12:30–22:30, max 2 trades/day, 80 % safety margin.
    """
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


# ---------------------------------------------------------------------------
# PatternSignal factory
# ---------------------------------------------------------------------------

@pytest.fixture
def signal_factory():
    """Build a PatternSignal with sensible defaults.

    Default is a tiny BUY on EURUSD that respects the 5 invariants
    enforced by `PatternSignal.__post_init__` and produces a small
    worst-case loss for compliance tests.
    """
    def _build(
        *,
        symbol: str = "EURUSD",
        direction: Direction = Direction.BUY,
        entry: float = 1.10000,
        sl: float | None = None,
        tp: float | None = None,
        confidence: float = 0.9,
        grade: Grade = Grade.A,
        confluences_met=("asian_sweep_low", "LONDON", "bias_neutral", "q9",
                         "tp1_1.10010"),
        bar_time_msc: int = 0,
        risk_pts: float = 10.0,
        rr: float = 2.5,
    ) -> PatternSignal:
        # Determine sl/tp from risk_pts if not given.
        # 5-decimal pairs: 1 pt = 0.00001. For risk_pts=10 → 0.0001.
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
