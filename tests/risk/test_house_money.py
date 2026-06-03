"""HouseMoneyManager — Griff per-trade risk allocator.

Trade 1: STANDARD = base_risk_pct.
Trade 2:
  prior pnl > 0 → HOUSE_MONEY: base + (pnl/equity)*100*fraction, capped at base*mult
  prior pnl <= 0 → DEFENSIVE: base * defensive_mult
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from risk.house_money import (
    DEFAULT_BASE_RISK_PCT, DEFENSIVE_MULT, HOUSE_MONEY_FRACTION,
    HouseMoneyManager, MAX_HOUSE_MONEY_MULT, RiskAllocation,
)
from strategy.patterns.base import Grade


@pytest.fixture
def manager():
    return HouseMoneyManager()


# ===========================================================================
# 1. Module constants
# ===========================================================================

class TestConstants:
    def test_default_base_a(self):
        assert DEFAULT_BASE_RISK_PCT[Grade.A] == 1.0

    def test_default_base_b(self):
        assert DEFAULT_BASE_RISK_PCT[Grade.B] == 0.5

    def test_default_base_c_zero(self):
        assert DEFAULT_BASE_RISK_PCT[Grade.C] == 0.0

    def test_house_money_fraction(self):
        assert HOUSE_MONEY_FRACTION == 0.5

    def test_max_house_money_mult(self):
        assert MAX_HOUSE_MONEY_MULT == 2.0

    def test_defensive_mult(self):
        assert DEFENSIVE_MULT == 0.5


# ===========================================================================
# 2. Constructor validation
# ===========================================================================

class TestConstructor:
    def test_default_constructs(self):
        HouseMoneyManager()

    def test_negative_fraction_raises(self):
        with pytest.raises(ValueError):
            HouseMoneyManager(house_money_fraction=-0.1)

    def test_max_mult_below_one_raises(self):
        with pytest.raises(ValueError):
            HouseMoneyManager(max_house_money_mult=0.9)

    def test_max_mult_exactly_one_allowed(self):
        HouseMoneyManager(max_house_money_mult=1.0)

    def test_defensive_mult_above_one_raises(self):
        with pytest.raises(ValueError):
            HouseMoneyManager(defensive_mult=1.1)

    def test_defensive_mult_negative_raises(self):
        with pytest.raises(ValueError):
            HouseMoneyManager(defensive_mult=-0.1)

    def test_custom_base_overrides_default(self):
        m = HouseMoneyManager(base_risk_pct={Grade.A: 2.0, Grade.B: 1.0})
        assert m.base_pct_for(Grade.A) == 2.0
        assert m.base_pct_for(Grade.B) == 1.0


# ===========================================================================
# 3. base_pct_for
# ===========================================================================

class TestBasePctFor:
    def test_a(self, manager):
        assert manager.base_pct_for(Grade.A) == 1.0

    def test_b(self, manager):
        assert manager.base_pct_for(Grade.B) == 0.5

    def test_c(self, manager):
        assert manager.base_pct_for(Grade.C) == 0.0


# ===========================================================================
# 4. calc_trade_risk — validation
# ===========================================================================

class TestCalcTradeRiskValidation:
    def test_grade_c_raises(self, manager):
        with pytest.raises(ValueError, match="C-grade"):
            manager.calc_trade_risk(Grade.C, 10_000.0, 0.0, 1)

    def test_zero_equity_raises(self, manager):
        with pytest.raises(ValueError, match="equity"):
            manager.calc_trade_risk(Grade.A, 0.0, 0.0, 1)

    def test_negative_equity_raises(self, manager):
        with pytest.raises(ValueError, match="equity"):
            manager.calc_trade_risk(Grade.A, -1.0, 0.0, 1)

    @pytest.mark.parametrize("n", [0, 3, 4, 100, -1])
    def test_invalid_trade_number_raises(self, manager, n):
        with pytest.raises(ValueError, match="trade_number_today"):
            manager.calc_trade_risk(Grade.A, 10_000.0, 0.0, n)


# ===========================================================================
# 5. Trade 1 — STANDARD
# ===========================================================================

class TestTrade1Standard:
    def test_grade_a_trade1(self, manager):
        alloc = manager.calc_trade_risk(Grade.A, 10_000.0, 0.0, 1)
        assert alloc.mode == "STANDARD"
        assert alloc.final_risk_pct == 1.0
        assert alloc.base_risk_pct == 1.0
        assert alloc.trade_number_today == 1
        assert alloc.grade == Grade.A

    def test_grade_b_trade1(self, manager):
        alloc = manager.calc_trade_risk(Grade.B, 10_000.0, 0.0, 1)
        assert alloc.mode == "STANDARD"
        assert alloc.final_risk_pct == 0.5

    def test_trade1_ignores_prior_pnl(self, manager):
        a = manager.calc_trade_risk(Grade.A, 10_000.0, 999.0, 1)
        b = manager.calc_trade_risk(Grade.A, 10_000.0, -999.0, 1)
        # Trade 1 risk is independent of prior pnl.
        assert a.final_risk_pct == b.final_risk_pct == 1.0

    def test_trade1_rationale_mentions_grade(self, manager):
        alloc = manager.calc_trade_risk(Grade.A, 10_000.0, 0.0, 1)
        assert "trade 1" in alloc.rationale.lower()


# ===========================================================================
# 6. Trade 2 HOUSE_MONEY — positive prior PnL
# ===========================================================================

class TestTrade2HouseMoney:
    def test_basic_house_money(self, manager):
        # equity=10000, base=1.0%, prior pnl=$200, fraction=0.5.
        # extra = (200/10000)*100*0.5 = 1.0 → raw = 2.0%
        # cap = base * mult = 2.0% → capped at 2.0%.
        alloc = manager.calc_trade_risk(Grade.A, 10_000.0, 200.0, 2)
        assert alloc.mode == "HOUSE_MONEY"
        assert alloc.final_risk_pct == pytest.approx(2.0)

    def test_small_win_no_cap(self, manager):
        # +$10 → extra = (10/10000)*100*0.5 = 0.05% → raw=1.05%, cap=2.0%
        alloc = manager.calc_trade_risk(Grade.A, 10_000.0, 10.0, 2)
        assert alloc.final_risk_pct == pytest.approx(1.05)

    def test_big_win_caps_at_2x(self, manager):
        # +$10000 win → extra = 50% → way over cap.
        alloc = manager.calc_trade_risk(Grade.A, 10_000.0, 10_000.0, 2)
        assert alloc.final_risk_pct == pytest.approx(2.0)
        assert alloc.mode == "HOUSE_MONEY"

    def test_house_money_rationale(self, manager):
        alloc = manager.calc_trade_risk(Grade.A, 10_000.0, 100.0, 2)
        assert "HOUSE_MONEY" == alloc.mode
        assert "+$100.00" in alloc.rationale

    @pytest.mark.parametrize("pnl,expected_pct", [
        (10.0, 1.05),
        (50.0, 1.25),
        (100.0, 1.5),
        (200.0, 2.0),
        (500.0, 2.0),  # capped
    ])
    def test_pnl_scaling(self, manager, pnl, expected_pct):
        alloc = manager.calc_trade_risk(Grade.A, 10_000.0, pnl, 2)
        assert alloc.final_risk_pct == pytest.approx(expected_pct, abs=1e-6)


# ===========================================================================
# 7. Trade 2 DEFENSIVE — non-positive prior PnL
# ===========================================================================

class TestTrade2Defensive:
    def test_loss(self, manager):
        alloc = manager.calc_trade_risk(Grade.A, 10_000.0, -100.0, 2)
        assert alloc.mode == "DEFENSIVE"
        assert alloc.final_risk_pct == pytest.approx(0.5)

    def test_exactly_flat(self, manager):
        alloc = manager.calc_trade_risk(Grade.A, 10_000.0, 0.0, 2)
        assert alloc.mode == "DEFENSIVE"
        # 1.0% * 0.5 = 0.5%
        assert alloc.final_risk_pct == pytest.approx(0.5)

    def test_grade_b_defensive(self, manager):
        alloc = manager.calc_trade_risk(Grade.B, 10_000.0, -50.0, 2)
        assert alloc.mode == "DEFENSIVE"
        # 0.5% * 0.5 = 0.25%
        assert alloc.final_risk_pct == pytest.approx(0.25)

    def test_defensive_rationale(self, manager):
        alloc = manager.calc_trade_risk(Grade.A, 10_000.0, -100.0, 2)
        assert "defensive" in alloc.rationale.lower()

    @pytest.mark.parametrize("loss", [-1.0, -10.0, -50.0, -200.0, -500.0])
    def test_defensive_always_half_of_base(self, manager, loss):
        alloc = manager.calc_trade_risk(Grade.A, 10_000.0, loss, 2)
        assert alloc.final_risk_pct == pytest.approx(0.5)


# ===========================================================================
# 8. Boundary: pnl exactly 0 → defensive (per spec)
# ===========================================================================

def test_exactly_zero_pnl_treated_defensive(manager):
    alloc = manager.calc_trade_risk(Grade.A, 10_000.0, 0.0, 2)
    assert alloc.mode == "DEFENSIVE"


def test_one_cent_positive_pnl_is_house_money(manager):
    alloc = manager.calc_trade_risk(Grade.A, 10_000.0, 0.01, 2)
    assert alloc.mode == "HOUSE_MONEY"


def test_one_cent_negative_pnl_is_defensive(manager):
    alloc = manager.calc_trade_risk(Grade.A, 10_000.0, -0.01, 2)
    assert alloc.mode == "DEFENSIVE"


# ===========================================================================
# 9. Skip mode (base_risk_pct = 0 grade)
# ===========================================================================

class TestSkipMode:
    def test_zero_base_returns_skip(self):
        manager = HouseMoneyManager(base_risk_pct={Grade.A: 0.0,
                                                   Grade.B: 0.5})
        alloc = manager.calc_trade_risk(Grade.A, 10_000.0, 0.0, 1)
        assert alloc.mode == "SKIP"
        assert alloc.final_risk_pct == 0.0


# ===========================================================================
# 10. RiskAllocation dataclass
# ===========================================================================

class TestRiskAllocation:
    def test_frozen(self):
        alloc = RiskAllocation(
            grade=Grade.A, trade_number_today=1,
            base_risk_pct=1.0, final_risk_pct=1.0,
            mode="STANDARD", rationale="",
        )
        with pytest.raises(Exception):
            alloc.mode = "DEFENSIVE"  # type: ignore[misc]


# ===========================================================================
# 11. Per-equity sweep (linearity check)
# ===========================================================================

@pytest.mark.parametrize("equity", [1_000.0, 5_000.0, 10_000.0,
                                    25_000.0, 50_000.0, 100_000.0])
def test_per_equity_trade1_pct_independent(manager, equity):
    alloc = manager.calc_trade_risk(Grade.A, equity, 0.0, 1)
    assert alloc.final_risk_pct == 1.0


@pytest.mark.parametrize("equity", [1_000.0, 5_000.0, 10_000.0,
                                    25_000.0, 50_000.0, 100_000.0])
def test_per_equity_defensive_invariant(manager, equity):
    alloc = manager.calc_trade_risk(Grade.A, equity, -100.0, 2)
    assert alloc.final_risk_pct == pytest.approx(0.5)


# ===========================================================================
# 12. Equity-tier-style scenarios (the user spec calls this "tier mapping")
# ===========================================================================

EQUITY_TIERS = [
    (10_000.0,   "fresh"),
    (15_000.0,   "+50% growth"),
    (20_000.0,   "+100% growth"),
    (25_000.0,   "+150% growth"),
    (50_000.0,   "scaling"),
    (100_000.0,  "scaled"),
]


@pytest.mark.parametrize("equity,label", EQUITY_TIERS)
def test_house_money_tier(manager, equity, label):
    alloc = manager.calc_trade_risk(Grade.A, equity, equity * 0.01, 2)
    # +1% win → extra = 1% * 0.5 = 0.5%; raw = 1.5%; cap = 2.0%.
    assert alloc.mode == "HOUSE_MONEY"
    assert alloc.final_risk_pct == pytest.approx(1.5)


# ===========================================================================
# 13. Custom fraction / cap
# ===========================================================================

class TestCustomFraction:
    def test_zero_fraction_means_no_uplift(self):
        m = HouseMoneyManager(house_money_fraction=0.0)
        alloc = m.calc_trade_risk(Grade.A, 10_000.0, 200.0, 2)
        assert alloc.final_risk_pct == pytest.approx(1.0)
        assert alloc.mode == "HOUSE_MONEY"  # still house-money mode

    def test_full_fraction_uplift(self):
        m = HouseMoneyManager(house_money_fraction=1.0,
                              max_house_money_mult=10.0)
        alloc = m.calc_trade_risk(Grade.A, 10_000.0, 200.0, 2)
        # extra = 200/10000 * 100 * 1.0 = 2.0%, raw=3.0%, cap=10%
        assert alloc.final_risk_pct == pytest.approx(3.0)


class TestCustomCap:
    def test_cap_1_5x(self):
        m = HouseMoneyManager(max_house_money_mult=1.5)
        alloc = m.calc_trade_risk(Grade.A, 10_000.0, 1_000.0, 2)
        # raw = 1 + 0.05*100*0.5 = 6.0%; cap = 1.5%
        assert alloc.final_risk_pct == pytest.approx(1.5)


class TestCustomDefensive:
    def test_zero_defensive_zeros_out_trade2(self):
        m = HouseMoneyManager(defensive_mult=0.0)
        alloc = m.calc_trade_risk(Grade.A, 10_000.0, -100.0, 2)
        assert alloc.final_risk_pct == pytest.approx(0.0)
        assert alloc.mode == "DEFENSIVE"

    def test_one_defensive_keeps_base(self):
        m = HouseMoneyManager(defensive_mult=1.0)
        alloc = m.calc_trade_risk(Grade.A, 10_000.0, -100.0, 2)
        assert alloc.final_risk_pct == pytest.approx(1.0)


# ===========================================================================
# 14. daily_summary
# ===========================================================================

class TestDailySummary:
    def test_returns_dict(self, manager):
        s = manager.daily_summary(10_000.0)
        assert isinstance(s, dict)
        for k in ("worst_pct", "best_pct", "base_pct"):
            assert k in s

    def test_zero_equity_returns_zeros(self, manager):
        s = manager.daily_summary(0.0)
        assert s == {"worst_pct": 0.0, "best_pct": 0.0, "base_pct": 1.0}

    def test_a_grade_worst_negative(self, manager):
        s = manager.daily_summary(10_000.0, Grade.A)
        # both losses, T2 defensive: -1% + (-0.5%) = -1.5%
        assert s["worst_pct"] == pytest.approx(-1.5)

    def test_b_grade_worst(self, manager):
        s = manager.daily_summary(10_000.0, Grade.B)
        # 0.5% base, T2 defensive 0.25% → worst = -0.75%
        assert s["worst_pct"] == pytest.approx(-0.75)

    def test_grade_c_zero_base(self, manager):
        s = manager.daily_summary(10_000.0, Grade.C)
        assert s == {"worst_pct": 0.0, "best_pct": 0.0, "base_pct": 0.0}


# ===========================================================================
# 15. Hypothesis property: trade1 risk == base
# ===========================================================================

@settings(max_examples=80, deadline=None)
@given(
    equity=st.floats(min_value=100.0, max_value=10_000_000.0,
                     allow_nan=False, allow_infinity=False),
    prior_pnl=st.floats(min_value=-1_000_000.0, max_value=1_000_000.0,
                        allow_nan=False, allow_infinity=False),
)
def test_trade1_risk_is_base(equity, prior_pnl):
    m = HouseMoneyManager()
    alloc = m.calc_trade_risk(Grade.A, equity, prior_pnl, 1)
    assert alloc.final_risk_pct == 1.0
    assert alloc.mode == "STANDARD"


@settings(max_examples=80, deadline=None)
@given(
    equity=st.floats(min_value=100.0, max_value=10_000_000.0,
                     allow_nan=False, allow_infinity=False),
    win=st.floats(min_value=0.01, max_value=100_000.0,
                  allow_nan=False, allow_infinity=False),
)
def test_house_money_capped_at_2x(equity, win):
    m = HouseMoneyManager()
    alloc = m.calc_trade_risk(Grade.A, equity, win, 2)
    assert alloc.final_risk_pct <= 2.0
    assert alloc.mode == "HOUSE_MONEY"


@settings(max_examples=80, deadline=None)
@given(
    equity=st.floats(min_value=100.0, max_value=10_000_000.0,
                     allow_nan=False, allow_infinity=False),
    loss=st.floats(min_value=-100_000.0, max_value=0.0,
                   allow_nan=False, allow_infinity=False),
)
def test_defensive_always_half(equity, loss):
    m = HouseMoneyManager()
    alloc = m.calc_trade_risk(Grade.A, equity, loss, 2)
    assert alloc.final_risk_pct == pytest.approx(0.5)
    assert alloc.mode == "DEFENSIVE"


# ===========================================================================
# 16. Per-grade matrix
# ===========================================================================

@pytest.mark.parametrize("grade,base", [
    (Grade.A, 1.0),
    (Grade.B, 0.5),
])
@pytest.mark.parametrize("trade,prior_pnl,expected_mode", [
    (1, 0.0, "STANDARD"),
    (1, 100.0, "STANDARD"),
    (1, -100.0, "STANDARD"),
    (2, 100.0, "HOUSE_MONEY"),
    (2, 0.0, "DEFENSIVE"),
    (2, -100.0, "DEFENSIVE"),
])
def test_full_matrix(manager, grade, base, trade, prior_pnl, expected_mode):
    alloc = manager.calc_trade_risk(grade, 10_000.0, prior_pnl, trade)
    assert alloc.mode == expected_mode
    assert alloc.base_risk_pct == base


# ===========================================================================
# 17. Weak-month dampener is NOT enforced by HouseMoneyManager
# ===========================================================================

class TestWeakMonthNotInHouseMoney:
    """house_money.py doesn't reference month-of-year. The weak-month
    dampener lives in `config.asian_sweep_config.risk_pct_for` and is
    applied at the position-sizer / size_position layer.

    These tests document that contract."""
    def test_grade_a_constant_across_months(self, manager):
        # Same trade #1, same equity, same prior pnl → same result regardless
        # of what month it is. Just verify the API has no `month` parameter
        # by inspecting the signature.
        import inspect
        sig = inspect.signature(manager.calc_trade_risk)
        assert "month" not in sig.parameters


# ===========================================================================
# 18. Idempotency: same inputs → same allocation
# ===========================================================================

@pytest.mark.parametrize("equity,prior,trade", [
    (10_000.0, 0.0, 1),
    (50_000.0, 200.0, 2),
    (100_000.0, -50.0, 2),
    (5_000.0, 1_000.0, 2),
])
def test_idempotent(manager, equity, prior, trade):
    a = manager.calc_trade_risk(Grade.A, equity, prior, trade)
    b = manager.calc_trade_risk(Grade.A, equity, prior, trade)
    assert a == b
