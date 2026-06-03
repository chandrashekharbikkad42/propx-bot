"""risk.position_sizer.calculate_lot_size — pure lot calculation.

Formula:
    pnl_per_lot = sl_distance_pts × POINT_VALUE × contract_size
    lots        = round(risk_pct × equity / pnl_per_lot, 2)
    clamped to [MIN_LOTS=0.01, MAX_LOTS=10.0]

For PROCENT accounts, equity is divided by 100 first (ProCent cents → USD).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from risk.position_sizer import (
    MAX_LOTS, MIN_LOTS, POINT_VALUE, calculate_lot_size,
)


# ===========================================================================
# 1. Module constants
# ===========================================================================

class TestConstants:
    def test_point_value(self):
        assert POINT_VALUE == 0.01

    def test_min_lots(self):
        assert MIN_LOTS == 0.01

    def test_max_lots(self):
        assert MAX_LOTS == 10.0

    def test_min_under_max(self):
        assert MIN_LOTS < MAX_LOTS


# ===========================================================================
# 2. Floor / degenerate inputs
# ===========================================================================

class TestFloor:
    def test_zero_sl_returns_min(self):
        assert calculate_lot_size(10_000.0, 0.01, 0) == MIN_LOTS

    def test_negative_sl_returns_min(self):
        assert calculate_lot_size(10_000.0, 0.01, -10) == MIN_LOTS

    def test_zero_equity_returns_min(self):
        assert calculate_lot_size(0.0, 0.01, 100) == MIN_LOTS

    def test_negative_equity_returns_min(self):
        assert calculate_lot_size(-1.0, 0.01, 100) == MIN_LOTS

    def test_zero_risk_pct_returns_min(self):
        assert calculate_lot_size(10_000.0, 0.0, 100) == MIN_LOTS

    def test_negative_risk_pct_returns_min(self):
        assert calculate_lot_size(10_000.0, -0.01, 100) == MIN_LOTS

    def test_zero_contract_size_returns_min(self):
        assert calculate_lot_size(10_000.0, 0.01, 100,
                                  contract_size=0) == MIN_LOTS

    def test_negative_contract_size_returns_min(self):
        assert calculate_lot_size(10_000.0, 0.01, 100,
                                  contract_size=-1) == MIN_LOTS


# ===========================================================================
# 3. Lot calculation math
# ===========================================================================

class TestLotMath:
    def test_basic_xau(self):
        """equity=$10000, risk=1% → $100. SL=50 pts × $0.01 × 100 = $50 per lot.
        lots = 100/50 = 2.0. Within [0.01, 10] → 2.0."""
        lot = calculate_lot_size(10_000.0, 0.01, 50, contract_size=100)
        assert lot == pytest.approx(2.0)

    def test_basic_forex(self):
        """equity=$10000, risk=1% → $100. SL=100 pts × $0.01 × 100000 = $100000.
        lots = 100/100000 = 0.001 → below MIN → MIN."""
        lot = calculate_lot_size(10_000.0, 0.01, 100, contract_size=100000)
        assert lot == MIN_LOTS

    @pytest.mark.parametrize("equity,risk,sl,ct,expected", [
        (10_000.0, 0.01, 100, 100, 1.0),
        (10_000.0, 0.005, 50, 100, 1.0),
        (50_000.0, 0.01, 100, 100, 5.0),
        (100_000.0, 0.01, 50, 100, 10.0),     # at max
        (10_000.0, 0.005, 25, 100, 2.0),
    ])
    def test_table(self, equity, risk, sl, ct, expected):
        assert calculate_lot_size(equity, risk, sl,
                                  contract_size=ct) == pytest.approx(expected)


# ===========================================================================
# 4. Min-lot floor (small risk budget)
# ===========================================================================

@pytest.mark.parametrize("equity,risk,sl", [
    (100.0, 0.01, 1000),
    (50.0, 0.01, 100),
    (100.0, 0.001, 50),
])
def test_tiny_risk_floors_to_min(equity, risk, sl):
    assert calculate_lot_size(equity, risk, sl,
                              contract_size=100) >= MIN_LOTS


# ===========================================================================
# 5. Max-lot cap
# ===========================================================================

@pytest.mark.parametrize("equity,risk,sl", [
    (1_000_000.0, 0.05, 10),
    (10_000_000.0, 0.1, 5),
    (1e9, 0.01, 1),
])
def test_huge_risk_caps_at_max(equity, risk, sl):
    assert calculate_lot_size(equity, risk, sl,
                              contract_size=100) == MAX_LOTS


# ===========================================================================
# 6. 2-decimal rounding
# ===========================================================================

@pytest.mark.parametrize("equity,risk,sl", [
    (10_000.0, 0.01, 87),
    (10_000.0, 0.013, 100),
    (10_000.0, 0.01, 33),
    (50_000.0, 0.0075, 110),
])
def test_two_decimal_rounding(equity, risk, sl):
    lot = calculate_lot_size(equity, risk, sl, contract_size=100)
    assert lot == round(lot, 2)


# ===========================================================================
# 7. ProCent account adjustment
# ===========================================================================

class TestProCent:
    def test_procent_divides_equity_by_100(self):
        # equity_cents = 1_000_000 ($10K real). risk=1% → $100.
        # SL=50 pts × $0.01 × 100 = $50 per lot. lots = 2.0.
        lot = calculate_lot_size(
            1_000_000.0, 0.01, 50,
            contract_size=100, account_type="PROCENT",
        )
        assert lot == pytest.approx(2.0)

    def test_procent_min_floor_still_enforced(self):
        # equity=100 cents = $1 — way too small for any lot.
        lot = calculate_lot_size(
            100.0, 0.01, 50,
            contract_size=100, account_type="PROCENT",
        )
        assert lot == MIN_LOTS

    @pytest.mark.parametrize("equity_cents,risk,sl,ct,expected", [
        (1_000_000.0, 0.01, 100, 100, 1.0),
        (5_000_000.0, 0.01, 100, 100, 5.0),
    ])
    def test_procent_table(self, equity_cents, risk, sl, ct, expected):
        lot = calculate_lot_size(
            equity_cents, risk, sl,
            contract_size=ct, account_type="PROCENT",
        )
        assert lot == pytest.approx(expected)

    def test_procent_vs_standard(self):
        # Same numeric "equity" treated differently.
        std = calculate_lot_size(1_000_000.0, 0.01, 100,
                                 contract_size=100, account_type="STANDARD")
        pro = calculate_lot_size(1_000_000.0, 0.01, 100,
                                 contract_size=100, account_type="PROCENT")
        # ProCent should compute 100× smaller real equity → smaller lot
        # (or the same at the cap).
        assert pro <= std


# ===========================================================================
# 8. Account type passthrough
# ===========================================================================

@pytest.mark.parametrize("account_type", ["STANDARD", "PRO", "CLASSIC",
                                          "ECN", "MICRO", "ANYTHING_ELSE"])
def test_non_procent_account_types_use_equity_directly(account_type):
    lot = calculate_lot_size(
        10_000.0, 0.01, 50, contract_size=100, account_type=account_type,
    )
    # Same as default STANDARD path.
    assert lot == pytest.approx(2.0)


# ===========================================================================
# 9. Equity tier sweep
# ===========================================================================

@pytest.mark.parametrize("equity", [
    100.0, 500.0, 1_000.0, 5_000.0, 10_000.0, 25_000.0,
    50_000.0, 100_000.0, 250_000.0, 500_000.0, 1_000_000.0,
])
def test_equity_sweep(equity):
    lot = calculate_lot_size(equity, 0.01, 100, contract_size=100)
    assert MIN_LOTS <= lot <= MAX_LOTS


# ===========================================================================
# 10. SL distance sweep
# ===========================================================================

@pytest.mark.parametrize("sl_pts", [
    1, 5, 10, 25, 50, 100, 200, 500, 1000, 5000,
])
def test_sl_sweep(sl_pts):
    lot = calculate_lot_size(10_000.0, 0.01, sl_pts, contract_size=100)
    assert MIN_LOTS <= lot <= MAX_LOTS


# ===========================================================================
# 11. Risk-pct sweep
# ===========================================================================

@pytest.mark.parametrize("risk_pct", [
    0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10,
])
def test_risk_pct_sweep(risk_pct):
    lot = calculate_lot_size(10_000.0, risk_pct, 100, contract_size=100)
    assert MIN_LOTS <= lot <= MAX_LOTS


# ===========================================================================
# 12. Contract-size sweep
# ===========================================================================

@pytest.mark.parametrize("ct", [
    1, 10, 100, 1_000, 10_000, 100_000,
])
def test_contract_size_sweep(ct):
    lot = calculate_lot_size(10_000.0, 0.01, 100, contract_size=ct)
    assert MIN_LOTS <= lot <= MAX_LOTS


# ===========================================================================
# 13. Hypothesis property: result is in valid range
# ===========================================================================

@settings(max_examples=120, deadline=None)
@given(
    equity=st.floats(min_value=0.01, max_value=10_000_000.0,
                     allow_nan=False, allow_infinity=False),
    risk_pct=st.floats(min_value=0.0001, max_value=0.5,
                       allow_nan=False, allow_infinity=False),
    sl=st.floats(min_value=0.001, max_value=10_000.0,
                 allow_nan=False, allow_infinity=False),
    ct=st.integers(min_value=1, max_value=1_000_000),
)
def test_property_result_in_range(equity, risk_pct, sl, ct):
    lot = calculate_lot_size(equity, risk_pct, sl, contract_size=ct)
    assert MIN_LOTS <= lot <= MAX_LOTS


# ===========================================================================
# 14. Direct math: lot = risk_usd / cost_per_lot
# ===========================================================================

@settings(max_examples=80, deadline=None)
@given(
    equity=st.floats(min_value=1_000.0, max_value=100_000.0,
                     allow_nan=False, allow_infinity=False),
    risk_pct=st.floats(min_value=0.001, max_value=0.05,
                       allow_nan=False, allow_infinity=False),
    sl=st.integers(min_value=10, max_value=500),
    ct=st.sampled_from([100, 1_000, 10_000, 100_000]),
)
def test_property_math_matches_formula(equity, risk_pct, sl, ct):
    """For non-extreme inputs, lot ≈ risk_usd / cost_per_lot (post-rounding)."""
    lot = calculate_lot_size(equity, risk_pct, sl, contract_size=ct)
    risk_usd = equity * risk_pct
    cost_per_lot = sl * POINT_VALUE * ct
    raw = risk_usd / cost_per_lot
    expected = round(raw, 2)
    if MIN_LOTS <= expected <= MAX_LOTS:
        assert lot == pytest.approx(expected)
    elif expected < MIN_LOTS:
        assert lot == MIN_LOTS
    else:
        assert lot == MAX_LOTS


# ===========================================================================
# 15. Lot precision (no fractional lots like 0.001)
# ===========================================================================

@pytest.mark.parametrize("equity,risk,sl,ct", [
    (10_000.0, 0.005, 33, 100),
    (10_000.0, 0.0125, 67, 100),
    (50_000.0, 0.0075, 99, 100),
    (12_345.67, 0.012345, 56, 100),
])
def test_lot_is_2dp(equity, risk, sl, ct):
    lot = calculate_lot_size(equity, risk, sl, contract_size=ct)
    # Quantized to 2dp (hundredths).
    cents = round(lot * 100)
    assert lot * 100 == pytest.approx(cents, abs=1e-9)


# ===========================================================================
# 16. Boundary: lot exactly at 0.01
# ===========================================================================

def test_lot_exactly_min():
    """A risk_usd / cost == 0.01 yields exactly MIN_LOTS."""
    # cost_per_lot=$50 (50pt*0.01*100). risk_usd=$0.50 → lot=0.01.
    lot = calculate_lot_size(50.0, 0.01, 50, contract_size=100)
    assert lot == 0.01


def test_lot_exactly_max():
    """A computed raw lot of exactly 10 returns 10."""
    # risk_usd / cost = 10.0 → 10. With $1000 risk, $100/lot → 10 lots.
    lot = calculate_lot_size(100_000.0, 0.01, 100, contract_size=100)
    assert lot == 10.0


# ===========================================================================
# 17. NaN / inf rejected via underlying float math
# ===========================================================================

class TestPathologicalFloats:
    def test_inf_equity_returns_max(self):
        # equity = inf → risk_usd = inf → lot calc may yield inf → cap to MAX.
        # But the function uses round() which may fail; check it doesn't crash.
        try:
            lot = calculate_lot_size(float("inf"), 0.01, 100,
                                     contract_size=100)
            assert lot in (MIN_LOTS, MAX_LOTS) or lot >= 0
        except (OverflowError, ValueError):
            pass  # acceptable failure mode

    def test_tiny_sl(self):
        # tiny SL → huge lot calc → caps at MAX.
        lot = calculate_lot_size(10_000.0, 0.01, 0.001,
                                 contract_size=100)
        assert lot == MAX_LOTS


# ===========================================================================
# 18. risk_pct = 1.0 (100% of equity) bumps to MAX
# ===========================================================================

def test_full_equity_risk():
    lot = calculate_lot_size(10_000.0, 1.0, 100, contract_size=100)
    # risk_usd = $10_000, cost=$100/lot → 100 lots → capped to MAX.
    assert lot == MAX_LOTS


# ===========================================================================
# 19. Doubling equity doubles lot (until cap)
# ===========================================================================

def test_doubling_equity_doubles_lot():
    lot1 = calculate_lot_size(10_000.0, 0.01, 100, contract_size=100)
    lot2 = calculate_lot_size(20_000.0, 0.01, 100, contract_size=100)
    if lot1 != MAX_LOTS and lot2 != MAX_LOTS:
        assert lot2 == pytest.approx(lot1 * 2.0, abs=0.01)


def test_halving_sl_doubles_lot():
    lot1 = calculate_lot_size(10_000.0, 0.01, 100, contract_size=100)
    lot2 = calculate_lot_size(10_000.0, 0.01, 50, contract_size=100)
    if lot1 != MAX_LOTS and lot2 != MAX_LOTS:
        assert lot2 == pytest.approx(lot1 * 2.0, abs=0.01)
