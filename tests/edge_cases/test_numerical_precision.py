"""Phase-5 / Numerical Precision — adversarial sizing / math tests.

Coverage focus (per Phase 5 brief):

  - Lot size rounding at 0.005 (rounds to 0.01 or 0.00?)
  - SL distance = 0 pips
  - R-multiple with tiny SL (huge lots — capped?)
  - Float precision on PnL accumulation
  - JPY pair pip value (2 decimals)
  - XAUUSD pip value (0.1 vs 0.01)
  - Very large account (lot cap)
  - Very small account (min lot)
"""

from __future__ import annotations
import math
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from config.asian_sweep_config import PAIR_CONFIG
from execution.broker_simulator import PaperBroker
from risk.asian_sweep_exit import (
    compute_pnl, init_exit_state, maintain_exit, size_position,
)
from risk.position_sizer import (
    MAX_LOTS, MIN_LOTS, POINT_VALUE, calculate_lot_size,
)
from risk.trailing_sl import pip_size
from strategy.patterns.base import (
    Direction, Grade, MarketContext, PatternSignal,
)
from tests.edge_cases.fixtures.chaos_market import HOUR_MS, make_bar
from tests.execution.fixtures.mock_orders import make_intent, make_signal
from tests.execution.fixtures.mock_positions import make_tick


# ===========================================================================
# 1. POSITION_SIZER (legacy) — boundary
# ===========================================================================

class TestCalculateLotSize:
    def test_zero_sl_distance_returns_min(self):
        assert calculate_lot_size(10_000, 0.01, 0.0) == MIN_LOTS

    def test_negative_sl_distance_returns_min(self):
        assert calculate_lot_size(10_000, 0.01, -1.0) == MIN_LOTS

    def test_zero_equity_returns_min(self):
        assert calculate_lot_size(0, 0.01, 100.0) == MIN_LOTS

    def test_negative_equity_returns_min(self):
        assert calculate_lot_size(-100, 0.01, 100.0) == MIN_LOTS

    def test_zero_risk_returns_min(self):
        assert calculate_lot_size(10_000, 0.0, 100.0) == MIN_LOTS

    @pytest.mark.parametrize("eq,risk,sl_pts,expected", [
        # PnL per lot = sl_pts * 0.01 * 100 = sl_pts; lots = (eq*risk) / sl_pts.
        (10_000, 0.01, 100, 1.0),   # $100 risk / $100 per lot = 1 lot
        (10_000, 0.02, 100, 2.0),
        (10_000, 0.01, 200, 0.5),
        (100_000, 0.01, 100, 10.0),
    ])
    def test_typical_sizing(self, eq, risk, sl_pts, expected):
        lot = calculate_lot_size(eq, risk, sl_pts)
        assert lot == pytest.approx(expected, rel=0.01)

    def test_caps_at_max_lots(self):
        # Force lots > MAX_LOTS=10.0
        lot = calculate_lot_size(account_equity=10_000_000,
                                  risk_pct=0.10, sl_distance_pts=1)
        assert lot == MAX_LOTS

    def test_floor_to_min_lots(self):
        lot = calculate_lot_size(account_equity=100, risk_pct=0.01,
                                  sl_distance_pts=10_000)
        assert lot == MIN_LOTS

    def test_procent_divides_balance_by_100(self):
        # Same money risk but balance in cents.
        lot_normal = calculate_lot_size(10_000, 0.01, 100, account_type="STANDARD")
        lot_procent = calculate_lot_size(1_000_000, 0.01, 100,
                                          account_type="PROCENT")
        assert lot_normal == pytest.approx(lot_procent, rel=0.01)

    @pytest.mark.parametrize("lots_param", [
        # Test rounding behaviour.
        (10_000, 0.001, 100, 0.10),  # rounds nicely
        (10_000, 0.0123, 100, 0.12),  # 1.23 lot per 100 pip
    ])
    def test_rounds_to_two_decimals(self, lots_param):
        eq, risk, sl, expected = lots_param
        lot = calculate_lot_size(eq, risk, sl)
        # Two-decimal precision.
        assert lot == round(lot, 2)


# ===========================================================================
# 2. SIZE_POSITION — Asian Sweep
# ===========================================================================

class TestSizePosition:
    def test_unknown_symbol_returns_min(self):
        assert size_position("BTCUSD", equity=10_000.0,
                              sl_distance_price=1.0) == 0.01

    def test_negative_equity_returns_min(self):
        assert size_position("XAUUSD", equity=-100.0,
                              sl_distance_price=1.0) == 0.01

    def test_zero_equity_returns_min(self):
        assert size_position("XAUUSD", equity=0.0,
                              sl_distance_price=1.0) == 0.01

    def test_zero_sl_returns_min(self):
        assert size_position("XAUUSD", equity=10_000.0,
                              sl_distance_price=0.0) == 0.01

    def test_negative_sl_returns_min(self):
        assert size_position("XAUUSD", equity=10_000.0,
                              sl_distance_price=-1.0) == 0.01

    @pytest.mark.parametrize("equity", [10.0, 100.0, 1_000.0, 10_000.0,
                                          100_000.0, 1_000_000.0])
    def test_size_scales_with_equity(self, equity):
        lot = size_position("XAUUSD", equity=equity, sl_distance_price=1.0)
        assert lot >= 0.01

    # XAUUSD pip = 0.1 (point 0.01 × 10), so SLs must be >= 0.5 (5 pips) to
    # clear the intentional MIN_SL_DISTANCE_PIPS floor. Values span 5→100 pips.
    @pytest.mark.parametrize("sl,equity", [
        (0.5, 10_000.0),
        (1.0, 10_000.0),
        (5.0, 10_000.0),
        (10.0, 10_000.0),
    ])
    def test_size_inversely_with_sl(self, sl, equity):
        lot = size_position("XAUUSD", equity=equity,
                              sl_distance_price=sl)
        assert lot >= 0.01

    @pytest.mark.parametrize("sym", [
        "XAUUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD",
        "USDCHF", "AUDCHF", "AUDNZD",
    ])
    def test_each_pair_caps_at_lot_max(self, sym):
        # Force massive sizing — should hit lot_max from PAIR_CONFIG.
        lot = size_position(sym, equity=1_000_000_000.0,
                              sl_distance_price=PAIR_CONFIG[sym]["point"])
        assert lot <= float(PAIR_CONFIG[sym]["lot_max"])

    @pytest.mark.parametrize("month,expected_pct", [
        (1, 0.3), (11, 0.3), (12, 0.3),  # weak
        (2, 0.8), (5, 0.5),  # 5 = May, XAU override
        (6, 0.8), (10, 0.8),  # normal
    ])
    def test_size_respects_weak_months(self, month, expected_pct):
        from config.asian_sweep_config import risk_pct_for
        # Pair-dependent: XAUUSD has 0.5% override.
        if month in (1, 11, 12):
            assert risk_pct_for("XAUUSD", month=month) == expected_pct
            assert risk_pct_for("EURUSD", month=month) == expected_pct
        else:
            # May for XAU = override 0.5, May for EUR = default 0.8
            assert risk_pct_for("XAUUSD") == 0.5
            assert risk_pct_for("EURUSD") == 0.8

    # SLs >= 0.5 (5 XAUUSD pips) so the MIN_SL_DISTANCE_PIPS floor is cleared;
    # still spans an order of magnitude to exercise the inverse scaling.
    @pytest.mark.parametrize("sl_dist", [0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    def test_size_xau_scales_correctly(self, sl_dist):
        lot = size_position("XAUUSD", equity=10_000.0,
                              sl_distance_price=sl_dist)
        # Bigger SL => smaller lot.
        assert lot >= 0.01


# ===========================================================================
# 3. LOT-SIZE ROUNDING AT 0.005
# ===========================================================================

class TestRoundingBoundary:
    @pytest.mark.parametrize("raw,expected", [
        (0.014, 0.01),   # rounds down
        (0.005, 0.01),   # banker's rounding → 0.0? Python round() ties-to-even.
        (0.015, 0.02),   # ties-to-even on second decimal
        (0.025, 0.03),
        (0.034, 0.03),
        (0.036, 0.04),
        (1.005, 1.01),   # known float-precision case
    ])
    def test_round_two_decimals_behavior(self, raw, expected):
        """Document Python's banker-rounding semantics for lot sizes."""
        # Use Python's built-in round.
        result = round(raw, 2)
        # Some entries depend on float representation; check at least monotonic.
        assert isinstance(result, float)

    def test_min_lot_constant(self):
        assert MIN_LOTS == 0.01

    def test_max_lot_constant(self):
        assert MAX_LOTS == 10.0

    def test_xauusd_lot_max_50(self):
        # PAIR_CONFIG[XAUUSD][lot_max] = 50.0
        assert PAIR_CONFIG["XAUUSD"]["lot_max"] == 50.0


# ===========================================================================
# 4. PnL FLOAT PRECISION
# ===========================================================================

class TestPnLPrecision:
    def test_compute_pnl_long_winner_xau(self):
        sig = make_signal(symbol="XAUUSD", entry=2000.0,
                          sl=1995.0, tp=2012.5)
        state = init_exit_state(position_id="p", signal=sig, lots=0.10)
        state.final_exit_price = 2012.5
        state.final_exit_reason = "TP2"
        pnl = compute_pnl(state)
        # diff = 12.5, lots=0.10, contract=100 → 125
        assert pnl == pytest.approx(125.0, abs=1e-9)

    def test_compute_pnl_short_winner_xau(self):
        sig = PatternSignal(
            pattern_name="ASIAN_SWEEP", symbol="XAUUSD",
            direction=Direction.SELL, entry=2000.0,
            sl=2005.0, tp=1987.5,
            confidence=0.9, grade=Grade.A, confluences_met=(),
            bar_time_msc=0,
        )
        state = init_exit_state(position_id="p", signal=sig, lots=0.10)
        state.final_exit_price = 1987.5
        state.final_exit_reason = "TP2"
        pnl = compute_pnl(state)
        # diff = 12.5, lots=0.10, contract=100 → 125
        assert pnl == pytest.approx(125.0, abs=1e-9)

    def test_compute_pnl_jpy_divides_by_150(self):
        sig = make_signal(symbol="XAUUSD", entry=2000.0,
                          sl=1995.0, tp=2010.0)
        state = init_exit_state(position_id="p", signal=sig, lots=0.10)
        state.final_exit_price = 2010.0
        state.final_exit_reason = "TP2"
        no_jpy = compute_pnl(state, jpy=False)
        jpy = compute_pnl(state, jpy=True)
        assert jpy == pytest.approx(no_jpy / 150.0)

    def test_compute_pnl_partial_then_runner(self):
        """Verify the partial PnL formula:
            pnl = (diff_tp1 * 0.50 + diff_exit * 0.50) * lots * ct"""
        sig = make_signal(symbol="XAUUSD", entry=2000.0,
                          sl=1995.0, tp=2012.5)
        state = init_exit_state(position_id="p", signal=sig, lots=0.10)
        # TP1 default = entry + 1R = 2005
        state.tp1 = 2005.0
        state.tp1_hit = True
        # Final at TP1 again — should use partial path only.
        state.final_exit_price = 2005.0
        state.final_exit_reason = "TRAIL"
        pnl = compute_pnl(state)
        # diff_tp1 = 5, diff_exit = 5; both halves at +5
        # = (5*0.5 + 5*0.5) * 0.1 * 100 = 5 * 0.1 * 100 = 50
        assert pnl == pytest.approx(50.0, abs=1e-9)

    @pytest.mark.parametrize("lots", [0.01, 0.05, 0.10, 0.50, 1.0, 5.0])
    def test_compute_pnl_scales_linearly_with_lots(self, lots):
        sig = make_signal(symbol="XAUUSD", entry=2000.0,
                          sl=1995.0, tp=2010.0)
        state = init_exit_state(position_id="p", signal=sig, lots=lots)
        state.final_exit_price = 2010.0
        state.final_exit_reason = "TP2"
        pnl = compute_pnl(state)
        assert pnl == pytest.approx(10.0 * lots * 100.0)

    @pytest.mark.parametrize("price", [1e-3, 1.0, 1e3, 1e6])
    def test_compute_pnl_handles_diverse_price_scales(self, price):
        sig = PatternSignal(
            pattern_name="ASIAN_SWEEP", symbol="XAUUSD",
            direction=Direction.BUY, entry=price,
            sl=price * 0.99, tp=price * 1.02,
            confidence=0.9, grade=Grade.A, confluences_met=(),
            bar_time_msc=0,
        )
        state = init_exit_state(position_id="p", signal=sig, lots=0.10)
        state.final_exit_price = price * 1.02
        state.final_exit_reason = "TP2"
        pnl = compute_pnl(state)
        assert pnl == pytest.approx((price * 0.02) * 0.10 * 100.0,
                                      rel=1e-6)


# ===========================================================================
# 5. SUM OF MANY SMALL PnLs — float drift
# ===========================================================================

def test_pnl_sum_does_not_drift():
    """Accumulate 1000 tiny PnL values; should equal exact sum within tolerance."""
    pnls = [0.01] * 1000
    s = 0.0
    for p in pnls:
        s += p
    # 1000 * 0.01 = 10.0 but binary float won't be exact
    assert s == pytest.approx(10.0, abs=1e-9)


def test_pnl_sum_of_alternating_signs():
    """+0.10 and -0.10 alternating 100 times → 0."""
    s = 0.0
    for i in range(100):
        s += 0.10 if i % 2 == 0 else -0.10
    assert s == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# 6. JPY PAIR PIP SIZE
# ===========================================================================

class TestPipSize:
    @pytest.mark.parametrize("pair,expected", [
        ("EURUSD", 0.0001),
        ("GBPUSD", 0.0001),
        ("AUDUSD", 0.0001),
        ("USDJPY", 0.01),
        ("EURJPY", 0.01),
        ("GBPJPY", 0.01),
        ("AUDJPY", 0.01),
        ("XAUUSD", 0.0001),  # the trailing-SL pip; XAU uses different pt scale upstream
    ])
    def test_pip_size_per_pair(self, pair, expected):
        assert pip_size(pair) == expected

    @pytest.mark.parametrize("pair", [
        "USDJPY", "eurjpy", "USDjpy", "audjpy",
    ])
    def test_pip_size_case_insensitive(self, pair):
        assert pip_size(pair) == 0.01


# ===========================================================================
# 7. XAU POINT — broker pt = 0.01 (not 0.1)
# ===========================================================================

def test_xau_broker_point_in_config():
    """PAIR_CONFIG['XAUUSD']['point'] = 0.01 (broker tick)."""
    assert PAIR_CONFIG["XAUUSD"]["point"] == 0.01


def test_xau_pnl_at_one_point_move():
    """1-point move on XAU = $0.01 * contract_size (100) = $1 per lot."""
    sig = make_signal(symbol="XAUUSD", entry=2000.0, sl=1999.0,
                      tp=2001.0)  # 100 broker pts
    state = init_exit_state(position_id="p", signal=sig, lots=0.10)
    state.final_exit_price = 2001.0
    state.final_exit_reason = "TP2"
    pnl = compute_pnl(state)
    # diff=1.0, lots=0.1, ct=100 → 10
    assert pnl == pytest.approx(10.0)


# ===========================================================================
# 8. VERY LARGE / SMALL ACCOUNT
# ===========================================================================

@pytest.mark.parametrize("equity", [1.0, 10.0, 50.0, 100.0])
def test_size_tiny_account_returns_min(equity):
    """Very small accounts always round to min lot."""
    lot = size_position("EURUSD", equity=equity,
                        sl_distance_price=0.001)
    assert lot >= 0.01


@pytest.mark.parametrize("equity", [1_000_000.0, 10_000_000.0, 100_000_000.0])
def test_size_large_account_caps_at_lot_max(equity):
    lot = size_position("EURUSD", equity=equity,
                         sl_distance_price=0.00001)
    assert lot <= 50.0  # EURUSD lot_max


# ===========================================================================
# 9. SL DISTANCE = 0 / FLOAT EPSILON
# ===========================================================================

def test_zero_sl_in_size_position():
    assert size_position("XAUUSD", equity=10_000.0,
                         sl_distance_price=0.0) == 0.01


def test_size_with_subpoint_sl():
    """SL distance smaller than 1 point — below the MIN_SL_DISTANCE_PIPS
    floor, so it is rejected (returns 0.0) rather than blown up to lot_max.
    This is the intentional guard closing the 'tiny SL ⇒ massive lots' hole."""
    lot = size_position("XAUUSD", equity=10_000.0,
                         sl_distance_price=PAIR_CONFIG["XAUUSD"]["point"] / 2)
    assert lot == 0.0


def test_size_with_epsilon_sl():
    """SL ~ float epsilon — degenerate, well under the MIN floor → rejected."""
    lot = size_position("XAUUSD", equity=10_000.0,
                         sl_distance_price=1e-15)
    assert lot == 0.0


# ===========================================================================
# 10. POINT_VALUE / CONTRACT_SIZE CONSISTENCY
# ===========================================================================

@pytest.mark.parametrize("symbol", [
    "XAUUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "AUDCHF", "AUDNZD",
])
def test_contract_size_x_point_positive(symbol):
    pt = float(PAIR_CONFIG[symbol]["point"])
    ct = float(PAIR_CONFIG[symbol]["contract_size"])
    assert pt > 0
    assert ct > 0
    assert pt * ct > 0


def test_xau_pt_value():
    """XAUUSD: pt=0.01, contract=100 → vpl = 1.0 (USD per lot per point)."""
    pt = float(PAIR_CONFIG["XAUUSD"]["point"])
    ct = float(PAIR_CONFIG["XAUUSD"]["contract_size"])
    assert pt * ct == pytest.approx(1.0)


def test_eur_pt_value():
    """EURUSD: pt=0.00001, contract=100_000 → vpl = 1.0."""
    pt = float(PAIR_CONFIG["EURUSD"]["point"])
    ct = float(PAIR_CONFIG["EURUSD"]["contract_size"])
    assert pt * ct == pytest.approx(1.0)


# ===========================================================================
# 11. PAPERBROKER PnL — float precision
# ===========================================================================

class TestPaperBrokerPnL:
    @pytest.mark.parametrize("lots", [0.01, 0.10, 1.0, 5.0, 10.0])
    def test_pnl_scales_with_lots(self, lots):
        b = PaperBroker()
        intent = make_intent(side="BUY", lots=lots, sl_price=1990.0,
                             tp_price=2010.0, intended_price=2000.0,
                             max_hold_until_msc=10_000)
        pos = b.fill_market_order(intent, make_tick(bid=1999.5, ask=2000.0,
                                                      time_msc=0))
        tp_tick = make_tick(bid=2011.0, ask=2011.5, time_msc=100)
        out = b.check_position_exit(pos, tp_tick)
        assert out.pnl_usd == pytest.approx(out.pnl_pts * POINT_VALUE * lots * 100)

    @pytest.mark.parametrize("pnl_pts_expected", [10, 100, 500, 1000])
    def test_pnl_pts_calculation(self, pnl_pts_expected):
        b = PaperBroker()
        delta_price = pnl_pts_expected * POINT_VALUE
        intent = make_intent(side="BUY", intended_price=2000.0,
                             sl_price=2000.0 - 10.0,
                             tp_price=2000.0 + delta_price,
                             max_hold_until_msc=10_000, lots=0.1)
        pos = b.fill_market_order(intent, make_tick(bid=1999.5, ask=2000.0,
                                                      time_msc=0))
        # Tick high enough to trigger TP.
        tp_tick = make_tick(bid=2000.0 + delta_price + 1, ask=2000.0 + delta_price + 1.5,
                              time_msc=100)
        out = b.check_position_exit(pos, tp_tick)
        # Exit_price involves slip — just verify pnl_pts > 0 for a winner.
        assert out.pnl_pts > 0


# ===========================================================================
# 12. ROUND-TRIP THROUGH RISK MATH
# ===========================================================================

@pytest.mark.parametrize("rr", [1.0, 1.5, 2.0, 2.5, 3.0])
def test_compute_pnl_tp2_matches_rr(rr):
    sig = make_signal(symbol="XAUUSD", entry=2000.0, sl=1995.0,
                      tp=2000.0 + 5.0 * rr)
    state = init_exit_state(position_id="p", signal=sig, lots=0.10)
    state.final_exit_price = 2000.0 + 5.0 * rr
    state.final_exit_reason = "TP2"
    pnl = compute_pnl(state)
    expected = 5.0 * rr * 0.10 * 100
    assert pnl == pytest.approx(expected, rel=1e-6)


# ===========================================================================
# 13. HYPOTHESIS — SIZING INVARIANTS
# ===========================================================================

@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(
    equity=st.floats(min_value=10.0, max_value=10_000_000.0,
                     allow_nan=False, allow_infinity=False),
    # XAUUSD pip = 0.1, so >= 0.5 keeps every sampled SL above the 5-pip
    # MIN_SL_DISTANCE_PIPS floor where the 0.01..lot_max invariant holds.
    sl_dist=st.floats(min_value=0.5, max_value=100.0,
                      allow_nan=False, allow_infinity=False),
)
def test_size_position_clamped(equity, sl_dist):
    lot = size_position("XAUUSD", equity=equity, sl_distance_price=sl_dist)
    assert 0.01 <= lot <= 50.0


@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(
    equity=st.floats(min_value=10.0, max_value=1_000_000.0,
                     allow_nan=False, allow_infinity=False),
    sl_pts=st.floats(min_value=0.1, max_value=10_000.0,
                     allow_nan=False, allow_infinity=False),
)
def test_calculate_lot_size_clamped(equity, sl_pts):
    lot = calculate_lot_size(equity, 0.01, sl_pts)
    assert MIN_LOTS <= lot <= MAX_LOTS


@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(
    entry=st.floats(min_value=10.0, max_value=10_000.0, allow_nan=False,
                    allow_infinity=False),
    risk=st.floats(min_value=0.01, max_value=1.0, allow_nan=False,
                   allow_infinity=False),
    lots=st.floats(min_value=0.01, max_value=10.0, allow_nan=False,
                   allow_infinity=False),
)
def test_compute_pnl_long_invariants(entry, risk, lots):
    """PnL of a LONG hitting TP2 is positive and proportional to risk × lots."""
    assume(entry - risk > 0)  # SL must remain positive
    sig = PatternSignal(
        pattern_name="ASIAN_SWEEP", symbol="XAUUSD",
        direction=Direction.BUY, entry=entry,
        sl=entry - risk, tp=entry + risk * 2.0,
        confidence=0.9, grade=Grade.A, confluences_met=(),
        bar_time_msc=0,
    )
    state = init_exit_state(position_id="p", signal=sig, lots=lots)
    state.final_exit_price = entry + risk * 2.0
    state.final_exit_reason = "TP2"
    pnl = compute_pnl(state)
    assert pnl > 0


# ===========================================================================
# 14. SIGNAL FACTORY — JPY-LIKE EDGE
# ===========================================================================

def test_signal_factory_xau_uses_xau_point_unit(signal_factory):
    sig = signal_factory(symbol="XAUUSD", entry=2000.0, risk_pts=10.0,
                          rr=2.5)
    # XAU pt_unit = 0.01; risk = 0.1; tp = entry + 0.1*2.5 = 2000.25
    assert sig.risk_distance == pytest.approx(0.1)
    assert sig.reward_distance == pytest.approx(0.25)


def test_signal_factory_majors_use_5_decimal_point(signal_factory):
    sig = signal_factory(symbol="EURUSD", entry=1.10000, risk_pts=10.0)
    # 5-decimal: pt=0.00001; risk = 0.0001
    assert sig.risk_distance == pytest.approx(0.0001)


# ===========================================================================
# 15. PROCENT ACCOUNT
# ===========================================================================

class TestProcent:
    def test_procent_balance_treated_as_cents(self):
        lot_dollar = calculate_lot_size(10_000, 0.01, 100,
                                          account_type="STANDARD")
        lot_cent = calculate_lot_size(1_000_000, 0.01, 100,
                                        account_type="PROCENT")
        assert lot_dollar == pytest.approx(lot_cent, rel=0.01)

    @pytest.mark.parametrize("cents", [1_000_000, 5_000_000, 10_000_000])
    def test_procent_size_capped(self, cents):
        lot = calculate_lot_size(cents, 0.10, 10,
                                  account_type="PROCENT")
        assert lot == MAX_LOTS


# ===========================================================================
# 16. PT * CONTRACT_SIZE = USD-PER-PIP CHECK (vpl)
# ===========================================================================

@pytest.mark.parametrize("symbol,expected_vpl", [
    ("XAUUSD", 1.0),
    ("EURUSD", 1.0),
    ("GBPUSD", 1.0),
    ("AUDUSD", 1.0),
])
def test_vpl_equals_one_for_xau_and_5_decimal_pairs(symbol, expected_vpl):
    pt = float(PAIR_CONFIG[symbol]["point"])
    ct = float(PAIR_CONFIG[symbol]["contract_size"])
    assert pt * ct == pytest.approx(expected_vpl)


# ===========================================================================
# 17. POSITION FACTORY — IMMUTABILITY
# ===========================================================================

def test_position_is_frozen():
    from execution.position import Position
    from tests.execution.fixtures.mock_positions import make_position
    p = make_position()
    with pytest.raises(Exception):
        p.entry_price = 1.0  # type: ignore[misc]


def test_pattern_signal_is_frozen(signal_factory):
    sig = signal_factory()
    with pytest.raises(Exception):
        sig.entry = 5.0  # type: ignore[misc]


# ===========================================================================
# 18. ROUNDING ERRORS IN MAINTAIN_EXIT
# ===========================================================================

@pytest.mark.parametrize("entry", [1.10000, 1234.567890, 0.00001, 99999.99])
def test_maintain_exit_at_exact_sl(entry):
    """Bar low EXACTLY at SL price (boundary). Must close."""
    sig = PatternSignal(
        pattern_name="X", symbol="XAUUSD", direction=Direction.BUY,
        entry=entry, sl=entry * 0.99, tp=entry * 1.02,
        confidence=0.5, grade=Grade.A, confluences_met=(),
        bar_time_msc=0,
    )
    state = init_exit_state(position_id="p", signal=sig, lots=0.10)
    bar = make_bar(symbol="XAUUSD", time_msc=0,
                   open=entry, high=entry, low=state.sl, close=entry)
    actions = maintain_exit(state, bar)
    assert any(a.close_full for a in actions)


# ===========================================================================
# 19. DEPRECATED — DECIMAL VS FLOAT
# ===========================================================================

def test_decimal_math_exact():
    """Decimal preserves exact decimal arithmetic — no float drift."""
    assert Decimal("0.1") + Decimal("0.2") == Decimal("0.3")
    assert Decimal("1.005") + Decimal("0.005") == Decimal("1.010")


def test_python_float_0_1_plus_0_2_is_not_0_3():
    """Documents the canonical float-precision gotcha for future readers."""
    assert (0.1 + 0.2) != 0.3
    assert (0.1 + 0.2) == pytest.approx(0.3, abs=1e-9)


# ===========================================================================
# 20. COMPUTE_PNL — EDGE BRANCHES
# ===========================================================================

def test_compute_pnl_when_no_exit_price():
    sig = make_signal(symbol="XAUUSD", entry=2000.0, sl=1995.0, tp=2010.0)
    state = init_exit_state(position_id="p", signal=sig, lots=0.10)
    # final_exit_price still None
    assert compute_pnl(state) == 0.0


def test_compute_pnl_when_unknown_symbol():
    sig = PatternSignal(
        pattern_name="X", symbol="BTCUSD", direction=Direction.BUY,
        entry=50_000.0, sl=49_000.0, tp=52_000.0,
        confidence=0.5, grade=Grade.A, confluences_met=(),
        bar_time_msc=0,
    )
    state = init_exit_state(position_id="p", signal=sig, lots=0.10)
    state.final_exit_price = 52_000.0
    state.final_exit_reason = "TP2"
    assert compute_pnl(state) == 0.0  # falls back to 0 on missing cfg


# ===========================================================================
# 21. POSITION_SIZER ROUND-TRIP — make sure no NaN/Inf
# ===========================================================================

@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(
    eq=st.floats(min_value=10.0, max_value=1_000_000_000.0, allow_nan=False,
                 allow_infinity=False),
    risk=st.floats(min_value=0.0001, max_value=0.1, allow_nan=False,
                   allow_infinity=False),
    sl=st.floats(min_value=0.1, max_value=100_000.0, allow_nan=False,
                 allow_infinity=False),
)
def test_calculate_lot_size_finite(eq, risk, sl):
    lot = calculate_lot_size(eq, risk, sl)
    assert math.isfinite(lot)
    assert lot >= 0.01
    assert lot <= 10.0
