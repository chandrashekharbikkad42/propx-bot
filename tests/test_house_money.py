"""Phase 8C — HouseMoneyManager tests.

Per spec:
  Trade 1 → base risk per grade (A=1.0%, B=0.5%).
  Trade 2 after win → base + (todays_pnl/equity)*HM_FRACTION, capped at 2× base.
  Trade 2 after loss → base × DEFENSIVE_MULT.
  Trade 2 after exactly 0 → defensive (treat as loss; neither buffer to redeploy
  nor evidence of a winning setup).
  C-grade → raises (scanner filters first).
  trade_number_today must be 1 or 2.
"""

from __future__ import annotations

import pytest

from risk.house_money import (
    DEFAULT_BASE_RISK_PCT,
    DEFENSIVE_MULT,
    HOUSE_MONEY_FRACTION,
    HouseMoneyManager,
    MAX_HOUSE_MONEY_MULT,
    RiskAllocation,
)
from strategy.patterns.base import Grade


@pytest.fixture
def hm():
    return HouseMoneyManager()


class TestBaseRiskPct:
    def test_a_grade_is_one_percent(self):
        assert DEFAULT_BASE_RISK_PCT[Grade.A] == 1.0

    def test_b_grade_is_half_percent(self):
        assert DEFAULT_BASE_RISK_PCT[Grade.B] == 0.5

    def test_c_grade_is_zero(self):
        assert DEFAULT_BASE_RISK_PCT[Grade.C] == 0.0


class TestTrade1Standard:
    def test_grade_a_returns_one_pct(self, hm):
        a = hm.calc_trade_risk(Grade.A, equity=10_000, todays_pnl_usd=0.0, trade_number_today=1)
        assert isinstance(a, RiskAllocation)
        assert a.mode == "STANDARD"
        assert a.final_risk_pct == 1.0
        assert a.base_risk_pct == 1.0

    def test_grade_b_returns_half_pct(self, hm):
        a = hm.calc_trade_risk(Grade.B, equity=10_000, todays_pnl_usd=0.0, trade_number_today=1)
        assert a.mode == "STANDARD"
        assert a.final_risk_pct == 0.5

    def test_trade_1_ignores_pnl(self, hm):
        # PnL is from yesterday or open positions — irrelevant for trade 1 today.
        a1 = hm.calc_trade_risk(Grade.A, equity=10_000, todays_pnl_usd=100.0, trade_number_today=1)
        a2 = hm.calc_trade_risk(Grade.A, equity=10_000, todays_pnl_usd=-100.0, trade_number_today=1)
        assert a1.final_risk_pct == a2.final_risk_pct == 1.0


class TestTrade2HouseMoney:
    def test_after_win_extra_risk_applied(self, hm):
        # Equity 10k, today's profit $200 (= 2% of equity).
        # extra = (200/10000)*100*0.5 = 1.0%. base 1.0% + 1.0% = 2.0%. Cap = 1.0 * 2 = 2.0%.
        a = hm.calc_trade_risk(Grade.A, equity=10_000, todays_pnl_usd=200.0, trade_number_today=2)
        assert a.mode == "HOUSE_MONEY"
        assert a.final_risk_pct == pytest.approx(2.0)

    def test_house_money_cap_enforced(self, hm):
        # Huge profit — extra would push way past cap. Should clamp at 2× base.
        a = hm.calc_trade_risk(Grade.A, equity=10_000, todays_pnl_usd=10_000.0, trade_number_today=2)
        assert a.mode == "HOUSE_MONEY"
        assert a.final_risk_pct == pytest.approx(DEFAULT_BASE_RISK_PCT[Grade.A] * MAX_HOUSE_MONEY_MULT)

    def test_small_win_under_cap(self, hm):
        # Equity 10k, $50 win. extra = (50/10000)*100*0.5 = 0.25%. base 1% + 0.25% = 1.25%.
        a = hm.calc_trade_risk(Grade.A, equity=10_000, todays_pnl_usd=50.0, trade_number_today=2)
        assert a.final_risk_pct == pytest.approx(1.25)

    def test_grade_b_house_money(self, hm):
        # Equity 10k, $100 win. extra = 0.5%. B base 0.5%. Sum 1.0%. Cap 1.0%. Exactly at cap.
        a = hm.calc_trade_risk(Grade.B, equity=10_000, todays_pnl_usd=100.0, trade_number_today=2)
        assert a.final_risk_pct == pytest.approx(1.0)


class TestTrade2Defensive:
    def test_after_loss_halves_risk(self, hm):
        a = hm.calc_trade_risk(Grade.A, equity=10_000, todays_pnl_usd=-100.0, trade_number_today=2)
        assert a.mode == "DEFENSIVE"
        assert a.final_risk_pct == pytest.approx(0.5)

    def test_after_zero_treated_as_defensive(self, hm):
        a = hm.calc_trade_risk(Grade.A, equity=10_000, todays_pnl_usd=0.0, trade_number_today=2)
        assert a.mode == "DEFENSIVE"
        assert a.final_risk_pct == pytest.approx(0.5)

    def test_b_grade_defensive(self, hm):
        a = hm.calc_trade_risk(Grade.B, equity=10_000, todays_pnl_usd=-50.0, trade_number_today=2)
        assert a.final_risk_pct == pytest.approx(0.25)


class TestGuards:
    def test_c_grade_rejected(self, hm):
        with pytest.raises(ValueError):
            hm.calc_trade_risk(Grade.C, equity=10_000, todays_pnl_usd=0.0, trade_number_today=1)

    def test_trade_3_rejected(self, hm):
        with pytest.raises(ValueError):
            hm.calc_trade_risk(Grade.A, equity=10_000, todays_pnl_usd=0.0, trade_number_today=3)

    def test_zero_equity_rejected(self, hm):
        with pytest.raises(ValueError):
            hm.calc_trade_risk(Grade.A, equity=0.0, todays_pnl_usd=0.0, trade_number_today=1)

    def test_negative_equity_rejected(self, hm):
        with pytest.raises(ValueError):
            hm.calc_trade_risk(Grade.A, equity=-100, todays_pnl_usd=0.0, trade_number_today=1)


class TestConstructorValidation:
    def test_negative_house_money_fraction_rejected(self):
        with pytest.raises(ValueError):
            HouseMoneyManager(house_money_fraction=-0.1)

    def test_house_money_cap_below_one_rejected(self):
        with pytest.raises(ValueError):
            HouseMoneyManager(max_house_money_mult=0.5)

    def test_defensive_mult_above_one_rejected(self):
        with pytest.raises(ValueError):
            HouseMoneyManager(defensive_mult=1.5)

    def test_defensive_mult_negative_rejected(self):
        with pytest.raises(ValueError):
            HouseMoneyManager(defensive_mult=-0.1)


class TestCustomTunables:
    def test_custom_base_risk(self):
        hm = HouseMoneyManager(base_risk_pct={Grade.A: 2.0, Grade.B: 1.0, Grade.C: 0.0})
        a = hm.calc_trade_risk(Grade.A, 10_000, 0.0, 1)
        assert a.final_risk_pct == 2.0

    def test_custom_defensive_mult(self):
        hm = HouseMoneyManager(defensive_mult=0.25)
        a = hm.calc_trade_risk(Grade.A, 10_000, -100, 2)
        assert a.final_risk_pct == pytest.approx(0.25)


class TestDailySummary:
    def test_summary_keys(self, hm):
        s = hm.daily_summary(equity=10_000, grade_for_both=Grade.A)
        assert {"worst_pct", "best_pct", "base_pct", "trade_2_house_money_pct"} <= set(s.keys())

    def test_worst_negative_best_positive(self, hm):
        s = hm.daily_summary(equity=10_000, grade_for_both=Grade.A)
        assert s["worst_pct"] < 0
        assert s["best_pct"] > 0

    def test_zero_equity_returns_zeroes(self, hm):
        s = hm.daily_summary(equity=0, grade_for_both=Grade.A)
        assert s["worst_pct"] == 0.0 and s["best_pct"] == 0.0
