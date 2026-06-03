"""Phase 8C — prop firm rules tests.

All rule sets load. Required fields present. Logical sanity (target should
be sized appropriately vs caps; daily ≤ total).
"""

from __future__ import annotations

import pytest

from risk.prop_firm.rules import (
    PropFirmRules,
    RULES_DB,
    get_rules,
    list_rule_keys,
)


REQUIRED_KEYS = {
    "ftmo_2step_challenge",
    "ftmo_2step_verification",
    "ftmo_2step_funded",
    "ftmo_1step_challenge",
    "ftmo_1step_funded",
    "the5ers_bootcamp_step1",
    "the5ers_bootcamp_step2",
    "the5ers_bootcamp_step3",
    "the5ers_bootcamp_funded",
    "the5ers_hyper_growth_step1",
    "the5ers_hyper_growth_funded",
    "the5ers_high_stakes_step1",
    "the5ers_high_stakes_step2",
    "the5ers_high_stakes_funded",
}


class TestRulesDb:
    def test_all_required_keys_present(self):
        missing = REQUIRED_KEYS - set(RULES_DB.keys())
        assert not missing, f"missing rule keys: {missing}"

    def test_list_rule_keys_sorted(self):
        keys = list_rule_keys()
        assert keys == tuple(sorted(keys))
        assert set(keys) == set(RULES_DB.keys())

    def test_get_rules_returns_object(self):
        r = get_rules("ftmo_1step_challenge")
        assert isinstance(r, PropFirmRules)
        assert r.name.startswith("FTMO")

    def test_get_rules_unknown_raises(self):
        with pytest.raises(KeyError):
            get_rules("nonexistent_firm")


class TestRuleSanity:
    @pytest.mark.parametrize("key", sorted(REQUIRED_KEYS))
    def test_daily_le_total(self, key):
        r = RULES_DB[key]
        assert r.max_daily_loss_pct <= r.max_total_loss_pct, (
            f"{key}: daily {r.max_daily_loss_pct}% > total {r.max_total_loss_pct}%"
        )

    @pytest.mark.parametrize("key", sorted(REQUIRED_KEYS))
    def test_positive_caps(self, key):
        r = RULES_DB[key]
        assert r.max_daily_loss_pct > 0
        assert r.max_total_loss_pct > 0

    @pytest.mark.parametrize("key", sorted(REQUIRED_KEYS))
    def test_non_negative_target(self, key):
        assert RULES_DB[key].profit_target_pct >= 0

    @pytest.mark.parametrize("key", sorted(REQUIRED_KEYS))
    def test_news_blackout_present(self, key):
        r = RULES_DB[key]
        assert r.news_blackout_minutes_before >= 0
        assert r.news_blackout_minutes_after >= 0

    @pytest.mark.parametrize("key", sorted(REQUIRED_KEYS))
    def test_leverage_positive(self, key):
        r = RULES_DB[key]
        assert r.leverage_forex > 0
        assert r.leverage_metals > 0


class TestSpecificFirmExpectations:
    def test_ftmo_1step_is_strictest_daily(self):
        # 1-Step's defining feature vs 2-Step: 3% / 6% caps.
        r = RULES_DB["ftmo_1step_challenge"]
        assert r.max_daily_loss_pct == 3.0
        assert r.max_total_loss_pct == 6.0

    def test_ftmo_2step_standard_caps(self):
        r = RULES_DB["ftmo_2step_challenge"]
        assert r.max_daily_loss_pct == 5.0
        assert r.max_total_loss_pct == 10.0

    def test_funded_stages_no_profit_target(self):
        for k in ("ftmo_1step_funded", "ftmo_2step_funded",
                  "the5ers_bootcamp_funded", "the5ers_hyper_growth_funded",
                  "the5ers_high_stakes_funded"):
            assert RULES_DB[k].profit_target_pct == 0.0, k

    def test_ftmo_1step_has_consistency_rule(self):
        r = RULES_DB["ftmo_1step_challenge"]
        assert r.consistency_rule is not None
        assert "max_best_day_pct_of_total" in r.consistency_rule

    def test_all_firms_disable_tick_scalping(self):
        # Phase-8 strategy is 1H — both firms forbid tick scalping.
        for k, r in RULES_DB.items():
            assert r.tick_scalp_allowed is False, k


class TestPropFirmRulesValidation:
    def test_rejects_daily_above_total(self):
        with pytest.raises(ValueError):
            PropFirmRules(
                name="bad", max_daily_loss_pct=10.0, max_total_loss_pct=5.0,
                profit_target_pct=5.0, min_trading_days=3,
                leverage_forex=100, leverage_metals=30,
                news_blackout_minutes_before=2, news_blackout_minutes_after=2,
            )

    def test_rejects_zero_total_cap(self):
        with pytest.raises(ValueError):
            PropFirmRules(
                name="bad", max_daily_loss_pct=5.0, max_total_loss_pct=0.0,
                profit_target_pct=5.0, min_trading_days=3,
                leverage_forex=100, leverage_metals=30,
                news_blackout_minutes_before=2, news_blackout_minutes_after=2,
            )

    def test_rejects_negative_target(self):
        with pytest.raises(ValueError):
            PropFirmRules(
                name="bad", max_daily_loss_pct=3.0, max_total_loss_pct=6.0,
                profit_target_pct=-1.0, min_trading_days=3,
                leverage_forex=100, leverage_metals=30,
                news_blackout_minutes_before=2, news_blackout_minutes_after=2,
            )

    def test_rejects_zero_leverage(self):
        with pytest.raises(ValueError):
            PropFirmRules(
                name="bad", max_daily_loss_pct=3.0, max_total_loss_pct=6.0,
                profit_target_pct=5.0, min_trading_days=3,
                leverage_forex=0, leverage_metals=30,
                news_blackout_minutes_before=2, news_blackout_minutes_after=2,
            )
