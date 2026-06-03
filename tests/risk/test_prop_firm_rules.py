"""Prop-firm rule registry tests.

Verifies pinned values for each `PropFirmRules` entry plus the
`__post_init__` invariants and the registry helpers.
"""

from __future__ import annotations

import pytest

from risk.prop_firm.rules import (
    PropFirmRules, RULES_DB, get_rules, list_rule_keys,
)
from tests.risk.fixtures.prop_firm_configs import (
    ALL_KEYS, ALL_FTMO_KEYS, ALL_THE5ERS_KEYS,
    CHALLENGE_KEYS, FUNDED_KEYS, STEP_KEYS,
    FTMO_2STEP_KEYS, FTMO_1STEP_KEYS,
    THE5ERS_BOOTCAMP_KEYS, THE5ERS_HYPER_GROWTH_KEYS,
    THE5ERS_HIGH_STAKES_KEYS,
)


# ===========================================================================
# 1. RULES_DB universe
# ===========================================================================

class TestRulesDb:
    def test_count_matches_fixture(self):
        assert len(RULES_DB) == len(ALL_KEYS)

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_key_present(self, key):
        assert key in RULES_DB

    def test_keys_unique(self):
        assert len(set(RULES_DB.keys())) == len(RULES_DB)

    def test_all_values_are_propfirmrules(self):
        for k, v in RULES_DB.items():
            assert isinstance(v, PropFirmRules), f"{k} not a PropFirmRules"

    def test_list_rule_keys_returns_sorted(self):
        keys = list_rule_keys()
        assert keys == tuple(sorted(keys))

    def test_list_rule_keys_complete(self):
        assert set(list_rule_keys()) == set(RULES_DB.keys())


# ===========================================================================
# 2. get_rules
# ===========================================================================

class TestGetRules:
    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_returns_rules(self, key):
        r = get_rules(key)
        assert isinstance(r, PropFirmRules)
        assert r is RULES_DB[key]

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown prop firm rule key"):
            get_rules("does_not_exist")

    def test_unknown_message_lists_available(self):
        with pytest.raises(KeyError) as exc:
            get_rules("nope")
        assert "Available" in str(exc.value)


# ===========================================================================
# 3. Generic invariants on every rule entry
# ===========================================================================

@pytest.mark.parametrize("key", ALL_KEYS)
class TestRuleInvariants:
    def test_has_name(self, key):
        r = get_rules(key)
        assert isinstance(r.name, str) and r.name.strip()

    def test_max_daily_loss_positive(self, key):
        assert get_rules(key).max_daily_loss_pct > 0

    def test_max_total_loss_positive(self, key):
        assert get_rules(key).max_total_loss_pct > 0

    def test_daily_le_total(self, key):
        r = get_rules(key)
        assert r.max_daily_loss_pct <= r.max_total_loss_pct

    def test_profit_target_non_negative(self, key):
        assert get_rules(key).profit_target_pct >= 0

    def test_min_trading_days_non_negative(self, key):
        assert get_rules(key).min_trading_days >= 0

    def test_leverage_forex_positive(self, key):
        assert get_rules(key).leverage_forex > 0

    def test_leverage_metals_positive(self, key):
        assert get_rules(key).leverage_metals > 0

    def test_news_before_non_negative(self, key):
        assert get_rules(key).news_blackout_minutes_before >= 0

    def test_news_after_non_negative(self, key):
        assert get_rules(key).news_blackout_minutes_after >= 0


# ===========================================================================
# 4. Funded-stage shape: profit_target == 0, min_trading_days == 0
# ===========================================================================

@pytest.mark.parametrize("key", FUNDED_KEYS)
class TestFundedStage:
    def test_no_profit_target(self, key):
        assert get_rules(key).profit_target_pct == 0

    def test_no_min_trading_days(self, key):
        assert get_rules(key).min_trading_days == 0


# ===========================================================================
# 5. Challenge-stage shape: profit_target > 0
# ===========================================================================

@pytest.mark.parametrize("key", CHALLENGE_KEYS)
class TestChallengeStage:
    def test_has_profit_target(self, key):
        assert get_rules(key).profit_target_pct > 0


# ===========================================================================
# 6. Pinned values: FTMO 2-Step Challenge
# ===========================================================================

class TestFtmo2StepChallenge:
    KEY = "ftmo_2step_challenge"

    def test_max_daily_loss(self):
        assert get_rules(self.KEY).max_daily_loss_pct == 5.0

    def test_max_total_loss(self):
        assert get_rules(self.KEY).max_total_loss_pct == 10.0

    def test_profit_target(self):
        assert get_rules(self.KEY).profit_target_pct == 10.0

    def test_min_trading_days(self):
        assert get_rules(self.KEY).min_trading_days == 4

    def test_leverage_forex(self):
        assert get_rules(self.KEY).leverage_forex == 100

    def test_leverage_metals(self):
        assert get_rules(self.KEY).leverage_metals == 30

    def test_news_before(self):
        assert get_rules(self.KEY).news_blackout_minutes_before == 2

    def test_news_after(self):
        assert get_rules(self.KEY).news_blackout_minutes_after == 2

    def test_no_consistency_rule(self):
        assert get_rules(self.KEY).consistency_rule is None

    def test_no_tick_scalp(self):
        assert get_rules(self.KEY).tick_scalp_allowed is False

    def test_overnight_allowed(self):
        assert get_rules(self.KEY).overnight_allowed is True

    def test_no_weekend(self):
        assert get_rules(self.KEY).weekend_allowed is False


# ===========================================================================
# 7. Pinned values: FTMO 1-Step (strictest mode + consistency rule)
# ===========================================================================

class TestFtmo1Step:
    def test_challenge_daily_loss(self):
        assert get_rules("ftmo_1step_challenge").max_daily_loss_pct == 3.0

    def test_challenge_total_loss(self):
        assert get_rules("ftmo_1step_challenge").max_total_loss_pct == 6.0

    def test_funded_daily_loss(self):
        assert get_rules("ftmo_1step_funded").max_daily_loss_pct == 3.0

    def test_funded_total_loss(self):
        assert get_rules("ftmo_1step_funded").max_total_loss_pct == 6.0

    def test_consistency_rule_on_challenge(self):
        rule = get_rules("ftmo_1step_challenge").consistency_rule
        assert rule is not None
        assert rule["max_best_day_pct_of_total"] == 50.0

    def test_consistency_rule_on_funded(self):
        rule = get_rules("ftmo_1step_funded").consistency_rule
        assert rule is not None
        assert rule["max_best_day_pct_of_total"] == 50.0

    def test_funded_no_profit_target(self):
        assert get_rules("ftmo_1step_funded").profit_target_pct == 0

    def test_funded_no_min_days(self):
        assert get_rules("ftmo_1step_funded").min_trading_days == 0


# ===========================================================================
# 8. Pinned values: The5ers Bootcamp
# ===========================================================================

@pytest.mark.parametrize("key", THE5ERS_BOOTCAMP_KEYS)
class TestThe5ersBootcamp:
    def test_daily_loss(self, key):
        assert get_rules(key).max_daily_loss_pct == 4.0

    def test_total_loss(self, key):
        assert get_rules(key).max_total_loss_pct == 5.0

    def test_leverage_forex(self, key):
        assert get_rules(key).leverage_forex == 30

    def test_leverage_metals(self, key):
        assert get_rules(key).leverage_metals == 20

    def test_news_before(self, key):
        assert get_rules(key).news_blackout_minutes_before == 2


@pytest.mark.parametrize("key", [
    "the5ers_bootcamp_step1",
    "the5ers_bootcamp_step2",
    "the5ers_bootcamp_step3",
])
def test_bootcamp_step_profit_target(key):
    assert get_rules(key).profit_target_pct == 6.0


@pytest.mark.parametrize("key", [
    "the5ers_bootcamp_step1",
    "the5ers_bootcamp_step2",
    "the5ers_bootcamp_step3",
])
def test_bootcamp_step_min_days(key):
    assert get_rules(key).min_trading_days == 10


def test_bootcamp_funded_profit_target_zero():
    assert get_rules("the5ers_bootcamp_funded").profit_target_pct == 0


def test_bootcamp_funded_min_days_zero():
    assert get_rules("the5ers_bootcamp_funded").min_trading_days == 0


# ===========================================================================
# 9. Pinned values: The5ers Hyper Growth
# ===========================================================================

@pytest.mark.parametrize("key", THE5ERS_HYPER_GROWTH_KEYS)
class TestThe5ersHyperGrowth:
    def test_daily_loss(self, key):
        assert get_rules(key).max_daily_loss_pct == 5.0

    def test_total_loss(self, key):
        assert get_rules(key).max_total_loss_pct == 10.0


def test_hyper_growth_step_profit_target():
    assert get_rules("the5ers_hyper_growth_step1").profit_target_pct == 10.0


def test_hyper_growth_step_min_days():
    assert get_rules("the5ers_hyper_growth_step1").min_trading_days == 10


def test_hyper_growth_funded_profit_target_zero():
    assert get_rules("the5ers_hyper_growth_funded").profit_target_pct == 0


# ===========================================================================
# 10. Pinned values: The5ers High Stakes
# ===========================================================================

@pytest.mark.parametrize("key", THE5ERS_HIGH_STAKES_KEYS)
class TestThe5ersHighStakes:
    def test_daily_loss(self, key):
        assert get_rules(key).max_daily_loss_pct == 5.0

    def test_total_loss(self, key):
        assert get_rules(key).max_total_loss_pct == 10.0


def test_high_stakes_step1_profit_target():
    assert get_rules("the5ers_high_stakes_step1").profit_target_pct == 8.0


def test_high_stakes_step2_profit_target():
    assert get_rules("the5ers_high_stakes_step2").profit_target_pct == 5.0


def test_high_stakes_step_min_days():
    assert get_rules("the5ers_high_stakes_step1").min_trading_days == 5
    assert get_rules("the5ers_high_stakes_step2").min_trading_days == 5


# ===========================================================================
# 11. PropFirmRules __post_init__ validation
# ===========================================================================

class TestRulesValidation:
    def _kw(self, **over):
        defaults = dict(
            name="X", max_daily_loss_pct=5.0, max_total_loss_pct=10.0,
            profit_target_pct=10.0, min_trading_days=4,
            leverage_forex=100, leverage_metals=30,
            news_blackout_minutes_before=2, news_blackout_minutes_after=2,
        )
        defaults.update(over)
        return defaults

    def test_valid_constructs(self):
        PropFirmRules(**self._kw())

    def test_zero_total_loss_raises(self):
        with pytest.raises(ValueError):
            PropFirmRules(**self._kw(max_total_loss_pct=0.0))

    def test_negative_total_loss_raises(self):
        with pytest.raises(ValueError):
            PropFirmRules(**self._kw(max_total_loss_pct=-1.0))

    def test_zero_daily_loss_raises(self):
        with pytest.raises(ValueError):
            PropFirmRules(**self._kw(max_daily_loss_pct=0.0))

    def test_negative_daily_loss_raises(self):
        with pytest.raises(ValueError):
            PropFirmRules(**self._kw(max_daily_loss_pct=-1.0))

    def test_daily_exceeds_total_raises(self):
        with pytest.raises(ValueError, match="exceeds total"):
            PropFirmRules(**self._kw(
                max_daily_loss_pct=12.0, max_total_loss_pct=10.0,
            ))

    def test_negative_profit_target_raises(self):
        with pytest.raises(ValueError):
            PropFirmRules(**self._kw(profit_target_pct=-1.0))

    def test_zero_profit_target_allowed(self):
        # Funded accounts have zero profit target.
        PropFirmRules(**self._kw(profit_target_pct=0.0))

    def test_negative_min_days_raises(self):
        with pytest.raises(ValueError):
            PropFirmRules(**self._kw(min_trading_days=-1))

    def test_zero_leverage_forex_raises(self):
        with pytest.raises(ValueError):
            PropFirmRules(**self._kw(leverage_forex=0))

    def test_negative_leverage_forex_raises(self):
        with pytest.raises(ValueError):
            PropFirmRules(**self._kw(leverage_forex=-1))

    def test_zero_leverage_metals_raises(self):
        with pytest.raises(ValueError):
            PropFirmRules(**self._kw(leverage_metals=0))


# ===========================================================================
# 12. Frozen dataclass invariants
# ===========================================================================

@pytest.mark.parametrize("key", ALL_KEYS)
def test_rules_are_frozen(key):
    r = get_rules(key)
    with pytest.raises(Exception):
        r.max_daily_loss_pct = 9.99  # type: ignore[misc]


# ===========================================================================
# 13. Family-level cross checks
# ===========================================================================

class TestFamilyShape:
    def test_ftmo_2step_max_daily(self):
        for k in FTMO_2STEP_KEYS:
            assert get_rules(k).max_daily_loss_pct == 5.0

    def test_ftmo_2step_max_total(self):
        for k in FTMO_2STEP_KEYS:
            assert get_rules(k).max_total_loss_pct == 10.0

    def test_ftmo_1step_max_daily(self):
        for k in FTMO_1STEP_KEYS:
            assert get_rules(k).max_daily_loss_pct == 3.0

    def test_ftmo_1step_max_total(self):
        for k in FTMO_1STEP_KEYS:
            assert get_rules(k).max_total_loss_pct == 6.0

    def test_ftmo_leverage(self):
        for k in (*FTMO_2STEP_KEYS, *FTMO_1STEP_KEYS):
            r = get_rules(k)
            assert r.leverage_forex == 100
            assert r.leverage_metals == 30

    def test_the5ers_leverage_forex(self):
        for k in ALL_THE5ERS_KEYS:
            assert get_rules(k).leverage_forex == 30

    def test_the5ers_leverage_metals(self):
        for k in ALL_THE5ERS_KEYS:
            assert get_rules(k).leverage_metals == 20

    def test_all_news_blackout_2min(self):
        for k in ALL_KEYS:
            r = get_rules(k)
            assert r.news_blackout_minutes_before == 2
            assert r.news_blackout_minutes_after == 2


# ===========================================================================
# 14. Phase transitions (lower target as you advance)
# ===========================================================================

class TestPhaseProgression:
    def test_ftmo_2step_target_decreases(self):
        challenge_target = get_rules("ftmo_2step_challenge").profit_target_pct
        verification_target = get_rules("ftmo_2step_verification").profit_target_pct
        funded_target = get_rules("ftmo_2step_funded").profit_target_pct
        assert challenge_target > verification_target > funded_target

    def test_high_stakes_target_decreases(self):
        s1 = get_rules("the5ers_high_stakes_step1").profit_target_pct
        s2 = get_rules("the5ers_high_stakes_step2").profit_target_pct
        funded = get_rules("the5ers_high_stakes_funded").profit_target_pct
        assert s1 > s2 > funded

    def test_ftmo_1step_strictest_daily(self):
        # 1-step has the strictest daily cap (3%) vs 2-step (5%).
        assert (get_rules("ftmo_1step_challenge").max_daily_loss_pct
                < get_rules("ftmo_2step_challenge").max_daily_loss_pct)

    def test_ftmo_1step_strictest_total(self):
        assert (get_rules("ftmo_1step_challenge").max_total_loss_pct
                < get_rules("ftmo_2step_challenge").max_total_loss_pct)


# ===========================================================================
# 15. Names are unique
# ===========================================================================

def test_rule_names_unique():
    names = {get_rules(k).name for k in ALL_KEYS}
    assert len(names) == len(ALL_KEYS)


# ===========================================================================
# 16. Consistency rule structure
# ===========================================================================

class TestConsistencyRule:
    def test_only_ftmo_1step_has_consistency_in_v5_db(self):
        with_rule = [
            k for k in ALL_KEYS if get_rules(k).consistency_rule is not None
        ]
        assert set(with_rule) == {"ftmo_1step_challenge", "ftmo_1step_funded"}

    def test_consistency_rule_value(self):
        for k in ("ftmo_1step_challenge", "ftmo_1step_funded"):
            r = get_rules(k).consistency_rule
            assert r == {"max_best_day_pct_of_total": 50.0}


# ===========================================================================
# 17. Defaults of optional fields
# ===========================================================================

class TestDefaults:
    def test_overnight_default_true(self):
        rules = PropFirmRules(
            name="X", max_daily_loss_pct=1.0, max_total_loss_pct=2.0,
            profit_target_pct=5.0, min_trading_days=0,
            leverage_forex=100, leverage_metals=30,
            news_blackout_minutes_before=2, news_blackout_minutes_after=2,
        )
        assert rules.overnight_allowed is True

    def test_weekend_default_false(self):
        rules = PropFirmRules(
            name="X", max_daily_loss_pct=1.0, max_total_loss_pct=2.0,
            profit_target_pct=5.0, min_trading_days=0,
            leverage_forex=100, leverage_metals=30,
            news_blackout_minutes_before=2, news_blackout_minutes_after=2,
        )
        assert rules.weekend_allowed is False

    def test_tick_scalp_default_false(self):
        rules = PropFirmRules(
            name="X", max_daily_loss_pct=1.0, max_total_loss_pct=2.0,
            profit_target_pct=5.0, min_trading_days=0,
            leverage_forex=100, leverage_metals=30,
            news_blackout_minutes_before=2, news_blackout_minutes_after=2,
        )
        assert rules.tick_scalp_allowed is False

    def test_consistency_default_none(self):
        rules = PropFirmRules(
            name="X", max_daily_loss_pct=1.0, max_total_loss_pct=2.0,
            profit_target_pct=5.0, min_trading_days=0,
            leverage_forex=100, leverage_metals=30,
            news_blackout_minutes_before=2, news_blackout_minutes_after=2,
        )
        assert rules.consistency_rule is None
