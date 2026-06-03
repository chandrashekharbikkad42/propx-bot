"""PropFirmDetector — server-name / company-name pattern matching."""

from __future__ import annotations

import pytest

from risk.prop_firm.detector import (
    AccountInfo, PropFirmDetector,
)
from risk.prop_firm.rules import RULES_DB


@pytest.fixture
def detector():
    return PropFirmDetector()


# ===========================================================================
# 1. AccountInfo dataclass
# ===========================================================================

class TestAccountInfo:
    def test_defaults(self):
        ai = AccountInfo(server="FTMO-Demo")
        assert ai.server == "FTMO-Demo"
        assert ai.company == ""
        assert ai.login == 0
        assert ai.balance == 0.0

    def test_frozen(self):
        ai = AccountInfo(server="FTMO")
        with pytest.raises(Exception):
            ai.server = "X"  # type: ignore[misc]


# ===========================================================================
# 2. detect_from_mt5 — FTMO patterns
# ===========================================================================

@pytest.mark.parametrize("server", [
    "FTMO",
    "FTMO-Demo",
    "FTMO_Live",
    "FTMO-Server",
    "FTMO-Server-01",
    "ftmo-demo",       # case-insensitive
    "Ftmo-Live",
])
def test_ftmo_server_detected(detector, server):
    rule = detector.detect_from_mt5(AccountInfo(server=server))
    assert rule == "ftmo_2step_challenge"


@pytest.mark.parametrize("company", [
    "FTMO",
    "FTMO-Demo",
    "FTMO_Capital",
    "ftmo",
    "Ftmo Trading",
])
def test_ftmo_company_detected(detector, company):
    rule = detector.detect_from_mt5(AccountInfo(server="", company=company))
    assert rule == "ftmo_2step_challenge"


# ===========================================================================
# 3. The5ers patterns
# ===========================================================================

@pytest.mark.parametrize("server", [
    "The5ers",
    "The5ers-Live",
    "The5ers-Demo",
    "The5ers_Server",
    "the5ers",
    "THE5ERS-Live",
])
def test_the5ers_server_detected(detector, server):
    rule = detector.detect_from_mt5(AccountInfo(server=server))
    assert rule == "the5ers_bootcamp_step1"


@pytest.mark.parametrize("server", [
    "MyServer-5ers",
    "5ers-Server",
    "Funded5ers-Demo",
])
def test_5ers_substring_detected(detector, server):
    rule = detector.detect_from_mt5(AccountInfo(server=server))
    assert rule == "the5ers_bootcamp_step1"


# ===========================================================================
# 4. Non-prop brokers → None
# ===========================================================================

@pytest.mark.parametrize("server", [
    "RoboForex",
    "RoboForex-Demo",
    "RoboForex-Live",
    "roboforex-ecn",
])
def test_roboforex_non_prop(detector, server):
    assert detector.detect_from_mt5(AccountInfo(server=server)) is None


@pytest.mark.parametrize("server", [
    "ICMarkets",
    "ICMarkets-Live",
    "icmarkets-demo",
    "IC-Markets",
    "IC_Markets-Live",
    "IC Markets",
])
def test_icmarkets_non_prop(detector, server):
    assert detector.detect_from_mt5(AccountInfo(server=server)) is None


@pytest.mark.parametrize("server", [
    "Pepperstone",
    "Pepperstone-Live",
    "pepperstone-demo",
])
def test_pepperstone_non_prop(detector, server):
    assert detector.detect_from_mt5(AccountInfo(server=server)) is None


@pytest.mark.parametrize("company", [
    "RoboForex",
    "ICMarkets",
    "IC Markets",
    "Pepperstone",
])
def test_non_prop_company(detector, company):
    assert detector.detect_from_mt5(
        AccountInfo(server="", company=company)
    ) is None


# ===========================================================================
# 5. Edge cases: empty input
# ===========================================================================

class TestEmptyInput:
    def test_empty_both(self, detector):
        assert detector.detect_from_mt5(AccountInfo(server="", company="")) is None

    def test_whitespace_only(self, detector):
        # `.strip()` makes whitespace → empty → None
        assert detector.detect_from_mt5(
            AccountInfo(server="   ", company="   ")
        ) is None

    def test_none_handled_as_empty_string_by_default(self, detector):
        # AccountInfo defaults company to "" — server is required.
        assert detector.detect_from_mt5(AccountInfo(server="")) is None


# ===========================================================================
# 6. Unknown brokers → None
# ===========================================================================

@pytest.mark.parametrize("server", [
    "Unknown-Broker",
    "MyBroker-Server",
    "DemoServer-Forex",
    "12345-Server",
    "RandomString",
])
def test_unknown_broker_returns_none(detector, server):
    assert detector.detect_from_mt5(AccountInfo(server=server)) is None


# ===========================================================================
# 7. Misspelled / partial matches
# ===========================================================================

class TestMisspelled:
    @pytest.mark.parametrize("server", [
        "FT-MO",        # break in pattern
        "F.T.M.O",
        "FT MO",
    ])
    def test_ftmo_misspelled_not_detected(self, detector, server):
        assert detector.detect_from_mt5(
            AccountInfo(server=server)
        ) is None

    @pytest.mark.parametrize("server", [
        "MyFTMO-Server",   # FTMO must be at start
        "Demo-FTMO",
    ])
    def test_ftmo_must_be_at_start(self, detector, server):
        assert detector.detect_from_mt5(
            AccountInfo(server=server)
        ) is None

    def test_partial_5ers_matched(self, detector):
        # 5ers pattern uses .search(), so anywhere in string matches.
        assert detector.detect_from_mt5(
            AccountInfo(server="x5ers-Server")
        ) == "the5ers_bootcamp_step1"


# ===========================================================================
# 8. Non-prop takes precedence over prop
# ===========================================================================

class TestPrecedence:
    def test_roboforex_with_ftmo_substring_treated_as_non_prop(self, detector):
        # An odd hypothetical server name combining both — the non-prop
        # allow-list short-circuits because RoboForex matches first.
        assert detector.detect_from_mt5(
            AccountInfo(server="RoboForex-FTMO")
        ) is None


# ===========================================================================
# 9. detect_from_config
# ===========================================================================

class TestDetectFromConfig:
    @pytest.mark.parametrize("key", list(RULES_DB.keys()))
    def test_valid_key_returns_self(self, detector, key):
        assert detector.detect_from_config(key) == key

    def test_none_returns_none(self, detector):
        assert detector.detect_from_config(None) is None

    def test_empty_string_returns_none(self, detector):
        assert detector.detect_from_config("") is None

    def test_unknown_key_raises(self, detector):
        with pytest.raises(ValueError, match="Unknown prop firm rule key"):
            detector.detect_from_config("not_a_real_key")

    def test_unknown_key_message_lists_available(self, detector):
        with pytest.raises(ValueError) as exc:
            detector.detect_from_config("xyz")
        assert "Available" in str(exc.value)


# ===========================================================================
# 10. detect — config wins over MT5 server match
# ===========================================================================

class TestDetectComposite:
    def test_config_wins_over_server_match(self, detector):
        ai = AccountInfo(server="FTMO-Demo")
        assert detector.detect(ai, configured_key="ftmo_1step_funded") == \
            "ftmo_1step_funded"

    def test_no_config_falls_back_to_server(self, detector):
        ai = AccountInfo(server="FTMO-Demo")
        assert detector.detect(ai, configured_key=None) == \
            "ftmo_2step_challenge"

    def test_no_config_no_server_match_returns_none(self, detector):
        ai = AccountInfo(server="Unknown")
        assert detector.detect(ai, configured_key=None) is None

    def test_empty_config_treated_as_none(self, detector):
        ai = AccountInfo(server="FTMO-Demo")
        assert detector.detect(ai, configured_key="") == \
            "ftmo_2step_challenge"

    def test_invalid_config_raises(self, detector):
        ai = AccountInfo(server="FTMO-Demo")
        with pytest.raises(ValueError):
            detector.detect(ai, configured_key="invalid-key")


# ===========================================================================
# 11. Per-pattern smoke (sanity that the patterns compile & match)
# ===========================================================================

FTMO_VARIANTS = [
    "FTMO", "FTMO-Demo", "FTMO_Live", "ftmo", "Ftmo-Server",
    "FTMO-Server-01", "FTMO_Server", "FTMO-",
]


@pytest.mark.parametrize("server", FTMO_VARIANTS)
def test_ftmo_variants(detector, server):
    assert detector.detect_from_mt5(AccountInfo(server=server)) == \
        "ftmo_2step_challenge"


@pytest.mark.parametrize("server", [
    "The5ers", "the5ers", "THE5ERS-Live", "The5ers-Demo",
    "The5ers Capital", "The5ers_Server",
])
def test_the5ers_variants(detector, server):
    assert detector.detect_from_mt5(AccountInfo(server=server)) == \
        "the5ers_bootcamp_step1"


# ===========================================================================
# 12. Subclassing for custom brokers
# ===========================================================================

class TestSubclass:
    def test_can_subclass(self):
        class MyDetector(PropFirmDetector):
            pass
        assert MyDetector().detect_from_mt5(
            AccountInfo(server="FTMO-Demo")
        ) == "ftmo_2step_challenge"


# ===========================================================================
# 13. Detector keyed result always present in RULES_DB
# ===========================================================================

DETECTABLE_SERVERS = [
    "FTMO", "FTMO-Demo", "FTMO_Live",
    "The5ers", "The5ers-Live", "The5ers-Demo",
    "MyServer-5ers", "Funded5ers-Server",
]


@pytest.mark.parametrize("server", DETECTABLE_SERVERS)
def test_detected_key_in_rules_db(detector, server):
    rule = detector.detect_from_mt5(AccountInfo(server=server))
    assert rule in RULES_DB


# ===========================================================================
# 14. AccountInfo extra fields don't break detection
# ===========================================================================

def test_balance_and_login_ignored(detector):
    ai = AccountInfo(
        server="FTMO-Demo", company="FTMO",
        login=12345, balance=50_000.0,
    )
    assert detector.detect_from_mt5(ai) == "ftmo_2step_challenge"


# ===========================================================================
# 15. Whitespace tolerance
# ===========================================================================

def test_leading_trailing_whitespace_stripped(detector):
    ai = AccountInfo(server="  FTMO-Demo  ")
    assert detector.detect_from_mt5(ai) == "ftmo_2step_challenge"


def test_company_whitespace_stripped(detector):
    ai = AccountInfo(server="", company="  FTMO  ")
    assert detector.detect_from_mt5(ai) == "ftmo_2step_challenge"
