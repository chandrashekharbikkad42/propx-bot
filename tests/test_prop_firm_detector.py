"""Phase 8C — PropFirmDetector tests."""

from __future__ import annotations

import pytest

from risk.prop_firm.detector import AccountInfo, PropFirmDetector


@pytest.fixture
def det():
    return PropFirmDetector()


class TestServerDetection:
    def test_ftmo_demo(self, det):
        assert det.detect_from_mt5(AccountInfo(server="FTMO-Demo")) == "ftmo_2step_challenge"

    def test_ftmo_server(self, det):
        assert det.detect_from_mt5(AccountInfo(server="FTMO-Server01")) == "ftmo_2step_challenge"

    def test_ftmo_lowercase(self, det):
        assert det.detect_from_mt5(AccountInfo(server="ftmo-server")) == "ftmo_2step_challenge"

    def test_the5ers(self, det):
        assert det.detect_from_mt5(AccountInfo(server="The5ers-Server")) == "the5ers_bootcamp_step1"

    def test_5ers_short(self, det):
        # "5ers" pattern alone — Sometimes branded that way.
        assert det.detect_from_mt5(AccountInfo(server="MyBroker-5ers-Live")) == "the5ers_bootcamp_step1"


class TestNonPropBrokers:
    def test_roboforex_returns_none(self, det):
        assert det.detect_from_mt5(AccountInfo(server="RoboForex-Pro")) is None

    def test_ic_markets_returns_none(self, det):
        assert det.detect_from_mt5(AccountInfo(server="ICMarkets-Live01")) is None
        assert det.detect_from_mt5(AccountInfo(server="IC-Markets-Demo")) is None

    def test_pepperstone_returns_none(self, det):
        assert det.detect_from_mt5(AccountInfo(server="Pepperstone-Live")) is None


class TestUnknownBroker:
    def test_returns_none_for_unknown(self, det):
        assert det.detect_from_mt5(AccountInfo(server="MysteryBroker-X")) is None

    def test_empty_account_info_returns_none(self, det):
        assert det.detect_from_mt5(AccountInfo(server="", company="")) is None


class TestCompanyFallback:
    def test_company_used_when_server_empty(self, det):
        assert det.detect_from_mt5(AccountInfo(server="", company="FTMO Ltd")) == "ftmo_2step_challenge"

    def test_company_can_mark_non_prop(self, det):
        assert det.detect_from_mt5(AccountInfo(server="", company="RoboForex Ltd")) is None


class TestConfigOverride:
    def test_valid_config_returned(self, det):
        assert det.detect_from_config("ftmo_1step_funded") == "ftmo_1step_funded"

    def test_invalid_config_raises(self, det):
        with pytest.raises(ValueError):
            det.detect_from_config("nope_firm")

    def test_none_config_returns_none(self, det):
        assert det.detect_from_config(None) is None

    def test_empty_config_returns_none(self, det):
        assert det.detect_from_config("") is None


class TestDetectPriority:
    def test_config_beats_mt5(self, det):
        """User override always wins, even if MT5 matches a different firm."""
        result = det.detect(
            AccountInfo(server="FTMO-Server"),
            configured_key="the5ers_high_stakes_funded",
        )
        assert result == "the5ers_high_stakes_funded"

    def test_mt5_used_when_no_config(self, det):
        assert det.detect(AccountInfo(server="FTMO-Demo")) == "ftmo_2step_challenge"

    def test_non_prop_returns_none(self, det):
        assert det.detect(AccountInfo(server="RoboForex-Pro")) is None

    def test_invalid_config_still_raises(self, det):
        # Even with a matching MT5 server, an invalid config raises rather
        # than silently falling through.
        with pytest.raises(ValueError):
            det.detect(AccountInfo(server="FTMO-Demo"), configured_key="bogus")
