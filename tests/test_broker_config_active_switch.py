"""Phase 9 — ACTIVE_BROKER env-routing tests.

Verifies that:
  - ACTIVE_BROKER=FTMO returns FTMO_* credentials
  - ACTIVE_BROKER=ROBOFOREX returns MT5_* credentials
  - missing ACTIVE_BROKER defaults to ROBOFOREX
  - invalid ACTIVE_BROKER raises
  - missing required vars under a broker trigger graceful fallback to the
    other broker with a warning
  - both sides missing → raises BrokerCredentialsMissing
"""

from __future__ import annotations
import os
import unittest
from unittest.mock import patch

from config.broker_config import (
    BrokerCredentialsMissing,
    active_broker_name,
    get_active_credentials,
    get_credentials_for,
)


FTMO_ENV = {
    "ACTIVE_BROKER": "FTMO",
    "FTMO_LOGIN": "1513426156",
    "FTMO_PASSWORD": "ftmo-pass",
    "FTMO_SERVER": "FTMO-Demo",
    "MT5_PATH": r"C:\Program Files\MetaTrader 5\terminal64.exe",
}

ROBOFOREX_ENV = {
    "ACTIVE_BROKER": "ROBOFOREX",
    "MT5_LOGIN": "37345118",
    "MT5_PASSWORD": "robo-pass",
    "MT5_SERVER": "RoboForex-Pro",
    "MT5_PATH": r"C:\Program Files\MetaTrader 5\terminal64.exe",
}


def _clear_env() -> dict[str, str]:
    """Snapshot then clear ALL broker-related env vars for an isolated test.

    Covers three families that all feed `config.broker_config`:
      - ACTIVE_BROKER selector
      - Legacy Phase-4 prefixes (FTMO_*, MT5_*, ROBOFOREX_*)
      - New Phase-5/B profile keys (BROKER_<NAME>_*)
    Real `.env` populates BROKER_FTMO_LOGIN and BROKER_ROBOFOREX_LOGIN, so
    if these aren't stripped, the profile lookup in `_credentials_for`
    short-circuits past the legacy fallback the tests are exercising.
    """
    snapshot: dict[str, str] = {}
    for k in list(os.environ):
        if (
            k == "ACTIVE_BROKER"
            or k.startswith(("BROKER_", "FTMO_", "ROBOFOREX_"))
            or k in {
                "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH",
            }
        ):
            snapshot[k] = os.environ.pop(k)
    return snapshot


def _restore_env(snapshot: dict[str, str]) -> None:
    for k, v in snapshot.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


class _EnvCase(unittest.TestCase):
    """Base — snapshot ALL broker env vars and restore after each test so
    parallel runs / .env-loaded prod values don't bleed into expectations."""

    def setUp(self):
        self._snapshot = _clear_env()

    def tearDown(self):
        _restore_env(self._snapshot)


# ============================================================================


class TestActiveBrokerName(_EnvCase):
    def test_explicit_ftmo(self):
        os.environ["ACTIVE_BROKER"] = "FTMO"
        self.assertEqual(active_broker_name(), "FTMO")

    def test_explicit_roboforex(self):
        os.environ["ACTIVE_BROKER"] = "ROBOFOREX"
        self.assertEqual(active_broker_name(), "ROBOFOREX")

    def test_lowercase_normalised(self):
        os.environ["ACTIVE_BROKER"] = "ftmo"
        self.assertEqual(active_broker_name(), "FTMO")

    def test_unset_defaults_to_roboforex(self):
        self.assertEqual(active_broker_name(), "ROBOFOREX")

    def test_unknown_passes_through_unvalidated(self):
        # Phase-5/B: active_broker_name() no longer validates against a
        # hardcoded enum — any uppercase name is returned and the credential
        # lookup is the safety net (see TestGetCredentialsFor below).
        os.environ["ACTIVE_BROKER"] = "BLOFIN"
        self.assertEqual(active_broker_name(), "BLOFIN")
        with self.assertRaises(BrokerCredentialsMissing):
            get_credentials_for("BLOFIN")


# ============================================================================


class TestGetActiveCredentials(_EnvCase):
    def test_ftmo_path(self):
        os.environ.update(FTMO_ENV)
        c = get_active_credentials()
        self.assertEqual(c.broker, "FTMO")
        self.assertEqual(c.login, 1_513_426_156)
        self.assertEqual(c.password, "ftmo-pass")
        self.assertEqual(c.server, "FTMO-Demo")

    def test_roboforex_path(self):
        os.environ.update(ROBOFOREX_ENV)
        c = get_active_credentials()
        self.assertEqual(c.broker, "ROBOFOREX")
        self.assertEqual(c.login, 37_345_118)
        self.assertEqual(c.password, "robo-pass")
        self.assertEqual(c.server, "RoboForex-Pro")

    def test_ftmo_falls_back_to_roboforex_when_ftmo_missing(self):
        # ACTIVE_BROKER says FTMO but only RoboForex env is present.
        os.environ.update(ROBOFOREX_ENV)
        os.environ["ACTIVE_BROKER"] = "FTMO"
        c = get_active_credentials()
        self.assertEqual(c.broker, "ROBOFOREX")
        self.assertEqual(c.login, 37_345_118)

    def test_roboforex_falls_back_to_ftmo_when_mt5_missing(self):
        os.environ.update(FTMO_ENV)
        os.environ["ACTIVE_BROKER"] = "ROBOFOREX"
        c = get_active_credentials()
        self.assertEqual(c.broker, "FTMO")
        self.assertEqual(c.login, 1_513_426_156)

    def test_both_sides_missing_raises(self):
        os.environ["ACTIVE_BROKER"] = "FTMO"
        with self.assertRaises(BrokerCredentialsMissing):
            get_active_credentials()


class TestGetCredentialsFor(_EnvCase):
    def test_force_ftmo(self):
        os.environ.update(FTMO_ENV | ROBOFOREX_ENV)
        os.environ["ACTIVE_BROKER"] = "ROBOFOREX"  # FTMO override ignores this
        c = get_credentials_for("FTMO")
        self.assertEqual(c.broker, "FTMO")
        self.assertEqual(c.login, 1_513_426_156)

    def test_force_roboforex(self):
        os.environ.update(FTMO_ENV | ROBOFOREX_ENV)
        os.environ["ACTIVE_BROKER"] = "FTMO"
        c = get_credentials_for("ROBOFOREX")
        self.assertEqual(c.broker, "ROBOFOREX")
        self.assertEqual(c.login, 37_345_118)

    def test_force_unknown_raises(self):
        with self.assertRaises(RuntimeError):
            get_credentials_for("BLOFIN")

    def test_non_integer_login_raises(self):
        os.environ["FTMO_LOGIN"] = "not-a-number"
        os.environ["FTMO_PASSWORD"] = "x"
        os.environ["FTMO_SERVER"] = "y"
        with self.assertRaises(BrokerCredentialsMissing):
            get_credentials_for("FTMO")


if __name__ == "__main__":
    unittest.main()
