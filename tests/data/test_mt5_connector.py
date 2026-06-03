"""MT5Connector — wraps the real MetaTrader5 module.

Tests monkey-patch `data.mt5_connector.mt5` with MockMT5 so no real terminal
is needed. The `settings` singleton is patched per-test so connection
parameters are deterministic.
"""

from __future__ import annotations
import time as time_mod
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from data.mt5_connector import (
    AccountInfo, MT5ConnectionError, MT5Connector, SymbolInfo,
)

from tests.execution.fixtures.mock_mt5 import (
    MockMT5, AccountInfoResult, DealInfo, OrderSendResult,
    PositionInfo, SymbolInfoResult, TerminalInfoResult, TickInfoResult,
    TIMEFRAME_H1,
)


# ---------------------------------------------------------------------------
# Settings stub (mirrors only the fields the connector uses)
# ---------------------------------------------------------------------------

class _SettingsStub:
    def __init__(
        self,
        symbol="XAUUSD",
        mt5_login=12345,
        mt5_password="pw",
        mt5_server="DemoServer",
        mt5_path="C:/MT5/terminal64.exe",
        mt5_timeout_ms=5000,
        mt5_retry_attempts=2,
        mt5_retry_delay_sec=0,
    ):
        self.symbol = symbol
        self.mt5_login = mt5_login
        self.mt5_password = mt5_password
        self.mt5_server = mt5_server
        self.mt5_path = mt5_path
        self.mt5_timeout_ms = mt5_timeout_ms
        self.mt5_retry_attempts = mt5_retry_attempts
        self.mt5_retry_delay_sec = mt5_retry_delay_sec


@pytest.fixture
def patch_settings(monkeypatch):
    """Replace data.mt5_connector.settings with a stub."""
    s = _SettingsStub()
    from data import mt5_connector
    monkeypatch.setattr(mt5_connector, "settings", s)
    return s


@pytest.fixture
def mock_mt5(monkeypatch):
    """Patch data.mt5_connector.mt5 with a fresh MockMT5."""
    m = MockMT5()
    from data import mt5_connector
    monkeypatch.setattr(mt5_connector, "mt5", m)
    return m


@pytest.fixture
def fast_sleep(monkeypatch):
    """Make time.sleep a no-op so retry loops don't slow tests."""
    monkeypatch.setattr(time_mod, "sleep", lambda *_a, **_k: None)


# ===========================================================================
# 1. Dataclasses
# ===========================================================================

class TestDataclasses:
    def test_account_info_frozen(self):
        a = AccountInfo(login=1, server="X", balance=1.0, equity=1.0,
                        leverage=100, currency="USD", company="C")
        import dataclasses
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.login = 2  # type: ignore[misc]

    def test_symbol_info_frozen(self):
        s = SymbolInfo(name="X", digits=2, point=0.01, spread_points=20,
                       trade_tick_size=0.01, trade_tick_value=1.0,
                       contract_size=100.0)
        import dataclasses
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.digits = 5  # type: ignore[misc]


# ===========================================================================
# 2. Construction
# ===========================================================================

class TestConstructor:
    def test_default_uses_settings_symbol(self, patch_settings):
        c = MT5Connector()
        assert c.symbol == patch_settings.symbol

    @pytest.mark.parametrize("sym", ["EURUSD", "GBPUSD", "USDJPY",
                                     "XAUUSD", "BTCUSD"])
    def test_custom_symbol(self, sym, patch_settings):
        c = MT5Connector(symbol=sym)
        assert c.symbol == sym

    def test_starts_disconnected(self, patch_settings):
        c = MT5Connector()
        assert c.is_connected is False


# ===========================================================================
# 3. connect — success
# ===========================================================================

class TestConnect:
    def test_success_sets_connected(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        assert c.is_connected is True

    def test_idempotent_when_already_connected(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        # Second call should no-op
        c.connect()
        assert c.is_connected is True

    def test_initialize_called(self, patch_settings, mock_mt5):
        spy = MagicMock(return_value=True)
        mock_mt5.initialize = spy
        c = MT5Connector()
        c.connect()
        spy.assert_called_once()

    def test_login_called(self, patch_settings, mock_mt5):
        spy = MagicMock(return_value=True)
        mock_mt5.login = spy
        c = MT5Connector()
        c.connect()
        spy.assert_called_once()

    def test_symbol_select_when_not_visible(self, patch_settings, mock_mt5):
        # symbol_info_obj.visible=False forces symbol_select
        mock_mt5.symbol_info_obj.visible = False
        spy = MagicMock(return_value=True)
        mock_mt5.symbol_select = spy
        c = MT5Connector()
        c.connect()
        spy.assert_called_once()

    def test_symbol_select_skipped_when_visible(self, patch_settings, mock_mt5):
        mock_mt5.symbol_info_obj.visible = True
        spy = MagicMock(return_value=True)
        mock_mt5.symbol_select = spy
        c = MT5Connector()
        c.connect()
        spy.assert_not_called()


# ===========================================================================
# 4. connect — failure paths
# ===========================================================================

class TestConnectFailures:
    def test_initialize_failure_then_retry_succeeds(self, patch_settings,
                                                     mock_mt5, fast_sleep):
        attempts = {"count": 0}

        def init(*a, **kw):
            attempts["count"] += 1
            return attempts["count"] >= 2

        mock_mt5.initialize = init
        c = MT5Connector()
        c.connect()
        assert c.is_connected
        assert attempts["count"] == 2

    def test_initialize_exhausts_retries(self, patch_settings,
                                          mock_mt5, fast_sleep):
        mock_mt5.initialize_returns = False
        c = MT5Connector()
        with pytest.raises(MT5ConnectionError, match="connection failed"):
            c.connect()
        assert c.is_connected is False

    def test_login_failure(self, patch_settings, mock_mt5, fast_sleep):
        mock_mt5.login_returns = False
        c = MT5Connector()
        with pytest.raises(MT5ConnectionError):
            c.connect()

    def test_symbol_not_in_market_watch(self, patch_settings, mock_mt5,
                                         fast_sleep):
        mock_mt5.symbol_info_obj = None
        c = MT5Connector()
        with pytest.raises(MT5ConnectionError):
            c.connect()

    def test_symbol_select_failure(self, patch_settings, mock_mt5, fast_sleep):
        mock_mt5.symbol_info_obj.visible = False
        mock_mt5.symbol_select_returns = False
        c = MT5Connector()
        with pytest.raises(MT5ConnectionError):
            c.connect()


# ===========================================================================
# 5. disconnect
# ===========================================================================

class TestDisconnect:
    def test_disconnect_after_connect(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        c.disconnect()
        assert c.is_connected is False
        assert mock_mt5.shutdown_calls == 1

    def test_disconnect_when_not_connected(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.disconnect()
        assert mock_mt5.shutdown_calls == 0


# ===========================================================================
# 6. Context manager
# ===========================================================================

class TestContextManager:
    def test_enter_connects(self, patch_settings, mock_mt5):
        with MT5Connector() as c:
            assert c.is_connected is True

    def test_exit_disconnects(self, patch_settings, mock_mt5):
        with MT5Connector() as c:
            pass
        assert c.is_connected is False
        assert mock_mt5.shutdown_calls == 1

    def test_exit_on_exception_still_disconnects(self, patch_settings,
                                                  mock_mt5):
        with pytest.raises(RuntimeError):
            with MT5Connector():
                raise RuntimeError("boom")
        assert mock_mt5.shutdown_calls == 1


# ===========================================================================
# 7. account_info
# ===========================================================================

class TestAccountInfo:
    def test_basic(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        info = c.account_info()
        assert isinstance(info, AccountInfo)
        assert info.login == mock_mt5.account.login
        assert info.balance == mock_mt5.account.balance

    def test_requires_connection(self, patch_settings, mock_mt5):
        c = MT5Connector()
        with pytest.raises(MT5ConnectionError, match="Not connected"):
            c.account_info()

    def test_none_response_raises(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        mock_mt5.account = None  # type: ignore[assignment]
        with pytest.raises(MT5ConnectionError, match="account_info None"):
            c.account_info()


# ===========================================================================
# 8. symbol_info
# ===========================================================================

class TestSymbolInfo:
    def test_basic(self, patch_settings, mock_mt5):
        c = MT5Connector(symbol="EURUSD")
        c.connect()
        info = c.symbol_info()
        assert isinstance(info, SymbolInfo)
        assert info.name == "EURUSD"

    def test_requires_connection(self, patch_settings, mock_mt5):
        c = MT5Connector()
        with pytest.raises(MT5ConnectionError):
            c.symbol_info()

    def test_none_response_raises(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        mock_mt5.symbol_info_obj = None
        with pytest.raises(MT5ConnectionError, match="symbol_info None"):
            c.symbol_info()


# ===========================================================================
# 9. terminal_info
# ===========================================================================

class TestTerminalInfo:
    def test_basic(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        out = c.terminal_info()
        assert isinstance(out, dict)
        assert out["connected"] is True

    def test_none_returns_empty(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        mock_mt5.terminal_info_obj = None
        assert c.terminal_info() == {}


# ===========================================================================
# 10. last_tick_msc
# ===========================================================================

class TestLastTickMsc:
    def test_basic(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        mock_mt5.tick_obj = TickInfoResult(time_msc=12345)
        assert c.last_tick_msc() == 12345

    def test_none_raises(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        mock_mt5.tick_obj = None
        with pytest.raises(MT5ConnectionError, match="symbol_info_tick None"):
            c.last_tick_msc()


# ===========================================================================
# 11. copy_rates_range
# ===========================================================================

class TestCopyRatesRange:
    def test_basic(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        mock_mt5.copy_rates_result = [1, 2, 3]
        out = c.copy_rates_range("XAUUSD", TIMEFRAME_H1,
                                  datetime(2025, 1, 1, tzinfo=timezone.utc),
                                  datetime(2025, 1, 2, tzinfo=timezone.utc))
        assert out == [1, 2, 3]

    def test_requires_connection(self, patch_settings, mock_mt5):
        c = MT5Connector()
        with pytest.raises(MT5ConnectionError, match="Not connected"):
            c.copy_rates_range("XAUUSD", TIMEFRAME_H1,
                                datetime.now(timezone.utc),
                                datetime.now(timezone.utc))

    def test_none_raises(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        mock_mt5.copy_rates_range = lambda *a, **kw: None  # type: ignore[method-assign]
        with pytest.raises(MT5ConnectionError, match="copy_rates_range"):
            c.copy_rates_range("EURUSD", TIMEFRAME_H1,
                                datetime.now(timezone.utc),
                                datetime.now(timezone.utc))


# ===========================================================================
# 12. copy_ticks_from
# ===========================================================================

class TestCopyTicksFrom:
    def test_count_zero_returns_empty(self, patch_settings, mock_mt5):
        import numpy as np
        c = MT5Connector()
        c.connect()
        out = c.copy_ticks_from(from_msc=0, count=0)
        assert isinstance(out, np.ndarray)
        assert out.size == 0

    def test_negative_count_returns_empty(self, patch_settings, mock_mt5):
        import numpy as np
        c = MT5Connector()
        c.connect()
        out = c.copy_ticks_from(from_msc=0, count=-1)
        assert isinstance(out, np.ndarray)
        assert out.size == 0

    def test_positive_count_proxies(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        mock_mt5.copy_ticks_result = [1, 2, 3]
        out = c.copy_ticks_from(from_msc=1_700_000_000_000, count=100)
        assert out == [1, 2, 3]

    def test_none_response_raises(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        mock_mt5.copy_ticks_from = lambda *a, **kw: None  # type: ignore[method-assign]
        with pytest.raises(MT5ConnectionError, match="copy_ticks_from"):
            c.copy_ticks_from(from_msc=0, count=10)

    def test_requires_connection(self, patch_settings, mock_mt5):
        c = MT5Connector()
        with pytest.raises(MT5ConnectionError, match="Not connected"):
            c.copy_ticks_from(from_msc=0, count=10)


# ===========================================================================
# 13. _require_connected
# ===========================================================================

@pytest.mark.parametrize("method,args", [
    ("account_info", ()),
    ("symbol_info", ()),
    ("terminal_info", ()),
    ("last_tick_msc", ()),
])
def test_methods_require_connection(method, args, patch_settings, mock_mt5):
    c = MT5Connector()
    with pytest.raises(MT5ConnectionError, match="Not connected"):
        getattr(c, method)(*args)


# ===========================================================================
# 14. Retry behaviour calls shutdown between attempts
# ===========================================================================

class TestRetryShutdown:
    def test_shutdown_called_between_failed_attempts(
        self, patch_settings, mock_mt5, fast_sleep,
    ):
        mock_mt5.initialize_returns = False
        c = MT5Connector()
        with pytest.raises(MT5ConnectionError):
            c.connect()
        assert mock_mt5.shutdown_calls >= patch_settings.mt5_retry_attempts


# ===========================================================================
# 15. Multiple symbols across connector instances
# ===========================================================================

@pytest.mark.parametrize("sym", [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "XAUUSD",
])
def test_connector_per_symbol(sym, patch_settings, mock_mt5):
    c = MT5Connector(symbol=sym)
    c.connect()
    assert c.symbol == sym
    info = c.symbol_info()
    assert info.name == sym


# ===========================================================================
# 16. Reconnect after disconnect
# ===========================================================================

class TestReconnect:
    def test_disconnect_then_reconnect(self, patch_settings, mock_mt5):
        c = MT5Connector()
        c.connect()
        c.disconnect()
        c.connect()
        assert c.is_connected is True
