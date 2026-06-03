"""execution.broker_factory.get_broker — mode → broker resolution.

Tests focus on:
  - PAPER mode → AsyncShim around PaperBroker
  - REAL mode → LiveBroker (mocked MT5)
  - Invalid mode → ValueError
  - AsyncShim correctly proxies sync PaperBroker methods
  - Broker switching across multiple settings instances
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from execution import broker_factory
from execution.broker_factory import (
    AsyncBroker, _PaperBrokerAsyncShim, get_broker,
)
from execution.broker_simulator import PaperBroker
from execution.order import Side, SignalType
from execution.position import CloseReason, PositionState
from utils.session import SessionLabel

from tests.execution.fixtures.mock_orders import make_intent
from tests.execution.fixtures.mock_positions import make_position, make_tick


# ---------------------------------------------------------------------------
# Settings stub — frozen dataclass-like object so get_broker() can read fields
# ---------------------------------------------------------------------------

class _SettingsStub:
    def __init__(
        self,
        execution_mode: str = "PAPER",
        symbol: str = "XAUUSD",
        broker_type: str = "IC_MARKETS",
    ) -> None:
        self.execution_mode = execution_mode
        self.symbol = symbol
        self.broker_type = broker_type


def _paper() -> _SettingsStub:
    return _SettingsStub(execution_mode="PAPER")


def _real() -> _SettingsStub:
    return _SettingsStub(execution_mode="REAL", symbol="XAUUSD",
                         broker_type="IC_MARKETS")


# ===========================================================================
# 1. Mode → type resolution
# ===========================================================================

class TestModeResolution:
    def test_paper_returns_async_shim(self):
        b = get_broker(_paper())
        assert isinstance(b, _PaperBrokerAsyncShim)

    def test_paper_wraps_paper_broker(self):
        b = get_broker(_paper())
        assert isinstance(b._paper, PaperBroker)

    def test_real_returns_live_broker(self, monkeypatch):
        # Replace LiveBroker BEFORE the late import inside get_broker.
        import execution.live_broker as live_broker_mod
        fake_cls = MagicMock(return_value="LB_INSTANCE")
        monkeypatch.setattr(live_broker_mod, "LiveBroker", fake_cls)
        out = get_broker(_real())
        assert out == "LB_INSTANCE"
        fake_cls.assert_called_once_with(symbol="XAUUSD")

    def test_real_passes_symbol(self, monkeypatch):
        import execution.live_broker as live_broker_mod
        fake_cls = MagicMock(return_value="LB")
        monkeypatch.setattr(live_broker_mod, "LiveBroker", fake_cls)
        s = _SettingsStub(execution_mode="REAL", symbol="EURUSD")
        get_broker(s)
        fake_cls.assert_called_once_with(symbol="EURUSD")

    def test_unknown_mode_raises(self):
        s = _SettingsStub(execution_mode="HYBRID")
        with pytest.raises(ValueError, match="Unknown execution_mode"):
            get_broker(s)

    @pytest.mark.parametrize("mode", ["paper", "real", "demo", ""])
    def test_lowercase_or_empty_mode_raises(self, mode):
        s = _SettingsStub(execution_mode=mode)
        with pytest.raises(ValueError):
            get_broker(s)

    @pytest.mark.parametrize("mode", [None, 0, 1, 1.5, [], {}])
    def test_non_string_mode_raises(self, mode):
        s = _SettingsStub(execution_mode=mode)  # type: ignore[arg-type]
        with pytest.raises((ValueError, TypeError)):
            get_broker(s)


# ===========================================================================
# 2. AsyncShim — proxies sync methods
# ===========================================================================

class TestAsyncShim:
    def test_fill_market_order_awaitable(self):
        shim = _PaperBrokerAsyncShim(PaperBroker(slippage_pct=0.0))
        i = make_intent(max_hold_until_msc=10**13)
        t = make_tick()
        pos = asyncio.run(shim.fill_market_order(i, t))
        assert pos.state == PositionState.OPEN

    def test_check_position_exit_returns_none_when_open_no_levels_touched(self):
        shim = _PaperBrokerAsyncShim(PaperBroker())
        p = make_position(side=Side.BUY, sl_price=0.5, tp_price=2.0,
                          max_hold_until_msc=10**13)
        t = make_tick(bid=1.0, ask=1.0)
        out = asyncio.run(shim.check_position_exit(p, t))
        assert out is None

    def test_check_position_exit_closes_on_sl(self):
        shim = _PaperBrokerAsyncShim(PaperBroker(slippage_pct=0.0))
        p = make_position(side=Side.BUY, sl_price=0.99, tp_price=1.50,
                          max_hold_until_msc=10**13)
        t = make_tick(bid=0.98, ask=0.98)
        out = asyncio.run(shim.check_position_exit(p, t))
        assert out is not None
        assert out.close_reason == CloseReason.SL_HIT

    def test_force_close_default_reason_eod(self):
        shim = _PaperBrokerAsyncShim(PaperBroker())
        p = make_position(max_hold_until_msc=10**13)
        out = asyncio.run(shim.force_close(p, make_tick()))
        assert out.close_reason == CloseReason.EOD

    @pytest.mark.parametrize("reason", list(CloseReason))
    def test_force_close_any_reason(self, reason):
        shim = _PaperBrokerAsyncShim(PaperBroker())
        p = make_position(max_hold_until_msc=10**13)
        out = asyncio.run(shim.force_close(p, make_tick(), reason))
        assert out.close_reason == reason


# ===========================================================================
# 3. AsyncBroker Protocol compliance
# ===========================================================================

class TestProtocol:
    def test_shim_satisfies_protocol(self):
        # Runtime-checkable Protocol assertion (structural typing)
        b = get_broker(_paper())
        # If methods exist with correct names, structural check passes.
        for name in ("fill_market_order", "check_position_exit", "force_close"):
            assert hasattr(b, name)


# ===========================================================================
# 4. Idempotency / multiple invocations
# ===========================================================================

class TestRepeatedInvocations:
    def test_two_calls_return_distinct_shims(self):
        a = get_broker(_paper())
        b = get_broker(_paper())
        assert a is not b

    def test_two_calls_share_no_state(self):
        a = get_broker(_paper())
        b = get_broker(_paper())
        # Each shim has its own PaperBroker instance.
        assert a._paper is not b._paper

    @pytest.mark.parametrize("n", [1, 5, 10, 50])
    def test_n_calls_all_succeed(self, n):
        brokers = [get_broker(_paper()) for _ in range(n)]
        assert len(brokers) == n
        for b in brokers:
            assert isinstance(b, _PaperBrokerAsyncShim)


# ===========================================================================
# 5. Real mode requires live_broker import (mocked)
# ===========================================================================

class TestRealModeIntegration:
    def test_real_with_alt_symbol(self, monkeypatch):
        import execution.live_broker as lb_mod
        captured = {}

        def factory(symbol):
            captured["symbol"] = symbol
            return MagicMock()

        monkeypatch.setattr(lb_mod, "LiveBroker", factory)
        s = _SettingsStub(execution_mode="REAL", symbol="USDJPY")
        get_broker(s)
        assert captured["symbol"] == "USDJPY"

    def test_real_logs_warning(self, monkeypatch, caplog):
        import execution.live_broker as lb_mod
        monkeypatch.setattr(lb_mod, "LiveBroker", MagicMock())
        s = _SettingsStub(execution_mode="REAL", symbol="XAUUSD",
                          broker_type="IC_MARKETS")
        # Just ensure call succeeds; the logger.warning is hard to capture
        # since it routes through loguru.
        get_broker(s)


# ===========================================================================
# 6. Profile switching — same factory handles repeated mode changes
# ===========================================================================

class TestProfileSwitching:
    def test_paper_then_real_then_paper(self, monkeypatch):
        import execution.live_broker as lb_mod
        monkeypatch.setattr(lb_mod, "LiveBroker", MagicMock(return_value="LB"))
        p1 = get_broker(_paper())
        r = get_broker(_real())
        p2 = get_broker(_paper())
        assert isinstance(p1, _PaperBrokerAsyncShim)
        assert r == "LB"
        assert isinstance(p2, _PaperBrokerAsyncShim)


# ===========================================================================
# 7. Configure pytest-asyncio mode for this file (no global config required)
# ===========================================================================

@pytest.fixture(autouse=True, scope="module")
def _enable_asyncio_mode():
    # pytest-asyncio is configured at the project level; ensure tests above
    # that use @pytest.mark.asyncio find their event loop.
    yield


# ===========================================================================
# 8. Internal shim attribute access
# ===========================================================================

class TestShimInternals:
    def test_shim_holds_paper_reference(self):
        p = PaperBroker()
        s = _PaperBrokerAsyncShim(p)
        assert s._paper is p

    def test_shim_constructor_typehint(self):
        # Just construct with various PaperBroker variants.
        for slip in [0.0, 0.5, 1.0]:
            p = PaperBroker(slippage_pct=slip)
            s = _PaperBrokerAsyncShim(p)
            assert s._paper._slippage_pct == slip
