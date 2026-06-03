"""E2E — broker mode flows (DRY_RUN / PAPER / REAL).

The propX live engine routes orders through GriffOrderRouter. The router
has two paths:
  - dry_run=True  → synthetic position with ticket=-1, no MT5 call
  - dry_run=False → real MT5 call (we mock with MockMT5)

The broker_factory module also exposes a PAPER / REAL switch for the older
single-symbol microstructure path (now retired); for the propX engine,
dry_run is the equivalent.

Coverage:
  - DRY_RUN full cycle (open + close)
  - REAL mode mocked — MT5 receives correct request payload
  - REAL mode transient retcode → retry → success
  - REAL mode permanent reject → GriffOrderError
  - REAL mode partial fill — production bug logged in PROD_BUGS.md
"""

from __future__ import annotations
import asyncio
from unittest.mock import MagicMock

import pytest

from config.asian_sweep_config import PAIR_CONFIG, PAIRS
from execution.order_router import (
    GriffOrderError, GriffOrderRouter,
    DEDUP_WINDOW_MS, MAGIC, COMMENT,
)
from strategy.patterns.base import Direction, Grade, PatternSignal

from tests.e2e.fixtures.scenario_runner import (
    ScenarioRunner, long_sweep_bars, hour_msc,
)
from tests.execution.fixtures.mock_mt5 import MockMT5, OrderSendResult


def _sig(symbol: str = "EURUSD", hour: int = 8,
          bar_time_msc: int = None) -> PatternSignal:
    pt = float(PAIR_CONFIG[symbol]["point"])
    entry = 1.10000 if symbol != "XAUUSD" else 2000.00
    risk = 100 * pt
    if bar_time_msc is None:
        bar_time_msc = hour_msc(2026, 4, 15, hour)
    return PatternSignal(
        pattern_name="ASIAN_SWEEP", symbol=symbol,
        direction=Direction.BUY, entry=entry,
        sl=entry - risk, tp=entry + risk * 2.5,
        confidence=0.9, grade=Grade.A,
        confluences_met=("asian_sweep_low", "LONDON", "bias_neutral",
                          "q9", f"tp1_{entry + risk:.5f}"),
        bar_time_msc=bar_time_msc,
    )


# ===========================================================================
# 1. DRY_RUN — full cycle, all pairs
# ===========================================================================

class TestDryRunFullCycle:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_dry_run_open_returns_synthetic_position(self, pair):
        router = GriffOrderRouter(dry_run=True)
        s = _sig(pair)
        pos = asyncio.run(router.place_market(
            s, lots=0.05, ask=s.entry + 0.0001,
            bid=s.entry, now_msc=s.bar_time_msc,
        ))
        assert pos.mt5_ticket == -1
        assert pos.symbol == pair
        assert pos.side == Direction.BUY
        assert pos.lots == 0.05

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_dry_run_pending_returns_synthetic(self, pair):
        router = GriffOrderRouter(dry_run=True)
        s = _sig(pair)
        pending = asyncio.run(router.place_pending_stop(
            s, lots=0.05, expiry_msc=s.bar_time_msc + 3_600_000,
            now_msc=s.bar_time_msc,
        ))
        assert pending.mt5_ticket == -1
        assert pending.is_limit is False

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_dry_run_pending_limit(self, pair):
        router = GriffOrderRouter(dry_run=True)
        s = _sig(pair)
        pending = asyncio.run(router.place_pending_limit(
            s, lots=0.05, expiry_msc=s.bar_time_msc + 3_600_000,
            now_msc=s.bar_time_msc,
        ))
        assert pending.is_limit is True

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_dry_run_close_returns_price(self, pair):
        from tests.execution.fixtures.mock_positions import make_griff_open
        router = GriffOrderRouter(dry_run=True)
        pos = make_griff_open(symbol=pair)
        price = asyncio.run(router.close_position(
            pos, bid=pos.entry_price - 0.0001,
            ask=pos.entry_price + 0.0001,
            now_msc=pos.opened_msc + 3_600_000,
        ))
        assert price > 0

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_dry_run_modify_sl_returns_true(self, pair):
        from tests.execution.fixtures.mock_positions import make_griff_open
        router = GriffOrderRouter(dry_run=True)
        pos = make_griff_open(symbol=pair)
        ok = asyncio.run(router.modify_sl(pos, new_sl=pos.sl_price + 0.0001))
        assert ok is True


# ===========================================================================
# 2. REAL mode (MockMT5) — request payload correct
# ===========================================================================

class TestRealModeMockedRequestShape:
    @pytest.fixture
    def patched_router(self, monkeypatch):
        from execution import order_router as router_mod
        mock = MockMT5()
        monkeypatch.setattr(router_mod, "mt5", mock)
        return mock

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_market_request_fields(self, pair, patched_router):
        router = GriffOrderRouter(dry_run=False)
        s = _sig(pair)
        pos = asyncio.run(router.place_market(
            s, lots=0.1, ask=s.entry + 0.0001,
            bid=s.entry, now_msc=s.bar_time_msc,
        ))
        assert len(patched_router.sent_requests) == 1
        req = patched_router.sent_requests[0]
        assert req["symbol"] == pair
        assert req["volume"] == 0.1
        assert req["magic"] == MAGIC
        assert COMMENT in req["comment"]
        assert pos.lots > 0

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_pending_stop_request_fields(self, pair, patched_router):
        router = GriffOrderRouter(dry_run=False)
        s = _sig(pair)
        expiry = s.bar_time_msc + 3_600_000
        pending = asyncio.run(router.place_pending_stop(
            s, lots=0.05, expiry_msc=expiry, now_msc=s.bar_time_msc,
        ))
        req = patched_router.sent_requests[0]
        assert req["action"] == patched_router.TRADE_ACTION_PENDING
        # MT5 expects expiration in SECONDS, not ms.
        assert req["expiration"] == expiry // 1000


# ===========================================================================
# 3. REAL mode transient retcode → retry → success
# ===========================================================================

class TestRealModeRetry:
    @pytest.fixture
    def patched_router(self, monkeypatch):
        from execution import order_router as router_mod
        mock = MockMT5()
        monkeypatch.setattr(router_mod, "mt5", mock)
        return mock

    @pytest.mark.parametrize("transient_rc", [10004, 10006, 10021,
                                                10018, 10031])
    def test_transient_then_done(self, transient_rc, patched_router,
                                   monkeypatch):
        # Avoid actual sleep delays — patch RETRY_BACKOFF_SEC to all-zero.
        from execution import order_router as router_mod
        monkeypatch.setattr(router_mod, "RETRY_BACKOFF_SEC",
                             (0.0, 0.0, 0.0))
        patched_router.queue_retcodes(transient_rc)  # first attempt fails
        # Next attempt: default (DONE).
        router = GriffOrderRouter(dry_run=False)
        s = _sig("EURUSD")
        pos = asyncio.run(router.place_market(
            s, lots=0.1, ask=s.entry + 0.0001,
            bid=s.entry, now_msc=s.bar_time_msc,
        ))
        assert pos is not None
        assert len(patched_router.sent_requests) == 2

    def test_exhausted_retries_raises(self, patched_router, monkeypatch):
        from execution import order_router as router_mod
        monkeypatch.setattr(router_mod, "RETRY_BACKOFF_SEC",
                             (0.0, 0.0, 0.0))
        # All retries return transient → exhaust.
        patched_router.queue_retcodes(10004, 10004, 10004)
        router = GriffOrderRouter(dry_run=False)
        s = _sig("EURUSD")
        with pytest.raises(GriffOrderError, match="exhausted retries"):
            asyncio.run(router.place_market(
                s, lots=0.1, ask=s.entry + 0.0001,
                bid=s.entry, now_msc=s.bar_time_msc,
            ))


# ===========================================================================
# 4. REAL mode permanent reject
# ===========================================================================

class TestRealModePermanentReject:
    @pytest.fixture
    def patched_router(self, monkeypatch):
        from execution import order_router as router_mod
        mock = MockMT5()
        monkeypatch.setattr(router_mod, "mt5", mock)
        return mock

    @pytest.mark.parametrize("perm_rc", [10016, 10019])
    def test_permanent_reject_raises(self, perm_rc, patched_router):
        patched_router.queue_result(retcode=perm_rc, comment="bad stops")
        router = GriffOrderRouter(dry_run=False)
        s = _sig("EURUSD")
        with pytest.raises(GriffOrderError, match="permanent reject"):
            asyncio.run(router.place_market(
                s, lots=0.1, ask=s.entry + 0.0001,
                bid=s.entry, now_msc=s.bar_time_msc,
            ))


# ===========================================================================
# 5. REAL mode partial fill (PROD BUG #1)
# ===========================================================================

class TestRealModePartialFill:
    @pytest.fixture
    def patched_router(self, monkeypatch):
        from execution import order_router as router_mod
        mock = MockMT5()
        monkeypatch.setattr(router_mod, "mt5", mock)
        return mock

    def test_partial_fill_records_actual_volume(self, patched_router):
        """The router has been patched (see commit a8f4647) to read
        result.volume; this test verifies that fix end-to-end."""
        patched_router.queue_result(
            retcode=patched_router.TRADE_RETCODE_DONE,
            order=12345, volume=0.5,  # requested 1.0, filled 0.5
            price=1.10005,
        )
        router = GriffOrderRouter(dry_run=False)
        s = _sig("EURUSD")
        pos = asyncio.run(router.place_market(
            s, lots=1.0, ask=s.entry + 0.0001,
            bid=s.entry, now_msc=s.bar_time_msc,
        ))
        # If the partial-fill bug was fixed → pos.lots == 0.5.
        assert pos.lots == 0.5


# ===========================================================================
# 6. REAL mode close + modify_sl + cancel pending
# ===========================================================================

class TestRealModeOperations:
    @pytest.fixture
    def patched_router(self, monkeypatch):
        from execution import order_router as router_mod
        mock = MockMT5()
        monkeypatch.setattr(router_mod, "mt5", mock)
        return mock

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_close_position_sends_request(self, pair, patched_router):
        from tests.execution.fixtures.mock_positions import make_griff_open
        router = GriffOrderRouter(dry_run=False)
        pos = make_griff_open(symbol=pair)
        price = asyncio.run(router.close_position(
            pos, bid=pos.entry_price - 0.0001,
            ask=pos.entry_price + 0.0001,
            now_msc=pos.opened_msc + 3_600_000,
        ))
        assert len(patched_router.sent_requests) == 1
        req = patched_router.sent_requests[0]
        assert req["action"] == patched_router.TRADE_ACTION_DEAL
        assert req["position"] == pos.mt5_ticket
        assert price > 0

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_modify_sl_sends_sltp(self, pair, patched_router):
        from tests.execution.fixtures.mock_positions import make_griff_open
        router = GriffOrderRouter(dry_run=False)
        pos = make_griff_open(symbol=pair)
        ok = asyncio.run(router.modify_sl(pos, new_sl=pos.sl_price + 0.0001))
        assert ok is True
        req = patched_router.sent_requests[0]
        assert req["action"] == patched_router.TRADE_ACTION_SLTP

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_cancel_pending_sends_remove(self, pair, patched_router):
        from tests.execution.fixtures.mock_positions import make_griff_pending
        router = GriffOrderRouter(dry_run=False)
        order = make_griff_pending(symbol=pair)
        ok = asyncio.run(router.cancel_pending(order))
        assert ok is True
        req = patched_router.sent_requests[0]
        assert req["action"] == patched_router.TRADE_ACTION_REMOVE

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_cancel_pending_already_gone(self, pair, patched_router):
        """Retcode 10027 (ORDER_NOT_FOUND) treated as cancelled."""
        from tests.execution.fixtures.mock_positions import make_griff_pending
        patched_router.queue_result(retcode=10027)
        router = GriffOrderRouter(dry_run=False)
        order = make_griff_pending(symbol=pair)
        ok = asyncio.run(router.cancel_pending(order))
        assert ok is True


# ===========================================================================
# 7. Dedup window — same signal twice within DEDUP_WINDOW_MS rejects
# ===========================================================================

class TestDedupWindow:
    def test_inside_window_blocked(self):
        router = GriffOrderRouter(dry_run=True)
        s = _sig("EURUSD")
        asyncio.run(router.place_market(
            s, lots=0.1, ask=s.entry + 0.0001,
            bid=s.entry, now_msc=s.bar_time_msc,
        ))
        with pytest.raises(GriffOrderError, match="duplicate"):
            asyncio.run(router.place_market(
                s, lots=0.1, ask=s.entry + 0.0001,
                bid=s.entry, now_msc=s.bar_time_msc + DEDUP_WINDOW_MS - 1,
            ))

    def test_outside_window_allowed(self):
        router = GriffOrderRouter(dry_run=True)
        s = _sig("EURUSD")
        asyncio.run(router.place_market(
            s, lots=0.1, ask=s.entry + 0.0001,
            bid=s.entry, now_msc=s.bar_time_msc,
        ))
        # Second submission outside dedup window with a different bar
        # → fresh key, accepted. Use a fresh signal bar (timestamp + window).
        s2 = _sig("EURUSD", bar_time_msc=s.bar_time_msc + DEDUP_WINDOW_MS + 1)
        asyncio.run(router.place_market(
            s2, lots=0.1, ask=s2.entry + 0.0001,
            bid=s2.entry, now_msc=s2.bar_time_msc,
        ))
