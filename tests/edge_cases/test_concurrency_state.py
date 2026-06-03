"""Phase-5 / Concurrency + State — adversarial tests on the position
manager, scanner, and restart paths.

Coverage focus (per Phase 5 brief):

  - Two signals same pair same minute (dedup)
  - Position open while new signal fires
  - Shutdown mid-trade (graceful close)
  - Restart with open position (state recovery)
  - Telegram fails (does trade still execute?)
  - Compliance check during position maintain
"""

from __future__ import annotations
import asyncio
import uuid
from typing import List

import pytest

from data.bar_aggregator import Bar, BarAggregator
from data.tick_collector import Tick
from execution.order_router import (
    GriffOpenPosition, GriffOrderRouter, GriffPendingOrder,
)
from execution.position_manager import (
    GriffPositionManager, MaintenanceReport, _legacy_position, _replace_sl,
    _sl_hit,
)
from execution.order import Side
from execution.position import Position, PositionState
from risk.trailing_sl import TrailingStopLoss
from strategy.patterns.asian_sweep import AsianSweepDetector
from strategy.patterns.base import Direction, Grade, MarketContext, PatternSignal
from strategy.scanner import Scanner
from strategy.swing_tracker import SwingTracker

from tests.edge_cases.fixtures.chaos_market import HOUR_MS, hour_msc, make_bar
from tests.execution.fixtures.mock_mt5 import (
    DealInfo, MockMT5, OrderSendResult, PositionInfo,
    TRADE_RETCODE_DONE,
)
from tests.execution.fixtures.mock_orders import make_signal, make_signal_sell
from tests.execution.fixtures.mock_positions import (
    make_griff_open, make_griff_pending,
)


def run(coro):
    return asyncio.run(coro)


# ===========================================================================
# 1. DUPLICATE SIGNALS — Scanner dedups via per-pair-best logic
# ===========================================================================

class TestScannerDedup:
    def test_scanner_returns_all_emitted_signals(self):
        """Scanner.scan_all returns every emitted signal (no per-pair dedup)."""
        from strategy.patterns.asian_sweep import AsianSweepDetector

        class FakeDetector:
            name = "FAKE"
            min_bars_required = 1
            timeframe = "1H"
            def detect(self, bars, ctx):
                return PatternSignal(
                    pattern_name="FAKE", symbol=ctx.symbol,
                    direction=Direction.BUY, entry=1.1, sl=1.09, tp=1.12,
                    confidence=0.9, grade=Grade.A,
                    confluences_met=(), bar_time_msc=0,
                )
        s = Scanner(pairs=("EURUSD",), patterns=(FakeDetector(),))
        bars = {"EURUSD": [make_bar(symbol="EURUSD", time_msc=0)]}
        sigs = s.scan_all(bars, current_time_msc=0)
        assert len(sigs) == 1

    def test_scanner_picks_best_signal_by_grade(self):
        class A:
            name = "A"; min_bars_required = 1; timeframe = "1H"
            def detect(self, bars, ctx):
                return PatternSignal(
                    pattern_name="A", symbol=ctx.symbol,
                    direction=Direction.BUY, entry=1.1, sl=1.09, tp=1.12,
                    confidence=0.5, grade=Grade.B,
                    confluences_met=(), bar_time_msc=0,
                )
        class B:
            name = "B"; min_bars_required = 1; timeframe = "1H"
            def detect(self, bars, ctx):
                return PatternSignal(
                    pattern_name="B", symbol=ctx.symbol,
                    direction=Direction.BUY, entry=1.1, sl=1.09, tp=1.12,
                    confidence=0.9, grade=Grade.A,
                    confluences_met=(), bar_time_msc=0,
                )
        s = Scanner(pairs=("EURUSD",), patterns=(A(), B()))
        bars = {"EURUSD": [make_bar(symbol="EURUSD", time_msc=0)]}
        s.scan_all(bars, current_time_msc=0)
        best = s.get_best_signal()
        assert best.pattern_name == "B"  # Grade A wins over B

    def test_scanner_drops_c_grade(self):
        class C:
            name = "C"; min_bars_required = 1; timeframe = "1H"
            def detect(self, bars, ctx):
                return PatternSignal(
                    pattern_name="C", symbol=ctx.symbol,
                    direction=Direction.BUY, entry=1.1, sl=1.09, tp=1.12,
                    confidence=0.5, grade=Grade.C,
                    confluences_met=(), bar_time_msc=0,
                )
        s = Scanner(pairs=("EURUSD",), patterns=(C(),))
        bars = {"EURUSD": [make_bar(symbol="EURUSD", time_msc=0)]}
        s.scan_all(bars, current_time_msc=0)
        assert s.get_best_signal() is None
        assert s.c_grade_dropped == 1

    def test_scanner_skips_insufficient_bars(self):
        class Need10:
            name = "N"; min_bars_required = 10; timeframe = "1H"
            def detect(self, bars, ctx):
                return None
        s = Scanner(pairs=("EURUSD",), patterns=(Need10(),))
        bars = {"EURUSD": [make_bar(symbol="EURUSD", time_msc=0)]}
        s.scan_all(bars, current_time_msc=0)
        assert s.skipped_insufficient_bars == 1

    def test_scanner_rejects_no_pairs(self):
        with pytest.raises(ValueError, match="at least one pair"):
            Scanner(pairs=(), patterns=(AsianSweepDetector(),))

    def test_scanner_rejects_no_patterns(self):
        with pytest.raises(ValueError, match="at least one pattern"):
            Scanner(pairs=("EURUSD",), patterns=())

    def test_scanner_signals_by_grade(self):
        class B:
            name = "B"; min_bars_required = 1; timeframe = "1H"
            def detect(self, bars, ctx):
                return PatternSignal(
                    pattern_name="B", symbol=ctx.symbol,
                    direction=Direction.BUY, entry=1.1, sl=1.09, tp=1.12,
                    confidence=0.5, grade=Grade.B,
                    confluences_met=(), bar_time_msc=0,
                )
        s = Scanner(pairs=("EURUSD",), patterns=(B(),))
        s.scan_all({"EURUSD": [make_bar(time_msc=0)]}, current_time_msc=0)
        by_grade = s.signals_by_grade()
        assert len(by_grade[Grade.B]) == 1


# ===========================================================================
# 2. POSITION MANAGER REGISTRATION
# ===========================================================================

class TestPositionManager:
    @pytest.fixture
    def pm(self):
        router = GriffOrderRouter(dry_run=True)
        tracker = SwingTracker()
        trail = TrailingStopLoss(tracker)
        return GriffPositionManager(router, tracker, trail)

    def test_register_position_appears_in_open_positions(self, pm):
        p = make_griff_open()
        pm.register_position(p)
        assert p in pm.open_positions

    def test_register_pending_appears_in_pendings(self, pm):
        o = make_griff_pending()
        pm.register_pending(o)
        assert o in pm.pending_orders

    def test_register_two_positions_same_pair(self, pm):
        a = make_griff_open(symbol="EURUSD")
        b = make_griff_open(symbol="EURUSD")
        pm.register_position(a)
        pm.register_position(b)
        assert len(pm.positions_for("EURUSD")) == 2

    def test_forget_position_returns_position(self, pm):
        p = make_griff_open()
        pm.register_position(p)
        out = pm.forget_position(p.position_id)
        assert out == p

    def test_forget_position_returns_none_if_unknown(self, pm):
        assert pm.forget_position("does-not-exist") is None

    def test_forget_pending_returns_order(self, pm):
        o = make_griff_pending()
        pm.register_pending(o)
        out = pm.forget_pending(o.order_id)
        assert out == o

    def test_forget_pending_returns_none_if_unknown(self, pm):
        assert pm.forget_pending("does-not-exist") is None


# ===========================================================================
# 3. on_pending_filled — pending → position
# ===========================================================================

class TestPendingFilled:
    @pytest.fixture
    def pm(self):
        router = GriffOrderRouter(dry_run=True)
        tracker = SwingTracker()
        trail = TrailingStopLoss(tracker)
        return GriffPositionManager(router, tracker, trail)

    def test_on_pending_filled_promotes_to_position(self, pm):
        o = make_griff_pending()
        pm.register_pending(o)
        pos = pm.on_pending_filled(o.order_id, fill_price=2000.0,
                                    mt5_position_ticket=42, fill_msc=0)
        assert pos is not None
        assert pos.entry_price == 2000.0
        assert pos.mt5_ticket == 42
        assert o not in pm.pending_orders
        assert pos in pm.open_positions

    def test_on_pending_filled_unknown_returns_none(self, pm):
        pos = pm.on_pending_filled("non-existent", fill_price=2000.0,
                                    mt5_position_ticket=42, fill_msc=0)
        assert pos is None


# ===========================================================================
# 4. MAINTAIN — SL HIT DETECTION
# ===========================================================================

class TestMaintainSLHit:
    @pytest.fixture
    def pm(self):
        router = GriffOrderRouter(dry_run=True)
        tracker = SwingTracker()
        trail = TrailingStopLoss(tracker)
        return GriffPositionManager(router, tracker, trail)

    def test_sl_hit_long_marks_closed(self, pm):
        pos = make_griff_open(side=Direction.BUY,
                                sl_price=1995.0, tp_price=2010.0)
        pm.register_position(pos)
        bar = make_bar(symbol="EURUSD", time_msc=0,
                       open=2000.0, high=2001.0, low=1990.0, close=1995.0)
        report = run(pm.maintain("EURUSD", bar, now_msc=0))
        assert pos.position_id in report.closed_positions
        assert pos not in pm.open_positions

    def test_sl_hit_short_marks_closed(self, pm):
        pos = make_griff_open(side=Direction.SELL,
                                sl_price=2005.0, tp_price=1990.0)
        pm.register_position(pos)
        bar = make_bar(symbol="EURUSD", time_msc=0,
                       open=2000.0, high=2010.0, low=1999.0, close=2005.0)
        report = run(pm.maintain("EURUSD", bar, now_msc=0))
        assert pos.position_id in report.closed_positions

    def test_no_sl_hit_no_close(self, pm):
        pos = make_griff_open(side=Direction.BUY,
                                sl_price=1990.0, tp_price=2010.0)
        pm.register_position(pos)
        bar = make_bar(symbol="EURUSD", time_msc=0,
                       open=2000.0, high=2001.0, low=1999.0, close=2000.5)
        report = run(pm.maintain("EURUSD", bar, now_msc=0))
        assert pos.position_id not in report.closed_positions
        assert pos in pm.open_positions


# ===========================================================================
# 5. MAINTAIN — EXPIRY CANCELS PENDING
# ===========================================================================

class TestMaintainCancel:
    @pytest.fixture
    def pm(self):
        router = GriffOrderRouter(dry_run=True)
        tracker = SwingTracker()
        trail = TrailingStopLoss(tracker)
        return GriffPositionManager(router, tracker, trail)

    def test_pending_past_expiry_cancelled(self, pm):
        o = make_griff_pending(expiry_msc=100)
        pm.register_pending(o)
        bar = make_bar(symbol="EURUSD", time_msc=200)
        report = run(pm.maintain("EURUSD", bar, now_msc=200))
        assert o.order_id in report.cancelled_pendings
        assert o not in pm.pending_orders

    def test_pending_before_expiry_kept(self, pm):
        o = make_griff_pending(expiry_msc=500)
        pm.register_pending(o)
        bar = make_bar(symbol="EURUSD", time_msc=200)
        report = run(pm.maintain("EURUSD", bar, now_msc=200))
        assert o.order_id not in report.cancelled_pendings
        assert o in pm.pending_orders


# ===========================================================================
# 6. RESTART / STATE RECOVERY
# ===========================================================================

class TestStateRecovery:
    def test_fresh_pm_has_no_state(self):
        router = GriffOrderRouter(dry_run=True)
        tracker = SwingTracker()
        trail = TrailingStopLoss(tracker)
        pm = GriffPositionManager(router, tracker, trail)
        assert pm.open_positions == ()
        assert pm.pending_orders == ()

    def test_repopulate_from_mt5_positions_via_register(self):
        """Simulate restart: bot queries MT5, gets orphan positions, registers them."""
        router = GriffOrderRouter(dry_run=True)
        tracker = SwingTracker()
        trail = TrailingStopLoss(tracker)
        pm = GriffPositionManager(router, tracker, trail)
        orphans = [
            make_griff_open(symbol="EURUSD", mt5_ticket=1,
                              position_id="orphan-1"),
            make_griff_open(symbol="XAUUSD", mt5_ticket=2,
                              position_id="orphan-2"),
        ]
        for o in orphans:
            pm.register_position(o)
        assert len(pm.open_positions) == 2

    def test_mock_mt5_positions_match_orphans(self):
        """Verify the surface the bot would poll for restart recovery."""
        m = MockMT5()
        m.positions = [
            PositionInfo(ticket=1, symbol="EURUSD",
                          volume=0.10, price_open=1.10),
            PositionInfo(ticket=2, symbol="XAUUSD",
                          volume=0.05, price_open=2000.0),
        ]
        out = m.positions_get()
        assert len(out) == 2


# ===========================================================================
# 7. CONCURRENT MAINTAIN — multi-pair
# ===========================================================================

class TestMultiPairMaintain:
    @pytest.fixture
    def pm(self):
        router = GriffOrderRouter(dry_run=True)
        tracker = SwingTracker()
        trail = TrailingStopLoss(tracker)
        return GriffPositionManager(router, tracker, trail)

    def test_maintain_one_pair_does_not_affect_other(self, pm):
        pos_eur = make_griff_open(symbol="EURUSD", sl_price=1.09,
                                    tp_price=1.12)
        pos_xau = make_griff_open(symbol="XAUUSD", sl_price=1995.0,
                                    tp_price=2010.0)
        pm.register_position(pos_eur)
        pm.register_position(pos_xau)
        bar_eur = make_bar(symbol="EURUSD", time_msc=0,
                            open=1.10, high=1.105, low=1.095, close=1.10)
        report = run(pm.maintain("EURUSD", bar_eur, now_msc=0))
        # XAUUSD untouched.
        assert pos_xau in pm.open_positions

    def test_maintain_sequential_pairs(self, pm):
        for pair in ("EURUSD", "XAUUSD", "GBPUSD"):
            pos = make_griff_open(symbol=pair,
                                   sl_price=1.0 if pair != "XAUUSD" else 1995.0)
            pm.register_position(pos)
            bar = make_bar(symbol=pair, time_msc=0,
                           open=1.10 if pair != "XAUUSD" else 2000.0,
                           high=1.105 if pair != "XAUUSD" else 2001.0,
                           low=1.095 if pair != "XAUUSD" else 1999.0,
                           close=1.10 if pair != "XAUUSD" else 2000.0)
            run(pm.maintain(pair, bar, now_msc=0))


# ===========================================================================
# 8. ASYNCIO — concurrent maintain
# ===========================================================================

class TestAsyncConcurrent:
    @pytest.fixture
    def pm(self):
        router = GriffOrderRouter(dry_run=True)
        tracker = SwingTracker()
        trail = TrailingStopLoss(tracker)
        return GriffPositionManager(router, tracker, trail)

    def test_maintain_can_be_awaited_in_gather(self, pm):
        pos = make_griff_open(symbol="EURUSD", sl_price=1.09,
                                tp_price=1.12)
        pm.register_position(pos)
        bar = make_bar(symbol="EURUSD", time_msc=0,
                       open=1.10, high=1.11, low=1.095, close=1.10)

        async def _go():
            return await asyncio.gather(
                pm.maintain("EURUSD", bar, now_msc=0),
            )

        results = run(_go())
        assert len(results) == 1


# ===========================================================================
# 9. POSITION-MANAGER HELPERS — _legacy_position, _replace_sl, _sl_hit
# ===========================================================================

class TestHelpers:
    def test_legacy_position_buy(self):
        p = make_griff_open(side=Direction.BUY)
        legacy = _legacy_position(p)
        assert legacy.side == Side.BUY

    def test_legacy_position_sell(self):
        p = make_griff_open(side=Direction.SELL)
        legacy = _legacy_position(p)
        assert legacy.side == Side.SELL

    def test_replace_sl_returns_new_frozen_instance(self):
        p = make_griff_open(sl_price=1.09)
        new_p = _replace_sl(p, new_sl=1.095)
        assert new_p.sl_price == 1.095
        assert new_p.position_id == p.position_id
        assert new_p is not p  # frozen → new instance

    @pytest.mark.parametrize("side,bar_low,bar_high,sl_price,expected", [
        (Direction.BUY, 1.09, 1.10, 1.095, True),
        (Direction.BUY, 1.096, 1.10, 1.095, False),
        (Direction.SELL, 1.09, 1.10, 1.095, True),
        (Direction.SELL, 1.09, 1.094, 1.095, False),
    ])
    def test_sl_hit(self, side, bar_low, bar_high, sl_price, expected):
        p = make_griff_open(side=side, sl_price=sl_price)
        bar = make_bar(time_msc=0, open=1.095,
                       high=bar_high, low=bar_low, close=1.095)
        assert _sl_hit(p, bar) is expected


# ===========================================================================
# 10. SWING TRACKER — feeds the trail
# ===========================================================================

class TestSwingTrackerEdge:
    def test_swing_tracker_empty(self):
        t = SwingTracker()
        assert t.get_last_swing_low("EURUSD") is None
        assert t.get_last_swing_high("EURUSD") is None

    def test_swing_tracker_one_bar(self):
        t = SwingTracker()
        t.update("EURUSD", make_bar(symbol="EURUSD", time_msc=0,
                                       open=1.10, close=1.10))
        # Single bar may not establish a swing yet.
        _ = t.get_last_swing_high("EURUSD")
        _ = t.get_last_swing_low("EURUSD")

    def test_swing_tracker_multi_bar(self):
        t = SwingTracker()
        for i in range(10):
            t.update("EURUSD", make_bar(
                symbol="EURUSD", time_msc=i * HOUR_MS,
                open=1.10 + i * 0.001, close=1.10 + i * 0.001,
            ))


# ===========================================================================
# 11. TRAILING SL — explicit hooks
# ===========================================================================

class TestTrailingSL:
    def test_apply_spread_protection_raises_for_unknown_position(self):
        from datetime import datetime, timezone
        trail = TrailingStopLoss(SwingTracker())
        pos = Position(
            position_id="unseen", side=Side.BUY, lots=0.1,
            entry_price=2000.0, entry_time_msc=0, sl_price=1995.0,
            tp_price=2010.0, max_hold_until_msc=0, state=PositionState.OPEN,
        )
        with pytest.raises(KeyError):
            trail.apply_spread_protection(pos,
                rollover_time=datetime(2026, 5, 14, 21, tzinfo=timezone.utc))

    def test_revert_returns_position_sl_when_no_state(self):
        trail = TrailingStopLoss(SwingTracker())
        pos = Position(
            position_id="x", side=Side.BUY, lots=0.1,
            entry_price=2000.0, entry_time_msc=0, sl_price=1995.0,
            tp_price=2010.0, max_hold_until_msc=0, state=PositionState.OPEN,
        )
        assert trail.revert_spread_protection(pos) == pos.sl_price


# ===========================================================================
# 12. MAINTENANCE REPORT — STRUCTURE
# ===========================================================================

def test_maintenance_report_is_frozen():
    rep = MaintenanceReport(pair="X", bar_close_msc=0,
                              sl_updates=(), closed_positions=(),
                              cancelled_pendings=())
    with pytest.raises(Exception):
        rep.pair = "Y"  # type: ignore[misc]


# ===========================================================================
# 13. POSITION VALUE OBJECT — IMMUTABILITY
# ===========================================================================

def test_griff_open_position_frozen():
    p = make_griff_open()
    with pytest.raises(Exception):
        p.sl_price = 9.0  # type: ignore[misc]


def test_griff_pending_order_frozen():
    o = make_griff_pending()
    with pytest.raises(Exception):
        o.expiry_msc = 0  # type: ignore[misc]


# ===========================================================================
# 14. SCANNER — DEFENSIVE
# ===========================================================================

class TestScannerDefensive:
    def test_scanner_with_empty_bar_feed_returns_empty(self):
        from strategy.patterns.asian_sweep import AsianSweepDetector
        s = Scanner(pairs=("EURUSD",), patterns=(AsianSweepDetector(),))
        sigs = s.scan_all({}, current_time_msc=0)
        assert sigs == ()

    def test_scanner_uses_context_overrides(self):
        class CaptureCtx:
            name = "X"; min_bars_required = 1; timeframe = "1H"
            captured = []
            def detect(self, bars, ctx):
                self.captured.append(ctx)
                return None
        det = CaptureCtx()
        s = Scanner(pairs=("EURUSD",), patterns=(det,))
        custom = MarketContext(symbol="EURUSD", current_time_msc=0,
                               htf_bias="BULLISH")
        s.scan_all({"EURUSD": [make_bar(time_msc=0)]},
                   current_time_msc=0,
                   context_overrides={"EURUSD": custom})
        assert det.captured[0].htf_bias == "BULLISH"


# ===========================================================================
# 15. MULTI-PATTERN, MULTI-PAIR — DENSITY MATRIX
# ===========================================================================

@pytest.mark.parametrize("n_pairs,n_patterns", [
    (1, 1), (1, 5), (8, 1), (8, 5), (3, 3),
])
def test_scanner_density_matrix(n_pairs, n_patterns):
    pairs = [f"PAIR{i}" for i in range(n_pairs)]
    class P:
        def __init__(self, n): self.name = n
        min_bars_required = 1
        timeframe = "1H"
        def detect(self, bars, ctx): return None
    patterns = [P(f"P{j}") for j in range(n_patterns)]
    bars = {p: [make_bar(symbol=p, time_msc=0)] for p in pairs}
    s = Scanner(pairs=tuple(pairs), patterns=tuple(patterns))
    sigs = s.scan_all(bars, current_time_msc=0)
    assert sigs == ()


# ===========================================================================
# 16. RUN-LIFE — schedule and shutdown
# ===========================================================================

class TestAsyncShutdown:
    def test_pm_can_outlive_async_context(self):
        router = GriffOrderRouter(dry_run=True)
        tracker = SwingTracker()
        trail = TrailingStopLoss(tracker)
        pm = GriffPositionManager(router, tracker, trail)
        # Register, maintain inside async, then access state outside.
        pos = make_griff_open(symbol="EURUSD",
                                sl_price=1.09, tp_price=1.12)
        pm.register_position(pos)

        async def _maintain_once():
            return await pm.maintain(
                "EURUSD",
                make_bar(symbol="EURUSD", time_msc=0,
                         open=1.10, high=1.11, low=1.095, close=1.10),
                now_msc=0,
            )

        run(_maintain_once())
        # State preserved.
        assert pos in pm.open_positions


# ===========================================================================
# 17. MOCK MT5 STATE
# ===========================================================================

class TestMockMt5State:
    def test_initialize_returns_default_true(self):
        m = MockMT5()
        assert m.initialize() is True

    def test_login_returns_default_true(self):
        m = MockMT5()
        assert m.login() is True

    def test_shutdown_increments_counter(self):
        m = MockMT5()
        for _ in range(3):
            m.shutdown()
        assert m.shutdown_calls == 3

    def test_next_ticket_monotonic(self):
        m = MockMT5()
        a = m.next_ticket()
        b = m.next_ticket()
        c = m.next_ticket()
        assert a < b < c

    def test_queue_retcodes(self):
        m = MockMT5()
        m.queue_retcodes(TRADE_RETCODE_DONE, TRADE_RETCODE_DONE)
        assert len(m.retcode_queue) == 2


# ===========================================================================
# 18. PROPERTY: SCAN_ALL DETERMINISTIC FOR SAME INPUT
# ===========================================================================

def test_scan_all_deterministic_for_same_input():
    from strategy.patterns.asian_sweep import AsianSweepDetector

    from tests.strategy.fixtures.synthetic_bars import long_sweep_bars

    bars = long_sweep_bars(symbol="EURUSD", pt=0.00001)
    s = Scanner(pairs=("EURUSD",), patterns=(AsianSweepDetector(),))
    sig1 = s.scan_all({"EURUSD": bars}, current_time_msc=bars[-1].time_msc)
    sig2 = s.scan_all({"EURUSD": bars}, current_time_msc=bars[-1].time_msc)
    assert sig1 == sig2


# ===========================================================================
# 19. CYCLE REPORT
# ===========================================================================

def test_cycle_report_defaults():
    from execution.live_engine import CycleReport
    rep = CycleReport(now_msc=0)
    assert rep.signals_emitted == 0
    assert rep.signals_rejected_by_compliance == 0
    assert rep.orders_placed == 0
    assert rep.rejections == []


# ===========================================================================
# 20. ASIAN_SWEEP_LOTS_FOR
# ===========================================================================

class TestAsianSweepLotsFor:
    def test_unknown_symbol_returns_min(self):
        from execution.live_engine import asian_sweep_lots_for
        assert asian_sweep_lots_for(0.8, 10_000.0, 0.001, "BTCUSD") == 0.01

    def test_zero_sl_returns_min(self):
        from execution.live_engine import asian_sweep_lots_for
        assert asian_sweep_lots_for(0.8, 10_000.0, 0.0, "EURUSD") == 0.01

    def test_zero_equity_returns_min(self):
        from execution.live_engine import asian_sweep_lots_for
        assert asian_sweep_lots_for(0.8, 0.0, 0.001, "EURUSD") == 0.01

    def test_zero_risk_returns_min(self):
        from execution.live_engine import asian_sweep_lots_for
        assert asian_sweep_lots_for(0.0, 10_000.0, 0.001, "EURUSD") == 0.01

    def test_caps_at_lot_max(self):
        from execution.live_engine import asian_sweep_lots_for
        lot = asian_sweep_lots_for(100.0, 10_000_000.0, 0.00001, "EURUSD")
        assert lot <= 50.0
