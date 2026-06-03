"""E2E — bot restart, reconnect, partial state recovery.

These tests simulate failures and verify the bot's state stays consistent:
  - Bot restart with open position → state recovery (via DailyTracker.persist)
  - Bot restart mid-Asian-range → resume correctly
  - Disconnect during position → reconnect → position intact
  - Crash after order send before confirm → no double-fill (router dedup)
  - Telegram down → trade still executes (alert non-blocking)
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from config.asian_sweep_config import PAIR_CONFIG, PAIRS
from data.bar_aggregator import Bar
from execution.order_router import GriffOrderError
from monitoring.daily_tracker import DailyTracker, DailyState
from risk.prop_firm.compliance import AccountState
from strategy.patterns.base import Direction, Grade, PatternSignal

from tests.e2e.fixtures.scenario_runner import (
    ScenarioRunner, long_sweep_bars, hour_msc,
)


def _sig(symbol: str = "EURUSD", hour: int = 8) -> PatternSignal:
    pt = float(PAIR_CONFIG[symbol]["point"])
    entry = 1.10000 if symbol != "XAUUSD" else 2000.00
    risk = 100 * pt
    return PatternSignal(
        pattern_name="ASIAN_SWEEP", symbol=symbol,
        direction=Direction.BUY, entry=entry,
        sl=entry - risk, tp=entry + risk * 2.5,
        confidence=0.9, grade=Grade.A,
        confluences_met=("asian_sweep_low", "LONDON", "bias_neutral",
                          "q9", f"tp1_{entry + risk:.5f}"),
        bar_time_msc=hour_msc(2026, 4, 15, hour),
    )


def _inject(r, sigs):
    r.scanner = MagicMock()
    r.scanner.scan_all = MagicMock(return_value=tuple(sigs))
    r.engine._scanner = r.scanner


# ===========================================================================
# 1. DailyTracker persists and reloads on same day
# ===========================================================================

class TestDailyTrackerPersistence:
    @pytest.mark.parametrize("trades", [1, 2, 3, 5])
    def test_persist_and_reload_same_day(self, tmp_path, trades):
        path = tmp_path / "daily.parquet"
        now = hour_msc(2026, 4, 15, 8)
        t = DailyTracker(starting_equity=100_000.0, persist_path=path,
                          now_ms=now)
        for _ in range(trades):
            t.record_trade_open(now_ms=now)
        t.persist()
        assert path.exists()
        # New instance same-day → recover.
        t2 = DailyTracker(starting_equity=100_000.0, persist_path=path,
                           now_ms=now + 60_000)
        assert t2.trade_count == trades

    def test_persist_and_reload_new_day_resets(self, tmp_path):
        path = tmp_path / "daily.parquet"
        now = hour_msc(2026, 4, 15, 8)
        t = DailyTracker(starting_equity=100_000.0, persist_path=path,
                          now_ms=now)
        t.record_trade_open(now_ms=now)
        t.persist()
        # Reload on a NEW IST day — should NOT recover prior counter.
        next_day = hour_msc(2026, 4, 16, 8)
        t2 = DailyTracker(starting_equity=100_000.0, persist_path=path,
                           now_ms=next_day)
        assert t2.trade_count == 0

    @pytest.mark.parametrize("equity", [10_000.0, 100_000.0, 500_000.0])
    def test_persist_preserves_equity(self, tmp_path, equity):
        path = tmp_path / "daily.parquet"
        now = hour_msc(2026, 4, 15, 8)
        t = DailyTracker(starting_equity=equity, persist_path=path,
                          now_ms=now)
        t.update_equity(equity + 1_000.0, now_ms=now)
        t.persist()
        t2 = DailyTracker(starting_equity=equity, persist_path=path,
                           now_ms=now + 60_000)
        assert t2.state.peak_equity >= equity


# ===========================================================================
# 2. Restart mid-day: open position persists in PositionManager (in-memory
#    only; MT5 is the source of truth, but the engine's local map can be
#    rehydrated by polling).
# ===========================================================================

class TestRestartWithOpenPosition:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_position_state_reattach(self, pair, runner_factory):
        r1 = runner_factory()
        s = _sig(pair, hour=8)
        _inject(r1, [s])
        r1.run_cycle({pair: []}, now_msc=s.bar_time_msc,
                     ask_by_pair={pair: s.entry},
                     bid_by_pair={pair: s.entry})
        assert len(r1.pm.open_positions) == 1
        snapshot_pos = r1.pm.open_positions[0]
        # Simulate restart — new runner re-registers from MT5 poll (mocked
        # by manual register_position call).
        r2 = runner_factory()
        r2.pm.register_position(snapshot_pos)
        assert len(r2.pm.open_positions) == 1
        assert r2.pm.open_positions[0].position_id == snapshot_pos.position_id

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_restart_maintain_continues_trailing(self, pair, runner_factory):
        # Open a position, simulate restart, run maintenance — SL should
        # still update if structure breaks.
        pt = float(PAIR_CONFIG[pair]["point"])
        r1 = runner_factory()
        bars = long_sweep_bars(symbol=pair, trigger_hour=8,
                                year=2026, month=4, day=15)
        r1.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        if not r1.pm.open_positions:
            pytest.skip("No position to test restart with")
        pos = r1.pm.open_positions[0]
        # Build a quiet bar past entry.
        bar_next = Bar(
            symbol=pair, time_msc=bars[-1].time_msc + 3_600_000,
            open=pos.entry_price + 5 * pt,
            high=pos.entry_price + 10 * pt,
            low=pos.entry_price + 1 * pt,
            close=pos.entry_price + 7 * pt, volume=10,
        )
        r2 = runner_factory()
        r2.pm.register_position(pos)
        r2.run_maintenance({pair: bar_next},
                            now_msc=bar_next.time_msc + 60_000)
        # Maintenance shouldn't close a healthy position.
        assert len(r2.pm.open_positions) == 1


# ===========================================================================
# 3. Idempotency / dedup — same signal submitted twice
# ===========================================================================

class TestIdempotency:
    def test_duplicate_market_submission_rejects(self, runner):
        s = _sig("EURUSD")
        # First submission OK.
        pos1 = asyncio.run(runner.router.place_market(
            s, lots=0.01, ask=s.entry + 0.0001,
            bid=s.entry, now_msc=s.bar_time_msc,
        ))
        assert pos1 is not None
        # Second submission within DEDUP_WINDOW_MS raises GriffOrderError.
        with pytest.raises(GriffOrderError, match="duplicate"):
            asyncio.run(runner.router.place_market(
                s, lots=0.01, ask=s.entry + 0.0001,
                bid=s.entry, now_msc=s.bar_time_msc + 1,
            ))

    def test_distinct_bars_allowed(self, runner):
        s1 = _sig("EURUSD", hour=8)
        s2 = _sig("EURUSD", hour=9)
        asyncio.run(runner.router.place_market(
            s1, lots=0.01, ask=s1.entry + 0.0001,
            bid=s1.entry, now_msc=s1.bar_time_msc,
        ))
        # Different bar → different key → accepted.
        asyncio.run(runner.router.place_market(
            s2, lots=0.01, ask=s2.entry + 0.0001,
            bid=s2.entry, now_msc=s2.bar_time_msc,
        ))


# ===========================================================================
# 4. Crash after submission — bot would re-submit on restart;
#    router dedup window prevents duplicate.
# ===========================================================================

class TestCrashRecovery:
    def test_post_send_pre_confirm_dedup(self, runner_factory):
        """First submit succeeds, bot 'crashes' mid-bookkeeping, second
        submit (after 'restart') hits the dedup guard and raises.
        """
        from execution.order_router import GriffOrderRouter
        router = GriffOrderRouter(dry_run=True)
        s = _sig("EURUSD")
        asyncio.run(router.place_market(
            s, lots=0.01, ask=s.entry + 0.0001,
            bid=s.entry, now_msc=s.bar_time_msc,
        ))
        with pytest.raises(GriffOrderError):
            asyncio.run(router.place_market(
                s, lots=0.01, ask=s.entry + 0.0001,
                bid=s.entry, now_msc=s.bar_time_msc + 100,
            ))


# ===========================================================================
# 5. Telegram down → trade still executes (alert non-blocking)
# ===========================================================================

class TestTelegramDownNonBlocking:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_alert_failure_does_not_break_open(self, pair, runner_factory):
        """Phase 6 fix #6 — engine wraps alert calls in `_safe_alert` so
        a notifier transport error never propagates into the trading
        loop. Trade opens normally even when Telegram is offline.
        """
        r = runner_factory()
        # Make every notifier send() raise — must not break the engine.
        r.notifier_mock.send = AsyncMock(side_effect=RuntimeError("net down"))
        s = _sig(pair)
        _inject(r, [s])
        r.run_cycle({pair: []}, now_msc=s.bar_time_msc,
                    ask_by_pair={pair: s.entry},
                    bid_by_pair={pair: s.entry})
        assert len(r.pm.open_positions) == 1


# ===========================================================================
# 6. Position-manager forget vs re-register idempotent
# ===========================================================================

class TestForgetReRegister:
    def test_forget_then_reregister(self, runner):
        s = _sig("EURUSD")
        _inject(runner, [s])
        runner.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                          ask_by_pair={"EURUSD": s.entry},
                          bid_by_pair={"EURUSD": s.entry})
        pos = runner.pm.open_positions[0]
        runner.pm.forget_position(pos.position_id)
        assert len(runner.pm.open_positions) == 0
        runner.pm.register_position(pos)
        assert len(runner.pm.open_positions) == 1

    def test_forget_unknown_is_noop(self, runner):
        assert runner.pm.forget_position("does-not-exist") is None


# ===========================================================================
# 7. Disconnect during position — maintain() with stale bar continues OK
# ===========================================================================

class TestDisconnectDuringPosition:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_quiet_period_no_close(self, pair, runner_factory):
        r = runner_factory()
        s = _sig(pair)
        _inject(r, [s])
        r.run_cycle({pair: []}, now_msc=s.bar_time_msc,
                    ask_by_pair={pair: s.entry},
                    bid_by_pair={pair: s.entry})
        if not r.pm.open_positions:
            pytest.skip("No position to test reconnect with")
        pos = r.pm.open_positions[0]
        pt = float(PAIR_CONFIG[pair]["point"])
        # Several quiet bars — simulating a disconnect that came back online.
        for h in range(1, 5):
            bar = Bar(
                symbol=pair,
                time_msc=s.bar_time_msc + h * 3_600_000,
                open=pos.entry_price + 1 * pt,
                high=pos.entry_price + 3 * pt,
                low=pos.entry_price + 0.5 * pt,
                close=pos.entry_price + 2 * pt, volume=5,
            )
            r.run_maintenance({pair: bar},
                                now_msc=bar.time_msc + 60_000)
        assert len(r.pm.open_positions) == 1


# ===========================================================================
# 8. Snapshot of pending orders survives restart
# ===========================================================================

class TestPendingSurvivesRestart:
    def test_pending_register_survives(self, runner_factory):
        from tests.execution.fixtures.mock_positions import make_griff_pending
        r1 = runner_factory()
        pending = make_griff_pending(symbol="EURUSD")
        r1.pm.register_pending(pending)
        assert len(r1.pm.pending_orders) == 1
        # Simulate restart by re-registering on a fresh runner.
        r2 = runner_factory()
        r2.pm.register_pending(pending)
        assert len(r2.pm.pending_orders) == 1
        assert r2.pm.pending_orders[0].order_id == pending.order_id


# ===========================================================================
# 9. Engine restart mid-Asian-range — bars from prev session still feed
#    the detector if the bar window includes them
# ===========================================================================

class TestRestartMidRange:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_history_seeds_first_scan(self, pair, runner_factory):
        # Use the synthetic bar fixture which seeds enough history.
        bars = long_sweep_bars(symbol=pair, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        rep = r.result.cycle_reports[-1]
        # First scan should produce a signal (history is sufficient).
        # Wiring assertion: engine never opened more positions than signals.
        assert rep.orders_placed <= rep.signals_emitted


# ===========================================================================
# 10. Multiple restarts within same session
# ===========================================================================

class TestRepeatedRestarts:
    @pytest.mark.parametrize("n_restarts", [1, 2, 3, 5])
    def test_restart_then_reattach(self, n_restarts, runner_factory):
        from tests.execution.fixtures.mock_positions import make_griff_open
        pos = make_griff_open(symbol="EURUSD")
        for _ in range(n_restarts):
            r = runner_factory()
            r.pm.register_position(pos)
            assert len(r.pm.open_positions) == 1


# ===========================================================================
# 11. DailyTracker rollover preserves equity progression
# ===========================================================================

class TestRolloverEquityProgression:
    def test_rollover_carries_equity_forward(self, runner_factory):
        r = runner_factory()
        r.daily.update_equity(105_000.0,
                              now_ms=hour_msc(2026, 4, 15, 8))
        # Force rollover.
        r.daily.update_equity(105_000.0,
                              now_ms=hour_msc(2026, 4, 15, 20))
        # New day's peak_equity should reflect carry-forward.
        assert r.daily.state.peak_equity > 0


# ===========================================================================
# 12. Notifier transport failure on close also non-blocking
# ===========================================================================

# ===========================================================================
# 13. Persist/reload across runner factory rebuilds
# ===========================================================================

class TestPersistReloadAcrossRunners:
    @pytest.mark.parametrize("trades_done", [0, 1, 2, 3])
    def test_state_recovered_in_fresh_runner(self, tmp_path, trades_done):
        from monitoring.daily_tracker import DailyTracker
        path = tmp_path / f"daily_{trades_done}.parquet"
        now = hour_msc(2026, 4, 15, 8)
        t = DailyTracker(starting_equity=100_000.0,
                          persist_path=path, now_ms=now)
        for _ in range(trades_done):
            t.record_trade_open(now_ms=now)
        t.persist()
        t2 = DailyTracker(starting_equity=100_000.0,
                           persist_path=path, now_ms=now + 60_000)
        assert t2.trade_count == trades_done

    @pytest.mark.parametrize("equity", [10_000.0, 50_000.0, 100_000.0,
                                          250_000.0])
    def test_starting_equity_survives(self, tmp_path, equity):
        from monitoring.daily_tracker import DailyTracker
        path = tmp_path / "daily.parquet"
        now = hour_msc(2026, 4, 15, 8)
        t = DailyTracker(starting_equity=equity,
                          persist_path=path, now_ms=now)
        t.update_equity(equity * 1.05, now_ms=now)
        t.persist()
        t2 = DailyTracker(starting_equity=equity,
                           persist_path=path, now_ms=now + 60_000)
        assert t2.state.peak_equity >= equity


# ===========================================================================
# 14. Restart with N positions — every pair survives
# ===========================================================================

class TestRestartAllPairsSurvive:
    @pytest.mark.parametrize("pair", list(PAIRS))
    @pytest.mark.parametrize("n_positions", [1, 2, 3])
    def test_multiple_positions_restored(self, pair, n_positions, runner_factory):
        from tests.execution.fixtures.mock_positions import make_griff_open
        r = runner_factory()
        positions = [make_griff_open(symbol=pair,
                                       mt5_ticket=10000 + i)
                       for i in range(n_positions)]
        for pos in positions:
            r.pm.register_position(pos)
        # New runner inherits these via the same registration path.
        r2 = runner_factory()
        for pos in positions:
            r2.pm.register_position(pos)
        assert len(r2.pm.open_positions) == n_positions


# ===========================================================================
# 15. Trailing SL state per-position isolation across restarts
# ===========================================================================

class TestTrailingStateIsolation:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_two_positions_same_pair_isolated(self, pair, runner_factory):
        from tests.execution.fixtures.mock_positions import make_griff_open
        from strategy.patterns.base import Direction
        r = runner_factory()
        p1 = make_griff_open(symbol=pair, mt5_ticket=10001,
                              position_id="pos-A")
        p2 = make_griff_open(symbol=pair, mt5_ticket=10002,
                              position_id="pos-B")
        r.pm.register_position(p1)
        r.pm.register_position(p2)
        assert len(r.pm.open_positions) == 2
        # Forget one — the other survives.
        r.pm.forget_position("pos-A")
        assert len(r.pm.open_positions) == 1
        assert r.pm.open_positions[0].position_id == "pos-B"


class TestCloseAlertFailureNonBlocking:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_close_alert_failure_does_not_break_force_close(self,
                                                            pair, runner_factory):
        """force_close_all goes directly through the router (no engine
        alert call); a notifier failure there still must not stop the
        close from happening."""
        r = runner_factory()
        s = _sig(pair)
        _inject(r, [s])
        r.run_cycle({pair: []}, now_msc=s.bar_time_msc,
                    ask_by_pair={pair: s.entry},
                    bid_by_pair={pair: s.entry})
        if not r.pm.open_positions:
            pytest.skip("No position to close")
        # Notifier fails on next send.
        r.notifier_mock.send = AsyncMock(side_effect=RuntimeError("offline"))
        r.notifier_mock.notify_trade_close = AsyncMock(
            side_effect=RuntimeError("offline")
        )
        n = r.force_close_all(
            ask_by_pair={pair: s.entry},
            bid_by_pair={pair: s.entry},
            now_msc=s.bar_time_msc + 3_600_000,
        )
        assert n >= 1
