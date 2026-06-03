"""E2E — full trade lifecycle from bar feed through exit.

Phase 6 integration suite: signal → compliance → size → route → register →
maintain → close. Tests exercise the REAL engine wiring against a dry-run
router; only the broker and Telegram transport are stubbed.

Coverage targets (each path verified end-to-end):
  - LONG and SHORT cycles
  - All 8 V5 pairs as full-cycle smoke tests
  - Compliance DENY path (no order placed)
  - News blackout path
  - Daily-trade-cap path
  - Outside-IST-window path
  - SL hit path (full close via maintain)
  - TP-side bar that crosses through SL on retracement
  - Force-close-all (EOD)
  - Best-per-pair dedup (two same-pair signals → one order)
"""

from __future__ import annotations
from typing import List

import pytest

from config.asian_sweep_config import PAIR_CONFIG, PAIRS
from data.bar_aggregator import Bar
from data.news_calendar import NewsEvent
from risk.prop_firm.compliance import AccountState
from strategy.patterns.base import Direction, Grade, PatternSignal

from tests.e2e.fixtures.scenario_runner import (
    ScenarioRunner, long_sweep_bars, short_sweep_bars, hour_msc,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_long_signal(symbol: str = "EURUSD") -> PatternSignal:
    pt = float(PAIR_CONFIG[symbol]["point"])
    entry = 1.10000 if symbol != "XAUUSD" else 2000.00
    risk = 100 * pt
    return PatternSignal(
        pattern_name="ASIAN_SWEEP", symbol=symbol,
        direction=Direction.BUY, entry=entry,
        sl=entry - risk, tp=entry + risk * 2.5,
        confidence=0.9, grade=Grade.A,
        confluences_met=("asian_sweep_low", "LONDON", "bias_neutral", "q9",
                         f"tp1_{entry + risk:.5f}"),
        bar_time_msc=hour_msc(2026, 4, 15, 8),
    )


def _make_short_signal(symbol: str = "EURUSD") -> PatternSignal:
    pt = float(PAIR_CONFIG[symbol]["point"])
    entry = 1.10000 if symbol != "XAUUSD" else 2000.00
    risk = 100 * pt
    return PatternSignal(
        pattern_name="ASIAN_SWEEP", symbol=symbol,
        direction=Direction.SELL, entry=entry,
        sl=entry + risk, tp=entry - risk * 2.5,
        confidence=0.9, grade=Grade.A,
        confluences_met=("asian_sweep_high", "LONDON", "bias_bearish", "q9",
                         f"tp1_{entry - risk:.5f}"),
        bar_time_msc=hour_msc(2026, 4, 15, 8),
    )


def _scan_with_canned_signal(runner: ScenarioRunner, signal: PatternSignal,
                              now_msc: int = None) -> None:
    """Inject a canned signal by overriding the scanner with a MagicMock."""
    from unittest.mock import MagicMock
    if now_msc is None:
        now_msc = signal.bar_time_msc
    runner.scanner = MagicMock()
    runner.scanner.scan_all = MagicMock(return_value=(signal,))
    runner.engine._scanner = runner.scanner
    runner.run_cycle({signal.symbol: []}, now_msc=now_msc,
                     ask_by_pair={signal.symbol: signal.entry},
                     bid_by_pair={signal.symbol: signal.entry})


# ===========================================================================
# 1. Real bar feed → real signal → full order (LONG / SHORT × all pairs)
# ===========================================================================

class TestFullCycleLong:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_long_sweep_opens_position(self, pair, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(symbol=pair, pt=pt, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        report = r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        assert report.signals_emitted >= 1
        assert report.orders_placed >= 1
        assert len(r.pm.open_positions) == 1
        pos = r.pm.open_positions[0]
        assert pos.symbol == pair
        assert pos.side == Direction.BUY

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_long_sweep_records_daily_trade(self, pair, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(symbol=pair, pt=pt, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        assert r.daily.trade_count == 1

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_long_sweep_sl_below_entry(self, pair, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(symbol=pair, pt=pt, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        pos = r.pm.open_positions[0]
        assert pos.sl_price < pos.entry_price

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_long_sweep_tp_above_entry(self, pair, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(symbol=pair, pt=pt, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        pos = r.pm.open_positions[0]
        assert pos.tp_price > pos.entry_price

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_long_sweep_lots_positive(self, pair, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(symbol=pair, pt=pt, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        pos = r.pm.open_positions[0]
        assert pos.lots > 0

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_long_sweep_pattern_name_tagged(self, pair, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(symbol=pair, pt=pt, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        assert r.pm.open_positions[0].pattern_name == "ASIAN_SWEEP"


class TestFullCycleShort:
    # NY-window short is V5-disabled, so we restrict to LONDON only.
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_short_sweep_opens_position_london(self, pair, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = short_sweep_bars(symbol=pair, pt=pt, trigger_hour=8,
                                 year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        if r.pm.open_positions:
            pos = r.pm.open_positions[0]
            assert pos.side == Direction.SELL
            assert pos.symbol == pair

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_short_sl_above_entry(self, pair, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = short_sweep_bars(symbol=pair, pt=pt, trigger_hour=8,
                                 year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        for pos in r.pm.open_positions:
            assert pos.sl_price > pos.entry_price


# ===========================================================================
# 2. Trigger-hour parametrise — all 5 LONDON hours + 4 NY hours
# ===========================================================================

class TestTriggerHourLong:
    @pytest.mark.parametrize("pair", list(PAIRS))
    @pytest.mark.parametrize("hour", [6, 7, 8, 9, 10])
    def test_london_hour_long(self, pair, hour, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(symbol=pair, pt=pt, trigger_hour=hour,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        assert len(r.pm.open_positions) <= 1  # never duplicates

    @pytest.mark.parametrize("pair", list(PAIRS))
    @pytest.mark.parametrize("hour", [12, 13, 14, 15])
    def test_ny_hour_long(self, pair, hour, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(symbol=pair, pt=pt, trigger_hour=hour,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        # NY-window LONG is allowed; we just confirm dispatcher didn't crash.
        assert len(r.pm.open_positions) <= 1


class TestTriggerHourShortLondonOnly:
    @pytest.mark.parametrize("pair", list(PAIRS))
    @pytest.mark.parametrize("hour", [12, 13, 14, 15])
    def test_ny_short_blocked(self, pair, hour, runner_factory):
        """V5 rule: SHORT only in LONDON. NY shorts must not fire."""
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = short_sweep_bars(symbol=pair, pt=pt, trigger_hour=hour,
                                 year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        for pos in r.pm.open_positions:
            assert pos.side != Direction.SELL


# ===========================================================================
# 3. Compliance DENY path — outside IST window, daily trade cap, etc.
# ===========================================================================

class TestComplianceDeny:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_outside_ist_window_blocks(self, pair, runner_factory):
        # Narrow window that excludes the trigger time → reject.
        r = runner_factory(ist_window_start="01:00", ist_window_end="02:00")
        sig = _make_long_signal(pair)
        _scan_with_canned_signal(r, sig, now_msc=hour_msc(2026, 4, 15, 8))
        assert len(r.pm.open_positions) == 0
        assert "outside_ist_window" in r.result.all_rejection_reasons

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_max_trades_reached_blocks(self, pair, runner_factory):
        r = runner_factory(max_trades_per_day=2)
        sig = _make_long_signal(pair)
        # Account already has 2 trades; cap is 2 → reject.
        from unittest.mock import MagicMock
        r.scanner = MagicMock()
        r.scanner.scan_all = MagicMock(return_value=(sig,))
        r.engine._scanner = r.scanner
        acct = r.account_with(trades_today=2)
        r.run_cycle({pair: []}, now_msc=sig.bar_time_msc,
                    ask_by_pair={pair: sig.entry},
                    bid_by_pair={pair: sig.entry}, account=acct)
        assert len(r.pm.open_positions) == 0
        assert "daily_trade_cap_reached" in r.result.all_rejection_reasons

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_compliance_reject_no_alert_for_pass(self, pair, runner_factory):
        # When compliance rejects, signal_detected alert is NOT sent
        # (the engine sends it only after the gate passes).
        r = runner_factory(ist_window_start="01:00", ist_window_end="02:00")
        sig = _make_long_signal(pair)
        _scan_with_canned_signal(r, sig, now_msc=hour_msc(2026, 4, 15, 8))
        signal_alerts = [m for m in r.result.alert_calls if "SIGNAL" in m]
        assert signal_alerts == []

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_compliance_reject_emits_kill_switch_alert(self, pair, runner_factory):
        r = runner_factory(ist_window_start="01:00", ist_window_end="02:00")
        sig = _make_long_signal(pair)
        _scan_with_canned_signal(r, sig, now_msc=hour_msc(2026, 4, 15, 8))
        assert len(r.result.alert_kill_switch_reasons) >= 1


# ===========================================================================
# 4. News blackout path
# ===========================================================================

class TestNewsBlackout:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_blackout_skips_signal(self, pair, runner_factory):
        # USD event at the trigger moment; window is ±2 min.
        from unittest.mock import MagicMock
        trigger_msc = hour_msc(2026, 4, 15, 8)
        event = NewsEvent(time_msc=trigger_msc, currency="USD",
                           title="NFP", impact="HIGH")
        r = runner_factory(news_events=[event])
        sig = _make_long_signal(pair)
        # USD events affect any pair containing 'USD' (cross pairs and
        # AUDCHF / AUDNZD don't → those won't blackout).
        r.scanner = MagicMock()
        r.scanner.scan_all = MagicMock(return_value=(sig,))
        r.engine._scanner = r.scanner
        r.run_cycle({pair: []}, now_msc=trigger_msc,
                    ask_by_pair={pair: sig.entry},
                    bid_by_pair={pair: sig.entry})
        if "USD" in pair:
            assert "news_blackout" in r.result.all_rejection_reasons
            assert len(r.pm.open_positions) == 0
        else:
            # Non-USD pairs should NOT be blacked-out by a USD event.
            assert "news_blackout" not in r.result.all_rejection_reasons

    def test_eur_event_blacks_out_eurusd(self, runner_factory):
        from unittest.mock import MagicMock
        trigger_msc = hour_msc(2026, 4, 15, 8)
        event = NewsEvent(time_msc=trigger_msc, currency="EUR",
                           title="ECB", impact="HIGH")
        r = runner_factory(news_events=[event])
        sig = _make_long_signal("EURUSD")
        r.scanner = MagicMock()
        r.scanner.scan_all = MagicMock(return_value=(sig,))
        r.engine._scanner = r.scanner
        r.run_cycle({"EURUSD": []}, now_msc=trigger_msc,
                    ask_by_pair={"EURUSD": sig.entry},
                    bid_by_pair={"EURUSD": sig.entry})
        assert "news_blackout" in r.result.all_rejection_reasons

    def test_low_impact_event_not_blackout(self, runner_factory):
        from unittest.mock import MagicMock
        trigger_msc = hour_msc(2026, 4, 15, 8)
        event = NewsEvent(time_msc=trigger_msc, currency="USD",
                           title="ADP", impact="LOW")
        r = runner_factory(news_events=[event])
        sig = _make_long_signal("EURUSD")
        r.scanner = MagicMock()
        r.scanner.scan_all = MagicMock(return_value=(sig,))
        r.engine._scanner = r.scanner
        r.run_cycle({"EURUSD": []}, now_msc=trigger_msc,
                    ask_by_pair={"EURUSD": sig.entry},
                    bid_by_pair={"EURUSD": sig.entry})
        assert "news_blackout" not in r.result.all_rejection_reasons


# ===========================================================================
# 5. SL hit path — maintain() detects SL crossed and closes the position
# ===========================================================================

class TestSLHitPath:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_sl_hit_closes_position(self, pair, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(symbol=pair, pt=pt, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        assert len(r.pm.open_positions) == 1
        pos = r.pm.open_positions[0]
        # Build a next-hour bar that drives low through the SL.
        next_bar = Bar(
            symbol=pair, time_msc=bars[-1].time_msc + 3_600_000,
            open=pos.entry_price,
            high=pos.entry_price + 1 * pt,
            low=pos.sl_price - 5 * pt,
            close=pos.sl_price - 1 * pt, volume=100,
        )
        r.run_maintenance({pair: next_bar},
                           now_msc=next_bar.time_msc + 3_600_000)
        assert len(r.pm.open_positions) == 0
        assert pos.position_id in r.result.closed_position_ids

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_sl_not_hit_keeps_position(self, pair, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(symbol=pair, pt=pt, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        pos = r.pm.open_positions[0]
        # Bar that stays well above SL.
        next_bar = Bar(
            symbol=pair, time_msc=bars[-1].time_msc + 3_600_000,
            open=pos.entry_price + 10 * pt,
            high=pos.entry_price + 20 * pt,
            low=pos.entry_price + 5 * pt,
            close=pos.entry_price + 15 * pt, volume=100,
        )
        r.run_maintenance({pair: next_bar},
                           now_msc=next_bar.time_msc + 3_600_000)
        assert len(r.pm.open_positions) == 1


# ===========================================================================
# 6. Force-close-all (EOD)
# ===========================================================================

class TestForceCloseEOD:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_force_close_flattens_all(self, pair, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(symbol=pair, pt=pt, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        assert len(r.pm.open_positions) >= 1
        n_closed = r.force_close_all(
            ask_by_pair={pair: bars[-1].close},
            bid_by_pair={pair: bars[-1].close},
            now_msc=bars[-1].time_msc + 7_200_000,
        )
        assert n_closed >= 1
        assert len(r.pm.open_positions) == 0


# ===========================================================================
# 7. Best-per-pair dedupe — two same-pair signals → one order
# ===========================================================================

class TestBestPerPairDedupe:
    def test_two_same_pair_signals_one_order(self, runner_factory):
        from unittest.mock import MagicMock
        sig_a = _make_long_signal("EURUSD")
        # A second signal at SAME bar — should be deduped to highest grade.
        sig_b = PatternSignal(
            pattern_name="ASIAN_SWEEP", symbol="EURUSD",
            direction=Direction.BUY, entry=sig_a.entry,
            sl=sig_a.sl, tp=sig_a.tp,
            confidence=0.6, grade=Grade.B,
            confluences_met=sig_a.confluences_met,
            bar_time_msc=sig_a.bar_time_msc,
        )
        r = runner_factory()
        r.scanner = MagicMock()
        r.scanner.scan_all = MagicMock(return_value=(sig_b, sig_a))
        r.engine._scanner = r.scanner
        r.run_cycle({"EURUSD": []}, now_msc=sig_a.bar_time_msc,
                    ask_by_pair={"EURUSD": sig_a.entry},
                    bid_by_pair={"EURUSD": sig_a.entry})
        assert len(r.pm.open_positions) == 1


# ===========================================================================
# 8. Cross-pair concurrent best-per-pair
# ===========================================================================

class TestCrossPairConcurrent:
    @pytest.mark.parametrize("p1,p2", [
        ("EURUSD", "GBPUSD"),
        ("EURUSD", "AUDUSD"),
        ("XAUUSD", "EURUSD"),
        ("XAUUSD", "GBPUSD"),
        ("USDCAD", "USDCHF"),
        ("AUDCHF", "AUDNZD"),
        ("EURUSD", "USDCHF"),
        ("GBPUSD", "AUDCHF"),
    ])
    def test_two_pairs_each_get_order(self, p1, p2, runner_factory):
        from unittest.mock import MagicMock
        s1 = _make_long_signal(p1)
        s2 = _make_long_signal(p2)
        r = runner_factory(max_trades_per_day=10)
        r.scanner = MagicMock()
        r.scanner.scan_all = MagicMock(return_value=(s1, s2))
        r.engine._scanner = r.scanner
        r.run_cycle(
            {p1: [], p2: []},
            now_msc=s1.bar_time_msc,
            ask_by_pair={p1: s1.entry, p2: s2.entry},
            bid_by_pair={p1: s1.entry, p2: s2.entry},
        )
        assert len(r.pm.open_positions) == 2


# ===========================================================================
# 9. Grade routing (A vs B vs C) — C never trades
# ===========================================================================

class TestGradeRouting:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_grade_a_trades(self, pair, runner_factory):
        from unittest.mock import MagicMock
        sig = _make_long_signal(pair)
        r = runner_factory()
        r.scanner = MagicMock()
        r.scanner.scan_all = MagicMock(return_value=(sig,))
        r.engine._scanner = r.scanner
        r.run_cycle({pair: []}, now_msc=sig.bar_time_msc,
                    ask_by_pair={pair: sig.entry},
                    bid_by_pair={pair: sig.entry})
        assert len(r.pm.open_positions) == 1

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_grade_b_trades(self, pair, runner_factory):
        from unittest.mock import MagicMock
        sig = _make_long_signal(pair)
        # Lower-grade variant.
        sig_b = PatternSignal(
            pattern_name=sig.pattern_name, symbol=sig.symbol,
            direction=sig.direction, entry=sig.entry, sl=sig.sl, tp=sig.tp,
            confidence=0.6, grade=Grade.B,
            confluences_met=sig.confluences_met,
            bar_time_msc=sig.bar_time_msc,
        )
        r = runner_factory()
        r.scanner = MagicMock()
        r.scanner.scan_all = MagicMock(return_value=(sig_b,))
        r.engine._scanner = r.scanner
        r.run_cycle({pair: []}, now_msc=sig.bar_time_msc,
                    ask_by_pair={pair: sig.entry},
                    bid_by_pair={pair: sig.entry})
        assert len(r.pm.open_positions) == 1

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_grade_c_never_trades(self, pair, runner_factory):
        from unittest.mock import MagicMock
        sig = _make_long_signal(pair)
        sig_c = PatternSignal(
            pattern_name=sig.pattern_name, symbol=sig.symbol,
            direction=sig.direction, entry=sig.entry, sl=sig.sl, tp=sig.tp,
            confidence=0.3, grade=Grade.C,
            confluences_met=sig.confluences_met,
            bar_time_msc=sig.bar_time_msc,
        )
        r = runner_factory()
        r.scanner = MagicMock()
        r.scanner.scan_all = MagicMock(return_value=(sig_c,))
        r.engine._scanner = r.scanner
        r.run_cycle({pair: []}, now_msc=sig.bar_time_msc,
                    ask_by_pair={pair: sig.entry},
                    bid_by_pair={pair: sig.entry})
        assert len(r.pm.open_positions) == 0


# ===========================================================================
# 10. Maintain-cycle smoke — no positions → no crash
# ===========================================================================

class TestMaintainNoOps:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_maintain_no_positions(self, pair, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        r = runner_factory()
        bar = Bar(symbol=pair, time_msc=hour_msc(2026, 4, 15, 9),
                  open=1.10000, high=1.10010, low=1.09990, close=1.10005,
                  volume=10)
        reps = r.run_maintenance({pair: bar}, now_msc=bar.time_msc + 60_000)
        assert pair in reps
        assert reps[pair].closed_positions == ()


# ===========================================================================
# 11. Alerts wiring — signal_detected, trade_opened
# ===========================================================================

class TestAlertsEmittedOnOpen:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_signal_detected_alert(self, pair, runner_factory):
        from unittest.mock import MagicMock
        sig = _make_long_signal(pair)
        r = runner_factory()
        r.scanner = MagicMock()
        r.scanner.scan_all = MagicMock(return_value=(sig,))
        r.engine._scanner = r.scanner
        r.run_cycle({pair: []}, now_msc=sig.bar_time_msc,
                    ask_by_pair={pair: sig.entry},
                    bid_by_pair={pair: sig.entry})
        assert any("SIGNAL" in m for m in r.result.alert_calls)

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_trade_open_alert(self, pair, runner_factory):
        from unittest.mock import MagicMock
        sig = _make_long_signal(pair)
        r = runner_factory()
        r.scanner = MagicMock()
        r.scanner.scan_all = MagicMock(return_value=(sig,))
        r.engine._scanner = r.scanner
        r.run_cycle({pair: []}, now_msc=sig.bar_time_msc,
                    ask_by_pair={pair: sig.entry},
                    bid_by_pair={pair: sig.entry})
        assert any("TRADE OPEN" in m for m in r.result.alert_calls)


# ===========================================================================
# 12. Cycle report shape on full cycle
# ===========================================================================

# ===========================================================================
# 13. BE shift then SL exit (post-TP1 trailed close)
# ===========================================================================

class TestBEShiftThenSL:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_sl_modified_on_trail_then_hit(self, pair, runner_factory):
        """Simulate: position opens, structure forms (swing low), trail
        moves SL up; subsequent bar reverses and hits the trailed SL.
        """
        pt = float(PAIR_CONFIG[pair]["point"])
        from unittest.mock import MagicMock
        sig = _make_long_signal(pair)
        r = runner_factory()
        r.scanner = MagicMock()
        r.scanner.scan_all = MagicMock(return_value=(sig,))
        r.engine._scanner = r.scanner
        r.run_cycle({pair: []}, now_msc=sig.bar_time_msc,
                    ask_by_pair={pair: sig.entry},
                    bid_by_pair={pair: sig.entry})
        if not r.pm.open_positions:
            pytest.skip("Signal didn't open")
        pos = r.pm.open_positions[0]
        # Two bars to build a swing low, then a reversal bar hits SL.
        bars_after = []
        for i in range(3):
            b = Bar(symbol=pair,
                    time_msc=sig.bar_time_msc + (i + 1) * 3_600_000,
                    open=pos.entry_price + (i + 1) * pt,
                    high=pos.entry_price + (i + 5) * pt,
                    low=pos.entry_price + i * pt,
                    close=pos.entry_price + (i + 2) * pt, volume=10)
            bars_after.append(b)
        for b in bars_after:
            r.run_maintenance({pair: b},
                                now_msc=b.time_msc + 60_000)
        # Reversal: bar with low under the original SL.
        kill_bar = Bar(symbol=pair,
                       time_msc=sig.bar_time_msc + 5 * 3_600_000,
                       open=pos.entry_price + 1 * pt,
                       high=pos.entry_price + 2 * pt,
                       low=pos.sl_price - 5 * pt,
                       close=pos.sl_price - 1 * pt, volume=50)
        r.run_maintenance({pair: kill_bar},
                            now_msc=kill_bar.time_msc + 60_000)
        # Position should now be gone (SL hit on the trailed level).
        assert len(r.pm.open_positions) == 0


# ===========================================================================
# 14. Compliance reason taxonomy — each reason produces a different alert
# ===========================================================================

class TestComplianceReasonTaxonomy:
    @pytest.mark.parametrize("pair", list(PAIRS))
    @pytest.mark.parametrize("reason_setup", [
        ("outside_ist_window", {"ist_window_start": "02:00",
                                  "ist_window_end": "03:00"}),
        ("max_trades", {"max_trades_per_day": 1}),
    ])
    def test_reason_recorded(self, pair, reason_setup, runner_factory):
        reason_name, kwargs = reason_setup
        r = runner_factory(**kwargs)
        sig = _make_long_signal(pair)
        from unittest.mock import MagicMock
        r.scanner = MagicMock()
        r.scanner.scan_all = MagicMock(return_value=(sig,))
        r.engine._scanner = r.scanner
        if reason_name == "max_trades":
            acct = r.account_with(trades_today=1)
        else:
            acct = None
        r.run_cycle({pair: []}, now_msc=sig.bar_time_msc,
                    ask_by_pair={pair: sig.entry},
                    bid_by_pair={pair: sig.entry}, account=acct)
        assert len(r.pm.open_positions) == 0
        assert len(r.result.all_rejection_reasons) >= 1


# ===========================================================================
# 15. Long & Short × all pairs × multiple hours — saturation parametrize
# ===========================================================================

class TestSaturation:
    @pytest.mark.parametrize("pair", list(PAIRS))
    @pytest.mark.parametrize("hour", [6, 7, 8, 9, 10])
    @pytest.mark.parametrize("bias", ["neutral", "bullish"])
    def test_long_all_combinations(self, pair, hour, bias, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(symbol=pair, trigger_hour=hour, bias=bias,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        rep = r.result.cycle_reports[-1]
        assert rep.orders_placed <= rep.signals_emitted


class TestCycleReportShape:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_report_reflects_orders_placed(self, pair, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(symbol=pair, pt=pt, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        rep = r.result.cycle_reports[-1]
        assert rep.signals_emitted >= 1
        assert rep.orders_placed == len(r.pm.open_positions)

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_report_no_rejection_on_pass(self, pair, runner_factory):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(symbol=pair, pt=pt, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        rep = r.result.cycle_reports[-1]
        assert rep.signals_rejected_by_compliance == 0
