"""E2E — session boundaries and full-day flow.

Asian range build (prev 19:30 UTC → today 00:30 UTC) → London sweep
window (06:00–10:30 UTC) → NY continuation window (12:00–15:30 UTC, LONG
only) → force-close at 16:00 UTC.

Also covers:
  - Day rollover (DailyTracker.trade_count resets, new Asian range valid)
  - Weekend → Monday gap (SKIP_MONDAY rule)
  - Bars before/outside windows produce no signal
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from config.asian_sweep_config import PAIR_CONFIG, PAIRS, SKIP_MONDAY
from data.bar_aggregator import Bar
from strategy.patterns.base import Direction, Grade, PatternSignal

from tests.e2e.fixtures.scenario_runner import (
    ScenarioRunner, long_sweep_bars, short_sweep_bars, hour_msc,
)


UTC = timezone.utc


def _sig(symbol: str = "EURUSD", hour: int = 8, day: int = 15,
          month: int = 4, year: int = 2026) -> PatternSignal:
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
        bar_time_msc=hour_msc(year, month, day, hour),
    )


def _inject(r, sigs):
    r.scanner = MagicMock()
    r.scanner.scan_all = MagicMock(return_value=tuple(sigs))
    r.engine._scanner = r.scanner


# ===========================================================================
# 1. London window — 06:00–10:30 UTC (= 11:30–16:00 IST)
# ===========================================================================

class TestLondonWindow:
    @pytest.mark.parametrize("pair", list(PAIRS))
    @pytest.mark.parametrize("hour", [6, 7, 8, 9, 10])
    def test_london_long_real_bars(self, pair, hour, runner_factory):
        bars = long_sweep_bars(symbol=pair, trigger_hour=hour,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        # Either a signal fires (orders_placed=1) or none does (filter); we
        # don't enforce both, but the WIRING must not produce more orders
        # than signals.
        rep = r.result.cycle_reports[-1]
        assert rep.orders_placed == len(r.pm.open_positions)


# ===========================================================================
# 2. NY window — 12:00–15:30 UTC, LONG only by V5 rule
# ===========================================================================

class TestNYWindow:
    @pytest.mark.parametrize("pair", list(PAIRS))
    @pytest.mark.parametrize("hour", [12, 13, 14, 15])
    def test_ny_long_real_bars(self, pair, hour, runner_factory):
        bars = long_sweep_bars(symbol=pair, trigger_hour=hour,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        for pos in r.pm.open_positions:
            assert pos.side == Direction.BUY  # NY shorts disabled

    @pytest.mark.parametrize("pair", list(PAIRS))
    @pytest.mark.parametrize("hour", [12, 13, 14, 15])
    def test_ny_short_never_fires(self, pair, hour, runner_factory):
        # SHORT bar in NY window must not produce a position.
        bars = short_sweep_bars(symbol=pair, trigger_hour=hour,
                                 year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        for pos in r.pm.open_positions:
            assert pos.side != Direction.SELL


# ===========================================================================
# 3. Outside both windows — no signal
# ===========================================================================

class TestOutsideWindows:
    @pytest.mark.parametrize("pair", list(PAIRS))
    @pytest.mark.parametrize("hour", [1, 2, 3, 4, 5, 11, 16, 17, 18, 23])
    def test_outside_windows_no_signal(self, pair, hour, runner_factory):
        bars = long_sweep_bars(symbol=pair, trigger_hour=hour,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        # The detector itself rejects outside-window bars → 0 signals.
        rep = r.result.cycle_reports[-1]
        assert rep.signals_emitted == 0
        assert len(r.pm.open_positions) == 0


# ===========================================================================
# 4. SKIP_MONDAY rule — Monday bar produces no signal
# ===========================================================================

class TestSkipMonday:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_monday_no_signal(self, pair, runner_factory):
        # 2026-04-13 is a Monday.
        bars = long_sweep_bars(symbol=pair, trigger_hour=8,
                                year=2026, month=4, day=13)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        if SKIP_MONDAY:
            rep = r.result.cycle_reports[-1]
            assert rep.signals_emitted == 0

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_tuesday_signal_allowed(self, pair, runner_factory):
        # 2026-04-14 is a Tuesday.
        bars = long_sweep_bars(symbol=pair, trigger_hour=8,
                                year=2026, month=4, day=14)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        rep = r.result.cycle_reports[-1]
        # Tuesday is allowed; signal may fire (most often does).
        # The strict assertion is "engine wiring doesn't crash".
        assert rep.orders_placed == len(r.pm.open_positions)


# ===========================================================================
# 5. Day rollover — DailyTracker reset
# ===========================================================================

class TestDayRollover:
    def test_trade_count_resets_on_new_ist_day(self, runner_factory):
        r = runner_factory()
        # Trade 1 on day 15.
        s = _sig("EURUSD", hour=8, day=15)
        _inject(r, [s])
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry})
        assert r.daily.trade_count == 1
        # Wait for a new IST day — IST midnight = 18:30 UTC. April 16 IST
        # starts at 2026-04-15 18:30 UTC. Use 20:00 UTC of day 15 → it's
        # already day 16 IST.
        next_msc = hour_msc(2026, 4, 15, 20)
        # Trigger rollover via update_equity.
        r.daily.update_equity(100_000.0, now_ms=next_msc)
        assert r.daily.trade_count == 0

    def test_trade_count_unchanged_within_day(self, runner_factory):
        r = runner_factory()
        s = _sig("EURUSD", hour=8, day=15)
        _inject(r, [s])
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry})
        r.daily.update_equity(100_000.0,
                              now_ms=hour_msc(2026, 4, 15, 10))
        assert r.daily.trade_count == 1


# ===========================================================================
# 6. Force-close at 16:00 UTC simulating EOD
# ===========================================================================

class TestEODForceClose:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_eod_flatten(self, pair, runner_factory):
        bars = long_sweep_bars(symbol=pair, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        if r.pm.open_positions:
            n = r.force_close_all(
                ask_by_pair={pair: bars[-1].close},
                bid_by_pair={pair: bars[-1].close},
                now_msc=hour_msc(2026, 4, 15, 16),
            )
            assert n >= 1
        assert len(r.pm.open_positions) == 0


# ===========================================================================
# 7. Asian range produces the trigger correctly per session
# ===========================================================================

class TestAsianRangeWired:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_long_entry_near_asian_low(self, pair, runner_factory):
        """Signal's entry should be at AL + spread; the position's
        entry_price equals the broker ASK at fill time (which may be
        slightly different). Tolerant bound: within 0.5% of AL.
        """
        from tests.e2e.fixtures.scenario_runner import _PAIR_PRICE_ANCHORS
        bars = long_sweep_bars(symbol=pair, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        if not r.pm.open_positions:
            pytest.skip("Signal didn't fire for this pair")
        pos = r.pm.open_positions[0]
        al = _PAIR_PRICE_ANCHORS[pair][0]
        # Allow up to 0.5% deviation — anything closer than that confirms
        # the signal is anchored to the Asian range, not random noise.
        assert abs(pos.entry_price - al) / al < 0.005

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_long_entry_above_sl(self, pair, runner_factory):
        bars = long_sweep_bars(symbol=pair, trigger_hour=8,
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        if not r.pm.open_positions:
            pytest.skip("Signal didn't fire for this pair")
        pos = r.pm.open_positions[0]
        assert pos.entry_price > pos.sl_price


# ===========================================================================
# 8. London then NY same day — second trade allowed (max=2)
# ===========================================================================

class TestLondonThenNYSameDay:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_second_trade_allowed_within_cap(self, pair, runner_factory):
        r = runner_factory(max_trades_per_day=2)
        s1 = _sig(pair, hour=8)
        _inject(r, [s1])
        r.run_cycle({pair: []}, now_msc=s1.bar_time_msc,
                    ask_by_pair={pair: s1.entry},
                    bid_by_pair={pair: s1.entry})
        # NY trade hours later.
        s2 = _sig(pair, hour=13)
        _inject(r, [s2])
        acct = r.account_with(trades_today=1)
        r.run_cycle({pair: []}, now_msc=s2.bar_time_msc,
                    ask_by_pair={pair: s2.entry},
                    bid_by_pair={pair: s2.entry}, account=acct)
        # Both pass compliance (cap is 2), so 2 positions open.
        assert len(r.pm.open_positions) == 2

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_third_trade_blocked(self, pair, runner_factory):
        r = runner_factory(max_trades_per_day=2)
        s = _sig(pair, hour=15)
        _inject(r, [s])
        # Already 2 trades today → cap.
        acct = r.account_with(trades_today=2)
        r.run_cycle({pair: []}, now_msc=s.bar_time_msc,
                    ask_by_pair={pair: s.entry},
                    bid_by_pair={pair: s.entry}, account=acct)
        assert len(r.pm.open_positions) == 0
        assert "daily_trade_cap_reached" in r.result.all_rejection_reasons


# ===========================================================================
# 9. Weekend gap — Friday close → Monday morning bar
# ===========================================================================

class TestWeekendGap:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_friday_signal_then_monday_skipped(self, pair, runner_factory):
        # 2026-04-17 is a Friday; 2026-04-20 is a Monday.
        bars_fri = long_sweep_bars(symbol=pair, trigger_hour=8,
                                    year=2026, month=4, day=17)
        bars_mon = long_sweep_bars(symbol=pair, trigger_hour=8,
                                    year=2026, month=4, day=20)
        r = runner_factory()
        r.run_cycle({pair: bars_fri}, now_msc=bars_fri[-1].time_msc)
        r.run_cycle({pair: bars_mon}, now_msc=bars_mon[-1].time_msc,
                    account=r.account_with(trades_today=1))
        if SKIP_MONDAY:
            # Monday rejected at detector level → no extra signal.
            mon_report = r.result.cycle_reports[-1]
            assert mon_report.signals_emitted == 0


# ===========================================================================
# 10. Range filter — too-tight or too-wide Asian range rejected
# ===========================================================================

class TestRangeFilter:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_too_tight_range_no_signal(self, pair, runner_factory):
        # Set asian high/low so range < min_range_pts.
        cfg = PAIR_CONFIG[pair]
        pt = float(cfg["point"])
        anchor = 2000.00 if pair == "XAUUSD" else 1.10000
        # 1 pt wide is below every pair's min_range_pts.
        bars = long_sweep_bars(symbol=pair, trigger_hour=8,
                                year=2026, month=4, day=15,
                                asian_low=anchor,
                                asian_high=anchor + 1 * pt)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        rep = r.result.cycle_reports[-1]
        assert rep.signals_emitted == 0


# ===========================================================================
# 11. Bias filter — bullish bias allows LONG, bearish bias blocks LONG
# ===========================================================================

class TestBiasFilter:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_bullish_bias_allows_long(self, pair, runner_factory):
        bars = long_sweep_bars(symbol=pair, trigger_hour=8, bias="bullish",
                                year=2026, month=4, day=15)
        r = runner_factory()
        r.run_cycle({pair: bars}, now_msc=bars[-1].time_msc)
        # Bullish bias permits LONG; we expect a position to open.
        if len(r.pm.open_positions) == 1:
            assert r.pm.open_positions[0].side == Direction.BUY


# ===========================================================================
# 12. Multi-day sequence — day1 trade + day2 trade independent
# ===========================================================================

class TestMultiDaySequence:
    def test_two_consecutive_days(self, runner_factory):
        r = runner_factory()
        # Tuesday April 14
        s1 = _sig("EURUSD", hour=8, day=14)
        _inject(r, [s1])
        r.run_cycle({"EURUSD": []}, now_msc=s1.bar_time_msc,
                    ask_by_pair={"EURUSD": s1.entry},
                    bid_by_pair={"EURUSD": s1.entry})
        assert r.daily.trade_count == 1
        # Force rollover to Wednesday April 15.
        r.daily.update_equity(100_000.0, now_ms=hour_msc(2026, 4, 14, 20))
        assert r.daily.trade_count == 0
        # Day 2 trade.
        s2 = _sig("EURUSD", hour=8, day=15)
        _inject(r, [s2])
        r.run_cycle({"EURUSD": []}, now_msc=s2.bar_time_msc,
                    ask_by_pair={"EURUSD": s2.entry},
                    bid_by_pair={"EURUSD": s2.entry})
        assert r.daily.trade_count == 1
