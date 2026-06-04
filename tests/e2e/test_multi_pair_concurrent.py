"""E2E — multi-pair concurrent signals.

Targets the wiring around `process_scan_cycle` when signals appear on
multiple pairs in the same cycle:
  - Per-pair position tracking isolation
  - Global trade-count cap shared across pairs
  - Compliance-blocked pair does not affect others
  - One-pair error doesn't poison the rest
  - Best-per-pair dedupe still applies per symbol
"""

from __future__ import annotations
from typing import List
from itertools import combinations
from unittest.mock import MagicMock

import pytest

from config.asian_sweep_config import PAIR_CONFIG, PAIRS
from data.bar_aggregator import Bar
from strategy.patterns.base import Direction, Grade, PatternSignal

from tests.e2e.fixtures.scenario_runner import (
    ScenarioRunner, long_sweep_bars, hour_msc,
)


def _sig(symbol: str, *, grade=Grade.A, confidence=0.9,
         direction: Direction = Direction.BUY) -> PatternSignal:
    pt = float(PAIR_CONFIG[symbol]["point"])
    # Big-point instruments (XAUUSD + index .cash pairs) need a price well above
    # 2.5×risk so a SHORT's tp stays positive; FX pairs use a 1.1000 anchor.
    entry = 2000.00 if pt >= 0.01 else 1.10000
    risk = 100 * pt
    if direction == Direction.BUY:
        sl, tp, sweep = entry - risk, entry + risk * 2.5, "asian_sweep_low"
    else:
        sl, tp, sweep = entry + risk, entry - risk * 2.5, "asian_sweep_high"
    return PatternSignal(
        pattern_name="ASIAN_SWEEP", symbol=symbol,
        direction=direction, entry=entry,
        sl=sl, tp=tp,
        confidence=confidence, grade=grade,
        confluences_met=(sweep, "LONDON", "bias_neutral",
                          "q9", f"tp1_{entry + risk:.5f}"),
        bar_time_msc=hour_msc(2026, 4, 15, 8),
    )


def _inject_signals(r: ScenarioRunner, sigs):
    r.scanner = MagicMock()
    r.scanner.scan_all = MagicMock(return_value=tuple(sigs))
    r.engine._scanner = r.scanner


# ===========================================================================
# 1. Two-pair pairs — all C(8,2) = 28 ordered pairs
# ===========================================================================

ALL_PAIR_COMBOS = list(combinations(PAIRS, 2))


class TestTwoPairConcurrent:
    @pytest.mark.parametrize("p1,p2", ALL_PAIR_COMBOS)
    def test_two_pairs_both_open(self, p1, p2, runner_factory):
        # Opposite directions — the 1-per-direction/day gate is global across
        # symbols, so two same-direction pairs would cap at one.
        s1, s2 = _sig(p1), _sig(p2, direction=Direction.SELL)
        r = runner_factory(max_trades_per_day=10)
        _inject_signals(r, [s1, s2])
        r.run_cycle({p1: [], p2: []}, now_msc=s1.bar_time_msc,
                    ask_by_pair={p1: s1.entry, p2: s2.entry},
                    bid_by_pair={p1: s1.entry, p2: s2.entry})
        assert len(r.pm.open_positions) == 2
        symbols = {p.symbol for p in r.pm.open_positions}
        assert symbols == {p1, p2}

    @pytest.mark.parametrize("p1,p2", ALL_PAIR_COMBOS)
    def test_two_pairs_isolated_positions(self, p1, p2, runner_factory):
        s1, s2 = _sig(p1), _sig(p2, direction=Direction.SELL)
        r = runner_factory(max_trades_per_day=10)
        _inject_signals(r, [s1, s2])
        r.run_cycle({p1: [], p2: []}, now_msc=s1.bar_time_msc,
                    ask_by_pair={p1: s1.entry, p2: s2.entry},
                    bid_by_pair={p1: s1.entry, p2: s2.entry})
        for_p1 = r.pm.positions_for(p1)
        for_p2 = r.pm.positions_for(p2)
        assert len(for_p1) == 1
        assert len(for_p2) == 1
        assert for_p1[0].symbol == p1
        assert for_p2[0].symbol == p2

    @pytest.mark.parametrize("p1,p2", ALL_PAIR_COMBOS)
    def test_two_pairs_each_recorded_in_daily_count(self, p1, p2, runner_factory):
        s1, s2 = _sig(p1), _sig(p2, direction=Direction.SELL)
        r = runner_factory(max_trades_per_day=10)
        _inject_signals(r, [s1, s2])
        r.run_cycle({p1: [], p2: []}, now_msc=s1.bar_time_msc,
                    ask_by_pair={p1: s1.entry, p2: s2.entry},
                    bid_by_pair={p1: s1.entry, p2: s2.entry})
        assert r.daily.trade_count == 2


# ===========================================================================
# 2. Global trade cap across pairs
# ===========================================================================

class TestGlobalTradeCap:
    @pytest.mark.parametrize("p1,p2", ALL_PAIR_COMBOS[:14])  # sample
    def test_cap_at_one_blocks_second_pair(self, p1, p2, runner_factory):
        s1, s2 = _sig(p1), _sig(p2)
        r = runner_factory(max_trades_per_day=1)
        _inject_signals(r, [s1, s2])
        # Compliance is stateful via AccountState; we simulate one already
        # taken by passing trades_today=1, so only the second can be checked,
        # which will be rejected when added to the running count (the engine
        # records via daily.record_trade_open after each order; the second
        # signal in THIS cycle will go through compliance against the same
        # account snapshot, so for a strict 1-cap test we set trades_today=1
        # to force both rejections).
        acct = r.account_with(trades_today=1)
        r.run_cycle({p1: [], p2: []}, now_msc=s1.bar_time_msc,
                    ask_by_pair={p1: s1.entry, p2: s2.entry},
                    bid_by_pair={p1: s1.entry, p2: s2.entry},
                    account=acct)
        assert len(r.pm.open_positions) == 0
        assert "daily_trade_cap_reached" in r.result.all_rejection_reasons

    @pytest.mark.parametrize("p1,p2", ALL_PAIR_COMBOS[:14])
    def test_cap_at_zero_blocks_everything(self, p1, p2, runner_factory):
        # Compliance ctor requires max >= 1; we use trades_today >= cap
        # to model the equivalent. Already 2/2 → both rejected.
        s1, s2 = _sig(p1), _sig(p2)
        r = runner_factory(max_trades_per_day=2)
        _inject_signals(r, [s1, s2])
        acct = r.account_with(trades_today=2)
        r.run_cycle({p1: [], p2: []}, now_msc=s1.bar_time_msc,
                    ask_by_pair={p1: s1.entry, p2: s2.entry},
                    bid_by_pair={p1: s1.entry, p2: s2.entry},
                    account=acct)
        assert len(r.pm.open_positions) == 0


# ===========================================================================
# 3. Per-pair best-of dedupe vs cross-pair independence
# ===========================================================================

class TestPerPairBest:
    @pytest.mark.parametrize("p1,p2", ALL_PAIR_COMBOS[:14])
    def test_each_pair_keeps_own_best(self, p1, p2, runner_factory):
        # Two signals per pair — only the best one each fires. Distinct
        # directions per pair so both winners clear the global per-direction gate.
        sigs = []
        for p, d in ((p1, Direction.BUY), (p2, Direction.SELL)):
            sigs.append(_sig(p, grade=Grade.B, confidence=0.5, direction=d))
            sigs.append(_sig(p, grade=Grade.A, confidence=0.9, direction=d))
        r = runner_factory(max_trades_per_day=10)
        _inject_signals(r, sigs)
        r.run_cycle({p1: [], p2: []}, now_msc=sigs[0].bar_time_msc,
                    ask_by_pair={p1: sigs[0].entry, p2: sigs[0].entry},
                    bid_by_pair={p1: sigs[0].entry, p2: sigs[0].entry})
        assert len(r.pm.open_positions) == 2


# ===========================================================================
# 4. Three-pair concurrency from real bar feeds
# ===========================================================================

class TestThreePairConcurrent:
    @pytest.mark.parametrize("p1,p2,p3", [
        ("EURUSD", "GBPUSD", "AUDUSD"),
        ("EURUSD", "GBPUSD", "USDCAD"),
        ("EURUSD", "AUDUSD", "USDCAD"),
        ("EURUSD", "AUDUSD", "USDCHF"),
        ("XAUUSD", "EURUSD", "GBPUSD"),
        ("XAUUSD", "AUDUSD", "USDCAD"),
        ("AUDCHF", "AUDNZD", "EURUSD"),
        ("USDCAD", "USDCHF", "AUDCHF"),
    ])
    def test_three_pairs_clamped_to_two(self, p1, p2, p3, runner_factory):
        """Three simultaneous signals → first 2 open (1 LONG + 1 SHORT), 3rd
        deferred. The global 1-per-direction/day gate is what clamps here: the
        3rd signal repeats a direction already taken, so it's rejected with
        `direction_already_traded_today` before reaching the trade-count cap.
        """
        dirs = (Direction.BUY, Direction.SELL, Direction.BUY)
        sigs = [_sig(p, direction=d) for p, d in zip((p1, p2, p3), dirs)]
        r = runner_factory(max_trades_per_day=10)
        _inject_signals(r, sigs)
        r.run_cycle(
            {p1: [], p2: [], p3: []},
            now_msc=sigs[0].bar_time_msc,
            ask_by_pair={p: s.entry for p, s in zip((p1, p2, p3), sigs)},
            bid_by_pair={p: s.entry for p, s in zip((p1, p2, p3), sigs)},
        )
        assert len(r.pm.open_positions) == 2
        assert "direction_already_traded_today" in r.result.all_rejection_reasons


# ===========================================================================
# 5. Real-feed multi-pair: bars built from synthetic scenarios
# ===========================================================================

class TestRealFeedMultiPair:
    @pytest.mark.parametrize("p1,p2", ALL_PAIR_COMBOS[:14])
    def test_real_bars_multi_pair_signals(self, p1, p2, runner_factory):
        feeds = {
            p1: long_sweep_bars(symbol=p1, trigger_hour=8,
                                  year=2026, month=4, day=15),
            p2: long_sweep_bars(symbol=p2, trigger_hour=8,
                                  year=2026, month=4, day=15),
        }
        # Same trigger bar across pairs.
        now = feeds[p1][-1].time_msc
        r = runner_factory(max_trades_per_day=10)
        r.run_cycle(feeds, now_msc=now,
                    ask_by_pair={p1: feeds[p1][-1].close,
                                 p2: feeds[p2][-1].close},
                    bid_by_pair={p1: feeds[p1][-1].close,
                                 p2: feeds[p2][-1].close})
        # We don't require both signals fired (some pairs / hours yield none)
        # but the engine MUST never open more positions than signals emitted.
        rep = r.result.cycle_reports[-1]
        assert rep.orders_placed <= rep.signals_emitted
        assert rep.orders_placed == len(r.pm.open_positions)


# ===========================================================================
# 6. Compliance rejection on ONE pair does not affect another
# ===========================================================================

class TestIsolatedRejection:
    @pytest.mark.parametrize("p1,p2", ALL_PAIR_COMBOS[:14])
    def test_one_pair_blocked_other_passes(self, p1, p2, runner_factory):
        # Build a news event that only affects USD pairs. Pick a non-USD
        # pair as the "ok" pair if possible.
        from data.news_calendar import NewsEvent
        sigs = [_sig(p) for p in (p1, p2)]
        now = sigs[0].bar_time_msc
        r = runner_factory(
            news_events=[NewsEvent(time_msc=now, currency="USD",
                                     title="NFP", impact="HIGH")],
            max_trades_per_day=10,
        )
        _inject_signals(r, sigs)
        r.run_cycle({p1: [], p2: []}, now_msc=now,
                    ask_by_pair={p1: sigs[0].entry, p2: sigs[1].entry},
                    bid_by_pair={p1: sigs[0].entry, p2: sigs[1].entry})
        usd_count = sum(1 for p in (p1, p2) if "USD" in p)
        non_usd_count = 2 - usd_count
        # All non-USD pairs should have opened; USD ones blocked.
        assert len(r.pm.open_positions) == non_usd_count


# ===========================================================================
# 7. Position-manager forget vs the engine's open_positions view
# ===========================================================================

class TestPMForget:
    def test_forget_position_removes_only_one(self, runner_factory):
        sigs = [_sig("EURUSD"), _sig("GBPUSD", direction=Direction.SELL)]
        r = runner_factory(max_trades_per_day=10)
        _inject_signals(r, sigs)
        r.run_cycle({"EURUSD": [], "GBPUSD": []},
                    now_msc=sigs[0].bar_time_msc,
                    ask_by_pair={"EURUSD": sigs[0].entry,
                                 "GBPUSD": sigs[1].entry},
                    bid_by_pair={"EURUSD": sigs[0].entry,
                                 "GBPUSD": sigs[1].entry})
        assert len(r.pm.open_positions) == 2
        first_id = r.pm.open_positions[0].position_id
        r.pm.forget_position(first_id)
        assert len(r.pm.open_positions) == 1


# ===========================================================================
# 8. Eight pairs in one cycle — stress wiring
# ===========================================================================

class TestEightPairCycle:
    def test_all_eight_pairs_with_real_bars_clamps_to_two(self, runner_factory):
        """Phase 6 fix #4 — engine no longer crashes when 8 pairs signal
        the same cycle. House Money cap clamps fills to 2."""
        r = runner_factory(max_trades_per_day=10)
        feeds = {
            p: long_sweep_bars(symbol=p, trigger_hour=8,
                                year=2026, month=4, day=15)
            for p in PAIRS
        }
        now = feeds["EURUSD"][-1].time_msc
        r.run_cycle(feeds, now_msc=now,
                    ask_by_pair={p: f[-1].close for p, f in feeds.items()},
                    bid_by_pair={p: f[-1].close for p, f in feeds.items()})
        rep = r.result.cycle_reports[-1]
        # No more than 2 fills regardless of how many signals fired.
        assert rep.orders_placed <= 2
        assert len(r.pm.open_positions) <= 2

    def test_all_eight_canned_signals_clamps_to_two(self, runner_factory):
        r = runner_factory(max_trades_per_day=20)
        # Alternate directions: the global 1-per-direction/day gate fills the
        # 1 LONG + 1 SHORT slots, then defers every further signal — regardless
        # of trade-count cap, since both directions are already taken.
        dirs = [Direction.BUY if i % 2 == 0 else Direction.SELL
                for i in range(len(PAIRS))]
        sigs = [_sig(p, direction=d) for p, d in zip(PAIRS, dirs)]
        _inject_signals(r, sigs)
        r.run_cycle(
            {p: [] for p in PAIRS},
            now_msc=sigs[0].bar_time_msc,
            ask_by_pair={p: s.entry for p, s in zip(PAIRS, sigs)},
            bid_by_pair={p: s.entry for p, s in zip(PAIRS, sigs)},
        )
        assert len(r.pm.open_positions) == 2
        # The other signals are all deferred by the per-direction gate.
        dir_rejections = [r for r in r.result.all_rejection_reasons
                           if r == "direction_already_traded_today"]
        assert len(dir_rejections) == len(PAIRS) - 2


# ===========================================================================
# 9. Sequential cycles — each cycle adds independently
# ===========================================================================

class TestSequentialCycles:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_two_cycles_two_pairs(self, pair, runner_factory):
        other = "USDCHF" if pair != "USDCHF" else "EURUSD"
        r = runner_factory(max_trades_per_day=10)
        s1 = _sig(pair)
        _inject_signals(r, [s1])
        r.run_cycle({pair: []}, now_msc=s1.bar_time_msc,
                    ask_by_pair={pair: s1.entry},
                    bid_by_pair={pair: s1.entry})
        # Second cycle: new pair, opposite direction (same IST day, so the
        # per-direction ledger still holds the BUY from cycle 1).
        s2 = _sig(other, direction=Direction.SELL)
        _inject_signals(r, [s2])
        r.run_cycle({other: []}, now_msc=s2.bar_time_msc + 3_600_000,
                    ask_by_pair={other: s2.entry},
                    bid_by_pair={other: s2.entry},
                    account=r.account_with(trades_today=1))
        assert len(r.pm.open_positions) == 2

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_quiet_cycle_then_active_cycle(self, pair, runner_factory):
        r = runner_factory()
        # Quiet cycle.
        _inject_signals(r, [])
        r.run_cycle({pair: []}, now_msc=hour_msc(2026, 4, 15, 4))
        assert len(r.pm.open_positions) == 0
        # Active cycle.
        sig = _sig(pair)
        _inject_signals(r, [sig])
        r.run_cycle({pair: []}, now_msc=sig.bar_time_msc,
                    ask_by_pair={pair: sig.entry},
                    bid_by_pair={pair: sig.entry})
        assert len(r.pm.open_positions) == 1
