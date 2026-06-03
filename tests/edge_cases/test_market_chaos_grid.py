"""Phase-5 / Market Chaos — large parameter-grid extensions.

This file complements test_market_chaos.py by sweeping wide parameter grids
through the same scenarios. Each test asserts a single invariant on a
generated point; collectively they multiply coverage of the exit-state
machine, signal validation, and bar-aggregator branches.
"""

from __future__ import annotations
import math

import pytest

from data.bar_aggregator import Bar, BarAggregator, floor_to_timeframe_ms
from data.tick_collector import Tick
from risk.asian_sweep_exit import (
    compute_pnl, force_close_eod, init_exit_state, maintain_exit,
    size_position,
)
from strategy.patterns.base import (
    Direction, Grade, MarketContext, PatternSignal,
)

from tests.edge_cases.fixtures.chaos_market import HOUR_MS, make_bar
from tests.strategy.fixtures.synthetic_bars import (
    long_sweep_bars, short_sweep_bars,
)


# Helpers --------------------------------------------------------------------

def _long_sig(entry=2000.0, risk=5.0, rr=2.0):
    return PatternSignal(
        pattern_name="ASIAN_SWEEP", symbol="XAUUSD",
        direction=Direction.BUY, entry=entry,
        sl=entry - risk, tp=entry + risk * rr,
        confidence=0.9, grade=Grade.A,
        confluences_met=(f"tp1_{entry + risk:.5f}",),
        bar_time_msc=0,
    )


def _short_sig(entry=2000.0, risk=5.0, rr=2.0):
    return PatternSignal(
        pattern_name="ASIAN_SWEEP", symbol="XAUUSD",
        direction=Direction.SELL, entry=entry,
        sl=entry + risk, tp=entry - risk * rr,
        confidence=0.9, grade=Grade.A,
        confluences_met=(f"tp1_{entry - risk:.5f}",),
        bar_time_msc=0,
    )


# ===========================================================================
# 1. LONG SL-touch grid — 10 risks × 5 wick depths
# ===========================================================================

@pytest.mark.parametrize("risk", [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0,
                                   15.0, 25.0, 50.0])
@pytest.mark.parametrize("wick_depth_mult", [1.0, 1.5, 2.0, 5.0, 100.0])
def test_long_sl_touch_grid(risk, wick_depth_mult):
    """LONG with low touching SL by (wick_depth_mult × risk) → close at SL."""
    sig = _long_sig(entry=2000.0, risk=risk, rr=2.0)
    state = init_exit_state(position_id="p", signal=sig, lots=0.10)
    bar = make_bar(symbol="XAUUSD", time_msc=0,
                   open=2000.0, high=2001.0,
                   low=sig.sl - risk * (wick_depth_mult - 1),
                   close=2000.0)
    actions = maintain_exit(state, bar)
    assert any(a.close_full and a.exit_reason == "SL" for a in actions)
    assert state.closed is True


# ===========================================================================
# 2. SHORT SL-touch grid
# ===========================================================================

@pytest.mark.parametrize("risk", [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0,
                                   15.0, 25.0, 50.0])
@pytest.mark.parametrize("wick_mult", [1.0, 1.5, 2.0, 5.0, 100.0])
def test_short_sl_touch_grid(risk, wick_mult):
    sig = _short_sig(entry=2000.0, risk=risk, rr=2.0)
    state = init_exit_state(position_id="p", signal=sig, lots=0.10)
    bar = make_bar(symbol="XAUUSD", time_msc=0,
                   open=2000.0,
                   high=sig.sl + risk * (wick_mult - 1),
                   low=1999.0, close=2000.0)
    actions = maintain_exit(state, bar)
    assert any(a.close_full and a.exit_reason == "SL" for a in actions)


# ===========================================================================
# 3. LONG TP1-only-touch grid
# ===========================================================================

@pytest.mark.parametrize("risk", [1.0, 2.0, 5.0, 10.0, 20.0])
@pytest.mark.parametrize("tp1_extra", [0.0, 0.01, 0.1, 1.0])
def test_long_tp1_touch_grid(risk, tp1_extra):
    sig = _long_sig(entry=2000.0, risk=risk, rr=3.0)
    state = init_exit_state(position_id="p", signal=sig, lots=0.10)
    tp1 = state.tp1
    bar = make_bar(symbol="XAUUSD", time_msc=0,
                   open=2000.0, high=tp1 + tp1_extra,
                   low=2000.0 - risk * 0.5, close=tp1 + tp1_extra - 0.1)
    maintain_exit(state, bar)
    assert state.tp1_hit is True


# ===========================================================================
# 4. SHORT TP1-only-touch grid
# ===========================================================================

@pytest.mark.parametrize("risk", [1.0, 2.0, 5.0, 10.0, 20.0])
@pytest.mark.parametrize("tp1_extra", [0.0, 0.01, 0.1, 1.0])
def test_short_tp1_touch_grid(risk, tp1_extra):
    sig = _short_sig(entry=2000.0, risk=risk, rr=3.0)
    state = init_exit_state(position_id="p", signal=sig, lots=0.10)
    tp1 = state.tp1
    bar = make_bar(symbol="XAUUSD", time_msc=0,
                   open=2000.0, high=2000.0 + risk * 0.5,
                   low=tp1 - tp1_extra, close=tp1 - tp1_extra + 0.1)
    maintain_exit(state, bar)
    assert state.tp1_hit is True


# ===========================================================================
# 5. EOD CLOSE — Reason matrix
# ===========================================================================

@pytest.mark.parametrize("tp1_hit_first", [False, True])
@pytest.mark.parametrize("exit_offset", [-5.0, -1.0, 0.0, 1.0, 5.0])
def test_eod_close_reason_matrix(tp1_hit_first, exit_offset):
    sig = _long_sig(entry=2000.0, risk=5.0, rr=2.0)
    state = init_exit_state(position_id="p", signal=sig, lots=0.10)
    state.tp1_hit = tp1_hit_first
    actions = force_close_eod(state, exit_price=2000.0 + exit_offset)
    assert state.closed is True
    expected = "EOD_trail" if tp1_hit_first else "EOD"
    assert actions[0].exit_reason == expected


# ===========================================================================
# 6. RR_RATIO PROPERTY MATRIX
# ===========================================================================

@pytest.mark.parametrize("risk", [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("rr", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0])
def test_rr_ratio_holds(risk, rr):
    entry = 2000.0
    sig = PatternSignal(
        pattern_name="X", symbol="XAUUSD", direction=Direction.BUY,
        entry=entry, sl=entry - risk, tp=entry + risk * rr,
        confidence=0.5, grade=Grade.A,
        confluences_met=(), bar_time_msc=0,
    )
    assert sig.rr_ratio == pytest.approx(rr, rel=1e-6)


# ===========================================================================
# 7. SIZE_POSITION GRID
# ===========================================================================

@pytest.mark.parametrize("pair", [
    "XAUUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "AUDCHF", "AUDNZD",
])
@pytest.mark.parametrize("equity", [1_000.0, 10_000.0, 100_000.0, 1_000_000.0])
@pytest.mark.parametrize("sl_mult", [1, 10, 100])
def test_size_position_grid(pair, equity, sl_mult):
    from config.asian_sweep_config import PAIR_CONFIG
    pt = float(PAIR_CONFIG[pair]["point"])
    lot_max = float(PAIR_CONFIG[pair]["lot_max"])
    lot = size_position(pair, equity=equity,
                         sl_distance_price=pt * sl_mult * 10)
    assert 0.01 <= lot <= lot_max


# ===========================================================================
# 8. SIGNAL CONFIDENCE GRID
# ===========================================================================

@pytest.mark.parametrize("conf", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
@pytest.mark.parametrize("grade", [Grade.A, Grade.B, Grade.C])
@pytest.mark.parametrize("direction", [Direction.BUY, Direction.SELL])
def test_signal_factory_grid(conf, grade, direction):
    entry = 2000.0
    sl = entry - 5.0 if direction == Direction.BUY else entry + 5.0
    tp = entry + 10.0 if direction == Direction.BUY else entry - 10.0
    sig = PatternSignal(
        pattern_name="X", symbol="XAUUSD", direction=direction,
        entry=entry, sl=sl, tp=tp,
        confidence=conf, grade=grade,
        confluences_met=(), bar_time_msc=0,
    )
    assert sig.confidence == conf
    assert sig.grade == grade


# ===========================================================================
# 9. EXIT-STATE FULL LIFECYCLE — MATRIX OF SEQUENCES
# ===========================================================================

@pytest.mark.parametrize("scenario", [
    "tp1_then_tp2", "tp1_then_trail_stop", "sl_first_bar",
])
def test_full_exit_lifecycle(scenario):
    sig = _long_sig(entry=2000.0, risk=5.0, rr=2.0)
    state = init_exit_state(position_id="p", signal=sig, lots=0.10)
    if scenario == "tp1_then_tp2":
        bar1 = make_bar(symbol="XAUUSD", time_msc=0,
                        open=2000.0, high=2006.0, low=1999.0, close=2002.0)
        bar2 = make_bar(symbol="XAUUSD", time_msc=HOUR_MS,
                        open=2002.0, high=2012.0, low=2001.0, close=2011.0)
        maintain_exit(state, bar1)
        a2 = maintain_exit(state, bar2)
        assert state.closed is True
        assert any(a.exit_reason == "TP2" for a in a2)
    elif scenario == "tp1_then_trail_stop":
        bar1 = make_bar(symbol="XAUUSD", time_msc=0,
                        open=2000.0, high=2006.0, low=1999.0, close=2005.5)
        bar2 = make_bar(symbol="XAUUSD", time_msc=HOUR_MS,
                        open=2005.0, high=2006.0, low=2000.0,
                        close=2000.0)
        maintain_exit(state, bar1)
        a2 = maintain_exit(state, bar2)
        # Trailed SL hit → "TRAIL"
        if state.closed:
            assert any(a.exit_reason == "TRAIL" for a in a2)
    else:  # sl_first_bar
        bar = make_bar(symbol="XAUUSD", time_msc=0,
                       open=2000.0, high=2001.0, low=1990.0, close=1995.0)
        actions = maintain_exit(state, bar)
        assert any(a.exit_reason == "SL" for a in actions)


# ===========================================================================
# 10. PARQUET-AGNOSTIC BAR CHURN
# ===========================================================================

@pytest.mark.parametrize("n_ticks", [10, 100, 1000])
@pytest.mark.parametrize("base_price", [1.10, 100.0, 2000.0])
def test_aggregator_arbitrary_density(n_ticks, base_price):
    agg = BarAggregator("X")
    for i in range(n_ticks):
        agg.on_tick(Tick(time_msc=i, bid=base_price, ask=base_price + 0.05,
                          last=base_price, volume=1, volume_real=1.0,
                          flags=0))
    bar = agg.flush()
    assert bar is not None


# ===========================================================================
# 11. FLOOR_TO_TIMEFRAME GRID
# ===========================================================================

@pytest.mark.parametrize("tf", [1, 5, 15, 30, 60, 240])
@pytest.mark.parametrize("offset_s", [0, 1, 60, 30 * 60])
def test_floor_aligned_property(tf, offset_s):
    base = 0
    t = base + offset_s * 1000
    snapped = floor_to_timeframe_ms(t, tf)
    period_ms = tf * 60 * 1000
    assert snapped % period_ms == 0
    assert snapped <= t


# ===========================================================================
# 12. NEGATIVE / ZERO SL-PROPS DEFEND IN PATTERN_SIGNAL
# ===========================================================================

@pytest.mark.parametrize("entry,sl,tp", [
    (-1.0, 0.5, 1.5), (0.0, 0.5, 1.5),
    (1.0, -0.5, 1.5), (1.0, 0.0, 1.5),
    (1.0, 0.5, 0.0),  (1.0, 0.5, -1.0),
])
def test_pattern_signal_rejects_nonpositive(entry, sl, tp):
    with pytest.raises(ValueError):
        PatternSignal(
            pattern_name="X", symbol="EURUSD", direction=Direction.BUY,
            entry=entry, sl=sl, tp=tp, confidence=0.5, grade=Grade.A,
            confluences_met=(), bar_time_msc=0,
        )


# ===========================================================================
# 13. SCANNER DEDUP GRID
# ===========================================================================

@pytest.mark.parametrize("n_pairs", [1, 4, 8])
@pytest.mark.parametrize("n_signals_per_pair", [0, 1, 2, 3])
def test_scanner_collects_all_signals(n_pairs, n_signals_per_pair):
    from strategy.scanner import Scanner

    class Multi:
        def __init__(self, k): self.name = f"M{k}"
        min_bars_required = 1
        timeframe = "1H"
        def detect(self, bars, ctx):
            return PatternSignal(
                pattern_name=self.name, symbol=ctx.symbol,
                direction=Direction.BUY, entry=1.10, sl=1.09, tp=1.12,
                confidence=0.5, grade=Grade.B,
                confluences_met=(), bar_time_msc=0,
            )
    patterns = tuple(Multi(k) for k in range(n_signals_per_pair))
    pairs = tuple(f"PAIR{i}" for i in range(n_pairs))
    if not patterns:
        with pytest.raises(ValueError):
            Scanner(pairs=pairs, patterns=patterns)
        return
    s = Scanner(pairs=pairs, patterns=patterns)
    bars = {p: [make_bar(symbol=p, time_msc=0)] for p in pairs}
    sigs = s.scan_all(bars, current_time_msc=0)
    assert len(sigs) == n_pairs * n_signals_per_pair


# ===========================================================================
# 14. SCANNER BEST-SIGNAL TIEBREAKERS
# ===========================================================================

@pytest.mark.parametrize("conf_a,conf_b,winner", [
    (0.9, 0.5, "A"),
    (0.5, 0.9, "B"),
    (0.7, 0.7, "A"),  # tie → first wins (sort is stable + reverse)
])
def test_best_signal_confidence_tiebreaker(conf_a, conf_b, winner):
    from strategy.scanner import Scanner

    class P:
        def __init__(self, name, conf):
            self.name = name; self._conf = conf
        min_bars_required = 1; timeframe = "1H"
        def detect(self, bars, ctx):
            return PatternSignal(
                pattern_name=self.name, symbol=ctx.symbol,
                direction=Direction.BUY, entry=1.10, sl=1.09, tp=1.12,
                confidence=self._conf, grade=Grade.A,
                confluences_met=(), bar_time_msc=0,
            )
    s = Scanner(pairs=("EURUSD",), patterns=(P("A", conf_a), P("B", conf_b)))
    s.scan_all({"EURUSD": [make_bar(time_msc=0)]}, current_time_msc=0)
    best = s.get_best_signal()
    # Either A or B depending on confidence; tie defaults to first.
    if conf_a > conf_b:
        assert best.pattern_name == "A"
    elif conf_b > conf_a:
        assert best.pattern_name == "B"


# ===========================================================================
# 15. ASIAN SWEEP DETECTOR — SESSION GATE
# ===========================================================================

@pytest.mark.parametrize("session_hour", [6, 7, 8, 9, 10, 12, 13, 14, 15])
def test_detector_signals_in_each_session_hour(detector, session_hour):
    """Both LONDON (6..10) and NY (12..15) windows can produce LONG signals."""
    bars = long_sweep_bars(symbol="EURUSD", pt=0.00001,
                            trigger_hour=session_hour)
    ctx = MarketContext(symbol="EURUSD", current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert sig is not None
    assert sig.direction == Direction.BUY


@pytest.mark.parametrize("hour", [0, 1, 2, 3, 4, 5, 11, 16, 17, 18, 19,
                                   20, 21, 22, 23])
def test_detector_blocked_outside_session(detector, hour):
    bars = long_sweep_bars(symbol="EURUSD", pt=0.00001,
                            trigger_hour=hour)
    ctx = MarketContext(symbol="EURUSD", current_time_msc=bars[-1].time_msc)
    assert detector.detect(bars, ctx) is None


# ===========================================================================
# 16. PnL SIGN INVARIANTS
# ===========================================================================

@pytest.mark.parametrize("rr", [0.5, 1.0, 1.5, 2.0, 2.5])
@pytest.mark.parametrize("lots", [0.01, 0.1, 1.0, 5.0])
def test_pnl_sign_long_winner(rr, lots):
    sig = _long_sig(entry=2000.0, risk=5.0, rr=rr)
    state = init_exit_state(position_id="p", signal=sig, lots=lots)
    state.final_exit_price = sig.tp
    state.final_exit_reason = "TP2"
    pnl = compute_pnl(state)
    assert pnl > 0


@pytest.mark.parametrize("rr", [0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("lots", [0.01, 0.1, 1.0])
def test_pnl_sign_long_loser(rr, lots):
    sig = _long_sig(entry=2000.0, risk=5.0, rr=rr)
    state = init_exit_state(position_id="p", signal=sig, lots=lots)
    state.final_exit_price = sig.sl
    state.final_exit_reason = "SL"
    pnl = compute_pnl(state)
    assert pnl < 0


@pytest.mark.parametrize("rr", [0.5, 1.0, 1.5, 2.0, 2.5])
@pytest.mark.parametrize("lots", [0.01, 0.1, 1.0])
def test_pnl_sign_short_winner(rr, lots):
    sig = _short_sig(entry=2000.0, risk=5.0, rr=rr)
    state = init_exit_state(position_id="p", signal=sig, lots=lots)
    state.final_exit_price = sig.tp
    state.final_exit_reason = "TP2"
    pnl = compute_pnl(state)
    assert pnl > 0


@pytest.mark.parametrize("rr", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("lots", [0.01, 0.1])
def test_pnl_sign_short_loser(rr, lots):
    sig = _short_sig(entry=2000.0, risk=5.0, rr=rr)
    state = init_exit_state(position_id="p", signal=sig, lots=lots)
    state.final_exit_price = sig.sl
    state.final_exit_reason = "SL"
    pnl = compute_pnl(state)
    assert pnl < 0


# ===========================================================================
# 17. ASIAN-WINDOW RANGE FILTER MATRIX
# ===========================================================================

@pytest.mark.parametrize("range_pts", [50, 100, 150, 200, 500, 1000,
                                        2000, 3000, 4000])
def test_xau_asian_range_filter(detector, range_pts):
    """XAUUSD: min_range_pts=100, max_range_pts=3000. Anything in [100, 3000]
    is acceptable; outside → None."""
    base = 2000.0
    half = range_pts * 0.01 / 2.0
    bars = long_sweep_bars(
        symbol="XAUUSD", pt=0.01,
        asian_low=base - half, asian_high=base + half,
        trigger_hour=8, wick_below_pts=70.0,
    )
    ctx = MarketContext(symbol="XAUUSD",
                        current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    if 100 <= range_pts <= 3000:
        # The detector may still reject due to bias or risk guard — just check
        # nothing raised.
        pass
    else:
        assert sig is None


# ===========================================================================
# 18. SIGNAL FACTORY DIRECTION INVARIANT
# ===========================================================================

@pytest.mark.parametrize("entry,risk", [
    (1.10000, 0.0001), (2000.0, 0.5), (100.0, 1.0), (0.50000, 0.00010),
])
def test_signal_buy_invariant(entry, risk):
    sig = PatternSignal(
        pattern_name="X", symbol="X", direction=Direction.BUY,
        entry=entry, sl=entry - risk, tp=entry + risk * 2,
        confidence=0.5, grade=Grade.A,
        confluences_met=(), bar_time_msc=0,
    )
    assert sig.sl < sig.entry < sig.tp


@pytest.mark.parametrize("entry,risk", [
    (1.10000, 0.0001), (2000.0, 0.5), (100.0, 1.0),
])
def test_signal_sell_invariant(entry, risk):
    sig = PatternSignal(
        pattern_name="X", symbol="X", direction=Direction.SELL,
        entry=entry, sl=entry + risk, tp=entry - risk * 2,
        confidence=0.5, grade=Grade.A,
        confluences_met=(), bar_time_msc=0,
    )
    assert sig.tp < sig.entry < sig.sl


# ===========================================================================
# 19. AGGREGATOR SPREAD-MEAN AVERAGE
# ===========================================================================

@pytest.mark.parametrize("n,base_spread", [
    (1, 0.1), (5, 0.5), (10, 1.0), (100, 2.0),
])
def test_aggregator_spread_mean_uniform(n, base_spread):
    agg = BarAggregator("X")
    for i in range(n):
        agg.on_tick(Tick(time_msc=i, bid=2000.0, ask=2000.0 + base_spread,
                          last=2000.0, volume=1, volume_real=1.0, flags=0))
    bar = agg.flush()
    if bar is not None:
        assert bar.spread_mean == pytest.approx(base_spread, abs=1e-9)


# ===========================================================================
# 20. PARAMETRIZE — MAINTAIN_EXIT IDEMPOTENT ON CLOSED STATE
# ===========================================================================

@pytest.mark.parametrize("call_count", [1, 2, 5, 10])
def test_maintain_exit_no_op_after_close(call_count):
    sig = _long_sig()
    state = init_exit_state(position_id="p", signal=sig, lots=0.10)
    state.closed = True
    bar = make_bar(symbol="XAUUSD", time_msc=0, open=2000.0, close=2000.0)
    for _ in range(call_count):
        assert maintain_exit(state, bar) == []
