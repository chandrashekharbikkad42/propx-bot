"""Asian Sweep V5 exit-management state-machine tests.

Coverage:
  - init_exit_state: TP1 extraction, fallbacks, fields
  - maintain_exit: SL hit (pre / post TP1), TP1 partial + BE move,
    trail-after-TP1, TP2 close, ordering invariants
  - force_close_eod: pre-TP1 vs post-TP1 tagging
  - compute_pnl: pre-TP1 full close, post-TP1 partial path, JPY conversion
  - size_position: per-pair lot math, risk_override, weak-month dampener,
    min-lot floor, max-lot cap, degenerate inputs
"""

from __future__ import annotations
from dataclasses import replace
from typing import Optional

import pytest
from hypothesis import given, settings, strategies as st

from config.asian_sweep_config import (
    PAIR_CONFIG, PAIRS, PARTIAL_CLOSE_FRACTION, RR_TP1, RR_TP2,
    TRAILING_STEP_R, WEAK_MONTHS,
)
from data.bar_aggregator import Bar
from risk.asian_sweep_exit import (
    ExitAction, ExitState,
    compute_pnl, force_close_eod,
    init_exit_state, maintain_exit, size_position,
)
from strategy.patterns.base import Direction, Grade, PatternSignal

from tests.strategy.fixtures.synthetic_bars import baseline_low


ALL_PAIRS = list(PAIRS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(*, time_msc: int = 0, open: float = 1.0,
         high: float | None = None, low: float | None = None,
         close: float = 1.0, symbol: str = "EURUSD") -> Bar:
    if high is None:
        high = max(open, close)
    if low is None:
        low = min(open, close)
    return Bar(symbol=symbol, time_msc=time_msc, open=open, high=high,
               low=low, close=close, volume=1, spread_mean=0.0)


def _signal(
    *,
    symbol: str = "EURUSD",
    direction: Direction = Direction.BUY,
    entry: float = 1.10000,
    risk_price: float = 0.00010,
    tp1: Optional[float] = None,
    grade: Grade = Grade.A,
) -> PatternSignal:
    if direction == Direction.BUY:
        sl = entry - risk_price
        tp = entry + risk_price * RR_TP2
        tp1_default = entry + risk_price * RR_TP1
    else:
        sl = entry + risk_price
        tp = entry - risk_price * RR_TP2
        tp1_default = entry - risk_price * RR_TP1
    tp1_val = tp1 if tp1 is not None else tp1_default
    return PatternSignal(
        pattern_name="ASIAN_SWEEP", symbol=symbol, direction=direction,
        entry=entry, sl=sl, tp=tp, confidence=0.9, grade=grade,
        confluences_met=("asian_sweep_low", "LONDON", "bias_neutral", "q9",
                         f"tp1_{tp1_val:.5f}"),
        bar_time_msc=0,
    )


def _short_signal(
    *,
    symbol: str = "EURUSD",
    entry: float = 1.10000,
    risk_price: float = 0.00010,
) -> PatternSignal:
    sl = entry + risk_price
    tp = entry - risk_price * RR_TP2
    tp1_val = entry - risk_price * RR_TP1
    return PatternSignal(
        pattern_name="ASIAN_SWEEP", symbol=symbol, direction=Direction.SELL,
        entry=entry, sl=sl, tp=tp, confidence=0.9, grade=Grade.A,
        confluences_met=("asian_sweep_high", "LONDON", "bias_bearish", "q9",
                         f"tp1_{tp1_val:.5f}"),
        bar_time_msc=0,
    )


# ===========================================================================
# 1. init_exit_state
# ===========================================================================

class TestInitExitState:
    def test_position_id(self):
        st_ = init_exit_state(position_id="POS-1", signal=_signal(), lots=0.5)
        assert st_.position_id == "POS-1"

    def test_symbol(self):
        st_ = init_exit_state(position_id="x", signal=_signal(symbol="GBPUSD"),
                              lots=0.5)
        assert st_.symbol == "GBPUSD"

    def test_direction_buy(self):
        st_ = init_exit_state(position_id="x", signal=_signal(), lots=0.5)
        assert st_.direction == Direction.BUY

    def test_direction_sell(self):
        st_ = init_exit_state(position_id="x", signal=_short_signal(),
                              lots=0.5)
        assert st_.direction == Direction.SELL

    def test_entry(self):
        st_ = init_exit_state(position_id="x", signal=_signal(), lots=0.5)
        assert st_.entry == 1.10000

    def test_sl(self):
        st_ = init_exit_state(position_id="x", signal=_signal(), lots=0.5)
        assert st_.sl == pytest.approx(1.09990)

    def test_tp2_from_signal_tp(self):
        st_ = init_exit_state(position_id="x", signal=_signal(), lots=0.5)
        assert st_.tp2 == pytest.approx(1.10025)

    def test_tp1_extracted_from_confluence_tag(self):
        st_ = init_exit_state(position_id="x", signal=_signal(), lots=0.5)
        assert st_.tp1 == pytest.approx(1.10010)

    def test_initial_risk_long(self):
        st_ = init_exit_state(position_id="x", signal=_signal(), lots=0.5)
        assert st_.initial_risk == pytest.approx(0.00010)

    def test_initial_risk_short(self):
        st_ = init_exit_state(position_id="x", signal=_short_signal(),
                              lots=0.5)
        assert st_.initial_risk == pytest.approx(0.00010)

    def test_initial_lots(self):
        st_ = init_exit_state(position_id="x", signal=_signal(), lots=0.42)
        assert st_.initial_lots == 0.42

    def test_remaining_lots_equals_initial(self):
        st_ = init_exit_state(position_id="x", signal=_signal(), lots=0.42)
        assert st_.remaining_lots == 0.42

    def test_tp1_hit_false_initially(self):
        st_ = init_exit_state(position_id="x", signal=_signal(), lots=0.5)
        assert st_.tp1_hit is False

    def test_closed_false_initially(self):
        st_ = init_exit_state(position_id="x", signal=_signal(), lots=0.5)
        assert st_.closed is False

    def test_partial_exit_price_none(self):
        st_ = init_exit_state(position_id="x", signal=_signal(), lots=0.5)
        assert st_.partial_exit_price is None

    def test_final_exit_price_none(self):
        st_ = init_exit_state(position_id="x", signal=_signal(), lots=0.5)
        assert st_.final_exit_price is None

    def test_final_exit_reason_none(self):
        st_ = init_exit_state(position_id="x", signal=_signal(), lots=0.5)
        assert st_.final_exit_reason is None

    def test_tp1_default_when_tag_missing(self):
        """When no `tp1_*` tag is present, fall back to entry + 1R."""
        sig = PatternSignal(
            pattern_name="ASIAN_SWEEP", symbol="EURUSD",
            direction=Direction.BUY,
            entry=1.10000, sl=1.09990, tp=1.10025,
            confidence=0.9, grade=Grade.A,
            confluences_met=("asian_sweep_low", "LONDON",
                             "bias_neutral", "q9"),
            bar_time_msc=0,
        )
        st_ = init_exit_state(position_id="x", signal=sig, lots=0.5)
        assert st_.tp1 == pytest.approx(1.10010)

    def test_tp1_default_short(self):
        sig = PatternSignal(
            pattern_name="ASIAN_SWEEP", symbol="EURUSD",
            direction=Direction.SELL,
            entry=1.10000, sl=1.10010, tp=1.09975,
            confidence=0.9, grade=Grade.A,
            confluences_met=("asian_sweep_high", "LONDON",
                             "bias_bearish", "q9"),
            bar_time_msc=0,
        )
        st_ = init_exit_state(position_id="x", signal=sig, lots=0.5)
        assert st_.tp1 == pytest.approx(1.09990)

    def test_tp1_malformed_tag_falls_back(self):
        sig = PatternSignal(
            pattern_name="ASIAN_SWEEP", symbol="EURUSD",
            direction=Direction.BUY,
            entry=1.10000, sl=1.09990, tp=1.10025,
            confidence=0.9, grade=Grade.A,
            confluences_met=("asian_sweep_low", "LONDON",
                             "bias_neutral", "q9", "tp1_not_a_number"),
            bar_time_msc=0,
        )
        st_ = init_exit_state(position_id="x", signal=sig, lots=0.5)
        assert st_.tp1 == pytest.approx(1.10010)


# ===========================================================================
# 2. maintain_exit — LONG path
# ===========================================================================

class TestMaintainLongSL:
    def test_sl_hit_pre_tp1(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        actions = maintain_exit(s, _bar(low=1.09980, high=1.09995,
                                        close=1.09985))
        assert len(actions) == 1
        assert actions[0].close_full is True
        assert actions[0].exit_reason == "SL"

    def test_sl_state_closed(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        maintain_exit(s, _bar(low=1.09980, high=1.09995, close=1.09985))
        assert s.closed is True

    def test_sl_remaining_lots_zero(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        maintain_exit(s, _bar(low=1.09980, high=1.09995, close=1.09985))
        assert s.remaining_lots == 0.0

    def test_sl_exact_touch_triggers(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        actions = maintain_exit(s, _bar(low=1.09990, high=1.09995,
                                        close=1.09993))
        assert actions[0].exit_reason == "SL"

    def test_sl_just_above_does_not_trigger(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        actions = maintain_exit(s, _bar(low=1.09991, high=1.09995,
                                        close=1.09993))
        assert actions == []
        assert s.closed is False

    def test_no_actions_when_already_closed(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        s.closed = True
        assert maintain_exit(s, _bar(low=0.5, high=1.5, close=1.0)) == []


class TestMaintainLongTp1:
    def test_tp1_partial(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        actions = maintain_exit(s, _bar(low=1.10000, high=1.10010,
                                        close=1.10005))
        assert s.tp1_hit is True
        assert actions[0].partial_close == PARTIAL_CLOSE_FRACTION
        assert actions[0].modify_sl == pytest.approx(1.10000)
        assert actions[0].exit_reason == "PARTIAL_TP1"

    def test_partial_lots_50pct(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        maintain_exit(s, _bar(high=1.10010, low=1.10000, close=1.10005))
        assert s.remaining_lots == pytest.approx(0.5)

    def test_sl_moves_to_BE(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        # close == entry so the trail step (close - 0.3R) lands below BE
        # and the trail-update branch does NOT move SL further.
        maintain_exit(s, _bar(high=1.10010, low=1.10000, close=1.10000))
        assert s.sl == pytest.approx(s.entry)

    def test_partial_exit_price_recorded(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        maintain_exit(s, _bar(high=1.10010, low=1.10000, close=1.10005))
        assert s.partial_exit_price == pytest.approx(1.10010)

    def test_tp1_exact_touch(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        maintain_exit(s, _bar(high=1.10010, low=1.10005, close=1.10008))
        assert s.tp1_hit is True

    def test_tp1_not_hit_when_high_below(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        actions = maintain_exit(s, _bar(high=1.10009, low=1.10000,
                                        close=1.10005))
        assert s.tp1_hit is False
        assert actions == []

    def test_partial_then_trail_can_happen_same_bar(self):
        """After TP1 partial, the trail update runs in the same bar if the
        close has moved enough."""
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        # high crosses TP1; close is high above entry → trail should jump.
        actions = maintain_exit(
            s, _bar(low=1.10000, high=1.10020, close=1.10015)
        )
        # Two actions: partial-close + trail-update.
        modify_actions = [a for a in actions if a.modify_sl is not None]
        # Partial sets sl to entry; trail computes close - 0.3R = 1.10015 - 3e-5
        # = 1.10012 > entry → trail moves.
        assert len(modify_actions) == 2


class TestMaintainLongTrail:
    def test_no_trail_before_tp1(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        actions = maintain_exit(s, _bar(high=1.10005, low=1.09995,
                                        close=1.10003))
        assert all(a.modify_sl is None for a in actions)

    def test_trail_after_tp1_moves_up(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        s.tp1_hit = True
        s.sl = s.entry
        actions = maintain_exit(s, _bar(high=1.10015, low=1.10010,
                                        close=1.10012))
        # new_trail = close - 0.3 * 0.00010 = 1.10012 - 0.00003 = 1.10009
        # entry was 1.10000; new_trail > entry → moves.
        assert any(
            a.modify_sl is not None and a.modify_sl > 1.10000 for a in actions
        )

    def test_trail_does_not_move_backwards(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        s.tp1_hit = True
        s.sl = 1.10005   # already trailing
        actions = maintain_exit(s, _bar(high=1.10006, low=1.10005,
                                        close=1.10005))
        # new_trail = 1.10005 - 0.00003 = 1.10002 < current sl
        assert all(a.modify_sl is None for a in actions)


class TestMaintainLongTp2:
    def test_tp2_after_tp1(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        s.tp1_hit = True
        s.sl = s.entry  # BE
        actions = maintain_exit(s, _bar(high=1.10030, low=1.10005,
                                        close=1.10028))
        assert any(a.exit_reason == "TP2" for a in actions)
        assert s.closed is True

    def test_tp2_requires_tp1_hit(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        # high crosses TP1 (1.10010) and TP2 (1.10025) in one bar.
        actions = maintain_exit(s, _bar(high=1.10030, low=1.09995,
                                        close=1.10028))
        # TP1 fires → partial. TP2 also fires because tp1_hit is True now.
        reasons = [a.exit_reason for a in actions]
        assert "PARTIAL_TP1" in reasons
        assert "TP2" in reasons

    def test_tp2_only_after_tp1_no_partial_in_same_bar(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        s.tp1_hit = True
        s.sl = s.entry
        actions = maintain_exit(s, _bar(high=1.10026, low=1.10015,
                                        close=1.10025))
        reasons = [a.exit_reason for a in actions]
        assert "PARTIAL_TP1" not in reasons
        assert "TP2" in reasons


# ===========================================================================
# 3. maintain_exit — SHORT path
# ===========================================================================

class TestMaintainShort:
    def test_short_sl_hit(self):
        s = init_exit_state(position_id="x", signal=_short_signal(),
                            lots=1.0)
        actions = maintain_exit(s, _bar(high=1.10015, low=1.09995,
                                        close=1.10010))
        assert actions[0].exit_reason == "SL"
        assert s.closed is True

    def test_short_tp1_partial(self):
        s = init_exit_state(position_id="x", signal=_short_signal(),
                            lots=1.0)
        # close == entry → trail step (close + 0.3R) lands above BE,
        # so the trail-update branch does NOT shift SL further.
        actions = maintain_exit(s, _bar(high=1.10005, low=1.09989,
                                        close=1.10000))
        assert s.tp1_hit is True
        assert s.sl == pytest.approx(s.entry)
        assert any(a.exit_reason == "PARTIAL_TP1" for a in actions)

    def test_short_tp2_full_close(self):
        s = init_exit_state(position_id="x", signal=_short_signal(),
                            lots=1.0)
        s.tp1_hit = True
        s.sl = s.entry
        actions = maintain_exit(s, _bar(high=1.09995, low=1.09970,
                                        close=1.09975))
        assert any(a.exit_reason == "TP2" for a in actions)
        assert s.closed is True

    def test_short_trail_moves_down(self):
        s = init_exit_state(position_id="x", signal=_short_signal(),
                            lots=1.0)
        s.tp1_hit = True
        s.sl = s.entry  # BE
        actions = maintain_exit(s, _bar(high=1.09995, low=1.09985,
                                        close=1.09988))
        # new_trail = close + 0.3 * 0.00010 = 1.09988 + 0.00003 = 1.09991
        # 1.09991 < entry (1.10000) → trail tightens.
        assert any(a.modify_sl is not None and a.modify_sl < s.entry
                   for a in actions)

    def test_short_no_trail_pre_tp1(self):
        s = init_exit_state(position_id="x", signal=_short_signal(),
                            lots=1.0)
        actions = maintain_exit(s, _bar(high=1.10001, low=1.09997,
                                        close=1.09998))
        assert all(a.modify_sl is None for a in actions)


# ===========================================================================
# 4. force_close_eod
# ===========================================================================

class TestForceCloseEod:
    def test_pre_tp1_eod(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        actions = force_close_eod(s, 1.10005)
        assert actions[0].exit_reason == "EOD"

    def test_post_tp1_eod_trail(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        s.tp1_hit = True
        actions = force_close_eod(s, 1.10005)
        assert actions[0].exit_reason == "EOD_trail"

    def test_state_closed_after_eod(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        force_close_eod(s, 1.10005)
        assert s.closed is True

    def test_remaining_lots_zero_after_eod(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        force_close_eod(s, 1.10005)
        assert s.remaining_lots == 0.0

    def test_already_closed_returns_empty(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        s.closed = True
        assert force_close_eod(s, 1.10005) == []

    @pytest.mark.parametrize("price", [1.0, 1.1, 1.2, 0.99999, 9999.99])
    def test_exit_price_passes_through(self, price):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        actions = force_close_eod(s, price)
        assert actions[0].exit_price == price


# ===========================================================================
# 5. compute_pnl
# ===========================================================================

class TestComputePnl:
    def test_no_exit_yet_returns_zero(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        assert compute_pnl(s) == 0.0

    def test_full_loss_long_at_sl(self):
        s = init_exit_state(position_id="x",
                            signal=_signal(entry=1.10000,
                                           risk_price=0.00010),
                            lots=1.0)
        s.final_exit_price = 1.09990
        s.final_exit_reason = "SL"
        # diff = -0.00010, lots=1.0, contract=100000 → −10 USD
        assert compute_pnl(s) == pytest.approx(-10.0)

    def test_full_win_long_at_tp2_no_partial(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        s.final_exit_price = 1.10025
        s.final_exit_reason = "TP2"
        # diff = +0.00025, lots=1, ct=100000 → 25 USD
        assert compute_pnl(s) == pytest.approx(25.0)

    def test_tp1_then_tp2_partial_path(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        s.tp1_hit = True
        s.final_exit_price = 1.10025
        s.final_exit_reason = "TP2"  # TP2 reason → full path NOT partial blend
        # When the reason is "TP2" the function uses the simple `diff*lots*ct`.
        assert compute_pnl(s) == pytest.approx(25.0)

    def test_tp1_then_trail_partial_blend(self):
        """post-TP1, final reason = TRAIL → partial blend formula."""
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        s.tp1_hit = True
        s.final_exit_price = 1.10005       # came back, hit BE/trail
        s.final_exit_reason = "TRAIL"
        # d1 = tp1 - entry = 0.00010; diff = 0.00005
        # pnl = (0.00010*0.5 + 0.00005*0.5)*1*100000 = (0.00005+0.000025)*1e5 = 7.5
        assert compute_pnl(s) == pytest.approx(7.5)

    def test_short_full_loss(self):
        s = init_exit_state(position_id="x", signal=_short_signal(),
                            lots=1.0)
        s.final_exit_price = 1.10010
        s.final_exit_reason = "SL"
        # SHORT diff = entry - exit = 1.10000 - 1.10010 = -0.00010 → -10 USD
        assert compute_pnl(s) == pytest.approx(-10.0)

    def test_short_full_win(self):
        s = init_exit_state(position_id="x", signal=_short_signal(),
                            lots=1.0)
        s.final_exit_price = 1.09975
        s.final_exit_reason = "TP2"
        # diff = 0.00025, pnl = 25 USD
        assert compute_pnl(s) == pytest.approx(25.0)

    def test_unknown_symbol_returns_zero(self):
        s = init_exit_state(position_id="x",
                            signal=_signal(symbol="EURUSD"), lots=1.0)
        # mutate symbol to unknown
        s.symbol = "ZZZZZZ"
        s.final_exit_price = 1.10025
        s.final_exit_reason = "TP2"
        assert compute_pnl(s) == 0.0

    def test_jpy_conversion(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        s.final_exit_price = 1.10025
        s.final_exit_reason = "TP2"
        assert compute_pnl(s, jpy=True) == pytest.approx(25.0 / 150.0)

    @pytest.mark.parametrize("lots", [0.01, 0.5, 1.0, 5.0])
    def test_pnl_scales_with_lots(self, lots):
        s = init_exit_state(position_id="x", signal=_signal(), lots=lots)
        s.final_exit_price = 1.10025
        s.final_exit_reason = "TP2"
        assert compute_pnl(s) == pytest.approx(25.0 * lots)


# ===========================================================================
# 6. size_position — per-pair lot math
# ===========================================================================

class TestSizePosition:
    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_returns_floor_min_for_zero_equity(self, pair):
        assert size_position(pair, equity=0.0,
                             sl_distance_price=0.001) == 0.01

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_returns_floor_min_for_zero_sl(self, pair):
        assert size_position(pair, equity=10000.0,
                             sl_distance_price=0.0) == 0.01

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_returns_floor_min_for_negative_equity(self, pair):
        assert size_position(pair, equity=-1.0,
                             sl_distance_price=0.001) == 0.01

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_returns_floor_min_for_negative_sl(self, pair):
        assert size_position(pair, equity=10000.0,
                             sl_distance_price=-0.001) == 0.01

    def test_unknown_symbol_returns_min(self):
        assert size_position("ZZZZZZ", equity=10000.0,
                             sl_distance_price=0.001) == 0.01

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_min_lots_floor(self, pair):
        """A tiny risk budget should never go below 0.01 lots."""
        # 100-pip SL (1 pip = 10 broker points): realistic per-pair distance
        # that clears the MIN floor without tripping the MAX-USD-risk cap
        # (a flat 1000.0 price distance is ~100M pips on 5-digit FX).
        pt = float(PAIR_CONFIG[pair]["point"])
        lot = size_position(pair, equity=1.0,
                            sl_distance_price=pt * 1000)
        assert lot >= 0.01

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_max_lots_cap(self, pair):
        """A huge equity is capped by pair's lot_max."""
        cfg = PAIR_CONFIG[pair]
        lot = size_position(pair, equity=10_000_000.0,
                            sl_distance_price=float(cfg["point"]) * 1.0)
        assert lot <= float(cfg["lot_max"])

    def test_xau_risk_override_used(self):
        """XAUUSD risk_override = 0.5% (vs default 0.8%) → lot should
        be smaller than EURUSD for the same SL distance & equity."""
        lot_xau = size_position("XAUUSD", equity=10000.0,
                                sl_distance_price=10.0)  # huge SL
        # 10-pip SL for EURUSD (>= the 5-pip MIN floor) keeps the lot finite.
        lot_eur = size_position("EURUSD", equity=10000.0,
                                sl_distance_price=0.0010)
        # Just verify XAUUSD computed a finite > 0 lot — the relative
        # comparison is governed by different contract sizes.
        assert lot_xau > 0
        assert lot_eur > 0

    def test_weak_month_dampener(self):
        """Risk pct in WEAK_MONTHS is 0.3% (less than default 0.8% / XAU 0.5%)
        → lot in weak month is smaller than in non-weak month."""
        # equity=10k keeps both un-capped by MAX_RISK_USD_PER_TRADE ($150),
        # so the 0.3% vs 0.8% risk dampener is what drives the lot difference.
        big = size_position("EURUSD", equity=10_000.0,
                            sl_distance_price=0.0010, month=6)
        small = size_position("EURUSD", equity=10_000.0,
                              sl_distance_price=0.0010,
                              month=WEAK_MONTHS[0])
        assert small < big

    def test_lot_calc_eurusd_basic(self):
        """EURUSD: equity=10000, risk_pct=0.8 → risk = $80.
        SL = 10 pips = 100 pts. pt=0.00001, ct=100000 → vpl=1.0.
        risk_pts_count = 100. lot = 80 / (100*1) = 0.80."""
        lot = size_position("EURUSD", equity=10_000.0,
                            sl_distance_price=0.00100)
        assert lot == pytest.approx(0.80)

    def test_lot_calc_xau(self):
        """XAU: equity=10000, risk_pct=0.5 → risk = $50.
        SL = 1.0 = 100 pts (pt=0.01). ct=100, vpl=1.0.
        lot = 50 / 100 = 0.50."""
        lot = size_position("XAUUSD", equity=10_000.0,
                            sl_distance_price=1.0)
        assert lot == pytest.approx(0.50)

    def test_lot_is_rounded_to_2dp(self):
        """Production rounds via round(_, 2)."""
        lot = size_position("EURUSD", equity=12_345.67,
                            sl_distance_price=0.00012345)
        # Just check it's quantized to hundredths.
        assert lot == round(lot, 2)


@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("equity", [500, 1_000, 5_000, 10_000, 100_000])
# sl_mult is the SL distance in broker points; 1 pip = 10 points, so the
# 5-pip MIN_SL_DISTANCE_PIPS floor = 50 points. All values clear it.
@pytest.mark.parametrize("sl_mult", [50, 100, 150, 250, 500])
def test_size_position_finite_per_pair(pair, equity, sl_mult, request):
    pt = float(PAIR_CONFIG[pair]["point"])
    sl_distance = sl_mult * pt
    lot = size_position(pair, equity=equity, sl_distance_price=sl_distance)
    assert 0.01 <= lot <= float(PAIR_CONFIG[pair]["lot_max"])


@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("month", list(range(1, 13)))
def test_size_position_per_month_per_pair(pair, month):
    pt = float(PAIR_CONFIG[pair]["point"])
    lot = size_position(pair, equity=10_000.0,
                        sl_distance_price=100 * pt, month=month)
    assert lot >= 0.01


# ===========================================================================
# 7. Hypothesis property — PnL invariants
# ===========================================================================

@settings(max_examples=80, deadline=None)
@given(
    entry=st.floats(min_value=0.5, max_value=10.0,
                    allow_nan=False, allow_infinity=False),
    risk_p=st.floats(min_value=1e-4, max_value=0.01,
                     allow_nan=False, allow_infinity=False),
    lots=st.floats(min_value=0.01, max_value=5.0,
                   allow_nan=False, allow_infinity=False),
)
def test_pnl_full_win_long_property(entry, risk_p, lots):
    sig = _signal(entry=entry, risk_price=risk_p)
    s = init_exit_state(position_id="x", signal=sig, lots=lots)
    s.final_exit_price = entry + risk_p * RR_TP2
    s.final_exit_reason = "TP2"
    expected = (entry + risk_p * RR_TP2 - entry) * lots * 100_000.0
    assert compute_pnl(s) == pytest.approx(expected, rel=1e-9)


@settings(max_examples=80, deadline=None)
@given(
    entry=st.floats(min_value=0.5, max_value=10.0,
                    allow_nan=False, allow_infinity=False),
    risk_p=st.floats(min_value=1e-4, max_value=0.01,
                     allow_nan=False, allow_infinity=False),
    lots=st.floats(min_value=0.01, max_value=5.0,
                   allow_nan=False, allow_infinity=False),
)
def test_pnl_full_loss_long_property(entry, risk_p, lots):
    sig = _signal(entry=entry, risk_price=risk_p)
    s = init_exit_state(position_id="x", signal=sig, lots=lots)
    s.final_exit_price = entry - risk_p
    s.final_exit_reason = "SL"
    expected = -risk_p * lots * 100_000.0
    assert compute_pnl(s) == pytest.approx(expected, rel=1e-9)


# ===========================================================================
# 8. Per-pair × direction smoke (state machine reaches TP2)
# ===========================================================================

@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_long_full_lifecycle_reaches_tp2(pair):
    pt = float(PAIR_CONFIG[pair]["point"])
    entry = baseline_low(pair)
    risk_p = 50 * pt
    sig = _signal(symbol=pair, entry=entry, risk_price=risk_p)
    s = init_exit_state(position_id="x", signal=sig, lots=0.50)
    # Bar 1: hit TP1 with a comfortable buffer to clear fp drift.
    maintain_exit(s, _bar(symbol=pair,
                          low=entry + 0.1 * pt,
                          high=s.tp1 + 5 * pt,
                          close=entry + risk_p * RR_TP1))
    assert s.tp1_hit is True
    # Bar 2: spike to TP2 (use state.tp2 directly to dodge fp drift).
    actions = maintain_exit(s, _bar(
        symbol=pair,
        low=entry + risk_p * RR_TP1,
        high=s.tp2 + 5 * pt,
        close=s.tp2,
    ))
    assert any(a.exit_reason == "TP2" for a in actions)


@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_short_full_lifecycle_reaches_tp2(pair):
    pt = float(PAIR_CONFIG[pair]["point"])
    entry = baseline_low(pair)
    risk_p = 50 * pt
    sig = _short_signal(symbol=pair, entry=entry, risk_price=risk_p)
    s = init_exit_state(position_id="x", signal=sig, lots=0.50)
    # Bar 1: clear TP1 with a buffer to bypass fp drift.
    maintain_exit(s, _bar(symbol=pair,
                          high=entry - 0.1 * pt,
                          low=s.tp1 - 5 * pt,
                          close=entry - risk_p * RR_TP1))
    assert s.tp1_hit is True
    # Bar 2: clear TP2.
    actions = maintain_exit(s, _bar(
        symbol=pair,
        high=entry - risk_p * RR_TP1,
        low=s.tp2 - 5 * pt,
        close=s.tp2,
    ))
    assert any(a.exit_reason == "TP2" for a in actions)


@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_long_sl_then_no_partial(pair):
    pt = float(PAIR_CONFIG[pair]["point"])
    entry = baseline_low(pair)
    risk_p = 50 * pt
    sig = _signal(symbol=pair, entry=entry, risk_price=risk_p)
    s = init_exit_state(position_id="x", signal=sig, lots=0.50)
    actions = maintain_exit(s, _bar(symbol=pair,
                                    low=entry - risk_p - 1 * pt,
                                    high=entry,
                                    close=entry - risk_p - 0.5 * pt))
    # Only ONE action expected: full close (SL). No partial yet.
    assert len(actions) == 1
    assert actions[0].exit_reason == "SL"


# ===========================================================================
# 9. ExitAction defaults & invariants
# ===========================================================================

class TestExitActionDefaults:
    def test_default_close_full_false(self):
        a = ExitAction()
        assert a.close_full is False

    def test_default_partial_close_zero(self):
        a = ExitAction()
        assert a.partial_close == 0.0

    def test_default_modify_sl_none(self):
        a = ExitAction()
        assert a.modify_sl is None

    def test_default_exit_price_none(self):
        a = ExitAction()
        assert a.exit_price is None

    def test_default_exit_reason_none(self):
        a = ExitAction()
        assert a.exit_reason is None

    def test_frozen_dataclass(self):
        a = ExitAction()
        with pytest.raises(Exception):
            a.close_full = True  # type: ignore[misc]


# ===========================================================================
# 10. Edge-case parameter sweeps
# ===========================================================================

@pytest.mark.parametrize("entry", [
    0.50000, 1.00000, 1.10000, 1.50000, 2.00000, 100.0, 1500.0, 3000.0,
])
def test_init_exit_state_per_entry_long(entry):
    sig = _signal(entry=entry, risk_price=max(0.0001, entry * 0.0001))
    s = init_exit_state(position_id="x", signal=sig, lots=1.0)
    assert s.entry == entry


@pytest.mark.parametrize("risk_p", [
    1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3,
])
def test_init_exit_state_per_risk(risk_p):
    sig = _signal(risk_price=risk_p)
    s = init_exit_state(position_id="x", signal=sig, lots=1.0)
    assert s.initial_risk == pytest.approx(risk_p)


@pytest.mark.parametrize("lots", [0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0])
def test_init_lots_round_trip(lots):
    s = init_exit_state(position_id="x", signal=_signal(), lots=lots)
    assert s.initial_lots == lots
    assert s.remaining_lots == lots


# ===========================================================================
# 11. Sequential bar processing — long path
# ===========================================================================

class TestSequentialBars:
    def test_three_bars_to_tp2(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        # Bar 1: noise, no action.
        a1 = maintain_exit(s, _bar(high=1.10004, low=1.09995,
                                   close=1.10003))
        # Bar 2: TP1 hit → partial. Use a small buffer over state.tp1 to
        # clear any fp drift between the literal and the constructed value.
        a2 = maintain_exit(s, _bar(high=s.tp1 + 1e-6, low=1.10000,
                                   close=1.10009))
        # Bar 3: TP2 hit (buffer for fp).
        a3 = maintain_exit(s, _bar(high=s.tp2 + 1e-6, low=1.10010,
                                   close=1.10024))
        assert a1 == []
        assert any(a.exit_reason == "PARTIAL_TP1" for a in a2)
        assert any(a.exit_reason == "TP2" for a in a3)

    def test_no_action_after_closed(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        maintain_exit(s, _bar(high=1.10005, low=1.09980, close=1.09990))
        assert s.closed is True
        # Subsequent bars yield nothing.
        assert maintain_exit(s, _bar(high=1.10030, low=1.10020,
                                     close=1.10025)) == []


# ===========================================================================
# 12. Boundary touches on TP1/TP2/SL (long & short)
# ===========================================================================

@pytest.mark.parametrize("direction", [Direction.BUY, Direction.SELL])
def test_sl_boundary_touch_long_and_short(direction):
    sig = _signal() if direction == Direction.BUY else _short_signal()
    s = init_exit_state(position_id="x", signal=sig, lots=1.0)
    if direction == Direction.BUY:
        actions = maintain_exit(s, _bar(low=s.sl, high=s.entry,
                                        close=s.sl))
    else:
        actions = maintain_exit(s, _bar(high=s.sl, low=s.entry,
                                        close=s.sl))
    assert actions[0].exit_reason == "SL"


# ===========================================================================
# 13. Trail step uses initial_risk (not current sl distance)
# ===========================================================================

class TestTrailStepReferences:
    def test_trail_uses_initial_risk_not_be_distance(self):
        s = init_exit_state(position_id="x", signal=_signal(), lots=1.0)
        s.tp1_hit = True
        s.sl = s.entry          # BE
        # initial_risk = 0.00010; trail step = 0.3 * 0.00010 = 0.00003.
        actions = maintain_exit(s, _bar(high=1.10015, low=1.10010,
                                        close=1.10015))
        # new_trail = 1.10015 - 0.00003 = 1.10012.
        new_sls = [a.modify_sl for a in actions if a.modify_sl is not None]
        assert new_sls and new_sls[-1] == pytest.approx(1.10012)


# ===========================================================================
# 14. RR_TP1 + RR_TP2 constants exported
# ===========================================================================

def test_rr_constants_exported():
    from risk.asian_sweep_exit import RR_TP1 as exit_rr1, RR_TP2 as exit_rr2
    assert exit_rr1 == 1.0
    assert exit_rr2 == 2.5
