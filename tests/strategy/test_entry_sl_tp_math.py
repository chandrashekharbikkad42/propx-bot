"""Entry / SL / TP1 / TP2 math + risk guard for AsianSweepDetector.

V5 formulas:
  LONG  : entry = AL + spread_pts * pt
          SL    = trigger.low - sl_pts * pt
          TP1   = entry + risk * RR_TP1
          TP2   = entry + risk * RR_TP2

  SHORT : entry = AH - spread_pts * pt
          SL    = trigger.high + sl_pts * pt
          TP1   = entry - risk * RR_TP1
          TP2   = entry - risk * RR_TP2

  Guard : if |entry-SL| < 3 * pt → reject (degenerate).

Plus hypothesis property-based tests on the helper formulas.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from config.asian_sweep_config import (
    PAIR_CONFIG, PAIRS, RR_TP1, RR_TP2,
)
from strategy.patterns.asian_sweep import AsianSweepDetector
from strategy.patterns.base import Direction, MarketContext

from tests.strategy.fixtures.synthetic_bars import (
    baseline_low, build_scenario, long_sweep_bars, short_sweep_bars,
)


ALL_PAIRS = list(PAIRS)


def _baseline_low(pair: str) -> float:
    return baseline_low(pair)


def _baseline_range_pts(pair: str) -> float:
    cfg = PAIR_CONFIG[pair]
    return (float(cfg["min_range_pts"])
            + float(cfg["max_range_pts"])) / 2.0


def _extract_tp1(sig) -> float:
    for tag in sig.confluences_met:
        if tag.startswith("tp1_"):
            return float(tag[len("tp1_"):])
    raise AssertionError("tp1 tag missing")


# ---------------------------------------------------------------------------
# 1. LONG entry = AL + spread * pt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestLongEntryFormula:
    def test_long_entry_above_al(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al,
            asian_high=al + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.entry > al

    def test_long_entry_equals_al_plus_spread(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        sp = float(PAIR_CONFIG[pair]["spread_pts"])
        al = _baseline_low(pair)
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al,
            asian_high=al + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.entry == pytest.approx(al + sp * pt, rel=1e-9, abs=pt / 2)


# ---------------------------------------------------------------------------
# 2. SHORT entry = AH - spread * pt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestShortEntryFormula:
    def test_short_entry_below_ah(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(
            symbol=pair, pt=pt, asian_low=al, asian_high=ah,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.entry < ah

    def test_short_entry_equals_ah_minus_spread(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        sp = float(PAIR_CONFIG[pair]["spread_pts"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(
            symbol=pair, pt=pt, asian_low=al, asian_high=ah,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.entry == pytest.approx(ah - sp * pt, rel=1e-9, abs=pt / 2)


# ---------------------------------------------------------------------------
# 3. LONG SL = trigger.low - sl_pts * pt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestLongSlFormula:
    def test_sl_below_entry(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al,
            asian_high=al + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.sl < sig.entry

    def test_sl_equals_low_minus_buffer(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        sl_pts = float(PAIR_CONFIG[pair]["sl_pts"])
        al = _baseline_low(pair)
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al,
            asian_high=al + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        # Trigger bar low: set by long_sweep_bars helper.
        trigger_low = bars[-1].low
        assert sig.sl == pytest.approx(trigger_low - sl_pts * pt,
                                       rel=1e-9, abs=pt / 2)


# ---------------------------------------------------------------------------
# 4. SHORT SL = trigger.high + sl_pts * pt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestShortSlFormula:
    def test_sl_above_entry(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(symbol=pair, pt=pt,
                                asian_low=al, asian_high=ah)
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.sl > sig.entry

    def test_sl_equals_high_plus_buffer(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        sl_pts = float(PAIR_CONFIG[pair]["sl_pts"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(symbol=pair, pt=pt,
                                asian_low=al, asian_high=ah)
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        trigger_high = bars[-1].high
        assert sig.sl == pytest.approx(trigger_high + sl_pts * pt,
                                       rel=1e-9, abs=pt / 2)


# ---------------------------------------------------------------------------
# 5. TP1 (1.0R) and TP2 (2.5R)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestLongTpFormulas:
    def test_tp1_at_1r(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al,
            asian_high=al + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        risk = sig.risk_distance
        tp1 = _extract_tp1(sig)
        assert tp1 == pytest.approx(sig.entry + risk * RR_TP1,
                                    rel=1e-9, abs=pt / 2)

    def test_tp2_at_2_5r(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=al,
            asian_high=al + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        risk = sig.risk_distance
        assert sig.tp == pytest.approx(sig.entry + risk * RR_TP2,
                                       rel=1e-9, abs=pt / 2)

    def test_tp2_greater_than_tp1(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        tp1 = _extract_tp1(sig)
        assert sig.tp > tp1


@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestShortTpFormulas:
    def test_tp1_at_1r(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(symbol=pair, pt=pt,
                                asian_low=al, asian_high=ah)
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        risk = sig.risk_distance
        tp1 = _extract_tp1(sig)
        assert tp1 == pytest.approx(sig.entry - risk * RR_TP1,
                                    rel=1e-9, abs=pt / 2)

    def test_tp2_at_2_5r(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(symbol=pair, pt=pt,
                                asian_low=al, asian_high=ah)
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        risk = sig.risk_distance
        assert sig.tp == pytest.approx(sig.entry - risk * RR_TP2,
                                       rel=1e-9, abs=pt / 2)

    def test_tp2_less_than_tp1(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(symbol=pair, pt=pt,
                                asian_low=al, asian_high=ah)
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        tp1 = _extract_tp1(sig)
        assert sig.tp < tp1


# ---------------------------------------------------------------------------
# 6. RR ratio
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestRRRatio:
    def test_long_rr_ratio_close_to_2_5(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.rr_ratio == pytest.approx(RR_TP2, rel=1e-6)

    def test_short_rr_ratio_close_to_2_5(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(symbol=pair, pt=pt,
                                asian_low=al, asian_high=ah)
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.rr_ratio == pytest.approx(RR_TP2, rel=1e-6)


# ---------------------------------------------------------------------------
# 7. Price ordering invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestPriceOrdering:
    def test_long_sl_below_entry_below_tp(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.sl < sig.entry < sig.tp

    def test_short_tp_below_entry_below_sl(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(symbol=pair, pt=pt,
                                asian_low=al, asian_high=ah)
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.tp < sig.entry < sig.sl


# ---------------------------------------------------------------------------
# 8. Min-risk guard: |entry-SL| < 3 * pt → reject
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestMinRiskGuard:
    def test_degenerate_long_rejected(self, pair, detector):
        """Construct a setup whose risk distance is < 3 * pt → must reject.

        Approach: manipulate sl_pts indirectly by setting trigger_low VERY
        close to AL — but the SL buffer (per-pair sl_pts * pt) keeps risk
        large. The cleanest path is to monkey-patch PAIR_CONFIG sl_pts to 0
        and spread_pts to 0; then risk = AL - trigger_low.
        """
        pt = float(PAIR_CONFIG[pair]["point"])
        # Monkey-patch a copy of the pair config with sl_pts=0/spread_pts=0.
        from strategy.patterns import asian_sweep as mod
        orig = dict(PAIR_CONFIG[pair])
        patched = dict(orig)
        patched["sl_pts"] = 0
        patched["spread_pts"] = 0
        from types import MappingProxyType
        import config.asian_sweep_config as cfg_mod
        orig_outer = cfg_mod.PAIR_CONFIG
        mp = dict(orig_outer)
        mp[pair] = MappingProxyType(patched)
        cfg_mod.PAIR_CONFIG = MappingProxyType(mp)
        mod.PAIR_CONFIG = cfg_mod.PAIR_CONFIG
        try:
            al = _baseline_low(pair)
            ah = al + _baseline_range_pts(pair) * pt
            # Wick only 1 pt below AL → |entry - SL| = 1*pt < 3*pt → reject.
            bars = build_scenario(
                symbol=pair, year=2026, month=4, day=15,
                asian_high=ah, asian_low=al,
                trigger_hour=8,
                trigger_high=al + 5 * pt,
                trigger_low=al - 1 * pt,
                trigger_close=al + 5 * pt,
            )
            ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
            sig = detector.detect(bars, ctx)
            assert sig is None
        finally:
            cfg_mod.PAIR_CONFIG = orig_outer
            mod.PAIR_CONFIG = orig_outer


# ---------------------------------------------------------------------------
# 9. Risk distance is positive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("wick_pts", [10, 30, 60, 100, 200])
class TestRiskDistancePositive:
    def test_long_risk_positive(self, pair, wick_pts, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
            wick_below_pts=wick_pts,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.risk_distance > 0

    def test_short_risk_positive(self, pair, wick_pts, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(symbol=pair, pt=pt,
                                asian_low=al, asian_high=ah,
                                wick_above_pts=wick_pts)
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.risk_distance > 0


# ---------------------------------------------------------------------------
# 10. Risk distance grows with wick depth (LONG)
# ---------------------------------------------------------------------------

class TestRiskMonotonicity:
    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_deeper_wick_means_more_risk_long(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        risks = []
        for wick in [20, 50, 100, 200]:
            bars = long_sweep_bars(
                symbol=pair, pt=pt,
                asian_low=_baseline_low(pair),
                asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
                wick_below_pts=wick,
            )
            ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
            sig = detector.detect(bars, ctx)
            risks.append(sig.risk_distance)
        assert risks == sorted(risks)

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_deeper_wick_means_more_risk_short(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        risks = []
        for wick in [20, 50, 100, 200]:
            bars = short_sweep_bars(symbol=pair, pt=pt,
                                    asian_low=al, asian_high=ah,
                                    wick_above_pts=wick)
            ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
            sig = detector.detect(bars, ctx)
            risks.append(sig.risk_distance)
        assert risks == sorted(risks)


# ---------------------------------------------------------------------------
# 11. Spread variation in per-pair config affects entry placement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_entry_offset_matches_spread(pair, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    sp = float(PAIR_CONFIG[pair]["spread_pts"])
    al = _baseline_low(pair)
    bars = long_sweep_bars(
        symbol=pair, pt=pt,
        asian_low=al,
        asian_high=al + _baseline_range_pts(pair) * pt,
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    expected_offset_pts = sp
    actual_offset_pts = (sig.entry - al) / pt
    assert actual_offset_pts == pytest.approx(expected_offset_pts, abs=0.5)


# ---------------------------------------------------------------------------
# 12. SL distance from extreme matches sl_pts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_long_sl_distance_matches_buffer(pair, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    sl_pts = float(PAIR_CONFIG[pair]["sl_pts"])
    bars = long_sweep_bars(
        symbol=pair, pt=pt,
        asian_low=_baseline_low(pair),
        asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    sl_offset_pts = (bars[-1].low - sig.sl) / pt
    assert sl_offset_pts == pytest.approx(sl_pts, abs=0.5)


@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_short_sl_distance_matches_buffer(pair, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    sl_pts = float(PAIR_CONFIG[pair]["sl_pts"])
    al = _baseline_low(pair)
    ah = al + _baseline_range_pts(pair) * pt
    bars = short_sweep_bars(symbol=pair, pt=pt,
                            asian_low=al, asian_high=ah)
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    sl_offset_pts = (sig.sl - bars[-1].high) / pt
    assert sl_offset_pts == pytest.approx(sl_pts, abs=0.5)


# ---------------------------------------------------------------------------
# 13. Hypothesis property-based on R-multiple math (pure)
# ---------------------------------------------------------------------------

@settings(max_examples=120, deadline=None)
@given(
    entry=st.floats(min_value=1.0, max_value=100.0,
                    allow_nan=False, allow_infinity=False),
    risk_pts=st.floats(min_value=0.001, max_value=10.0,
                       allow_nan=False, allow_infinity=False),
    direction=st.sampled_from(["LONG", "SHORT"]),
)
def test_tp_math_property(entry, risk_pts, direction):
    """For any entry/risk and direction, TP1 = entry ± risk * 1.0,
    TP2 = entry ± risk * 2.5, and TP2 strictly past TP1."""
    if direction == "LONG":
        sl = entry - risk_pts
        tp1 = entry + risk_pts * RR_TP1
        tp2 = entry + risk_pts * RR_TP2
        assert sl < entry < tp1 < tp2
        assert tp2 - entry == pytest.approx(2.5 * (entry - sl), rel=1e-6)
    else:
        sl = entry + risk_pts
        tp1 = entry - risk_pts * RR_TP1
        tp2 = entry - risk_pts * RR_TP2
        assert tp2 < tp1 < entry < sl
        assert entry - tp2 == pytest.approx(2.5 * (sl - entry), rel=1e-6)


@settings(max_examples=80, deadline=None)
@given(
    al=st.floats(min_value=0.5, max_value=1000.0,
                 allow_nan=False, allow_infinity=False),
    range_pts=st.integers(min_value=200, max_value=2000),
    spread_pts=st.integers(min_value=0, max_value=50),
    sl_pts=st.integers(min_value=10, max_value=200),
    wick_pts=st.integers(min_value=10, max_value=500),
    pt=st.sampled_from([0.01, 0.0001, 0.00001]),
)
def test_long_entry_sl_property(al, range_pts, spread_pts, sl_pts, wick_pts, pt):
    """Closed-form check of LONG entry / SL placement formulas."""
    entry = al + spread_pts * pt
    trigger_low = al - wick_pts * pt
    sl = trigger_low - sl_pts * pt
    risk = abs(entry - sl)
    assert risk > 0
    assert sl < entry


# ---------------------------------------------------------------------------
# 14. Property: rr_ratio is constant across reasonable wick depths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("wick_pts", [25, 50, 75, 100, 150, 250, 400])
def test_rr_ratio_is_constant(pair, wick_pts, detector):
    """RR ratio (=TP2/risk) must always equal RR_TP2 regardless of wick depth."""
    pt = float(PAIR_CONFIG[pair]["point"])
    bars = long_sweep_bars(
        symbol=pair, pt=pt,
        asian_low=_baseline_low(pair),
        asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        wick_below_pts=wick_pts,
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert sig.rr_ratio == pytest.approx(RR_TP2, rel=1e-6)


# ---------------------------------------------------------------------------
# 15. Reward distance = 2.5 × risk distance (sanity)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_reward_equals_2_5_risk_long(pair, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    bars = long_sweep_bars(
        symbol=pair, pt=pt,
        asian_low=_baseline_low(pair),
        asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert sig.reward_distance == pytest.approx(2.5 * sig.risk_distance,
                                                rel=1e-6)


@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_reward_equals_2_5_risk_short(pair, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    al = _baseline_low(pair)
    ah = al + _baseline_range_pts(pair) * pt
    bars = short_sweep_bars(symbol=pair, pt=pt,
                            asian_low=al, asian_high=ah)
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert sig.reward_distance == pytest.approx(2.5 * sig.risk_distance,
                                                rel=1e-6)


# ---------------------------------------------------------------------------
# 16. TP1 distance = 1.0 × risk (sanity)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_tp1_distance_equals_risk_long(pair, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    bars = long_sweep_bars(
        symbol=pair, pt=pt,
        asian_low=_baseline_low(pair),
        asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    tp1 = _extract_tp1(sig)
    assert (tp1 - sig.entry) == pytest.approx(sig.risk_distance, rel=1e-6)


@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_tp1_distance_equals_risk_short(pair, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    al = _baseline_low(pair)
    ah = al + _baseline_range_pts(pair) * pt
    bars = short_sweep_bars(symbol=pair, pt=pt,
                            asian_low=al, asian_high=ah)
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    tp1 = _extract_tp1(sig)
    assert (sig.entry - tp1) == pytest.approx(sig.risk_distance, rel=1e-6)


# ---------------------------------------------------------------------------
# 17. Wider range × per-pair × direction matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("range_frac", [0.3, 0.5, 0.7, 0.9])
def test_long_math_across_range_fractions(pair, range_frac, detector):
    cfg = PAIR_CONFIG[pair]
    pt = float(cfg["point"])
    mn = float(cfg["min_range_pts"])
    mx = float(cfg["max_range_pts"])
    range_pts = mn + (mx - mn) * range_frac
    al = _baseline_low(pair)
    ah = al + range_pts * pt
    bars = long_sweep_bars(symbol=pair, pt=pt,
                           asian_low=al, asian_high=ah)
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert sig is not None
    # Math invariants hold regardless of range size.
    assert sig.sl < sig.entry < sig.tp
    assert sig.rr_ratio == pytest.approx(RR_TP2, rel=1e-6)


@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("range_frac", [0.3, 0.5, 0.7, 0.9])
def test_short_math_across_range_fractions(pair, range_frac, detector):
    cfg = PAIR_CONFIG[pair]
    pt = float(cfg["point"])
    mn = float(cfg["min_range_pts"])
    mx = float(cfg["max_range_pts"])
    range_pts = mn + (mx - mn) * range_frac
    al = _baseline_low(pair)
    ah = al + range_pts * pt
    bars = short_sweep_bars(symbol=pair, pt=pt,
                            asian_low=al, asian_high=ah)
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert sig is not None
    assert sig.tp < sig.entry < sig.sl
    assert sig.rr_ratio == pytest.approx(RR_TP2, rel=1e-6)


# ---------------------------------------------------------------------------
# 18. Internal helper: _ema seeded-with-first-value invariant
# ---------------------------------------------------------------------------

class TestEmaHelper:
    def test_empty_returns_zero(self):
        from strategy.patterns.asian_sweep import _ema
        assert _ema([], 20) == 0.0

    def test_single_value(self):
        from strategy.patterns.asian_sweep import _ema
        assert _ema([1.5], 20) == 1.5

    def test_constant_values(self):
        from strategy.patterns.asian_sweep import _ema
        assert _ema([2.0] * 50, 20) == pytest.approx(2.0)

    def test_increasing_values(self):
        from strategy.patterns.asian_sweep import _ema
        e = _ema([1.0, 2.0, 3.0, 4.0, 5.0], 5)
        assert 1.0 < e < 5.0

    def test_matches_pandas_ewm(self):
        """Sanity check against pandas.ewm(span, adjust=False).mean()."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas unavailable")
        from strategy.patterns.asian_sweep import _ema
        series = [1.0, 1.1, 0.95, 1.05, 1.2, 1.15, 1.0, 0.9, 0.95, 1.0]
        expected = pd.Series(series).ewm(span=5, adjust=False).mean().iloc[-1]
        assert _ema(series, 5) == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("span", [5, 10, 50, 100, 200])
    def test_constant_values_various_spans(self, span):
        from strategy.patterns.asian_sweep import _ema
        assert _ema([7.0] * 250, span) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# 19. Bias detection helper
# ---------------------------------------------------------------------------

class TestBiasHelper:
    def test_neutral_when_fewer_than_200_closes(self):
        from strategy.patterns.asian_sweep import _compute_bias
        from tests.strategy.fixtures.synthetic_bars import (
            build_scenario,
        )
        bars = build_scenario(
            symbol="EURUSD", year=2026, month=4, day=15,
            asian_high=1.105, asian_low=1.103,
            trigger_hour=8,
            trigger_high=1.1031, trigger_low=1.10250, trigger_close=1.1031,
            bias="neutral", history_bars=30,
        )
        from datetime import datetime, timezone
        cur_dt = datetime.fromtimestamp(bars[-1].time_msc / 1000.0,
                                        tz=timezone.utc)
        assert _compute_bias(bars, cur_dt) == "neutral"


# ---------------------------------------------------------------------------
# 20. ALL_PAIRS spread×sl combinations preserve invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_spread_pts_below_sl_pts_keeps_long_valid(pair, detector):
    """Production constraint: spread_pts < sl_pts, so entry never lands
    inside [trigger_low, AL]; SL stays strictly below entry."""
    cfg = PAIR_CONFIG[pair]
    assert cfg["spread_pts"] < cfg["sl_pts"]
    # Verify with an actual signal.
    pt = float(cfg["point"])
    bars = long_sweep_bars(
        symbol=pair, pt=pt,
        asian_low=_baseline_low(pair),
        asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert sig.sl < sig.entry
