"""Grade / confidence / quality-ranking tests for AsianSweepDetector.

Mapping (from `_GRADE_A_QUALITY_CUTOFF = 9`):
    quality 9–10 → Grade.A
    quality 4–8  → Grade.B
    quality 1–3  → Grade.B (still, since `else Grade.B`)

The detector clamps confidence to [0, 1] via max(0, min(1, quality/10)),
so unknown pairs (quality=0) → confidence=0 — but unknown pairs are
rejected earlier, so we never see confidence=0 in practice.
"""

from __future__ import annotations

import pytest

from config.asian_sweep_config import PAIR_CONFIG, PAIRS
from strategy.patterns.asian_sweep import (
    AsianSweepDetector, _GRADE_A_QUALITY_CUTOFF, _build_signal,
)
from strategy.patterns.base import Direction, Grade, MarketContext

from tests.strategy.fixtures.synthetic_bars import (
    baseline_low, long_sweep_bars, short_sweep_bars,
)

ALL_PAIRS = list(PAIRS)


def _baseline_low(pair: str) -> float:
    return baseline_low(pair)


def _baseline_range_pts(pair: str) -> float:
    cfg = PAIR_CONFIG[pair]
    return (float(cfg["min_range_pts"])
            + float(cfg["max_range_pts"])) / 2.0


# ---------------------------------------------------------------------------
# 1. Grade cutoff constant
# ---------------------------------------------------------------------------

class TestGradeCutoff:
    def test_grade_a_cutoff_is_9(self):
        assert _GRADE_A_QUALITY_CUTOFF == 9


# ---------------------------------------------------------------------------
# 2. Quality → Grade.A pairs
# ---------------------------------------------------------------------------

GRADE_A_PAIRS = [p for p in ALL_PAIRS if PAIR_CONFIG[p]["quality"] >= 9]
GRADE_B_PAIRS = [p for p in ALL_PAIRS if PAIR_CONFIG[p]["quality"] < 9]


class TestGradeAPairsLong:
    @pytest.mark.parametrize("pair", GRADE_A_PAIRS)
    def test_long_emits_grade_a(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.grade == Grade.A

    @pytest.mark.parametrize("pair", GRADE_A_PAIRS)
    def test_short_emits_grade_a(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(symbol=pair, pt=pt,
                                asian_low=al, asian_high=ah)
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.grade == Grade.A


class TestGradeBPairsLong:
    @pytest.mark.parametrize("pair", GRADE_B_PAIRS)
    def test_long_emits_grade_b(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.grade == Grade.B

    @pytest.mark.parametrize("pair", GRADE_B_PAIRS)
    def test_short_emits_grade_b(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(symbol=pair, pt=pt,
                                asian_low=al, asian_high=ah)
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.grade == Grade.B


# ---------------------------------------------------------------------------
# 3. Confidence ranking matches quality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestConfidenceFromQuality:
    def test_long_confidence(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        q = PAIR_CONFIG[pair]["quality"]
        assert sig.confidence == pytest.approx(q / 10.0)

    def test_short_confidence(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(symbol=pair, pt=pt,
                                asian_low=al, asian_high=ah)
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        q = PAIR_CONFIG[pair]["quality"]
        assert sig.confidence == pytest.approx(q / 10.0)


# ---------------------------------------------------------------------------
# 4. Q-tag in confluences matches quality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
class TestQTagMatchesQuality:
    def test_q_tag_format_long(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        q = PAIR_CONFIG[pair]["quality"]
        assert f"q{q}" in sig.confluences_met

    def test_q_tag_format_short(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(symbol=pair, pt=pt,
                                asian_low=al, asian_high=ah)
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        q = PAIR_CONFIG[pair]["quality"]
        assert f"q{q}" in sig.confluences_met


# ---------------------------------------------------------------------------
# 5. Ordering: Grade.A > Grade.B (.rank)
# ---------------------------------------------------------------------------

class TestGradeRanking:
    def test_a_rank_above_b(self):
        assert Grade.A.rank > Grade.B.rank

    def test_b_rank_above_c(self):
        assert Grade.B.rank > Grade.C.rank

    def test_a_ranks_above_c(self):
        assert Grade.A.rank > Grade.C.rank


# ---------------------------------------------------------------------------
# 6. _build_signal direct unit tests
# ---------------------------------------------------------------------------

class TestBuildSignal:
    @pytest.mark.parametrize("q", [9, 10])
    def test_grade_a_when_above_cutoff(self, q):
        sig = _build_signal(
            symbol="EURUSD", direction=Direction.BUY,
            entry=1.105, sl=1.10, tp2=1.11, tp1=1.107,
            quality=q, session="LONDON", bias="neutral",
            bar_time_msc=1, sweep_tag="asian_sweep_low",
        )
        assert sig.grade == Grade.A

    @pytest.mark.parametrize("q", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    def test_grade_b_when_below_cutoff(self, q):
        sig = _build_signal(
            symbol="EURUSD", direction=Direction.BUY,
            entry=1.105, sl=1.10, tp2=1.11, tp1=1.107,
            quality=q, session="LONDON", bias="neutral",
            bar_time_msc=1, sweep_tag="asian_sweep_low",
        )
        assert sig.grade == Grade.B

    @pytest.mark.parametrize("q,expected", [
        (0, 0.0), (1, 0.1), (3, 0.3), (5, 0.5),
        (7, 0.7), (8, 0.8), (9, 0.9), (10, 1.0),
    ])
    def test_confidence_formula(self, q, expected):
        sig = _build_signal(
            symbol="EURUSD", direction=Direction.BUY,
            entry=1.105, sl=1.10, tp2=1.11, tp1=1.107,
            quality=q, session="LONDON", bias="neutral",
            bar_time_msc=1, sweep_tag="asian_sweep_low",
        )
        assert sig.confidence == pytest.approx(expected)

    @pytest.mark.parametrize("q", [11, 100, 1000])
    def test_confidence_clamped_above(self, q):
        sig = _build_signal(
            symbol="EURUSD", direction=Direction.BUY,
            entry=1.105, sl=1.10, tp2=1.11, tp1=1.107,
            quality=q, session="LONDON", bias="neutral",
            bar_time_msc=1, sweep_tag="asian_sweep_low",
        )
        assert sig.confidence == 1.0
        # Grade is A because q >= 9.
        assert sig.grade == Grade.A

    @pytest.mark.parametrize("q", [-1, -5, -100])
    def test_confidence_clamped_below(self, q):
        sig = _build_signal(
            symbol="EURUSD", direction=Direction.BUY,
            entry=1.105, sl=1.10, tp2=1.11, tp1=1.107,
            quality=q, session="LONDON", bias="neutral",
            bar_time_msc=1, sweep_tag="asian_sweep_low",
        )
        assert sig.confidence == 0.0
        # Grade is B because q < 9.
        assert sig.grade == Grade.B


# ---------------------------------------------------------------------------
# 7. Confluence tag content from _build_signal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("session", ["LONDON", "NY"])
@pytest.mark.parametrize("bias", ["bullish", "bearish", "neutral"])
@pytest.mark.parametrize("sweep_tag",
                         ["asian_sweep_low", "asian_sweep_high"])
class TestConfluenceContent:
    def test_session_tag_present(self, session, bias, sweep_tag):
        sig = _build_signal(
            symbol="EURUSD", direction=Direction.BUY,
            entry=1.105, sl=1.10, tp2=1.11, tp1=1.107,
            quality=8, session=session, bias=bias,
            bar_time_msc=1, sweep_tag=sweep_tag,
        )
        assert session in sig.confluences_met

    def test_bias_tag_present(self, session, bias, sweep_tag):
        sig = _build_signal(
            symbol="EURUSD", direction=Direction.BUY,
            entry=1.105, sl=1.10, tp2=1.11, tp1=1.107,
            quality=8, session=session, bias=bias,
            bar_time_msc=1, sweep_tag=sweep_tag,
        )
        assert f"bias_{bias}" in sig.confluences_met

    def test_sweep_tag_present(self, session, bias, sweep_tag):
        sig = _build_signal(
            symbol="EURUSD", direction=Direction.BUY,
            entry=1.105, sl=1.10, tp2=1.11, tp1=1.107,
            quality=8, session=session, bias=bias,
            bar_time_msc=1, sweep_tag=sweep_tag,
        )
        assert sweep_tag in sig.confluences_met

    def test_confluence_tuple_length_5(self, session, bias, sweep_tag):
        sig = _build_signal(
            symbol="EURUSD", direction=Direction.BUY,
            entry=1.105, sl=1.10, tp2=1.11, tp1=1.107,
            quality=8, session=session, bias=bias,
            bar_time_msc=1, sweep_tag=sweep_tag,
        )
        assert len(sig.confluences_met) == 5


# ---------------------------------------------------------------------------
# 8. TP1 tag formatting precision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tp1", [
    1.10000, 1.10005, 1.10009, 1.0, 100.0, 1234.56789, 0.50001,
])
def test_tp1_tag_5_decimal_places(tp1):
    sig = _build_signal(
        symbol="EURUSD", direction=Direction.BUY,
        entry=1.105, sl=1.10, tp2=tp1 + 1.0, tp1=tp1,
        quality=8, session="LONDON", bias="neutral",
        bar_time_msc=1, sweep_tag="asian_sweep_low",
    )
    tp1_tag = [t for t in sig.confluences_met if t.startswith("tp1_")][0]
    # Format is tp1_{:.5f} → exactly 5 fractional digits.
    value_part = tp1_tag[len("tp1_"):]
    if "." in value_part:
        frac = value_part.split(".", 1)[1]
        assert len(frac) == 5


# ---------------------------------------------------------------------------
# 9. Per-pair × direction → consistent grade outputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_grade_consistent_across_direction(pair, direction, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    al = _baseline_low(pair)
    ah = al + _baseline_range_pts(pair) * pt
    if direction == "LONG":
        bars = long_sweep_bars(symbol=pair, pt=pt,
                               asian_low=al, asian_high=ah)
    else:
        bars = short_sweep_bars(symbol=pair, pt=pt,
                                asian_low=al, asian_high=ah)
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    q = PAIR_CONFIG[pair]["quality"]
    expected = Grade.A if q >= _GRADE_A_QUALITY_CUTOFF else Grade.B
    assert sig.grade == expected


# ---------------------------------------------------------------------------
# 10. Pair ordering by confidence (higher quality wins)
# ---------------------------------------------------------------------------

class TestQualityOrdering:
    def test_xau_outranks_audnzd(self, detector):
        # Build a signal for both. XAU has q=10 → conf=1.0; AUDNZD q=4
        # → conf=0.4. Sort by confidence desc: XAU first.
        from config.asian_sweep_config import PAIR_CONFIG as cfg
        pairs = ["XAUUSD", "AUDNZD"]
        sigs = []
        for p in pairs:
            pt = float(cfg[p]["point"])
            bars = long_sweep_bars(
                symbol=p, pt=pt,
                asian_low=_baseline_low(p),
                asian_high=_baseline_low(p) + _baseline_range_pts(p) * pt,
            )
            ctx = MarketContext(symbol=p, current_time_msc=bars[-1].time_msc)
            sigs.append(detector.detect(bars, ctx))
        sigs_sorted = sorted(sigs,
                             key=lambda s: (s.grade.rank, s.confidence, s.rr_ratio),
                             reverse=True)
        assert sigs_sorted[0].symbol == "XAUUSD"

    def test_eurusd_outranks_gbpusd_on_quality(self, detector):
        from config.asian_sweep_config import PAIR_CONFIG as cfg
        pairs = ["EURUSD", "GBPUSD"]
        sigs = []
        for p in pairs:
            pt = float(cfg[p]["point"])
            bars = long_sweep_bars(
                symbol=p, pt=pt,
                asian_low=_baseline_low(p),
                asian_high=_baseline_low(p) + _baseline_range_pts(p) * pt,
            )
            ctx = MarketContext(symbol=p, current_time_msc=bars[-1].time_msc)
            sigs.append(detector.detect(bars, ctx))
        # EURUSD q=9 (A), GBPUSD q=8 (B). Grade A first.
        sigs_sorted = sorted(sigs,
                             key=lambda s: (s.grade.rank, s.confidence, s.rr_ratio),
                             reverse=True)
        assert sigs_sorted[0].symbol == "EURUSD"


# ---------------------------------------------------------------------------
# 11. Tie-breakers (rr_ratio = 2.5 for every signal — so ties on RR)
# ---------------------------------------------------------------------------

class TestRrTieBreaker:
    def test_rr_equal_across_signals(self, detector):
        from config.asian_sweep_config import PAIR_CONFIG as cfg, RR_TP2
        sigs = []
        for p in ALL_PAIRS:
            pt = float(cfg[p]["point"])
            bars = long_sweep_bars(
                symbol=p, pt=pt,
                asian_low=_baseline_low(p),
                asian_high=_baseline_low(p) + _baseline_range_pts(p) * pt,
            )
            ctx = MarketContext(symbol=p, current_time_msc=bars[-1].time_msc)
            sigs.append(detector.detect(bars, ctx))
        for s in sigs:
            assert s.rr_ratio == pytest.approx(RR_TP2, rel=1e-6)


# ---------------------------------------------------------------------------
# 12. Grade.C never emitted by AsianSweepDetector
# ---------------------------------------------------------------------------

class TestNoGradeC:
    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_long_grade_is_a_or_b(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        bars = long_sweep_bars(
            symbol=pair, pt=pt,
            asian_low=_baseline_low(pair),
            asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
        )
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.grade != Grade.C

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_short_grade_is_a_or_b(self, pair, detector):
        pt = float(PAIR_CONFIG[pair]["point"])
        al = _baseline_low(pair)
        ah = al + _baseline_range_pts(pair) * pt
        bars = short_sweep_bars(symbol=pair, pt=pt,
                                asian_low=al, asian_high=ah)
        ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
        sig = detector.detect(bars, ctx)
        assert sig.grade != Grade.C


# ---------------------------------------------------------------------------
# 13. Direct mapping table verification
# ---------------------------------------------------------------------------

QUALITY_GRADE_MAP = [
    (0, Grade.B), (1, Grade.B), (2, Grade.B), (3, Grade.B),
    (4, Grade.B), (5, Grade.B), (6, Grade.B), (7, Grade.B), (8, Grade.B),
    (9, Grade.A), (10, Grade.A),
]


@pytest.mark.parametrize("quality,expected", QUALITY_GRADE_MAP)
def test_quality_to_grade_table(quality, expected):
    sig = _build_signal(
        symbol="EURUSD", direction=Direction.BUY,
        entry=1.105, sl=1.10, tp2=1.11, tp1=1.107,
        quality=quality, session="LONDON", bias="neutral",
        bar_time_msc=1, sweep_tag="asian_sweep_low",
    )
    assert sig.grade == expected


# ---------------------------------------------------------------------------
# 14. Real-trade CSV cross-check (when available)
# ---------------------------------------------------------------------------

from tests.strategy.fixtures.csv_scenarios import TRADES, trades_by_sym

CSV_GRADE_A_PAIRS = {"XAUUSD", "EURUSD", "AUDUSD"}


@pytest.mark.parametrize("pair", sorted(CSV_GRADE_A_PAIRS))
def test_csv_grade_a_pair_quality(pair):
    if not TRADES:
        pytest.skip("CSV unavailable")
    q = PAIR_CONFIG[pair]["quality"]
    assert q >= 9


# ---------------------------------------------------------------------------
# 15. CSV trade quality matches config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_csv_quality_matches_config(pair):
    if not TRADES:
        pytest.skip("CSV unavailable")
    rows = list(trades_by_sym(pair))
    if not rows:
        pytest.skip(f"no rows for {pair}")
    expected_q = PAIR_CONFIG[pair]["quality"]
    for r in rows:
        assert r.quality == expected_q


# ---------------------------------------------------------------------------
# 16. Direction tag matches CSV historical direction (LONG/SHORT)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_csv_direction_field(direction):
    if not TRADES:
        pytest.skip("CSV unavailable")
    rows = [t for t in TRADES if t.direction == direction]
    assert all(r.direction == direction for r in rows)


# ---------------------------------------------------------------------------
# 17. Cross-pair: at any one time the highest-quality pair has top confidence
# ---------------------------------------------------------------------------

def test_max_confidence_belongs_to_top_quality_pair(detector):
    """If we detect across all pairs in parallel, the pair with q=10
    yields confidence=1.0 — strictly the maximum."""
    from config.asian_sweep_config import PAIR_CONFIG as cfg
    sigs = []
    for p in ALL_PAIRS:
        pt = float(cfg[p]["point"])
        bars = long_sweep_bars(
            symbol=p, pt=pt,
            asian_low=_baseline_low(p),
            asian_high=_baseline_low(p) + _baseline_range_pts(p) * pt,
        )
        ctx = MarketContext(symbol=p, current_time_msc=bars[-1].time_msc)
        sigs.append(detector.detect(bars, ctx))
    best = max(sigs, key=lambda s: s.confidence)
    assert best.symbol == "XAUUSD"


# ---------------------------------------------------------------------------
# 18. Sweep-tag check by direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_long_sweep_tag_is_low(pair, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    bars = long_sweep_bars(
        symbol=pair, pt=pt,
        asian_low=_baseline_low(pair),
        asian_high=_baseline_low(pair) + _baseline_range_pts(pair) * pt,
    )
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert "asian_sweep_low" in sig.confluences_met
    assert "asian_sweep_high" not in sig.confluences_met


@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_short_sweep_tag_is_high(pair, detector):
    pt = float(PAIR_CONFIG[pair]["point"])
    al = _baseline_low(pair)
    ah = al + _baseline_range_pts(pair) * pt
    bars = short_sweep_bars(symbol=pair, pt=pt,
                            asian_low=al, asian_high=ah)
    ctx = MarketContext(symbol=pair, current_time_msc=bars[-1].time_msc)
    sig = detector.detect(bars, ctx)
    assert "asian_sweep_high" in sig.confluences_met
    assert "asian_sweep_low" not in sig.confluences_met


# ---------------------------------------------------------------------------
# 19. Confidence equals quality/10 across all known qualities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", list(range(0, 11)))
def test_confidence_equals_q_over_10(q):
    sig = _build_signal(
        symbol="EURUSD", direction=Direction.BUY,
        entry=1.105, sl=1.10, tp2=1.11, tp1=1.107,
        quality=q, session="LONDON", bias="neutral",
        bar_time_msc=1, sweep_tag="asian_sweep_low",
    )
    assert sig.confidence == pytest.approx(q / 10.0)


# ---------------------------------------------------------------------------
# 20. _build_signal raises for malformed BUY (sl >= entry)
# ---------------------------------------------------------------------------

class TestBuildSignalValidation:
    def test_long_sl_not_below_entry_raises(self):
        with pytest.raises(ValueError):
            _build_signal(
                symbol="EURUSD", direction=Direction.BUY,
                entry=1.10, sl=1.10, tp2=1.11, tp1=1.105,
                quality=8, session="LONDON", bias="neutral",
                bar_time_msc=1, sweep_tag="asian_sweep_low",
            )

    def test_long_tp_not_above_entry_raises(self):
        with pytest.raises(ValueError):
            _build_signal(
                symbol="EURUSD", direction=Direction.BUY,
                entry=1.105, sl=1.10, tp2=1.105, tp1=1.105,
                quality=8, session="LONDON", bias="neutral",
                bar_time_msc=1, sweep_tag="asian_sweep_low",
            )

    def test_short_sl_not_above_entry_raises(self):
        with pytest.raises(ValueError):
            _build_signal(
                symbol="EURUSD", direction=Direction.SELL,
                entry=1.105, sl=1.105, tp2=1.10, tp1=1.103,
                quality=8, session="LONDON", bias="bearish",
                bar_time_msc=1, sweep_tag="asian_sweep_high",
            )

    def test_short_tp_not_below_entry_raises(self):
        with pytest.raises(ValueError):
            _build_signal(
                symbol="EURUSD", direction=Direction.SELL,
                entry=1.105, sl=1.11, tp2=1.105, tp1=1.103,
                quality=8, session="LONDON", bias="bearish",
                bar_time_msc=1, sweep_tag="asian_sweep_high",
            )

    def test_negative_entry_raises(self):
        with pytest.raises(ValueError):
            _build_signal(
                symbol="EURUSD", direction=Direction.BUY,
                entry=-1.0, sl=-2.0, tp2=0.5, tp1=0.0,
                quality=8, session="LONDON", bias="neutral",
                bar_time_msc=1, sweep_tag="asian_sweep_low",
            )

    def test_zero_entry_raises(self):
        with pytest.raises(ValueError):
            _build_signal(
                symbol="EURUSD", direction=Direction.BUY,
                entry=0.0, sl=-1.0, tp2=1.0, tp1=0.5,
                quality=8, session="LONDON", bias="neutral",
                bar_time_msc=1, sweep_tag="asian_sweep_low",
            )


# ---------------------------------------------------------------------------
# 21. Grade.B is the dominant grade in V5 universe
# ---------------------------------------------------------------------------

def test_majority_grade_b_pairs():
    grade_a = sum(1 for p in ALL_PAIRS if PAIR_CONFIG[p]["quality"] >= 9)
    grade_b = sum(1 for p in ALL_PAIRS if PAIR_CONFIG[p]["quality"] < 9)
    # V5 13-pair universe: 4 A-grade (XAU q10, EUR q9, AUD q9, HK50 q9) and
    # 9 B-grade — Grade.B stays the dominant bucket.
    assert grade_a == 4
    assert grade_b == 9
    assert grade_a + grade_b == len(ALL_PAIRS)
    assert grade_b > grade_a
