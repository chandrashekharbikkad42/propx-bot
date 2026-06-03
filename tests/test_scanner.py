"""Phase 8C — Scanner tests.

Uses a fake `StubPattern` that returns a pre-configured PatternSignal per
pair so we can test ranking, grade filtering, and tie-breaking without
real Griff patterns (which arrive in Phase 8C-Patterns).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence

import pytest

from data.bar_aggregator import Bar
from strategy.patterns.base import (
    Direction,
    Grade,
    MarketContext,
    PatternDetector,
    PatternSignal,
)
from strategy.scanner import Scanner


def _bar(t: int = 0, p: float = 1.10) -> Bar:
    return Bar("X", t, p, p + 0.001, p - 0.001, p, 1)


def _sig(
    pair: str = "EURUSD",
    grade: Grade = Grade.A,
    confidence: float = 0.7,
    pattern_name: str = "P",
    entry: float = 1.10,
    sl: float = 1.09,
    tp: float = 1.13,
    direction: Direction = Direction.BUY,
) -> PatternSignal:
    return PatternSignal(
        pattern_name=pattern_name, symbol=pair, direction=direction,
        entry=entry, sl=sl, tp=tp, confidence=confidence, grade=grade,
        confluences_met=("c1", "c2") if grade == Grade.A else ("c1",),
        bar_time_msc=1_000_000,
    )


class StubPattern(PatternDetector):
    """Test stub. Returns a pre-set signal for matching symbol, else None."""
    name = "STUB"
    min_bars_required = 1
    timeframe = "1H"

    def __init__(
        self,
        per_pair_signals: Optional[Dict[str, Optional[PatternSignal]]] = None,
        min_bars_required: int = 1,
        name: str = "STUB",
    ) -> None:
        self._sigs = per_pair_signals or {}
        # Allow per-instance overrides of class attrs without subclassing
        self.min_bars_required = min_bars_required
        self.name = name

    def detect(
        self, bars: Sequence[Bar], context: MarketContext
    ) -> Optional[PatternSignal]:
        return self._sigs.get(context.symbol)


# ---------------------------------------------------------------------------

class TestConstruction:
    def test_no_pairs_rejected(self):
        with pytest.raises(ValueError):
            Scanner([], [StubPattern()])

    def test_no_patterns_rejected(self):
        with pytest.raises(ValueError):
            Scanner(["EURUSD"], [])


class TestScanAll:
    def test_no_bars_no_signals(self):
        s = Scanner(["EURUSD"], [StubPattern({"EURUSD": _sig()})])
        sigs = s.scan_all({}, 1_000_000)
        assert sigs == ()
        assert s.get_best_signal() is None

    def test_single_pair_one_signal(self):
        s = Scanner(["EURUSD"], [StubPattern({"EURUSD": _sig()})])
        sigs = s.scan_all({"EURUSD": [_bar()]}, 1_000_000)
        assert len(sigs) == 1
        assert sigs[0].symbol == "EURUSD"

    def test_skips_pair_with_insufficient_bars(self):
        s = Scanner(
            ["EURUSD"],
            [StubPattern({"EURUSD": _sig()}, min_bars_required=10)],
        )
        sigs = s.scan_all({"EURUSD": [_bar()]}, 1_000_000)
        assert sigs == ()
        assert s.skipped_insufficient_bars == 1

    def test_multiple_patterns_same_pair(self):
        p1 = StubPattern({"EURUSD": _sig(pattern_name="P1")}, name="P1")
        p2 = StubPattern({"EURUSD": _sig(pattern_name="P2")}, name="P2")
        s = Scanner(["EURUSD"], [p1, p2])
        sigs = s.scan_all({"EURUSD": [_bar()]}, 1_000_000)
        assert len(sigs) == 2

    def test_same_pattern_multiple_pairs(self):
        p = StubPattern({
            "EURUSD": _sig(pair="EURUSD"),
            "GBPUSD": _sig(pair="GBPUSD"),
        })
        s = Scanner(["EURUSD", "GBPUSD"], [p])
        sigs = s.scan_all({"EURUSD": [_bar()], "GBPUSD": [_bar()]}, 1_000_000)
        assert {x.symbol for x in sigs} == {"EURUSD", "GBPUSD"}

    def test_context_override_propagates(self):
        seen: Dict[str, Optional[str]] = {}

        class Spy(PatternDetector):
            name = "SPY"
            min_bars_required = 1

            def detect(self, bars, context):
                seen[context.symbol] = context.htf_bias
                return None

        s = Scanner(["EURUSD"], [Spy()])
        s.scan_all(
            {"EURUSD": [_bar()]}, 1_000_000,
            context_overrides={"EURUSD": MarketContext("EURUSD", 0, htf_bias="BULLISH")},
        )
        assert seen["EURUSD"] == "BULLISH"


class TestGetBestSignal:
    def test_returns_none_when_only_c_grade(self):
        p = StubPattern({"EURUSD": _sig(grade=Grade.C)})
        s = Scanner(["EURUSD"], [p])
        s.scan_all({"EURUSD": [_bar()]}, 1_000_000)
        assert s.get_best_signal() is None
        assert s.c_grade_dropped == 1

    def test_a_beats_b(self):
        p1 = StubPattern({"EURUSD": _sig(grade=Grade.B, confidence=0.99)}, name="P1")
        p2 = StubPattern({"EURUSD": _sig(grade=Grade.A, confidence=0.50)}, name="P2")
        s = Scanner(["EURUSD"], [p1, p2])
        s.scan_all({"EURUSD": [_bar()]}, 1_000_000)
        best = s.get_best_signal()
        assert best is not None and best.grade == Grade.A

    def test_higher_confidence_wins_same_grade(self):
        p1 = StubPattern({"EURUSD": _sig(grade=Grade.A, confidence=0.5, pattern_name="LOW")}, name="P1")
        p2 = StubPattern({"EURUSD": _sig(grade=Grade.A, confidence=0.9, pattern_name="HIGH")}, name="P2")
        s = Scanner(["EURUSD"], [p1, p2])
        s.scan_all({"EURUSD": [_bar()]}, 1_000_000)
        best = s.get_best_signal()
        assert best is not None and best.confidence == pytest.approx(0.9)

    def test_rr_tiebreaker(self):
        # Same grade + confidence — higher R:R wins.
        sig_low_rr = _sig(grade=Grade.A, confidence=0.7, entry=1.10, sl=1.09, tp=1.12)  # R:R=2
        sig_high_rr = _sig(grade=Grade.A, confidence=0.7, entry=1.10, sl=1.09, tp=1.16)  # R:R=6
        p1 = StubPattern({"EURUSD": sig_low_rr}, name="P1")
        p2 = StubPattern({"EURUSD": sig_high_rr}, name="P2")
        s = Scanner(["EURUSD"], [p1, p2])
        s.scan_all({"EURUSD": [_bar()]}, 1_000_000)
        best = s.get_best_signal()
        assert best is not None and best.rr_ratio == pytest.approx(6.0)

    def test_picks_best_across_pairs(self):
        # EURUSD B-grade, GBPUSD A-grade → GBPUSD wins.
        p = StubPattern({
            "EURUSD": _sig(pair="EURUSD", grade=Grade.B),
            "GBPUSD": _sig(pair="GBPUSD", grade=Grade.A),
        })
        s = Scanner(["EURUSD", "GBPUSD"], [p])
        s.scan_all({"EURUSD": [_bar()], "GBPUSD": [_bar()]}, 1_000_000)
        best = s.get_best_signal()
        assert best is not None and best.symbol == "GBPUSD"


class TestSignalsByGrade:
    def test_groups_correctly(self):
        p = StubPattern({
            "EURUSD": _sig(pair="EURUSD", grade=Grade.A),
            "GBPUSD": _sig(pair="GBPUSD", grade=Grade.B),
            "USDJPY": _sig(pair="USDJPY", grade=Grade.C, entry=150.0, sl=148.0, tp=156.0),
        })
        s = Scanner(["EURUSD", "GBPUSD", "USDJPY"], [p])
        s.scan_all({sym: [_bar()] for sym in ("EURUSD", "GBPUSD", "USDJPY")}, 1_000_000)
        bg = s.signals_by_grade()
        assert len(bg[Grade.A]) == 1
        assert len(bg[Grade.B]) == 1
        assert len(bg[Grade.C]) == 1


class TestDiagnostics:
    def test_counts_reset_per_scan(self):
        p_c = StubPattern({"EURUSD": _sig(grade=Grade.C)})
        s = Scanner(["EURUSD"], [p_c])
        s.scan_all({"EURUSD": [_bar()]}, 1_000_000)
        assert s.c_grade_dropped == 1
        # Second scan with no C signals
        p_a = StubPattern({"EURUSD": _sig(grade=Grade.A)})
        s2 = Scanner(["EURUSD"], [p_a])
        s2.scan_all({"EURUSD": [_bar()]}, 1_000_000)
        assert s2.c_grade_dropped == 0
