"""Phase 8C — pattern framework foundation tests.

Covers: Grade rank, Direction enum, MarketContext, PatternSignal validation
(price ordering, confidence range, positivity), PatternDetector ABC contract.
"""

from __future__ import annotations
from typing import List, Optional, Sequence

import pytest

from data.bar_aggregator import Bar
from strategy.patterns.base import (
    Direction,
    Grade,
    MarketContext,
    PatternDetector,
    PatternSignal,
)


# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------

class TestGrade:
    def test_three_values(self):
        assert {Grade.A, Grade.B, Grade.C} == set(Grade)

    def test_rank_ordering(self):
        assert Grade.A.rank > Grade.B.rank > Grade.C.rank

    def test_rank_values(self):
        assert Grade.A.rank == 2
        assert Grade.B.rank == 1
        assert Grade.C.rank == 0

    def test_string_value(self):
        assert Grade.A.value == "A"


class TestDirection:
    def test_values(self):
        assert Direction.BUY.value == "BUY"
        assert Direction.SELL.value == "SELL"


# ---------------------------------------------------------------------------
# MarketContext
# ---------------------------------------------------------------------------

class TestMarketContext:
    def test_minimum_fields(self):
        ctx = MarketContext(symbol="EURUSD", current_time_msc=123)
        assert ctx.symbol == "EURUSD"
        assert ctx.current_time_msc == 123
        assert ctx.htf_bias is None
        assert ctx.spread_pts == 0.0
        assert ctx.session is None

    def test_optional_fields_propagate(self):
        ctx = MarketContext(
            symbol="GBPJPY", current_time_msc=999,
            htf_bias="BULLISH", spread_pts=2.5, session="LONDON",
        )
        assert ctx.htf_bias == "BULLISH"
        assert ctx.spread_pts == 2.5
        assert ctx.session == "LONDON"

    def test_is_frozen(self):
        ctx = MarketContext(symbol="EURUSD", current_time_msc=0)
        with pytest.raises(Exception):  # FrozenInstanceError subclasses AttributeError
            ctx.symbol = "FOO"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PatternSignal validation
# ---------------------------------------------------------------------------

def _buy(entry=1.10, sl=1.09, tp=1.13, **kw) -> PatternSignal:
    defaults = dict(
        pattern_name="X", symbol="EURUSD", direction=Direction.BUY,
        entry=entry, sl=sl, tp=tp, confidence=0.7, grade=Grade.A,
        confluences_met=("c1", "c2"), bar_time_msc=1000,
    )
    defaults.update(kw)
    return PatternSignal(**defaults)


def _sell(entry=1.10, sl=1.11, tp=1.07, **kw) -> PatternSignal:
    defaults = dict(
        pattern_name="X", symbol="EURUSD", direction=Direction.SELL,
        entry=entry, sl=sl, tp=tp, confidence=0.7, grade=Grade.A,
        confluences_met=("c1",), bar_time_msc=1000,
    )
    defaults.update(kw)
    return PatternSignal(**defaults)


class TestPatternSignalValidation:
    def test_valid_buy(self):
        s = _buy()
        assert s.direction == Direction.BUY

    def test_valid_sell(self):
        s = _sell()
        assert s.direction == Direction.SELL

    def test_buy_requires_sl_below_entry(self):
        with pytest.raises(ValueError):
            _buy(entry=1.10, sl=1.11, tp=1.13)

    def test_buy_requires_tp_above_entry(self):
        with pytest.raises(ValueError):
            _buy(entry=1.10, sl=1.09, tp=1.08)

    def test_sell_requires_sl_above_entry(self):
        with pytest.raises(ValueError):
            _sell(entry=1.10, sl=1.09, tp=1.05)

    def test_sell_requires_tp_below_entry(self):
        with pytest.raises(ValueError):
            _sell(entry=1.10, sl=1.11, tp=1.15)

    def test_confidence_outside_range_rejected(self):
        with pytest.raises(ValueError):
            _buy(confidence=1.1)
        with pytest.raises(ValueError):
            _buy(confidence=-0.01)

    def test_negative_prices_rejected(self):
        with pytest.raises(ValueError):
            _buy(entry=-1.0)
        with pytest.raises(ValueError):
            _buy(sl=0)

    def test_confluences_list_cast_to_tuple(self):
        s = _buy(confluences_met=["a", "b", "c"])
        assert isinstance(s.confluences_met, tuple)
        assert s.confluences_met == ("a", "b", "c")


class TestPatternSignalGeometry:
    def test_risk_distance(self):
        s = _buy(entry=1.10, sl=1.09, tp=1.13)
        assert s.risk_distance == pytest.approx(0.01)

    def test_reward_distance(self):
        s = _buy(entry=1.10, sl=1.09, tp=1.13)
        assert s.reward_distance == pytest.approx(0.03)

    def test_rr_ratio_for_buy(self):
        s = _buy(entry=1.10, sl=1.09, tp=1.13)
        assert s.rr_ratio == pytest.approx(3.0)

    def test_rr_ratio_for_sell(self):
        s = _sell(entry=1.10, sl=1.11, tp=1.07)
        assert s.rr_ratio == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# PatternDetector ABC
# ---------------------------------------------------------------------------

class TestPatternDetectorAbstract:
    def test_cannot_instantiate_abstract_directly(self):
        with pytest.raises(TypeError):
            PatternDetector()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        class Dummy(PatternDetector):
            name = "DUMMY"
            min_bars_required = 5
            timeframe = "1H"

            def detect(
                self, bars: Sequence[Bar], context: MarketContext
            ) -> Optional[PatternSignal]:
                if len(bars) < self.min_bars_required:
                    return None
                return _buy()

        d = Dummy()
        assert d.name == "DUMMY"
        assert d.min_bars_required == 5
        # Below required → None
        assert d.detect([], MarketContext("EURUSD", 0)) is None
        # Above → signal
        bars = [Bar("EURUSD", i * 3_600_000, 1.0, 1.0, 1.0, 1.0, 1) for i in range(5)]
        s = d.detect(bars, MarketContext("EURUSD", 0))
        assert isinstance(s, PatternSignal)

    def test_detect_not_implemented_must_be_overridden(self):
        class Bad(PatternDetector):
            name = "BAD"
        with pytest.raises(TypeError):
            Bad()  # type: ignore[abstract]
