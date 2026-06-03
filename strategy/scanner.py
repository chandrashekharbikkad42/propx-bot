"""Multi-pair × multi-pattern probability scanner (Phase 8C).

Drives every registered pattern detector across every configured pair on
each new 1H bar close. Outputs ranked PatternSignals. The orchestrator
asks for the single best signal — scanner picks the highest-grade /
highest-confidence one and silently drops C-grade.

The scanner is STATELESS between scans: feed it the current bar-feeds dict
and it returns the result. `last_signals` is exposed for diagnostics only.

Ranking order (highest first):
  1. Grade rank (A > B; C filtered out entirely)
  2. Confidence
  3. R:R ratio (as tiebreaker — bigger payoff per unit risk wins)

Hinglish: sab pair sab pattern dekho, sab signals collect karo, C-grade
phenk do, baaki me sabse strong ek nikalo. Bot us ek ko trade karta hai.
"""

from __future__ import annotations
from typing import Dict, List, Mapping, Optional, Sequence

from data.bar_aggregator import Bar
from strategy.patterns.base import (
    Grade,
    MarketContext,
    PatternDetector,
    PatternSignal,
)


class Scanner:
    def __init__(
        self,
        pairs: Sequence[str],
        patterns: Sequence[PatternDetector],
    ) -> None:
        if not pairs:
            raise ValueError("Scanner requires at least one pair")
        if not patterns:
            raise ValueError("Scanner requires at least one pattern")
        self._pairs: tuple[str, ...] = tuple(pairs)
        self._patterns: tuple[PatternDetector, ...] = tuple(patterns)
        self._last_signals: tuple[PatternSignal, ...] = ()
        self._last_c_dropped: int = 0
        self._last_skipped_insufficient_bars: int = 0

    # ---------------------------------------------------------- public API

    @property
    def pairs(self) -> tuple[str, ...]:
        return self._pairs

    @property
    def patterns(self) -> tuple[PatternDetector, ...]:
        return self._patterns

    @property
    def last_signals(self) -> tuple[PatternSignal, ...]:
        return self._last_signals

    @property
    def c_grade_dropped(self) -> int:
        """Count of C-grade signals filtered out in the last scan."""
        return self._last_c_dropped

    @property
    def skipped_insufficient_bars(self) -> int:
        """Count of pattern×pair pairs skipped because bars < min_bars_required."""
        return self._last_skipped_insufficient_bars

    def scan_all(
        self,
        bar_feeds: Mapping[str, Sequence[Bar]],
        current_time_msc: int,
        context_overrides: Optional[Mapping[str, MarketContext]] = None,
    ) -> tuple[PatternSignal, ...]:
        """Run every pattern against every pair. Returns ALL signals (any grade).

        `bar_feeds`: pair → list of bars (most recent last).
        `context_overrides`: optional per-pair MarketContext (e.g. with htf_bias
        pre-computed); defaults to a fresh MarketContext per pair otherwise.
        """
        all_sigs: list[PatternSignal] = []
        skipped = 0
        for pair in self._pairs:
            bars = bar_feeds.get(pair) or ()
            if not bars:
                continue
            if context_overrides and pair in context_overrides:
                ctx = context_overrides[pair]
            else:
                ctx = MarketContext(
                    symbol=pair, current_time_msc=current_time_msc
                )
            for p in self._patterns:
                if len(bars) < p.min_bars_required:
                    skipped += 1
                    continue
                sig = p.detect(bars, ctx)
                if sig is not None:
                    all_sigs.append(sig)

        self._last_signals = tuple(all_sigs)
        self._last_c_dropped = sum(1 for s in all_sigs if s.grade == Grade.C)
        self._last_skipped_insufficient_bars = skipped
        return self._last_signals

    def get_best_signal(self) -> Optional[PatternSignal]:
        """Best tradeable signal from the last scan; None if only C-grade or empty."""
        tradeable = [s for s in self._last_signals if s.grade != Grade.C]
        if not tradeable:
            return None
        # Sort descending by (grade rank, confidence, rr_ratio).
        tradeable.sort(
            key=lambda s: (s.grade.rank, s.confidence, s.rr_ratio),
            reverse=True,
        )
        return tradeable[0]

    def signals_by_grade(self) -> Dict[Grade, list[PatternSignal]]:
        """Group last scan's signals by grade. Useful for the dashboard."""
        out: Dict[Grade, list[PatternSignal]] = {g: [] for g in Grade}
        for s in self._last_signals:
            out[s.grade].append(s)
        return out
