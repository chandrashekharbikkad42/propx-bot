"""Pattern detector framework — abstract contract + value objects.

A Griff pattern detector consumes a list of recent 1H bars + a MarketContext
and either emits a PatternSignal or returns None. Concrete patterns subclass
PatternDetector and implement `detect(bars, context)`.

Design choices:
  - Signal is FROZEN — once emitted, the scanner and risk engine treat it as
    a value, not a mutable container. Any new info means a new Signal.
  - Direction lives here as BUY/SELL (matches execution-layer Side semantics)
    rather than reusing strategy/signals/base.Direction (UP/DOWN), because
    pattern detectors output a trading intent — buy/sell is the natural verb.
  - Grade ranks A>B>C via `.rank`; the scanner uses this to compare combos
    of grade+confidence+R:R when picking the best signal across pairs.
  - PatternSignal validates basic price ordering (BUY: sl<entry<tp, SELL flipped)
    so a malformed detector can't poison the scanner downstream.

Hinglish: ek pattern ek "kya karu?" lekar aata hai. Direction, kahaan ghusna,
SL kahaan, TP kahaan, kitna confident, aur grade. Scanner sabko dekh ke best
pick karta hai.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

from data.bar_aggregator import Bar


class Grade(str, Enum):
    """Signal quality grade. A = all confluences, B = strong (1 missing),
    C = weak (2+ missing) — C never gets traded, only logged for analytics."""
    A = "A"
    B = "B"
    C = "C"

    @property
    def rank(self) -> int:
        """Higher is better. A=2, B=1, C=0."""
        return {"A": 2, "B": 1, "C": 0}[self.value]


class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class MarketContext:
    """Per-scan context passed to every detector.

    `htf_bias` is optional — set by an upstream higher-timeframe filter
    (Phase 8C-Patterns) when Griff's rules require it. Detectors that don't
    care simply ignore it.

    Multi-timeframe support (propX Multi-Setup, Phase 2):
      `htf_bars` and `ltf_bars` are optional secondary bar feeds. Single-TF
      detectors (e.g. AsianSweep) ignore them and consume the `bars` arg
      passed directly to `detect()`. Multi-TF detectors expect:
        - the `bars` arg = primary feed for the detector's `timeframe`
          (defaults to "1H" matching AsianSweep; multi-setup overrides to "15M"
          since the trigger is on LTF)
        - `context.htf_bars` = the higher-timeframe feed (1H) for structure
        - `context.ltf_bars` = redundant pointer to the LTF feed (15M) for
          callers that prefer reading via context rather than the `bars` arg.
      Both default to None; populated only when the orchestrator runs the
      multi-setup scanner. Adding here keeps the PatternDetector signature
      unchanged so AsianSweep continues to compile and run.
    """
    symbol: str
    current_time_msc: int
    htf_bias: Optional[str] = None       # "BULLISH" / "BEARISH" / "NEUTRAL" / None
    spread_pts: float = 0.0
    session: Optional[str] = None        # propagated from utils.session if known
    htf_bars: Optional[Sequence[Bar]] = None  # 1H feed (multi-setup only)
    ltf_bars: Optional[Sequence[Bar]] = None  # 15M feed (multi-setup only)


@dataclass(frozen=True)
class PatternSignal:
    """Frozen output of a pattern detector. Scanner ranks, risk engine sizes.

    Invariants enforced in __post_init__:
      - 0 <= confidence <= 1
      - entry / sl / tp all positive
      - BUY: sl < entry < tp
      - SELL: tp < entry < sl
      - confluences_met is a tuple (caller may pass list; we cast)
    """
    pattern_name: str
    symbol: str
    direction: Direction
    entry: float
    sl: float
    tp: float
    confidence: float                  # in [0, 1]
    grade: Grade
    confluences_met: tuple[str, ...]
    bar_time_msc: int                  # open time of the bar that triggered

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0,1], got {self.confidence}"
            )
        if self.entry <= 0 or self.sl <= 0 or self.tp <= 0:
            raise ValueError("entry, sl, tp must all be positive")
        # Cast confluences_met to a tuple if caller passed a list.
        if not isinstance(self.confluences_met, tuple):
            object.__setattr__(
                self, "confluences_met", tuple(self.confluences_met)
            )
        # Direction-aware price ordering.
        if self.direction == Direction.BUY:
            if not (self.sl < self.entry < self.tp):
                raise ValueError(
                    f"BUY signal requires sl<entry<tp; "
                    f"got sl={self.sl} entry={self.entry} tp={self.tp}"
                )
        else:  # SELL
            if not (self.tp < self.entry < self.sl):
                raise ValueError(
                    f"SELL signal requires tp<entry<sl; "
                    f"got sl={self.sl} entry={self.entry} tp={self.tp}"
                )

    @property
    def risk_distance(self) -> float:
        """Absolute price distance from entry to SL (positive)."""
        return abs(self.entry - self.sl)

    @property
    def reward_distance(self) -> float:
        """Absolute price distance from entry to TP (positive)."""
        return abs(self.tp - self.entry)

    @property
    def rr_ratio(self) -> float:
        """Reward / risk ratio. 0.0 if risk_distance is 0 (defensive)."""
        return self.reward_distance / self.risk_distance if self.risk_distance > 0 else 0.0


class PatternDetector(ABC):
    """Subclass and implement `detect`. Class attrs declare metadata.

    Subclasses MUST override:
      - name              (str)        — short id for logging / dashboards
      - min_bars_required (int)        — how many bars before detector is meaningful
    They MAY override:
      - timeframe         (str, default "1H")
    """
    name: str = "BASE"
    min_bars_required: int = 1
    timeframe: str = "1H"

    @abstractmethod
    def detect(
        self, bars: Sequence[Bar], context: MarketContext
    ) -> Optional[PatternSignal]:
        """Return a PatternSignal or None. MUST NOT mutate `bars` or `context`."""
        raise NotImplementedError
