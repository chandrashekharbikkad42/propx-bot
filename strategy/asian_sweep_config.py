"""Asian Sweep — detector factory.

Mirrors `strategy.patterns.build_griff_detectors`: a tiny zero-arg builder
the live engine wires up. Keeping the factory in `strategy/` (not in
`strategy/patterns/__init__.py`) so the Asian Sweep wiring stays separate
from the existing Griff bundle until Phase 4 deletes Griff.

Hinglish: ek banane wala — Scanner ko detector tuple dena hai bas.
"""

from __future__ import annotations

from strategy.patterns.asian_sweep import AsianSweepDetector
from strategy.patterns.base import PatternDetector


def build_asian_sweep_detector() -> tuple[PatternDetector, ...]:
    """Canonical Asian Sweep detector set — currently just the V5 sweep.

    Returned as a tuple to keep the Scanner constructor signature happy
    (it accepts any Sequence[PatternDetector]) and to leave room for
    additional pattern siblings later without changing call sites.
    """
    return (AsianSweepDetector(),)


__all__ = ["AsianSweepDetector", "build_asian_sweep_detector"]
