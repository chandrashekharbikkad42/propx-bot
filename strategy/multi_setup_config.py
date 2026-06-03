"""propX Multi-Setup — detector factory + confluence resolver.

Sibling to `strategy/asian_sweep_config.py`. Live engine wires up the
4-setup bundle by calling `build_multi_setup_detectors()`.

Confluence (spec §6) is resolved AFTER the scanner emits raw signals — the
detectors are independent. Use `resolve_confluence(signals)` to collapse
overlapping same-direction signals on the same pair into a single
high-confidence signal, and to discard conflicting opposite-direction ones.

Hinglish: ek factory chaaron detector deta hai. Confluence resolver alag se
chalata hai — orchestrator scanner ke output pe.
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple

from strategy.patterns.base import (
    Direction, Grade, PatternDetector, PatternSignal,
)
from strategy.patterns.break_of_structure import BreakOfStructureDetector
from strategy.patterns.liquidity_sweep import LiquiditySweepDetector
from strategy.patterns.order_block import OrderBlockDetector
from strategy.patterns.sr_rejection import SRRejectionDetector
from config.multi_setup_config import (
    CONFLUENCE_BAR_TOL_LTF,
    CONFLUENCE_CONFIDENCE_BOOST,
    CONFLUENCE_PRICE_TOL_PIPS,
    pip_size_for, setup_rank_for,
)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_multi_setup_detectors() -> Tuple[PatternDetector, ...]:
    """Canonical bundle — 4 detectors in spec-rank order (highest first).

    Returned as a tuple so callers can pass directly to the Scanner. The
    bundle is stateless; safe to share across pairs / threads.
    """
    return (
        BreakOfStructureDetector(),
        OrderBlockDetector(),
        LiquiditySweepDetector(),
        SRRejectionDetector(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Confluence resolver (spec §6)
# ─────────────────────────────────────────────────────────────────────────────

# 15M bar duration in milliseconds — used to compare bar_time_msc deltas
# against CONFLUENCE_BAR_TOL_LTF in units of bars.
_LTF_BAR_MS: int = 15 * 60 * 1000


def _within_confluence_window(a: PatternSignal, b: PatternSignal) -> bool:
    """True if two signals on the same symbol fall within the spec §6
    price + bar tolerance windows."""
    if a.symbol != b.symbol:
        return False
    price_tol = CONFLUENCE_PRICE_TOL_PIPS * pip_size_for(a.symbol)
    if abs(a.entry - b.entry) > price_tol:
        return False
    bar_delta = abs(a.bar_time_msc - b.bar_time_msc)
    if bar_delta > CONFLUENCE_BAR_TOL_LTF * _LTF_BAR_MS:
        return False
    return True


def _merge_confluence(group: List[PatternSignal]) -> PatternSignal:
    """Collapse a confluent group into one signal (spec §6).

    Use the EARLIEST signal's entry/SL/TP (it triggered first; subsequent
    setups merely confirm). Confidence = min(1.0, max(child confidences) +
    CONFLUENCE_CONFIDENCE_BOOST). Grade promotes to A if any child is A.
    """
    group.sort(key=lambda s: s.bar_time_msc)
    primary = group[0]
    max_conf = max(s.confidence for s in group)
    boosted = min(1.0, max_conf + CONFLUENCE_CONFIDENCE_BOOST)
    grade = Grade.A if any(s.grade == Grade.A for s in group) else primary.grade
    tags = (
        "confluence",
        *sorted({f"setup_{s.pattern_name}" for s in group}),
        f"n_confluent_{len(group)}",
        *primary.confluences_met,
    )
    return PatternSignal(
        pattern_name=f"CONFLUENCE_{'+'.join(sorted({s.pattern_name for s in group}))}",
        symbol=primary.symbol,
        direction=primary.direction,
        entry=primary.entry, sl=primary.sl, tp=primary.tp,
        confidence=boosted,
        grade=grade,
        confluences_met=tuple(tags),
        bar_time_msc=primary.bar_time_msc,
    )


def resolve_confluence(
    signals: Iterable[PatternSignal],
) -> Tuple[PatternSignal, ...]:
    """Apply spec §6 confluence + conflict resolution to raw scanner output.

    Steps per pair:
      1. Discard pairs with conflicting opposite-direction signals within
         the confluence window (`reason='conflict'` is dropped silently).
      2. Group remaining same-direction signals that fall within the
         price+bar window; merge each group into one signal.
      3. Leave isolated signals untouched.

    Returns a new tuple. Original signals are not mutated.
    """
    by_symbol: dict[str, List[PatternSignal]] = {}
    for s in signals:
        by_symbol.setdefault(s.symbol, []).append(s)

    out: List[PatternSignal] = []
    for sym, sigs in by_symbol.items():
        # Step 1 — conflict detection (opposite directions, within window).
        conflict_ids: set[int] = set()
        for i, a in enumerate(sigs):
            for j in range(i + 1, len(sigs)):
                b = sigs[j]
                if a.direction != b.direction and _within_confluence_window(a, b):
                    conflict_ids.add(id(a))
                    conflict_ids.add(id(b))
        survivors = [s for s in sigs if id(s) not in conflict_ids]

        # Step 2 — group same-direction confluent clusters (greedy).
        used: set[int] = set()
        for i, a in enumerate(survivors):
            if id(a) in used:
                continue
            group = [a]
            used.add(id(a))
            for j in range(i + 1, len(survivors)):
                b = survivors[j]
                if id(b) in used:
                    continue
                if a.direction == b.direction and _within_confluence_window(a, b):
                    group.append(b)
                    used.add(id(b))
            if len(group) >= 2:
                out.append(_merge_confluence(group))
            else:
                out.append(group[0])

    # Final ordering — by (grade rank, confidence, setup rank) desc, deterministic.
    out.sort(
        key=lambda s: (
            s.grade.rank, s.confidence,
            setup_rank_for(s.pattern_name.replace("CONFLUENCE_", "").split("+")[0]),
        ),
        reverse=True,
    )
    return tuple(out)


__all__ = [
    "build_multi_setup_detectors",
    "resolve_confluence",
    "LiquiditySweepDetector",
    "OrderBlockDetector",
    "BreakOfStructureDetector",
    "SRRejectionDetector",
]
