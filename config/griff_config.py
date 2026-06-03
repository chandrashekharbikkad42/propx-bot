"""Griff strategy configuration constants — Phase 8C.

Single source of truth for the magic numbers the trailing-SL + spread-hour
machinery needs. Future calibration / alt-broker tuning lives here.

SPREAD_WIDEN_PIPS:
  Per-pair amount (in standard forex pips — 0.0001 for non-JPY pairs, 0.01
  for JPY pairs) to widen the trailing SL during the broker rollover window.
  Calibrated from observed mean spread blowout around 21:00 UTC ± 30 min on
  retail brokers; values larger for JPY crosses where blowout is worst.

ROLLOVER_UTC:
  HH:MM string. The "5 PM EST" convention is 21:00 UTC during EST (Nov-Mar)
  and 20:00 UTC during EDT (Mar-Nov). We pin 21:00 as a single-value default
  — the wide protection band absorbs the DST shift.

protection_window_before_min / protection_window_after_min:
  Minutes BEFORE rollover when widening starts (default 15) and minutes AFTER
  when it reverts (default 60).

trail_offset_pips:
  Pips of safety BELOW (long) / ABOVE (short) the structural swing the trail
  is anchored to. Griff calls 2 pips; tweak per-broker if noise demands more.

Hinglish: yeh config saare Griff knobs ek jagah rakhta hai — kabhi naye
broker pe spread blowout zyada lage to bas yahaan tune kar do, baaki sab
modules ko chhua nahi jaata.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping
from types import MappingProxyType


_DEFAULT_SPREAD_WIDEN_PIPS: Mapping[str, int] = MappingProxyType(
    {
        "AUDJPY": 50,
        "AUDUSD": 45,
        "EURJPY": 55,
        "EURUSD": 40,
        "GBPUSD": 45,
        "NZDJPY": 60,
    }
)


@dataclass(frozen=True)
class GriffConfig:
    spread_widen_pips: Mapping[str, int] = field(
        default_factory=lambda: _DEFAULT_SPREAD_WIDEN_PIPS
    )
    rollover_utc: str = "21:00"
    protection_window_before_min: int = 15
    protection_window_after_min: int = 60
    trail_offset_pips: float = 2.0
    default_widen_pips: int = 50  # for pairs missing from the map

    def widen_pips_for(self, pair: str) -> int:
        """Per-pair widen amount; falls back to `default_widen_pips`."""
        return self.spread_widen_pips.get(pair.upper(), self.default_widen_pips)
