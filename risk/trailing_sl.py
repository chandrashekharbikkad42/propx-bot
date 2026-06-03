"""Mechanical swing-based trailing stop + spread-hour protection (Phase 8C).

Trail logic:
  Long  — query SwingTracker for latest swing low; target SL = swing_low
          - trail_offset_pips. Raise structural SL if target > current.
  Short — symmetric on swing high; lower structural SL if target < current.

The trail never moves UNFAVORABLY: a new lower swing low (during retracement)
does not pull a long's SL back down. Once structure is locked in, only better
structure replaces it.

Spread-hour protection:
  Around the broker rollover (default 21:00 UTC), bid/ask blows out for
  ~30-90 min. We WIDEN the SL by SPREAD_WIDEN_PIPS[pair] starting
  `protection_window_before_min` ahead and revert `protection_window_after_min`
  after. Widening is SUPPRESSED for positions already at or beyond break-even
  — no point handing a winner back.

State (per position_id):
  - structural_sl   : swing-derived SL, never includes widening
  - spread_active   : whether protection band is currently applied
  - pair            : cached on first sight (Position is symbol-less)

Position is frozen, so this module owns the trail history. `update()`
returns the new SL for the orchestrator to thread through to the broker.

Hinglish: long trade me higher swing-low banti hai to SL upar utha do, bas
2 pip safety. Rollover ke aas-paas spread fatti hai — temporary widen kar
lete hain, par profit me aaye trade ko chhua nahi jaata.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

from config.griff_config import GriffConfig
from data.bar_aggregator import Bar
from execution.order import Side
from execution.position import Position
from strategy.swing_tracker import SwingTracker


def pip_size(pair: str) -> float:
    """Standard forex pip — 0.01 for JPY pairs, 0.0001 otherwise.

    XAU/metals use different scales; caller must override for those. For
    Griff's currency universe this is sufficient.
    """
    return 0.01 if "JPY" in pair.upper() else 0.0001


@dataclass
class _Trail:
    """Per-position trailing state. Mutable — owned by TrailingStopLoss."""
    pair: str
    structural_sl: float
    spread_active: bool = False
    pre_spread_sl: Optional[float] = None  # captured at start of widening


class TrailingStopLoss:
    """Stateful trail manager. One instance per running bot; positions are
    keyed by position_id.

    Caller contract:
      - call SwingTracker.update(pair, bar) FIRST so the tracker has fresh
        swings for this bar.
      - then call trailing_sl.update(position, bar, current_time) for each
        open position on that pair.
      - if `update` returns a float, apply it as the new SL externally
        (Position is frozen — broker constructs a fresh one).
    """

    def __init__(
        self,
        swing_tracker: SwingTracker,
        config: Optional[GriffConfig] = None,
    ) -> None:
        self._tracker = swing_tracker
        self._cfg = config or GriffConfig()
        self._state: Dict[str, _Trail] = {}

    # --------------------------------------------------------------- main API

    def update(
        self,
        position: Position,
        bar: Bar,
        current_time: datetime,
    ) -> Optional[float]:
        """Apply structural trail + spread-hour protection.

        Returns the new SL price, or None if effectively unchanged.
        """
        pid = position.position_id
        pair = bar.symbol

        # Seed state on first sight of this position.
        if pid not in self._state:
            self._state[pid] = _Trail(pair=pair, structural_sl=position.sl_price)
        trail = self._state[pid]

        # 1) Structural trail from latest swings — favorable-only.
        offset = self._cfg.trail_offset_pips * pip_size(pair)
        if position.side == Side.BUY:
            anchor = self._tracker.get_last_swing_low(pair)
            if anchor is not None:
                target = anchor - offset
                if target > trail.structural_sl:
                    trail.structural_sl = target
        else:  # SELL
            anchor = self._tracker.get_last_swing_high(pair)
            if anchor is not None:
                target = anchor + offset
                if target < trail.structural_sl:
                    trail.structural_sl = target

        # 2) Spread-hour window — widen only if not in profit.
        in_window = self._in_protection_window(current_time)
        in_profit = self._is_in_profit(position, trail.structural_sl)

        if in_window and not in_profit:
            if not trail.spread_active:
                trail.spread_active = True
                trail.pre_spread_sl = trail.structural_sl
            widen = self._cfg.widen_pips_for(pair) * pip_size(pair)
            effective = (
                trail.structural_sl - widen
                if position.side == Side.BUY
                else trail.structural_sl + widen
            )
        else:
            if trail.spread_active:
                trail.spread_active = False
                trail.pre_spread_sl = None
            effective = trail.structural_sl

        # 3) Return new SL only on material change.
        if abs(effective - position.sl_price) < 1e-9:
            return None
        return effective

    # ------------------------------ explicit hooks for orchestrator usage

    def apply_spread_protection(
        self, position: Position, rollover_time: datetime
    ) -> float:
        """Force widening for a known position. Requires that `update` has
        already been called once on this position (pair must be cached).
        """
        pid = position.position_id
        if pid not in self._state:
            raise KeyError(
                f"Position {pid!r} not seen by update() yet — "
                f"pair unknown, cannot apply spread protection."
            )
        trail = self._state[pid]
        trail.spread_active = True
        trail.pre_spread_sl = trail.structural_sl
        widen = self._cfg.widen_pips_for(trail.pair) * pip_size(trail.pair)
        if position.side == Side.BUY:
            return trail.structural_sl - widen
        return trail.structural_sl + widen

    def revert_spread_protection(self, position: Position) -> float:
        """Restore the pre-widen structural SL. If protection wasn't active,
        returns position.sl_price unchanged.
        """
        pid = position.position_id
        trail = self._state.get(pid)
        if trail is None or not trail.spread_active:
            return position.sl_price
        trail.spread_active = False
        trail.pre_spread_sl = None
        return trail.structural_sl

    def _calc_widen_pips(self, pair: str) -> int:
        """Pip count this pair widens by during spread hour."""
        return self._cfg.widen_pips_for(pair)

    # ---------------------------------------------------------------- internals

    def _in_protection_window(self, current_time: datetime) -> bool:
        """True if `current_time` (UTC) is inside the rollover protection band.

        Window: [rollover - before_min, rollover + after_min). Built on the
        same calendar date as `current_time` — does not handle window
        wrap-around past midnight (default config never crosses).
        """
        hh, mm = (int(x) for x in self._cfg.rollover_utc.split(":"))
        rollover_today = current_time.replace(
            hour=hh, minute=mm, second=0, microsecond=0
        )
        start = rollover_today - timedelta(
            minutes=self._cfg.protection_window_before_min
        )
        end = rollover_today + timedelta(
            minutes=self._cfg.protection_window_after_min
        )
        return start <= current_time < end

    @staticmethod
    def _is_in_profit(position: Position, structural_sl: float) -> bool:
        """SL at or beyond entry — protection skipped to preserve locked PnL."""
        if position.side == Side.BUY:
            return structural_sl >= position.entry_price
        return structural_sl <= position.entry_price
