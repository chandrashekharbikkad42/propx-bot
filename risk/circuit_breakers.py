"""Stateful trading kill-switch.

Three independent gates, ANY of which blocks `can_trade`:
  - Daily loss cap: stop trading if today's realised PnL <= -daily_cap_pct of starting equity.
  - Loss streak: after N consecutive losses, pause for `pause_minutes`.
  - Session filter: only trade LONDON / LONDON_NY_OVERLAP / NY.

Day rollover (UTC) resets the daily counters and the loss streak.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Tuple

from execution.position import Position
from utils.session import SessionLabel, session_for_msc


_ACTIVE_SESSIONS = {
    SessionLabel.LONDON,
    SessionLabel.LONDON_NY_OVERLAP,
    SessionLabel.NY,
}


@dataclass
class CircuitBreakerState:
    daily_pnl_usd: float = 0.0
    daily_starting_equity: float = 0.0
    consecutive_losses: int = 0
    streak_pause_until_msc: int = 0
    daily_cap_hit: bool = False
    current_day_utc: Optional[str] = None  # YYYY-MM-DD


class CircuitBreakers:
    def __init__(
        self,
        daily_cap_pct: float = 0.02,
        streak_threshold: int = 3,
        pause_minutes: int = 30,
    ) -> None:
        self._daily_cap_pct = daily_cap_pct
        self._streak_threshold = streak_threshold
        self._pause_ms = pause_minutes * 60 * 1000
        self.state = CircuitBreakerState()

    # ------------------------------------------------------------- public API

    def can_trade(
        self, current_time_msc: int, account_equity: float
    ) -> Tuple[bool, str]:
        self._roll_day_if_needed(current_time_msc, account_equity)

        if self.state.daily_cap_hit:
            return False, "daily_cap_hit"

        if current_time_msc < self.state.streak_pause_until_msc:
            return False, "loss_streak_pause"

        session = session_for_msc(current_time_msc)
        if session not in _ACTIVE_SESSIONS:
            return False, f"session_blocked:{session.value}"

        return True, "ok"

    def record_trade_close(self, position: Position) -> None:
        pnl = position.pnl_usd or 0.0
        self.state.daily_pnl_usd += pnl

        if pnl < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= self._streak_threshold:
                # Pause from the close timestamp — gives the cooldown
                # a deterministic anchor regardless of when the next tick lands.
                close_msc = position.exit_time_msc or position.entry_time_msc
                self.state.streak_pause_until_msc = close_msc + self._pause_ms
        else:
            self.state.consecutive_losses = 0

        cap_usd = self.state.daily_starting_equity * self._daily_cap_pct
        if cap_usd > 0 and self.state.daily_pnl_usd <= -cap_usd:
            self.state.daily_cap_hit = True

    # ---------------------------------------------------------------- helpers

    def _roll_day_if_needed(self, current_time_msc: int, account_equity: float) -> None:
        day = datetime.fromtimestamp(current_time_msc / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        if self.state.current_day_utc != day:
            self.state = CircuitBreakerState(
                daily_pnl_usd=0.0,
                daily_starting_equity=account_equity,
                consecutive_losses=0,
                streak_pause_until_msc=0,
                daily_cap_hit=False,
                current_day_utc=day,
            )
