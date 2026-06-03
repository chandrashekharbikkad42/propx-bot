"""Pre-trade compliance engine (Phase 8C foundation).

Wraps the 7 hard kill-switches the user spec calls out:
  1. Within IST trade window?
  2. Daily loss < 80% of cap? (early-stop margin)
  3. Total loss < 80% of cap?
  4. Trade count < max_trades_per_day?
  5. Not in news blackout?
  6. SL distance respects max_loss (signal won't single-handedly breach daily cap)?
  7. Position size within leverage limits?

`can_trade(signal, current_time_ms)` returns (allowed: bool, reason: str).
Reason starts with "ok" when allowed; otherwise a short snake_case code.

The engine is STATEFUL only for emergency_stop + status reporting; the
real account state (equity, daily PnL, trades count) is passed in as a
dataclass per check — that keeps the engine pure and easy to test, and
lets the orchestrator own the account state lifecycle.

Hinglish: yeh sare hard rules dekhta hai — agar koi bhi violate, signal
reject. Compliance se panga = funded account gone, isliye 80% me hi ruk
jaate hain — full cap tak jaane ka risk hi nahi lete.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

from data.news_calendar import NewsCalendar, StaticNewsCalendar
from risk.prop_firm.rules import PropFirmRules
from strategy.patterns.base import Direction, PatternSignal
from utils.session import is_within_ist_window


# Early-stop margin — we halt at 80% of any cap so a final-bar slip can't
# tip us over. Single source of truth.
SAFETY_MARGIN_PCT = 0.80


@dataclass(frozen=True)
class AccountState:
    """Snapshot of account at the moment a check is made."""
    equity: float
    starting_equity: float            # account-start, for total-loss cap
    daily_start_equity: float         # equity at IST 00:00 today
    daily_pnl_usd: float              # realised PnL today
    trades_today: int                 # 0, 1, 2 (cap 2 per Griff spec)
    open_position_count: int = 0


class ComplianceEngine:
    """Pre-trade gate. One instance per running bot, holds rules + emergency
    stop flag. All checks are pure on `AccountState` snapshots."""

    def __init__(
        self,
        rules: PropFirmRules,
        max_trades_per_day: int = 2,
        ist_window_start: str = "12:30",
        ist_window_end: str = "22:30",
        news_calendar: Optional[NewsCalendar] = None,
        safety_margin_pct: float = SAFETY_MARGIN_PCT,
    ) -> None:
        if not (0.0 < safety_margin_pct <= 1.0):
            raise ValueError("safety_margin_pct must be in (0, 1]")
        if max_trades_per_day < 1:
            raise ValueError("max_trades_per_day must be >= 1")
        self._rules = rules
        self._max_trades = max_trades_per_day
        self._win_start = ist_window_start
        self._win_end = ist_window_end
        self._news: NewsCalendar = news_calendar or StaticNewsCalendar([])
        self._margin = safety_margin_pct
        self._emergency_stopped: bool = False
        self._emergency_reason: Optional[str] = None

    # ---------------------------------------------------------- public API

    @property
    def rules(self) -> PropFirmRules:
        return self._rules

    @property
    def emergency_stopped(self) -> bool:
        return self._emergency_stopped

    @property
    def emergency_reason(self) -> Optional[str]:
        return self._emergency_reason

    def emergency_stop(self, reason: str) -> None:
        """Latch the engine into a permanent block until cleared.

        Caller should also force-close positions externally — the engine
        itself doesn't touch the broker.
        """
        self._emergency_stopped = True
        self._emergency_reason = reason

    def clear_emergency(self) -> None:
        """Manual reset — used after operator review."""
        self._emergency_stopped = False
        self._emergency_reason = None

    def can_trade(
        self,
        signal: PatternSignal,
        current_time_msc: int,
        account: AccountState,
        lots: float = 1.0,
        contract_size: float = 100_000.0,
    ) -> Tuple[bool, str]:
        """Run all 7 checks in cheap-to-expensive order. Returns (ok, reason)."""
        if self._emergency_stopped:
            return False, f"emergency_stop:{self._emergency_reason or ''}"

        # 1. IST window
        if not is_within_ist_window(current_time_msc, self._win_start, self._win_end):
            return False, "outside_ist_window"

        # 2. Daily loss cap (with safety margin)
        daily_cap_usd = account.daily_start_equity * (self._rules.max_daily_loss_pct / 100.0)
        if -account.daily_pnl_usd >= daily_cap_usd * self._margin:
            return False, "daily_loss_near_cap"

        # 3. Total loss cap (account drawdown vs starting equity)
        total_loss_usd = account.starting_equity - account.equity
        total_cap_usd = account.starting_equity * (self._rules.max_total_loss_pct / 100.0)
        if total_loss_usd >= total_cap_usd * self._margin:
            return False, "total_loss_near_cap"

        # 4. Max trades per day
        if account.trades_today >= self._max_trades:
            return False, "daily_trade_cap_reached"

        # 5. News blackout
        if self._news.is_news_blackout(
            signal.symbol, current_time_msc,
            window_min=max(
                self._rules.news_blackout_minutes_before,
                self._rules.news_blackout_minutes_after,
            ),
        ):
            return False, "news_blackout"

        # 6. SL respects max-loss — would this signal's worst-case eat
        #    more than the *remaining* daily cap? We treat the SL distance
        #    as the worst-case loss in pips × usd_per_pip_per_lot × lots.
        #    Without a per-pair pip-value table here we approximate using
        #    risk_distance × contract_size × lots as a USD loss estimate.
        worst_loss = signal.risk_distance * contract_size * lots
        remaining_daily_room = (daily_cap_usd * self._margin) + account.daily_pnl_usd
        if worst_loss > remaining_daily_room:
            return False, "sl_exceeds_remaining_daily_room"

        # 7. Position size within leverage cap
        # leverage = (notional / equity). notional = price × contract × lots.
        # We use signal.entry as the price.
        notional = signal.entry * contract_size * lots
        max_leverage = self._rules.leverage_forex  # caller picks metals where needed
        if account.equity > 0 and (notional / account.equity) > max_leverage:
            return False, "exceeds_leverage_cap"

        return True, "ok"

    def get_status_report(
        self, account: AccountState, current_time_msc: int
    ) -> dict:
        """Snapshot for dashboards / Telegram alerts. No mutation."""
        daily_cap_usd = account.daily_start_equity * (self._rules.max_daily_loss_pct / 100.0)
        total_cap_usd = account.starting_equity * (self._rules.max_total_loss_pct / 100.0)
        return {
            "firm": self._rules.name,
            "equity": account.equity,
            "starting_equity": account.starting_equity,
            "daily_pnl_usd": account.daily_pnl_usd,
            "daily_cap_usd": daily_cap_usd,
            "daily_used_pct": (
                (-account.daily_pnl_usd / daily_cap_usd * 100.0)
                if daily_cap_usd > 0 else 0.0
            ),
            "total_loss_usd": account.starting_equity - account.equity,
            "total_cap_usd": total_cap_usd,
            "trades_today": account.trades_today,
            "max_trades_per_day": self._max_trades,
            "in_ist_window": is_within_ist_window(
                current_time_msc, self._win_start, self._win_end
            ),
            "emergency_stopped": self._emergency_stopped,
            "emergency_reason": self._emergency_reason,
        }
