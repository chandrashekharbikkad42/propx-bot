"""Griff-specific Telegram alert formatters.

Wraps the existing `alerts.TelegramNotifier` (which handles transport,
rate limits, and failure-safe sends) with formatters for Griff event
types. Reuses the underlying notifier's `send()`; never opens its own
HTTP session.

Event types covered:
  - signal_detected     : scanner emitted a tradeable PatternSignal
  - trade_opened        : market fill or pending fill turned into a position
  - trade_closed        : SL/TP/manual close
  - kill_switch         : compliance gate rejected (or emergency stop)
  - daily_summary       : end-of-session digest

All formatters degrade to a no-op (return False) if the notifier is
disabled — never raise, never block.

Hinglish: Griff ka apna alerts wrapper. Send karne ka kaam pehla wala
TelegramNotifier hi karta hai; yeh sirf message ka shape banata hai.
"""

from __future__ import annotations
from typing import Optional, Sequence

from alerts.telegram_notifier import TelegramNotifier
from execution.order_router import GriffOpenPosition
from monitoring.daily_tracker import DailyState
from strategy.patterns.base import Direction, PatternSignal


class GriffTelegramAlerts:
    def __init__(
        self,
        notifier: TelegramNotifier,
        *,
        bot_label: str = "propX",
    ) -> None:
        """`bot_label` is interpolated into the bot_started / bot_stopped
        envelope (e.g. "propX" → "propX Bot LIVE TRADING", "AsianSweep"
        → "AsianSweep Bot DRY_RUN"). Defaults to "propX" — the current
        bot identity. The Asian Sweep entry point overrides this to
        "AsianSweep".
        """
        self._n = notifier
        self._label = bot_label

    @property
    def enabled(self) -> bool:
        return self._n.enabled

    @property
    def bot_label(self) -> str:
        return self._label

    # ----- formatters

    async def signal_detected(self, signal: PatternSignal) -> bool:
        msg = (
            f"<b>SIGNAL</b> {signal.pattern_name} {signal.symbol} "
            f"{signal.direction.value}\n"
            f"grade={signal.grade.value} conf={signal.confidence:.2f}\n"
            f"entry={signal.entry:.5f}  sl={signal.sl:.5f}"
        )
        return await self._n.send(msg)

    async def trade_opened(
        self, position: GriffOpenPosition, *, lots: float,
    ) -> bool:
        msg = (
            f"<b>TRADE OPEN</b> {position.pattern_name} {position.symbol} "
            f"{position.side.value}\n"
            f"entry={position.entry_price:.5f}  lots={lots:g}\n"
            f"sl={position.sl_price:.5f}"
        )
        return await self._n.send(msg)

    async def trade_closed(
        self, position: GriffOpenPosition, *, exit_price: float,
        pnl_usd: float, reason: str = "SL",
    ) -> bool:
        sign = "+" if pnl_usd >= 0 else "-"
        msg = (
            f"<b>TRADE CLOSED</b> {position.pattern_name} {position.symbol} "
            f"{position.side.value} ({reason})\n"
            f"exit={exit_price:.5f}  pnl={sign}${abs(pnl_usd):.2f}"
        )
        return await self._n.send(msg)

    async def kill_switch_triggered(self, reason: str) -> bool:
        return await self._n.send(
            f"<b>KILL SWITCH</b> trading halted: {reason}"
        )

    async def daily_summary(self, state: DailyState) -> bool:
        sign = "+" if state.closed_pnl >= 0 else "-"
        msg = (
            f"<b>DAILY SUMMARY</b> {state.trade_day}\n"
            f"trades={state.trade_count}  "
            f"closed_pnl={sign}${abs(state.closed_pnl):.2f}\n"
            f"max_dd_today=${state.max_dd_today:.2f}  "
            f"peak_equity=${state.peak_equity:.2f}"
        )
        return await self._n.send(msg)

    async def bot_started(
        self, *, dry_run: bool, pairs: Sequence[str],
        broker_name: Optional[str] = None,
        prop_firm_key: Optional[str] = None,
        account_balance: Optional[float] = None,
        account_currency: Optional[str] = None,
    ) -> bool:
        """Rich startup message. Extra fields are optional — when omitted,
        the message gracefully degrades to the original single-line form so
        DRY_RUN bring-ups and tests without a live broker still work.
        """
        if dry_run:
            mode_label = "DRY_RUN"
            status_emoji = "🟡"
            status_text = "DRY RUN — no orders will be placed"
        else:
            mode_label = "LIVE TRADING"
            status_emoji = "🟢"
            status_text = "OPERATIONAL"

        lines: list[str] = [f"🤖 <b>{self._label} Bot {mode_label}</b>"]
        if broker_name:
            if prop_firm_key:
                lines.append(
                    f"Broker: {broker_name} ({prop_firm_key} detected)"
                )
            else:
                lines.append(f"Broker: {broker_name}")
        if account_balance is not None and account_currency:
            lines.append(
                f"Account: ${account_balance:,.2f} {account_currency}"
            )
        lines.append(f"Pairs: {','.join(pairs)}")
        lines.append(f"Mode: {mode_label}")
        lines.append(f"Status: {status_emoji} {status_text}")
        return await self._n.send("\n".join(lines))

    async def bot_stopped(self, reason: str = "graceful") -> bool:
        return await self._n.send(
            f"<b>{self._label.upper()} BOT STOPPED</b> reason={reason}"
        )
