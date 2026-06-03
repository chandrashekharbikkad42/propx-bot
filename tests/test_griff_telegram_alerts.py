"""Phase 8D-Live — Griff Telegram alert wrapper tests."""

from __future__ import annotations
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from alerts.telegram_notifier import TelegramNotifier
from execution.order_router import GriffOpenPosition
from monitoring.daily_tracker import DailyState
from monitoring.telegram_alerts import GriffTelegramAlerts
from strategy.patterns.base import Direction, Grade, PatternSignal


def _signal() -> PatternSignal:
    return PatternSignal(
        pattern_name="FLAG", symbol="EURUSD", direction=Direction.BUY,
        entry=1.1000, sl=1.0980, tp=1.1040, confidence=0.8, grade=Grade.A,
        confluences_met=("a",), bar_time_msc=0,
    )


def _position() -> GriffOpenPosition:
    return GriffOpenPosition(
        position_id="p", mt5_ticket=1, symbol="EURUSD", side=Direction.BUY,
        lots=0.1, entry_price=1.10, sl_price=1.099, tp_price=1.102,
        opened_msc=0, signal_id="s", pattern_name="FLAG",
    )


def _make_notifier_with_send_mock():
    n = TelegramNotifier(token="x", chat_id="y")
    n.send = AsyncMock(return_value=True)
    return n


# ============================================================================
# Disabled notifier — formatters return False, never raise
# ============================================================================

class TestDisabledNotifier(unittest.TestCase):
    def test_disabled_signal_returns_false(self):
        n = TelegramNotifier(token=None, chat_id=None)
        a = GriffTelegramAlerts(n)
        self.assertFalse(a.enabled)
        ok = asyncio.run(a.signal_detected(_signal()))
        self.assertFalse(ok)


# ============================================================================
# Formatters — verify message shape via the underlying notifier mock
# ============================================================================

class TestFormatters(unittest.TestCase):
    def test_signal_detected_includes_pattern_and_grade(self):
        n = _make_notifier_with_send_mock()
        a = GriffTelegramAlerts(n)
        asyncio.run(a.signal_detected(_signal()))
        sent = n.send.call_args.args[0]
        self.assertIn("FLAG", sent)
        self.assertIn("EURUSD", sent)
        self.assertIn("grade=A", sent)

    def test_trade_opened_includes_entry_and_lots(self):
        n = _make_notifier_with_send_mock()
        a = GriffTelegramAlerts(n)
        asyncio.run(a.trade_opened(_position(), lots=0.1))
        sent = n.send.call_args.args[0]
        self.assertIn("TRADE OPEN", sent)
        self.assertIn("1.10000", sent)

    def test_trade_closed_positive_pnl(self):
        n = _make_notifier_with_send_mock()
        a = GriffTelegramAlerts(n)
        asyncio.run(a.trade_closed(_position(), exit_price=1.102,
                                   pnl_usd=20.0, reason="TP"))
        sent = n.send.call_args.args[0]
        self.assertIn("+$20.00", sent)
        self.assertIn("TP", sent)

    def test_trade_closed_negative_pnl(self):
        n = _make_notifier_with_send_mock()
        a = GriffTelegramAlerts(n)
        asyncio.run(a.trade_closed(_position(), exit_price=1.099,
                                   pnl_usd=-10.0, reason="SL"))
        sent = n.send.call_args.args[0]
        self.assertIn("-$10.00", sent)

    def test_kill_switch_message(self):
        n = _make_notifier_with_send_mock()
        a = GriffTelegramAlerts(n)
        asyncio.run(a.kill_switch_triggered("daily-loss-cap"))
        sent = n.send.call_args.args[0]
        self.assertIn("KILL SWITCH", sent)
        self.assertIn("daily-loss-cap", sent)

    def test_daily_summary_includes_counts(self):
        n = _make_notifier_with_send_mock()
        a = GriffTelegramAlerts(n)
        s = DailyState(
            trade_day="2026-05-17", peak_equity=10_100, closed_pnl=80.0,
            floating_pnl=20.0, trade_count=2, max_dd_today=25.0,
            last_update_ms=0,
        )
        asyncio.run(a.daily_summary(s))
        sent = n.send.call_args.args[0]
        self.assertIn("DAILY SUMMARY", sent)
        self.assertIn("2026-05-17", sent)
        self.assertIn("trades=2", sent)

    def test_bot_started_dry_run_label(self):
        n = _make_notifier_with_send_mock()
        a = GriffTelegramAlerts(n)
        asyncio.run(a.bot_started(dry_run=True, pairs=("EURUSD", "AUDJPY")))
        sent = n.send.call_args.args[0]
        self.assertIn("DRY_RUN", sent)
        self.assertIn("EURUSD,AUDJPY", sent)

    def test_bot_started_live_label(self):
        n = _make_notifier_with_send_mock()
        a = GriffTelegramAlerts(n)
        asyncio.run(a.bot_started(dry_run=False, pairs=("EURUSD",)))
        sent = n.send.call_args.args[0]
        self.assertIn("LIVE", sent)
        self.assertNotIn("DRY_RUN", sent)

    def test_bot_started_live_with_full_context(self):
        n = _make_notifier_with_send_mock()
        a = GriffTelegramAlerts(n)
        asyncio.run(a.bot_started(
            dry_run=False, pairs=("EURUSD", "AUDJPY"),
            broker_name="FTMO", prop_firm_key="ftmo_2step_challenge",
            account_balance=10_000.0, account_currency="USD",
        ))
        sent = n.send.call_args.args[0]
        # Header + each enrichment line.
        self.assertIn("propX Bot LIVE TRADING", sent)
        self.assertIn("Broker: FTMO (ftmo_2step_challenge detected)", sent)
        self.assertIn("Account: $10,000.00 USD", sent)
        self.assertIn("Mode: LIVE TRADING", sent)
        self.assertIn("🟢 OPERATIONAL", sent)

    def test_bot_started_dry_run_with_context_shows_yellow(self):
        n = _make_notifier_with_send_mock()
        a = GriffTelegramAlerts(n)
        asyncio.run(a.bot_started(
            dry_run=True, pairs=("EURUSD",),
            broker_name="FTMO", account_balance=10_000.0,
            account_currency="USD",
        ))
        sent = n.send.call_args.args[0]
        self.assertIn("DRY_RUN", sent)
        self.assertIn("🟡", sent)
        self.assertIn("no orders will be placed", sent)
