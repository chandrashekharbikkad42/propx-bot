"""Circuit-breaker state machine."""

from __future__ import annotations
import unittest
from datetime import datetime, timezone

from execution.order import Side
from execution.position import CloseReason, Position, PositionState
from risk.circuit_breakers import CircuitBreakers


def _msc(year: int, month: int, day: int, hour: int = 12) -> int:
    return int(datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


def _closed_position(pnl_usd: float, exit_msc: int) -> Position:
    return Position(
        position_id="p1", side=Side.BUY, lots=1.0, entry_price=2000.0,
        entry_time_msc=exit_msc - 1000,
        sl_price=1990.0, tp_price=2010.0, max_hold_until_msc=exit_msc + 1000,
        state=PositionState.CLOSED,
        exit_price=2000.0 + (pnl_usd / 100.0),
        exit_time_msc=exit_msc,
        close_reason=CloseReason.TP_HIT if pnl_usd > 0 else CloseReason.SL_HIT,
        pnl_pts=pnl_usd, pnl_usd=pnl_usd,
    )


class TestCircuitBreakers(unittest.TestCase):
    def test_daily_cap_blocks_after_2pct_loss(self) -> None:
        cb = CircuitBreakers(daily_cap_pct=0.02, streak_threshold=99)
        t = _msc(2026, 5, 12, hour=12)  # London/NY overlap
        ok, _ = cb.can_trade(t, account_equity=10_000.0)
        self.assertTrue(ok)

        # Single trade loses 2% of $10K = $200. Cap hit immediately.
        cb.record_trade_close(_closed_position(-200.0, t))
        ok, reason = cb.can_trade(t + 1000, account_equity=9_800.0)
        self.assertFalse(ok)
        self.assertEqual(reason, "daily_cap_hit")

    def test_three_losses_trigger_pause(self) -> None:
        cb = CircuitBreakers(daily_cap_pct=0.5, streak_threshold=3, pause_minutes=30)
        t = _msc(2026, 5, 12, hour=12)
        cb.can_trade(t, account_equity=10_000.0)  # priming the day

        for i in range(3):
            cb.record_trade_close(_closed_position(-10.0, t + i * 1000))

        # Inside pause window → blocked.
        ok, reason = cb.can_trade(t + 5 * 60 * 1000, account_equity=10_000.0)
        self.assertFalse(ok)
        self.assertEqual(reason, "loss_streak_pause")

        # After pause window → unblocked.
        ok, _ = cb.can_trade(t + (2 * 1000 + 30 * 60 * 1000 + 1), account_equity=10_000.0)
        self.assertTrue(ok)

    def test_win_resets_streak(self) -> None:
        cb = CircuitBreakers(daily_cap_pct=0.5, streak_threshold=3, pause_minutes=30)
        t = _msc(2026, 5, 12, hour=12)
        cb.can_trade(t, account_equity=10_000.0)

        cb.record_trade_close(_closed_position(-10.0, t))
        cb.record_trade_close(_closed_position(-10.0, t + 1000))
        cb.record_trade_close(_closed_position(15.0, t + 2000))  # win — resets

        self.assertEqual(cb.state.consecutive_losses, 0)
        self.assertEqual(cb.state.streak_pause_until_msc, 0)

    def test_day_rollover_resets_state(self) -> None:
        cb = CircuitBreakers(daily_cap_pct=0.02, streak_threshold=3)
        d1 = _msc(2026, 5, 12, hour=12)
        cb.can_trade(d1, account_equity=10_000.0)
        cb.record_trade_close(_closed_position(-200.0, d1))
        self.assertTrue(cb.state.daily_cap_hit)

        d2 = _msc(2026, 5, 13, hour=12)
        ok, _ = cb.can_trade(d2, account_equity=9_800.0)
        self.assertTrue(ok)
        self.assertFalse(cb.state.daily_cap_hit)
        self.assertEqual(cb.state.daily_pnl_usd, 0.0)

    def test_asian_session_blocked(self) -> None:
        cb = CircuitBreakers()
        t = _msc(2026, 5, 12, hour=3)  # ASIAN
        ok, reason = cb.can_trade(t, account_equity=10_000.0)
        self.assertFalse(ok)
        self.assertIn("ASIAN", reason)

    def test_off_session_blocked(self) -> None:
        cb = CircuitBreakers()
        t = _msc(2026, 5, 12, hour=22)  # OFF
        ok, reason = cb.can_trade(t, account_equity=10_000.0)
        self.assertFalse(ok)
        self.assertIn("OFF", reason)

    def test_london_session_allowed(self) -> None:
        cb = CircuitBreakers()
        t = _msc(2026, 5, 12, hour=9)
        ok, _ = cb.can_trade(t, account_equity=10_000.0)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
