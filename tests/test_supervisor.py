"""Supervisor — restart loop, backoff, balance guard."""

from __future__ import annotations
import asyncio
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from supervisor import BotSupervisor, _backoff_seconds


class TestBackoff(unittest.TestCase):
    def test_exponential(self) -> None:
        self.assertEqual(_backoff_seconds(0, 30), 30)
        self.assertEqual(_backoff_seconds(1, 30), 60)
        self.assertEqual(_backoff_seconds(2, 30), 120)
        self.assertEqual(_backoff_seconds(3, 30), 240)

    def test_caps_at_30_min(self) -> None:
        self.assertEqual(_backoff_seconds(20, 30), 1800)


class TestSupervisorRestartLoop(unittest.TestCase):
    def _supervisor(self, max_attempts: int = 2) -> BotSupervisor:
        return BotSupervisor(
            bot_args=["--mode", "live"],
            max_attempts=max_attempts,
            base_backoff_sec=1,
            notifier=None,
        )

    def test_clean_exit_no_restart(self) -> None:
        sup = self._supervisor()
        spawn = AsyncMock(return_value=0)
        with patch.object(sup, "_spawn_and_wait", spawn), \
                patch.object(sup, "_install_signal_handlers"):
            code = asyncio.run(sup.run())
        self.assertEqual(code, 0)
        self.assertEqual(spawn.await_count, 1)

    def test_crash_then_clean(self) -> None:
        sup = self._supervisor(max_attempts=3)
        spawn = AsyncMock(side_effect=[1, 0])
        with patch.object(sup, "_spawn_and_wait", spawn), \
                patch.object(sup, "_install_signal_handlers"), \
                patch("supervisor.asyncio.sleep", new=AsyncMock(return_value=None)), \
                patch.object(sup, "_safe_to_restart", new=AsyncMock(return_value=True)):
            code = asyncio.run(sup.run())
        self.assertEqual(code, 0)
        self.assertEqual(spawn.await_count, 2)

    def test_max_attempts_exhausted(self) -> None:
        sup = self._supervisor(max_attempts=2)
        spawn = AsyncMock(return_value=1)  # always crashes
        with patch.object(sup, "_spawn_and_wait", spawn), \
                patch.object(sup, "_install_signal_handlers"), \
                patch("supervisor.asyncio.sleep", new=AsyncMock(return_value=None)), \
                patch.object(sup, "_safe_to_restart", new=AsyncMock(return_value=True)):
            code = asyncio.run(sup.run())
        self.assertEqual(code, 3)
        # 1 initial + 2 retries = 3 attempts
        self.assertEqual(spawn.await_count, 3)

    def test_balance_guard_blocks_restart(self) -> None:
        sup = self._supervisor(max_attempts=2)
        spawn = AsyncMock(return_value=1)
        with patch.object(sup, "_spawn_and_wait", spawn), \
                patch.object(sup, "_install_signal_handlers"), \
                patch("supervisor.asyncio.sleep", new=AsyncMock(return_value=None)), \
                patch.object(sup, "_safe_to_restart", new=AsyncMock(return_value=False)):
            code = asyncio.run(sup.run())
        self.assertEqual(code, 2)


class TestBalanceGuard(unittest.TestCase):
    def test_paper_mode_always_safe(self) -> None:
        sup = BotSupervisor(bot_args=[], max_attempts=1, base_backoff_sec=1)
        with patch("supervisor.settings") as mock_settings:
            mock_settings.execution_mode = "PAPER"
            result = asyncio.run(sup._safe_to_restart())
        self.assertTrue(result)

    def test_real_mode_block_on_large_change(self) -> None:
        sup = BotSupervisor(bot_args=[], max_attempts=1, base_backoff_sec=1)
        sup._last_real_balance = 1000.0
        with patch("supervisor.settings") as mock_settings, \
                patch("supervisor._query_real_balance",
                      new=AsyncMock(return_value=900.0)):  # -10%
            mock_settings.execution_mode = "REAL"
            result = asyncio.run(sup._safe_to_restart())
        self.assertFalse(result)

    def test_real_mode_pass_on_small_change(self) -> None:
        sup = BotSupervisor(bot_args=[], max_attempts=1, base_backoff_sec=1)
        sup._last_real_balance = 1000.0
        with patch("supervisor.settings") as mock_settings, \
                patch("supervisor._query_real_balance",
                      new=AsyncMock(return_value=990.0)):  # -1%
            mock_settings.execution_mode = "REAL"
            result = asyncio.run(sup._safe_to_restart())
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
