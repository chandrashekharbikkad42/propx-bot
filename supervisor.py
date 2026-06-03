"""Process supervisor — keeps bot.py alive across crashes.

Wraps `python -m bot ...` as a child process and restarts on non-zero exit
with exponential backoff. Clean (exit 0) shutdowns are NOT restarted — that's
graceful Ctrl+C. Restart count is bounded by settings.max_restart_attempts.

REAL-mode safety: if EXECUTION_MODE=REAL and balance has changed dramatically
since the last attempt (>5%), refuse to restart — likely the strategy is in a
bad state and re-launching could compound the loss.
"""

from __future__ import annotations
import argparse
import asyncio
import os
import signal
import subprocess
import sys
from typing import Optional

from config.settings import settings
from utils.logger import logger


BALANCE_CHANGE_GUARD_PCT = 5.0
BALANCE_PROBE_TIMEOUT_SEC = 10.0


def _backoff_seconds(attempt: int, base: int) -> int:
    """Exponential: base, base*2, base*4, ... capped at 30 minutes."""
    return min(base * (2 ** attempt), 1800)


async def _query_real_balance() -> Optional[float]:
    """Fetch live MT5 balance, or None if MT5 unavailable / not REAL mode."""
    try:
        from data.mt5_connector import MT5Connector
        connector = MT5Connector()
        await asyncio.wait_for(
            asyncio.to_thread(connector.connect),
            timeout=BALANCE_PROBE_TIMEOUT_SEC,
        )
        try:
            info = await asyncio.to_thread(connector.account_info)
            return float(info.balance)
        finally:
            await asyncio.to_thread(connector.disconnect)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"balance probe failed: {exc}")
        return None


class BotSupervisor:
    def __init__(
        self,
        bot_args: list[str],
        max_attempts: int,
        base_backoff_sec: int,
        notifier=None,
    ) -> None:
        self._bot_args = bot_args
        self._max_attempts = max_attempts
        self._base_backoff_sec = base_backoff_sec
        self._notifier = notifier
        self._stop_requested = False
        self._child: Optional[asyncio.subprocess.Process] = None
        self._last_real_balance: Optional[float] = None

    # ----------------------------------------------------------------- run

    async def run(self) -> int:
        self._install_signal_handlers()

        attempt = 0
        while attempt <= self._max_attempts:
            if attempt > 0:
                if not await self._safe_to_restart():
                    logger.error("supervisor refusing to restart (balance guard)")
                    await self._notify(
                        "supervisor refusing restart — balance changed too much"
                    )
                    return 2
                backoff = _backoff_seconds(attempt - 1, self._base_backoff_sec)
                logger.warning(
                    f"supervisor: restart {attempt}/{self._max_attempts} "
                    f"after {backoff}s backoff"
                )
                await self._notify(
                    f"restart {attempt}/{self._max_attempts} in {backoff}s"
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    return 130
                if self._stop_requested:
                    return 0

            exit_code = await self._spawn_and_wait()
            if self._stop_requested:
                logger.info("supervisor: graceful stop")
                return 0
            if exit_code == 0:
                logger.success("supervisor: child exited cleanly — done")
                return 0
            attempt += 1
            logger.error(f"supervisor: child crashed exit={exit_code}")
            await self._notify(f"bot crashed exit={exit_code}")

        logger.error("supervisor: max restart attempts exhausted")
        await self._notify("supervisor giving up — max restart attempts")
        return 3

    # ------------------------------------------------------------- internals

    async def _spawn_and_wait(self) -> int:
        cmd = [sys.executable, "-u", "-m", "bot", *self._bot_args]
        logger.info(f"supervisor: launching {' '.join(cmd)}")
        self._child = await asyncio.create_subprocess_exec(*cmd)
        try:
            return await self._child.wait()
        except asyncio.CancelledError:
            await self._terminate_child()
            raise

    async def _terminate_child(self) -> None:
        if self._child is None or self._child.returncode is not None:
            return
        try:
            self._child.terminate()
            await asyncio.wait_for(self._child.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("supervisor: child unresponsive — killing")
            self._child.kill()
            await self._child.wait()
        except ProcessLookupError:
            pass

    async def _safe_to_restart(self) -> bool:
        """Block restart if REAL-mode balance changed too much."""
        if settings.execution_mode != "REAL":
            return True
        cur = await _query_real_balance()
        if cur is None:
            # Can't probe — fail safe by allowing restart with a warning.
            logger.warning("supervisor: balance probe unavailable, proceeding")
            return True
        if self._last_real_balance is None:
            self._last_real_balance = cur
            return True
        prev = self._last_real_balance
        if prev <= 0:
            self._last_real_balance = cur
            return True
        pct = abs(cur - prev) / prev * 100.0
        if pct >= BALANCE_CHANGE_GUARD_PCT:
            logger.error(
                f"supervisor: balance change {pct:.2f}% "
                f"(prev=${prev:,.2f} cur=${cur:,.2f}) — blocking restart"
            )
            return False
        self._last_real_balance = cur
        return True

    async def _notify(self, message: str) -> None:
        if self._notifier is None or not getattr(self._notifier, "enabled", False):
            return
        try:
            await self._notifier.send(f"[supervisor] {message}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"supervisor notifier error: {exc}")

    def _install_signal_handlers(self) -> None:
        def _request_stop(*_args):
            if not self._stop_requested:
                logger.info("supervisor: stop requested — propagating to child")
                self._stop_requested = True
            # Propagate to child if it's still alive.
            if self._child is not None and self._child.returncode is None:
                try:
                    self._child.terminate()
                except ProcessLookupError:
                    pass

        if sys.platform == "win32":
            signal.signal(signal.SIGINT, _request_stop)
            signal.signal(signal.SIGTERM, _request_stop)
        else:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _request_stop)


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="supervisor", description="bot.py auto-restart wrapper"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=None,
        help="Override settings.max_restart_attempts.",
    )
    parser.add_argument(
        "--backoff", type=int, default=None,
        help="Override settings.restart_backoff_sec (base, exponential).",
    )
    # Everything after `--` is passed verbatim to bot.py
    known, rest = parser.parse_known_args(argv)
    return known, rest


async def _async_main(argv: list[str] | None = None) -> int:
    args, bot_args = _parse_args(argv)
    max_attempts = args.max_attempts if args.max_attempts is not None \
        else settings.max_restart_attempts
    base_backoff = args.backoff if args.backoff is not None \
        else settings.restart_backoff_sec

    from alerts import TelegramNotifier
    notifier = TelegramNotifier(
        settings.telegram_bot_token, settings.telegram_chat_id
    )

    supervisor = BotSupervisor(
        bot_args=bot_args,
        max_attempts=max_attempts,
        base_backoff_sec=base_backoff,
        notifier=notifier,
    )
    return await supervisor.run()


def main() -> int:
    try:
        return asyncio.run(_async_main())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
