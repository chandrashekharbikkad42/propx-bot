"""Telegram bot notifier — async, fail-safe.

Disabled (no-op) when token or chat_id is missing. All network errors are
swallowed and logged so the trading loop never blocks on a flaky Telegram
endpoint. Signal-type notifications are rate-limited per-type to avoid
spamming the chat during a noisy regime.
"""

from __future__ import annotations
import asyncio
import ssl
import time
from typing import Optional

import aiohttp
import certifi

from execution.position import Position
from utils.logger import logger


TELEGRAM_API = "https://api.telegram.org"
DEFAULT_TIMEOUT_SEC = 5.0
SIGNAL_RATE_LIMIT_SEC = 300  # 5 minutes per signal type


def _build_ssl_context() -> ssl.SSLContext:
    # Windows trust store often misses the Telegram CA chain; fall back to
    # certifi's bundle so SSLCertVerificationError doesn't kill alerts.
    return ssl.create_default_context(cafile=certifi.where())


def _fmt_duration_ms(ms: int) -> str:
    if ms <= 0:
        return "0s"
    secs = ms // 1000
    if secs < 60:
        return f"{secs}s"
    mins, s = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m{s:02d}s"
    h, m = divmod(mins, 60)
    return f"{h}h{m:02d}m"


class TelegramNotifier:
    def __init__(
        self,
        token: Optional[str],
        chat_id: Optional[str],
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self._token = (token or "").strip()
        self._chat_id = (chat_id or "").strip()
        self._timeout_sec = timeout_sec
        self._last_signal_sent: dict[str, float] = {}
        self._ssl_context: Optional[ssl.SSLContext] = None

        self.enabled = bool(self._token and self._chat_id)
        if not self.enabled:
            logger.warning("TelegramNotifier disabled (missing token or chat_id)")

    def _get_ssl_context(self) -> ssl.SSLContext:
        if self._ssl_context is None:
            self._ssl_context = _build_ssl_context()
        return self._ssl_context

    # -------------------------------------------------------------- transport

    async def send(self, message: str, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            return False

        url = f"{TELEGRAM_API}/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        timeout = aiohttp.ClientTimeout(total=self._timeout_sec)
        connector = aiohttp.TCPConnector(ssl=self._get_ssl_context())
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, connector=connector
            ) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning(
                            f"Telegram send failed status={resp.status} body={body[:200]}"
                        )
                        return False
                    return True
        except asyncio.TimeoutError:
            logger.warning("Telegram send timed out")
            return False
        except Exception as exc:  # noqa: BLE001 — must not crash trading loop
            logger.warning(f"Telegram send error: {exc}")
            return False

    # ------------------------------------------------------------- formatters

    async def notify_trade_open(self, position: Position) -> bool:
        side = position.side.value
        msg = (
            f"<b>OPEN</b> {side} {position.signal_type or '?'}\n"
            f"entry={position.entry_price:.2f}  lots={position.lots:g}\n"
            f"SL={position.sl_price:.2f}  TP={position.tp_price:.2f}\n"
            f"session={position.session or '?'}"
        )
        return await self.send(msg)

    async def notify_trade_close(self, position: Position) -> bool:
        pnl = position.pnl_usd or 0.0
        pts = position.pnl_pts or 0.0
        reason = position.close_reason.value if position.close_reason else "?"
        hold = _fmt_duration_ms(
            (position.exit_time_msc or 0) - position.entry_time_msc
        )
        emoji = "+" if pnl >= 0 else "-"
        msg = (
            f"<b>CLOSE</b> {position.side.value} {position.signal_type or '?'} "
            f"({reason})\n"
            f"pnl={emoji}${abs(pnl):.2f}  pts={pts:+.1f}  hold={hold}"
        )
        return await self.send(msg)

    # `notify_signal(Signal)` was removed in Phase 5 cleanup along with the
    # legacy strategy/signals/* tick-microstructure modules. V5 emits signal
    # alerts via `monitoring.telegram_alerts.GriffTelegramAlerts
    # .signal_detected(PatternSignal)` instead.

    async def notify_circuit_breaker(self, reason: str) -> bool:
        return await self.send(f"<b>CIRCUIT BREAKER</b> trading paused: {reason}")

    async def notify_bot_started(self, mode: str, symbol: str) -> bool:
        return await self.send(
            f"<b>BOT STARTED</b> mode={mode} symbol={symbol}"
        )

    async def notify_bot_stopped(self, reason: str = "graceful") -> bool:
        return await self.send(f"<b>BOT STOPPED</b> reason={reason}")

    async def notify_bot_crashed(self, reason: str) -> bool:
        return await self.send(f"<b>BOT CRASHED</b> {reason}")
