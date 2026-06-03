"""HourlyReporter — periodic Telegram status during live trading.

Pushes an hourly digest to the configured Telegram chat so the operator
always knows the bot is alive AND the broker feed is healthy. Two flavours:

  * Full report     : every hour by default
  * Idle short msg  : during silent IST hours (00:00–12:00) when literally
                      nothing has happened (no bars seen, no signals, no
                      open positions, no closed trades). Just a heartbeat.

Format (full):

    📊 Griff Bot Status — HH:MM IST

    📈 Today:
    • Trades: X
    • P/L: ±$X (X.X%)
    • DD: X.X% / X.X% cap

    🔍 Last hour:
    • Bars received: X/N pairs
    • Signals detected: X
    • Compliance verdicts: pass=X blocked=X
    • Open positions: X

    ✅ Healthy

Format (idle):

    📊 Griff Bot Status — HH:MM IST — idle (silent window)

Inputs the engine fills as cycles run:

    HourlyStats — append-only counter for the past hour. Reset on each send.

Scheduling helper:

    next_top_of_hour_ms(now_msc) → ms timestamp of the next HH:00:00 UTC.
    Use with `await asyncio.sleep((nxt - now)/1000)` from the run loop.

Hinglish: har ghante ek status snap. Operator ko bharosa rahe ki bot zinda
hai aur feed bhi sahi aa raha hai. Silent IST raat me ek-line "idle" bhej
dete hain — full report ka traffic bachata hai. Send ke baad stats reset.
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Set

from alerts.telegram_notifier import TelegramNotifier
from execution.position_manager import GriffPositionManager
from monitoring.daily_tracker import DailyTracker


IST_OFFSET = timedelta(hours=5, minutes=30)
HOUR_MS = 60 * 60 * 1000
MINUTE_MS = 60 * 1000

# Status-report cadence. Historically the digest fired once per hour; the
# operator wanted a tighter pulse, so the default is now every 15 minutes.
# Configurable end-to-end: run_periodic takes interval_ms, and the live
# runner exposes --report-interval-min. next_top_of_hour_ms stays as the
# hour-aligned special case for callers/tests that still want it.
DEFAULT_REPORT_INTERVAL_MS = 15 * MINUTE_MS


# ----------------------------------------------------------------- stats


@dataclass
class HourlyStats:
    """Mutable per-hour counters. Engine increments; reporter reads + resets."""

    bars_received: int = 0
    pairs_with_bars: Set[str] = field(default_factory=set)
    signals_detected: int = 0
    compliance_passed: int = 0
    compliance_blocked: int = 0

    def record_bar(self, pair: str) -> None:
        self.bars_received += 1
        self.pairs_with_bars.add(pair)

    def record_signal(self, n: int = 1) -> None:
        self.signals_detected += n

    def record_compliance(self, *, passed: bool) -> None:
        if passed:
            self.compliance_passed += 1
        else:
            self.compliance_blocked += 1

    def is_idle(self) -> bool:
        return (
            self.bars_received == 0
            and self.signals_detected == 0
            and self.compliance_passed == 0
            and self.compliance_blocked == 0
        )

    def reset(self) -> None:
        self.bars_received = 0
        self.pairs_with_bars = set()
        self.signals_detected = 0
        self.compliance_passed = 0
        self.compliance_blocked = 0


# ----------------------------------------------------------------- helpers


def next_tick_ms(
    now_msc: int, interval_ms: int = DEFAULT_REPORT_INTERVAL_MS
) -> int:
    """Return the next interval-aligned ms boundary strictly after now_msc.

    Boundaries are aligned to the UTC epoch, so a 15-minute interval fires at
    :00/:15/:30/:45. Advances even when now is already exactly on a boundary
    so the loop never busy-waits on a zero-length sleep.
    """
    interval_ms = max(1, int(interval_ms))
    rem = now_msc % interval_ms
    return now_msc + (interval_ms - rem) if rem else now_msc + interval_ms


def next_top_of_hour_ms(now_msc: int) -> int:
    """Return the ms timestamp of the next HH:00:00 UTC (>= now+1ms)."""
    return next_tick_ms(now_msc, HOUR_MS)


def _ist_hhmm(now_msc: int) -> str:
    utc_dt = datetime.fromtimestamp(now_msc / 1000.0, tz=timezone.utc)
    ist_dt = utc_dt + IST_OFFSET
    return ist_dt.strftime("%H:%M")


def _ist_hour(now_msc: int) -> int:
    utc_dt = datetime.fromtimestamp(now_msc / 1000.0, tz=timezone.utc)
    ist_dt = utc_dt + IST_OFFSET
    return ist_dt.hour


# ----------------------------------------------------------------- reporter


class HourlyReporter:
    """Renders + sends the hourly status snapshot."""

    def __init__(
        self,
        *,
        notifier: TelegramNotifier,
        daily: DailyTracker,
        position_mgr: GriffPositionManager,
        stats: HourlyStats,
        num_pairs: int,
        daily_loss_cap_pct: float,
        starting_equity: Optional[float] = None,
        idle_silent_hour_start: int = 0,
        idle_silent_hour_end: int = 12,
    ) -> None:
        self._n = notifier
        self._daily = daily
        self._pm = position_mgr
        self._stats = stats
        self._num_pairs = max(1, num_pairs)
        self._daily_cap_pct = daily_loss_cap_pct
        # Starting equity baseline for % conversions. Defaults to the daily
        # tracker's peak at construction time so % math is well-defined.
        self._starting_equity = (
            starting_equity if starting_equity is not None
            else (daily.state.peak_equity or 10_000.0)
        )
        self._silent_start = idle_silent_hour_start
        self._silent_end = idle_silent_hour_end

    # ----- formatting --------------------------------------------------

    def _in_silent_window(self, now_msc: int) -> bool:
        h = _ist_hour(now_msc)
        return self._silent_start <= h < self._silent_end

    def _should_abbreviate(self, now_msc: int) -> bool:
        if not self._in_silent_window(now_msc):
            return False
        if not self._stats.is_idle():
            return False
        if self._pm.open_positions:
            return False
        if self._daily.trade_count > 0:
            return False
        return True

    def format(self, now_msc: int) -> str:
        if self._should_abbreviate(now_msc):
            return self._format_idle(now_msc)
        return self._format_full(now_msc)

    def _format_idle(self, now_msc: int) -> str:
        return (
            f"📊 propX Bot Status — {_ist_hhmm(now_msc)} IST — "
            f"idle (silent window)"
        )

    def _format_full(self, now_msc: int) -> str:
        s = self._daily.state
        equity_base = max(self._starting_equity, 1.0)
        pnl_pct = (s.closed_pnl / equity_base) * 100.0
        dd_pct = (s.max_dd_today / equity_base) * 100.0
        pnl_sign = "+" if s.closed_pnl >= 0 else "-"

        bars_part = (
            f"{len(self._stats.pairs_with_bars)}/{self._num_pairs}"
        )
        compliance_part = (
            f"pass={self._stats.compliance_passed} "
            f"blocked={self._stats.compliance_blocked}"
        )
        open_count = len(self._pm.open_positions)
        health_marker = "✅ Healthy"
        # Soft warning when bars-received pairs < num_pairs during active
        # hours. Operator action: check feed.
        if (
            not self._in_silent_window(now_msc)
            and len(self._stats.pairs_with_bars) == 0
        ):
            health_marker = "⚠️ No bars received last hour"

        return (
            f"📊 propX Bot Status — {_ist_hhmm(now_msc)} IST\n"
            f"\n"
            f"📈 Today:\n"
            f"• Trades: {s.trade_count}\n"
            f"• P/L: {pnl_sign}${abs(s.closed_pnl):.2f} "
            f"({pnl_pct:+.2f}%)\n"
            f"• DD: {dd_pct:.2f}% / {self._daily_cap_pct:.1f}% cap\n"
            f"\n"
            f"🔍 Last hour:\n"
            f"• Bars received: {bars_part} pairs\n"
            f"• Signals detected: {self._stats.signals_detected}\n"
            f"• Compliance verdicts: {compliance_part}\n"
            f"• Open positions: {open_count}\n"
            f"\n"
            f"{health_marker}"
        )

    # ----- transport ---------------------------------------------------

    async def send(self, now_msc: int) -> bool:
        """Format + send; ALWAYS reset stats afterward (even on send failure)
        so a single Telegram outage doesn't double-count the next window."""
        msg = self.format(now_msc)
        try:
            ok = await self._n.send(msg)
        finally:
            self._stats.reset()
        return bool(ok)

    # ----- run loop ----------------------------------------------------

    async def run_periodic(
        self,
        stop_event: asyncio.Event,
        *,
        clock_ms,  # callable returning current epoch-ms; injected for tests
        interval_ms: int = DEFAULT_REPORT_INTERVAL_MS,
    ) -> None:
        """Sleep until the next interval boundary, send, repeat — until
        stop_event. Defaults to a 15-minute cadence; pass interval_ms to
        change it (the live runner derives this from --report-interval-min)."""
        while not stop_event.is_set():
            now = clock_ms()
            next_ms = next_tick_ms(now, interval_ms)
            wait_s = max(0.0, (next_ms - now) / 1000.0)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=wait_s)
                return  # stop signalled mid-wait
            except asyncio.TimeoutError:
                pass
            if stop_event.is_set():
                return
            await self.send(clock_ms())
