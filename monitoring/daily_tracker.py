"""DailyTracker — running per-day P/L, drawdown, trade count.

Owns the bot's view of the trading day. Rolls at 00:00 IST (= 18:30 UTC
previous day). Persists the per-day snapshot to a parquet file so a
restart of the bot mid-day picks up the day's progress instead of
treating itself as fresh.

State recorded per day (UTC date stamp of the IST "trade day"):
  - trade_day      : the IST calendar date string (YYYY-MM-DD).
  - peak_equity    : highest equity (closed + floating) seen this day.
  - closed_pnl     : sum of closed-trade PnL today.
  - floating_pnl   : last-observed floating PnL (open positions).
  - trade_count    : number of closed + filled-pending trades today.
  - max_dd_today   : largest equity drawdown from intraday peak.
  - last_update_ms : wall-clock ms of last `update_equity()` call.

What this module DOES NOT do:
  - It doesn't compute floating PnL from MT5 — caller passes it in.
  - It doesn't sign off on trades; the ComplianceEngine consumes the
    `trade_count` and DD to decide.

Hinglish: ek din ke ledger ki rakhwali. IST midnight pe naya din shuru.
Parquet me snapshot dump rehta hai taaki restart ke baad bot apne hi
purane numbers se aage chale.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.logger import logger


IST_OFFSET = timedelta(hours=5, minutes=30)


def ist_trade_day(now_msc: int) -> str:
    """Return YYYY-MM-DD of the IST calendar day for a UTC ms timestamp.

    The IST trading day rolls at 00:00 IST = 18:30 UTC. A UTC moment at
    18:35 already belongs to the NEXT IST day.
    """
    utc_dt = datetime.fromtimestamp(now_msc / 1000.0, tz=timezone.utc)
    ist_dt = utc_dt + IST_OFFSET
    return ist_dt.date().isoformat()


@dataclass(frozen=True)
class DailyState:
    trade_day: str
    peak_equity: float
    closed_pnl: float
    floating_pnl: float
    trade_count: int
    max_dd_today: float
    last_update_ms: int

    @classmethod
    def empty(cls, trade_day: str, starting_equity: float, now_ms: int) -> "DailyState":
        return cls(
            trade_day=trade_day, peak_equity=starting_equity,
            closed_pnl=0.0, floating_pnl=0.0, trade_count=0,
            max_dd_today=0.0, last_update_ms=now_ms,
        )

    @property
    def equity(self) -> float:
        return self.peak_equity + (self.floating_pnl - self.peak_equity + self.closed_pnl) \
            if False else self._equity_explicit()

    def _equity_explicit(self) -> float:
        # Equity here means: equity AS OF the last update. We don't track
        # opening balance separately; peak_equity is the running max of the
        # equity we've observed via update_equity, so equity = peak - dd.
        # In practice callers should use `equity_now` below.
        return self.peak_equity - self.max_dd_today

    @property
    def equity_now(self) -> float:
        """Approximation of current equity: opening balance + closed + floating.
        Without an explicit `opening_equity` field we treat peak_equity as the
        running max — caller passes equity into `update_equity` so this is
        kept self-consistent."""
        # The most-recent equity seen is peak_equity - drawdown_from_peak.
        # Since drawdown_from_peak is exactly max_dd_today only when we're
        # AT the trough, callers should treat this as a soft accessor.
        return self.peak_equity - self.max_dd_today


class DailyTracker:
    def __init__(
        self,
        starting_equity: float,
        *,
        persist_path: Optional[Path] = None,
        now_ms: Optional[int] = None,
    ) -> None:
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        self._persist_path = persist_path
        self._state = DailyState.empty(
            trade_day=ist_trade_day(now),
            starting_equity=starting_equity,
            now_ms=now,
        )
        if persist_path is not None and persist_path.exists():
            loaded = self._load_if_same_day(persist_path, now)
            if loaded is not None:
                self._state = loaded

    # --------------------------------------------------------------- views

    @property
    def state(self) -> DailyState:
        return self._state

    @property
    def trade_day(self) -> str:
        return self._state.trade_day

    @property
    def trade_count(self) -> int:
        return self._state.trade_count

    @property
    def max_dd_today(self) -> float:
        return self._state.max_dd_today

    # --------------------------------------------------------- mutate ops

    def update_equity(self, equity: float, *, now_ms: Optional[int] = None) -> None:
        """Record a fresh equity (closed + floating) observation."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        self._roll_if_new_day(now)
        s = self._state
        new_peak = max(s.peak_equity, equity)
        dd = max(0.0, new_peak - equity)
        new_max_dd = max(s.max_dd_today, dd)
        self._state = replace(
            s, peak_equity=new_peak, max_dd_today=new_max_dd,
            last_update_ms=now,
        )

    def record_trade_open(self, *, now_ms: Optional[int] = None) -> None:
        """Bump the today-trade counter on each open. Called by the engine."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        self._roll_if_new_day(now)
        self._state = replace(
            self._state, trade_count=self._state.trade_count + 1,
            last_update_ms=now,
        )

    def record_trade_closed(
        self, pnl_usd: float, *, now_ms: Optional[int] = None
    ) -> None:
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        self._roll_if_new_day(now)
        self._state = replace(
            self._state,
            closed_pnl=self._state.closed_pnl + pnl_usd,
            last_update_ms=now,
        )

    def set_floating_pnl(
        self, floating: float, *, now_ms: Optional[int] = None
    ) -> None:
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        self._roll_if_new_day(now)
        self._state = replace(
            self._state, floating_pnl=floating, last_update_ms=now,
        )

    def persist(self) -> None:
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([self._state.__dict__])
        df.to_parquet(self._persist_path, index=False)

    # ----------------------------------------------------------- helpers

    def _roll_if_new_day(self, now_ms: int) -> None:
        today = ist_trade_day(now_ms)
        if today == self._state.trade_day:
            return
        logger.info(
            f"DailyTracker rolling from {self._state.trade_day} → {today}; "
            f"prior day closed_pnl={self._state.closed_pnl:.2f} "
            f"max_dd={self._state.max_dd_today:.2f} "
            f"trades={self._state.trade_count}"
        )
        # Carry only peak_equity forward as the new day's starting peak (the
        # equity at the rollover moment). Other counters reset.
        prior_peak = self._state.peak_equity - self._state.max_dd_today + \
            self._state.closed_pnl + self._state.floating_pnl
        self._state = DailyState.empty(
            trade_day=today, starting_equity=prior_peak, now_ms=now_ms,
        )

    def _load_if_same_day(
        self, path: Path, now_ms: int
    ) -> Optional[DailyState]:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"DailyTracker could not read {path}: {exc}")
            return None
        if df.empty:
            return None
        row = df.iloc[-1].to_dict()
        if row.get("trade_day") != ist_trade_day(now_ms):
            return None
        return DailyState(
            trade_day=str(row["trade_day"]),
            peak_equity=float(row["peak_equity"]),
            closed_pnl=float(row["closed_pnl"]),
            floating_pnl=float(row["floating_pnl"]),
            trade_count=int(row["trade_count"]),
            max_dd_today=float(row["max_dd_today"]),
            last_update_ms=int(row["last_update_ms"]),
        )
