"""Trade ledger + summary statistics. No persistence — backtest owns lifetime."""

from __future__ import annotations
from collections import defaultdict
from typing import List

import pandas as pd

from execution.position import CloseReason, Position


class TradeJournal:
    def __init__(self) -> None:
        self._closed: List[Position] = []
        self._opens: int = 0

    def log_open(self, position: Position) -> None:
        self._opens += 1

    def log_close(self, position: Position) -> None:
        self._closed.append(position)

    # ------------------------------------------------------------------ summary

    def summary(self) -> dict:
        n = len(self._closed)
        if n == 0:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "gross_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "expectancy": 0.0,
                "max_consecutive_losses": 0,
                "max_drawdown_pct": 0.0,
                "max_drawdown_usd": 0.0,
                "max_loss_from_start_usd": 0.0,
                "by_signal_type": {},
                "by_session": {},
                "by_close_reason": {},
            }

        wins = [p.pnl_usd for p in self._closed if (p.pnl_usd or 0.0) > 0]
        losses = [p.pnl_usd for p in self._closed if (p.pnl_usd or 0.0) <= 0]
        gross = sum((p.pnl_usd or 0.0) for p in self._closed)

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        win_rate = len(wins) / n
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        # Streaks + drawdown computed over the realised sequence.
        max_streak = 0
        run = 0
        peak = 0.0
        equity = 0.0
        max_dd = 0.0
        max_dd_usd = 0.0
        max_loss_from_start = 0.0  # most-negative cumulative pnl seen
        for p in self._closed:
            pnl = p.pnl_usd or 0.0
            equity += pnl
            if equity > peak:
                peak = equity
            # Peak-to-trough drawdown (USD + % of peak) and cumulative pnl
            # vs the starting line — the latter is what the CircuitBreakers
            # daily cap actually compares against.
            dd_usd = peak - equity
            if dd_usd > max_dd_usd:
                max_dd_usd = dd_usd
            if peak > 0:
                dd_pct = (peak - equity) / peak * 100.0
                if dd_pct > max_dd:
                    max_dd = dd_pct
            if equity < max_loss_from_start:
                max_loss_from_start = equity
            if pnl <= 0:
                run += 1
                if run > max_streak:
                    max_streak = run
            else:
                run = 0

        return {
            "total_trades": n,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "gross_pnl": gross,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "max_consecutive_losses": max_streak,
            "max_drawdown_pct": max_dd,
            "max_drawdown_usd": max_dd_usd,
            "max_loss_from_start_usd": -max_loss_from_start,  # positive = how far below start
            "by_signal_type": self._bucket(lambda p: p.signal_type or "UNKNOWN"),
            "by_session": self._bucket(lambda p: p.session or "UNKNOWN"),
            "by_close_reason": self._bucket(
                lambda p: p.close_reason.value if isinstance(p.close_reason, CloseReason) else "UNKNOWN"
            ),
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for p in self._closed:
            rows.append(
                {
                    "position_id": p.position_id,
                    "side": p.side.value,
                    "lots": p.lots,
                    "entry_price": p.entry_price,
                    "exit_price": p.exit_price,
                    "entry_time_msc": p.entry_time_msc,
                    "exit_time_msc": p.exit_time_msc,
                    "sl_price": p.sl_price,
                    "tp_price": p.tp_price,
                    "close_reason": p.close_reason.value if p.close_reason else None,
                    "signal_type": p.signal_type,
                    "session": p.session,
                    "pnl_pts": p.pnl_pts,
                    "pnl_usd": p.pnl_usd,
                }
            )
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- helpers

    def _bucket(self, key_fn) -> dict:
        agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
        for p in self._closed:
            key = key_fn(p)
            agg[key]["count"] += 1
            agg[key]["pnl"] += p.pnl_usd or 0.0
            if (p.pnl_usd or 0.0) > 0:
                agg[key]["wins"] += 1
        # Cast back to dict for clean printing.
        return {k: dict(v) for k, v in agg.items()}

    @property
    def closed_positions(self) -> List[Position]:
        return list(self._closed)
