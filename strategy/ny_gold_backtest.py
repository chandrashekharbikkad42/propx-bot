"""NY Gold Sweep — backtest harness (NY_GOLD_SWEEP_SPEC.md v1.1 §6–§9).

Single-pair (XAUUSD), single-position, NY-session only, static SL/TP.

Run from the repo root:
    venv\\Scripts\\python.exe -m strategy.ny_gold_backtest

This script:
  - loads the 2-yr XAUUSD parquet bars,
  - iterates closed 1M bars in NY session (per §1),
  - calls the detector at each decision time (§0.1 invariants enforced
    via assert_no_lookahead on EVERY decision in the main loop),
  - emits entries at R+1.open with ask/bid + slippage (§5.2 / §5.3),
  - manages open positions with bid/ask exit checks (§5.4 / §8.2),
  - applies the §7 compliance gates (news / DD / cooldown / cap /
    concurrency),
  - SL-wins-same-bar (§9 ¶4),
  - 60-second min-hold deferral (§6),
  - 45-min time stop, 17:00 UTC session flatten (§6),
  - writes report + equity CSV + closed-trade CSV,
  - prints a §12 SHIP/TUNE/CUT verdict.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from config import ny_gold_sweep_config as cfg
from data.news_calendar import StaticNewsCalendar
from strategy.ny_gold_data import (
    NYGoldData, in_session, utc_date_key, utc_hms,
)
from strategy.ny_gold_detector import NYGoldSweepDetector, NYGoldSignal


# ─── trade record ───────────────────────────────────────────────────────────
@dataclass
class OpenTrade:
    direction: str             # "BUY" | "SELL"
    entry_idx: int             # 1M index of entry bar (= R+1)
    entry_time_msc: int        # R+1 open time
    entry_price: float         # cost-adjusted (ask + slip for BUY)
    sl: float
    tp: float
    lot: float
    grade: str
    sweep_depth_pips: float
    reversal_kind: str
    tp_mode: str
    level_price: float
    risk_pips: float
    risk_usd: float
    # Min-hold deferral: if SL/TP would have triggered in the entry bar,
    # we defer the exit to R+2 open at the breached price.
    deferred_exit_price: Optional[float] = None
    deferred_exit_reason: Optional[str] = None  # "sl_minhold" | "tp_minhold"


@dataclass
class ClosedTrade:
    open: OpenTrade
    exit_idx: int
    exit_time_msc: int
    exit_price: float
    exit_reason: str           # "sl" | "tp" | "time_stop" | "session_flatten" | "sl_minhold" | "tp_minhold"
    pnl_pips: float
    pnl_usd: float             # net of commission
    commission_usd: float
    balance_after: float
    r_multiple: float          # pnl_pips / risk_pips (signed)


# ─── helpers ────────────────────────────────────────────────────────────────
def _round_lot(raw: float) -> float:
    """Round to LOT_STEP, clamp to [LOT_MIN, LOT_MAX]."""
    step = cfg.LOT_STEP
    lot = math.floor(raw / step + 1e-9) * step
    lot = max(cfg.LOT_MIN, min(cfg.LOT_MAX, lot))
    return round(lot, 2)


def _half_spread_price(spread_pts: float) -> float:
    """spread_pts -> half-spread in price units (pips → price via PIP_SIZE)."""
    if math.isnan(spread_pts) or spread_pts <= 0:
        spread_pts = cfg.DEFAULT_SPREAD_PIPS * cfg.POINTS_PER_PIP
    spread_pips = spread_pts / cfg.POINTS_PER_PIP
    return (spread_pips / 2.0) * cfg.PIP_SIZE


# ─── main backtest ──────────────────────────────────────────────────────────
@dataclass
class BacktestResult:
    closed: list[ClosedTrade] = field(default_factory=list)
    equity_curve: list[tuple[int, float]] = field(default_factory=list)  # (time_msc, equity)
    spread_missing_bars: int = 0
    spread_total_bars: int = 0
    # Compliance counters
    news_blocked: int = 0
    cooldown_blocked: int = 0
    cap_blocked: int = 0
    daily_dd_halted_days: int = 0
    total_dd_halted: bool = False
    risk_floor_skips: int = 0
    intrabar_dd_breach_count: int = 0   # any single-trade % loss > 5%
    min_hold_breach_count: int = 0      # always 0 by construction (we defer)


def run_backtest(
    ngd: Optional[NYGoldData] = None,
    news_calendar: Optional[StaticNewsCalendar] = None,
    verbose: bool = True,
) -> BacktestResult:
    if ngd is None:
        ngd = NYGoldData.load()
    if news_calendar is None:
        news_calendar = StaticNewsCalendar()

    detector = NYGoldSweepDetector()
    result = BacktestResult()

    balance = cfg.INITIAL_BALANCE
    starting_balance = cfg.INITIAL_BALANCE
    sod_balance = balance       # start-of-day for §7.5 daily DD
    sod_day = -1
    daily_trade_count = 0
    daily_dd_halted = False
    last_closed_loss_time: Optional[int] = None
    consecutive_losses_today = 0
    cooldown_until_ms: Optional[int] = None
    open_trade: Optional[OpenTrade] = None
    pending_signal: Optional[NYGoldSignal] = None

    bars_1m = ngd.one_m
    n_1m = len(bars_1m)

    # Helper: convert a 1M index → its open_time
    def _open_t(i: int) -> int:
        return int(bars_1m.time_msc[i])

    # Helper: close an open trade and record
    def _close_trade(
        ot: OpenTrade, exit_idx: int, exit_time_msc: int,
        exit_price: float, reason: str,
    ) -> ClosedTrade:
        nonlocal balance, daily_trade_count, consecutive_losses_today
        nonlocal last_closed_loss_time, cooldown_until_ms

        if ot.direction == "BUY":
            pnl_price = exit_price - ot.entry_price
        else:
            pnl_price = ot.entry_price - exit_price
        pnl_pips = pnl_price / cfg.PIP_SIZE
        pnl_gross_usd = pnl_pips * ot.lot * cfg.USD_PER_PIP_PER_LOT
        commission = ot.lot * cfg.COMMISSION_USD_PER_LOT_ROUND_TURN
        pnl_net = pnl_gross_usd - commission

        balance += pnl_net
        r_mult = pnl_pips / ot.risk_pips if ot.risk_pips > 0 else 0.0

        # Compliance counter: intrabar DD breach (single-trade % loss > 5%)
        if pnl_net < 0 and abs(pnl_net) > 0.05 * sod_balance:
            result.intrabar_dd_breach_count += 1

        # Cooldown rule (§7.8)
        if pnl_net <= 0:
            consecutive_losses_today += 1
            last_closed_loss_time = exit_time_msc
            cd_min = (
                cfg.COOLDOWN_AFTER_TWO_LOSSES_MIN
                if consecutive_losses_today >= 2
                else cfg.COOLDOWN_AFTER_LOSS_MIN
            )
            cooldown_until_ms = exit_time_msc + cd_min * 60_000
        else:
            consecutive_losses_today = 0
            cooldown_until_ms = None

        return ClosedTrade(
            open=ot, exit_idx=exit_idx, exit_time_msc=exit_time_msc,
            exit_price=exit_price, exit_reason=reason,
            pnl_pips=pnl_pips, pnl_usd=pnl_net,
            commission_usd=commission, balance_after=balance, r_multiple=r_mult,
        )

    # ─── main loop ──────────────────────────────────────────────────
    t0 = time.time()
    for t, idx in ngd.iter_decisions():
        # Skip weekends entirely (no NY session)
        open_t = _open_t(idx)
        # We need to manage existing open trades EVERY bar — including
        # bars outside the NY entry window, in case time-stop / flatten
        # / SL / TP triggers. So we process management first, then entry.

        # ─── per-bar position management (every bar) ────────────────
        if open_trade is not None:
            mgmt_result = _manage_position(
                open_trade, bars_1m, idx, t, _close_trade, result,
            )
            if mgmt_result is not None:
                result.closed.append(mgmt_result)
                open_trade = None

        # Day rollover — reset daily counters at UTC midnight
        day = utc_date_key(open_t)
        if day != sod_day:
            sod_day = day
            sod_balance = balance
            daily_trade_count = 0
            daily_dd_halted = False
            consecutive_losses_today = 0

        # Total DD halt (§7.6)
        if balance <= starting_balance * (1 - cfg.TOTAL_DD_HALT_PCT / 100.0):
            result.total_dd_halted = True
            # Continue managing open trade, but no new entries
            result.equity_curve.append((open_t, balance))
            continue

        # Daily DD halt (§7.5)
        if not daily_dd_halted and balance <= sod_balance * (1 - cfg.DAILY_DD_HALT_PCT / 100.0):
            daily_dd_halted = True
            result.daily_dd_halted_days += 1
        # daily_dd_halted blocks new entries until next UTC day

        # ─── pending-signal fill (next-bar-open) ────────────────────
        # If we got a signal at the previous decision (t_prev = idx-1's close),
        # the entry bar IS the current bar `idx`. Fill at this bar's open.
        if pending_signal is not None and open_trade is None:
            entry_ok = (
                not daily_dd_halted
                and not result.total_dd_halted
                and daily_trade_count < cfg.MAX_TRADES_PER_DAY
            )
            if entry_ok:
                open_trade = _fill_entry(pending_signal, bars_1m, idx, result)
                if open_trade is not None:
                    daily_trade_count += 1
            pending_signal = None

        # ─── detector call (only during NY session) ─────────────────
        in_ny = in_session(
            open_t, cfg.NY_SESSION_START_UTC, cfg.NY_SESSION_END_UTC,
        )
        # Track spread coverage on ALL session bars for QA
        if in_ny:
            sp = float(bars_1m.spread_pts[idx])
            result.spread_total_bars += 1
            if math.isnan(sp) or sp <= 0:
                result.spread_missing_bars += 1

        if not in_ny:
            result.equity_curve.append((open_t, balance))
            continue

        # §1 — skip first 1M of session (open_t hh:mm == 12:00)
        h, m, _ = utc_hms(open_t)
        sh, sm, _ = cfg.NY_SESSION_START_UTC
        if h == sh and m < sm + cfg.SESSION_SKIP_FIRST_MIN:
            result.equity_curve.append((open_t, balance))
            continue
        eh, em, _ = cfg.NY_SESSION_END_UTC
        # §1 — no new entries in the last 5M
        last_5m_start = (eh * 60 + em) - cfg.SESSION_NO_NEW_ENTRY_LAST_MIN
        cur_min = h * 60 + m
        no_new_entry_window = cur_min >= last_5m_start

        # Gates: only run detector if entry is plausible
        gates_block = (
            open_trade is not None
            or daily_dd_halted
            or result.total_dd_halted
            or daily_trade_count >= cfg.MAX_TRADES_PER_DAY
            or no_new_entry_window
        )

        if cooldown_until_ms is not None and t < cooldown_until_ms:
            gates_block = True

        # News blackout — check at t (signal trigger time)
        in_blackout = news_calendar.is_news_blackout(
            cfg.SYMBOL, t, window_min=cfg.NEWS_BLACKOUT_WINDOW_MIN,
        )

        if not gates_block:
            v1 = ngd.get_visible("1M", t)
            v5 = ngd.get_visible("5M", t)
            v15 = ngd.get_visible("15M", t)
            ngd.assert_no_lookahead(t, v1, v5, v15)
            sig = detector.detect(t, idx, v1, v5, v15)
            if sig is not None:
                if in_blackout:
                    result.news_blocked += 1
                else:
                    pending_signal = sig
        else:
            # Count gate reasons for compliance reporting
            if open_trade is None and not result.total_dd_halted:
                if cooldown_until_ms is not None and t < cooldown_until_ms:
                    result.cooldown_blocked += 1
                elif daily_trade_count >= cfg.MAX_TRADES_PER_DAY:
                    result.cap_blocked += 1

        result.equity_curve.append((open_t, balance))

    # Close any still-open trade at the last bar (defensive — should be 0 due to
    # session flatten, but cover if data ends mid-session)
    if open_trade is not None:
        last_idx = n_1m - 1
        last_open_t = _open_t(last_idx)
        final_close = _close_trade(
            open_trade, last_idx, last_open_t,
            float(bars_1m.open[last_idx]) - (
                _half_spread_price(float(bars_1m.spread_pts[last_idx]))
                * (1 if open_trade.direction == "BUY" else -1)
            ),
            "data_end",
        )
        result.closed.append(final_close)
        open_trade = None

    if verbose:
        print(f"backtest loop done in {time.time() - t0:.1f}s")
    return result


def _fill_entry(
    sig: NYGoldSignal, bars_1m, idx: int, result: BacktestResult,
) -> Optional[OpenTrade]:
    """Fill the pending entry at bars_1m[idx].open with ask/bid + slip.

    Returns None if final risk falls below MIN_RISK_PIPS at the actual fill
    (the detector's risk-floor check used R.close as proxy — fill price may
    be slightly different, so we re-check).
    """
    mid_open = float(bars_1m.open[idx])
    spread_pts = float(bars_1m.spread_pts[idx])
    half_spread = _half_spread_price(spread_pts)
    slip_price = cfg.SLIPPAGE_PIPS * cfg.PIP_SIZE

    if sig.direction == "BUY":
        entry_price = mid_open + half_spread + slip_price
        risk_price = entry_price - sig.sl
    else:
        entry_price = mid_open - half_spread - slip_price
        risk_price = sig.sl - entry_price

    risk_pips = risk_price / cfg.PIP_SIZE
    if risk_pips < cfg.MIN_RISK_PIPS:
        result.risk_floor_skips += 1
        return None

    # §7.4 sizing
    # Need running balance — pull from caller via result is awkward; the
    # caller passes balance via closure. Simpler: read closing balance off
    # the last equity_curve entry, fall back to INITIAL.
    if result.equity_curve:
        balance = result.equity_curve[-1][1]
    elif result.closed:
        balance = result.closed[-1].balance_after
    else:
        balance = cfg.INITIAL_BALANCE
    risk_usd = balance * cfg.RISK_PCT / 100.0
    raw_lot = risk_usd / (risk_pips * cfg.USD_PER_PIP_PER_LOT)
    lot = _round_lot(raw_lot)
    if lot <= 0:
        result.risk_floor_skips += 1
        return None

    # §6 min-hold check — would SL/TP have triggered in the entry bar itself?
    entry_bar_high = float(bars_1m.high[idx])
    entry_bar_low = float(bars_1m.low[idx])

    deferred_price: Optional[float] = None
    deferred_reason: Optional[str] = None
    if sig.direction == "BUY":
        # Long exit checked vs. bid = mid - half_spread
        bid_low = entry_bar_low - half_spread
        bid_high = entry_bar_high - half_spread
        # §9 ¶4 — same-bar SL+TP precedence: SL wins
        if bid_low <= sig.sl:
            deferred_price = sig.sl
            deferred_reason = "sl_minhold"
        elif bid_high >= sig.tp:
            deferred_price = sig.tp
            deferred_reason = "tp_minhold"
    else:
        ask_low = entry_bar_low + half_spread
        ask_high = entry_bar_high + half_spread
        if ask_high >= sig.sl:
            deferred_price = sig.sl
            deferred_reason = "sl_minhold"
        elif ask_low <= sig.tp:
            deferred_price = sig.tp
            deferred_reason = "tp_minhold"

    return OpenTrade(
        direction=sig.direction,
        entry_idx=idx,
        entry_time_msc=int(bars_1m.time_msc[idx]),
        entry_price=entry_price,
        sl=sig.sl, tp=sig.tp, lot=lot,
        grade=sig.grade,
        sweep_depth_pips=sig.sweep_depth_pips,
        reversal_kind=sig.reversal_kind,
        tp_mode=sig.tp_mode,
        level_price=sig.level_price,
        risk_pips=risk_pips, risk_usd=risk_usd,
        deferred_exit_price=deferred_price,
        deferred_exit_reason=deferred_reason,
    )


def _manage_position(
    ot: OpenTrade, bars_1m, idx: int, t: int, close_fn, result: BacktestResult,
) -> Optional[ClosedTrade]:
    """One bar of position management. Returns closed trade or None."""
    # Skip the entry bar itself — exits start from R+2 (§6 min-hold deferral).
    if idx == ot.entry_idx:
        return None

    bar_open_t = int(bars_1m.time_msc[idx])

    # §6 deferred exit (resolves on R+2 open)
    if ot.deferred_exit_price is not None and idx == ot.entry_idx + 1:
        return close_fn(
            ot, idx, bar_open_t, ot.deferred_exit_price,
            ot.deferred_exit_reason,
        )

    # §6 session flatten — open_t >= 17:00 UTC closes at this bar's open
    eh, em, es = cfg.NY_SESSION_END_UTC
    h, m, _ = utc_hms(bar_open_t)
    if (h, m) >= (eh, em):
        # Use bar open (mid) adjusted to bid/ask for exit
        spread_pts = float(bars_1m.spread_pts[idx])
        half_spread = _half_spread_price(spread_pts)
        if ot.direction == "BUY":
            exit_price = float(bars_1m.open[idx]) - half_spread
        else:
            exit_price = float(bars_1m.open[idx]) + half_spread
        return close_fn(ot, idx, bar_open_t, exit_price, "session_flatten")

    # §6 time stop
    age_min = (bar_open_t - ot.entry_time_msc) / 60_000.0
    if age_min >= cfg.TIME_STOP_MIN:
        spread_pts = float(bars_1m.spread_pts[idx])
        half_spread = _half_spread_price(spread_pts)
        if ot.direction == "BUY":
            exit_price = float(bars_1m.open[idx]) - half_spread
        else:
            exit_price = float(bars_1m.open[idx]) + half_spread
        return close_fn(ot, idx, bar_open_t, exit_price, "time_stop")

    # SL / TP — bid/ask exit check (§5.4 / §8.2)
    spread_pts = float(bars_1m.spread_pts[idx])
    half_spread = _half_spread_price(spread_pts)
    hi = float(bars_1m.high[idx])
    lo = float(bars_1m.low[idx])

    if ot.direction == "BUY":
        bid_low = lo - half_spread
        bid_high = hi - half_spread
        hit_sl = bid_low <= ot.sl
        hit_tp = bid_high >= ot.tp
        if hit_sl and hit_tp:
            # §9 ¶4 SL wins
            return close_fn(ot, idx, bar_open_t, ot.sl, "sl")
        if hit_sl:
            return close_fn(ot, idx, bar_open_t, ot.sl, "sl")
        if hit_tp:
            return close_fn(ot, idx, bar_open_t, ot.tp, "tp")
    else:
        ask_low = lo + half_spread
        ask_high = hi + half_spread
        hit_sl = ask_high >= ot.sl
        hit_tp = ask_low <= ot.tp
        if hit_sl and hit_tp:
            return close_fn(ot, idx, bar_open_t, ot.sl, "sl")
        if hit_sl:
            return close_fn(ot, idx, bar_open_t, ot.sl, "sl")
        if hit_tp:
            return close_fn(ot, idx, bar_open_t, ot.tp, "tp")

    return None


# ─── reporting ──────────────────────────────────────────────────────────────
def _summarise(result: BacktestResult) -> dict:
    n = len(result.closed)
    if n == 0:
        return {
            "trades": 0, "wins": 0, "losses": 0,
            "wr": 0.0, "pf": 0.0, "exp_r": 0.0, "exp_usd": 0.0,
            "mdd_pct": 0.0, "mdd_usd": 0.0,
            "avg_hold_min": 0.0, "profitable_days": 0,
            "grade_a": 0, "grade_b": 0,
            "final_balance": cfg.INITIAL_BALANCE,
            "net_pnl_usd": 0.0,
        }
    wins = [c for c in result.closed if c.pnl_usd > 0]
    losses = [c for c in result.closed if c.pnl_usd <= 0]
    gross_win = sum(c.pnl_usd for c in wins)
    gross_loss = -sum(c.pnl_usd for c in losses)
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    wr = len(wins) / n
    exp_r = sum(c.r_multiple for c in result.closed) / n
    exp_usd = sum(c.pnl_usd for c in result.closed) / n
    avg_hold_min = sum(
        (c.exit_time_msc - c.open.entry_time_msc) / 60_000.0
        for c in result.closed
    ) / n

    # Max drawdown on equity curve
    if result.equity_curve:
        eq = np.asarray([e[1] for e in result.equity_curve], dtype=np.float64)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        mdd_pct = float(dd.max() * 100.0)
        mdd_usd = float((peak - eq).max())
        final_balance = float(eq[-1])
    else:
        mdd_pct = mdd_usd = 0.0
        final_balance = cfg.INITIAL_BALANCE

    # Profitable days (UTC)
    by_day: dict[int, float] = {}
    for c in result.closed:
        d = utc_date_key(c.exit_time_msc)
        by_day[d] = by_day.get(d, 0.0) + c.pnl_usd
    profitable_days = sum(1 for v in by_day.values() if v > 0)
    total_days = len(by_day)

    grade_a = sum(1 for c in result.closed if c.open.grade == "A")
    grade_b = sum(1 for c in result.closed if c.open.grade == "B")

    return {
        "trades": n,
        "wins": len(wins), "losses": len(losses),
        "wr": wr, "pf": pf,
        "exp_r": exp_r, "exp_usd": exp_usd,
        "mdd_pct": mdd_pct, "mdd_usd": mdd_usd,
        "avg_hold_min": avg_hold_min,
        "profitable_days": profitable_days, "total_days": total_days,
        "grade_a": grade_a, "grade_b": grade_b,
        "final_balance": final_balance,
        "net_pnl_usd": final_balance - cfg.INITIAL_BALANCE,
        "gross_win_usd": gross_win, "gross_loss_usd": gross_loss,
    }


def _verdict(summary: dict) -> str:
    """§12 SHIP / TUNE / CUT."""
    pf = summary["pf"]
    wr = summary["wr"] * 100.0
    mdd = summary["mdd_pct"]
    if pf >= 1.40 and wr >= 48.0 and mdd <= 6.0:
        return "SHIP"
    if pf < 1.20 or wr < 45.0 or mdd > 8.0:
        return "CUT"
    return "TUNE"


def _write_equity_csv(result: BacktestResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_msc", "balance_usd"])
        for ts, eq in result.equity_curve:
            w.writerow([ts, f"{eq:.4f}"])


def _write_trades_csv(result: BacktestResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "entry_time_msc", "exit_time_msc", "direction", "grade", "tp_mode",
            "entry_price", "sl", "tp", "exit_price", "exit_reason",
            "lot", "risk_pips", "risk_usd",
            "pnl_pips", "pnl_usd", "commission_usd", "r_multiple",
            "balance_after", "sweep_depth_pips", "reversal_kind",
            "level_price",
        ])
        for c in result.closed:
            ot = c.open
            w.writerow([
                ot.entry_time_msc, c.exit_time_msc, ot.direction, ot.grade, ot.tp_mode,
                f"{ot.entry_price:.4f}", f"{ot.sl:.4f}", f"{ot.tp:.4f}",
                f"{c.exit_price:.4f}", c.exit_reason,
                f"{ot.lot:.2f}", f"{ot.risk_pips:.3f}", f"{ot.risk_usd:.2f}",
                f"{c.pnl_pips:.3f}", f"{c.pnl_usd:.2f}", f"{c.commission_usd:.2f}",
                f"{c.r_multiple:.3f}",
                f"{c.balance_after:.2f}",
                f"{ot.sweep_depth_pips:.2f}", ot.reversal_kind,
                f"{ot.level_price:.4f}",
            ])


def _write_report_md(
    result: BacktestResult, summary: dict, verdict: str, path: Path,
) -> None:
    spread_cov_pct = (
        100.0 * (1.0 - result.spread_missing_bars / max(1, result.spread_total_bars))
    )
    pf_str = f"{summary['pf']:.2f}" if summary["pf"] != float("inf") else "inf"
    body = [
        "# NY Gold Sweep — Baseline Backtest Report v1.1",
        "",
        f"Run: full 2y XAUUSD, NY session 12:00-17:00 UTC, TP_MODE='C' (hybrid),",
        f"all defaults per `config/ny_gold_sweep_config.py`.",
        "",
        "## Summary",
        "",
        f"- Trades                  : **{summary['trades']}**",
        f"- Wins / Losses           : {summary['wins']} / {summary['losses']}",
        f"- Win rate                : **{summary['wr']*100:.2f}%**",
        f"- Profit factor           : **{pf_str}**",
        f"- Expectancy per trade    : **{summary['exp_r']:+.3f} R**  /  **${summary['exp_usd']:+.2f}**",
        f"- Max drawdown            : **{summary['mdd_pct']:.2f}%**  /  ${summary['mdd_usd']:,.2f}",
        f"- Avg hold                : {summary['avg_hold_min']:.1f} min",
        f"- Profitable days         : {summary['profitable_days']} / {summary['total_days']}",
        f"- Grade A / B split       : {summary['grade_a']} A  /  {summary['grade_b']} B",
        f"- Net P&L                 : **${summary['net_pnl_usd']:+,.2f}** "
        f"(final balance ${summary['final_balance']:,.2f})",
        f"- Gross win / loss        : ${summary['gross_win_usd']:,.2f} / ${summary['gross_loss_usd']:,.2f}",
        "",
        "## Compliance flags (§7)",
        "",
        f"- Signals blocked by news blackout : {result.news_blocked}",
        f"- Signals blocked by cooldown      : {result.cooldown_blocked}",
        f"- Signals blocked by daily cap     : {result.cap_blocked}",
        f"- Daily-DD halted days             : {result.daily_dd_halted_days}",
        f"- Total-DD halted (10% breached?)  : {result.total_dd_halted}",
        f"- Risk-floor skips (entry-side)    : {result.risk_floor_skips}",
        f"- Min-hold (<60s) breaches         : {result.min_hold_breach_count}  (0 by construction)",
        f"- Single-trade DD > 5% breaches    : {result.intrabar_dd_breach_count}",
        f"- Spread coverage                  : {spread_cov_pct:.2f}% "
        f"({result.spread_missing_bars} missing / {result.spread_total_bars} session bars)",
        "",
        "## Verdict (§12)",
        "",
        f"### **{verdict}**",
        "",
        _verdict_explanation(verdict, summary),
        "",
        "## Files written",
        "",
        "- `out/ny_gold_equity.csv`   — equity curve, 1 row per 1M bar",
        "- `out/ny_gold_trades.csv`   — closed trades, one row per trade",
        "",
        "## Methodology notes",
        "",
        "- Look-ahead invariant (§0.1 / §9) enforced via `assert_no_lookahead`",
        "  called on EVERY decision in the main loop (not just smoke).",
        "- Entries: next-bar-open only. Long fills at `ask = mid + spread/2 + slip`;",
        "  short mirror. SL/TP exits checked against bid (long) / ask (short) on",
        "  every closed bar after entry, using current-bar spread (§5.4 / §8.2).",
        "- Same-bar SL+TP precedence: SL wins (§9 ¶4).",
        "- 60s min-hold (§6): if SL/TP would trigger in the entry bar (R+1), the",
        "  exit is deferred to R+2 open at the breached price.",
        "- Static SL/TP only (§6 v1.1): no partials, no BE shift, no trailing.",
        "  TP_MODE='C' picks opposing-level TP if RR≥1.0 (capped at 3.0R),",
        "  else falls back to fixed 1.5R.",
        "- Per-bar spread from data (`spread_mean` column / 100 = pips, §0.2).",
        "- Position sizing: 0.5% risk on running balance, lot rounded to 0.01.",
        "- Commission: $7 / lot / round-turn (§8.4).",
        "- News blackout: ±5 min, static calendar (`data/news_calendar.py`).",
        "  The static calendar's coverage is limited — actual NFP/FOMC events",
        "  outside its window were NOT blocked in this backtest.",
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def _verdict_explanation(verdict: str, summary: dict) -> str:
    pf = summary["pf"]
    wr = summary["wr"] * 100.0
    mdd = summary["mdd_pct"]
    pf_str = f"{pf:.2f}" if pf != float('inf') else "inf"
    if verdict == "SHIP":
        return (
            f"PF {pf_str} ≥ 1.40, WR {wr:.2f}% ≥ 48%, MDD {mdd:.2f}% ≤ 6%. "
            "All §12 SHIP thresholds met. Move to live paper-trade validation."
        )
    if verdict == "CUT":
        reasons = []
        if pf < 1.20:
            reasons.append(f"PF {pf_str} < 1.20")
        if wr < 45.0:
            reasons.append(f"WR {wr:.2f}% < 45%")
        if mdd > 8.0:
            reasons.append(f"MDD {mdd:.2f}% > 8%")
        return (
            f"§12 CUT triggered: {', '.join(reasons) if reasons else 'unknown reason'}. "
            "No tuning — move on. Per spec §12: 'No second chances; move to next "
            "pair or next strategy.'"
        )
    # TUNE
    return (
        f"PF {pf_str} or MDD {mdd:.2f}% fall in the TUNE band (PF 1.20-1.40 "
        f"or MDD 6-8%). Spec §12 candidate tweaks: tighten penetration bounds, "
        "restrict TP to Mode A only, or increase min-touches to 3 for grade A."
    )


# ─── CLI entry ──────────────────────────────────────────────────────────────
def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "out"
    out_dir.mkdir(exist_ok=True)
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    docs_dir.mkdir(exist_ok=True)

    print("Loading data...")
    ngd = NYGoldData.load()
    print(f"  1M={len(ngd.one_m):,}  5M={len(ngd.five_m):,}  "
          f"15M={len(ngd.fifteen_m):,}  1H={len(ngd.one_h):,}")
    cal = StaticNewsCalendar()
    print(f"  news events loaded: {len(cal.events)}")

    print("Running backtest (TP_MODE='C', full defaults)...")
    result = run_backtest(ngd=ngd, news_calendar=cal, verbose=True)
    summary = _summarise(result)
    verdict = _verdict(summary)

    # CSVs
    eq_path = out_dir / "ny_gold_equity.csv"
    tr_path = out_dir / "ny_gold_trades.csv"
    _write_equity_csv(result, eq_path)
    _write_trades_csv(result, tr_path)

    # Report
    rep_path = docs_dir / "NY_GOLD_BACKTEST_REPORT.md"
    _write_report_md(result, summary, verdict, rep_path)

    # Console summary
    pf_str = f"{summary['pf']:.2f}" if summary['pf'] != float('inf') else "inf"
    print()
    print("=" * 72)
    print("  NY GOLD SWEEP — BASELINE RESULT  (v1.1, TP_MODE='C')")
    print("=" * 72)
    print(f"  trades         : {summary['trades']:,}")
    print(f"  wins / losses  : {summary['wins']} / {summary['losses']}")
    print(f"  win rate       : {summary['wr']*100:.2f} %")
    print(f"  profit factor  : {pf_str}")
    print(f"  expectancy     : {summary['exp_r']:+.3f} R   /   ${summary['exp_usd']:+.2f}")
    print(f"  max drawdown   : {summary['mdd_pct']:.2f} %  /  ${summary['mdd_usd']:,.2f}")
    print(f"  avg hold       : {summary['avg_hold_min']:.1f} min")
    print(f"  profitable days: {summary['profitable_days']} / {summary['total_days']}")
    print(f"  grade A / B    : {summary['grade_a']} / {summary['grade_b']}")
    print(f"  net P&L        : ${summary['net_pnl_usd']:+,.2f}  "
          f"(final ${summary['final_balance']:,.2f})")
    print(f"  --- compliance ---")
    print(f"  news-blocked   : {result.news_blocked}")
    print(f"  cooldown-block : {result.cooldown_blocked}")
    print(f"  cap-blocked    : {result.cap_blocked}")
    print(f"  daily-DD halts : {result.daily_dd_halted_days}")
    print(f"  total-DD halt? : {result.total_dd_halted}")
    print(f"  risk-floor skip: {result.risk_floor_skips}")
    print(f"  min-hold breach: {result.min_hold_breach_count}")
    print(f"  trade-DD>5%    : {result.intrabar_dd_breach_count}")
    print("-" * 72)
    print(f"  VERDICT (§12)  : {verdict}")
    print("=" * 72)
    print(f"  report : {rep_path}")
    print(f"  equity : {eq_path}")
    print(f"  trades : {tr_path}")


if __name__ == "__main__":
    main()
