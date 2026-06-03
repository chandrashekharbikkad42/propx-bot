"""propX Multi-Setup — full backtest (28 pairs, 1H+15M, ~729d).

Walks 15M bars chronologically across all 28 pairs, runs the 4 detectors
with strict no-look-ahead bias, applies confluence resolution, enforces
prop-firm constraints (max 2 trades/day global, 0.5 % risk per trade,
daily/total DD halts, news blackout), simulates the spec §7 trade
lifecycle (TP1=1.5R partial 50 % → BE → trail 0.3R → TP2=2.5R, 48h
time stop, Friday flatten), books realistic spread + commission +
slippage, and writes a full report + equity PNG + trade CSV.

Honesty notes:
  - Entries: ALL setups simulated as MARKET at the signal bar's CLOSE
    with `0.5 × MAX_SPREAD_PIPS` slippage (spec §8.3). Real LIMIT entries
    (Liquidity Sweep retest, Order Block top) may fill or expire — this
    simplification slightly OVER-trades the strategy. Discussed in report.
  - SL fills: assumed exact at SL price (spec §8.3). On gap-down bars this
    is OPTIMISTIC. Documented.
  - Same-bar SL+TP: SL wins (conservative — assume worst path).
  - News blackout: wired via `data.news_calendar`. The static fallback
    only covers May–July 2026, so over the 2024–2026 backtest the filter
    blocks ~0 trades. Real news data would tighten results; flagged.
  - Pip value: computed per-bar from quote currency → USD using current
    cross-rates (USDJPY, GBPUSD, etc.) read off the same parquet stream.

Run:
    python scripts/multi_setup_backtest.py
    python scripts/multi_setup_backtest.py --days 365
    python scripts/multi_setup_backtest.py --pairs EURUSD,GBPUSD,XAUUSD
"""

from __future__ import annotations
import argparse
import io
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Force UTF-8 stdout on Windows (cp1252 chokes on em-dashes / arrows).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, io.UnsupportedOperation):
    pass

# Allow running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Lazy matplotlib import (only for plotting) to avoid heavy boot.
from config.multi_setup_config import (  # noqa: E402
    BE_SHIFT_R, CONTRACT_SIZE, DAILY_DD_PCT, FRIDAY_FLATTEN_UTC,
    MAX_CONCURRENT, MAX_SPREAD_PIPS, MAX_TRADES_PER_DAY,
    MIN_RISK_PIPS, NEWS_BLACKOUT_MIN,
    PAIRS as ALL_PAIRS,
    PARTIAL_FRACTION, PIP_SIZE, RISK_PCT,
    SLIPPAGE_MARKET_FRAC_OF_MAX_SPREAD,
    TIME_STOP_HOURS, TP1_R, TP2_R, TRAIL_STEP_R,
    pip_size_for,
)
from data.bar_aggregator import Bar, read_bars_parquet  # noqa: E402
from data.news_calendar import DEFAULT_CALENDAR  # noqa: E402
from strategy.multi_setup_config import (  # noqa: E402
    build_multi_setup_detectors, resolve_confluence,
)
from strategy.patterns._multi_setup_common import (  # noqa: E402
    is_rejection_bearish, is_rejection_bullish,
)
from strategy.patterns.base import Direction, MarketContext  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Config (backtest-only — strategy constants live in config/multi_setup_config)
# ─────────────────────────────────────────────────────────────────────────────
UTC = timezone.utc
INITIAL_BALANCE: float = 10_000.0
DAILY_DD_PCT_HARD: float = 5.0       # The5%ers / FTMO-style daily soft cap
TOTAL_DD_PCT_HARD: float = 10.0
COMMISSION_PER_LOT_ROUNDTURN_USD: float = 7.0
LOT_MIN: float = 0.01
LOT_STEP: float = 0.01
LOT_MAX: float = 50.0

# Bar slice limits — bound the windows passed to detectors for speed.
HTF_WINDOW: int = 260
LTF_WINDOW: int = 80


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def df_to_bars(df: pd.DataFrame, symbol: str) -> List[Bar]:
    out: List[Bar] = []
    for r in df.itertuples():
        out.append(Bar(
            symbol=symbol, time_msc=int(r.time_msc),
            open=float(r.open), high=float(r.high), low=float(r.low),
            close=float(r.close), volume=int(r.volume),
            spread_mean=float(getattr(r, "spread_mean", 0.0)),
        ))
    return out


def load_all_pairs(
    pairs: Tuple[str, ...], lookback_days: Optional[int] = None,
) -> Tuple[Dict[str, List[Bar]], Dict[str, List[Bar]]]:
    htf: Dict[str, List[Bar]] = {}
    ltf: Dict[str, List[Bar]] = {}
    for sym in pairs:
        try:
            d1 = read_bars_parquet(sym, "1H").sort_values("time_msc")
            d15 = read_bars_parquet(sym, "15M").sort_values("time_msc")
        except FileNotFoundError as exc:
            print(f"  [skip] {sym}: {exc}")
            continue
        if lookback_days is not None:
            cutoff_msc = int(d15["time_msc"].max()) - lookback_days * 24 * 60 * 60 * 1000
            d1 = d1[d1["time_msc"] >= cutoff_msc - 7 * 24 * 60 * 60 * 1000]
            d15 = d15[d15["time_msc"] >= cutoff_msc]
        htf[sym] = df_to_bars(d1.reset_index(drop=True), sym)
        ltf[sym] = df_to_bars(d15.reset_index(drop=True), sym)
        print(f"  {sym:<8} 1H={len(htf[sym]):>6}  15M={len(ltf[sym]):>6}")
    return htf, ltf


# ─────────────────────────────────────────────────────────────────────────────
# Pip-value (quote → USD) calculator
# ─────────────────────────────────────────────────────────────────────────────

def pip_value_usd_per_lot(
    symbol: str, current_prices: Dict[str, float],
) -> float:
    """USD value of 1 pip movement for 1 standard lot.

    For USD-quoted pairs (EURUSD etc) and XAUUSD → already in USD.
    For USD/XXX pairs → divide by current USD/XXX price.
    For crosses → convert quote currency to USD via the quote/USD rate
    (uses USDJPY for JPY-quoted, XXXUSD spot for major-quoted).
    Returns 10.0 as a defensive fallback when we cannot resolve the rate.
    """
    pip = PIP_SIZE[symbol]
    contract = CONTRACT_SIZE[symbol]
    raw = pip * contract  # in quote currency

    if symbol == "XAUUSD":
        return raw  # already USD
    quote = symbol[3:6]
    if quote == "USD":
        return raw  # already USD
    # Quote != USD — need quote → USD conversion at current spot.
    usd_quote_sym = f"USD{quote}"
    quote_usd_sym = f"{quote}USD"
    if usd_quote_sym in current_prices and current_prices[usd_quote_sym] > 0:
        return raw / current_prices[usd_quote_sym]
    if quote_usd_sym in current_prices and current_prices[quote_usd_sym] > 0:
        return raw * current_prices[quote_usd_sym]
    return 10.0  # ~$10/pip/lot fallback (warn once)


# ─────────────────────────────────────────────────────────────────────────────
# Open-trade record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpenTrade:
    pattern: str
    symbol: str
    direction: str             # "BUY" or "SELL"
    entry: float
    sl: float                  # mutates after BE shift / trail
    tp1: float
    tp2: float
    risk_price: float          # initial |entry - sl|
    lot_full: float            # initial lot size
    lot_remaining: float
    pip_value_usd: float       # at entry; assume constant for trade
    entry_time_msc: int
    bar_index_at_entry: int
    tp1_hit: bool = False
    realized_pnl_usd: float = 0.0
    commission_usd: float = 0.0
    confluence: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Trade lifecycle (per-bar update; no look-ahead)
# ─────────────────────────────────────────────────────────────────────────────

def update_open_trade(
    trade: OpenTrade, bar: Bar, symbol_max_spread_pips: float,
) -> Optional[Dict]:
    """Walk a single LTF bar against an open trade. Returns a CLOSED-trade
    dict if the trade closes during this bar (SL/TP2/timeout/friday/EOD),
    else None.

    Order-of-events convention (conservative):
      1. If SL is hit (low ≤ SL for long, high ≥ SL for short), close
         remaining at SL price. SL fills exact (spec §8.3).
      2. Else if TP1 not yet hit AND TP1 is hit this bar: close 50 %, shift
         SL to BE + 0.5 × spread, start trailing.
      3. Else if TP1 already hit AND TP2 hit this bar: close remaining at TP2.
      4. Update trailing SL after TP1 hit, based on bar.close.

    If SL and TP both touch this bar, we assume SL hits first (worst path).
    Returns None if trade is still open after this bar.
    """
    long = trade.direction == "BUY"
    pip = pip_size_for(trade.symbol)
    half_spread_price = 0.5 * symbol_max_spread_pips * pip

    # 1. SL hit?
    if long:
        if bar.low <= trade.sl:
            # Close remainder at SL price. SL fills exact (spec §8.3).
            pnl_remainder = (trade.sl - trade.entry) * trade.lot_remaining * \
                (1 / pip) * trade.pip_value_usd
            trade.realized_pnl_usd += pnl_remainder
            return _close_trade(trade, bar, exit_price=trade.sl,
                                exit_reason="SL" if not trade.tp1_hit else "TRAIL")
    else:
        if bar.high >= trade.sl:
            pnl_remainder = (trade.entry - trade.sl) * trade.lot_remaining * \
                (1 / pip) * trade.pip_value_usd
            trade.realized_pnl_usd += pnl_remainder
            return _close_trade(trade, bar, exit_price=trade.sl,
                                exit_reason="SL" if not trade.tp1_hit else "TRAIL")

    # 2. TP1 hit?
    if not trade.tp1_hit:
        tp1_hit_this_bar = (long and bar.high >= trade.tp1) or \
                           ((not long) and bar.low <= trade.tp1)
        if tp1_hit_this_bar:
            lot_to_close = trade.lot_remaining * PARTIAL_FRACTION
            lot_remaining_after = trade.lot_remaining - lot_to_close
            if long:
                pnl = (trade.tp1 - trade.entry) * lot_to_close * (1 / pip) * trade.pip_value_usd
            else:
                pnl = (trade.entry - trade.tp1) * lot_to_close * (1 / pip) * trade.pip_value_usd
            trade.realized_pnl_usd += pnl
            trade.lot_remaining = lot_remaining_after
            trade.tp1_hit = True
            # Shift SL to BE + half spread (cover round-trip cost).
            trade.sl = trade.entry + (half_spread_price if long else -half_spread_price)

    # 3. TP2 hit (only after tp1)?
    if trade.tp1_hit:
        tp2_hit_this_bar = (long and bar.high >= trade.tp2) or \
                           ((not long) and bar.low <= trade.tp2)
        if tp2_hit_this_bar:
            if long:
                pnl = (trade.tp2 - trade.entry) * trade.lot_remaining * (1 / pip) * trade.pip_value_usd
            else:
                pnl = (trade.entry - trade.tp2) * trade.lot_remaining * (1 / pip) * trade.pip_value_usd
            trade.realized_pnl_usd += pnl
            return _close_trade(trade, bar, exit_price=trade.tp2, exit_reason="TP2")

    # 4. Trail SL after TP1, based on bar.close.
    if trade.tp1_hit:
        trail_step_price = TRAIL_STEP_R * trade.risk_price
        if long:
            new_sl = bar.close - trail_step_price
            if new_sl > trade.sl:
                trade.sl = new_sl
        else:
            new_sl = bar.close + trail_step_price
            if new_sl < trade.sl:
                trade.sl = new_sl

    return None


def _close_trade(
    trade: OpenTrade, bar: Bar, exit_price: float, exit_reason: str,
) -> Dict:
    return {
        "symbol": trade.symbol, "pattern": trade.pattern,
        "direction": trade.direction,
        "entry_time": datetime.fromtimestamp(trade.entry_time_msc / 1000, tz=UTC),
        "exit_time": datetime.fromtimestamp(bar.time_msc / 1000, tz=UTC),
        "entry": trade.entry, "sl_initial": _initial_sl(trade),
        "tp1": trade.tp1, "tp2": trade.tp2,
        "exit_price": exit_price, "exit_reason": exit_reason,
        "tp1_hit": trade.tp1_hit, "lot": trade.lot_full,
        "risk_price": trade.risk_price,
        "pnl_gross_usd": round(trade.realized_pnl_usd, 4),
        "commission_usd": round(trade.commission_usd, 4),
        "pnl_net_usd": round(trade.realized_pnl_usd - trade.commission_usd, 4),
        "confluence": trade.confluence,
    }


def _initial_sl(trade: OpenTrade) -> float:
    """Recover the pre-trail SL value for the trade log."""
    if trade.direction == "BUY":
        return trade.entry - trade.risk_price
    return trade.entry + trade.risk_price


def force_close_at_market(
    trade: OpenTrade, bar: Bar, reason: str,
) -> Dict:
    """Close at bar.close (used by 48h time stop, Friday flatten)."""
    pip = pip_size_for(trade.symbol)
    if trade.direction == "BUY":
        pnl = (bar.close - trade.entry) * trade.lot_remaining * (1 / pip) * trade.pip_value_usd
    else:
        pnl = (trade.entry - bar.close) * trade.lot_remaining * (1 / pip) * trade.pip_value_usd
    trade.realized_pnl_usd += pnl
    return _close_trade(trade, bar, exit_price=bar.close, exit_reason=reason)


# ─────────────────────────────────────────────────────────────────────────────
# Event-driven backtest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestState:
    balance: float = INITIAL_BALANCE
    peak_balance: float = INITIAL_BALANCE
    open_trades: List[OpenTrade] = field(default_factory=list)
    closed_trades: List[Dict] = field(default_factory=list)
    day_trade_count: Dict[str, int] = field(default_factory=dict)
    day_start_balance: Dict[str, float] = field(default_factory=dict)
    equity_curve: List[Tuple[int, float]] = field(default_factory=list)  # (time_msc, balance)
    halted_until_day: Optional[str] = None
    total_dd_halted: bool = False
    current_prices: Dict[str, float] = field(default_factory=dict)
    blocked_news: int = 0
    blocked_spread: int = 0
    blocked_daily_cap: int = 0
    blocked_daily_dd: int = 0
    blocked_total_dd: int = 0
    blocked_concurrent: int = 0


def _date_key(time_msc: int) -> str:
    return datetime.fromtimestamp(time_msc / 1000, tz=UTC).strftime("%Y-%m-%d")


def run_backtest(
    htf_bars: Dict[str, List[Bar]],
    ltf_bars: Dict[str, List[Bar]],
    quiet: bool = False,
) -> BacktestState:
    detectors = build_multi_setup_detectors()
    news = DEFAULT_CALENDAR

    state = BacktestState()

    # Build a global time-ordered event stream of (time_msc, symbol, ltf_idx).
    events: List[Tuple[int, str, int]] = []
    for sym, bars in ltf_bars.items():
        for i, b in enumerate(bars):
            events.append((b.time_msc, sym, i))
    events.sort()
    if not events:
        return state
    start_msc = events[0][0]
    state.equity_curve.append((start_msc, state.balance))

    # HTF cursor cache: sym → last_htf_idx (closed bar strictly before LTF time).
    htf_cursor: Dict[str, int] = {sym: -1 for sym in htf_bars}

    t_wall_start = time.time()
    report_every = max(1, len(events) // 50)
    last_progress_msc: Optional[int] = None

    for ev_no, (t_msc, sym, k) in enumerate(events):
        ltf = ltf_bars[sym]
        htf = htf_bars[sym]
        cur_bar = ltf[k]

        # ── 1. Update current price for pip-value calcs.
        state.current_prices[sym] = cur_bar.close

        # ── 2. Walk open trades on THIS pair against THIS bar.
        still_open: List[OpenTrade] = []
        for t in state.open_trades:
            if t.symbol != sym:
                still_open.append(t)
                continue
            # Time stop (48h).
            elapsed_h = (t_msc - t.entry_time_msc) / (3600 * 1000)
            if elapsed_h >= TIME_STOP_HOURS:
                closed = force_close_at_market(t, cur_bar, "TIME_STOP")
                _book(state, closed)
                continue
            # Friday flatten — 23:45 UTC and later on Fri (15M cadence; spec 23:55).
            dt = datetime.fromtimestamp(t_msc / 1000, tz=UTC)
            if dt.weekday() == 4 and dt.hour >= 23 and dt.minute >= 45:
                closed = force_close_at_market(t, cur_bar, "FRIDAY_FLATTEN")
                _book(state, closed)
                continue
            # Normal SL / TP walk.
            max_spread = MAX_SPREAD_PIPS[sym]
            result = update_open_trade(t, cur_bar, max_spread)
            if result is not None:
                _book(state, result)
            else:
                still_open.append(t)
        state.open_trades = still_open

        # ── 3. Daily DD / Total DD halts (use BALANCE, conservative).
        day_key = _date_key(t_msc)
        if day_key not in state.day_start_balance:
            state.day_start_balance[day_key] = state.balance
            state.halted_until_day = None  # new day, reset halt
            state.day_trade_count[day_key] = 0

        if state.total_dd_halted:
            # Once we've blown 10 % total — STOP forever.
            continue

        day_loss_pct = (
            (state.day_start_balance[day_key] - state.balance)
            / state.day_start_balance[day_key] * 100
        )
        if day_loss_pct >= DAILY_DD_PCT_HARD:
            if state.halted_until_day != day_key:
                state.halted_until_day = day_key
            state.blocked_daily_dd += 1
            _equity_log(state, t_msc)
            continue

        total_dd_pct = (state.peak_balance - state.balance) / state.peak_balance * 100
        if total_dd_pct >= TOTAL_DD_PCT_HARD:
            state.total_dd_halted = True
            state.blocked_total_dd += 1
            _equity_log(state, t_msc)
            continue

        # ── 4. Global concurrent-trades cap.
        if len(state.open_trades) >= MAX_CONCURRENT:
            state.blocked_concurrent += 1
            _equity_log(state, t_msc)
            continue

        # ── 5. Daily entry cap (global across pairs).
        if state.day_trade_count.get(day_key, 0) >= MAX_TRADES_PER_DAY:
            state.blocked_daily_cap += 1
            _equity_log(state, t_msc)
            continue

        # ── 6. Pre-gate: ALL 4 detectors require a rejection candle on the
        # current LTF bar (perf optimisation — skips ~90 % of bars).
        if not (is_rejection_bullish(ltf, k) or is_rejection_bearish(ltf, k)):
            _equity_log(state, t_msc)
            continue

        # ── 7. News blackout (±2 min, per-pair).
        if news.is_news_blackout(sym, t_msc, NEWS_BLACKOUT_MIN):
            state.blocked_news += 1
            _equity_log(state, t_msc)
            continue

        # ── 8. HTF window (closed bars strictly before current LTF time).
        # Cache cursor — htf advances roughly every 4 LTF bars.
        cursor = htf_cursor[sym]
        while cursor + 1 < len(htf) and htf[cursor + 1].time_msc < t_msc:
            cursor += 1
        htf_cursor[sym] = cursor
        if cursor < HTF_WINDOW // 4:   # need some history
            _equity_log(state, t_msc)
            continue

        htf_lo = max(0, cursor - HTF_WINDOW + 1)
        htf_window = htf[htf_lo: cursor + 1]
        ltf_lo = max(0, k - LTF_WINDOW + 1)
        ltf_window = ltf[ltf_lo: k + 1]

        ctx = MarketContext(
            symbol=sym, current_time_msc=t_msc,
            htf_bars=tuple(htf_window), ltf_bars=tuple(ltf_window),
        )

        # ── 9. Run 4 detectors → raw signals.
        raw_sigs = []
        for d in detectors:
            try:
                sig = d.detect(ltf_window, ctx)
            except Exception as exc:  # noqa: BLE001
                if not quiet:
                    print(f"  [detector-err] {sym} {d.name}: {exc}")
                continue
            if sig is not None:
                raw_sigs.append(sig)
        if not raw_sigs:
            _equity_log(state, t_msc)
            continue

        # ── 10. Confluence resolution (merge same-dir, drop opposite-dir).
        resolved = resolve_confluence(raw_sigs)
        if not resolved:
            _equity_log(state, t_msc)
            continue

        # Pick best by (grade, confidence).
        best = resolved[0]

        # ── 11. Spread guard (per spec §1.6).
        # `spread_mean` in the parquet is stored in MT5 integer POINTS
        # (1 pip = 10 points on 5-digit brokers — including XAUUSD where
        # point=0.01 and pip=0.10). Convert points → pips before comparison.
        # Bars with spread_mean=0 (unpopulated) are treated as "unknown" and
        # pass through (we cannot reject on missing data).
        max_spread = MAX_SPREAD_PIPS[sym]
        spread_pips = cur_bar.spread_mean / 10.0
        if spread_pips > max_spread:
            state.blocked_spread += 1
            _equity_log(state, t_msc)
            continue

        # ── 12. Slippage on market entry (spec §8.3).
        pip = PIP_SIZE[sym]
        slip_price = SLIPPAGE_MARKET_FRAC_OF_MAX_SPREAD * max_spread * pip
        is_long = (best.direction == Direction.BUY)
        entry_px = best.entry + (slip_price if is_long else -slip_price)
        # Re-validate price ordering after slippage.
        if is_long and not (best.sl < entry_px < best.tp):
            _equity_log(state, t_msc)
            continue
        if (not is_long) and not (best.tp < entry_px < best.sl):
            _equity_log(state, t_msc)
            continue

        risk_price = abs(entry_px - best.sl)
        risk_pips = risk_price / pip
        if risk_pips < MIN_RISK_PIPS:
            _equity_log(state, t_msc)
            continue

        # Extract TP1 from confluences_met if present (spec §7 — TP1 = 1.5R).
        tp1_price = _extract_tp1(best.confluences_met, entry_px, risk_price, is_long)

        # ── 13. Sizing — 0.5 % of CURRENT balance.
        risk_usd = state.balance * (RISK_PCT / 100.0)
        pv_usd = pip_value_usd_per_lot(sym, state.current_prices)
        if pv_usd <= 0:
            _equity_log(state, t_msc)
            continue
        lot_raw = risk_usd / (risk_pips * pv_usd)
        lot = _round_lot(lot_raw)
        if lot < LOT_MIN:
            _equity_log(state, t_msc)
            continue
        lot = min(lot, LOT_MAX)

        # ── 14. Commission (round-turn).
        commission = COMMISSION_PER_LOT_ROUNDTURN_USD * lot

        trade = OpenTrade(
            pattern=best.pattern_name, symbol=sym,
            direction="BUY" if is_long else "SELL",
            entry=entry_px, sl=best.sl,
            tp1=tp1_price, tp2=best.tp,
            risk_price=risk_price, lot_full=lot, lot_remaining=lot,
            pip_value_usd=pv_usd, entry_time_msc=t_msc,
            bar_index_at_entry=k,
            commission_usd=commission,
            confluence="CONFLUENCE_" in best.pattern_name,
        )
        state.open_trades.append(trade)
        state.day_trade_count[day_key] = state.day_trade_count.get(day_key, 0) + 1
        _equity_log(state, t_msc)

        # ── Progress.
        if not quiet and ev_no % report_every == 0:
            pct = (ev_no / len(events)) * 100
            elapsed = time.time() - t_wall_start
            print(f"  {pct:5.1f}%  | bal=${state.balance:>9.2f} "
                  f"| trades_closed={len(state.closed_trades)} "
                  f"| open={len(state.open_trades)} "
                  f"| elapsed={elapsed:.0f}s")

    # End-of-stream: force-close any open trades at last bar.
    if state.open_trades:
        last_msc = events[-1][0]
        for t in list(state.open_trades):
            bars = ltf_bars[t.symbol]
            if not bars:
                continue
            last_bar = bars[-1]
            closed = force_close_at_market(t, last_bar, "END_OF_DATA")
            _book(state, closed)
        state.open_trades.clear()
        _equity_log(state, last_msc)

    return state


def _book(state: BacktestState, closed: Dict) -> None:
    state.balance += closed["pnl_net_usd"]
    if state.balance > state.peak_balance:
        state.peak_balance = state.balance
    state.closed_trades.append(closed)


def _equity_log(state: BacktestState, t_msc: int) -> None:
    # Append only when balance actually changed since last sample (compress).
    if state.equity_curve and state.equity_curve[-1][1] == state.balance:
        # Update timestamp only if last entry is the same balance — keep
        # the most recent t_msc so the plot extends to the end.
        state.equity_curve[-1] = (t_msc, state.balance)
        return
    state.equity_curve.append((t_msc, state.balance))


def _round_lot(lot: float) -> float:
    return math.floor(lot / LOT_STEP) * LOT_STEP


def _extract_tp1(confluences: tuple, entry: float, risk_price: float, is_long: bool) -> float:
    """Parse 'tp1_<price>' from confluences_met if present; else compute from spec."""
    for tag in confluences:
        if isinstance(tag, str) and tag.startswith("tp1_"):
            try:
                return float(tag[len("tp1_"):])
            except ValueError:
                pass
    if is_long:
        return entry + TP1_R * risk_price
    return entry - TP1_R * risk_price


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _max_dd_from_curve(equity: List[Tuple[int, float]]) -> float:
    peak = equity[0][1]
    mdd = 0.0
    for _, v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100.0
        if dd > mdd:
            mdd = dd
    return mdd


def _streaks(trades: List[Dict]) -> Tuple[int, int]:
    """Longest losing streak, longest winning streak."""
    longest_loss = longest_win = cur_loss = cur_win = 0
    for t in trades:
        if t["pnl_net_usd"] > 0:
            cur_win += 1; cur_loss = 0
            longest_win = max(longest_win, cur_win)
        elif t["pnl_net_usd"] < 0:
            cur_loss += 1; cur_win = 0
            longest_loss = max(longest_loss, cur_loss)
        else:
            cur_win = cur_loss = 0
    return longest_loss, longest_win


def print_report(state: BacktestState, ltf_bars: Dict[str, List[Bar]]) -> None:
    trades = state.closed_trades
    if not trades:
        print("\n[!] No trades generated.\n")
        print(f"  Filters: news={state.blocked_news} spread={state.blocked_spread} "
              f"daily_cap={state.blocked_daily_cap} daily_dd={state.blocked_daily_dd} "
              f"total_dd={state.blocked_total_dd} concurrent={state.blocked_concurrent}")
        return

    df = pd.DataFrame(trades)
    df["pnl"] = df["pnl_net_usd"]
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    flats = df[df["pnl"] == 0]

    wr = len(wins) / len(df) * 100.0 if len(df) else 0.0
    avg_win = wins["pnl"].mean() if len(wins) else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) else 0.0
    gross_win = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    net = state.balance - INITIAL_BALANCE
    ret_pct = net / INITIAL_BALANCE * 100
    mdd_pct = _max_dd_from_curve(state.equity_curve)

    # Time span & frequency.
    df["entry_date"] = pd.to_datetime(df["entry_time"]).dt.tz_convert(UTC).dt.date
    n_days = (df["entry_date"].max() - df["entry_date"].min()).days + 1
    trades_per_day = len(df) / max(1, n_days)
    trades_per_week = trades_per_day * 7
    trades_per_month = trades_per_day * 30

    avg_r = (df["pnl"] / (df.apply(
        lambda r: state.balance * (RISK_PCT / 100) if r["risk_price"] == 0
        else INITIAL_BALANCE * (RISK_PCT / 100), axis=1))).mean()

    longest_loss, longest_win = _streaks(trades)

    print("\n" + "=" * 72)
    print("  propX MULTI-SETUP — BACKTEST RESULTS")
    print("=" * 72)
    print(f"  Period           : {df['entry_date'].min()}  ->  {df['entry_date'].max()}  ({n_days} days)")
    print(f"  Trades           : {len(df):>6}   wins={len(wins)}  losses={len(losses)}  flat={len(flats)}")
    print(f"  Win rate         : {wr:>6.1f} %")
    print(f"  Profit factor    : {pf:>6.2f}")
    print(f"  Avg win / loss   : ${avg_win:>+7.2f} / ${avg_loss:>+7.2f}")
    print(f"  Avg R per trade  : {avg_r:>+6.2f}")
    print(f"  Initial balance  : ${INITIAL_BALANCE:>10,.2f}")
    print(f"  Final balance    : ${state.balance:>10,.2f}")
    print(f"  Net P&L          : ${net:>+10,.2f}   ({ret_pct:+.2f} %)")
    print(f"  Peak balance     : ${state.peak_balance:>10,.2f}")
    print(f"  Max DD (peak)    : {mdd_pct:>6.2f} %")
    print(f"  Longest win run  : {longest_win}")
    print(f"  Longest loss run : {longest_loss}")
    print(f"  Trades / day     : {trades_per_day:>6.2f}")
    print(f"  Trades / week    : {trades_per_week:>6.1f}")
    print(f"  Trades / month   : {trades_per_month:>6.1f}")
    print(f"  Filters blocked  : news={state.blocked_news}  spread={state.blocked_spread}  "
          f"day_cap={state.blocked_daily_cap}  daily_dd={state.blocked_daily_dd}  "
          f"total_dd={state.blocked_total_dd}  concurrent={state.blocked_concurrent}")

    # ── Per-setup breakdown
    print(f"\n  {'SETUP':<32} {'N':>5} {'WR%':>7} {'PF':>6} {'NetUSD':>10} {'AvgUSD':>10}")
    print(f"  {'-'*32} {'-'*5} {'-'*7} {'-'*6} {'-'*10} {'-'*10}")
    for pat, g in df.groupby("pattern"):
        gw = g[g["pnl"] > 0]
        gl = g[g["pnl"] < 0]
        gw_sum = gw["pnl"].sum()
        gl_sum = abs(gl["pnl"].sum())
        ppf = gw_sum / gl_sum if gl_sum > 0 else float("inf")
        wr_p = len(gw) / len(g) * 100
        print(f"  {pat:<32} {len(g):>5} {wr_p:>6.1f}% {ppf:>6.2f} "
              f"{g['pnl'].sum():>+10.2f} {g['pnl'].mean():>+10.2f}")

    # ── Per-pair breakdown — top 10 / bottom 10
    by_sym = df.groupby("symbol")["pnl"].agg(['count', 'sum', 'mean'])
    by_sym["wr"] = df.groupby("symbol").apply(lambda g: (g["pnl"] > 0).mean() * 100)
    by_sym = by_sym.sort_values("sum", ascending=False)
    print(f"\n  TOP 10 PAIRS BY P&L")
    print(f"  {'Symbol':<8} {'N':>5} {'WR%':>7} {'Net':>10} {'Avg':>10}")
    print(f"  {'-'*8} {'-'*5} {'-'*7} {'-'*10} {'-'*10}")
    for sym, row in by_sym.head(10).iterrows():
        print(f"  {sym:<8} {int(row['count']):>5} {row['wr']:>6.1f}% "
              f"{row['sum']:>+10.2f} {row['mean']:>+10.2f}")
    print(f"\n  BOTTOM 10 PAIRS BY P&L")
    print(f"  {'Symbol':<8} {'N':>5} {'WR%':>7} {'Net':>10} {'Avg':>10}")
    print(f"  {'-'*8} {'-'*5} {'-'*7} {'-'*10} {'-'*10}")
    for sym, row in by_sym.tail(10).iterrows():
        print(f"  {sym:<8} {int(row['count']):>5} {row['wr']:>6.1f}% "
              f"{row['sum']:>+10.2f} {row['mean']:>+10.2f}")

    # ── Monthly P&L
    df["month"] = pd.to_datetime(df["entry_time"]).dt.tz_convert(UTC).dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum()
    print(f"\n  MONTHLY P&L")
    print(f"  {'Month':<10} {'P&L':>10} {'Cum':>12}")
    cum = INITIAL_BALANCE
    red_months = 0
    for m, p in monthly.items():
        cum += p
        if p < 0:
            red_months += 1
        print(f"  {str(m):<10} {p:>+10.2f}  ${cum:>10,.2f}")
    print(f"  Red months: {red_months} / {len(monthly)}")

    # ── Prop-firm check (The5%ers Phase 1 typical: 8% target, 4% daily, 6% MDD)
    print(f"\n  PROP-FIRM CHECK")
    print(f"  Hit 8 % profit target?      : {'YES' if ret_pct >= 8 else 'NO'} ({ret_pct:+.2f}%)")
    print(f"  Stayed under 5 % daily DD?  : {'YES' if state.blocked_daily_dd == 0 else 'NO'}")
    print(f"  Stayed under 10 % total DD? : {'YES' if not state.total_dd_halted else 'NO'}")
    print(f"  Max trades/day cap hit?     : {'YES' if state.blocked_daily_cap > 0 else 'NO'}")
    print("=" * 72)


def write_outputs(state: BacktestState, out_csv: Path, out_png: Path) -> None:
    if not state.closed_trades:
        return
    df = pd.DataFrame(state.closed_trades)
    df.to_csv(out_csv, index=False)
    print(f"\n[SAVED] {out_csv}")

    try:
        _plot(state, out_png)
        print(f"[SAVED] {out_png}")
    except Exception as exc:  # noqa: BLE001
        print(f"[!] plot failed: {exc}")


def _plot(state: BacktestState, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    times = [datetime.fromtimestamp(t / 1000, tz=UTC) for t, _ in state.equity_curve]
    eq = np.array([v for _, v in state.equity_curve])

    df = pd.DataFrame(state.closed_trades)
    df["entry_dt"] = pd.to_datetime(df["entry_time"]).dt.tz_convert(UTC)
    df["month"] = df["entry_dt"].dt.to_period("M")
    monthly = df.groupby("month")["pnl_net_usd"].sum()
    by_sym = df.groupby("symbol")["pnl_net_usd"].sum().sort_values()

    wr = (df["pnl_net_usd"] > 0).mean() * 100
    gross_win = df.loc[df["pnl_net_usd"] > 0, "pnl_net_usd"].sum()
    gross_loss = abs(df.loc[df["pnl_net_usd"] < 0, "pnl_net_usd"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else 999

    fig, axes = plt.subplots(3, 1, figsize=(15, 13))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(
        f"propX Multi-Setup  —  28 pairs  —  0.5% risk  —  {len(df)} trades  "
        f"|  {wr:.1f}% WR  |  PF {pf:.2f}  |  ${df['pnl_net_usd'].sum():+,.0f} net",
        fontsize=12, fontweight="bold", color="#eee",
    )
    for ax in axes:
        ax.set_facecolor("#141414")
        ax.tick_params(colors="#aaa")
        for sp in ax.spines.values():
            sp.set_color("#333")
        ax.title.set_color("#eee")
        ax.yaxis.label.set_color("#aaa")

    # Equity curve
    ax = axes[0]
    ax.plot(times, eq, color="#22c55e", lw=1.3)
    ax.fill_between(times, eq, INITIAL_BALANCE,
                    where=eq >= INITIAL_BALANCE, alpha=0.15, color="#22c55e")
    ax.fill_between(times, eq, INITIAL_BALANCE,
                    where=eq < INITIAL_BALANCE, alpha=0.15, color="#ef4444")
    ax.axhline(INITIAL_BALANCE, color="#555", lw=0.8, ls="--")
    ax.set_title("Equity Curve", fontsize=11)
    ax.set_ylabel("Balance ($)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Monthly P&L
    ax = axes[1]
    mths = [str(m) for m in monthly.index]
    pnls = monthly.values
    clrs = ["#22c55e" if p >= 0 else "#ef4444" for p in pnls]
    ax.bar(mths, pnls, color=clrs, width=0.6)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_title("Monthly P&L", fontsize=11)
    ax.set_ylabel("P&L ($)")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Per-symbol P&L
    ax = axes[2]
    clrs = ["#22c55e" if v >= 0 else "#ef4444" for v in by_sym.values]
    ax.bar(by_sym.index, by_sym.values, color=clrs, width=0.6)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_title("P&L by Symbol", fontsize=11)
    ax.set_ylabel("Total P&L ($)")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=70, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="propX Multi-Setup backtest")
    p.add_argument("--pairs", default=None,
                   help="Comma-separated subset; default = all 28")
    p.add_argument("--days", type=int, default=None,
                   help="Lookback days (most recent N days only)")
    p.add_argument("--out-csv", default="multi_setup_trades.csv")
    p.add_argument("--out-png", default="multi_setup_results.png")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.pairs:
        pairs = tuple(p.strip().upper() for p in args.pairs.split(",") if p.strip())
    else:
        pairs = ALL_PAIRS

    print("\n" + "=" * 72)
    print(f"  Loading data | pairs={len(pairs)} | "
          f"lookback={args.days or 'full'} days")
    print("=" * 72)
    htf, ltf = load_all_pairs(pairs, lookback_days=args.days)
    if not ltf:
        print("[!] No data loaded.")
        return 1

    print(f"\n  Running backtest...")
    t0 = time.time()
    state = run_backtest(htf, ltf, quiet=args.quiet)
    elapsed = time.time() - t0
    print(f"  Backtest done in {elapsed:.1f}s")

    print_report(state, ltf)
    write_outputs(state, Path(args.out_csv), Path(args.out_png))
    return 0


if __name__ == "__main__":
    sys.exit(main())
