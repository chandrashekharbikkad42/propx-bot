"""ICT Silver Bullet — STANDALONE candidate backtest (30 pairs, 1H+15M, 2y).

Walks 15M bars chronologically across all 30 pairs, runs ONLY the
SilverBulletDetector with strict no-look-ahead, applies the spec §5
filters (spread, DD halts), simulates the spec §4 trade lifecycle
(TP1=1.5R partial 50% → BE → trail 0.3R → TP2=2.5R, 48h time stop,
Friday flatten), books realistic spread + commission + slippage, and
writes a full report + equity PNG + trade CSV.

Differences from `multi_setup_backtest.py`:
  - Only ONE detector (no confluence merge).
  - No rejection-candle pre-gate (SB doesn't require it).
  - No global "max 2/day" cap — we want to see TRUE signal rate. Instead
    a per-window cap of 1 trade per (pair, day, window) prevents
    intra-window stacking.
  - Per-window breakdown in the report (LO / AM / PM).
  - 30-pair universe (includes CADCHF + XAGUSD vs. multi_setup's 28).

Honesty notes:
  - Entries: simulated MARKET at signal-bar CLOSE + slippage = 0.5 ×
    MAX_SPREAD_PIPS. Real LIMIT fills at FVG.high/low may miss; this
    slightly OVER-trades the strategy (flagged in report).
  - SL fills: exact at SL price (optimistic on gap-down bars).
  - Same-bar SL+TP: SL wins (conservative).
  - News blackout: off (static calendar covers May–Jul 2026 only).
  - Pip value: per-bar quote→USD via cross rates on the same stream.

Run:
    venv\\Scripts\\python.exe scripts/silver_bullet_backtest.py
    venv\\Scripts\\python.exe scripts/silver_bullet_backtest.py --days 365
    venv\\Scripts\\python.exe scripts/silver_bullet_backtest.py --pairs EURUSD,XAUUSD
"""

from __future__ import annotations
import argparse
import io
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, io.UnsupportedOperation):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.silver_bullet_config import (  # noqa: E402
    BE_SHIFT_R, COMMISSION_PER_LOT_ROUNDTURN_USD, CONTRACT_SIZE,
    MAX_SPREAD_PIPS, MAX_TRADES_PER_WINDOW_PER_PAIR,
    MIN_RISK_PIPS, PAIRS as ALL_PAIRS,
    PARTIAL_FRACTION, PIP_SIZE, RISK_PCT,
    SLIPPAGE_MARKET_FRAC_OF_MAX_SPREAD,
    TIME_STOP_HOURS, TP1_R, TP2_R, TRAIL_STEP_R,
    pip_size_for,
)
from data.bar_aggregator import Bar, read_bars_parquet  # noqa: E402
from strategy.patterns.base import Direction, MarketContext  # noqa: E402
from strategy.patterns.silver_bullet import (  # noqa: E402
    SilverBulletDetector, window_id_for,
)


UTC = timezone.utc
INITIAL_BALANCE: float = 10_000.0
DAILY_DD_PCT_HARD: float = 5.0
TOTAL_DD_PCT_HARD: float = 10.0
MAX_CONCURRENT: int = 2
LOT_MIN: float = 0.01
LOT_STEP: float = 0.01
LOT_MAX: float = 50.0

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
# Pip-value (quote → USD)
# ─────────────────────────────────────────────────────────────────────────────

def pip_value_usd_per_lot(
    symbol: str, current_prices: Dict[str, float],
) -> float:
    pip = PIP_SIZE[symbol]
    contract = CONTRACT_SIZE[symbol]
    raw = pip * contract

    if symbol in ("XAUUSD", "XAGUSD"):
        return raw
    quote = symbol[3:6]
    if quote == "USD":
        return raw
    usd_quote_sym = f"USD{quote}"
    quote_usd_sym = f"{quote}USD"
    if usd_quote_sym in current_prices and current_prices[usd_quote_sym] > 0:
        return raw / current_prices[usd_quote_sym]
    if quote_usd_sym in current_prices and current_prices[quote_usd_sym] > 0:
        return raw * current_prices[quote_usd_sym]
    return 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Open-trade record + lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpenTrade:
    pattern: str
    symbol: str
    window: str
    direction: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    risk_price: float
    lot_full: float
    lot_remaining: float
    pip_value_usd: float
    entry_time_msc: int
    bar_index_at_entry: int
    tp1_hit: bool = False
    realized_pnl_usd: float = 0.0
    commission_usd: float = 0.0


def update_open_trade(
    trade: OpenTrade, bar: Bar, symbol_max_spread_pips: float,
) -> Optional[Dict]:
    long = trade.direction == "BUY"
    pip = pip_size_for(trade.symbol)
    half_spread_price = 0.5 * symbol_max_spread_pips * pip

    # 1. SL
    if long and bar.low <= trade.sl:
        pnl = (trade.sl - trade.entry) * trade.lot_remaining * (1 / pip) * trade.pip_value_usd
        trade.realized_pnl_usd += pnl
        return _close_trade(trade, bar, trade.sl, "SL" if not trade.tp1_hit else "TRAIL")
    if (not long) and bar.high >= trade.sl:
        pnl = (trade.entry - trade.sl) * trade.lot_remaining * (1 / pip) * trade.pip_value_usd
        trade.realized_pnl_usd += pnl
        return _close_trade(trade, bar, trade.sl, "SL" if not trade.tp1_hit else "TRAIL")

    # 2. TP1
    if not trade.tp1_hit:
        if (long and bar.high >= trade.tp1) or ((not long) and bar.low <= trade.tp1):
            lot_to_close = trade.lot_remaining * PARTIAL_FRACTION
            lot_after = trade.lot_remaining - lot_to_close
            if long:
                pnl = (trade.tp1 - trade.entry) * lot_to_close * (1 / pip) * trade.pip_value_usd
            else:
                pnl = (trade.entry - trade.tp1) * lot_to_close * (1 / pip) * trade.pip_value_usd
            trade.realized_pnl_usd += pnl
            trade.lot_remaining = lot_after
            trade.tp1_hit = True
            trade.sl = trade.entry + (half_spread_price if long else -half_spread_price)

    # 3. TP2 (only after TP1)
    if trade.tp1_hit:
        if (long and bar.high >= trade.tp2) or ((not long) and bar.low <= trade.tp2):
            if long:
                pnl = (trade.tp2 - trade.entry) * trade.lot_remaining * (1 / pip) * trade.pip_value_usd
            else:
                pnl = (trade.entry - trade.tp2) * trade.lot_remaining * (1 / pip) * trade.pip_value_usd
            trade.realized_pnl_usd += pnl
            return _close_trade(trade, bar, trade.tp2, "TP2")

    # 4. Trail after TP1
    if trade.tp1_hit:
        trail = TRAIL_STEP_R * trade.risk_price
        if long:
            new_sl = bar.close - trail
            if new_sl > trade.sl:
                trade.sl = new_sl
        else:
            new_sl = bar.close + trail
            if new_sl < trade.sl:
                trade.sl = new_sl

    return None


def _close_trade(trade: OpenTrade, bar: Bar, exit_price: float, reason: str) -> Dict:
    return {
        "symbol": trade.symbol, "pattern": trade.pattern, "window": trade.window,
        "direction": trade.direction,
        "entry_time": datetime.fromtimestamp(trade.entry_time_msc / 1000, tz=UTC),
        "exit_time": datetime.fromtimestamp(bar.time_msc / 1000, tz=UTC),
        "entry": trade.entry, "sl_initial": _initial_sl(trade),
        "tp1": trade.tp1, "tp2": trade.tp2,
        "exit_price": exit_price, "exit_reason": reason,
        "tp1_hit": trade.tp1_hit, "lot": trade.lot_full,
        "risk_price": trade.risk_price,
        "pnl_gross_usd": round(trade.realized_pnl_usd, 4),
        "commission_usd": round(trade.commission_usd, 4),
        "pnl_net_usd": round(trade.realized_pnl_usd - trade.commission_usd, 4),
    }


def _initial_sl(trade: OpenTrade) -> float:
    if trade.direction == "BUY":
        return trade.entry - trade.risk_price
    return trade.entry + trade.risk_price


def force_close_at_market(trade: OpenTrade, bar: Bar, reason: str) -> Dict:
    pip = pip_size_for(trade.symbol)
    if trade.direction == "BUY":
        pnl = (bar.close - trade.entry) * trade.lot_remaining * (1 / pip) * trade.pip_value_usd
    else:
        pnl = (trade.entry - bar.close) * trade.lot_remaining * (1 / pip) * trade.pip_value_usd
    trade.realized_pnl_usd += pnl
    return _close_trade(trade, bar, bar.close, reason)


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestState:
    balance: float = INITIAL_BALANCE
    peak_balance: float = INITIAL_BALANCE
    open_trades: List[OpenTrade] = field(default_factory=list)
    closed_trades: List[Dict] = field(default_factory=list)
    day_start_balance: Dict[str, float] = field(default_factory=dict)
    # Per-(pair, day, window) trade counter for the cap.
    window_trade_count: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    equity_curve: List[Tuple[int, float]] = field(default_factory=list)
    halted_until_day: Optional[str] = None
    total_dd_halted: bool = False
    current_prices: Dict[str, float] = field(default_factory=dict)
    blocked_spread: int = 0
    blocked_window_cap: int = 0
    blocked_daily_dd: int = 0
    blocked_total_dd: int = 0
    blocked_concurrent: int = 0
    signals_emitted: int = 0
    signals_in_window: Dict[str, int] = field(default_factory=lambda: {"LO": 0, "AM": 0, "PM": 0})


def _date_key(time_msc: int) -> str:
    return datetime.fromtimestamp(time_msc / 1000, tz=UTC).strftime("%Y-%m-%d")


def run_backtest(
    htf_bars: Dict[str, List[Bar]],
    ltf_bars: Dict[str, List[Bar]],
    quiet: bool = False,
) -> BacktestState:
    detector = SilverBulletDetector()
    state = BacktestState()

    events: List[Tuple[int, str, int]] = []
    for sym, bars in ltf_bars.items():
        for i, b in enumerate(bars):
            events.append((b.time_msc, sym, i))
    events.sort()
    if not events:
        return state
    state.equity_curve.append((events[0][0], state.balance))

    htf_cursor: Dict[str, int] = {sym: -1 for sym in htf_bars}
    t_wall_start = time.time()
    report_every = max(1, len(events) // 50)

    for ev_no, (t_msc, sym, k) in enumerate(events):
        ltf = ltf_bars[sym]
        htf = htf_bars[sym]
        cur_bar = ltf[k]

        # 1. Update price for pip-value calcs.
        state.current_prices[sym] = cur_bar.close

        # 2. Walk open trades on this pair against this bar.
        still_open: List[OpenTrade] = []
        for t in state.open_trades:
            if t.symbol != sym:
                still_open.append(t)
                continue
            elapsed_h = (t_msc - t.entry_time_msc) / (3600 * 1000)
            if elapsed_h >= TIME_STOP_HOURS:
                _book(state, force_close_at_market(t, cur_bar, "TIME_STOP"))
                continue
            dt = datetime.fromtimestamp(t_msc / 1000, tz=UTC)
            if dt.weekday() == 4 and dt.hour >= 23 and dt.minute >= 45:
                _book(state, force_close_at_market(t, cur_bar, "FRIDAY_FLATTEN"))
                continue
            max_spread = MAX_SPREAD_PIPS[sym]
            result = update_open_trade(t, cur_bar, max_spread)
            if result is not None:
                _book(state, result)
            else:
                still_open.append(t)
        state.open_trades = still_open

        # 3. Day key + DD halts.
        day_key = _date_key(t_msc)
        if day_key not in state.day_start_balance:
            state.day_start_balance[day_key] = state.balance
            state.halted_until_day = None

        if state.total_dd_halted:
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

        # 4. Concurrent cap.
        if len(state.open_trades) >= MAX_CONCURRENT:
            state.blocked_concurrent += 1
            _equity_log(state, t_msc)
            continue

        # 5. Pre-filter: only run detector during the 3 ET windows (skip
        # ~87.5% of bars: 3h windowed / 24h).
        wid = window_id_for(t_msc)
        if wid is None:
            _equity_log(state, t_msc)
            continue

        # 6. Window cap.
        wkey = (sym, day_key, wid)
        if state.window_trade_count.get(wkey, 0) >= MAX_TRADES_PER_WINDOW_PER_PAIR:
            state.blocked_window_cap += 1
            _equity_log(state, t_msc)
            continue

        # 7. HTF window (closed bars strictly before current LTF time).
        cursor = htf_cursor[sym]
        while cursor + 1 < len(htf) and htf[cursor + 1].time_msc < t_msc:
            cursor += 1
        htf_cursor[sym] = cursor
        if cursor < HTF_WINDOW // 4:
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

        # 8. Run detector.
        try:
            sig = detector.detect(ltf_window, ctx)
        except Exception as exc:  # noqa: BLE001
            if not quiet:
                print(f"  [detector-err] {sym}: {exc}")
            _equity_log(state, t_msc)
            continue
        if sig is None:
            _equity_log(state, t_msc)
            continue

        state.signals_emitted += 1
        state.signals_in_window[wid] = state.signals_in_window.get(wid, 0) + 1

        # 9. Spread guard.
        max_spread = MAX_SPREAD_PIPS[sym]
        spread_pips = cur_bar.spread_mean / 10.0
        if spread_pips > max_spread:
            state.blocked_spread += 1
            _equity_log(state, t_msc)
            continue

        # 10. Slippage on market entry.
        pip = PIP_SIZE[sym]
        slip_price = SLIPPAGE_MARKET_FRAC_OF_MAX_SPREAD * max_spread * pip
        is_long = (sig.direction == Direction.BUY)
        entry_px = sig.entry + (slip_price if is_long else -slip_price)
        if is_long and not (sig.sl < entry_px < sig.tp):
            _equity_log(state, t_msc); continue
        if (not is_long) and not (sig.tp < entry_px < sig.sl):
            _equity_log(state, t_msc); continue

        risk_price = abs(entry_px - sig.sl)
        risk_pips = risk_price / pip
        if risk_pips < MIN_RISK_PIPS:
            _equity_log(state, t_msc); continue

        tp1_price = _extract_tp1(sig.confluences_met, entry_px, risk_price, is_long)

        # 11. Sizing.
        risk_usd = state.balance * (RISK_PCT / 100.0)
        pv_usd = pip_value_usd_per_lot(sym, state.current_prices)
        if pv_usd <= 0:
            _equity_log(state, t_msc); continue
        lot_raw = risk_usd / (risk_pips * pv_usd)
        lot = math.floor(lot_raw / LOT_STEP) * LOT_STEP
        if lot < LOT_MIN:
            _equity_log(state, t_msc); continue
        lot = min(lot, LOT_MAX)

        commission = COMMISSION_PER_LOT_ROUNDTURN_USD * lot

        trade = OpenTrade(
            pattern=sig.pattern_name, symbol=sym, window=wid,
            direction="BUY" if is_long else "SELL",
            entry=entry_px, sl=sig.sl,
            tp1=tp1_price, tp2=sig.tp,
            risk_price=risk_price, lot_full=lot, lot_remaining=lot,
            pip_value_usd=pv_usd, entry_time_msc=t_msc,
            bar_index_at_entry=k, commission_usd=commission,
        )
        state.open_trades.append(trade)
        state.window_trade_count[wkey] = state.window_trade_count.get(wkey, 0) + 1
        _equity_log(state, t_msc)

        if not quiet and ev_no % report_every == 0:
            pct = (ev_no / len(events)) * 100
            elapsed = time.time() - t_wall_start
            print(f"  {pct:5.1f}%  | bal=${state.balance:>9.2f} "
                  f"| sigs={state.signals_emitted} "
                  f"| closed={len(state.closed_trades)} "
                  f"| open={len(state.open_trades)} "
                  f"| elapsed={elapsed:.0f}s")

    # Force-close remaining open trades at last bar of their pair.
    if state.open_trades:
        for t in list(state.open_trades):
            bars = ltf_bars[t.symbol]
            if not bars:
                continue
            _book(state, force_close_at_market(t, bars[-1], "END_OF_DATA"))
        state.open_trades.clear()
        _equity_log(state, events[-1][0])

    return state


def _book(state: BacktestState, closed: Dict) -> None:
    state.balance += closed["pnl_net_usd"]
    if state.balance > state.peak_balance:
        state.peak_balance = state.balance
    state.closed_trades.append(closed)


def _equity_log(state: BacktestState, t_msc: int) -> None:
    if state.equity_curve and state.equity_curve[-1][1] == state.balance:
        state.equity_curve[-1] = (t_msc, state.balance)
        return
    state.equity_curve.append((t_msc, state.balance))


def _extract_tp1(confluences: tuple, entry: float, risk_price: float, is_long: bool) -> float:
    for tag in confluences:
        if isinstance(tag, str) and tag.startswith("tp1_"):
            try:
                return float(tag[len("tp1_"):])
            except ValueError:
                pass
    return entry + (TP1_R * risk_price if is_long else -TP1_R * risk_price)


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
    print("\n" + "=" * 72)
    print("  ICT SILVER BULLET — STANDALONE BACKTEST RESULTS")
    print("=" * 72)
    print(f"  Signals emitted    : {state.signals_emitted}")
    print(f"  Signals per window : LO={state.signals_in_window.get('LO',0)}  "
          f"AM={state.signals_in_window.get('AM',0)}  "
          f"PM={state.signals_in_window.get('PM',0)}")
    print(f"  Filters blocked    : spread={state.blocked_spread}  "
          f"window_cap={state.blocked_window_cap}  daily_dd={state.blocked_daily_dd}  "
          f"total_dd={state.blocked_total_dd}  concurrent={state.blocked_concurrent}")

    trades = state.closed_trades
    if not trades:
        print("\n[!] No trades closed.\n")
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

    df["entry_date"] = pd.to_datetime(df["entry_time"]).dt.tz_convert(UTC).dt.date
    n_days = (df["entry_date"].max() - df["entry_date"].min()).days + 1
    n_weeks = n_days / 7
    trades_per_day = len(df) / max(1, n_days)
    trades_per_week = len(df) / max(1, n_weeks)

    longest_loss, longest_win = _streaks(trades)

    print(f"\n  Period             : {df['entry_date'].min()}  ->  {df['entry_date'].max()}  ({n_days} days)")
    print(f"  Trades             : {len(df):>6}   wins={len(wins)}  losses={len(losses)}  flat={len(flats)}")
    print(f"  Win rate           : {wr:>6.1f} %")
    print(f"  Profit factor      : {pf:>6.2f}")
    print(f"  Avg win / loss     : ${avg_win:>+7.2f} / ${avg_loss:>+7.2f}")
    print(f"  Initial balance    : ${INITIAL_BALANCE:>10,.2f}")
    print(f"  Final balance      : ${state.balance:>10,.2f}")
    print(f"  Net P&L            : ${net:>+10,.2f}   ({ret_pct:+.2f} %)")
    print(f"  Peak balance       : ${state.peak_balance:>10,.2f}")
    print(f"  Max DD (peak)      : {mdd_pct:>6.2f} %")
    print(f"  Longest win run    : {longest_win}")
    print(f"  Longest loss run   : {longest_loss}")
    print(f"  Trades / day       : {trades_per_day:>6.2f}")
    print(f"  Trades / week      : {trades_per_week:>6.1f}")

    # Per-window breakdown
    print(f"\n  {'WINDOW':<8} {'N':>5} {'WR%':>7} {'PF':>6} {'NetUSD':>10} {'AvgUSD':>10}")
    print(f"  {'-'*8} {'-'*5} {'-'*7} {'-'*6} {'-'*10} {'-'*10}")
    for wid in ("LO", "AM", "PM"):
        g = df[df["window"] == wid]
        if len(g) == 0:
            print(f"  {wid:<8} {0:>5} {'-':>7} {'-':>6} {'-':>10} {'-':>10}")
            continue
        gw = g[g["pnl"] > 0]
        gl = g[g["pnl"] < 0]
        gw_sum = gw["pnl"].sum()
        gl_sum = abs(gl["pnl"].sum())
        ppf = gw_sum / gl_sum if gl_sum > 0 else float("inf")
        print(f"  {wid:<8} {len(g):>5} {len(gw)/len(g)*100:>6.1f}% {ppf:>6.2f} "
              f"{g['pnl'].sum():>+10.2f} {g['pnl'].mean():>+10.2f}")

    # Direction breakdown
    print(f"\n  {'DIR':<8} {'N':>5} {'WR%':>7} {'PF':>6} {'NetUSD':>10}")
    print(f"  {'-'*8} {'-'*5} {'-'*7} {'-'*6} {'-'*10}")
    for d in ("BUY", "SELL"):
        g = df[df["direction"] == d]
        if len(g) == 0:
            continue
        gw_sum = g[g["pnl"] > 0]["pnl"].sum()
        gl_sum = abs(g[g["pnl"] < 0]["pnl"].sum())
        ppf = gw_sum / gl_sum if gl_sum > 0 else float("inf")
        print(f"  {d:<8} {len(g):>5} {(g['pnl']>0).mean()*100:>6.1f}% {ppf:>6.2f} {g['pnl'].sum():>+10.2f}")

    # Per-symbol breakdown
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

    # Profitable vs unprofitable symbols
    profitable = by_sym[by_sym["sum"] > 0]
    unprofitable = by_sym[by_sym["sum"] <= 0]
    print(f"\n  Symbols profitable : {len(profitable)} / {len(by_sym)} traded")
    print(f"  Symbols unprofitable: {len(unprofitable)} / {len(by_sym)} traded")

    # Monthly P&L
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

    # Verdict (spec §9 thresholds)
    print(f"\n  VERDICT (spec §9 thresholds)")
    keep = pf >= 1.3 and wr >= 45.0 and mdd_pct <= 8.0
    fix = (not keep) and (1.1 <= pf < 1.3)
    cut = pf < 1.1 or wr < 40.0
    if keep:
        verdict = "KEEP  (PF >= 1.3, WR >= 45 %, MDD <= 8 %)"
    elif fix:
        verdict = "FIX   (PF in 1.1–1.3 — tighten params)"
    elif cut:
        verdict = "CUT   (PF < 1.1 or WR < 40 %)"
    else:
        verdict = "BORDERLINE — operator decision"
    print(f"  PF={pf:.2f}  WR={wr:.1f}%  MDD={mdd_pct:.2f}%  → {verdict}")
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
    by_win = df.groupby("window")["pnl_net_usd"].sum()

    wr = (df["pnl_net_usd"] > 0).mean() * 100
    gross_win = df.loc[df["pnl_net_usd"] > 0, "pnl_net_usd"].sum()
    gross_loss = abs(df.loc[df["pnl_net_usd"] < 0, "pnl_net_usd"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else 999

    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(
        f"ICT Silver Bullet (standalone)  —  30 pairs  —  0.5% risk  —  "
        f"{len(df)} trades  |  {wr:.1f}% WR  |  PF {pf:.2f}  |  "
        f"${df['pnl_net_usd'].sum():+,.0f} net",
        fontsize=12, fontweight="bold", color="#eee",
    )
    for ax in axes:
        ax.set_facecolor("#141414")
        ax.tick_params(colors="#aaa")
        for sp in ax.spines.values():
            sp.set_color("#333")
        ax.title.set_color("#eee")
        ax.yaxis.label.set_color("#aaa")

    # Equity
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

    # Per-window P&L
    ax = axes[2]
    win_ids = list(by_win.index)
    win_vals = by_win.values
    clrs = ["#22c55e" if v >= 0 else "#ef4444" for v in win_vals]
    ax.bar(win_ids, win_vals, color=clrs, width=0.5)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_title("P&L by SB Window (LO=London Open, AM=NY AM, PM=NY PM)", fontsize=11)
    ax.set_ylabel("Total P&L ($)")

    # Per-symbol P&L
    ax = axes[3]
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
    p = argparse.ArgumentParser(description="ICT Silver Bullet standalone backtest")
    p.add_argument("--pairs", default=None, help="Comma-separated subset; default = all 30")
    p.add_argument("--days", type=int, default=None, help="Lookback days (most recent N)")
    p.add_argument("--out-csv", default="silver_bullet_trades.csv")
    p.add_argument("--out-png", default="silver_bullet_results.png")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.pairs:
        pairs = tuple(p.strip().upper() for p in args.pairs.split(",") if p.strip())
    else:
        pairs = ALL_PAIRS

    print("\n" + "=" * 72)
    print(f"  Loading data | pairs={len(pairs)} | lookback={args.days or 'full'} days")
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
