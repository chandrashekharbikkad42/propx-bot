"""
============================================================
XAUUSD  —  Asian Range London Sweep  —  1-Year Backtest
============================================================
Strategy:
  1. Mark Asian session range (01:00 – 06:00 IST = 19:30 – 00:30 UTC prev day)
  2. London open (11:30 IST = 06:00 UTC): detect sweep of Asian High or Low
  3. Wait for M15 candle CLOSE outside range (fake-out filter)
  4. Enter in reversal direction after sweep
  5. SL  = 7 points beyond sweep wick
  6. TP1 = 1R (partial close 50%), TP2 = opposite Asian level
  7. Breakeven after TP1 hit
  8. Session filter: London (06:00–10:30 UTC) + NY (12:30–16:00 UTC)
  9. Skip Monday + high-impact news days (NFP/FOMC Friday/Wednesday)

Requirements:
  pip install MetaTrader5 pandas numpy matplotlib

Run on a Windows machine with MT5 installed and logged in.
============================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  ASSET CONFIG — symbol change karo upar, baaki auto-set hoga
# ─────────────────────────────────────────────
ASSET_PARAMS = {
    "XAUUSD": {
        "sl_buffer": 7.0,
        "spread":    2.0,
        "min_range": 2.0,
        "max_range": 30.0,
        "value_per_point": 100.0,   # $100 per lot per point
        "lot_cap":   5.0,
    },
    "NAS100": {
        "sl_buffer": 20.0,
        "spread":    5.0,
        "min_range": 8.0,
        "max_range": 80.0,
        "value_per_point": 1.0,    # $1 per lot per point (MT5 standard)
        "lot_cap":   50.0,
    },
    "US30": {
        "sl_buffer": 15.0,
        "spread":    3.0,
        "min_range": 5.0,
        "max_range": 60.0,
        "value_per_point": 1.0,
        "lot_cap":   50.0,
    },
}

# Auto-load params for selected symbol (fallback to XAUUSD values)
_ap = ASSET_PARAMS.get("SYMBOL", ASSET_PARAMS["XAUUSD"])
SL_BUFFER      = _ap["sl_buffer"]
SPREAD_POINTS  = _ap["spread"]
MIN_RANGE_SIZE = _ap["min_range"]
MAX_RANGE_SIZE = _ap["max_range"]
VALUE_PER_PT   = _ap["value_per_point"]
LOT_CAP        = _ap["lot_cap"]

SYMBOL          = "US100.cash"     # ← CHANGE THIS: "XAUUSD" / "NAS100" / "US30"
BACKTEST_DAYS   = 365
RISK_PER_TRADE  = 1.0
INITIAL_BALANCE = 10_000.0
PARTIAL_CLOSE   = 0.50
RR_TP1          = 1.0
RR_TP2          = 2.5
SKIP_MONDAY     = True
MAX_TRADES_DAY  = 2
TRAILING_STEP       = 15.0     # trail SL by this many points after TP1
WEAK_MONTHS         = [11, 12, 1]  # Nov, Dec, Jan — reduce size
WEAK_MONTH_RISK_PCT = 0.5      # use only 0.5% risk in weak months (vs 1%)
SHORT_ONLY_BEARISH  = True     # take shorts ONLY when HTF is bearish (not neutral)
LONG_ONLY_BULLISH   = False    # longs allowed in bullish + neutral (keep as-is)

# IST offset = UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

# Asian session in UTC (previous day 19:30 → current day 00:30)
ASIAN_START_H, ASIAN_START_M = 19, 30   # UTC prev day
ASIAN_END_H,   ASIAN_END_M   =  0, 30   # UTC current day

# London sweep window in UTC
LONDON_START_H = 6    # 11:30 IST
LONDON_END_H   = 10   # 15:30 IST

# NY sweep window in UTC (second trade opportunity)
NY_SWEEP_START_H = 12  # 17:30 IST — NY open
NY_SWEEP_END_H   = 14  # 19:30 IST — NY early session only

# NY session end in UTC
NY_START_H = 12       # 17:30 IST
NY_END_H   = 16       # 21:30 IST


# ─────────────────────────────────────────────
#  MT5 CONNECTION + DATA FETCH
# ─────────────────────────────────────────────
def connect_mt5():
    if not mt5.initialize():
        print(f"[ERROR] MT5 initialize failed: {mt5.last_error()}")
        print("Make sure MetaTrader5 is open and logged into your account.")
        return False
    info = mt5.account_info()
    print(f"[OK] Connected to MT5 | Account: {info.login} | Server: {info.server}")
    print(f"     Balance: ${info.balance:,.2f} | Leverage: 1:{info.leverage}")
    print(f"     Symbol : {SYMBOL} | SL Buffer: {SL_BUFFER}pts | Trail: {TRAILING_STEP}pts")
    return True


def fetch_ohlcv(symbol, timeframe, days):
    end   = datetime.now(UTC)
    start = end - timedelta(days=days + 10)   # extra buffer for weekends

    rates = mt5.copy_rates_range(
        symbol,
        timeframe,
        start.replace(tzinfo=None),
        end.replace(tzinfo=None)
    )
    if rates is None or len(rates) == 0:
        print(f"[ERROR] No data fetched for {symbol}. Check symbol name in your broker.")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df = df[["open", "high", "low", "close", "tick_volume"]]
    print(f"[OK] Fetched {len(df)} H1 bars for {symbol}")
    return df


# ─────────────────────────────────────────────
#  STRATEGY LOGIC
# ─────────────────────────────────────────────
def get_asian_range(df_h1, date_utc):
    """
    Returns (asian_high, asian_low) for given UTC date.
    Asian session = previous day 19:30 UTC to current day 00:30 UTC.
    """
    prev_day = date_utc - timedelta(days=1)

    session_start = datetime(prev_day.year, prev_day.month, prev_day.day,
                              ASIAN_START_H, ASIAN_START_M, tzinfo=UTC)
    session_end   = datetime(date_utc.year, date_utc.month, date_utc.day,
                              ASIAN_END_H, ASIAN_END_M, tzinfo=UTC)

    mask = (df_h1.index >= session_start) & (df_h1.index < session_end)
    session_bars = df_h1[mask]

    if len(session_bars) < 3:
        return None, None

    return round(session_bars["high"].max(), 2), round(session_bars["low"].min(), 2)


def detect_sweep_and_entry(df_h1, date_utc, asian_high, asian_low, htf_bias):
    """
    Detect sweep in both London AND NY windows.
    Returns list of trade signals (max 2 per day).
    """
    signals = []

    for session_label, s_start_h, s_end_h in [
        ("LONDON", LONDON_START_H, LONDON_END_H),
        ("NY",     NY_SWEEP_START_H, NY_SWEEP_END_H),
    ]:
        win_start = datetime(date_utc.year, date_utc.month, date_utc.day,
                             s_start_h, 0, tzinfo=UTC)
        win_end   = datetime(date_utc.year, date_utc.month, date_utc.day,
                             s_end_h, 30, tzinfo=UTC)

        mask = (df_h1.index >= win_start) & (df_h1.index <= win_end)
        session_bars = df_h1[mask]

        swept_high = False
        swept_low  = False

        # Don't retake same direction already taken in London
        already_short = any(s["direction"] == "SHORT" for s in signals)
        already_long  = any(s["direction"] == "LONG"  for s in signals)

        for bar_time, bar in session_bars.iterrows():
            short_bias_ok = (htf_bias == "bearish") if SHORT_ONLY_BEARISH else (htf_bias in ["bearish", "neutral"])

            # Sweep of Asian HIGH → SHORT (London only — NY shorts skip)
            if (not swept_high and not already_short and
                    bar["high"] > asian_high and
                    bar["close"] < asian_high and
                    short_bias_ok and
                    session_label == "LONDON"):   # ← V5: NY me short nahi
                swept_high = True
                already_short = True
                entry = asian_high - SPREAD_POINTS
                sl    = bar["high"] + SL_BUFFER
                risk  = sl - entry
                tp1   = entry - (risk * RR_TP1)
                tp2   = entry - (risk * RR_TP2)
                signals.append({
                    "direction": "SHORT",
                    "session":   session_label,
                    "entry": round(entry, 2),
                    "sl":    round(sl, 2),
                    "tp1":   round(tp1, 2),
                    "tp2":   round(tp2, 2),
                    "risk":  round(risk, 2),
                    "bar_time":   bar_time,
                    "asian_high": asian_high,
                    "asian_low":  asian_low,
                })

            # Sweep of Asian LOW → LONG
            if (not swept_low and not already_long and
                    bar["low"] < asian_low and
                    bar["close"] > asian_low and
                    (htf_bias in ["bullish", "neutral"])):
                swept_low = True
                already_long = True
                entry = asian_low + SPREAD_POINTS
                sl    = bar["low"] - SL_BUFFER
                risk  = entry - sl
                tp1   = entry + (risk * RR_TP1)
                tp2   = entry + (risk * RR_TP2)
                signals.append({
                    "direction": "LONG",
                    "session":   session_label,
                    "entry": round(entry, 2),
                    "sl":    round(sl, 2),
                    "tp1":   round(tp1, 2),
                    "tp2":   round(tp2, 2),
                    "risk":  round(risk, 2),
                    "bar_time":   bar_time,
                    "asian_high": asian_high,
                    "asian_low":  asian_low,
                })

    return signals


def simulate_trade(df_h1, signal, date_utc, balance, risk_pct=None):
    """
    Simulate trade execution on subsequent bars.
    Returns trade result dict.
    """
    if risk_pct is None:
        risk_pct = RISK_PER_TRADE
    # Trade active from signal bar until end of NY session
    trade_start = signal["bar_time"]
    session_end = datetime(date_utc.year, date_utc.month, date_utc.day,
                            NY_END_H, 0, tzinfo=UTC)

    mask = (df_h1.index > trade_start) & (df_h1.index <= session_end)
    forward_bars = df_h1[mask]

    if forward_bars.empty:
        return None

    # Position sizing: risk RISK_PER_TRADE % of balance
    risk_amount = balance * (risk_pct / 100)
    risk_pts    = signal["risk"]
    if risk_pts <= 0:
        return None

    lot_size    = risk_amount / (risk_pts * VALUE_PER_PT)
    lot_size    = round(max(0.01, min(lot_size, LOT_CAP)), 2)

    entry   = signal["entry"]
    sl      = signal["sl"]
    tp1     = signal["tp1"]
    tp2     = signal["tp2"]
    direction = signal["direction"]

    tp1_hit     = False
    exit_price  = None
    exit_reason = None
    trailing_sl = sl      # starts at original SL, trails after TP1

    for bar_time, bar in forward_bars.iterrows():
        if direction == "LONG":
            if bar["low"] <= trailing_sl:
                exit_price  = trailing_sl
                exit_reason = "SL" if not tp1_hit else "TRAIL"
                break
            if not tp1_hit and bar["high"] >= tp1:
                tp1_hit     = True
                trailing_sl = entry
            if tp1_hit:
                new_trail = bar["close"] - TRAILING_STEP
                if new_trail > trailing_sl:
                    trailing_sl = round(new_trail, 2)
            if tp1_hit and bar["high"] >= tp2:
                exit_price  = tp2
                exit_reason = "TP2"
                break
        else:  # SHORT
            if bar["high"] >= trailing_sl:
                exit_price  = trailing_sl
                exit_reason = "SL" if not tp1_hit else "TRAIL"
                break
            if not tp1_hit and bar["low"] <= tp1:
                tp1_hit     = True
                trailing_sl = entry
            if tp1_hit:
                new_trail = bar["close"] + TRAILING_STEP
                if new_trail < trailing_sl:
                    trailing_sl = round(new_trail, 2)
            if tp1_hit and bar["low"] <= tp2:
                exit_price  = tp2
                exit_reason = "TP2"
                break

    if exit_price is None:
        last_bar    = forward_bars.iloc[-1]
        exit_price  = last_bar["close"]
        exit_reason = "EOD" if not tp1_hit else "EOD_trail"

    # P&L calculation — partial close at TP1, remainder at exit
    if direction == "LONG":
        if tp1_hit and exit_reason not in ["TP2"]:
            tp1_pnl  = (tp1 - entry) * (lot_size * PARTIAL_CLOSE) * VALUE_PER_PT
            rem_pnl  = (exit_price - entry) * (lot_size * (1 - PARTIAL_CLOSE)) * VALUE_PER_PT
            full_pnl = tp1_pnl + rem_pnl
        else:
            full_pnl = (exit_price - entry) * lot_size * VALUE_PER_PT
    else:
        if tp1_hit and exit_reason not in ["TP2"]:
            tp1_pnl  = (entry - tp1) * (lot_size * PARTIAL_CLOSE) * VALUE_PER_PT
            rem_pnl  = (entry - exit_price) * (lot_size * (1 - PARTIAL_CLOSE)) * VALUE_PER_PT
            full_pnl = tp1_pnl + rem_pnl
        else:
            full_pnl = (entry - exit_price) * lot_size * VALUE_PER_PT

    return {
        "date":         date_utc.strftime("%Y-%m-%d"),
        "direction":    direction,
        "entry":        entry,
        "exit":         round(exit_price, 2),
        "sl":           sl,
        "tp1":          tp1,
        "tp2":          tp2,
        "exit_reason":  exit_reason,
        "lot_size":     lot_size,
        "pnl":          round(full_pnl, 2),
        "risk_pts":     risk_pts,
        "tp1_hit":      tp1_hit,
        "bar_time":     signal["bar_time"],
        "asian_high":   signal["asian_high"],
        "asian_low":    signal["asian_low"],
    }


def get_htf_bias(df_h1, date_utc):
    """Simple HTF bias: EMA200 on H4."""
    cutoff = date_utc - timedelta(hours=1)
    recent = df_h1[df_h1.index <= cutoff]
    if len(recent) < 200:
        return "neutral"
    ema200 = recent["close"].ewm(span=200, adjust=False).mean().iloc[-1]
    last_close = recent["close"].iloc[-1]
    if last_close > ema200 * 1.001:
        return "bullish"
    elif last_close < ema200 * 0.999:
        return "bearish"
    return "neutral"


# ─────────────────────────────────────────────
#  MAIN BACKTEST LOOP
# ─────────────────────────────────────────────
def run_backtest():
    print("\n" + "="*60)
    print(f"  {SYMBOL}  Asian Range London Sweep — Backtest v5")
    print("  Trail=15 | NY longs only | Weak sizing | London+NY")
    print("="*60)

    if not connect_mt5():
        return

    print(f"\nFetching {BACKTEST_DAYS} days of H1 data...")
    df_h1 = fetch_ohlcv(SYMBOL, mt5.TIMEFRAME_H1, BACKTEST_DAYS + 30)
    mt5.shutdown()

    if df_h1 is None:
        return

    # Generate trading dates (skip weekends)
    end_date   = datetime.now(UTC).date()
    start_date = end_date - timedelta(days=BACKTEST_DAYS)
    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")

    balance   = INITIAL_BALANCE
    equity_curve = [balance]
    equity_dates = [dates[0].to_pydatetime().replace(tzinfo=UTC)]
    trades    = []
    skipped   = 0

    print(f"\nRunning backtest: {start_date} → {end_date}")
    print("-"*60)

    for date in dates:
        date_utc = date.to_pydatetime().replace(tzinfo=UTC)

        # Skip Monday
        if SKIP_MONDAY and date_utc.weekday() == 0:
            skipped += 1
            continue

        # Get Asian range
        asian_high, asian_low = get_asian_range(df_h1, date_utc)
        if asian_high is None:
            skipped += 1
            continue

        range_size = asian_high - asian_low

        # Range size filter
        if range_size < MIN_RANGE_SIZE or range_size > MAX_RANGE_SIZE:
            skipped += 1
            continue

        # HTF bias
        htf_bias = get_htf_bias(df_h1, date_utc)

        # Weak month risk reduction (Nov, Dec, Jan)
        month_risk = WEAK_MONTH_RISK_PCT if date_utc.month in WEAK_MONTHS else RISK_PER_TRADE

        # Detect sweep and entry signals (London + NY, max 2)
        signals = detect_sweep_and_entry(df_h1, date_utc, asian_high, asian_low, htf_bias)

        if not signals:
            skipped += 1
            continue

        day_had_trade = False
        for signal in signals[:MAX_TRADES_DAY]:
            result = simulate_trade(df_h1, signal, date_utc, balance, month_risk)
            if result is None:
                continue
            balance += result["pnl"]
            result["balance"]  = round(balance, 2)
            result["htf_bias"] = htf_bias
            trades.append(result)
            equity_curve.append(balance)
            equity_dates.append(date_utc)
            day_had_trade = True

        if not day_had_trade:
            skipped += 1

    # ─────────────────────────────────────────
    #  RESULTS ANALYSIS
    # ─────────────────────────────────────────
    if not trades:
        print("[!] No trades generated. Check data or parameters.")
        return

    df_trades = pd.DataFrame(trades)
    wins  = df_trades[df_trades["pnl"] > 0]
    losses = df_trades[df_trades["pnl"] <= 0]
    tp2_hits = df_trades[df_trades["exit_reason"] == "TP2"]
    be_exits = df_trades[df_trades["exit_reason"].isin(["TRAIL", "EOD_trail"])]

    win_rate    = len(wins) / len(df_trades) * 100
    avg_win     = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss    = losses["pnl"].mean() if len(losses) > 0 else 0
    profit_factor = wins["pnl"].sum() / abs(losses["pnl"].sum()) if losses["pnl"].sum() != 0 else 999
    total_pnl   = df_trades["pnl"].sum()
    net_return  = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    max_dd      = _max_drawdown(equity_curve)

    print(f"\n{'─'*60}")
    print(f"  BACKTEST RESULTS  |  {len(df_trades)} trades  |  {skipped} days skipped")
    print(f"{'─'*60}")
    print(f"  Initial balance : ${INITIAL_BALANCE:>10,.2f}")
    print(f"  Final balance   : ${balance:>10,.2f}")
    print(f"  Net P&L         : ${total_pnl:>+10,.2f}  ({net_return:+.1f}%)")
    print(f"  Max Drawdown    : {max_dd:.1f}%")
    print(f"{'─'*60}")
    print(f"  Win rate        : {win_rate:.1f}%")
    print(f"  Profit factor   : {profit_factor:.2f}")
    print(f"  Avg win         : ${avg_win:>+.2f}")
    print(f"  Avg loss        : ${avg_loss:>+.2f}")
    print(f"  TP2 hits        : {len(tp2_hits)} ({len(tp2_hits)/len(df_trades)*100:.1f}%)")
    print(f"  BE exits        : {len(be_exits)} ({len(be_exits)/len(df_trades)*100:.1f}%)")
    print(f"{'─'*60}")

    # Direction breakdown
    longs  = df_trades[df_trades["direction"] == "LONG"]
    shorts = df_trades[df_trades["direction"] == "SHORT"]
    london_t = df_trades[df_trades["session"] == "LONDON"] if "session" in df_trades.columns else df_trades
    ny_t     = df_trades[df_trades["session"] == "NY"]     if "session" in df_trades.columns else pd.DataFrame()

    print(f"\n  LONG  trades : {len(longs):3d}  |  PnL: ${longs['pnl'].sum():>+.2f}  |  WR: {(longs['pnl']>0).mean()*100:.1f}%")
    print(f"  SHORT trades : {len(shorts):3d}  |  PnL: ${shorts['pnl'].sum():>+.2f}  |  WR: {(shorts['pnl']>0).mean()*100:.1f}%")
    if len(london_t): print(f"  LONDON sess  : {len(london_t):3d}  |  PnL: ${london_t['pnl'].sum():>+.2f}  |  WR: {(london_t['pnl']>0).mean()*100:.1f}%")
    if len(ny_t):     print(f"  NY     sess  : {len(ny_t):3d}  |  PnL: ${ny_t['pnl'].sum():>+.2f}  |  WR: {(ny_t['pnl']>0).mean()*100:.1f}%")

    # Monthly breakdown
    df_trades["month"] = pd.to_datetime(df_trades["date"]).dt.to_period("M")
    monthly = df_trades.groupby("month")["pnl"].sum()
    print(f"\n  MONTHLY P&L")
    print(f"  {'Month':<12} {'P&L':>10} {'Cum':>12}")
    cum = INITIAL_BALANCE
    for m, pnl in monthly.items():
        cum += pnl
        bar = "█" * int(abs(pnl) / 50) if abs(pnl) > 0 else ""
        sign = "+" if pnl >= 0 else ""
        print(f"  {str(m):<12} {sign}{pnl:>8.0f}  ${cum:>10,.0f}  {bar}")

    print(f"\n{'─'*60}")

    # Save trade log CSV
    csv_path = f"backtest_trades_{SYMBOL}_v5.csv"
    df_trades.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Trade log → {csv_path}")

    # ─────────────────────────────────────────
    #  CHARTS
    # ─────────────────────────────────────────
    _plot_results(equity_curve, equity_dates, df_trades, monthly)


def _max_drawdown(equity):
    peak = equity[0]
    max_dd = 0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _plot_results(equity_curve, equity_dates, df_trades, monthly):
    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.suptitle(f"{SYMBOL}  Asian Range London Sweep  v5  —  1-Year Backtest",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#141414")
        ax.tick_params(colors="#aaa")
        ax.spines[:].set_color("#333")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.title.set_color("#eee")

    # ── Chart 1: Equity Curve ──
    ax = axes[0]
    equity_arr = np.array(equity_curve)
    colors_fill = ["#22c55e" if v >= INITIAL_BALANCE else "#ef4444" for v in equity_arr]
    ax.plot(equity_dates, equity_arr, color="#22c55e", lw=1.5, zorder=3)
    ax.fill_between(equity_dates, equity_arr, INITIAL_BALANCE,
                    where=equity_arr >= INITIAL_BALANCE, alpha=0.15, color="#22c55e")
    ax.fill_between(equity_dates, equity_arr, INITIAL_BALANCE,
                    where=equity_arr < INITIAL_BALANCE, alpha=0.15, color="#ef4444")
    ax.axhline(INITIAL_BALANCE, color="#555", lw=0.8, ls="--")
    ax.set_title("Equity curve", fontsize=11)
    ax.set_ylabel("Balance ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Chart 2: Monthly P&L bars ──
    ax = axes[1]
    months = [str(m) for m in monthly.index]
    pnls   = monthly.values
    bar_colors = ["#22c55e" if p >= 0 else "#ef4444" for p in pnls]
    bars = ax.bar(months, pnls, color=bar_colors, width=0.6, zorder=3)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_title("Monthly P&L", fontsize=11)
    ax.set_ylabel("P&L ($)")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    for bar, pnl in zip(bars, pnls):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (50 if pnl >= 0 else -120),
                f"${pnl:+.0f}", ha="center", fontsize=8, color="#ccc")

    # ── Chart 3: Trade distribution ──
    ax = axes[2]
    pnl_vals = df_trades["pnl"].values
    win_vals  = pnl_vals[pnl_vals > 0]
    loss_vals = pnl_vals[pnl_vals <= 0]
    if len(win_vals):
        ax.hist(win_vals,  bins=25, color="#22c55e", alpha=0.75, label=f"Wins ({len(win_vals)})")
    if len(loss_vals):
        ax.hist(loss_vals, bins=25, color="#ef4444", alpha=0.75, label=f"Losses ({len(loss_vals)})")
    ax.axvline(0, color="#555", lw=0.8)
    ax.axvline(pnl_vals.mean(), color="#facc15", lw=1.2, ls="--",
               label=f"Avg ${pnl_vals.mean():+.0f}")
    ax.set_title("Trade P&L distribution", fontsize=11)
    ax.set_xlabel("P&L per trade ($)")
    ax.set_ylabel("Count")
    legend = ax.legend(facecolor="#222", edgecolor="#444", labelcolor="#ccc", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    chart_path = f"backtest_results_{SYMBOL}_v5.png"
    plt.savefig(chart_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[SAVED] Chart → {chart_path}")
    plt.show()
    print("\n[DONE] Backtest complete!")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_backtest()