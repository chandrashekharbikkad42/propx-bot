"""
============================================================
MULTI-PAIR Asian Range London Sweep — 1-Year Backtest
============================================================
14 pairs from FTMO Demo — exact broker data from symbols_info.csv
Target: ~480 trades/year | Max 2 trades/day | 1% risk/trade

Pairs:
  Metals : XAUUSD, XAGUSD
  Majors : EURUSD, AUDUSD, GBPUSD, USDCAD, USDCHF, USDJPY
  Crosses: AUDCAD, AUDCHF, AUDJPY, AUDNZD, NZDJPY, EURJPY

Daily slot selection: best quality sweep wins
Max 2 trades/day, max 1 per direction (1 LONG + 1 SHORT max)
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

UTC = timezone.utc

# ─────────────────────────────────────────────
#  GLOBAL CONFIG
# ─────────────────────────────────────────────
BACKTEST_DAYS   = 365
INITIAL_BALANCE = 10_000.0
RISK_PER_TRADE  = 0.8       # reduced from 1.0 → DD fix
MAX_TRADES_DAY  = 2
PARTIAL_CLOSE   = 0.50
RR_TP1          = 1.0
RR_TP2          = 2.5
TRAILING_STEP   = 0.30
SKIP_MONDAY     = True
WEAK_MONTHS     = [11, 12, 1]
WEAK_RISK       = 0.3
MAX_DAILY_DD    = 3.0

# Sessions (UTC)
ASIAN_PREV_H = 19
ASIAN_END_H  =  0
LONDON_S, LONDON_E = 6, 10
NY_S,     NY_E     = 12, 15
SESS_CLOSE         = 16

# ─────────────────────────────────────────────
#  SYMBOL CONFIG
#  Built from exact symbols_info.csv data
#  point, contract, spread_pts all exact from broker
#  sl_pts: SL buffer in points beyond sweep wick
#  min_r/max_r: valid Asian range in points
#  quality: preference score for daily slot (higher = picked first)
# ─────────────────────────────────────────────
SYMBOLS = {
    # ── Metals ──────────────────────────────────────────────────────
    # XAUUSD: special risk_override=0.5% (large swings → DD control)
    "XAUUSD": {"spread": 45,  "point": 0.01,    "contract": 100.0,    "lot_max": 50.0,
               "sl_pts": 70,  "min_r": 100, "max_r": 3000, "quality": 10, "cat": "Metal",
               "jpy": False,  "risk_override": 0.5},
    # XAGUSD removed — 46.7% WR, only $117 P&L (slot waste)
    # USDJPY removed — 66.7% WR but only $42 P&L (JPY conversion eating gains)
    # ── Majors ──────────────────────────────────────────────────────
    "EURUSD": {"spread":  4,  "point": 0.00001, "contract": 100000.0, "lot_max": 50.0,
               "sl_pts": 80,  "min_r": 200, "max_r": 2000, "quality": 9,  "cat": "Major",
               "jpy": False,  "risk_override": None},
    "AUDUSD": {"spread":  3,  "point": 0.00001, "contract": 100000.0, "lot_max": 50.0,
               "sl_pts": 80,  "min_r": 150, "max_r": 1800, "quality": 9,  "cat": "Major",
               "jpy": False,  "risk_override": None},
    "GBPUSD": {"spread":  8,  "point": 0.00001, "contract": 100000.0, "lot_max": 50.0,
               "sl_pts": 100, "min_r": 200, "max_r": 2500, "quality": 8,  "cat": "Major",
               "jpy": False,  "risk_override": None},
    "USDCAD": {"spread":  5,  "point": 0.00001, "contract": 100000.0, "lot_max": 50.0,
               "sl_pts": 80,  "min_r": 150, "max_r": 2000, "quality": 7,  "cat": "Major",
               "jpy": False,  "risk_override": None},
    "USDCHF": {"spread":  6,  "point": 0.00001, "contract": 100000.0, "lot_max": 50.0,
               "sl_pts": 80,  "min_r": 150, "max_r": 2000, "quality": 7,  "cat": "Major",
               "jpy": False,  "risk_override": None},
    # ── Crosses ─────────────────────────────────────────────────────
    "AUDCHF": {"spread":  8,  "point": 0.00001, "contract": 100000.0, "lot_max": 50.0,
               "sl_pts": 80,  "min_r": 150, "max_r": 1800, "quality": 5,  "cat": "Cross",
               "jpy": False,  "risk_override": None},
    "NZDUSD": {"spread":  7,  "point": 0.00001, "contract": 100000.0, "lot_max": 50.0,
             "sl_pts": 80,  "min_r": 150, "max_r": 1800, "quality": 7,  "cat": "Major",
             "jpy": False,  "risk_override": None},
    "EURNZD": {"spread": 12, "point": 0.00001, "contract": 100000.0, "lot_max": 50.0,
             "sl_pts": 80,  "min_r": 150, "max_r": 1800, "quality": 8,  "cat": "Cross",
             "jpy": False,  "risk_override": None},
    "GBPCAD": {"spread": 12, "point": 0.00001, "contract": 100000.0, "lot_max": 50.0,
             "sl_pts": 80,  "min_r": 150, "max_r": 1800, "quality": 8,  "cat": "Cross",
             "jpy": False,  "risk_override": None},
    "GBPAUD": {"spread": 12, "point": 0.00001, "contract": 100000.0, "lot_max": 50.0,
             "sl_pts": 80,  "min_r": 150, "max_r": 1800, "quality": 8,  "cat": "Cross",
             "jpy": False,  "risk_override": None},
    "HK50.cash": {"spread": 50, "point": 0.01, "contract": 1.0, "lot_max": 50.0,
             "sl_pts": 2000, "min_r": 100, "max_r": 30000, "quality": 9, "cat": "Index",
             "jpy": False,  "risk_override": None},
    "GER40.cash": {"spread": 30, "point": 0.01, "contract": 1.0, "lot_max": 50.0,
             "sl_pts": 2000, "min_r": 100, "max_r": 30000, "quality": 8, "cat": "Index",
             "jpy": False,  "risk_override": None},
    "AUDNZD": {"spread": 12,  "point": 0.00001, "contract": 100000.0, "lot_max": 50.0,
               "sl_pts": 80,  "min_r": 150, "max_r": 1800, "quality": 4,  "cat": "Cross",
               "jpy": False,  "risk_override": None},
    # NZDJPY removed — 33.3% WR, −$6 P&L (slot waste)
    # EURJPY removed — 65.9% WR but only $27 P&L (JPY conversion issue)
}

ALL_SYMS = list(SYMBOLS.keys())


# ─────────────────────────────────────────────
#  MT5 HELPERS
# ─────────────────────────────────────────────
def connect_mt5():
    if not mt5.initialize():
        print(f"[ERROR] {mt5.last_error()}")
        return False
    a = mt5.account_info()
    print(f"[OK] {a.login} | {a.server} | ${a.balance:,.2f} | 1:{a.leverage}")
    return True


def fetch_h1(sym, days):
    end = datetime.now(UTC)
    st  = end - timedelta(days=days + 20)
    r   = mt5.copy_rates_range(sym, mt5.TIMEFRAME_H1,
                                st.replace(tzinfo=None), end.replace(tzinfo=None))
    if r is None or len(r) == 0:
        return None
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df[["open", "high", "low", "close"]]


# ─────────────────────────────────────────────
#  STRATEGY FUNCTIONS
# ─────────────────────────────────────────────
def get_asian_range(df, du):
    prev = du - timedelta(days=1)
    s = datetime(prev.year, prev.month, prev.day, ASIAN_PREV_H, 30, tzinfo=UTC)
    e = datetime(du.year,   du.month,   du.day,   ASIAN_END_H,  30, tzinfo=UTC)
    bars = df[(df.index >= s) & (df.index < e)]
    if len(bars) < 2:
        return None, None
    return float(bars["high"].max()), float(bars["low"].min())


def get_bias(df, du):
    rec = df[df.index <= du - timedelta(hours=1)]
    if len(rec) < 200:
        return "neutral"
    ema = rec["close"].ewm(span=200, adjust=False).mean().iloc[-1]
    cl  = rec["close"].iloc[-1]
    if cl > ema * 1.001:  return "bullish"
    if cl < ema * 0.999:  return "bearish"
    return "neutral"


def scan_signals(df, sym, du, ah, al, bias, cfg):
    """Scan London + NY for sweep signals. Return list."""
    sigs   = []
    pt     = cfg["point"]
    sl_pts = cfg["sl_pts"]
    sp     = cfg["spread"]
    min_r  = cfg["min_r"]
    max_r  = cfg["max_r"]
    q      = cfg["quality"]

    rng = round((ah - al) / pt)
    if rng < min_r or rng > max_r:
        return []

    for sess, sh, eh, allow_short in [
        ("LONDON", LONDON_S, LONDON_E, True),
        ("NY",     NY_S,     NY_E,     False),
    ]:
        ws   = datetime(du.year, du.month, du.day, sh, 0,  tzinfo=UTC)
        we   = datetime(du.year, du.month, du.day, eh, 30, tzinfo=UTC)
        bars = df[(df.index >= ws) & (df.index <= we)]
        if bars.empty:
            continue

        hit_h = hit_l = False
        for bt, bar in bars.iterrows():
            # SHORT — sweep Asian High (London only, bearish HTF)
            if (allow_short and not hit_h and bias == "bearish" and
                    bar["high"] > ah and bar["close"] < ah):
                hit_h = True
                entry = ah - sp * pt
                sl    = bar["high"] + sl_pts * pt
                risk  = abs(sl - entry)
                if risk < pt * 3:
                    continue
                sigs.append({"sym": sym, "dir": "SHORT", "sess": sess,
                              "entry": entry, "sl": sl,
                              "tp1": entry - risk * RR_TP1,
                              "tp2": entry - risk * RR_TP2,
                              "risk": risk, "bt": bt, "q": q})

            # LONG — sweep Asian Low (both sessions, bullish/neutral HTF)
            if (not hit_l and bias in ["bullish", "neutral"] and
                    bar["low"] < al and bar["close"] > al):
                hit_l = True
                entry = al + sp * pt
                sl    = bar["low"] - sl_pts * pt
                risk  = abs(entry - sl)
                if risk < pt * 3:
                    continue
                sigs.append({"sym": sym, "dir": "LONG", "sess": sess,
                              "entry": entry, "sl": sl,
                              "tp1": entry + risk * RR_TP1,
                              "tp2": entry + risk * RR_TP2,
                              "risk": risk, "bt": bt, "q": q})
    return sigs


def simulate(df, sig, du, balance, rp, cfg):
    fwd_end = datetime(du.year, du.month, du.day, SESS_CLOSE, 0, tzinfo=UTC)
    fwd     = df[(df.index > sig["bt"]) & (df.index <= fwd_end)]
    if fwd.empty:
        return None

    pt   = cfg["point"]
    ct   = cfg["contract"]
    lmax = cfg["lot_max"]
    is_jpy = cfg["jpy"]

    risk_amt = balance * (rp / 100)
    # Per-symbol risk override (e.g. XAUUSD uses 0.5% instead of global)
    override = cfg.get("risk_override")
    if override is not None:
        risk_amt = balance * (override / 100)
    risk_p   = sig["risk"]
    if risk_p <= 0:
        return None

    # Lot sizing: risk_amt = lot * (risk_p / pt) * (ct * pt)
    # Simplifies to: lot = risk_amt / (risk_pts_count * ct * pt)
    # where risk_pts_count = risk_p / pt
    risk_pts_count = risk_p / pt
    vpl  = ct * pt          # value per lot per point in profit currency
    lot  = round(max(0.01, min(risk_amt / (risk_pts_count * vpl), lmax)), 2)

    e, sl, tp1, tp2 = sig["entry"], sig["sl"], sig["tp1"], sig["tp2"]
    dr     = sig["dir"]
    trail  = sl
    tp1hit = False
    ep, er = None, None

    for _, bar in fwd.iterrows():
        if dr == "LONG":
            if bar["low"] <= trail:
                ep = trail; er = "SL" if not tp1hit else "TRAIL"; break
            if not tp1hit and bar["high"] >= tp1:
                tp1hit = True; trail = e
            if tp1hit:
                nt = bar["close"] - TRAILING_STEP * risk_p
                if nt > trail: trail = nt
            if tp1hit and bar["high"] >= tp2:
                ep = tp2; er = "TP2"; break
        else:
            if bar["high"] >= trail:
                ep = trail; er = "SL" if not tp1hit else "TRAIL"; break
            if not tp1hit and bar["low"] <= tp1:
                tp1hit = True; trail = e
            if tp1hit:
                nt = bar["close"] + TRAILING_STEP * risk_p
                if nt < trail: trail = nt
            if tp1hit and bar["low"] <= tp2:
                ep = tp2; er = "TP2"; break

    if ep is None:
        ep = fwd.iloc[-1]["close"]
        er = "EOD" if not tp1hit else "EOD_trail"

    diff = (ep - e) if dr == "LONG" else (e - ep)

    if tp1hit and er != "TP2":
        d1 = (tp1 - e) if dr == "LONG" else (e - tp1)
        pnl = (d1 * PARTIAL_CLOSE + diff * (1 - PARTIAL_CLOSE)) * lot * ct
    else:
        pnl = diff * lot * ct

    # JPY pairs: profit is in JPY → convert to USD (~150)
    if is_jpy:
        pnl /= 150.0

    return {"sym": sig["sym"], "date": du.strftime("%Y-%m-%d"),
            "dir": dr, "sess": sig["sess"], "cat": cfg["cat"],
            "entry": round(e, 5), "exit": round(ep, 5),
            "er": er, "lot": lot, "pnl": round(pnl, 2),
            "tp1hit": tp1hit, "q": sig["q"]}


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    print("\n" + "="*65)
    print("  MULTI-PAIR Asian Range London Sweep — Backtest v5")
    print(f"  8 pairs | 0.8% risk | XAUUSD=0.5% | DD circuit {MAX_DAILY_DD}%")
    print("="*65)

    if not connect_mt5():
        return

    print(f"\nFetching H1 data for {len(ALL_SYMS)} symbols...")
    data = {}
    for sym in ALL_SYMS:
        df = fetch_h1(sym, BACKTEST_DAYS + 15)
        if df is not None:
            data[sym] = df
            print(f"  ✓ {sym:<10} {len(df)} bars")
        else:
            print(f"  ✗ {sym:<10} not available")
    mt5.shutdown()

    loaded = list(data.keys())
    print(f"\n  {len(loaded)}/{len(ALL_SYMS)} symbols loaded")

    end_dt   = datetime.now(UTC).date()
    start_dt = end_dt - timedelta(days=BACKTEST_DAYS)
    dates    = pd.bdate_range(start=start_dt, end=end_dt)

    balance  = INITIAL_BALANCE
    eq_curve = [balance]
    eq_dates = [dates[0].to_pydatetime().replace(tzinfo=UTC)]
    trades   = []
    skipped  = 0

    print(f"\nRunning: {start_dt} → {end_dt}")
    print("-"*65)

    for date in dates:
        du = date.to_pydatetime().replace(tzinfo=UTC)

        if SKIP_MONDAY and du.weekday() == 0:
            skipped += 1
            continue

        rp = WEAK_RISK if du.month in WEAK_MONTHS else RISK_PER_TRADE

        # Collect all signals from all symbols
        all_sigs = []
        for sym in loaded:
            ah, al = get_asian_range(data[sym], du)
            if ah is None:
                continue
            bias = get_bias(data[sym], du)
            sigs = scan_signals(data[sym], sym, du, ah, al, bias, SYMBOLS[sym])
            all_sigs.extend(sigs)

        if not all_sigs:
            skipped += 1
            continue

        # Rank by quality → max 2 trades, max 1 per direction
        all_sigs.sort(key=lambda x: x["q"], reverse=True)
        picked    = []
        dirs_used = []
        for sig in all_sigs:
            if len(picked) >= MAX_TRADES_DAY:
                break
            if sig["dir"] in dirs_used:
                continue
            picked.append(sig)
            dirs_used.append(sig["dir"])

        day_start_balance = balance   # track daily DD
        day_ok = False
        for sig in picked:
            # Circuit breaker — stop if daily loss exceeds MAX_DAILY_DD%
            daily_loss_pct = (day_start_balance - balance) / day_start_balance * 100
            if daily_loss_pct >= MAX_DAILY_DD:
                break

            res = simulate(data[sig["sym"]], sig, du, balance, rp, SYMBOLS[sig["sym"]])
            if res is None:
                continue
            balance += res["pnl"]
            res["balance"] = round(balance, 2)
            trades.append(res)
            day_ok = True

        if day_ok:
            eq_curve.append(balance)
            eq_dates.append(du)
        else:
            skipped += 1

    # ── Results ──────────────────────────────
    if not trades:
        print("[!] No trades generated.")
        return

    df_t   = pd.DataFrame(trades)
    wins   = df_t[df_t["pnl"] > 0]
    losses = df_t[df_t["pnl"] <= 0]
    wr     = len(wins) / len(df_t) * 100
    pf     = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) else 999
    net    = balance - INITIAL_BALANCE
    ret    = net / INITIAL_BALANCE * 100
    mdd    = _max_dd(eq_curve)

    print(f"\n{'─'*65}")
    print(f"  RESULTS  |  {len(df_t)} trades  |  {skipped} days skipped")
    print(f"{'─'*65}")
    print(f"  Initial  : ${INITIAL_BALANCE:>10,.2f}")
    print(f"  Final    : ${balance:>10,.2f}")
    print(f"  Net P&L  : ${net:>+10,.2f}  ({ret:+.1f}%)")
    print(f"  Max DD   : {mdd:.1f}%")
    print(f"  Win rate : {wr:.1f}%")
    print(f"  Prof fac : {pf:.2f}")
    print(f"  Avg win  : ${wins['pnl'].mean():>+.2f}" if len(wins) else "  Avg win  : N/A")
    print(f"  Avg loss : ${losses['pnl'].mean():>+.2f}" if len(losses) else "  Avg loss : N/A")
    tp2c = (df_t["er"] == "TP2").sum()
    print(f"  TP2 hits : {tp2c} ({tp2c/len(df_t)*100:.1f}%)")
    print(f"{'─'*65}")

    # Per-symbol
    print(f"\n  {'Symbol':<10} {'N':>4} {'PnL':>10} {'WR%':>7} {'Cat'}")
    print(f"  {'─'*42}")
    for sym, g in df_t.groupby("sym"):
        print(f"  {sym:<10} {len(g):>4} {g['pnl'].sum():>+10.2f} {(g['pnl']>0).mean()*100:>6.1f}%  {SYMBOLS[sym]['cat']}")

    # Session
    print(f"\n  {'Session':<10} {'N':>4} {'PnL':>10} {'WR%':>7}")
    print(f"  {'─'*34}")
    for sess, g in df_t.groupby("sess"):
        print(f"  {sess:<10} {len(g):>4} {g['pnl'].sum():>+10.2f} {(g['pnl']>0).mean()*100:>6.1f}%")

    # Monthly
    df_t["month"] = pd.to_datetime(df_t["date"]).dt.to_period("M")
    monthly = df_t.groupby("month")["pnl"].sum()
    print(f"\n  MONTHLY P&L")
    cum = INITIAL_BALANCE
    for m, p in monthly.items():
        cum += p
        bar = "█" * max(0, int(abs(p) / 100))
        print(f"  {str(m)}  {'+'if p>=0 else ''}{p:>7.0f}  ${cum:>10,.0f}  {bar}")

    df_t.to_csv("multi_pair_trades_v5.csv", index=False)
    print(f"\n[SAVED] multi_pair_trades_v5.csv")

    _plot(eq_curve, eq_dates, df_t, monthly)


def _max_dd(eq):
    peak, mdd = eq[0], 0
    for v in eq:
        if v > peak: peak = v
        dd = (peak - v) / peak * 100
        if dd > mdd: mdd = dd
    return mdd


def _plot(eq_curve, eq_dates, df_t, monthly):
    fig, axes = plt.subplots(3, 1, figsize=(15, 13))
    wr  = (df_t["pnl"] > 0).mean() * 100
    pf  = df_t[df_t["pnl"]>0]["pnl"].sum() / abs(df_t[df_t["pnl"]<=0]["pnl"].sum())
    fig.suptitle(
        f"Multi-Pair Asian Range Sweep v5  —  8 Pairs  —  1-Year Backtest\n"
        f"{len(df_t)} trades  |  {wr:.1f}% WR  |  PF {pf:.2f}  |  "
        f"${df_t['pnl'].sum():+,.0f} net",
        fontsize=12, fontweight="bold")
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#141414")
        ax.tick_params(colors="#aaa")
        ax.spines[:].set_color("#333")
        ax.title.set_color("#eee")
        ax.yaxis.label.set_color("#aaa")

    # 1. Equity curve
    ax  = axes[0]
    arr = np.array(eq_curve)
    ax.plot(eq_dates, arr, color="#22c55e", lw=1.5)
    ax.fill_between(eq_dates, arr, INITIAL_BALANCE,
                    where=arr >= INITIAL_BALANCE, alpha=0.15, color="#22c55e")
    ax.fill_between(eq_dates, arr, INITIAL_BALANCE,
                    where=arr < INITIAL_BALANCE,  alpha=0.15, color="#ef4444")
    ax.axhline(INITIAL_BALANCE, color="#555", lw=0.8, ls="--")
    ax.set_title("Equity Curve", fontsize=11)
    ax.set_ylabel("Balance ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # 2. Monthly P&L
    ax  = axes[1]
    mths  = [str(m) for m in monthly.index]
    pnls  = monthly.values
    clrs  = ["#22c55e" if p >= 0 else "#ef4444" for p in pnls]
    bars  = ax.bar(mths, pnls, color=clrs, width=0.6)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_title("Monthly P&L", fontsize=11)
    ax.set_ylabel("P&L ($)")
    for b, p in zip(bars, pnls):
        ax.text(b.get_x() + b.get_width()/2,
                b.get_height() + (20 if p >= 0 else -60),
                f"${p:+.0f}", ha="center", fontsize=8, color="#ccc")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # 3. Per-symbol P&L
    ax      = axes[2]
    sym_pnl = df_t.groupby("sym")["pnl"].sum().sort_values(ascending=False)
    sc      = ["#22c55e" if v >= 0 else "#ef4444" for v in sym_pnl.values]
    ax.bar(sym_pnl.index, sym_pnl.values, color=sc, width=0.6)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_title("P&L by Symbol", fontsize=11)
    ax.set_ylabel("Total P&L ($)")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig("multi_pair_results_v5.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("[SAVED] multi_pair_results_v5.png")
    plt.show()
    print("\n[DONE] Backtest complete!")


if __name__ == "__main__":
    run()

