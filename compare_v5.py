"""
Side-by-side V5 backtest: BEFORE (14-pair) vs AFTER (edit applied).
Edit = remove NZDUSD, remove GER40.cash, GBPAUD quality -2.
Fetches H1 ONCE (union of all symbols) so both configs run on identical bars.
Replicates multi_pair_backtest.run()'s slot loop exactly. Does not modify SYMBOLS.
"""
import copy
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import multi_pair_backtest as m

UTC = timezone.utc


def build_configs():
    before = copy.deepcopy(m.SYMBOLS)            # current module state = 14-pair
    after = copy.deepcopy(before)
    for sym in ("NZDUSD", "GER40.cash"):
        after.pop(sym, None)
    if "GBPAUD" in after:
        after["GBPAUD"]["quality"] -= 2
    return before, after


def fetch_all(symbols):
    data = {}
    for sym in symbols:
        df = m.fetch_h1(sym, m.BACKTEST_DAYS + 15)
        if df is not None and len(df):
            data[sym] = df
    return data


def run_config(symbols, data, dates):
    """Faithful replica of m.run()'s daily slot loop."""
    loaded = [s for s in symbols if s in data]
    balance = m.INITIAL_BALANCE
    eq_curve = [balance]
    trades = []
    for date in dates:
        du = date.to_pydatetime().replace(tzinfo=UTC)
        if m.SKIP_MONDAY and du.weekday() == 0:
            continue
        rp = m.WEAK_RISK if du.month in m.WEAK_MONTHS else m.RISK_PER_TRADE
        all_sigs = []
        for sym in loaded:
            ah, al = m.get_asian_range(data[sym], du)
            if ah is None:
                continue
            bias = m.get_bias(data[sym], du)
            all_sigs.extend(m.scan_signals(data[sym], sym, du, ah, al, bias, symbols[sym]))
        if not all_sigs:
            continue
        all_sigs.sort(key=lambda x: x["q"], reverse=True)
        picked, dirs_used = [], []
        for sig in all_sigs:
            if len(picked) >= m.MAX_TRADES_DAY:
                break
            if sig["dir"] in dirs_used:
                continue
            picked.append(sig)
            dirs_used.append(sig["dir"])
        day_start = balance
        day_ok = False
        for sig in picked:
            if (day_start - balance) / day_start * 100 >= m.MAX_DAILY_DD:
                break
            res = m.simulate(data[sig["sym"]], sig, du, balance, rp, symbols[sig["sym"]])
            if res is None:
                continue
            balance += res["pnl"]
            res["balance"] = round(balance, 2)
            res["date"] = du
            trades.append(res)
            day_ok = True
        if day_ok:
            eq_curve.append(balance)
    return pd.DataFrame(trades), eq_curve, balance


def pair_maxdd(pnls, start=10000.0):
    eq = start + np.cumsum(pnls)
    peak = np.maximum.accumulate(eq)
    return ((eq - peak) / peak).min() * 100 if len(eq) else 0.0


def per_pair(df_t):
    rows = []
    df_t = df_t.copy()
    df_t["mo"] = pd.to_datetime(df_t["date"]).dt.to_period("M")
    for sym, g in df_t.groupby("sym"):
        p = g["pnl"].values
        wins, losses = p[p > 0], p[p <= 0]
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
        mo = g.groupby("mo")["pnl"].sum()
        rows.append(dict(sym=sym, n=len(p), wr=len(wins) / len(p) * 100, pf=pf,
                         pnl=p.sum(), dd=pair_maxdd(p),
                         posm=int((mo > 0).sum()), totm=len(mo)))
    return sorted(rows, key=lambda r: -r["pnl"])


def portfolio(df_t, eq_curve, balance):
    p = df_t["pnl"].values
    wins, losses = p[p > 0], p[p <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    net = balance - m.INITIAL_BALANCE
    return dict(trades=len(p), pnl=net, ret=net / m.INITIAL_BALANCE * 100,
                wr=len(wins) / len(p) * 100, pf=pf, dd=m._max_dd(eq_curve))


def fmt_pf(v):
    return "inf" if v == float("inf") else f"{v:.2f}"


def print_table(label, rows):
    print(f"\n=== {label} — per pair ===")
    print(f"{'sym':11}{'n':>4}{'WR%':>7}{'PF':>7}{'P&L$':>9}{'maxDD%':>8}{'+mo/mo':>9}")
    for r in rows:
        print(f"{r['sym']:11}{r['n']:>4}{r['wr']:>7.1f}{fmt_pf(r['pf']):>7}"
              f"{r['pnl']:>9.0f}{r['dd']:>8.1f}{r['posm']:>5}/{r['totm']:<3}")


def main():
    if not m.connect_mt5():
        return
    before_cfg, after_cfg = build_configs()
    union = list(before_cfg.keys())
    print(f"Fetching {len(union)} symbols once (shared data)...")
    data = fetch_all(union)
    m.mt5.shutdown()
    print(f"  {len(data)}/{len(union)} loaded")

    end_dt = datetime.now(UTC).date()
    start_dt = end_dt - timedelta(days=m.BACKTEST_DAYS)
    dates = pd.bdate_range(start=start_dt, end=end_dt)

    bt, beq, bbal = run_config(before_cfg, data, dates)
    at, aeq, abal = run_config(after_cfg, data, dates)

    print_table("BEFORE (14-pair)", per_pair(bt))
    print_table("AFTER (12-pair: -NZDUSD -GER40, GBPAUD q-2)", per_pair(at))

    bp, ap = portfolio(bt, beq, bbal), portfolio(at, aeq, abal)
    print("\n=== PORTFOLIO — side by side ===")
    print(f"{'metric':12}{'BEFORE(14)':>14}{'AFTER(12)':>14}{'delta':>12}")
    print(f"{'trades':12}{bp['trades']:>14}{ap['trades']:>14}{ap['trades']-bp['trades']:>+12}")
    print(f"{'P&L %':12}{bp['ret']:>13.1f}%{ap['ret']:>13.1f}%{ap['ret']-bp['ret']:>+11.1f}%")
    print(f"{'P&L $':12}{bp['pnl']:>14.0f}{ap['pnl']:>14.0f}{ap['pnl']-bp['pnl']:>+12.0f}")
    print(f"{'PF':12}{fmt_pf(bp['pf']):>14}{fmt_pf(ap['pf']):>14}{ap['pf']-bp['pf']:>+12.2f}")
    print(f"{'WR %':12}{bp['wr']:>13.1f}%{ap['wr']:>13.1f}%{ap['wr']-bp['wr']:>+11.1f}%")
    print(f"{'maxDD %':12}{bp['dd']:>13.1f}%{ap['dd']:>13.1f}%{ap['dd']-bp['dd']:>+11.1f}%")


if __name__ == "__main__":
    main()
