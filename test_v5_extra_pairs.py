"""
Test V5 Asian Range Sweep edge on EURGBP and NZDCAD.
Reuses V5's own functions from multi_pair_backtest.py — does NOT modify it.
"""
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import multi_pair_backtest as m

UTC = timezone.utc

TEST_CONFIG = {
    "EURGBP": {"spread":  4, "point": 0.00001, "contract": 100000.0, "lot_max": 50.0,
               "sl_pts": 80, "min_r": 150, "max_r": 1800, "quality": 6, "cat": "Cross",
               "jpy": False, "risk_override": None},
    "NZDCAD": {"spread": 10, "point": 0.00001, "contract": 100000.0, "lot_max": 50.0,
               "sl_pts": 80, "min_r": 150, "max_r": 1800, "quality": 5, "cat": "Cross",
               "jpy": False, "risk_override": None},
}

BACKTEST_DAYS = getattr(m, "BACKTEST_DAYS", 365)
INITIAL_BALANCE = getattr(m, "INITIAL_BALANCE", 10_000.0)
RISK_PCT = getattr(m, "RISK_PCT", 0.8)


def run_pair(sym, cfg):
    df = m.fetch_h1(sym, BACKTEST_DAYS + 15)
    if df is None or len(df) == 0:
        return None
    end_dt = datetime.now(UTC)
    start_dt = end_dt - timedelta(days=BACKTEST_DAYS)
    dates = pd.bdate_range(start=start_dt, end=end_dt)

    balance = INITIAL_BALANCE
    trades = []
    for d in dates:
        du = d.to_pydatetime().replace(tzinfo=UTC)
        try:
            ah, al = m.get_asian_range(df, du)
        except Exception:
            continue
        if ah is None:
            continue
        try:
            bias = m.get_bias(df, du)
            sigs = m.scan_signals(df, sym, du, ah, al, bias, cfg)
        except Exception:
            continue
        if not sigs:
            continue
        sigs.sort(key=lambda x: x.get("q", 0), reverse=True)
        sig = sigs[0]
        try:
            res = m.simulate(df, sig, du, balance, RISK_PCT, cfg)
        except Exception:
            continue
        if res is None:
            continue
        pnl = res.get("pnl", 0.0) if isinstance(res, dict) else res
        balance += pnl
        trades.append(pnl)

    if not trades:
        return {"sym": sym, "trades": 0, "wr": 0, "pf": 0, "pnl": 0, "bal": balance}

    t = pd.Series(trades)
    wins = t[t > 0]
    losses = t[t <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    wr = len(wins) / len(t) * 100
    return {"sym": sym, "trades": len(t), "wr": wr, "pf": pf,
            "pnl": t.sum(), "bal": balance}


def main():
    if not m.connect_mt5():
        print("MT5 connect failed")
        sys.exit(1)
    print(f"Testing V5 edge on EURGBP/NZDCAD | {BACKTEST_DAYS}d | risk {RISK_PCT}%")
    print("=" * 60)
    rows = []
    for sym, cfg in TEST_CONFIG.items():
        print(f"\n--- {sym} ---")
        r = run_pair(sym, cfg)
        if r is None:
            print(f"  {sym}: no data")
            continue
        rows.append(r)
        print(f"  trades={r['trades']}  WR={r['wr']:.1f}%  PF={r['pf']:.2f}  "
              f"P&L=${r['pnl']:+.2f}  bal=${r['bal']:.2f}")

    print("\n" + "=" * 60)
    print("VERDICT (keep if PF >= 1.4 AND trades >= 20 AND P&L >= $200):")
    for r in rows:
        v = "KEEP" if r["pf"] >= 1.4 and r["trades"] >= 20 and r["pnl"] >= 200 else "SKIP"
        print(f"  {r['sym']:8} PF {r['pf']:.2f}  WR {r['wr']:.1f}%  "
              f"{r['trades']} trades  P&L ${r['pnl']:+.2f}  ->  {v}")


if __name__ == "__main__":
    main()
