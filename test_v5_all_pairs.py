"""
Test V5 Asian Range Sweep edge on EVERY untested tradable symbol on FTMO-Demo.
Auto-discovers symbols, filters tradable categories (FX + Metals + Indices),
skips pairs already in V5, tests each individually with V5's own functions.
Reports only pairs that pass: PF >= 1.4 AND trades >= 20 AND P&L >= $200.
V5 source file is NOT modified.
"""
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import MetaTrader5 as mt5
import multi_pair_backtest as m

UTC = timezone.utc
BACKTEST_DAYS = 365
INITIAL_BALANCE = 10_000.0
RISK_PCT = 0.8

# V5 already has these — skip
ALREADY_IN_V5 = {"XAUUSD", "GBPUSD", "AUDUSD", "EURUSD", "USDCAD",
                 "USDCHF", "AUDCHF", "AUDNZD", "NZDUSD"}

# Already proven losers (don't waste cycles)
KNOWN_LOSERS = {"XAGUSD", "USDJPY", "AUDJPY", "NZDJPY", "GBPJPY", "EURJPY",
                "NZDCAD", "EURGBP", "CADJPY"}


def is_tradable_candidate(name):
    """Filter to FX majors/crosses + metals + indices, skip exotics."""
    n = name.upper().replace(".CASH", "")
    # 6-char FX or 3-letter metal/index
    if len(name) > 12:
        return False
    # Skip CFDs we don't want
    bad = ["XPT", "XPD", "USDCNH", "USDHKD", "USDSGD", "USDZAR", "USDMXN",
           "USDTRY", "USDNOK", "USDSEK", "USDDKK", "USDPLN", "USDHUF",
           "USDCZK", "USDILS", "USDRUB", "BTC", "ETH"]
    if any(b in n for b in bad):
        return False
    # Must be 6-letter FX OR known index/metal pattern
    if len(name) == 6 and name.isalpha():
        return True
    if name.endswith(".cash"):
        return True
    if name in ("XAUUSD", "XAGUSD"):
        return True
    return False


def get_cfg(sym, info):
    point = info.point
    is_jpy = "JPY" in sym
    is_index = sym.endswith(".cash")
    is_metal = sym in ("XAUUSD", "XAGUSD")
    contract = info.trade_contract_size
    cat = "Index" if is_index else ("Metal" if is_metal else
                                     ("Cross" if is_jpy or sym[:3] not in
                                      ("EUR", "GBP", "AUD", "USD", "NZD")
                                      else "Major"))
    spread = max(int(info.spread), 3) if info.spread > 0 else 10
    if is_index:
        sl_pts, min_r, max_r = 2000, 100, 30000
    elif is_metal:
        sl_pts, min_r, max_r = 70, 100, 3000
    else:
        sl_pts, min_r, max_r = 80, 150, 1800
    return {"spread": spread, "point": point, "contract": contract,
            "lot_max": 50.0, "sl_pts": sl_pts, "min_r": min_r, "max_r": max_r,
            "quality": 6, "cat": cat, "jpy": is_jpy, "risk_override": None}


def run_pair(sym, cfg):
    df = m.fetch_h1(sym, BACKTEST_DAYS + 15)
    if df is None or len(df) < 100:
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
            if ah is None:
                continue
            bias = m.get_bias(df, du)
            sigs = m.scan_signals(df, sym, du, ah, al, bias, cfg)
            if not sigs:
                continue
            sigs.sort(key=lambda x: x.get("q", 0), reverse=True)
            res = m.simulate(df, sigs[0], du, balance, RISK_PCT, cfg)
            if res is None:
                continue
            pnl = res.get("pnl", 0.0) if isinstance(res, dict) else res
            balance += pnl
            trades.append(pnl)
        except Exception:
            continue

    if len(trades) < 5:
        return None
    t = pd.Series(trades)
    wins = t[t > 0]
    losses = t[t <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 999.0
    wr = len(wins) / len(t) * 100
    return {"sym": sym, "trades": len(t), "wr": wr, "pf": pf,
            "pnl": float(t.sum()), "bal": balance}


def main():
    if not m.connect_mt5():
        print("MT5 connect failed")
        sys.exit(1)

    all_syms = mt5.symbols_get()
    candidates = []
    for s in all_syms:
        if s.name in ALREADY_IN_V5 or s.name in KNOWN_LOSERS:
            continue
        if not is_tradable_candidate(s.name):
            continue
        # Select to ensure rates fetch works
        mt5.symbol_select(s.name, True)
        candidates.append(s)

    print(f"Testing {len(candidates)} untested pairs on V5 edge")
    print("=" * 70)

    results = []
    for s in candidates:
        cfg = get_cfg(s.name, s)
        r = run_pair(s.name, cfg)
        if r is None:
            print(f"  {s.name:14} no data / too few trades")
            continue
        results.append(r)
        flag = "*" if r["pf"] >= 1.4 and r["trades"] >= 20 and r["pnl"] >= 200 else " "
        print(f" {flag}{r['sym']:14} trades={r['trades']:3}  WR={r['wr']:5.1f}%  "
              f"PF={r['pf']:5.2f}  P&L=${r['pnl']:+9.2f}")

    print("\n" + "=" * 70)
    print("WINNERS (PF>=1.4 AND trades>=20 AND P&L>=$200):")
    winners = [r for r in results
               if r["pf"] >= 1.4 and r["trades"] >= 20 and r["pnl"] >= 200]
    winners.sort(key=lambda x: x["pnl"], reverse=True)
    if not winners:
        print("  (none)")
    for r in winners:
        print(f"  {r['sym']:14} PF {r['pf']:.2f}  WR {r['wr']:.1f}%  "
              f"{r['trades']} trades  P&L ${r['pnl']:+.2f}")

    mt5.shutdown()


if __name__ == "__main__":
    main()
