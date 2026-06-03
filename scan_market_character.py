"""
Market character scan — ranks all pairs on 15M data for scalp-suitability.
Measures: volatility (ATR% of price), trend-vs-range (efficiency ratio),
session activity, and spread cost. Data decides which pair + approach fits.
"""
import glob, os
import pandas as pd
import numpy as np

BARS_DIR = "data/bars"

def load_15m(pair):
    df = pd.read_parquet(f"{BARS_DIR}/{pair}_15M.parquet")
    df["dt"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    return df

def atr_pct(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    return (atr / c * 100).median()  # median ATR as % of price

def efficiency_ratio(df, n=20):
    # Kaufman ER: net move / total path. ~1 = strong trend, ~0 = choppy/range
    c = df["close"]
    change = (c - c.shift(n)).abs()
    vol = c.diff().abs().rolling(n).sum()
    er = (change / vol).replace([np.inf, -np.inf], np.nan)
    return er.median()

def spread_pips(df, pair):
    # spread_mean in points; pip = 10 points for FX, varies for metals/indices
    pip_pts = 100 if pair in ("XAUUSD",) else (1000 if pair in ("XAGUSD",) else 10)
    return (df["spread_mean"].median() / pip_pts)

def session_atr(df):
    # ATR% by UTC session: Asia 0-7, London 7-12, NY 12-17
    df = df.copy()
    df["hr"] = df["dt"].dt.hour
    rng = (df["high"] - df["low"]) / df["close"] * 100
    out = {}
    for name, (a, b) in {"Asia": (0, 7), "London": (7, 12), "NY": (12, 17)}.items():
        m = (df["hr"] >= a) & (df["hr"] < b)
        out[name] = rng[m].median()
    return out

def main():
    pairs = sorted(set(
        os.path.basename(f).replace("_15M.parquet", "")
        for f in glob.glob(f"{BARS_DIR}/*_15M.parquet")
    ))
    rows = []
    for p in pairs:
        try:
            df = load_15m(p)
            if len(df) < 1000:
                continue
            sess = session_atr(df)
            best_sess = max(sess, key=sess.get)
            rows.append({
                "pair": p,
                "atr_pct": round(atr_pct(df), 4),
                "eff_ratio": round(efficiency_ratio(df), 3),
                "spread_pip": round(spread_pips(df, p), 2),
                "best_session": best_sess,
                "asia": round(sess["Asia"], 4),
                "london": round(sess["London"], 4),
                "ny": round(sess["NY"], 4),
            })
        except Exception as e:
            print(f"skip {p}: {e}")
    r = pd.DataFrame(rows)
    # scalp score: high volatility + clear trend tendency, low spread drag
    r["vol_rank"] = r["atr_pct"].rank(ascending=False)
    r["trend_rank"] = r["eff_ratio"].rank(ascending=False)
    r["cost_rank"] = r["spread_pip"].rank(ascending=True)  # low spread = better
    r["scalp_score"] = (r["vol_rank"] + r["trend_rank"] + r["cost_rank"]).rank()
    r = r.sort_values("scalp_score")

    pd.set_option("display.width", 200, "display.max_rows", 100)
    print("\n=== SCALP-SUITABILITY RANKING (lower scalp_score = better) ===\n")
    print(r[["pair", "atr_pct", "eff_ratio", "spread_pip", "best_session", "scalp_score"]].to_string(index=False))
    print("\n=== TOP 5 — character detail ===\n")
    print(r.head(5)[["pair", "atr_pct", "eff_ratio", "spread_pip", "asia", "london", "ny", "best_session"]].to_string(index=False))
    r.to_csv("out/market_character_scan.csv", index=False)
    print("\nsaved out/market_character_scan.csv")

if __name__ == "__main__":
    main()
