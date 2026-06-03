"""Audit scalping-data capture: per (symbol, tf) bars, span, file size, integrity.

Reports:
  - rows, date range (UTC), span days
  - file size (MB)
  - monotonic + aligned + ohlc_consistent flags
  - missing-bar count (gap detection)
  - per-pair total disk
  - grand total disk

Run:
    venv\\Scripts\\python.exe scripts/audit_scalping_data.py
"""
from __future__ import annotations
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, io.UnsupportedOperation):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.bar_aggregator import (  # noqa: E402
    bars_path, check_bar_integrity, read_bars_parquet,
)


PAIRS = ("EURUSD", "GBPUSD", "USDJPY")
TFS = (("1M", 1), ("5M", 5), ("15M", 15), ("1H", 60))
UTC = timezone.utc


def fmt_size(b: int) -> str:
    if b >= 1024 * 1024:
        return f"{b / (1024 * 1024):.1f} MB"
    return f"{b / 1024:.1f} KB"


def main() -> int:
    print("=" * 96)
    print("  SCALPING DATA AUDIT — EURUSD / GBPUSD / USDJPY × {1M, 5M, 15M, 1H}")
    print("=" * 96)
    print(f"  {'Sym':<7} {'TF':<4} {'Rows':>9} {'Start (UTC)':<20} {'End (UTC)':<20} "
          f"{'Days':>5} {'Mono':<5} {'Algn':<5} {'OHLC':<5} {'Miss':>6} {'Size':>10}")
    print(f"  {'-'*7} {'-'*4} {'-'*9} {'-'*20} {'-'*20} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*10}")

    grand_bytes = 0
    per_pair_bytes = {p: 0 for p in PAIRS}
    missing_files: list[str] = []
    for sym in PAIRS:
        for tf_label, tf_min in TFS:
            path = bars_path(sym, tf_label)
            if not path.exists():
                missing_files.append(f"{sym}_{tf_label}")
                print(f"  {sym:<7} {tf_label:<4} {'<missing>':>9} {'':<20} {'':<20} "
                      f"{'-':>5} {'-':<5} {'-':<5} {'-':<5} {'-':>6} {'-':>10}")
                continue
            size = path.stat().st_size
            grand_bytes += size
            per_pair_bytes[sym] += size
            try:
                df = read_bars_parquet(sym, tf_label)
                integ = check_bar_integrity(df, timeframe_minutes=tf_min)
                if integ["rows"] == 0:
                    print(f"  {sym:<7} {tf_label:<4} {0:>9} {'<empty>':<20} {'':<20} "
                          f"{'-':>5} {'-':<5} {'-':<5} {'-':<5} {'-':>6} {fmt_size(size):>10}")
                    continue
                t0 = datetime.fromtimestamp(int(df['time_msc'].iloc[0]) / 1000, tz=UTC)
                t1 = datetime.fromtimestamp(int(df['time_msc'].iloc[-1]) / 1000, tz=UTC)
                span_days = (t1 - t0).total_seconds() / 86400
                print(f"  {sym:<7} {tf_label:<4} {integ['rows']:>9,} "
                      f"{t0.strftime('%Y-%m-%d %H:%M'):<20} "
                      f"{t1.strftime('%Y-%m-%d %H:%M'):<20} "
                      f"{span_days:>5.0f} "
                      f"{'Y' if integ['monotonic'] else 'N':<5} "
                      f"{'Y' if integ['aligned'] else 'N':<5} "
                      f"{'Y' if integ['ohlc_consistent'] else 'N':<5} "
                      f"{integ['missing_count']:>6,} "
                      f"{fmt_size(size):>10}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {sym:<7} {tf_label:<4} {'ERR':>9} {str(exc)[:40]:<40} "
                      f"{'':<5} {'':<5} {'':<5} {'':<5} {'':>6} {fmt_size(size):>10}")
    print(f"  {'-'*96}")
    for sym in PAIRS:
        print(f"  {sym:<7} subtotal:           {fmt_size(per_pair_bytes[sym]):>20}")
    print(f"  {'GRAND TOTAL':<7}                  {fmt_size(grand_bytes):>20}")
    print("=" * 96)

    if missing_files:
        print(f"\n[!] Missing files: {', '.join(missing_files)}")
        return 1

    # Readiness check
    print("\n  READINESS — 1M-entry scalping backtest:")
    ready = True
    for sym in PAIRS:
        for tf_label, _ in TFS:
            path = bars_path(sym, tf_label)
            if not path.exists():
                ready = False
                print(f"    {sym} {tf_label} ABSENT")
    if ready:
        print("    All 12 (pair × tf) files present — ready to build 4-TF scalping detector.")
    print("=" * 96)
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
