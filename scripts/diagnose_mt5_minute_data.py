"""Probe MT5 minute-data availability with multiple fetch strategies.

Goal: figure out why copy_rates_range / copy_rates_from_pos both return
empty for M1 / M5 on FTMO-Demo, while H1/M15 work fine.
"""
from __future__ import annotations
import io
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, io.UnsupportedOperation):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import MetaTrader5 as mt5  # noqa: E402
from data.mt5_connector import MT5Connector  # noqa: E402


def probe(symbol: str) -> None:
    print(f"\n--- {symbol} ---")
    ok = mt5.symbol_select(symbol, True)
    info = mt5.symbol_info(symbol)
    print(f"  symbol_select={ok}  visible={info.visible if info else 'no_info'}  "
          f"trade_mode={info.trade_mode if info else '?'}  "
          f"last_error_after_select={mt5.last_error()}")
    time.sleep(0.3)  # let terminal stabilise after select

    tests = [
        ("M1  from_pos(0, 100)",    lambda: mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)),
        ("M1  from_pos(0, 10_000)", lambda: mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10_000)),
        ("M5  from_pos(0, 100)",    lambda: mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)),
        ("M5  from_pos(0, 50_000)", lambda: mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 50_000)),
        ("M15 from_pos(0, 100)",    lambda: mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)),
        ("H1  from_pos(0, 100)",    lambda: mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)),
    ]
    for label, fn in tests:
        try:
            r = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<24} EXCEPTION {exc}  err={mt5.last_error()}")
            continue
        if r is None:
            print(f"  {label:<24} None  err={mt5.last_error()}")
        else:
            n = len(r)
            if n == 0:
                print(f"  {label:<24} 0 bars  err={mt5.last_error()}")
            else:
                t0 = datetime.fromtimestamp(int(r[0]['time']), tz=timezone.utc)
                t1 = datetime.fromtimestamp(int(r[-1]['time']), tz=timezone.utc)
                print(f"  {label:<24} {n} bars   {t0}  -> {t1}")

    # copy_rates_from with small count + int timestamp
    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    for n_req in (50_000, 99_999):
        r = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, now_ts, n_req)
        if r is None or len(r) == 0:
            print(f"  M1 from_ts(now,{n_req:>6}) EMPTY err={mt5.last_error()}")
            continue
        t0 = datetime.fromtimestamp(int(r[0]['time']), tz=timezone.utc)
        t1 = datetime.fromtimestamp(int(r[-1]['time']), tz=timezone.utc)
        days = (t1 - t0).total_seconds() / 86400
        print(f"  M1 from_ts(now,{n_req:>6}) {len(r):>6} bars {t0} -> {t1} ({days:.1f}d)")

    # Walk-back chunks of 50k each
    print("  -- walk-back 50k chunks for M1 --")
    cursor = int(datetime.now(tz=timezone.utc).timestamp())
    total = 0; earliest = None
    for i in range(10):
        r = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, cursor, 50_000)
        if r is None or len(r) == 0:
            print(f"    chunk #{i}: EMPTY  err={mt5.last_error()}  cursor_ts={cursor}")
            break
        t0 = int(r[0]['time']); t1 = int(r[-1]['time'])
        total += len(r)
        earliest = t0 if earliest is None or t0 < earliest else earliest
        d0 = datetime.fromtimestamp(t0, tz=timezone.utc)
        d1 = datetime.fromtimestamp(t1, tz=timezone.utc)
        print(f"    chunk #{i}: {len(r):>5} bars  {d0} -> {d1}  total={total}")
        if t0 - 1 >= cursor:
            print(f"    cursor not advancing; depth exhausted")
            break
        cursor = t0 - 1
    if earliest is not None:
        print(f"  M1 deepest reachable: {datetime.fromtimestamp(earliest, tz=timezone.utc)}")


def main() -> int:
    conn = MT5Connector()
    conn.connect()
    info_term = mt5.terminal_info()
    print(f"Terminal: connected={info_term.connected}  trade_allowed={info_term.trade_allowed}  "
          f"build={info_term.build}  data_path={info_term.data_path}")
    print(f"Total symbols: {mt5.symbols_total()}")
    for sym in ("EURUSD", "GBPUSD", "USDJPY"):
        probe(sym)
    conn.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
