"""Phase 8B / propX Multi-Setup — capture historical bars for prop-firm pairs.

Usage:
    python scripts/capture_historical_bars.py                       # default: priority + asian sweep, 1H, 2yr
    python scripts/capture_historical_bars.py --pairs EURUSD,XAUUSD # custom subset
    python scripts/capture_historical_bars.py --years 1             # shorter range
    python scripts/capture_historical_bars.py --force               # overwrite existing
    python scripts/capture_historical_bars.py --multi-setup         # 28-pair universe + 1H AND 15M
    python scripts/capture_historical_bars.py --timeframes 1H,15M   # comma-separated TFs

Output: `data/bars/{SYMBOL}_{TF}.parquet` per (pair, tf).

Idempotent: skips a (pair, tf) if its parquet already exists, unless --force.

Hinglish: ye script ek baar chala lo, 2 saal ka data mil jaayega. Backtest
ke liye chahiye. Live MT5 connection required — terminal khula hona chahiye.
"""

from __future__ import annotations
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running as a script: `python scripts/capture_historical_bars.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import MetaTrader5 as mt5  # noqa: E402

from config.settings import settings  # noqa: E402
from data.bar_aggregator import bars_path, check_bar_integrity, read_bars_parquet, write_bars_parquet  # noqa: E402
from data.bar_capture_utils import bars_summary, mt5_rates_to_bars  # noqa: E402
from data.mt5_connector import MT5Connector, MT5ConnectionError  # noqa: E402
from utils.logger import logger  # noqa: E402


# User-priority pairs from the Phase-8B spec — captured first.
PRIORITY_PAIRS: tuple[str, ...] = (
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
    "USDCAD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY",
)

# Asian Sweep V5 strategy universe — kept in sync with
# `config.asian_sweep_config.PAIRS`. Default fetch list = union of these and
# `PRIORITY_PAIRS` so capturing once gives both bots their parquet feeds.
ASIAN_SWEEP_PAIRS: tuple[str, ...] = (
    "XAUUSD", "GBPUSD", "AUDUSD", "EURUSD",
    "USDCAD", "USDCHF", "AUDCHF", "AUDNZD",
)

# propX Multi-Setup universe — 28 FX/metal pairs confirmed tradeable on
# FTMO-Demo (trade_mode=4 / FULL). Verified 2026-05-26 against the live
# broker. Includes 7 USD majors, EUR/GBP/AUD/NZD crosses, JPY/CHF crosses,
# and XAUUSD (gold). 4-setup detectors will scan this universe on 1H + 15M.
MULTI_SETUP_PAIRS: tuple[str, ...] = (
    # USD majors (7)
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
    # EUR crosses (6)
    "EURJPY", "EURGBP", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
    # GBP crosses (5)
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
    # AUD crosses (4)
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    # NZD crosses (3)
    "NZDJPY", "NZDCHF", "NZDCAD",
    # JPY crosses (2)
    "CADJPY", "CHFJPY",
    # Metal (1)
    "XAUUSD",
)

# Timeframe label → (MT5 constant, minutes). Single source of truth.
# Extend here to add 4H/D — caller doesn't need to touch other code.
TIMEFRAME_MAP: dict[str, tuple[int, int]] = {
    "1M":  (mt5.TIMEFRAME_M1,  1),
    "5M":  (mt5.TIMEFRAME_M5,  5),
    "15M": (mt5.TIMEFRAME_M15, 15),
    "1H":  (mt5.TIMEFRAME_H1,  60),
    "4H":  (mt5.TIMEFRAME_H4,  240),
}


def _default_pair_universe() -> tuple[str, ...]:
    """Union of PRIORITY_PAIRS and ASIAN_SWEEP_PAIRS, preserving insertion order."""
    seen: dict[str, None] = {}
    for p in PRIORITY_PAIRS + ASIAN_SWEEP_PAIRS:
        seen.setdefault(p, None)
    return tuple(seen.keys())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture historical bars from MT5.")
    default_pairs = _default_pair_universe()
    p.add_argument(
        "--pairs",
        default=",".join(default_pairs),
        help=f"Comma-separated symbols. Default = "
             f"{len(default_pairs)} pairs (PRIORITY + ASIAN_SWEEP universe).",
    )
    p.add_argument(
        "--asian-sweep-only",
        action="store_true",
        help="Capture only the 8 Asian Sweep V5 strategy pairs.",
    )
    p.add_argument(
        "--multi-setup",
        action="store_true",
        help="Capture the 28-pair propX Multi-Setup universe on 1H AND 15M.",
    )
    p.add_argument("--years", type=float, default=2.0, help="Lookback years. Default 2.")
    p.add_argument("--force", action="store_true", help="Overwrite existing parquet files.")
    p.add_argument(
        "--timeframe", default=None,
        choices=list(TIMEFRAME_MAP.keys()),
        help="Single bar timeframe. Use --timeframes for multiple.",
    )
    p.add_argument(
        "--timeframes", default=None,
        help=f"Comma-separated timeframes (any of {','.join(TIMEFRAME_MAP)}). "
             f"Default = 1H unless --multi-setup is set.",
    )
    return p.parse_args(argv)


def fetch_one_pair(
    symbol: str,
    years: float,
    timeframe_label: str = "1H",
    force: bool = False,
) -> dict:
    """Fetch + persist one pair. Returns a per-pair summary dict.

    Status values: "skipped" / "fetched" / "empty" / "error".
    """
    out_path = bars_path(symbol, timeframe_label)

    if timeframe_label not in TIMEFRAME_MAP:
        return {
            "symbol": symbol, "status": "error",
            "error": f"unknown timeframe {timeframe_label!r}; "
                     f"known: {list(TIMEFRAME_MAP)}",
        }
    mt5_tf, tf_minutes = TIMEFRAME_MAP[timeframe_label]

    if out_path.exists() and not force:
        try:
            df = read_bars_parquet(symbol, timeframe_label)
            integ = check_bar_integrity(df, timeframe_minutes=tf_minutes)
            return {
                "symbol": symbol, "status": "skipped",
                "path": str(out_path), "rows": integ["rows"],
                "monotonic": integ["monotonic"], "aligned": integ["aligned"],
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{symbol}: existing file unreadable ({exc}) — refetching")

    date_to = datetime.now(tz=timezone.utc)
    date_from = date_to - timedelta(days=int(years * 365))

    try:
        # Need MT5 to recognise this symbol — try to select it.
        # symbol_info_or None hits the terminal; if the pair isn't in Market Watch
        # the user must enable it. We attempt symbol_select to be friendly.
        if not mt5.symbol_select(symbol, True):
            return {
                "symbol": symbol, "status": "error",
                "error": f"symbol_select failed: {mt5.last_error()}",
            }

        rates = mt5.copy_rates_range(symbol, mt5_tf, date_from, date_to)
        if rates is None or len(rates) == 0:
            # MT5 quirk: short timeframes (1M/5M) hit the terminal's
            # MAX_BARS_IN_HISTORY cap and `copy_rates_range` over a long
            # period returns empty. Workaround: walk back in 50k chunks via
            # `copy_rates_from(ts, count)`. Stops when the cursor stops
            # advancing (broker depth exhausted).
            import numpy as np
            from_ts = int(date_from.timestamp())
            cursor_ts = int(date_to.timestamp())
            chunks: list = []
            seen_ts: set[int] = set()
            chunk_dtype = None
            chunk_size = 50_000
            max_chunks = 80
            for _ in range(max_chunks):
                r = mt5.copy_rates_from(symbol, mt5_tf, cursor_ts, chunk_size)
                if r is None or len(r) == 0:
                    break
                if chunk_dtype is None:
                    chunk_dtype = r.dtype
                mask = np.array([int(t) not in seen_ts for t in r["time"]])
                new_rows = r[mask]
                if len(new_rows) == 0:
                    break
                chunks.append(new_rows)
                seen_ts.update(int(t) for t in new_rows["time"])
                oldest_ts = int(new_rows["time"].min())
                if oldest_ts <= from_ts:
                    break
                if oldest_ts - 1 >= cursor_ts:
                    break
                cursor_ts = oldest_ts - 1
            if not chunks:
                return {
                    "symbol": symbol, "status": "empty",
                    "error": f"chunked walk-back returned no bars "
                             f"(last_error={mt5.last_error()})",
                }
            rates = np.concatenate(chunks)
            rates = rates[rates["time"] >= from_ts]
            if len(rates) == 0:
                return {"symbol": symbol, "status": "empty",
                        "error": "broker history shallower than requested range"}
            rates = np.sort(rates, order="time")

        bars = mt5_rates_to_bars(rates, symbol)
        write_bars_parquet(bars, symbol, timeframe_label)
        summ = bars_summary(bars)

        # Integrity check post-write — Belt + braces.
        df = read_bars_parquet(symbol, timeframe_label)
        integ = check_bar_integrity(df, timeframe_minutes=tf_minutes)

        return {
            "symbol": symbol, "status": "fetched",
            "path": str(out_path),
            "rows": summ["count"],
            "span_days": round(summ["span_days"], 1),
            "monotonic": integ["monotonic"],
            "aligned": integ["aligned"],
            "missing_count": integ["missing_count"],
        }
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "status": "error", "error": str(exc)}


def _resolve_pair_universe(args: argparse.Namespace) -> tuple[str, ...]:
    if args.multi_setup:
        return MULTI_SETUP_PAIRS
    if args.asian_sweep_only:
        return ASIAN_SWEEP_PAIRS
    return tuple(p.strip().upper() for p in args.pairs.split(",") if p.strip())


def _resolve_timeframes(args: argparse.Namespace) -> tuple[str, ...]:
    if args.timeframes:
        tfs = tuple(t.strip().upper() for t in args.timeframes.split(",") if t.strip())
    elif args.timeframe:
        tfs = (args.timeframe,)
    elif args.multi_setup:
        tfs = ("1H", "15M")
    else:
        tfs = ("1H",)
    for tf in tfs:
        if tf not in TIMEFRAME_MAP:
            raise SystemExit(f"Unknown timeframe {tf!r}; known: {list(TIMEFRAME_MAP)}")
    return tfs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pairs = _resolve_pair_universe(args)
    timeframes = _resolve_timeframes(args)
    if not pairs:
        logger.error("No pairs to fetch.")
        return 2

    settings.bars_dir.mkdir(parents=True, exist_ok=True)

    connector = MT5Connector()
    try:
        connector.connect()
    except MT5ConnectionError as exc:
        logger.error(f"MT5 connect failed: {exc}")
        return 1

    logger.info(
        f"Capturing bars | pairs={len(pairs)} tfs={','.join(timeframes)} "
        f"years={args.years} force={args.force} out={settings.bars_dir}"
    )

    results: list[dict] = []
    try:
        for tf in timeframes:
            logger.info(f"========== TIMEFRAME {tf} ==========")
            for sym in pairs:
                logger.info(f"--- {sym} {tf} ---")
                r = fetch_one_pair(sym, args.years, tf, args.force)
                r["timeframe"] = tf
                results.append(r)
                if r["status"] == "fetched":
                    logger.success(
                        f"{sym} {tf}: {r['rows']} bars, ~{r['span_days']}d span, "
                        f"mono={r['monotonic']} aligned={r['aligned']} "
                        f"missing={r['missing_count']}"
                    )
                elif r["status"] == "skipped":
                    logger.info(
                        f"{sym} {tf}: skipped (already have {r.get('rows', '?')} bars)"
                    )
                elif r["status"] == "empty":
                    logger.warning(f"{sym} {tf}: no bars returned in range")
                else:
                    logger.error(f"{sym} {tf}: {r.get('error', 'unknown error')}")
    finally:
        connector.disconnect()

    # ---- Summary ----
    n_ok = sum(1 for r in results if r["status"] == "fetched")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_err = sum(1 for r in results if r["status"] == "error")
    n_empty = sum(1 for r in results if r["status"] == "empty")
    logger.info("=" * 60)
    logger.info(
        f"DONE | fetched={n_ok} skipped={n_skip} empty={n_empty} errors={n_err}"
    )

    # ---- Strategy coverage check (Asian Sweep V5 universe per timeframe) ----
    for tf in timeframes:
        report_strategy_coverage(tf)
        if args.multi_setup:
            report_multi_setup_coverage(tf)

    logger.info("=" * 60)
    return 0 if n_err == 0 else 1


def report_multi_setup_coverage(timeframe_label: str = "1H") -> dict:
    """Walk the propX Multi-Setup 28-pair universe and report parquet presence."""
    present: list[str] = []
    missing: list[str] = []
    for sym in MULTI_SETUP_PAIRS:
        path = bars_path(sym, timeframe_label)
        if path.exists():
            present.append(sym)
        else:
            missing.append(sym)
    logger.info(f"--- propX Multi-Setup coverage ({timeframe_label}) ---")
    logger.info(
        f"{timeframe_label} parquet bars present "
        f"({len(present)}/{len(MULTI_SETUP_PAIRS)})"
    )
    if missing:
        logger.warning(
            f"{timeframe_label} MISSING ({len(missing)}/"
            f"{len(MULTI_SETUP_PAIRS)}): {', '.join(missing)}"
        )
    else:
        logger.success(
            f"All {len(MULTI_SETUP_PAIRS)} Multi-Setup pairs have "
            f"{timeframe_label} bars."
        )
    return {"present": present, "missing": missing}


def report_strategy_coverage(timeframe_label: str = "1H") -> dict:
    """Walk the Asian Sweep universe and report which pairs have parquet bars.

    Side-effect: logs a present/missing table. Returns a dict the caller
    can introspect (`{"present": [...], "missing": [...]}`).
    """
    present: list[str] = []
    missing: list[str] = []
    for sym in ASIAN_SWEEP_PAIRS:
        path = bars_path(sym, timeframe_label)
        if path.exists():
            present.append(sym)
        else:
            missing.append(sym)
    logger.info("--- Asian Sweep V5 strategy coverage ---")
    logger.info(
        f"{timeframe_label} parquet bars present "
        f"({len(present)}/{len(ASIAN_SWEEP_PAIRS)}): "
        f"{', '.join(present) if present else '<none>'}"
    )
    if missing:
        logger.warning(
            f"{timeframe_label} parquet bars MISSING "
            f"({len(missing)}/{len(ASIAN_SWEEP_PAIRS)}): "
            f"{', '.join(missing)}"
        )
    else:
        logger.success(
            f"All {len(ASIAN_SWEEP_PAIRS)} Asian Sweep pairs have "
            f"{timeframe_label} bars."
        )
    return {"present": present, "missing": missing}


if __name__ == "__main__":
    sys.exit(main())
