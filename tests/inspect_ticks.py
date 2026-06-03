"""Read-back utility. Inspects the most recent parquet partition for the configured symbol."""

from __future__ import annotations
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import settings
from data.tick_writer import TICK_SCHEMA
from utils.logger import logger


# XAUUSD point size on IC Markets (2-digit pricing). Used for human-readable spread.
XAU_POINT = 0.01


def _find_latest_partition(symbol: str) -> Optional[Path]:
    symbol_dir = settings.data_dir / f"symbol={symbol}"
    if not symbol_dir.exists():
        return None
    date_dirs = sorted(
        (p for p in symbol_dir.iterdir() if p.is_dir() and p.name.startswith("date=")),
        key=lambda p: p.name,
    )
    return date_dirs[-1] if date_dirs else None


def main() -> int:
    symbol = settings.symbol
    logger.info("=" * 60)
    logger.info("PARQUET PARTITION INSPECTION")
    logger.info("=" * 60)
    logger.info(f"Data root : {settings.data_dir}")
    logger.info(f"Symbol    : {symbol}")

    partition = _find_latest_partition(symbol)
    if partition is None:
        logger.error("No partitions found")
        return 1

    parts = sorted(partition.glob("part-*.parquet"))
    if not parts:
        logger.error(f"No part files in {partition}")
        return 1

    logger.info(f"Partition : {partition.name}")
    logger.info(f"Files     : {len(parts)}")

    # On-disk schema check uses the file's own arrow schema (no path-derived
    # partition columns). The data load below also restricts to TICK_SCHEMA
    # columns so downstream stats see exactly what was persisted.
    file_schema = pq.ParquetFile(parts[0]).schema_arrow
    if file_schema.equals(TICK_SCHEMA):
        logger.info("Schema    : matches TICK_SCHEMA")
    else:
        logger.warning("Schema    : MISMATCH vs TICK_SCHEMA")
        logger.warning(f"  actual : {file_schema}")
        logger.warning(f"  expect : {TICK_SCHEMA}")

    columns = [f.name for f in TICK_SCHEMA]
    tables = [pq.read_table(p, columns=columns) for p in parts]
    table = pa.concat_tables(tables)

    n = table.num_rows
    if n == 0:
        logger.warning("Partition has 0 rows")
        return 0

    time_msc = table.column("time_msc").to_numpy()
    bid = table.column("bid").to_numpy()
    ask = table.column("ask").to_numpy()
    spread = ask - bid

    t_min_msc = int(time_msc.min())
    t_max_msc = int(time_msc.max())
    t_min = datetime.fromtimestamp(t_min_msc / 1000.0, tz=timezone.utc)
    t_max = datetime.fromtimestamp(t_max_msc / 1000.0, tz=timezone.utc)
    span_sec = (t_max_msc - t_min_msc) / 1000.0
    rate = n / span_sec if span_sec > 0 else 0.0

    logger.info("--- ROW STATS ---")
    logger.info(f"  Rows               : {n}")
    logger.info(f"  First tick (UTC)   : {t_min.isoformat()}")
    logger.info(f"  Last tick  (UTC)   : {t_max.isoformat()}")
    logger.info(f"  Span               : {span_sec:.3f} s")
    logger.info(f"  Tick rate          : {rate:.2f} ticks/sec")

    logger.info("--- PRICE STATS ---")
    logger.info(f"  Bid min / max      : {bid.min():.2f} / {bid.max():.2f}")
    logger.info(f"  Ask min / max      : {ask.min():.2f} / {ask.max():.2f}")

    logger.info("--- SPREAD STATS ---")
    logger.info(f"  Mean (price)       : {spread.mean():.4f}")
    logger.info(f"  Max  (price)       : {spread.max():.4f}")
    logger.info(f"  Min  (price)       : {spread.min():.4f}")
    logger.info(f"  Mean (points)      : {spread.mean() / XAU_POINT:.2f}")
    logger.info(f"  Max  (points)      : {spread.max() / XAU_POINT:.2f}")

    logger.info("--- FILES ---")
    for p in parts:
        rows = pq.read_metadata(p).num_rows
        size_kb = p.stat().st_size / 1024.0
        logger.info(f"  {p.name}  rows={rows:<5}  size={size_kb:.1f} KB")

    logger.success("Inspection complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
