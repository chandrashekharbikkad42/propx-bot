"""Pre-replay validation. Returns IntegrityReport (frozen). No MT5, no network."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from config.settings import settings
from data.tick_writer import TICK_SCHEMA
from utils.logger import logger


WARN_GAP_MS = 5_000
ERROR_GAP_MS = 60_000


@dataclass(frozen=True)
class IntegrityReport:
    symbol: str
    date: str
    partition_dir: Path
    file_count: int
    row_count: int
    first_msc: int
    last_msc: int
    monotonic: bool
    duplicate_count: int
    warn_gaps: tuple[tuple[int, int, int], ...]   # (prev_msc, next_msc, gap_ms)
    error_gaps: tuple[tuple[int, int, int], ...]
    schema_match: bool
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            self.row_count > 0
            and self.monotonic
            and self.duplicate_count == 0
            and not self.error_gaps
            and self.schema_match
            and not self.errors
        )


def check_partition(symbol: str, date: str) -> IntegrityReport:
    """Inspect a single (symbol, date) partition. Reads only the time_msc column."""
    partition_dir = settings.data_dir / f"symbol={symbol}" / f"date={date}"
    errors: list[str] = []

    if not partition_dir.exists():
        return IntegrityReport(
            symbol=symbol, date=date, partition_dir=partition_dir,
            file_count=0, row_count=0, first_msc=0, last_msc=0,
            monotonic=True, duplicate_count=0,
            warn_gaps=(), error_gaps=(), schema_match=False,
            errors=(f"partition not found: {partition_dir}",),
        )

    parts = sorted(partition_dir.glob("part-*.parquet"))
    if not parts:
        return IntegrityReport(
            symbol=symbol, date=date, partition_dir=partition_dir,
            file_count=0, row_count=0, first_msc=0, last_msc=0,
            monotonic=True, duplicate_count=0,
            warn_gaps=(), error_gaps=(), schema_match=False,
            errors=("no part files in partition",),
        )

    schema_match = True
    all_msc: list[int] = []
    for p in parts:
        try:
            file_schema = pq.ParquetFile(p).schema_arrow
            if not file_schema.equals(TICK_SCHEMA):
                schema_match = False
                errors.append(f"schema mismatch in {p.name}")
            table = pq.read_table(p, columns=["time_msc"])
            all_msc.extend(int(x) for x in table.column("time_msc").to_pylist())
        except Exception as exc:
            errors.append(f"read failed for {p.name}: {exc}")

    n = len(all_msc)
    if n == 0:
        return IntegrityReport(
            symbol=symbol, date=date, partition_dir=partition_dir,
            file_count=len(parts), row_count=0,
            first_msc=0, last_msc=0,
            monotonic=True, duplicate_count=0,
            warn_gaps=(), error_gaps=(), schema_match=schema_match,
            errors=tuple(errors) or ("partition has 0 rows",),
        )

    monotonic = True
    duplicates = 0
    warns: list[tuple[int, int, int]] = []
    errs: list[tuple[int, int, int]] = []
    for i in range(n - 1):
        a, b = all_msc[i], all_msc[i + 1]
        if b < a:
            monotonic = False
        elif b == a:
            duplicates += 1
        else:
            gap = b - a
            if gap > ERROR_GAP_MS:
                errs.append((a, b, gap))
            elif gap > WARN_GAP_MS:
                warns.append((a, b, gap))

    return IntegrityReport(
        symbol=symbol, date=date, partition_dir=partition_dir,
        file_count=len(parts), row_count=n,
        first_msc=all_msc[0], last_msc=all_msc[-1],
        monotonic=monotonic, duplicate_count=duplicates,
        warn_gaps=tuple(warns), error_gaps=tuple(errs),
        schema_match=schema_match, errors=tuple(errors),
    )


def log_report(report: IntegrityReport) -> None:
    """Pretty-print a report through the centralized logger."""
    logger.info("--- INTEGRITY REPORT ---")
    logger.info(f"  Partition       : {report.partition_dir}")
    logger.info(f"  Files / Rows    : {report.file_count} / {report.row_count}")
    if report.row_count:
        logger.info(f"  msc range       : {report.first_msc} → {report.last_msc}")
    logger.info(f"  Monotonic       : {report.monotonic}")
    logger.info(f"  Duplicates      : {report.duplicate_count}")
    logger.info(f"  Schema match    : {report.schema_match}")
    logger.info(f"  Gaps  (warn>5s) : {len(report.warn_gaps)}")
    logger.info(f"  Gaps  (err>60s) : {len(report.error_gaps)}")
    for a, b, gap in report.error_gaps:
        logger.error(f"  ERROR gap {gap} ms between {a} and {b}")
    for err in report.errors:
        logger.error(f"  ERROR: {err}")
    if report.ok:
        logger.success("Integrity OK")
    else:
        logger.warning("Integrity NOT OK")
