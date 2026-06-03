"""Helpers for parquet round-trip tests."""

from __future__ import annotations
from pathlib import Path
from typing import List

import pyarrow.parquet as pq


def list_date_partitions(symbol_dir: Path) -> List[Path]:
    """Return sorted date= partition directories under a writer's symbol dir."""
    if not symbol_dir.exists():
        return []
    return sorted(p for p in symbol_dir.iterdir() if p.is_dir() and p.name.startswith("date="))


def list_part_files(date_dir: Path) -> List[Path]:
    """Return sorted part-NNNNN.parquet files in a date partition."""
    if not date_dir.exists():
        return []
    return sorted(p for p in date_dir.iterdir() if p.suffix == ".parquet")


def read_partition_rows(date_dir: Path) -> int:
    """Sum row counts across all part files in a date partition."""
    rows = 0
    for f in list_part_files(date_dir):
        rows += pq.read_table(f).num_rows
    return rows


def read_partition_table(date_dir: Path):
    """Concatenate all part files in a partition into a single pyarrow Table."""
    import pyarrow as pa
    tables = [pq.read_table(f) for f in list_part_files(date_dir)]
    if not tables:
        return pa.table({})
    return pa.concat_tables(tables)
