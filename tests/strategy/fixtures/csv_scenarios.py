"""Load real-trade rows from multi_pair_trades_v5.csv (verified backtest).

The CSV captures the OUTCOME of each trade (entry price, exit price, lot,
P&L, grade) — it does NOT include the OHLC of the trigger bar, so we can
only use it to parametrise *configuration* slices (per-pair, per-session,
per-direction) and verify that the detector's grade/quality mapping
matches the historical ranking.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable, List, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = REPO_ROOT / "multi_pair_trades_v5.csv"


class TradeRow(NamedTuple):
    sym: str
    date: str
    direction: str   # "LONG" / "SHORT"
    session: str     # "LONDON" / "NY"
    category: str    # "Major" / "Metal" / "Cross"
    entry: float
    exit: float
    er: str
    lot: float
    pnl: float
    tp1hit: bool
    quality: int
    balance: float
    month: str


def load_trades() -> List[TradeRow]:
    if not CSV_PATH.exists():
        return []
    out: List[TradeRow] = []
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            try:
                out.append(TradeRow(
                    sym=r["sym"],
                    date=r["date"],
                    direction=r["dir"],
                    session=r["sess"],
                    category=r["cat"],
                    entry=float(r["entry"]),
                    exit=float(r["exit"]),
                    er=r["er"],
                    lot=float(r["lot"]),
                    pnl=float(r["pnl"]),
                    tp1hit=r["tp1hit"].lower() == "true",
                    quality=int(r["q"]),
                    balance=float(r["balance"]),
                    month=r["month"],
                ))
            except (KeyError, ValueError):
                continue
    return out


TRADES: List[TradeRow] = load_trades()


def trades_by_sym(sym: str) -> Iterable[TradeRow]:
    return [t for t in TRADES if t.sym == sym]


def trades_by_session(session: str) -> Iterable[TradeRow]:
    return [t for t in TRADES if t.session == session]


def trades_by_direction(direction: str) -> Iterable[TradeRow]:
    return [t for t in TRADES if t.direction == direction]
