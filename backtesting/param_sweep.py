"""Parameter sweep over (sigma_mult, abs_floor_pts, cooldown_sec).

Cartesian product = 3 × 4 × 3 = 36 combos. Each combo is applied by mutating
the detector class attributes (LiquiditySweepDetector / TickMomentumDetector /
RejectionDetector) before running a backtest, and restored afterwards. Phase 5
detectors are untouched — we only flip class attrs in/out around each run.

The sweep is intentionally serial: detector class attrs are global state.
Cross-process parallelism would work but the per-run cost on 31k–70k ticks
is small enough that the simpler serial implementation wins in practice.

Output: rows of metrics, one per (combo × date), writable to CSV via
:meth:`write_csv`.
"""

from __future__ import annotations
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from backtesting.backtest_runner import run_backtest_sync
from backtesting.walk_forward import (
    WalkForwardResult,
    WalkForwardRunner,
    extract_metrics,
)
from strategy.signals.liquidity_sweep import LiquiditySweepDetector
from strategy.signals.rejection import RejectionDetector
from strategy.signals.tick_momentum import TickMomentumDetector


# Default grid. Keep here so callers can override per-experiment without
# touching the runner internals.
SWEEP_SIGMA_MULT: List[float] = [2.5, 3.0, 3.5]
SWEEP_ABS_FLOOR_PTS: List[float] = [22.0, 25.0, 28.0, 32.0]
SWEEP_COOLDOWN_SEC: List[float] = [2.0, 3.0, 5.0]


def all_combos(
    sigma: Optional[List[float]] = None,
    floor: Optional[List[float]] = None,
    cooldown: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    s = sigma if sigma is not None else SWEEP_SIGMA_MULT
    f = floor if floor is not None else SWEEP_ABS_FLOOR_PTS
    c = cooldown if cooldown is not None else SWEEP_COOLDOWN_SEC
    out: List[Dict[str, Any]] = []
    for sm, fl, cd in itertools.product(s, f, c):
        out.append(
            {"sigma_mult": sm, "abs_floor_pts": fl, "cooldown_sec": cd}
        )
    return out


def apply_detector_params(params: Dict[str, Any]) -> Callable[[], None]:
    """Mutate detector class attributes; return a teardown to restore them.

    Why class attrs and not constructors: Phase 5 detectors take no knobs
    via __init__, and we're explicitly avoiding modifying that code in this
    phase. The trade-off is global state, mitigated by always running the
    teardown in a try/finally on the caller side.
    """
    sigma = float(params["sigma_mult"])
    floor = float(params["abs_floor_pts"])
    cooldown_ms = int(float(params["cooldown_sec"]) * 1000)

    # Snapshot originals so restore is exact (including for fields we don't
    # touch — defensive against future grid expansions).
    saved = {
        "sweep_std_mult": LiquiditySweepDetector.STD_MULT,
        "sweep_abs_floor": LiquiditySweepDetector.ABS_FLOOR_PTS,
        "sweep_cooldown_ms": LiquiditySweepDetector.COOLDOWN_MS,
        "rejection_std_mult": RejectionDetector.STD_MULT,
        "rejection_abs_floor": RejectionDetector.ABS_FLOOR_PTS,
        "momentum_cooldown_ms": TickMomentumDetector.COOLDOWN_MS,
    }

    # Sweep + rejection share the same trigger by Phase-5 design (they both
    # arm off the same spike), so we drive their thresholds together.
    LiquiditySweepDetector.STD_MULT = sigma
    LiquiditySweepDetector.ABS_FLOOR_PTS = floor
    LiquiditySweepDetector.COOLDOWN_MS = cooldown_ms
    RejectionDetector.STD_MULT = sigma
    RejectionDetector.ABS_FLOOR_PTS = floor
    # Momentum runs on cumulative — only its cooldown is interesting in this
    # grid. Floor / sigma are not analogues for the 20-tick window.
    TickMomentumDetector.COOLDOWN_MS = cooldown_ms

    def restore() -> None:
        LiquiditySweepDetector.STD_MULT = saved["sweep_std_mult"]
        LiquiditySweepDetector.ABS_FLOOR_PTS = saved["sweep_abs_floor"]
        LiquiditySweepDetector.COOLDOWN_MS = saved["sweep_cooldown_ms"]
        RejectionDetector.STD_MULT = saved["rejection_std_mult"]
        RejectionDetector.ABS_FLOOR_PTS = saved["rejection_abs_floor"]
        TickMomentumDetector.COOLDOWN_MS = saved["momentum_cooldown_ms"]

    return restore


@dataclass
class SweepRunResult:
    params: Dict[str, Any]
    date: str
    metrics: Dict[str, Any]
    signals_seen: int
    signals_blocked: int
    final_equity: float
    return_pct: float

    def to_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"date": self.date}
        for k, v in self.params.items():
            row[f"p_{k}"] = v
        for k, v in self.metrics.items():
            row[k] = v
        row["signals_seen"] = self.signals_seen
        row["signals_blocked"] = self.signals_blocked
        row["final_equity"] = self.final_equity
        row["return_pct"] = self.return_pct
        return row


class ParamSweep:
    """Sweeps a grid against one or more partition dates.

    For walk-forward usage, pass two dates and a WalkForwardRunner instead —
    this class is the simpler "report metrics per combo per date" sibling.
    """

    def __init__(
        self,
        dates: Iterable[str],
        symbol: Optional[str] = None,
        capital: float = 10_000.0,
        broker_type: Optional[str] = None,
        account_type: Optional[str] = None,
        combos: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.dates = list(dates)
        self.symbol = symbol
        self.capital = capital
        self.broker_type = broker_type
        self.account_type = account_type
        self.combos = combos if combos is not None else all_combos()
        self.results: List[SweepRunResult] = []

    def run(self) -> List[SweepRunResult]:
        out: List[SweepRunResult] = []
        for combo in self.combos:
            teardown = apply_detector_params(combo)
            try:
                for d in self.dates:
                    summary = run_backtest_sync(
                        d,
                        symbol=self.symbol,
                        capital=self.capital,
                        broker_type=self.broker_type,
                        account_type=self.account_type,
                    )
                    out.append(
                        SweepRunResult(
                            params=dict(combo),
                            date=d,
                            metrics=extract_metrics(summary),
                            signals_seen=summary.get("signals_seen", 0),
                            signals_blocked=summary.get("signals_blocked", 0),
                            final_equity=summary.get("final_equity", 0.0),
                            return_pct=summary.get("return_pct", 0.0),
                        )
                    )
            finally:
                teardown()
        self.results = out
        return out

    def write_csv(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.results:
            path.write_text("")
            return path
        rows = [r.to_row() for r in self.results]
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        return path


def run_walk_forward_sweep(
    train_date: str,
    test_date: str,
    symbol: Optional[str] = None,
    capital: float = 10_000.0,
    broker_type: Optional[str] = None,
    account_type: Optional[str] = None,
    combos: Optional[List[Dict[str, Any]]] = None,
) -> List[WalkForwardResult]:
    """Run the grid through a WalkForwardRunner. Returns ranked results."""
    runner = WalkForwardRunner(
        train_date=train_date,
        test_date=test_date,
        symbol=symbol,
        capital=capital,
        broker_type=broker_type,
        account_type=account_type,
        apply_params=apply_detector_params,
    )
    return runner.run_many(combos or all_combos())
