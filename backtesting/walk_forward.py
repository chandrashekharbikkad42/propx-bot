"""Walk-forward validation.

Splits the available tick data into a train partition and a test partition,
runs the same parameter combo over each, and computes an overfitting score
defined as train_pnl / test_pnl. Robust combos have:
    - positive test_pnl
    - overfit_score close to 1.0 (train ~ test)

The runner is data-driven — train/test dates are passed in by the caller,
no hard-coded splits. ParamSweep can stack 36 combos × 2 partitions; the
shared monkey-patch context is owned there, not here.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from backtesting.backtest_runner import run_backtest_sync


def _profit_factor(gross_wins: float, gross_losses: float) -> float:
    if gross_losses == 0.0:
        return float("inf") if gross_wins > 0 else 0.0
    return gross_wins / abs(gross_losses)


def extract_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the journal summary into the metric set requested in spec."""
    avg_win = summary.get("avg_win", 0.0)
    avg_loss = summary.get("avg_loss", 0.0)
    wins = summary.get("wins", 0)
    losses = summary.get("losses", 0)
    gross_wins = avg_win * wins
    gross_losses = avg_loss * losses
    return {
        "total_trades": summary.get("total_trades", 0),
        "win_rate": summary.get("win_rate", 0.0),
        "gross_pnl": summary.get("gross_pnl", 0.0),
        "max_dd_usd": summary.get("max_drawdown_usd", 0.0),
        "profit_factor": _profit_factor(gross_wins, gross_losses),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


@dataclass
class WalkForwardResult:
    params: Dict[str, Any]
    train: Dict[str, Any]
    test: Dict[str, Any]
    overfit_score: float  # train_pnl / test_pnl
    train_pnl: float
    test_pnl: float
    robust: bool

    def to_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for k, v in self.params.items():
            row[f"p_{k}"] = v
        for prefix, m in (("train", self.train), ("test", self.test)):
            for k, v in m.items():
                row[f"{prefix}_{k}"] = v
        row["overfit_score"] = self.overfit_score
        row["train_pnl"] = self.train_pnl
        row["test_pnl"] = self.test_pnl
        row["robust"] = self.robust
        return row


def _overfit_score(train_pnl: float, test_pnl: float) -> float:
    """Ratio of train pnl to test pnl. NaN/inf semantics:
      - test_pnl == 0: infinite (any positive train looks overfit vs zero test)
      - both <= 0:     0.0 (not robust either way — caller flags via `robust`)
    """
    if test_pnl == 0.0:
        return float("inf") if train_pnl != 0.0 else 0.0
    return train_pnl / test_pnl


def _is_robust(train_pnl: float, test_pnl: float, ratio_band: Tuple[float, float]) -> bool:
    """Robust if test is profitable AND train/test ratio lies in band."""
    if test_pnl <= 0.0:
        return False
    if train_pnl <= 0.0:
        return False
    ratio = train_pnl / test_pnl
    lo, hi = ratio_band
    return lo <= ratio <= hi


class WalkForwardRunner:
    """Runs a list of param combos across (train_date, test_date) and ranks them.

    Parameters
    ----------
    train_date / test_date : YYYY-MM-DD partition strings.
    symbol : optional symbol override (defaults to global settings).
    capital : starting equity for each run.
    apply_params : callable(params_dict) -> teardown_callable.
        Receives a parameter combo and applies it to the live detector
        class attributes; returns a callable that restores the original
        values. ParamSweep injects this — keeping it pluggable means the
        runner stays unaware of *which* parameters are being swept.
    robust_band : (low, high) range for train/test ratio considered robust.
        Default (0.5, 2.0) — train should not produce more than 2× test
        nor less than half. Tunable per-experiment.
    """

    DEFAULT_ROBUST_BAND: Tuple[float, float] = (0.5, 2.0)

    def __init__(
        self,
        train_date: str,
        test_date: str,
        symbol: Optional[str] = None,
        capital: float = 10_000.0,
        broker_type: Optional[str] = None,
        account_type: Optional[str] = None,
        apply_params=None,
        robust_band: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.train_date = train_date
        self.test_date = test_date
        self.symbol = symbol
        self.capital = capital
        self.broker_type = broker_type
        self.account_type = account_type
        self._apply_params = apply_params or (lambda params: (lambda: None))
        self.robust_band = robust_band or self.DEFAULT_ROBUST_BAND
        self.results: List[WalkForwardResult] = []

    def run_one(self, params: Dict[str, Any]) -> WalkForwardResult:
        teardown = self._apply_params(params)
        try:
            train_summary = run_backtest_sync(
                self.train_date,
                symbol=self.symbol,
                capital=self.capital,
                broker_type=self.broker_type,
                account_type=self.account_type,
            )
            test_summary = run_backtest_sync(
                self.test_date,
                symbol=self.symbol,
                capital=self.capital,
                broker_type=self.broker_type,
                account_type=self.account_type,
            )
        finally:
            teardown()

        train_m = extract_metrics(train_summary)
        test_m = extract_metrics(test_summary)
        train_pnl = train_m["gross_pnl"]
        test_pnl = test_m["gross_pnl"]

        return WalkForwardResult(
            params=dict(params),
            train=train_m,
            test=test_m,
            overfit_score=_overfit_score(train_pnl, test_pnl),
            train_pnl=train_pnl,
            test_pnl=test_pnl,
            robust=_is_robust(train_pnl, test_pnl, self.robust_band),
        )

    def run_many(self, combos: List[Dict[str, Any]]) -> List[WalkForwardResult]:
        out: List[WalkForwardResult] = []
        for combo in combos:
            out.append(self.run_one(combo))
        self.results = out
        return out

    def ranked_by_test_pnl(self) -> List[WalkForwardResult]:
        return sorted(self.results, key=lambda r: r.test_pnl, reverse=True)

    def best_robust(self) -> Optional[WalkForwardResult]:
        robust = [r for r in self.results if r.robust]
        if not robust:
            return None
        return max(robust, key=lambda r: r.test_pnl)
