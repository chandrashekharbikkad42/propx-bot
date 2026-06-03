"""Generate the Phase-6 markdown report from walk-forward results.

The report has three sections:
  1. Top-5 combos by test_pnl — the candidates that work on unseen data.
  2. Overfitting analysis — train/test ratio histogram + flagged combos.
  3. Robust recommendation — best combo whose train/test ratio sits in the
     configured band AND that produces positive test_pnl.

Reproducible: same input list → identical output bytes.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from backtesting.walk_forward import WalkForwardResult


def _fmt_pnl(v: float) -> str:
    return f"${v:,.2f}"


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _fmt_score(v: float) -> str:
    if v == float("inf"):
        return "inf"
    if v != v:  # NaN
        return "n/a"
    return f"{v:.2f}"


def _fmt_params(p: Dict[str, Any]) -> str:
    return (
        f"σ={p.get('sigma_mult')}, "
        f"floor={p.get('abs_floor_pts')}pt, "
        f"cd={p.get('cooldown_sec')}s"
    )


def _row(cells: Iterable[Any]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _metric_table(results: List[WalkForwardResult], top_n: int = 5) -> str:
    header = _row(
        [
            "Rank",
            "Params",
            "Train PnL",
            "Test PnL",
            "Train Trades",
            "Test Trades",
            "Train WR",
            "Test WR",
            "Overfit",
            "Robust",
        ]
    )
    sep = "|" + "|".join(["---"] * 10) + "|"
    lines = [header, sep]
    ranked = sorted(results, key=lambda r: r.test_pnl, reverse=True)
    for i, r in enumerate(ranked[:top_n], start=1):
        lines.append(
            _row(
                [
                    i,
                    _fmt_params(r.params),
                    _fmt_pnl(r.train_pnl),
                    _fmt_pnl(r.test_pnl),
                    r.train["total_trades"],
                    r.test["total_trades"],
                    _fmt_pct(r.train["win_rate"]),
                    _fmt_pct(r.test["win_rate"]),
                    _fmt_score(r.overfit_score),
                    "yes" if r.robust else "no",
                ]
            )
        )
    return "\n".join(lines)


def _overfit_buckets(results: List[WalkForwardResult]) -> str:
    buckets = {
        "test losing (test_pnl <= 0)": 0,
        "severe overfit (ratio > 2.0)": 0,
        "robust (0.5 <= ratio <= 2.0)": 0,
        "test outperforms (ratio < 0.5)": 0,
    }
    for r in results:
        if r.test_pnl <= 0.0:
            buckets["test losing (test_pnl <= 0)"] += 1
            continue
        if r.train_pnl <= 0.0:
            # Train negative but test positive — unusual; counted as test-outperforms
            buckets["test outperforms (ratio < 0.5)"] += 1
            continue
        ratio = r.train_pnl / r.test_pnl
        if ratio > 2.0:
            buckets["severe overfit (ratio > 2.0)"] += 1
        elif ratio < 0.5:
            buckets["test outperforms (ratio < 0.5)"] += 1
        else:
            buckets["robust (0.5 <= ratio <= 2.0)"] += 1

    lines = [_row(["Bucket", "Count"]), "|---|---|"]
    for k, v in buckets.items():
        lines.append(_row([k, v]))
    return "\n".join(lines)


@dataclass
class Recommendation:
    params: Optional[Dict[str, Any]]
    rationale: str
    train_pnl: float
    test_pnl: float
    overfit_score: float


def best_recommendation(results: List[WalkForwardResult]) -> Recommendation:
    robust = [r for r in results if r.robust]
    if robust:
        best = max(robust, key=lambda r: r.test_pnl)
        return Recommendation(
            params=best.params,
            rationale=(
                "Highest test_pnl among combos with positive train+test PnL and "
                f"overfit ratio in [0.5, 2.0] (actual {best.overfit_score:.2f})."
            ),
            train_pnl=best.train_pnl,
            test_pnl=best.test_pnl,
            overfit_score=best.overfit_score,
        )
    positive_test = [r for r in results if r.test_pnl > 0.0]
    if positive_test:
        fallback = max(positive_test, key=lambda r: r.test_pnl)
        return Recommendation(
            params=fallback.params,
            rationale=(
                "No combo met the robust criteria; falling back to highest "
                "test_pnl combo (treat as candidate, not validated)."
            ),
            train_pnl=fallback.train_pnl,
            test_pnl=fallback.test_pnl,
            overfit_score=fallback.overfit_score,
        )
    return Recommendation(
        params=None,
        rationale="No combo produced positive test_pnl. Strategy needs review before deployment.",
        train_pnl=0.0,
        test_pnl=0.0,
        overfit_score=0.0,
    )


def generate_report(
    results: List[WalkForwardResult],
    train_date: str,
    test_date: str,
    symbol: str = "XAUUSD",
    broker_label: str = "",
) -> str:
    rec = best_recommendation(results)
    n_total = len(results)
    n_positive_test = sum(1 for r in results if r.test_pnl > 0.0)
    n_robust = sum(1 for r in results if r.robust)

    lines: List[str] = []
    lines.append("# Phase 6 — Walk-Forward Validation Report")
    lines.append("")
    lines.append(f"- Symbol: `{symbol}`")
    lines.append(f"- Train partition: `{train_date}`")
    lines.append(f"- Test partition: `{test_date}`")
    if broker_label:
        lines.append(f"- Broker context: {broker_label}")
    lines.append(f"- Combos evaluated: {n_total}")
    lines.append(f"- Combos with positive test PnL: {n_positive_test}")
    lines.append(f"- Robust combos (positive train+test, ratio in [0.5,2.0]): {n_robust}")
    lines.append("")
    lines.append("## 1. Top 5 Combos by Test PnL")
    lines.append("")
    lines.append(_metric_table(results, top_n=5))
    lines.append("")
    lines.append("## 2. Overfitting Distribution")
    lines.append("")
    lines.append(_overfit_buckets(results))
    lines.append("")
    lines.append("## 3. Recommendation")
    lines.append("")
    if rec.params is None:
        lines.append(f"**No deployable parameters.** {rec.rationale}")
    else:
        lines.append(f"**Deploy:** `{_fmt_params(rec.params)}`")
        lines.append("")
        lines.append(f"- Train PnL: {_fmt_pnl(rec.train_pnl)}")
        lines.append(f"- Test PnL: {_fmt_pnl(rec.test_pnl)}")
        lines.append(f"- Overfit score: {_fmt_score(rec.overfit_score)}")
        lines.append(f"- Rationale: {rec.rationale}")
        lines.append("")
        lines.append("### Suggested .env updates")
        lines.append("")
        lines.append("```env")
        lines.append(f"SWEEP_SIGMA_MULT={rec.params['sigma_mult']}")
        lines.append(f"SWEEP_ABS_FLOOR_PTS={rec.params['abs_floor_pts']}")
        lines.append(f"SWEEP_COOLDOWN_SEC={rec.params['cooldown_sec']}")
        lines.append("```")
    lines.append("")
    lines.append("## Full Results")
    lines.append("")
    lines.append(_metric_table(results, top_n=n_total))
    lines.append("")
    return "\n".join(lines)


def write_report(
    results: List[WalkForwardResult],
    path: Path,
    train_date: str,
    test_date: str,
    symbol: str = "XAUUSD",
    broker_label: str = "",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = generate_report(
        results,
        train_date=train_date,
        test_date=test_date,
        symbol=symbol,
        broker_label=broker_label,
    )
    path.write_text(content, encoding="utf-8")
    return path
