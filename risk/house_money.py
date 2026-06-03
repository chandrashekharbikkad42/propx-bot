"""House-money risk allocator (Phase 8C — Griff protocol).

Per IST trading day the bot takes up to 2 trades. Risk per trade depends on:
  - signal grade (A/B/C — C never trades)
  - which trade of the day this is (1st vs 2nd)
  - if 2nd, whether the 1st was a winner ("house money") or loser (defensive)

Math:
  Trade 1 (any grade A/B): base_risk_pct[grade] × equity.
  Trade 2 (any grade A/B), with todays_pnl_usd from Trade 1:
    if todays_pnl_usd > 0:  # house money mode
        # Reinvest half the prior win on top of base risk, capped at 2× base.
        risk_pct = base_pct + (todays_pnl_usd / equity) * HOUSE_MONEY_FRACTION
        cap   = base_pct * MAX_HOUSE_MONEY_MULT
        risk_pct = min(risk_pct, cap)
    elif todays_pnl_usd < 0:  # defensive
        risk_pct = base_pct * DEFENSIVE_MULT
    else:  # exactly flat — treat as defensive (neither win nor loss to leverage)
        risk_pct = base_pct * DEFENSIVE_MULT

  Trade 3+ : not supported (compliance caps trades_today at 2). Caller asks
  for trade_number_today >= 3 → raises.

Hinglish: pehla trade jeeta = doosre me thoda extra risk lo (jeeti hui kamai
ka half re-deploy), haara = doosre me kam risk (defensive). C-grade signals
ko function call hi mat karo — they're never traded.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Optional

from strategy.patterns.base import Grade


# ---------------------------------------------------------------------------
# Tunables — per Griff spec. Centralised so future calibration is one place.
# ---------------------------------------------------------------------------

DEFAULT_BASE_RISK_PCT: Mapping[Grade, float] = {
    Grade.A: 1.0,
    Grade.B: 0.5,
    Grade.C: 0.0,    # never traded — function raises if asked
}

# Fraction of yesterday-style prior win to redeploy in Trade 2.
HOUSE_MONEY_FRACTION: float = 0.5

# Trade 2 never risks more than this multiple of base, regardless of win size.
MAX_HOUSE_MONEY_MULT: float = 2.0

# Trade 2 after a loss: defensive scale-down vs base.
DEFENSIVE_MULT: float = 0.5


@dataclass(frozen=True)
class RiskAllocation:
    """Computed sizing decision. Pure value object."""
    grade: Grade
    trade_number_today: int
    base_risk_pct: float
    final_risk_pct: float
    mode: str        # "STANDARD", "HOUSE_MONEY", "DEFENSIVE", or "SKIP"
    rationale: str   # short human-readable explanation for logs


class HouseMoneyManager:
    def __init__(
        self,
        base_risk_pct: Optional[Mapping[Grade, float]] = None,
        house_money_fraction: float = HOUSE_MONEY_FRACTION,
        max_house_money_mult: float = MAX_HOUSE_MONEY_MULT,
        defensive_mult: float = DEFENSIVE_MULT,
    ) -> None:
        if house_money_fraction < 0:
            raise ValueError("house_money_fraction must be >= 0")
        if max_house_money_mult < 1.0:
            raise ValueError("max_house_money_mult must be >= 1.0")
        if not (0.0 <= defensive_mult <= 1.0):
            raise ValueError("defensive_mult must be in [0,1]")
        self._base = dict(base_risk_pct or DEFAULT_BASE_RISK_PCT)
        self._fraction = house_money_fraction
        self._cap_mult = max_house_money_mult
        self._defensive_mult = defensive_mult

    # ----------------------------------------------------------- public API

    def base_pct_for(self, grade: Grade) -> float:
        """Base risk % for a grade BEFORE house-money / defensive adjustments."""
        return self._base.get(grade, 0.0)

    def calc_trade_risk(
        self,
        grade: Grade,
        equity: float,
        todays_pnl_usd: float,
        trade_number_today: int,
    ) -> RiskAllocation:
        """Return the risk allocation (as a %) for this trade.

        Raises ValueError if `grade is Grade.C` (those are filtered earlier
        by the scanner) or if trade_number_today not in {1, 2}.
        """
        if grade == Grade.C:
            raise ValueError("C-grade signals must not be sized (scanner skips them)")
        if equity <= 0:
            raise ValueError(f"equity must be > 0, got {equity}")
        if trade_number_today not in (1, 2):
            raise ValueError(
                f"trade_number_today must be 1 or 2, got {trade_number_today}"
            )

        base = self._base.get(grade, 0.0)
        if base <= 0:
            return RiskAllocation(
                grade=grade, trade_number_today=trade_number_today,
                base_risk_pct=0.0, final_risk_pct=0.0, mode="SKIP",
                rationale=f"grade {grade.value} has zero base risk",
            )

        if trade_number_today == 1:
            return RiskAllocation(
                grade=grade, trade_number_today=1,
                base_risk_pct=base, final_risk_pct=base, mode="STANDARD",
                rationale=f"trade 1 of day, base risk for grade {grade.value}",
            )

        # Trade 2 — branch on prior PnL.
        if todays_pnl_usd > 0.0:
            extra = (todays_pnl_usd / equity) * 100.0 * self._fraction
            raw = base + extra
            cap = base * self._cap_mult
            final = min(raw, cap)
            return RiskAllocation(
                grade=grade, trade_number_today=2,
                base_risk_pct=base, final_risk_pct=final, mode="HOUSE_MONEY",
                rationale=(
                    f"trade 2 after +${todays_pnl_usd:.2f} win: "
                    f"base {base:.2f}% + {extra:.2f}% (cap {cap:.2f}%) "
                    f"= {final:.2f}%"
                ),
            )
        # Loss or exactly flat → defensive.
        defensive = base * self._defensive_mult
        return RiskAllocation(
            grade=grade, trade_number_today=2,
            base_risk_pct=base, final_risk_pct=defensive, mode="DEFENSIVE",
            rationale=(
                f"trade 2 after ${todays_pnl_usd:.2f}: defensive "
                f"({self._defensive_mult:g}× base) = {defensive:.2f}%"
            ),
        )

    def daily_summary(
        self, equity: float, grade_for_both: Grade = Grade.A,
    ) -> dict:
        """Hypothetical worst / best / expected outcomes for one full day.

        Useful for the dashboard. Assumes both trades are `grade_for_both`.
        Worst = trade 1 loss (-base%) + trade 2 defensive loss.
        Best  = trade 1 win (+R:R × base%) + trade 2 house-money win.
        We don't know R:R here, so we use a 1:2 assumption for illustration.
        """
        base = self._base.get(grade_for_both, 0.0)
        if base <= 0 or equity <= 0:
            return {"worst_pct": 0.0, "best_pct": 0.0, "base_pct": base}
        rr_assumption = 2.0  # 1:2 assumption — easy to override later
        win_pct = base * rr_assumption
        worst = -base + (-base * self._defensive_mult)  # both losses, T2 defensive
        # Best case: T1 wins (+win_pct% × equity), T2 sized larger by house money.
        t1_profit_usd = (win_pct / 100.0) * equity
        t2_alloc = self.calc_trade_risk(grade_for_both, equity, t1_profit_usd, 2)
        best = win_pct + (t2_alloc.final_risk_pct * rr_assumption)
        return {
            "worst_pct": worst,
            "best_pct": best,
            "base_pct": base,
            "trade_2_house_money_pct": t2_alloc.final_risk_pct,
        }
