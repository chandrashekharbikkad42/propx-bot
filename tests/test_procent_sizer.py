"""ProCent (cents-denominated) account sizing tests (Phase 5)."""

from __future__ import annotations
import unittest

from risk.position_sizer import calculate_lot_size, MAX_LOTS, MIN_LOTS


class TestProCentSizer(unittest.TestCase):
    def test_procent_divides_balance_by_100(self) -> None:
        # 100,000 cents = $1,000 real. 0.5% = $5 risk. SL=30pt costs $30/lot
        # → 5/30 = 0.166... → 0.17 lots.
        lots = calculate_lot_size(
            account_equity=100_000.0,
            risk_pct=0.005,
            sl_distance_pts=30.0,
            account_type="PROCENT",
        )
        self.assertAlmostEqual(lots, 0.17, places=2)

    def test_standard_unchanged_when_account_type_default(self) -> None:
        # Default ("STANDARD") path must match Phase 4 numbers.
        lots = calculate_lot_size(
            account_equity=10_000.0,
            risk_pct=0.005,
            sl_distance_pts=30.0,
        )
        self.assertAlmostEqual(lots, 1.67, places=2)

    def test_procent_explicit_standard_matches_default(self) -> None:
        a = calculate_lot_size(10_000.0, 0.005, 30.0)
        b = calculate_lot_size(10_000.0, 0.005, 30.0, account_type="STANDARD")
        self.assertEqual(a, b)

    def test_procent_clamps_to_min(self) -> None:
        # Tiny cent balance → lots round below MIN_LOTS → clamp.
        lots = calculate_lot_size(
            account_equity=10.0,
            risk_pct=0.005,
            sl_distance_pts=30.0,
            account_type="PROCENT",
        )
        self.assertEqual(lots, MIN_LOTS)

    def test_procent_clamps_to_max(self) -> None:
        # 1B cents = $10M, ample to blow past MAX_LOTS.
        lots = calculate_lot_size(
            account_equity=1_000_000_000.0,
            risk_pct=0.005,
            sl_distance_pts=30.0,
            account_type="PROCENT",
        )
        self.assertEqual(lots, MAX_LOTS)

    def test_zero_equity_returns_min_under_procent(self) -> None:
        lots = calculate_lot_size(0.0, 0.005, 30.0, account_type="PROCENT")
        self.assertEqual(lots, MIN_LOTS)


if __name__ == "__main__":
    unittest.main()
