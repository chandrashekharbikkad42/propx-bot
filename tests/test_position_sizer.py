"""Position sizing unit tests."""

from __future__ import annotations
import unittest

from risk.position_sizer import MAX_LOTS, MIN_LOTS, calculate_lot_size


class TestPositionSizer(unittest.TestCase):
    def test_half_pct_of_10k_30pt_sl(self) -> None:
        # 0.5% of $10K = $50. SL distance 30pt * $0.01 * 100 = $30 per lot.
        # 50 / 30 = 1.666... → 1.67 lots.
        lots = calculate_lot_size(10_000.0, 0.005, 30.0)
        self.assertAlmostEqual(lots, 1.67, places=2)

    def test_zero_sl_distance_returns_min(self) -> None:
        lots = calculate_lot_size(10_000.0, 0.005, 0.0)
        self.assertEqual(lots, MIN_LOTS)

    def test_negative_sl_distance_returns_min(self) -> None:
        lots = calculate_lot_size(10_000.0, 0.005, -5.0)
        self.assertEqual(lots, MIN_LOTS)

    def test_very_large_equity_clamped_to_max(self) -> None:
        lots = calculate_lot_size(10_000_000.0, 0.005, 30.0)
        self.assertEqual(lots, MAX_LOTS)

    def test_very_small_equity_clamped_to_min(self) -> None:
        lots = calculate_lot_size(10.0, 0.005, 30.0)
        self.assertEqual(lots, MIN_LOTS)

    def test_zero_equity_returns_min(self) -> None:
        self.assertEqual(calculate_lot_size(0.0, 0.005, 30.0), MIN_LOTS)


if __name__ == "__main__":
    unittest.main()
