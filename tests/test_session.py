"""Session boundary tests. UTC only — no DST handling needed."""

from __future__ import annotations
import unittest
from datetime import datetime, timezone

from utils.session import SessionLabel, session_for, session_for_msc


class TestSessionBoundaries(unittest.TestCase):
    def _dt(self, h: int, m: int = 0, s: int = 0) -> datetime:
        return datetime(2026, 5, 12, h, m, s, tzinfo=timezone.utc)

    def test_06_59_is_asian(self):
        self.assertEqual(session_for(self._dt(6, 59)), SessionLabel.ASIAN)

    def test_07_00_is_london(self):
        self.assertEqual(session_for(self._dt(7, 0)), SessionLabel.LONDON)

    def test_11_59_is_london(self):
        self.assertEqual(session_for(self._dt(11, 59)), SessionLabel.LONDON)

    def test_12_00_is_overlap(self):
        self.assertEqual(
            session_for(self._dt(12, 0)), SessionLabel.LONDON_NY_OVERLAP
        )

    def test_15_59_is_overlap(self):
        self.assertEqual(
            session_for(self._dt(15, 59)), SessionLabel.LONDON_NY_OVERLAP
        )

    def test_16_00_is_ny(self):
        self.assertEqual(session_for(self._dt(16, 0)), SessionLabel.NY)

    def test_20_59_is_ny(self):
        self.assertEqual(session_for(self._dt(20, 59)), SessionLabel.NY)

    def test_21_00_is_off(self):
        self.assertEqual(session_for(self._dt(21, 0)), SessionLabel.OFF)

    def test_00_00_is_asian(self):
        self.assertEqual(session_for(self._dt(0, 0)), SessionLabel.ASIAN)

    def test_23_59_is_off(self):
        self.assertEqual(session_for(self._dt(23, 59, 59)), SessionLabel.OFF)


class TestSessionFromMsc(unittest.TestCase):
    def test_msc_helper_matches_dt_helper(self):
        dt = datetime(2026, 5, 12, 9, 30, tzinfo=timezone.utc)
        msc = int(dt.timestamp() * 1000)
        self.assertEqual(session_for_msc(msc), session_for(dt))
        self.assertEqual(session_for_msc(msc), SessionLabel.LONDON)


class TestNoDST(unittest.TestCase):
    """Same UTC instant always maps to same session, regardless of any caller TZ."""

    def test_dst_summer_does_not_shift(self):
        # July (BST in London) — 07:00 UTC is still LONDON, no DST adjustment.
        self.assertEqual(
            session_for(datetime(2026, 7, 15, 7, 0, tzinfo=timezone.utc)),
            SessionLabel.LONDON,
        )

    def test_dst_winter_does_not_shift(self):
        # January (GMT in London) — same mapping.
        self.assertEqual(
            session_for(datetime(2026, 1, 15, 7, 0, tzinfo=timezone.utc)),
            SessionLabel.LONDON,
        )


if __name__ == "__main__":
    unittest.main()
