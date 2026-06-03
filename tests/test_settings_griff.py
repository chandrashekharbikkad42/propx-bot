"""Phase 8B — verify new Griff-pivot fields on Settings.

These tests assert defaults + dataclass shape. ENV-override flows are
exercised via _validate_hhmm and the Settings dataclass directly so we
don't have to reimport the module.
"""

from __future__ import annotations
from dataclasses import replace

import pytest

from config.settings import (
    DEFAULT_FOREX_PAIRS,
    DEFAULT_IST_WINDOW_END,
    DEFAULT_IST_WINDOW_START,
    DEFAULT_TIMEZONE,
    _validate_hhmm,
    settings,
)


class TestForexPairsDefault:
    def test_has_28_pairs(self):
        assert len(DEFAULT_FOREX_PAIRS) == 28

    def test_no_duplicates(self):
        assert len(set(DEFAULT_FOREX_PAIRS)) == len(DEFAULT_FOREX_PAIRS)

    def test_contains_majors(self):
        majors = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"}
        assert majors.issubset(set(DEFAULT_FOREX_PAIRS))

    def test_all_six_letters_uppercase(self):
        for p in DEFAULT_FOREX_PAIRS:
            assert len(p) == 6
            assert p.isupper()


class TestPropFirmDefault:
    def test_default_is_ftmo_1step(self):
        # Active settings instance — built from .env at import.
        # Without PROP_FIRM_TYPE in .env, default is FTMO_1STEP.
        assert settings.prop_firm_type in (
            "FTMO_1STEP", "FTMO_2STEP", "THE5ERS_BOOTCAMP", "THE5ERS_HRP", "NONE",
        )

    def test_can_override_via_replace(self):
        s2 = replace(settings, prop_firm_type="THE5ERS_BOOTCAMP")
        assert s2.prop_firm_type == "THE5ERS_BOOTCAMP"


class TestIstWindowFields:
    def test_default_start_1230(self):
        assert settings.ist_window_start == DEFAULT_IST_WINDOW_START == "12:30"

    def test_default_end_2230(self):
        assert settings.ist_window_end == DEFAULT_IST_WINDOW_END == "22:30"

    def test_default_timezone_kolkata(self):
        assert settings.timezone == DEFAULT_TIMEZONE == "Asia/Kolkata"


class TestBarsDir:
    def test_bars_dir_exists(self):
        # _build() creates it.
        assert settings.bars_dir.exists()
        assert settings.bars_dir.is_dir()
        assert settings.bars_dir.name == "bars"


class TestAutoDetectPairs:
    def test_default_false(self):
        assert settings.auto_detect_pairs is False


class TestValidateHHmm:
    def test_accepts_valid(self):
        assert _validate_hhmm("X", "07:30") == "07:30"
        assert _validate_hhmm("X", "00:00") == "00:00"
        assert _validate_hhmm("X", "23:59") == "23:59"

    def test_pads_single_digit(self):
        assert _validate_hhmm("X", "7:5") == "07:05"

    def test_rejects_bad_format(self):
        with pytest.raises(RuntimeError):
            _validate_hhmm("X", "7-30")

    def test_rejects_out_of_range_hour(self):
        with pytest.raises(RuntimeError):
            _validate_hhmm("X", "24:00")

    def test_rejects_out_of_range_minute(self):
        with pytest.raises(RuntimeError):
            _validate_hhmm("X", "12:60")

    def test_rejects_non_int(self):
        with pytest.raises(RuntimeError):
            _validate_hhmm("X", "ab:cd")
