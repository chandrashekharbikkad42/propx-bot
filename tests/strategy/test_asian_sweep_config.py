"""Config validation tests for `config.asian_sweep_config`.

Pinned to the verified V5 backtest (`multi_pair_backtest.py`, PF 2.27,
239 trades). Each constant has at least one test; per-pair fields are
parametrised over all 14 PAIRS (8 original V5 majors/crosses/metal + 6
extension pairs, two of which are index CFDs: HK50.cash, GER40.cash).
"""

from __future__ import annotations

import pytest

from config import asian_sweep_config as cfg
from config.asian_sweep_config import (
    ASIAN_END_IST, ASIAN_END_UTC_H, ASIAN_END_UTC_M,
    ASIAN_START_IST, ASIAN_START_UTC_H, ASIAN_START_UTC_M,
    LONDON_SWEEP_IST_END, LONDON_SWEEP_IST_START,
    LONDON_SWEEP_UTC_H_END, LONDON_SWEEP_UTC_H_START,
    MAX_DAILY_DD_PCT, MAX_TRADES_PER_DAY,
    NEWS_BLACKOUT_MIN,
    NY_END_IST, NY_SWEEP_IST_END, NY_SWEEP_IST_START,
    NY_SWEEP_UTC_H_END, NY_SWEEP_UTC_H_START,
    PAIR_CONFIG, PAIRS,
    PARTIAL_CLOSE_FRACTION,
    RISK_PCT, RR_TP1, RR_TP2,
    SESSION_FORCE_CLOSE_UTC_H, SKIP_MONDAY,
    TRAILING_STEP_R,
    WEAK_MONTH_RISK_PCT, WEAK_MONTHS,
    point_for, quality_for, risk_pct_for,
)


ALL_PAIRS = list(PAIRS)

# Symbol families — the 14-pair universe mixes plain FX/metal symbols with
# index CFDs that follow a "<UPPER>.cash" convention and carry point=0.01.
FX_PAIRS = [p for p in ALL_PAIRS if "." not in p]      # 6-char uppercase FX/metal
INDEX_PAIRS = [p for p in ALL_PAIRS if "." in p]       # e.g. HK50.cash, GER40.cash
FIVE_DP_PAIRS = [p for p in FX_PAIRS if p != "XAUUSD"] # 5-digit FX (point 0.00001)


# ---------------------------------------------------------------------------
# 1. PAIRS universe
# ---------------------------------------------------------------------------

class TestPairsUniverse:
    def test_pairs_is_tuple(self):
        assert isinstance(PAIRS, tuple)

    def test_pairs_count_fourteen(self):
        assert len(PAIRS) == 14

    def test_pairs_unique(self):
        assert len(set(PAIRS)) == len(PAIRS)

    def test_pairs_all_strings(self):
        assert all(isinstance(p, str) for p in PAIRS)

    def test_pairs_no_whitespace(self):
        assert all(p.strip() == p for p in PAIRS)

    def test_fx_pairs_all_uppercase(self):
        # FX/metal symbols are plain uppercase (e.g. EURUSD, XAUUSD).
        assert all(p.isupper() for p in FX_PAIRS)

    def test_fx_pairs_all_six_chars(self):
        assert all(len(p) == 6 for p in FX_PAIRS)

    def test_index_pairs_format(self):
        # Index CFDs follow "<UPPER>.cash" (e.g. HK50.cash, GER40.cash).
        for p in INDEX_PAIRS:
            base, _, suffix = p.partition(".")
            assert base.isupper() and suffix == "cash", p

    def test_index_pairs_present(self):
        # Two index CFDs are part of the extension universe.
        assert set(INDEX_PAIRS) == {"HK50.cash", "GER40.cash"}

    @pytest.mark.parametrize("expected", [
        "XAUUSD", "GBPUSD", "AUDUSD", "EURUSD",
        "USDCAD", "USDCHF", "AUDCHF", "AUDNZD",
        "NZDUSD", "EURNZD", "GBPCAD", "GBPAUD",
        "HK50.cash", "GER40.cash",
    ])
    def test_each_expected_pair_present(self, expected):
        assert expected in PAIRS

    def test_no_jpy_pairs_remaining(self):
        # V5 removed USDJPY/EURJPY/AUDJPY/NZDJPY (JPY conversion drag).
        assert not any("JPY" in p for p in PAIRS)

    def test_no_xagusd(self):
        # V5 removed XAGUSD (slot waste).
        assert "XAGUSD" not in PAIRS

    def test_pair_config_keys_match_pairs(self):
        assert set(PAIR_CONFIG.keys()) == set(PAIRS)

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_pair_config_has_entry(self, pair):
        assert pair in PAIR_CONFIG


# ---------------------------------------------------------------------------
# 2. Session window constants
# ---------------------------------------------------------------------------

class TestSessionWindows:
    def test_asian_start_utc_h(self):
        assert ASIAN_START_UTC_H == 19

    def test_asian_start_utc_m(self):
        assert ASIAN_START_UTC_M == 30

    def test_asian_end_utc_h(self):
        assert ASIAN_END_UTC_H == 0

    def test_asian_end_utc_m(self):
        assert ASIAN_END_UTC_M == 30

    def test_asian_start_ist_label(self):
        assert ASIAN_START_IST == "01:00"

    def test_asian_end_ist_label(self):
        assert ASIAN_END_IST == "06:00"

    def test_asian_ist_5h_30m_span(self):
        # 01:00 → 06:00 IST = 5 h.
        sh, sm = map(int, ASIAN_START_IST.split(":"))
        eh, em = map(int, ASIAN_END_IST.split(":"))
        assert (eh * 60 + em) - (sh * 60 + sm) == 5 * 60

    def test_london_sweep_start_utc(self):
        assert LONDON_SWEEP_UTC_H_START == 6

    def test_london_sweep_end_utc(self):
        assert LONDON_SWEEP_UTC_H_END == 10

    def test_london_sweep_window_5_bars(self):
        # bars 06..10 inclusive → 5 hours.
        assert LONDON_SWEEP_UTC_H_END - LONDON_SWEEP_UTC_H_START + 1 == 5

    def test_london_sweep_ist_start_label(self):
        assert LONDON_SWEEP_IST_START == "11:30"

    def test_london_sweep_ist_end_label(self):
        assert LONDON_SWEEP_IST_END == "16:00"

    def test_ny_sweep_start_utc(self):
        assert NY_SWEEP_UTC_H_START == 12

    def test_ny_sweep_end_utc(self):
        assert NY_SWEEP_UTC_H_END == 15

    def test_ny_sweep_window_4_bars(self):
        assert NY_SWEEP_UTC_H_END - NY_SWEEP_UTC_H_START + 1 == 4

    def test_ny_sweep_ist_start_label(self):
        assert NY_SWEEP_IST_START == "17:30"

    def test_ny_sweep_ist_end_label(self):
        assert NY_SWEEP_IST_END == "21:00"

    def test_session_force_close_utc(self):
        assert SESSION_FORCE_CLOSE_UTC_H == 16

    def test_ny_end_ist_label(self):
        assert NY_END_IST == "21:30"

    def test_london_does_not_overlap_ny(self):
        # London ends at 10 UTC, NY starts at 12 UTC.
        assert LONDON_SWEEP_UTC_H_END < NY_SWEEP_UTC_H_START

    def test_ny_ends_before_force_close(self):
        assert NY_SWEEP_UTC_H_END < SESSION_FORCE_CLOSE_UTC_H

    def test_utc_ist_offset_consistent_london(self):
        # 06:00 UTC ≈ 11:30 IST (UTC + 5:30).
        ist_h, ist_m = map(int, LONDON_SWEEP_IST_START.split(":"))
        utc_total = LONDON_SWEEP_UTC_H_START * 60
        ist_total = ist_h * 60 + ist_m
        assert ist_total - utc_total == 5 * 60 + 30

    def test_utc_ist_offset_consistent_ny(self):
        ist_h, ist_m = map(int, NY_SWEEP_IST_START.split(":"))
        utc_total = NY_SWEEP_UTC_H_START * 60
        ist_total = ist_h * 60 + ist_m
        assert ist_total - utc_total == 5 * 60 + 30

    def test_utc_ist_offset_consistent_asian_start(self):
        # 19:30 UTC = 01:00 IST next day (so +5:30 modulo 24).
        utc_total = ASIAN_START_UTC_H * 60 + ASIAN_START_UTC_M
        ist_h, ist_m = map(int, ASIAN_START_IST.split(":"))
        ist_total = ist_h * 60 + ist_m
        # Account for next-day rollover.
        assert (utc_total + 5 * 60 + 30) % (24 * 60) == ist_total

    def test_utc_ist_offset_consistent_asian_end(self):
        utc_total = ASIAN_END_UTC_H * 60 + ASIAN_END_UTC_M
        ist_h, ist_m = map(int, ASIAN_END_IST.split(":"))
        ist_total = ist_h * 60 + ist_m
        assert (utc_total + 5 * 60 + 30) % (24 * 60) == ist_total


# ---------------------------------------------------------------------------
# 3. Trade-management constants
# ---------------------------------------------------------------------------

class TestTradeManagementConstants:
    def test_max_trades_per_day(self):
        assert MAX_TRADES_PER_DAY == 2

    def test_max_trades_per_day_is_int(self):
        assert isinstance(MAX_TRADES_PER_DAY, int)

    def test_partial_close_fraction(self):
        assert PARTIAL_CLOSE_FRACTION == 0.50

    def test_partial_close_in_range(self):
        assert 0.0 < PARTIAL_CLOSE_FRACTION < 1.0

    def test_rr_tp1(self):
        assert RR_TP1 == 1.0

    def test_rr_tp2(self):
        assert RR_TP2 == 2.5

    def test_rr_tp2_greater_than_tp1(self):
        assert RR_TP2 > RR_TP1

    def test_trailing_step_r(self):
        assert TRAILING_STEP_R == 0.30

    def test_trailing_step_smaller_than_rr_tp1(self):
        assert TRAILING_STEP_R < RR_TP1

    def test_max_daily_dd_pct(self):
        assert MAX_DAILY_DD_PCT == 3.0

    def test_max_daily_dd_positive(self):
        assert MAX_DAILY_DD_PCT > 0

    def test_skip_monday(self):
        assert SKIP_MONDAY is True

    def test_skip_monday_is_bool(self):
        assert isinstance(SKIP_MONDAY, bool)

    def test_news_blackout_minutes(self):
        assert NEWS_BLACKOUT_MIN == 2

    def test_news_blackout_positive(self):
        assert NEWS_BLACKOUT_MIN > 0


# ---------------------------------------------------------------------------
# 4. Risk %
# ---------------------------------------------------------------------------

class TestRiskPct:
    def test_default_risk_pct(self):
        assert RISK_PCT["default"] == 0.8

    def test_xauusd_risk_override(self):
        assert PAIR_CONFIG["XAUUSD"]["risk_override"] == 0.5

    @pytest.mark.parametrize("pair", [p for p in ALL_PAIRS if p != "XAUUSD"])
    def test_non_xau_has_no_risk_override(self, pair):
        assert PAIR_CONFIG[pair]["risk_override"] is None

    def test_weak_months(self):
        assert WEAK_MONTHS == (11, 12, 1)

    def test_weak_month_risk(self):
        assert WEAK_MONTH_RISK_PCT == 0.3

    def test_weak_month_lower_than_default(self):
        assert WEAK_MONTH_RISK_PCT < RISK_PCT["default"]

    @pytest.mark.parametrize("m", [11, 12, 1])
    def test_risk_pct_for_weak_month(self, m):
        assert risk_pct_for("EURUSD", month=m) == WEAK_MONTH_RISK_PCT

    @pytest.mark.parametrize("m", [2, 3, 4, 5, 6, 7, 8, 9, 10])
    def test_risk_pct_for_non_weak_months(self, m):
        # EURUSD has no override → falls through to default.
        assert risk_pct_for("EURUSD", month=m) == RISK_PCT["default"]

    def test_risk_pct_xau_no_month(self):
        assert risk_pct_for("XAUUSD") == 0.5

    @pytest.mark.parametrize("m", [11, 12, 1])
    def test_risk_pct_xau_weak_month_wins(self, m):
        # Weak month dampener takes precedence over per-pair override.
        assert risk_pct_for("XAUUSD", month=m) == WEAK_MONTH_RISK_PCT

    def test_risk_pct_unknown_symbol_falls_to_default(self):
        assert risk_pct_for("ZZZZZZ") == RISK_PCT["default"]

    def test_risk_pct_no_month_arg_equiv_to_default(self):
        assert risk_pct_for("EURUSD") == RISK_PCT["default"]


# ---------------------------------------------------------------------------
# 5. Per-pair PAIR_CONFIG structure
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "point", "contract_size", "lot_max", "spread_pts", "sl_pts",
    "min_range_pts", "max_range_pts", "quality", "category", "jpy",
    "risk_override",
}

# Canonical key set — every PAIR_CONFIG entry must carry exactly these and
# nothing else. Frozen here so a copy-paste from `multi_pair_backtest.SYMBOLS`
# (which uses the legacy `min_r`/`max_r`/`cat` names) is rejected loudly.
CANONICAL_KEYS = frozenset(REQUIRED_KEYS)

# Legacy field names from the backtest dict. The detector reads the canonical
# names with a direct subscript, so any of these slipping into PAIR_CONFIG is a
# latent KeyError on the live scan path (and crashes the whole scan loop).
LEGACY_KEYS = frozenset({"min_r", "max_r", "cat", "spread"})


class TestPairConfigStructure:
    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_all_required_keys_present(self, pair):
        assert REQUIRED_KEYS.issubset(PAIR_CONFIG[pair].keys())

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_point_positive(self, pair):
        assert PAIR_CONFIG[pair]["point"] > 0

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_contract_size_positive(self, pair):
        assert PAIR_CONFIG[pair]["contract_size"] > 0

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_lot_max_positive(self, pair):
        assert PAIR_CONFIG[pair]["lot_max"] > 0

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_spread_pts_non_negative(self, pair):
        assert PAIR_CONFIG[pair]["spread_pts"] >= 0

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_sl_pts_positive(self, pair):
        assert PAIR_CONFIG[pair]["sl_pts"] > 0

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_min_range_pts_positive(self, pair):
        assert PAIR_CONFIG[pair]["min_range_pts"] > 0

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_max_range_pts_positive(self, pair):
        assert PAIR_CONFIG[pair]["max_range_pts"] > 0

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_max_range_gt_min_range(self, pair):
        cfg_p = PAIR_CONFIG[pair]
        assert cfg_p["max_range_pts"] > cfg_p["min_range_pts"]

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_quality_in_range_1_10(self, pair):
        q = PAIR_CONFIG[pair]["quality"]
        assert 1 <= q <= 10

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_category_string(self, pair):
        assert isinstance(PAIR_CONFIG[pair]["category"], str)

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_category_known(self, pair):
        assert PAIR_CONFIG[pair]["category"] in {"Metal", "Major", "Cross", "Index"}

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_jpy_flag_bool(self, pair):
        assert isinstance(PAIR_CONFIG[pair]["jpy"], bool)

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_jpy_flag_false_for_v5(self, pair):
        # V5 universe contains no JPY pairs.
        assert PAIR_CONFIG[pair]["jpy"] is False


# ---------------------------------------------------------------------------
# 6. Per-pair pinned values (from backtest)
# ---------------------------------------------------------------------------

EXPECTED_PAIR_VALUES = {
    # pair         point    spread sl_pts min_r  max_r quality category
    "XAUUSD":     (0.01,     45,    70,    100,   3000, 10,    "Metal"),
    "EURUSD":     (0.00001,   4,    80,    200,   2000,  9,    "Major"),
    "AUDUSD":     (0.00001,   3,    80,    150,   1800,  9,    "Major"),
    "GBPUSD":     (0.00001,   8,   100,    200,   2500,  8,    "Major"),
    "USDCAD":     (0.00001,   5,    80,    150,   2000,  7,    "Major"),
    "USDCHF":     (0.00001,   6,    80,    150,   2000,  7,    "Major"),
    "AUDCHF":     (0.00001,   8,    80,    150,   1800,  5,    "Cross"),
    "AUDNZD":     (0.00001,  12,    80,    150,   1800,  4,    "Cross"),
    # ── Extension pairs ────────────────────────────────────────────────
    "NZDUSD":     (0.00001,   7,    80,    150,   1800,  7,    "Major"),
    "EURNZD":     (0.00001,  12,    80,    150,   1800,  8,    "Cross"),
    "GBPCAD":     (0.00001,  12,    80,    150,   1800,  8,    "Cross"),
    "GBPAUD":     (0.00001,  12,    80,    150,   1800,  8,    "Cross"),
    "HK50.cash":  (0.01,     50,  2000,    100,  30000,  9,    "Index"),
    "GER40.cash": (0.01,     30,  2000,    100,  30000,  8,    "Index"),
}


class TestPairConfigPinnedValues:
    @pytest.mark.parametrize("pair,pt,_sp,_sl,_min,_max,_q,_c",
                             [(k, *v) for k, v in EXPECTED_PAIR_VALUES.items()])
    def test_point(self, pair, pt, _sp, _sl, _min, _max, _q, _c):
        assert PAIR_CONFIG[pair]["point"] == pt

    @pytest.mark.parametrize("pair,_pt,sp,_sl,_min,_max,_q,_c",
                             [(k, *v) for k, v in EXPECTED_PAIR_VALUES.items()])
    def test_spread(self, pair, _pt, sp, _sl, _min, _max, _q, _c):
        assert PAIR_CONFIG[pair]["spread_pts"] == sp

    @pytest.mark.parametrize("pair,_pt,_sp,sl,_min,_max,_q,_c",
                             [(k, *v) for k, v in EXPECTED_PAIR_VALUES.items()])
    def test_sl_pts(self, pair, _pt, _sp, sl, _min, _max, _q, _c):
        assert PAIR_CONFIG[pair]["sl_pts"] == sl

    @pytest.mark.parametrize("pair,_pt,_sp,_sl,mn,_max,_q,_c",
                             [(k, *v) for k, v in EXPECTED_PAIR_VALUES.items()])
    def test_min_range(self, pair, _pt, _sp, _sl, mn, _max, _q, _c):
        assert PAIR_CONFIG[pair]["min_range_pts"] == mn

    @pytest.mark.parametrize("pair,_pt,_sp,_sl,_min,mx,_q,_c",
                             [(k, *v) for k, v in EXPECTED_PAIR_VALUES.items()])
    def test_max_range(self, pair, _pt, _sp, _sl, _min, mx, _q, _c):
        assert PAIR_CONFIG[pair]["max_range_pts"] == mx

    @pytest.mark.parametrize("pair,_pt,_sp,_sl,_min,_max,q,_c",
                             [(k, *v) for k, v in EXPECTED_PAIR_VALUES.items()])
    def test_quality(self, pair, _pt, _sp, _sl, _min, _max, q, _c):
        assert PAIR_CONFIG[pair]["quality"] == q

    @pytest.mark.parametrize("pair,_pt,_sp,_sl,_min,_max,_q,cat",
                             [(k, *v) for k, v in EXPECTED_PAIR_VALUES.items()])
    def test_category(self, pair, _pt, _sp, _sl, _min, _max, _q, cat):
        assert PAIR_CONFIG[pair]["category"] == cat


# ---------------------------------------------------------------------------
# 7. Helper functions
# ---------------------------------------------------------------------------

class TestPointFor:
    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_returns_float(self, pair):
        assert isinstance(point_for(pair), float)

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_matches_config(self, pair):
        assert point_for(pair) == PAIR_CONFIG[pair]["point"]

    def test_xauusd_point(self):
        assert point_for("XAUUSD") == 0.01

    @pytest.mark.parametrize("pair", FIVE_DP_PAIRS)
    def test_5dp_pairs(self, pair):
        assert point_for(pair) == 0.00001

    @pytest.mark.parametrize("pair", INDEX_PAIRS)
    def test_index_pairs_point_001(self, pair):
        # Index CFDs quote in 0.01 increments like XAUUSD, not 5-digit FX.
        assert point_for(pair) == 0.01

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            point_for("ZZZZZZ")


class TestQualityFor:
    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_returns_int(self, pair):
        assert isinstance(quality_for(pair), int)

    @pytest.mark.parametrize("pair,_pt,_sp,_sl,_min,_max,q,_c",
                             [(k, *v) for k, v in EXPECTED_PAIR_VALUES.items()])
    def test_quality_matches_pinned(self, pair, _pt, _sp, _sl, _min, _max, q, _c):
        assert quality_for(pair) == q

    def test_unknown_pair_zero(self):
        assert quality_for("ZZZZZZ") == 0

    def test_unknown_pair_does_not_raise(self):
        quality_for("ANYTHING")  # no exception

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_in_valid_range(self, pair):
        q = quality_for(pair)
        assert 1 <= q <= 10


class TestImmutability:
    def test_pair_config_immutable_top(self):
        with pytest.raises(TypeError):
            PAIR_CONFIG["NEWPAIR"] = {}

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_pair_config_per_pair_immutable(self, pair):
        with pytest.raises(TypeError):
            PAIR_CONFIG[pair]["spread_pts"] = 999

    def test_risk_pct_immutable(self):
        with pytest.raises(TypeError):
            RISK_PCT["default"] = 5.0


class TestQualityRanking:
    def test_xauusd_top_quality(self):
        assert quality_for("XAUUSD") == max(
            quality_for(p) for p in ALL_PAIRS
        )

    def test_audnzd_lowest_quality(self):
        assert quality_for("AUDNZD") == min(
            quality_for(p) for p in ALL_PAIRS
        )

    def test_xauusd_strictly_greater_than_eurusd(self):
        assert quality_for("XAUUSD") > quality_for("EURUSD")

    def test_eurusd_audusd_same_quality(self):
        assert quality_for("EURUSD") == quality_for("AUDUSD")

    def test_usdcad_usdchf_same_quality(self):
        assert quality_for("USDCAD") == quality_for("USDCHF")

    def test_metal_outranks_every_other_pair(self):
        # XAUUSD (10) is the single highest-quality symbol in the universe.
        metals = [p for p in ALL_PAIRS if PAIR_CONFIG[p]["category"] == "Metal"]
        others = [p for p in ALL_PAIRS if PAIR_CONFIG[p]["category"] != "Metal"]
        assert min(quality_for(m) for m in metals) > max(
            quality_for(o) for o in others
        )

    def test_quality_within_1_10_band(self):
        # NOTE: the old "all majors outrank all crosses" invariant no longer
        # holds — extension crosses (EURNZD/GBPCAD/GBPAUD, q=8) outrank some
        # majors (USDCAD/USDCHF/NZDUSD, q=7). Quality is a manual per-pair
        # preference score, not a function of category in the 14-pair universe.
        assert all(1 <= quality_for(p) <= 10 for p in ALL_PAIRS)

    def test_metals_top_quality(self):
        metals = [p for p in ALL_PAIRS if PAIR_CONFIG[p]["category"] == "Metal"]
        assert all(quality_for(m) == 10 for m in metals)


# ---------------------------------------------------------------------------
# 8. Cross-constant invariants
# ---------------------------------------------------------------------------

class TestCrossInvariants:
    def test_force_close_after_ny_end(self):
        # NY sweep window ends at hour 15 (bars 12..15). Force-close at 16.
        assert SESSION_FORCE_CLOSE_UTC_H > NY_SWEEP_UTC_H_END

    def test_asian_window_does_not_overlap_london(self):
        # Asian ends at 00:30 UTC; London starts at 06:00 UTC.
        assert ASIAN_END_UTC_H < LONDON_SWEEP_UTC_H_START

    @pytest.mark.parametrize("pair", ALL_PAIRS)
    def test_spread_strictly_less_than_sl_buffer(self, pair):
        # Entry offset must be smaller than the SL buffer; otherwise risk
        # collapses below the _MIN_RISK_PT_MULT * point guard for all sweeps.
        cfg_p = PAIR_CONFIG[pair]
        assert cfg_p["spread_pts"] < cfg_p["sl_pts"]


# ---------------------------------------------------------------------------
# 9. Schema consistency — every PAIR_CONFIG entry carries the canonical keys
#
# Regression guard for the min_r/max_r/cat drift: six pairs (NZDUSD, EURNZD,
# GBPCAD, GBPAUD, HK50.cash, GER40.cash) were copy-pasted from
# multi_pair_backtest.SYMBOLS with the backtest's legacy key names. The
# detector reads cfg["min_range_pts"]/["max_range_pts"] with a direct
# subscript, so those entries raised KeyError and — since scanner.scan() wraps
# detect() in no try/except — crashed the entire scan during London/NY windows.
#
# These tests iterate PAIR_CONFIG directly (not the stale 8-pair ALL_PAIRS) so
# they cover every pair actually shipped, regardless of the PAIRS count.
# ---------------------------------------------------------------------------

class TestSchemaConsistency:
    @pytest.mark.parametrize("pair", list(PAIR_CONFIG.keys()))
    def test_keys_exactly_canonical(self, pair):
        # Exact match: catches both missing canonical keys AND stray extras.
        assert set(PAIR_CONFIG[pair].keys()) == set(CANONICAL_KEYS), (
            f"{pair} keys {set(PAIR_CONFIG[pair].keys())} "
            f"!= canonical {set(CANONICAL_KEYS)}"
        )

    @pytest.mark.parametrize("pair", list(PAIR_CONFIG.keys()))
    def test_no_legacy_backtest_keys(self, pair):
        # Explicit, readable failure if a backtest-style name slips back in.
        leaked = set(PAIR_CONFIG[pair].keys()) & set(LEGACY_KEYS)
        assert not leaked, f"{pair} carries legacy backtest keys {leaked}"

    def test_every_pair_has_identical_key_set(self):
        # No drift between entries — all share one schema.
        key_sets = {frozenset(v.keys()) for v in PAIR_CONFIG.values()}
        assert len(key_sets) == 1, f"PAIR_CONFIG entries disagree on keys: {key_sets}"

    @pytest.mark.parametrize("pair", list(PAIR_CONFIG.keys()))
    def test_range_filter_readable_by_detector(self, pair):
        # Direct subscript mirrors AsianSweepDetector.detect — must not raise.
        cfg_p = PAIR_CONFIG[pair]
        assert float(cfg_p["min_range_pts"]) > 0
        assert float(cfg_p["max_range_pts"]) > float(cfg_p["min_range_pts"])
