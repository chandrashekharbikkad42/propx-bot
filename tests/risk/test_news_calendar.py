"""StaticNewsCalendar — pre-trade news blackout gate.

Spec under test (`data/news_calendar.py`):
  - HIGH-impact events block; MEDIUM/LOW are ignored.
  - `is_blackout` uses the default ±2 min window
    (DEFAULT_BLACKOUT_WINDOW_MIN).
  - `is_news_blackout(symbol, time_msc, window_min=N)` blocks if
    `|evt.time_msc - time_msc| <= N*60_000` AND `evt.currency in symbol`.
  - Currency match is uppercase substring on the symbol code.
  - `upcoming_events(after_msc, limit)` returns events strictly after the
    timestamp, ascending, capped at `limit`.

Tests aim for ~200 cases:
  - Per-currency / per-symbol matching matrix (USD, EUR, GBP, JPY, CHF,
    CAD, AUD, NZD + XAU cross with USD).
  - Boundary timing at exactly ±N minutes for window sizes 0, 1, 2, 5, 10.
  - Impact filter: HIGH passes, MED/LOW are no-ops.
  - Overlapping events (two events at the same time, different currencies).
  - upcoming_events — strict-after ordering, limit, sort order.
  - Constructor: empty list, single event, default fallback.
  - Validation: negative window_min, negative limit.
  - DEFAULT_CALENDAR is a populated StaticNewsCalendar.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from data.news_calendar import (
    DEFAULT_BLACKOUT_WINDOW_MIN,
    DEFAULT_CALENDAR,
    DEFAULT_HIGH_IMPACT_EVENTS,
    NewsEvent,
    StaticNewsCalendar,
)

from tests.risk.fixtures.news_events import event, utc_ms, SPREAD


# ===========================================================================
# 0. Constants under test
# ===========================================================================

class TestModuleConstants:
    def test_default_window_is_two_minutes(self):
        assert DEFAULT_BLACKOUT_WINDOW_MIN == 2

    def test_default_calendar_is_static(self):
        assert isinstance(DEFAULT_CALENDAR, StaticNewsCalendar)

    def test_default_calendar_has_events(self):
        assert len(DEFAULT_CALENDAR.events) > 0

    def test_default_calendar_uses_module_events(self):
        # The constructor should default to DEFAULT_HIGH_IMPACT_EVENTS.
        cal = StaticNewsCalendar()
        assert len(cal.events) == len(DEFAULT_HIGH_IMPACT_EVENTS)

    def test_default_events_all_high_impact(self):
        for evt in DEFAULT_HIGH_IMPACT_EVENTS:
            assert evt.impact == "HIGH"


# ===========================================================================
# 1. NewsEvent dataclass
# ===========================================================================

class TestNewsEvent:
    def test_construct_default_impact(self):
        evt = NewsEvent(time_msc=0, currency="USD", title="NFP")
        assert evt.impact == "HIGH"

    def test_explicit_impact(self):
        evt = NewsEvent(time_msc=1, currency="USD", title="t", impact="LOW")
        assert evt.impact == "LOW"

    def test_frozen(self):
        evt = NewsEvent(time_msc=0, currency="USD", title="NFP")
        with pytest.raises(Exception):
            evt.title = "X"  # type: ignore[misc]

    @pytest.mark.parametrize("ccy", ["USD", "EUR", "GBP", "JPY",
                                     "CHF", "CAD", "AUD", "NZD"])
    def test_currency_roundtrips(self, ccy):
        evt = NewsEvent(time_msc=0, currency=ccy, title="t")
        assert evt.currency == ccy

    def test_equality_by_value(self):
        a = NewsEvent(time_msc=10, currency="USD", title="NFP")
        b = NewsEvent(time_msc=10, currency="USD", title="NFP")
        assert a == b


# ===========================================================================
# 2. Constructor variants
# ===========================================================================

class TestConstructor:
    def test_no_arg_uses_default(self):
        cal = StaticNewsCalendar()
        assert len(cal.events) == len(DEFAULT_HIGH_IMPACT_EVENTS)

    def test_none_uses_default(self):
        cal = StaticNewsCalendar(events=None)
        assert len(cal.events) == len(DEFAULT_HIGH_IMPACT_EVENTS)

    def test_empty_list(self):
        cal = StaticNewsCalendar(events=[])
        assert cal.events == ()

    def test_single_event(self):
        e = event(2026, 5, 14, 12, 30, "USD", "CPI")
        cal = StaticNewsCalendar(events=[e])
        assert cal.events == (e,)

    def test_events_are_sorted_ascending(self):
        unsorted_events = [
            event(2026, 6, 5,  12, 30, "USD", "NFP"),
            event(2026, 5, 14, 12, 30, "USD", "CPI"),
            event(2026, 5, 28, 18, 0,  "USD", "FOMC"),
        ]
        cal = StaticNewsCalendar(events=unsorted_events)
        times = [e.time_msc for e in cal.events]
        assert times == sorted(times)

    def test_events_property_is_tuple(self):
        cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30)])
        assert isinstance(cal.events, tuple)

    def test_iterable_input_consumed(self):
        # generator is OK
        cal = StaticNewsCalendar(events=(event(2026, 5, 14, 12, 30, "USD")
                                         for _ in range(1)))
        assert len(cal.events) == 1


# ===========================================================================
# 3. is_blackout — uses default ±2 min window
# ===========================================================================

USD_EVENT_MSC = utc_ms(2026, 5, 14, 12, 30)
USD_CAL = StaticNewsCalendar(events=[
    event(2026, 5, 14, 12, 30, "USD", "CPI")
])


@pytest.mark.parametrize("delta_sec", [
    -120, -119, -60, -30, -1, 0, 1, 30, 60, 119, 120,
])
def test_is_blackout_within_two_minutes(delta_sec):
    """All deltas inside ±120s should block."""
    assert USD_CAL.is_blackout("EURUSD", USD_EVENT_MSC + delta_sec * 1000)


@pytest.mark.parametrize("delta_sec", [
    -121, -180, -3600, 121, 180, 3600,
])
def test_is_blackout_outside_two_minutes(delta_sec):
    """All deltas outside ±120s should NOT block."""
    assert not USD_CAL.is_blackout("EURUSD",
                                   USD_EVENT_MSC + delta_sec * 1000)


def test_is_blackout_currency_mismatch():
    # USD event, EURGBP pair (no USD) → no block.
    assert not USD_CAL.is_blackout("EURGBP", USD_EVENT_MSC)


def test_is_blackout_default_window_is_two():
    # Boundary at exactly +2 min should block (window is inclusive).
    assert USD_CAL.is_blackout("EURUSD", USD_EVENT_MSC + 2 * 60 * 1000)
    # At +2 min + 1 ms should not block.
    assert not USD_CAL.is_blackout(
        "EURUSD", USD_EVENT_MSC + 2 * 60 * 1000 + 1
    )


# ===========================================================================
# 4. Per-currency × per-symbol matrix
# ===========================================================================

CURRENCY_TO_BLOCKED_SYMBOLS = {
    "USD": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
            "AUDUSD", "NZDUSD", "XAUUSD"],
    "EUR": ["EURUSD", "EURGBP", "EURJPY", "EURCHF", "EURAUD"],
    "GBP": ["GBPUSD", "EURGBP", "GBPJPY", "GBPCHF"],
    "JPY": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"],
    "CHF": ["USDCHF", "EURCHF", "GBPCHF", "CHFJPY"],
    "CAD": ["USDCAD", "CADJPY", "EURCAD", "GBPCAD"],
    "AUD": ["AUDUSD", "AUDJPY", "EURAUD", "GBPAUD"],
    "NZD": ["NZDUSD", "NZDJPY", "EURNZD"],
}


@pytest.mark.parametrize("ccy,symbols", list(CURRENCY_TO_BLOCKED_SYMBOLS.items()))
def test_currency_blocks_its_pairs(ccy, symbols):
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, ccy, "X")])
    for sym in symbols:
        assert cal.is_blackout(sym, msc), f"{ccy} should block {sym}"


CURRENCY_TO_UNBLOCKED_SYMBOLS = {
    "USD": ["EURGBP", "EURJPY", "GBPJPY", "EURCHF",
            "GBPCHF", "CHFJPY", "AUDNZD"],
    "EUR": ["GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"],
    "GBP": ["USDJPY", "EURUSD", "AUDUSD", "XAUUSD"],
    "JPY": ["EURUSD", "GBPUSD", "AUDUSD", "XAUUSD"],
    "CHF": ["EURUSD", "GBPUSD", "AUDUSD", "XAUUSD"],
    "CAD": ["EURUSD", "GBPUSD", "AUDUSD", "XAUUSD"],
    "AUD": ["EURUSD", "GBPUSD", "USDJPY", "EURGBP"],
    "NZD": ["EURUSD", "GBPUSD", "USDJPY", "EURGBP"],
}


@pytest.mark.parametrize("ccy,symbols",
                         list(CURRENCY_TO_UNBLOCKED_SYMBOLS.items()))
def test_currency_does_not_block_unrelated_pairs(ccy, symbols):
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, ccy, "X")])
    for sym in symbols:
        assert not cal.is_blackout(sym, msc), f"{ccy} should NOT block {sym}"


# ===========================================================================
# 5. XAUUSD special — gold blocks on USD only
# ===========================================================================

class TestXauUsd:
    def test_xauusd_blocked_on_usd_event(self):
        msc = utc_ms(2026, 5, 14, 12, 30)
        cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
        assert cal.is_blackout("XAUUSD", msc)

    @pytest.mark.parametrize("ccy", ["EUR", "GBP", "JPY", "CHF",
                                     "CAD", "AUD", "NZD"])
    def test_xauusd_not_blocked_on_non_usd_event(self, ccy):
        msc = utc_ms(2026, 5, 14, 12, 30)
        cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, ccy)])
        assert not cal.is_blackout("XAUUSD", msc)


# ===========================================================================
# 6. Impact filter — only HIGH blocks
# ===========================================================================

@pytest.mark.parametrize("impact", ["MEDIUM", "LOW", "medium", "low",
                                    "med", "info", "", "HIGH-ISH"])
def test_non_high_impact_does_not_block(impact):
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[
        event(2026, 5, 14, 12, 30, "USD", "JOLTS", impact=impact)
    ])
    assert not cal.is_blackout("EURUSD", msc)


def test_high_impact_blocks():
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[
        event(2026, 5, 14, 12, 30, "USD", "NFP", impact="HIGH")
    ])
    assert cal.is_blackout("EURUSD", msc)


def test_high_impact_is_case_sensitive():
    """Implementation uses `evt.impact != 'HIGH'` — lowercase doesn't match."""
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[
        event(2026, 5, 14, 12, 30, "USD", "NFP", impact="high")
    ])
    assert not cal.is_blackout("EURUSD", msc)


# ===========================================================================
# 7. Window sizes 0..10
# ===========================================================================

@pytest.mark.parametrize("window_min", [0, 1, 2, 3, 5, 10, 30, 60])
def test_window_size_exact_boundary(window_min):
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
    # Exactly window_min minutes after — should block (inclusive).
    assert cal.is_news_blackout(
        "EURUSD", msc + window_min * 60 * 1000, window_min=window_min
    )


@pytest.mark.parametrize("window_min", [0, 1, 2, 3, 5, 10, 30, 60])
def test_window_size_exceeded(window_min):
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
    assert not cal.is_news_blackout(
        "EURUSD", msc + window_min * 60 * 1000 + 1, window_min=window_min
    )


def test_window_zero_blocks_only_exact_time():
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
    assert cal.is_news_blackout("EURUSD", msc, window_min=0)
    assert not cal.is_news_blackout("EURUSD", msc + 1, window_min=0)
    assert not cal.is_news_blackout("EURUSD", msc - 1, window_min=0)


def test_window_min_negative_raises():
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
    with pytest.raises(ValueError, match="window_min must be >= 0"):
        cal.is_news_blackout("EURUSD", 0, window_min=-1)


@pytest.mark.parametrize("window_min", [-1, -2, -100])
def test_window_min_negative_raises_param(window_min):
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
    with pytest.raises(ValueError):
        cal.is_news_blackout("EURUSD", 0, window_min=window_min)


# ===========================================================================
# 8. Symbol case — comparison uppercases the symbol
# ===========================================================================

@pytest.mark.parametrize("symbol", ["eurusd", "EURUSD", "EurUsd",
                                    "eUrUsD"])
def test_symbol_case_insensitive(symbol):
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
    assert cal.is_blackout(symbol, msc)


def test_symbol_with_suffix_blocks():
    # MT5 sometimes uses suffix-encoded symbols like 'EURUSD.r'.
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
    assert cal.is_blackout("EURUSD.r", msc)


def test_symbol_with_prefix_blocks():
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
    assert cal.is_blackout("FX_EURUSD", msc)


# ===========================================================================
# 9. Empty / no-event behaviour
# ===========================================================================

class TestNoEvents:
    def test_empty_calendar_never_blocks(self):
        cal = StaticNewsCalendar(events=[])
        for sym in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]:
            assert not cal.is_blackout(sym, utc_ms(2026, 5, 14, 12, 30))

    def test_empty_calendar_zero_window_does_not_block(self):
        cal = StaticNewsCalendar(events=[])
        assert not cal.is_news_blackout(
            "EURUSD", utc_ms(2026, 5, 14, 12, 30), window_min=0
        )

    def test_empty_calendar_huge_window_does_not_block(self):
        cal = StaticNewsCalendar(events=[])
        assert not cal.is_news_blackout(
            "EURUSD", utc_ms(2026, 5, 14, 12, 30), window_min=10_000
        )


# ===========================================================================
# 10. Overlapping / simultaneous events
# ===========================================================================

class TestOverlapping:
    def test_two_events_same_time_one_matches(self):
        msc = utc_ms(2026, 5, 14, 12, 30)
        cal = StaticNewsCalendar(events=[
            event(2026, 5, 14, 12, 30, "USD", "CPI"),
            event(2026, 5, 14, 12, 30, "EUR", "ECB"),
        ])
        assert cal.is_blackout("EURUSD", msc)   # both relevant
        assert cal.is_blackout("EURGBP", msc)   # EUR matches
        assert cal.is_blackout("USDJPY", msc)   # USD matches

    def test_overlapping_independent_currencies(self):
        msc = utc_ms(2026, 5, 14, 12, 30)
        cal = StaticNewsCalendar(events=[
            event(2026, 5, 14, 12, 30, "USD"),
            event(2026, 5, 14, 12, 31, "GBP"),
            event(2026, 5, 14, 12, 32, "JPY"),
        ])
        assert cal.is_blackout("USDJPY", msc)
        assert cal.is_blackout("GBPUSD", msc + 60_000)
        assert cal.is_blackout("USDJPY", msc + 2 * 60_000)

    def test_two_close_events_each_block(self):
        msc1 = utc_ms(2026, 5, 14, 12, 30)
        msc2 = utc_ms(2026, 5, 14, 12, 35)
        cal = StaticNewsCalendar(events=[
            event(2026, 5, 14, 12, 30, "USD"),
            event(2026, 5, 14, 12, 35, "USD"),
        ])
        assert cal.is_blackout("EURUSD", msc1)
        assert cal.is_blackout("EURUSD", msc2)
        # Right between them — outside ±2 min of both.
        midpoint = (msc1 + msc2) // 2  # 2.5 min after msc1
        assert not cal.is_blackout("EURUSD", midpoint)


# ===========================================================================
# 11. is_blackout vs is_news_blackout — same default
# ===========================================================================

class TestShorthand:
    def test_is_blackout_uses_default_two(self):
        msc = utc_ms(2026, 5, 14, 12, 30)
        cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
        # Two-minute boundary blocks via is_blackout
        assert cal.is_blackout("EURUSD", msc + 2 * 60_000)
        assert not cal.is_blackout("EURUSD", msc + 2 * 60_000 + 1)

    def test_is_blackout_equivalent_to_default_window(self):
        cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
        msc = utc_ms(2026, 5, 14, 12, 30)
        for delta in range(-200, 201, 30):
            t = msc + delta * 1000
            assert cal.is_blackout("EURUSD", t) == \
                cal.is_news_blackout("EURUSD", t,
                                     window_min=DEFAULT_BLACKOUT_WINDOW_MIN)


# ===========================================================================
# 12. upcoming_events
# ===========================================================================

class TestUpcomingEvents:
    def test_returns_strict_after(self):
        cal = StaticNewsCalendar(events=SPREAD)
        msc = utc_ms(2026, 5, 14, 12, 30)
        out = cal.upcoming_events(after_msc=msc, limit=5)
        for e in out:
            assert e.time_msc > msc

    def test_returns_in_ascending_order(self):
        cal = StaticNewsCalendar(events=SPREAD)
        out = cal.upcoming_events(after_msc=0, limit=100)
        times = [e.time_msc for e in out]
        assert times == sorted(times)

    def test_respects_limit(self):
        cal = StaticNewsCalendar(events=SPREAD)
        out = cal.upcoming_events(after_msc=0, limit=3)
        assert len(out) == 3

    def test_limit_larger_than_available(self):
        cal = StaticNewsCalendar(events=[
            event(2026, 5, 14, 12, 30, "USD"),
        ])
        out = cal.upcoming_events(after_msc=0, limit=10)
        assert len(out) == 1

    def test_limit_zero_returns_empty(self):
        cal = StaticNewsCalendar(events=SPREAD)
        out = cal.upcoming_events(after_msc=0, limit=0)
        assert out == []

    def test_limit_negative_raises(self):
        cal = StaticNewsCalendar(events=SPREAD)
        with pytest.raises(ValueError, match="limit must be >= 0"):
            cal.upcoming_events(after_msc=0, limit=-1)

    def test_no_events_after_returns_empty(self):
        cal = StaticNewsCalendar(events=[
            event(2026, 5, 14, 12, 30, "USD")
        ])
        out = cal.upcoming_events(after_msc=utc_ms(2030, 1, 1, 0, 0))
        assert out == []

    def test_after_equals_event_excludes(self):
        msc = utc_ms(2026, 5, 14, 12, 30)
        cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
        out = cal.upcoming_events(after_msc=msc)
        assert out == []

    def test_after_one_msc_before_event_includes(self):
        msc = utc_ms(2026, 5, 14, 12, 30)
        cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
        out = cal.upcoming_events(after_msc=msc - 1)
        assert len(out) == 1

    def test_empty_calendar_empty_upcoming(self):
        cal = StaticNewsCalendar(events=[])
        assert cal.upcoming_events(after_msc=0) == []

    def test_default_limit_is_ten(self):
        # Build 15 events; default limit caps at 10.
        events = [
            event(2026, 1, 1, h, 0, "USD") for h in range(15)
        ]
        cal = StaticNewsCalendar(events=events)
        out = cal.upcoming_events(after_msc=0)
        assert len(out) == 10

    def test_includes_non_high_impact(self):
        """Per protocol, upcoming returns ALL events — not just HIGH."""
        cal = StaticNewsCalendar(events=[
            event(2026, 5, 14, 12, 30, "USD", "X", impact="LOW"),
        ])
        out = cal.upcoming_events(after_msc=0)
        assert len(out) == 1


# ===========================================================================
# 13. Smoke per default-calendar entry
# ===========================================================================

@pytest.mark.parametrize("evt", list(DEFAULT_HIGH_IMPACT_EVENTS))
def test_default_event_blocks_its_currency_pair(evt):
    pair = f"EUR{evt.currency}" if evt.currency != "EUR" else "EURUSD"
    if pair == evt.currency * 2:
        pair = "USD" + evt.currency  # fallback
    cal = StaticNewsCalendar(events=[evt])
    assert cal.is_blackout(pair, evt.time_msc)


@pytest.mark.parametrize("evt", list(DEFAULT_HIGH_IMPACT_EVENTS))
def test_default_event_unrelated_pair_does_not_block(evt):
    # CHFJPY contains neither USD nor GBP nor EUR
    unrelated = "CHFJPY" if evt.currency not in ("CHF", "JPY") else "AUDNZD"
    cal = StaticNewsCalendar(events=[evt])
    assert not cal.is_blackout(unrelated, evt.time_msc)


# ===========================================================================
# 14. Cross-pair: both legs are inspected (substring)
# ===========================================================================

@pytest.mark.parametrize("symbol,blocks_on", [
    ("GBPUSD", ["GBP", "USD"]),
    ("EURUSD", ["EUR", "USD"]),
    ("USDJPY", ["USD", "JPY"]),
    ("EURGBP", ["EUR", "GBP"]),
    ("AUDJPY", ["AUD", "JPY"]),
    ("NZDCAD", ["NZD", "CAD"]),
    ("CHFJPY", ["CHF", "JPY"]),
    ("XAUUSD", ["XAU", "USD"]),
])
def test_cross_pair_blocks_on_each_leg(symbol, blocks_on):
    msc = utc_ms(2026, 5, 14, 12, 30)
    for ccy in blocks_on:
        cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, ccy)])
        assert cal.is_blackout(symbol, msc), f"{ccy} should block {symbol}"


# ===========================================================================
# 15. SPREAD fixture parametrisation
# ===========================================================================

class TestSpread:
    def test_spread_loads(self):
        cal = StaticNewsCalendar(events=SPREAD)
        assert len(cal.events) == len(SPREAD)

    def test_low_impact_in_spread_ignored(self):
        cal = StaticNewsCalendar(events=SPREAD)
        # JOLTS LOW at 2026-06-05 13:00 — well outside any HIGH window.
        msc = utc_ms(2026, 6, 5, 13, 0)
        assert not cal.is_blackout("EURUSD", msc)

    def test_medium_impact_in_spread_ignored(self):
        cal = StaticNewsCalendar(events=SPREAD)
        msc = utc_ms(2026, 6, 5, 13, 30)
        assert not cal.is_blackout("EURUSD", msc)

    def test_usd_cpi_blocks_eurusd(self):
        cal = StaticNewsCalendar(events=SPREAD)
        assert cal.is_blackout("EURUSD", utc_ms(2026, 5, 14, 12, 30))

    def test_eur_ecb_blocks_eurgbp(self):
        cal = StaticNewsCalendar(events=SPREAD)
        assert cal.is_blackout("EURGBP", utc_ms(2026, 5, 14, 12, 30))

    def test_jpy_boj_blocks_usdjpy(self):
        cal = StaticNewsCalendar(events=SPREAD)
        assert cal.is_blackout("USDJPY", utc_ms(2026, 6, 25, 9, 0))

    def test_cad_cpi_blocks_usdcad(self):
        cal = StaticNewsCalendar(events=SPREAD)
        assert cal.is_blackout("USDCAD", utc_ms(2026, 6, 26, 0, 30))


# ===========================================================================
# 16. Edge cases — symbols, time
# ===========================================================================

class TestEdgeCases:
    def test_empty_symbol_no_block(self):
        cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
        assert not cal.is_blackout("", utc_ms(2026, 5, 14, 12, 30))

    def test_symbol_shorter_than_currency_no_block(self):
        cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
        assert not cal.is_blackout("US", utc_ms(2026, 5, 14, 12, 30))

    def test_time_far_in_past_no_block(self):
        cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
        assert not cal.is_blackout("EURUSD", 0)

    def test_time_far_in_future_no_block(self):
        cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
        assert not cal.is_blackout(
            "EURUSD", utc_ms(2030, 12, 31, 23, 59)
        )

    def test_negative_time_supported(self):
        # Implementation does arithmetic on integers; negative msc is
        # exotic but should still work.
        cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
        assert not cal.is_blackout("EURUSD", -1)


# ===========================================================================
# 17. Currency 3-letter substring catches metals too
# ===========================================================================

@pytest.mark.parametrize("ccy,symbol", [
    ("XAU", "XAUUSD"),
    ("XAG", "XAGUSD"),
    ("BTC", "BTCUSD"),
])
def test_metal_or_crypto_substring_blocks(ccy, symbol):
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, ccy)])
    assert cal.is_blackout(symbol, msc)


# ===========================================================================
# 18. Hypothesis — properties of the blackout function
# ===========================================================================

@settings(max_examples=200, deadline=None)
@given(
    delta_sec=st.integers(min_value=-3600, max_value=3600),
    window_min=st.integers(min_value=0, max_value=60),
)
def test_blackout_window_is_symmetric(delta_sec, window_min):
    """is_news_blackout is symmetric around the event time."""
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
    pos = cal.is_news_blackout(
        "EURUSD", msc + delta_sec * 1000, window_min=window_min
    )
    neg = cal.is_news_blackout(
        "EURUSD", msc - delta_sec * 1000, window_min=window_min
    )
    assert pos == neg


@settings(max_examples=200, deadline=None)
@given(
    delta_sec=st.integers(min_value=-3600, max_value=3600),
    window_min=st.integers(min_value=0, max_value=60),
)
def test_blackout_matches_inclusive_window(delta_sec, window_min):
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
    expected = abs(delta_sec) * 1000 <= window_min * 60 * 1000
    actual = cal.is_news_blackout(
        "EURUSD", msc + delta_sec * 1000, window_min=window_min
    )
    assert actual is expected


@settings(max_examples=100, deadline=None)
@given(window_min=st.integers(min_value=0, max_value=120))
def test_currency_mismatch_never_blocks(window_min):
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
    for delta in range(-window_min * 60 - 60,
                       window_min * 60 + 60,
                       max(1, window_min) * 30):
        assert not cal.is_news_blackout(
            "EURGBP",
            utc_ms(2026, 5, 14, 12, 30) + delta * 1000,
            window_min=window_min,
        )


@settings(max_examples=100, deadline=None)
@given(limit=st.integers(min_value=0, max_value=100))
def test_upcoming_respects_arbitrary_limit(limit):
    cal = StaticNewsCalendar(events=SPREAD)
    out = cal.upcoming_events(after_msc=0, limit=limit)
    assert len(out) <= limit


# ===========================================================================
# 19. Multiple windows per event simultaneously
# ===========================================================================

@pytest.mark.parametrize("window_min,delta_min,expected", [
    (2,  1,  True),
    (2,  2,  True),
    (2,  3,  False),
    (5,  4,  True),
    (5,  5,  True),
    (5,  6,  False),
    (10, 9,  True),
    (10, 10, True),
    (10, 11, False),
    (30, 29, True),
    (30, 30, True),
    (30, 31, False),
])
def test_window_minutes_matrix(window_min, delta_min, expected):
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
    assert cal.is_news_blackout(
        "EURUSD", msc + delta_min * 60_000, window_min=window_min
    ) is expected


# ===========================================================================
# 20. Mixed-impact events in the same calendar
# ===========================================================================

class TestMixedImpact:
    def test_only_high_in_mixed_calendar_blocks(self):
        msc_low = utc_ms(2026, 5, 14, 12, 0)
        msc_high = utc_ms(2026, 5, 14, 13, 0)
        cal = StaticNewsCalendar(events=[
            event(2026, 5, 14, 12, 0, "USD", impact="LOW"),
            event(2026, 5, 14, 13, 0, "USD", impact="HIGH"),
        ])
        assert not cal.is_blackout("EURUSD", msc_low)
        assert cal.is_blackout("EURUSD", msc_high)

    def test_high_first_low_second_only_high_blocks(self):
        msc_high = utc_ms(2026, 5, 14, 12, 0)
        msc_low = utc_ms(2026, 5, 14, 13, 0)
        cal = StaticNewsCalendar(events=[
            event(2026, 5, 14, 12, 0, "USD", impact="HIGH"),
            event(2026, 5, 14, 13, 0, "USD", impact="LOW"),
        ])
        assert cal.is_blackout("EURUSD", msc_high)
        assert not cal.is_blackout("EURUSD", msc_low)

    def test_clustered_low_high_low(self):
        center = utc_ms(2026, 5, 14, 12, 30)
        cal = StaticNewsCalendar(events=[
            event(2026, 5, 14, 12, 28, "USD", impact="LOW"),
            event(2026, 5, 14, 12, 30, "USD", impact="HIGH"),
            event(2026, 5, 14, 12, 32, "USD", impact="LOW"),
        ])
        # Center is the HIGH event itself → blocks.
        assert cal.is_blackout("EURUSD", center)
        # 2 minutes before — only the LOW is within reach, no block.
        # (Actually the HIGH is 2 min away too — blocks via the HIGH event.)
        assert cal.is_blackout("EURUSD", center - 2 * 60_000)
        # 3 minutes before — neither HIGH event within ±2 min → no block.
        assert not cal.is_blackout("EURUSD", center - 3 * 60_000)


# ===========================================================================
# 21. Long-tail per-symbol smoke (single USD event)
# ===========================================================================

USD_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
    "AUDUSD", "NZDUSD", "XAUUSD",
]


@pytest.mark.parametrize("symbol", USD_PAIRS)
@pytest.mark.parametrize("delta_sec", [-120, -60, 0, 60, 120])
def test_usd_pairs_all_blocked_in_window(symbol, delta_sec):
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
    assert cal.is_blackout(symbol, msc + delta_sec * 1000)


@pytest.mark.parametrize("symbol", USD_PAIRS)
@pytest.mark.parametrize("delta_sec", [-180, -121, 121, 180, 3600])
def test_usd_pairs_all_clear_outside_window(symbol, delta_sec):
    msc = utc_ms(2026, 5, 14, 12, 30)
    cal = StaticNewsCalendar(events=[event(2026, 5, 14, 12, 30, "USD")])
    assert not cal.is_blackout(symbol, msc + delta_sec * 1000)


# ===========================================================================
# 22. Sanity: events tuple is immutable
# ===========================================================================

class TestEventsImmutable:
    def test_cannot_set_events_attribute(self):
        cal = StaticNewsCalendar(events=[])
        with pytest.raises(AttributeError):
            cal.events = ()  # type: ignore[misc]

    def test_returned_tuple_does_not_share_with_input(self):
        src = [event(2026, 5, 14, 12, 30, "USD")]
        cal = StaticNewsCalendar(events=src)
        # Tuple cast must have been made.
        assert isinstance(cal.events, tuple)
        # Mutating src must not affect cal.
        src.append(event(2026, 5, 14, 13, 30, "USD"))
        assert len(cal.events) == 1
