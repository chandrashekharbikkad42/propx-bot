"""High-impact economic-news calendar — interface + static fallback.

Used by the pre-trade compliance check (Phase 8D) to block entries inside a
news blackout window. Spec from the user:
  - Block trades within 2 minutes BEFORE and AFTER high-impact events
    (NFP, CPI, FOMC, etc.) for funded prop accounts.
  - Pluggable source: today = `StaticNewsCalendar`, later = ForexFactory
    scraper / paid API / MT5 plugin.

Per-symbol applicability:
  - USD events affect any pair containing USD (EURUSD, GBPUSD, USDJPY, XAUUSD ...).
  - EUR events affect EUR pairs; GBP events affect GBP pairs; etc.
  - Currency match is by substring — pair codes are 6 chars like "EURUSD"
    so the test is `currency in symbol`.

Hinglish: news event ke ±2 min me trade nahi karna — broker spread blow up
kar deta hai aur jhatak ke SL hit ho jaata hai. FTMO funded account pe news
ke time trade le liya toh rule violation = account gone.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional, Protocol

# Default symmetric blackout window in minutes (±N). Aligns with the
# Asian Sweep V5 config (NEWS_BLACKOUT_MIN=2) and the prop-firm rule book.
DEFAULT_BLACKOUT_WINDOW_MIN: int = 2


@dataclass(frozen=True)
class NewsEvent:
    """One high-impact macro event."""
    time_msc: int          # UTC ms at the event print time
    currency: str          # 3-letter code: USD, EUR, GBP, JPY, ...
    title: str             # "NFP", "CPI", "FOMC Statement", etc.
    impact: str = "HIGH"   # HIGH / MEDIUM / LOW — bot only blocks HIGH today


class NewsCalendar(Protocol):
    """Strategy plugs against this. Replaceable per source.

    `window_min` is the symmetric blackout in minutes — default 2.
    """

    def is_news_blackout(
        self, symbol: str, time_msc: int, window_min: int = 2
    ) -> bool: ...

    def is_blackout(self, symbol: str, time_msc: int) -> bool: ...

    def upcoming_events(self, after_msc: int, limit: int = 10) -> list[NewsEvent]: ...


# ---------------------------------------------------------------------------
# Static fallback — small hardcoded list, refreshed manually.
# ---------------------------------------------------------------------------

def _utc_ms(year: int, month: int, day: int, hour: int, minute: int) -> int:
    return int(
        datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp() * 1000
    )


# Curated high-impact events. Edit this list as the calendar moves forward.
# Times are the scheduled print time in UTC.
# TODO Phase 8D — replace with ForexFactoryNewsCalendar (scraper).
DEFAULT_HIGH_IMPACT_EVENTS: tuple[NewsEvent, ...] = (
    # ---- May 2026 ----
    NewsEvent(_utc_ms(2026, 5, 14, 12, 30), "USD", "CPI"),
    NewsEvent(_utc_ms(2026, 5, 15, 12, 30), "USD", "PPI"),
    NewsEvent(_utc_ms(2026, 5, 28, 18, 0),  "USD", "FOMC Minutes"),
    # ---- June 2026 ----
    NewsEvent(_utc_ms(2026, 6, 5,  12, 30), "USD", "NFP"),    # 1st Friday
    NewsEvent(_utc_ms(2026, 6, 11, 12, 30), "USD", "CPI"),
    NewsEvent(_utc_ms(2026, 6, 17, 18, 0),  "USD", "FOMC Statement"),
    NewsEvent(_utc_ms(2026, 6, 18, 11, 0),  "GBP", "BoE Rate"),
    # ---- July 2026 (NFP placeholder) ----
    NewsEvent(_utc_ms(2026, 7, 3,  12, 30), "USD", "NFP"),
    NewsEvent(_utc_ms(2026, 7, 15, 12, 30), "USD", "CPI"),
    NewsEvent(_utc_ms(2026, 7, 29, 18, 0),  "USD", "FOMC Statement"),
)


class StaticNewsCalendar:
    """In-memory NewsCalendar backed by a constant list. Default fallback.

    Useful for tests + first-pass production use until ForexFactory scraper
    lands. Events are sorted by time at construction so lookups can early-exit.
    """

    def __init__(self, events: Optional[Iterable[NewsEvent]] = None) -> None:
        src = list(events) if events is not None else list(DEFAULT_HIGH_IMPACT_EVENTS)
        # Sort ascending by time so upcoming_events and blackout-scan are O(log n) / linear-early-exit.
        self._events: tuple[NewsEvent, ...] = tuple(sorted(src, key=lambda e: e.time_msc))

    @property
    def events(self) -> tuple[NewsEvent, ...]:
        return self._events

    def is_blackout(self, symbol: str, time_msc: int) -> bool:
        """Shorthand for `is_news_blackout` using the default ±2 min window.

        Asian Sweep V5 callers should prefer this — keeps the blackout
        policy a single source of truth (`DEFAULT_BLACKOUT_WINDOW_MIN`).
        """
        return self.is_news_blackout(
            symbol, time_msc, window_min=DEFAULT_BLACKOUT_WINDOW_MIN
        )

    def is_news_blackout(
        self, symbol: str, time_msc: int, window_min: int = 2
    ) -> bool:
        """True if `time_msc` falls within ±window_min minutes of a HIGH event
        whose currency is present in `symbol`."""
        if window_min < 0:
            raise ValueError("window_min must be >= 0")
        if not self._events:
            return False
        window_ms = window_min * 60 * 1000
        sym = symbol.upper()
        for evt in self._events:
            if evt.impact != "HIGH":
                continue
            if evt.currency not in sym:
                continue
            if abs(evt.time_msc - time_msc) <= window_ms:
                return True
            # Optimisation: events are sorted; if we're far past this one we
            # could break, but the symbol-mismatch path keeps us moving anyway.
        return False

    def upcoming_events(self, after_msc: int, limit: int = 10) -> list[NewsEvent]:
        """Next `limit` events strictly after `after_msc`. Useful for dashboards."""
        if limit < 0:
            raise ValueError("limit must be >= 0")
        if limit == 0:
            return []
        out: list[NewsEvent] = []
        for evt in self._events:
            if evt.time_msc <= after_msc:
                continue
            out.append(evt)
            if len(out) >= limit:
                break
        return out


# Module-level default singleton — caller can replace via constructor injection.
DEFAULT_CALENDAR: NewsCalendar = StaticNewsCalendar()
