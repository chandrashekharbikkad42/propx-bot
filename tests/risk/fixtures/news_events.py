"""News-calendar helpers + canned event sets."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import List

from data.news_calendar import NewsEvent


UTC = timezone.utc


def utc_ms(year: int, month: int, day: int, hour: int, minute: int = 0,
           second: int = 0) -> int:
    return int(
        datetime(year, month, day, hour, minute, second, tzinfo=UTC)
        .timestamp() * 1000
    )


def event(year: int, month: int, day: int, hour: int, minute: int = 0,
          currency: str = "USD", title: str = "NFP",
          impact: str = "HIGH") -> NewsEvent:
    return NewsEvent(
        time_msc=utc_ms(year, month, day, hour, minute),
        currency=currency,
        title=title,
        impact=impact,
    )


def usd_event_2026(hour: int = 12, minute: int = 30,
                   title: str = "CPI") -> NewsEvent:
    return event(2026, 5, 14, hour, minute, "USD", title)


# A pre-built spread of events for parametrised tests.
SPREAD: List[NewsEvent] = [
    event(2026, 5, 14, 12, 30, "USD", "CPI"),
    event(2026, 5, 14, 12, 30, "EUR", "ECB"),     # simultaneous EUR
    event(2026, 5, 15, 12, 30, "USD", "PPI"),
    event(2026, 5, 28, 18, 0,  "USD", "FOMC"),
    event(2026, 6, 5,  12, 30, "USD", "NFP"),
    event(2026, 6, 18, 11, 0,  "GBP", "BoE"),
    event(2026, 6, 25, 9, 0,   "JPY", "BoJ"),
    event(2026, 6, 26, 0, 30,  "CAD", "CPI"),
    event(2026, 6, 26, 0, 30,  "CHF", "SNB"),
    event(2026, 6, 26, 0, 30,  "AUD", "RBA"),
    event(2026, 6, 26, 0, 30,  "NZD", "RBNZ"),
    # A LOW-impact event we should ignore.
    event(2026, 6, 5,  13, 0,  "USD", "JOLTS", impact="LOW"),
    # A MEDIUM-impact event we should ignore.
    event(2026, 6, 5,  13, 30, "USD", "Manufacturing PMI",
          impact="MEDIUM"),
]
