"""UTC-based session tagger + IST window helpers. Pure functions — no I/O, no globals."""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from zoneinfo import ZoneInfo


class SessionLabel(str, Enum):
    ASIAN = "ASIAN"
    LONDON = "LONDON"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"
    NY = "NY"
    OFF = "OFF"


# UTC session windows (half-open intervals on the hour):
#   ASIAN              : [00:00, 07:00)
#   LONDON             : [07:00, 12:00)
#   LONDON_NY_OVERLAP  : [12:00, 16:00)
#   NY                 : [16:00, 21:00)
#   OFF                : [21:00, 24:00)
#
# DST-free by construction: pure UTC, no zone conversion.

def session_for(dt: datetime) -> SessionLabel:
    """Map a UTC datetime to its session label. Caller must pass an aware UTC datetime."""
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) != timezone.utc.utcoffset(dt):
        # Normalize to UTC if input has a non-UTC zone or is naive.
        dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    h = dt.hour
    if h < 7:
        return SessionLabel.ASIAN
    if h < 12:
        return SessionLabel.LONDON
    if h < 16:
        return SessionLabel.LONDON_NY_OVERLAP
    if h < 21:
        return SessionLabel.NY
    return SessionLabel.OFF


def session_for_msc(time_msc: int) -> SessionLabel:
    """Convenience: derive session from MT5-style epoch milliseconds."""
    return session_for(datetime.fromtimestamp(time_msc / 1000.0, tz=timezone.utc))


# ---------------------------------------------------------------------------
# Phase 8B — IST trading window helpers
#
# Bot trades only inside 12:30–22:30 IST (London open through NY close).
# Daily counters (trade-count, daily-loss reset, consistency rule) are also
# anchored to the IST calendar day — user is in India, prop firm "trading day"
# semantically aligns with user's local day.
# ---------------------------------------------------------------------------

IST = ZoneInfo("Asia/Kolkata")


def to_ist(time_msc: int) -> datetime:
    """Convert MT5-style epoch milliseconds (UTC) to an IST-aware datetime."""
    return datetime.fromtimestamp(time_msc / 1000.0, tz=timezone.utc).astimezone(IST)


def ist_date(time_msc: int) -> str:
    """IST calendar date as YYYY-MM-DD. Day rollover at 00:00 IST."""
    return to_ist(time_msc).strftime("%Y-%m-%d")


def _parse_hhmm(value: str) -> tuple[int, int]:
    h_str, m_str = value.split(":")
    return int(h_str), int(m_str)


def is_within_ist_window(
    time_msc: int,
    window_start: str = "12:30",
    window_end: str = "22:30",
) -> bool:
    """True if the given UTC ms timestamp falls in [start, end] in IST.

    Windows are half-open on the end: [start, end). A window that wraps
    midnight is supported (start > end means "from start today to end tomorrow"),
    though Griff's window 12:30–22:30 never wraps.

    `window_start` / `window_end` are HH:MM strings.
    """
    ist_dt = to_ist(time_msc)
    minutes_now = ist_dt.hour * 60 + ist_dt.minute
    sh, sm = _parse_hhmm(window_start)
    eh, em = _parse_hhmm(window_end)
    start_min = sh * 60 + sm
    end_min = eh * 60 + em
    if start_min <= end_min:
        return start_min <= minutes_now < end_min
    # Wrap-midnight case
    return minutes_now >= start_min or minutes_now < end_min
