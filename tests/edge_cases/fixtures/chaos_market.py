"""Chaos market generators.

Produces deterministic-but-pathological bar/tick sequences used by
test_market_chaos.py and others. Every helper returns plain values
(``list[Bar]`` / ``list[Tick]``) so tests can compose them freely.

All bar timestamps are UTC ms, aligned to hour boundaries unless the
test explicitly wants misalignment (a chaos scenario in itself).
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Sequence

from data.bar_aggregator import Bar
from data.tick_collector import Tick


UTC = timezone.utc
HOUR_MS = 60 * 60 * 1000


def hour_msc(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def make_bar(
    *,
    symbol: str = "XAUUSD",
    time_msc: int = 0,
    open: float = 2000.0,
    high: Optional[float] = None,
    low: Optional[float] = None,
    close: float = 2000.0,
    volume: int = 1,
    spread_mean: float = 0.0,
) -> Bar:
    """Build a Bar; derive high/low when omitted to stay self-consistent."""
    if high is None:
        high = max(open, close)
    if low is None:
        low = min(open, close)
    return Bar(
        symbol=symbol, time_msc=time_msc,
        open=open, high=high, low=low, close=close,
        volume=volume, spread_mean=spread_mean,
    )


def flat_bars(
    *,
    symbol: str = "XAUUSD",
    start_msc: int = 0,
    count: int = 10,
    price: float = 2000.0,
    volume: int = 100,
    period_ms: int = HOUR_MS,
) -> List[Bar]:
    """N equal bars at the same price — a stale-feed canvas to perturb."""
    return [
        make_bar(
            symbol=symbol,
            time_msc=start_msc + i * period_ms,
            open=price, high=price, low=price, close=price,
            volume=volume,
        )
        for i in range(count)
    ]


def gap_open_bars(
    *,
    symbol: str = "XAUUSD",
    start_msc: int = 0,
    pre_price: float = 2000.0,
    post_price: float = 1900.0,
    pre_bars: int = 5,
    gap_at_index: int = 5,
    post_bars: int = 5,
) -> List[Bar]:
    """Sequence with a single hard gap in price at `gap_at_index`."""
    bars: List[Bar] = []
    for i in range(pre_bars):
        bars.append(make_bar(symbol=symbol, time_msc=start_msc + i * HOUR_MS,
                             open=pre_price, close=pre_price))
    # gap bar — opens at post_price, no overlap with pre_price.
    bars.append(make_bar(
        symbol=symbol,
        time_msc=start_msc + gap_at_index * HOUR_MS,
        open=post_price, close=post_price,
    ))
    for j in range(1, post_bars):
        bars.append(make_bar(
            symbol=symbol,
            time_msc=start_msc + (gap_at_index + j) * HOUR_MS,
            open=post_price, close=post_price,
        ))
    return bars


def flash_crash_bar(
    *,
    symbol: str = "XAUUSD",
    time_msc: int = 0,
    open_price: float = 2000.0,
    crash_low: float = 1500.0,
    close_price: float = 1980.0,
) -> Bar:
    """Single bar with extreme wick down (price gaps through SL)."""
    return make_bar(
        symbol=symbol, time_msc=time_msc,
        open=open_price, high=open_price + 5.0,
        low=crash_low, close=close_price, volume=10_000,
    )


def spread_explosion_tick(
    *,
    bid: float = 2000.0,
    spread: float = 50.0,
    time_msc: int = 0,
) -> Tick:
    """Tick with 10x normal spread."""
    return Tick(
        time_msc=time_msc, bid=bid, ask=bid + spread, last=bid + spread / 2,
        volume=1, volume_real=1.0, flags=0,
    )


def stale_feed_bars(
    *,
    symbol: str = "XAUUSD",
    start_msc: int = 0,
    count: int = 20,
    frozen_price: float = 2000.0,
) -> List[Bar]:
    """All bars identical OHLC — price hasn't moved at all."""
    return [
        make_bar(
            symbol=symbol, time_msc=start_msc + i * HOUR_MS,
            open=frozen_price, high=frozen_price,
            low=frozen_price, close=frozen_price, volume=0,
        )
        for i in range(count)
    ]


def tick_storm(
    *,
    base_msc: int = 0,
    count: int = 1000,
    bid: float = 2000.0,
    ask: float = 2000.05,
) -> List[Tick]:
    """Burst of N back-to-back ticks within a single millisecond window."""
    return [
        Tick(
            time_msc=base_msc + (i // 100),
            bid=bid + (i % 5) * 0.01,
            ask=ask + (i % 5) * 0.01,
            last=bid + (i % 5) * 0.01,
            volume=1, volume_real=1.0, flags=0,
        )
        for i in range(count)
    ]


def zero_volume_bars(
    *, symbol: str = "XAUUSD", start_msc: int = 0, count: int = 5,
    price: float = 2000.0,
) -> List[Bar]:
    return [
        make_bar(
            symbol=symbol, time_msc=start_msc + i * HOUR_MS,
            open=price, high=price, low=price, close=price, volume=0,
        )
        for i in range(count)
    ]


def inverted_bar(
    *, symbol: str = "XAUUSD", time_msc: int = 0,
    bad_high: float = 1990.0, bad_low: float = 2010.0,
    open: float = 2000.0, close: float = 2000.0,
) -> Bar:
    """High < low — corrupt bar. Bar dataclass does NOT validate so this
    constructs successfully; we use it to probe whether downstream code
    catches the invariant violation.
    """
    return Bar(
        symbol=symbol, time_msc=time_msc,
        open=open, high=bad_high, low=bad_low, close=close,
        volume=1, spread_mean=0.0,
    )


def degenerate_asian_range_bars(
    *,
    symbol: str = "XAUUSD",
    year: int = 2026,
    month: int = 5,
    day: int = 15,
    single_price: float = 2000.0,
) -> List[Bar]:
    """5 Asian-window bars all at one tick — Asian range collapses to zero."""
    cur_date = datetime(year, month, day, 0, 0, tzinfo=UTC).date()
    prev_date = cur_date - timedelta(days=1)
    hours = [
        (prev_date, 20), (prev_date, 21), (prev_date, 22),
        (prev_date, 23), (cur_date, 0),
    ]
    return [
        make_bar(
            symbol=symbol,
            time_msc=int(datetime(d.year, d.month, d.day, h, 0,
                                  tzinfo=UTC).timestamp() * 1000),
            open=single_price, high=single_price,
            low=single_price, close=single_price, volume=1,
        )
        for d, h in hours
    ]


def negative_price_bar(*, symbol: str = "XAUUSD", time_msc: int = 0) -> Bar:
    """Bar with negative price — Bar dataclass allows it but downstream
    sizing / PnL should reject."""
    return Bar(
        symbol=symbol, time_msc=time_msc,
        open=-2000.0, high=-1999.0, low=-2001.0, close=-2000.0,
        volume=1, spread_mean=0.0,
    )


def duplicate_timestamp_bars(
    *, symbol: str = "XAUUSD", time_msc: int = 0, count: int = 3,
    price: float = 2000.0,
) -> List[Bar]:
    """N bars sharing the SAME time_msc — duplicate ingestion."""
    return [
        make_bar(symbol=symbol, time_msc=time_msc,
                 open=price, high=price, low=price, close=price)
        for _ in range(count)
    ]


def out_of_order_bars(
    *, symbol: str = "XAUUSD", start_msc: int = 0,
    count: int = 5, price: float = 2000.0,
) -> List[Bar]:
    """Bars with monotonically DECREASING timestamps — backwards order."""
    return [
        make_bar(symbol=symbol, time_msc=start_msc + (count - i) * HOUR_MS,
                 open=price, high=price, low=price, close=price)
        for i in range(count)
    ]


def future_dated_bar(
    *, symbol: str = "XAUUSD", price: float = 2000.0,
    base_year: int = 2099,
) -> Bar:
    """Bar timestamped far in the future — clock-skew probe."""
    return make_bar(
        symbol=symbol,
        time_msc=hour_msc(base_year, 1, 1, 0, 0),
        open=price, high=price, low=price, close=price,
    )


def whipsaw_bars(
    *, symbol: str = "XAUUSD", start_msc: int = 0,
    base_price: float = 2000.0, amp: float = 10.0,
    count: int = 20,
) -> List[Bar]:
    """Alternating up/down bars — pure noise."""
    out: List[Bar] = []
    for i in range(count):
        sign = 1 if i % 2 == 0 else -1
        p = base_price + sign * amp
        out.append(make_bar(symbol=symbol, time_msc=start_msc + i * HOUR_MS,
                            open=base_price, close=p))
    return out


def asian_window_with_missing_bars(
    *, symbol: str = "XAUUSD", year: int = 2026, month: int = 5, day: int = 15,
    asian_high: float = 2010.0, asian_low: float = 1990.0,
    keep_indices: Sequence[int] = (0, 4),
) -> List[Bar]:
    """Asian window where only SOME bars are present (e.g. broker downtime).
    `keep_indices` selects which of the 5 hours (20,21,22,23,00) survive.
    """
    cur_date = datetime(year, month, day, 0, 0, tzinfo=UTC).date()
    prev_date = cur_date - timedelta(days=1)
    hours = [
        (prev_date, 20), (prev_date, 21), (prev_date, 22),
        (prev_date, 23), (cur_date, 0),
    ]
    out: List[Bar] = []
    for i, (d, h) in enumerate(hours):
        if i not in keep_indices:
            continue
        # Stamp the highest/lowest on the first/last kept bar.
        hi = asian_high if i == max(keep_indices) else (asian_high + asian_low) / 2
        lo = asian_low if i == min(keep_indices) else (asian_high + asian_low) / 2
        out.append(make_bar(
            symbol=symbol,
            time_msc=int(datetime(d.year, d.month, d.day, h, 0,
                                  tzinfo=UTC).timestamp() * 1000),
            open=(asian_high + asian_low) / 2,
            high=hi, low=lo, close=(asian_high + asian_low) / 2,
        ))
    return out
