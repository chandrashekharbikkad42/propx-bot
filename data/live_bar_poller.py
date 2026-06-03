"""LiveBarPoller — drive the Griff engine off MT5's H1 bar stream.

Polls MT5 for closed H1 bars per pair. When a fresh close is detected:
  1. Update the rolling history buffer (used as `bar_feeds` for the scanner).
  2. Build a `now_msc` from wall-clock time.
  3. Read live ask/bid prices via the caller-supplied `prices_provider`.
  4. Snapshot account state via the caller-supplied `account_provider`.
  5. Drive `engine.process_scan_cycle()` + `engine.maintain_open()`.

The poller is intentionally a thin async loop:
  - MT5 module is INJECTED so tests don't need a live terminal.
  - Account + prices providers are CALLABLES so the run-loop owner can wire
    them to MT5 / mocks without the poller learning about either.
  - Exceptions inside one poll are caught and logged — the loop must not
    die because one MT5 call hiccuped.

Bar timing convention (mirrors `data.bar_aggregator.Bar`):
  - `Bar.time_msc` is the bar's OPEN ms timestamp.
  - We request `start_pos=1` from MT5 to SKIP the currently-forming bar
    — only fully-closed bars enter the buffer.

Hinglish: yeh polling shell hai. Har `poll_sec` second me MT5 se latest
closed H1 bars laata hai, naya close mile to engine ka scan cycle fire
karta hai. DRY_RUN se yeh use NAHI hota — sirf LIVE mode pe wire hota hai.
"""

from __future__ import annotations
import asyncio
import time
from typing import Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

from data.bar_aggregator import Bar
from data.bar_capture_utils import mt5_rates_to_bars
from utils.logger import logger


AccountProvider = Callable[[], object]
PricesProvider = Callable[[], "tuple[Mapping[str, float], Mapping[str, float]]"]


class LiveBarPoller:
    """Polls MT5 for closed H1 bars; drives the live engine per close."""

    def __init__(
        self,
        *,
        pairs: Sequence[str],
        mt5_module,
        history_bars: int = 50,
        poll_sec: float = 30.0,
    ) -> None:
        self._pairs: tuple[str, ...] = tuple(pairs)
        self._mt5 = mt5_module
        self._history_bars = max(1, int(history_bars))
        self._poll_sec = max(0.01, float(poll_sec))
        self._last_bar_msc: Dict[str, int] = {p: 0 for p in self._pairs}
        self._buffer: Dict[str, List[Bar]] = {p: [] for p in self._pairs}

    # ----- views -------------------------------------------------------

    @property
    def buffer(self) -> Mapping[str, List[Bar]]:
        return self._buffer

    @property
    def last_bar_msc(self) -> Mapping[str, int]:
        return dict(self._last_bar_msc)

    # ----- per-poll work ---------------------------------------------

    def fetch_closed_bars(self, pair: str) -> List[Bar]:
        """Pull the last N CLOSED H1 bars from MT5 (skips current forming bar)."""
        rates = self._mt5.copy_rates_from_pos(
            pair, self._mt5.TIMEFRAME_H1, 1, self._history_bars,
        )
        if rates is None:
            return []
        return mt5_rates_to_bars(rates, pair)

    def poll_once(self) -> Dict[str, Bar]:
        """Update buffers; return `{pair: newest_bar}` for NEW closes only."""
        new_closes: Dict[str, Bar] = {}
        for pair in self._pairs:
            bars = self.fetch_closed_bars(pair)
            if not bars:
                continue
            self._buffer[pair] = bars
            newest = bars[-1]
            if newest.time_msc > self._last_bar_msc[pair]:
                self._last_bar_msc[pair] = newest.time_msc
                new_closes[pair] = newest
        return new_closes

    # ----- run loop ---------------------------------------------------

    async def run(
        self,
        *,
        engine,
        stop: asyncio.Event,
        account_provider: AccountProvider,
        prices_provider: PricesProvider,
    ) -> None:
        """Loop until stop is set; on each new close, drive the engine."""
        while not stop.is_set():
            try:
                new_closes = self.poll_once()
            except Exception as exc:  # noqa: BLE001 — must never kill the loop
                logger.warning(f"LiveBarPoller poll_once failed: {exc}")
                new_closes = {}

            if new_closes:
                try:
                    await self._fire_engine(
                        engine=engine,
                        new_closes=new_closes,
                        account_provider=account_provider,
                        prices_provider=prices_provider,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"LiveBarPoller engine call failed: {exc}"
                    )

            try:
                await asyncio.wait_for(stop.wait(), timeout=self._poll_sec)
                return  # stop signalled
            except asyncio.TimeoutError:
                pass

    async def _fire_engine(
        self,
        *,
        engine,
        new_closes: Mapping[str, Bar],
        account_provider: AccountProvider,
        prices_provider: PricesProvider,
    ) -> None:
        now_msc = int(time.time() * 1000)
        ask_by_pair, bid_by_pair = prices_provider()
        account = account_provider()
        logger.info(
            f"GRIFF live tick: {len(new_closes)} new bar(s) "
            f"({','.join(new_closes.keys())})"
        )
        await engine.process_scan_cycle(
            self._buffer,
            now_msc=now_msc,
            ask_by_pair=ask_by_pair,
            bid_by_pair=bid_by_pair,
            account=account,
        )
        await engine.maintain_open(new_closes, now_msc=now_msc)
