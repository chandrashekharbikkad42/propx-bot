"""Phase 3 integration proof: replay 31k captured ticks → SignalEngine.

Reports:
    - total signals by type (SWEEP / MOMENTUM / REJECTION)
    - signals by session
    - top 5 highest-magnitude signals
    - replay-vs-consumer reconciliation (no orphan ticks)

Usage:
    python -m tests.integration_phase3 [--date YYYY-MM-DD]
"""

from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from pathlib import Path

from config.settings import settings
from data.tick_collector import Tick
from replay.replay_engine import ReplayConfig, ReplayEngine
from strategy.signal_engine import SignalEngine
from strategy.signals.base import (
    MomentumSignal,
    RejectionSignal,
    Signal,
    SignalType,
    SweepSignal,
)
from utils.logger import logger


def _signal_magnitude(s: Signal) -> float:
    if isinstance(s, SweepSignal):
        return abs(s.spike_pts)
    if isinstance(s, MomentumSignal):
        return abs(s.cumulative_pts)
    if isinstance(s, RejectionSignal):
        return abs(s.spike_pts)
    return 0.0


async def _run(date: str) -> int:
    symbol = settings.symbol
    partition = settings.data_dir / f"symbol={symbol}" / f"date={date}"
    if not partition.exists():
        logger.error(f"Partition not found: {partition}")
        return 1
    files = tuple(sorted(partition.glob("part-*.parquet")))
    logger.info(f"integration | partition={partition} files={len(files)}")

    tick_q: asyncio.Queue[Tick] = asyncio.Queue(maxsize=10_000)
    sig_q: asyncio.Queue[Signal] = asyncio.Queue()

    replay = ReplayEngine(
        ReplayConfig(symbol=symbol, date=date, speed=0.0, files=files), tick_q
    )
    engine = SignalEngine(tick_q, sig_q, log_each_signal=False)

    replay_task = asyncio.create_task(replay.run(), name="replay")
    engine_task = asyncio.create_task(engine.run(), name="engine")

    try:
        await replay_task
    except Exception as exc:  # noqa: BLE001 — top-level integration boundary
        logger.exception(f"replay crashed: {exc}")
        engine_task.cancel()
        return 2

    # Wait for engine to consume the residual queue, then cancel.
    while engine.consumed < replay.emitted:
        await asyncio.sleep(0.01)
    engine_task.cancel()
    try:
        await engine_task
    except asyncio.CancelledError:
        pass

    # Drain signal queue into list.
    signals: list[Signal] = []
    while not sig_q.empty():
        signals.append(sig_q.get_nowait())

    by_type: Counter[str] = Counter(s.type.value for s in signals)
    by_session: Counter[str] = Counter(s.session.value for s in signals)

    print()
    print("=" * 64)
    print(f"Phase 3 integration | date={date} symbol={symbol}")
    print("=" * 64)
    print(f"Replay      : emitted={replay.emitted}  skipped={replay.skipped}")
    print(f"Engine      : consumed={engine.consumed}  signals={engine.emitted}")
    print(f"Reconciled  : {engine.consumed == replay.emitted}")
    print()
    print("Signals by type:")
    for t in SignalType:
        print(f"  {t.value:<10s} {by_type.get(t.value, 0):>6d}")
    print()
    print("Signals by session:")
    for sess, n in sorted(by_session.items(), key=lambda x: -x[1]):
        print(f"  {sess:<20s} {n:>6d}")
    print()
    print("Top 5 by magnitude (points):")
    top = sorted(signals, key=_signal_magnitude, reverse=True)[:5]
    for s in top:
        mag = _signal_magnitude(s)
        extra = ""
        if isinstance(s, MomentumSignal):
            extra = f" cum={s.cumulative_pts:+.1f} cons={s.consistency:.2f}"
        elif isinstance(s, SweepSignal):
            extra = f" spike={s.spike_pts:+.1f} std={s.rolling_std_pts:.1f}"
        elif isinstance(s, RejectionSignal):
            extra = (
                f" spike={s.spike_pts:+.1f} rev={s.reversal_pts:+.1f} "
                f"ticks={s.ticks_to_reject}"
            )
        print(
            f"  {s.type.value:<10s} t={s.time_msc} "
            f"dir={s.direction.name:<4s} session={s.session.value:<18s} "
            f"mag={mag:.1f}{extra}"
        )
    print()
    no_orphans = engine.consumed == replay.emitted
    print(f"Zero exceptions, no orphan ticks: {no_orphans}")
    print("=" * 64)
    return 0 if no_orphans else 3


def main() -> int:
    parser = argparse.ArgumentParser(prog="integration_phase3")
    parser.add_argument("--date", default="2026-05-12")
    args = parser.parse_args()
    return asyncio.run(_run(args.date))


if __name__ == "__main__":
    raise SystemExit(main())
