"""End-to-end backtest. Replay → strategy → risk → broker → journal.

In-process pipeline — no asyncio queues between stages: at backtest speed
the queue churn dwarfs the actual work, and a single tight loop keeps
ordering deterministic and the logs sane.
"""

from __future__ import annotations
import asyncio
from dataclasses import replace
from typing import Optional

import pyarrow.parquet as pq

from config.settings import settings as global_settings
from data.tick_collector import Tick
from data.tick_writer import TICK_SCHEMA
from backtesting.trade_journal import TradeJournal
from execution.broker_simulator import PaperBroker
from execution.position import CloseReason, Position
from replay.replay_engine import ReplayConfig
from risk.risk_engine import RiskEngine
from strategy.microstructure import MicrostructureState
from strategy.signal_confirmation import SignalConfirmationBuffer
from strategy.signals import (
    LiquiditySweepDetector,
    RejectionDetector,
    Signal,
    TickMomentumDetector,
)
from utils.logger import logger


POINT_VALUE = 0.01


def _enrich(tick: Tick, prev_mid: Optional[float]) -> tuple[dict, float]:
    mid = (tick.bid + tick.ask) * 0.5
    mid_change_pts = 0.0 if prev_mid is None else (mid - prev_mid) / POINT_VALUE
    spread_pts = (tick.ask - tick.bid) / POINT_VALUE
    if mid_change_pts > 0:
        direction = 1
    elif mid_change_pts < 0:
        direction = -1
    else:
        direction = 0
    return (
        {
            "time_msc": tick.time_msc,
            "bid": tick.bid,
            "ask": tick.ask,
            "mid": mid,
            "mid_change_pts": mid_change_pts,
            "spread_pts": spread_pts,
            "direction": direction,
        },
        mid,
    )


def _iter_ticks(symbol: str, date_str: str):
    """Yield Ticks from a partition without going through asyncio.Queue."""
    partition_dir = (
        global_settings.data_dir
        / f"symbol={symbol}"
        / f"date={date_str}"
    )
    if not partition_dir.exists():
        raise FileNotFoundError(f"Partition not found: {partition_dir}")
    files = sorted(partition_dir.glob("part-*.parquet"))
    if not files:
        raise ValueError(f"No parquet files for replay: {partition_dir}")

    columns = [f.name for f in TICK_SCHEMA]
    last_msc = 0
    for file in files:
        pf = pq.ParquetFile(str(file))
        for batch in pf.iter_batches(batch_size=2000, columns=columns):
            time_msc = batch.column("time_msc").to_numpy()
            bid = batch.column("bid").to_numpy()
            ask = batch.column("ask").to_numpy()
            last = batch.column("last").to_numpy()
            volume = batch.column("volume").to_numpy()
            volume_real = batch.column("volume_real").to_numpy()
            flags = batch.column("flags").to_numpy()
            for i in range(len(time_msc)):
                tmsc = int(time_msc[i])
                if tmsc <= last_msc:
                    continue
                last_msc = tmsc
                yield Tick(
                    time_msc=tmsc,
                    bid=float(bid[i]),
                    ask=float(ask[i]),
                    last=float(last[i]),
                    volume=int(volume[i]),
                    volume_real=float(volume_real[i]),
                    flags=int(flags[i]),
                )


def run_backtest_sync(
    date_str: str,
    symbol: Optional[str] = None,
    capital: float = 10_000.0,
    broker_type: Optional[str] = None,
    account_type: Optional[str] = None,
) -> dict:
    """Synchronous backtest core. Returns the summary dict.

    `broker_type` and `account_type` default to whatever the .env loader
    resolved on global_settings. Callers (typically tests) can override to
    pin a known configuration regardless of the active .env values.
    """
    sym = symbol or global_settings.symbol

    overrides: dict = {"simulated_starting_capital": capital}
    if broker_type is not None:
        overrides["broker_type"] = broker_type
    if account_type is not None:
        overrides["account_type"] = account_type
    bt_settings = replace(global_settings, **overrides)

    state = MicrostructureState()
    sweep = LiquiditySweepDetector(broker_type=bt_settings.broker_type)
    momentum = TickMomentumDetector(broker_type=bt_settings.broker_type)
    rejection = RejectionDetector(broker_type=bt_settings.broker_type)
    confirmer = SignalConfirmationBuffer()
    risk = RiskEngine(bt_settings)
    broker = PaperBroker()
    journal = TradeJournal()

    prev_mid: Optional[float] = None
    last_tick: Optional[Tick] = None
    consumed = 0
    signals_seen = 0
    blocked = 0

    for tick in _iter_ticks(sym, date_str):
        last_tick = tick
        consumed += 1

        enriched, prev_mid = _enrich(tick, prev_mid)
        state.update(enriched)

        # 1. Exit check on any open position FIRST — a tick that would both
        #    close the position AND trigger a new signal must not double-book.
        if risk.open_positions:
            pos = risk.open_positions[0]
            closed = broker.check_position_exit(pos, tick)
            if closed is not None:
                risk.register_close(closed)
                journal.log_close(closed)

        # 2. Age the confirmation buffer first — signals from earlier ticks
        #    may release this tick (3-tick post-fire wait, Phase 5).
        confirmed_now: list[Signal] = list(
            confirmer.on_tick(enriched["mid_change_pts"])
        )

        # 3. Run detectors (mirror SignalEngine ordering) and buffer fresh
        #    candidates — they will not be eligible to fill until 3 ticks elapse.
        s = sweep.on_tick(enriched, state)
        if s is not None:
            confirmer.add(s)
        m = momentum.on_tick(enriched, state)
        if m is not None:
            confirmer.add(m)
        r = rejection.on_tick(enriched, state)
        if r is not None:
            confirmer.add(r)

        for sig in confirmed_now:
            signals_seen += 1
            ok, reason = risk.can_open_position(sig, tick)
            if not ok:
                blocked += 1
                continue
            intent = risk.build_order_intent(sig, tick)
            pos = broker.fill_market_order(intent, tick)
            risk.register_open(pos)
            journal.log_open(pos)

    # Close anything still open at the last tick.
    if last_tick is not None and risk.open_positions:
        for pos in list(risk.open_positions):
            closed = broker.force_close(pos, last_tick, reason=CloseReason.EOD)
            risk.register_close(closed)
            journal.log_close(closed)

    summary = journal.summary()
    summary["ticks_consumed"] = consumed
    summary["signals_seen"] = signals_seen
    summary["signals_blocked"] = blocked
    summary["final_equity"] = risk.account_equity
    summary["starting_equity"] = capital
    summary["return_pct"] = (
        (risk.account_equity - capital) / capital * 100.0 if capital > 0 else 0.0
    )
    return summary


async def run_backtest(
    date_str: str,
    speed: float = 0.0,
    capital: float = 10_000.0,
    symbol: Optional[str] = None,
) -> dict:
    """Async wrapper. `speed` is kept for CLI parity but the sync runner
    always replays max-speed — pacing buys nothing for an offline backtest."""
    summary = await asyncio.to_thread(run_backtest_sync, date_str, symbol, capital)
    print_summary(summary)
    return summary


def print_summary(s: dict) -> None:
    logger.info("=" * 60)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Ticks consumed   : {s.get('ticks_consumed', 0):,}")
    logger.info(f"Signals seen     : {s.get('signals_seen', 0):,}")
    logger.info(f"Signals blocked  : {s.get('signals_blocked', 0):,}")
    logger.info(f"Total trades     : {s['total_trades']}")
    logger.info(f"Wins / Losses    : {s['wins']} / {s['losses']}")
    logger.info(f"Win rate         : {s['win_rate'] * 100:.2f}%")
    logger.info(f"Gross PnL        : ${s['gross_pnl']:,.2f}")
    logger.info(f"Avg win          : ${s['avg_win']:,.2f}")
    logger.info(f"Avg loss         : ${s['avg_loss']:,.2f}")
    logger.info(f"Expectancy/trade : ${s['expectancy']:,.2f}")
    logger.info(f"Max loss streak  : {s['max_consecutive_losses']}")
    logger.info(f"Max drawdown     : {s['max_drawdown_pct']:.2f}% (${s.get('max_drawdown_usd', 0):,.2f})")
    logger.info(f"Max loss vs start: ${s.get('max_loss_from_start_usd', 0):,.2f}")
    logger.info(f"Starting equity  : ${s.get('starting_equity', 0):,.2f}")
    logger.info(f"Final equity     : ${s.get('final_equity', 0):,.2f}")
    logger.info(f"Return           : {s.get('return_pct', 0):.2f}%")
    logger.info("-" * 60)
    logger.info("By signal type:")
    for k, v in s["by_signal_type"].items():
        logger.info(f"  {k:10s} n={v['count']:3d} pnl=${v['pnl']:>10,.2f} wins={v['wins']}")
    logger.info("By session:")
    for k, v in s["by_session"].items():
        logger.info(f"  {k:18s} n={v['count']:3d} pnl=${v['pnl']:>10,.2f} wins={v['wins']}")
    logger.info("By close reason:")
    for k, v in s["by_close_reason"].items():
        logger.info(f"  {k:10s} n={v['count']:3d} pnl=${v['pnl']:>10,.2f}")
    logger.info("=" * 60)
