"""xau_hft_engine orchestrator. Live capture or offline replay, sharing one shutdown path."""

from __future__ import annotations
import argparse
import asyncio
import os
import signal
import sys
import time

from config.settings import settings
from data.mt5_connector import MT5Connector, MT5ConnectionError
from data.tick_collector import Tick, TickCollector
from data.tick_writer import TickWriter
from replay.integrity_checker import check_partition, log_report
from replay.replay_engine import ReplayConfig, ReplayEngine
from utils.logger import logger


QUEUE_MAX_SIZE = 10000
DRAIN_TIMEOUT_SEC = 10.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bot", description="xau_hft_engine")
    parser.add_argument(
        "--mode",
        choices=["live", "replay", "paper-backtest", "live-demo"],
        default="live",
        help=(
            "live: poll MT5. replay: emit ticks. paper-backtest: full sim w/ trades. "
            "live-demo: live MT5 → strategy → paper broker (Phase 6 demo)."
        ),
    )
    parser.add_argument(
        "--date", default=None,
        help="YYYY-MM-DD partition date (required for replay / paper-backtest).",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Replay speed: 0=max (no sleeps), 1.0=real-time, N>1=N× faster. Default 1.0.",
    )
    parser.add_argument(
        "--symbol", default=None,
        help="Override settings.symbol (defaults to .env SYMBOL).",
    )
    parser.add_argument(
        "--capital", type=float, default=10_000.0,
        help="Simulated starting capital for paper-backtest (default 10000).",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Print a periodic console dashboard in live-demo mode.",
    )
    args = parser.parse_args(argv)

    if args.mode in ("replay", "paper-backtest") and not args.date:
        parser.error(f"--date YYYY-MM-DD is required when --mode {args.mode}")
    if args.speed < 0:
        parser.error("--speed must be >= 0 (0 = max-speed)")
    return args


def _install_signal_handlers(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        if not stop.is_set():
            logger.info("shutdown signal received")
            stop.set()

    if sys.platform == "win32":
        def _win_handler(signum, frame):
            loop.call_soon_threadsafe(_request_stop)
        signal.signal(signal.SIGINT, _win_handler)
        signal.signal(signal.SIGTERM, _win_handler)
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _request_stop)


async def _cancel_and_wait(task: asyncio.Task, name: str) -> None:
    if task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.exception(f"{name} raised during shutdown: {exc}")


async def _drain_queue(queue: asyncio.Queue, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    while queue.qsize() > 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    return queue.qsize()


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

async def _run_live(stop: asyncio.Event, symbol: str | None) -> int:
    connector = MT5Connector(symbol=symbol) if symbol else MT5Connector()
    try:
        await asyncio.to_thread(connector.connect)
    except MT5ConnectionError as exc:
        logger.error(f"MT5 connect failed: {exc}")
        return 1

    queue: asyncio.Queue[Tick] = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    collector = TickCollector(connector, queue)
    writer = TickWriter(queue, symbol=connector.symbol)

    collector_task = asyncio.create_task(collector.run(), name="collector")
    writer_task = asyncio.create_task(writer.run(), name="writer")
    stop_task = asyncio.create_task(stop.wait(), name="stop")

    logger.info(
        f"xau_hft_engine LIVE | symbol={connector.symbol} "
        f"queue_max={QUEUE_MAX_SIZE} | press Ctrl+C to stop"
    )

    try:
        await asyncio.wait(
            {collector_task, writer_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        stop_task.cancel()
        for t, name in ((collector_task, "collector"), (writer_task, "writer")):
            if t.done() and not t.cancelled():
                exc = t.exception()
                if exc is not None:
                    logger.opt(exception=exc).error(f"{name} task ended unexpectedly")

        await _cancel_and_wait(collector_task, "collector")
        remaining = await _drain_queue(queue, DRAIN_TIMEOUT_SEC)
        if remaining > 0:
            logger.warning(f"drain timeout | {remaining} ticks may not reach disk")
        await _cancel_and_wait(writer_task, "writer")

        try:
            await asyncio.to_thread(connector.disconnect)
        except Exception as exc:
            logger.exception(f"MT5 disconnect failed: {exc}")

        logger.success(
            f"shutdown complete | collected={collector.collected} "
            f"dropped={collector.dropped} written={writer.written_ticks} "
            f"files={writer.written_files}"
        )
    return 0


# ---------------------------------------------------------------------------
# Replay mode
# ---------------------------------------------------------------------------

class _TickCounter:
    """Minimal consumer for replay: drain the queue, count, log progress."""

    def __init__(self, queue: asyncio.Queue[Tick]) -> None:
        self._queue = queue
        self._consumed = 0
        self._first_msc = 0
        self._last_msc = 0

    async def run(self) -> None:
        try:
            while True:
                tick = await self._queue.get()
                if self._consumed == 0:
                    self._first_msc = tick.time_msc
                self._last_msc = tick.time_msc
                self._consumed += 1
                if self._consumed % 5000 == 0:
                    logger.info(f"replay progress | consumed={self._consumed}")
        except asyncio.CancelledError:
            logger.info(
                f"TickCounter stopped | consumed={self._consumed} "
                f"first_msc={self._first_msc} last_msc={self._last_msc}"
            )
            raise

    @property
    def consumed(self) -> int:
        return self._consumed


async def _run_replay(stop: asyncio.Event, args: argparse.Namespace) -> int:
    symbol = args.symbol or settings.symbol

    report = check_partition(symbol, args.date)
    log_report(report)
    if report.row_count == 0:
        logger.error("Partition has no rows. Aborting.")
        return 1
    if not report.schema_match:
        logger.error("Schema mismatch. Aborting.")
        return 1
    if report.error_gaps:
        logger.warning(f"{len(report.error_gaps)} severe gap(s); continuing replay anyway")

    queue: asyncio.Queue[Tick] = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    config = ReplayConfig(symbol=symbol, date=args.date, speed=args.speed)
    engine = ReplayEngine(config, queue)
    counter = _TickCounter(queue)

    engine_task = asyncio.create_task(engine.run(), name="replay")
    counter_task = asyncio.create_task(counter.run(), name="counter")
    stop_task = asyncio.create_task(stop.wait(), name="stop")

    mode_label = "max-speed" if args.speed == 0.0 else f"{args.speed:.2f}x"
    logger.info(
        f"xau_hft_engine REPLAY | symbol={symbol} date={args.date} "
        f"mode={mode_label} | press Ctrl+C to stop"
    )

    try:
        await asyncio.wait(
            {engine_task, counter_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        stop_task.cancel()
        for t, name in ((engine_task, "replay"), (counter_task, "counter")):
            if t.done() and not t.cancelled():
                exc = t.exception()
                if exc is not None:
                    logger.opt(exception=exc).error(f"{name} task ended unexpectedly")

        await _cancel_and_wait(engine_task, "replay")
        remaining = await _drain_queue(queue, DRAIN_TIMEOUT_SEC)
        if remaining > 0:
            logger.warning(f"drain timeout | {remaining} ticks in queue at shutdown")
        await _cancel_and_wait(counter_task, "counter")

        logger.success(
            f"shutdown complete | emitted={engine.emitted} "
            f"skipped={engine.skipped} consumed={counter.consumed}"
        )
    return 0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> int:
    stop = asyncio.Event()
    _install_signal_handlers(stop)

    auto_stop_sec = float(os.getenv("BOT_AUTO_STOP_SEC", "0") or "0")
    if auto_stop_sec > 0:
        asyncio.get_running_loop().call_later(auto_stop_sec, stop.set)
        logger.info(f"auto-stop scheduled in {auto_stop_sec:.1f}s")

    if args.mode == "live":
        return await _run_live(stop, args.symbol)
    if args.mode == "paper-backtest":
        return await _run_paper_backtest(args)
    if args.mode == "live-demo":
        return await _run_live_demo(stop, args)
    return await _run_replay(stop, args)


async def _run_paper_backtest(args: argparse.Namespace) -> int:
    from backtesting.backtest_runner import run_backtest

    symbol = args.symbol or settings.symbol
    logger.info(
        f"xau_hft_engine PAPER-BACKTEST | symbol={symbol} date={args.date} "
        f"capital=${args.capital:,.2f}"
    )
    try:
        await run_backtest(args.date, speed=args.speed, capital=args.capital, symbol=symbol)
    except FileNotFoundError as exc:
        logger.error(f"backtest failed: {exc}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Live-demo mode (Phase 6)
# ---------------------------------------------------------------------------

async def _run_live_demo(stop: asyncio.Event, args: argparse.Namespace) -> int:
    """Live MT5 ticks → strategy → RiskEngine → PaperBroker → TradeJournal.

    Paper still — Phase 7 wires real order placement. The point of this mode
    is to expose live-vs-backtest divergence: real spread regime, real tick
    cadence, real session boundaries — but no broker risk.
    """
    from alerts import TelegramNotifier
    from backtesting.trade_journal import TradeJournal
    from backtesting.backtest_runner import _enrich, print_summary
    from execution.broker_factory import get_broker
    from execution.position import CloseReason
    from monitoring import ConsoleDashboard
    from risk.risk_engine import RiskEngine
    from strategy.microstructure import MicrostructureState
    from strategy.signal_confirmation import SignalConfirmationBuffer
    from strategy.signals import (
        LiquiditySweepDetector,
        RejectionDetector,
        Signal,
        TickMomentumDetector,
    )

    symbol = args.symbol or settings.symbol
    connector = MT5Connector(symbol=symbol)
    try:
        await asyncio.to_thread(connector.connect)
    except MT5ConnectionError as exc:
        logger.error(f"MT5 connect failed: {exc}")
        return 1

    queue: asyncio.Queue[Tick] = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    collector = TickCollector(connector, queue)

    state = MicrostructureState()
    sweep = LiquiditySweepDetector(broker_type=settings.broker_type)
    momentum = TickMomentumDetector(broker_type=settings.broker_type)
    rejection = RejectionDetector(broker_type=settings.broker_type)
    confirmer = SignalConfirmationBuffer()
    risk = RiskEngine(settings)
    broker = get_broker(settings)
    journal = TradeJournal()
    notifier = TelegramNotifier(
        settings.telegram_bot_token, settings.telegram_chat_id
    )
    dashboard = (
        ConsoleDashboard(
            mode=settings.execution_mode, symbol=symbol,
            risk=risk, state=state, settings=settings,
        )
        if getattr(args, "dashboard", False)
        else None
    )
    blocked_reasons_alerted: set[str] = set()
    await notifier.notify_bot_started(settings.execution_mode, symbol)

    consumed = 0
    signals_seen = 0
    blocked = 0
    prev_mid: float | None = None
    last_tick: Tick | None = None

    async def _strategy_loop() -> None:
        nonlocal consumed, signals_seen, blocked, prev_mid, last_tick
        while True:
            tick = await queue.get()
            last_tick = tick
            consumed += 1
            enriched, prev_mid = _enrich(tick, prev_mid)
            state.update(enriched)

            if dashboard is not None:
                dashboard.update_tick(tick, queue.qsize(), QUEUE_MAX_SIZE)

            if risk.open_positions:
                pos = risk.open_positions[0]
                closed = await broker.check_position_exit(pos, tick)
                if closed is not None:
                    risk.register_close(closed)
                    journal.log_close(closed)
                    logger.info(
                        f"CLOSE {closed.signal_type} {closed.close_reason.value} "
                        f"pnl=${closed.pnl_usd:.2f}"
                    )
                    asyncio.create_task(notifier.notify_trade_close(closed))

            confirmed_now: list[Signal] = list(
                confirmer.on_tick(enriched["mid_change_pts"])
            )
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
                if dashboard is not None:
                    dashboard.update_signal(sig, tick.time_msc)
                ok, reason = risk.can_open_position(sig, tick)
                if not ok:
                    blocked += 1
                    logger.debug(f"signal blocked: {reason}")
                    if reason in ("daily_cap_hit", "loss_streak_pause") \
                            and reason not in blocked_reasons_alerted:
                        blocked_reasons_alerted.add(reason)
                        asyncio.create_task(notifier.notify_circuit_breaker(reason))
                    continue
                intent = risk.build_order_intent(sig, tick)
                pos = await broker.fill_market_order(intent, tick)
                risk.register_open(pos)
                journal.log_open(pos)
                logger.info(
                    f"OPEN {sig.type.value} {pos.side.value} "
                    f"@{pos.entry_price:.2f} sl={pos.sl_price:.2f} tp={pos.tp_price:.2f}"
                )
                asyncio.create_task(notifier.notify_trade_open(pos))

    collector_task = asyncio.create_task(collector.run(), name="collector")
    strategy_task = asyncio.create_task(_strategy_loop(), name="strategy")
    stop_task = asyncio.create_task(stop.wait(), name="stop")
    dashboard_task = (
        asyncio.create_task(dashboard.run(stop), name="dashboard")
        if dashboard is not None
        else None
    )

    logger.info(
        f"xau_hft_engine LIVE-DEMO | symbol={symbol} broker={settings.broker_type} "
        f"account={settings.account_type} capital=${settings.simulated_starting_capital:,.2f} "
        f"| mode={settings.execution_mode} | Ctrl+C to stop"
    )

    waitable = {collector_task, strategy_task, stop_task}
    try:
        await asyncio.wait(waitable, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop_task.cancel()
        for t, name in ((collector_task, "collector"), (strategy_task, "strategy")):
            if t.done() and not t.cancelled():
                exc = t.exception()
                if exc is not None:
                    logger.opt(exception=exc).error(f"{name} task ended unexpectedly")

        await _cancel_and_wait(collector_task, "collector")
        await _cancel_and_wait(strategy_task, "strategy")
        if dashboard_task is not None:
            await _cancel_and_wait(dashboard_task, "dashboard")

        # Force-close any open position at the last tick we saw.
        if last_tick is not None and risk.open_positions:
            for pos in list(risk.open_positions):
                closed = await broker.force_close(pos, last_tick, reason=CloseReason.EOD)
                risk.register_close(closed)
                journal.log_close(closed)
                try:
                    await notifier.notify_trade_close(closed)
                except Exception:  # noqa: BLE001
                    pass

        try:
            await notifier.notify_bot_stopped("graceful")
        except Exception:  # noqa: BLE001
            pass

        try:
            await asyncio.to_thread(connector.disconnect)
        except Exception as exc:
            logger.exception(f"MT5 disconnect failed: {exc}")

        summary = journal.summary()
        summary["ticks_consumed"] = consumed
        summary["signals_seen"] = signals_seen
        summary["signals_blocked"] = blocked
        summary["final_equity"] = risk.account_equity
        summary["starting_equity"] = settings.simulated_starting_capital
        cap = settings.simulated_starting_capital
        summary["return_pct"] = (
            (risk.account_equity - cap) / cap * 100.0 if cap > 0 else 0.0
        )
        print_summary(summary)
    return 0


def main() -> int:
    args = _parse_args()
    # Branded startup banner. Mode label maps the CLI --mode onto the
    # PAPER/DRY_RUN/REAL palette used by the console dashboard so the
    # colour cue is consistent across entry points.
    try:
        from monitoring.banner import print_banner
        mode_label = {
            "live": "REAL",
            "live-demo": "PAPER",
            "paper-backtest": "PAPER",
            "replay": "DRY_RUN",
        }.get(args.mode, "DRY_RUN")
        print_banner(mode=mode_label, pairs=(args.symbol,) if args.symbol else None)
    except Exception:
        # Banner is cosmetic — never let it block bot startup.
        pass
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.warning("interrupted before signal handler engaged")
        return 130


if __name__ == "__main__":
    sys.exit(main())
