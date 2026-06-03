"""Asian Sweep V5 live-trading CLI entry point.

DRY_RUN is the DEFAULT. To send real orders to MT5 you must pass
`--no-dry-run` AND `EXECUTION_MODE=REAL` must be set in `.env`. Same
two-key safety as the Griff entrypoint — single-fault → log-only.

Usage:
    python scripts/run_asian_sweep_live.py                        # dry run, default
    python scripts/run_asian_sweep_live.py --dry-run              # explicit
    python scripts/run_asian_sweep_live.py --no-dry-run           # arms LIVE
    python scripts/run_asian_sweep_live.py --pairs XAUUSD,EURUSD  # subset
    python scripts/run_asian_sweep_live.py --no-dashboard         # skip HTTP server
    python scripts/run_asian_sweep_live.py --no-telegram          # disable alerts
    python scripts/run_asian_sweep_live.py --once                 # one scan then exit

Default `--history-bars 250` so EMA200 HTF-bias has its full window from
the first scan (Asian Sweep needs > 200 closes of history to leave neutral).

Hinglish: Griff script ka twin — Asian Sweep V5 ke detector factory aur
engine class plugged in. Saari safety, dashboard, telegram, MT5 wiring
intact rakhi gayi hai.
"""

from __future__ import annotations
import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional, Sequence

# Ensure project root is importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alerts.telegram_notifier import TelegramNotifier
from config.asian_sweep_config import PAIRS as ASIAN_SWEEP_PAIRS
from config.credential_prompt import (
    ensure_active_credentials,
    get_active_profile,
)
from data.live_bar_poller import LiveBarPoller
from data.news_calendar import StaticNewsCalendar
from execution.order_router import GriffOrderRouter
from execution.position_manager import GriffPositionManager
from execution.live_engine import AsianSweepLiveEngine
from monitoring.banner import print_banner
from monitoring.console_dashboard import ConsoleDashboard
from monitoring.daily_tracker import DailyTracker
from monitoring.dashboard import GriffDashboard
from monitoring.telegram_alerts import GriffTelegramAlerts
from monitoring.hourly_reporter import HourlyReporter, HourlyStats
from risk.house_money import HouseMoneyManager
from risk.prop_firm.compliance import AccountState, ComplianceEngine
from risk.prop_firm.rules import RULES_DB
from risk.trailing_sl import TrailingStopLoss
from strategy.asian_sweep_config import build_asian_sweep_detector
from strategy.scanner import Scanner
from strategy.swing_tracker import SwingTracker
from utils.logger import logger


DEFAULT_DASHBOARD_PORT = 8080
# EMA200 bias needs ≥ 200 prior closes + ~24h of intraday before signals
# stabilise. 250 gives the scanner a working bias on the first scan.
DEFAULT_HISTORY_BARS = 250


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_asian_sweep_live",
        description="Live Asian Range London Sweep V5 bot — DRY_RUN by default.",
    )
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   default=True,
                   help="Log orders without placing (default).")
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                   help="Arm LIVE orders. Requires EXECUTION_MODE=REAL in env.")
    p.add_argument("--pairs", default=",".join(ASIAN_SWEEP_PAIRS),
                   help=f"Comma-separated pair list. Default: {','.join(ASIAN_SWEEP_PAIRS)}.")
    p.add_argument("--no-dashboard", dest="dashboard", action="store_false",
                   default=True, help="Skip the localhost:8080 HTTP dashboard.")
    p.add_argument("--dashboard-port", type=int, default=DEFAULT_DASHBOARD_PORT)
    p.add_argument("--no-console-dashboard", dest="console_dashboard",
                   action="store_false", default=True,
                   help="Skip the periodic ANSI console dashboard panel "
                        "(useful when redirecting logs to a file).")
    p.add_argument("--no-telegram", dest="telegram", action="store_false",
                   default=True, help="Disable Telegram alerts.")
    p.add_argument("--once", action="store_true",
                   help="Run one scan cycle and exit (smoke test).")
    p.add_argument("--duration",
                   help="Auto-stop after N (e.g. '5min', '2h', '90s'). Smoke-test only.")
    p.add_argument("--starting-equity", type=float, default=10_000.0)
    p.add_argument("--no-hourly", dest="hourly", action="store_false",
                   default=True, help="Disable periodic Telegram status digests.")
    p.add_argument("--report-interval-min", type=float, default=15.0,
                   help="Status digest cadence in minutes (default 15).")
    p.add_argument("--poll-sec", type=float, default=30.0,
                   help="LIVE bar-poll interval (default 30s). LIVE mode only.")
    p.add_argument("--history-bars", type=int, default=DEFAULT_HISTORY_BARS,
                   help=f"H1 history fed to the scanner (default {DEFAULT_HISTORY_BARS}; "
                        f"EMA200 bias needs >= 200).")
    p.add_argument("--reset", action="store_true",
                   help="Re-enter MT5 credentials for the active broker profile "
                        "and overwrite .env.")
    p.add_argument("--switch", metavar="BROKER",
                   help="Switch ACTIVE_BROKER to the named profile (e.g. FTMO, "
                        "THE5ERS). Creates the profile interactively if missing.")
    p.add_argument("--no-prompt", action="store_true",
                   help="Fail instead of prompting when credentials are missing "
                        "(non-interactive / CI mode).")
    return p.parse_args(argv)


_DURATION_UNITS = {
    "s": 1, "sec": 1, "secs": 1,
    "m": 60, "min": 60, "mins": 60,
    "h": 3600, "hr": 3600, "hrs": 3600,
}


def parse_duration(raw: Optional[str]) -> Optional[float]:
    """Parse '5min' / '90s' / '2h' → seconds. None passes through."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    i = 0
    while i < len(s) and (s[i].isdigit() or s[i] == "."):
        i += 1
    if i == 0:
        raise ValueError(f"Invalid --duration: {raw!r}")
    n = float(s[:i])
    unit = s[i:] or "s"
    if unit not in _DURATION_UNITS:
        raise ValueError(
            f"Invalid --duration unit {unit!r}; allowed: {sorted(_DURATION_UNITS)}"
        )
    return n * _DURATION_UNITS[unit]


def safety_gate(args: argparse.Namespace) -> tuple[bool, str]:
    """Returns (effective_dry_run, reason). Two-key safety — same as Griff."""
    if args.dry_run:
        return True, "cli_default_dry_run"
    env_mode = os.environ.get("EXECUTION_MODE", "PAPER").upper()
    if env_mode != "REAL":
        return True, f"env_EXECUTION_MODE={env_mode}_not_REAL"
    return False, "live_armed"


async def main_async(args: argparse.Namespace) -> int:
    pairs = tuple(s.strip() for s in args.pairs.split(",") if s.strip())
    dry_run, reason = safety_gate(args)

    # Credential gate. LIVE always demands creds. DRY_RUN: prompt if the
    # profile is missing (or --reset/--switch asked) and the caller hasn't
    # forbidden prompts. Without this, the polished banner would show
    # "(no broker yet)" forever.
    profile = None
    needs_creds = (
        not dry_run
        or args.reset
        or args.switch
        or (get_active_profile() is None and not args.no_prompt)
    )
    if needs_creds:
        profile = ensure_active_credentials(
            reset=args.reset,
            switch_to=args.switch,
            interactive=not args.no_prompt,
        )
        # Hot-reload .env into os.environ (settings module may already
        # have been imported transitively — utils.logger pulls it in to
        # learn LOG_DIR — so we rebuild the singleton too).
        from dotenv import load_dotenv as _reload_env
        from pathlib import Path as _Path
        _reload_env(_Path(__file__).resolve().parents[1] / ".env", override=True)
        # Bug #6: `import config.settings as X` resolves to the Settings
        # dataclass instance (because `config/__init__.py` does
        # `from .settings import settings`, which makes `from config
        # import settings as X` — the equivalent of `import config.settings
        # as X` for submodules — return the attribute rather than the
        # module). `importlib.import_module` bypasses that and always
        # returns the actual module from sys.modules.
        import importlib
        _settings_module = importlib.import_module("config.settings")
        _settings_module.settings = _settings_module._build()
        logger.info(
            f"AsianSweepLive using broker profile {profile.name} "
            f"(label={profile.label or profile.name!r}) "
            f"login={profile.login} server={profile.server}"
        )
    else:
        # DRY_RUN with --no-prompt: still try to read whatever profile
        # exists so the banner displays it; don't fail when none.
        profile = get_active_profile()
    from config.settings import settings  # noqa: F401

    # ── Branded startup banner ──────────────────────────────────────────
    mode_label = (
        "REAL" if not dry_run else (
            "PAPER" if os.environ.get("EXECUTION_MODE", "").upper() == "PAPER"
            else "DRY_RUN"
        )
    )
    print_banner(
        broker_label=getattr(profile, "label", None) or (
            getattr(profile, "name", None) if profile else None),
        broker_name=getattr(profile, "name", None) if profile else None,
        account=getattr(profile, "login", None) if profile else None,
        server=getattr(profile, "server", None) if profile else None,
        mode=mode_label,
        pairs=pairs,
    )

    logger.info(
        f"AsianSweepLive starting dry_run={dry_run} (safety_reason={reason}) "
        f"pairs={pairs}"
    )

    # Build deps. Detector factory swapped — pipeline identical to Griff.
    scanner = Scanner(pairs, build_asian_sweep_detector())
    router = GriffOrderRouter(dry_run=dry_run)
    swing_tracker = SwingTracker()
    trail = TrailingStopLoss(swing_tracker)
    pm = GriffPositionManager(router, swing_tracker, trail)
    rules = RULES_DB.get("ftmo_2step_challenge")
    # Real news calendar — without this, ComplianceEngine falls back to an
    # empty StaticNewsCalendar (no events) and the news blackout becomes a
    # no-op. The5%ers / FTMO rule: no trades within ±2 min of HIGH-impact
    # events; the calendar's DEFAULT_HIGH_IMPACT_EVENTS provides the curated
    # list (NFP / CPI / FOMC / BoE rate / etc.).
    news_calendar = StaticNewsCalendar()
    # V5 max-2-trades/day is honoured here; per-direction cap is enforced
    # by the per-pair-best dedupe inside the engine (one signal per pair).
    compliance = ComplianceEngine(
        rules, max_trades_per_day=2, news_calendar=news_calendar
    )
    house = HouseMoneyManager()
    daily = DailyTracker(starting_equity=args.starting_equity)
    notifier = TelegramNotifier(
        token=os.environ.get("TELEGRAM_BOT_TOKEN") if args.telegram else None,
        chat_id=os.environ.get("TELEGRAM_CHAT_ID") if args.telegram else None,
    )
    alerts = GriffTelegramAlerts(notifier, bot_label="AsianSweep")

    stats = HourlyStats() if args.hourly else None
    engine = AsianSweepLiveEngine(
        scanner=scanner, router=router, position_mgr=pm,
        compliance=compliance, house_money=house, daily=daily, alerts=alerts,
        hourly_stats=stats,
        news_calendar=news_calendar,
    )

    reporter: Optional[HourlyReporter] = None
    if args.hourly and stats is not None:
        reporter = HourlyReporter(
            notifier=notifier, daily=daily, position_mgr=pm,
            stats=stats, num_pairs=len(pairs),
            daily_loss_cap_pct=float(rules.max_daily_loss_pct) if rules else 5.0,
            starting_equity=args.starting_equity,
        )

    # Optional dashboard.
    dash = None
    if args.dashboard:
        dash = GriffDashboard(
            pm, daily,
            signals_provider=lambda: list(scanner.last_signals),
            health_provider=lambda: {
                "ok": True, "mt5_connected": not dry_run,
                "last_bar_ms": daily.state.last_update_ms,
            },
        )
        await dash.start(host="127.0.0.1", port=args.dashboard_port)
        logger.info(f"AsianSweepDashboard http://127.0.0.1:{args.dashboard_port}/")

    # LIVE: connect MT5 + subscribe pairs.
    mt5_module = None
    live_account_balance: Optional[float] = None
    live_account_currency: Optional[str] = None
    live_broker_name: Optional[str] = None
    live_prop_firm_key: Optional[str] = None
    if not dry_run:
        try:
            import MetaTrader5 as _mt5  # noqa: N814
        except ImportError as exc:
            logger.error(f"MetaTrader5 not importable: {exc}")
            return 1
        mt5_module = _mt5
        if not mt5_module.initialize(
            path=settings.mt5_path,
            login=settings.mt5_login,
            password=settings.mt5_password,
            server=settings.mt5_server,
            timeout=15_000,
        ):
            logger.error(f"mt5.initialize failed: {mt5_module.last_error()}")
            return 1
        if not mt5_module.login(
            login=settings.mt5_login,
            password=settings.mt5_password,
            server=settings.mt5_server,
            timeout=15_000,
        ):
            logger.error(f"mt5.login failed: {mt5_module.last_error()}")
            mt5_module.shutdown()
            return 1
        for pair in pairs:
            info = mt5_module.symbol_info(pair)
            if info is None or not info.visible:
                if not mt5_module.symbol_select(pair, True):
                    logger.warning(
                        f"LIVE: could not select {pair} in Market Watch"
                    )
        acct = mt5_module.account_info()
        if acct is not None:
            logger.success(
                f"MT5 LIVE connected — login={acct.login} "
                f"server={acct.server} balance=${acct.balance:.2f} "
                f"currency={acct.currency} leverage=1:{acct.leverage}"
            )
            live_account_balance = float(acct.balance)
            live_account_currency = str(acct.currency)
            from config.broker_config import active_broker_name
            from risk.prop_firm.detector import (
                AccountInfo as _PFAcct, PropFirmDetector,
            )
            live_broker_name = active_broker_name()
            live_prop_firm_key = PropFirmDetector().detect_from_mt5(
                _PFAcct(server=acct.server, company=getattr(acct, "company", ""),
                        login=acct.login, balance=acct.balance)
            )

    await alerts.bot_started(
        dry_run=dry_run, pairs=pairs,
        broker_name=live_broker_name,
        prop_firm_key=live_prop_firm_key,
        account_balance=live_account_balance,
        account_currency=live_account_currency,
    )

    try:
        if args.once:
            logger.info("AsianSweepLive --once: smoke scan, then exit")
            await engine.process_scan_cycle(
                {p: [] for p in pairs},
                now_msc=daily.state.last_update_ms,
                ask_by_pair={p: 0.0 for p in pairs},
                bid_by_pair={p: 0.0 for p in pairs},
                account=AccountState(
                    equity=args.starting_equity,
                    starting_equity=args.starting_equity,
                    daily_start_equity=args.starting_equity,
                    daily_pnl_usd=0.0, trades_today=0,
                ),
            )
            return 0

        stop = asyncio.Event()
        _install_signal_handlers(stop)

        duration_sec = parse_duration(args.duration)
        if duration_sec is not None:
            asyncio.get_running_loop().call_later(
                duration_sec, stop.set,
            )
            logger.info(f"AsianSweepLive auto-stop in {duration_sec:.0f}s")

        bg_tasks: list[asyncio.Task] = []
        if reporter is not None:
            import time as _time
            report_interval_ms = max(1, int(args.report_interval_min * 60_000))
            bg_tasks.append(asyncio.create_task(
                reporter.run_periodic(
                    stop, clock_ms=lambda: int(_time.time() * 1000),
                    interval_ms=report_interval_ms,
                )
            ))
            logger.info(
                f"AsianSweepLive status reporter scheduled "
                f"(every {args.report_interval_min:g}m)"
            )

        # ── Console dashboard ──────────────────────────────────────────
        # Display-only — pulls from pm/daily/(optional)MT5. Refresh cadence
        # matches the bar poller so a fresh panel lands right after each
        # scan cycle. Skipped via --no-console-dashboard for headless runs.
        console = None
        if args.console_dashboard:
            def _dash_account_provider():
                if mt5_module is None:
                    return {"balance": args.starting_equity,
                            "equity": args.starting_equity}
                a = mt5_module.account_info()
                if a is None:
                    return {"balance": args.starting_equity,
                            "equity": args.starting_equity}
                return {
                    "balance": float(a.balance),
                    "equity": float(a.equity),
                    "currency": str(getattr(a, "currency", "USD")),
                }

            def _dash_spread_provider():
                if mt5_module is None:
                    return {}
                out: dict[str, float] = {}
                for pair in pairs:
                    tick = mt5_module.symbol_info_tick(pair)
                    info = mt5_module.symbol_info(pair)
                    if tick is None or info is None:
                        continue
                    pts = max(0.0, (float(tick.ask) - float(tick.bid)) / info.point)
                    out[pair] = pts
                return out

            def _dash_status_provider():
                if compliance.emergency_stopped:
                    return (f"Halted: {compliance.emergency_reason}", "halt")
                if pm.open_positions:
                    return ("Position open", "ok")
                if daily.trade_count >= 2:
                    return ("Idle — daily trade cap reached", "warn")
                return ("Scanning", "ok")

            broker_label_disp = (
                getattr(profile, "label", None)
                or getattr(profile, "name", None)
                or "—"
            )
            account_disp = str(getattr(profile, "login", "—") or "—")
            console = ConsoleDashboard(
                position_manager=pm,
                daily=daily,
                pairs=pairs,
                starting_equity=args.starting_equity,
                broker_label=broker_label_disp,
                account=account_disp,
                mode=mode_label,
                daily_cap_pct=float(rules.max_daily_loss_pct) if rules else 5.0,
                max_trades_per_day=2,
                account_provider=_dash_account_provider,
                spread_provider=_dash_spread_provider,
                status_provider=_dash_status_provider,
                refresh_sec=max(5.0, float(args.poll_sec)),
            )
            bg_tasks.append(asyncio.create_task(
                console.run(stop), name="console_dashboard",
            ))
            logger.info(
                f"AsianSweepLive ConsoleDashboard refresh="
                f"{console._refresh_sec:.0f}s"
            )

        if mt5_module is not None:
            poller = LiveBarPoller(
                pairs=pairs, mt5_module=mt5_module,
                history_bars=args.history_bars, poll_sec=args.poll_sec,
            )

            def _account_provider():
                a = mt5_module.account_info()
                eq = float(a.equity) if a else args.starting_equity
                bal = float(a.balance) if a else args.starting_equity
                return AccountState(
                    equity=eq, starting_equity=args.starting_equity,
                    daily_start_equity=bal,
                    daily_pnl_usd=daily.state.closed_pnl,
                    trades_today=daily.trade_count,
                    open_position_count=len(pm.open_positions),
                )

            def _prices_provider():
                ask = {}
                bid = {}
                for pair in pairs:
                    tick = mt5_module.symbol_info_tick(pair)
                    if tick is None:
                        continue
                    ask[pair] = float(tick.ask)
                    bid[pair] = float(tick.bid)
                return ask, bid

            bg_tasks.append(asyncio.create_task(
                poller.run(
                    engine=engine, stop=stop,
                    account_provider=_account_provider,
                    prices_provider=_prices_provider,
                )
            ))
            logger.info(
                f"AsianSweepLive LIVE bar poller scheduled "
                f"(every {args.poll_sec:.0f}s, {args.history_bars} bars history)"
            )

        logger.info("AsianSweepLive idle loop: awaiting SIGINT (Ctrl+C)")
        await stop.wait()
        for t in bg_tasks:
            t.cancel()
        for t in bg_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        return 0
    finally:
        await alerts.bot_stopped()
        if dash is not None:
            await dash.stop()
        if mt5_module is not None:
            try:
                mt5_module.shutdown()
                logger.info("MT5 LIVE disconnected")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"MT5 shutdown error: {exc}")
        daily.persist()
        logger.info("AsianSweepLive shutdown complete")


def _install_signal_handlers(stop: asyncio.Event) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    def _request_stop():
        if not stop.is_set():
            stop.set()

    if sys.platform == "win32":
        signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(_request_stop))
        signal.signal(signal.SIGTERM, lambda *_: loop.call_soon_threadsafe(_request_stop))
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _request_stop)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
