"""FTMO Demo pre-flight verification.

Forces ACTIVE_BROKER=FTMO, connects to MT5, prints a full health report
(account, broker, prop-firm detection, per-pair availability + spread,
loaded compliance rules), and optionally pings Telegram so the operator
gets a green-light heartbeat before the bot is armed.

Usage:
    python scripts/ftmo_preflight.py
    python scripts/ftmo_preflight.py --no-telegram
    python scripts/ftmo_preflight.py --expect-balance 10000

Exit code:
    0  — all checks passed
    1  — at least one verification failed (look for ❌ lines)
"""

from __future__ import annotations
import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows console default is cp1252 — reconfigure to utf-8 so emoji status
# markers render. Best-effort; falls back to ASCII if reconfigure fails.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

# FORCE FTMO before anything else imports settings — that's how settings.py
# picks up the active broker creds (it reads env at module import time).
os.environ["ACTIVE_BROKER"] = "FTMO"

# Now safe to import the rest.
from config.broker_config import get_active_credentials  # noqa: E402
from risk.prop_firm.detector import (  # noqa: E402
    AccountInfo as PFAccountInfo, PropFirmDetector,
)
from risk.prop_firm.rules import RULES_DB  # noqa: E402
from strategy.patterns import GRIFF_PAIRS  # noqa: E402


def _line(ok: bool, label: str, value: str = "") -> str:
    mark = "✅" if ok else "❌"
    return f"{mark} {label}" + (f": {value}" if value else "")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ftmo_preflight",
        description="FTMO Demo account verification (Phase 9 pre-go-live).",
    )
    p.add_argument(
        "--no-telegram", dest="telegram", action="store_false", default=True,
        help="Skip the success Telegram ping.",
    )
    p.add_argument(
        "--expect-balance", type=float, default=10_000.0,
        help="Balance to assert against (default 10000 for FTMO 10K demo).",
    )
    p.add_argument(
        "--balance-tolerance", type=float, default=200.0,
        help="USD tolerance around expected balance (default 200).",
    )
    return p.parse_args()


async def _telegram_ping(message: str) -> bool:
    from alerts.telegram_notifier import TelegramNotifier
    n = TelegramNotifier(
        token=os.environ.get("TELEGRAM_BOT_TOKEN"),
        chat_id=os.environ.get("TELEGRAM_CHAT_ID"),
    )
    if not n.enabled:
        print("• Telegram notifier disabled (no token/chat_id) — skipping ping.")
        return False
    return await n.send(message)


def main() -> int:
    args = _parse_args()
    all_ok = True
    print("=" * 72)
    print("FTMO Demo Pre-Flight — Phase 9")
    print("=" * 72)

    # ----- 1. credentials route via ACTIVE_BROKER
    try:
        creds = get_active_credentials()
    except Exception as exc:
        print(_line(False, "broker_config", f"{exc}"))
        return 1

    if creds.broker != "FTMO":
        print(_line(False, "ACTIVE_BROKER routing",
                    f"expected FTMO, got {creds.broker} (fallback?)"))
        return 1
    print(_line(True, "ACTIVE_BROKER", f"FTMO → login={creds.login} "
                                       f"server={creds.server}"))

    # ----- 2. connect MT5
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        print(_line(False, "MetaTrader5 import", str(exc)))
        return 1

    if not mt5.initialize(
        path=creds.path or None, login=creds.login,
        password=creds.password, server=creds.server, timeout=15_000,
    ):
        last = mt5.last_error()
        print(_line(False, "mt5.initialize", f"{last}"))
        return 1
    try:
        if not mt5.login(
            login=creds.login, password=creds.password,
            server=creds.server, timeout=15_000,
        ):
            print(_line(False, "mt5.login", f"{mt5.last_error()}"))
            return 1
        print(_line(True, "MT5 terminal initialised + logged in"))

        # ----- 3. account snapshot
        acct = mt5.account_info()
        if acct is None:
            print(_line(False, "account_info", f"{mt5.last_error()}"))
            return 1

        print("-" * 72)
        print(f"Account number  : {acct.login}")
        print(f"Server          : {acct.server}")
        print(f"Company         : {acct.company}")
        print(f"Currency        : {acct.currency}")
        print(f"Balance         : ${acct.balance:,.2f}")
        print(f"Equity          : ${acct.equity:,.2f}")
        print(f"Leverage        : 1:{acct.leverage}")
        print("-" * 72)

        # balance assertion
        bal_diff = abs(acct.balance - args.expect_balance)
        bal_ok = bal_diff <= args.balance_tolerance
        all_ok = all_ok and bal_ok
        print(_line(
            bal_ok,
            f"Balance within ±${args.balance_tolerance:.0f} of "
            f"${args.expect_balance:.0f}",
            f"diff=${bal_diff:.2f}",
        ))

        # currency assertion
        curr_ok = acct.currency.upper() == "USD"
        all_ok = all_ok and curr_ok
        print(_line(curr_ok, "Currency = USD", acct.currency))

        # ----- 4. prop-firm auto detection
        detector = PropFirmDetector()
        pf_info = PFAccountInfo(
            server=acct.server, company=getattr(acct, "company", ""),
            login=acct.login, balance=acct.balance,
        )
        detected = detector.detect_from_mt5(pf_info)
        pf_ok = detected == "ftmo_2step_challenge"
        all_ok = all_ok and pf_ok
        print(_line(
            pf_ok,
            "Prop firm auto-detected = ftmo_2step_challenge",
            str(detected),
        ))

        # ----- 5. compliance rules
        rules = RULES_DB.get(detected or "ftmo_2step_challenge")
        if rules is not None:
            print("-" * 72)
            print(f"Rule pack            : {rules.name}")
            print(f"Daily loss cap       : {rules.max_daily_loss_pct:.1f}%")
            print(f"Total loss cap       : {rules.max_total_loss_pct:.1f}%")
            print(f"Min trading days     : {rules.min_trading_days}")
            print(f"News blackout (min)  : "
                  f"{rules.news_blackout_minutes_before}/"
                  f"{rules.news_blackout_minutes_after}")
            print(f"Leverage (forex)     : 1:{rules.leverage_forex}")
            print(f"Tick scalp allowed   : {rules.tick_scalp_allowed}")
            print("-" * 72)
            rules_ok = (
                rules.max_daily_loss_pct == 5.0
                and rules.max_total_loss_pct == 10.0
                and rules.min_trading_days == 4
            )
            all_ok = all_ok and rules_ok
            print(_line(rules_ok, "Rule caps match FTMO 2-Step (5% / 10% / 4d)"))
        else:
            print(_line(False, "No rule pack for detected firm"))
            all_ok = False

        # ----- 6. per-pair availability + spread
        print("-" * 72)
        print("Griff pair availability:")
        per_pair_ok = True
        for pair in GRIFF_PAIRS:
            info = mt5.symbol_info(pair)
            if info is None:
                print(f"  ❌ {pair:8s} NOT FOUND in Market Watch")
                per_pair_ok = False
                continue
            if not info.visible:
                if not mt5.symbol_select(pair, True):
                    print(f"  ❌ {pair:8s} could not select")
                    per_pair_ok = False
                    continue
                info = mt5.symbol_info(pair)
            tradable = bool(getattr(info, "trade_mode", 4)) and info.visible
            tick = mt5.symbol_info_tick(pair)
            spread_pts = info.spread if info else 0
            spread_disp = (
                f"spread={spread_pts}pt" if spread_pts else "spread=n/a"
            )
            mark = "✅" if tradable else "❌"
            tick_age = (
                "tick OK" if tick and tick.time > 0
                else "no tick"
            )
            print(f"  {mark} {pair:8s} visible={info.visible} "
                  f"trade_mode={info.trade_mode} {spread_disp} {tick_age}")
            if not tradable:
                per_pair_ok = False
        all_ok = all_ok and per_pair_ok
        print(_line(per_pair_ok, "All 6 Griff pairs available + tradable"))

        # ----- 7. final marker
        print("=" * 72)
        if all_ok:
            print("✅ FTMO Demo Pre-Flight: ALL CHECKS PASSED")
        else:
            print("❌ FTMO Demo Pre-Flight: ONE OR MORE CHECKS FAILED")
        print("=" * 72)

        # ----- 8. Telegram ping (best-effort)
        if args.telegram:
            tg_msg = (
                f"<b>FTMO Pre-Flight</b> "
                f"{'✅' if all_ok else '❌'} "
                f"account={acct.login} balance=${acct.balance:,.0f} "
                f"detected={detected} pairs_ok={per_pair_ok}"
            )
            try:
                sent = asyncio.run(_telegram_ping(tg_msg))
                print(_line(sent, "Telegram ping sent"))
            except Exception as exc:  # noqa: BLE001
                print(_line(False, "Telegram ping", str(exc)))

        return 0 if all_ok else 1
    finally:
        try:
            mt5.shutdown()
            print("• MT5 terminal shut down cleanly")
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
