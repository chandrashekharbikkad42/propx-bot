"""Scenario runner — wires the REAL propX pipeline for E2E tests.

This is the heart of the Phase 6 integration suite. It composes the same
objects the live entry point (`scripts/run_asian_sweep_live.py`) builds,
but with two swaps:

  - `GriffOrderRouter(dry_run=True)`  — no MT5; orders return synthetic
                                         `GriffOpenPosition` / `GriffPendingOrder`
                                         with ticket=-1.
  - `AsyncMock` Telegram notifier      — no HTTP; alert spies are inspected
                                         to assert messaging behaviour.

Everything else is real: `AsianSweepLiveEngine`, `ComplianceEngine`,
`HouseMoneyManager`, `DailyTracker`, `Scanner` w/ `AsianSweepDetector`,
`GriffPositionManager` (running `maintain()` against real `SwingTracker`
+ `TrailingStopLoss`).

The runner captures every side effect (orders placed, positions opened,
maintenance reports, alerts emitted, compliance rejections) into a
`ScenarioResult` dataclass so tests assert end-state without re-implementing
the engine's invariants.

Hinglish: ek aisi knob — bars de do, account de do, engine ki real wiring
chala ke result le lo. Test sirf "kya hua" check karta hai, "kaise hua"
nahi.
"""

from __future__ import annotations
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from unittest.mock import AsyncMock, MagicMock

# Make repo importable when pytest is launched from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alerts.telegram_notifier import TelegramNotifier
from config.asian_sweep_config import PAIRS as ASIAN_SWEEP_PAIRS
from data.bar_aggregator import Bar
from data.news_calendar import StaticNewsCalendar
from execution.live_engine import AsianSweepLiveEngine, CycleReport
from execution.order_router import GriffOpenPosition, GriffOrderRouter
from execution.position_manager import GriffPositionManager, MaintenanceReport
from monitoring.daily_tracker import DailyTracker
from monitoring.telegram_alerts import GriffTelegramAlerts
from risk.house_money import HouseMoneyManager
from risk.prop_firm.compliance import AccountState, ComplianceEngine
from risk.prop_firm.rules import RULES_DB
from risk.trailing_sl import TrailingStopLoss
from strategy.asian_sweep_config import build_asian_sweep_detector
from strategy.scanner import Scanner
from strategy.swing_tracker import SwingTracker


def make_notifier_mock() -> MagicMock:
    """Mock TelegramNotifier that records every send() call but returns False."""
    m = MagicMock(spec=TelegramNotifier)
    m.enabled = True  # so GriffTelegramAlerts treats it as live
    m.send = AsyncMock(return_value=True)
    m.notify_trade_open = AsyncMock(return_value=True)
    m.notify_trade_close = AsyncMock(return_value=True)
    m.notify_circuit_breaker = AsyncMock(return_value=True)
    m.notify_bot_started = AsyncMock(return_value=True)
    m.notify_bot_stopped = AsyncMock(return_value=True)
    return m


@dataclass
class ScenarioResult:
    """Captured side effects of a full scenario run.

    Tests assert against this snapshot rather than poking the engine
    internals directly.
    """
    cycle_reports: List[CycleReport] = field(default_factory=list)
    maintenance_reports: List[Dict[str, MaintenanceReport]] = field(default_factory=list)
    positions_opened: List[GriffOpenPosition] = field(default_factory=list)
    closed_position_ids: List[str] = field(default_factory=list)
    alert_calls: List[str] = field(default_factory=list)         # method names
    alert_kill_switch_reasons: List[str] = field(default_factory=list)
    final_open_positions: Tuple[GriffOpenPosition, ...] = ()
    final_daily_state: Optional[Any] = None
    final_account_equity: float = 0.0
    compliance_status: Dict[str, Any] = field(default_factory=dict)

    # Convenience accessors --------------------------------------------------

    @property
    def total_orders_placed(self) -> int:
        return sum(r.orders_placed for r in self.cycle_reports)

    @property
    def total_signals_emitted(self) -> int:
        return sum(r.signals_emitted for r in self.cycle_reports)

    @property
    def total_compliance_rejections(self) -> int:
        return sum(r.signals_rejected_by_compliance for r in self.cycle_reports)

    @property
    def all_rejection_reasons(self) -> List[str]:
        out: List[str] = []
        for r in self.cycle_reports:
            out.extend(reason for _, reason in r.rejections)
        return out

    @property
    def total_closed(self) -> int:
        return len(self.closed_position_ids)


@dataclass
class ScenarioRunner:
    """Composed live-engine wiring + capture buffers.

    Usage:
        runner = ScenarioRunner.build()
        result = runner.run_cycle(bar_feeds, now_msc, ask, bid, account)
        assert result.total_orders_placed == 1
    """
    engine: AsianSweepLiveEngine
    router: GriffOrderRouter
    pm: GriffPositionManager
    compliance: ComplianceEngine
    house: HouseMoneyManager
    daily: DailyTracker
    alerts: GriffTelegramAlerts
    notifier_mock: MagicMock
    scanner: Scanner
    swing_tracker: SwingTracker
    trail: TrailingStopLoss
    result: ScenarioResult = field(default_factory=ScenarioResult)

    @classmethod
    def build(
        cls,
        *,
        pairs: Sequence[str] = ASIAN_SWEEP_PAIRS,
        rules_key: str = "ftmo_2step_challenge",
        starting_equity: float = 100_000.0,
        max_trades_per_day: int = 2,
        ist_window_start: str = "00:00",
        ist_window_end: str = "23:59",
        news_events: Optional[Sequence[Any]] = None,
        safety_margin_pct: float = 0.80,
        now_ms: int = 1_700_000_000_000,
    ) -> "ScenarioRunner":
        """Build a real-engine runner with sensible test defaults.

        Defaults open the IST window for the entire day so window logic is
        opt-in (tests that want to assert window blocking pass explicit
        bounds). News calendar defaults to an empty list — caller injects
        events for blackout assertions.
        """
        scanner = Scanner(pairs, build_asian_sweep_detector())
        router = GriffOrderRouter(dry_run=True)
        swing_tracker = SwingTracker()
        trail = TrailingStopLoss(swing_tracker)
        pm = GriffPositionManager(router, swing_tracker, trail)
        news_calendar = StaticNewsCalendar(list(news_events) if news_events else [])
        rules = RULES_DB[rules_key]
        compliance = ComplianceEngine(
            rules,
            max_trades_per_day=max_trades_per_day,
            ist_window_start=ist_window_start,
            ist_window_end=ist_window_end,
            news_calendar=news_calendar,
            safety_margin_pct=safety_margin_pct,
        )
        house = HouseMoneyManager()
        daily = DailyTracker(starting_equity=starting_equity, now_ms=now_ms)
        notifier_mock = make_notifier_mock()
        alerts = GriffTelegramAlerts(notifier_mock, bot_label="propX_test")
        engine = AsianSweepLiveEngine(
            scanner=scanner, router=router, position_mgr=pm,
            compliance=compliance, house_money=house, daily=daily,
            alerts=alerts, news_calendar=news_calendar,
        )
        runner = cls(
            engine=engine, router=router, pm=pm, compliance=compliance,
            house=house, daily=daily, alerts=alerts,
            notifier_mock=notifier_mock, scanner=scanner,
            swing_tracker=swing_tracker, trail=trail,
        )
        runner.result.final_account_equity = starting_equity
        return runner

    # ---------------------------------------------------------- core operations

    def run_cycle(
        self,
        bar_feeds: Mapping[str, Sequence[Bar]],
        *,
        now_msc: int,
        ask_by_pair: Optional[Mapping[str, float]] = None,
        bid_by_pair: Optional[Mapping[str, float]] = None,
        account: Optional[AccountState] = None,
    ) -> CycleReport:
        """Run one scan cycle, capture orders, alerts, rejections."""
        if account is None:
            account = self.default_account()
        if ask_by_pair is None:
            ask_by_pair = self._derive_prices(bar_feeds, side="ask")
        if bid_by_pair is None:
            bid_by_pair = self._derive_prices(bar_feeds, side="bid")
        before = set(p.position_id for p in self.pm.open_positions)
        report = asyncio.run(self.engine.process_scan_cycle(
            bar_feeds=bar_feeds, now_msc=now_msc,
            ask_by_pair=ask_by_pair, bid_by_pair=bid_by_pair,
            account=account,
        ))
        self.result.cycle_reports.append(report)
        after = self.pm.open_positions
        for pos in after:
            if pos.position_id not in before:
                self.result.positions_opened.append(pos)
        self._capture_alerts()
        self.result.final_open_positions = self.pm.open_positions
        self.result.final_daily_state = self.daily.state
        return report

    def run_maintenance(
        self, latest_bar_per_pair: Mapping[str, Bar], *, now_msc: int,
    ) -> Dict[str, MaintenanceReport]:
        """Run per-pair maintenance, capture closes."""
        reports = asyncio.run(
            self.engine.maintain_open(latest_bar_per_pair, now_msc=now_msc)
        )
        self.result.maintenance_reports.append(dict(reports))
        for rep in reports.values():
            self.result.closed_position_ids.extend(rep.closed_positions)
        self.result.final_open_positions = self.pm.open_positions
        return reports

    def force_close_all(
        self, ask_by_pair: Mapping[str, float], bid_by_pair: Mapping[str, float],
        now_msc: int,
    ) -> int:
        """Flatten every open position via the router. Returns count closed."""
        n = 0
        for pos in list(self.pm.open_positions):
            asyncio.run(self.router.close_position(
                pos, bid=bid_by_pair.get(pos.symbol, pos.entry_price),
                ask=ask_by_pair.get(pos.symbol, pos.entry_price),
                now_msc=now_msc,
            ))
            self.pm.forget_position(pos.position_id)
            self.result.closed_position_ids.append(pos.position_id)
            n += 1
        self.result.final_open_positions = self.pm.open_positions
        return n

    # -------------------------------------------------------- account helpers

    def default_account(self, *, equity: float = 100_000.0) -> AccountState:
        return AccountState(
            equity=equity, starting_equity=equity,
            daily_start_equity=equity,
            daily_pnl_usd=self.daily.state.closed_pnl,
            trades_today=self.daily.trade_count,
            open_position_count=len(self.pm.open_positions),
        )

    def account_with(
        self,
        *,
        equity: float = 100_000.0,
        starting_equity: float = 100_000.0,
        daily_start_equity: Optional[float] = None,
        daily_pnl_usd: float = 0.0,
        trades_today: Optional[int] = None,
        open_position_count: Optional[int] = None,
    ) -> AccountState:
        return AccountState(
            equity=equity,
            starting_equity=starting_equity,
            daily_start_equity=daily_start_equity if daily_start_equity is not None else equity,
            daily_pnl_usd=daily_pnl_usd,
            trades_today=(
                trades_today if trades_today is not None else self.daily.trade_count
            ),
            open_position_count=(
                open_position_count if open_position_count is not None
                else len(self.pm.open_positions)
            ),
        )

    # ------------------------------------------------------------ internals

    def _capture_alerts(self) -> None:
        """Record what the notifier mock received this cycle."""
        for call in self.notifier_mock.send.call_args_list:
            args, kwargs = call
            if not args:
                continue
            msg = args[0] if isinstance(args[0], str) else ""
            self.result.alert_calls.append(msg)
            if "KILL SWITCH" in msg:
                self.result.alert_kill_switch_reasons.append(msg)
        # Reset for next cycle so capture is per-call.
        self.notifier_mock.send.reset_mock()

    @staticmethod
    def _derive_prices(
        bar_feeds: Mapping[str, Sequence[Bar]], *, side: str,
    ) -> Dict[str, float]:
        """Derive a sensible ask/bid from the last bar close for each pair.

        Tests that need slippage / specific fills pass `ask_by_pair`
        explicitly; this helper just avoids forcing every test to construct
        the dict for the no-fill / pending paths.
        """
        out: Dict[str, float] = {}
        for pair, bars in bar_feeds.items():
            if not bars:
                continue
            last = bars[-1]
            spread = 0.0001
            out[pair] = (last.close + spread) if side == "ask" else (last.close - spread)
        return out


# ---------------------------------------------------------------------------
# Bar-builder helpers — thin wrappers around the strategy synthetic fixtures
# ---------------------------------------------------------------------------

from tests.strategy.fixtures.synthetic_bars import (  # noqa: E402
    long_sweep_bars as _long_sweep_bars,
    short_sweep_bars as _short_sweep_bars,
    build_scenario,
    hour_msc, make_bar,
)
from config.asian_sweep_config import PAIR_CONFIG  # noqa: E402


# Per-pair price anchors that satisfy `min_range_pts` / `max_range_pts`.
# Values are chosen large enough to clear the per-pair minimum (asian_range_pts
# = (high-low)/point must lie inside [min_range_pts, max_range_pts]).
_PAIR_PRICE_ANCHORS: Dict[str, Tuple[float, float]] = {
    "XAUUSD": (2000.00, 2005.00),   # 500 pts at pt=0.01, min 100, max 3000
    "EURUSD": (1.10300, 1.10800),   # 500 pts at pt=0.00001, min 200, max 2000
    "AUDUSD": (0.65000, 0.65500),
    "GBPUSD": (1.25000, 1.25600),
    "USDCAD": (1.35000, 1.35500),
    "USDCHF": (0.91000, 0.91500),
    "AUDCHF": (0.59000, 0.59500),
    "AUDNZD": (1.08000, 1.08500),
}


def long_sweep_bars(*, symbol: str, pt: float = None, trigger_hour: int = 8,
                     year: int = 2026, month: int = 4, day: int = 15,
                     bias: str = "neutral", wick_below_pts: float = 50.0,
                     close_above_pts: float = 10.0,
                     asian_low: float = None, asian_high: float = None):
    """Pair-aware LONG sweep bar generator.

    Picks sensible defaults per pair so the asian range size passes
    `min_range_pts` / `max_range_pts` for every V5 pair, including XAUUSD
    (which the upstream `long_sweep_bars` doesn't handle by default).
    """
    if asian_low is None or asian_high is None:
        anchor_low, anchor_high = _PAIR_PRICE_ANCHORS.get(
            symbol, _PAIR_PRICE_ANCHORS["EURUSD"]
        )
        asian_low = anchor_low if asian_low is None else asian_low
        asian_high = anchor_high if asian_high is None else asian_high
    if pt is None:
        pt = float(PAIR_CONFIG[symbol]["point"])
    return _long_sweep_bars(
        symbol=symbol, pt=pt,
        asian_low=asian_low, asian_high=asian_high,
        trigger_hour=trigger_hour, year=year, month=month, day=day,
        wick_below_pts=wick_below_pts, close_above_pts=close_above_pts,
        bias=bias,
    )


def short_sweep_bars(*, symbol: str, pt: float = None, trigger_hour: int = 8,
                      year: int = 2026, month: int = 4, day: int = 15,
                      wick_above_pts: float = 50.0,
                      close_below_pts: float = 10.0,
                      asian_low: float = None, asian_high: float = None):
    """Pair-aware SHORT sweep bar generator."""
    if asian_low is None or asian_high is None:
        anchor_low, anchor_high = _PAIR_PRICE_ANCHORS.get(
            symbol, _PAIR_PRICE_ANCHORS["EURUSD"]
        )
        asian_low = anchor_low if asian_low is None else asian_low
        asian_high = anchor_high if asian_high is None else asian_high
    if pt is None:
        pt = float(PAIR_CONFIG[symbol]["point"])
    return _short_sweep_bars(
        symbol=symbol, pt=pt,
        asian_low=asian_low, asian_high=asian_high,
        trigger_hour=trigger_hour, year=year, month=month, day=day,
        wick_above_pts=wick_above_pts, close_below_pts=close_below_pts,
    )


__all__ = [
    "ScenarioRunner",
    "ScenarioResult",
    "make_notifier_mock",
    "long_sweep_bars",
    "short_sweep_bars",
    "build_scenario",
    "hour_msc",
    "make_bar",
]
