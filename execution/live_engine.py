"""AsianSweepLiveEngine — orchestration loop for the V5 strategy.

A faithful copy of `execution.griff_live_engine.GriffLiveEngine` with three
targeted swaps so the existing Scanner/Compliance/Router pipeline can be
reused as-is:

  1. Scanner is built with `build_asian_sweep_detector()` (caller wires this).
  2. `pattern_name == "ASIAN_SWEEP"` is routed via market entry (the V5
     strategy enters at bar close on the sweep, not via pending). FLAG / other
     names retain their original routing for backward compatibility.
  3. Lot sizing uses `asian_sweep_lots_for(...)` — point-based, mirrors
     `multi_pair_backtest.simulate` (which the verified PF 2.27 / 239-trade
     backtest uses). The Griff `pip_size × 10 USD` MVP formula is unsuitable
     for XAUUSD (point 0.01) and the cross pairs in V5.

Everything else — CycleReport shape, per-pair-best dedupe, compliance gate,
HourlyStats hooks, `maintain_open` per-bar bookkeeping — stays identical to
the Griff engine so the dashboard, Telegram alerts, daily tracker, and
position-manager keep working without modification.

Hinglish: pipeline wahi puraana — scan/compliance/size/router. Bas detector
factory aur lot-sizing badla. ASIAN_SWEEP ko bhi FLAG ki tarah market entry
milti hai. Baaki sab geometry pre-existing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from config.asian_sweep_config import (
    PAIR_CONFIG,
    MIN_SL_DISTANCE_PIPS,
    MAX_RISK_USD_PER_TRADE,
    MAX_TOTAL_DD_PCT,
)
from data.bar_aggregator import Bar
# Constant shared with HouseMoneyManager — the manager only supports
# trade_number_today in {1, 2}; anything beyond that is invalid per spec
# (the V5 strategy caps at 2 trades/day). When the in-cycle position
# count would push past this, the engine defers the signal rather than
# calling HouseMoneyManager with an out-of-range index.
_HOUSE_MONEY_MAX_TRADE_NUMBER: int = 2
from data.news_calendar import NewsCalendar
from execution.order_router import (
    GriffOpenPosition,
    GriffOrderRouter,
    GriffPendingOrder,
)
from execution.position_manager import GriffPositionManager
from monitoring.daily_tracker import DailyTracker
from monitoring.telegram_alerts import GriffTelegramAlerts
from monitoring.hourly_reporter import HourlyStats
from risk.asian_sweep_exit import size_position as _asian_sweep_size_position
from risk.house_money import HouseMoneyManager, RiskAllocation
from risk.prop_firm.compliance import AccountState, ComplianceEngine
from strategy.patterns.base import Direction, Grade, PatternSignal
from strategy.scanner import Scanner
from utils.logger import logger


# Patterns that take a market entry instead of a pending order. FLAG is the
# legacy Griff market-entry pattern; ASIAN_SWEEP is V5 (sweep + close on
# the current bar → enter at close).
_MARKET_ENTRY_PATTERNS: frozenset[str] = frozenset({"FLAG", "ASIAN_SWEEP"})


def asian_sweep_lots_for(
    risk_pct: float,
    equity: float,
    sl_distance_price: float,
    symbol: str,
    *,
    min_lots: float = 0.01,
) -> float:
    """Point-based lot sizing — direct port of `multi_pair_backtest.simulate`.

        risk_amt        = equity * risk_pct / 100
        risk_pts_count  = sl_distance_price / point
        vpl             = contract_size * point      # value per lot per point
        lots            = clamp(risk_amt / (risk_pts_count * vpl),
                                min_lots, lot_max)

    Falls back to `min_lots` for unknown symbols or degenerate inputs.
    Per-pair `risk_override` (XAUUSD = 0.5%) is NOT applied here — caller
    decides which risk_pct to pass in (HouseMoneyManager has already
    consumed risk grading by the time we get here). To honour the
    XAUUSD-0.5% override end-to-end, prefer `risk.asian_sweep_exit.size_position`.
    """
    cfg = PAIR_CONFIG.get(symbol)
    if cfg is None:
        return min_lots
    pt = float(cfg["point"])             # type: ignore[arg-type]
    ct = float(cfg["contract_size"])     # type: ignore[arg-type]
    lmax = float(cfg["lot_max"])         # type: ignore[arg-type]

    if sl_distance_price <= 0 or equity <= 0 or risk_pct <= 0:
        return min_lots

    # SAFETY CAP #1 — MIN SL FLOOR. Reject degenerate SLs (1 pip = 10
    # broker points). Mirrors the same gate in size_position.
    pip_size = pt * 10.0
    sl_pips = sl_distance_price / pip_size
    if sl_pips < MIN_SL_DISTANCE_PIPS:
        logger.warning(
            f"asian_sweep_lots_for REJECT {symbol}: SL distance "
            f"{sl_pips:.2f} pips < MIN_SL_DISTANCE_PIPS="
            f"{MIN_SL_DISTANCE_PIPS} (sl_distance_price={sl_distance_price})"
        )
        return 0.0

    risk_amt = equity * risk_pct / 100.0
    risk_pts_count = sl_distance_price / pt
    vpl = ct * pt
    if risk_pts_count <= 0 or vpl <= 0:
        return min_lots
    raw = risk_amt / (risk_pts_count * vpl)
    lots = round(max(min_lots, min(raw, lmax)), 2)
    lots = max(min_lots, lots)

    # SAFETY CAP #2 — MAX USD RISK PER TRADE. Absolute ceiling.
    actual_risk_usd = lots * risk_pts_count * vpl
    if actual_risk_usd > MAX_RISK_USD_PER_TRADE:
        capped_raw = MAX_RISK_USD_PER_TRADE / (risk_pts_count * vpl)
        capped_lot = round(min(capped_raw, lmax), 2)
        logger.warning(
            f"asian_sweep_lots_for SCALE {symbol}: actual_risk_usd "
            f"${actual_risk_usd:.2f} > MAX_RISK_USD_PER_TRADE="
            f"${MAX_RISK_USD_PER_TRADE:.2f}; lots {lots:.2f} → {capped_lot:.2f}"
        )
        if capped_lot < min_lots:
            return 0.0
        lots = capped_lot
    return lots


@dataclass
class CycleReport:
    """Per-scan-cycle observation — same shape as the Griff engine for parity."""
    now_msc: int
    signals_emitted: int = 0
    signals_rejected_by_compliance: int = 0
    orders_placed: int = 0
    rejections: List[Tuple[str, str]] = field(default_factory=list)


class AsianSweepLiveEngine:
    """V5 live engine. Drop-in replacement for `GriffLiveEngine` — same
    constructor shape so existing wiring code can swap in cleanly.
    """

    def __init__(
        self,
        scanner: Scanner,
        router: GriffOrderRouter,
        position_mgr: GriffPositionManager,
        compliance: ComplianceEngine,
        house_money: HouseMoneyManager,
        daily: DailyTracker,
        alerts: GriffTelegramAlerts,
        *,
        contract_size: float = 100_000.0,
        pending_expiry_hours: int = 1,
        hourly_stats: Optional[HourlyStats] = None,
        news_calendar: Optional[NewsCalendar] = None,
    ) -> None:
        self._scanner = scanner
        self._router = router
        self._pm = position_mgr
        self._compliance = compliance
        self._house = house_money
        self._daily = daily
        self._alerts = alerts
        # Legacy fallback for symbols not present in PAIR_CONFIG. The
        # actual per-signal contract size is resolved via
        # `_contract_size_for(symbol)` so XAUUSD (100) and the
        # 100k-contract FX majors get the right worst-loss estimate in
        # the compliance gate.
        self._contract_size = contract_size
        self._pending_expiry_hours = pending_expiry_hours
        self._stats = hourly_stats
        # The5%ers / FTMO rule: no entries within ±2 min of HIGH-impact news.
        # ComplianceEngine ALSO checks this (defense-in-depth) but the engine
        # has its own gate so blackout rejections produce a distinct log /
        # Telegram alert before the compliance pipeline even runs.
        self._news = news_calendar
        # SAFETY CAP #3 state — observed equity high-water mark. Lazy-init
        # on the first cycle (None ⇒ adopt the first equity we see). Drives
        # the MAX_TOTAL_DD_PCT kill switch in process_scan_cycle.
        self._equity_hwm: Optional[float] = None

    @property
    def hourly_stats(self) -> Optional[HourlyStats]:
        return self._stats

    # ============================================================ cycle API

    async def process_scan_cycle(
        self,
        bar_feeds: Mapping[str, Sequence[Bar]],
        *,
        now_msc: int,
        ask_by_pair: Mapping[str, float],
        bid_by_pair: Mapping[str, float],
        account: AccountState,
    ) -> CycleReport:
        """Run the scanner once and place orders for allowed signals."""
        report = CycleReport(now_msc=now_msc)
        if self._stats is not None:
            for pair, bars in bar_feeds.items():
                if bars:
                    self._stats.record_bar(pair)
        signals = self._scanner.scan_all(bar_feeds, now_msc)
        report.signals_emitted = len(signals)
        if self._stats is not None and signals:
            self._stats.record_signal(len(signals))

        # Pick best signal per symbol — V5 also caps 2 trades/day with 1
        # per direction, but compliance owns the global cap so we just
        # forward our best-per-pair here. The router handles ordering.
        per_pair_best: dict[str, PatternSignal] = {}
        for s in signals:
            if s.grade == Grade.C:
                continue
            key = s.symbol
            cur = per_pair_best.get(key)
            if cur is None or _sig_rank(s) > _sig_rank(cur):
                per_pair_best[key] = s

        # SAFETY CAP #3 — MAX_TOTAL_DD_PCT kill switch. Update the bot's
        # observed HWM with the latest equity, then halt all new entries
        # this cycle if drawdown from HWM has breached the threshold.
        # Daily DD (3%) is owned by DailyTracker / compliance; this is
        # the total-account guard, sized below the prop firm's 10% cap.
        if self._equity_hwm is None or account.equity > self._equity_hwm:
            self._equity_hwm = account.equity
        if self._equity_hwm > 0 and per_pair_best:
            dd_pct = (
                (self._equity_hwm - account.equity) / self._equity_hwm * 100.0
            )
            if dd_pct >= MAX_TOTAL_DD_PCT:
                logger.warning(
                    f"AsianSweepLiveEngine HALT new entries: total DD "
                    f"{dd_pct:.2f}% >= MAX_TOTAL_DD_PCT={MAX_TOTAL_DD_PCT}% "
                    f"(equity ${account.equity:,.2f}, "
                    f"HWM ${self._equity_hwm:,.2f})"
                )
                for symbol, signal in per_pair_best.items():
                    report.signals_rejected_by_compliance += 1
                    report.rejections.append(
                        (f"{symbol}:{signal.pattern_name}",
                         "max_total_dd_breached")
                    )
                await _safe_alert(
                    self._alerts.kill_switch_triggered,
                    f"max_total_dd_breached: dd_pct={dd_pct:.2f}% "
                    f"(hwm=${self._equity_hwm:,.2f}, "
                    f"eq=${account.equity:,.2f})",
                )
                return report

        # Phase 6 fix #4 — clamp signals past HouseMoneyManager's max
        # trade_number_today. Compliance sees a per-cycle SNAPSHOT
        # (`account.trades_today`) so multiple signals in the same cycle
        # all pass that check, but House Money is keyed on the LIVE
        # running count and raises ValueError beyond 2. We use
        # `daily.trade_count` directly here because `record_trade_open`
        # bumps it synchronously each loop iteration, so it already
        # reflects everything opened in the current cycle.
        cap = _HOUSE_MONEY_MAX_TRADE_NUMBER
        for symbol, signal in per_pair_best.items():
            # News blackout — early skip so we don't churn HouseMoney /
            # compliance for a signal that will never trade. Compliance
            # also rejects on news but this branch yields a cleaner reason.
            if self._news is not None and self._news.is_blackout(symbol, now_msc):
                report.signals_rejected_by_compliance += 1
                report.rejections.append(
                    (f"{symbol}:{signal.pattern_name}", "news_blackout")
                )
                logger.info(
                    f"ASIAN_SWEEP news_blackout SKIP {symbol} {signal.pattern_name}"
                )
                await _safe_alert(
                    self._alerts.kill_switch_triggered,
                    f"{symbol} {signal.pattern_name}: news_blackout",
                )
                continue

            # Phase 6 fix #4 — stop accepting new signals when the
            # running per-cycle count would exceed House Money's cap.
            # Defer (count as compliance rejection) rather than raising.
            prospective_trade_no = self._daily.trade_count + 1
            if prospective_trade_no > cap:
                report.signals_rejected_by_compliance += 1
                report.rejections.append(
                    (f"{symbol}:{signal.pattern_name}",
                     "daily_trade_cap_reached")
                )
                logger.info(
                    f"ASIAN_SWEEP daily_trade_cap_reached SKIP "
                    f"{symbol} {signal.pattern_name} "
                    f"(daily.trade_count={self._daily.trade_count})"
                )
                await _safe_alert(
                    self._alerts.kill_switch_triggered,
                    f"{symbol} {signal.pattern_name}: daily_trade_cap_reached",
                )
                continue

            # Phase 6 fix #5 — per-pair contract_size lookup. The
            # compliance worst-loss estimate is sensitive to this; XAUUSD
            # has contract_size=100, not 100_000, so passing the wrong
            # value over-rejects valid XAUUSD signals on small accounts.
            contract_size = self._contract_size_for(symbol)
            ok, reason = self._compliance.can_trade(
                signal, now_msc, account, lots=0.01,
                contract_size=contract_size,
            )
            if self._stats is not None:
                self._stats.record_compliance(passed=ok)
            if not ok:
                report.signals_rejected_by_compliance += 1
                report.rejections.append((f"{symbol}:{signal.pattern_name}", reason))
                logger.info(
                    f"ASIAN_SWEEP compliance REJECT {symbol} {signal.pattern_name}: {reason}"
                )
                await _safe_alert(
                    self._alerts.kill_switch_triggered,
                    f"{symbol} {signal.pattern_name}: {reason}",
                )
                continue

            # Size.
            #
            # Two-stage policy:
            #   1. HouseMoneyManager grades risk by signal quality + daily
            #      PnL state — same as the Griff path.
            #   2. `size_position` then resolves the per-pair `risk_override`
            #      from PAIR_CONFIG (XAUUSD=0.5%) and the weak-month dampener
            #      from the trade's calendar month, so the backtest's per-pair
            #      risk discipline is preserved end-to-end.
            trade_no = prospective_trade_no
            allocation = self._house.calc_trade_risk(
                signal.grade, account.equity,
                account.daily_pnl_usd, trade_no,
            )
            from datetime import datetime as _dt, timezone as _tz
            trade_month = _dt.fromtimestamp(now_msc / 1000.0, tz=_tz.utc).month
            base_lots = _asian_sweep_size_position(
                signal.symbol,
                equity=account.equity,
                sl_distance_price=signal.risk_distance,
                month=trade_month,
            )
            # HouseMoney can dampen further (e.g. on a B-grade or a hot
            # day); we scale `base_lots` by the ratio of allocation vs the
            # default 0.8% / 0.5% the backtest used.
            base_risk_pct = (
                _resolve_base_risk_pct(signal.symbol, trade_month)
            )
            if base_risk_pct > 0 and allocation.final_risk_pct < base_risk_pct:
                lots = round(
                    max(0.01, base_lots * allocation.final_risk_pct / base_risk_pct),
                    2,
                )
            else:
                lots = base_lots
            await _safe_alert(self._alerts.signal_detected, signal)

            # Place. ASIAN_SWEEP (and FLAG legacy) → market; everything
            # else falls through to the pending-stop / pending-limit
            # branches preserved from the Griff engine.
            if signal.pattern_name in _MARKET_ENTRY_PATTERNS:
                pos = await self._router.place_market(
                    signal, lots,
                    ask=ask_by_pair.get(signal.symbol, signal.entry),
                    bid=bid_by_pair.get(signal.symbol, signal.entry),
                    now_msc=now_msc,
                )
                self._pm.register_position(pos)
                await _safe_alert(self._alerts.trade_opened, pos, lots=lots)
            else:
                expiry_msc = now_msc + self._pending_expiry_hours * 3_600_000
                is_combo = signal.pattern_name == "COMBO"
                placer = (
                    self._router.place_pending_limit if is_combo
                    else self._router.place_pending_stop
                )
                pending = await placer(
                    signal, lots, expiry_msc=expiry_msc, now_msc=now_msc,
                )
                self._pm.register_pending(pending)

            self._daily.record_trade_open(now_ms=now_msc)
            report.orders_placed += 1

        return report

    def _contract_size_for(self, symbol: str) -> float:
        """Per-pair contract size for compliance worst-loss estimation.

        Falls back to `self._contract_size` (legacy 100_000 default) for
        symbols not in PAIR_CONFIG so previously-working FX paths are
        unchanged.
        """
        cfg = PAIR_CONFIG.get(symbol)
        if cfg is None:
            return self._contract_size
        return float(cfg["contract_size"])  # type: ignore[arg-type]

    async def maintain_open(
        self,
        latest_bar_per_pair: Mapping[str, Bar],
        *,
        now_msc: int,
    ) -> Dict[str, "MaintenanceReport"]:
        """Per-bar maintenance for every pair we have a new bar for."""
        from execution.position_manager import MaintenanceReport
        out: dict[str, MaintenanceReport] = {}
        for pair, bar in latest_bar_per_pair.items():
            rep = await self._pm.maintain(pair, bar, now_msc=now_msc)
            out[pair] = rep
        return out


async def _safe_alert(coro_fn, *args, **kwargs) -> None:
    """Best-effort alert dispatch (Phase 6 fix #6).

    The real `TelegramNotifier.send` swallows its own transport errors,
    but the engine should not depend on that invariant. Wrapping each
    alert call here guarantees that a regression in the notifier — or a
    mock in tests — never propagates an exception into the trading loop.
    """
    try:
        await coro_fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — must not break trading loop
        logger.warning(f"alert dispatch failed (non-fatal): {exc}")


def _sig_rank(s: PatternSignal) -> tuple:
    """Same key Scanner.get_best_signal uses — grade rank, confidence, rr."""
    return (s.grade.rank, s.confidence, s.rr_ratio)


def _resolve_base_risk_pct(symbol: str, month: int) -> float:
    """Return the V5 base risk-% the backtest would have used for this trade.

    Mirrors `risk.asian_sweep_exit.size_position`'s lookup so the scaler in
    `process_scan_cycle` can compare HouseMoney's final_risk_pct on the
    same axis.
    """
    from config.asian_sweep_config import risk_pct_for
    return risk_pct_for(symbol, month=month)


__all__ = [
    "AsianSweepLiveEngine",
    "CycleReport",
    "asian_sweep_lots_for",
]
