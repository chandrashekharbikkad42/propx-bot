"""propX live-state HTTP dashboard.

aiohttp web server on localhost:8080 (configurable) exposing a JSON
snapshot of the bot's state. Tiny on purpose: 4 endpoints, no templating,
no auth (bound to 127.0.0.1 by default), no DB.

Endpoints:
  GET /              — full snapshot (positions + pendings + daily + health)
  GET /positions     — just the open-positions list
  GET /pendings      — just pending orders
  GET /daily         — DailyTracker snapshot
  GET /signals       — last N detected signals (engine-provided)
  GET /health        — { ok: bool, last_bar_ms, mt5_connected }

The dashboard pulls state lazily via callbacks the caller supplies — no
shared mutable state across modules. To run:

    from monitoring.dashboard import GriffDashboard
    dash = GriffDashboard(pm, daily, signals_provider=lambda: [], health=lambda: {...})
    await dash.start(host="127.0.0.1", port=8080)
    ...
    await dash.stop()

aiohttp is already a project dep (used by TelegramNotifier) — no new pip
install required. FastAPI was the spec's first instinct; aiohttp keeps
the dep surface unchanged.

Hinglish: chhota sa HTTP server, sab JSON me. Koi shared mutable state
nahi — caller hi callbacks bhejta hai, dashboard sirf serve karta hai.
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Callable, List, Optional

from aiohttp import web

from execution.position_manager import GriffPositionManager
from monitoring.daily_tracker import DailyTracker
from strategy.patterns.base import PatternSignal


SignalsProvider = Callable[[], List[PatternSignal]]
HealthProvider = Callable[[], dict]


def _signal_to_dict(s: PatternSignal) -> dict:
    return {
        "pattern_name": s.pattern_name,
        "symbol": s.symbol,
        "direction": s.direction.value,
        "entry": s.entry, "sl": s.sl, "tp": s.tp,
        "confidence": s.confidence, "grade": s.grade.value,
        "confluences_met": list(s.confluences_met),
        "bar_time_msc": s.bar_time_msc,
    }


class GriffDashboard:
    def __init__(
        self,
        position_mgr: GriffPositionManager,
        daily: DailyTracker,
        *,
        signals_provider: Optional[SignalsProvider] = None,
        health_provider: Optional[HealthProvider] = None,
    ) -> None:
        self._pm = position_mgr
        self._daily = daily
        self._signals = signals_provider or (lambda: [])
        self._health = health_provider or (lambda: {
            "ok": True, "last_bar_ms": None, "mt5_connected": None,
        })
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    # --------------------------------------------------------------- app

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.handle_snapshot)
        app.router.add_get("/positions", self.handle_positions)
        app.router.add_get("/pendings", self.handle_pendings)
        app.router.add_get("/daily", self.handle_daily)
        app.router.add_get("/signals", self.handle_signals)
        app.router.add_get("/health", self.handle_health)
        return app

    # ----------------------------------------------------------- handlers

    async def handle_snapshot(self, _req):
        return web.json_response({
            "positions": [self._pos_to_dict(p) for p in self._pm.open_positions],
            "pendings": [self._pending_to_dict(o) for o in self._pm.pending_orders],
            "daily": _daily_dict(self._daily),
            "signals": [_signal_to_dict(s) for s in self._signals()],
            "health": self._health(),
        })

    async def handle_positions(self, _req):
        return web.json_response(
            [self._pos_to_dict(p) for p in self._pm.open_positions]
        )

    async def handle_pendings(self, _req):
        return web.json_response(
            [self._pending_to_dict(o) for o in self._pm.pending_orders]
        )

    async def handle_daily(self, _req):
        return web.json_response(_daily_dict(self._daily))

    async def handle_signals(self, _req):
        return web.json_response([_signal_to_dict(s) for s in self._signals()])

    async def handle_health(self, _req):
        return web.json_response(self._health())

    # ----------------------------------------------------------- lifecycle

    async def start(self, *, host: str = "127.0.0.1", port: int = 8080) -> None:
        self._runner = web.AppRunner(self.build_app())
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, host=host, port=port)
        await self._site.start()

    async def stop(self) -> None:
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    # --------------------------------------------------------- formatters

    @staticmethod
    def _pos_to_dict(p) -> dict:
        return {
            "position_id": p.position_id, "mt5_ticket": p.mt5_ticket,
            "symbol": p.symbol, "side": p.side.value, "lots": p.lots,
            "entry_price": p.entry_price, "sl_price": p.sl_price,
            "tp_price": p.tp_price, "opened_msc": p.opened_msc,
            "pattern_name": p.pattern_name,
        }

    @staticmethod
    def _pending_to_dict(o) -> dict:
        return {
            "order_id": o.order_id, "mt5_ticket": o.mt5_ticket,
            "symbol": o.symbol, "side": o.side.value, "lots": o.lots,
            "pending_price": o.pending_price, "sl_price": o.sl_price,
            "tp_price": o.tp_price, "expiry_msc": o.expiry_msc,
            "is_limit": o.is_limit, "pattern_name": o.pattern_name,
        }


def _daily_dict(d: DailyTracker) -> dict:
    s = d.state
    return {
        "trade_day": s.trade_day, "peak_equity": s.peak_equity,
        "closed_pnl": s.closed_pnl, "floating_pnl": s.floating_pnl,
        "trade_count": s.trade_count, "max_dd_today": s.max_dd_today,
        "last_update_ms": s.last_update_ms,
    }
