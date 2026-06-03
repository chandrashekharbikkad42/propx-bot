"""propX — live console dashboard.

Display-only. Pulls state from `DailyTracker`, `GriffPositionManager`,
and caller-supplied providers (account balance / per-pair spread / halt
reason), then renders a coloured ANSI panel to stdout.

The renderer is split into:
  * `render(...)` — pure, returns the panel as a string (testable).
  * `ConsoleDashboard.render_once(...)` — snapshot using bound providers.
  * `ConsoleDashboard.run(...)` — periodic refresh loop (background task).

No new dependencies: pure colorama (already pinned in requirements.txt).
This module sits ALONGSIDE the existing aiohttp `GriffDashboard` (HTTP) —
they don't share state, the HTTP server is left untouched.

Hinglish: terminal me ek polished panel — balance, PnL, W/L, session,
positions. Trading logic ko sirf padhta hai, kabhi modify nahi karta.
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional, Sequence

from monitoring.banner import THEME
from monitoring.daily_tracker import DailyTracker


# Width of the panel body (excluding borders) — chosen to fit an 80-col
# terminal with room for the border chars (│ + │ + 2 padding = 4).
_PANEL_W: int = 76


# ─── Session detection ───────────────────────────────────────────────────────
# UTC hour-of-day → session label. Tied to `config/asian_sweep_config.py`
# constants but duplicated here (display-only — keep the deps minimal).
_ASIAN_START_UTC_H = 19   # previous day 19:30 UTC
_ASIAN_END_UTC_H = 0      # current day 00:30 UTC
_LONDON_START_UTC_H = 6
_LONDON_END_UTC_H = 10
_NY_START_UTC_H = 12
_NY_END_UTC_H = 15
_FORCE_CLOSE_UTC_H = 16


def session_label(now_utc: datetime) -> tuple[str, str]:
    """Return (label, level) for the current UTC moment.

    level ∈ {"active", "neutral", "closed"} — drives the panel colour.
    """
    h = now_utc.hour
    m = now_utc.minute
    # Asian range build: 19:30 prev → 00:30 today
    if h >= _ASIAN_START_UTC_H and (h > _ASIAN_START_UTC_H or m >= 30):
        return "ASIAN RANGE BUILDING", "neutral"
    if h == _ASIAN_END_UTC_H and m < 30:
        return "ASIAN RANGE BUILDING", "neutral"
    if _LONDON_START_UTC_H <= h <= _LONDON_END_UTC_H:
        return "LONDON SWEEP WINDOW", "active"
    if _NY_START_UTC_H <= h <= _NY_END_UTC_H:
        return "NY SWEEP WINDOW", "active"
    if h == _FORCE_CLOSE_UTC_H:
        return "EOD FORCE-CLOSE", "neutral"
    return "OUT OF SESSION", "closed"


# ─── Provider type aliases ───────────────────────────────────────────────────
# All providers are optional callables returning dicts — keeps the
# dashboard decoupled from MT5 / Settings / engine internals.
AccountProvider = Callable[[], dict]        # {"balance", "equity", "currency"}
SpreadProvider = Callable[[], dict]         # {pair: points}
StatusProvider = Callable[[], tuple[str, str]]  # (label, level)


# ─── Pure renderers (helpers) ────────────────────────────────────────────────

def _fmt_money(v: float, *, sign: bool = False, currency: str = "$") -> str:
    """`$1,234.50`, optionally `+$23.40` / `-$5.00`."""
    sgn = ""
    if sign and v > 0:
        sgn = "+"
    elif v < 0:
        sgn = "-"
        v = -v
    return f"{sgn}{currency}{v:,.2f}"


def _color_for_pnl(v: float) -> str:
    if v > 0:
        return THEME.profit
    if v < 0:
        return THEME.loss
    return THEME.value


def _color_for_dd(used_pct: float, cap_pct: float) -> str:
    if cap_pct <= 0:
        return THEME.value
    ratio = used_pct / cap_pct
    if ratio >= 0.8:
        return THEME.loss
    if ratio >= 0.5:
        return THEME.warn
    return THEME.profit


def _hr(char: str = "─") -> str:
    return char * (_PANEL_W + 2)


def _box_line(content: str = "", visible_len: Optional[int] = None) -> str:
    """Wrap a (possibly ANSI-coloured) string in │ borders, padded to width.

    `visible_len` is the printable width of `content` (no ANSI). When not
    given we recompute it — but the caller usually knows.
    """
    if visible_len is None:
        visible_len = _visible_len(content)
    pad = max(0, _PANEL_W - visible_len)
    return f"│ {content}{' ' * pad} │"


_ANSI_RE = None


def _visible_len(s: str) -> int:
    """Length of `s` minus ANSI escape sequences."""
    global _ANSI_RE
    if _ANSI_RE is None:
        import re
        _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
    return len(_ANSI_RE.sub("", s))


# ─── Public renderer ─────────────────────────────────────────────────────────

@dataclass
class DashboardSnapshot:
    """Inputs gathered by `ConsoleDashboard.render_once()` and passed to
    `render(...)`. Exposed so tests can build snapshots without standing
    up a full bot."""
    # Identity
    broker_label: str
    account: str
    mode: str
    # Account state
    balance: float
    starting_equity: float
    closed_pnl: float
    floating_pnl: float
    # Daily stats
    wins: int
    losses: int
    trades_today: int
    max_trades_per_day: int
    daily_dd_pct: float
    daily_cap_pct: float
    # Session / market
    pairs: tuple[str, ...]
    spread_by_pair: dict[str, float]
    session_label: str
    session_level: str
    # Positions
    open_positions: list[dict]   # {symbol, side, lots, entry, pnl}
    # Status
    status_label: str
    status_level: str            # ok / halt / warn / idle
    # Timestamp
    now_utc_str: str


def render(snap: DashboardSnapshot) -> str:
    """Render one panel string from a snapshot. Pure — easy to unit-test."""
    lines: list[str] = []

    top = f"{THEME.accent}┌{_hr('─')}┐{THEME.reset}"
    bot = f"{THEME.accent}└{_hr('─')}┘{THEME.reset}"
    mid = f"{THEME.accent}├{_hr('─')}┤{THEME.reset}"

    lines.append(top)

    # Header row: brand + tagline + UTC stamp
    header_left = (
        f"{THEME.accent}propX{THEME.reset}"
        f"  {THEME.label}Asian Range London Sweep{THEME.reset}"
    )
    header_right = f"{THEME.muted}{snap.now_utc_str} UTC{THEME.reset}"
    hl_vis = _visible_len(header_left)
    hr_vis = _visible_len(header_right)
    pad = max(1, _PANEL_W - hl_vis - hr_vis)
    header = f"{header_left}{' ' * pad}{header_right}"
    lines.append(_box_line(header, visible_len=hl_vis + pad + hr_vis))
    lines.append(mid)

    # Identity row
    mode_col = THEME.loss if snap.mode.upper() == "REAL" else (
        THEME.warn if snap.mode.upper() == "PAPER" else THEME.muted)
    identity = (
        f"{THEME.label}Broker:{THEME.reset} {THEME.value}{snap.broker_label}{THEME.reset}"
        f"   {THEME.label}Account:{THEME.reset} {THEME.value}{snap.account}{THEME.reset}"
        f"   {THEME.label}Mode:{THEME.reset} {mode_col}{snap.mode.upper()}{THEME.reset}"
    )
    lines.append(_box_line(identity))
    lines.append(mid)

    # Balance / PnL / DD block
    net_pnl = snap.closed_pnl + snap.floating_pnl
    pct = (net_pnl / snap.starting_equity * 100.0) if snap.starting_equity else 0.0
    bal_col = _color_for_pnl(net_pnl)
    bal_line = (
        f"{THEME.label}Balance:{THEME.reset} {bal_col}{_fmt_money(snap.balance)}{THEME.reset}"
        f" ({bal_col}{pct:+.2f}%{THEME.reset})"
        f"     {THEME.label}Start:{THEME.reset} {_fmt_money(snap.starting_equity)}"
    )
    lines.append(_box_line(bal_line))

    wr = (snap.wins / max(1, snap.wins + snap.losses)) * 100.0
    wr_col = THEME.profit if wr >= 50 else (THEME.warn if wr >= 33 else THEME.loss)
    if snap.wins + snap.losses == 0:
        wr_col = THEME.label
    pnl_col = _color_for_pnl(net_pnl)
    pnl_line = (
        f"{THEME.label}PnL day:{THEME.reset} {pnl_col}{_fmt_money(net_pnl, sign=True)}{THEME.reset}"
        f"   {THEME.label}W:{THEME.reset}{THEME.profit}{snap.wins}{THEME.reset}"
        f" {THEME.label}L:{THEME.reset}{THEME.loss}{snap.losses}{THEME.reset}"
        f"   {THEME.label}WR:{THEME.reset} {wr_col}{wr:5.1f}%{THEME.reset}"
        f"   {THEME.label}Trades:{THEME.reset} {THEME.value}{snap.trades_today}/{snap.max_trades_per_day}{THEME.reset}"
    )
    lines.append(_box_line(pnl_line))

    dd_col = _color_for_dd(snap.daily_dd_pct, snap.daily_cap_pct)
    dd_line = (
        f"{THEME.label}Daily DD:{THEME.reset} "
        f"{dd_col}{snap.daily_dd_pct:.2f}%{THEME.reset}"
        f" / {snap.daily_cap_pct:.2f}%"
    )
    lines.append(_box_line(dd_line))
    lines.append(mid)

    # Session row
    if snap.session_level == "active":
        sess_col = THEME.profit
    elif snap.session_level == "neutral":
        sess_col = THEME.warn
    else:
        sess_col = THEME.muted
    # Spread: show XAU (usually present) or first pair available.
    spread_pair = "XAUUSD" if "XAUUSD" in snap.spread_by_pair else (
        next(iter(snap.spread_by_pair), None))
    if spread_pair is not None:
        spread_pts = snap.spread_by_pair[spread_pair]
        spread_line_part = (
            f"   {THEME.label}Spread:{THEME.reset} "
            f"{THEME.value}{spread_pts:.0f}pts{THEME.reset}"
            f" {THEME.label}({spread_pair}){THEME.reset}"
        )
    else:
        spread_line_part = (
            f"   {THEME.label}Spread:{THEME.reset} {THEME.muted}n/a{THEME.reset}"
        )
    sess_line = (
        f"{THEME.label}Session:{THEME.reset} {sess_col}{snap.session_label}{THEME.reset}"
        f"{spread_line_part}"
    )
    lines.append(_box_line(sess_line))

    if snap.pairs:
        pairs_line = (
            f"{THEME.label}Pairs:{THEME.reset} "
            f"{THEME.accent_soft}{' '.join(snap.pairs)}{THEME.reset}"
        )
        lines.append(_box_line(pairs_line))
    lines.append(mid)

    # Positions block
    if snap.open_positions:
        lines.append(_box_line(f"{THEME.label}Open positions:{THEME.reset}"))
        for p in snap.open_positions:
            side = p.get("side", "?")
            side_col = THEME.profit if side.upper() == "BUY" else THEME.loss
            pnl = float(p.get("pnl", 0.0))
            pnl_col = _color_for_pnl(pnl)
            row = (
                f"  {THEME.value}{p['symbol']:<7}{THEME.reset}"
                f" {side_col}{side:<4}{THEME.reset}"
                f" {THEME.value}{p['lots']:>5.2f}{THEME.reset}"
                f"  {THEME.label}@{THEME.reset} {p['entry']:>10.4f}"
                f"   {THEME.label}pnl:{THEME.reset} {pnl_col}{_fmt_money(pnl, sign=True)}{THEME.reset}"
            )
            lines.append(_box_line(row))
    else:
        lines.append(_box_line(
            f"{THEME.label}Open positions:{THEME.reset} {THEME.muted}none{THEME.reset}"
        ))
    lines.append(mid)

    # Status
    level = snap.status_level
    if level == "halt":
        st_col = THEME.loss
    elif level == "ok":
        st_col = THEME.profit
    elif level == "warn":
        st_col = THEME.warn
    else:
        st_col = THEME.muted
    status_line = (
        f"{THEME.label}Status:{THEME.reset} {st_col}{snap.status_label}{THEME.reset}"
    )
    lines.append(_box_line(status_line))

    lines.append(bot)
    lines.append(
        f"  {THEME.muted}Ctrl+C to stop safely{THEME.reset}"
    )

    return "\n".join(lines)


# ─── Live wrapper ────────────────────────────────────────────────────────────

class ConsoleDashboard:
    """Binds providers + state objects to a renderer + refresh loop.

    The dashboard reads ONLY — never mutates strategy/risk/execution state.
    """

    def __init__(
        self,
        *,
        position_manager,
        daily: DailyTracker,
        pairs: Sequence[str],
        starting_equity: float,
        broker_label: str,
        account: str,
        mode: str,
        daily_cap_pct: float = 5.0,
        max_trades_per_day: int = 2,
        account_provider: Optional[AccountProvider] = None,
        spread_provider: Optional[SpreadProvider] = None,
        status_provider: Optional[StatusProvider] = None,
        refresh_sec: float = 30.0,
        sink=print,
    ) -> None:
        self._pm = position_manager
        self._daily = daily
        self._pairs = tuple(pairs)
        self._starting_equity = float(starting_equity)
        self._broker_label = broker_label
        self._account = str(account)
        self._mode = mode
        self._cap_pct = float(daily_cap_pct)
        self._max_trades = int(max_trades_per_day)
        self._account_provider = account_provider
        self._spread_provider = spread_provider
        self._status_provider = status_provider
        self._refresh_sec = float(refresh_sec)
        self._sink = sink
        # Win/loss accumulators — fed via `record_trade_close(pnl)`.
        self._wins: int = 0
        self._losses: int = 0

    # ----------------------------------------------------------- mutators
    def record_trade_close(self, pnl_usd: float) -> None:
        """Caller hook: bump win/loss counter on each closed trade."""
        if pnl_usd > 0:
            self._wins += 1
        elif pnl_usd < 0:
            self._losses += 1

    # ----------------------------------------------------------- providers
    def _snapshot(self) -> DashboardSnapshot:
        now_utc = datetime.now(timezone.utc)
        sess, sess_level = session_label(now_utc)

        # Balance / equity: prefer live provider, fall back to starting eq.
        balance = self._starting_equity
        floating = self._daily.state.floating_pnl
        if self._account_provider is not None:
            try:
                acct = self._account_provider() or {}
                balance = float(acct.get("balance", balance))
                # If the provider reports equity, treat the delta vs balance
                # as floating PnL (more accurate than daily.floating_pnl
                # which is only set when the engine pushes it).
                if "equity" in acct:
                    floating = float(acct["equity"]) - balance
            except Exception:
                pass

        spread_by_pair: dict[str, float] = {}
        if self._spread_provider is not None:
            try:
                spread_by_pair = dict(self._spread_provider() or {})
            except Exception:
                spread_by_pair = {}

        # Status: provider > position-based default.
        if self._status_provider is not None:
            try:
                status_label, status_level = self._status_provider()
            except Exception:
                status_label, status_level = ("scanning", "ok")
        else:
            if self._pm.open_positions:
                status_label, status_level = ("Position open", "ok")
            else:
                status_label, status_level = ("Scanning", "ok")

        # Positions: GriffOpenPosition → display dict.
        positions: list[dict] = []
        for p in self._pm.open_positions:
            # Estimate PnL using mark price = entry (we don't see ticks).
            # Caller can wire spread_provider for a richer view later.
            positions.append({
                "symbol": p.symbol,
                "side": getattr(p.side, "name", str(p.side)),
                "lots": float(p.lots),
                "entry": float(p.entry_price),
                "pnl": 0.0,
            })

        return DashboardSnapshot(
            broker_label=self._broker_label,
            account=self._account,
            mode=self._mode,
            balance=balance,
            starting_equity=self._starting_equity,
            closed_pnl=self._daily.state.closed_pnl,
            floating_pnl=floating,
            wins=self._wins,
            losses=self._losses,
            trades_today=self._daily.trade_count,
            max_trades_per_day=self._max_trades,
            daily_dd_pct=(
                (self._daily.state.max_dd_today / self._starting_equity * 100.0)
                if self._starting_equity > 0 else 0.0
            ),
            daily_cap_pct=self._cap_pct,
            pairs=self._pairs,
            spread_by_pair=spread_by_pair,
            session_label=sess,
            session_level=sess_level,
            open_positions=positions,
            status_label=status_label,
            status_level=status_level,
            now_utc_str=now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        )

    # ----------------------------------------------------------- renderers
    def render_once(self) -> str:
        return render(self._snapshot())

    def print_once(self) -> None:
        self._sink(self.render_once())

    # ----------------------------------------------------------- async loop
    async def run(self, stop: asyncio.Event) -> None:
        """Periodic refresh loop. Cancelled by setting `stop`."""
        # First render immediately so the user sees the panel as soon as
        # the loop starts (rather than waiting `refresh_sec` for nothing).
        self.print_once()
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._refresh_sec)
            except asyncio.TimeoutError:
                self.print_once()


__all__ = [
    "ConsoleDashboard",
    "DashboardSnapshot",
    "render",
    "session_label",
]
