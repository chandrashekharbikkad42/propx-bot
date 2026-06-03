"""Asian Sweep V5 — exit-management state machine.

Ported from `multi_pair_backtest.simulate`. One state object per open
position, fed one closed 1H bar at a time. Returns a list of ExitActions
the engine applies (modify SL on the broker, partial-close, full-close).

Per-bar decision order (must match the backtest exactly — order changes
the trade count):

    1. SL / Trail hit       (close full, reason = "SL" or "TRAIL")
    2. TP1 hit (if not yet) (partial 50%, SL → BE, reason = "PARTIAL_TP1")
    3. Trail update after TP1 (move SL closer to price by 0.3R from close)
    4. TP2 hit (if TP1 was hit) (close runner, reason = "TP2")

Force-close at session boundary (16:00 UTC) and degenerate EOD scenarios
are handled by `force_close_eod`.

PnL accounting mirrors the backtest:
    point-based PnL = price_diff * lots * contract_size
    partial path     = diff(tp1, e) * fraction + diff(exit, e) * (1-fraction)
                       then × lots × contract_size
    JPY pairs        = pnl / 150

The exit module is the BACKTEST's mover. It does NOT call the broker —
the engine consumes ExitActions and issues `modify_sl` / `partial_close`
/ `close_position` against `GriffOrderRouter`.

Hinglish: ek bar pe ek bar ko process karo — pehle SL/Trail check, phir
TP1 partial, phir trail update, phir TP2. EOD pe force flat. Backtest ka
sequence as-is — order change matlab number badal jata hai.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from config.asian_sweep_config import (
    PAIR_CONFIG,
    PARTIAL_CLOSE_FRACTION,
    RR_TP1,
    RR_TP2,
    TRAILING_STEP_R,
    MIN_SL_DISTANCE_PIPS,
    MAX_RISK_USD_PER_TRADE,
    risk_pct_for,
)
from data.bar_aggregator import Bar
from strategy.patterns.base import Direction, PatternSignal
from utils.logger import logger


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExitAction:
    """One imperative for the engine. Fields are set ONLY for the relevant
    branch — engine looks at the booleans first.
    """
    # Mutually exclusive flavors:
    close_full: bool = False           # close the whole position
    partial_close: float = 0.0         # fraction of remaining lots to close
    modify_sl: Optional[float] = None  # new SL price
    # Bookkeeping (set on close_full / final partial outcome):
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "SL"|"TRAIL"|"TP2"|"EOD"|"EOD_trail"|"PARTIAL_TP1"


@dataclass
class ExitState:
    """Mutable per-position state. Engine creates via `init_exit_state`."""
    position_id: str
    symbol: str
    direction: Direction
    entry: float
    sl: float                          # CURRENT SL (mutates: BE → trail)
    tp1: float
    tp2: float
    initial_risk: float                # |entry - sl_initial| price distance
    initial_lots: float                # lot size at entry (before partial)
    remaining_lots: float              # what's still open
    tp1_hit: bool = False
    closed: bool = False
    # Realised PnL components, tracked for engine reporting (USD per
    # contract_size — NOT including JPY conversion; engine totals via
    # `compute_pnl` below for the final close-out).
    partial_exit_price: Optional[float] = None
    final_exit_price: Optional[float] = None
    final_exit_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------

def init_exit_state(
    *,
    position_id: str,
    signal: PatternSignal,
    lots: float,
) -> ExitState:
    """Build an ExitState from the signal + sized lot count.

    TP1 is recovered from `signal.confluences_met` if encoded (the
    AsianSweepDetector tags it as `"tp1_<float>"`); falls back to
    entry ± RR_TP1 × risk otherwise.
    """
    entry = signal.entry
    sl = signal.sl
    tp2 = signal.tp
    risk = abs(entry - sl)
    tp1 = _extract_tp1(signal) or _default_tp1(entry, sl, signal.direction, risk)
    return ExitState(
        position_id=position_id,
        symbol=signal.symbol,
        direction=signal.direction,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        initial_risk=risk,
        initial_lots=lots,
        remaining_lots=lots,
    )


def _extract_tp1(signal: PatternSignal) -> Optional[float]:
    for tag in signal.confluences_met:
        if isinstance(tag, str) and tag.startswith("tp1_"):
            try:
                return float(tag[4:])
            except ValueError:
                return None
    return None


def _default_tp1(
    entry: float, sl: float, direction: Direction, risk: float
) -> float:
    return entry + RR_TP1 * risk if direction == Direction.BUY else entry - RR_TP1 * risk


# ---------------------------------------------------------------------------
# Per-bar maintenance — the heart of the V5 exit
# ---------------------------------------------------------------------------

def maintain_exit(state: ExitState, bar: Bar) -> List[ExitAction]:
    """Apply the V5 exit ladder to a closed bar. Returns 0..N actions.

    Mutates `state` in-place (SL, tp1_hit, closed, remaining_lots, exit
    bookkeeping). Engine should iterate the returned actions in order.
    """
    if state.closed:
        return []

    actions: List[ExitAction] = []

    # 1) SL / Trail hit — checked with the CURRENT sl (which may already be
    #    BE / trailed).
    if state.direction == Direction.BUY:
        sl_hit = bar.low <= state.sl
    else:
        sl_hit = bar.high >= state.sl

    if sl_hit:
        reason = "TRAIL" if state.tp1_hit else "SL"
        state.closed = True
        state.final_exit_price = state.sl
        state.final_exit_reason = reason
        state.remaining_lots = 0.0
        actions.append(ExitAction(
            close_full=True,
            exit_price=state.sl,
            exit_reason=reason,
        ))
        return actions

    # 2) TP1 partial — only if not yet, and only on a bar that touched TP1.
    if not state.tp1_hit:
        if state.direction == Direction.BUY:
            tp1_touched = bar.high >= state.tp1
        else:
            tp1_touched = bar.low <= state.tp1
        if tp1_touched:
            state.tp1_hit = True
            state.partial_exit_price = state.tp1
            state.sl = state.entry  # SL → BE
            partial_lots = state.remaining_lots * PARTIAL_CLOSE_FRACTION
            state.remaining_lots -= partial_lots
            actions.append(ExitAction(
                partial_close=PARTIAL_CLOSE_FRACTION,
                modify_sl=state.entry,
                exit_price=state.tp1,
                exit_reason="PARTIAL_TP1",
            ))

    # 3) Trail update — only AFTER TP1 was hit. Trail = close ± 0.3R.
    if state.tp1_hit and not state.closed:
        if state.direction == Direction.BUY:
            new_trail = bar.close - TRAILING_STEP_R * state.initial_risk
            if new_trail > state.sl:
                state.sl = new_trail
                actions.append(ExitAction(modify_sl=new_trail))
        else:
            new_trail = bar.close + TRAILING_STEP_R * state.initial_risk
            if new_trail < state.sl:
                state.sl = new_trail
                actions.append(ExitAction(modify_sl=new_trail))

    # 4) TP2 — only after TP1; on touch we close the runner.
    if state.tp1_hit and not state.closed:
        if state.direction == Direction.BUY:
            tp2_touched = bar.high >= state.tp2
        else:
            tp2_touched = bar.low <= state.tp2
        if tp2_touched:
            state.closed = True
            state.final_exit_price = state.tp2
            state.final_exit_reason = "TP2"
            state.remaining_lots = 0.0
            actions.append(ExitAction(
                close_full=True,
                exit_price=state.tp2,
                exit_reason="TP2",
            ))

    return actions


def force_close_eod(state: ExitState, exit_price: float) -> List[ExitAction]:
    """Session boundary (16:00 UTC) flatten. Backtest tags as 'EOD'
    (pre-TP1) or 'EOD_trail' (post-TP1).
    """
    if state.closed:
        return []
    reason = "EOD_trail" if state.tp1_hit else "EOD"
    state.closed = True
    state.final_exit_price = exit_price
    state.final_exit_reason = reason
    state.remaining_lots = 0.0
    return [ExitAction(
        close_full=True,
        exit_price=exit_price,
        exit_reason=reason,
    )]


# ---------------------------------------------------------------------------
# PnL accounting — mirrors `simulate` exactly so backtest/live PnL agree
# ---------------------------------------------------------------------------

def compute_pnl(state: ExitState, *, jpy: bool = False) -> float:
    """Realised USD PnL on the full trade lifecycle.

    Replicates the simulate() formula:
        diff = exit - entry (LONG) | entry - exit (SHORT)
        if tp1_hit AND not pure-TP2:
            pnl = (diff(tp1,e)*FRAC + diff(exit,e)*(1-FRAC)) * lots * ct
        else:
            pnl = diff * lots * ct
        JPY pairs: pnl /= 150
    """
    if state.final_exit_price is None:
        return 0.0
    cfg = PAIR_CONFIG.get(state.symbol)
    if cfg is None:
        return 0.0
    ct = float(cfg["contract_size"])  # type: ignore[arg-type]
    lots = state.initial_lots
    e = state.entry
    ep = state.final_exit_price

    if state.direction == Direction.BUY:
        diff = ep - e
        d1 = state.tp1 - e
    else:
        diff = e - ep
        d1 = e - state.tp1

    if state.tp1_hit and state.final_exit_reason != "TP2":
        pnl = (d1 * PARTIAL_CLOSE_FRACTION
               + diff * (1.0 - PARTIAL_CLOSE_FRACTION)) * lots * ct
    else:
        pnl = diff * lots * ct

    if jpy:
        pnl /= 150.0
    return pnl


# ---------------------------------------------------------------------------
# Position sizing — point-based, mirrors simulate() exactly
# ---------------------------------------------------------------------------

def size_position(
    symbol: str,
    *,
    equity: float,
    sl_distance_price: float,
    month: Optional[int] = None,
    min_lots: float = 0.01,
) -> float:
    """Compute lots from risk budget + per-pair contract spec.

    Formula (backtest):
        risk_amt        = equity * risk_pct / 100
        risk_pts_count  = sl_distance_price / point
        vpl             = contract_size * point   # value per lot per point
        lot             = clamp(risk_amt / (risk_pts_count * vpl), 0.01, lot_max)

    `risk_override` per pair (XAUUSD=0.5%) is resolved via `risk_pct_for`,
    which also applies the weak-month dampener when `month ∈ WEAK_MONTHS`.
    """
    cfg = PAIR_CONFIG.get(symbol)
    if cfg is None:
        return min_lots
    pt = float(cfg["point"])             # type: ignore[arg-type]
    ct = float(cfg["contract_size"])     # type: ignore[arg-type]
    lmax = float(cfg["lot_max"])         # type: ignore[arg-type]

    if equity <= 0 or sl_distance_price <= 0 or pt <= 0:
        return min_lots

    # SAFETY CAP #1 — MIN SL FLOOR. Reject degenerate SLs before they can
    # produce a huge lot count. 1 pip = 10 broker points (MT5 convention).
    pip_size = pt * 10.0
    sl_pips = sl_distance_price / pip_size
    if sl_pips < MIN_SL_DISTANCE_PIPS:
        logger.warning(
            f"size_position REJECT {symbol}: SL distance {sl_pips:.2f} pips "
            f"< MIN_SL_DISTANCE_PIPS={MIN_SL_DISTANCE_PIPS} "
            f"(sl_distance_price={sl_distance_price})"
        )
        return 0.0

    risk_pct = risk_pct_for(symbol, month=month)
    risk_amt = equity * risk_pct / 100.0
    risk_pts_count = sl_distance_price / pt
    vpl = ct * pt
    if risk_pts_count <= 0 or vpl <= 0:
        return min_lots
    raw = risk_amt / (risk_pts_count * vpl)
    lot = round(max(min_lots, min(raw, lmax)), 2)
    lot = max(min_lots, lot)

    # SAFETY CAP #2 — MAX USD RISK PER TRADE. Absolute ceiling: regardless
    # of equity/risk_pct, no single trade may risk more than the cap.
    actual_risk_usd = lot * risk_pts_count * vpl
    if actual_risk_usd > MAX_RISK_USD_PER_TRADE:
        capped_raw = MAX_RISK_USD_PER_TRADE / (risk_pts_count * vpl)
        capped_lot = round(min(capped_raw, lmax), 2)
        logger.warning(
            f"size_position SCALE {symbol}: actual_risk_usd "
            f"${actual_risk_usd:.2f} > MAX_RISK_USD_PER_TRADE="
            f"${MAX_RISK_USD_PER_TRADE:.2f}; lots {lot:.2f} → {capped_lot:.2f}"
        )
        if capped_lot < min_lots:
            # Corner case (very wide SL on exotic instrument): even
            # min_lots breaches the cap — reject to preserve the
            # "absolute ceiling" guarantee.
            return 0.0
        lot = capped_lot
    return lot


__all__ = [
    "ExitAction",
    "ExitState",
    "init_exit_state",
    "maintain_exit",
    "force_close_eod",
    "compute_pnl",
    "size_position",
    "RR_TP1",
    "RR_TP2",
]
