"""S/R Flip — exit-variant sweep.

Tests 6 exit variants against the SAME S/R Flip detector signals (entries +
SL stay identical). The detector runs ONCE to cache every signal emission;
each variant then replays the cached signals through its own exit policy.

Variants:
  A: Fixed 2.0R TP, full close, no partial, no trail
  B: Fixed 1.5R TP, full close, no partial, no trail
  C: Fixed 3.0R TP, full close, no partial, no trail
  D: No TP; once price reaches 1R, trail SL to prior LTF swing low/high
  E: Full hold until opposite rejection candle prints (no fixed TP)
  F: 50% close at 2R + runner trails at 0.3R, SL→BE at 1R

Realistic costs preserved across all variants:
  - $10k start, 0.5% risk, 30 pairs, 1H+15M, 2yr
  - Slippage = 0.5 × MAX_SPREAD on market entry
  - Commission $7/lot/round-turn
  - Spread guard, daily DD 5%, total DD 10% (LIVE-realistic), MAX_CONCURRENT=2
  - Per-(pair, day) cap = 2, 48h time stop, Friday 23:45 flatten

Run:
    venv\\Scripts\\python.exe scripts/sr_flip_exit_sweep.py
    venv\\Scripts\\python.exe scripts/sr_flip_exit_sweep.py --days 365
"""
from __future__ import annotations
import argparse
import io
import math
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, io.UnsupportedOperation):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.sr_flip_config import (  # noqa: E402
    COMMISSION_PER_LOT_ROUNDTURN_USD, CONTRACT_SIZE,
    L_SWING, MAX_SPREAD_PIPS, MAX_TRADES_PER_DAY_PER_PAIR,
    MIN_RISK_PIPS, PAIRS as ALL_PAIRS, PIP_SIZE, RISK_PCT,
    SLIPPAGE_MARKET_FRAC_OF_MAX_SPREAD,
    TIME_STOP_HOURS, pip_size_for, sl_buffer_pips_for,
)
from data.bar_aggregator import Bar, read_bars_parquet  # noqa: E402
from strategy.patterns.base import Direction, MarketContext  # noqa: E402
from strategy.patterns.sr_flip import SRFlipDetector  # noqa: E402
from strategy.patterns._multi_setup_common import (  # noqa: E402
    is_bearish_pin, is_bearish_engulf,
    is_bullish_pin, is_bullish_engulf,
)


UTC = timezone.utc
INITIAL_BALANCE: float = 10_000.0
DAILY_DD_PCT_HARD: float = 5.0
TOTAL_DD_PCT_HARD: float = 10.0
MAX_CONCURRENT: int = 2
LOT_MIN: float = 0.01
LOT_STEP: float = 0.01
LOT_MAX: float = 50.0
HTF_WINDOW: int = 260
LTF_WINDOW: int = 80
TRAIL_STEP_R: float = 0.30


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def df_to_bars(df: pd.DataFrame, symbol: str) -> List[Bar]:
    out: List[Bar] = []
    for r in df.itertuples():
        out.append(Bar(
            symbol=symbol, time_msc=int(r.time_msc),
            open=float(r.open), high=float(r.high), low=float(r.low),
            close=float(r.close), volume=int(r.volume),
            spread_mean=float(getattr(r, "spread_mean", 0.0)),
        ))
    return out


def load_all_pairs(
    pairs: Tuple[str, ...], lookback_days: Optional[int] = None,
) -> Tuple[Dict[str, List[Bar]], Dict[str, List[Bar]]]:
    htf: Dict[str, List[Bar]] = {}
    ltf: Dict[str, List[Bar]] = {}
    for sym in pairs:
        try:
            d1 = read_bars_parquet(sym, "1H").sort_values("time_msc")
            d15 = read_bars_parquet(sym, "15M").sort_values("time_msc")
        except FileNotFoundError as exc:
            print(f"  [skip] {sym}: {exc}")
            continue
        if lookback_days is not None:
            cutoff_msc = int(d15["time_msc"].max()) - lookback_days * 24 * 60 * 60 * 1000
            d1 = d1[d1["time_msc"] >= cutoff_msc - 7 * 24 * 60 * 60 * 1000]
            d15 = d15[d15["time_msc"] >= cutoff_msc]
        htf[sym] = df_to_bars(d1.reset_index(drop=True), sym)
        ltf[sym] = df_to_bars(d15.reset_index(drop=True), sym)
        print(f"  {sym:<8} 1H={len(htf[sym]):>6}  15M={len(ltf[sym]):>6}")
    return htf, ltf


def pip_value_usd_per_lot(symbol: str, current_prices: Dict[str, float]) -> float:
    pip = PIP_SIZE[symbol]
    contract = CONTRACT_SIZE[symbol]
    raw = pip * contract
    if symbol in ("XAUUSD", "XAGUSD"):
        return raw
    quote = symbol[3:6]
    if quote == "USD":
        return raw
    usd_quote = f"USD{quote}"
    quote_usd = f"{quote}USD"
    if usd_quote in current_prices and current_prices[usd_quote] > 0:
        return raw / current_prices[usd_quote]
    if quote_usd in current_prices and current_prices[quote_usd] > 0:
        return raw * current_prices[quote_usd]
    return 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Cached signal
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SignalEvent:
    t_msc: int
    symbol: str
    ltf_idx: int
    direction: str          # "BUY" / "SELL"
    flip_dir: str           # "long" / "short"
    entry_intent: float     # raw detector entry (no slippage)
    sl: float               # raw detector SL
    tp2: float              # detector's TP2 (informational only — variants override)
    pip: float
    spread_pips_at_signal: float


def collect_signals(
    htf_bars: Dict[str, List[Bar]],
    ltf_bars: Dict[str, List[Bar]],
    quiet: bool = False,
) -> List[SignalEvent]:
    """Walk events chronologically and cache every signal the detector emits.
    Filters (spread / caps / DD halts) are NOT applied here — variants apply
    them in their own replay so each variant gets identical signal candidates.
    """
    detector = SRFlipDetector()
    cached: List[SignalEvent] = []

    events: List[Tuple[int, str, int]] = []
    for sym, bars in ltf_bars.items():
        for i, b in enumerate(bars):
            events.append((b.time_msc, sym, i))
    events.sort()
    if not events:
        return cached

    htf_cursor: Dict[str, int] = {sym: -1 for sym in htf_bars}
    t0 = time.time()
    report_every = max(1, len(events) // 40)

    for ev_no, (t_msc, sym, k) in enumerate(events):
        ltf = ltf_bars[sym]
        htf = htf_bars[sym]
        cursor = htf_cursor[sym]
        while cursor + 1 < len(htf) and htf[cursor + 1].time_msc < t_msc:
            cursor += 1
        htf_cursor[sym] = cursor
        if cursor < 30:
            continue
        htf_lo = max(0, cursor - HTF_WINDOW + 1)
        htf_win = htf[htf_lo: cursor + 1]
        ltf_lo = max(0, k - LTF_WINDOW + 1)
        ltf_win = ltf[ltf_lo: k + 1]
        ctx = MarketContext(
            symbol=sym, current_time_msc=t_msc,
            htf_bars=tuple(htf_win), ltf_bars=tuple(ltf_win),
        )
        try:
            sig = detector.detect(ltf_win, ctx)
        except Exception as exc:  # noqa: BLE001
            if not quiet:
                print(f"  [detector-err] {sym}: {exc}")
            continue
        if sig is None:
            continue

        cur = ltf[k]
        flip_dir = "long" if sig.direction == Direction.BUY else "short"
        cached.append(SignalEvent(
            t_msc=t_msc, symbol=sym, ltf_idx=k,
            direction="BUY" if sig.direction == Direction.BUY else "SELL",
            flip_dir=flip_dir,
            entry_intent=sig.entry, sl=sig.sl, tp2=sig.tp,
            pip=PIP_SIZE[sym],
            spread_pips_at_signal=cur.spread_mean / 10.0,
        ))

        if not quiet and ev_no % report_every == 0:
            pct = (ev_no / len(events)) * 100
            elapsed = time.time() - t0
            print(f"  cache {pct:5.1f}% | events={ev_no} | cached={len(cached)} "
                  f"| elapsed={elapsed:.0f}s")
    if not quiet:
        print(f"  cache done: {len(cached)} signals in {time.time()-t0:.1f}s")
    return cached


# ─────────────────────────────────────────────────────────────────────────────
# Open trade + exit variants
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpenTrade:
    variant: str
    symbol: str
    direction: str          # "BUY" / "SELL"
    flip_dir: str
    entry: float
    sl_initial: float
    sl: float
    tp: Optional[float]     # variant-set target (None for D, E)
    risk_price: float
    lot_full: float
    lot_remaining: float
    pip_value_usd: float
    entry_time_msc: int
    entry_ltf_idx: int
    commission_usd: float
    realized_pnl_usd: float = 0.0
    # Variant state
    armed: bool = False            # D: trail started; F: 1R reached → SL@BE
    partial_taken: bool = False    # F: 2R partial closed
    runner_high_water: float = 0.0 # F: trail high-water mark


def _close_trade(trade: OpenTrade, bar: Bar, exit_price: float, reason: str) -> Dict:
    return {
        "variant": trade.variant, "symbol": trade.symbol,
        "direction": trade.direction, "flip_dir": trade.flip_dir,
        "entry_time": datetime.fromtimestamp(trade.entry_time_msc / 1000, tz=UTC),
        "exit_time": datetime.fromtimestamp(bar.time_msc / 1000, tz=UTC),
        "entry": trade.entry, "sl_initial": trade.sl_initial,
        "exit_price": exit_price, "exit_reason": reason,
        "lot": trade.lot_full,
        "risk_price": trade.risk_price,
        "pnl_gross_usd": round(trade.realized_pnl_usd, 4),
        "commission_usd": round(trade.commission_usd, 4),
        "pnl_net_usd": round(trade.realized_pnl_usd - trade.commission_usd, 4),
    }


def _book_remaining(trade: OpenTrade, exit_price: float) -> None:
    pip = pip_size_for(trade.symbol)
    if trade.direction == "BUY":
        pnl = (exit_price - trade.entry) * trade.lot_remaining * (1 / pip) * trade.pip_value_usd
    else:
        pnl = (trade.entry - exit_price) * trade.lot_remaining * (1 / pip) * trade.pip_value_usd
    trade.realized_pnl_usd += pnl


def force_close_at_market(trade: OpenTrade, bar: Bar, reason: str) -> Dict:
    _book_remaining(trade, bar.close)
    return _close_trade(trade, bar, bar.close, reason)


# ─────────────────────── exit fn signature ───────────────────────
# fn(trade, bar, prev_bar, ltf_bars, ltf_idx, max_spread_pips) -> Optional[Dict]
# Returns trade dict if closed, None to keep open.

def _exit_fixed_tp(trade: OpenTrade, bar: Bar) -> Optional[Dict]:
    long = trade.direction == "BUY"
    # SL first (conservative on same-bar SL+TP).
    if long and bar.low <= trade.sl:
        _book_remaining(trade, trade.sl)
        return _close_trade(trade, bar, trade.sl, "SL")
    if (not long) and bar.high >= trade.sl:
        _book_remaining(trade, trade.sl)
        return _close_trade(trade, bar, trade.sl, "SL")
    # TP
    if trade.tp is not None:
        if long and bar.high >= trade.tp:
            _book_remaining(trade, trade.tp)
            return _close_trade(trade, bar, trade.tp, "TP")
        if (not long) and bar.low <= trade.tp:
            _book_remaining(trade, trade.tp)
            return _close_trade(trade, bar, trade.tp, "TP")
    return None


def exit_A(trade, bar, prev_bar, ltf_bars, ltf_idx, max_spread_pips):
    return _exit_fixed_tp(trade, bar)


def exit_B(trade, bar, prev_bar, ltf_bars, ltf_idx, max_spread_pips):
    return _exit_fixed_tp(trade, bar)


def exit_C(trade, bar, prev_bar, ltf_bars, ltf_idx, max_spread_pips):
    return _exit_fixed_tp(trade, bar)


def exit_D(trade, bar, prev_bar, ltf_bars, ltf_idx, max_spread_pips):
    """No TP. Once price.high/low touches entry ± 1R, trail SL to most recent
    confirmed swing low (long) / swing high (short) since entry, minus/plus
    SL buffer. Exit on SL hit.
    """
    long = trade.direction == "BUY"
    # SL check first.
    if long and bar.low <= trade.sl:
        _book_remaining(trade, trade.sl)
        return _close_trade(trade, bar, trade.sl, "TRAIL" if trade.armed else "SL")
    if (not long) and bar.high >= trade.sl:
        _book_remaining(trade, trade.sl)
        return _close_trade(trade, bar, trade.sl, "TRAIL" if trade.armed else "SL")

    one_R = trade.risk_price
    if not trade.armed:
        reached = (
            (long and bar.high >= trade.entry + one_R)
            or ((not long) and bar.low <= trade.entry - one_R)
        )
        if reached:
            trade.armed = True

    if trade.armed:
        # Find most recent confirmed swing low/high since entry.
        lo = max(0, trade.entry_ltf_idx - L_SWING)
        window = ltf_bars[lo: ltf_idx + 1]
        sl_buf_price = sl_buffer_pips_for(trade.symbol) * pip_size_for(trade.symbol)
        if long:
            best_low = None
            for j in range(L_SWING, len(window) - L_SWING):
                b = window[j]
                ok = all(
                    b.low < window[j - k].low and b.low < window[j + k].low
                    for k in range(1, L_SWING + 1)
                )
                if ok:
                    if best_low is None or b.low > best_low:
                        best_low = b.low
            if best_low is not None:
                new_sl = best_low - sl_buf_price
                if new_sl > trade.sl:
                    trade.sl = new_sl
        else:
            best_high = None
            for j in range(L_SWING, len(window) - L_SWING):
                b = window[j]
                ok = all(
                    b.high > window[j - k].high and b.high > window[j + k].high
                    for k in range(1, L_SWING + 1)
                )
                if ok:
                    if best_high is None or b.high < best_high:
                        best_high = b.high
            if best_high is not None:
                new_sl = best_high + sl_buf_price
                if new_sl < trade.sl:
                    trade.sl = new_sl
    return None


def exit_E(trade, bar, prev_bar, ltf_bars, ltf_idx, max_spread_pips):
    """No TP. Exit when opposite rejection candle prints OR SL hit."""
    long = trade.direction == "BUY"
    if long and bar.low <= trade.sl:
        _book_remaining(trade, trade.sl)
        return _close_trade(trade, bar, trade.sl, "SL")
    if (not long) and bar.high >= trade.sl:
        _book_remaining(trade, trade.sl)
        return _close_trade(trade, bar, trade.sl, "SL")

    if prev_bar is None:
        return None
    if long:
        opp = is_bearish_pin(bar) or is_bearish_engulf(prev_bar, bar)
    else:
        opp = is_bullish_pin(bar) or is_bullish_engulf(prev_bar, bar)
    if opp:
        _book_remaining(trade, bar.close)
        return _close_trade(trade, bar, bar.close, "OPP_REJ")
    return None


def exit_F(trade, bar, prev_bar, ltf_bars, ltf_idx, max_spread_pips):
    """SL→BE at 1R, 50% close at 2R, runner trails at 0.3R afterward."""
    long = trade.direction == "BUY"
    pip = pip_size_for(trade.symbol)
    half_spread_price = 0.5 * max_spread_pips * pip
    one_R = trade.risk_price
    two_R = 2.0 * trade.risk_price

    # SL check first.
    if long and bar.low <= trade.sl:
        _book_remaining(trade, trade.sl)
        reason = "TRAIL" if trade.partial_taken else ("BE" if trade.armed else "SL")
        return _close_trade(trade, bar, trade.sl, reason)
    if (not long) and bar.high >= trade.sl:
        _book_remaining(trade, trade.sl)
        reason = "TRAIL" if trade.partial_taken else ("BE" if trade.armed else "SL")
        return _close_trade(trade, bar, trade.sl, reason)

    # Phase 1 → 2: at 1R, SL→BE+half_spread
    if not trade.armed:
        reached = (
            (long and bar.high >= trade.entry + one_R)
            or ((not long) and bar.low <= trade.entry - one_R)
        )
        if reached:
            trade.armed = True
            be = trade.entry + (half_spread_price if long else -half_spread_price)
            if long and be > trade.sl:
                trade.sl = be
            elif (not long) and be < trade.sl:
                trade.sl = be

    # Phase 2 → 3: at 2R, close 50%, start runner trail
    if trade.armed and not trade.partial_taken:
        hit_2R = (
            (long and bar.high >= trade.entry + two_R)
            or ((not long) and bar.low <= trade.entry - two_R)
        )
        if hit_2R:
            target = trade.entry + (two_R if long else -two_R)
            close_lot = trade.lot_remaining * 0.5
            if long:
                pnl = (target - trade.entry) * close_lot * (1 / pip) * trade.pip_value_usd
            else:
                pnl = (trade.entry - target) * close_lot * (1 / pip) * trade.pip_value_usd
            trade.realized_pnl_usd += pnl
            trade.lot_remaining -= close_lot
            trade.partial_taken = True
            trade.runner_high_water = target

    # Phase 3: trail by 0.3R on each bar close beyond high-water
    if trade.partial_taken:
        trail = TRAIL_STEP_R * trade.risk_price
        if long:
            if bar.close > trade.runner_high_water:
                trade.runner_high_water = bar.close
            new_sl = trade.runner_high_water - trail
            if new_sl > trade.sl:
                trade.sl = new_sl
        else:
            if bar.close < trade.runner_high_water:
                trade.runner_high_water = bar.close
            new_sl = trade.runner_high_water + trail
            if new_sl < trade.sl:
                trade.sl = new_sl
    return None


@dataclass(frozen=True)
class Variant:
    name: str
    setup_tp_R: Optional[float]   # used at entry; None for D/E
    exit_fn: Callable
    label: str


VARIANTS: List[Variant] = [
    Variant("A", 2.0, exit_A, "Fixed 2.0R TP, full close"),
    Variant("B", 1.5, exit_B, "Fixed 1.5R TP, full close"),
    Variant("C", 3.0, exit_C, "Fixed 3.0R TP, full close"),
    Variant("D", None, exit_D, "Trail after 1R, no TP"),
    Variant("E", None, exit_E, "Full hold, opposite rejection exit"),
    Variant("F", 2.0, exit_F, "50% @ 2R + runner trail, BE @ 1R"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Per-variant simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VariantState:
    name: str
    balance: float = INITIAL_BALANCE
    peak_balance: float = INITIAL_BALANCE
    open_trades: List[OpenTrade] = field(default_factory=list)
    closed_trades: List[Dict] = field(default_factory=list)
    day_start_balance: Dict[str, float] = field(default_factory=dict)
    pair_day_trade_count: Dict[Tuple[str, str], int] = field(default_factory=dict)
    equity_curve: List[Tuple[int, float]] = field(default_factory=list)
    total_dd_halted: bool = False
    current_prices: Dict[str, float] = field(default_factory=dict)
    n_signals_seen: int = 0
    blk_spread: int = 0
    blk_pair_cap: int = 0
    blk_daily_dd: int = 0
    blk_total_dd: int = 0
    blk_concurrent: int = 0
    blk_bad_levels: int = 0
    blk_min_risk: int = 0


def _date_key(t_msc: int) -> str:
    return datetime.fromtimestamp(t_msc / 1000, tz=UTC).strftime("%Y-%m-%d")


def _equity_log(state: VariantState, t_msc: int) -> None:
    if state.equity_curve and state.equity_curve[-1][1] == state.balance:
        state.equity_curve[-1] = (t_msc, state.balance)
        return
    state.equity_curve.append((t_msc, state.balance))


def _book_closed(state: VariantState, closed: Dict) -> None:
    state.balance += closed["pnl_net_usd"]
    if state.balance > state.peak_balance:
        state.peak_balance = state.balance
    state.closed_trades.append(closed)


def simulate_variant(
    variant: Variant,
    signals_by_key: Dict[Tuple[str, int], SignalEvent],
    ltf_bars: Dict[str, List[Bar]],
    apply_dd_halts: bool = True,
) -> VariantState:
    state = VariantState(name=variant.name)

    events: List[Tuple[int, str, int]] = []
    for sym, bars in ltf_bars.items():
        for i, b in enumerate(bars):
            events.append((b.time_msc, sym, i))
    events.sort()
    if not events:
        return state
    state.equity_curve.append((events[0][0], state.balance))

    for t_msc, sym, k in events:
        ltf = ltf_bars[sym]
        cur = ltf[k]
        prev = ltf[k - 1] if k >= 1 else None
        state.current_prices[sym] = cur.close

        # 1. Update all open trades on this pair against this bar.
        still_open: List[OpenTrade] = []
        for t in state.open_trades:
            if t.symbol != sym:
                still_open.append(t)
                continue
            elapsed_h = (t_msc - t.entry_time_msc) / (3600 * 1000)
            if elapsed_h >= TIME_STOP_HOURS:
                _book_closed(state, force_close_at_market(t, cur, "TIME_STOP"))
                continue
            dt = datetime.fromtimestamp(t_msc / 1000, tz=UTC)
            if dt.weekday() == 4 and dt.hour >= 23 and dt.minute >= 45:
                _book_closed(state, force_close_at_market(t, cur, "FRIDAY_FLATTEN"))
                continue
            max_spread = MAX_SPREAD_PIPS[t.symbol]
            res = variant.exit_fn(t, cur, prev, ltf, k, max_spread)
            if res is not None:
                _book_closed(state, res)
            else:
                still_open.append(t)
        state.open_trades = still_open

        # 2. Day key + DD halts.
        day_key = _date_key(t_msc)
        if day_key not in state.day_start_balance:
            state.day_start_balance[day_key] = state.balance

        if apply_dd_halts:
            if state.total_dd_halted:
                continue
            day_loss_pct = (
                (state.day_start_balance[day_key] - state.balance)
                / state.day_start_balance[day_key] * 100
            )
            if day_loss_pct >= DAILY_DD_PCT_HARD:
                state.blk_daily_dd += 1
                _equity_log(state, t_msc)
                continue
            total_dd_pct = (state.peak_balance - state.balance) / state.peak_balance * 100
            if total_dd_pct >= TOTAL_DD_PCT_HARD:
                state.total_dd_halted = True
                state.blk_total_dd += 1
                _equity_log(state, t_msc)
                continue

        # 3. Is there a cached signal here?
        sig = signals_by_key.get((sym, k))
        if sig is None:
            _equity_log(state, t_msc)
            continue
        state.n_signals_seen += 1

        # 4. Concurrent + per-(pair,day) caps.
        if len(state.open_trades) >= MAX_CONCURRENT:
            state.blk_concurrent += 1
            _equity_log(state, t_msc)
            continue
        pkey = (sym, day_key)
        if state.pair_day_trade_count.get(pkey, 0) >= MAX_TRADES_PER_DAY_PER_PAIR:
            state.blk_pair_cap += 1
            _equity_log(state, t_msc)
            continue

        # 5. Spread guard at entry bar.
        max_spread = MAX_SPREAD_PIPS[sym]
        if cur.spread_mean / 10.0 > max_spread:
            state.blk_spread += 1
            _equity_log(state, t_msc)
            continue

        # 6. Slippage on market entry.
        pip = PIP_SIZE[sym]
        slip_price = SLIPPAGE_MARKET_FRAC_OF_MAX_SPREAD * max_spread * pip
        is_long = sig.direction == "BUY"
        entry_px = sig.entry_intent + (slip_price if is_long else -slip_price)
        if is_long and not (sig.sl < entry_px):
            state.blk_bad_levels += 1; _equity_log(state, t_msc); continue
        if (not is_long) and not (entry_px < sig.sl):
            state.blk_bad_levels += 1; _equity_log(state, t_msc); continue

        risk_price = abs(entry_px - sig.sl)
        risk_pips = risk_price / pip
        if risk_pips < MIN_RISK_PIPS:
            state.blk_min_risk += 1; _equity_log(state, t_msc); continue

        # 7. Build variant target.
        tp_R = variant.setup_tp_R
        if tp_R is not None:
            tp_px = entry_px + (tp_R * risk_price if is_long else -tp_R * risk_price)
        else:
            tp_px = None

        # 8. Sizing.
        risk_usd = state.balance * (RISK_PCT / 100.0)
        pv_usd = pip_value_usd_per_lot(sym, state.current_prices)
        if pv_usd <= 0:
            _equity_log(state, t_msc); continue
        lot_raw = risk_usd / (risk_pips * pv_usd)
        lot = math.floor(lot_raw / LOT_STEP) * LOT_STEP
        if lot < LOT_MIN:
            _equity_log(state, t_msc); continue
        lot = min(lot, LOT_MAX)
        commission = COMMISSION_PER_LOT_ROUNDTURN_USD * lot

        trade = OpenTrade(
            variant=variant.name, symbol=sym,
            direction="BUY" if is_long else "SELL",
            flip_dir=sig.flip_dir,
            entry=entry_px, sl_initial=sig.sl, sl=sig.sl, tp=tp_px,
            risk_price=risk_price, lot_full=lot, lot_remaining=lot,
            pip_value_usd=pv_usd, entry_time_msc=t_msc,
            entry_ltf_idx=k, commission_usd=commission,
        )
        state.open_trades.append(trade)
        state.pair_day_trade_count[pkey] = state.pair_day_trade_count.get(pkey, 0) + 1
        _equity_log(state, t_msc)

    # Force-close remaining
    for t in list(state.open_trades):
        bars = ltf_bars[t.symbol]
        if not bars:
            continue
        _book_closed(state, force_close_at_market(t, bars[-1], "END_OF_DATA"))
    state.open_trades.clear()
    _equity_log(state, events[-1][0])
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _max_dd(equity: List[Tuple[int, float]]) -> float:
    peak = equity[0][1]
    mdd = 0.0
    for _, v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100.0
        if dd > mdd:
            mdd = dd
    return mdd


def _metrics(state: VariantState) -> Dict:
    trades = state.closed_trades
    n = len(trades)
    if n == 0:
        return dict(N=0, WR=0.0, PF=0.0, NetUSD=0.0, NetPct=0.0, MDD=0.0,
                    AvgWin=0.0, AvgLoss=0.0, TPD=0.0)
    df = pd.DataFrame(trades)
    wins = df[df["pnl_net_usd"] > 0]
    losses = df[df["pnl_net_usd"] < 0]
    wr = len(wins) / n * 100.0
    gw = wins["pnl_net_usd"].sum()
    gl = abs(losses["pnl_net_usd"].sum())
    pf = gw / gl if gl > 0 else float("inf")
    avg_win = wins["pnl_net_usd"].mean() if len(wins) else 0.0
    avg_loss = losses["pnl_net_usd"].mean() if len(losses) else 0.0
    net = state.balance - INITIAL_BALANCE
    net_pct = net / INITIAL_BALANCE * 100.0
    mdd = _max_dd(state.equity_curve)
    df["entry_date"] = pd.to_datetime(df["entry_time"]).dt.tz_convert(UTC).dt.date
    n_days = (df["entry_date"].max() - df["entry_date"].min()).days + 1
    tpd = n / max(1, n_days)
    return dict(N=n, WR=wr, PF=pf, NetUSD=net, NetPct=net_pct, MDD=mdd,
                AvgWin=avg_win, AvgLoss=avg_loss, TPD=tpd)


def print_comparison(per_variant: Dict[str, Dict]) -> None:
    print("\n" + "=" * 96)
    print("  S/R FLIP — EXIT-VARIANT SWEEP RESULTS")
    print("=" * 96)
    print(f"  {'V':<2} {'N':>5} {'WR%':>6} {'PF':>6} {'NetUSD':>10} {'NetPct':>8} "
          f"{'MDD%':>7} {'AvgWin':>8} {'AvgLoss':>9} {'TPD':>5}  Variant")
    print(f"  {'-'*2} {'-'*5} {'-'*6} {'-'*6} {'-'*10} {'-'*8} {'-'*7} {'-'*8} {'-'*9} {'-'*5}  {'-'*40}")
    for v in VARIANTS:
        m = per_variant[v.name]
        if m["N"] == 0:
            print(f"  {v.name:<2} {0:>5} {'-':>6} {'-':>6} {'-':>10} {'-':>8} "
                  f"{'-':>7} {'-':>8} {'-':>9} {'-':>5}  {v.label}")
            continue
        print(f"  {v.name:<2} {m['N']:>5} {m['WR']:>5.1f}% {m['PF']:>6.2f} "
              f"{m['NetUSD']:>+10.2f} {m['NetPct']:>+7.2f}% {m['MDD']:>6.2f}% "
              f"{m['AvgWin']:>+8.2f} {m['AvgLoss']:>+9.2f} {m['TPD']:>5.2f}  {v.label}")
    print("=" * 96)

    # Find best PF variant
    best = None
    for v in VARIANTS:
        m = per_variant[v.name]
        if m["N"] == 0:
            continue
        if best is None or m["PF"] > per_variant[best.name]["PF"]:
            best = v

    print()
    if best is None:
        print("  All variants produced zero trades.")
        return
    bm = per_variant[best.name]
    print(f"  Best variant by PF: {best.name} — {best.label}")
    print(f"    PF={bm['PF']:.2f}  WR={bm['WR']:.1f}%  Net={bm['NetPct']:+.2f}%  MDD={bm['MDD']:.2f}%")

    # Verdict
    print()
    print("  VERDICT — does any variant clear the KEEP bar (PF >= 1.3 AND WR >= 45% AND MDD <= 8%)?")
    revived = False
    for v in VARIANTS:
        m = per_variant[v.name]
        if m["N"] == 0:
            continue
        keep = (m["PF"] >= 1.3) and (m["WR"] >= 45.0) and (m["MDD"] <= 8.0)
        fix = (not keep) and (1.1 <= m["PF"] < 1.3)
        cut = m["PF"] < 1.1
        tag = "KEEP" if keep else ("FIX" if fix else ("CUT" if cut else "BORDERLINE"))
        print(f"    {v.name}: PF={m['PF']:.2f}  WR={m['WR']:.1f}%  MDD={m['MDD']:.2f}%  → {tag}")
        if keep:
            revived = True
    print()
    if revived:
        print("  S/R FLIP STATUS: REVIVED — at least one exit variant clears KEEP threshold.")
    else:
        # any PF > 1.3 alone?
        any_pf = any(per_variant[v.name]["N"] > 0 and per_variant[v.name]["PF"] >= 1.3
                     for v in VARIANTS)
        if any_pf:
            print("  S/R FLIP STATUS: PARTIALLY REVIVED — a variant cleared PF≥1.3 but "
                  "missed WR/MDD. Operator review required.")
        else:
            print("  S/R FLIP STATUS: STAYS CUT — no exit variant clears PF≥1.3.")
            print("  The detector signal itself is the weakness, not just the exit ladder.")
    print("=" * 96)


def write_per_variant_csv(per_variant_states: Dict[str, VariantState], path: Path) -> None:
    all_rows: List[Dict] = []
    for name, st in per_variant_states.items():
        for tr in st.closed_trades:
            row = dict(tr)
            row["variant"] = name
            all_rows.append(row)
    if not all_rows:
        return
    pd.DataFrame(all_rows).to_csv(path, index=False)
    print(f"\n[SAVED] {path}  ({len(all_rows)} trades across all variants)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=None)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--out-csv", default="sr_flip_exit_sweep_trades.csv")
    p.add_argument("--cache-file", default="sr_flip_signal_cache.pkl",
                   help="Pickle of cached signals + pair list. Reused if matches.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.pairs:
        pairs = tuple(p.strip().upper() for p in args.pairs.split(",") if p.strip())
    else:
        pairs = ALL_PAIRS

    print("\n" + "=" * 72)
    print(f"  S/R Flip exit-variant sweep | pairs={len(pairs)} "
          f"| lookback={args.days or 'full'} days")
    print("=" * 72)
    htf, ltf = load_all_pairs(pairs, lookback_days=args.days)
    if not ltf:
        print("[!] No data loaded.")
        return 1

    cache_key = (tuple(sorted(ltf.keys())), args.days)
    cache_path = Path(args.cache_file)
    signals: Optional[List[SignalEvent]] = None
    if cache_path.exists():
        try:
            with cache_path.open("rb") as fh:
                payload = pickle.load(fh)
            if payload.get("key") == cache_key:
                signals = payload["signals"]
                print(f"\n  STEP 1 — loaded {len(signals)} signals from cache "
                      f"({cache_path})")
            else:
                print(f"\n  STEP 1 — cache mismatch (pairs/days changed); rebuilding")
        except Exception as exc:  # noqa: BLE001
            print(f"\n  STEP 1 — cache unreadable ({exc}); rebuilding")
    if signals is None:
        print("\n  STEP 1 — caching detector signals (one pass)...")
        t0 = time.time()
        signals = collect_signals(htf, ltf, quiet=args.quiet)
        print(f"  Cached {len(signals)} signals in {time.time()-t0:.1f}s")
        try:
            with cache_path.open("wb") as fh:
                pickle.dump({"key": cache_key, "signals": signals}, fh)
            print(f"  [saved cache → {cache_path}]")
        except Exception as exc:  # noqa: BLE001
            print(f"  [cache save failed: {exc}]")
    signals_by_key = {(s.symbol, s.ltf_idx): s for s in signals}

    # Run BOTH modes back-to-back: live-realistic (DD halts ON) + full-2yr
    # diagnostic (DD halts OFF). The diagnostic is the honest "is the exit
    # the lever?" answer — the live-realistic shows what a prop firm would see.
    for mode_name, apply_dd in (("LIVE-REALISTIC (DD halts ON)", True),
                                ("FULL-2YR DIAGNOSTIC (DD halts OFF)", False)):
        print(f"\n  STEP 2/{mode_name} — replaying each variant against cached signals...")
        per_variant_states: Dict[str, VariantState] = {}
        per_variant_metrics: Dict[str, Dict] = {}
        for v in VARIANTS:
            t1 = time.time()
            st = simulate_variant(v, signals_by_key, ltf, apply_dd_halts=apply_dd)
            per_variant_states[v.name] = st
            per_variant_metrics[v.name] = _metrics(st)
            m = per_variant_metrics[v.name]
            dt = time.time() - t1
            print(f"    Variant {v.name} done in {dt:.1f}s | N={m['N']} | PF={m['PF']:.2f} | "
                  f"Net={m['NetPct']:+.2f}% | MDD={m['MDD']:.2f}% | {v.label}")

        print(f"\n  === {mode_name} ===")
        print_comparison(per_variant_metrics)
        out_path = Path(args.out_csv)
        if not apply_dd:
            out_path = out_path.with_name(out_path.stem + "_nodd" + out_path.suffix)
        write_per_variant_csv(per_variant_states, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
