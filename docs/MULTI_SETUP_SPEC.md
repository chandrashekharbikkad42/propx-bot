# propX Multi-Setup — Strategy Specification v1.0

**Status:** DRAFT (Phase 1 — design + data). Detector code not yet written.
**Author:** Claude (under direction of @chandrashekharbikkad42)
**Date:** 2026-05-26
**Sibling-strategy:** `strategy/patterns/asian_sweep.py` (V5, untouched).

This document is the **contract** that the four setup detectors must implement.
Every threshold below has a sensible default we can tune in walk-forward
optimisation. Any rule that says "see §X" is binding — no detector may invent
its own definition.

> Hinglish: yeh spec ek baar pakka lock karo, fir detector code likhna. Har
> threshold tunable hai but default abhi yahin tay hai. Vague rules = unbacktestable.

---

## 0. Scope, universe, risk envelope

| Item | Value |
|---|---|
| Universe | 28 pairs (`MULTI_SETUP_PAIRS` in `scripts/capture_historical_bars.py`) |
| Structure timeframe (HTF) | **1H** |
| Entry timeframe (LTF) | **15M** |
| Sessions | **24h** (Asia + London + NY) — no session filter |
| Risk per trade | **0.5 %** of account equity (FIXED) |
| Max concurrent open trades | **2 globally** (across all pairs/setups) |
| Max trades/day | **2 globally** (entries — not closes) |
| TP ladder | TP1 = **1.5 R** (close 50 %), TP2 = **2.5 R** (close remainder) |
| Break-even shift | At **+1.0 R** unrealised, move SL to entry ± 1× spread |
| Trailing | After TP1 hit, trail SL by **0.3 R** behind close (15M close) |
| Confluence boost | 2+ setups firing on same pair/level within ±5 pips & ±3 bars (15M) → mark `confluence=True`. Risk stays 0.5 % (no upsize); flag is for analytics & priority selection. |
| Conflict rule | If two setups fire opposite directions on same pair within ±5 pips & ±3 bars → SKIP both (regime noise). |
| Daily DD circuit | Reuse existing 3 % intraday DD kill-switch (compliance layer). |
| Account DD | Reuse existing 5 % daily / 10 % total prop-firm caps (compliance layer). |
| News blackout | Reuse `data/news_calendar.py` — high-impact news ± 2 min, no new entries. |
| Kill-switches | Reuse all 7 existing kill-switches (see compliance layer). |

**Priority selection (when >2 setups qualify same day):**
1. `confluence=True` setups first.
2. Then by setup type rank: `BOS_RETEST` > `ORDER_BLOCK` > `LIQ_SWEEP` > `SR_REJECTION` (empirically structural setups tend higher PF; we will revisit after backtest).
3. Then by pair quality (reuse `quality_for(symbol)` from `asian_sweep_config`, fall back to 5 for unknown pairs in a new `multi_setup_config`).

---

## 1. Glossary & universal primitives

These primitives are referenced by every setup below. **Detectors MUST import
from a single helper module** (`strategy/multi_setup/primitives.py`, future) so
that the same definition is used everywhere.

### 1.1 Pip

`pip_size(symbol)`:
- JPY-quoted (`*JPY`): `0.01` (= 10 broker points at 3-digit)
- XAUUSD: `0.10` (= 10 broker points at 2-digit)
- All other FX: `0.0001` (= 10 broker points at 5-digit)

All pip-denominated thresholds below are converted at detector load time via
this helper; **never hard-code pips per pair**.

### 1.2 Swing high / swing low (HTF only — 1H)

A bar at index `i` is a **swing high** if:
- `high[i] > high[i-k]` and `high[i] > high[i+k]` for all `k ∈ [1, L_SWING]`.
- `L_SWING = 3` bars (default).

Equivalent for **swing low** with `low`.

- A bar can be both swing high and swing low only if `L_SWING` window is empty
  (cannot happen at `L_SWING ≥ 1`).
- "Last swing" = the most recent swing that has fully formed (i.e. the latest
  bar where `t > t_swing + L_SWING * 1h`); in-progress swings don't count.

### 1.3 Structure: HH / HL / LH / LL

Walk the sequence of confirmed swings (alternating high/low). Label each new
swing relative to the prior same-type swing:
- `HH` — higher high than prior swing high
- `LH` — lower high than prior swing high
- `HL` — higher low than prior swing low
- `LL` — lower low than prior swing low

**Trend state** (1H):
- `BULLISH` ↔ last two same-type pairs are both HH and HL.
- `BEARISH` ↔ last two same-type pairs are both LL and LH.
- `RANGE` ↔ neither (one HH + one LL, or insufficient swings).

### 1.4 Impulsive move

A sequence of `N_IMP = 3` consecutive same-direction 15M closes where:
- Total displacement ≥ `IMP_MIN_PIPS = 15 × atr_factor` pips, where
  `atr_factor = ATR14_15M / median(ATR14_15M, last 100 bars)` clamped to
  `[0.5, 2.0]`. (Default min ~7.5 pips weak regime, ~30 pips strong regime.)
- No single bar in the run reverses by > 30 % of that bar's range.
- Volume (tick count) of run ≥ 1.2 × median 15M volume over last 50 bars.

This definition is **direction-symmetric** (use absolute displacement).

### 1.5 ATR

`ATR14_1H` and `ATR14_15M` — standard Wilder ATR(14), computed on closed bars
only. Used for adaptive thresholds. **Never use the in-progress bar.**

### 1.6 Spread / cost guard

Per pair, define `MAX_SPREAD_PIPS`:
- Majors (EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF): **2.0 pips**
- EUR/GBP crosses (EURJPY, GBPJPY, EURGBP, EURCHF, GBPCHF): **3.0 pips**
- AUD/NZD/CAD crosses (AUDJPY, AUDCAD, AUDCHF, AUDNZD, NZDJPY, NZDCHF, NZDCAD, CADJPY, CHFJPY): **4.0 pips**
- Exotic crosses (EURAUD, EURNZD, EURCAD, GBPAUD, GBPNZD, GBPCAD): **5.0 pips**
- XAUUSD: **5.0 pips** (= 50 broker points at 2-digit)

If live tick spread > `MAX_SPREAD_PIPS` at signal time → **DO NOT ENTER**.

### 1.7 Buffers

- `SL_BUFFER_PIPS`: 2 pips for majors, 3 pips for crosses, 5 pips for XAUUSD.
  This is the cushion **beyond** the structural level (e.g. swing wick, OB extreme).
- `ENTRY_BUFFER_PIPS`: 0 pips by default — entries are limit/stop at structural
  price. Detector may add `0.5 × spread_pips` for slippage absorption (live only).

### 1.8 Rejection candle (LTF — 15M)

A 15M closed bar is a **rejection candle** if it satisfies ONE of:

**(a) Pin bar (long pin = bullish rejection):**
- `body = |close - open|`
- `range = high - low`, `range > 0`
- `lower_wick = min(open, close) - low`
- `body ≤ 0.33 × range`
- `lower_wick ≥ 2.0 × body`
- `lower_wick ≥ 0.55 × range`
- `close > open` (bullish body) — OPTIONAL relax: allow doji body if wick rule is met.

Bearish pin: mirror — `upper_wick ≥ 2.0 × body` and `≥ 0.55 × range`.

**(b) Bullish engulfing:**
- Previous bar was bearish (`close[-1] < open[-1]`).
- Current bar `open ≤ close[-1]` AND `close ≥ open[-1]`.
- Current bar body `(close - open) ≥ 1.0 × |close[-1] - open[-1]|`.
- Current bar body `(close - open) ≥ 0.4 × (high - low)` (real-body strength).

Bearish engulfing: mirror.

**(c) Volume confirmation (required for both pin & engulf):**
- Current bar tick volume ≥ 1.1 × mean tick volume of last 20 bars.

---

## 2. Setup #1 — Liquidity Sweep + Reversal

### 2.1 Trigger (long example; short = mirror)

1. Identify last confirmed **1H swing low** `SL` (per §1.2). The swing low
   price is `P_sweep_long`.
2. On 15M, detect a bar `B_sweep` where:
   - `low[B_sweep] < P_sweep_long - SWEEP_PENETRATION_PIPS`
   - `SWEEP_PENETRATION_PIPS = max(1.0, 0.10 × ATR14_15M_in_pips)` (default ≥ 1 pip but adaptive).
3. Within the next `SWEEP_RECLAIM_BARS = 3` 15M bars (inclusive of sweep bar):
   - At least one 15M bar closes back **above** `P_sweep_long`
     (call this `B_reclaim`; it may equal `B_sweep`).
4. `B_reclaim` must itself be a **rejection candle** (§1.8), OR the next closed
   15M bar after `B_reclaim` must be one.

### 2.2 Entry

- Type: **limit BUY** at `P_sweep_long + ENTRY_BUFFER_PIPS` (i.e. on retest of
  the swept level from above).
- Validity: limit expires `SWEEP_LIMIT_EXPIRY_BARS = 4` 15M bars after `B_reclaim`.
- Alternative (configurable): **market BUY** on close of confirming rejection
  candle. Default = **limit retest** (less slippage, fewer fills).

### 2.3 Stop loss

- `SL = low[B_sweep] - SL_BUFFER_PIPS`.
- This is the swept wick low, not the swing low.

### 2.4 Take profit

- `risk_pips = entry - SL`.
- `TP1 = entry + 1.5 × risk_pips` (close 50 %).
- `TP2 = entry + 2.5 × risk_pips` (close 50 %).
- Invariants: `risk_pips ≥ 5 pips` (FX) / `5 × pip` (XAU). If smaller → SKIP.

### 2.5 Invalidations / skip rules

- HTF trend `BEARISH` AND swept level is HH (not HL) → SKIP (counter-trend sweep
  of a high inside a downtrend = continuation, not reversal).
  - Equivalent: long sweep is taken only when swept level is a **swing low**
    (any trend) OR the prior structure is `RANGE` / `BULLISH`.
- News blackout active on this pair within ±2 min of `B_reclaim` close → SKIP.
- `spread > MAX_SPREAD_PIPS` at signal moment → SKIP.
- `B_sweep.low` is more than `SWEEP_MAX_PENETRATION_PIPS = 4 × ATR14_15M_in_pips`
  below `P_sweep_long` → SKIP (too deep; likely a real breakdown).

---

## 3. Setup #2 — Order Block

### 3.1 Order block definition (HTF — 1H)

A **bullish order block** (`OB_bull`) is the most recent **bearish 1H bar**
immediately preceding an **impulsive bullish move** (§1.4 adapted to 1H):
- The candidate OB bar `B_OB` has `close < open`.
- Bars `B_OB+1 .. B_OB+N_IMP_HTF` are all bullish closes.
- `N_IMP_HTF = 2` (1H impulse = 2 strong same-direction bars).
- Total displacement of impulse ≥ `OB_IMP_MIN_PIPS = max(20, 1.5 × ATR14_1H_in_pips)`.
- `B_OB+1.low ≥ B_OB.high` is NOT required (impulse may start from inside OB) —
  but `B_OB+N_IMP_HTF.close - B_OB.high ≥ OB_IMP_MIN_PIPS / 2` IS required
  (impulse must clear the OB by half its minimum displacement).

OB price zone: `[B_OB.low, B_OB.high]` (the full bar body+wicks).
- "OB extreme" (used for SL): `B_OB.low` for bullish OB; `B_OB.high` for bearish OB.

OB expires after **`OB_MAX_AGE_BARS = 200` 1H bars** (~8 trading days) without
a fill, OR after the OB zone is fully traded through (a 1H close beyond the
OB extreme invalidates it permanently — "OB mitigated by full breach").

Bearish OB: mirror.

### 3.2 Trigger

1. A live `OB_bull` exists with `now < created_at + OB_MAX_AGE_BARS × 1h`.
2. On 15M, price re-enters the OB zone: `low[B_retest] ≤ B_OB.high`.
3. Within the OB zone (`B_retest` and up to `OB_RETEST_BARS = 3` subsequent
   bars), a 15M **rejection candle** (§1.8) prints with bullish bias.

### 3.3 Entry

- Type: **limit BUY** at `B_OB.high + ENTRY_BUFFER_PIPS` (top of OB; first
  touch). Alternative aggressive entry: `(B_OB.high + B_OB.low) / 2` (mid-OB).
- Default = **top of OB**.
- Limit valid until OB invalidation OR `OB_RETEST_BARS × 15min` after first
  zone touch.

### 3.4 Stop loss

- `SL = B_OB.low - SL_BUFFER_PIPS`.

### 3.5 Take profit

- Same TP1=1.5R / TP2=2.5R ladder as §2.4. `risk_pips` invariant applies.

### 3.6 Invalidations / skip rules

- 1H close beyond `B_OB.low` (bullish OB) → OB invalidated, cancel any pending
  limit, exit any open OB-trade at market.
- HTF trend `BEARISH` → SKIP (counter-trend OB; we wait for trend alignment).
- 15M `B_retest` is itself a strong bearish bar that engulfs the OB high in one
  shot → SKIP (this is a breakdown, not a retest).

---

## 4. Setup #3 — Break of Structure + Retest

### 4.1 BoS definition (HTF — 1H)

A **bullish BoS** event fires on 1H when:
1. The most recent **confirmed swing high** is `SH` at price `P_BoS`.
2. A subsequent 1H bar `B_break` prints `close > P_BoS + BOS_BUFFER_PIPS`.
3. `BOS_BUFFER_PIPS = max(2, 0.10 × ATR14_1H_in_pips)`.
4. The trend state BEFORE `B_break` was `BEARISH` or `RANGE` — i.e. this
   break **changes the structure** (we don't count continuation breaks here;
   continuation BOS is implicitly handled by Order Block §3).

Bearish BoS: mirror with swing low + close below.

Once `B_break` fires, mark the broken level `P_BoS` as the **retest zone**:
- Zone = `[P_BoS - BOS_RETEST_TOLERANCE, P_BoS + BOS_RETEST_TOLERANCE]`.
- `BOS_RETEST_TOLERANCE = max(3, 0.15 × ATR14_1H_in_pips)` pips.
- Zone expires after `BOS_MAX_AGE_BARS = 100` 1H bars OR after a 1H close
  back below `P_BoS - BOS_BUFFER_PIPS` (BoS invalidated — structure flip
  rejected).

### 4.2 Trigger

1. A live bullish BoS retest zone exists.
2. On 15M, price re-enters zone from above: `low[B_retest] ≤ P_BoS + BOS_RETEST_TOLERANCE`.
3. Within `BOS_RETEST_BARS = 4` 15M bars of zone touch, a 15M bullish
   rejection candle (§1.8) prints with `low ≥ P_BoS - BOS_RETEST_TOLERANCE`.
   - If price closes below `P_BoS - BOS_RETEST_TOLERANCE` at any point during
     the retest window → SKIP and invalidate (deep break-back).

### 4.3 Entry

- Type: **market BUY** on close of confirming rejection candle.
  (Limit at `P_BoS` exact would miss many fills due to the tolerance.)
- Alternative: limit at `P_BoS + ENTRY_BUFFER_PIPS` valid for 2 bars. Default = market.

### 4.4 Stop loss

- `SL = min(P_BoS - BOS_RETEST_TOLERANCE, low of last 3 15M bars) - SL_BUFFER_PIPS`.
- Use the lower of the two so we always sit beyond the deepest recent wick.

### 4.5 Take profit

- TP1=1.5R, TP2=2.5R. `risk_pips ≥ 5 pips` invariant.

### 4.6 Invalidations / skip rules

- HTF trend was already `BULLISH` before `B_break` → SKIP (no structure change,
  use Order Block setup instead).
- News blackout window active at `B_retest` close → SKIP.
- The 1H bar containing `B_break` was driven by a news spike (news event
  within ±10 min of `B_break` open) → SKIP this BoS (news-driven breaks
  often fade).

---

## 5. Setup #4 — Support/Resistance Rejection

### 5.1 Level definition (HTF — 1H)

A **horizontal S/R level** is a price `P_level` such that:
- At least `LEVEL_MIN_TOUCHES = 3` 1H wicks have touched within
  `LEVEL_CLUSTER_TOLERANCE = max(3, 0.20 × ATR14_1H_in_pips)` pips of `P_level`
  over the lookback window `LEVEL_LOOKBACK_BARS = 200` 1H bars (~8 trading days).
- Two touches in the same 1H bar count as one.
- Touches are separated by at least `LEVEL_MIN_GAP_BARS = 5` 1H bars
  (filter out hugging price action that's not a true revisit).
- The level is a **resistance** if ≥ 2 of the touches were from below
  (high near level, close below); a **support** if ≥ 2 were from above
  (low near level, close above). A level can be both (flip zone).

The detector maintains a per-pair `level_book` of all currently-valid levels.
A level is **invalidated** when 1H closes beyond it by ≥ `LEVEL_BREAK_PIPS =
max(5, 0.30 × ATR14_1H_in_pips)`.

### 5.2 Cleanliness filter

A level is "clean" if:
- No competing level within ±`LEVEL_CLUSTER_TOLERANCE × 3` pips
  (no double S/R muddying the zone).
- Price has been away from the level for at least `LEVEL_MIN_GAP_BARS` bars
  since the last touch (so the rejection has time to develop).

**Only clean levels generate signals.**

### 5.3 Trigger (long example at support)

1. A clean support `P_sup` exists.
2. On 15M, a bar `B_test` tags the level: `low[B_test] ≤ P_sup + LEVEL_CLUSTER_TOLERANCE`.
3. Within `SR_REJECT_BARS = 3` 15M bars of tag, a 15M bullish rejection
   candle (§1.8) prints with `low ≥ P_sup - LEVEL_CLUSTER_TOLERANCE × 1.5`
   (allow slight wick break; reject if body closes below tolerance).

### 5.4 Entry

- Type: **market BUY** on close of confirming rejection candle.

### 5.5 Stop loss

- `SL = min(low of B_test .. confirming bar) - SL_BUFFER_PIPS`.

### 5.6 Take profit

- TP1=1.5R, TP2=2.5R.
- **Cap**: TP2 must not exceed the next opposing S/R level minus
  `LEVEL_CLUSTER_TOLERANCE`. If 2.5R is beyond that, set TP2 = (next_level -
  tol). If 1.5R is also beyond, SKIP (no room).

### 5.7 Invalidations / skip rules

- HTF trend `BEARISH` AND level is a support → reduce confidence but allow
  (range-bound bounce). If trend is `BEARISH` AND level is a **resistance**
  → SKIP (don't fight trend at trend resistance — handled by setups 1/3).
- News blackout within ±2 min → SKIP.
- Level age > `LEVEL_LOOKBACK_BARS` since last touch → mark stale, SKIP.

---

## 6. Confluence logic

Confluence = `True` when **two or more** setups fire on the **same pair, same
direction**, with both signals' entry prices within `CONFLUENCE_PRICE_TOL =
5 pips` and confirming bars within `CONFLUENCE_BAR_TOL = 3` 15M bars of
each other.

When confluence is detected:
- Use the **earlier** signal's entry/SL/TP for execution.
- Tag the trade `confluence=True` and `confluence_setups=["LIQ_SWEEP","OB"]`.
- Risk stays at 0.5 % (no upsize — prop-firm conservative).
- Confluence trades take priority over solo trades for the daily 2-trade cap.

When **opposite-direction** signals fire on the same pair within the same
tolerance window → both are **DISCARDED** (`reason="conflict"`).

---

## 7. Trade management (applies to all setups)

| Phase | Rule |
|---|---|
| Pre-fill | Limit orders expire per setup; market orders execute on close. |
| Post-fill | Compute `R = |entry - SL|`. Set TP1 at `entry ± 1.5R`, TP2 at `entry ± 2.5R`. |
| +0.5R | No action (no early scratch). |
| +1.0R | Move SL to `entry ± 1 × spread` (BE+spread). Log `BE_SHIFT` event. |
| TP1 hit | Close 50 % of position. Cancel TP1 leg. Start 0.3R trailing stop behind 15M closes. |
| Trail | Each new 15M close that advances by ≥ 0.3R → SL := `close - 0.3R` (long) / `close + 0.3R` (short). Never move SL backward. |
| TP2 hit | Close remainder. End trade. |
| SL hit | Close remainder. End trade. Log `SL_HIT` event with setup tag. |
| Time stop | After **48 hours** open without TP1 → close at market. (Most setups should resolve in <12h; 48h = stale.) |
| Daily flatten | At `23:55 UTC` Friday → close all open trades regardless of P&L. |

---

## 8. Backtest invariants (must hold for spec to be considered valid)

The following invariants must hold against the captured 2-year dataset:

1. **No look-ahead.** A signal at 15M bar `B` is computed using only bars
   `< B` (closed bars only). The HTF (1H) bar containing `B` is **only**
   available once it closes (so 15M bars `00:00..00:45` see 1H bar `23:00`
   of previous hour, not the in-progress one).
2. **Reproducible.** Same inputs → same outputs. No RNG, no time-of-day
   drift, no broker tick-by-tick variance in the backtest.
3. **Slippage model.** Apply `+0.5 × MAX_SPREAD_PIPS[pair]` to entry on
   market orders, `0` on limit fills, `+0` on SL fills (broker fills at
   stop). Conservative.
4. **Commission.** $7 per round-turn per standard lot (FTMO default).
   Subtract from R-multiple calculation.
5. **Equity tracking.** Daily DD must never exceed 4 % in any single day of
   the backtest. If it does, the spec needs a tighter daily kill-switch.

---

## 9. What this spec deliberately does NOT define

- **Indicator filters** (RSI, MACD, MAs) — the user requested **indicator-free**.
- **Machine-learning scoring** — out of scope for v1. Confluence is the only
  multi-signal logic.
- **Position scaling-in** — single fill per trade.
- **News-event direction trades** — strictly avoided via blackout.
- **Correlation hedging** — handled at portfolio layer (future); for now the
  global 2-trade cap is the only correlation guard.

---

## 10. Open questions for v1.1 (post-backtest)

1. Are the `SWEEP_RECLAIM_BARS`, `OB_RETEST_BARS`, `BOS_RETEST_BARS`,
   `SR_REJECT_BARS` defaults right? Walk-forward grid 2..6.
2. Should TP1 partial be 50 % or 33 %? Sensitivity test.
3. Is the 24h session window too loose? Try London-NY only as ablation.
4. Should confluence boost risk to 0.75 %? Test in walk-forward.
5. Per-pair tuning vs global tuning — start global, only specialise if a pair
   shows persistent over/under-performance.

---

## Appendix A — Constants table (single source of truth for v1.0 defaults)

```python
# All values v1.0 — to be moved to config/multi_setup_config.py when detectors land.

# Universal
HTF = "1H"
LTF = "15M"
RISK_PCT = 0.5
MAX_TRADES_PER_DAY = 2
MAX_CONCURRENT = 2
TP1_R = 1.5
TP2_R = 2.5
PARTIAL_FRACTION = 0.50
BE_SHIFT_R = 1.0
TRAIL_STEP_R = 0.30
TIME_STOP_HOURS = 48
FRIDAY_FLATTEN_UTC = "23:55"

# Swings / structure
L_SWING = 3

# Impulsive move (15M)
N_IMP = 3
IMP_MIN_PIPS_BASE = 15.0  # multiplied by adaptive ATR factor

# Confluence
CONFLUENCE_PRICE_TOL_PIPS = 5.0
CONFLUENCE_BAR_TOL = 3

# Setup #1 Liquidity Sweep
SWEEP_RECLAIM_BARS = 3
SWEEP_LIMIT_EXPIRY_BARS = 4
SWEEP_PENETRATION_PIPS_FLOOR = 1.0  # adaptive: max(floor, 0.10*ATR15M)
SWEEP_MAX_PENETRATION_ATR_MULT = 4.0

# Setup #2 Order Block
N_IMP_HTF = 2
OB_IMP_MIN_PIPS_FLOOR = 20.0  # adaptive: max(floor, 1.5*ATR1H)
OB_MAX_AGE_BARS_1H = 200
OB_RETEST_BARS = 3

# Setup #3 BoS + Retest
BOS_BUFFER_PIPS_FLOOR = 2.0  # adaptive: max(floor, 0.10*ATR1H)
BOS_RETEST_TOLERANCE_FLOOR = 3.0  # adaptive: max(floor, 0.15*ATR1H)
BOS_MAX_AGE_BARS_1H = 100
BOS_RETEST_BARS = 4

# Setup #4 S/R Rejection
LEVEL_MIN_TOUCHES = 3
LEVEL_CLUSTER_TOLERANCE_FLOOR = 3.0  # adaptive: max(floor, 0.20*ATR1H)
LEVEL_LOOKBACK_BARS_1H = 200
LEVEL_MIN_GAP_BARS = 5
LEVEL_BREAK_PIPS_FLOOR = 5.0  # adaptive: max(floor, 0.30*ATR1H)
SR_REJECT_BARS = 3

# Rejection candle
PIN_BODY_MAX_FRAC = 0.33
PIN_WICK_MIN_BODY_MULT = 2.0
PIN_WICK_MIN_RANGE_FRAC = 0.55
ENGULF_BODY_MIN_MULT = 1.0
ENGULF_BODY_MIN_RANGE_FRAC = 0.40
REJECT_VOL_MIN_MULT = 1.1

# Spread / cost guard
MAX_SPREAD_PIPS = {
    # Majors
    "EURUSD": 2.0, "GBPUSD": 2.0, "USDJPY": 2.0, "AUDUSD": 2.0,
    "NZDUSD": 2.0, "USDCAD": 2.0, "USDCHF": 2.0,
    # EUR/GBP crosses
    "EURJPY": 3.0, "GBPJPY": 3.0, "EURGBP": 3.0,
    "EURCHF": 3.0, "GBPCHF": 3.0,
    # AUD/NZD/CAD crosses
    "AUDJPY": 4.0, "AUDCAD": 4.0, "AUDCHF": 4.0, "AUDNZD": 4.0,
    "NZDJPY": 4.0, "NZDCHF": 4.0, "NZDCAD": 4.0,
    "CADJPY": 4.0, "CHFJPY": 4.0,
    # Exotic crosses
    "EURAUD": 5.0, "EURNZD": 5.0, "EURCAD": 5.0,
    "GBPAUD": 5.0, "GBPNZD": 5.0, "GBPCAD": 5.0,
    # Metal
    "XAUUSD": 5.0,
}

# SL buffer (pips)
SL_BUFFER_PIPS = {
    # Majors
    **{p: 2.0 for p in ("EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD","USDCHF")},
    # All crosses
    **{p: 3.0 for p in (
        "EURJPY","EURGBP","EURCHF","EURAUD","EURNZD","EURCAD",
        "GBPJPY","GBPCHF","GBPAUD","GBPNZD","GBPCAD",
        "AUDJPY","AUDCHF","AUDCAD","AUDNZD",
        "NZDJPY","NZDCHF","NZDCAD","CADJPY","CHFJPY",
    )},
    "XAUUSD": 5.0,
}
```

---

**END OF SPEC v1.0**
