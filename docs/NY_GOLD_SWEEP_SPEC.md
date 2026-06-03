# NY Gold Sweep — Strategy Spec v1.1

Standalone scalping bot for **The5%ers High Stakes** evaluation.
Single-pair (XAUUSD), NY-session only, sweep-and-reverse scalp built on
a multi-timeframe stack (15M bias → 5M zone → 1M trigger).

This document is the **only source of truth** for the detector. Magic
numbers live in `config/ny_gold_sweep_config.py` (to be created — not in
this PR). If the implementation diverges from any rule below, the spec
wins and the code is wrong.

> **Do not touch**: Asian Sweep V5 modules, the live engine, `.env`.
> This strategy ships in **parallel** alongside V5, not on top of it.

---

## §0. Universe, timeframes, risk envelope

- **Pair (v1):** `XAUUSD` only. Selected by data: highest scalp-suitability
  score in the 2-year audit (`scan_market_character.py`) — best
  ATR%/spread ratio of any tradable instrument we have data for, and the
  NY session dominates its directional move budget.
- **Higher-timeframe (bias):** `15M` — used to locate recent confirmed
  swing highs/lows and decide which side of the book holds tradable
  liquidity.
- **Mid-timeframe (zone):** `5M` — used as a proximity filter: don't even
  look at 1M trigger candles until 5M price has approached a fresh 15M
  level.
- **Trigger timeframe:** `1M` — sweep penetration + rejection + reversal
  candle all detected on 1M.
- **Decision cadence:** evaluated on every **closed** 1M bar during the
  NY session window (§1).
- **Risk per trade:** `0.5 %` of running balance — The5%ers hard rule.
- **Stop-distance floor:** `0.30 pips` (30 broker points). Stops tighter
  than this get blown by spread + slippage; skip the setup.

### §0.1 — Time-honesty invariant (CRITICAL)

This was the failure mode of prior phases. Locked in here at the top so
every downstream rule inherits it.

At any 1M decision time **t** (= close timestamp of the most recent
closed 1M bar), the detector MUST use ONLY bars whose
`close_time <= t`:

- **1M slice:** all 1M bars with `close_time <= t`. The bar with
  `close_time == t` is the trigger candidate; anything beyond is the
  future and must not be read.
- **5M slice:** the most recent 5M bar with `close_time <= t`. The 5M
  bar currently forming (i.e. `open_time <= t < close_time`) is **not
  visible**. If t = 12:03:00 UTC, the latest visible 5M bar is the
  11:55–12:00 bar; the 12:00–12:05 bar is forming and excluded.
- **15M slice:** same rule — only bars with `close_time <= t`.

**No** detector branch may read `bars[i+1]`, `current.next`, or the
high/low of any bar whose `close_time > t`. The next-bar-open entry
model in §5 commits at `bars_1m[trigger_idx + 1].open` — which is
**only** legal because that bar's open is observable at time `t+60s`,
which becomes the new decision time. The order is placed at `t`, filled
at `t + 60s`.

### §0.2 — XAUUSD pip convention (CONFIRMED)

This convention applies to **every** numeric value in this spec and the
config file:

| Term         | Value                                        |
|--------------|----------------------------------------------|
| Broker point | `0.01` of price (the MT5 `SYMBOL_POINT`)     |
| 1 pip        | `100 broker points` = `1.00` of price ($1.00) |
| 1 USD move   | `1 pip` = `100 points`                       |

So `spread_pts = 22` in the data stream means **0.22 pips of spread**
(≈ $0.22 per 0.01 lot). `ATR_5M = 1.40 pips` means a 5M ATR of $1.40 in
price terms. All defaults below are stated in **pips** unless explicitly
suffixed `_pts`.

This is consistent with `config/asian_sweep_config.py` where
`XAUUSD.point = 0.01` and `spread_pts = 45` (= 0.45 pips), and with the
ICT / FTMO / The5%ers convention used everywhere else in the repo.

---

## §1. NY session window

- **Default UTC window:** `12:00:00 — 17:00:00` UTC, both endpoints
  inclusive of bar **open** time. A 1M bar with `open_time = 11:59` is
  out; `open_time = 12:00` is in; `open_time = 16:59` is in;
  `open_time = 17:00` is out.
- **Configurable** via `NY_SESSION_START_UTC` / `NY_SESSION_END_UTC` in
  `ny_gold_sweep_config.py`.
- **Rationale:** the 2y audit shows XAUUSD volume and directional
  efficiency concentrate from NY open (13:30 UTC) through mid-afternoon.
  We pad slightly before NY open to catch pre-open sweeps that often
  lead the session.
- **DST:** the spec is in **UTC**, deliberately. Gold's volume curve is
  driven by COMEX open (13:30 UTC year-round), not by NY local clock —
  using UTC keeps the window stable across DST transitions. If a future
  v1.x wants ET-relative windows, gate it behind a flag.
- **No-trade buffers** inside the window:
  - **First 1M bar of session (12:00 bar):** skip — its 1M context is
    pre-session.
  - **Last 5M (16:55–17:00):** no new entries (existing trades manage
    out per §6).

---

## §2. 15M bias and key liquidity levels

The 15M timeframe answers two questions: *what swing levels exist nearby,
and which ones still hold tradable liquidity?*

### §2.1 — Swing detection
- Use fractal swings on the **closed** 15M series, with
  `L_SWING_15M = 2` (a swing high is a bar whose high is strictly
  greater than the 2 bars on each side; mirror for lows).
- Swings only **confirm** 2 bars after they print — so the most recent
  detectable swing has `swing_idx <= len(bars_15m) - 3`. This is
  inherent to fractal logic and double-protects against look-ahead.

### §2.2 — Fresh-level set
At decision time t, the **tradable level set** is:

- All confirmed swing highs `H_i` and swing lows `L_i` with
  `swing_time >= t - LEVEL_LOOKBACK_HOURS` (default `8 hours` =
  32 × 15M bars).
- Each level carries: `price`, `swing_time`, `direction`
  (`high`/`low`).

### §2.3 — Freshness rule (unswept = tradable)
A level is **fresh** iff, since its swing confirmation, NO 1M bar has
penetrated it by more than `LEVEL_INVALIDATE_PIPS` (default `0.50`
pips). Penetration = `low < price - 0.50` for a swing low, mirror for
high.

- A level swept and traded once is **burned** for the rest of the
  session — do not retrade it even if price returns. Tracked via a
  per-session `burned_levels: set[(price, swing_time)]`.
- A level whose freshness check fails for a non-tradable reason
  (e.g. swept while we were on cooldown) is also burned.

### §2.4 — Side selection
- **Long setups** target a fresh swing **low** below current price
  (sweep + reverse up = liquidity grab below).
- **Short setups** target a fresh swing **high** above current price.
- It is legal for both sides to be armed simultaneously; the 1M trigger
  (§4) decides which fires.

---

## §3. 5M zone (proximity gate)

Cheap pre-filter — keeps the 1M detector from grinding when price is
nowhere near a level.

### §3.1 — Zone definition
For each fresh level `L` (§2.2), define the **approach zone**:

- For a swing **low** (long target):
  `zone = [L - ZONE_PROXIMITY_PIPS,  L + ZONE_PROXIMITY_PIPS]`
- For a swing **high** (short target):
  `zone = [L - ZONE_PROXIMITY_PIPS,  L + ZONE_PROXIMITY_PIPS]`
- Default `ZONE_PROXIMITY_PIPS = 1.5` (= $1.50 = 150 broker points).

### §3.2 — Proximity test
At decision time t, the 5M is **inside the zone for level L** iff the
**last closed 5M bar** satisfies either:
- `5m.low <= L + ZONE_PROXIMITY_PIPS` (for a long-target swing low), OR
- `5m.high >= L - ZONE_PROXIMITY_PIPS` (for a short-target swing high).

If no fresh level has a zone-active 5M, **no 1M trigger is evaluated
this t**. Cheap rejection — exits the detector early.

### §3.3 — Stale-zone debounce
If price has been zone-active on `L` for more than
`ZONE_MAX_DWELL_MIN` (default `25` minutes = 5 × 5M bars) without a
sweep, **burn the level**. Long dwell = market is grinding through, not
sweeping; the level is no longer a liquidity prize.

---

## §4. 1M sweep + reversal trigger

This is the entry decision. Long path described; short is mirror.

### §4.1 — Sweep bar (`S`)
At decision time t, the **most recent closed 1M bar** is the sweep
candidate `S`. For a long setup targeting a fresh swing low `L`:

1. **Penetration:**
   `L - SWEEP_MAX_PENETRATION_PIPS <= S.low < L - SWEEP_MIN_PENETRATION_PIPS`
   - Default `SWEEP_MIN_PENETRATION_PIPS = 0.10` (10 points — must
     actually break the level, not graze it).
   - Default `SWEEP_MAX_PENETRATION_PIPS = max(0.80,
     SWEEP_MAX_ATR_MULT × ATR_5M_pips)`, with `SWEEP_MAX_ATR_MULT =
     0.40`. Anything deeper is a real breakout, not a sweep.
2. **Rejection (close back inside):**
   `S.close > L - SWEEP_REJECT_TOLERANCE_PIPS`, with default
   `SWEEP_REJECT_TOLERANCE_PIPS = 0.10`. The sweep wicks below, but the
   1M closes back at-or-above the level. A 1M that closes well below
   the level is a clean break, not a sweep — reject.
3. **Optional same-bar reversal short-circuit:** if `S` itself is a
   bullish engulfing or bullish pin (per §4.2 definitions) AND
   `S.close > L`, then `S` doubles as the reversal trigger `R` (skip
   §4.2 wait). This is common on fast NY sweeps.

### §4.2 — Reversal trigger candle (`R`)
If §4.1 fires but `S` is not itself the trigger, scan **forward in 1M
real-time** up to `REVERSAL_MAX_WAIT_BARS` (default `3` = 3 minutes) for
a confirming bar `R`:

- `R` is one of `bars_1m[S.idx + 1 .. S.idx + REVERSAL_MAX_WAIT_BARS]`,
  evaluated in order, FIRST match wins.
- `R` must satisfy:
  - `R.close > L` (level has been reclaimed), AND
  - `R.low >= L - SWEEP_MAX_PENETRATION_PIPS` (no fresh deeper wick —
    if a wait-bar wicks deeper than `S`, the sweep is failing; abort
    the setup and **burn the level**), AND
  - **either** of the following bullish reversal candle shapes:
    - **Bullish engulfing:** `R.close > R.open` AND
      `R.body >= ENGULF_MIN_BODY_PIPS` AND `R.close >= prev.open` AND
      `R.open <= prev.close` AND `prev.close < prev.open` (prev was
      bearish).
    - **Bullish pin:** `R.close > R.open` AND
      `lower_wick >= PIN_WICK_BODY_RATIO × body` AND
      `lower_wick >= PIN_MIN_WICK_PIPS` AND `R.close` lies in the
      upper third of `[R.low, R.high]`.
- Defaults: `ENGULF_MIN_BODY_PIPS = 0.20`, `PIN_WICK_BODY_RATIO = 2.0`,
  `PIN_MIN_WICK_PIPS = 0.30`.

If no bar in the wait window qualifies, the sweep is dead. **Burn the
level** (do not re-arm on the same swing low this session — once
liquidity is taken, it stops being liquidity).

### §4.3 — Concurrency lock
Only one trigger per (level, session) — once `R` is identified, do NOT
keep scanning further bars for the same level even if a "better" `R'`
later appears. First valid wins. The level is consumed.

### §4.4 — Short setup (mirror)
- Target: fresh swing **high** `H`.
- §4.1 penetration: `H + SWEEP_MIN_PENETRATION_PIPS < S.high <= H + SWEEP_MAX_PENETRATION_PIPS`.
- §4.1 rejection: `S.close < H + SWEEP_REJECT_TOLERANCE_PIPS`.
- §4.2 candle shapes: bearish engulfing / bearish pin (upper-wick
  variant).
- §4.2 freshness: `R.high <= H + SWEEP_MAX_PENETRATION_PIPS`.

---

## §5. Entry, stop, target

### §5.1 — Entry: next-bar-open ONLY
- Once `R` is identified at time `t_R` (close of bar `R`), the order is
  placed and **filled at the open of bar `R+1`**, i.e. at
  `t_R + 60s`.
- This is the only legal entry. No mid-bar fills, no limit orders, no
  "fill at R.close". The 1M next-bar-open rule is what makes the
  backtest honest: at `t_R` we know `R`'s close but not `R+1`'s
  internals, so the only price we can transact at without look-ahead is
  `bars_1m[R.idx + 1].open`.

### §5.2 — Cost-adjusted fill price (long)
```
mid_open       = bars_1m[R.idx + 1].open
spread_pips    = bars_1m[R.idx + 1].spread_pts / 100       # per-bar, from data
half_spread    = spread_pips / 2
slippage_pips  = SLIPPAGE_PIPS                             # default 0.20
ask_open       = mid_open + half_spread × pip_size         # textbook ask side
entry_price    = ask_open + slippage_pips × pip_size       # slippage on top
```
- The long pays the **ask** (`mid + spread/2`) at entry; slippage is
  added on top of the ask.
- SL and TP are evaluated against the **bid** (`mid - spread/2`) on
  every closed bar after entry — see §5.4 and §8.2.
- Spread is therefore charged at BOTH ends of the trade — half via the
  ask at entry, half via the bid at exit — totalling `1 × spread` over
  the round-trip. Slippage is charged **once**, on entry only.

### §5.3 — Cost-adjusted fill price (short, mirror)
```
bid_open    = mid_open - half_spread × pip_size            # textbook bid side
entry_price = bid_open - slippage_pips × pip_size          # slippage subtracted
```
- The short sells the **bid** (`mid - spread/2`) at entry; slippage is
  subtracted further (entry is worse than the quoted bid).
- SL and TP are evaluated against the **ask** (`mid + spread/2`) on
  every closed bar after entry — mirror of §5.2 / §5.4.
- Round-trip spread = `1 × spread` (half on each side); slippage once,
  on entry only. Identical cost structure to the long.

### §5.4 — Stop loss (mandatory — The5%ers rule)
- **Long:** `SL = S.low - SL_BUFFER_PIPS × pip_size`, with default
  `SL_BUFFER_PIPS = 0.20`. SL sits beneath the sweep wick `S.low`.
- **Short:** `SL = S.high + SL_BUFFER_PIPS × pip_size`.
- If `(entry_price - SL) / pip_size < MIN_RISK_PIPS` (default `0.30`,
  long; mirror for short), **skip the trade** — stop is too tight to
  survive normal noise.
- SL is **always** sent with the order. No naked entries.

**Exit-side execution (applies to SL AND TP, both LONG and SHORT):**
- **Long exit** is evaluated against the **bid** (`mid - spread/2`)
  using the current bar's `spread_pts` (not the entry-bar's):
  - SL fires on bar `B` (with `B.idx >= R.idx + 1`) when
    `B.low - half_spread_B × pip_size <= SL_price`.
  - TP fires on bar `B` when
    `B.high - half_spread_B × pip_size >= TP_price`.
- **Short exit** is evaluated against the **ask** (`mid + spread/2`):
  - SL fires when `B.high + half_spread_B × pip_size >= SL_price`.
  - TP fires when `B.low + half_spread_B × pip_size <= TP_price`.
- Spread is therefore applied on both ends of every trade via bid/ask,
  not just at entry. Total round-trip spread cost = `1 × spread`.

### §5.5 — Take profit (three modes; default = hybrid)

Mode is set by `TP_MODE` in config. All three return a single TP price
that goes on the order; partials/trails are handled by lifecycle (§6).

**Mode A — Fixed RR**
```
risk_price = entry_price - SL                       # long
TP         = entry_price + TP_RR × risk_price       # default TP_RR = 1.5
```

**Mode B — Opposing level**
- Long: `TP = nearest fresh swing high above entry, found via §2 logic
  but in the opposite direction`, less `OPPOSING_BUFFER_PIPS` (default
  `0.20`). Short: mirror.
- If no opposing fresh level exists within `OPPOSING_MAX_DISTANCE_PIPS`
  (default `8.0`), fall back to Mode A.

**Mode C — Hybrid (default)**
- Compute `TP_A` (fixed RR) and `TP_B` (opposing level).
- If `TP_B` exists AND `(TP_B - entry) / risk_price >= MIN_RR_FOR_OPPOSING`
  (default `1.0`), use `TP = min(TP_B, entry + TP_RR_MAX × risk_price)`
  with `TP_RR_MAX = 3.0` (cap so an absurdly far opposing level doesn't
  push TP past statistical reach).
- Else use `TP_A`.
- Short: mirror.

---

## §6. Trade lifecycle (intra-trade management) — static SL/TP only

Single-trade-at-a-time, single-pair, **static SL and TP** — no partials,
no breakeven shifts, no trailing. A position runs from entry to one of
four exits and nothing in between modifies SL or TP:

- **TP hit:** the single TP price from §5.5 (whatever mode is active) →
  close in full.
- **SL hit:** the single SL price from §5.4 → close in full.
- **Time stop:** if open longer than `TIME_STOP_MIN` (default
  `45 min`), force-close at market. NY scalps don't ripen — if it
  hasn't worked in 45 minutes, it's not working.
- **Session flatten:** at `17:00:00 UTC` (session end §1), force-close
  any still-open position. No overnight gold scalps. Ever.

**Min-hold deferral (60 s — The5%ers rule):** if SL or TP would trigger
inside the entry bar itself (i.e. `R+1`), the exit is **deferred to the
open of `R+2`** at the price of the level that was breached. This
satisfies The5%ers' 60-second min-hold and is intentionally
conservative (the deferral worsens the fill on losers and gives back a
tick on tight winners). See §9 for the backtest-side identity of this
rule.

**Note — partials, BE shift and trailing are deferred to v2.** Modelling
TP1 partials + BE shift + post-TP1 trailing on 1M bars introduces
exit-simulation optimism: a prior phase showed the engine assumes
intra-bar trail levels are touched in the most favourable order,
inflating PF by ~15–25 % vs. live. v1 ships with a **static-only**
baseline so the backtest measures the strategy's raw edge, not the
engine's optimism. Partial/trail logic will only be added in v2 **after
v1's static baseline prints PF ≥ 1.40** in the 2y backtest; if static
can't clear that bar, no amount of trail engineering will save it. If
it can, v1 becomes the clean reference against which v2's trail/partial
impact is measured.

---

## §7. The5%ers High Stakes compliance gates

These are **hard gates** — every entry must pass all of them. Encoded
once here so the implementation has a single checklist.

| #    | Gate                                | Default                              |
|------|-------------------------------------|--------------------------------------|
| 7.1  | Mandatory protective SL             | Always on every order (§5.4)         |
| 7.2  | Minimum hold time                   | 60 s (§6 min-hold)                   |
| 7.3  | News blackout                       | ±5 min around High-impact events     |
| 7.4  | Risk per trade                      | 0.50 % of running balance            |
| 7.5  | Daily loss halt                     | 5.0 % of starting-of-day balance     |
| 7.6  | Total loss halt                     | 10.0 % of initial account balance    |
| 7.7  | Max trades per day                  | 3 (configurable, range 3–5)          |
| 7.8  | Post-loss cooldown                  | 30 min after any losing trade        |
| 7.9  | One open position at a time         | Concurrency = 1                      |
| 7.10 | Weekend / Friday flatten            | 17:00 UTC Friday final flatten       |

### §7.3 — News blackout details
- Use `data/news_calendar.py::is_news_blackout(symbol, time_msc,
  before_ms, after_ms)` with `before_ms = after_ms = 5 × 60_000`.
- Symbols to consider: `XAUUSD` + `USD` general (FOMC, NFP, CPI,
  ISM, Powell speeches). The calendar's symbol mapping handles this.
- A 1M trigger inside the blackout window is **dropped silently**
  (logged but no signal emitted). A position already open before
  blackout enters is allowed to manage out per §6 — we just block new
  entries.
- The static calendar covers a limited window (May–Jul 2026 in the
  current file). v1.0 ships with the static calendar; report
  acknowledges the gap.

### §7.4 — Risk sizing
```
risk_pips        = (entry_price - SL) / pip_size              # long
risk_usd         = balance × RISK_PCT / 100                   # 0.50 %
usd_per_pip_lot  = 100   # 1 lot of XAUUSD = 100 oz, 1 pip = $1/oz × 100 = $100/lot/pip
lot_size         = risk_usd / (risk_pips × usd_per_pip_lot)
lot_size         = round_to_step(lot_size, LOT_STEP)          # broker step
lot_size         = clamp(lot_size, LOT_MIN, LOT_MAX)
```
Verify the contract size and pip value against `config/asian_sweep_config.py::PAIR_CONFIG["XAUUSD"]` at runtime — single source of truth for broker constants.

### §7.7 — Daily trade counter
- Reset at 00:00 UTC (NOT NY midnight — UTC, to align with §1 and the
  data stream).
- Once `daily_trade_count >= MAX_TRADES_PER_DAY`, no new entries until
  next UTC day. Existing trade is allowed to manage out.

### §7.8 — Post-loss cooldown
- A "loss" is any closed trade with `realized_pnl <= 0`.
- Cooldown begins at trade-close timestamp; new entries blocked for
  `COOLDOWN_AFTER_LOSS_MIN = 30 min`.
- Two consecutive losses in one day → extend cooldown to
  `COOLDOWN_AFTER_TWO_LOSSES_MIN = 60 min` (default; tunable).

---

## §8. Cost model (the bit prior phases got wrong)

### §8.1 — Per-bar spread from data
- Bars in the parquet stream carry a `spread_pts` column (broker points
  averaged over the bar's tick stream — provided by
  `scripts/capture_historical_bars.py`).
- For each backtest evaluation, the spread on bar `B` is
  `spread_pips_B = bars[B].spread_pts / 100` (§0.2).
- If `spread_pts` is missing (NaN) for a given bar, fall back to
  `DEFAULT_SPREAD_PIPS = 0.45` (the asian_sweep_config typical) — but
  log a warning; missing spread on more than 1 % of session bars
  invalidates the run.

### §8.2 — Ask / bid / mid (round-trip via the bid/ask sides)
- The parquet stream carries OHLC as **mid prices** (broker MT5
  convention).
- **Long entry** fills at the **ask** = `mid + spread/2` (plus
  slippage). **Long exit** (SL or TP) is evaluated against the **bid**
  = `mid - spread/2` (see §5.4 exit-side execution).
- **Short entry** fills at the **bid** = `mid - spread/2` (minus
  slippage). **Short exit** is evaluated against the **ask** = `mid +
  spread/2`.
- Spread is therefore applied at BOTH ends of every trade — never
  entry-only. Total round-trip spread cost in price terms = `1 × spread`
  (half attributed to each side). Slippage is charged **once**, on
  entry only.
- The `spread_pts` used at entry is the entry-bar's value; the
  `spread_pts` used for each exit check is the **current** bar's value
  (spreads drift through the session, especially around news — using
  the current-bar spread keeps the exit-side honest).

### §8.3 — Slippage
- `SLIPPAGE_PIPS = 0.20` (default, NY-session realistic for XAUUSD on
  a top-tier broker). Adds in the same direction as spread (worsens
  entry by total `spread + slippage`).

### §8.4 — Commission
- `COMMISSION_USD_PER_LOT_ROUND_TURN = 7.0` (FTMO / The5%ers default,
  charged once per closed trade, scaled by realized lot size).

### §8.5 — Round-turn cost estimate (sanity check, static 1.5R)
With v1's static-only TP (§6) every winner closes at exactly **+1.5 R**
and every loser at exactly **−1 R** — no partials, no trail, no
ambiguity. The PF↔WR mapping is closed-form, and breakeven math is
straightforward.

Worked example: 0.5 pip stop, 0.22 pip per-bar spread (round-tripped
via bid/ask sides per §8.2), 0.20 pip entry slippage, $7/lot commission,
0.10 lot:
```
spread (round-trip)   = 1 × 0.22  = 0.22 pips    # half via ask at entry, half via bid at exit
slippage (entry only) = 0.20 pips
total cost in pips    = 0.42
$-cost on 0.10 lot    = 0.42 × 0.10 × $100/pip   = $4.20
commission            = 0.10 × $7 round-turn     = $0.70
total round-turn $    = $4.90
risk @ 0.5 % of $100 000                          = $500
cost drag             = $4.90 / $500             ≈ 1.0 % of risk
gross breakeven WR @ 1.5R  = 1 / (1 + 1.5)        = 40.0 %
breakeven WR after costs                          ≈ 41–42 %
```

Closed-form PF↔WR for static 1.5R (ignoring within-trade variance):
```
PF = (WR × 1.5) / ((1 − WR) × 1.0)
PF = 1.40  ⇔  WR ≈ 48.3 %     ← ship threshold (§12)
PF = 1.20  ⇔  WR ≈ 44.4 %     ← cut threshold  (§12)
PF = 1.00  ⇔  WR =  40.0 %    ← gross breakeven
```

**This is the strategy's edge envelope.** Anything that prints
WR < 42 % in the 2y backtest is dead on arrival even before costs are
fully recognised. The static 1.5R model makes the verdict thresholds
in §12 directly readable off WR alone.

---

## §9. Backtest invariants (anti-look-ahead, restate)

Restating §0.1 in operational terms for the backtest harness:

1. The backtest loop iterates `t` over **closed** 1M bars in chronological
   order. At each iteration:
   - `bars_1m_visible  = bars_1m[bars_1m.close_time <= t]`
   - `bars_5m_visible  = bars_5m[bars_5m.close_time <= t]`
   - `bars_15m_visible = bars_15m[bars_15m.close_time <= t]`
   - The detector receives ONLY the above slices. It must not have
     access to the full DataFrame.
2. The detector returns at most one `PatternSignal` per `t` — or `None`.
3. If a signal is returned at `t`, the trade is **entered at**
   `bars_1m[idx_of(t) + 1].open` (the next bar's open). The harness
   handles this; the detector does not need to do anything beyond
   emitting the signal at `t`.
4. **Same-bar SL+TP precedence:** if entry-bar's [low, high] spans both
   SL and TP, **SL wins** (conservative). This combined with the
   §6 min-hold rule means the actual same-bar same-direction touch is
   deferred to the next bar's open, which is even more conservative.
5. **Spread used for the trade** = `bars_1m[idx_of(t)+1].spread_pts /
   100` (the entry bar's own spread, not `t`'s spread).
6. The detector is **deterministic** given a fixed input slice — no
   randomness, no clock reads.
7. **Forward-fill rule:** if the detector reads `ATR_5M` and the most
   recent visible 5M bar's ATR is stale (>3 closed 5M bars old vs. t),
   recompute on the visible slice rather than carrying forward.

---

## §10. Grading

- **Grade A (confidence 0.85):**
  - Swept level is a 15M swing that has **≥ 2 prior touches** in the
    lookback window (i.e. real liquidity, not just one fractal), AND
  - Reversal candle is an **engulfing** (not a pin), AND
  - Penetration depth ∈ `[0.20, 0.60]` pips (the sweet-spot bucket from
    audit).
- **Grade B (confidence 0.70):**
  - All other valid setups.
- **Grade C:** not emitted (we only fire A/B).

Grading is for logging + post-hoc analysis; the live engine treats A
and B identically in v1.0. v1.1 may gate position size or daily-cap
priority on grade.

---

## §11. `PatternSignal` contract

```
pattern_name     = "NY_GOLD_SWEEP"
symbol           = "XAUUSD"
direction        = BUY | SELL
entry            = bars_1m[R.idx + 1].open + cost-adjustment (§5.2/5.3)
sl               = beyond sweep wick + buffer (§5.4)
tp               = TP price per active TP_MODE (§5.5)
confidence       = 0.85 (A) | 0.70 (B)
grade            = "A" | "B"
confluences_met:
  - "ny_sweep_<long|short>"
  - "sweep_depth_<X.XX>p"
  - "rev_<engulf|pin>"
  - "level_<N>touches"
  - "tp_mode_<A|B|C>"
  - "tp1_<price>"
bar_time_msc     = R's close timestamp (trigger time, NOT entry time)
fill_time_msc    = R+1's open timestamp (entry time)
```

---

## §12. Honest expectations

NY Gold Sweep is a **mean-reversion scalp** in a market the audit
characterizes as choppy (efficiency ratio ~0.20). That is exactly the
regime mean-reversion eats; it is also a regime where trends, when they
appear, devour mean-reverters. Realistic envelope:

- **WR:** 50–60 % (sweeps are high-WR by setup nature, especially with
  next-bar-open and tight stops). Breakeven WR at 1.5R is ~42 % (§8.5),
  so the strategy has ~8–18 pp of margin to play with.
- **Trades/session:** 0–3 (most sessions print 1 setup; many print
  zero; news days print zero).
- **Trades/week:** ~5–10.
- **PF:** target ≥ 1.40 to be worth shipping live. <1.20 = cut.
- **Max DD (2y):** target ≤ 6 % running. The5%ers hard fail at 10 %.
- **Avg trade R:** target ≥ 0.30R after costs.

**Verdict thresholds (post-backtest, 2y XAUUSD-only):**
- **SHIP:** PF ≥ 1.40 AND WR ≥ 48 % AND MDD ≤ 6 % AND no rule violated
  (no SL absences, no 60s breach, no daily-cap breach).
- **TUNE:** PF 1.20–1.40 OR MDD 6–8 %. One of: tighten penetration
  bounds, restrict TP to Mode A only, increase min-touches to 3 for
  grade A.
- **CUT:** PF < 1.20 OR WR < 45 % OR MDD > 8 % OR any compliance gate
  breach. No second chances; move to next pair or next strategy.

---

## Appendix A — Parameter summary table

All defaults; live in `config/ny_gold_sweep_config.py`.

| §    | Param                                  | Default     | Units             |
|------|----------------------------------------|-------------|-------------------|
| 0    | `RISK_PCT`                             | `0.50`      | % of balance      |
| 0    | `MIN_RISK_PIPS`                        | `0.30`      | pips              |
| 1    | `NY_SESSION_START_UTC`                 | `12:00:00`  | UTC               |
| 1    | `NY_SESSION_END_UTC`                   | `17:00:00`  | UTC               |
| 2.1  | `L_SWING_15M`                          | `2`         | bars each side    |
| 2.2  | `LEVEL_LOOKBACK_HOURS`                 | `8`         | hours             |
| 2.3  | `LEVEL_INVALIDATE_PIPS`                | `0.50`      | pips              |
| 3.1  | `ZONE_PROXIMITY_PIPS`                  | `1.50`      | pips              |
| 3.3  | `ZONE_MAX_DWELL_MIN`                   | `25`        | minutes           |
| 4.1  | `SWEEP_MIN_PENETRATION_PIPS`           | `0.10`      | pips              |
| 4.1  | `SWEEP_MAX_PENETRATION_PIPS` (floor)   | `0.80`      | pips              |
| 4.1  | `SWEEP_MAX_ATR_MULT`                   | `0.40`      | × ATR_5M_pips     |
| 4.1  | `SWEEP_REJECT_TOLERANCE_PIPS`          | `0.10`      | pips              |
| 4.2  | `REVERSAL_MAX_WAIT_BARS`               | `3`         | 1M bars           |
| 4.2  | `ENGULF_MIN_BODY_PIPS`                 | `0.20`      | pips              |
| 4.2  | `PIN_WICK_BODY_RATIO`                  | `2.0`       | wick/body         |
| 4.2  | `PIN_MIN_WICK_PIPS`                    | `0.30`      | pips              |
| 5.4  | `SL_BUFFER_PIPS`                       | `0.20`      | pips              |
| 5.5  | `TP_MODE`                              | `"C"`       | A / B / C         |
| 5.5  | `TP_RR`                                | `1.50`      | R-multiple        |
| 5.5  | `TP_RR_MAX`                            | `3.00`      | R-multiple cap    |
| 5.5  | `MIN_RR_FOR_OPPOSING`                  | `1.00`      | R-multiple        |
| 5.5  | `OPPOSING_BUFFER_PIPS`                 | `0.20`      | pips              |
| 5.5  | `OPPOSING_MAX_DISTANCE_PIPS`           | `8.00`      | pips              |
| 6    | `TIME_STOP_MIN`                        | `45`        | minutes           |
| 7.2  | `MIN_HOLD_SEC`                         | `60`        | seconds           |
| 7.3  | `NEWS_BLACKOUT_BEFORE_MIN`             | `5`         | minutes           |
| 7.3  | `NEWS_BLACKOUT_AFTER_MIN`              | `5`         | minutes           |
| 7.5  | `DAILY_DD_HALT_PCT`                    | `5.0`       | %                 |
| 7.6  | `TOTAL_DD_HALT_PCT`                    | `10.0`      | %                 |
| 7.7  | `MAX_TRADES_PER_DAY`                   | `3`         | trades            |
| 7.8  | `COOLDOWN_AFTER_LOSS_MIN`              | `30`        | minutes           |
| 7.8  | `COOLDOWN_AFTER_TWO_LOSSES_MIN`        | `60`        | minutes           |
| 8.3  | `SLIPPAGE_PIPS`                        | `0.20`      | pips              |
| 8.1  | `DEFAULT_SPREAD_PIPS` (fallback)       | `0.45`      | pips              |
| 8.4  | `COMMISSION_USD_PER_LOT_ROUND_TURN`    | `7.00`      | USD               |
| 10   | Grade A min touches on level           | `2`         | touches           |
| 10   | Grade A penetration band               | `[0.20, 0.60]` | pips           |

---

## Appendix B — What this spec deliberately does NOT cover

- **No** multi-pair logic. v1 is XAUUSD only. Adding a second pair is
  a v2 decision after the 2y backtest verdict.
- **No** indicator filters (RSI, MA, etc.). The audit picked
  mean-reversion *because* the regime is choppy; layering trend
  filters defeats the premise.
- **No** session-rotation logic. NY only. London / Asia sessions are
  not in scope for this strategy — they are owned by Asian Sweep V5
  and any future London-specific bot.
- **No** ML signal fusion. v1 is rule-based. If/when v1 prints PF ≥
  1.40 over 2y, then we have a base to consider a meta-filter on top.

---

*End of spec — review and approve before any code is written. The
detector / backtest / live engine all reference the section numbers
here verbatim in their docstrings.*
