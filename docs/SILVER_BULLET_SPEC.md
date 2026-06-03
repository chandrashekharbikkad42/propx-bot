# ICT Silver Bullet — Strategy Spec v1.0

Standalone Multi-Strategy v2 candidate #1. Tested in isolation before any
combination with Asian Sweep V5 or the propX Multi-Setup bundle.

Pure price action + time. No indicators. The rules below are the **only**
source of truth for the `SilverBulletDetector` implementation; magic numbers
live in `config/silver_bullet_config.py`.

---

## §0. Universe, timeframes, risk envelope

- **Pair universe (30):** 7 FX majors, 21 G10 crosses, XAUUSD, XAGUSD.
  Defined in `config/silver_bullet_config.py::PAIRS`.
- **Higher timeframe (HTF):** 1H — used for swing structure & sweep targets.
- **Lower timeframe (LTF):** 15M — used for window detection, sweep
  confirmation, FVG formation, and entry trigger.
- **Risk per trade:** 0.5 % of running balance.
- **Standalone test daily cap:** **unlimited** (the global "max 2/day" cap
  applies only when combined; we want to see the strategy's raw signal rate
  here). Per-window cap = 1 trade per window per pair to prevent
  intra-window stacking.
- **Stop-distance floor:** 5 pips (same as multi-setup spec §2.4).

---

## §1. The three Silver Bullet windows (ET, with DST handling)

ICT's canonical Silver Bullet windows are defined in **New York local time**
(`America/New_York`). The detector converts each bar's UTC timestamp into
ET via `zoneinfo.ZoneInfo("America/New_York")` so DST is handled
automatically — no fixed UTC offsets.

| ID  | Name        | ET window     | UTC (EST winter) | UTC (EDT summer) |
|-----|-------------|---------------|------------------|------------------|
| LO  | London Open | 03:00–04:00   | 08:00–09:00      | 07:00–08:00      |
| AM  | NY AM       | 10:00–11:00   | 15:00–16:00      | 14:00–15:00      |
| PM  | NY PM       | 14:00–15:00   | 19:00–20:00      | 18:00–19:00      |

A 15M bar is **inside** a window if its **open** timestamp (`time_msc`)
maps to an ET hour that satisfies `window_start_hour <= ET.hour <
window_end_hour`. So each window contains exactly **4 × 15M bars**.

Configurable in `silver_bullet_config.py::WINDOWS` if the user wants to
extend a window or add a new one. Default windows match ICT's canonical
definitions.

---

## §2. Setup definition (long; short is mirror)

A long Silver Bullet setup requires, in chronological order, **all four**:

### §2.1 — In-window trigger
The current closed 15M bar's open timestamp falls inside one of the three
ET windows (§1).

### §2.2 — Liquidity sweep of a recent low
There exists a 15M bar `S` within the last `SWEEP_LOOKBACK_BARS` (default
20 = 5 hours) bars **before the current bar**, such that:

1. `S.low < swept_low_price - SWEEP_PENETRATION_PRICE`, where
   `swept_low_price` is one of:
   - the **last confirmed HTF (1H) swing low** (preferred, found via the
     shared `find_swings` helper from `_multi_setup_common.py`), OR
   - if no HTF swing low is in range, the **lowest low of the prior
     `SWEEP_REF_BARS` (default 12 = 3 hours) 15M bars before `S`**.
2. `S` is followed by **at least one bar that closes back above
   `swept_low_price`** — i.e. the sweep is **rejected**, not breakout.
   This bar is `S+k` where `k >= 1` and `S+k.close > swept_low_price`.
3. Penetration is bounded:
   - **floor:** `SWEEP_PENETRATION_PIPS_FLOOR` (default 1.0 pip), OR
     adaptive `SWEEP_PENETRATION_ATR_MULT × ATR15M` (default 0.10×),
     whichever is larger.
   - **ceiling:** `SWEEP_MAX_PENETRATION_ATR_MULT × ATR15M` (default 4.0×)
     to filter true breakouts.

### §2.3 — Bullish FVG formation **after** the sweep
A **fair value gap** (3-candle imbalance) appears in the sweep-direction
recovery, **strictly after** the sweep bar `S`.

Three consecutive 15M bars `(A, B, C)` form a **bullish FVG** when:

- `A.index >= S.index + 1` (FVG bars are all after the sweep)
- `C.index <= current.index` (FVG is fully formed by the trigger bar)
- **Gap condition:** `A.high < C.low` (a clean upward gap between bar A's
  high and bar C's low; bar B straddles the gap)
- **Min FVG size:** `(C.low - A.high) >= FVG_MIN_PIPS × pip_size`
  (default 2.0 pips; 3.0 for XAUUSD/XAGUSD)
- **Bar B is bullish or doji** (close >= open) — ensures the gap was
  printed by impulsive up-movement, not a 1-bar pause inside a downtrend.

The FVG `(low, high)` is `(A.high, C.low)`.

### §2.4 — Retracement entry
The setup is **armed** as soon as conditions §2.1–§2.3 are simultaneously
true on the current 15M bar (the trigger bar = bar `C`, or the bar that
closes immediately after the FVG completes).

**Entry mode:** the detector emits a signal with **entry = top of FVG
(`C.low`)** as a limit order conceptually — but for the standalone
backtest we model it as MARKET at the current bar's close with slippage,
**only if `current_bar.low <= FVG.high`** (i.e. price has already touched
or pierced into the FVG on the trigger bar). This matches how the
multi_setup backtest fills BOS/OB market entries and keeps results
comparable.

If `current_bar.low > FVG.high` (price is still above the FVG and hasn't
retraced yet), the detector emits **no signal this bar**; we wait. The
FVG remains valid for `FVG_VALIDITY_BARS` (default 6 = 90 minutes) bars
after formation, after which it expires unfilled.

### §2.5 — Stop-loss and take-profit

- **SL:** below the sweep wick — `SL = min(S.low, FVG.low) - SL_BUFFER_PIPS × pip`.
  Default `SL_BUFFER_PIPS = 2.0` for FX majors, 3.0 for crosses, 5.0 for metals.
- **TP:** fixed-R ladder, same convention as multi_setup spec §7:
  - `TP1 = entry + TP1_R × risk_price` (default `TP1_R = 1.5`)
  - `TP2 = entry + TP2_R × risk_price` (default `TP2_R = 2.5`)
- **Min risk distance:** `MIN_RISK_PIPS = 5.0`. If `risk_pips < 5`, skip.

`PatternSignal.tp` is set to `TP2` and `tp1_<price>` is encoded in
`confluences_met` (same convention as the existing 4 detectors).

---

## §3. Short setup (mirror)

Replace lows with highs, "below" with "above":

- §2.2: sweep of an HTF swing **high** (or the highest high of prior
  `SWEEP_REF_BARS` 15M bars). `S.high > swept_high_price + SWEEP_PENETRATION_PRICE`,
  rejected by `S+k.close < swept_high_price`.
- §2.3: **bearish FVG** — `A.low > C.high`, bar B bearish or doji.
- §2.4: entry triggers when `current_bar.high >= FVG.low`.
- §2.5: `SL = max(S.high, FVG.high) + SL_BUFFER_PIPS × pip`.

---

## §4. Trade lifecycle

Reuses the **multi_setup backtest engine's** lifecycle verbatim
(`update_open_trade` in `scripts/multi_setup_backtest.py`):

- TP1 hit → close 50 %, shift SL to BE + 0.5 × spread (cover round-turn).
- After TP1 → trail SL by `0.3R` on each new bar close.
- TP2 hit → close remainder.
- 48-hour time stop → force-close at market.
- Friday 23:45 UTC flatten → force-close.

---

## §5. Filters (lighter than multi_setup — diagnostic test)

For the standalone candidate test:

- **Spread guard:** identical to multi_setup §1.6 — skip if
  `spread_mean / 10 > MAX_SPREAD_PIPS[symbol]`.
- **News blackout:** off (the static news_calendar covers only May–Jul 2026
  and would block ~0 trades over 2y; flagged in report).
- **HTF trend filter:** **off** in v1.0 (Silver Bullet is canonically a
  session-timed setup, not trend-following). We will add an optional HTF
  filter in v1.1 if base results justify keeping the strategy.
- **Daily-DD halt:** 5 % intraday (same as multi_setup).
- **Total-DD halt:** 10 % (same).
- **Max concurrent:** 2 trades open at once across all pairs (same as
  multi_setup; prevents over-leveraging).

---

## §6. Grading

- **Grade A:** sweep was of a confirmed HTF (1H) swing point (not just an
  LTF local extreme) AND FVG size ≥ `FVG_GRADE_A_PIPS` (default 4.0 pips).
  Confidence `0.85`.
- **Grade B:** all other valid setups (LTF-only sweep reference OR
  smaller FVG). Confidence `0.70`.
- **Grade C:** not emitted (we only emit valid A/B candidates).

---

## §7. PatternSignal contract

```
pattern_name      = "SILVER_BULLET"
symbol            = <pair>
direction         = BUY | SELL
entry             = top of FVG (long) / bottom of FVG (short)
sl                = beyond sweep wick + buffer
tp                = TP2 price (1.5R/2.5R ladder; TP1 in confluences_met)
confidence        = 0.85 (A) or 0.70 (B)
grade             = A | B
confluences_met:
  - "sb_<window>"          # "sb_LO" / "sb_AM" / "sb_PM"
  - "long_sweep" | "short_sweep"
  - "fvg_<size_pips>p"
  - "sweep_ref_<htf|ltf>"
  - "tp1_<price>"
bar_time_msc      = trigger bar (the bar that confirms the entry condition)
```

---

## §8. Backtest invariants

- All entries simulated as MARKET at trigger-bar close + slippage
  `0.5 × MAX_SPREAD_PIPS` (same as multi_setup §8.3).
- SL fills exact at SL price.
- Same-bar SL+TP → SL wins (conservative).
- Commission: `$7 / lot / round-turn` (FTMO default).
- Pip value computed per-bar from quote currency → USD via cross rates
  read off the same parquet stream.

---

## §9. Honest expectations

ICT Silver Bullet is a popular but **not statistically validated** retail
setup. Realistic 2y backtest expectations on a 30-pair universe with
0.5 % risk, 1.5R / 2.5R targets, no HTF filter:

- **WR:** 40–55 % (FVG retracement entries tend to be hit-rate moderate)
- **PF:** 1.0–1.5 if the setup has edge, < 1.0 if it doesn't
- **Trades/day:** ~3–8 across 30 pairs (3 windows × 30 pairs × low fill rate)

**Verdict thresholds for this candidate test:**
- **KEEP:** PF ≥ 1.3 **and** WR ≥ 45 % **and** max DD ≤ 8 %
- **FIX:** PF 1.1–1.3 (tighten one of: FVG size, sweep penetration, HTF
  trend gate)
- **CUT:** PF < 1.1 or WR < 40 % — no fix is going to save it; move on
  to candidate #2.

This is a one-shot honest read. We cut losers.

---

## Appendix A — Default constants

```
# Windows (ET local; DST handled via zoneinfo)
WINDOWS = (
    ("LO", 3,  4),   # London Open
    ("AM", 10, 11),  # NY AM
    ("PM", 14, 15),  # NY PM
)

# Sweep
SWEEP_LOOKBACK_BARS         = 20    # 15M bars (5h) to scan for the sweep
SWEEP_REF_BARS              = 12    # 15M bars (3h) for LTF-only reference
SWEEP_PENETRATION_PIPS_FLOOR = 1.0
SWEEP_PENETRATION_ATR_MULT   = 0.10
SWEEP_MAX_PENETRATION_ATR_MULT = 4.0

# FVG
FVG_MIN_PIPS                = 2.0   # 3.0 for XAUUSD, XAGUSD (handled per-symbol)
FVG_GRADE_A_PIPS            = 4.0
FVG_VALIDITY_BARS           = 6     # 90 minutes

# Risk
TP1_R                       = 1.5
TP2_R                       = 2.5
MIN_RISK_PIPS               = 5.0
SL_BUFFER_PIPS_FX           = 2.0
SL_BUFFER_PIPS_CROSS        = 3.0
SL_BUFFER_PIPS_METAL        = 5.0

# Lifecycle (reused from multi_setup_config)
PARTIAL_FRACTION            = 0.50
BE_SHIFT_R                  = 1.0
TRAIL_STEP_R                = 0.30
TIME_STOP_HOURS             = 48
```
