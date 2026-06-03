# S/R Flip — Strategy Spec v1.0

Standalone Multi-Strategy v2 candidate #2. Tested in isolation before any
combination with Asian Sweep V5 / propX Multi-Setup / surviving v2 candidates.

Pure price action. No indicators. The rules below are the **only** source
of truth for the `SRFlipDetector` implementation; magic numbers live in
`config/sr_flip_config.py`.

---

## §0. Universe, timeframes, risk envelope

- **Pair universe (30):** identical to Silver Bullet candidate test —
  7 FX majors + 21 G10 crosses + XAUUSD + XAGUSD. Defined in
  `config/sr_flip_config.py::PAIRS`.
- **HTF:** 1H — used for S/R level discovery & break confirmation.
- **LTF:** 15M — used for retest detection + rejection trigger.
- **Risk per trade:** 0.5 % of running balance.
- **Cap:** **2 trades per (pair, day)** — relaxed vs. multi_setup's global
  daily 2, since this is a diagnostic on a single pattern. Per-pair gating
  prevents intra-day stacking on the same level.
- **Stop-distance floor:** 5 pips.

---

## §1. The S/R Flip concept

A horizontal level that has acted as resistance, once cleanly broken **above**,
should henceforth act as **support** on the first retest. Mirror: a former
support, broken **below**, should act as **resistance** on its retest.

The setup pays when the market **re-respects** the level after the structural
break — i.e. the breakout was not a fake-out and the level has been "flipped".

Four sequential conditions must all hold:

1. A horizontal **level** exists (multi-touch HTF cluster).
2. A **clean HTF break** of that level (close strictly beyond by ≥ break-margin).
3. The level has **not been re-broken** in the opposite direction since.
4. The current LTF bar **retests + rejects** the level in the flip direction.

---

## §2. Level discovery (HTF)

Levels are discovered from **confirmed HTF swing highs / lows** clustered by
price. We deliberately do **not** reuse the multi_setup `find_sr_levels`
because it depends on `config.multi_setup_config.pip_size_for`, which does
not include CADCHF / XAGUSD. The detector implements its own clustering with
the augmented `silver_bullet`/`sr_flip` pip map (see §10).

### §2.1 — Swing detection
Reuse `find_swings(htf, l_swing=3)` from `_multi_setup_common.py`
(L_SWING=3 fractals). Returns confirmed swings only — never the in-progress
right edge.

### §2.2 — Multi-touch clustering
For each direction (highs ⇒ resistance candidates; lows ⇒ support
candidates):

- Sort swing prices ascending.
- Greedy-merge any two consecutive prices if `|p_i - p_{i+1}| <= 2 × cluster_tol`.
- `cluster_tol_pips = max(LEVEL_CLUSTER_TOLERANCE_FLOOR,
  LEVEL_CLUSTER_TOLERANCE_ATR_MULT × ATR_HTF_pips)`.
- Keep clusters with **≥ MIN_LEVEL_TOUCHES** swings (default 2).
- Level price = **median** of the cluster's swing prices.
- Level "last_swing_idx" = max swing index in the cluster.

### §2.3 — Level recency
Drop levels whose `last_swing_idx` is older than
`LEVEL_MAX_AGE_HTF_BARS` (default 200 = ~8 days) — stale levels are
ignored.

---

## §3. Break detection (HTF, long setup)

Given a resistance level `L`, find the **earliest** HTF bar with index
`b > last_swing_idx` such that:

```
htf[b].close > L + break_margin_price
```

where

```
break_margin_pips  = max(BREAK_MARGIN_PIPS_FLOOR,
                         BREAK_MARGIN_ATR_MULT × ATR_HTF_pips)
break_margin_price = break_margin_pips × pip_size
```

Defaults: `BREAK_MARGIN_PIPS_FLOOR = 3.0`,
`BREAK_MARGIN_ATR_MULT = 0.20`.

### §3.1 — Break recency
The break index must satisfy
`len(htf) - b <= MAX_BREAK_AGE_HTF_BARS` (default 48 = 2 days).
Older breaks are considered "stale flips" — the level has reverted to noise.

### §3.2 — No failed re-cross since break
No HTF bar after `b` may have **closed back below** the level by more than
`REENTRY_BLOCK_PIPS × pip` (default 5.0 pips). If any such bar exists, the
flip has failed and the level is discarded.

---

## §4. Retest + rejection (LTF, long setup)

Evaluated on the **current closed 15M bar** only:

### §4.1 — Retest condition
```
cur.low  <= L + retest_tol_price          # price wicked to or below the level
cur.close >  L                            # but closed back above (flip held)
```

`retest_tol_pips = max(RETEST_TOL_PIPS_FLOOR,
                       RETEST_TOL_ATR_MULT × ATR_LTF_pips)`.
Defaults: floor 2.0 pips, multiplier 0.15.

### §4.2 — Rejection candle
Reuse `is_rejection_bullish(ltf, last_idx)` from `_multi_setup_common.py`
— passes if the current bar is a bullish **pin** OR a bullish **engulfing**
of the prior bar, with tick-volume ≥ 1.1× mean of last 20 bars.

### §4.3 — Single-touch debounce
Skip if the level has already been retested + rejected by a **prior** LTF
bar within `RETEST_LOOKBACK_LTF_BARS` (default 48 = 12 hours). Only the
**first** retest is tradable; subsequent touches dilute the edge.

---

## §5. Short setup (mirror)

Replace highs with lows, "above" with "below":

- §2: cluster swing **lows** → support levels.
- §3: `htf[b].close < L - break_margin_price`.
- §3.2: no HTF bar closes back above `L + REENTRY_BLOCK_PIPS × pip`.
- §4.1: `cur.high >= L - retest_tol_price AND cur.close < L`.
- §4.2: `is_rejection_bearish(ltf, last_idx)`.

---

## §6. Entry, stop, target

### §6.1 — Long
```
entry      = cur.close + slippage_price          # market fill model
sl         = min(cur.low,  L) - sl_buffer_price
risk_price = entry - sl
tp1        = entry + TP1_R × risk_price          # default TP1_R = 1.5
tp2        = entry + TP2_R × risk_price          # default TP2_R = 2.5
```

### §6.2 — Short (mirror)
```
sl         = max(cur.high, L) + sl_buffer_price
risk_price = sl - entry
tp1        = entry - TP1_R × risk_price
tp2        = entry - TP2_R × risk_price
```

### §6.3 — Risk floor
Reject the setup if `(risk_price / pip) < MIN_RISK_PIPS` (default 5.0).

`PatternSignal.tp` is set to `TP2`; `tp1_<price>` is encoded in
`confluences_met` (same convention as Silver Bullet / multi_setup).

---

## §7. Grading

- **Grade A:** level has **≥ 3 touches** in the cluster **AND** rejection
  was an **engulfing** candle. Confidence 0.85.
- **Grade B:** 2-touch level OR pin (not engulf). Confidence 0.70.
- **Grade C:** not emitted.

---

## §8. PatternSignal contract

```
pattern_name      = "SR_FLIP"
symbol            = <pair>
direction         = BUY | SELL
entry             = current bar close + slippage  (market model)
sl                = beyond retest wick + buffer
tp                = TP2 (1.5R / 2.5R ladder)
confidence        = 0.85 (A) | 0.70 (B)
grade             = A | B
confluences_met:
  - "flip_<dir>"            # "flip_long" / "flip_short"
  - "level_<touches>t"
  - "break_<bar_age>h"
  - "rejection_<pin|engulf>"
  - "tp1_<price>"
bar_time_msc      = trigger bar (retest + rejection)
```

---

## §9. Trade lifecycle (identical to spec §4 of Silver Bullet)

Reuse the same engine from `scripts/silver_bullet_backtest.py` —
`update_open_trade` is copied / adapted verbatim:

- TP1 hit → close 50 %, shift SL to BE + 0.5 × spread.
- After TP1 → trail SL by `0.3R` on each new bar close.
- TP2 hit → close remainder.
- 48-hour time stop → force-close.
- Friday 23:45 UTC flatten → force-close.

---

## §10. Filters (lighter than multi_setup — diagnostic test)

- **Spread guard:** skip if `spread_mean / 10 > MAX_SPREAD_PIPS[symbol]`.
- **News blackout:** off (static calendar covers only May–Jul 2026).
- **HTF trend filter:** **off** in v1.0. S/R Flip is a counter-trend
  pattern by construction; adding a trend filter would invert its premise.
- **Daily-DD halt:** 5 % intraday (live-realistic run).
- **Total-DD halt:** 10 % (live-realistic run).
- **Max concurrent:** 2 open at once across all pairs.
- The diagnostic run disables both DD halts to expose the full 2y
  statistical edge.

---

## §11. Backtest invariants

- Entries simulated as MARKET at trigger-bar close + slippage
  `0.5 × MAX_SPREAD_PIPS`.
- SL fills exact at SL price.
- Same-bar SL+TP → SL wins (conservative).
- Commission: `$7 / lot / round-turn` (FTMO default).
- Pip value computed per-bar from quote currency → USD via cross rates
  read off the same parquet stream.
- No look-ahead: HTF window passed to detector is sliced to bars whose
  `time_msc < cur_ltf.time_msc`. LTF window is the current bar and prior
  closed bars.

---

## §12. Honest expectations

S/R Flip is a textbook retail setup. The retest-after-break premise has
intuitive appeal but is widely traded — meaning the edge may already be
arbitraged out at the 30-pair universe level.

Realistic 2y projections on 30 pairs with 0.5 % risk, 1.5R / 2.5R ladder,
no trend filter:

- **WR:** 40–55 % (retest entries are hit-rate moderate; trail eats some
  upside on TP1-only outcomes)
- **PF:** 1.0–1.4 if the setup has edge; < 1.0 if it doesn't
- **Trades/day:** ~2–6 across 30 pairs (S/R Flips are infrequent but
  spread across the day, not session-gated)

**Verdict thresholds:**
- **KEEP:** PF ≥ 1.3 **and** WR ≥ 45 % **and** MDD ≤ 8 %
- **FIX:** PF 1.1–1.3 (one parameter tighten: break margin, retest tol,
  touches threshold)
- **CUT:** PF < 1.1 or WR < 40 %

We cut losers.

---

## Appendix A — Default constants

```
# Level discovery
LEVEL_CLUSTER_TOLERANCE_FLOOR     = 5.0    # pips
LEVEL_CLUSTER_TOLERANCE_ATR_MULT  = 0.30   # × ATR_HTF_pips
MIN_LEVEL_TOUCHES                 = 2
LEVEL_MAX_AGE_HTF_BARS            = 200    # ~8 days of 1H

# Break detection
BREAK_MARGIN_PIPS_FLOOR           = 3.0
BREAK_MARGIN_ATR_MULT             = 0.20   # × ATR_HTF_pips
MAX_BREAK_AGE_HTF_BARS            = 48     # 2 days
REENTRY_BLOCK_PIPS                = 5.0

# Retest
RETEST_TOL_PIPS_FLOOR             = 2.0
RETEST_TOL_ATR_MULT               = 0.15   # × ATR_LTF_pips
RETEST_LOOKBACK_LTF_BARS          = 48     # 12 h debounce

# Risk
TP1_R                             = 1.5
TP2_R                             = 2.5
MIN_RISK_PIPS                     = 5.0

# Lifecycle (reused)
PARTIAL_FRACTION                  = 0.50
BE_SHIFT_R                        = 1.0
TRAIL_STEP_R                      = 0.30
TIME_STOP_HOURS                   = 48
```
