# NY Gold Sweep — Baseline Backtest Report v1.1

**Run configuration**

- Strategy: NY Gold Sweep v1.1 (`docs/NY_GOLD_SWEEP_SPEC.md`)
- Pair: XAUUSD only
- Period: 2024-05-28 → 2026-05-28 (full 2-year parquet stream)
- Session: 12:00–17:00 UTC, NY only
- `TP_MODE = "C"` (hybrid: opposing-level if RR≥1.0, cap 3.0R; else fixed 1.5R)
- All other parameters at Appendix A defaults
- Starting balance: $100,000.00 (The5%ers High Stakes simulation)
- No tuning; no parameter sweeps; one-shot honest read.

---

## Headline result

| Metric | Value | Threshold (§12) | Status |
|---|---:|---:|:---:|
| Trades | 47 | n/a | — |
| Win rate | **23.40 %** | ≥ 48 % SHIP / ≥ 45 % to avoid CUT | **CUT** |
| Profit factor | **0.43** | ≥ 1.40 SHIP / ≥ 1.20 to avoid CUT | **CUT** |
| Expectancy / trade | −0.411 R / −$219.98 | ≥ +0.30 R | **CUT** |
| Max drawdown | **11.17 %** | ≤ 6 % SHIP / ≤ 8 % to avoid CUT | **CUT** |
| Profitable days | 9 / 34 (26.5 %) | n/a | — |
| Avg hold | 11.2 min | n/a | — |
| Net P&L | **−$10,339.13** (final $89,660.87) | ≥ 0 | **CUT** |

### Verdict — **CUT** (§12)

All four CUT conditions tripped simultaneously: PF below 1.20, WR below 45 %, MDD above 8 %, and the negative expectancy is monotone (not a streak — see breakdown below). No tuning. Per spec §12: *"No second chances; move to next pair or next strategy."*

---

## What actually happened — strategy busted at trade 47, sat dormant for 22 months

The 47 trades all happened in a **98-day window** at the start of the data:

| | Date (UTC) | Balance |
|---|---|---:|
| First entry | 2024-05-29 14:23 | $100,000.00 |
| Last entry  | 2024-09-04 15:30 | $89,660.87 |

Trade 47 on 2024-09-04 pushed the running balance below $90,000 — tripping the 10 % Total-DD halt (§7.6). From that moment until the data ends (2026-05-28), **no new entries were taken** (~22 months of dormant capital). The "1,231 cooldown blocks" and "29 cap-blocks" in the compliance counters are almost entirely post-halt artifacts where signals were also gated by total-DD halt; the strategy was effectively dead-on-arrival.

This means the 47-trade sample is the *actual* edge measurement; the rest of the 2-year window adds no information.

---

## Trade-level autopsy

### Exit-reason mix

| Exit reason | Count | WR within group |
|---|---:|---:|
| SL (clean) | 30 | 0.0 % |
| SL via min-hold deferral (SL touched in entry bar R+1) | 6 | 0.0 % |
| TP (clean) | 9 | 100.0 % |
| Time stop (45 min) | 2 | 100.0 % |
| **Total** | **47** | **23.4 %** |

- **36 / 47 = 76.6 % of trades stop out.** The "reversal" candle (§4.2 engulf / pin) is — in this XAUUSD-NY regime — far more often a *pause inside a continuation* than a real turn.
- **6 trades (12.8 %) hit SL inside the entry bar itself** and were deferred to R+2 open at the SL price per the §6 min-hold rule. That's the most diagnostic stat in this report: every eighth signal is so wrong that the very next minute breaks the stop. Min-hold deferral isn't masking these; it's just delaying the realised loss.
- Time-stop winners (2) are flat-trade survivors where TP wasn't reached in 45 min — these would have been losers under tighter time-stops or a 60-min trail; they don't represent a tail.

### R-multiples (static 1.5R model)

| Bucket | Mean R |
|---|---:|
| All trades | −0.41 |
| Winners (n=11) | +1.52 |
| Losers (n=36) | −1.00 |

Winners and losers behave **exactly** as static SL/TP predicts: every winner closes near +1.5R (TP fixed), every loser at −1R (SL fixed). The PF↔WR closed-form in spec §8.5 says WR ≈ 23.4 % implies PF = (0.234 × 1.5) / ((1 − 0.234) × 1.0) ≈ **0.46** — matches the realised **0.43** within commission/spread drift. The system is not behaving anomalously; it's just losing at the rate the math predicts.

### Grade A vs B

| Grade | n | WR | Net P&L |
|---|---:|---:|---:|
| A | 13 | 30.8 % | −$2,067 |
| B | 34 | 20.6 % | −$8,272 |

Grade A (the spec's "real liquidity + engulfing + sweet-spot depth" combo) is **also** losing, just less fast. There is no clean A-only subset that flips the result positive.

### Direction split

| Direction | n | WR |
|---|---:|---:|
| BUY (long) | 20 | 30.0 % |
| SELL (short) | 27 | 18.5 % |

Shorts performed notably worse — consistent with the data: the 2024-05 → 2024-09 window had XAUUSD in a sustained uptrend ($2,330 → $2,500), so every short was leaning against the trend. The "mean-reverting" premise of the strategy (§12) does not hold in this regime; the audit's "efficiency ratio ~0.20" must have been an average across regimes including strong trends.

### Sweep-depth conditioning

| Bucket | Loser avg | Winner avg |
|---|---:|---:|
| Sweep penetration (pips) | 0.26 | 0.31 |

Penetration depth does **not** discriminate winners from losers — the distributions overlap heavily. The §10 grade-A "sweet-spot band" [0.20, 0.60] pips contains both.

---

## Compliance flags (§7) — all green

| Gate | Result |
|---|:---:|
| 7.1 Mandatory protective SL | ✅ on every trade |
| 7.2 Min hold 60 s | ✅ 0 breaches (deferral worked as designed; 6 trades took the deferred-SL path) |
| 7.3 News blackout | ⚠️ 0 signals blocked — the static calendar has 10 events, mostly outside the 98-day active window |
| 7.4 0.5 % risk | ✅ all entries sized accordingly |
| 7.5 Daily DD halt 5 % | ✅ 0 days tripped |
| 7.6 Total DD halt 10 % | ⚠️ **TRIPPED at trade 47** — strategy dormant thereafter |
| 7.7 Max 3 trades/day | ✅ 29 cap-blocks (mostly post-halt) |
| 7.8 Post-loss cooldown | ✅ 1,231 cooldown-blocks (mostly post-halt) |
| 7.9 One open position | ✅ enforced |
| 7.10 Session flatten 17:00 UTC | ✅ enforced (no overnight trades observed) |

Spread coverage: **100.00 %** (1 missing spread tick out of 21,510 NY-session bars; below the 1 % invalidation threshold).

---

## Diagnosis (no fixing — per user instruction)

The strategy lost because the premise didn't hold in the 2024-05 → 2024-09 XAUUSD regime:

1. **The market trended strongly upward**, sweeping prior swing lows on the way up (a common feature of a bull run); reversal candles after these sweeps were *continuation pauses*, not turns. 20/47 longs only managed 30 % WR; 27 shorts hit 18.5 %.
2. **Tight stops (~1.4 pip median risk) amplified the regime mismatch.** Spread + slip alone is 0.42 pips per round-trip — that's 30 % of average risk; a 23 % WR is 17 percentage-points below the cost-adjusted breakeven of ~41 %.
3. **The 10 % Total-DD halt killed the run** before any later regime change (if there was one) could have rescued the strategy. This is correct, per The5%ers rules — but it tells us nothing about the strategy in 2025 or 2026, because no trades were taken there.

The §12 honest expectation ("WR 50–60 %, PF target 1.40, trades/week 5–10") is **not** met. The realised regime (early Q3 2024 XAUUSD bull-run) is exactly the regime spec §12 warned about: *"trends, when they appear, devour mean-reverters."*

---

## Files written

- `out/ny_gold_equity.csv` — equity curve, 1 row per 1M bar evaluated (699k rows ≈ 18 MB)
- `out/ny_gold_trades.csv` — closed-trade ledger, 47 rows

---

## Methodology notes (proof of honesty)

- **Look-ahead invariant (§0.1 / §9):** `assert_no_lookahead` is called on **every decision** in the main loop, not just smoke. Detector receives only `BarFrame` slices with `close_time <= t`.
- **Entries:** next-bar-open only. Long fills at `ask = mid + spread/2 + slippage`; short mirror. SL/TP exits are checked against **bid** (long) / **ask** (short) on every closed bar after entry, using the **current bar's** `spread_mean`, not the entry-bar's.
- **Same-bar SL+TP precedence (§9 ¶4):** SL wins.
- **60 s min-hold (§6):** if SL or TP would trigger in the entry bar (R+1), the exit is deferred to R+2 open at the breached price. 6 trades took this path.
- **Static SL/TP only (§6 v1.1):** no partial fills, no breakeven shift, no trailing — exactly the baseline the spec requires before any v2 partial/trail can be considered.
- **Cost model (§8):** per-bar `spread_mean / 100` → pip; 0.20 pip slippage on entry; commission $7 / lot round-turn.
- **Position sizing (§7.4):** 0.5 % risk on running balance, lot rounded to 0.01.
- **News blackout (§7.3):** `data/news_calendar.StaticNewsCalendar.is_news_blackout` with `window_min=5`. The static calendar holds 10 high-impact events; coverage during the active 98-day window was sparse. This is acknowledged in §7.3 of the spec.

---

## What this run does NOT prove

- It does not prove the strategy is bad in 2025 or 2026 regimes (no trades there).
- It does not prove a single-parameter tweak (penetration band, SL buffer, TP mode) couldn't help — but per spec §12 and user instruction, **no tuning** was attempted.
- It does not invalidate the broader v2 multi-setup family; this is one strategy, one pair, one regime.

What it DOES prove: with the v1.1 defaults, applied honestly to 2024 H2 XAUUSD, NY Gold Sweep loses money fast enough to bust a $100k The5%ers High Stakes account inside 100 days. Per §12: **CUT, move on.**
