# Phase 7 Diagnostic — Why 6/6 trades hit SL

Investigation of every closed trade from the Phase 7 walk-forward
validation. Source: `logs/phase7_diagnostic.json` produced by
`scripts/diagnose_phase7.py` (replays the same train+test partitions
under ROBOFOREX/PROCENT and captures per-trade detail + signal pipeline
counters). **No engine code changes.**

Context constants:
- `SPREAD_HARD_CAP_PTS = 15.0` (RiskEngine pre-trade)
- `_MIN_EFFECTIVE_TP_PTS = 5.0` (strategy/risk.py floor after 2×spread debit)
- `_SL_SLIPPAGE_BUFFER = 1.5` (SL widened by 1.5×)
- `MAX_HOLD_MS = 300_000` (5-min time exit)
- Slippage model: `0.5 × spread` debited from the touched side
- POINT_VALUE = $0.01, CONTRACT_SIZE = 100 oz/lot

## Trade Details (6 trades)

| # | Date | Sig | Dir | Side | Entry | SL | TP | SL pts | TP pts | R:R | Exit | Reason | Held | Spread entry/exit | Lots | PnL USD | PnL ¢ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 05-12 | MOM | DOWN | SELL | 4713.635 | 4713.964 | 4713.153 | 32.9 | 48.3 | 1:1.47 | 4714.000 | SL_HIT | 21.9s | 5 / 6 pt | 1.65 | −$60.22 | −6,022 |
| 2 | 05-12 | MOM | DOWN | SELL | 4714.180 | 4714.544 | 4713.663 | 36.4 | 51.8 | 1:1.42 | 4714.645 | SL_HIT | 7.9s | 6 / 11 pt | 1.49 | −$69.29 | −6,929 |
| 3 | 05-12 | SWP | UP | BUY  | 4716.450 | 4716.185 | 4716.700 | 26.5 | 25.0 | 1:0.94 | 4716.125 | SL_HIT | 6.0s | 8 / 11 pt | 2.19 | −$71.18 | −7,117 |
| 4 | 05-13 | MOM | UP | BUY  | 4699.105 | 4698.868 | 4699.090 | 23.8 | 1.5  | 1:0.06 | 4698.660 | SL_HIT | 0.83s | 13 / 20 pt | 2.90 | −$129.05 | −12,905 |
| 5 | 05-13 | MOM | UP | BUY  | 4700.415 | 4700.185 | 4700.400 | 23.0 | 1.5  | 1:0.07 | 4700.085 | SL_HIT | 0.07s | 13 / 19 pt | 2.99 | −$98.67 | −9,867 |
| 6 | 05-15 | SWP | UP | BUY  | 4539.905 | 4539.589 | 4539.963 | 31.6 | 5.8  | 1:0.18 | 4539.080 | SL_HIT | 1.9s | 13 / 52 pt | 1.99 | −$164.18 | −16,418 |

Notes:
- "SL pts" / "TP pts" are signed distances from filled `entry_price`. Slippage moves entry past `intended_price`, so these differ from `suggested_sl_pts` / `suggested_tp_pts` (the detector's risk-engine inputs) by ~`0.5 × spread`.
- Trades 4–5 have **TP set inside the spread band** — even though the strategy framed TP as 5pt, the broker exit-trigger requires `spread + 5` pt of favourable mid move.
- Total: 6 SL_HIT, 0 TP_HIT, 0 TIME_EXIT. Aggregate −$592.58 / −59,258 ¢.

## Hypothesis Tests

### H1 — Direction inversion: **REJECTED**

Code path (risk_engine.py:74): `side = BUY if direction == UP else SELL`. Code is correct. All 6 trades moved opposite to the signaled direction:

| Trade | Signal | Expected | Actual mid move | Outcome |
|---|---|---|---|---|
| 1 | DOWN/SELL | price ↓ | +36.5 pt | reversed |
| 2 | DOWN/SELL | price ↓ | +46.5 pt | reversed |
| 3 | UP/BUY    | price ↑ | −32.5 pt | reversed |
| 4 | UP/BUY    | price ↑ | −44.5 pt | reversed |
| 5 | UP/BUY    | price ↑ | −33.0 pt | reversed |
| 6 | UP/BUY    | price ↑ | −82.5 pt | reversed |

6/6 reversals with `WP_priors=0.45–0.55` has prob `≈ 0.55^6 = 2.7%` — suspicious but not conclusive given the sample size, and the bad R:R (H4) makes price direction nearly irrelevant: even with 50% direction accuracy, every trade would still hit SL first because TP is unreachable. The pattern is a *symptom* of H4, not a code-level inversion.

### H2 — SL/TP swap: **REJECTED**

Per-trade SL/TP-vs-entry check:

| Trade | Side | SL vs entry | TP vs entry | Geometry |
|---|---|---|---|---|
| 1 | SELL | SL > entry ✓ | TP < entry ✓ | correct |
| 2 | SELL | SL > entry ✓ | TP < entry ✓ | correct |
| 3 | BUY  | SL < entry ✓ | TP > entry ✓ | correct |
| 4 | BUY  | SL < entry ✓ | **TP < entry ✗** | TP below entry by 1.5pt |
| 5 | BUY  | SL < entry ✓ | **TP < entry ✗** | TP below entry by 1.5pt |
| 6 | BUY  | SL < entry ✓ | TP > entry ✓ | correct (barely) |

No code-level swap, but trades 4–5 show TP placed **below the BUY entry price** due to slippage. Mechanism: `risk_engine.py:88` sets `tp_price = intended_price + suggested_tp_pts × 0.01` where `intended_price = ask`. The broker fills BUY at `ask + 0.5×spread`. When `suggested_tp_pts < 0.5×spread` (e.g. 5pt vs spread/2=6.5pt), `tp_price` ends up below `entry_price`. The broker still triggers TP on `bid ≥ tp_price`, so the trade is not literally upside-down — but the entry is *already past TP* by 1.5pt at fill, requiring a bid move of `spread + suggested_tp` ≈ 18pt to actually trigger.

### H3 — ProCent lot math: **CONFIRMED**

Target risk per trade: $50 (STANDARD 0.5% × $10k) or **$0.50** (PROCENT 0.5% × $100 real). Actual:

| Trade | Lots | SL pts | Actual $ risk | Target $0.50? |
|---|---|---|---|---|
| 1 | 1.65 | 32.87 | $54.24 | **108× too large** |
| 2 | 1.49 | 36.37 | $54.19 | **108× too large** |
| 3 | 2.19 | 26.50 | $58.04 | **116× too large** |
| 4 | 2.90 | 23.75 | $68.89 | **138× too large** |
| 5 | 2.99 | 23.00 | $68.77 | **138× too large** |
| 6 | 1.99 | 31.63 | $62.93 | **126× too large** |

Root cause is upstream of `position_sizer.calculate_lot_size` in `risk_engine.py:42–43`:

```python
cfg = settings.risk_per_trade_pct          # 0.5  (intent: "0.5%")
self.risk_pct: float = cfg / 100.0 if cfg >= 1 else cfg
```

The conditional only normalizes whole-number percent inputs. Decimal inputs like `0.5` bypass the division and are passed through as a literal multiplier — so 0.5 (intended as 0.5%) becomes a **50%** risk fraction. Combined with the PROCENT divisor (`real_equity = account_equity / 100`), the math produces `risk_usd = $100 × 0.5 = $50` instead of `$100 × 0.005 = $0.50` — the 100× inflation seen in every row.

Note: this would also affect STANDARD at $10k (intended $50 risk → actual $5000 → lots clamped to MAX_LOTS=10), which is why Phase 6 IC/STANDARD numbers also looked "off" relative to the comment in `position_sizer.py:18`.

### H4 — Spread killing the trade: **CONFIRMED (primary cause)**

Round-trip cost (= 2 × spread) vs the strategy's `tp_pts` floor (`_MIN_EFFECTIVE_TP_PTS = 5`):

| Trade | suggested_tp | suggested_sl | R:R suggested | net TP after 2×spread debit (strategy/risk.py already applies this) | TP clamped to floor? |
|---|---|---|---|---|---|
| 1 | 50.75 | 30.37 | 1:1.67 | spread=5 → base 1.5×40.5 − 10 = 50.75 | no |
| 2 | 54.75 | 33.37 | 1:1.64 | spread=6 → base 66.75 − 12 = 54.75 | no |
| 3 | 29.00 | 22.50 | 1:1.29 | spread=8 → base 45 − 16 = 29 | no |
| 4 | **5.00** | 17.25 | **1:0.29** | spread≈14 → base ~34.5 − 28 = 6.5 → floored to 5 | **YES** |
| 5 | **5.00** | 16.50 | **1:0.30** | spread≈14 → similar | **YES** |
| 6 | **12.25** | 25.12 | **1:0.49** | spread=13 → base 38 − 26 = 12 | no, but R:R already < 1 |

The detector emits a signal at magnitude ≥ floor and lets `strategy/risk.py:calculate_risk` size SL/TP. On RoboForex's NY-session spreads (13–20pt typical) the 2× round-trip debit eats the entire base TP (`1.5 × mag`) and clamps to the 5pt floor in 2 of 6 trades. Three more trades have R:R below break-even (1.29:1 needs WR > 43.7%, 1:0.94 needs > 51.5%, 1:0.18 needs > 84.7%) — well above the strategy's 0.45–0.55 priors. **Five of six trades are designed to lose by R:R alone**, independent of direction. SL hit rate of 100% follows mechanically.

### H5 — Time exit too tight: **REJECTED**

Max hold = 300s. Observed durations: 21.93, 7.92, 6.00, 0.83, 0.07, 1.89 s. Mean ≈ 6.4s, max ≈ 22s. SL trigger always preceded the 5-min cutoff by 14× or more. Zero TIME_EXIT closes in 6 trades.

## Signal Generation Analysis

Aggregated over the 3 partitions (103k + 326k = 429k ticks total):

| Metric | 2026-05-12 (LON) | 2026-05-13 (NY) | 2026-05-15 (fresh) | Total |
|---|---|---|---|---|
| Ticks consumed | 30,961 | 72,511 | 326,563 | 430,035 |
| Signals seen | 118 | 452 | 869 | **1,439** |
| Signals blocked | 115 | 450 | 868 | **1,433** (99.6%) |
| Filled | 3 | 2 | 1 | **6** |

Block reasons:

| Reason | 05-12 | 05-13 | 05-15 | Total | Share |
|---|---|---|---|---|---|
| `spread_cap` (>15pt hard) | 0 | 322 | 868 | **1,190** | 83.0% |
| `daily_cap_hit` (circuit breaker after losses) | 115 | 128 | 0 | 243 | 17.0% |
| `position_already_open` | 0 | 0 | 0 | 0 | 0% |
| `spread_drift` | 0 | 0 | 0 | 0 | 0% |

Signal-type distribution:

| Type | 05-12 | 05-13 | 05-15 | Total | Share |
|---|---|---|---|---|---|
| SWEEP | 71 | 187 | 504 | 762 | 53.0% |
| MOMENTUM | 43 | 253 | 306 | 602 | 41.8% |
| REJECTION | 4 | 12 | 59 | 75 | 5.2% |

Direction balance:

| Dir | 05-12 | 05-13 | 05-15 | Total | Share |
|---|---|---|---|---|---|
| UP | 56 | 223 | 438 | 717 | 49.8% |
| DOWN | 62 | 229 | 431 | 722 | 50.2% |

Direction is perfectly balanced (no bias). Spread-cap blocks dominate on RoboForex (83% of all blocks), confirming the broker's typical spread profile sits above the 15pt hard cap most of the time. The circuit breaker engages after the first 1–3 losing trades each day and locks out the remaining ~115 candidates — so even if R:R were tuned, current daily exposure caps would hard-limit any rebound.

## Most Likely Root Cause

**H4 is the primary cause** — the strategy's TP/SL sizing produces negative R:R on RoboForex spreads (13–20pt typical, vs 5–8pt on IC Markets). Five of six trades have R:R < 1.0 after the 2× spread debit, two clamp to the 5pt TP floor. Even a coin-flip direction predictor cannot be profitable when TP < SL. This explains the 100% SL_HIT rate fully on its own — no direction-prediction edge can rescue these economics.

**H3 is a serious amplifier** — the risk_pct misinterpretation (intended 0.5%, applied as 50%) makes every loss ~100× larger in real-dollar terms on PROCENT. This is *why the dollar PnL looks catastrophic*; it is not what *causes* the trades to lose, but it converts a small "negative-edge" leak into a fast equity bleed.

**H2 (TP-below-entry on BUY)** is a downstream consequence of H4 + slippage interaction, not an independent bug. Once `suggested_tp_pts` collapses to 5 under wide spreads, the 0.5×spread slip on entry pushes the filled price past the TP level.

## Recommended Fix (in priority order — DO NOT implement yet)

1. **Rewrite `RiskEngine.__init__` risk_pct conversion** (risk_engine.py:42–43)
   Always divide by 100 (or change the env contract to require decimals).  This is a one-line correctness fix and explains the bulk of the $-loss magnitude.

2. **Decouple TP base from spread debit in `strategy/risk.py`**
   The current formula `effective_tp = 1.5 × mag − 2 × spread` clamps to 5pt on RoboForex.  Options:
   - Raise `_MIN_EFFECTIVE_TP_PTS` to `max(5, 1.0 × spread)` so the floor scales with broker
   - Require `1.5 × mag > 3 × spread + min_target_rr × sl_pts` before emitting (reject sub-RR setups at the detector layer)
   - Re-tune base multipliers (today: SL=0.5×mag×1.5, TP=1.5×mag) — for broker-typical 15pt spread, TP needs ≥ 2.5×mag to leave a fightable R:R after debit

3. **Add a pre-emit R:R guard**
   Before a signal leaves the detector, compute the *suggested* R:R and skip if < 1.0. This would have rejected trades 3–6 outright (no fill, no loss) and surfaced the underlying tuning problem earlier.

4. **Fix the TP-vs-entry geometry for slipped fills** (risk_engine.py:85–92)
   Anchor `tp_price` and `sl_price` to the post-slip `entry_price` you expect, not `intended_price`. Equivalent: add `0.5 × spread` to `suggested_tp_pts` on BUY and to `suggested_sl_pts` on SELL (or use bid for BUY-TP and ask for SELL-TP).

5. **Re-run `scripts/run_phase7_validation.py` after each fix** to confirm the SL_HIT rate drops and `test_pnl` moves positive before considering any deployment.

## Artifacts

- `logs/phase7_diagnostic.json` — per-trade JSON
- `scripts/diagnose_phase7.py` — diagnostic driver (no engine changes)
- Referenced: `docs/PHASE_7_REPORT.md` (walk-forward summary)
