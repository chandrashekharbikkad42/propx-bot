# Phase 7 — Walk-Forward Validation Report

True out-of-sample test of Phase 7 regime-aware detectors against a fresh
RoboForex capture (2026-05-15) the strategy has never seen.

- Symbol: `XAUUSD`
- Broker context: `ROBOFOREX` / `PROCENT`
- Capital: $10,000
- Train: `2026-05-12` (IC London capture, 31k ticks) + `2026-05-13` (RoboForex NY, 72k ticks) → 103k combined
- Test:  `2026-05-15` (RoboForex fresh, 326k ticks, all 3 sessions)
- Detector params: Phase 7 defaults (σ=3.0, floor=28pt, cd=3.0s) under regime-adaptive multipliers — **no manual tuning, no sweep**

## 1. Phase 6 Baseline (overfit failure)

Captured under `IC_MARKETS` / `STANDARD`:

| Split | Date | Trades | WR | PnL |
|---|---|---|---|---|
| Train | 2026-05-12 (IC London 31k) | 7 | 42.86% | **+$535.00** |
| Test  | 2026-05-13 (RoboForex NY 70k) | 1 | 0.00% | **−$465.00** |

Overfit score: **−1.15** (opposite signs ⇒ not robust). Verdict at Phase 6: *no
deployable parameters*.

## 2. Phase 7 Validation (Out-of-Sample)

Same code, regime-aware thresholds active, broker context flipped to the
deployment target (`ROBOFOREX` / `PROCENT`).

### 2.1 Aggregate

| Split | Dates | Ticks | Signals seen | Blocked | Trades | WR | Gross PnL |
|---|---|---|---|---|---|---|---|
| Train | 2026-05-12 + 2026-05-13 | 103,472 | 570 | 565 (99.1%) | 5 | 0.00% | **−$428.40** |
| Test  | 2026-05-15 | 326,563 | 869 | 868 (99.9%) | 1 | 0.00% | **−$164.18** |

**Overfit ratio**: `train_pnl / test_pnl = -428.40 / -164.18 = +2.609`

Both partitions are negative, so the +2.61 ratio reflects *equal-direction*
losses (train loses 2.6× as much as test) rather than overfit-vs-underfit
divergence. The Phase 6 −1.15 inversion is gone — both halves now move the
same way, just both losing.

### 2.2 By session

Train:

| Session | Trades | Wins | PnL |
|---|---|---|---|
| LONDON (2026-05-12) | 3 | 0 | −$200.69 |
| NY     (2026-05-13) | 2 | 0 | −$227.72 |

Test (2026-05-15):

| Session | Trades | Wins | PnL |
|---|---|---|---|
| LONDON | 0 | 0 | $0.00 |
| LONDON_NY_OVERLAP | 0 | 0 | $0.00 |
| NY | 1 | 0 | −$164.18 |

Only one trade fired across the entire 326k-tick test day — in NY. London
and Overlap saw signals (per `signals_seen=869`) but every London/Overlap
candidate was filtered out by spread / cooldown / warm-up gates.

### 2.3 By signal type

| Type | Train trades | Train PnL | Test trades | Test PnL |
|---|---|---|---|---|
| MOMENTUM | 4 | −$357.23 | 0 | $0.00 |
| SWEEP    | 1 | −$71.17  | 1 | −$164.18 |
| REJECTION | 0 | $0.00   | 0 | $0.00 |

All 6 trades across the entire experiment closed via `SL_HIT`. Zero trades
hit a TP across train+test.

### 2.4 Regime distribution in test data

Counted via a second pass of `MicrostructureState.volatility_regime` over
all 326,563 ticks of 2026-05-15:

| Regime | Ticks | Share |
|---|---|---|
| LOW    | 8,576   | 2.6% |
| MEDIUM | 216,817 | 66.4% |
| HIGH   | 101,170 | 31.0% |

The fresh day skews toward MEDIUM with a substantial HIGH-vol tail —
roughly 1 in 3 ticks classified as HIGH. Phase 7's HIGH-regime multipliers
(σ=3.5, floor×1.4, cooldown×1.5) raise the bar for those ticks, which is
consistent with the 99.9% signal-block rate observed in test.

## 3. Verdict

**STILL OVERFIT — more tuning needed.**

The Phase 6 train/test sign inversion (−1.15) is replaced by a same-sign
2.61 ratio, but the underlying problem has shifted from *overfit* to
*underperforming everywhere*: under the `ROBOFOREX` / `PROCENT` context the
detector defaults yield 0% win rate on both halves. The regime-aware
thresholds did not unlock profitability — they tightened the funnel
enough to suppress most candidates (99%+ block rate) but the ones that
survived all hit SL.

Note that Phase 6's +$535 train result was captured under
`IC_MARKETS`/`STANDARD`. Re-running the same 2026-05-12 data under
`ROBOFOREX`/`PROCENT` (this report) collapsed it to 3 trades, all
losses — broker/account context, not regime tuning, is the dominant
variable in this slice.

## 4. Recommendation

Do **not** deploy current detector defaults to the RoboForex demo. Next
actions, in priority order:

1. **Re-tune for ROBOFOREX**: run a focused param sweep (sigma_mult,
   abs_floor_pts, cooldown_sec) against 2026-05-13 + 2026-05-15
   under `ROBOFOREX`/`PROCENT`. Phase 6's IC-tuned defaults do not transfer.
2. **Validate ProCent math vs Standard**: the 2026-05-12 data shifted from
   +$535/7 trades (IC/STANDARD) to −$200.69/3 trades (ROBOFOREX/PROCENT).
   Confirm risk-engine lot sizing and SL distance are computed correctly
   under PROCENT — a wrong lot multiplier would explain both the trade
   count drop and the across-the-board SL_HIT outcome.
3. **Consider TP/exit logic**: 100% SL_HIT rate (6/6 trades) suggests TP
   placement is too far or trailing exits never engage. Worth reviewing
   `PaperBroker.check_position_exit` and the SL/TP point-distance derivation.
4. After (1)–(3), re-run this same validation script
   (`scripts/run_phase7_validation.py`) — same train/test partitions,
   same broker/account — and target `test_pnl > 0` with `overfit_ratio
   ∈ [0.5, 2.0]` for a DEPLOY-READY verdict.

## Artifacts

- Raw JSON: `logs/phase7_validation.json`
- Driver: `scripts/run_phase7_validation.py`
- Baseline: `docs/PHASE_6_REPORT.md`
