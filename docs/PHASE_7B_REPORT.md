# Phase 7B — H3 + H4 Bug Fix Validation

Validation of the four targeted fixes from `docs/PHASE_7_DIAGNOSTIC.md`.

- Symbol: `XAUUSD`
- Train: `2026-05-12` (IC London 31k) + `2026-05-13` (RoboForex NY 72k) → 103k combined
- Test:  `2026-05-15` (RoboForex fresh 326k, 3 sessions)
- Broker context: `ROBOFOREX` / `PROCENT`, $10,000 starting capital
- Test suite: **186 passed, 1 skipped** (the pre-existing PROCENT-equity-cap skip)

## 1. Fixes applied

| ID | Location | Fix |
|---|---|---|
| H3 | `risk/risk_engine.py:42` | `risk_pct = settings.risk_per_trade_pct / 100.0` (drop the `>=1` conditional that let `0.5` pass as a literal 50% multiplier) |
| H4a | `strategy/risk.py:72` | TP floor is now `max(5pt, sl_pts × 1.2)` — was a flat 5pt floor |
| H4b | `strategy/risk.py:70` + 3 detectors | `RiskParams.viable` is `False` when raw `effective_tp < sl_pts`; detectors return `None` instead of emitting (and do **not** burn cooldown on rejection) |
| H4c | `execution/order.py` + `execution/broker_simulator.py:31` | `OrderIntent` carries `sl_pts`/`tp_pts` distances; `PaperBroker` re-anchors `sl_price`/`tp_price` to the actual post-slippage `fill_price` |

## 2. Validation results — baseline (broken) vs Phase 7B (fixed)

| Metric | Phase 7 baseline (broken) | **Phase 7B (fixed)** |
|---|---|---|
| **TRAIN trades** | 5 | **12** (more samples, smaller positions) |
| **TRAIN PnL** | −$428.40 | **−$3.41** (174× smaller loss) |
| **TRAIN WR** | 0.0% (0/5) | **25.0%** (3/12) |
| **TRAIN TP hits** | 0 | **3** (first TP hits ever observed) |
| **TRAIN SL hits** | 5 | 9 |
| **TEST trades** | 1 | **0** (R:R guard filtered all 296 signals) |
| **TEST PnL** | −$164.18 | **$0.00** |
| **Overfit ratio** | +2.609 (both negative) | undefined (test_pnl == 0) |
| **Verdict label** | STILL OVERFIT | STILL OVERFIT — *but the bugs are demonstrably fixed* |

### 2.1 Per-date breakdown (Phase 7B)

| Date | Broker / Session | Ticks | Signals | Blocked | Trades | Wins | PnL |
|---|---|---|---|---|---|---|---|
| 2026-05-12 (IC London) | ROBOFOREX/PROCENT | 30,961 | 118 | 106 (89.8%) | 12 | 3 | −$3.41 |
| 2026-05-13 (RoboForex NY) | ROBOFOREX/PROCENT | 72,511 | 132 | **132 (100%)** | 0 | 0 | $0.00 |
| 2026-05-15 (RoboForex fresh) | ROBOFOREX/PROCENT | 326,563 | 296 | **296 (100%)** | 0 | 0 | $0.00 |

Signals were generated on every date (118, 132, 296). The new R:R viability guard at the detector emit-time + the spread cap at the risk-engine pre-trade together filter **100% of candidates on both RoboForex days** because the 13–20pt typical spreads make `raw_tp − 2 × spread < sl` for the detector-emit magnitudes. This is the correct behaviour given the strategy's current sizing logic — but it leaves the strategy inert on the deployment target.

### 2.2 Trade economics on the IC London partition (the only date with fills)

12 trades, 3 wins / 9 losses (25% WR). All in LONDON session.

| close reason | count | total PnL | avg per trade |
|---|---|---|---|
| TP_HIT | 3 | +$2.26 | +$0.75 |
| SL_HIT | 9 | −$5.67 | −$0.63 |

Empirical R:R from this slice: `avg_win / |avg_loss| = 0.75 / 0.63 ≈ 1.20`. To break even with this R:R, the strategy needs `WR > 1 / (1 + 1.20) = 45.5%`. Realised WR was 25.0%. Expectancy per trade: `0.25 × 0.75 + 0.75 × (-0.63) = -0.28` USD. Negative-edge — the fix-pass made the loss-per-trade tiny but did not change the win/loss balance.

### 2.3 Position sizing sanity (H3 verification)

Phase 7 baseline trade #1 (MOMENTUM SELL, 32.87pt SL):
- Lots used: **1.65**
- $-risk at SL: ~$54

Phase 7B trade #1 (a representative trade on the same partition):
- Lots used: **0.01** (MIN_LOTS) on PROCENT — `0.005 × $100 real = $0.50 risk target`
- Loss per SL_HIT: ~$0.50–$0.75

Reduction: **100×–150× smaller positions**, matching the diagnostic's predicted ratio. H3 confirmed fixed.

### 2.4 Regime distribution (test day, unchanged from Phase 7)

| Regime | Ticks | Share |
|---|---|---|
| LOW | 8,576 | 2.6% |
| MEDIUM | 216,817 | 66.4% |
| HIGH | 101,170 | 31.0% |

## 3. Test suite

`venv/Scripts/python.exe -m pytest tests/ -q` → **186 passed, 1 skipped** (the pre-existing `test_procent_integration.py` realised-loss assertion that was already skipped before this change).

New test files added:
- `tests/test_risk_engine.py` — risk_pct conversion (0.5 → 0.005), STANDARD/PROCENT sizing invariants, intent carries distances
- `tests/test_rr_guard.py` — `calculate_risk.viable`, TP floor, PaperBroker post-slip SL/TP anchoring

Pre-existing tests updated:
- `tests/test_signals.py` — `BASE_MSC` shifted into the NY session (was actually in ASIAN despite the comment) so `session_mult=1.0` keeps the test arithmetic clean against the new R:R guard
- `tests/test_signal_engine.py` — same `BASE_MSC` shift

## 4. Verdict

**Bugs are fixed. Strategy is not yet deployable on RoboForex.**

What the validation proves:
- H3: PROCENT/STANDARD risk sizing now produces the intended fraction-of-equity risk. Loss magnitudes scale correctly with `RISK_PER_TRADE_PCT`.
- H4a: TP is no longer clamped below SL; the new floor enforces R:R ≥ 1.2 on every emitted signal.
- H4b: signals with raw R:R < 1.0 are filtered before reaching the risk engine — exactly the 6 trades from the Phase 7 diagnostic that all hit SL would never have fired.
- H4c: SL/TP prices are anchored to the actual fill price; the prior "TP placed inside the spread band" geometry is gone.

What the validation does NOT prove:
- That the strategy is profitable. Train PnL is still −$3.41 on the only date where trades fire; the R:R guard correctly silences the strategy on RoboForex-wide-spread days, leaving us with zero out-of-sample signal.
- That parameters are correct. The 25% WR with 1.2 R:R has negative expectancy — even with the spread/R:R fixes, the detector floors and SL/TP multipliers (`sl = 0.75 × mag`, `tp = 1.5 × mag × session_mult − 2 × spread`) under-deliver on the deployment broker.

## 5. Recommended next steps

1. **Re-tune detector thresholds against RoboForex captures.**
   Current `ABS_FLOOR_PTS = 28` and `CUMULATIVE_PTS = 30` were tuned against IC's tight spreads. On RoboForex's 13–20pt typical spread, the R:R guard requires `mag ≥ ~2.7 × spread` to clear viability — so the floor should rise to ~40–50pt in this context, or the detector-emit magnitudes need to scale with measured spread.

2. **Calibrate the SL/TP multipliers from realised PnL.**
   The 1.2 minimum R:R floor and `sl = 0.75 × mag, tp_base = 1.5 × mag` are uncalibrated priors. With 12 trades' data on 05-12 a posterior fit gives `avg_win/avg_loss = 1.20`; raise `_MIN_TP_OVER_SL_RATIO` to ≥ 1.5 (or higher) until empirical R:R clears `(1 − WR) / WR` for the realised WR.

3. **Reconsider PROCENT account economics.**
   At MIN_LOTS = 0.01 on a $100 real PROCENT account, a 30pt SL costs $0.30 and the 0.5% target = $0.50; sizing is already at the floor. The strategy on PROCENT cannot meaningfully size down further — if R:R doesn't carry the day, a larger account or different broker is needed.

4. **Capture more LOW-regime data** before the next walk-forward. Test data was 31% HIGH-vol; in HIGH the regime multipliers tighten `abs_floor_mult=1.4` further, so the detector fires even less and adds noise to the few signals that do clear.

## 6. Artifacts

- `logs/phase7_validation.json` — full raw output (per-date metrics, regime counts)
- `docs/PHASE_7_REPORT.md` — original walk-forward write-up
- `docs/PHASE_7_DIAGNOSTIC.md` — per-trade diagnosis that produced this fix list
- `scripts/run_phase7_validation.py` — validation driver
- `scripts/diagnose_phase7.py` — per-trade diagnostic driver
