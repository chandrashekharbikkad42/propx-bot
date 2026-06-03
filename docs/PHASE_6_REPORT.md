# Phase 6 — Walk-Forward Validation Report

- Symbol: `XAUUSD`
- Train partition: `2026-05-12`
- Test partition: `2026-05-13`
- Broker context: IC_MARKETS / STANDARD
- Combos evaluated: 1
- Combos with positive test PnL: 0
- Robust combos (positive train+test, ratio in [0.5,2.0]): 0

## 1. Top 5 Combos by Test PnL

| Rank | Params | Train PnL | Test PnL | Train Trades | Test Trades | Train WR | Test WR | Overfit | Robust |
|---|---|---|---|---|---|---|---|---|---|
| 1 | σ=3.0, floor=28.0pt, cd=3.0s | $535.00 | $-465.00 | 7 | 1 | 42.86% | 0.00% | -1.15 | no |

## 2. Overfitting Distribution

| Bucket | Count |
|---|---|
| test losing (test_pnl <= 0) | 1 |
| severe overfit (ratio > 2.0) | 0 |
| robust (0.5 <= ratio <= 2.0) | 0 |
| test outperforms (ratio < 0.5) | 0 |

## 3. Recommendation

**No deployable parameters.** No combo produced positive test_pnl. Strategy needs review before deployment.

### Finding — train/test divergence

Train partition (2026-05-12, IC Markets London, 31k ticks) replicates the
Phase 5 +$535 result (7 trades, 42.86% WR). Test partition (2026-05-13,
RoboForex NY, 70k ticks) fired only **1 trade** and it lost (-$465).

What this means:
- The signal-fire rate collapses on the RoboForex/NY data. Likely causes
  to investigate in Phase 7:
  - Wider RoboForex spreads gating the dynamic spread filter
    (`STATIC_SPREAD_FALLBACK_PTS["ROBOFOREX"]=30pt` vs `IC_MARKETS`=10pt)
  - Different volatility profile in NY vs London sessions
  - Phase 5 cooldown/threshold tuning was implicit to the London capture
- The infrastructure (walk-forward runner, sweep grid, report generator)
  is in place and validated. The 36-combo grid sweep was infeasible at
  current backtest cost (~16s/run × 144 runs = ~40min wall) and was
  cancelled in favor of a single-point validation. Future runs should
  either parallelise the grid or trim it.

Action: do **not** deploy Phase 5 params to RoboForex demo. Phase 7 will
re-tune against the live RoboForex regime.

## Full Results

| Rank | Params | Train PnL | Test PnL | Train Trades | Test Trades | Train WR | Test WR | Overfit | Robust |
|---|---|---|---|---|---|---|---|---|---|
| 1 | σ=3.0, floor=28.0pt, cd=3.0s | $535.00 | $-465.00 | 7 | 1 | 42.86% | 0.00% | -1.15 | no |
