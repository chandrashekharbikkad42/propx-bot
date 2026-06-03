# Phase 8 — Pivot Plan: XAU Tick HFT → Griff 1H Prop-Firm Bot

> **Status:** Audit complete. No strategy code touched yet.
> **Decision:** Pivot off tick microstructure (broke on RoboForex 13–20pt spreads).
> Reuse infrastructure layer, swap strategy + risk for Griff 1H + prop-firm engine.
>
> *Bhai, ye audit puri hai — abhi sirf plan + skeleton. Strategy rules next session
> me Griff videos dekhne ke baad fill karna hai.*

---

## 1. Audit findings — per file

Categories: ✅ KEEP / 🔧 MODIFY / ❌ REPLACE / ➕ NEW.

### 1.1 Infrastructure (mostly KEEP)

| File | Status | Notes |
|------|--------|-------|
| `bot.py` | 🔧 MODIFY | Orchestrator is solid (signal-driven shutdown, mode switch live/replay/backtest/live-demo). Add new mode `griff-scan` for multi-pair 1H. Strip `MicrostructureState` from strategy loop; replace with `BarAggregator` + `GriffEngine`. |
| `supervisor.py` | ✅ KEEP | Auto-restart + balance-guard works regardless of strategy. No changes. |
| `config/settings.py` | 🔧 MODIFY | Add `prop_firm_type` (FTMO/THE5ERS), `forex_pairs` list, `time_window_ist_start/end`, `news_blackout_min`, `max_trades_per_day`. Remove tick-specific session caps (or repurpose). |
| `utils/logger.py` | ✅ KEEP | Loguru wiring — strategy-agnostic. |
| `utils/session.py` | 🔧 MODIFY | Add IST helpers (12:30–22:30 IST window). Keep UTC enum for legacy tests. Add `is_within_trading_window(time_msc)`. |
| `data/mt5_connector.py` | 🔧 MODIFY | Tick-only API today. Add `copy_rates_range(symbol, timeframe, ...)` for 1H bars + `positions_get`/`account_info` already partially used. Multi-symbol support — make `symbol` a per-call arg, not init. |
| `data/tick_collector.py` | ✅ KEEP | Useful for live execution monitoring (intra-bar SL/TP) even on 1H strategy. |
| `data/tick_writer.py` | ✅ KEEP | Parquet capture stays — bar-level data also stored as parquet (new writer reuses schema pattern). |
| `replay/replay_engine.py` | 🔧 MODIFY | Currently tick-only. Either (a) replay ticks → 1H bars on-the-fly OR (b) add a `BarReplayEngine` sibling. Recommend (a) — single source of truth, deterministic. |
| `replay/integrity_checker.py` | ✅ KEEP | Partition integrity logic is dataset-agnostic. |
| `alerts/telegram_notifier.py` | ✅ KEEP | Transport + formatters reusable. New formatters for prop-firm events (`notify_prop_breach_warning`, `notify_daily_cap_approaching`). |
| `monitoring/dashboard.py` | 🔧 MODIFY | Frame layout assumes tick state. Rebuild for: equity vs prop caps, daily trades 1/2, pair scan results, active position. |

### 1.2 Risk layer (significant MODIFY)

| File | Status | Notes |
|------|--------|-------|
| `risk/risk_engine.py` | 🔧 MODIFY | Keep structure (pre-trade gate + bookkeeping + circuit breakers). Replace XAU-specific constants (`POINT_VALUE=0.01`, `SPREAD_HARD_CAP_PTS=15`, `MAX_HOLD_MS=5min`) with per-symbol lookup + 1H hold time (1–4 bars). Replace `can_open_position` single-position rule with: `1 position + 2 trades/day + house-money rule`. |
| `risk/circuit_breakers.py` | 🔧 MODIFY | Daily-loss cap structure is reusable. ADD: prop-firm hard caps (max daily loss %, max total loss %), 80% early-stop, news blackout window, time-window gate, max-trades-per-day counter. REPLACE session filter with IST window check. |
| `risk/position_sizer.py` | 🔧 MODIFY | Keep clean signature. ADD: grade-based risk (`A=1.0%`, `B=0.5%`, `C=skip`), house-money buffer mode for trade 2, per-pair pip value lookup (XAU pip ≠ EURUSD pip). |
| `strategy/risk.py` | ❌ REPLACE | Built around tick microstructure (Nσ moves, REJECTION/SWEEP/MOMENTUM win-rate priors, 1.5× SL buffer). 1H Griff uses pattern-defined SL/TP from swing structure, not tick magnitudes. |

### 1.3 Strategy layer (mostly REPLACE)

| File | Status | Notes |
|------|--------|-------|
| `strategy/microstructure.py` | ❌ REPLACE | O(1) tick-level state — useless at 1H. REPLACE with `BarAggregator` (rolling 1H OHLC, swing-point detection, structure tracking). |
| `strategy/signal_engine.py` | 🔧 MODIFY | Async orchestrator pattern is fine. Rewire to feed bars (not ticks) to new detectors. Keep tick → bar aggregation upstream. |
| `strategy/signal_confirmation.py` | 🔧 MODIFY | 3-tick confirmation makes no sense at 1H. REPLACE with "1-bar close confirmation" OR drop entirely (Griff is bar-close trigger). |
| `strategy/signals/base.py` | 🔧 MODIFY | `Signal` dataclass stays. ADD: `grade: GradeLabel` (A/B/C), `confluence_count: int`, `pair: str`, `pattern_type` enum {FLAG, CONTINUATION, COMBO, REVERSAL}. REPLACE `SignalType` enum values. |
| `strategy/signals/liquidity_sweep.py` | ❌ REPLACE | Tick-Nσ trigger. Not relevant. Delete or archive. |
| `strategy/signals/tick_momentum.py` | ❌ REPLACE | 20-tick cumulative drift. Not relevant. Delete or archive. |
| `strategy/signals/rejection.py` | ❌ REPLACE | Tick state machine. Bar-level rejection is a different pattern. |
| `strategy/signals/regime.py` | ❌ REPLACE | Tick volatility regimes. Not applicable. (Could keep concept but reframed for 1H ATR regimes later.) |

### 1.4 Execution layer (mostly KEEP, some MODIFY)

| File | Status | Notes |
|------|--------|-------|
| `execution/order.py` | 🔧 MODIFY | Add `pair: str`, `grade: str`. Keep Side enum. |
| `execution/position.py` | 🔧 MODIFY | Add `pair`, `grade`, `pattern_type`. Already supports SL/TP/exit reasons. |
| `execution/broker_simulator.py` | 🔧 MODIFY | PaperBroker logic is per-tick exit polling. For 1H bar backtest, need per-bar exit check (high/low against SL/TP). Add per-pair `POINT_VALUE` + `contract_size` lookup. |
| `execution/live_broker.py` | 🔧 MODIFY | MT5 order placement is reusable. Per-symbol contract specs (already pulled from `symbol_info` partially). Visible SL/TP MANDATORY (already does this — good). |
| `execution/broker_factory.py` | ✅ KEEP | Pattern stays. |

### 1.5 Backtesting (KEEP infrastructure, REPLACE inner)

| File | Status | Notes |
|------|--------|-------|
| `backtesting/backtest_runner.py` | 🔧 MODIFY | Loop structure reusable. Replace tick-feed + per-tick detector calls with bar-feed + per-bar Griff pattern check + multi-pair scan. |
| `backtesting/trade_journal.py` | ✅ KEEP | Summary stats (win rate, expectancy, max DD, by-bucket) work for any strategy. Add `by_grade` and `by_pattern_type` buckets. |
| `backtesting/walk_forward.py` | ✅ KEEP | Train/test split + overfit ratio is strategy-agnostic. |
| `backtesting/param_sweep.py` | 🔧 MODIFY | Hard-codes `LiquiditySweepDetector` / `RejectionDetector` / `TickMomentumDetector` class attrs. Rewire to sweep Griff pattern thresholds (when defined). |
| `backtesting/report.py` | ✅ KEEP | Markdown report generator is generic over `WalkForwardResult`. |

### 1.6 Tests (mostly KEEP, some REPLACE)

| Group | Status | Notes |
|-------|--------|-------|
| `tests/test_mt5_connection.py`, `test_session.py`, `test_replay.py`, `test_telegram_notifier.py`, `test_broker_factory.py`, `test_live_broker.py`, `test_supervisor.py`, `test_dashboard.py` | ✅ KEEP | Infrastructure tests. |
| `tests/test_circuit_breakers.py`, `test_risk_engine.py`, `test_rr_guard.py`, `test_position_sizer.py`, `test_procent_sizer.py`, `test_procent_integration.py` | 🔧 MODIFY | Risk tests need new assertions (prop caps, grade-based sizing). Will likely add many new tests, modify ~half. |
| `tests/test_broker_simulator.py` | 🔧 MODIFY | Per-tick exit semantics → also bar-exit semantics. |
| `tests/test_microstructure.py`, `test_signals.py`, `test_signal_engine.py`, `test_signal_confirmation.py`, `test_dynamic_spread.py`, `test_volatility_regime.py`, `test_session_spread_cap.py`, `test_regime_adaptive_thresholds.py` | ❌ REPLACE | Tick-microstructure tests. Will be deleted as the modules they cover are deleted. Counts: ~70 tests gone, will be replaced 1:1+ by Griff pattern tests. |
| `tests/test_backtest_e2e.py` | 🔧 MODIFY | End-to-end stays; swap fixtures for 1H bars. |
| `tests/test_walk_forward.py`, `test_param_sweep.py` | 🔧 MODIFY | Param targets change. |
| `tests/test_live_demo_mode.py` | 🔧 MODIFY | Live-demo loop body rewires. |

### 1.7 Scripts (KEEP, ADD)

| File | Status | Notes |
|------|--------|-------|
| `scripts/run_phase6.py`, `run_phase7.py`, `run_phase7_validation.py`, `diagnose_phase7.py`, `validate_pattern_3candle.py` | ✅ KEEP | Reference scripts — keep for history. Will create `run_phase8.py` for Griff backtest. |

---

## 2. Modification roadmap

### Phase 8A — Foundation audit + docs (THIS SESSION)
- [x] Audit + categorize. ← *you're reading the output*.
- [x] Write `PHASE_8_PIVOT_PLAN.md` (this file).
- [x] Write `GRIFF_STRATEGY_SPEC.md` skeleton.
- [ ] **No code changes.** Existing 186 tests remain green.

### Phase 8B — Bar infrastructure (next session, after Griff rules captured)
1. `data/bar_aggregator.py` — NEW. Ticks → OHLCV 1H bars. Pure function, testable.
2. `data/mt5_connector.py` — add `copy_rates_range`, multi-symbol fetch.
3. `strategy/structure.py` — NEW. Swing-high / swing-low / trend bias on 1H bars.
4. `data/multi_pair_feed.py` — NEW. Periodic 1H bar fetch across all forex pairs on connected prop account.
5. Tests for each.

### Phase 8C — Griff pattern detectors (TDD, once rules captured)
1. `strategy/patterns/base.py` — NEW. `Pattern` enum + `PatternMatch` dataclass with `grade`, `confluence_count`, `entry_price`, `sl`, `tp`, `pair`.
2. `strategy/patterns/flag.py` — NEW.
3. `strategy/patterns/continuation.py` — NEW.
4. `strategy/patterns/combo.py` — NEW.
5. `strategy/patterns/reversal.py` — NEW.
6. `strategy/scanner.py` — NEW. Scans all pairs each new bar, returns highest-grade match.

### Phase 8D — Prop-firm rules engine
1. `risk/prop_firm/__init__.py` — NEW.
2. `risk/prop_firm/rules.py` — NEW. FTMO + The5ers rule sets as dataclasses.
3. `risk/prop_firm/detector.py` — NEW. Auto-detect firm from MT5 `company` / `server` strings.
4. `risk/prop_firm/compliance.py` — NEW. Pre-trade compliance check (daily loss %, total loss %, news blackout, time window, max trades, consistency rule).
5. `risk/prop_firm/news_calendar.py` — NEW (or stub). NFP/CPI/FOMC blackout windows.

### Phase 8E — Risk + sizing rewire
1. Modify `risk/risk_engine.py` — integrate prop_firm/compliance, grade-based sizing, house-money mode.
2. Modify `risk/position_sizer.py` — grade-aware risk %, per-pair pip values.
3. Modify `risk/circuit_breakers.py` — IST window + max-trades-per-day + 80% early-stop.

### Phase 8F — Backtest + walk-forward on 1H bars
1. Modify `backtesting/backtest_runner.py` for bar feed.
2. Modify `backtesting/param_sweep.py` for new param surface.
3. Run on captured historical pair data (TBD which pairs first).

### Phase 8G — Live execution
1. Modify `bot.py` for `griff-scan` mode.
2. Modify `monitoring/dashboard.py` for prop-firm display.
3. Demo first → fund-test → live.

---

## 3. Dependency graph (new modules)

```
                    ┌──────────────────────┐
                    │ data/multi_pair_feed │ (NEW — fetches 1H bars across all pairs)
                    └────────────┬─────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │ data/bar_aggregator  │ (NEW — tick→bar deterministic, or pull bars)
                    └────────────┬─────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────┐
              │ strategy/structure (NEW)        │
              │  - swing high/low               │
              │  - trend bias                   │
              └──────────────┬──────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ patterns/    │    │ patterns/    │    │ patterns/    │
│   flag       │    │ continuation │    │   reversal   │   (+ combo)
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       └────────┬──────────┴───────────────────┘
                ▼
       ┌──────────────────┐
       │ strategy/scanner │ (NEW — score by confluence, pick best across pairs)
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────────────┐
       │ risk/risk_engine (MOD)   │
       │   ├─ prop_firm/compliance│
       │   ├─ position_sizer (MOD)│
       │   └─ circuit_breakers(MOD)│
       └────────┬─────────────────┘
                │
                ▼
       ┌──────────────────┐
       │ execution/broker │ (MODIFY — multi-symbol)
       └──────────────────┘
```

---

## 4. Effort estimate (15-day deadline)

| Phase | Estimate | Risk |
|-------|----------|------|
| 8A (this session) | 1 session | None — audit only. |
| 8B (bar infra) | 1–2 sessions | Low — well-trodden ground. |
| 8C (Griff patterns) | 3–4 sessions | **High** — depends on rule precision from videos. Each pattern = TDD round. |
| 8D (prop-firm rules) | 2 sessions | Medium — FTMO/The5ers rule text is public, news calendar needs source. |
| 8E (risk rewire) | 1–2 sessions | Medium — careful test migration. |
| 8F (backtest) | 1–2 sessions | Medium — need historical bar data for chosen pairs. |
| 8G (live) | 1–2 sessions + monitoring time | Medium — demo first. |
| **Total** | **10–15 sessions** | Achievable if Griff rules are clean. |

---

## 5. Known blockers / open questions

See `GRIFF_STRATEGY_SPEC.md` § "Open questions for next session" — the strategy rules
themselves are the single biggest blocker. Everything else is execution.

---

*End of pivot plan. Strategy rules to be filled in next session.*
