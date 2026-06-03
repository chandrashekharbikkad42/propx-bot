# PRODUCTION BUGS FOUND — Phase 5 adversarial suite

Each entry is a real-world failure scenario the bot does NOT currently
handle correctly. Tests that surface them are marked `xfail(strict=False)`
so they remain visible without breaking CI; a normal pass after a fix
turns into an XPASS, which `--runxfail` upgrades to a hard failure
(forcing removal of the marker).

Resolve by either (a) fixing production code and dropping the xfail, or
(b) explicitly accepting the behaviour and rewriting the test to assert
the current contract.

---

## Format

```
### <short-title>

- File:    <production file>:<line>
- Test:    tests/edge_cases/<file>::<test_name>
- Scenario: <one-line repro>
- Expected: <what a correct implementation does>
- Actual:   <what the production code does today>
- Severity: LOW | MEDIUM | HIGH
- Likelihood: theoretical | rare | common
```

---

### 1. Partial-fill volume bookkeeping is ignored

- File: `execution/griff_order_router.py:162-174` (`place_market`)
- Test: `tests/edge_cases/test_broker_misbehavior.py::TestPartialFill::test_router_should_record_actual_filled_volume` (xfail)
- Scenario: MT5 returns `OrderSendResult(retcode=DONE, volume=0.5)` when the
  bot asked for `1.0` lots. The router never inspects `result.volume`; it
  stores the request's `lots` (`1.0`) into `GriffOpenPosition.lots`.
- Expected: `pos.lots == 0.5` so subsequent SL/TP/close requests target the
  ACTUAL filled volume; alternatively, the router should detect the
  shortfall and either issue a follow-up to flatten the partial or raise
  `GriffOrderError("partial_fill")`.
- Actual: `pos.lots == 1.0` (the request value). On close, the bot will
  try to flatten `1.0` lots — MT5 will close `0.5` and may reject the
  excess, leaving an orphan position or an unexpected MT5-side reject.
- Severity: **HIGH** (silent state divergence; can produce stuck lots on
  illiquid pairs / news spikes where partial fills are common).
- Likelihood: **rare to common** depending on broker + pair.

### 2. No idempotency / dedup key on `place_market`

- File: `execution/griff_order_router.py:127-174` (`place_market`)
- Test: `tests/edge_cases/test_broker_misbehavior.py::TestDuplicateOrder::test_duplicate_signal_should_be_deduped` (xfail)
- Scenario: A single `PatternSignal` is submitted twice in rapid succession
  (e.g. the scan loop fires twice because of a retry-after-timeout race, or
  the live engine's bar-feed produces two scan ticks for the same minute).
- Expected: Second call returns the prior `GriffOpenPosition` (or raises
  `GriffOrderError("duplicate")`). The router should keep an in-flight key
  derived from `(signal.symbol, signal.bar_time_msc, signal.pattern_name)`
  and drop dups.
- Actual: Both calls reach MT5; two MT5 tickets are opened. Position-manager
  registers both, doubling exposure and breaking the per-bar trade-count
  invariant the compliance engine assumes.
- Severity: **HIGH** (double exposure, compliance violation risk).
- Likelihood: **rare** (race-conditional) but catastrophic if it lands
  during a high-volatility move.

### 3. Detector can raise `ValueError` for absurdly small prices

- File: `strategy/patterns/asian_sweep.py:159-200` → `strategy/patterns/base.py:92`
- Test: `tests/edge_cases/test_market_chaos.py::test_detector_does_not_raise_for_tiny_prices` (xfail)
- Scenario: `asian_low ≈ 0.0001`, `sl_pts = 70 * 0.00001 = 0.0007`, so
  `current.low - 70*pt` falls below `0`. The detector then constructs a
  `PatternSignal` with `sl <= 0`, which raises `ValueError("entry, sl, tp
  must all be positive")` inside `PatternSignal.__post_init__`.
- Expected: detector returns `None` (defensive) on degenerate scales
  rather than propagating a ValueError to the scanner.
- Actual: ValueError surfaces; if the scanner doesn't trap it, the live
  loop aborts mid-cycle.
- Severity: **LOW** (no real symbol has price 0.0001).
- Likelihood: **theoretical** — could only happen on a synthetic test
  pair or a broker quoting bug that pegs price near 0.

---

---

## Phase 6 (end-to-end integration suite) findings

The Phase 6 suite wires every layer together (`AsianSweepLiveEngine`,
`ComplianceEngine`, `HouseMoneyManager`, `GriffOrderRouter(dry_run=True)`,
`GriffPositionManager`, `DailyTracker`) and feeds real bar sequences /
canned signals through `process_scan_cycle`. Below are the integration
issues it surfaced — none were in the unit suite because they only show
up when modules are composed.

### 4. HouseMoneyManager.calc_trade_risk raises for trade_number_today >= 3  — **FIXED**

**Fix landed:** `execution/live_engine.py` now clamps signals past
`_HOUSE_MONEY_MAX_TRADE_NUMBER` (= 2) inside `process_scan_cycle`. Excess
signals are deferred with reason `daily_trade_cap_reached` instead of
crashing the cycle. Tests in `tests/e2e/test_multi_pair_concurrent.py`
(`TestThreePairConcurrent::test_three_pairs_clamped_to_two`,
`TestEightPairCycle::test_all_eight_*_clamps_to_two`) now assert the
clamp behaviour and pass.

Original report:

- File: `risk/house_money.py:109` (validates `trade_number_today in (1, 2)`)
- Engine call site: `execution/live_engine.py:225-228` (passes
  `trade_no=self._daily.trade_count + 1` with no clamp)
- Test: `tests/e2e/test_multi_pair_concurrent.py::TestThreePairConcurrent` (xfail)
        `tests/e2e/test_multi_pair_concurrent.py::TestEightPairCycle::*` (xfail)
- Scenario: A single `process_scan_cycle` produces orders for >= 3 pairs
  in the SAME cycle. The engine records each open via `daily.record_trade_open`
  immediately after, so the 3rd iteration passes `trade_no=3` to
  HouseMoneyManager — which raises `ValueError`, killing the cycle mid-loop.
- Expected: engine should either clamp `trade_no` (HouseMoney degrades to
  defensive/standard at 3+) or skip the sizing call once the cap is reached.
  Alternatively, broaden HouseMoneyManager to support `trade_number_today`
  beyond 2 with a sensible default tier (likely STANDARD or DEFENSIVE).
- Actual: ValueError propagates out of `process_scan_cycle`, leaving any
  already-opened positions registered but the cycle's remaining signals
  silently dropped. Compliance's `max_trades_per_day=2` normally prevents
  this (compliance gate runs first), but the gate compares against
  `account.trades_today` — a snapshot at cycle START. Multiple opens in
  the same cycle all see the same snapshot and all pass compliance.
- Severity: **MEDIUM** (production caps trades at 2/day so the path is
  guarded, but ANY misconfiguration to >2 cap, or a future strategy that
  trades more pairs simultaneously, will blow up).
- Likelihood: **rare** in current config; **common** if cap raised.

### 5. Compliance worst-case loss uses constant 100_000 contract_size  — **FIXED**

**Fix landed:** `AsianSweepLiveEngine._contract_size_for(symbol)` now
resolves the per-pair `contract_size` from `PAIR_CONFIG` (XAUUSD = 100,
FX = 100_000) and passes it to `compliance.can_trade`. Small-account
XAUUSD signals are no longer over-rejected as
`sl_exceeds_remaining_daily_room`. The runner default equity in the
e2e suite was lowered back to $10k for the relevant assertions.

Original report:

- File: `execution/live_engine.py:122-126` (`self._contract_size = 100_000.0`)
       `risk/prop_firm/compliance.py:151` (worst_loss uses caller-provided value)
- Test: surfaced by `tests/e2e/test_full_trade_lifecycle.py::TestFullCycleLong`
        when `runner_factory(starting_equity=10_000)` is used — XAUUSD
        signals get rejected as `sl_exceeds_remaining_daily_room` even
        though their actual lot exposure is < 0.5% of equity.
- Scenario: For XAUUSD (`contract_size = 100`), the engine passes
  `contract_size=100_000` to `compliance.can_trade`. Worst-case loss is
  estimated as `risk_distance × 100_000 × lots` — for XAUUSD this
  overstates the real risk by 1000×. On a $10k account, a normal XAUUSD
  trade rejects on check #6.
- Expected: `process_scan_cycle` should look up `PAIR_CONFIG[symbol]["contract_size"]`
  per signal and pass that to `can_trade`. Even better: thread the
  precomputed `lots × contract_size × pip_value` USD figure directly.
- Actual: All XAUUSD signals (and any cross with non-100k contract) get
  spurious rejections on small accounts. Worked around in the e2e suite
  by raising `starting_equity` to $100k.
- Severity: **HIGH** for small accounts; rejects valid XAUUSD signals
  in production if the account is below ~$50k.
- Likelihood: **common** on funded XAUUSD accounts below $50k.

### 6. Telegram failure can crash the engine cycle (defensive concern)  — **FIXED**

**Fix landed:** All four alert call sites in `process_scan_cycle`
(`kill_switch_triggered` for blackout, `kill_switch_triggered` for
cap-reached, `kill_switch_triggered` for compliance reject,
`signal_detected`, `trade_opened`) now go through the new `_safe_alert`
helper, which wraps the await in `try/except Exception` and logs the
failure as `alert dispatch failed (non-fatal)`. The trading loop is no
longer coupled to notifier transport health.

Original report:

- File: `execution/live_engine.py:251, 263` (awaits `alerts.signal_detected`
        / `alerts.trade_opened` without try/except)
- Test: `tests/e2e/test_recovery_restart.py::TestTelegramDownNonBlocking` (xfail)
- Scenario: If the alerts layer (or the underlying `TelegramNotifier.send`)
  raises an exception, the live engine's `await self._alerts.signal_detected(sig)`
  propagates it up out of `process_scan_cycle`, breaking the cycle.
- Expected: alerts are best-effort — wrap in `try: ... except Exception:
  logger.warning(...)` so a flaky Telegram endpoint never blocks trading.
  The current `TelegramNotifier.send` catches its OWN exceptions, so this
  is purely a defense-in-depth concern; a regression in the notifier or
  the wrapper would re-introduce the risk.
- Actual: Hardened against by the notifier today; not hardened at the
  engine layer.
- Severity: **LOW** today, **MEDIUM** if any refactor changes the
  notifier's exception swallowing.
- Likelihood: **theoretical** today, **rare** after any refactor.

---

## Coverage delta (Phase 5 alone)

Tested production modules (relative to total LOC), Phase 5 suite only:

| module | stmts | covered | % |
|---|---:|---:|---:|
| `risk/prop_firm/compliance.py` | 74 | 74 | **100%** |
| `risk/prop_firm/rules.py` | 53 | 53 | **100%** |
| `execution/order.py` | 24 | 24 | **100%** |
| `execution/position.py` | 32 | 32 | **100%** |
| `data/news_calendar.py` | 53 | 53 | **100%** |
| `utils/session.py` | 43 | 43 | **100%** |
| `execution/griff_order_router.py` | 137 | 136 | **99%** |
| `risk/asian_sweep_exit.py` | 143 | 142 | **99%** |
| `data/bar_aggregator.py` | 133 | 130 | **98%** |
| `strategy/patterns/base.py` | 63 | 62 | **98%** |
| `risk/position_sizer.py` | 23 | 22 | **96%** |
| `strategy/scanner.py` | 62 | 59 | **95%** |
| `risk/house_money.py` | 61 | 58 | **95%** |
| `strategy/patterns/asian_sweep.py` | 115 | 107 | **93%** |
| `execution/live_broker.py` | 104 | 97 | **93%** |
| `execution/griff_position_manager.py` | 98 | 90 | **92%** |
| `config/asian_sweep_config.py` | 48 | 45 | **94%** |
| `execution/broker_simulator.py` | 62 | 51 | **82%** |
| `strategy/swing_tracker.py` | 47 | 39 | **83%** |
| `risk/trailing_sl.py` | 87 | 62 | **71%** |
| `risk/circuit_breakers.py` | 47 | 34 | **72%** |
| **TOTAL (touched modules)** | **1608** | **1413** | **~88%** |

Untouched (and intentionally out of Phase-5 scope): `live_engine.py` async
orchestration loop, `multi_pair_feed.py`, `mt5_connector.py`,
`bar_capture_utils.py`, `tick_writer.py`, `broker_factory.py`,
`prop_firm/detector.py` (auto-firm-detection from balance), and the
monitoring/dashboard modules. Those are integration-shaped and Phase-1..4
cover them.
