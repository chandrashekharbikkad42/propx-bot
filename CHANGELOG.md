# Changelog

All notable changes to this project documented per phase / commit.
Hinglish allowed in descriptions where helpful — yeh internal log hai.

## [Phase 9] — 2026-05-18

FTMO Demo pre-flight + go-live prep. Switches from RoboForex paper account
to FTMO 2-Step Challenge demo (login 1513426156, $10K). DRY_RUN stays OFF
from Day 1 because the FTMO side is a demo — no real money risk. Bot runs
24/7; the session gate inside ComplianceEngine handles trading hours.

### Shell 6 — Enriched bot_started Telegram alert (2026-05-18)
- `monitoring/griff_telegram_alerts.py` — `bot_started()` now accepts
  optional `broker_name`, `prop_firm_key`, `account_balance`,
  `account_currency`. Renders the multi-line ceremony format you've been
  expecting (🤖 / Broker: X (key) / Account: $X.XX USD / Pairs: ...
  / Mode: LIVE TRADING / Status: 🟢 OPERATIONAL). DRY_RUN gets a 🟡 +
  "no orders will be placed" status line so the operator can tell at a
  glance which mode the bot booted in.
- `scripts/run_griff_live.py` — LIVE path passes the real MT5 account
  snapshot (balance, currency, prop firm key from auto-detect, broker
  name from `active_broker_name()`) into `bot_started`. DRY_RUN keeps
  all four as None — the formatter gracefully omits those lines.
- `tests/test_griff_telegram_alerts.py` — +2 tests for the rich LIVE
  context format and the DRY_RUN yellow status; existing 9 unchanged.

### Shell 5 — Windows daemon wrapper (2026-05-18)
- `scripts/run_griff_live_daemon.bat` — daemon entry point. cd to repo,
  activates the venv, sets `ACTIVE_BROKER=FTMO`, runs the bot with
  `--no-dry-run` and appends stdout/stderr to
  `logs/griff_live_daemon.log`.
- `scripts/install_griff_service.ps1` — registers a Windows Scheduled
  Task (`GriffLiveBot`) that triggers on user logon, restarts on
  failure every 1 minute up to 5 retries, runs hidden, and respects the
  bot's own DRY_RUN/REAL gate. Idempotent (re-running re-creates).
- `scripts/uninstall_griff_service.ps1` — removes the task cleanly.
- `docs/GRIFF_DAEMON.md` — install / verify / tail / stop / uninstall
  procedure plus the laptop sleep-mode caveat (close-lid = trades miss).

### Shell 4 — LIVE bar-feed loop (2026-05-18)
- `data/live_bar_poller.py` — NEW. `LiveBarPoller` polls MT5's
  `copy_rates_from_pos(symbol, TIMEFRAME_H1, 1, N)` (skips the
  currently-forming bar so only fully-closed H1 bars enter the engine).
  Per-pair last-seen-msc tracker so the same bar never fires the engine
  twice. `poll_once()` is sync + testable; `run()` is the async loop
  that catches per-iteration exceptions so a single MT5 hiccup doesn't
  kill the bot. MT5 module is dependency-injected; tests mock it.
- `scripts/run_griff_live.py` — LIVE mode now (1) connects+logs into
  MT5 via the active broker creds, (2) subscribes the 6 Griff pairs in
  Market Watch, (3) schedules `LiveBarPoller.run()` as a non-blocking
  background task alongside the HourlyReporter. Account snapshot
  (`equity`, `daily_pnl_usd`, `trades_today`, `open_position_count`)
  and per-pair ask/bid are read fresh from MT5 each scan via callable
  providers — engine sees current state, not stale cache. Clean MT5
  shutdown on exit. New `--poll-sec` and `--history-bars` CLI flags.
- `tests/test_live_bar_poller.py` — 8 tests: empty MT5 returns empty,
  first-poll emits latest, same bars twice no duplicate, new bar after
  first emits, multi-pair independent, run() stops cleanly on event,
  run() invokes engine on new bar, run() survives a poll exception.

### Shell 3 — FTMO preflight (2026-05-18)
- `scripts/ftmo_preflight.py` — NEW. Forces ACTIVE_BROKER=FTMO and runs a
  full pre-go-live check: credentials routing, MT5 init+login, account
  number / balance (within ±$200 of $10K) / currency=USD / leverage,
  prop-firm auto-detection (must resolve to `ftmo_2step_challenge`),
  rule-pack caps (5% / 10% / 4d), per-pair availability + spread for all
  6 Griff pairs, and a Telegram success ping. Exits 0 only when every
  check is ✅; otherwise 1 with a clear marker per failure.
- `config/broker_config.py` — `_credentials_for()` now resolves the MT5
  terminal binary path per broker. `FTMO_PATH` / `ROBOFOREX_PATH` env
  vars take precedence; `MT5_PATH` is the shared fallback. Necessary
  because each prop firm ships its own white-labelled terminal install
  (FTMO Global Markets MT5 Terminal vs Five Percent Online MetaTrader 5).
- `.env` — `MT5_PATH` repointed to the existing The5ers/RoboForex
  terminal; new `FTMO_PATH` and `ROBOFOREX_PATH` added so each broker
  resolves to its own binary.
- Verified live: account 1513426156, balance $10,000.00, currency USD,
  leverage 1:100, server FTMO-Demo, prop firm auto-detect ✅, all 6 Griff
  pairs visible + tradable, Telegram ping delivered.

### Shell 2 — ACTIVE_BROKER env switch (2026-05-18)
- `config/broker_config.py` — NEW. `get_active_credentials()` reads
  `ACTIVE_BROKER` and returns FTMO_* or MT5_* (RoboForex) credentials.
  Graceful fallback to the other broker when the primary's env is
  incomplete (logs a warning); both sides missing → raises
  `BrokerCredentialsMissing`. `get_credentials_for(broker)` is the
  explicit-override hook for preflight scripts and tests.
- `config/settings.py` — `_build()` now sources `mt5_login/password/
  server` via `get_active_credentials()` so a `.env` flip from
  `ACTIVE_BROKER=ROBOFOREX` to `ACTIVE_BROKER=FTMO` is the ONLY change
  needed to swap accounts. Legacy MT5_PATH still required.
- `tests/test_broker_config_active_switch.py` — 14 tests: explicit FTMO,
  explicit ROBOFOREX, missing default, invalid value, both fallback
  directions, both-missing error, force-override, non-integer login.

### Shell 1 — HourlyReporter (2026-05-18)
- `monitoring/hourly_reporter.py` — NEW. `HourlyReporter` posts a 1-hour
  Telegram digest (today: trades / P&L / DD-vs-cap; last hour: bars
  received / signals / compliance pass-blocked / open positions; healthy
  marker). `HourlyStats` is a mutable counter the engine increments per
  scan cycle. Reset on each send. Silent-hour throttle (00:00–12:00 IST):
  if nothing happened AND no open positions AND no trades today, send a
  one-line "idle" heartbeat instead of the full body. `next_top_of_hour_ms`
  helper schedules the run loop to fire at HH:00:00 UTC.
- `execution/griff_live_engine.py` — `GriffLiveEngine` accepts an optional
  `hourly_stats` and increments it inline (bars, signals, compliance
  verdicts) so the reporter has fresh numbers without polling.
- `scripts/run_griff_live.py` — `--no-hourly` opt-out; `--duration 5min`
  / `--duration 2h` auto-stop for smoke tests; hourly task scheduled as
  a non-blocking `asyncio.create_task` alongside the main scan loop and
  cancelled cleanly on shutdown.
- `tests/test_hourly_reporter.py` — 17 tests covering HourlyStats unit,
  format-full vs format-idle, silent-window logic, IST timestamping,
  send routes through notifier and resets stats, disabled notifier
  no-ops, and scheduling helper.

## [Phase 8D-Live] — 2026-05-17

Live execution + monitoring stack. 9 shells. Tests 601 → 702 (+101).
Designed alongside the existing XAUUSD scalping bot (does NOT extend
LiveBroker/OrderIntent) — Griff and scalping live in parallel.

**STATUS: COMPLETE. Ready for demo paper trading on account 37345118.**

### Shell 9 — SSL hardening + smoke verification (2026-05-17)
- `alerts/telegram_notifier.py` — `TCPConnector(ssl=...)` wired with an
  `ssl.create_default_context(cafile=certifi.where())` so Windows trust-
  store gaps no longer trigger `SSLCertVerificationError: self-signed
  certificate in certificate chain` against `api.telegram.org`. Context
  is lazy + cached per notifier instance.
- `requirements.txt` — `certifi==2026.4.22` pinned.
- `tests/test_telegram_notifier.py` — `TestTelegramSSLContext` (2 tests):
  context is `CERT_REQUIRED` and reused; `aiohttp.TCPConnector` receives
  the ssl kwarg from `send()`.
- Live smoke: real send to chat 7410045685 succeeded (timeout bumped to
  30s for first-handshake latency on slow networks).
- DRY_RUN end-to-end: `python scripts/run_griff_live.py --dry-run --once`
  bootstraps router/PM/compliance/house-money/daily/alerts, starts the
  dashboard on 127.0.0.1:8080, runs one empty scan cycle, and shuts
  down cleanly with `GriffLive shutdown complete`.

### Added
- `execution/griff_order_router.py` — multi-symbol MT5 order issuer.
  Market (Flag) + pending STOP (Continuation / Reversal) + pending LIMIT
  (Combo). Hybrid expiry: `ORDER_TIME_SPECIFIED` broker-side AND bot-side
  `cancel_pending()` next-bar. Distinct magic 786544 from the scalping
  bot. Retry on transient retcodes (mirrors LiveBroker policy).
  DRY_RUN mode never touches MT5. 20 tests.
- `execution/griff_position_manager.py` — bookkeeper for open positions
  + pending orders. Per-bar `maintain(pair, bar)`: drives SwingTracker,
  asks TrailingStopLoss for new SL, applies modify_sl via the router,
  detects bot-side SL hits, expires past-due pending orders. Adapter
  pattern: `_legacy_position()` wraps GriffOpenPosition for the existing
  TrailingStopLoss (which expects execution.position.Position). 15 tests.
- `monitoring/daily_tracker.py` — per-IST-day equity / P&L / DD / trade-
  count ledger with parquet persistence. IST trade-day = UTC hour 18:30
  rollover. Same-day reload picks up state; cross-day reload resets.
  12 tests.
- `monitoring/griff_telegram_alerts.py` — thin wrapper over
  `alerts.TelegramNotifier` with Griff-specific formatters: signal
  detected, trade opened, trade closed, kill switch, daily summary,
  bot started/stopped. Reuses the existing notifier transport so no
  new aiohttp session. 9 tests.
- `execution/griff_live_engine.py` — orchestration loop. `process_scan_cycle`
  runs scanner → per-pair best signal → compliance check → house-money
  sizing → router → position manager → telegram. `maintain_open` does
  per-pair maintenance. Also exports `griff_lots_for()` lot sizer
  (10 USD/pip/lot approximation; refine for JPY via mt5.symbol_info_tick
  in production). 13 tests.
- `monitoring/griff_dashboard.py` — aiohttp.web HTTP dashboard on
  localhost:8080 (FastAPI in the brief; aiohttp keeps deps unchanged —
  already used by TelegramNotifier). Endpoints: `/`, `/positions`,
  `/pendings`, `/daily`, `/signals`, `/health`. 6 tests using
  AioHTTPTestCase.
- `scripts/run_griff_live.py` — CLI entry. `--dry-run` is DEFAULT.
  Two-key safety: real orders require `--no-dry-run` AND
  `EXECUTION_MODE=REAL` in env; any single misconfig falls back to
  DRY_RUN with a logged reason. `--once` for smoke tests. 8 tests.
- `tests/test_griff_full_pipeline.py` — end-to-end integration suite:
  Flag market-to-SL flow, Continuation pending-to-expiry, Combo
  pending-LIMIT, pending-fill-to-position promotion, kill switches
  (daily cap / IST window / emergency stop / total loss), trailing SL
  trigger, multi-pair scan + isolation, GBPUSD reversal exclusion via
  the real detector stack, house-money trade-2-after-winner sizing,
  DRY_RUN bookkeeping. 16 tests.

### Spec interpretations (user-confirmed before any code shipped)
- Pending order expiry → HYBRID (broker-side + bot-side cancel).
- Existing infra integration → NEW classes alongside (do not extend
  LiveBroker / OrderIntent — those remain scalping-only).
- PIP_VALUE_PER_LOT_USD constant = 10.0 (MVP approximation).

## [Phase 8C-Patterns] — 2026-05-17

Four Griff pattern detectors land + scanner integration. Tests 515 → 587
(+72, all green).

### Added
- `strategy/patterns/_griff_common.py` — shared constants & helpers:
  `GRIFF_PAIRS` (6 majors/crosses), `INITIAL_SL_PIPS` (per-pair fixed stops
  for Continuation / Combo / Reversal), `pip_size()`, `synthesize_tp()`
  placeholder at 1:2 R, candle-math helpers (`body`, `range_`, wicks,
  `avg_body`). The TP synthesis is a sentinel to satisfy the existing
  `PatternSignal` contract — real exits live in `risk/trailing_sl.py`.
  See `docs/GRIFF_PATTERN_AMBIGUITIES.md` item #1.
- `strategy/patterns/flag.py` — `FlagPattern`. 4-bar window: impulse →
  pullback (Flag Low/High) → breakout → entry candle. Market exec at
  CLOSE of the entry candle. SL at Flag Low/High ± 2 pips. Excessive-
  entry-candle guard (body > 2× avg body of last 10 bars). 16 tests.
- `strategy/patterns/continuation.py` — `ContinuationPattern`. 2-bar
  window: impulse + tight pullback (body < 40% impulse, rejection wick
  > 60% pullback range). Buy Stop / Sell Stop pending 2 pips beyond the
  pullback extreme. Fixed pip SL per pair. 21 tests.
- `strategy/patterns/combo.py` — `ComboPattern`. Fires when Flag AND
  Continuation criteria coincide on the same 4-bar window. Pending LIMIT
  at min(inside-bar level, swing breakout ± 2 pips), tighter side wins.
  Fixed pip SL. 16 tests.
- `strategy/patterns/reversal.py` — `ReversalPattern`. Detects a clean
  LH/LL corrective channel (≥ 2 confirmed swings each side, strictly
  monotonic), then the first swing-high break to the upside (or low
  break to the downside). Entry 2 pips beyond the broken swing.
  "Struggle to return" gate: break bar must not have already reached
  the pending entry level (rejects fakeouts). **GBPUSD hard exclusion.**
  20 tests.
- `strategy/patterns/__init__.py` — added `build_griff_detectors()`
  helper returning the canonical 4-detector tuple, plus re-exports of
  `GRIFF_PAIRS`, `INITIAL_SL_PIPS`, `REVERSAL_EXCLUDED_PAIRS`.
- `tests/test_scanner_griff_integration.py` — end-to-end scanner ×
  detectors tests: GBPUSD reversal exclusion through the scanner, multi-
  pair signal isolation, combo + flag concurrent fire, geometry survives
  the scanner pass. 14 tests.
- `docs/GRIFF_PATTERN_AMBIGUITIES.md` — running log of every spec
  interpretation made during Phase 8C-Patterns. Each item names the
  constant / function that anchors the decision so reviewers can edit
  one place to revise.

### Not changed
- `strategy/patterns/base.py` — `PatternSignal` contract untouched
  (per phase brief rule "don't redefine Signal").
- `strategy/scanner.py` — no logic changes; the GBPUSD reversal
  exclusion lives in the detector itself, so the scanner trivially
  inherits it.

## [Phase 8C] — 2026-05-17

Griff-strategy foundation — 8 modules wired in: pattern framework, prop-firm
rules DB, house-money allocator, multi-pair scanner, prop-firm auto-detect,
compliance engine, swing tracker, trailing-SL. Tests: 470 → 515 (+45) end
of phase; the SwingTracker + TrailingSL additions land 44 of those, and one
risk_engine.py stub was reverted (docstring promised a method that didn't
exist).

### Added
- `strategy/patterns/base.py` — `Grade` (A/B/C with rank), `Direction` (BUY/
  SELL), `MarketContext`, frozen `PatternSignal` (validates BUY: sl<entry<tp
  and SELL: tp<entry<sl, plus rr_ratio / risk_distance / reward_distance
  geometry), `PatternDetector` ABC.
- `risk/prop_firm/rules.py` — `PropFirmRules` dataclass + `RULES_DB` covering
  14 stages across FTMO (1-Step / 2-Step) and The5ers (Bootcamp Step 1+2,
  Funded, High-Stakes, HRP) — caps, leverage, blackout minutes per stage.
- `risk/house_money.py` — `HouseMoneyManager` Griff risk allocator:
  trade 1 = base %; trade 2 with prior win → base + 50% of prior PnL %
  (capped at 2× base); trade 2 after loss → 0.5× base; C-grade signals
  raise. Returns `RiskAllocation` value object with rationale string.
- `strategy/scanner.py` — multi-pair multi-pattern Scanner. Iterates pairs
  × detectors per bar, ranks candidates by `(grade.rank, confidence, rr_ratio)`,
  picks single best, skips C-grade.
- `risk/prop_firm/detector.py` — `PropFirmDetector` MT5-server pattern match
  (FTMO / The5ers / 5ers); config override beats MT5; non-prop brokers
  (RoboForex / IC Markets / Pepperstone) explicitly return None.
- `risk/prop_firm/compliance.py` — `ComplianceEngine` with 7 hard kill-switches
  (IST window, daily-loss-near-cap@80% margin, total-loss-near-cap, daily
  trade cap, news blackout, SL-exceeds-remaining-room, leverage cap) +
  latching emergency-stop + dashboard status reporter.
- `strategy/swing_tracker.py` — NEW. `SwingTracker` 1-bar fractal detector,
  strict-inequality, per-pair state. `update()` returns
  `{new_swing_high, new_swing_low, broke_high, broke_low}`. Wick-break
  semantics — close irrelevant. (19 tests)
- `config/griff_config.py` — NEW. `GriffConfig` dataclass: per-pair
  `SPREAD_WIDEN_PIPS` (AUDJPY=50, AUDUSD=45, EURJPY=55, EURUSD=40, GBPUSD=45,
  NZDJPY=60), rollover `21:00` UTC, `protection_window_before_min=15`,
  `protection_window_after_min=60`, `trail_offset_pips=2.0`.
- `risk/trailing_sl.py` — NEW. `TrailingStopLoss` mechanical swing trail:
  long SL rises to (last_swing_low - 2pip), short SL falls to
  (last_swing_high + 2pip); favorable-only. Spread-hour widens during
  `[rollover - 15min, rollover + 60min)`, suppressed if structural SL is
  already at break-even or better. Per-position state. `pip_size(pair)`
  returns 0.01 for JPY pairs / 0.0001 otherwise. (25 tests)

### Tests
- `tests/test_pattern_base.py` (24)
- `tests/test_prop_firm_rules.py` (24)
- `tests/test_house_money.py` (16)
- `tests/test_scanner.py` (~22)
- `tests/test_prop_firm_detector.py` (15)
- `tests/test_compliance.py` (~18)
- `tests/test_swing_tracker.py` (19) — NEW
- `tests/test_trailing_sl.py` (25) — NEW

### Data
- 1H bar capture verified for all 6 Griff pairs (AUDJPY, AUDUSD, EURJPY,
  EURUSD, GBPUSD, NZDJPY) — 12,341–12,350 rows each, 2.0y span (2024-05-16
  to 2026-05-15), monotonic / aligned / OHLC-consistent.

### Reverted
- `risk/risk_engine.py` Phase 8C stub — docstring announced `size_position`
  + `GriffAccountState` that were never implemented; imports unused. Clean
  revert to `1f2c2a3` (Phase 7B). Sizing path stays on `calculate_lot_size`
  + `HouseMoneyManager` until a focused 8C-Patterns wiring lands.

### Foundation status
- 515 tests collected; 514 passed, 1 skipped, 0 failures (full suite).
- All 8 Phase 8C-Foundation modules complete (patterns, rules, house money,
  scanner, detector, compliance, swing tracker, trailing SL).
- Ready for Phase 8C-Patterns (concrete pattern detectors implementing the
  `PatternDetector` ABC against the captured 2yr bar history).

## [Phase 8B] — 2026-05-17

Bar infrastructure scaffolding. No Griff strategy logic yet — that's Phase 8C
once user provides exact pattern rules.

### Added
- `CHANGELOG.md` (this file).
- `config/settings.py` — Griff / prop-firm fields:
  - `prop_firm_type` (FTMO_1STEP / FTMO_2STEP / THE5ERS_BOOTCAMP / THE5ERS_HRP / NONE)
  - `forex_pairs` — 28 default majors + minors (configurable via `FOREX_PAIRS` env)
  - `ist_window_start` / `ist_window_end` — IST trading window (default 12:30–22:30)
  - `timezone` — default "Asia/Kolkata"
  - `auto_detect_pairs` — default False (Phase 9+ feature)
  - `bars_dir` — `data/bars/` for 1H bar parquets
  - HH:MM validator (`_validate_hhmm`) for window config
- `utils/session.py` — IST window helpers:
  - `to_ist(time_msc)` / `ist_date(time_msc)` / `is_within_ist_window(...)`
- `data/bar_aggregator.py` — NEW. Tick → OHLCV 1H bar aggregator + parquet I/O:
  - `BarAggregator` (per-symbol, hour-boundary detection, gap-tolerant)
  - `Bar` (frozen dataclass with `is_bullish`, `range_pts`)
  - `floor_to_timeframe_ms`, `write_bars_parquet`, `read_bars_parquet`
  - `check_bar_integrity` — monotonic / aligned / missing / OHLC sanity
  - Frozen `BAR_SCHEMA` (pyarrow)
- `data/multi_pair_feed.py` — NEW. Manages N per-symbol aggregators, emits
  `BarCloseEvent` per closed bar; gap-tolerant; optional `register_missing`.
- `data/news_calendar.py` — NEW (stub interface + static fallback):
  - `NewsCalendar` Protocol — pluggable source
  - `StaticNewsCalendar` — hardcoded high-impact events (NFP/CPI/FOMC/BoE) May–Jul 2026
  - `is_news_blackout(symbol, time_msc, window_min=2)` — currency-substring match
  - `upcoming_events(after_msc, limit)` — for dashboards
- `data/bar_capture_utils.py` — NEW. Pure helpers (`mt5_rates_to_bars`, `bars_summary`)
  separated from script so they're testable without live MT5.
- `data/mt5_connector.py` — added `copy_rates_range(symbol, timeframe, from, to)`
  (additive — no existing callers affected).
- `scripts/capture_historical_bars.py` — NEW. Idempotent MT5 → parquet pull for
  10 priority pairs (EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF,
  EURGBP, EURJPY, GBPJPY); `--years` `--pairs` `--force` flags.
- Tests:
  - `tests/test_ist_window.py` (15)
  - `tests/test_settings_griff.py` (15)
  - `tests/test_bar_aggregator.py` (24)
  - `tests/test_multi_pair_feed.py` (11)
  - `tests/test_news_calendar.py` (22)
  - `tests/test_bar_capture_utils.py` (7)
  - Total NEW: 94 tests. Existing 187 untouched.

## [Phase 8A] — 2026-05-16

### Added
- `docs/PHASE_8_PIVOT_PLAN.md` — full per-file audit + roadmap + dependency graph + effort estimate.
- `docs/GRIFF_STRATEGY_SPEC.md` — skeleton for 4 patterns + grading + house-money + prop-firm hooks + 15 open questions.

### Audit
- Categorized existing codebase: ~40% KEEP (infra), ~40% MODIFY (orchestration + risk),
  REPLACE tick microstructure (~10%), ADD NEW Griff patterns + prop-firm engine (~10%).
- No code changed in 8A; 186/187 tests remained green.

## [Phase 7B] — earlier

- See git log for pre-pivot phases (1 → 7B).
