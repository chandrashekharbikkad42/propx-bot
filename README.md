# XAU HFT ENGINE

## PROJECT GOAL

Build an institutional-style ultra-fast XAUUSD scalping engine focused on:

- liquidity sweeps
- tick momentum
- absorption/rejection
- low-latency execution
- session-based filtering
- scalable architecture
- replay-driven validation

This is NOT a retail indicator bot.

The system will use:
- tick-level processing
- microstructure behavior
- fast execution logic
- modular architecture
- replay/backtesting systems

---

# DEVELOPMENT PHILOSOPHY

Core principles:

1. Data first
2. Replay before live
3. Modular architecture
4. Risk-first engineering
5. Lightweight infrastructure
6. No unnecessary indicators
7. Session-aware trading only

---

# TARGET MARKET

- XAUUSD
- CME Gold Futures concepts
- London + New York sessions

---

# BOT STYLE

- hyper scalping
- ultra-fast scalping
- short holding periods
- high-frequency decision logic
- low-latency execution

---

# PRIMARY SIGNALS

1. Liquidity sweeps
2. Tick acceleration
3. Absorption
4. Rejection
5. Momentum displacement
6. Spread filtering
7. Session timing

---

# TECH STACK

- Python
- MetaTrader5
- Pandas
- NumPy
- PyArrow
- AsyncIO
- WebSockets
- VS Code

---

# CURRENT PROJECT STRUCTURE

xau_hft_engine/
│
├── data/
├── strategy/
├── execution/
├── analytics/
├── backtesting/
├── config/
├── logs/
├── utils/
│
├── bot.py
├── backtest.py
├── requirements.txt
└── README.md

---

# DEVELOPMENT ROADMAP

PHASE 1
- environment setup
- MT5 connection
- tick collector

PHASE 2
- replay engine
- parquet storage
- session tagging

PHASE 3
- liquidity sweep detection
- tick momentum engine
- rejection logic

PHASE 4
- execution engine
- spread filters
- risk engine

PHASE 5
- replay testing
- optimization
- demo deployment

---

# IMPORTANT RULES

- Keep architecture modular
- Avoid bloated frameworks
- Avoid unnecessary indicators
- Optimize for low memory usage
- Use async processing where useful
- Keep systems production-oriented
- Always prioritize execution quality

---

# CURRENT STATUS

## Phase 1 — Live tick capture pipeline: COMPLETE

End-to-end verified against IC Markets Raw Spread demo on XAUUSD.

Pipeline:
- `data/mt5_connector.py` — sandboxed MT5 adapter (only file importing MetaTrader5)
- `data/tick_collector.py` — async producer, cursor-based dedup, drop-on-full back-pressure
- `data/tick_writer.py` — async consumer, Hive-partitioned parquet, frozen pyarrow schema, snappy compression
- `bot.py` — orchestrator with signal-based graceful shutdown (`SIGINT`/`SIGTERM`) and optional `BOT_AUTO_STOP_SEC` env var for timed captures
- `tests/inspect_ticks.py` — read-back validation utility

Live capture proof (30 s, 2026-05-12 08:01 UTC, London open):
- 154 XAUUSD ticks captured, 0 dropped
- 6 parquet files at `data/ticks/symbol=XAUUSD/date=2026-05-12/part-NNNNN.parquet`
- Tick rate: 5.20 ticks/sec
- Spread: mean 10.4 pts, min 6 pts, max 12 pts
- On-disk schema matches frozen `TICK_SCHEMA`

Architectural invariants enforced:
- MT5 imports isolated to a single file
- All blocking I/O (MT5 RPC, disk writes) runs in `asyncio.to_thread`
- Producer never blocks on consumer (bounded queue, drop-on-full)
- Settings immutable, loaded once from `.env` at import
- No runtime schema inference

Architectural notes: `docs/PHASE_1_NOTES.md`.

## Phase 2 — Replay engine + session tagging: NEXT

- Deterministic replay of captured parquet partitions for offline strategy work
- Session tagging on each tick (Asian / London / NY) using UTC windows
- Storage hardening: partition rollover, gap detection, integrity checks