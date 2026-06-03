# Phase 1 — Architecture Decisions

Brief log of the non-obvious design calls made while building the live tick capture pipeline. Each entry is short on purpose — context, decision, why-not-the-alternative.

---

## 1. Frozen pyarrow schema (no runtime inference)

**Context**: Tick writer (`data/tick_writer.py`) needs to convert a list of `Tick` dataclasses into a parquet file.

**Decision**: Declare `TICK_SCHEMA` as a module-level `pa.schema(...)` constant. Every `pa.table(...)` call passes it explicitly; the reader (`tests/inspect_ticks.py`) verifies on-disk file schema against it.

**Why**:
- Schema drift across part-files is a silent killer for replay. If `volume` is `int64` in one file and `int32` in the next, concat either explodes loudly or silently coerces.
- Letting pyarrow infer types from Python lists is non-deterministic across runs — `last=0.0` could be `float64` one batch and `int64` the next if Python ints sneak in.
- The on-disk format is now a **contract**, not an emergent property.

**Cost**: One extra line per column when the schema evolves. Trivial.

---

## 2. Drop-on-full back-pressure

**Context**: Producer (collector) and consumer (writer) run as separate asyncio tasks linked by a bounded `asyncio.Queue`.

**Decision**: `Queue(maxsize=10000)`. On `QueueFull`, the collector increments `dropped` and continues — it does NOT await space.

**Why**:
- HFT pipelines are inherently lossy at saturation. The real question is *which* ticks you lose. Losing the most recent (blocking producer) is worse than losing the oldest queued.
- The collector must stay current with market state. A blocked producer would build up a lag tail that's unrecoverable.
- 10 000 = 10× the writer's flush size — absorbs flush-window stalls without unbounded memory growth.

**Operational signal**: `dropped > 0` in shutdown logs = writer or disk is the bottleneck. So far 0 drops at 5 ticks/sec; nowhere near saturation.

**Alternatives rejected**:
- Unbounded queue → unbounded memory under sustained backpressure. No.
- `put()` (blocking) → defeats the purpose of an async producer; producer would couple to consumer's worst case.

---

## 3. `BOT_AUTO_STOP_SEC` env var (not a CLI flag, not a test wrapper)

**Context**: Needed a way to run timed captures and smoke tests without fighting Windows console signal plumbing.

**Decision**: Optional env var. If set to a positive number, `bot.py` schedules `stop.set()` via `loop.call_later()` at startup — same shutdown path as `SIGINT`.

**Why**:
- Cleanly delivering `SIGINT` to a Python subprocess from another Python on Windows is awkward (no PTY, `CTRL_C_EVENT` broadcasts to the whole console group, signal-vs-event semantics differ).
- A separate test wrapper that patches `bot.run()` would diverge from the production code path — defeats the point of an integration test.
- An env var costs ~3 lines, exercises the real shutdown sequence, and is also useful for CI / scheduled capture jobs / replay-data generation runs.

**Alternatives rejected**:
- CLI arg → pollutes the production binary's interface for behavior used mostly in tests.
- Adding a Windows `SIGBREAK` handler + subprocess wrapper → fixes only the test scenario, no operational upside.

---

## 4. Hive partition layout: `symbol=XAUUSD/date=YYYY-MM-DD/part-NNNNN.parquet`

**Context**: Need a directory layout for tick files that scales to multiple symbols and supports resume-safe writes.

**Decision**: Hive-style `key=value` directories, UTC date partitions, zero-padded `part-NNNNN` sequence.

**Why this layout**:
- **Date partition** = natural query unit. "Replay London open from 2026-05-12" is a single-directory read.
- **Symbol partition** = future-proof when we add XAUEUR, US500, etc. — no restructuring required.
- **`part-NNNNN`** (5-digit) = monotonic, resume-safe. Multiple bot restarts on the same day append without collision. Ceiling of 100 000 parts/day is unreachable with current flush triggers.

**Why Hive specifically**:
- DuckDB, PyArrow Dataset, Spark, Trino, etc. auto-discover Hive-partitioned data. No custom catalog needed.
- PyArrow's auto-discovery actually surfaced this in `inspect_ticks.py` — `pq.read_table()` injects `symbol`/`date` as dictionary columns from the path. We had to explicitly opt-out of the partition columns to validate the on-disk schema cleanly. Worth knowing.

**Why UTC dates** (not broker-server time, not local time):
- Broker server timezone drift across DST regimes would partition inconsistently.
- Strategy logic will need UTC for session windows anyway (London = 07–16 UTC, NY = 12–21 UTC).
- One canonical clock everywhere. Easier to reason about, easier to debug.

---
