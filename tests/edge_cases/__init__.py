"""Phase-5 adversarial / edge-case test suite.

These tests are intentionally not happy-path. They hunt for: market chaos
(gaps, spikes, halts, malformed bars), session-boundary surprises (DST,
midnight rollover, news edge), broker misbehavior (rejects, partial fills,
disconnects), data-integrity corruption, compliance boundary exploits,
numerical precision traps, and concurrency / state-recovery anomalies.

A failing test here is more likely to surface a production bug than a
detector-spec bug. The convention used by the suite is:

  - if the bot ALREADY guards against the scenario, the test asserts it.
  - if the scenario reveals a real production gap, the test is marked
    `xfail(strict=False, reason="PROD BUG: ...")` and the bug is logged
    in tests/edge_cases/PROD_BUGS.md for human review.

No production code is modified by this suite.
"""
