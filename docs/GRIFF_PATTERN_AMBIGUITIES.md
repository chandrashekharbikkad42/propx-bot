# Griff Pattern Detectors — Ambiguities & Interpretation Decisions

Companion to Phase 8C-Patterns. Captures every place the spec was loose
and the concrete numeric / structural choice we shipped, so reviewers can
either bless or revise.

Date: 2026-05-17.

---

## 1. NO take-profit in Griff spec

**Spec:** "Return Signal dataclass (no TP — only entry + initial SL)".
**Conflict:** `strategy.patterns.base.PatternSignal` REQUIRES a positive
`tp` AND enforces `sl < entry < tp` (BUY) / `tp < entry < sl` (SELL).
**Decision (user-approved):** Synthesise a placeholder TP at **1:2 R**:
`tp = entry ± 2 × |entry − sl|` in the favourable direction. Real exits
are governed by `risk/trailing_sl.py`; the TP field is contract-satisfying
sentinel only.
**Code:** `strategy/patterns/_griff_common.py:synthesize_tp`.

## 2. Flag — "Flag Low break" semantics

**Spec:** "Market execution at CLOSE of first 1H retrace candle immediately
after Flag Low break".
**Interpretation:** "Flag Low break" = the breakout bar resumes the impulse
direction by closing beyond the pullback extreme. "First 1H retrace candle"
= the IMMEDIATELY following bar (a small counter-trend candle).
4-bar window: impulse → pullback (Flag Low) → breakout → entry candle.
**Code:** `strategy/patterns/flag.py:_is_bull_flag` / `_is_bear_flag`.

## 3. Flag — "excessively large entry candle"

**Spec:** "Reject if entry candle is excessively large (>2× avg body of
last 10 bars)".
**Decision:** Implemented literally — body > `2.0 × mean(body)` over the
trailing 10 bars (`AVG_BODY_LOOKBACK=10`, `EXCESSIVE_BODY_MULT=2.0`).
**Code:** `_griff_common.AVG_BODY_LOOKBACK`, `_griff_common.EXCESSIVE_BODY_MULT`.

## 4. Continuation — impulse-strength threshold

**Spec:** silent on what makes a bar "large body, directional".
**Decision:** body / range >= **0.60**. Same threshold reused by Flag.
**Code:** `flag.py:_IMPULSE_BODY_RATIO_MIN`, `continuation.py:_IMPULSE_BODY_RATIO_MIN`.

## 5. Continuation — pullback wick threshold

**Spec:** "body < 40% of impulse body; wick on rejection side > 60% of
pullback range".
**Decision:** Implemented literally:
`CONT_PULLBACK_BODY_PCT_MAX = 0.40`, `CONT_PULLBACK_WICK_PCT_MIN = 0.60`.

## 6. Continuation — "cancel if not triggered next bar"

**Spec:** The pending order cancels on the next bar if untriggered.
**Decision:** The DETECTOR only emits at the moment the pullback bar just
closed (bars[-1]). One hour later the same setup is at bars[-2..-1+1] →
no longer the immediate window, so the detector naturally stops re-emitting.
Order-cancellation timing lives in the execution layer (Phase 8D-Live).

## 7. Combo — "inside bar level OR 2 pips beyond swing breakout (tighter)"

**Spec:** Limit at inside-bar level OR 2 pips beyond swing breakout,
whichever is tighter.
**Decision:**
  - "Inside bar" check: `pullback.high <= impulse.high AND
    pullback.low >= impulse.low`.
  - "Swing breakout level": pullback.low + 2 pips (BUY) /
    pullback.high − 2 pips (SELL).
  - "Tighter" for a long LIMIT = HIGHER level (less retrace needed) →
    `entry = max(inside, swing)`; mirrored for shorts.
**Code:** `combo.py:ComboPattern.detect`.

## 8. Combo — "cleanly retraces to entry zone"

**Spec:** Price must cleanly retrace to entry zone.
**Decision:** At scan time we cannot wait for a future fill, so the
detector emits the LIMIT level and the execution layer decides whether
to leave it pending and how long.

## 9. Reversal — "consecutive low breaks + retrace fails"

**Spec:** Lower lows + retrace fails to break previous swing high.
**Decision:** Formalised as ALL confirmed swing highs strictly decreasing
AND ALL confirmed swing lows strictly decreasing within the input window.
Requires at least 2 of each. A single higher high anywhere invalidates.
**Code:** `reversal.py:_is_strictly_decreasing` + the bull-reversal block.

## 10. Reversal — "first structural break of a swing high"

**Spec:** Trigger fires on the first swing-high break after the
corrective channel.
**Decision:** The break must occur on the most recent bar (`bars[-1]`).
The detector does not "remember" a break from an earlier bar — if the
break printed two bars ago, the post-break candle would have already
either filled or invalidated the pending order, which is a decision the
execution layer owns.

## 11. Reversal — "struggle to return to entry zone"

**Spec:** Price must STRUGGLE to return; immediate fill rejected.
**Decision:** Implemented as: the break bar itself must NOT have already
reached the pending entry level. For BUY: `bars[-1].high < entry`. The
break is then a "shallow break" (between `swing_high` and
`swing_high + 2 pips`), and the pending order requires further upside on
a subsequent bar before it fills.

## 12. Reversal — GBPUSD hard exclusion

**Spec:** GBPUSD never takes a reversal trade.
**Decision:** Implemented at detector level (`REVERSAL_EXCLUDED_PAIRS =
{"GBPUSD"}`) — returns `None` unconditionally. Scanner-level integration
test verifies the end-to-end exclusion.

---

## Items needing user review

Items 2, 7, 8, 10, 11 above involve subjective interpretation of phrases
from the prompt. None blocks Phase 8D-Live, but a Griff-video re-check
might tighten or loosen them. Ping us with corrections and we'll port
the changes — every interpretation has a single constant or a small
function as its anchor.
