# Griff Strategy Specification — SKELETON

> **Status:** Skeleton only. Pattern rules to be filled after watching Griff's
> videos in the next session.
>
> *Bhai, ye sirf dhanche ka document hai. Rules abhi blank rakhe hain — agle
> session me Griff videos dekh ke har pattern ke confluence rules fill karenge.*

---

## 0. Strategy charter (locked specs)

- **Single-trader replication:** pure Griff, no hybridization.
- **Timeframe:** 1H entries only. (HTF context optional — TBD if Griff uses 4H/D for bias.)
- **Universe:** all forex pairs available on the connected prop account.
- **Trade selection:** scan all pairs each new 1H bar close; take **only 1 trade per cycle** — the highest-grade match.
- **Max trades per day:** 2.
- **Time window:** 12:30 PM – 10:30 PM IST (London open → NY close + overlap).
- **Verified outcome (Griff):** 80% return in 30 days on FTMO (per source — to verify).

---

## 1. Pattern catalogue (4 patterns)

### 1.1 FLAG
- **Description:** *(TBD — fill from videos)*
- **HTF context required?** *(TBD)*
- **Trigger bar:** *(TBD — close above/below something? rejection wick?)*
- **Entry rule:** *(TBD — bar close? next bar open? limit at fib?)*
- **Stop loss:** *(TBD — beyond what swing structure)*
- **Take profit / target:** *(TBD — fixed R:R? structure-based?)*
- **Required confluences for A-grade (all of):**
  - [ ] *(TBD)*
  - [ ] *(TBD)*
  - [ ] *(TBD)*
- **Confluence count thresholds:**
  - A-grade: all required confluences present → risk 1.0%
  - B-grade: all but 1 → risk 0.5%
  - C-grade: 2+ missing → **SKIP** (do not trade)
- **Notes:** *(TBD)*

### 1.2 CONTINUATION
- **Description:** *(TBD)*
- **HTF context required?** *(TBD)*
- **Trigger bar:** *(TBD)*
- **Entry rule:** *(TBD)*
- **Stop loss:** *(TBD)*
- **Take profit:** *(TBD)*
- **Required confluences for A-grade:**
  - [ ] *(TBD)*
- **Notes:** *(TBD)*

### 1.3 COMBO
- **Description:** *(TBD — combination of which two? Flag + Continuation?)*
- **HTF context required?** *(TBD)*
- **Trigger bar:** *(TBD)*
- **Entry rule:** *(TBD)*
- **Stop loss:** *(TBD)*
- **Take profit:** *(TBD)*
- **Required confluences for A-grade:**
  - [ ] *(TBD)*
- **Notes:** *(TBD)*

### 1.4 REVERSAL
- **Description:** *(TBD)*
- **HTF context required?** *(TBD)*
- **Trigger bar:** *(TBD — pin bar? engulfing? structure break?)*
- **Entry rule:** *(TBD)*
- **Stop loss:** *(TBD)*
- **Take profit:** *(TBD)*
- **Required confluences for A-grade:**
  - [ ] *(TBD)*
- **Notes:** *(TBD)*

---

## 2. Grading + risk allocation

| Grade | Confluence rule | Risk % of equity | Action |
|-------|-----------------|------------------|--------|
| A | All required confluences | 1.0% | Trade |
| B | Strong but 1 missing | 0.5% | Trade |
| C | 2+ missing | n/a | **Skip** |

Implementation: `risk/position_sizer.py:calculate_lot_size_griff(equity, grade, sl_pips, pair)`.

---

## 3. House-money protocol (compounding)

Per trading day:
- **Trade 1:** Normal grade-based risk (A=1.0%, B=0.5%).
- **Trade 2 — depends on Trade 1 result:**
  - **Trade 1 WON →** "house money" mode. Use Trade 1's *profit* as additional SL buffer.
    Effective risk on Trade 2 can be larger because the cushion is house money, not principal.
    Worst case Trade 2 loses its profit-buffer + standard risk; net day still positive.
  - **Trade 1 LOST →** Defensive mode. Use a smaller risk % (TBD — likely 0.25%) on Trade 2.
    Goal: cap worst-case day at -1%, not -2%.

Worst-case day: -1% (one loss + skip remaining).
Best-case day: +4% to +6% (two compounded wins).

Implementation:
- `risk/risk_engine.py` tracks `trade_1_result` per UTC day (or IST day — TBD which calendar).
- `position_sizer` consults `trade_count_today` + `trade_1_result` before sizing Trade 2.

---

## 4. Prop-firm compliance (hard constraints)

Auto-detected from MT5 `account_info().company` / `server` at connect time:

| Constraint | FTMO | The5ers | Bot enforcement |
|------------|------|---------|-----------------|
| Max daily loss | 3–5% | 3–5% | Stop trading at **80% of cap** (early kill). |
| Max total loss | 6–10% | 5–10% | Stop trading at **80% of cap**. |
| Min trading days | 4 (1-step), 10 (2-step) | varies | Tracked, alerted, not strictly enforceable by bot. |
| Consistency rule | 1-Step: best day ≤50% of total profit | n/a | Pre-trade gate: if today's projected PnL would breach, scale down. |
| Tick scalping ban | Yes | Yes | Strategy is 1H — safe. |
| HFT ban | Yes | Yes | Strategy is 1H — safe. |
| Visible SL/TP | Mandatory | Mandatory | LiveBroker already sends `sl=` `tp=` in MT5 request. ✅ |
| News blackout | 2min before/after high-impact (NFP/CPI/FOMC) | Similar | Hard pre-trade veto if within window. |

**Kill-switches (in order):**
1. Outside IST time window → no trade.
2. News window active → no trade.
3. 2 trades already done today → no trade.
4. Daily loss ≥ 80% of cap → freeze for the day.
5. Total loss ≥ 80% of cap → freeze indefinitely, require manual reset + Telegram alert.

---

## 5. Time window

- **Active hours:** 12:30 IST – 22:30 IST (= 07:00 UTC – 17:00 UTC).
- Covers London open (07:00 UTC), London-NY overlap (12:00–16:00 UTC), NY open (12:30 UTC).
- Outside this window: scanner can still run for context but no trades placed.

---

## 6. Scanner logic (multi-pair, 1H bar close)

```
On every 1H bar close (top of UTC hour):
  if outside IST window OR news blackout OR daily-cap hit OR trades_today >= 2:
      return

  matches: list[PatternMatch] = []
  for pair in all_forex_pairs:
      bars = fetch_last_N_bars_1h(pair)
      structure = compute_structure(bars)  # swing-H/L, trend bias
      for pattern in [FLAG, CONTINUATION, COMBO, REVERSAL]:
          m = pattern.match(bars, structure)
          if m is not None:
              matches.append(m)

  matches = [m for m in matches if m.grade in (A, B)]  # skip C
  if not matches:
      return

  best = max(matches, key=lambda m: (m.grade_rank, m.confluence_count))
  intent = build_order_intent(best, grade_to_risk_pct(best.grade), trade_count_today)
  if compliance_ok(intent):
      execute(intent)
```

---

## 7. Open questions for next session

These need answers from the Griff videos before pattern code can be written:

1. **Bias filter** — does Griff require HTF bias (4H/D)? If yes, what defines bullish vs bearish bias?
2. **Pattern definitions** — exact bar mechanics per pattern (close above structure? engulfing? specific wick ratios?).
3. **Confluence list per pattern** — what factors qualify (RSI? volume? specific S/R zones? Fib levels? session-of-day? prior-day high/low?).
4. **Entry execution** — market on bar close, or pending order somewhere on the next bar?
5. **SL placement** — fixed pips beyond swing? ATR-based? structural?
6. **TP / target** — fixed R:R (1:2? 1:3?), structural target, or trail?
7. **Pair selection priority** — when two pairs score equal A-grade, what's the tiebreaker (spread? historical win-rate? majors over crosses)?
8. **Pair exclusions** — does Griff explicitly avoid certain pairs (exotic crosses, XAU during NFP, etc.)?
9. **Calendar definition** — UTC day vs IST day for daily reset and trade counting? Prop firms usually use server-time, but IST window suggests broker-day might differ.
10. **House-money exact formula** — what risk % for Trade 2 after a win? Is it "previous profit + standard %" or "standard % × multiplier"?
11. **Defensive Trade 2 risk %** — confirmed 0.25%, or something else?
12. **News filter source** — ForexFactory? MT5 economic calendar plugin? Manual JSON?
13. **News whitelist** — only NFP/CPI/FOMC, or full high-impact list?
14. **Re-entry rule** — if a trade hits SL within minutes, can the same pattern re-trigger? (Probably no — 2 trades/day cap covers it, but worth confirming.)
15. **Weekend / Friday-close** — any "no new positions after Friday X UTC" rule?

---

*End of skeleton. Fill rules in next session.*
