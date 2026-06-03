"""E2E — compliance gate wired into the live pipeline.

Exercises each of the 7 kill-switches in the REAL engine path (not in
isolation against ComplianceEngine.can_trade — that is unit-tested
elsewhere). The point is to confirm the engine consumes the gate's verdict
correctly and produces the right rejection / alert / no-order outcome.

Also covers:
  - The5%ers ruleset end-to-end
  - FTMO challenge / verification / funded end-to-end
  - Pre-trade reject when SL would breach daily DD
  - Drawdown halt → no new trades → recovery → resume
"""

from __future__ import annotations
from unittest.mock import MagicMock

import pytest

from config.asian_sweep_config import PAIR_CONFIG, PAIRS
from data.news_calendar import NewsEvent
from risk.prop_firm.compliance import AccountState
from risk.prop_firm.rules import RULES_DB
from strategy.patterns.base import Direction, Grade, PatternSignal

from tests.e2e.fixtures.scenario_runner import (
    ScenarioRunner, long_sweep_bars, hour_msc,
)


ALL_RULE_KEYS = list(RULES_DB.keys())


def _sig(symbol: str = "EURUSD", hour: int = 8) -> PatternSignal:
    pt = float(PAIR_CONFIG[symbol]["point"])
    entry = 1.10000 if symbol != "XAUUSD" else 2000.00
    risk = 100 * pt
    return PatternSignal(
        pattern_name="ASIAN_SWEEP", symbol=symbol,
        direction=Direction.BUY, entry=entry,
        sl=entry - risk, tp=entry + risk * 2.5,
        confidence=0.9, grade=Grade.A,
        confluences_met=("asian_sweep_low", "LONDON", "bias_neutral", "q9",
                          f"tp1_{entry + risk:.5f}"),
        bar_time_msc=hour_msc(2026, 4, 15, hour),
    )


def _inject(r, sigs):
    r.scanner = MagicMock()
    r.scanner.scan_all = MagicMock(return_value=tuple(sigs))
    r.engine._scanner = r.scanner


# ===========================================================================
# 1. IST window kill-switch (#1)
# ===========================================================================

class TestKillSwitchIstWindow:
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_outside_window_rejects(self, pair, runner_factory):
        r = runner_factory(ist_window_start="02:00", ist_window_end="03:00")
        s = _sig(pair, hour=8)
        _inject(r, [s])
        r.run_cycle({pair: []}, now_msc=s.bar_time_msc,
                    ask_by_pair={pair: s.entry},
                    bid_by_pair={pair: s.entry})
        assert "outside_ist_window" in r.result.all_rejection_reasons
        assert len(r.pm.open_positions) == 0

    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_inside_window_allows(self, pair, runner_factory):
        r = runner_factory(ist_window_start="00:00", ist_window_end="23:59")
        s = _sig(pair, hour=8)
        _inject(r, [s])
        r.run_cycle({pair: []}, now_msc=s.bar_time_msc,
                    ask_by_pair={pair: s.entry},
                    bid_by_pair={pair: s.entry})
        assert "outside_ist_window" not in r.result.all_rejection_reasons


# ===========================================================================
# 2. Daily loss cap kill-switch (#2)
# ===========================================================================

class TestKillSwitchDailyLoss:
    @pytest.mark.parametrize("rule_key", ALL_RULE_KEYS)
    def test_daily_loss_near_cap_rejects(self, rule_key, runner_factory):
        r = runner_factory(rules_key=rule_key, starting_equity=100_000.0)
        # Push daily_pnl to -80% of cap → reject.
        rules = RULES_DB[rule_key]
        daily_cap = 100_000.0 * (rules.max_daily_loss_pct / 100.0)
        acct = r.account_with(daily_pnl_usd=-(daily_cap * 0.85))
        s = _sig("EURUSD")
        _inject(r, [s])
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry}, account=acct)
        assert "daily_loss_near_cap" in r.result.all_rejection_reasons

    @pytest.mark.parametrize("rule_key", ALL_RULE_KEYS)
    def test_daily_pnl_well_below_cap_passes(self, rule_key, runner_factory):
        r = runner_factory(rules_key=rule_key, starting_equity=100_000.0)
        rules = RULES_DB[rule_key]
        daily_cap = 100_000.0 * (rules.max_daily_loss_pct / 100.0)
        # 10% of cap consumed → far from limit.
        acct = r.account_with(daily_pnl_usd=-(daily_cap * 0.10))
        s = _sig("EURUSD")
        _inject(r, [s])
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry}, account=acct)
        assert "daily_loss_near_cap" not in r.result.all_rejection_reasons


# ===========================================================================
# 3. Total loss cap kill-switch (#3)
# ===========================================================================

class TestKillSwitchTotalLoss:
    @pytest.mark.parametrize("rule_key", ALL_RULE_KEYS)
    def test_total_loss_near_cap_rejects(self, rule_key, runner_factory):
        r = runner_factory(rules_key=rule_key, starting_equity=100_000.0)
        rules = RULES_DB[rule_key]
        total_cap = 100_000.0 * (rules.max_total_loss_pct / 100.0)
        # Equity dropped to 80% of total_cap loss.
        eroded = 100_000.0 - (total_cap * 0.85)
        acct = r.account_with(equity=eroded)
        s = _sig("EURUSD")
        _inject(r, [s])
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry}, account=acct)
        assert "total_loss_near_cap" in r.result.all_rejection_reasons


# ===========================================================================
# 4. Max trades per day kill-switch (#4)
# ===========================================================================

class TestKillSwitchTradeCount:
    @pytest.mark.parametrize("cap", [1, 2, 3, 5, 10])
    def test_at_cap_rejects(self, cap, runner_factory):
        r = runner_factory(max_trades_per_day=cap)
        acct = r.account_with(trades_today=cap)
        s = _sig("EURUSD")
        _inject(r, [s])
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry}, account=acct)
        assert "daily_trade_cap_reached" in r.result.all_rejection_reasons

    @pytest.mark.parametrize("cap", [2, 3, 5, 10])
    def test_below_cap_allows(self, cap, runner_factory):
        r = runner_factory(max_trades_per_day=cap)
        acct = r.account_with(trades_today=cap - 1)
        s = _sig("EURUSD")
        _inject(r, [s])
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry}, account=acct)
        assert "daily_trade_cap_reached" not in r.result.all_rejection_reasons


# ===========================================================================
# 5. News blackout kill-switch (#5)
# ===========================================================================

class TestKillSwitchNewsBlackout:
    @pytest.mark.parametrize("pair", [p for p in PAIRS if "USD" in p])
    def test_usd_news_blackout_blocks_usd_pair(self, pair, runner_factory):
        s = _sig(pair)
        event = NewsEvent(time_msc=s.bar_time_msc, currency="USD",
                           title="NFP", impact="HIGH")
        r = runner_factory(news_events=[event])
        _inject(r, [s])
        r.run_cycle({pair: []}, now_msc=s.bar_time_msc,
                    ask_by_pair={pair: s.entry},
                    bid_by_pair={pair: s.entry})
        assert "news_blackout" in r.result.all_rejection_reasons

    @pytest.mark.parametrize("pair", [p for p in PAIRS if "USD" not in p])
    def test_usd_news_does_not_blackout_non_usd_pair(self, pair, runner_factory):
        s = _sig(pair)
        event = NewsEvent(time_msc=s.bar_time_msc, currency="USD",
                           title="NFP", impact="HIGH")
        r = runner_factory(news_events=[event])
        _inject(r, [s])
        r.run_cycle({pair: []}, now_msc=s.bar_time_msc,
                    ask_by_pair={pair: s.entry},
                    bid_by_pair={pair: s.entry})
        assert "news_blackout" not in r.result.all_rejection_reasons


# ===========================================================================
# 6. SL exceeds remaining daily room (#6) — would single-handedly breach cap
# ===========================================================================

class TestKillSwitchSLBudget:
    def test_giant_sl_rejects(self, runner_factory):
        # Build a signal whose risk_distance × 100k × 0.01 > daily cap budget.
        # $1k account, 5% daily cap = $50 → 80% margin = $40 room.
        # risk_distance × 1000 must be > 40 → risk_distance > 0.04.
        s = PatternSignal(
            pattern_name="ASIAN_SWEEP", symbol="EURUSD",
            direction=Direction.BUY, entry=1.10000,
            sl=1.04000, tp=1.30000,  # 6000-pip risk
            confidence=0.9, grade=Grade.A,
            confluences_met=("asian_sweep_low", "LONDON",
                              "bias_neutral", "q9", "tp1_1.10100"),
            bar_time_msc=hour_msc(2026, 4, 15, 8),
        )
        r = runner_factory(starting_equity=1_000.0)
        acct = r.account_with(equity=1_000.0, starting_equity=1_000.0,
                                daily_start_equity=1_000.0)
        _inject(r, [s])
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry}, account=acct)
        assert "sl_exceeds_remaining_daily_room" in r.result.all_rejection_reasons


# ===========================================================================
# 7. Leverage cap kill-switch (#7)
# ===========================================================================

class TestKillSwitchLeverage:
    @pytest.mark.parametrize("rule_key", ALL_RULE_KEYS)
    def test_low_equity_blocks_for_leverage(self, rule_key, runner_factory):
        # notional = signal.entry × contract × lots. With contract=100_000
        # and equity=$10, leverage spectacularly exceeded.
        r = runner_factory(rules_key=rule_key, starting_equity=10.0)
        acct = r.account_with(equity=10.0, starting_equity=10.0,
                                daily_start_equity=10.0)
        s = _sig("EURUSD")
        _inject(r, [s])
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry}, account=acct)
        # Either leverage cap or sl_exceeds_remaining_daily_room reject.
        reasons = r.result.all_rejection_reasons
        assert any(
            x in reasons for x in
            ("exceeds_leverage_cap", "sl_exceeds_remaining_daily_room",
             "total_loss_near_cap")
        )


# ===========================================================================
# 8. Emergency stop (latch) — no trades while stopped
# ===========================================================================

class TestEmergencyStop:
    def test_emergency_stop_blocks_all_trades(self, runner_factory):
        r = runner_factory()
        r.compliance.emergency_stop("ops-paused")
        s = _sig("EURUSD")
        _inject(r, [s])
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry})
        assert any("emergency_stop" in r for r in r.result.all_rejection_reasons)
        assert len(r.pm.open_positions) == 0

    def test_clear_emergency_resumes_trading(self, runner_factory):
        r = runner_factory()
        r.compliance.emergency_stop("test")
        r.compliance.clear_emergency()
        s = _sig("EURUSD")
        _inject(r, [s])
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry})
        assert len(r.pm.open_positions) == 1


# ===========================================================================
# 9. The5%ers ruleset end-to-end across all stages
# ===========================================================================

THE5ERS_KEYS = [k for k in RULES_DB.keys() if k.startswith("the5ers_")]


class TestThe5ersEndToEnd:
    @pytest.mark.parametrize("rule_key", THE5ERS_KEYS)
    def test_normal_flow_allows_trade(self, rule_key, runner_factory):
        r = runner_factory(rules_key=rule_key)
        s = _sig("EURUSD")
        _inject(r, [s])
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry})
        # Healthy state → trade should open.
        assert len(r.pm.open_positions) == 1


# ===========================================================================
# 10. FTMO ruleset end-to-end across all stages
# ===========================================================================

FTMO_KEYS = [k for k in RULES_DB.keys() if k.startswith("ftmo_")]


class TestFTMOEndToEnd:
    @pytest.mark.parametrize("rule_key", FTMO_KEYS)
    def test_normal_flow_allows_trade(self, rule_key, runner_factory):
        r = runner_factory(rules_key=rule_key)
        s = _sig("EURUSD")
        _inject(r, [s])
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry})
        assert len(r.pm.open_positions) == 1


# ===========================================================================
# 11. Drawdown halt → no trades → recovery → resume
# ===========================================================================

class TestDrawdownHaltRecovery:
    def test_halt_no_trades_then_recovery_resumes(self, runner_factory):
        r = runner_factory()
        # Cycle 1: in drawdown, no trade.
        s = _sig("EURUSD", hour=8)
        _inject(r, [s])
        rules = r.compliance.rules
        daily_cap = 100_000.0 * (rules.max_daily_loss_pct / 100.0)
        acct_halt = r.account_with(daily_pnl_usd=-(daily_cap * 0.9))
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry}, account=acct_halt)
        assert len(r.pm.open_positions) == 0
        # Cycle 2: equity recovered.
        s2 = _sig("EURUSD", hour=9)
        _inject(r, [s2])
        acct_ok = r.account_with(daily_pnl_usd=-(daily_cap * 0.1))
        r.run_cycle({"EURUSD": []}, now_msc=s2.bar_time_msc,
                    ask_by_pair={"EURUSD": s2.entry},
                    bid_by_pair={"EURUSD": s2.entry}, account=acct_ok)
        assert len(r.pm.open_positions) == 1


# ===========================================================================
# 12. Multiple rejections on same cycle — all captured
# ===========================================================================

class TestMultipleRejections:
    @pytest.mark.parametrize("p1,p2", [
        ("EURUSD", "GBPUSD"),
        ("EURUSD", "USDCAD"),
        ("XAUUSD", "USDCHF"),
        ("AUDCHF", "AUDNZD"),
    ])
    def test_two_pairs_both_rejected(self, p1, p2, runner_factory):
        r = runner_factory(ist_window_start="02:00", ist_window_end="03:00")
        s1, s2 = _sig(p1), _sig(p2)
        _inject(r, [s1, s2])
        r.run_cycle({p1: [], p2: []}, now_msc=s1.bar_time_msc,
                    ask_by_pair={p1: s1.entry, p2: s2.entry},
                    bid_by_pair={p1: s1.entry, p2: s2.entry})
        # Both rejected with same reason.
        rejections = [reason for _, reason in
                       r.result.cycle_reports[-1].rejections]
        assert rejections.count("outside_ist_window") == 2


# ===========================================================================
# 13. Compliance status accessible after cycle
# ===========================================================================

# ===========================================================================
# 14. Per-pair × per-rule kill-switch matrix
# ===========================================================================

class TestPairRuleKillSwitchMatrix:
    @pytest.mark.parametrize("rule_key", ALL_RULE_KEYS)
    @pytest.mark.parametrize("pair", list(PAIRS))
    def test_outside_window_per_pair_per_rule(self, rule_key, pair, runner_factory):
        r = runner_factory(rules_key=rule_key,
                            ist_window_start="02:00",
                            ist_window_end="03:00")
        s = _sig(pair, hour=8)
        _inject(r, [s])
        r.run_cycle({pair: []}, now_msc=s.bar_time_msc,
                    ask_by_pair={pair: s.entry},
                    bid_by_pair={pair: s.entry})
        assert "outside_ist_window" in r.result.all_rejection_reasons


# ===========================================================================
# 15. Safety margin sensitivity — different margins yield different outcomes
# ===========================================================================

class TestSafetyMarginSensitivity:
    @pytest.mark.parametrize("margin", [0.5, 0.7, 0.8, 0.9, 1.0])
    def test_strict_margin_rejects_sooner(self, margin, runner_factory):
        r = runner_factory(safety_margin_pct=margin,
                            starting_equity=100_000.0)
        rules = r.compliance.rules
        daily_cap = 100_000.0 * (rules.max_daily_loss_pct / 100.0)
        # Place PnL just under the strictest margin boundary.
        loss = -(daily_cap * margin) - 1.0
        acct = r.account_with(daily_pnl_usd=loss)
        s = _sig("EURUSD")
        _inject(r, [s])
        r.run_cycle({"EURUSD": []}, now_msc=s.bar_time_msc,
                    ask_by_pair={"EURUSD": s.entry},
                    bid_by_pair={"EURUSD": s.entry}, account=acct)
        assert "daily_loss_near_cap" in r.result.all_rejection_reasons


class TestComplianceStatus:
    @pytest.mark.parametrize("rule_key", ALL_RULE_KEYS)
    def test_status_report_shape(self, rule_key, runner_factory):
        r = runner_factory(rules_key=rule_key)
        report = r.compliance.get_status_report(
            r.default_account(), hour_msc(2026, 4, 15, 8),
        )
        for k in ("equity", "starting_equity", "daily_pnl_usd",
                  "daily_cap_usd", "trades_today", "max_trades_per_day"):
            assert k in report
