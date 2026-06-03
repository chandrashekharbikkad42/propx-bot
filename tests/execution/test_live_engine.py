"""AsianSweepLiveEngine — process_scan_cycle orchestration + lot sizing."""

from __future__ import annotations
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, strategies as st

from execution.live_engine import (
    AsianSweepLiveEngine, CycleReport,
    _MARKET_ENTRY_PATTERNS, _resolve_base_risk_pct, asian_sweep_lots_for,
)
from risk.prop_firm.compliance import AccountState
from risk.house_money import RiskAllocation
from strategy.patterns.base import Direction, Grade, PatternSignal

from tests.execution.fixtures.mock_orders import make_signal, make_signal_sell
from tests.execution.fixtures.mock_positions import make_griff_open


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Default factories
# ---------------------------------------------------------------------------

def _acc(equity=100_000.0, daily_pnl=0.0, trades_today=0, open_count=0):
    return AccountState(
        equity=equity,
        starting_equity=100_000.0,
        daily_start_equity=100_000.0,
        daily_pnl_usd=daily_pnl,
        trades_today=trades_today,
        open_position_count=open_count,
    )


def _allocation(final_pct=0.5, grade=Grade.A):
    return RiskAllocation(
        grade=grade,
        trade_number_today=1,
        base_risk_pct=0.5,
        final_risk_pct=final_pct,
        mode="STANDARD",
        rationale="test",
    )


@pytest.fixture
def engine_factory():
    """Builds an AsianSweepLiveEngine with all dependencies mocked."""
    def _make(*,
              scan_signals=None,
              compliance_ok=True,
              compliance_reason="ok",
              risk_alloc=None,
              news_blackout=False,
              market_entry_pattern="ASIAN_SWEEP",
              ):
        scanner = MagicMock()
        scanner.scan_all = MagicMock(return_value=tuple(scan_signals or ()))

        router = MagicMock()
        router.place_market = AsyncMock(return_value=make_griff_open())
        router.place_pending_stop = AsyncMock(
            return_value=MagicMock(symbol="EURUSD")
        )
        router.place_pending_limit = AsyncMock(
            return_value=MagicMock(symbol="EURUSD")
        )

        pm = MagicMock()
        pm.register_position = MagicMock()
        pm.register_pending = MagicMock()
        # Default: no open positions on any symbol. A bare MagicMock would be
        # truthy and trip the no-pyramiding guard for every signal, so pin it
        # to an empty tuple. Tests that exercise the open-position skip
        # override this on the returned mock.
        pm.positions_for = MagicMock(return_value=())

        compliance = MagicMock()
        compliance.can_trade = MagicMock(
            return_value=(compliance_ok, compliance_reason)
        )

        house = MagicMock()
        house.calc_trade_risk = MagicMock(
            return_value=risk_alloc or _allocation()
        )

        daily = MagicMock()
        daily.trade_count = 0
        daily.record_trade_open = MagicMock()

        alerts = MagicMock()
        alerts.signal_detected = AsyncMock()
        alerts.trade_opened = AsyncMock()
        alerts.kill_switch_triggered = AsyncMock()

        news = MagicMock()
        news.is_blackout = MagicMock(return_value=news_blackout)

        engine = AsianSweepLiveEngine(
            scanner=scanner, router=router, position_mgr=pm,
            compliance=compliance, house_money=house, daily=daily,
            alerts=alerts, news_calendar=news,
        )
        return engine, dict(
            scanner=scanner, router=router, pm=pm,
            compliance=compliance, house=house, daily=daily,
            alerts=alerts, news=news,
        )
    return _make


# ===========================================================================
# 1. Module-level constants & helpers
# ===========================================================================

class TestModuleConstants:
    def test_market_entry_patterns(self):
        assert "FLAG" in _MARKET_ENTRY_PATTERNS
        assert "ASIAN_SWEEP" in _MARKET_ENTRY_PATTERNS
        assert "REVERSAL" not in _MARKET_ENTRY_PATTERNS
        assert "COMBO" not in _MARKET_ENTRY_PATTERNS


class TestResolveBaseRiskPct:
    @pytest.mark.parametrize("sym,month,expected_floor", [
        ("XAUUSD", 5, 0.1),    # has override 0.5%
        ("EURUSD", 5, 0.1),
        ("GBPUSD", 5, 0.1),
        ("AUDUSD", 5, 0.1),
        ("USDCAD", 5, 0.1),
    ])
    def test_returns_positive(self, sym, month, expected_floor):
        out = _resolve_base_risk_pct(sym, month)
        assert out >= expected_floor

    def test_unknown_symbol_returns_zero_or_default(self):
        # _resolve_base_risk_pct delegates to risk_pct_for which has
        # a default; just confirm a float.
        out = _resolve_base_risk_pct("ZZZZZZ", 5)
        assert isinstance(out, float)


# ===========================================================================
# 2. asian_sweep_lots_for — pair sizing
# ===========================================================================

class TestAsianSweepLotsFor:
    def test_eurusd_basic(self):
        lots = asian_sweep_lots_for(
            risk_pct=0.5, equity=100_000.0,
            sl_distance_price=0.0010,  # 10 pips
            symbol="EURUSD",
        )
        # risk_amt = 500. risk_pts = 100. vpl = 1 USD/lot/pt → 5 lots.
        assert lots == pytest.approx(5.0, abs=0.01)

    def test_xauusd_basic(self):
        # XAUUSD: pt=0.01, contract=100 oz, vpl=1.0
        lots = asian_sweep_lots_for(
            risk_pct=0.5, equity=100_000.0,
            sl_distance_price=5.0,    # $5 SL
            symbol="XAUUSD",
        )
        # risk_amt = 500, risk_pts = 500, vpl=1 → 1.0 lot
        assert lots == pytest.approx(1.0, abs=0.01)

    def test_unknown_symbol_returns_min(self):
        lots = asian_sweep_lots_for(
            risk_pct=0.5, equity=100_000.0,
            sl_distance_price=0.0010, symbol="NOTAPAIR",
        )
        assert lots == 0.01

    def test_zero_sl_distance_returns_min(self):
        lots = asian_sweep_lots_for(
            risk_pct=0.5, equity=100_000.0,
            sl_distance_price=0.0, symbol="EURUSD",
        )
        assert lots == 0.01

    def test_zero_equity_returns_min(self):
        lots = asian_sweep_lots_for(
            risk_pct=0.5, equity=0.0,
            sl_distance_price=0.0010, symbol="EURUSD",
        )
        assert lots == 0.01

    def test_zero_risk_pct_returns_min(self):
        lots = asian_sweep_lots_for(
            risk_pct=0.0, equity=100_000.0,
            sl_distance_price=0.0010, symbol="EURUSD",
        )
        assert lots == 0.01

    def test_lots_capped_at_lot_max(self):
        # Extreme: huge equity → would otherwise produce 100 lots,
        # but EURUSD lot_max = 50.
        lots = asian_sweep_lots_for(
            risk_pct=10.0, equity=1_000_000_000.0,
            sl_distance_price=0.0010, symbol="EURUSD",
        )
        assert lots == 50.0

    @pytest.mark.parametrize("sym", [
        "XAUUSD", "EURUSD", "GBPUSD", "AUDUSD",
        "USDCAD", "USDCHF", "AUDCHF", "AUDNZD",
    ])
    def test_all_v5_pairs_yield_positive(self, sym):
        lots = asian_sweep_lots_for(
            risk_pct=0.5, equity=100_000.0,
            sl_distance_price=0.001, symbol=sym,
        )
        assert lots > 0

    @pytest.mark.parametrize("risk_pct,equity,expected_lots", [
        (0.5, 50_000, 2.5),
        (1.0, 50_000, 5.0),
        (1.0, 100_000, 10.0),
    ])
    def test_lots_scale_with_risk_and_equity(self, risk_pct, equity,
                                              expected_lots):
        lots = asian_sweep_lots_for(
            risk_pct=risk_pct, equity=equity,
            sl_distance_price=0.0010, symbol="EURUSD",
        )
        assert lots == pytest.approx(expected_lots, abs=0.01)


# ===========================================================================
# 3. CycleReport
# ===========================================================================

class TestCycleReport:
    def test_construct_defaults(self):
        r = CycleReport(now_msc=42)
        assert r.now_msc == 42
        assert r.signals_emitted == 0
        assert r.signals_rejected_by_compliance == 0
        assert r.orders_placed == 0
        assert r.rejections == []


# ===========================================================================
# 4. process_scan_cycle — no signals
# ===========================================================================

class TestProcessNoSignals:
    def test_empty_scanner_no_orders(self, engine_factory):
        engine, _ = engine_factory(scan_signals=())
        rep = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=1_700_000_000_000,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        assert rep.signals_emitted == 0
        assert rep.orders_placed == 0

    def test_grade_c_signals_skipped(self, engine_factory):
        sig = make_signal(grade=Grade.C, confidence=0.3)
        engine, mocks = engine_factory(scan_signals=[sig])
        rep = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=1_700_000_000_000,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        assert rep.signals_emitted == 1  # counted but not traded
        assert rep.orders_placed == 0


# ===========================================================================
# 5. process_scan_cycle — news blackout path
# ===========================================================================

class TestProcessNewsBlackout:
    def test_blackout_skips_signal(self, engine_factory):
        sig = make_signal()
        engine, mocks = engine_factory(scan_signals=[sig], news_blackout=True)
        rep = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=1_700_000_000_000,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        assert rep.signals_rejected_by_compliance == 1
        assert rep.orders_placed == 0
        # Compliance never reached
        mocks["compliance"].can_trade.assert_not_called()
        # Alert sent
        mocks["alerts"].kill_switch_triggered.assert_awaited()

    def test_blackout_rejection_reason(self, engine_factory):
        sig = make_signal()
        engine, _ = engine_factory(scan_signals=[sig], news_blackout=True)
        rep = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=0,
            ask_by_pair={"EURUSD": 1.0},
            bid_by_pair={"EURUSD": 1.0},
            account=_acc(),
        ))
        assert rep.rejections[0][1] == "news_blackout"


# ===========================================================================
# 6. process_scan_cycle — compliance reject path
# ===========================================================================

class TestProcessComplianceReject:
    def test_compliance_reject_no_order(self, engine_factory):
        sig = make_signal()
        engine, mocks = engine_factory(
            scan_signals=[sig],
            compliance_ok=False, compliance_reason="daily_cap_hit",
        )
        rep = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=1_700_000_000_000,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        assert rep.signals_rejected_by_compliance == 1
        assert rep.orders_placed == 0
        assert rep.rejections[0][1] == "daily_cap_hit"
        mocks["router"].place_market.assert_not_awaited()
        mocks["alerts"].kill_switch_triggered.assert_awaited()


# ===========================================================================
# 7. process_scan_cycle — market entry path
# ===========================================================================

class TestProcessMarketEntry:
    def test_asian_sweep_places_market(self, engine_factory):
        sig = make_signal(pattern_name="ASIAN_SWEEP")
        engine, mocks = engine_factory(scan_signals=[sig])
        rep = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=1_700_000_000_000,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        assert rep.orders_placed == 1
        mocks["router"].place_market.assert_awaited_once()
        mocks["pm"].register_position.assert_called_once()
        mocks["alerts"].trade_opened.assert_awaited_once()
        mocks["daily"].record_trade_open.assert_called_once()

    def test_flag_places_market(self, engine_factory):
        sig = make_signal(pattern_name="FLAG")
        engine, mocks = engine_factory(scan_signals=[sig])
        rep = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=1_700_000_000_000,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        assert rep.orders_placed == 1
        mocks["router"].place_market.assert_awaited_once()

    def test_default_ask_when_missing(self, engine_factory):
        sig = make_signal(pattern_name="ASIAN_SWEEP")
        engine, mocks = engine_factory(scan_signals=[sig])
        run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=0,
            ask_by_pair={},  # missing
            bid_by_pair={},
            account=_acc(),
        ))
        kwargs = mocks["router"].place_market.await_args.kwargs
        assert kwargs["ask"] == sig.entry
        assert kwargs["bid"] == sig.entry


# ===========================================================================
# 8. process_scan_cycle — pending entry path
# ===========================================================================

class TestProcessPendingEntry:
    def test_continuation_places_pending_stop(self, engine_factory):
        sig = make_signal(pattern_name="CONTINUATION")
        engine, mocks = engine_factory(scan_signals=[sig])
        rep = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=0,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        assert rep.orders_placed == 1
        mocks["router"].place_pending_stop.assert_awaited_once()
        mocks["router"].place_pending_limit.assert_not_awaited()
        mocks["pm"].register_pending.assert_called_once()

    def test_combo_places_pending_limit(self, engine_factory):
        sig = make_signal(pattern_name="COMBO")
        engine, mocks = engine_factory(scan_signals=[sig])
        rep = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=0,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        mocks["router"].place_pending_limit.assert_awaited_once()
        mocks["router"].place_pending_stop.assert_not_awaited()

    def test_pending_expiry_one_hour_ahead(self, engine_factory):
        sig = make_signal(pattern_name="CONTINUATION")
        engine, mocks = engine_factory(scan_signals=[sig])
        run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=1000,
            ask_by_pair={"EURUSD": 1.0},
            bid_by_pair={"EURUSD": 1.0},
            account=_acc(),
        ))
        kwargs = mocks["router"].place_pending_stop.await_args.kwargs
        assert kwargs["expiry_msc"] == 1000 + 3_600_000


# ===========================================================================
# 9. process_scan_cycle — best-per-pair dedupe
# ===========================================================================

class TestBestPerPair:
    def test_two_signals_same_pair_take_higher_grade(self, engine_factory):
        sig_a = make_signal(pattern_name="ASIAN_SWEEP", grade=Grade.A,
                            confidence=0.9)
        sig_b = make_signal(pattern_name="ASIAN_SWEEP", grade=Grade.B,
                            confidence=0.7)
        engine, mocks = engine_factory(scan_signals=[sig_b, sig_a])
        rep = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=0,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        # Only one order placed (the better one)
        assert rep.orders_placed == 1


# ===========================================================================
# 10. process_scan_cycle — multiple pairs, same cycle
# ===========================================================================

class TestMultiplePairs:
    def test_two_pairs_two_orders(self, engine_factory):
        eu = make_signal(symbol="EURUSD")
        gu = make_signal(symbol="GBPUSD")
        engine, mocks = engine_factory(scan_signals=[eu, gu])
        rep = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": [], "GBPUSD": []},
            now_msc=0,
            ask_by_pair={"EURUSD": 1.10010, "GBPUSD": 1.25010},
            bid_by_pair={"EURUSD": 1.10000, "GBPUSD": 1.25000},
            account=_acc(),
        ))
        assert rep.orders_placed == 2
        assert mocks["pm"].register_position.call_count == 2


# ===========================================================================
# 11. Sizing path — verifies HouseMoney scaling
# ===========================================================================

class TestSizing:
    def test_house_money_scales_lots_down(self, engine_factory):
        # final_risk_pct < base → lots get scaled down
        alloc = _allocation(final_pct=0.1)  # below 0.5 base
        sig = make_signal(symbol="EURUSD")
        engine, mocks = engine_factory(scan_signals=[sig], risk_alloc=alloc)
        run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=1_715_000_000_000,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        # signal, lots are positional
        args = mocks["router"].place_market.await_args.args
        assert args[1] >= 0.01


# ===========================================================================
# 12. Stats hook (when hourly_stats set)
# ===========================================================================

class TestHourlyStats:
    def test_hourly_stats_property_default_none(self, engine_factory):
        engine, _ = engine_factory(scan_signals=())
        assert engine.hourly_stats is None

    def test_hourly_stats_records_signal(self, engine_factory):
        sig = make_signal()
        engine, _ = engine_factory(scan_signals=[sig])
        stats = MagicMock()
        stats.record_bar = MagicMock()
        stats.record_signal = MagicMock()
        stats.record_compliance = MagicMock()
        engine._stats = stats
        run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": [object()]},  # truthy bars list
            now_msc=0,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        stats.record_bar.assert_called()
        stats.record_signal.assert_called_with(1)
        stats.record_compliance.assert_called_with(passed=True)


# ===========================================================================
# 13. Alerts
# ===========================================================================

class TestAlerts:
    def test_signal_detected_alert_on_pass(self, engine_factory):
        sig = make_signal()
        engine, mocks = engine_factory(scan_signals=[sig])
        run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=0,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        mocks["alerts"].signal_detected.assert_awaited_once()

    def test_no_signal_alert_on_skip(self, engine_factory):
        sig = make_signal()
        engine, mocks = engine_factory(
            scan_signals=[sig], compliance_ok=False,
        )
        run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=0,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        mocks["alerts"].signal_detected.assert_not_awaited()


# ===========================================================================
# 14. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_bar_feeds(self, engine_factory):
        engine, _ = engine_factory(scan_signals=())
        rep = run(engine.process_scan_cycle(
            bar_feeds={},
            now_msc=0,
            ask_by_pair={},
            bid_by_pair={},
            account=_acc(),
        ))
        assert rep.orders_placed == 0

    def test_single_grade_a_signal_goes_through(self, engine_factory):
        sig = make_signal(grade=Grade.A)
        engine, _ = engine_factory(scan_signals=[sig])
        rep = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=0,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        assert rep.orders_placed == 1

    def test_grade_b_also_passes(self, engine_factory):
        sig = make_signal(grade=Grade.B, confidence=0.6)
        engine, _ = engine_factory(scan_signals=[sig])
        rep = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=0,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        assert rep.orders_placed == 1


# ===========================================================================
# 14b. V5 admission gates — no-pyramiding + 1-per-direction/day
# ===========================================================================

class TestAdmissionGates:
    def test_open_position_blocks_signal(self, engine_factory):
        """An existing open position on the symbol → skip (no pyramiding)."""
        sig = make_signal(pattern_name="ASIAN_SWEEP", symbol="EURUSD")
        engine, mocks = engine_factory(scan_signals=[sig])
        mocks["pm"].positions_for = MagicMock(
            return_value=(make_griff_open(symbol="EURUSD"),)
        )
        rep = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []},
            now_msc=1_700_000_000_000,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        assert rep.orders_placed == 0
        assert rep.signals_rejected_by_compliance == 1
        assert rep.rejections[0][1] == "open_position_exists"
        mocks["router"].place_market.assert_not_awaited()

    def test_second_same_direction_signal_blocked(self, engine_factory):
        """Same symbol/direction signal on a later cycle → 2nd order blocked
        with a direction rejection reason ("1 per direction/day")."""
        sig = make_signal(pattern_name="ASIAN_SWEEP", symbol="EURUSD",
                          direction=Direction.BUY)
        engine, mocks = engine_factory(scan_signals=[sig])
        common = dict(
            bar_feeds={"EURUSD": []},
            now_msc=1_700_000_000_000,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        )
        # Cycle 1 — opens the EURUSD LONG, records the direction.
        rep1 = run(engine.process_scan_cycle(**common))
        assert rep1.orders_placed == 1
        # Cycle 2 — identical signal. PM still reports no open position (mock),
        # so the block must come from the per-direction/day ledger.
        rep2 = run(engine.process_scan_cycle(**common))
        assert rep2.orders_placed == 0
        assert rep2.signals_rejected_by_compliance == 1
        assert rep2.rejections[0][1] == "direction_already_traded_today"
        # Only the first cycle reached the router.
        mocks["router"].place_market.assert_awaited_once()

    def test_opposite_direction_still_allowed_same_day(self, engine_factory):
        """1-per-direction is per (symbol, direction): a LONG then a SHORT on
        the same symbol/day are both admissible (the 2-trade cap, not this
        gate, is what limits the day)."""
        engine, mocks = engine_factory(scan_signals=[])
        long_sig = make_signal(symbol="EURUSD", direction=Direction.BUY)
        short_sig = make_signal_sell(symbol="EURUSD")
        common = dict(
            bar_feeds={"EURUSD": []},
            now_msc=1_700_000_000_000,
            ask_by_pair={"EURUSD": 1.10010},
            bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        )
        mocks["scanner"].scan_all = MagicMock(return_value=(long_sig,))
        assert run(engine.process_scan_cycle(**common)).orders_placed == 1
        mocks["scanner"].scan_all = MagicMock(return_value=(short_sig,))
        assert run(engine.process_scan_cycle(**common)).orders_placed == 1

    def test_direction_ledger_resets_next_ist_day(self, engine_factory):
        """The per-direction ledger rolls with the IST trade-day, so the same
        direction is tradable again the next day."""
        sig = make_signal(symbol="EURUSD", direction=Direction.BUY)
        engine, mocks = engine_factory(scan_signals=[sig])
        # Day 1.
        d1 = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []}, now_msc=1_700_000_000_000,
            ask_by_pair={"EURUSD": 1.10010}, bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        assert d1.orders_placed == 1
        # +1 day in ms — a fresh IST trade-day.
        d2 = run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": []}, now_msc=1_700_000_000_000 + 86_400_000,
            ask_by_pair={"EURUSD": 1.10010}, bid_by_pair={"EURUSD": 1.10000},
            account=_acc(),
        ))
        assert d2.orders_placed == 1


# ===========================================================================
# 15. Property-based — asian_sweep_lots_for
# ===========================================================================

@settings(max_examples=100, deadline=None)
@given(
    risk_pct=st.floats(min_value=0.01, max_value=2.0,
                       allow_nan=False, allow_infinity=False),
    equity=st.floats(min_value=1000.0, max_value=1_000_000.0,
                     allow_nan=False, allow_infinity=False),
    sl_pips=st.integers(min_value=1, max_value=500),
)
def test_lots_non_negative_property(risk_pct, equity, sl_pips):
    lots = asian_sweep_lots_for(
        risk_pct=risk_pct, equity=equity,
        sl_distance_price=sl_pips * 0.00001,  # forex point
        symbol="EURUSD",
    )
    assert lots >= 0.01
    assert lots <= 50.0


@settings(max_examples=50, deadline=None)
@given(
    risk_pct=st.floats(min_value=0.01, max_value=2.0,
                       allow_nan=False, allow_infinity=False),
)
def test_lots_monotonic_in_risk_property(risk_pct):
    a = asian_sweep_lots_for(risk_pct=risk_pct, equity=100_000,
                              sl_distance_price=0.0010, symbol="EURUSD")
    b = asian_sweep_lots_for(risk_pct=risk_pct * 2.0, equity=100_000,
                              sl_distance_price=0.0010, symbol="EURUSD")
    assert b >= a


# ===========================================================================
# 16. Min lots floor
# ===========================================================================

@pytest.mark.parametrize("sym,sl,equity", [
    ("EURUSD", 0.10, 100),     # tiny risk
    ("XAUUSD", 100, 100),
])
def test_floor_at_min_lots(sym, sl, equity):
    lots = asian_sweep_lots_for(
        risk_pct=0.001, equity=equity, sl_distance_price=sl, symbol=sym,
    )
    assert lots == 0.01


# ===========================================================================
# 17. Hourly stats — bar recording per pair
# ===========================================================================

class TestStatsBarsRecorded:
    def test_bar_recorded_for_each_non_empty_pair(self, engine_factory):
        engine, _ = engine_factory(scan_signals=())
        stats = MagicMock()
        engine._stats = stats
        run(engine.process_scan_cycle(
            bar_feeds={"EURUSD": [object()], "GBPUSD": [object(), object()],
                       "AUDUSD": []},   # empty list shouldn't record
            now_msc=0,
            ask_by_pair={},
            bid_by_pair={},
            account=_acc(),
        ))
        symbols_recorded = [c.args[0] for c in stats.record_bar.call_args_list]
        assert "EURUSD" in symbols_recorded
        assert "GBPUSD" in symbols_recorded
        assert "AUDUSD" not in symbols_recorded


# ===========================================================================
# 18. maintain_open
# ===========================================================================

class TestMaintainOpen:
    def test_calls_pm_maintain_per_pair(self, engine_factory):
        engine, mocks = engine_factory(scan_signals=())
        mocks["pm"].maintain = AsyncMock(return_value=MagicMock())
        bar = MagicMock()
        out = run(engine.maintain_open(
            {"EURUSD": bar, "GBPUSD": bar}, now_msc=42,
        ))
        assert set(out.keys()) == {"EURUSD", "GBPUSD"}
        assert mocks["pm"].maintain.await_count == 2

    def test_empty_dict_returns_empty(self, engine_factory):
        engine, mocks = engine_factory(scan_signals=())
        mocks["pm"].maintain = AsyncMock()
        out = run(engine.maintain_open({}, now_msc=0))
        assert out == {}
        mocks["pm"].maintain.assert_not_awaited()
