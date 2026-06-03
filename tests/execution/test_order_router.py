"""GriffOrderRouter — issue MT5 orders. Dry-run + mocked-MT5 happy paths."""

from __future__ import annotations
import asyncio

import pytest

from execution.order_router import (
    COMMENT, DEFAULT_DEVIATION_POINTS, MAGIC, MAX_RETRIES,
    GriffOpenPosition, GriffOrderError, GriffOrderRouter,
    GriffPendingOrder, _ticket_from_result,
)
from strategy.patterns.base import Direction

from tests.execution.fixtures.mock_mt5 import (
    OrderSendResult, TRADE_RETCODE_DONE, TRADE_RETCODE_REQUOTE,
    TRADE_RETCODE_REJECT, TRADE_RETCODE_MARKET_CLOSED,
    TRADE_RETCODE_INVALID_STOPS, TRADE_RETCODE_NO_MONEY,
    TRADE_RETCODE_ORDER_NOT_FOUND,
)
from tests.execution.fixtures.mock_orders import make_signal, make_signal_sell
from tests.execution.fixtures.mock_positions import make_griff_open, make_griff_pending


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    """Make `asyncio.sleep` in the router a no-op so retry tests don't wait."""
    import execution.order_router as gor
    async def _no_sleep(*_a, **_k):
        return None
    monkeypatch.setattr(gor.asyncio, "sleep", _no_sleep)


# ===========================================================================
# 1. Constructor
# ===========================================================================

class TestConstructor:
    def test_defaults(self):
        r = GriffOrderRouter()
        assert r.dry_run is True
        assert r._deviation == DEFAULT_DEVIATION_POINTS
        assert r._magic == MAGIC

    @pytest.mark.parametrize("dry", [True, False])
    def test_dry_run_param(self, dry):
        assert GriffOrderRouter(dry_run=dry).dry_run is dry

    @pytest.mark.parametrize("dev", [1, 5, 20, 100])
    def test_deviation(self, dev):
        assert GriffOrderRouter(deviation_points=dev)._deviation == dev

    @pytest.mark.parametrize("magic", [786543, 786544, 99999])
    def test_magic(self, magic):
        assert GriffOrderRouter(magic=magic)._magic == magic


# ===========================================================================
# 2. _ticket_from_result helper
# ===========================================================================

class TestTicketHelper:
    def test_order_field_used(self):
        r = OrderSendResult(order=42, deal=0)
        assert _ticket_from_result(r) == 42

    def test_deal_falls_back(self):
        r = OrderSendResult(order=0, deal=99)
        assert _ticket_from_result(r) == 99

    def test_neither_returns_zero(self):
        r = OrderSendResult(order=0, deal=0)
        assert _ticket_from_result(r) == 0


# ===========================================================================
# 3. place_market — dry_run
# ===========================================================================

class TestPlaceMarketDry:
    def test_buy_uses_ask(self):
        r = GriffOrderRouter(dry_run=True)
        s = make_signal()
        pos = run(r.place_market(s, lots=0.1, ask=1.10010, bid=1.10000,
                                  now_msc=1000))
        assert pos.entry_price == 1.10010
        assert pos.mt5_ticket == -1
        assert isinstance(pos, GriffOpenPosition)

    def test_sell_uses_bid(self):
        r = GriffOrderRouter(dry_run=True)
        s = make_signal_sell()
        pos = run(r.place_market(s, lots=0.1, ask=1.10010, bid=1.10000,
                                  now_msc=1000))
        assert pos.entry_price == 1.10000

    def test_dry_signal_id_contains_pattern(self):
        r = GriffOrderRouter(dry_run=True)
        s = make_signal(pattern_name="ASIAN_SWEEP")
        pos = run(r.place_market(s, lots=0.1, ask=1.0, bid=1.0, now_msc=0))
        assert "ASIAN_SWEEP" in pos.signal_id

    def test_dry_position_id_unique(self):
        r = GriffOrderRouter(dry_run=True)
        s1 = make_signal(bar_time_msc=1_700_000_000_000)
        s2 = make_signal(bar_time_msc=1_700_000_060_000)
        a = run(r.place_market(s1, lots=0.1, ask=1.0, bid=1.0, now_msc=0))
        b = run(r.place_market(s2, lots=0.1, ask=1.0, bid=1.0, now_msc=0))
        assert a.position_id != b.position_id

    def test_dry_pattern_pass_through(self):
        r = GriffOrderRouter(dry_run=True)
        for pname in ("FLAG", "ASIAN_SWEEP", "CONTINUATION"):
            s = make_signal(pattern_name=pname)
            pos = run(r.place_market(s, lots=0.1, ask=1.0, bid=1.0, now_msc=0))
            assert pos.pattern_name == pname

    @pytest.mark.parametrize("lots", [0.01, 0.10, 1.0, 5.0])
    def test_lots_pass_through(self, lots):
        r = GriffOrderRouter(dry_run=True)
        pos = run(r.place_market(make_signal(), lots=lots, ask=1.0, bid=1.0,
                                  now_msc=0))
        assert pos.lots == lots

    def test_dry_opened_msc(self):
        r = GriffOrderRouter(dry_run=True)
        pos = run(r.place_market(make_signal(), lots=0.1, ask=1.0, bid=1.0,
                                  now_msc=999_999))
        assert pos.opened_msc == 999_999

    def test_dry_sl_tp_from_signal(self):
        r = GriffOrderRouter(dry_run=True)
        s = make_signal(sl=1.09800, tp=1.10400)
        pos = run(r.place_market(s, lots=0.1, ask=1.0, bid=1.0, now_msc=0))
        assert pos.sl_price == 1.09800
        assert pos.tp_price == 1.10400


# ===========================================================================
# 4. place_market — mocked MT5 happy path
# ===========================================================================

class TestPlaceMarketMocked:
    def test_buy_sends_correct_request(self, patch_router_mt5):
        m = patch_router_mt5
        r = GriffOrderRouter(dry_run=False)
        s = make_signal()
        run(r.place_market(s, lots=0.1, ask=1.10010, bid=1.10000, now_msc=0))
        req = m.sent_requests[-1]
        assert req["action"] == m.TRADE_ACTION_DEAL
        assert req["type"] == m.ORDER_TYPE_BUY
        assert req["price"] == 1.10010
        assert req["sl"] == s.sl
        assert req["tp"] == s.tp
        assert req["magic"] == MAGIC

    def test_sell_sends_correct_request(self, patch_router_mt5):
        m = patch_router_mt5
        r = GriffOrderRouter(dry_run=False)
        s = make_signal_sell()
        run(r.place_market(s, lots=0.1, ask=1.10010, bid=1.10000, now_msc=0))
        req = m.sent_requests[-1]
        assert req["type"] == m.ORDER_TYPE_SELL
        assert req["price"] == 1.10000

    def test_position_returned(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, order=12345, price=1.10005)
        r = GriffOrderRouter(dry_run=False)
        pos = run(r.place_market(make_signal(), lots=0.1,
                                  ask=1.10010, bid=1.10000, now_msc=42))
        assert pos.mt5_ticket == 12345
        assert pos.entry_price == 1.10005
        assert pos.opened_msc == 42


# ===========================================================================
# 5. place_market — non-transient retcode raises
# ===========================================================================

class TestPlaceMarketReject:
    @pytest.mark.parametrize("retcode", [
        TRADE_RETCODE_INVALID_STOPS,
        TRADE_RETCODE_NO_MONEY,
        99999,  # unknown
    ])
    def test_permanent_reject_raises(self, patch_router_mt5, retcode):
        m = patch_router_mt5
        m.queue_retcodes(retcode)
        r = GriffOrderRouter(dry_run=False)
        with pytest.raises(GriffOrderError, match="permanent reject"):
            run(r.place_market(make_signal(), lots=0.1,
                                ask=1.0, bid=1.0, now_msc=0))

    def test_transient_retried(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_retcodes(TRADE_RETCODE_REQUOTE,
                         TRADE_RETCODE_REQUOTE,
                         TRADE_RETCODE_DONE)
        r = GriffOrderRouter(dry_run=False)
        pos = run(r.place_market(make_signal(), lots=0.1,
                                  ask=1.0, bid=1.0, now_msc=0))
        assert isinstance(pos, GriffOpenPosition)
        assert len(m.sent_requests) == 3

    def test_exhausts_retries_raises(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_retcodes(*([TRADE_RETCODE_REQUOTE] * MAX_RETRIES))
        r = GriffOrderRouter(dry_run=False)
        with pytest.raises(GriffOrderError, match="exhausted retries"):
            run(r.place_market(make_signal(), lots=0.1,
                                ask=1.0, bid=1.0, now_msc=0))


# ===========================================================================
# 6. place_pending_stop — dry & mocked
# ===========================================================================

class TestPlacePendingStop:
    def test_dry_buy_stop(self):
        r = GriffOrderRouter(dry_run=True)
        s = make_signal(pattern_name="CONTINUATION")
        out = run(r.place_pending_stop(s, lots=0.1, expiry_msc=999_999,
                                        now_msc=0))
        assert isinstance(out, GriffPendingOrder)
        assert out.is_limit is False
        assert out.mt5_ticket == -1
        assert out.pending_price == s.entry
        assert out.expiry_msc == 999_999

    def test_dry_sell_stop(self):
        r = GriffOrderRouter(dry_run=True)
        s = make_signal_sell(pattern_name="REVERSAL")
        out = run(r.place_pending_stop(s, lots=0.1, expiry_msc=999_999,
                                        now_msc=0))
        assert out.is_limit is False
        assert out.side == Direction.SELL

    def test_mocked_buy_stop_request(self, patch_router_mt5):
        m = patch_router_mt5
        r = GriffOrderRouter(dry_run=False)
        s = make_signal(pattern_name="CONTINUATION")
        run(r.place_pending_stop(s, lots=0.1, expiry_msc=1_700_000_000_000,
                                  now_msc=0))
        req = m.sent_requests[-1]
        assert req["action"] == m.TRADE_ACTION_PENDING
        assert req["type"] == m.ORDER_TYPE_BUY_STOP
        # MT5 wants seconds, not ms
        assert req["expiration"] == 1_700_000_000

    def test_mocked_sell_stop_request(self, patch_router_mt5):
        m = patch_router_mt5
        r = GriffOrderRouter(dry_run=False)
        s = make_signal_sell()
        run(r.place_pending_stop(s, lots=0.1, expiry_msc=1_700_000_000_000,
                                  now_msc=0))
        req = m.sent_requests[-1]
        assert req["type"] == m.ORDER_TYPE_SELL_STOP


# ===========================================================================
# 7. place_pending_limit — dry & mocked
# ===========================================================================

class TestPlacePendingLimit:
    def test_dry_buy_limit(self):
        r = GriffOrderRouter(dry_run=True)
        s = make_signal(pattern_name="COMBO")
        out = run(r.place_pending_limit(s, lots=0.1, expiry_msc=999,
                                         now_msc=0))
        assert out.is_limit is True
        assert out.side == Direction.BUY

    def test_dry_sell_limit(self):
        r = GriffOrderRouter(dry_run=True)
        s = make_signal_sell(pattern_name="COMBO")
        out = run(r.place_pending_limit(s, lots=0.1, expiry_msc=999,
                                         now_msc=0))
        assert out.is_limit is True
        assert out.side == Direction.SELL

    def test_mocked_buy_limit_request(self, patch_router_mt5):
        m = patch_router_mt5
        r = GriffOrderRouter(dry_run=False)
        s = make_signal()
        run(r.place_pending_limit(s, lots=0.1, expiry_msc=1_000,
                                   now_msc=0))
        req = m.sent_requests[-1]
        assert req["type"] == m.ORDER_TYPE_BUY_LIMIT

    def test_mocked_sell_limit_request(self, patch_router_mt5):
        m = patch_router_mt5
        r = GriffOrderRouter(dry_run=False)
        s = make_signal_sell()
        run(r.place_pending_limit(s, lots=0.1, expiry_msc=1_000,
                                   now_msc=0))
        req = m.sent_requests[-1]
        assert req["type"] == m.ORDER_TYPE_SELL_LIMIT


# ===========================================================================
# 8. cancel_pending
# ===========================================================================

class TestCancelPending:
    def test_dry_run_returns_true(self):
        r = GriffOrderRouter(dry_run=True)
        order = make_griff_pending()
        assert run(r.cancel_pending(order)) is True

    def test_mocked_success(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_retcodes(TRADE_RETCODE_DONE)
        r = GriffOrderRouter(dry_run=False)
        assert run(r.cancel_pending(make_griff_pending())) is True
        req = m.sent_requests[-1]
        assert req["action"] == m.TRADE_ACTION_REMOVE

    def test_already_gone_returns_true(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_retcodes(TRADE_RETCODE_ORDER_NOT_FOUND)
        r = GriffOrderRouter(dry_run=False)
        assert run(r.cancel_pending(make_griff_pending())) is True

    def test_unknown_failure_returns_false(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_retcodes(TRADE_RETCODE_REJECT)
        r = GriffOrderRouter(dry_run=False)
        assert run(r.cancel_pending(make_griff_pending())) is False


# ===========================================================================
# 9. close_position
# ===========================================================================

class TestClosePosition:
    def test_dry_buy_uses_bid(self):
        r = GriffOrderRouter(dry_run=True)
        pos = make_griff_open(side=Direction.BUY)
        out = run(r.close_position(pos, bid=1.10000, ask=1.10010, now_msc=0))
        assert out == 1.10000

    def test_dry_sell_uses_ask(self):
        r = GriffOrderRouter(dry_run=True)
        pos = make_griff_open(side=Direction.SELL)
        out = run(r.close_position(pos, bid=1.10000, ask=1.10010, now_msc=0))
        assert out == 1.10010

    def test_mocked_request(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, price=1.10000)
        r = GriffOrderRouter(dry_run=False)
        pos = make_griff_open(side=Direction.BUY, mt5_ticket=777)
        out = run(r.close_position(pos, bid=1.10000, ask=1.10010, now_msc=0))
        req = m.sent_requests[-1]
        assert req["action"] == m.TRADE_ACTION_DEAL
        assert req["position"] == 777
        assert req["type"] == m.ORDER_TYPE_SELL  # close BUY → sell
        assert out == 1.10000


# ===========================================================================
# 10. modify_sl
# ===========================================================================

class TestModifySl:
    def test_dry_run_returns_true(self):
        r = GriffOrderRouter(dry_run=True)
        pos = make_griff_open()
        assert run(r.modify_sl(pos, new_sl=1.09000)) is True

    def test_mocked_success(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_retcodes(TRADE_RETCODE_DONE)
        r = GriffOrderRouter(dry_run=False)
        pos = make_griff_open(mt5_ticket=42)
        out = run(r.modify_sl(pos, new_sl=1.09500))
        assert out is True
        req = m.sent_requests[-1]
        assert req["action"] == m.TRADE_ACTION_SLTP
        assert req["sl"] == 1.09500
        assert req["position"] == 42
        assert req["tp"] == pos.tp_price

    def test_mocked_failure(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_retcodes(TRADE_RETCODE_REJECT)
        r = GriffOrderRouter(dry_run=False)
        assert run(r.modify_sl(make_griff_open(), new_sl=1.0)) is False


# ===========================================================================
# 11. dry_run vs not — observable diff
# ===========================================================================

class TestDryVsLive:
    def test_dry_no_mt5_calls(self, patch_router_mt5):
        m = patch_router_mt5
        r = GriffOrderRouter(dry_run=True)
        run(r.place_market(make_signal(), lots=0.1,
                            ask=1.0, bid=1.0, now_msc=0))
        assert len(m.sent_requests) == 0

    def test_live_sends_one_request(self, patch_router_mt5):
        m = patch_router_mt5
        r = GriffOrderRouter(dry_run=False)
        run(r.place_market(make_signal(), lots=0.1,
                            ask=1.0, bid=1.0, now_msc=0))
        assert len(m.sent_requests) == 1


# ===========================================================================
# 12. Per-pattern smoke
# ===========================================================================

@pytest.mark.parametrize("pname", [
    "FLAG", "ASIAN_SWEEP", "CONTINUATION", "REVERSAL", "COMBO",
])
def test_dry_market_any_pattern(pname):
    r = GriffOrderRouter(dry_run=True)
    pos = run(r.place_market(make_signal(pattern_name=pname),
                              lots=0.1, ask=1.0, bid=1.0, now_msc=0))
    assert pos.pattern_name == pname


@pytest.mark.parametrize("pname", [
    "CONTINUATION", "REVERSAL",
])
def test_dry_stop_per_pattern(pname):
    r = GriffOrderRouter(dry_run=True)
    out = run(r.place_pending_stop(make_signal(pattern_name=pname),
                                     lots=0.1, expiry_msc=999,
                                     now_msc=0))
    assert out.pattern_name == pname
    assert out.is_limit is False


# ===========================================================================
# 13. Per-side smoke
# ===========================================================================

@pytest.mark.parametrize("side", [Direction.BUY, Direction.SELL])
def test_market_either_side(side, patch_router_mt5):
    r = GriffOrderRouter(dry_run=False)
    if side == Direction.BUY:
        s = make_signal()
    else:
        s = make_signal_sell()
    pos = run(r.place_market(s, lots=0.1, ask=1.10010, bid=1.10000, now_msc=0))
    assert pos.side == side


# ===========================================================================
# 14. Retry edge cases
# ===========================================================================

class TestRetryEdge:
    @pytest.mark.parametrize("retcode", [
        TRADE_RETCODE_REQUOTE,
        TRADE_RETCODE_REJECT,
        TRADE_RETCODE_MARKET_CLOSED,
    ])
    def test_retries_then_succeeds(self, patch_router_mt5, retcode):
        m = patch_router_mt5
        m.queue_retcodes(retcode, TRADE_RETCODE_DONE)
        r = GriffOrderRouter(dry_run=False)
        pos = run(r.place_market(make_signal(), lots=0.1,
                                  ask=1.0, bid=1.0, now_msc=0))
        assert isinstance(pos, GriffOpenPosition)
        assert len(m.sent_requests) == 2


# ===========================================================================
# 15. lots float pass-through
# ===========================================================================

@pytest.mark.parametrize("lots", [0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
def test_lots_pass_through_market(lots, patch_router_mt5):
    m = patch_router_mt5
    r = GriffOrderRouter(dry_run=False)
    run(r.place_market(make_signal(), lots=lots, ask=1.0, bid=1.0, now_msc=0))
    req = m.sent_requests[-1]
    assert req["volume"] == lots


@pytest.mark.parametrize("lots", [0.01, 0.05, 0.1])
def test_lots_pass_through_pending(lots, patch_router_mt5):
    m = patch_router_mt5
    r = GriffOrderRouter(dry_run=False)
    run(r.place_pending_stop(make_signal(), lots=lots, expiry_msc=999,
                              now_msc=0))
    req = m.sent_requests[-1]
    assert req["volume"] == lots


# ===========================================================================
# 16. comment + magic
# ===========================================================================

class TestCommentAndMagic:
    def test_market_comment_includes_pattern_name(self, patch_router_mt5):
        m = patch_router_mt5
        r = GriffOrderRouter(dry_run=False)
        run(r.place_market(make_signal(pattern_name="ASIAN_SWEEP"),
                            lots=0.1, ask=1.0, bid=1.0, now_msc=0))
        assert COMMENT in m.sent_requests[-1]["comment"]
        assert "ASIAN_SWEEP" in m.sent_requests[-1]["comment"]

    def test_close_comment_is_close(self, patch_router_mt5):
        m = patch_router_mt5
        m.queue_result(retcode=TRADE_RETCODE_DONE, price=1.0)
        r = GriffOrderRouter(dry_run=False)
        pos = make_griff_open()
        run(r.close_position(pos, bid=1.0, ask=1.0, now_msc=0))
        assert "close" in m.sent_requests[-1]["comment"]

    def test_default_magic_is_griff(self, patch_router_mt5):
        m = patch_router_mt5
        r = GriffOrderRouter(dry_run=False)
        run(r.place_market(make_signal(), lots=0.1, ask=1.0, bid=1.0,
                            now_msc=0))
        assert m.sent_requests[-1]["magic"] == MAGIC == 786544


# ===========================================================================
# 17. Pending expiry as MT5 seconds
# ===========================================================================

class TestExpirySeconds:
    @pytest.mark.parametrize("ms,sec", [
        (1_000, 1),
        (1_500, 1),  # int() truncates
        (1_000_000, 1000),
        (1_700_000_000_000, 1_700_000_000),
    ])
    def test_seconds_conversion(self, patch_router_mt5, ms, sec):
        m = patch_router_mt5
        r = GriffOrderRouter(dry_run=False)
        run(r.place_pending_stop(make_signal(), lots=0.1,
                                  expiry_msc=ms, now_msc=0))
        assert m.sent_requests[-1]["expiration"] == sec
