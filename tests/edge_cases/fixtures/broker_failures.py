"""Adversarial broker stubs — extend MockMT5 with failure-injection helpers.

We don't subclass MockMT5 (its constructor already sets sensible state);
we provide functions that PATCH an existing MockMT5 into the desired
failure mode in-place. Tests pass a fresh `mock_mt5` fixture, then call
``inject_*`` to shape the broker's behavior for that test only.
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple

from tests.execution.fixtures.mock_mt5 import (
    MockMT5, OrderSendResult,
    TRADE_RETCODE_DONE,
    TRADE_RETCODE_REQUOTE, TRADE_RETCODE_REJECT,
    TRADE_RETCODE_PRICE_OFF, TRADE_RETCODE_MARKET_CLOSED,
    TRADE_RETCODE_CONNECTION, TRADE_RETCODE_INVALID_STOPS,
    TRADE_RETCODE_NO_MONEY, TRADE_RETCODE_ORDER_NOT_FOUND,
)


def inject_reject_then_success(mock: MockMT5, n_rejects: int = 1) -> None:
    """Reject the first N order_send calls, then accept."""
    for _ in range(n_rejects):
        mock.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_REJECT))
    mock.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_DONE,
                                              order=mock.next_ticket()))


def inject_requote_loop(mock: MockMT5, n: int = 4) -> None:
    """N consecutive REQUOTE responses — exceeds router MAX_RETRIES=3."""
    for _ in range(n):
        mock.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_REQUOTE))


def inject_partial_fill(
    mock: MockMT5, requested_volume: float, filled_volume: float,
) -> None:
    """order_send returns DONE but with volume < requested.

    Note the router/live-broker code does NOT inspect `result.volume` against
    request volume — partial fills currently slip past. This fixture is used
    to PROBE that gap.
    """
    mock.retcode_queue.append(OrderSendResult(
        retcode=TRADE_RETCODE_DONE,
        order=mock.next_ticket(),
        volume=filled_volume,
    ))


def inject_disconnect(mock: MockMT5) -> None:
    """A single CONNECTION-error response. Router treats as transient → retry."""
    mock.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_CONNECTION))


def inject_permanent_no_money(mock: MockMT5) -> None:
    """NO_MONEY retcode — non-transient → router/live raise immediately."""
    mock.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_NO_MONEY))


def inject_invalid_stops(mock: MockMT5) -> None:
    """INVALID_STOPS — non-transient → router raises immediately."""
    mock.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_INVALID_STOPS))


def inject_order_not_found_on_cancel(mock: MockMT5) -> None:
    """Pending-cancel hits 10027 (already-gone). Router treats as success."""
    mock.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_ORDER_NOT_FOUND))


def inject_none_result(mock: MockMT5) -> None:
    """order_send returns None — only possible if MT5 has internal failure.

    Implemented by swapping the bound method on the mock instance for one
    invocation."""
    original = mock.order_send

    def _once_none(request: dict):  # type: ignore[override]
        mock.order_send = original  # type: ignore[assignment]
        mock.sent_requests.append(dict(request))
        return None

    mock.order_send = _once_none  # type: ignore[assignment]


def inject_market_closed(mock: MockMT5, n: int = 3) -> None:
    """N MARKET_CLOSED responses (transient — router retries)."""
    for _ in range(n):
        mock.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_MARKET_CLOSED))


def inject_price_off(mock: MockMT5) -> None:
    mock.retcode_queue.append(OrderSendResult(retcode=TRADE_RETCODE_PRICE_OFF))


def inject_slippage(
    mock: MockMT5, requested_price: float, fill_price: float,
) -> None:
    """order_send returns DONE but at a different price than requested."""
    mock.retcode_queue.append(OrderSendResult(
        retcode=TRADE_RETCODE_DONE,
        order=mock.next_ticket(),
        price=fill_price,
    ))


def inject_zero_ticket_done(mock: MockMT5) -> None:
    """DONE but with ticket=0 — pathological response.

    MockMT5 auto-replaces ticket=0 with next_ticket() inside order_send, but
    we explicitly clear that behaviour by injecting deal=0 AND order=0 AND
    a *non-zero* price so the auto-assign branch doesn't fire (it only fires
    when DONE arrives with both order==0 and deal==0 — which it does here).
    """
    mock.retcode_queue.append(OrderSendResult(
        retcode=TRADE_RETCODE_DONE, order=0, deal=0, price=1.0, volume=0.01,
    ))


def inject_deal_ticket_only(mock: MockMT5, deal_ticket: int = 12345) -> None:
    """DONE with `deal` populated but `order` = 0 — common for FILL_OR_KILL."""
    mock.retcode_queue.append(OrderSendResult(
        retcode=TRADE_RETCODE_DONE, order=0, deal=deal_ticket,
    ))


def queue_retcode_sequence(mock: MockMT5, retcodes: Iterable[int]) -> None:
    """Generic helper: drop a sequence of raw retcodes onto the queue."""
    for rc in retcodes:
        mock.retcode_queue.append(OrderSendResult(retcode=rc))
