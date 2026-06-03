"""In-memory MT5 mock — covers the surface used by live_broker, order_router,
and mt5_connector.

The real `MetaTrader5` module is installed (pip-installed shim on Windows) so
imports succeed, but tests must NEVER call real MT5 functions. Tests inject
this mock by monkey-patching the symbol the module-under-test imported:
  - execution.live_broker.mt5  → MockMT5()
  - execution.order_router.mt5 → MockMT5()
  - data.mt5_connector.mt5 → MockMT5()

Retcodes / constants mirror the real MT5 values where the code-under-test
relies on them. Order side / action constants are typed integers so equality
checks behave correctly in the production code.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants (mirror MT5 — values match the real shim where it matters)
# ---------------------------------------------------------------------------

TRADE_ACTION_DEAL = 1
TRADE_ACTION_PENDING = 5
TRADE_ACTION_SLTP = 6
TRADE_ACTION_REMOVE = 8

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TYPE_BUY_STOP = 4
ORDER_TYPE_SELL_STOP = 5

ORDER_TIME_GTC = 0
ORDER_TIME_SPECIFIED = 1

ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2

# retcodes
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_REQUOTE = 10004
TRADE_RETCODE_REJECT = 10006
TRADE_RETCODE_PRICE_OFF = 10021
TRADE_RETCODE_MARKET_CLOSED = 10018
TRADE_RETCODE_CONNECTION = 10031
TRADE_RETCODE_INVALID_STOPS = 10016
TRADE_RETCODE_NO_MONEY = 10019
TRADE_RETCODE_ORDER_NOT_FOUND = 10027

COPY_TICKS_ALL = 0

TIMEFRAME_H1 = 16385  # value matches the real lib


# ---------------------------------------------------------------------------
# Result/Info structures used by getattr in production code
# ---------------------------------------------------------------------------


@dataclass
class OrderSendResult:
    retcode: int = TRADE_RETCODE_DONE
    order: int = 0
    deal: int = 0
    price: float = 0.0
    volume: float = 0.0
    comment: str = ""
    request_id: int = 0


@dataclass
class SymbolInfoResult:
    name: str = "XAUUSD"
    digits: int = 2
    point: float = 0.01
    spread: int = 20
    trade_tick_size: float = 0.01
    trade_tick_value: float = 1.0
    trade_contract_size: float = 100.0
    visible: bool = True


@dataclass
class AccountInfoResult:
    login: int = 12345
    server: str = "TestServer"
    balance: float = 100_000.0
    equity: float = 100_000.0
    leverage: int = 100
    currency: str = "USD"
    company: str = "TestBroker"


@dataclass
class TickInfoResult:
    bid: float = 1.10000
    ask: float = 1.10010
    last: float = 1.10005
    volume: int = 1
    time_msc: int = 0


@dataclass
class TerminalInfoResult:
    connected: bool = True
    name: str = "MT5"

    def _asdict(self) -> dict:
        return {"connected": self.connected, "name": self.name}


@dataclass
class DealInfo:
    price: float = 0.0
    profit: float = 0.0
    time_msc: int = 0
    type: int = ORDER_TYPE_BUY


@dataclass
class PositionInfo:
    ticket: int = 0
    symbol: str = "XAUUSD"
    volume: float = 0.01
    price_open: float = 0.0


# ---------------------------------------------------------------------------
# The mock module
# ---------------------------------------------------------------------------


class MockMT5:
    """Stand-in for the `MetaTrader5` module.

    Attributes:
      - sent_requests : list of dicts passed to order_send()
      - retcode_queue : queue of OrderSendResult instances returned in FIFO.
                        If empty, order_send returns a default DONE result with
                        ticket=`next_ticket()`.
      - positions     : list of PositionInfo for positions_get to return.
      - deals         : list of DealInfo for history_deals_get to return.
      - initialize_returns / login_returns : configurable booleans.
    """

    # Constants mirrored
    TRADE_ACTION_DEAL = TRADE_ACTION_DEAL
    TRADE_ACTION_PENDING = TRADE_ACTION_PENDING
    TRADE_ACTION_SLTP = TRADE_ACTION_SLTP
    TRADE_ACTION_REMOVE = TRADE_ACTION_REMOVE
    ORDER_TYPE_BUY = ORDER_TYPE_BUY
    ORDER_TYPE_SELL = ORDER_TYPE_SELL
    ORDER_TYPE_BUY_LIMIT = ORDER_TYPE_BUY_LIMIT
    ORDER_TYPE_SELL_LIMIT = ORDER_TYPE_SELL_LIMIT
    ORDER_TYPE_BUY_STOP = ORDER_TYPE_BUY_STOP
    ORDER_TYPE_SELL_STOP = ORDER_TYPE_SELL_STOP
    ORDER_TIME_GTC = ORDER_TIME_GTC
    ORDER_TIME_SPECIFIED = ORDER_TIME_SPECIFIED
    ORDER_FILLING_IOC = ORDER_FILLING_IOC
    ORDER_FILLING_RETURN = ORDER_FILLING_RETURN
    TRADE_RETCODE_DONE = TRADE_RETCODE_DONE
    TRADE_RETCODE_REQUOTE = TRADE_RETCODE_REQUOTE
    TRADE_RETCODE_REJECT = TRADE_RETCODE_REJECT
    TRADE_RETCODE_PRICE_OFF = TRADE_RETCODE_PRICE_OFF
    TRADE_RETCODE_MARKET_CLOSED = TRADE_RETCODE_MARKET_CLOSED
    TRADE_RETCODE_CONNECTION = TRADE_RETCODE_CONNECTION
    TRADE_RETCODE_INVALID_STOPS = TRADE_RETCODE_INVALID_STOPS
    TRADE_RETCODE_NO_MONEY = TRADE_RETCODE_NO_MONEY
    COPY_TICKS_ALL = COPY_TICKS_ALL
    TIMEFRAME_H1 = TIMEFRAME_H1

    def __init__(self) -> None:
        self.sent_requests: List[dict] = []
        self.retcode_queue: List[OrderSendResult] = []
        self.positions: List[PositionInfo] = []
        self.deals: List[DealInfo] = []
        self.account: AccountInfoResult = AccountInfoResult()
        self.symbol_info_obj: Optional[SymbolInfoResult] = SymbolInfoResult()
        self.terminal_info_obj: Optional[TerminalInfoResult] = TerminalInfoResult()
        self.tick_obj: Optional[TickInfoResult] = TickInfoResult()
        self.initialize_returns: bool = True
        self.login_returns: bool = True
        self.symbol_select_returns: bool = True
        self.shutdown_calls: int = 0
        self.copy_rates_result: Any = []
        self.copy_ticks_result: Any = []
        self._ticket_seq: int = 1000
        self._last_error: Tuple[int, str] = (0, "no error")

    # --------------------------------------------------------------- helpers

    def next_ticket(self) -> int:
        self._ticket_seq += 1
        return self._ticket_seq

    def queue_result(self, **kwargs: Any) -> "OrderSendResult":
        """Queue an OrderSendResult to be returned by the NEXT order_send()."""
        r = OrderSendResult(**kwargs)
        self.retcode_queue.append(r)
        return r

    def queue_retcodes(self, *retcodes: int) -> None:
        """Queue raw retcodes in order."""
        for rc in retcodes:
            self.retcode_queue.append(OrderSendResult(retcode=rc))

    def set_last_error(self, code: int, msg: str = "err") -> None:
        self._last_error = (code, msg)

    # ----------------------------------------------------- MT5 API surface

    def initialize(self, *args: Any, **kwargs: Any) -> bool:
        return self.initialize_returns

    def login(self, *args: Any, **kwargs: Any) -> bool:
        return self.login_returns

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def symbol_info(self, name: str) -> Optional[SymbolInfoResult]:
        if self.symbol_info_obj is None:
            return None
        return SymbolInfoResult(
            name=name,
            digits=self.symbol_info_obj.digits,
            point=self.symbol_info_obj.point,
            spread=self.symbol_info_obj.spread,
            trade_tick_size=self.symbol_info_obj.trade_tick_size,
            trade_tick_value=self.symbol_info_obj.trade_tick_value,
            trade_contract_size=self.symbol_info_obj.trade_contract_size,
            visible=self.symbol_info_obj.visible,
        )

    def symbol_select(self, name: str, enable: bool) -> bool:
        return self.symbol_select_returns

    def symbol_info_tick(self, name: str) -> Optional[TickInfoResult]:
        return self.tick_obj

    def account_info(self) -> Optional[AccountInfoResult]:
        return self.account

    def terminal_info(self) -> Optional[TerminalInfoResult]:
        return self.terminal_info_obj

    def order_send(self, request: dict) -> Optional[OrderSendResult]:
        self.sent_requests.append(dict(request))
        if self.retcode_queue:
            r = self.retcode_queue.pop(0)
            # If the queued result has retcode=DONE but no ticket, give it one.
            if r.retcode == TRADE_RETCODE_DONE and r.order == 0 and r.deal == 0:
                r.order = self.next_ticket()
                r.price = float(request.get("price", 0.0))
                r.volume = float(request.get("volume", 0.0))
            return r
        # default: success
        return OrderSendResult(
            retcode=TRADE_RETCODE_DONE,
            order=self.next_ticket(),
            price=float(request.get("price", 0.0)),
            volume=float(request.get("volume", 0.0)),
        )

    def positions_get(self, ticket: Optional[int] = None,
                      symbol: Optional[str] = None) -> Tuple[PositionInfo, ...]:
        if ticket is not None:
            return tuple(p for p in self.positions if p.ticket == ticket)
        if symbol is not None:
            return tuple(p for p in self.positions if p.symbol == symbol)
        return tuple(self.positions)

    def history_deals_get(self, position: Optional[int] = None,
                          **kwargs: Any) -> Tuple[DealInfo, ...]:
        return tuple(self.deals)

    def copy_rates_range(self, symbol: str, timeframe: int,
                         date_from: Any, date_to: Any) -> Any:
        return self.copy_rates_result

    def copy_ticks_from(self, symbol: str, date_from: Any,
                        count: int, flags: int) -> Any:
        return self.copy_ticks_result

    def last_error(self) -> Tuple[int, str]:
        return self._last_error
