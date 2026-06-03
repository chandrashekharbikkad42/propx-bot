"""MT5 terminal adapter. Only file that imports MetaTrader5 directly."""

from __future__ import annotations
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5
import numpy as np

from config.settings import settings
from utils.logger import logger


@dataclass(frozen=True)
class AccountInfo:
    login: int
    server: str
    balance: float
    equity: float
    leverage: int
    currency: str
    company: str


@dataclass(frozen=True)
class SymbolInfo:
    name: str
    digits: int
    point: float
    spread_points: int
    trade_tick_size: float
    trade_tick_value: float
    contract_size: float


class MT5ConnectionError(RuntimeError):
    pass


class MT5Connector:
    def __init__(self, symbol: Optional[str] = None) -> None:
        self._symbol = symbol or settings.symbol
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            return

        last_err: Optional[Exception] = None
        for attempt in range(1, settings.mt5_retry_attempts + 1):
            try:
                self._init_terminal()
                self._login()
                self._select_symbol()
                self._connected = True
                logger.success(
                    f"MT5 connected | account={settings.mt5_login} "
                    f"server={settings.mt5_server} symbol={self._symbol}"
                )
                return
            except Exception as exc:
                last_err = exc
                logger.warning(
                    f"MT5 connect attempt {attempt}/"
                    f"{settings.mt5_retry_attempts} failed: {exc}"
                )
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                time.sleep(settings.mt5_retry_delay_sec)

        raise MT5ConnectionError(
            f"MT5 connection failed after {settings.mt5_retry_attempts} attempts. "
            f"Last error: {last_err}"
        )

    def disconnect(self) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 terminal shut down")

    def __enter__(self) -> "MT5Connector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    def _init_terminal(self) -> None:
        ok = mt5.initialize(
            path=settings.mt5_path,
            login=settings.mt5_login,
            password=settings.mt5_password,
            server=settings.mt5_server,
            timeout=settings.mt5_timeout_ms,
            portable=False,
        )
        if not ok:
            raise MT5ConnectionError(f"mt5.initialize failed: {mt5.last_error()}")

    def _login(self) -> None:
        ok = mt5.login(
            login=settings.mt5_login,
            password=settings.mt5_password,
            server=settings.mt5_server,
            timeout=settings.mt5_timeout_ms,
        )
        if not ok:
            raise MT5ConnectionError(f"mt5.login failed: {mt5.last_error()}")

    def _select_symbol(self) -> None:
        info = mt5.symbol_info(self._symbol)
        if info is None:
            raise MT5ConnectionError(
                f"Symbol {self._symbol} not found in Market Watch"
            )
        if not info.visible:
            if not mt5.symbol_select(self._symbol, True):
                raise MT5ConnectionError(
                    f"Failed to select symbol {self._symbol}"
                )

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def symbol(self) -> str:
        return self._symbol

    def account_info(self) -> AccountInfo:
        self._require_connected()
        a = mt5.account_info()
        if a is None:
            raise MT5ConnectionError(f"account_info None: {mt5.last_error()}")
        return AccountInfo(
            login=a.login,
            server=a.server,
            balance=a.balance,
            equity=a.equity,
            leverage=a.leverage,
            currency=a.currency,
            company=a.company,
        )

    def symbol_info(self) -> SymbolInfo:
        self._require_connected()
        s = mt5.symbol_info(self._symbol)
        if s is None:
            raise MT5ConnectionError(f"symbol_info None: {mt5.last_error()}")
        return SymbolInfo(
            name=s.name,
            digits=s.digits,
            point=s.point,
            spread_points=s.spread,
            trade_tick_size=s.trade_tick_size,
            trade_tick_value=s.trade_tick_value,
            contract_size=s.trade_contract_size,
        )

    def terminal_info(self) -> dict:
        self._require_connected()
        t = mt5.terminal_info()
        return t._asdict() if t else {}

    def last_tick_msc(self) -> int:
        """Return latest tick timestamp in milliseconds. Used as collector cursor seed."""
        self._require_connected()
        t = mt5.symbol_info_tick(self._symbol)
        if t is None:
            raise MT5ConnectionError(
                f"symbol_info_tick None: {mt5.last_error()}"
            )
        return int(t.time_msc)

    def copy_rates_range(
        self, symbol: str, timeframe: int, date_from: datetime, date_to: datetime
    ) -> np.ndarray:
        """Fetch OHLCV bars for `symbol` in `[date_from, date_to]` (inclusive).

        `timeframe` is an MT5 constant — e.g. `mt5.TIMEFRAME_H1` for 1H bars.
        Returns a structured numpy array with fields:
        time, open, high, low, close, tick_volume, spread, real_volume.

        Phase 8B: used by `scripts/capture_historical_bars.py`. We keep
        symbol as a per-call arg (not connector instance) so the same
        connection can pull data for any pair the broker exposes.
        """
        self._require_connected()
        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
        if rates is None:
            raise MT5ConnectionError(
                f"copy_rates_range None for {symbol}: {mt5.last_error()}"
            )
        return rates

    def copy_ticks_from(self, from_msc: int, count: int) -> np.ndarray:
        """Fetch up to `count` ticks starting at `from_msc` (epoch ms, UTC).

        Returns a structured numpy array with fields:
        time, bid, ask, last, volume, time_msc, flags, volume_real.
        Returns an empty array if no ticks are available.
        """
        self._require_connected()
        if count <= 0:
            return np.empty(0)
        from_dt = datetime.fromtimestamp(from_msc / 1000.0, tz=timezone.utc)
        ticks = mt5.copy_ticks_from(
            self._symbol, from_dt, count, mt5.COPY_TICKS_ALL
        )
        if ticks is None:
            raise MT5ConnectionError(
                f"copy_ticks_from None: {mt5.last_error()}"
            )
        return ticks

    def _require_connected(self) -> None:
        if not self._connected:
            raise MT5ConnectionError("Not connected. Call connect() first.")