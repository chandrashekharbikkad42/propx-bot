"""MT5 connection smoke test."""

from __future__ import annotations
import sys

from data.mt5_connector import MT5Connector, MT5ConnectionError
from utils.logger import logger


def main() -> int:
    logger.info("=" * 60)
    logger.info("MT5 CONNECTION SMOKE TEST")
    logger.info("=" * 60)

    try:
        with MT5Connector() as conn:
            acc = conn.account_info()
            logger.info("--- ACCOUNT ---")
            logger.info(f"  Login    : {acc.login}")
            logger.info(f"  Server   : {acc.server}")
            logger.info(f"  Broker   : {acc.company}")
            logger.info(f"  Currency : {acc.currency}")
            logger.info(f"  Balance  : {acc.balance:.2f}")
            logger.info(f"  Equity   : {acc.equity:.2f}")
            logger.info(f"  Leverage : 1:{acc.leverage}")

            sym = conn.symbol_info()
            logger.info("--- SYMBOL ---")
            logger.info(f"  Name           : {sym.name}")
            logger.info(f"  Digits         : {sym.digits}")
            logger.info(f"  Point          : {sym.point}")
            logger.info(f"  Spread (pts)   : {sym.spread_points}")
            logger.info(f"  Tick size      : {sym.trade_tick_size}")
            logger.info(f"  Tick value     : {sym.trade_tick_value}")
            logger.info(f"  Contract size  : {sym.contract_size}")

            term = conn.terminal_info()
            logger.info("--- TERMINAL ---")
            logger.info(f"  Connected     : {term.get('connected')}")
            logger.info(f"  Trade allowed : {term.get('trade_allowed')}")
            logger.info(f"  Build         : {term.get('build')}")
            logger.info(f"  Ping (ms)     : {term.get('ping_last', 0) / 1000:.2f}")

    except MT5ConnectionError as e:
        logger.error(f"Connection test FAILED: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 2

    logger.success("MT5 connection test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())