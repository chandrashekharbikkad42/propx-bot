"""AccountState factory + canned profiles for compliance tests."""

from __future__ import annotations
from dataclasses import replace

from risk.prop_firm.compliance import AccountState


def make_account(
    *,
    equity: float = 10_000.0,
    starting_equity: float = 10_000.0,
    daily_start_equity: float = 10_000.0,
    daily_pnl_usd: float = 0.0,
    trades_today: int = 0,
    open_position_count: int = 0,
) -> AccountState:
    return AccountState(
        equity=equity,
        starting_equity=starting_equity,
        daily_start_equity=daily_start_equity,
        daily_pnl_usd=daily_pnl_usd,
        trades_today=trades_today,
        open_position_count=open_position_count,
    )


# Common canned states ------------------------------------------------------

FRESH_FUNDED = make_account()


def with_daily_pnl(account: AccountState, pnl_usd: float) -> AccountState:
    return replace(account, daily_pnl_usd=pnl_usd)


def with_equity(account: AccountState, equity: float) -> AccountState:
    return replace(account, equity=equity)


def with_trades(account: AccountState, n: int) -> AccountState:
    return replace(account, trades_today=n)
