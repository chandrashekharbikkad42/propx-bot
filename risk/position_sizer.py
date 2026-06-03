"""Pure lot-sizing function. No state, no I/O."""

from __future__ import annotations

POINT_VALUE = 0.01  # XAUUSD: 1 pt = $0.01

MIN_LOTS = 0.01
MAX_LOTS = 10.0


def calculate_lot_size(
    account_equity: float,
    risk_pct: float,
    sl_distance_pts: float,
    contract_size: int = 100,
    account_type: str = "STANDARD",
) -> float:
    """Lots such that an SL hit costs `risk_pct * account_equity` USD.

    PnL per lot at SL = sl_distance_pts * POINT_VALUE * contract_size.
    lots = risk_usd / pnl_per_lot.
    Clamped to broker bounds [0.01, 10.0] lots.

    For ROBOFOREX ProCent accounts the balance is denominated in USD-cents,
    so we divide by 100 before computing risk in real USD.
    """
    if sl_distance_pts <= 0:
        return MIN_LOTS
    if account_equity <= 0 or risk_pct <= 0:
        return MIN_LOTS

    if account_type == "PROCENT":
        real_equity = account_equity / 100.0
    else:
        real_equity = account_equity

    risk_usd = real_equity * risk_pct
    cost_per_lot = sl_distance_pts * POINT_VALUE * contract_size
    if cost_per_lot <= 0:
        return MIN_LOTS

    lots = risk_usd / cost_per_lot
    lots = round(lots, 2)
    if lots < MIN_LOTS:
        return MIN_LOTS
    if lots > MAX_LOTS:
        return MAX_LOTS
    return lots
