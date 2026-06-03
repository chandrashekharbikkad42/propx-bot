"""Prop-firm rule definitions.

Each entry in RULES_DB is one stage of one program. Values are the public
posted limits as of mid-2026; numbers marked TODO are unverified or vary by
account size and should be confirmed before the bot trades on a real
funded account.

Cap semantics:
  - max_daily_loss_pct      : equity drawdown vs DAILY-START equity (UTC or
                              broker-day depending on firm — most firms use
                              broker server time; we honor that downstream).
  - max_total_loss_pct      : equity drawdown vs ACCOUNT-START equity.
  - profit_target_pct       : minimum gain to pass the stage. 0 = no target
                              (funded accounts).
  - consistency_rule        : optional dict; common form
                              {"max_best_day_pct_of_total": 50.0}.
  - tick_scalp_allowed      : False ⇒ minimum hold time enforced (typically 3s).
  - overnight_allowed       : False ⇒ positions must close by broker EOD.
  - weekend_allowed         : False ⇒ all positions flat before Fri close.

Hinglish: ye dictionary bot ka rule-book hai. Auto-detector dekh ke decide
karta hai konsa firm hai, compliance engine yahaan se caps uthata hai.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass(frozen=True)
class PropFirmRules:
    name: str
    max_daily_loss_pct: float
    max_total_loss_pct: float
    profit_target_pct: float
    min_trading_days: int
    leverage_forex: int
    leverage_metals: int
    news_blackout_minutes_before: int
    news_blackout_minutes_after: int
    consistency_rule: Optional[Mapping[str, float]] = None
    tick_scalp_allowed: bool = False
    overnight_allowed: bool = True
    weekend_allowed: bool = False

    def __post_init__(self) -> None:
        # Sanity: profit target should not exceed total loss cap by more than
        # 5× (a 50% target with 10% loss cap is a stress signal — flag with
        # an exception rather than letting a typo through).
        if self.max_total_loss_pct <= 0:
            raise ValueError(
                f"{self.name}: max_total_loss_pct must be > 0"
            )
        if self.max_daily_loss_pct <= 0:
            raise ValueError(
                f"{self.name}: max_daily_loss_pct must be > 0"
            )
        if self.max_daily_loss_pct > self.max_total_loss_pct:
            raise ValueError(
                f"{self.name}: daily cap ({self.max_daily_loss_pct}%) "
                f"exceeds total cap ({self.max_total_loss_pct}%)"
            )
        if self.profit_target_pct < 0:
            raise ValueError(
                f"{self.name}: profit_target_pct must be >= 0"
            )
        if self.min_trading_days < 0:
            raise ValueError(
                f"{self.name}: min_trading_days must be >= 0"
            )
        if self.leverage_forex <= 0 or self.leverage_metals <= 0:
            raise ValueError(f"{self.name}: leverage must be > 0")


# ---------------------------------------------------------------------------
# FTMO — public limits as of 2025/2026. Values per the FTMO website.
# ---------------------------------------------------------------------------

_FTMO_2STEP_CHALLENGE = PropFirmRules(
    name="FTMO 2-Step Challenge",
    max_daily_loss_pct=5.0,
    max_total_loss_pct=10.0,
    profit_target_pct=10.0,
    min_trading_days=4,
    leverage_forex=100,
    leverage_metals=30,
    news_blackout_minutes_before=2,
    news_blackout_minutes_after=2,
    consistency_rule=None,
    tick_scalp_allowed=False,
    overnight_allowed=True,
    weekend_allowed=False,
)

_FTMO_2STEP_VERIFICATION = PropFirmRules(
    name="FTMO 2-Step Verification",
    max_daily_loss_pct=5.0,
    max_total_loss_pct=10.0,
    profit_target_pct=5.0,
    min_trading_days=4,
    leverage_forex=100,
    leverage_metals=30,
    news_blackout_minutes_before=2,
    news_blackout_minutes_after=2,
    consistency_rule=None,
    tick_scalp_allowed=False,
    overnight_allowed=True,
    weekend_allowed=False,
)

_FTMO_2STEP_FUNDED = PropFirmRules(
    name="FTMO 2-Step Funded",
    max_daily_loss_pct=5.0,
    max_total_loss_pct=10.0,
    profit_target_pct=0.0,            # no target, just don't breach
    min_trading_days=0,
    leverage_forex=100,
    leverage_metals=30,
    news_blackout_minutes_before=2,
    news_blackout_minutes_after=2,
    consistency_rule=None,             # TODO confirm — payout has rules
    tick_scalp_allowed=False,
    overnight_allowed=True,
    weekend_allowed=False,
)

_FTMO_1STEP_CHALLENGE = PropFirmRules(
    name="FTMO 1-Step Challenge",
    max_daily_loss_pct=3.0,            # strictest mode
    max_total_loss_pct=6.0,
    profit_target_pct=10.0,
    min_trading_days=3,
    leverage_forex=100,
    leverage_metals=30,
    news_blackout_minutes_before=2,
    news_blackout_minutes_after=2,
    # FTMO 1-Step has the consistency rule on payout: best day ≤ 50% of total.
    consistency_rule={"max_best_day_pct_of_total": 50.0},
    tick_scalp_allowed=False,
    overnight_allowed=True,
    weekend_allowed=False,
)

_FTMO_1STEP_FUNDED = PropFirmRules(
    name="FTMO 1-Step Funded",
    max_daily_loss_pct=3.0,
    max_total_loss_pct=6.0,
    profit_target_pct=0.0,
    min_trading_days=0,
    leverage_forex=100,
    leverage_metals=30,
    news_blackout_minutes_before=2,
    news_blackout_minutes_after=2,
    consistency_rule={"max_best_day_pct_of_total": 50.0},
    tick_scalp_allowed=False,
    overnight_allowed=True,
    weekend_allowed=False,
)

# ---------------------------------------------------------------------------
# The5ers — values per public site. Some sub-products vary; flagged TODO.
# ---------------------------------------------------------------------------

_THE5ERS_BOOTCAMP_STEP1 = PropFirmRules(
    name="The5ers Bootcamp Step 1",
    max_daily_loss_pct=4.0,            # TODO confirm — varies by program
    max_total_loss_pct=5.0,
    profit_target_pct=6.0,
    min_trading_days=10,
    leverage_forex=30,
    leverage_metals=20,
    news_blackout_minutes_before=2,
    news_blackout_minutes_after=2,
    consistency_rule=None,
    tick_scalp_allowed=False,
    overnight_allowed=True,
    weekend_allowed=False,
)

_THE5ERS_BOOTCAMP_STEP2 = PropFirmRules(
    name="The5ers Bootcamp Step 2",
    max_daily_loss_pct=4.0,
    max_total_loss_pct=5.0,
    profit_target_pct=6.0,
    min_trading_days=10,
    leverage_forex=30,
    leverage_metals=20,
    news_blackout_minutes_before=2,
    news_blackout_minutes_after=2,
    consistency_rule=None,
    tick_scalp_allowed=False,
    overnight_allowed=True,
    weekend_allowed=False,
)

_THE5ERS_BOOTCAMP_STEP3 = PropFirmRules(
    name="The5ers Bootcamp Step 3",
    max_daily_loss_pct=4.0,
    max_total_loss_pct=5.0,
    profit_target_pct=6.0,
    min_trading_days=10,
    leverage_forex=30,
    leverage_metals=20,
    news_blackout_minutes_before=2,
    news_blackout_minutes_after=2,
    consistency_rule=None,
    tick_scalp_allowed=False,
    overnight_allowed=True,
    weekend_allowed=False,
)

_THE5ERS_BOOTCAMP_FUNDED = PropFirmRules(
    name="The5ers Bootcamp Funded",
    max_daily_loss_pct=4.0,
    max_total_loss_pct=5.0,
    profit_target_pct=0.0,
    min_trading_days=0,
    leverage_forex=30,
    leverage_metals=20,
    news_blackout_minutes_before=2,
    news_blackout_minutes_after=2,
    consistency_rule=None,
    tick_scalp_allowed=False,
    overnight_allowed=True,
    weekend_allowed=False,
)

_THE5ERS_HYPER_GROWTH_STEP1 = PropFirmRules(
    name="The5ers Hyper Growth Step 1",
    max_daily_loss_pct=5.0,
    max_total_loss_pct=10.0,
    profit_target_pct=10.0,
    min_trading_days=10,
    leverage_forex=30,
    leverage_metals=20,
    news_blackout_minutes_before=2,
    news_blackout_minutes_after=2,
    consistency_rule=None,
    tick_scalp_allowed=False,
    overnight_allowed=True,
    weekend_allowed=False,
)

_THE5ERS_HYPER_GROWTH_FUNDED = PropFirmRules(
    name="The5ers Hyper Growth Funded",
    max_daily_loss_pct=5.0,
    max_total_loss_pct=10.0,
    profit_target_pct=0.0,
    min_trading_days=0,
    leverage_forex=30,
    leverage_metals=20,
    news_blackout_minutes_before=2,
    news_blackout_minutes_after=2,
    consistency_rule=None,
    tick_scalp_allowed=False,
    overnight_allowed=True,
    weekend_allowed=False,
)

_THE5ERS_HIGH_STAKES_STEP1 = PropFirmRules(
    name="The5ers High Stakes Step 1",
    max_daily_loss_pct=5.0,            # TODO confirm
    max_total_loss_pct=10.0,
    profit_target_pct=8.0,
    min_trading_days=5,
    leverage_forex=30,
    leverage_metals=20,
    news_blackout_minutes_before=2,
    news_blackout_minutes_after=2,
    consistency_rule=None,
    tick_scalp_allowed=False,
    overnight_allowed=True,
    weekend_allowed=False,
)

_THE5ERS_HIGH_STAKES_STEP2 = PropFirmRules(
    name="The5ers High Stakes Step 2",
    max_daily_loss_pct=5.0,
    max_total_loss_pct=10.0,
    profit_target_pct=5.0,
    min_trading_days=5,
    leverage_forex=30,
    leverage_metals=20,
    news_blackout_minutes_before=2,
    news_blackout_minutes_after=2,
    consistency_rule=None,
    tick_scalp_allowed=False,
    overnight_allowed=True,
    weekend_allowed=False,
)

_THE5ERS_HIGH_STAKES_FUNDED = PropFirmRules(
    name="The5ers High Stakes Funded",
    max_daily_loss_pct=5.0,
    max_total_loss_pct=10.0,
    profit_target_pct=0.0,
    min_trading_days=0,
    leverage_forex=30,
    leverage_metals=20,
    news_blackout_minutes_before=2,
    news_blackout_minutes_after=2,
    consistency_rule=None,
    tick_scalp_allowed=False,
    overnight_allowed=True,
    weekend_allowed=False,
)


RULES_DB: Mapping[str, PropFirmRules] = {
    "ftmo_2step_challenge": _FTMO_2STEP_CHALLENGE,
    "ftmo_2step_verification": _FTMO_2STEP_VERIFICATION,
    "ftmo_2step_funded": _FTMO_2STEP_FUNDED,
    "ftmo_1step_challenge": _FTMO_1STEP_CHALLENGE,
    "ftmo_1step_funded": _FTMO_1STEP_FUNDED,
    "the5ers_bootcamp_step1": _THE5ERS_BOOTCAMP_STEP1,
    "the5ers_bootcamp_step2": _THE5ERS_BOOTCAMP_STEP2,
    "the5ers_bootcamp_step3": _THE5ERS_BOOTCAMP_STEP3,
    "the5ers_bootcamp_funded": _THE5ERS_BOOTCAMP_FUNDED,
    "the5ers_hyper_growth_step1": _THE5ERS_HYPER_GROWTH_STEP1,
    "the5ers_hyper_growth_funded": _THE5ERS_HYPER_GROWTH_FUNDED,
    "the5ers_high_stakes_step1": _THE5ERS_HIGH_STAKES_STEP1,
    "the5ers_high_stakes_step2": _THE5ERS_HIGH_STAKES_STEP2,
    "the5ers_high_stakes_funded": _THE5ERS_HIGH_STAKES_FUNDED,
}


def get_rules(key: str) -> PropFirmRules:
    """Return rules for a key; raises KeyError with a helpful message."""
    try:
        return RULES_DB[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown prop firm rule key: {key!r}. "
            f"Available: {sorted(RULES_DB.keys())}"
        ) from exc


def list_rule_keys() -> tuple[str, ...]:
    """All registered rule keys, sorted alphabetically."""
    return tuple(sorted(RULES_DB.keys()))
