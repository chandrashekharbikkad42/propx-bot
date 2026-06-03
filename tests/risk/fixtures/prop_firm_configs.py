"""Compact registry of all prop-firm rule keys, grouped by family/phase."""

from __future__ import annotations

FTMO_2STEP_KEYS = (
    "ftmo_2step_challenge",
    "ftmo_2step_verification",
    "ftmo_2step_funded",
)
FTMO_1STEP_KEYS = (
    "ftmo_1step_challenge",
    "ftmo_1step_funded",
)
THE5ERS_BOOTCAMP_KEYS = (
    "the5ers_bootcamp_step1",
    "the5ers_bootcamp_step2",
    "the5ers_bootcamp_step3",
    "the5ers_bootcamp_funded",
)
THE5ERS_HYPER_GROWTH_KEYS = (
    "the5ers_hyper_growth_step1",
    "the5ers_hyper_growth_funded",
)
THE5ERS_HIGH_STAKES_KEYS = (
    "the5ers_high_stakes_step1",
    "the5ers_high_stakes_step2",
    "the5ers_high_stakes_funded",
)
ALL_FTMO_KEYS = FTMO_2STEP_KEYS + FTMO_1STEP_KEYS
ALL_THE5ERS_KEYS = (
    THE5ERS_BOOTCAMP_KEYS
    + THE5ERS_HYPER_GROWTH_KEYS
    + THE5ERS_HIGH_STAKES_KEYS
)
ALL_KEYS = ALL_FTMO_KEYS + ALL_THE5ERS_KEYS

FUNDED_KEYS = tuple(k for k in ALL_KEYS if k.endswith("_funded"))
CHALLENGE_KEYS = tuple(k for k in ALL_KEYS if "challenge" in k)
STEP_KEYS = tuple(k for k in ALL_KEYS if "step" in k)
