"""Prop firm auto-detection from MT5 connection metadata.

Strategy: pattern-match against the broker's server name and (optionally)
company string. If detection succeeds, return a rule key from RULES_DB
that the user can override in .env.

Detection precedence (highest first):
  1. Explicit user config (`detect_from_config`).
  2. MT5 server match (`detect_from_mt5`).
  3. None — caller must fall back or raise.

Hinglish: MT5 terminal pe konsa server connected hai uss se prop firm
identify hoti hai. RoboForex jaise regular brokers ko prop firm nahi
mante — None return karte hain, bot prop-rules engine ko skip kar deta hai.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Mapping, Optional, Pattern

from risk.prop_firm.rules import RULES_DB


@dataclass(frozen=True)
class AccountInfo:
    """Minimal subset of MT5 account_info used for detection. Decoupled from
    the live MT5 connector so tests can pass plain dataclasses."""
    server: str
    company: str = ""
    login: int = 0
    balance: float = 0.0


# Server / company → default rule-key. The default is the most conservative
# stage (challenge / step 1) so the bot never assumes a more permissive set
# of caps than it can verify. Caller upgrades to "_funded" / later steps
# via config override.
_SERVER_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    (re.compile(r"^FTMO[-_]?", re.IGNORECASE), "ftmo_2step_challenge"),
    (re.compile(r"^The5ers", re.IGNORECASE), "the5ers_bootcamp_step1"),
    (re.compile(r"5ers[-_]?", re.IGNORECASE), "the5ers_bootcamp_step1"),
)

# Brokers we recognise as NOT a prop firm. Returning None here means the
# compliance engine is bypassed — useful for personal-funds testing.
_NON_PROP_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"^RoboForex", re.IGNORECASE),
    re.compile(r"^ICMarkets", re.IGNORECASE),
    re.compile(r"^IC[-_ ]Markets", re.IGNORECASE),
    re.compile(r"^Pepperstone", re.IGNORECASE),
)


class PropFirmDetector:
    """Pluggable detector. Subclass and override `_match_server` to add brokers."""

    def detect_from_mt5(self, account: AccountInfo) -> Optional[str]:
        """Return a rule key or None when the broker is non-prop / unknown."""
        server = (account.server or "").strip()
        company = (account.company or "").strip()
        if not server and not company:
            return None
        # Non-prop first — explicit allow-list of retail brokers.
        for pat in _NON_PROP_PATTERNS:
            if server and pat.match(server):
                return None
            if company and pat.match(company):
                return None
        # Prop firm patterns.
        for pat, key in _SERVER_PATTERNS:
            if server and pat.search(server):
                return key
            if company and pat.search(company):
                return key
        return None

    def detect_from_config(self, configured_key: Optional[str]) -> Optional[str]:
        """Validate a user-provided override key against RULES_DB."""
        if not configured_key:
            return None
        if configured_key not in RULES_DB:
            raise ValueError(
                f"Unknown prop firm rule key in config: {configured_key!r}. "
                f"Available: {sorted(RULES_DB.keys())}"
            )
        return configured_key

    def detect(
        self,
        account: AccountInfo,
        configured_key: Optional[str] = None,
    ) -> Optional[str]:
        """Config override wins; otherwise fall back to MT5 server match."""
        # Validates and returns if set; else falls through.
        from_cfg = self.detect_from_config(configured_key)
        if from_cfg is not None:
            return from_cfg
        return self.detect_from_mt5(account)
