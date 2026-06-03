"""propX — branded startup banner + colour theme.

Pure display module. No state. No side effects unless you ask for them
(call `print_banner(...)` to write to stdout). The colour palette is
exposed as module constants so the console dashboard can share it.

Hinglish: bot start hote hi screen ke top pe ASCII art "propX" + ek
tagline aur active broker/account/mode ki info. Sirf cosmetic — no
trading logic touched here.
"""

from __future__ import annotations
import sys
from typing import Optional

from colorama import Fore, Style, init as _colorama_init


# Initialise colorama once at import — wraps stdout on Windows so ANSI
# escapes render natively. `autoreset=False` lets us compose sequences
# without a forced RESET between every Fore/Style change.
_colorama_init(autoreset=False)


# Force stdout/stderr to UTF-8 so the block-character ASCII art renders
# in PowerShell / cmd.exe (default cp1252 cannot encode `█`, `╔`, etc.).
# Failure is harmless — falls back to whatever encoding the host had.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is None:
        continue
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


# ─── Palette ─────────────────────────────────────────────────────────────────
# Shared by banner + console dashboard. Keep names semantic — callers say
# `THEME.profit` not `Fore.GREEN`, so we can re-skin without hunting.
class _Theme:
    profit = Fore.GREEN + Style.BRIGHT       # gains, OK status, active session
    loss = Fore.RED + Style.BRIGHT           # losses, halts, breaches
    warn = Fore.YELLOW + Style.BRIGHT        # neutral, DRY_RUN, warnings
    label = Style.DIM                        # row labels, hints
    accent = Fore.CYAN + Style.BRIGHT        # title, dividers
    accent_soft = Fore.CYAN                  # secondary brand colour
    value = Style.BRIGHT                     # bold white-ish — numeric values
    muted = Style.DIM                        # timestamps, footers
    reset = Style.RESET_ALL


THEME = _Theme()


# ─── ASCII art ───────────────────────────────────────────────────────────────
# "ANSI Shadow" figlet style — 6 lines tall, ~42 cols wide. Fits inside an
# 80-col terminal with breathing room. Hand-tweaked so block letters align
# even when the terminal font has slightly non-square cells.
_PROPX_ART = r"""
██████╗ ██████╗  ██████╗ ██████╗ ██╗  ██╗
██╔══██╗██╔══██╗██╔═══██╗██╔══██╗╚██╗██╔╝
██████╔╝██████╔╝██║   ██║██████╔╝ ╚███╔╝
██╔═══╝ ██╔══██╗██║   ██║██╔═══╝  ██╔██╗
██║     ██║  ██║╚██████╔╝██║     ██╔╝ ██╗
╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝  ╚═╝
"""

TAGLINE = "Asian Range London Sweep  |  Automated Multi-Pair"


def _mode_color(mode: str) -> str:
    """Mode label colour — REAL is red (live money), PAPER yellow, DRY_RUN dim."""
    m = (mode or "").upper()
    if m == "REAL":
        return THEME.loss
    if m == "PAPER":
        return THEME.warn
    return THEME.muted  # DRY_RUN / anything else


def render_banner(
    *,
    broker_label: Optional[str] = None,
    broker_name: Optional[str] = None,
    account: Optional[int | str] = None,
    server: Optional[str] = None,
    mode: str = "DRY_RUN",
    pairs: Optional[tuple[str, ...]] = None,
) -> str:
    """Return the full startup banner as one multi-line string (with ANSI).

    Caller chooses where to render — `print_banner(...)` is the obvious sink
    but tests just check the rendered text.
    """
    art_lines = [ln for ln in _PROPX_ART.splitlines() if ln.strip()]
    art_block = "\n".join(f"{THEME.accent}{ln}{THEME.reset}" for ln in art_lines)

    tagline = f"{THEME.accent_soft}{TAGLINE}{THEME.reset}"

    # Identity row — broker, account, mode.
    broker_display = broker_label or broker_name or "(no broker yet)"
    if broker_label and broker_name and broker_label != broker_name:
        broker_display = f"{broker_label}  [{broker_name}]"
    acct_display = str(account) if account else "—"
    mode_display = f"{_mode_color(mode)}{mode.upper()}{THEME.reset}"
    server_part = f"  {THEME.label}server:{THEME.reset} {server}" if server else ""
    identity = (
        f"  {THEME.label}broker:{THEME.reset} {THEME.value}{broker_display}{THEME.reset}"
        f"    {THEME.label}account:{THEME.reset} {THEME.value}{acct_display}{THEME.reset}"
        f"    {THEME.label}mode:{THEME.reset} {mode_display}"
        f"{server_part}"
    )

    pairs_line = ""
    if pairs:
        pairs_line = (
            f"\n  {THEME.label}pairs:{THEME.reset} "
            f"{THEME.accent_soft}{' '.join(pairs)}{THEME.reset}"
        )

    return (
        f"\n{art_block}\n"
        f"  {tagline}\n"
        f"{identity}{pairs_line}\n"
    )


def print_banner(**kwargs) -> None:
    """Render and write the banner to stdout. Thin sugar over `render_banner`."""
    print(render_banner(**kwargs))


__all__ = ["THEME", "TAGLINE", "render_banner", "print_banner"]
