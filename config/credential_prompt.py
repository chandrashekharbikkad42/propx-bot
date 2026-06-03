"""Interactive MT5-credential entry + .env persistence (Phase 5/B).

Profile model:
  Every broker profile is a flat group of env vars in the project `.env`
  file, all sharing the `BROKER_<NAME>_` prefix:

      BROKER_THE5ERS_LOGIN=...
      BROKER_THE5ERS_PASSWORD=...
      BROKER_THE5ERS_SERVER=...
      BROKER_THE5ERS_PATH=...
      BROKER_THE5ERS_LABEL=The5%ers

  The active profile is selected by:

      ACTIVE_BROKER=THE5ERS

  We deliberately stay flat-file (no INI sections) because python-dotenv
  only understands flat KEY=VALUE; the prefix convention preserves grouping
  without losing dotenv compatibility.

Legacy compatibility (Phase 4 .env style):
  The5%ers-style `FTMO_LOGIN` / `MT5_LOGIN` (ROBOFOREX) groupings still
  resolve via `config.broker_config`, so an unmigrated .env continues to
  work. The interactive prompt rewrites the file into the new
  `BROKER_<NAME>_*` convention as a side effect.

Hinglish: bot start hote hi .env me credentials check karte hain. Agar
missing ya `--reset` flag mila to interactively poochho aur save kar do.
Multi-account support: alag-alag `BROKER_<NAME>_*` blocks save karke
`ACTIVE_BROKER` flip karke switch karo.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import Iterable, Optional


# Required env keys per profile (besides the optional LABEL + PATH).
PROFILE_REQUIRED_SUFFIXES: tuple[str, ...] = ("LOGIN", "PASSWORD", "SERVER")
PROFILE_OPTIONAL_SUFFIXES: tuple[str, ...] = ("PATH", "LABEL")
# Anything matching this regex in .env is treated as a broker profile key.
_PROFILE_KEY_RE = re.compile(
    r"^BROKER_(?P<name>[A-Z0-9_]+)_(?P<suffix>LOGIN|PASSWORD|SERVER|PATH|LABEL)$"
)


@dataclass(frozen=True)
class BrokerProfile:
    """One broker profile, sourced from `.env` BROKER_<NAME>_* keys."""
    name: str          # canonical UPPER_SNAKE — used in env keys & --switch
    login: int
    password: str
    server: str
    path: str = ""     # MT5 terminal binary; may be empty (uses MT5_PATH)
    label: str = ""    # human-readable display name (Telegram, dashboard)

    @property
    def is_complete(self) -> bool:
        return bool(self.login and self.password and self.server)


def _project_env_path() -> Path:
    """`.env` always lives at the project root, one level above `config/`."""
    return Path(__file__).resolve().parent.parent / ".env"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_env_file(env_path: Optional[Path] = None) -> dict[str, str]:
    """Parse `.env` into a flat dict. Preserves comment-free key=value pairs.

    NOTE: we deliberately do NOT consult `os.environ` here — this function
    sees the FILE only, so the prompt can edit it without interference from
    shell exports. Use `load_dotenv()` or `os.environ` for runtime lookups.
    """
    path = env_path or _project_env_path()
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def list_profiles(env: Optional[dict[str, str]] = None) -> list[BrokerProfile]:
    """All BROKER_<NAME>_* profiles present in `.env`, sorted by name."""
    e = env if env is not None else read_env_file()
    grouped: dict[str, dict[str, str]] = {}
    for k, v in e.items():
        m = _PROFILE_KEY_RE.match(k)
        if not m:
            continue
        grouped.setdefault(m.group("name"), {})[m.group("suffix")] = v
    profiles: list[BrokerProfile] = []
    for name, parts in sorted(grouped.items()):
        try:
            login = int(parts.get("LOGIN", "0") or "0")
        except ValueError:
            login = 0
        profiles.append(BrokerProfile(
            name=name,
            login=login,
            password=parts.get("PASSWORD", ""),
            server=parts.get("SERVER", ""),
            path=parts.get("PATH", ""),
            label=parts.get("LABEL", ""),
        ))
    return profiles


def get_active_profile(env: Optional[dict[str, str]] = None) -> Optional[BrokerProfile]:
    """Resolve ACTIVE_BROKER → profile. None if either is missing/incomplete."""
    e = env if env is not None else read_env_file()
    name = (e.get("ACTIVE_BROKER") or os.environ.get("ACTIVE_BROKER") or "").strip().upper()
    if not name:
        return None
    for p in list_profiles(e):
        if p.name == name and p.is_complete:
            return p
    return None


# ---------------------------------------------------------------------------
# Writing — preserves all NON-broker env keys (telegram tokens, log level...)
# ---------------------------------------------------------------------------

def _serialize_env(
    non_profile: dict[str, str],
    profiles: Iterable[BrokerProfile],
    active_broker: str,
) -> str:
    """Build the `.env` body. Non-profile keys come first; profiles are
    grouped into clearly delimited sections.
    """
    lines: list[str] = []
    if non_profile:
        lines.append("# === Global settings ===")
        for k in sorted(non_profile):
            lines.append(f"{k}={non_profile[k]}")
        lines.append("")
    lines.append(f"ACTIVE_BROKER={active_broker}")
    lines.append("")
    for p in profiles:
        title = p.label or p.name
        lines.append(f"# === Broker profile: {title} ===")
        lines.append(f"BROKER_{p.name}_LOGIN={p.login}")
        lines.append(f"BROKER_{p.name}_PASSWORD={p.password}")
        lines.append(f"BROKER_{p.name}_SERVER={p.server}")
        if p.path:
            lines.append(f"BROKER_{p.name}_PATH={p.path}")
        if p.label:
            lines.append(f"BROKER_{p.name}_LABEL={p.label}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_env_file(
    profiles: Iterable[BrokerProfile],
    active_broker: str,
    *,
    env_path: Optional[Path] = None,
    preserve_unknown: bool = True,
) -> Path:
    """Persist profiles + active selection. Returns the written path.

    `preserve_unknown=True` keeps non-broker keys (telegram tokens, log
    settings, etc.) intact across rewrites.
    """
    path = env_path or _project_env_path()
    existing = read_env_file(path) if preserve_unknown else {}
    # Strip out anything that would collide with the new broker section /
    # the ACTIVE_BROKER key — those are owned by this writer.
    non_profile = {
        k: v for k, v in existing.items()
        if not _PROFILE_KEY_RE.match(k) and k != "ACTIVE_BROKER"
    }
    body = _serialize_env(non_profile, profiles, active_broker)
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Interactive prompt
# ---------------------------------------------------------------------------

def _normalise_name(raw: str) -> str:
    """User-typed broker name → canonical uppercase profile key.

    Single-word inputs ('The5%ers', 'FTMO', 'MyForexFunds') drop all
    non-alphanum and uppercase, producing dotenv-safe identifiers without
    underscores. Multi-word inputs containing literal separators
    (spaces or dashes — but NOT punctuation like `%`) are joined with
    `_` for readability:

        'The5%ers'        → 'THE5ERS'
        'FTMO'            → 'FTMO'
        'MyForexFunds'    → 'MYFOREXFUNDS'
        'My Forex Funds'  → 'MY_FOREX_FUNDS'
        'My-Forex-Funds'  → 'MY_FOREX_FUNDS'
    """
    # Step 1: word-separators → underscore; other punctuation dropped.
    s = re.sub(r"[ \-]+", "_", raw)
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_").upper()
    return s or "BROKER"


def prompt_credentials(
    *,
    suggested_name: Optional[str] = None,
    input_fn=input,
    password_fn=getpass,
    output_fn=print,
) -> BrokerProfile:
    """Interactive single-profile capture.

    `input_fn` / `password_fn` are injectable for testing (otherwise stdin).
    """
    output_fn("")
    output_fn("=== MT5 broker credentials ===")
    output_fn("Enter the credentials your MT5 terminal uses to log in.")
    output_fn("(Press Ctrl+C at any time to abort.)")
    output_fn("")
    label_default = suggested_name or "The5%ers"
    label = input_fn(f"Broker display label (e.g. 'The5%ers', 'FTMO') "
                     f"[{label_default}]: ").strip() or label_default
    name_default = _normalise_name(label)
    name_raw = input_fn(f"Profile key (UPPER_SNAKE) [{name_default}]: ").strip()
    name = _normalise_name(name_raw) if name_raw else name_default

    login_raw = input_fn("MT5 account number: ").strip()
    try:
        login = int(login_raw)
    except ValueError as exc:
        raise ValueError(
            f"Account number must be an integer, got {login_raw!r}"
        ) from exc

    password = password_fn("MT5 password (hidden): ").strip()
    if not password:
        raise ValueError("Password may not be empty")

    server = input_fn("MT5 server (e.g. 'FTMO-Demo', 'RoboForex-Pro'): ").strip()
    if not server:
        raise ValueError("Server may not be empty")

    path = input_fn(
        "MT5 terminal path (optional, full path to terminal64.exe; press "
        "Enter to use MT5_PATH): "
    ).strip()

    return BrokerProfile(
        name=name, login=login, password=password,
        server=server, path=path, label=label,
    )


def ensure_active_credentials(
    *,
    reset: bool = False,
    switch_to: Optional[str] = None,
    env_path: Optional[Path] = None,
    interactive: bool = True,
    input_fn=input,
    password_fn=getpass,
    output_fn=print,
) -> BrokerProfile:
    """Top-level entry point used by `run_asian_sweep_live.py`.

    Resolution order:
      1. If `switch_to` is given, set ACTIVE_BROKER to that profile name.
         Raise if no profile with that name exists (use `--reset` to add).
      2. If `reset` is True OR the active profile is missing/incomplete,
         enter the interactive prompt (when `interactive=True`); the
         resulting profile becomes ACTIVE_BROKER.
      3. Otherwise return the existing active profile unchanged.

    Always re-reads `.env` AFTER any mutation so the returned profile and
    the on-disk file agree.
    """
    env = read_env_file(env_path)
    profiles = {p.name: p for p in list_profiles(env)}

    if switch_to:
        target = _normalise_name(switch_to)
        if target in profiles and profiles[target].is_complete:
            write_env_file(profiles.values(), active_broker=target, env_path=env_path)
            output_fn(f"[broker] Active profile -> {target}")
            return profiles[target]
        if not interactive:
            raise RuntimeError(
                f"--switch {switch_to}: no complete profile named {target!r} "
                f"in .env. Run with --reset to create one."
            )
        output_fn(
            f"[broker] No complete profile named {target!r}. Creating one now."
        )
        new_profile = prompt_credentials(
            suggested_name=target,
            input_fn=input_fn, password_fn=password_fn, output_fn=output_fn,
        )
        profiles[new_profile.name] = new_profile
        write_env_file(profiles.values(), active_broker=new_profile.name,
                       env_path=env_path)
        return new_profile

    active = get_active_profile(env)
    if active is not None and not reset:
        return active

    if not interactive:
        raise RuntimeError(
            "No active broker credentials in .env and interactive=False. "
            "Set ACTIVE_BROKER + BROKER_<NAME>_LOGIN/PASSWORD/SERVER, or "
            "run the CLI without --no-prompt to enter them."
        )

    if reset:
        output_fn("[broker] --reset: re-entering credentials for active profile.")
    else:
        output_fn(
            "[broker] No active broker credentials found in .env. "
            "Let's set them up."
        )
    new_profile = prompt_credentials(
        input_fn=input_fn, password_fn=password_fn, output_fn=output_fn,
    )
    profiles[new_profile.name] = new_profile
    write_env_file(profiles.values(), active_broker=new_profile.name,
                   env_path=env_path)
    output_fn(
        f"[broker] Saved profile BROKER_{new_profile.name}_* to .env "
        f"and set ACTIVE_BROKER={new_profile.name}."
    )
    return new_profile


__all__ = [
    "BrokerProfile",
    "PROFILE_REQUIRED_SUFFIXES",
    "PROFILE_OPTIONAL_SUFFIXES",
    "read_env_file",
    "list_profiles",
    "get_active_profile",
    "write_env_file",
    "prompt_credentials",
    "ensure_active_credentials",
]
