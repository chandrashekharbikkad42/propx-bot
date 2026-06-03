# Griff Live-Trading Daemon (Phase 9/5)

Run the bot 24/7 on a Windows machine without keeping a terminal open.

## What you get

- Bot auto-starts when you log into Windows.
- Bot auto-restarts every 1 minute on crash (max 5 retries per cycle).
- No console window — runs hidden in the background.
- All output appended to `logs\griff_live_daemon.log`.
- Survives terminal close. Does **not** survive PC power-off (the laptop
  has to be on and signed in).

This uses a Windows **Scheduled Task** triggered on user logon. No third-
party services (NSSM etc.) and no admin rights required.

## Install

1. Confirm `EXECUTION_MODE=REAL` is set in `.env` (the bot's own two-key
   safety gate — without this it stays in DRY_RUN even with `--no-dry-run`).
2. Confirm `ACTIVE_BROKER=FTMO` in `.env` (the wrapper script also sets
   this at runtime, but keep the .env consistent).
3. Run the pre-flight first — it'll fail fast if MT5 / FTMO terminal
   isn't reachable, which is much easier to debug than a silent daemon:
   ```powershell
   .\venv\Scripts\python.exe scripts\ftmo_preflight.py
   ```
4. Register the Scheduled Task:
   ```powershell
   powershell.exe -ExecutionPolicy Bypass -File scripts\install_griff_service.ps1
   ```
5. Start it now (it'll also start on next logon automatically):
   ```powershell
   Start-ScheduledTask -TaskName GriffLiveBot
   ```

## Tail the logs

```powershell
Get-Content logs\griff_live_daemon.log -Wait -Tail 50
```

## Stop it

```powershell
Stop-ScheduledTask -TaskName GriffLiveBot
```

The task will still re-trigger on your next logon. To permanently remove:

```powershell
powershell.exe -ExecutionPolicy Bypass -File scripts\uninstall_griff_service.ps1
```

## Laptop power settings (important)

A Scheduled Task can't trade while the laptop is asleep. On any laptop
that's the deployment host, in Windows Settings:

- System → Power & battery → Screen and sleep
  - **When plugged in, put my device to sleep after**: Never
- System → Power & battery → Power mode: Best Performance (when plugged in)
- Close-lid behaviour (Control Panel → Power Options → Choose what
  closing the lid does): **Do nothing** when plugged in

If the lid closes and the machine sleeps, MT5's connection drops, the
poll loop pauses, and you miss bar closes. The bot will resume when the
machine wakes — but trades that needed to fire while it was asleep are
gone.

## How it works

- `scripts\run_griff_live_daemon.bat` is the entry point. It cd's to the
  repo, activates the venv, sets `ACTIVE_BROKER=FTMO`, and invokes
  `scripts\run_griff_live.py --no-dry-run`. stdout + stderr are appended
  to `logs\griff_live_daemon.log`.
- `scripts\install_griff_service.ps1` registers a Scheduled Task that
  runs that .bat on user logon, with restart-on-failure and hidden window.
- The bot still respects its own two-key safety: `--no-dry-run` plus
  `EXECUTION_MODE=REAL`. Either missing → it stays in DRY_RUN even when
  the daemon is running.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Task shows `Last Run Result: 0x1` and bot never opens MT5 | venv missing — check `venv\Scripts\python.exe` exists. |
| Bot connects briefly then dies with IPC timeout | MT5 terminal opened manually for a different account first; let the bot launch its own instance. |
| Telegram silent for hours | Check `logs\griff_live_daemon.log` for SSL or token errors; `tests/test_telegram_notifier.py` covers the SSL hardening. |
| Task disappears after Windows update | Re-run the installer. |

## Caveat

If you reboot or sleep, the daemon resumes from a clean state — open
positions still exist with FTMO (broker-side state), but any in-flight
pending orders the bot was tracking in memory are reconstructed from
MT5's `positions_get` / `orders_get` on the next maintenance cycle.
The existing `GriffPositionManager` has the bookkeeping methods — but
the boot path doesn't currently reconcile from MT5 state. That's a
future hardening (Phase 10).
