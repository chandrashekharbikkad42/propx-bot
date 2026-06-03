<#
.SYNOPSIS
    Install Griff live-trading bot as a Windows Scheduled Task.

.DESCRIPTION
    Registers a Scheduled Task that:
      - Triggers on user logon (so MT5's user-session requirement is met).
      - Runs scripts\run_griff_live_daemon.bat (which calls the bot with
        --no-dry-run after activating the venv).
      - Restarts every 1 minute on failure, up to 5 retries.
      - Runs hidden (no console window).
      - Logs to logs\griff_live_daemon.log (append).

    Requires:
      - EXECUTION_MODE=REAL set in .env (the bot's own two-key gate).
      - ACTIVE_BROKER=FTMO (set by the wrapper script).
      - MT5 terminals (FTMO + RoboForex) installed at the paths in .env.
      - The Windows account is the one that runs MT5 manually (Scheduled
        Task triggers on logon for the current user).

.PARAMETER TaskName
    Name to register under. Default: GriffLiveBot

.PARAMETER ExtraArgs
    Extra args to pass to run_griff_live.py (e.g. "--poll-sec 60").

.EXAMPLE
    # Register the task (run from project root, in an elevated PowerShell):
    powershell.exe -ExecutionPolicy Bypass -File scripts\install_griff_service.ps1

.EXAMPLE
    # Verify task installed:
    Get-ScheduledTask -TaskName GriffLiveBot | Format-List *

.EXAMPLE
    # Start it immediately (also runs automatically on next logon):
    Start-ScheduledTask -TaskName GriffLiveBot

.NOTES
    Laptops: set Power Options to "Never sleep" when plugged in or the bot
    will pause when the lid closes. Settings → System → Power & battery →
    Screen and sleep → Sleep = Never (when plugged in).
#>

[CmdletBinding()]
param(
    [string]$TaskName = "GriffLiveBot",
    [string]$ExtraArgs = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$daemonBat = Join-Path $scriptDir "run_griff_live_daemon.bat"

if (-not (Test-Path $daemonBat)) {
    throw "Daemon wrapper not found: $daemonBat"
}

Write-Host "Repo:        $repoRoot"
Write-Host "Daemon bat:  $daemonBat"
Write-Host "Task name:   $TaskName"

# Remove an existing registration with the same name (idempotent).
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task '$TaskName'..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Action — run the daemon batch with any extra args.
$action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$daemonBat`" $ExtraArgs" `
    -WorkingDirectory $repoRoot

# Trigger — at logon of the current user.
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"

# Settings — restart on failure, no idle stop, no time limit, run hidden.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -RestartCount 5 `
    -ExecutionTimeLimit (New-TimeSpan -Days 0) `
    -Hidden

# Principal — run as the current user, NOT elevated (MT5 doesn't need it).
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Griff Bot — FTMO live trading daemon (Phase 9)." | Out-Null

Write-Host ""
Write-Host "✅ Scheduled Task '$TaskName' installed."
Write-Host "   Triggers on logon of $env:USERDOMAIN\$env:USERNAME"
Write-Host "   Logs: $repoRoot\logs\griff_live_daemon.log"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Confirm EXECUTION_MODE=REAL in .env"
Write-Host "  2. Confirm FTMO terminal can launch unattended (run preflight)"
Write-Host "  3. Start now:   Start-ScheduledTask -TaskName $TaskName"
Write-Host "  4. Or sign out / sign back in — bot will auto-start"
Write-Host ""
Write-Host "Tail logs:"
Write-Host "  Get-Content $repoRoot\logs\griff_live_daemon.log -Wait -Tail 50"
