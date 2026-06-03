<#
.SYNOPSIS
    Remove the Griff live-trading Scheduled Task.

.PARAMETER TaskName
    Task to remove. Default: GriffLiveBot
#>

[CmdletBinding()]
param(
    [string]$TaskName = "GriffLiveBot"
)

$ErrorActionPreference = "Stop"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $existing) {
    Write-Host "No task named '$TaskName' registered. Nothing to do."
    exit 0
}

# Stop if running, then unregister.
try {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    Write-Host "Stopped running task '$TaskName'."
} catch {
    # Task wasn't running — fine.
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "✅ Scheduled Task '$TaskName' removed."
