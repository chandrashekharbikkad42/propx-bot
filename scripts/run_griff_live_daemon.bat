@echo off
REM Griff live-trading daemon wrapper.
REM Invoked by the Windows Scheduled Task registered via
REM scripts\install_griff_service.ps1. Activates the project venv,
REM sets ACTIVE_BROKER=FTMO, then runs the bot with output captured
REM to logs\griff_live_daemon.log.

setlocal

set REPO=%~dp0..
cd /d "%REPO%"

if not exist "venv\Scripts\python.exe" (
    echo [griff-daemon] venv not found at %REPO%\venv; aborting.
    exit /b 1
)

set ACTIVE_BROKER=FTMO

if not exist "logs" mkdir logs

REM Append-only log so restarts don't truncate history.
"venv\Scripts\python.exe" scripts\run_griff_live.py --no-dry-run %* 1>>logs\griff_live_daemon.log 2>&1

set EC=%ERRORLEVEL%
echo [griff-daemon] exit code %EC% at %DATE% %TIME% >>logs\griff_live_daemon.log
exit /b %EC%
