<#
.SYNOPSIS
  Restart the background Fin Analyst Agent UI (Streamlit): stop, then start.

.DESCRIPTION
  Runs stop_ui.ps1, waits for the port to free up, then runs start_ui.ps1, which
  waits for /_stcore/health. Use after code or .env changes; Streamlit's own
  file watcher does not reload modules imported by app/ui.py reliably.

.PARAMETER NoBrowser
  Restart without opening a browser tab.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\restart_ui.ps1
#>
param(
    [switch]$NoBrowser,
    [int]$Port = 8501
)

$ErrorActionPreference = 'Stop'

& (Join-Path $PSScriptRoot 'stop_ui.ps1')

# Give Windows a moment to release the listener so start_ui.ps1 doesn't see the old server.
$deadline = (Get-Date).AddSeconds(15)
while ((Get-Date) -lt $deadline) {
    try { Invoke-WebRequest "http://localhost:$Port/_stcore/health" -UseBasicParsing -TimeoutSec 1 | Out-Null }
    catch { break }
    Start-Sleep -Milliseconds 300
}

& (Join-Path $PSScriptRoot 'start_ui.ps1') -NoBrowser:$NoBrowser -Port $Port
"UI restarted at http://localhost:$Port"
