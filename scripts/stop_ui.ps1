<#
.SYNOPSIS
  Stop the background Fin Analyst Agent UI started by start_ui.ps1.

.DESCRIPTION
  Finds this repo's `streamlit run app/ui.py` processes by command line (port
  state on Windows is unreliable for Streamlit's listener) and stops them,
  including the venv launcher shim.
#>
$repo = Split-Path -Parent $PSScriptRoot
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='streamlit.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine -like "*$repo*" -and $_.CommandLine -match 'streamlit(\.exe)?"?\s+run\s+app[\\/]ui\.py' }
if (-not $procs) { "The UI is not running."; return }
foreach ($p in $procs) {
    Stop-Process -Id $p.ProcessId -Force -Confirm:$false -ErrorAction SilentlyContinue
    "Stopped $($p.Name) (pid $($p.ProcessId))."
}
