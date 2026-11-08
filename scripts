# Start the avatar bridge.
#
#   powershell -ExecutionPolicy Bypass -File scripts\start.ps1
#
# VTube Studio should already be running with its API enabled.

param(
    [string]$Config = "config.toml"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$pythonExe = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "No .venv found. Run scripts\install.ps1 first."
}

Set-Location $root
Write-Host "starting bridge (ws://127.0.0.1:8765)..."
& $pythonExe -m dsh_vts_avatar --config $Config serve
