# Self-healing runner for a single native agent CLI inside a psmux session.
# Mirrors the WSL tmux-agents.sh pattern: if the program crashes or exits,
# it relaunches automatically, so no external watchdog is needed.
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Cmd
)

$log = "$env:USERPROFILE\.psmux-agents.log"

Set-Location $env:USERPROFILE

while ($true) {
    $ts = (Get-Date).ToUniversalTime().ToString("o")
    Add-Content -Path $log -Value "$ts launching: $Cmd ($Name)"
    Invoke-Expression $Cmd
    $ts2 = (Get-Date).ToUniversalTime().ToString("o")
    Add-Content -Path $log -Value "$ts2 $Name exited, restarting in 3s..."
    Start-Sleep -Seconds 3
}
