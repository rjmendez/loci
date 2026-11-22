# Ensures the "claude" and "copilot" psmux sessions exist and are running
# their respective native CLI, self-healing via agent-loop.ps1. Idempotent —
# safe to run repeatedly (used both at logon and on a periodic safety-net
# schedule in case the psmux server itself dies). Mirrors
# /usr/local/bin/tmux-agents.sh inside WSL.

$psmux      = "C:\Users\rjmendez-admin\AppData\Local\Microsoft\WinGet\Packages\marlocarlo.psmux_Microsoft.Winget.Source_8wekyb3d8bbwe\psmux.exe"
$loopScript = "C:\Users\rjmendez-admin\Scripts\agent-loop.ps1"
$log        = "$env:USERPROFILE\.psmux-agents.log"

# psmux spawns the session shell via a raw CreateProcessW call, which cannot
# resolve a bare "pwsh.exe" through the WindowsApps App Execution Alias
# reparse point (fails with os error 2, "file not found") now that the only
# pwsh.exe on this machine is the WindowsApps/MSIX package rather than a
# traditional Program Files install. Must pass the fully-resolved path.
$realPwsh = (Get-Command pwsh.exe).Source

# If this script is ever run from inside a live claude/copilot session (as
# opposed to a clean logon/scheduled-task context), a first-time psmux server
# spawn would otherwise inherit CLAUDE_CODE_* vars and the child CLI would
# misidentify itself as a nested/child session. Strip them defensively.
Get-ChildItem Env: | Where-Object { $_.Name -match '^CLAUDE' } | ForEach-Object {
    Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
}

function Ensure-Session {
    param([string]$Name, [string]$Cmd)
    & $psmux has-session -t $Name 2>$null
    if ($LASTEXITCODE -ne 0) {
        & $psmux new-session -d -s $Name -- $realPwsh -NoLogo -NoProfile -File $loopScript -Name $Name -Cmd $Cmd
        $ts = (Get-Date).ToUniversalTime().ToString("o")
        Add-Content -Path $log -Value "$ts started missing session: $Name"
    }
}

Ensure-Session -Name "claude"  -Cmd "claude"
Ensure-Session -Name "copilot" -Cmd "copilot --yolo"
