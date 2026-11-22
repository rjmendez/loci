# Switch which GPUs the Windows Ollama server may use. Needs an elevated shell (edits the SYSTEM task "Ollama").
#   .\ollama-gpu.ps1 status
#   .\ollama-gpu.ps1 pin     # RTX 2080 Ti only (default state): leaves the 4070 Ti free for tests and games
#   .\ollama-gpu.ps1 both    # both cards, for the huge models (gemma4:26b, qwen3.8); the 4070 Ti is then shared
# Vulkan is always off: Ollama would otherwise reach the 4070 Ti through Vulkan whatever CUDA_VISIBLE_DEVICES says.
# Restarts the server (about 5 s); every change backs up the task definition first. Rollback: Register-ScheduledTask
# -Xml from the newest Ollama.task.xml.bak-* in this folder.
param([Parameter(Mandatory)][ValidateSet('status', 'pin', 'both')][string]$Mode)

$ErrorActionPreference = 'Stop'
$u2080 = 'GPU-0a10e32f-d9c0-cdaf-9d49-fa91a000af04'
$u4070 = 'GPU-371c24ef-483f-051c-e75b-b2de6a76a1f7'
$exe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $exe)) { $exe = 'C:\Users\rjmendez-admin\AppData\Local\Programs\Ollama\ollama.exe' }

function Show-State {
    $a = (Get-ScheduledTask -TaskName Ollama).Actions.Arguments
    $devs = if ($a -match 'CUDA_VISIBLE_DEVICES=([^"]+)"') { $Matches[1] } else { '(unset: all GPUs)' }
    $mode = if ($devs -eq $u2080) { 'pin (2080 Ti only)' } elseif ($devs -like "*$u2080*$u4070*") { 'both' } else { 'custom/unset' }
    "mode: $mode"; "devices: $devs"; "vulkan off: $($a -match 'OLLAMA_VULKAN=0')"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
}

if ($Mode -eq 'status') { Show-State; return }

$devs = if ($Mode -eq 'pin') { $u2080 } else { "$u2080,$u4070" }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
Export-ScheduledTask -TaskName Ollama | Out-File "$PSScriptRoot\Ollama.task.xml.bak-$stamp" -Encoding utf8
$args2 = "/c set `"OLLAMA_HOST=0.0.0.0:11434`" && set `"OLLAMA_VULKAN=0`" && set `"CUDA_VISIBLE_DEVICES=$devs`" && $exe serve"
Set-ScheduledTask -TaskName Ollama -Action (New-ScheduledTaskAction -Execute 'cmd' -Argument $args2) | Out-Null
Stop-ScheduledTask -TaskName Ollama
Start-Sleep 4
Start-ScheduledTask -TaskName Ollama
$up = $false
foreach ($i in 1..20) { Start-Sleep 2; try { Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 3 | Out-Null; $up = $true; break } catch {} }
if (-not $up) { Write-Warning "server did not come back within 40 s; roll back from the .bak file" }
Show-State
