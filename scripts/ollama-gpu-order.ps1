# Keep the RTX 4070 Ti free of default Ollama placement, switchable on and off.
#   .\ollama-gpu-order.ps1 status
#   .\ollama-gpu-order.ps1 free4070 [-DryRun]   # server sees ONLY the 2080 Ti (device 0); Loci pins use 0
#   .\ollama-gpu-order.ps1 default  [-DryRun]   # both cards, Ollama's own order (device 0 = 4070 Ti, 1 = 2080 Ti); pins use 1
# Why hide the card: with both visible, a request WITHOUT main_gpu goes to whichever card has the most free VRAM
# (measured 2026-10-12: reordering did not keep default traffic off the 4070 Ti), so only hiding it keeps it free.
# Consequence in free4070: main_gpu=1 is a HTTP 500 (measured), and models that need both cards (gemma4:26b,
# qwen3.8-35b/27b) do not fit one card: switch back to 'default' for those.
#
# What it changes (both files are backed up first as <file>.bak-gpuorder-<stamp>):
#   1. ollama_serve.bat      a marked block with OLLAMA_VULKAN=0 (Vulkan ignores CUDA_VISIBLE_DEVICES) and
#                            CUDA_VISIBLE_DEVICES=<2080 Ti>.
#   2. ~/.loci/backends.toml `gpu = N` pool pins and `gpu_vram_gb` swapped to the new numbering. Loci re-reads the
#                            file on change (PR #466), no Loci restart.
# Then it restarts ONLY the Ollama server that listens on :11440 (and its runners) and starts task OllamaServe.
# Loaded models drop and reload on the next request (about 10-30 s). The resident proxy is not touched.
# Rollback: copy the .bak-gpuorder-* files back, or run this script with the opposite mode.
param(
    [Parameter(Mandatory)][ValidateSet('status', 'free4070', 'default')][string]$Mode,
    [switch]$DryRun,
    [switch]$NoRestart
)
$ErrorActionPreference = 'Stop'
$bat = 'C:\Users\rjmendez-admin\ollama_serve.bat'
$toml = 'C:\Users\rjmendez-admin\.loci\backends.toml'
$log = 'C:\Users\rjmendez-admin\ollama_serve.log'
$u2080 = 'GPU-0a10e32f-d9c0-cdaf-9d49-fa91a000af04'
$u4070 = 'GPU-371c24ef-483f-051c-e75b-b2de6a76a1f7'
$begin = 'rem gpu-order-begin'
$end = 'rem gpu-order-end'

function Current-Mode {
    $t = Get-Content $bat -Raw
    if ($t -match [regex]::Escape($begin) + '(?s)(.*?)' + [regex]::Escape($end)) {
        if ($Matches[1] -match 'CUDA_VISIBLE_DEVICES=' + [regex]::Escape($u2080) + ',' + [regex]::Escape($u4070)) { return 'ordered' }   # legacy reorder, same pin numbering as free4070
        if ($Matches[1] -match 'CUDA_VISIBLE_DEVICES=' + [regex]::Escape($u2080)) { return 'free4070' }
    }
    return 'default'
}
function Show-State {
    "mode (ollama_serve.bat): $(Current-Mode)"
    $g = (Select-String -Path $toml -Pattern '^\s*gpu\s*=\s*([01])' | ForEach-Object { $_.Matches[0].Groups[1].Value } | Group-Object | ForEach-Object { "gpu=$($_.Name) x$($_.Count)" }) -join ', '
    "Loci pool pins: $g"
    (Select-String -Path $toml -Pattern '^\s*gpu_vram_gb').Line
    $own = (Get-NetTCPConnection -LocalPort 11440 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess
    "server on :11440: pid $own"
    try {
        $ps = Invoke-RestMethod http://127.0.0.1:11440/api/ps -TimeoutSec 4
        "loaded: " + (($ps.models | ForEach-Object { $_.name }) -join ', ')
    } catch { "api/ps: $($_.Exception.Message)" }
    Get-Content $log -Tail 20000 | Select-String 'msg="inference compute"' | Select-Object -Last 2 | ForEach-Object {
        if ($_.Line -match 'id=(\d).*description="([^"]+)"') { "  Ollama device $($Matches[1]) = $($Matches[2])" } }
}

if ($Mode -eq 'status') { Show-State; return }
$cur = Current-Mode
if ($cur -eq $Mode) { "already in mode '$Mode' (nothing to change)"; Show-State; return }

# ---- plan
# Loci pins mean "the 2080 Ti": index 1 in 'default' (both cards, 4070 Ti first), index 0 when the 2080 Ti is first/only.
$idx = @{ default = '1'; ordered = '0'; free4070 = '0' }
$flipFrom, $flipTo = $idx[$cur], $idx[$Mode]
$doFlip = ($flipFrom -ne $flipTo)
$pins = if ($doFlip) { @(Select-String -Path $toml -Pattern "^\s*gpu\s*=\s*$flipFrom\b") } else { @() }
$vram = if ($Mode -eq 'default') { 'gpu_vram_gb = { 0 = 12.0, 1 = 11.0 }' } else { 'gpu_vram_gb = { 0 = 11.0, 1 = 12.0 }' }
"plan: $cur -> $Mode"
"  ollama_serve.bat : $(if ($Mode -eq 'free4070') { 'CUDA_VISIBLE_DEVICES = 2080 Ti only (4070 Ti hidden from Ollama)' } else { 'remove the CUDA_VISIBLE_DEVICES block (both cards, Ollama order)' })"
"  backends.toml    : $(if ($doFlip) { "flip $($pins.Count) pool pins gpu=$flipFrom -> gpu=$flipTo; $vram" } else { 'pins already use gpu=' + $flipTo + ' (no change)' })"
"  restart          : Ollama on :11440 + its runners, then task OllamaServe $(if ($NoRestart) { '(SKIPPED: -NoRestart)' })"
if ($DryRun) { "dry run: nothing changed"; return }

# ---- backups
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
Copy-Item $bat "$bat.bak-gpuorder-$stamp"
Copy-Item $toml "$toml.bak-gpuorder-$stamp"
"backups: *.bak-gpuorder-$stamp"

# ---- bat
$t = Get-Content $bat -Raw
$block = if ($Mode -eq 'free4070') { "$begin`r`nset OLLAMA_VULKAN=0`r`nset CUDA_VISIBLE_DEVICES=$u2080`r`n$end" } else { "$begin`r`n$end" }
if ($t -match '(?s)' + [regex]::Escape($begin) + '.*?' + [regex]::Escape($end)) {
    $t = [regex]::Replace($t, '(?s)' + [regex]::Escape($begin) + '.*?' + [regex]::Escape($end), { param($m) $block })
} else {
    $t = $t -replace '(set OLLAMA_DEBUG=1\r?\n)', "`$1$block`r`n"
}
if ($t -notmatch [regex]::Escape($begin)) { throw "could not place the gpu-order block in $bat (no 'set OLLAMA_DEBUG=1' anchor); restore from backup" }
[IO.File]::WriteAllText($bat, $t, [Text.Encoding]::ASCII)

# ---- toml (line-wise; only `gpu = N` lines and the gpu_vram_gb line)
if ($doFlip) {
    $lines = Get-Content $toml
    $out = foreach ($ln in $lines) {
        if ($ln -match '^(\s*gpu\s*=\s*)([01])(\b.*)$') {
            "$($Matches[1])$flipTo$($Matches[3])"
        } elseif ($ln -match '^(\s*)gpu_vram_gb\s*=\s*\{[^}]*\}(.*)$') {
            "$($Matches[1])$vram$($Matches[2])"
        } else { $ln }
    }
    # no BOM: Windows PowerShell 5.1 Set-Content -Encoding UTF8 adds one, and tomllib (Loci, ops_logs shipper) rejects it
    [IO.File]::WriteAllText($toml, (($out -join "`n") + "`n"), (New-Object System.Text.UTF8Encoding($false)))
}
"edited bat$(if ($doFlip) { ' and backends.toml' })"
if ($NoRestart) { "restart skipped; the running server still has the old order"; Show-State; return }

# ---- restart only the :11440 server and its runners
$srv = (Get-NetTCPConnection -LocalPort 11440 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess
if ($srv) {
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$srv"
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$srv" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Stop-Process -Id $srv -Force
    Stop-Process -Id $p.ParentProcessId -Force -ErrorAction SilentlyContinue
    Start-Sleep 3
}
if ((Get-ScheduledTask -TaskName OllamaServe).State -eq 'Running') { Stop-ScheduledTask -TaskName OllamaServe; Start-Sleep 2 }
Start-ScheduledTask -TaskName OllamaServe
$up = $false
foreach ($i in 1..30) { Start-Sleep 2; try { Invoke-RestMethod http://127.0.0.1:11440/api/tags -TimeoutSec 3 | Out-Null; $up = $true; break } catch {} }
if (-not $up) { Write-Warning "server did not answer on :11440 within 60 s; roll back from the .bak-gpuorder-$stamp files"; return }
Start-Sleep 3
Show-State
