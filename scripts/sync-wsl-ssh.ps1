[CmdletBinding()]
param(
    [string]$WslDistro = "",
    [switch]$IncludeKnownHosts
)

$ErrorActionPreference = "Stop"

function Invoke-Wsl {
    param([string]$Command)
    if ([string]::IsNullOrWhiteSpace($WslDistro)) {
        return & wsl -e bash -lc $Command
    }
    return & wsl -d $WslDistro -e bash -lc $Command
}

function Get-WslWindowsPath {
    param([string]$LinuxPath)
    $escaped = $LinuxPath.Replace("'", "'\\''")
    $cmd = "wslpath -w '$escaped'"
    return (Invoke-Wsl $cmd | Out-String).Trim()
}

$winSshDir = Join-Path $env:USERPROFILE ".ssh"
New-Item -ItemType Directory -Path $winSshDir -Force | Out-Null

$wslHome = (Invoke-Wsl 'printf "%s" "$HOME"' | Out-String).Trim()
if ([string]::IsNullOrWhiteSpace($wslHome)) {
    throw "Unable to resolve WSL HOME."
}

$wslKey = "$wslHome/.ssh/id_ed25519"
$wslPub = "$wslHome/.ssh/id_ed25519.pub"
$wslConfig = "$wslHome/.ssh/config"
$wslKnownHosts = "$wslHome/.ssh/known_hosts"

$targets = @(
    @{ Src = $wslKey; Dst = (Join-Path $winSshDir "id_ed25519") },
    @{ Src = $wslPub; Dst = (Join-Path $winSshDir "id_ed25519.pub") }
)
if ($IncludeKnownHosts) {
    $targets += @{ Src = $wslKnownHosts; Dst = (Join-Path $winSshDir "known_hosts") }
}

foreach ($t in $targets) {
    $srcEscaped = $t.Src.Replace("'", "'\\''")
    $exists = (Invoke-Wsl "[ -f '$srcEscaped' ] && echo yes || echo no" | Out-String).Trim()
    if ($exists -ne "yes") {
        throw "Missing source file in WSL: $($t.Src)"
    }
}

foreach ($t in $targets) {
    $srcWinPath = Get-WslWindowsPath -LinuxPath $t.Src
    Copy-Item -LiteralPath $srcWinPath -Destination $t.Dst -Force
}

icacls $winSshDir /grant:r "${env:USERNAME}:(OI)(CI)F" | Out-Null
foreach ($t in $targets) {
    icacls $t.Dst /grant:r "${env:USERNAME}:F" | Out-Null
}

$winFp = (ssh-keygen -lf (Join-Path $winSshDir "id_ed25519.pub")).Trim()
$wslFp = (Invoke-Wsl "ssh-keygen -lf '$wslPub'").Trim()
if ($winFp -ne $wslFp) {
    throw "Fingerprint mismatch after sync.`nWindows: $winFp`nWSL: $wslFp"
}

$winHash = (Get-FileHash (Join-Path $winSshDir "id_ed25519") -Algorithm SHA256).Hash.ToLowerInvariant()
$wslHash = (Invoke-Wsl "sha256sum '$wslKey' | awk '{print `$1}'" | Out-String).Trim().ToLowerInvariant()
if ($winHash -ne $wslHash) {
    throw "Private key hash mismatch after sync."
}

Write-Output "WSL->Windows SSH sync OK"
Write-Output "fingerprint: $winFp"
Write-Output "hash: $winHash"
