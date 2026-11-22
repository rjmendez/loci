# home-git-guard.ps1 -- keeps the home-folder git repo from ever exhausting Windows commit again.
#
# Root cause (2026-10-11 .. 10-16): C:\Users\<you> is a git repo with 7 tracked files and ~2000 untracked top-level entries.
# Claude Code runs `git ls-files --others --exclude-standard` in the session folder; against the home repo that walked the whole
# profile, held ~18 GB of commit per process, and orphaned copies piled up until commit hit 100% (lightcraft/Ollama/git aborts,
# 0xc0000409, "not enough memory"). The fix is in .git/info/exclude (home-git-guard block: only the 7 tracked paths are visible).
# This script is the safety net:
#   1. re-asserts that fix: if the untracked listing is not tiny or does not finish fast, regenerate the block;
#   2. kills any git.exe running `ls-files --others` for more than $MaxAgeS seconds (a healthy one takes < 1 s) or holding > $MaxGB.
# It logs every action (and only actions) to home-git-guard.log. Run with -DryRun to see what it would do.
param([switch]$DryRun, [int]$MaxAgeS = 300, [double]$MaxGB = 2.0)
$ErrorActionPreference = 'Continue'
$home_ = $env:USERPROFILE
$log = Join-Path $PSScriptRoot 'home-git-guard.log'
function Log($m) { $line = "{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m; if (-not $DryRun) { Add-Content -LiteralPath $log -Value $line } ; Write-Output $line }

# 1. leaked listers
$now = Get-Date
foreach ($p in Get-CimInstance Win32_Process -Filter "Name='git.exe'") {
    $cl = [string]$p.CommandLine
    if ($cl -notmatch 'ls-files' -or $cl -notmatch '--others') { continue }
    $age = ($now - $p.CreationDate).TotalSeconds
    $gb = ((Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue).PrivateMemorySize64) / 1GB
    $parentAlive = [bool](Get-Process -Id $p.ParentProcessId -ErrorAction SilentlyContinue)
    if ($age -gt $MaxAgeS -or $gb -gt $MaxGB) {
        Log ("KILL git pid {0} age {1:N0}s commit {2:N1} GB parentAlive={3} cmd={4}" -f $p.ProcessId, $age, $gb, $parentAlive, ($cl -replace '\s+', ' ').Substring(0, [Math]::Min(160, ($cl -replace '\s+', ' ').Length)))
        if (-not $DryRun) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    }
}

# 2. the exclusion must still hold
$ex = Join-Path $home_ '.git\info\exclude'
$ok = $false
if (Test-Path -LiteralPath $ex) {
    $txt = Get-Content -LiteralPath $ex -Raw
    if ($txt -match 'home-git-guard-begin' -and $txt -match 'home-git-guard-end') {
        $psi = [Diagnostics.ProcessStartInfo]::new('git', "-C `"$home_`" -c core.hooksPath=NUL ls-files --others --exclude-standard")
        $psi.RedirectStandardOutput = $true; $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
        $pr = [Diagnostics.Process]::Start($psi); $t = $pr.StandardOutput.ReadToEndAsync()
        if ($pr.WaitForExit(30000)) { $n = @(($t.Result -split "`n") | Where-Object { $_ }).Count; $ok = ($n -le 25) } else { $pr.Kill($true); Log 'listing did not finish in 30 s' }
    }
}
if (-not $ok) {
    Log 'exclusion missing or ineffective: regenerating the home-git-guard block'
    if (-not $DryRun) { & python -I (Join-Path $PSScriptRoot 'home-git-guard-generate.py') 2>&1 | ForEach-Object { Log "generate: $_" } }
}
