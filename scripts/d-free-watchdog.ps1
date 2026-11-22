# D: holds the Ubuntu VHDX. 2026-10-14 it hit 0 GB free and WSL could not boot (see memory reference-wsl-disk-and-memory).
# Sparse VHD mode is on, so `fstrim` returns space to D:. This script logs free space every run, trims when it is low, and
# leaves a flag file the launchers can check.
$ErrorActionPreference = 'Continue'
$dir  = 'C:\Users\rjmendez-admin\Scripts'
$log  = Join-Path $dir 'd-free-watchdog.log'
$flag = Join-Path $dir 'LOW_DISK.flag'
# warn line = the 60 GB floor of wsl-redesign-20261014.md (was 100, which kept the flag set during any restore); crit = 30 GB
$warnGB = 60; $critGB = 30
$free = [math]::Round((Get-Volume -DriveLetter D).SizeRemaining / 1GB, 1)
$line = "{0} D: free {1} GB" -f (Get-Date -Format 's'), $free
if ($free -lt $warnGB) {
  $line += " < ${warnGB} GB: trimming"
  # only if WSL is already up: never start a VM from a watchdog
  $up = (& wsl.exe --list --running 2>&1 | Out-String) -match 'Ubuntu'
  if ($up) {
    # prune regenerable build output and old logs first (wsl-cache-prune.sh ends with fstrim), then fall back to a bare fstrim
    $j = Start-Job { & wsl.exe -d Ubuntu -u root -e sh -c 'PATH=/usr/sbin:/sbin:/usr/bin:/bin; if [ -x /mnt/c/Users/rjmendez-admin/Scripts/wsl-cache-prune.sh ] || [ -f /mnt/c/Users/rjmendez-admin/Scripts/wsl-cache-prune.sh ]; then sh /mnt/c/Users/rjmendez-admin/Scripts/wsl-cache-prune.sh; else fstrim -v /; fi' 2>&1 }
    if (Wait-Job $j -Timeout 900) { $line += " | " + ((Receive-Job $j | ForEach-Object { ($_ -replace "`0",'') }) -join ' ') } else { Stop-Job $j; $line += " | fstrim timed out" }
    Remove-Job $j -Force
    $free = [math]::Round((Get-Volume -DriveLetter D).SizeRemaining / 1GB, 1)
    $line += " | now $free GB"
  } else { $line += " | WSL not running, no trim" }
}
if ($free -lt $warnGB) { Set-Content -Path $flag -Value $line } elseif (Test-Path $flag) { Remove-Item $flag -Force }
if ($free -lt $critGB) {
  try {
    if (-not [System.Diagnostics.EventLog]::SourceExists('WSLDiskWatch')) { New-EventLog -LogName Application -Source 'WSLDiskWatch' }
    Write-EventLog -LogName Application -Source 'WSLDiskWatch' -EntryType Error -EventId 4001 -Message "D: has $free GB free; the Ubuntu VHDX lives there and WSL will stop booting near 0."
  } catch {}
  & msg.exe * "D: has $free GB free (WSL VHDX). Free space or WSL stops booting." 2>$null
}
Add-Content -Path $log -Value $line
if ((Get-Item $log).Length -gt 1MB) { Get-Content $log -Tail 500 | Set-Content $log }
