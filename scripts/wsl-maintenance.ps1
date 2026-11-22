$logPath = "C:\Users\rjmendez-admin\Scripts\wsl-maintenance.log"
$compactScript = "C:\Users\rjmendez-admin\Scripts\compact_vhdx.txt"
$vhdxPath = "D:\WSL\Ubuntu\rootfs\ext4.vhdx"

function Write-Log($msg) {
    Add-Content -Path $logPath -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg)
}

Write-Log "=== Maintenance run started ==="

$before = Get-PSDrive D
Write-Log ("D: free before: {0:N2} GB" -f ($before.Free / 1GB))

Write-Log "Running fstrim inside Ubuntu..."
$trimOutput = wsl -d Ubuntu -u root -- fstrim -v / 2>&1
Write-Log ("fstrim output: {0}" -f ($trimOutput -join ' '))

Write-Log "Shutting down WSL..."
wsl --shutdown
Start-Sleep -Seconds 5

Write-Log "Compacting vhdx via diskpart..."
$diskpartOutput = diskpart /s $compactScript 2>&1
Write-Log ("diskpart output: {0}" -f ($diskpartOutput -join ' | '))

$after = Get-PSDrive D
Write-Log ("D: free after: {0:N2} GB" -f ($after.Free / 1GB))

# Bring WSL back up. Without this the distro stays shut down after the weekly
# compaction, which silently stops everything inside it (radsense-bridge, the
# k3s cluster, Loci) until something happens to invoke wsl.exe. Starting the
# distro also starts its systemd user services (lingering is enabled).
Write-Log "Restarting WSL (Ubuntu)..."
$restart = wsl -d Ubuntu -- true 2>&1
Write-Log ("restart exit code: {0} {1}" -f $LASTEXITCODE, ($restart -join ' '))

Write-Log "=== Maintenance run finished ==="
