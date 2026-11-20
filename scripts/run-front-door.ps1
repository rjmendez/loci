# run-front-door.ps1: keep the Caddy front door running, starting it only when it can actually bind.
#
# Why this exists: at boot the scheduled task started caddy.exe before Tailscale had assigned the tailnet
# address, Caddy exited with "bind: The requested address is not valid in its context", and every consumer of
# the lab names (LiteLLM local tiers, embeddings, grounding, the ops_logs shipper) degraded silently.
# The task's own restart-on-failure did NOT restart it (observed 2026-10-11: exit code 1, task left "Ready"),
# so this script supervises Caddy itself instead of relying on it.
#
# Loop, until the task is stopped:
#   1. read the addresses the Caddyfile binds to (the Caddyfile is the source of truth, nothing hard-coded)
#   2. wait until each one is assigned to an interface (logs every 30 s while waiting)
#   3. check the certificate and key files the Caddyfile names exist (else log and retry in 30 s)
#   4. run caddy in the foreground; when it exits, log the exit code, back off, and go to 1
# Backoff: 5 s after a run that lasted at least 60 s, otherwise doubling from 5 s up to 60 s, so a crash
# loop cannot spin.
#
# Stopping: Task Scheduler ends this script with TerminateProcess, which runs no cleanup, and a plain child
# process would be orphaned and keep serving (observed: the new instance then failed to bind and looped, and
# a certificate renewal's Stop+Start would never have loaded the new cert). So caddy is started inside a
# Windows job object with KILL_ON_JOB_CLOSE: when this script dies the job closes and caddy dies with it.
# Any caddy.exe of the SAME path still running when the supervisor starts one (an orphan from before this
# was added) is stopped first, and logged.
#
# -Check does steps 1-3 once and exits (0 ok, 2 address never appeared within -WaitSeconds, 3 cert files
# missing) without starting Caddy; it exists for testing.
param(
    [string]$Caddyfile = 'F:\services\tls\caddy\Caddyfile',
    [string]$CaddyExe = 'F:\services\tls\caddy\caddy.exe',
    [int]$WaitSeconds = 900,
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
$log = 'F:\services\tls\logs\front-door-wrapper.log'
function Log([string]$m) {
    try { Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date -Format 's'), $m) } catch { }
    Write-Host $m
}

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class FrontDoorJob {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern IntPtr CreateJobObject(IntPtr attrs, string name);
    [DllImport("kernel32.dll")] static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint len);
    [DllImport("kernel32.dll")] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [StructLayout(LayoutKind.Sequential)] struct IoCounters { public ulong a, b, c, d, e, f; }
    [StructLayout(LayoutKind.Sequential)] struct Basic {
        public long PerProcessUserTimeLimit; public long PerJobUserTimeLimit; public uint LimitFlags;
        public UIntPtr MinWs; public UIntPtr MaxWs; public uint ActiveProcessLimit; public UIntPtr Affinity;
        public uint PriorityClass; public uint SchedulingClass; }
    [StructLayout(LayoutKind.Sequential)] struct Extended {
        public Basic Basic; public IoCounters Io; public UIntPtr ProcessMemoryLimit; public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed; public UIntPtr PeakJobMemoryUsed; }
    static IntPtr job = IntPtr.Zero;
    public static bool Init() {
        job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) return false;
        var info = new Extended();
        info.Basic.LimitFlags = 0x2000;   // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        int len = Marshal.SizeOf(typeof(Extended));
        IntPtr p = Marshal.AllocHGlobal(len);
        try { Marshal.StructureToPtr(info, p, false); return SetInformationJobObject(job, 9, p, (uint)len); }
        finally { Marshal.FreeHGlobal(p); }
    }
    public static bool Add(IntPtr process) { return job != IntPtr.Zero && AssignProcessToJobObject(job, process); }
}
'@

function Read-Caddyfile {
    $text = Get-Content -LiteralPath $Caddyfile -Raw
    [pscustomobject]@{
        Binds = @([regex]::Matches($text, '(?m)^\s*bind\s+(\S+)') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
        Files = @([regex]::Matches($text, '(?m)^\s*tls\s+(\S+)\s+(\S+)') | ForEach-Object { $_.Groups[1].Value; $_.Groups[2].Value } | Sort-Object -Unique)
    }
}

function Missing-Addresses([string[]]$binds) {
    @($binds | Where-Object {
        -not (Get-NetIPAddress -IPAddress $_ -ErrorAction SilentlyContinue | Where-Object { $_.AddressState -eq 'Preferred' })
    })
}

# Waits for the addresses. $limit = $null waits forever. Returns $true when all are present.
function Wait-Addresses([string[]]$binds, $limit) {
    $deadline = if ($null -ne $limit) { (Get-Date).AddSeconds($limit) } else { $null }
    $lastNote = Get-Date
    while ($true) {
        $missing = Missing-Addresses $binds
        if ($missing.Count -eq 0) { return $true }
        if ($null -ne $deadline -and (Get-Date) -ge $deadline) {
            Log "ERROR: address(es) not assigned after ${limit}s: $($missing -join ',')"
            return $false
        }
        if (((Get-Date) - $lastNote).TotalSeconds -ge 30) { Log "waiting for address(es): $($missing -join ',')"; $lastNote = Get-Date }
        Start-Sleep -Seconds 2
    }
}

function Absent-Files([string[]]$files) {
    @($files | Where-Object { $_ -match '^[A-Za-z]:[\\/]' -and -not (Test-Path -LiteralPath $_) })
}

if ($Check) {
    $cfg = Read-Caddyfile
    Log "check: binds=[$($cfg.Binds -join ',')] certfiles=$($cfg.Files.Count) wait<=${WaitSeconds}s"
    if (-not (Wait-Addresses $cfg.Binds $WaitSeconds)) { exit 2 }
    $absent = Absent-Files $cfg.Files
    if ($absent.Count) { Log "ERROR: certificate file(s) missing: $($absent -join '; ')"; exit 3 }
    Log 'check ok (caddy not started)'
    exit 0
}

if (-not [FrontDoorJob]::Init()) { Log 'WARNING: could not create the job object; caddy may outlive a task stop' }
$backoff = 5
while ($true) {
    $cfg = Read-Caddyfile
    Log "supervisor: binds=[$($cfg.Binds -join ',')] certfiles=$($cfg.Files.Count)"
    [void](Wait-Addresses $cfg.Binds $null)
    Log "addresses present: $($cfg.Binds -join ',')"

    $absent = Absent-Files $cfg.Files
    if ($absent.Count) {
        Log "ERROR: certificate file(s) missing: $($absent -join '; '); retrying in 30 s"
        Start-Sleep -Seconds 30
        continue
    }

    # an orphan of the same binary (started before the job object existed) would hold the port
    foreach ($stray in @(Get-Process -Name caddy -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $CaddyExe })) {
        Log "stopping stray caddy pid $($stray.Id) (not owned by this supervisor)"
        Stop-Process -Id $stray.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 500
    }

    Log "starting caddy: $CaddyExe run --config $Caddyfile"
    $started = Get-Date
    $proc = Start-Process -FilePath $CaddyExe -ArgumentList @('run', '--config', "`"$Caddyfile`"", '--adapter', 'caddyfile') `
            -WorkingDirectory (Split-Path -Parent $Caddyfile) -NoNewWindow -PassThru
    if (-not [FrontDoorJob]::Add($proc.Handle)) { Log "WARNING: could not add caddy pid $($proc.Id) to the job object" }
    $proc.WaitForExit()
    $code = $proc.ExitCode
    $ran = [int]((Get-Date) - $started).TotalSeconds
    Log "caddy exited with code $code after ${ran}s"

    if ($ran -ge 60) { $backoff = 5 } else { $backoff = [Math]::Min($backoff * 2, 60) }
    Log "restarting in ${backoff}s"
    Start-Sleep -Seconds $backoff
}
