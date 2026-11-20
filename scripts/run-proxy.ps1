# run-proxy.ps1: supervise the Ollama resident proxy (start.bat -> python proxy.py).
#
# Why: the task scheduler's restart-on-failure was observed NOT to restart a failed service on this machine
# (2026-10-11: Caddy exited 1 and the task stayed "Ready"), and a proxy that is alive but hung would never
# exit at all. Every consumer of the local models goes through this proxy, directly or via the lab name.
#
# Loop, until the task is stopped:
#   1. stop a stray proxy of this install still holding the port (one not owned by this supervisor)
#   2. run start.bat as a child inside a KILL_ON_JOB_CLOSE job object: when this script dies (task stop,
#      hard kill) cmd and python die with it, so Stop+Start always yields a fresh proxy
#   3. watch it: restart when the process exits, or when GET /health/models fails -ProbeFailures times in a
#      row (-ProbeSeconds apart), i.e. it is listening but not answering
#   4. back off before restarting: 5 s after a run of at least 120 s, otherwise doubling up to 60 s
# Log: supervisor.log next to this script. Every restart says why.
#
# Test knobs: -StartBat / -Port / -ProbeSeconds / -ProbeFailures let a throwaway instance run on another
# port (the production task uses the defaults). -Check validates paths and exits.
param(
    [string]$StartBat = 'F:\services\ollama-resident-proxy\start.bat',
    [int]$Port = 11435,
    [int]$ProbeSeconds = 30,
    [int]$ProbeFailures = 3,
    [int]$ProbeTimeoutSeconds = 10,
    [int]$GraceSeconds = 20,
    [string]$Log = 'F:\services\ollama-resident-proxy\supervisor.log',
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
function Log([string]$m) {
    try { Add-Content -Path $Log -Value ("{0} {1}" -f (Get-Date -Format 's'), $m) } catch { }
    Write-Host $m
}

if (-not (Test-Path -LiteralPath $StartBat)) { Log "ERROR: $StartBat not found"; exit 3 }
if ($Check) { Log "check ok: $StartBat port=$Port probe=${ProbeSeconds}s x$ProbeFailures"; exit 0 }

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ProxyJob {
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

function Test-Health {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health/models" -UseBasicParsing -TimeoutSec $ProbeTimeoutSeconds
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}

# a listener on $Port that is this install's proxy (matched by its command line), not owned by this supervisor
function Stop-Stray {
    $pids = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | ForEach-Object { $_.OwningProcess } | Sort-Object -Unique)
    foreach ($p in $pids) {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$p" -ErrorAction SilentlyContinue).CommandLine
        if ($cmd -and $cmd -match 'ollama-resident-proxy') {
            Log "stopping stray proxy pid $p holding port $Port (not owned by this supervisor)"
            & taskkill.exe /PID $p /T /F *> $null
            Start-Sleep -Seconds 1
        } elseif ($cmd) {
            Log "WARNING: port $Port is held by pid $p which is not this proxy: $($cmd.Substring(0, [Math]::Min(120, $cmd.Length)))"
        }
    }
}

if (-not [ProxyJob]::Init()) { Log 'WARNING: could not create the job object; the proxy may outlive a task stop' }
$backoff = 5
while ($true) {
    Stop-Stray
    Log "starting $StartBat (port $Port)"
    $started = Get-Date
    $proc = Start-Process -FilePath $env:ComSpec -ArgumentList @('/c', "`"$StartBat`"") -NoNewWindow -PassThru
    if (-not [ProxyJob]::Add($proc.Handle)) { Log "WARNING: could not add pid $($proc.Id) to the job object" }

    $reason = $null
    $fails = 0
    $nextProbe = (Get-Date).AddSeconds($GraceSeconds)     # let it bind before the first probe
    while ($true) {
        if ($proc.WaitForExit(2000)) { $reason = "process exited with code $($proc.ExitCode)"; break }
        if ((Get-Date) -lt $nextProbe) { continue }
        $nextProbe = (Get-Date).AddSeconds($ProbeSeconds)
        if (Test-Health) {
            if ($fails -gt 0) { Log "health probe recovered after $fails failure(s)" }
            $fails = 0
        } else {
            $fails++
            Log "health probe failed ($fails/$ProbeFailures)"
            if ($fails -ge $ProbeFailures) { $reason = "health probe failed $fails times in a row (listening but not answering)"; break }
        }
    }

    Log "restarting: $reason"
    if (-not $proc.HasExited) { & taskkill.exe /PID $proc.Id /T /F *> $null }
    $ran = [int]((Get-Date) - $started).TotalSeconds
    if ($ran -ge 120) { $backoff = 5 } else { $backoff = [Math]::Min($backoff * 2, 60) }
    Log "ran ${ran}s; waiting ${backoff}s"
    Start-Sleep -Seconds $backoff
}
