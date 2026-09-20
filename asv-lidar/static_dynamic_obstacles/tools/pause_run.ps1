# Pause or resume a training run in place (suspends its learner and workers).
#
#   powershell -ExecutionPolicy Bypass -File tools\pause_run.ps1 -Tag v10 -Action pause
#   powershell -ExecutionPolicy Bypass -File tools\pause_run.ps1 -Tag v10 -Action resume
#
# Suspension keeps everything in memory, so a resumed run continues exactly where
# it stopped. It does not survive a reboot or sign-out: after one, continue from
# the last checkpoint with `train_formulation.py --resume` instead.
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [Parameter(Mandatory = $true)][ValidateSet("pause", "resume", "status")][string]$Action
)

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ProcCtl {
    [DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
    [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
    [DllImport("kernel32.dll")] public static extern IntPtr OpenProcess(int access, bool inherit, int pid);
    [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
}
"@

$learner = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'python' -and $_.CommandLine -match 'train_formulation\.py' -and $_.CommandLine -match "--tag $Tag(\s|$)"
}
if (-not $learner) { Write-Output "no running learner for tag $Tag"; exit 1 }
$ids = @()
foreach ($l in $learner) {
    $ids += $l.ProcessId
    $ids += (Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $l.ProcessId } | ForEach-Object { $_.ProcessId })
}

if ($Action -eq "status") {
    foreach ($id in $ids) {
        $p = Get-Process -Id $id -ErrorAction SilentlyContinue
        $suspended = ($p.Threads | Where-Object { $_.WaitReason -ne 'Suspended' }).Count -eq 0
        "{0,6} {1}" -f $id, ($(if ($suspended) { "suspended" } else { "running" }))
    }
    exit 0
}

$PROCESS_SUSPEND_RESUME = 0x0800
foreach ($id in $ids) {
    $h = [ProcCtl]::OpenProcess($PROCESS_SUSPEND_RESUME, $false, $id)
    if ($h -eq [IntPtr]::Zero) { Write-Output "cannot open $id"; continue }
    $rc = if ($Action -eq "pause") { [ProcCtl]::NtSuspendProcess($h) } else { [ProcCtl]::NtResumeProcess($h) }
    [void][ProcCtl]::CloseHandle($h)
    Write-Output ("{0} {1} (status {2})" -f $Action, $id, $rc)
}
