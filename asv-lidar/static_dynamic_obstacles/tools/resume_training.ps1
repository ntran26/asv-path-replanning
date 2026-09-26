# Resume every suspended python process (the learner and its workers).
# The pair to a cooldown pause: suspension keeps the run in memory, so nothing
# is lost -- but every worker must be resumed or the learner blocks on the ones
# left suspended (F86, and seen again on 2026-09-25).
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class PR {
    [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
    [DllImport("kernel32.dll")] public static extern IntPtr OpenProcess(int a, bool i, int p);
    [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
}
"@
$stuck = Get-Process python3.10 -ErrorAction SilentlyContinue | Where-Object {
    ($_.Threads | Where-Object { $_.WaitReason -ne 'Suspended' }).Count -eq 0 }
foreach ($p in $stuck) {
    $h = [PR]::OpenProcess(0x0800, $false, $p.Id)
    if ($h -ne [IntPtr]::Zero) { [void][PR]::NtResumeProcess($h); [void][PR]::CloseHandle($h); "resumed $($p.Id)" }
}
"resumed $($stuck.Count) processes at $(Get-Date -Format 'HH:mm:ss'); clock $((Get-CimInstance Win32_Processor).CurrentClockSpeed) MHz"
