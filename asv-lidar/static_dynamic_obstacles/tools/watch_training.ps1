<#
.SYNOPSIS
  Progress bar for the baseline-v3 training run in progress (Windows PowerShell 5.1 or later).

.DESCRIPTION
  Shows steps done out of 3.0 M, speed over the last 30 minutes, curriculum stage, ETA, and the
  latest and best validation-set scores (goal - 2 x collision, safety off). Without -Algo/-Seed
  it follows the most recently written run log, so it moves on to the next run by itself.
  Reads only the end of the training log and the small eval_summary.json; refreshes every
  -Interval seconds (default 60). Stop with Ctrl+C.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\watch_training.ps1
  powershell -ExecutionPolicy Bypass -File tools\watch_training.ps1 -Algo tqc -Seed 1
  powershell -ExecutionPolicy Bypass -File tools\watch_training.ps1 -Once
#>
param(
    [string]$Algo = '',
    [int]$Seed = -1,
    [int]$Interval = 60,
    [int]$Total = 3000000,
    [switch]$Once
)

$root = Split-Path -Parent $PSScriptRoot
$runs = Join-Path $root 'runs'
$milestones = Join-Path $root 'results\train_seed.log'
$stageStarts = @(0, 160000, 360000, 640000, 1000000, 1500000, 2000000)

function Get-RunLog {
    if ($Algo -and $Seed -ge 0) {
        return Join-Path $runs ('{0}_formulation_seed{1}_bl3.log' -f $Algo, $Seed)
    }
    Get-ChildItem -Path $runs -Filter '*_formulation_seed*_bl3.log' |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
}

function Format-Score([double]$s) {
    if ($s -ge 0) { return ('+{0:0.000}' -f $s) }
    return ('{0:0.000}' -f $s)
}

do {
    $log = Get-RunLog
    if (-not $log -or -not (Test-Path $log)) {
        Write-Host 'No run log found under runs\.'
        if ($Once) { break }
        Start-Sleep -Seconds $Interval
        continue
    }
    $name = [IO.Path]::GetFileNameWithoutExtension($log)          # e.g. tqc_formulation_seed1_bl3
    $runDir = Join-Path $runs $name

    # Last (timesteps, time_elapsed) pairs from the end of the log.
    $pairs = New-Object System.Collections.Generic.List[object]
    $elapsed = $null
    foreach ($l in (Get-Content -Path $log -Tail 4000)) {
        if ($l -match '\|\s+time_elapsed\s+\|\s+(\d+)') { $elapsed = [int]$matches[1] }
        elseif ($l -match '\|\s+total_timesteps\s+\|\s+(\d+)' -and $null -ne $elapsed) {
            $pairs.Add(@([int]$matches[1], $elapsed))
        }
    }
    if ($pairs.Count -eq 0) {
        Write-Host ('{0}: no progress lines yet' -f $name)
        if ($Once) { break }
        Start-Sleep -Seconds $Interval
        continue
    }
    $last = $pairs[$pairs.Count - 1]
    $steps = $last[0]
    $speed = 0.0
    $first = $null
    foreach ($p in $pairs) {
        if ($p[1] -ge $last[1] - 1800 -and $p[1] -lt $last[1] -and $p[0] -lt $steps) { $first = $p; break }
    }
    if ($first) { $speed = ($steps - $first[0]) / [double]($last[1] - $first[1]) }

    $stage = 1
    for ($i = 0; $i -lt $stageStarts.Count; $i++) { if ($steps -ge $stageStarts[$i]) { $stage = $i + 1 } }
    $pct = [math]::Min(100, [math]::Round(100.0 * $steps / $Total, 1))
    if ($speed -gt 0) {
        $secs = ($Total - $steps) / $speed
        $eta = (Get-Date).AddSeconds($secs).ToString('ddd dd MMM HH:mm')
    } else { $secs = -1; $eta = 'unknown' }

    # Validation-set checks (safety off) from eval_summary.json.
    $evalText = 'no validation-set check yet'
    $summary = Join-Path $runDir 'eval_summary.json'
    if (Test-Path $summary) {
        $evals = @(Get-Content -Raw -Path $summary | ConvertFrom-Json) |
            Where-Object { $_.safety -eq 'off' -or $_.supervisor -eq 'off' }
        if ($evals.Count -gt 0) {
            $latest = $evals[$evals.Count - 1]
            $ls = $latest.goal_rate - 2 * $latest.collision_rate
            $best = $evals | Sort-Object { $_.goal_rate - 2 * $_.collision_rate } -Descending | Select-Object -First 1
            $bs = $best.goal_rate - 2 * $best.collision_rate
            $evalText = ('latest check {0} at {1:N0} (decoupled {2:0.00}, coupled {3:0.00}); best {4} at {5:N0}' -f
                (Format-Score $ls), $latest.timesteps, $latest.'dev/goal', $latest.'field/goal',
                (Format-Score $bs), $best.timesteps)
        }
    }

    $done = Test-Path (Join-Path $runDir 'final_model.zip')
    $activity = ('{0}: {1:N0} / {2:N0} steps' -f $name, $steps, $Total)
    if ($done) {
        $status = 'training finished; post-training evaluations or the next run follow'
        $pct = 100
    } else {
        $status = ('{0}% | {1:0.0} steps/s (last 30 min) | stage {2} of 7 | ETA {3}' -f $pct, $speed, $stage, $eta)
    }
    $progress = @{ Activity = $activity; Status = $status; PercentComplete = [int]$pct; CurrentOperation = $evalText }
    if ($secs -gt 0 -and -not $done) { $progress.SecondsRemaining = [int]$secs }
    Write-Progress @progress

    Write-Host ('[{0}] {1} | {2}' -f (Get-Date).ToString('HH:mm'), $activity, $status)
    Write-Host ('        {0}' -f $evalText)
    if ($done -and (Test-Path $milestones)) {
        Write-Host ('        last milestone: {0}' -f (Get-Content -Path $milestones -Tail 1))
    }

    if ($Once) { break }
    Start-Sleep -Seconds $Interval
} while ($true)
