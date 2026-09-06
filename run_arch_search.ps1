<#
.SYNOPSIS
  Runs architecture_search_strategies.py `run` for one or more config folders (sequentially),
  optionally followed by `populate` + build_search_report.py.

.DESCRIPTION
  Every config folder under -ConfigRoot must contain f_architectures.csv and h_architectures.csv.
  All runs share the SAME --models_dir, so combos already trained (same f/h/strategy AND same
  hyperparameters -> same hp_hash) are skipped and reused. Keep the hyperparameter flags equal
  to the previous sweep (defaults below == hp_hash e42bf99b) or the reuse won't kick in.

.EXAMPLE
  .\run_arch_search.ps1
      # runs configs/arch_search_6to8 then configs/arch_search_9to16, 32 workers

.EXAMPLE
  .\run_arch_search.ps1 -Configs arch_search_6to8 -Workers 16

.EXAMPLE
  .\run_arch_search.ps1 -Populate -LooGmTop 50
      # runs both configs, then rebuilds results.csv (with gm) + search_report.html

.EXAMPLE
  .\run_arch_search.ps1 -GmSmoothing none -ModelsDir C:\...\archsearch_models_gmnone -Gm1Weight 0.075
      # gm1 target with NO smoothing -> hp_hash differs from savgol, so use a fresh -ModelsDir
      # (loop this over your gm1_weight values). -GmSmoothing savgol (default) keeps the old hash.
#>
[CmdletBinding()]
param(
    [string[]] $Configs        = @('arch_search_6to8', 'arch_search_9to16'),
    [int]      $Workers        = 32,
    [int]      $Epochs         = 1000,
    [int]      $DeltaBaseEpochs = 1000,
    [double]   $Gm1Weight      = 0.0,
    [ValidateSet('savgol', 'cascade', 'none')]
    [string]   $GmSmoothing    = 'savgol',
    [string]   $ConfigRoot     = 'configs',
    [string]   $ModelsDir      = 'C:\Users\acost\repos\transistor_modeling_hyper_outputs\archsearch_models',
    [string]   $CsvDir         = 'C:\Users\acost\repos\csvs',
    [string]   $Python         = 'python',
    [string[]] $ExtraRunArgs   = @(),
    [switch]   $Populate,
    [int]      $LooGmTop       = 50
)

$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'architecture_search_strategies.py'
if (-not (Test-Path $script)) { throw "architecture_search_strategies.py not found next to this .ps1 ($script)" }

# --- resolve + validate every config folder up front ---
$runs = foreach ($c in $Configs) {
    $dir = if (Test-Path $c) { (Resolve-Path $c).Path } else { Join-Path $PSScriptRoot (Join-Path $ConfigRoot $c) }
    $fcsv = Join-Path $dir 'f_architectures.csv'
    $hcsv = Join-Path $dir 'h_architectures.csv'
    if (-not (Test-Path $fcsv)) { throw "missing $fcsv" }
    if (-not (Test-Path $hcsv)) { throw "missing $hcsv" }
    [pscustomobject]@{ Name = $c; FCsv = $fcsv; HCsv = $hcsv }
}

Write-Host "models_dir : $ModelsDir"
Write-Host "configs    : $($runs.Name -join ', ')"
Write-Host "hyperparams: --epochs $Epochs --delta_base_epochs $DeltaBaseEpochs --gm1_weight $Gm1Weight --gm_smoothing $GmSmoothing  (--workers $Workers)"
Write-Host ''

$overall = [System.Diagnostics.Stopwatch]::StartNew()
foreach ($r in $runs) {
    Write-Host "==== run: $($r.Name) ====" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $runArgs = @(
        $script, 'run',
        '--workers', $Workers,
        '--epochs', $Epochs,
        '--delta_base_epochs', $DeltaBaseEpochs,
        '--gm1_weight', $Gm1Weight,
        '--gm_smoothing', $GmSmoothing,
        '--models_dir', $ModelsDir,
        '--csv_dir', $CsvDir,
        '--f_csv', $r.FCsv,
        '--h_csv', $r.HCsv
    ) + $ExtraRunArgs
    & $Python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "run for $($r.Name) exited with code $LASTEXITCODE" }
    Write-Host "---- $($r.Name) done in $([int]$sw.Elapsed.TotalMinutes) min ----`n" -ForegroundColor Green
}

if ($Populate) {
    Write-Host "==== populate (+gm, +loo_gm top $LooGmTop) ====" -ForegroundColor Cyan
    & $Python $script 'populate' '--with_gm' '--with_loo_gm' '--loo_gm_top' $LooGmTop `
        '--models_dir' $ModelsDir '--csv_dir' $CsvDir '--gm_workers' $Workers
    if ($LASTEXITCODE -ne 0) { throw "populate exited with code $LASTEXITCODE" }

    $report = Join-Path $PSScriptRoot 'build_search_report.py'
    if (Test-Path $report) {
        Write-Host "==== build_search_report ====" -ForegroundColor Cyan
        & $Python $report
        if ($LASTEXITCODE -ne 0) { throw "build_search_report exited with code $LASTEXITCODE" }
    }
}

Write-Host "ALL DONE in $([int]$overall.Elapsed.TotalMinutes) min" -ForegroundColor Green
