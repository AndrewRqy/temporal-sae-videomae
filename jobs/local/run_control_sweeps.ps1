<#
  Dimensionality sweep (PCA/ICA, both sign modes, fixed-768 cutoff) on the two CONTROLS,
  mirroring the VideoMAE sweep (jobs/local/run_sweep_local.ps1):

    - DINO negative control : NFP-only (MS is circular on DINOv2). Caches DINOv2 ball-token
                              acts once; fits on the SSv2-train DINO patch acts already
                              extracted by run_dino_pca_ica.ps1 (local_runs/dino_patch_acts).
    - Synthetic positive    : NFP + ground-truth W_tau / W_static alignment vs D, for the
                              100- and 763-static variants (pure linear algebra on cached reps).

      powershell -ExecutionPolicy Bypass -File jobs\local\run_control_sweeps.ps1
#>
param(
    [int[]]$Grid    = @(16, 32, 64, 128, 256, 512, 768),
    [string[]]$Modes = @("sign_split", "signed"),
    [int]$FixedDenom = 768,
    [int]$Workers   = 0,
    [string]$Device = "cuda:0",
    [string]$Py = "C:\Users\relay\miniconda3\python.exe"
)
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Work = Join-Path $Repo "local_runs"
$Res  = Join-Path $Repo "results\pca_ica_baselines"
Set-Location $Repo
$env:PYTHONPATH = $Repo

function Run([string[]]$argv) {
    Write-Host "`n>>> $Py $($argv -join ' ')" -ForegroundColor Cyan
    & $Py @argv
    if ($LASTEXITCODE -ne 0) { throw "step failed (exit $LASTEXITCODE)" }
}

$gridArgs = $Grid | ForEach-Object { "$_" }

# --- Synthetic (100 + 763) ---
foreach ($v in @(@{tag="100"; data="synth_data"}, @{tag="763"; data="synth_data_763"})) {
    Write-Host "`n=== synthetic-$($v.tag) dimension sweep ===" -ForegroundColor Green
    Run (@("analysis/sweep_synth_dim.py",
        "--all_videos_path", (Join-Path $Work "$($v.data)\all_videos.pt"),
        "--matrices_path",   (Join-Path $Work "$($v.data)\matrices.pt"),
        "--output_csv",      (Join-Path $Res "sweep_dim_synth$($v.tag).csv"),
        "--grid") + $gridArgs + @("--modes") + $Modes + @(
        "--fixed_denom", "$FixedDenom", "--device", $Device))
}

# --- DINO negative control ---
Write-Host "`n=== DINO dimension sweep ===" -ForegroundColor Green
Run (@("analysis/sweep_dino_dim.py",
    "--dataset_dir", (Join-Path $Repo "data\output\nfp"),
    "--train_dir",   (Join-Path $Work "dino_patch_acts"),
    "--output_csv",  (Join-Path $Res "sweep_dim_dino.csv"),
    "--grid") + $gridArgs + @("--modes") + $Modes + @(
    "--fixed_denom", "$FixedDenom", "--workers", "$Workers", "--device", $Device))

Write-Host "`nDONE -> $Res\sweep_dim_{synth100,synth763,dino}.csv" -ForegroundColor Green
