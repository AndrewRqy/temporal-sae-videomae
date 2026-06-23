<#
  Local (no-SLURM) dimensionality sweep for the PCA/ICA NFP + MS baselines.

  Sweeps D (incl. full-rank 768) for both sign_split and signed modes, and reports the
  NFP significance count under BOTH:
    - the adaptive Bonferroni cutoff  (alpha / #features), and
    - a fixed, D-independent cutoff   (alpha / 768)  -- see analysis/sweep_pca_ica_dim.py.

  Reuses the cached train activations + DINOv2 val embeddings from run_pca_ica_ms_local.ps1
  (the VideoMAE forward passes are re-run once here to cache MS + NFP acts; D is then swept
  as cheap linear algebra). Run from anywhere:
      powershell -ExecutionPolicy Bypass -File jobs\local\run_sweep_local.ps1
#>
param(
    [int[]]$Grid   = @(16, 32, 64, 128, 256, 512, 768),
    [string[]]$Modes = @("sign_split", "signed"),
    [int]$FixedDenom = 768,
    [int]$BVmae    = 2,
    [int]$Workers  = 0,
    [string]$Device = "cuda:0",
    [string]$Py = "C:\Users\relay\miniconda3\python.exe"
)
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # ...\sae-for-vlm
$SSv2 = Join-Path (Split-Path -Parent $Repo) "SSv2"            # ...\project\SSv2
$Work = Join-Path $Repo "local_runs"
Set-Location $Repo
$env:PYTHONPATH = $Repo

$argv = @(
    "analysis/sweep_pca_ica_dim.py",
    "--ssv2_path", $SSv2,
    "--nfp_dir", (Join-Path $Repo "data\output\nfp"),
    "--train_dir", (Join-Path $Work "train_acts"),
    "--embeds_path", (Join-Path $Work "embeds\ssv2_val_dinov2.pt"),
    "--output_csv", (Join-Path $Work "sweep_dim_fixed$FixedDenom.csv"),
    "--grid"
) + ($Grid | ForEach-Object { "$_" }) + @("--modes") + $Modes + @(
    "--fixed_denom", "$FixedDenom",
    "--batch", "$BVmae", "--workers", "$Workers", "--device", $Device
)

Write-Host ">>> $Py $($argv -join ' ')" -ForegroundColor Cyan
& $Py @argv
if ($LASTEXITCODE -ne 0) { throw "sweep failed (exit $LASTEXITCODE)" }
Write-Host "`nDONE -> $Work\sweep_dim_fixed$FixedDenom.csv" -ForegroundColor Green
