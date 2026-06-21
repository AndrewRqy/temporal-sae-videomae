<#
  Local (no-SLURM) NFP no-false-positive test for the raw layer, PCA, and ICA,
  on the synthetic ball dataset. The SAE column is the existing cluster result
  (75/6144) — the trained SAE checkpoint is not local.

  Reuses the PCA/ICA decompositions fit by run_pca_ica_ms_local.ps1
  (local_runs/decomp/{pca,ica}.pt). The no-false-positive guarantee is a property
  of the covariance statistic + stimulus design, so it holds for all bases.

  Run:
      powershell -ExecutionPolicy Bypass -File jobs\local\run_nfp_local.ps1
  Smoke (first 200 videos only, faster):
      powershell -ExecutionPolicy Bypass -File jobs\local\run_nfp_local.ps1 -Smoke
#>
param(
    [switch]$Smoke,
    [int]$Batch   = 4,
    [int]$Workers = 0,        # 0 = safest on Windows
    [string]$Mode = "sign_split",
    [string]$Device = "cuda:0",
    [string]$Py = "C:\Users\relay\miniconda3\python.exe"
)
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # ...\sae-for-vlm
$NFP  = Join-Path $Repo "data\output\nfp"
$Dec  = Join-Path $Repo "local_runs\decomp"
$Out  = Join-Path $Repo "local_runs\nfp_results"
Set-Location $Repo
$env:PYTHONPATH = $Repo
New-Item -ItemType Directory -Force -Path $Out | Out-Null

# nfp_test.py has no max-videos flag; for a smoke run, point at a temp dir with a
# symlinked subset. Simplest: just run full — it's ~15-25 min/condition. Smoke just
# warns the user.
if ($Smoke) { Write-Host "(-Smoke has no effect: nfp_test.py runs the full dataset; expect ~15-25 min/condition)" -ForegroundColor Yellow }

function Run([string[]]$argv) {
    Write-Host "`n>>> $Py $($argv -join ' ')" -ForegroundColor Cyan
    & $Py @argv
    if ($LASTEXITCODE -ne 0) { throw "step failed (exit $LASTEXITCODE)" }
}

# raw layer (identity) -------------------------------------------------------
Write-Host "=== NFP: raw layer (768 dims, identity) ===" -ForegroundColor Green
Run @("analysis/nfp_test.py","--dataset_dir",$NFP,"--sae_model","identity",
    "--output_path",(Join-Path $Out "raw_nfp.pt"),"--label","Raw layer (768 dims)",
    "--batch_size","$Batch","--num_workers","$Workers","--device",$Device)

# PCA ------------------------------------------------------------------------
Write-Host "=== NFP: PCA ($Mode) ===" -ForegroundColor Green
Run @("analysis/nfp_test.py","--dataset_dir",$NFP,"--sae_model","pca",
    "--sae_path",(Join-Path $Dec "pca.pt"),"--decomp_mode",$Mode,
    "--output_path",(Join-Path $Out "pca_${Mode}_nfp.pt"),"--label","PCA ($Mode)",
    "--batch_size","$Batch","--num_workers","$Workers","--device",$Device)

# ICA ------------------------------------------------------------------------
Write-Host "=== NFP: ICA ($Mode) ===" -ForegroundColor Green
Run @("analysis/nfp_test.py","--dataset_dir",$NFP,"--sae_model","ica",
    "--sae_path",(Join-Path $Dec "ica.pt"),"--decomp_mode",$Mode,
    "--output_path",(Join-Path $Out "ica_${Mode}_nfp.pt"),"--label","ICA ($Mode)",
    "--batch_size","$Batch","--num_workers","$Workers","--device",$Device)

Write-Host "`nDONE. Result tensors saved under $Out" -ForegroundColor Green
