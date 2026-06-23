<#
  Local PCA/ICA baseline on the DINOv2 NEGATIVE control, mirroring VideoMAE:
    1. extract DINOv2 spatial patch activations over an independent SSv2-train corpus,
    2. fit PCA/ICA on them (same Dictionary interface as the SAE),
    3. run the NFP test on the ball dataset for pca, ica, and identity (raw DINO patch dims).

  DINOv2 processes each frame independently (no temporal context), so the expected
  result for EVERY basis is ~0 temporal features — the negative control should hold
  regardless of the decomposition. The reported DINO-SAE row stays the cluster number
  (0 / 6144); the SAE itself is not re-run here.

  Each NFP run re-runs DINOv2 over the 3000-video ball dataset (~24k frames), so the
  three runs are the bulk of the wall-clock here.

      powershell -ExecutionPolicy Bypass -File jobs\local\run_dino_pca_ica.ps1
#>
param(
    [int]$NVid    = 200,                     # SSv2-train videos for the fit corpus (~600k patch tokens)
    [int]$NComp   = 256,                     # PCA/ICA components (-> 512 sign-split features)
    [string]$Mode = "sign_split",
    [int]$BExtract = 8,
    [int]$BNfp    = 8,
    [int]$Workers = 0,   # Windows: the DINO collate is a local closure -> not picklable; keep 0
    [string]$Device = "cuda:0",
    [string]$Py = "C:\Users\relay\miniconda3\python.exe"
)
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$SSv2 = Join-Path (Split-Path -Parent $Repo) "SSv2"
$Work = Join-Path $Repo "local_runs"
$Ball = Join-Path $Repo "data\output\nfp"
$Acts = Join-Path $Work "dino_patch_acts"
$Dcmp = Join-Path $Work "decomp_dino"
$Res  = Join-Path $Repo "results\pca_ica_baselines\dino_nfp"
Set-Location $Repo
$env:PYTHONPATH = $Repo
New-Item -ItemType Directory -Force -Path $Res | Out-Null

function Run([string[]]$argv) {
    Write-Host "`n>>> $Py $($argv -join ' ')" -ForegroundColor Cyan
    & $Py @argv
    if ($LASTEXITCODE -ne 0) { throw "step failed (exit $LASTEXITCODE)" }
}

Write-Host "=== 1/3  Extract DINOv2 patch activations ($NVid SSv2-train videos) ===" -ForegroundColor Green
if (Test-Path (Join-Path $Acts "activations_part0.pt")) {
    Write-Host "    patch activations already exist in $Acts (skipping extraction)"
} else {
    Run @("training/extract_dino_patch_activations.py","--data_path",$SSv2,"--split","train",
        "--output_dir",$Acts,"--max_videos","$NVid","--batch_size","$BExtract",
        "--save_every","50000","--num_workers","$Workers","--device",$Device)
}

Write-Host "=== 2/3  Fit PCA + ICA on DINO patch activations ($NComp comps) ===" -ForegroundColor Green
Run @("analysis/fit_pca_ica.py","--activations_dir",$Acts,"--output_dir",$Dcmp,
    "--n_components","$NComp","--n_samples","500000","--methods","pca","ica",
    "--ica_max_iter","2000","--ica_tol","1e-3","--seed","0")

Write-Host "=== 3/3  NFP on the ball dataset (pca, ica, identity) ===" -ForegroundColor Green
foreach ($m in @("pca","ica")) {
    Run @("analysis/nfp_test_dino_patch.py","--dataset_dir",$Ball,
        "--sae_model",$m,"--decomp_mode",$Mode,"--sae_path",(Join-Path $Dcmp "$m.pt"),
        "--output_path",(Join-Path $Res "dino_${m}_${Mode}.pt"),
        "--batch_size","$BNfp","--num_workers","$Workers","--device",$Device)
}
# Raw DINO patch dims (identity) — no decomposition checkpoint needed.
Run @("analysis/nfp_test_dino_patch.py","--dataset_dir",$Ball,
    "--sae_model","identity",
    "--output_path",(Join-Path $Res "dino_identity.pt"),
    "--batch_size","$BNfp","--num_workers","$Workers","--device",$Device)

Write-Host "`nDONE -> $Res" -ForegroundColor Green
