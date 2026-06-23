<#
  Local PCA/ICA baseline on the SYNTHETIC positive control (100- and 763-static
  variants), mirroring what we do for VideoMAE: fit PCA/ICA on the same reps the
  synthetic SAE was trained on, then run the NFP test + ground-truth W_tau alignment.

  For each variant we run pca, ica, and (for a same-pipeline reference) the trained
  synthetic SAE. The reported SAE result of record stays the cluster number; the
  local SAE row is a pipeline check, like the VideoMAE convention.

  No VideoMAE/DINOv2 forward passes here — everything is linear algebra on the cached
  synthetic reps, so this is fast and can even run on CPU (-Device cpu).

      powershell -ExecutionPolicy Bypass -File jobs\local\run_synth_pca_ica.ps1
#>
param(
    [int]$NComp = 256,                       # PCA/ICA components (-> 512 sign-split features)
    [string]$Mode = "sign_split",
    [string]$Device = "cuda:0",
    [string]$Py = "C:\Users\relay\miniconda3\python.exe"
)
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Work = Join-Path $Repo "local_runs"
$Res  = Join-Path $Repo "results\pca_ica_baselines\synth_nfp"
Set-Location $Repo
$env:PYTHONPATH = $Repo
New-Item -ItemType Directory -Force -Path $Res | Out-Null

function Run([string[]]$argv) {
    Write-Host "`n>>> $Py $($argv -join ' ')" -ForegroundColor Cyan
    & $Py @argv
    if ($LASTEXITCODE -ne 0) { throw "step failed (exit $LASTEXITCODE)" }
}

# (variant tag, data dir, trained synthetic SAE)
$variants = @(
    @{ tag = "100"; data = "synth_data";     sae = "sae_synth100\ae.pt" },
    @{ tag = "763"; data = "synth_data_763"; sae = "sae_synth763\ae.pt" }
)

foreach ($v in $variants) {
    $tag   = $v.tag
    $allV  = Join-Path $Work "$($v.data)\all_videos.pt"
    $mats  = Join-Path $Work "$($v.data)\matrices.pt"
    $dcmp  = Join-Path $Work "decomp_synth_$tag"
    $saeP  = Join-Path $Work $v.sae

    Write-Host "`n=== synthetic-$tag : dump reps + fit PCA/ICA ($NComp comps) ===" -ForegroundColor Green
    Run @("analysis/dump_synth_acts.py","--all_videos_path",$allV,"--output_dir",$dcmp)
    Run @("analysis/fit_pca_ica.py","--activations_dir",$dcmp,"--output_dir",$dcmp,
        "--n_components","$NComp","--n_samples","500000","--methods","pca","ica",
        "--ica_max_iter","2000","--ica_tol","1e-3","--seed","0")

    foreach ($m in @("pca","ica")) {
        Write-Host "=== synthetic-$tag : NFP + W_tau alignment ($m, $Mode) ===" -ForegroundColor Green
        Run @("analysis/nfp_test_synthetic.py","--all_videos_path",$allV,"--matrices_path",$mats,
            "--sae_model",$m,"--decomp_mode",$Mode,"--sae_path",(Join-Path $dcmp "$m.pt"),
            "--output_path",(Join-Path $Res "synth${tag}_${m}_${Mode}.pt"),"--device",$Device)
    }

    # Same-pipeline SAE reference (reported result stays the cluster number).
    if (Test-Path $saeP) {
        Write-Host "=== synthetic-$tag : NFP for the trained SAE (pipeline reference) ===" -ForegroundColor Green
        Run @("analysis/nfp_test_synthetic.py","--all_videos_path",$allV,"--matrices_path",$mats,
            "--sae_model","standard","--sae_path",$saeP,
            "--output_path",(Join-Path $Res "synth${tag}_sae.pt"),"--device",$Device)
    } else {
        Write-Host "    (synthetic SAE not found at $saeP — skipping the SAE reference row)" -ForegroundColor Yellow
    }
}

Write-Host "`nDONE -> $Res" -ForegroundColor Green
