<#
  Local (no-SLURM) MS + NFP for the trained VideoMAE SAE (recovered from the cluster),
  so the SAE sits in the SAME local pipeline as the PCA/ICA/raw baselines — a fully
  self-consistent SAE-vs-PCA-vs-ICA comparison.

  Reuses the DINOv2 val embeddings (local_runs/embeds) and the NFP ball dataset
  (data/output/nfp) already present from the baseline runs.

  Run:
      powershell -ExecutionPolicy Bypass -File jobs\local\run_sae_local.ps1
#>
param(
    [string]$SaePath = "local_runs\sae\ae.pt",
    [int]$NVal   = 800,
    [int]$BVmae  = 2,
    [int]$Workers = 0,
    [string]$Device = "cuda:0",
    [string]$Py = "C:\Users\relay\miniconda3\python.exe"
)
$ErrorActionPreference = "Stop"

$Repo  = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # ...\sae-for-vlm
$SSv2  = Join-Path (Split-Path -Parent $Repo) "SSv2"
$Work  = Join-Path $Repo "local_runs"
$NFP   = Join-Path $Repo "data\output\nfp"
$Model = "MCG-NJU/videomae-base-finetuned-ssv2"
$Embed = Join-Path $Work "embeds\ssv2_val_dinov2.pt"
Set-Location $Repo
$env:PYTHONPATH = $Repo

function Run([string[]]$argv) {
    Write-Host "`n>>> $Py $($argv -join ' ')" -ForegroundColor Cyan
    & $Py @argv
    if ($LASTEXITCODE -ne 0) { throw "step failed (exit $LASTEXITCODE)" }
}

# 1) MS: extract max-pooled SAE activations on the 800 val clips, then score ------
$msDir = Join-Path $Work "sae_val"
Write-Host "=== SAE MS: extract max-pooled features ($NVal val clips) ===" -ForegroundColor Green
Run @("training/extract_activations.py","--model_name",$Model,
    "--attachment_point","post_mlp_residual","--layer","11",
    "--dataset_name","ssv2","--data_path",$SSv2,"--split","val",
    "--batch_size","$BVmae","--num_workers","$Workers","--output_dir",$msDir,"--max_pool",
    "--sae_model","standard","--sae_path",$SaePath,"--max_clips","$NVal","--device",$Device)

Write-Host "=== SAE MS: monosemanticity score ===" -ForegroundColor Green
Run @("eval/metric.py","--activations_dir",$msDir,"--embeddings_path",$Embed,"--output_subdir","ms_dinov2")

# 2) NFP test with the SAE on the ball dataset ----------------------------------
Write-Host "=== SAE NFP test (3000 ball videos) ===" -ForegroundColor Green
Run @("analysis/nfp_test.py","--dataset_dir",$NFP,"--sae_model","standard","--sae_path",$SaePath,
    "--output_path",(Join-Path $Work "nfp_results\sae_nfp.pt"),"--label","VideoMAE SAE (local)",
    "--batch_size","4","--num_workers","$Workers","--device",$Device)

Write-Host "`nDONE." -ForegroundColor Green
$f = Join-Path $msDir "ms_dinov2\metric_stats_new.txt"
if (Test-Path $f) { Write-Host "--- SAE MS ---"; Get-Content $f | Select-Object -First 3 }
