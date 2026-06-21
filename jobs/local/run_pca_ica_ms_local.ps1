<#
  Local (no-SLURM) run of the PCA / ICA monosemanticity-score pipeline on this PC.

  Mirrors jobs/monosemanticity/{fit_pca_ica,eval_mono_pca,eval_mono_ica}.sh but:
    - extracts a small fitting corpus locally (the 44M-token cluster corpus is gone),
    - uses local SSv2 + small batch sizes for an 8.5 GB laptop GPU,
    - invokes python directly instead of sbatch.

  Run from anywhere:
      powershell -ExecutionPolicy Bypass -File jobs\local\run_pca_ica_ms_local.ps1
  Quick end-to-end smoke test (4 videos, tiny ICA):
      powershell -ExecutionPolicy Bypass -File jobs\local\run_pca_ica_ms_local.ps1 -Smoke
#>
param(
    [switch]$Smoke,
    [int]$NFit   = 400,     # SSv2 train videos used to fit PCA/ICA (~NFit*1568 tokens)
    [int]$NVal   = 800,     # SSv2 val clips scored (must match DINOv2 embeddings)
    [int]$NComp  = 256,     # PCA/ICA components (<=768; 256 is fast + stable locally)
    [int]$BVmae  = 2,       # VideoMAE batch (videos). Lower to 1 if you OOM.
    [int]$BDino  = 4,       # DINOv2 batch (clips).
    [int]$Workers = 2,      # DataLoader workers. Set 0 if Windows multiprocessing hangs.
    [string]$Device = "cuda:0",
    # IMPORTANT: use the miniconda interpreter that has torch+CUDA+sklearn+transformers+av.
    # PowerShell's bare `python` is a different Store Python 3.13 without these.
    [string]$Py = "C:\Users\relay\miniconda3\python.exe"
)
$ErrorActionPreference = "Stop"

if ($Smoke) { $NFit = 4; $NVal = 4; $NComp = 8; $BVmae = 1; $BDino = 2 }

$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # ...\sae-for-vlm
$SSv2 = Join-Path (Split-Path -Parent $Repo) "SSv2"             # ...\project\SSv2
$Work = Join-Path $Repo "local_runs"
$Model = "MCG-NJU/videomae-base-finetuned-ssv2"
$Embed = Join-Path $Work "embeds\ssv2_val_dinov2.pt"
Set-Location $Repo
# Put the repo root on PYTHONPATH so subdir scripts (training/, eval/, analysis/) can
# import the top-level packages utils / datasets / models / dictionary_learning. Running
# `python training/extract_activations.py` otherwise only puts training/ on sys.path.
$env:PYTHONPATH = $Repo
New-Item -ItemType Directory -Force -Path (Join-Path $Work "embeds") | Out-Null

function Run([string[]]$argv) {
    Write-Host "`n>>> $Py $($argv -join ' ')" -ForegroundColor Cyan
    & $Py @argv
    if ($LASTEXITCODE -ne 0) { throw "step failed (exit $LASTEXITCODE)" }
}

Write-Host "=== 1/5  Extract layer-11 fitting corpus ($NFit train videos) ===" -ForegroundColor Green
Run @("training/extract_activations.py","--model_name",$Model,
    "--attachment_point","post_mlp_residual","--layer","11",
    "--dataset_name","ssv2","--data_path",$SSv2,"--split","train",
    "--batch_size","$BVmae","--num_workers","$Workers",
    "--output_dir",(Join-Path $Work "train_acts"),"--max_clips","$NFit","--device",$Device)

Write-Host "=== 2/5  Fit PCA + ICA ($NComp components) ===" -ForegroundColor Green
Run @("analysis/fit_pca_ica.py","--activations_dir",(Join-Path $Work "train_acts"),
    "--output_dir",(Join-Path $Work "decomp"),"--n_components","$NComp",
    "--n_samples","500000","--max_chunks","50","--methods","pca","ica",
    "--ica_max_iter","2000","--ica_tol","1e-3","--seed","0")

Write-Host "=== 3/5  DINOv2 embeddings ($NVal val clips) ===" -ForegroundColor Green
if (Test-Path $Embed) {
    Write-Host "    embeddings already exist at $Embed (skipping)"
} else {
    Run @("eval/encode_videos.py","--embeddings_path",$Embed,"--model_name","dinov2-base",
        "--data_path",$SSv2,"--split","val","--batch_size","$BDino",
        "--num_workers","$Workers","--max_clips","$NVal","--device",$Device)
}

foreach ($m in @("pca","ica")) {
    $outDir = Join-Path $Work "${m}_sign_split_val"
    Write-Host "=== 4/5  Extract max-pooled $m activations (val, sign_split) ===" -ForegroundColor Green
    Run @("training/extract_activations.py","--model_name",$Model,
        "--attachment_point","post_mlp_residual","--layer","11",
        "--dataset_name","ssv2","--data_path",$SSv2,"--split","val",
        "--batch_size","$BVmae","--num_workers","$Workers","--output_dir",$outDir,"--max_pool",
        "--sae_model",$m,"--sae_path",(Join-Path $Work "decomp\$m.pt"),
        "--decomp_mode","sign_split","--max_clips","$NVal","--device",$Device)

    Write-Host "=== 5/5  Monosemanticity score for $m ===" -ForegroundColor Green
    Run @("eval/metric.py","--activations_dir",$outDir,"--embeddings_path",$Embed,"--output_subdir","ms_dinov2")
}

Write-Host "`nDONE. MS results:" -ForegroundColor Green
foreach ($m in @("pca","ica")) {
    $f = Join-Path $Work "${m}_sign_split_val\ms_dinov2\metric_stats_new.txt"
    if (Test-Path $f) { Write-Host "--- $m ---"; Get-Content $f | Select-Object -First 3 }
}
