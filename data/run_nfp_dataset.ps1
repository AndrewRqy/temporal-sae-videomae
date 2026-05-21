# Generate the no-false-positives ball dataset using Kubric Docker.
#
# Usage (from PowerShell):
#   .\run_nfp_dataset.ps1              # all 3000 videos
#   .\run_nfp_dataset.ps1 -NVideos 10  # quick smoke test
#   .\run_nfp_dataset.ps1 -Start 0 -End 499   # shard 0
#   .\run_nfp_dataset.ps1 -Start 500 -End 999  # shard 1
#
# Output: .\output\nfp\v00000\ ... v02999\

param(
    [int]$NVideos = 3000,
    [int]$Start   = 0,
    [int]$End     = -1,      # -1 means NVideos-1
    [int]$Seed    = 0
)

$ScriptDir  = $PSScriptRoot
$OutDir     = Join-Path $ScriptDir "output\nfp"
$Image      = "kubricdockerhub/kubruntu"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if ($End -lt 0) { $End = $NVideos - 1 }

Write-Host "=== NFP ball dataset: videos $Start .. $End (seed=$Seed) ===" -ForegroundColor Cyan
Write-Host "Output: $OutDir"

docker run --rm `
    --gpus all `
    --volume "${ScriptDir}:/kubric_scripts:ro" `
    --volume "${OutDir}:/output" `
    $Image `
    /usr/bin/python3 /kubric_scripts/nfp_ball_dataset.py `
        --output_dir /output `
        --n_videos   $NVideos `
        --start_idx  $Start `
        --end_idx    $End `
        --seed       $Seed

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker run failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "=== Done. Output in $OutDir ===" -ForegroundColor Green
