# Generate the back-and-forth ball dataset (50 videos) using Kubric Docker.
# Ball moves forward frames 0-7 then reverses frames 8-15 (net displacement ≈ 0).
#
# Usage (from PowerShell, inside the kubric/ directory):
#   .\run_backforth_dataset.ps1
#
# Output: .\output\backforth\

$ScriptDir  = $PSScriptRoot
$OutDir     = Join-Path $ScriptDir "output\backforth"
$Image      = "kubricdockerhub/kubruntu"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "=== Generating back-and-forth dataset (50 videos) ===" -ForegroundColor Cyan

docker run --rm `
    --gpus all `
    --volume "${ScriptDir}:/kubric_scripts:ro" `
    --volume "${OutDir}:/output" `
    $Image `
    /usr/bin/python3 /kubric_scripts/backforth_ball_dataset.py `
        --output_dir /output `
        --width 256 `
        --height 256

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker run failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "=== Done. Output in $OutDir ===" -ForegroundColor Green
