# Generate the static ball dataset (30 videos) using Kubric Docker.
#
# Usage (from PowerShell, inside the kubric/ directory):
#   .\run_static_dataset.ps1
#
# Output: .\output\static\

$ScriptDir  = $PSScriptRoot
$OutDir     = Join-Path $ScriptDir "output\static"
$Image      = "kubricdockerhub/kubruntu"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "=== Generating static dataset (30 videos) ===" -ForegroundColor Cyan

docker run --rm `
    --gpus all `
    --volume "${ScriptDir}:/kubric_scripts:ro" `
    --volume "${OutDir}:/output" `
    $Image `
    /usr/bin/python3 /kubric_scripts/static_ball_dataset.py `
        --output_dir /output `
        --width 256 `
        --height 256

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker run failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "=== Done. Output in $OutDir ===" -ForegroundColor Green
