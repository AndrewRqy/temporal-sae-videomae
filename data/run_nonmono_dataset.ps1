# Generate the non-monotonic ball dataset (50 videos) using Kubric Docker.
#
# Usage (from PowerShell, inside the kubric/ directory):
#   .\run_nonmono_dataset.ps1
#
# Output: .\output\nonmono\

$ScriptDir  = $PSScriptRoot
$OutDir     = Join-Path $ScriptDir "output\nonmono"
$Image      = "kubricdockerhub/kubruntu"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "=== Generating nonmono dataset (50 videos) ===" -ForegroundColor Cyan

docker run --rm `
    --gpus all `
    --volume "${ScriptDir}:/kubric_scripts:ro" `
    --volume "${OutDir}:/output" `
    $Image `
    /usr/bin/python3 /kubric_scripts/nonmono_ball_dataset.py `
        --output_dir /output `
        --width 256 `
        --height 256

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker run failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "=== Done. Output in $OutDir ===" -ForegroundColor Green
