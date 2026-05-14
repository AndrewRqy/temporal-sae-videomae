# Generate synthetic ball datasets using Kubric Docker.
#
# Usage (from PowerShell, inside the kubric/ directory):
#   .\run_ball_dataset.ps1 velocity       # 392 videos
#   .\run_ball_dataset.ps1 acceleration   # 280 videos
#   .\run_ball_dataset.ps1 all            # both sequentially (default)
#
# Output: .\output\velocity\  and  .\output\acceleration\

param(
    [string]$DatasetType = "all"
)

$ScriptDir = $PSScriptRoot
$OutputBase = Join-Path $ScriptDir "output"
$Image = "kubricdockerhub/kubruntu"

function Invoke-KubricDataset {
    param([string]$Type)

    $OutDir = Join-Path $OutputBase $Type
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

    Write-Host "=== Generating $Type dataset ===" -ForegroundColor Cyan

    docker run --rm `
        --gpus all `
        --volume "${ScriptDir}:/kubric_scripts:ro" `
        --volume "${OutDir}:/output" `
        $Image `
        /usr/bin/python3 /kubric_scripts/ball_dataset.py `
            --dataset_type $Type `
            --output_dir /output `
            --width 256 `
            --height 256

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker run failed for $Type dataset (exit code $LASTEXITCODE)"
        exit $LASTEXITCODE
    }

    Write-Host "=== Done: $Type ===" -ForegroundColor Green
}

switch ($DatasetType.ToLower()) {
    "velocity"     { Invoke-KubricDataset "velocity" }
    "acceleration" { Invoke-KubricDataset "acceleration" }
    "all" {
        Invoke-KubricDataset "velocity"
        Invoke-KubricDataset "acceleration"
    }
    default {
        Write-Error "Unknown dataset type '$DatasetType'. Use: velocity | acceleration | all"
        exit 1
    }
}

Write-Host "All done. Output in $OutputBase" -ForegroundColor Green
