param(
    [Parameter(Mandatory)]
    [ValidateSet('spc','finder','data_workspace','tabulation','explorer','distributions','power','diagnostics','examples','factor')]
    [string]$App,

    [Parameter(Mandatory)]
    [string]$Username
)

$appNames = @{
    spc            = "spc-control-charts"
    finder         = "statistical-test-finder"
    data_workspace = "data-workspace"
    tabulation     = "tabular-summary"
    explorer       = "graph-explorer"
    distributions  = "probability-distributions"
    power          = "power-analysis"
    diagnostics    = "data-screening-diagnostics"
    examples       = "solved-examples"
    factor         = "factor-analysis"
}

$spaceName = $appNames[$App]
$source = "app_$App.py"
$target = "app.py"

# 1. Copy the stub -> app.py
Copy-Item -LiteralPath $source -Destination $target -Force
Write-Host "✓ Copied $source -> $target"

# 2. Deploy
$remote = "https://huggingface.co/spaces/$Username/$spaceName"
if (-not (git remote get-url space 2>$null)) {
    git remote add space $remote
    Write-Host "✓ Added remote: $remote"
} else {
    git remote set-url space $remote
    Write-Host "✓ Updated remote -> $remote"
}

git add $target requirements.txt features/ core/ datasets/ .streamlit/
git commit -m "deploy: $App for HF Space $spaceName"
git push space main
Write-Host "✓ Pushed to https://huggingface.co/spaces/$Username/$spaceName"
