param(
    [switch]$CpuOnly,
    [switch]$SkipModels,
    [ValidateSet('large-v3-turbo', 'small')][string]$Model = 'large-v3-turbo'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Program $Arguments" }
}

try {
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface'
    & python -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), 'Python 3.10-3.13 required (3.12 recommended)'"
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 x64 and add it to PATH first.' }
    $projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $projectPython)) {
        Invoke-Checked 'python' @('-m', 'venv', '.venv')
    }
    Invoke-Checked $projectPython @('-m', 'pip', 'install', '--upgrade', 'pip')
    $installTarget = if ($CpuOnly) { '.' } else { '.[cuda]' }
    $wheelCache = Join-Path $PSScriptRoot '.cache\wheels'
    New-Item -ItemType Directory -Force -Path $wheelCache | Out-Null
    Invoke-Checked $projectPython @('-m', 'pip', 'install', '--find-links', $wheelCache, '-e', $installTarget)
    if (-not $SkipModels) {
        Invoke-Checked $projectPython @('-u', '-m', 'simulinterp.prepare_models', '--asr', $Model)
    }
    Write-Host "Setup complete. Launch start.cmd. CPU-only install: select CPU in the app." -ForegroundColor Green
} catch {
    Write-Host $_ -ForegroundColor Red
    Write-Host 'Setup did not complete. Fix the error above and run setup.cmd again; model downloads can resume.'
    exit 1
}
