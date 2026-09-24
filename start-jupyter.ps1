$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA 'uv-envs\Drug_Assist'
$env:PYTHONNOUSERSITE = '1'
Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
uv sync --locked
if ($LASTEXITCODE -ne 0) { throw 'Environment sync failed.' }
uv run --locked jupyter lab
