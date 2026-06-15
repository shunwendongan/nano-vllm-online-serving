[CmdletBinding()]
param(
    [string]$Python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
)

$ErrorActionPreference = "Stop"

function Invoke-CheckedStep {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "Repository: $repoRoot"
Write-Host "Python: $Python"

$versionJson = & $Python -c "import json, sys; print(json.dumps({'major': sys.version_info.major, 'minor': sys.version_info.minor, 'micro': sys.version_info.micro, 'executable': sys.executable}))"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to query Python version"
}
$version = $versionJson | ConvertFrom-Json
$versionText = "$($version.major).$($version.minor).$($version.micro)"
Write-Host "Python version: $versionText"
if ($version.major -ne 3 -or $version.minor -lt 10 -or $version.minor -ge 13) {
    throw "Python version must satisfy >=3.10,<3.13; got $versionText"
}

Invoke-CheckedStep "Check local test dependencies" {
    & $Python -c "import importlib; mods=['pytest','fastapi','starlette','uvicorn','httpx']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('required local packages: '+(', '.join(mods))); print('missing: '+(', '.join(missing) if missing else 'none')); raise SystemExit(1 if missing else 0)"
}

Write-Host ""
Write-Host "==> Check optional GPU/model dependencies"
& $Python -c "import importlib; mods=[('torch','torch'),('flash-attn','flash_attn'),('triton','triton'),('transformers','transformers')]; missing=[]; [missing.append(name) for name, module in mods if importlib.util.find_spec(module) is None]; print('missing optional GPU/model packages: '+(', '.join(missing) if missing else 'none')); print('These packages are required for Colab/CUDA model validation, not for local non-GPU tests.')"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to inspect optional GPU/model dependencies"
}

Invoke-CheckedStep "unittest discover" {
    & $Python -m unittest discover -s tests
}

Invoke-CheckedStep "pytest tests" {
    & $Python -m pytest tests
}

Invoke-CheckedStep "compileall" {
    & $Python -m compileall nanovllm tests scripts
}

Invoke-CheckedStep "git diff --check" {
    & git diff --check
}

Invoke-CheckedStep "serve CLI help" {
    & $Python -m nanovllm.serve --help
}

Invoke-CheckedStep "bench_online CLI help" {
    & $Python bench_online.py --help
}

Invoke-CheckedStep "validate_online_gpu CLI help" {
    & $Python scripts\validate_online_gpu.py --help
}

Invoke-CheckedStep "run_colab_config CLI help" {
    & $Python scripts\run_colab_config.py --help
}

Write-Host ""
Write-Host "Local non-GPU validation passed."
Write-Host "Run scripts\validate_online_gpu.py on a CUDA/Colab host for torch/flash-attn/triton/model serving validation."
