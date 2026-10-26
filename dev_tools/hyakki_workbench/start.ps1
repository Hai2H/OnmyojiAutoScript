$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到运行环境：$python"
}

Push-Location $projectRoot
try {
    & $python -m dev_tools.hyakki_workbench.app @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
