$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$bootstrapPython = Join-Path $projectRoot '.venv-train\Scripts\python.exe'
$inferRoot = Join-Path $projectRoot '.venv-infer'
$inferPython = Join-Path $inferRoot 'Scripts\python.exe'
$requirements = Join-Path $PSScriptRoot 'requirements-infer.txt'

if (-not (Test-Path -LiteralPath $bootstrapPython)) {
    throw "找不到用于创建环境的 Python：$bootstrapPython"
}

if (-not (Test-Path -LiteralPath $inferPython)) {
    & $bootstrapPython -m venv $inferRoot
}

& $inferPython -m pip install --upgrade pip
& $inferPython -m pip install -r $requirements
& $inferPython -c "import cv2, numpy, onnxruntime as ort; print('inference env ok', cv2.__version__, numpy.__version__, ort.get_available_providers())"
