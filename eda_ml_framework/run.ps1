param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonCandidates = @(
    (Join-Path $projectRoot '.venv\Scripts\python.exe'),
    (Join-Path (Split-Path $projectRoot -Parent) '.venv\Scripts\python.exe')
)
$runtimePath = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $runtimePath) {
    throw 'No existe un entorno virtual. Sigue las instrucciones del README para instalar.'
}
Push-Location -LiteralPath $projectRoot
try {
    & $runtimePath -m streamlit run app.py --server.address 127.0.0.1 --server.port $Port --browser.gatherUsageStats false
} finally {
    Pop-Location
}
