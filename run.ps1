<#
.SYNOPSIS
  One-stop helper for the PCOS early-screening project.

.EXAMPLE
  .\run.ps1 setup     # create the virtual environment and install dependencies
  .\run.ps1 train     # full nested-CV training (~30 min on 12 cores)
  .\run.ps1 quick     # fast smoke training (~2 min)
  .\run.ps1 test      # run the automated test suite
  .\run.ps1 serve     # start the web app on http://127.0.0.1:8000
#>
param(
  [Parameter(Position = 0)][ValidateSet("setup", "train", "quick", "test", "serve")] [string]$Task = "serve",
  # The venv lives outside OneDrive by default; syncing thousands of site-packages files is slow.
  [string]$Venv = "$env:USERPROFILE\.venvs\pcos-early-screening",
  # Extra arguments passed straight to the Python command, e.g. --port 8001
  [Parameter(ValueFromRemainingArguments = $true)] [string[]]$Rest = @()
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $Venv "Scripts\python.exe"

if ($Task -eq "setup") {
  if (-not (Test-Path $py)) { python -m venv $Venv }
  & $py -m pip install --upgrade pip
  & $py -m pip install -r requirements.txt
  exit $LASTEXITCODE
}
if (-not (Test-Path $py)) { Write-Error "Virtual environment not found at $Venv. Run '.\run.ps1 setup' first."; exit 1 }

switch ($Task) {
  "train" { & $py -m pcos_screen.train @Rest }
  "quick" { & $py -m pcos_screen.train --quick @Rest }
  "test"  { & $py -m pytest -q -p no:cacheprovider @Rest }
  "serve" { & $py -m app.main @Rest }
}
exit $LASTEXITCODE
