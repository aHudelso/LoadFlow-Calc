param([switch]$Dev)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$launcher = Get-Command py.exe -ErrorAction SilentlyContinue
$launcherArgs = @('-3')
if (-not $launcher) {
    $launcher = Get-Command python.exe -ErrorAction SilentlyContinue
    $launcherArgs = @()
}
if (-not $launcher) {
    throw 'Install Python 3.11 or newer with Tcl/Tk support from python.org, then run setup.ps1 again.'
}

& $launcher.Source @launcherArgs -c "import sys, tkinter; assert sys.version_info >= (3, 11), 'Python 3.11 or newer is required'"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11+ with Tkinter is required. Repair or install Python with Tcl/Tk support.' }

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $launcher.Source @launcherArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the local .venv environment.' }
}

& '.\.venv\Scripts\python.exe' -c "import sys, tkinter; assert sys.version_info >= (3, 11), 'Python 3.11 or newer is required'"
if ($LASTEXITCODE -ne 0) { throw 'The existing .venv needs Python 3.11+ with Tkinter. Rename that environment and run setup.ps1 again.' }

$requirementsFile = 'requirements.txt'
if ($Dev) { $requirementsFile = 'requirements-dev.txt' }
& '.\.venv\Scripts\python.exe' -m pip install -r $requirementsFile
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check internet access and the pip error above.' }
& '.\.venv\Scripts\python.exe' -c "import tkinter, numpy; root=tkinter.Tk(); root.withdraw(); root.destroy(); print('Python, Tkinter and NumPy are ready.')"
if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed.' }

Write-Host 'Setup complete. Double-click launch.cmd, or run .\.venv\Scripts\python.exe -m loadflow_app'
