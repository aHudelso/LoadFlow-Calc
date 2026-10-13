@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First run: powershell -ExecutionPolicy Bypass -File .\setup.ps1
    pause
    popd
    exit /b 1
)
".venv\Scripts\python.exe" -m loadflow_app
set "app_exit_code=%errorlevel%"
if not "%app_exit_code%"=="0" pause
popd
exit /b %app_exit_code%
