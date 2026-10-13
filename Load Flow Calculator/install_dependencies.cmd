@echo off
setlocal
pushd "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set "setup_exit_code=%errorlevel%"
if not "%setup_exit_code%"=="0" echo Setup did not finish. Review the error above.
pause
popd
exit /b %setup_exit_code%
