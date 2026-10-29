@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
