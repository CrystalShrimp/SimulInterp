@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.cmd first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -u -m simulinterp.prepare_models %*
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
