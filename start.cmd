@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.cmd first.
    pause
    exit /b 1
)
if exist ".venv\Scripts\pythonw.exe" (
    rem Detached GUI launch: no console window stays on the taskbar.
    rem Early tracebacks go to logs\app.stderr.log inside the app.
    start "" ".venv\Scripts\pythonw.exe" -m simulinterp
    exit /b 0
)
".venv\Scripts\python.exe" -m simulinterp
if errorlevel 1 pause
