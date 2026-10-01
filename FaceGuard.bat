@echo off
rem Start FaceGuard (no console window). Arguments are passed through, e.g.  FaceGuard.bat tray
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo FaceGuard is not set up yet - running setup first...
    call setup.bat || exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m faceguard %*
