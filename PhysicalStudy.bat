@echo off
rem FaceGuard physical study (close the FaceGuard app first - it uses the camera). Optional: PhysicalStudy.bat --camera 1
cd /d "%~dp0experiments"
"..\.venv\Scripts\python.exe" exp_physical.py %*
pause
