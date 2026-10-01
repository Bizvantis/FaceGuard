@echo off
rem FaceGuard real-world pilot study (close the FaceGuard app first - it uses the camera)
cd /d "%~dp0experiments"
"..\.venv\Scripts\python.exe" exp_realworld.py
pause
