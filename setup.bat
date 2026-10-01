@echo off
rem FaceGuard one-click setup: creates .venv, installs dependencies, downloads the AI models.
setlocal
cd /d "%~dp0"
echo.
echo  === FaceGuard setup ===
echo.

where uv >nul 2>nul
if %errorlevel%==0 (
    echo Using uv to create the Python environment...
    uv venv .venv --python 3.12 || goto :plainpython
    uv pip install --python .venv\Scripts\python.exe -r requirements.txt || uv pip install --python .venv\Scripts\python.exe -e .[dev] || goto :fail
    uv pip install --python .venv\Scripts\python.exe -e . || goto :fail
    goto :models
)

:plainpython
echo Creating the Python environment...
py -3.12 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv || goto :nopython
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || .venv\Scripts\python.exe -m pip install -e .[dev] || goto :fail
.venv\Scripts\python.exe -m pip install -e . || goto :fail

:models
echo.
echo Downloading and verifying the neural-network models (about 43 MB)...
.venv\Scripts\python.exe -m faceguard download-models || goto :fail
echo.
.venv\Scripts\python.exe -m faceguard doctor
echo.
echo  Setup complete. Start FaceGuard with FaceGuard.bat
echo  (run create_shortcut.bat if you also want a desktop icon).
echo.
pause
exit /b 0

:nopython
echo Python 3.10 or newer is required: https://www.python.org/downloads/
pause
exit /b 1

:fail
echo.
echo Setup failed - see the messages above.
pause
exit /b 1
