@echo off
rem Put a FaceGuard icon on your desktop.
cd /d "%~dp0"
powershell -NoProfile -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\FaceGuard.lnk'); $s.TargetPath = '%~dp0.venv\Scripts\pythonw.exe'; $s.Arguments = '-m faceguard'; $s.WorkingDirectory = '%~dp0'; $s.IconLocation = '%~dp0assets\faceguard.ico'; $s.Description = 'FaceGuard - face recognition and presence security'; $s.Save()" && echo Desktop shortcut created.
pause
