@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" goto brak
start "" ".venv\Scripts\pythonw.exe" aplikacja.py %*
exit /b

:brak
echo Brak ˜rodowiska Pythona. Uruchom najpierw instaluj.bat
pause
exit /b 1
