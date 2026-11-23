@echo off
title —ledzenie dˆoni - instalacja
cd /d "%~dp0"
rem Pliki tymczasowe instalacji na tym samym dysku co aplikacja (dysk C: bywa peˆny).
if not exist ".tmp" mkdir ".tmp"
set "TMP=%~dp0.tmp"
set "TEMP=%~dp0.tmp"
set "PIP_CACHE_DIR=%~dp0.tmp\pip-cache"
py -3 -m venv .venv
if errorlevel 1 goto brak_pythona
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto blad
".venv\Scripts\python.exe" -c "import dlonie; dlonie.pobierz_model(); print('Model dˆoni gotowy')"
if errorlevel 1 goto blad
rmdir /s /q ".tmp"
echo.
echo Gotowe. Teraz uruchom start.bat
pause
exit /b

:brak_pythona
echo Nie znaleziono Pythona. Zainstaluj go ze strony python.org i spr¢buj ponownie.
pause
exit /b 1

:blad
echo Instalacja si© nie udaˆa. Sprawd« poˆ¥czenie z internetem i wolne miejsce na dysku.
pause
exit /b 1
