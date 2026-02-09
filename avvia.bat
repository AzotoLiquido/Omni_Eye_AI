@echo off
REM Omni Eye AI - Avvio Rapido per Windows

title Omni Eye AI - Assistente Locale

echo.
echo ====================================================================
echo        OMNI EYE AI - Assistente AI Completamente Locale
echo ====================================================================
echo.

REM Vai nella directory dello script
cd /d "%~dp0"

REM Avvia l'applicazione con il Python del venv
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe start.py
) else (
    python start.py
)

pause
