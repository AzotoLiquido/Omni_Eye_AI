@echo off
REM Omni Eye AI - Avvio Rapido (senza verifiche)

title Omni Eye AI - Server

echo.
echo ===================================================================
echo   OMNI EYE AI - Avvio Rapido
echo ===================================================================
echo.

REM Vai nella directory dello script
cd /d "%~dp0"

REM Avvia direttamente il server
if exist "venv\Scripts\python.exe" (
    echo Avvio con ambiente virtuale...
    venv\Scripts\python.exe app\main.py
) else (
    echo Avvio con Python di sistema...
    python app\main.py
)

pause
