@echo off
setlocal

REM Start de Spurgeon GUI vanuit de projectmap.
cd /d "%~dp0"

where poetry >nul 2>nul
if %errorlevel%==0 (
    poetry run daily gui
) else (
    echo Poetry niet gevonden. Probeer met Python module...
    py -m spurgeon.cli gui
)

endlocal
