@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

title AnchorFlux - Starting...

echo.
echo ========================================
echo   AnchorFlux Unified Launcher
echo ========================================
echo.

REM Get script directory
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

REM Python detection (priority: embedded > venv > system)
set "EMBED_PYTHON=%PROJECT_ROOT%\tools\python\python.exe"
set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"

if exist "%EMBED_PYTHON%" (
    set "PYTHON_EXEC=%EMBED_PYTHON%"
    echo [OK] Using embedded Python
    goto :run_bootloader
)

if exist "%VENV_PYTHON%" (
    set "PYTHON_EXEC=%VENV_PYTHON%"
    echo [OK] Using venv Python
    goto :run_bootloader
)

REM Try system Python
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXEC=python"
    echo [OK] Using system Python
    goto :run_bootloader
)

echo [ERROR] Python not found!
echo Please install Python 3.10+ or place embedded Python in tools\python\
pause
exit /b 1

:run_bootloader
echo [INFO] Starting bootloader...
echo.

REM Run bootloader with all arguments passed through
"%PYTHON_EXEC%" "%PROJECT_ROOT%\bootloader.py" %*

REM Capture exit code
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% neq 0 (
    echo.
    echo [ERROR] Bootloader exited with code: %EXIT_CODE%
    pause
)

exit /b %EXIT_CODE%
