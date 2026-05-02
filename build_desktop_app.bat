@echo off
setlocal

REM Build a single-file Windows EXE for desktop_app.py
REM Usage:
REM   build_desktop_app.bat

py -3 -m pip install pyinstaller >nul 2>&1
py -3 -m PyInstaller --noconfirm --onefile --windowed --name "DChPipelineApp" "C:\DCh\desktop_app.py"

echo.
echo Build finished. EXE is in:
echo   C:\DCh\dist\DChPipelineApp.exe
endlocal
