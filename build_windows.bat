@echo off
REM Build script for KV4P-Desktop on Windows.
REM Usage: build_windows.bat
REM Requires: Python 3.10+, pip, Visual C++ Redistributable

echo ==^> Installing build dependencies...
pip install pyinstaller -q

echo ==^> Cleaning previous build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo ==^> Building kv4p-desktop.exe...
pyinstaller kv4p-ht.spec

echo.
echo ==^> Build complete!
echo     Binary: %cd%\dist\kv4p-desktop.exe
echo.
echo     Run: dist\kv4p-desktop.exe --help
