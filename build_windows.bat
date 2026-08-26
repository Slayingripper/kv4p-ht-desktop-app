@echo off
REM Build script for KV4P-Desktop on Windows.
REM Usage: build_windows.bat
REM Requires: Python 3.10+, pip, Visual C++ Redistributable, vcpkg (for opus)

echo ==^> Installing build dependencies...
pip install pyinstaller -q

echo ==^> Ensuring opus library is available...
if not exist vendor mkdir vendor
if not exist vendor\opus.dll (
    echo     opus.dll not found in vendor\ — attempting vcpkg install...
    where vcpkg >nul 2>&1
    if %errorlevel% equ 0 (
        vcpkg install opus:x64-windows
        copy "%VCPKG_ROOT%\installed\x64-windows\bin\opus.dll" vendor\opus.dll
    ) else if defined VCPKG_ROOT (
        "%VCPKG_ROOT%\vcpkg.exe" install opus:x64-windows
        copy "%VCPKG_ROOT%\installed\x64-windows\bin\opus.dll" vendor\opus.dll
    ) else (
        echo     ERROR: vcpkg not found. Install vcpkg or place opus.dll in vendor\
        echo     See: https://vcpkg.io/en/getting-started
        exit /b 1
    )
) else (
    echo     opus.dll already present.
)

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
