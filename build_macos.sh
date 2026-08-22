#!/usr/bin/env bash
set -euo pipefail
# Build script for KV4P-Desktop on macOS.
# Usage: ./build_macos.sh
# Produces: dist/kv4p-desktop (macOS .app bundle or binary)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Installing build dependencies..."
pip3 install pyinstaller -q

echo "==> Cleaning previous build..."
rm -rf build dist

echo "==> Building kv4p-desktop for macOS..."
# For .app bundle, use --windowed and add icon
# For CLI binary (recommended), use --onefile
pyinstaller --onefile --windowed \
    --name "KV4P-Desktop" \
    --add-data "kv4p_ht:kv4p_ht" \
    --hidden-import opuslib \
    --hidden-import sounddevice \
    --hidden-import serial \
    --hidden-import serial.tools.list_ports \
    --hidden-import numpy \
    --hidden-import PyQt6 \
    --hidden-import PyQt6.QtCore \
    --hidden-import PyQt6.QtWidgets \
    --hidden-import PyQt6.QtGui \
    --osx-bundle-identifier com.kv4p.ht \
    kv4p_ht/main.py 2>&1

echo ""
echo "==> Build complete!"
if [ -d "dist/KV4P-Desktop.app" ]; then
    echo "    App: $(pwd)/dist/KV4P-Desktop.app"
elif [ -f "dist/kv4p-desktop" ]; then
    echo "    Binary: $(pwd)/dist/kv4p-desktop"
fi
echo "    Size: $(du -sh dist/ 2>/dev/null | cut -f1)"
echo ""
echo "    Run: open dist/KV4P-Desktop.app"
