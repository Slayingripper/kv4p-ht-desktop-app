#!/usr/bin/env bash
set -euo pipefail
# Build script for KV4P-Desktop standalone binary.
# Usage: ./build.sh
# Produces: dist/kv4p-desktop (single-file executable)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Use venv if it exists
if [ -d .venv ]; then
    source .venv/bin/activate
fi

echo "==> Installing build dependencies..."
pip install pyinstaller -q

echo "==> Cleaning previous build..."
rm -rf build dist

echo "==> Building kv4p-desktop binary..."
pyinstaller kv4p-ht.spec 2>&1

echo ""
echo "==> Build complete!"
echo "    Binary: $(pwd)/dist/kv4p-desktop"
echo "    Size:   $(du -h dist/kv4p-desktop | cut -f1)"
echo ""
echo "    Run: ./dist/kv4p-desktop --help"
