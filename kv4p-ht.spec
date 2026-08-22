# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for KV4P-Desktop.
Build: pyinstaller kv4p-ht.spec
"""
import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['kv4p_ht/main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'opuslib',
        'sounddevice',
        'serial',
        'serial.tools.list_ports',
        'numpy',
        'scipy',
        'scipy.ndimage',
        'scipy.signal',
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtWidgets',
        'PyQt6.QtGui',
        'PyQt6.QtSvg',
        'PIL',
        'PIL.Image',
        'pysstv',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='kv4p-desktop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
