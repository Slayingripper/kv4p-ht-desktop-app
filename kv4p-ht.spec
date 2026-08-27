# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for KV4P-Desktop.
Build: pyinstaller kv4p-ht.spec
"""
import sys
import os
from pathlib import Path

block_cipher = None

binaries = []
if sys.platform == 'win32':
    vendor_dir = Path('vendor')
    for dll_name in ('opus.dll', 'libopus-0.dll'):
        dll_path = vendor_dir / dll_name
        if dll_path.exists():
            binaries.append((str(dll_path), '.'))
            break
    else:
        import ctypes.util
        lib_path = ctypes.util.find_library('opus')
        if lib_path:
            binaries.append((lib_path, '.'))
    for dll_name in os.environ.get('WIN_EXTRA_DLLS', '').split(os.pathsep):
        dll_path = Path(dll_name.strip()) if dll_name.strip() else None
        if dll_path and dll_path.is_file():
            binaries.append((str(dll_path), '.'))

a = Analysis(
    ['kv4p_ht/main.py'],
    pathex=[],
    binaries=binaries,
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
    strip=False,
    upx=False,
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
