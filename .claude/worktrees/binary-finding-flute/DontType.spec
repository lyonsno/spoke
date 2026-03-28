# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for DontType.app — macOS global hold-to-dictate."""

import os
import sys
import tomllib

from PyInstaller.utils.hooks import collect_all

# Read version from pyproject.toml
_spec_dir = os.path.dirname(os.path.abspath(SPEC))
with open(os.path.join(_spec_dir, 'pyproject.toml'), 'rb') as _f:
    _pyproject = tomllib.load(_f)
_version = _pyproject['project']['version']

block_cipher = None

# Collect native extensions (.so, .dylib, .metallib) and data files
mlx_datas, mlx_binaries, mlx_hiddenimports = collect_all('mlx')
mlx_whisper_datas, mlx_whisper_binaries, mlx_whisper_hiddenimports = collect_all('mlx_whisper')
certifi_datas, certifi_binaries, certifi_hiddenimports = collect_all('certifi')

# Find the metallib so we can place it adjacent to libmlx.dylib at the top level
import glob as _glob
_metallib_paths = _glob.glob(os.path.join(
    os.path.dirname(os.path.abspath(SPEC)),
    '.venv/lib/python*/site-packages/mlx_metal/lib/mlx.metallib'
))
_extra_datas = []
if _metallib_paths:
    # Place metallib in the root of the bundle's Frameworks dir
    _extra_datas.append((_metallib_paths[0], '.'))

a = Analysis(
    ['entry_point.py'],
    pathex=[],
    binaries=mlx_binaries + mlx_whisper_binaries + certifi_binaries,
    datas=mlx_datas + mlx_whisper_datas + certifi_datas + _extra_datas,
    hiddenimports=mlx_hiddenimports + mlx_whisper_hiddenimports + certifi_hiddenimports + [
        # donttype modules
        'donttype',
        'donttype.__main__',
        'donttype.capture',
        'donttype.glow',
        'donttype.inject',
        'donttype.input_tap',
        'donttype.menubar',
        'donttype.overlay',
        'donttype.transcribe',
        'donttype.transcribe_local',
        # third-party
        'sounddevice',
        'numpy',
        'httpx',
        'httpx._transports.default',
        'httpcore',
        'tiktoken',
        'huggingface_hub',
        'tqdm',
        'regex',
        # PyObjC frameworks
        'AppKit',
        'Foundation',
        'Quartz',
        'PyObjCTools',
        'PyObjCTools.AppHelper',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'PIL',
        'IPython',
        'jupyter',
        'pytest',
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
    [],
    exclude_binaries=True,
    name='DontType',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch=None,
    codesign_identity=os.environ.get('CODESIGN_IDENTITY', ''),
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='DontType',
)

app = BUNDLE(
    coll,
    name='DontType.app',
    icon=None,  # TODO: add icon.icns
    bundle_identifier='com.noahlyons.donttype',
    codesign_identity=os.environ.get('CODESIGN_IDENTITY', ''),
    info_plist={
        'CFBundleName': 'DontType',
        'CFBundleDisplayName': 'DontType',
        'CFBundleVersion': _version,
        'CFBundleShortVersionString': _version,
        'LSUIElement': True,
        'NSMicrophoneUsageDescription': 'DontType needs microphone access to record speech for transcription.',
        'NSAppleEventsUsageDescription': 'DontType needs accessibility access to type transcribed text.',
    },
)
