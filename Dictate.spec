# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Dictate.app — macOS global hold-to-dictate."""

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

a = Analysis(
    ['entry_point.py'],
    pathex=[],
    binaries=mlx_binaries + mlx_whisper_binaries + certifi_binaries,
    datas=mlx_datas + mlx_whisper_datas + certifi_datas,
    hiddenimports=mlx_hiddenimports + mlx_whisper_hiddenimports + certifi_hiddenimports + [
        # dictate modules
        'dictate',
        'dictate.__main__',
        'dictate.capture',
        'dictate.glow',
        'dictate.inject',
        'dictate.input_tap',
        'dictate.menubar',
        'dictate.overlay',
        'dictate.transcribe',
        'dictate.transcribe_local',
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
        'torch',
        'sympy',
        'mlx_whisper.torch_whisper',
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
    name='Dictate',
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
    name='Dictate',
)

app = BUNDLE(
    coll,
    name='Dictate.app',
    icon=None,  # TODO: add icon.icns
    bundle_identifier='com.noahlyons.dictate',
    codesign_identity=os.environ.get('CODESIGN_IDENTITY', ''),
    info_plist={
        'CFBundleName': 'Dictate',
        'CFBundleDisplayName': 'Dictate',
        'CFBundleVersion': _version,
        'CFBundleShortVersionString': _version,
        'LSUIElement': True,
        'NSMicrophoneUsageDescription': 'Dictate needs microphone access to record speech for transcription.',
        'NSAppleEventsUsageDescription': 'Dictate needs accessibility access to type transcribed text.',
    },
)
