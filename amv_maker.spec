# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for AMV Maker — single-file executable."""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# find the bundled ffmpeg binary from imageio-ffmpeg
from imageio_ffmpeg import get_ffmpeg_exe
ffmpeg_bin = get_ffmpeg_exe()

# find soundfile's libsndfile DLL
import soundfile
sf_dir = os.path.dirname(soundfile.__file__)
sf_dlls = []
for f in os.listdir(sf_dir):
    if f.endswith('.dll') or f.endswith('.so') or f.endswith('.dylib'):
        sf_dlls.append((os.path.join(sf_dir, f), '.'))

# collect librosa data files (e.g. example audio, filter coefficients)
librosa_datas = collect_data_files('librosa')

# collect soxr binary
soxr_datas = collect_data_files('soxr')

a = Analysis(
    ['gui.py'],
    pathex=['.'],
    binaries=[
        (ffmpeg_bin, 'imageio_ffmpeg/binaries'),
    ] + sf_dlls,
    datas=librosa_datas + soxr_datas,
    hiddenimports=[
        # librosa and its deep dependency tree
        'librosa', 'librosa.core', 'librosa.core.audio',
        'librosa.feature', 'librosa.onset', 'librosa.util',
        'audioread', 'soundfile', 'soxr',
        'scipy', 'scipy.signal', 'scipy.fft', 'scipy.io', 'scipy.io.wavfile',
        'sklearn', 'sklearn.utils', 'sklearn.utils._typedefs',
        'numba', 'llvmlite',
        # moviepy
        'moviepy', 'moviepy.video', 'moviepy.audio',
        'proglog', 'imageio', 'imageio_ffmpeg',
        # pygame
        'pygame', 'pygame.mixer',
        # PIL
        'PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont', 'PIL.ImageTk',
        # opencv
        'cv2',
        # our modules
        'constants', 'effects', 'visualizers', 'audio', 'compositor',
        'render', 'player',
    ] + collect_submodules('librosa') + collect_submodules('scipy'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='AMV Maker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # windowed app, no console
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,
)
