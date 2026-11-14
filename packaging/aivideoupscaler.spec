# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the AI 4K video upscaler.

Build a standalone executable with::

    pyinstaller packaging/aivideoupscaler.spec --noconfirm

The optional FFmpeg and ncnn/Vulkan binaries are added as data files when
they are present in the ``bin`` folder, so a released build can run without a
system-wide FFmpeg installation.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT_ROOT = Path(SPECPATH).parent
SRC = PROJECT_ROOT / "src"
BIN_DIR = PROJECT_ROOT / "bin"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

block_cipher = None

# The PyTorch backend is optional: bundling it would multiply the artefact
# size, so it is only included when it is actually installed.
bundle_torch = os.environ.get("AIVU_BUNDLE_TORCH", "0") == "1"

# PySide6 adds roughly 350 MB to the bundle, so the standalone executable
# ships the command line client and the terminal interface only.  Set
# AIVU_BUNDLE_GUI=1 to include the graphical interface as well.
bundle_gui = os.environ.get("AIVU_BUNDLE_GUI", "0") == "1"

hiddenimports = collect_submodules("aivideoupscaler")
hiddenimports += ["PIL._tkinter_finder"]

# The interactive terminal interface is built with Textual, which imports its
# widgets lazily.  Collecting the submodules makes sure every widget, theme and
# CSS parser is available inside the frozen executable.
hiddenimports += collect_submodules("textual")
hiddenimports += collect_submodules("textual.widgets")
hiddenimports += collect_submodules("textual.css")
hiddenimports += [
    "textual._context",
    "textual._xterm_parser",
    "textual.app",
    "textual.binding",
    "textual.containers",
    "textual.css.query",
    "textual.css.styles",
    "textual.events",
    "textual.message",
    "textual.message_pump",
    "textual.reactive",
    "textual.screen",
    "textual.widget",
    "textual.widgets.option_list",
    "textual.widgets._directory_tree",
    "textual.widgets._input",
    "textual.widgets._option_list",
    "textual.widgets._progress_bar",
    "textual.widgets._select",
    "textual.widgets._static",
    "textual.widgets.option_list",
]
excludes = ["tkinter", "matplotlib", "pytest", "IPython", "notebook"]

if not bundle_torch:
    excludes += ["torch", "torchvision", "torch_directml"]

if not bundle_gui:
    excludes += ["PySide6", "shiboken6"]

datas = []
binaries = []

# Textual ships the tree-sitter highlight definitions as data files.
datas += collect_data_files("textual", include_py_files=False, excludes=["*.pyc", "*.pyo"])

# Bundle the helper executables when they are available.
for name in ("ffmpeg", "ffprobe", "realesrgan-ncnn-vulkan"):
    for candidate in BIN_DIR.glob(f"{name}*"):
        if candidate.is_file():
            binaries.append((str(candidate), "bin"))
            break

# Bundle the ncnn model files when they are available.
NCNN_DIR = PROJECT_ROOT / "models" / "ncnn"
if NCNN_DIR.is_dir():
    for candidate in NCNN_DIR.glob("*.param"):
        datas.append((str(candidate), "models/ncnn"))
    for candidate in NCNN_DIR.glob("*.bin"):
        datas.append((str(candidate), "models/ncnn"))

analysis = Analysis(
    [str(PROJECT_ROOT / "packaging" / "launcher.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="aivideoupscaler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="aivideoupscaler",
)
