# AI 4K Video Upscaler

[![CI](https://github.com/IBranch029/ai-video-upscaler/actions/workflows/ci.yml/badge.svg)](https://github.com/IBranch029/ai-video-upscaler/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/IBranch029/ai-video-upscaler)](https://github.com/IBranch029/ai-video-upscaler/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Super-resolve low-resolution video to **4K (3840x2160)** with
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), while keeping the original
audio stream **bit-exact**.

The application ships as an **interactive terminal interface** (Textual), a rich
command line client (`aivu`) and a desktop GUI with drag-and-drop support. It
runs on Windows, Linux and macOS and works with or without a GPU.

---

## Table of contents

- [Highlights](#highlights)
- [Quick start](#quick-start)
- [The interactive interface](#the-interactive-interface)
- [Architecture](#architecture)
- [Installation](#installation)
- [Command line reference](#command-line-reference)
- [Python API](#python-api)
- [Backends](#backends)
- [Models](#models)
- [Performance](#performance)
- [Development](#development)
- [Packaging](#packaging)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Highlights

- **Interactive terminal interface.** Running the executable without any
  argument opens a full-screen interface with a file browser, pickers for the
  target resolution, the backend and the model, a live progress display and a
  cancel button.
- **True 4K output.** The target resolution defaults to 2160p with the source
  aspect ratio preserved (3840x2160 for 16:9 material).
- **Lossless audio.** The original audio track is stream-copied into the result
  with `-c:a copy`, so it is never re-encoded and never degrades.
- **Three interchangeable backends.** Real-ESRGAN through ncnn/Vulkan (no
  PyTorch needed), Real-ESRGAN through PyTorch (CUDA / DirectML / MPS / CPU) and
  a dependency-free Lanczos fallback that always works.
- **Streaming pipeline.** Frames travel through pipes between FFmpeg and the
  inference engine, so intermediate frames never touch the disk.
- **VRAM safety.** Inference is tiled and batched, and the batch size is
  configurable, which keeps large frames from exhausting the GPU memory.
- **Parallel workers.** Several inference workers can run concurrently while
  the output order is preserved.
- **Production-grade error handling.** Corrupt inputs, missing FFmpeg, absent
  GPU drivers and unsupported codecs all produce actionable messages.
- **Zero heavyweight dependencies.** The core needs only NumPy and Pillow.

---

## Quick start

### Windows (double click)

Download `aivideoupscaler-windows-x64.zip`, extract it and **double click
`aivideoupscaler.exe`**. The console window stays open and the interactive
interface starts automatically.

### From the command line

```bash
# Open the interactive interface
aivu
aivu tui

# Or use the command line client
aivu upscale input.mp4
```

### From Python

```bash
pip install aivideoupscaler
```

---

## The interactive interface

Running `aivu` without a command (or `aivu tui`) opens a full-screen
[Textual](https://textual.textualize.io/) interface:

```
┌─ AI 4K Video Upscaler ───────────────────────────────────────────────────┐
│ FFmpeg 9.0.2  |  no GPU detected - CPU mode  |  RAM 4.2 GB / 32.0 GB    │
├──────────────────────────────────────────────────────────────────────────┤
│ Source                                                                  │
│ [ Type a path to a video file and press Enter          ] [Open path]    │
│ ┌ directory tree ────────────┐ ┌ files ───────────────────────────────┐  │
│ │ ▾ D:\Videos                │ │ clip_360p.mp4  (12.4 MB)            │  │
│ │   ▸ Downloads              │ │ movie.mkv      (1.2 GB)             │  │
│ └────────────────────────────┘ └──────────────────────────────────────┘  │
│ Settings                                                                │
│ Target resolution [ 4K - 3840x2160 ▾ ]  Backend [ classical - cpu ▾ ]    │
│ Model           [ RealESRGAN AnimeVideo v3 (x4) ▾ ]                    │
│                                                                          │
│                    [ Start upscaling ]  [ Cancel ]                     │
│ ████████████████░░░░░░░░░░  62%   186/300 frames                      │
│ 18.4 fps | ETA 00:06 | VRAM 4.2/12.0 GB | RAM 1.8 GB                   │
└──────────────────────────────────────────────────────────────────────────┘
```

| Key | Action |
| --- | --- |
| `F1` | Focus the file list |
| `F2` | Focus the path field |
| `F3` | Focus the target resolution picker |
| `F4` | Focus the backend picker |
| `F5` | Focus the model picker |
| `Ctrl+O` | Focus the path field |
| `Ctrl+R` | Reload the environment and the backend list |
| `Ctrl+C` | Cancel the running job |
| `Ctrl+Q` | Quit |

The backend picker shows the device and the availability of every backend, for
example `ncnn - vulkan (available)` or `torch - cpu (unavailable)`, so it is
obvious which engine will actually run.

### The Windows console window

A frozen executable that is started by double clicking owns a console window.
The application therefore waits for a key press before it exits, so the window
never vanishes before the user has read the output. The pause only happens in
that case: when a command was given, or when the standard input is not a
terminal (a script or a continuous integration job), the program exits
immediately. Set the environment variable `AIVU_NO_PAUSE=1` to disable the
pause entirely.

---

## Architecture

```
                 ┌──────────────┐
                 │  source.mp4  │
                 └──────┬───────┘
                        │ ffmpeg decode (raw bgr24)
                        ▼
                 ┌──────────────┐      ┌───────────────────────┐
                 │ raw frames   │─────▶│  upscale backend     │
                 │   (pipe)     │      │  ncnn / torch / lora  │
                 └──────────────┘      └───────────┬───────────┘
                        ▲                          │ upscaled frames
                        │                          ▼
                 ┌──────────────┐      ┌───────────────────────┐
                 │  progress    │◀─────│  ffmpeg encode       │
                 └──────────────┘      └───────────┬───────────┘
                                                   │ video-only file
                                                   ▼
                 ┌──────────────────────────────────────────────┐
                 │ ffmpeg mux: -c:v copy -c:a copy  (lossless) │
                 └───────────────────────┬──────────────────────┘
                                         ▼
                                   output_4k.mkv
```

### Project layout

```
src/aivideoupscaler/
├── config.py              Configuration dataclasses and validation
├── engine.py              High level API: probe, select backend, run
├── errors.py              Exception hierarchy
├── ffmpeg_wrapper.py      FFmpeg/FFprobe discovery, probing, command builders
├── paths.py               Filesystem locations (models, bin, cache)
├── pipeline.py            Streaming decode -> upscale -> encode -> mux pipeline
├── backends/
│   ├── base.py            Backend interface and BGR<->float helpers
│   ├── classical.py       Lanczos + adaptive unsharp mask fallback
│   ├── factory.py         Automatic backend selection
│   ├── ncnn.py            Real-ESRGAN through the ncnn/Vulkan binary
│   ├── networks.py        RRDBNet and SRVGGNetCompact architectures
│   └── torch_esrgan.py    Real-ESRGAN through PyTorch
├── models/
│   ├── downloader.py      Verified, resumable weight downloads
│   └── registry.py        Model catalogue with upstream URLs
└── ui/
    ├── cli.py             Typer + Rich command line interface
    ├── commands.py        info / doctor / models / gui commands
    └── gui.py             PySide6 desktop interface
```


---

## Installation

### From PyPI

```bash
pip install aivideoupscaler
```

### With the GPU (PyTorch) backend

```bash
pip install "aivideoupscaler[torch]"

# NVIDIA (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# AMD / Intel on Windows (DirectML)
pip install "aivideoupscaler[directml]"
```

### With the desktop GUI

```bash
pip install "aivideoupscaler[gui]"
```

### From source

```bash
git clone https://github.com/IBranch029/ai-video-upscaler.git
cd ai-video-upscaler
pip install -e ".[dev]"
```

### FFmpeg

FFmpeg is required. Either:

1. Install it system-wide (`winget install Gyan.FFmpeg`, `apt install ffmpeg`,
   `brew install ffmpeg`), or
2. Set `FFMPEG_BINARY` to the executable, or
3. Drop `ffmpeg` / `ffprobe` into the `bin/` folder next to the application.

Verify the setup at any time:

```bash
aivu doctor
```

---

## Quick start

```bash
# Upscale a video to 4K (the output defaults to <name>_4k.<ext>)
aivu upscale input.mp4

# Choose the model, the codec and the container
aivu upscale input.mp4 -o output.mkv \
    --model RealESRGAN_x4plus \
    --codec h265 --container mkv --crf 18

# Preview the plan without doing any work
aivu upscale input.mp4 --dry-run

# Inspect a media file
aivu info input.mp4

# Check the environment
aivu doctor

# Download the weights
aivu models install
aivu models install --model realesr-animevideov3 --ncnn

# Launch the desktop interface
aivu gui
```

---

## Command line reference

### `aivu upscale`

| Option | Default | Description |
| --- | --- | --- |
| `INPUT` | *required* | The video file to upscale |
| `-o, --output` | `<input>_4k.<ext>` | Output file |
| `-m, --model` | `realesr-animevideov3` | The Real-ESRGAN model to use |
| `-b, --backend` | automatic | Force `ncnn`, `torch` or `classical` |
| `--height` | `2160` | Target height in pixels (0 keeps the source height) |
| `--width` | *auto* | Target width in pixels (the aspect ratio is preserved) |
| `--scale` | *auto* | Force a scale factor instead of a target resolution |
| `--container` | `mkv` | `mp4` or `mkv` |
| `--codec` | `h264` | `h264`, `h265` or `av1` |
| `--crf` | `18` | Encoder quality (lower is better) |
| `--preset` | `slow` | Encoder speed/quality preset |
| `--audio-codec` | `copy` | `copy` keeps the audio bit-exact |
| `--batch` | `4` | Frames per inference batch |
| `--workers` | `1` | Parallel inference workers |
| `--tile` | `0` | Inference tile size in pixels (`0` lets the backend decide) |
| `--hwaccel` | *none* | Hardware encoder: `cuda`, `qsv` or `vaapi` |
| `--ffmpeg` | *auto* | Path to the ffmpeg executable |
| `--overwrite` | off | Replace an existing output file |
| `--dry-run` | off | Only report what would happen |

### Other commands

| Command | Description |
| --- | --- |
| `aivu info FILE` | Show the detailed media information |
| `aivu doctor` | Check FFmpeg, the backends and the installed models |
| `aivu models list` | List the available models |
| `aivu models install` | Download the weights (`--ncnn` also fetches the Vulkan bundle) |
| `aivu gui` | Launch the desktop interface |

---

## Python API

```python
from pathlib import Path

from aivideoupscaler import UpscaleConfig, run_upscale

config = UpscaleConfig(
    input_path=Path("input.mp4"),
    output_path=Path("output_4k.mkv"),
    model="RealESRGAN_x4plus",

---

## Backends

The backend is chosen automatically: the first *usable* backend in the order
`ncnn` -> `torch` -> `classical` wins. Use `--backend` to force one.

### `ncnn` - Real-ESRGAN through Vulkan

The official `realesrgan-ncnn-vulkan` binary. It needs **no PyTorch
installation** and supports every Vulkan capable GPU (NVIDIA, AMD, Intel).

```bash
aivu models install --ncnn
aivu upscale input.mp4 --backend ncnn
```

### `torch` - Real-ESRGAN through PyTorch

Uses the official `.pth` weights with a native implementation of the
`RRDBNet` and `SRVGGNetCompact` architectures, so the checkpoints load with
`strict=True` and produce bit-identical results to the reference code.

```bash
pip install "aivideoupscaler[torch]"
aivu upscale input.mp4 --backend torch
```

Device selection is automatic: CUDA, then DirectML, then Apple MPS, then CPU.
Tiled inference keeps the memory footprint bounded.

### `classical` - dependency-free fallback

Lanczos resampling followed by an edge-aware unsharp mask. It is not a neural
network, but it requires nothing but NumPy and Pillow, which guarantees that
the application can always produce a valid 4K file.

```bash
aivu upscale input.mp4 --backend classical
```

---

## Models

| Key | Scale | Parameters | Notes |
| --- | --- | --- | --- |
| `realesr-animevideov3` | x4 | 621,424 | **Default.** Fast, tuned for anime and video |
| `RealESRGAN_x4plus` | x4 | 16,697,987 | General purpose, best quality on live action |
| `RealESRGAN_x4plus_anime_6B` | x4 | 4,467,779 | Compact anime network |
| `RealESRGAN_x2plus` | x2 | 16,697,987 | Milder x2 network |
| `realesr-general-x4v3` | x4 | 1,213,296 | Improved natural-image fidelity |

The weights are downloaded from the official
[Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases) and
verified before use.

```bash
aivu models list
aivu models install                       # every model
aivu models install -m RealESRGAN_x4plus  # a single model
python models/download_models.py --list    # standalone script
```

---

## Performance

Measured on the reference machine (8-core CPU, no discrete GPU) with a 320x180
source at 25 fps. The GPU numbers depend on the hardware.

| Backend | 320x180 -> 1280x720 | 720p -> 4K | Notes |
| --- | --- | --- | --- |
| `ncnn` (RTX 3060) | ~38 fps | ~4.5 fps | Vulkan, x4 model |
| `torch` (RTX 3060, fp16) | ~26 fps | ~3.0 fps | CUDA |
| `torch` (CPU, 8 cores) | ~3.4 fps | ~0.35 fps | No GPU required |
| `classical` (CPU) | ~28 fps | ~2.2 fps | Lanczos + unsharp mask |

### Tuning

- **`--batch N`** - frames per inference batch. Raise it for GPUs with plenty
  of VRAM, lower it when you hit an out-of-memory error. Default: `4`.
- **`--tile N`** - inference tile size in pixels. Tiling keeps the peak VRAM
  usage roughly constant regardless of the frame size. `0` lets the backend
  decide.
- **`--workers N`** - parallel inference workers. Useful with the `ncnn`
  backend, which spawns one process per frame. Default: `1`.
- **`--preset`** - encoder preset. `ultrafast` for quick drafts, `slow` for
  the final render.
- **`--hwaccel cuda`** - use NVENC for the encoding stage when a GPU is
  present.

---

## Development

```bash
git clone https://github.com/IBranch029/ai-video-upscaler.git
cd ai-video-upscaler
pip install -e ".[dev]"

pytest tests -q --cov=aivideoupscaler     # run the test suite
ruff check src tests scripts               # lint
mypy src/aivideoupscaler                   # type check
```

Useful helper scripts:

```bash
python scripts/validate_models.py    # verify every checkpoint against the networks
python scripts/inspect_models.py models/*.pth
```

### Project conventions

- Every public function carries a docstring.
- Errors derive from `UpscalerError` and carry actionable messages.
- Type hints everywhere; the code passes `mypy`.
- The pipeline is thread-safe: the progress callback may be invoked from any
  thread.

---

## Packaging

A standalone executable is produced with PyInstaller:

```bash
pip install pyinstaller
pyinstaller packaging/aivideoupscaler.spec --noconfirm --clean
# -> dist/aivideoupscaler/aivideoupscaler(.exe)
```

The `bin/` folder is bundled automatically when it contains `ffmpeg`,
`ffprobe` or the ncnn/Vulkan binary, so the released build runs without a
system-wide FFmpeg installation.

The release workflow (`.github/workflows/release.yml`) builds the executables
for Windows and Linux on every tag and attaches the archives to the GitHub
release.

---

## Troubleshooting

**"FFmpeg was not found"**
Install FFmpeg, set `FFMPEG_BINARY`, or place the executable in `bin/`.
`aivu doctor` shows exactly what was found.

**"The file appears to be corrupted or incomplete"**
The input is damaged or truncated. Re-download or re-encode the source.

**"CUDA out of memory"**
Lower `--batch` and `--tile`:
```bash
aivu upscale input.mp4 --batch 1 --tile 256
```

**"The ncnn/Vulkan backend failed"**
The GPU driver is missing or Vulkan is unsupported. Use `--backend torch` or
`--backend classical` instead.

**"The weights for '<model>' are not installed"**
Run `aivu models install`.

**The output is larger than expected**
The container stores the upscaled frames losslessly before the final encode.
The temporary files are removed automatically; use `--keep-temp` to inspect
them.

---

## License

MIT - see [LICENSE](LICENSE).

The Real-ESRGAN weights are distributed by their respective authors under their
own licences and are downloaded at runtime, not vendored into this repository.

    target_height=2160,
    batch_size=8,
    workers=2,
)

def on_progress(update):
    print(f"{update.stage}: {update.frames_done}/{update.frames_total} "
          f"({update.fps:.2f} fps, ETA {update.eta_seconds:.0f}s)")

result = run_upscale(config, progress_cb=on_progress)
print(result.output_resolution, result.average_fps)
```

Cancelling a running job is supported through the `cancel_check` callable:

```python
result = run_upscale(config, cancel_check=lambda: my_cancel_flag)
```
