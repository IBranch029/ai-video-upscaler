# Release notes

## v1.1.0 - Interactive terminal interface

The 1.1.0 release adds a full-screen interactive terminal interface and makes
it the default entry point of the Windows executable.

### What is new

- **Interactive terminal interface** built with
  [Textual](https://textual.textualize.io/).
  - Browse the folders with the directory tree and pick a video from the list of
    the files in the current folder, or type the path directly.
  - Choose the target resolution (2x, 4x, 4K - 3840x2160, Full HD or the source
    resolution).
  - Choose the backend. Every entry shows the device it would use and whether it
    is available, for example `ncnn - vulkan (available)`.
  - Choose the model out of the five downloadable Real-ESRGAN networks.
  - Watch the progress bar with the frame counter, the throughput, the ETA, the
    video memory and the system memory.
  - Stop a running job with the cancel button or with `Ctrl+C`.
- **Double clicking `aivideoupscaler.exe` now opens the interface** and the
  console window is kept open until the user presses Enter, so nothing is lost.
- **`aivu tui`** starts the interface explicitly.
- Live hardware statistics through `nvidia-smi` and `psutil`.

### Artefacts

| Platform | File |
| --- | --- |
| Windows x64 | `aivideoupscaler-windows.zip` |
| Linux x64 | `aivideoupscaler-linux.tar.gz` |

Each archive contains the standalone executable, the bundled FFmpeg helpers and
the documentation. The FFmpeg binaries are bundled, so no system-wide FFmpeg
installation is required.

### Notes

- The Real-ESRGAN weights are downloaded on demand with `aivu models install`
  and are not bundled into the release artefacts.
- The PyTorch backend is optional and is not part of the standalone build; use
  the `ncnn` backend for GPU acceleration without extra dependencies.
- Set `AIVU_NO_PAUSE=1` to stop the executable from waiting for a key press
  before it closes the console window. The pause only happens when the
  executable is started without any argument, so scripts are never blocked.

---

## v1.0.0 - Initial release

The first public release of the AI 4K video upscaler.

### Features

- **4K super-resolution** with Real-ESRGAN. The target resolution defaults to
  2160p with the source aspect ratio preserved.
- **Lossless audio.** The original audio track is stream-copied into the result
  with `-c:a copy`, so it is never re-encoded.
- **Three interchangeable backends**
  - `ncnn`: the official `realesrgan-ncnn-vulkan` binary, for any Vulkan GPU
    and without a PyTorch installation.
  - `torch`: the official `.pth` weights with a native implementation of the
    `RRDBNet` and `SRVGGNetCompact` architectures. CUDA, DirectML, Apple MPS
    and CPU are supported.
  - `classical`: a dependency-free Lanczos + edge-aware unsharp mask fallback
    that always works.
- **Streaming pipeline.** Frames are passed between FFmpeg and the inference
  engine through pipes, so intermediate frames never touch the disk.
- **VRAM-safe inference** through tiled processing, batching and a configurable
  batch size.
- **Parallel inference workers** with deterministic output ordering.
- **Rich command line interface** (`aivu`) with progress bars, live throughput
  and ETA reporting.
- **PySide6 desktop interface** with drag-and-drop, a cancellation button and
  live progress.
- **Five downloadable models** with verified, atomic downloads.
- **Rich diagnostics.** `aivu doctor` reports the FFmpeg installation, the
  available backends, the GPU device and the installed models.
- **Actionable error handling** for corrupt inputs, missing FFmpeg, absent GPU
  drivers and unsupported codecs.

### Artefacts

| Platform | File |
| --- | --- |
| Windows x64 | `aivideoupscaler-windows.zip` |
| Linux x64 | `aivideoupscaler-linux.tar.gz` |

Each archive contains the standalone executable, the FFmpeg helpers and the
model downloader. The FFmpeg binaries are bundled, so no system-wide FFmpeg
installation is required.

### Notes

- The Real-ESRGAN weights are downloaded on demand with `aivu models install`
  and are not bundled into the release artefacts.
- The PyTorch backend is optional and is not part of the standalone build; use
  the `ncnn` backend for GPU acceleration without extra dependencies.
