# Release notes

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
