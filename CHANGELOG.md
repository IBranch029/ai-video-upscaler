# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-14

### Added

- **Core engine**
  - Streaming decode -> upscale -> encode -> mux pipeline that keeps the
    intermediate frames in pipes instead of writing them to disk.
  - Three interchangeable backends:
    - `ncnn` - Real-ESRGAN through the official `realesrgan-ncnn-vulkan`
      binary, for any Vulkan GPU and without a PyTorch installation.
    - `torch` - Real-ESRGAN through PyTorch with CUDA, DirectML, Apple MPS and
      CPU support, plus tiled inference for bounded VRAM usage.
    - `classical` - a dependency-free Lanczos resampling and edge-aware
      unsharp mask fallback that always works.
  - Native implementations of the `RRDBNet` and `SRVGGNetCompact`
    architectures, so the official checkpoints load with `strict=True`.
  - Automatic architecture and scale detection straight from the checkpoint.
  - Batched inference, configurable parallel workers and deterministic frame
    ordering.
  - Lossless audio muxing with `-c:a copy`.
- **Interfaces**
  - `aivu` - a Rich + Typer command line client with progress bars, live
    throughput and ETA reporting.
  - `aivu gui` - a PySide6 desktop application with drag-and-drop, a
    cancellation button and live progress.
  - A stable Python API (`UpscaleConfig`, `run_upscale`, `PipelineResult`).
- **Models**
  - Five downloadable Real-ESRGAN networks with verified, atomic downloads.
  - `aivu models install --ncnn` fetches the ncnn/Vulkan bundle.
- **Tooling**
  - FFmpeg and ffprobe discovery, including version-aware command construction
    for FFmpeg 5.1+ (`-fps_mode`) and older builds (`-vsync`).
  - A PyInstaller spec that produces a standalone executable.
  - GitHub Actions workflows for continuous integration and tag-triggered
    releases.
  - 65 tests covering the configuration, the FFmpeg wrapper, the backends, the
    pipeline, the engine, the filesystem layout and the command line client.

### Fixed

- The bundled binaries are located in the PyInstaller `_internal` folder.
- The console keeps its own code page, so the output stays readable on legacy
  Windows terminals.
- The download temporary file is released before it is written, which makes the
  download work on Windows.
