# Architecture

This document describes how the AI 4K video upscaler is put together and why.

## Goals

1. **Always produce a valid file.** A missing GPU, an absent PyTorch or an
   unsupported codec must never prevent the tool from delivering a 4K render.
2. **Never touch the disk with intermediate frames.** Disk I/O dominates naive
   implementations.
3. **Never lose the audio.** The original track is copied bit-exactly.
4. **Never exceed the available VRAM.** Inference is tiled and batched.
5. **Fail with an actionable message.** Every error explains what to do.

## Module map

| Module | Responsibility |
| --- | --- |
| `config.py` | Typed configuration objects and their validation |
| `errors.py` | The exception hierarchy rooted at `UpscalerError` |
| `paths.py` | Where models, binaries and caches live |
| `ffmpeg_wrapper.py` | FFmpeg discovery, probing and command construction |
| `models/registry.py` | The catalogue of downloadable networks |
| `models/downloader.py` | Atomic, verified downloads |
| `backends/base.py` | The backend interface and BGR/float conversions |
| `backends/networks.py` | `RRDBNet` and `SRVGGNetCompact` |
| `backends/torch_esrgan.py` | PyTorch inference with tiling |
| `backends/ncnn.py` | Inference through the ncnn/Vulkan executable |
| `backends/classical.py` | The dependency-free fallback |
| `backends/factory.py` | Backend selection |
| `pipeline.py` | The streaming decode/upscale/encode/mux pipeline |
| `engine.py` | The public API used by the CLI, the GUI and library users |
| `ui/cli.py` | The Typer + Rich command line client |
| `ui/gui.py` | The PySide6 desktop application |

## The pipeline

```
source.mp4
   │
   │  ffmpeg -f rawvideo -pix_fmt bgr24 -
   ▼
┌─────────────┐   queue    ┌──────────────┐   queue   ┌─────────────┐
│  producer   │───────────▶│  consumer(s) │──────────▶│   writer    │
│  (thread)   │  raw bytes │  (threads)   │  frames   │  (main)     │
└─────────────┘            └──────────────┘            └──────┬──────┘
                                                                │ raw bytes
                                                                ▼
                                                        ┌─────────────┐
                                                        │  ffmpeg     │
                                                        │  encoder    │
                                                        └──────┬──────┘
                                                               │ video only
                                                               ▼
                                                        ┌─────────────┐
                                                        │  ffmpeg mux │  -c:v copy -c:a copy
                                                        └──────┬──────┘
                                                               ▼
                                                            output.mkv
```

### Why threads instead of processes

The heavy work happens inside two subprocesses (FFmpeg) and inside the
inference backend, all of which release the GIL. Threads therefore give real
parallelism without the cost and the fragility of transferring large frame
buffers between processes.

### Frame ordering

Several consumers may run concurrently, so the batches carry a sequence
number. The writer keeps a small reordering buffer and emits the batches in
the original order, which guarantees that the output video is temporally
correct regardless of how many workers are used.

### Deadlock avoidance

Three rules keep the pipeline deadlock free:

1. **Every queue operation has a timeout.** A blocking `put()` that is never
   drained would hang the job forever, so the producers use a bounded wait and
   honour a stop flag.
2. **The subprocess pipes are drained.** FFmpeg writes diagnostics to stderr.
   When that pipe is not read, FFmpeg blocks once the operating system's
   buffer is full - and the whole pipeline stalls. Dedicated threads consume
   both stderr streams continuously.
3. **A sentinel terminates every stage.** `None` travels through the raw queue
   to stop the consumers, and the last consumer to finish sends `None` through
   the output queue to stop the writer.

### Memory budget

Peak memory is roughly:

```
batch_size * (source_width * source_height * 3 + target_width * target_height * 3)
```

For 720p source material, a 4K target and a batch of four that is about 113 MiB
of frame data - independent of the length of the video.

## Backend selection

`auto_backend` evaluates the backends in the order `ncnn`, `torch`,
`classical` and returns the first one that is usable. "Usable" means:

- the required dependencies are importable,
- the required files are present,
- the model weights are installed.

The classical backend has no such requirements, so the selection always
succeeds.

## The colour space

Frames travel as BGR `uint8` arrays. This is the native layout of FFmpeg's
`rawvideo`/`bgr24` decoder, so no conversion is needed between the decoder, the
inference engine and the encoder. Only the PyTorch backend converts to RGB
floats, and it converts back immediately.

## Target resolution

`compute_target` resolves the output geometry:

- an explicit width and height are honoured exactly,
- an explicit height preserves the aspect ratio,
- otherwise the height becomes 2160 (4K) and the width follows the aspect
  ratio.

All dimensions are rounded down to even numbers because `yuv420p` requires it.
