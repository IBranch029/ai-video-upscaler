"""The frame processing pipeline.

The pipeline is a three stage streaming design:

1. **Decode** - FFmpeg writes raw BGR24 frames to a pipe.
2. **Upscale** - frames are consumed in batches, passed through the selected
   super-resolution backend and written back to another pipe.
3. **Encode** - FFmpeg reads the upscaled frames and writes the video track.

Because every stage communicates through pipes, the intermediate frames never
touch the disk, which removes the I/O bottleneck that dominates naive
implementations.  The original audio stream is copied over bit-exactly in a
final muxing step.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import numpy as np

from .backends.base import UpscalerBackend, lanczos_filter
from .config import MediaInfo, UpscaleConfig
from .errors import CancelledError, FFmpegError, PipelineError
from .ffmpeg_wrapper import (
    build_decode_command,
    build_encode_command,
    build_mux_command,
    require_ffmpeg,
)

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass
class Progress:
    """A snapshot of the pipeline's progress."""

    stage: str = "starting"
    frames_done: int = 0
    frames_total: int = 0
    fps: float = 0.0
    eta_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    message: str = ""
    source_resolution: str = ""
    output_resolution: str = ""

    @property
    def fraction(self) -> float:
        """Completion ratio in the ``0.0 - 1.0`` range."""
        if self.frames_total <= 0:
            return 0.0
        return min(1.0, self.frames_done / self.frames_total)

    @property
    def percent(self) -> float:
        """Completion percentage."""
        return self.fraction * 100.0


ProgressCallback = Callable[[Progress], None]


@dataclass
class PipelineResult:
    """The outcome of a completed upscale job."""

    output_path: Path
    frames_processed: int
    elapsed_seconds: float
    source_resolution: str
    output_resolution: str
    backend: str
    model: str
    audio_preserved: bool
    temporary_files: list[Path] = field(default_factory=list)

    @property
    def average_fps(self) -> float:
        """Average number of frames processed per second."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_processed / self.elapsed_seconds


def even(value: int) -> int:
    """Round ``value`` down to the nearest even number (required by yuv420p)."""
    return max(2, value - (value % 2))


def compute_target(
    src_width: int,
    src_height: int,
    target_width: int | None = None,
    target_height: int | None = None,
    scale: int | None = None,
    default_height: int = 2160,
) -> tuple[int, int]:
    """Compute the output resolution for a job.

    When both dimensions are given they are honoured exactly.  When only the
    height is given the aspect ratio is preserved.  When nothing is given the
    video is scaled so that its height matches ``default_height`` (4K).
    """
    if target_width and target_height:
        return even(target_width), even(target_height)

    if scale and scale > 1:
        return even(src_width * scale), even(src_height * scale)

    height = target_height or default_height
    if target_width:
        height = max(2, round(src_height * target_width / src_width))
        return even(target_width), even(height)

    if src_height <= 0:
        return even(src_width), even(height)

    ratio = src_width / src_height
    width = round(height * ratio)
    return even(width), even(height)


def _drain(stream: IO[bytes] | None, sink: list[str]) -> None:
    """Consume a subprocess' stderr stream into ``sink``.

    Draining the pipe continuously prevents FFmpeg from blocking when the
    operating system's pipe buffer fills up.
    """
    if stream is None:
        return
    try:
        for chunk in iter(lambda: stream.read(4096), b""):
            sink.append(chunk.decode("utf-8", "replace"))
    except (ValueError, OSError):  # pragma: no cover - the stream was closed
        pass


class _FrameReader:
    """Reads fixed-size raw frames from a pipe."""

    def __init__(self, stream: IO[bytes], frame_size: int) -> None:
        self._stream = stream
        self._frame_size = frame_size

    def read(self) -> bytes:
        """Read exactly one frame worth of bytes, or return empty on EOF."""
        chunks: list[bytes] = []
        remaining = self._frame_size
        while remaining > 0:
            block = self._stream.read(remaining)
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        return b"".join(chunks)

    def read_batch(self, count: int) -> list[bytes]:
        """Read up to ``count`` frames, stopping early at end of stream."""
        batch: list[bytes] = []
        for _ in range(count):
            frame = self.read()
            if not frame:
                break
            batch.append(frame)
        return batch


class UpscalePipeline:
    """Runs a complete decode -> upscale -> encode -> mux job."""

    def __init__(
        self,
        config: UpscaleConfig,
        media: MediaInfo,
        backend: UpscalerBackend,
        progress_cb: ProgressCallback | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self.media = media
        self.backend = backend
        self.progress_cb = progress_cb
        self.cancel_check = cancel_check

        self.ffmpeg_bin = require_ffmpeg(config.ffmpeg_path)
        self.source_width = media.video.width
        self.source_height = media.video.height
        self.fps = media.video.fps or 25.0

        self.target_width, self.target_height = compute_target(
            self.source_width,
            self.source_height,
            config.target_width,
            config.target_height,
            config.scale,
        )

        self._frames_done = 0
        self._start_time = 0.0
        self._cancelled = False
        self._lock = threading.Lock()
        self._error: BaseException | None = None

    # ------------------------------------------------------------------
    # Progress reporting
    # ------------------------------------------------------------------
    def _report(self, stage: str, message: str = "") -> None:
        """Push a progress snapshot to the callback."""
        if self.progress_cb is None:
            return
        try:
            elapsed = time.perf_counter() - self._start_time if self._start_time else 0.0
            frames_total = self.media.video.frame_count or 0
            fps = self._frames_done / elapsed if elapsed > 0 else 0.0
            eta = 0.0
            if fps > 0 and frames_total > self._frames_done:
                eta = (frames_total - self._frames_done) / fps

            progress = Progress(
                stage=stage,
                frames_done=self._frames_done,
                frames_total=frames_total,
                fps=fps,
                eta_seconds=eta,
                elapsed_seconds=elapsed,
                message=message,
                source_resolution=f"{self.source_width}x{self.source_height}",
                output_resolution=f"{self.target_width}x{self.target_height}",
            )
            self.progress_cb(progress)
        except Exception:
            import traceback

            traceback.print_exc()

    def _check_cancelled(self) -> None:
        """Raise :class:`CancelledError` when the user asked to abort."""
        if self._cancelled:
            raise CancelledError("The upscale job was cancelled.")
        if self.cancel_check is not None and self.cancel_check():
            self._cancelled = True
            raise CancelledError("The upscale job was cancelled.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self) -> PipelineResult:
        """Execute the job and return its result."""
        self._start_time = time.perf_counter()
        config = self.config
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        work_dir = output_path.parent / f".{output_path.stem}.aivup-tmp"
        work_dir.mkdir(parents=True, exist_ok=True)
        silent_video = work_dir / "video.mkv"

        try:
            self._check_cancelled()
            self._report("encoding", "Preparing the encoder")
            self._encode(silent_video)

            self._report("muxing", "Copying the audio stream")
            self._mux(silent_video, output_path)

            elapsed = time.perf_counter() - self._start_time
            self._report("done", "Finished")
            return PipelineResult(
                output_path=output_path,
                frames_processed=self._frames_done,
                elapsed_seconds=elapsed,
                source_resolution=f"{self.source_width}x{self.source_height}",
                output_resolution=f"{self.target_width}x{self.target_height}",
                backend=getattr(self.backend, "name", "unknown"),
                model=config.model,
                audio_preserved=self.media.has_audio,
            )
        finally:
            self._cleanup(work_dir, silent_video)

    def _cleanup(self, work_dir: Path, silent_video: Path) -> None:
        """Remove the temporary artefacts."""
        if not self.config.keep_temp:
            try:
                silent_video.unlink(missing_ok=True)
                if work_dir.exists():
                    work_dir.rmdir()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    def _encode(self, output_path: Path) -> None:
        """Run the decode -> upscale -> encode stages."""
        config = self.config
        source = Path(self.media.path)
        frame_bytes = self.source_width * self.source_height * 3

        decode_cmd = build_decode_command(
            self.ffmpeg_bin, source, self.source_width, self.source_height, self.fps
        )
        encode_cmd = build_encode_command(
            self.ffmpeg_bin,
            output_path,
            self.target_width,
            self.target_height,
            self.fps,
            codec=self._encoder_name(),
            crf=config.encoder.crf,
            preset=config.encoder.preset,
            pix_fmt=config.encoder.pix_fmt,
            hwaccel=config.encoder.hwaccel,
            extra_args=config.encoder.extra_args,
            overwrite=config.overwrite,
        )

        self._report("decoding", "Decoding source frames")
        decoder = subprocess.Popen(
            decode_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_CREATE_NO_WINDOW,
        )
        encoder = subprocess.Popen(
            encode_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_CREATE_NO_WINDOW,
        )

        # FFmpeg writes diagnostics to stderr.  If that pipe is not drained the
        # process blocks once the OS buffer is full, which would deadlock the
        # pipeline, so both streams are consumed by dedicated threads.
        decoder_log: list[str] = []
        encoder_log: list[str] = []
        drain_threads = [
            threading.Thread(
                target=_drain,
                args=(decoder.stderr, decoder_log),
                name="decode-log",
                daemon=True,
            ),
            threading.Thread(
                target=_drain,
                args=(encoder.stderr, encoder_log),
                name="encode-log",
                daemon=True,
            ),
        ]
        for thread in drain_threads:
            thread.start()

        if decoder.stdout is None:  # pragma: no cover - the decoder always has a pipe
            raise PipelineError("The FFmpeg decoder was started without an output pipe.")
        reader = _FrameReader(decoder.stdout, frame_bytes)
        try:
            self._pump(reader, encoder)
        finally:
            self._close_pipes(decoder, encoder)
            for thread in drain_threads:
                thread.join(timeout=5)

        decoder_err = "".join(decoder_log)
        encoder_err = "".join(encoder_log)

        if decoder.returncode not in (0, None):
            raise FFmpegError(
                "FFmpeg failed while decoding the source video.",
                returncode=decoder.returncode,
                stderr=decoder_err,
            )
        if encoder.returncode not in (0, None):
            raise FFmpegError(
                "FFmpeg failed while encoding the upscaled video.",
                returncode=encoder.returncode,
                stderr=encoder_err,
            )

    def _pump(self, reader: _FrameReader, encoder: subprocess.Popen) -> None:
        """Move frames from the decoder to the encoder through the backend.

        The producer thread reads raw frames, the consumer threads upscale
        them and the calling thread writes the results to the encoder.  A
        sentinel value travels through both queues so that the shutdown is
        deterministic and cannot deadlock.
        """
        batch_size = max(1, self.config.batch_size)
        workers = max(1, self.config.workers)

        raw_queue: queue.Queue = queue.Queue(maxsize=max(2, workers * 2))
        out_queue: queue.Queue = queue.Queue(maxsize=max(2, workers * 2))
        stop = threading.Event()
        finished_consumers = 0
        finished_lock = threading.Lock()

        def put_raw(item: object) -> None:
            """Put an item on the raw queue, honouring the stop flag."""
            while not stop.is_set():
                try:
                    raw_queue.put(item, timeout=0.5)
                    return
                except queue.Full:
                    continue
            if item is None:
                raw_queue.put_nowait(item)

        def put_out(item: object) -> None:
            """Put an item on the output queue, honouring the stop flag."""
            while not stop.is_set():
                try:
                    out_queue.put(item, timeout=0.5)
                    return
                except queue.Full:
                    continue
            try:
                out_queue.put_nowait(item)
            except queue.Full:
                pass

        def producer() -> None:
            """Read raw frames from FFmpeg into the queue."""
            sequence = 0
            try:
                while not stop.is_set():
                    self._check_cancelled()
                    batch = reader.read_batch(batch_size)
                    if not batch:
                        break
                    put_raw((sequence, batch))
                    sequence += 1
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
                stop.set()
            finally:
                put_raw(None)

        def consumer() -> None:
            """Upscale batches of frames, preserving their sequence number."""
            nonlocal finished_consumers
            try:
                while True:
                    item = raw_queue.get()
                    if item is None:
                        break
                    sequence, batch = item
                    try:
                        if stop.is_set():
                            continue
                        frames = [
                            np.frombuffer(payload, dtype=np.uint8).reshape(
                                self.source_height, self.source_width, 3
                            )
                            for payload in batch
                        ]
                        upscaled = self.backend.upscale_batch(frames)
                        put_out((sequence, upscaled))
                    except BaseException as exc:
                        if self._error is None:
                            self._error = exc
                        stop.set()
                    finally:
                        raw_queue.task_done()
            finally:
                # The last consumer to finish signals the end of the stream.
                with finished_lock:
                    finished_consumers += 1
                    last = finished_consumers >= workers
                if last:
                    put_out(None)

        threads = [threading.Thread(target=producer, name="decode", daemon=True)]
        for index in range(workers):
            threads.append(threading.Thread(target=consumer, name=f"upscale-{index}", daemon=True))
        for thread in threads:
            thread.start()

        # The writer stays on this thread so the frame order is preserved.
        self._report("upscaling", "Upscaling frames")
        pending: dict[int, list[np.ndarray]] = {}
        expected = 0
        try:
            while True:
                item = out_queue.get()
                if item is None:
                    break
                sequence, frames = item
                pending[sequence] = frames
                while expected in pending:
                    for frame in pending.pop(expected):
                        self._check_cancelled()
                        self._write_frame(encoder, frame)
                        with self._lock:
                            self._frames_done += 1
                        self._report("upscaling")
                    expected += 1
                if self._error is not None:
                    break
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=30)

        if self._error is not None:
            raise self._error

    def _write_frame(self, encoder: subprocess.Popen, frame: np.ndarray) -> None:
        """Write one upscaled frame to the encoder's stdin."""
        if frame.shape[0] != self.target_height or frame.shape[1] != self.target_width:
            frame = self._fit_frame(frame)
        stdin = encoder.stdin
        if stdin is None:  # pragma: no cover - the encoder always has a pipe
            raise PipelineError("The FFmpeg encoder was started without an input pipe.")
        stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def _fit_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame to the exact target geometry when it does not match."""
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover
            return frame
        image = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
        resized = image.resize((self.target_width, self.target_height), lanczos_filter())
        return np.ascontiguousarray(np.asarray(resized)[:, :, ::-1])

    def _close_pipes(self, decoder: subprocess.Popen, encoder: subprocess.Popen) -> None:
        """Close the pipes and wait for both FFmpeg processes to exit."""
        if encoder.stdin:
            try:
                encoder.stdin.close()
            except OSError:
                pass
        try:
            encoder.wait(timeout=120)
        except subprocess.TimeoutExpired:
            encoder.kill()
        if decoder.stdout:
            try:
                decoder.stdout.close()
            except OSError:
                pass
        try:
            decoder.wait(timeout=120)
        except subprocess.TimeoutExpired:
            decoder.kill()

    def _mux(self, silent_video: Path, output_path: Path) -> None:
        """Copy the original audio into the encoded video."""
        config = self.config
        audio_codec = config.encoder.audio_codec
        source = Path(self.media.path)

        if not self.media.has_audio:
            # No audio present: the encoded file already is the final file.
            silent_video.replace(output_path)
            return

        cmd = build_mux_command(
            self.ffmpeg_bin,
            silent_video,
            source,
            output_path,
            audio_codec=audio_codec,
            audio_bitrate=config.encoder.audio_bitrate,
            container=config.encoder.container.value,
            overwrite=config.overwrite,
        )
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=_CREATE_NO_WINDOW,
        )
        if result.returncode != 0:
            raise FFmpegError(
                "FFmpeg failed while muxing the audio stream into the output file.",
                returncode=result.returncode,
                stderr=result.stderr,
            )

    def _encoder_name(self) -> str:
        """Map the configured codec onto an FFmpeg encoder name."""
        from .config import Codec

        mapping = {
            Codec.H264: "libx264",
            Codec.H265: "libx265",
            Codec.AV1: "libsvtav1",
        }
        return mapping.get(self.config.encoder.codec, "libx264")
