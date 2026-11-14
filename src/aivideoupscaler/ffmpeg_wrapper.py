"""Robust wrapper around the ``ffmpeg`` / ``ffprobe`` binaries.

Everything that touches the video container goes through this module.  It
provides binary discovery, media probing, raw-frame extraction, encoding and
lossless audio muxing, with friendly errors for the failure modes users hit
most often (missing FFmpeg, corrupt inputs, unsupported codecs).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import AudioSpec, MediaInfo, VideoSpec
from .errors import (
    CorruptMediaError,
    FFmpegError,
    FFmpegNotFoundError,
    MediaProbeError,
)
from .paths import ENV_FFMPEG, ENV_FFPROBE, resolve_binary

#: Places FFmpeg commonly lives on Windows / macOS / Linux.
_COMMON_BIN_DIRS = (
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\ProgramData\chocolatey\bin",
    "/usr/local/bin",
    "/usr/bin",
    "/opt/homebrew/bin",
    "/opt/local/bin",
    "/snap/bin",
)

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _candidate_dirs() -> list[str]:
    """Return the directories searched for FFmpeg, most specific first."""
    dirs: list[str] = []
    extra = os.environ.get("FFMPEG_DIR")
    if extra:
        dirs.append(extra)
    dirs.extend(_COMMON_BIN_DIRS)
    return dirs


def _binary_names(name: str) -> tuple[str, ...]:
    """Return the platform-specific file names for a tool."""
    return (f"{name}.exe", name) if sys.platform == "win32" else (name,)


def _search_on_disk(name: str) -> str | None:
    """Look for an executable in well-known installation directories."""
    for directory in _candidate_dirs():
        for candidate_name in _binary_names(name):
            candidate = Path(directory) / candidate_name
            if candidate.is_file():
                return str(candidate)
    return None


def find_ffmpeg(explicit: str = "") -> str | None:
    """Locate the ``ffmpeg`` executable.

    Search order: explicit path, ``FFMPEG_BINARY`` env var, bundled ``bin/``
    directory, ``PATH``, then well-known install locations.
    """
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path.resolve())
        found = shutil.which(explicit)
        if found:
            return found
        raise FFmpegNotFoundError(f"ffmpeg executable not found at '{explicit}'")

    bundled = resolve_binary("ffmpeg", ENV_FFMPEG)
    if bundled:
        return str(bundled)

    found = shutil.which("ffmpeg")
    if found:
        return found

    return _search_on_disk("ffmpeg")


def find_ffprobe(explicit: str = "") -> str | None:
    """Locate the ``ffprobe`` executable."""
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path.resolve())
        found = shutil.which(explicit)
        if found:
            return found
        raise FFmpegNotFoundError(f"ffprobe executable not found at '{explicit}'")

    bundled = resolve_binary("ffprobe", ENV_FFPROBE)
    if bundled:
        return str(bundled)

    found = shutil.which("ffprobe")
    if found:
        return found

    return _search_on_disk("ffprobe")


def require_ffmpeg(explicit: str = "") -> str:
    """Return the ffmpeg path or raise :class:`FFmpegNotFoundError`."""
    path = find_ffmpeg(explicit)
    if not path:
        raise FFmpegNotFoundError()
    return path


def require_ffprobe(explicit: str = "") -> str:
    """Return the ffprobe path or raise :class:`FFmpegNotFoundError`."""
    path = find_ffprobe(explicit)
    if not path:
        raise FFmpegNotFoundError(
            "ffprobe was not found. It ships with FFmpeg and is required to inspect media."
        )
    return path


def run_command(cmd: Sequence[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a command, capturing stdout and stderr as text."""
    try:
        return subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=_CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"Command timed out: {' '.join(cmd[:3])}...") from exc
    except OSError as exc:
        raise FFmpegError(f"Could not execute '{cmd[0]}': {exc}") from exc


def ffmpeg_version(ffmpeg_bin: str = "") -> str:
    """Return the FFmpeg version banner, or an empty string when unavailable."""
    path = find_ffmpeg(ffmpeg_bin)
    if not path:
        return ""
    result = run_command([path, "-version"])
    if result.returncode != 0 or not result.stdout:
        return ""
    return result.stdout.splitlines()[0].strip()


def ffprobe_version(ffprobe_bin: str = "") -> str:
    """Return the ffprobe version banner, or an empty string."""
    path = find_ffprobe(ffprobe_bin)
    if not path:
        return ""
    result = run_command([path, "-version"])
    if result.returncode != 0 or not result.stdout:
        return ""
    return result.stdout.splitlines()[0].strip()


def list_encoders(ffmpeg_bin: str = "") -> list[str]:
    """Return the names of every encoder compiled into this FFmpeg build."""
    path = find_ffmpeg(ffmpeg_bin)
    if not path:
        return []
    result = run_command([path, "-hide_banner", "-encoders"])
    if result.returncode != 0:
        return []
    encoders: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        # Lines look like: " V....D libx264   H.264 encoder ..." -> flags, name, description
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in ("V", "A"):
            encoders.append(parts[1])
    return encoders


def has_encoder(encoder: str, ffmpeg_bin: str = "") -> bool:
    """Return ``True`` when ``encoder`` is compiled into this FFmpeg build."""
    if not encoder:
        return False
    return encoder in list_encoders(ffmpeg_bin)


def parse_rational(value: str) -> float:
    """Convert an FFmpeg rational value such as ``30000/1001`` into a float."""
    if not value:
        return 0.0
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            denominator = float(den)
            if denominator == 0:
                return 0.0
            return float(num) / denominator
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_or(value: object, default: int = 0) -> int:
    """Best-effort integer conversion that never raises."""
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def probe(path: Path | str, ffprobe_bin: str = "") -> MediaInfo:
    """Inspect a media file and return its :class:`MediaInfo`.

    Raises :class:`CorruptMediaError` for unreadable or truncated files and
    :class:`MediaProbeError` when the file contains no usable video track.
    """
    media_path = Path(path)
    if not media_path.exists():
        raise MediaProbeError(f"Input file does not exist: {media_path}")
    if media_path.is_dir():
        raise MediaProbeError(f"Input path is a directory, not a file: {media_path}")
    if media_path.stat().st_size == 0:
        raise CorruptMediaError(f"Input file is empty: {media_path}")

    probe_bin = require_ffprobe(ffprobe_bin)
    cmd = [
        probe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(media_path),
    ]
    result = run_command(cmd, timeout=120)
    if result.returncode != 0:
        message = (result.stderr or "").strip()
        lowered = message.lower()
        if any(
            token in lowered
            for token in ("invalid data", "moov atom not found", "truncated", "corrupt")
        ):
            raise CorruptMediaError(
                f"The file appears to be corrupted or incomplete: {media_path}\n{message}"
            )
        raise MediaProbeError(f"ffprobe failed on {media_path}: {message}")

    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaProbeError(f"ffprobe returned invalid JSON for {media_path}: {exc}") from exc

    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise MediaProbeError(f"No video track found in {media_path}.")

    duration = parse_rational(str(fmt.get("duration") or video_stream.get("duration") or "0"))
    if duration <= 0:
        duration = parse_rational(str(video_stream.get("duration") or "0"))

    fps = parse_rational(str(video_stream.get("avg_frame_rate") or "0"))
    if fps <= 0:
        fps = parse_rational(str(video_stream.get("r_frame_rate") or "0"))

    frame_count = _int_or(video_stream.get("nb_frames"), 0)
    if frame_count <= 0 and fps > 0 and duration > 0:
        frame_count = round(fps * duration)

    rotation = 0
    for side_data in video_stream.get("side_data_list") or []:
        if "rotation" in side_data:
            rotation = _int_or(side_data.get("rotation"), 0) % 360

    if rotation in (90, 270):
        width = _int_or(video_stream.get("height"))
        height = _int_or(video_stream.get("width"))
    else:
        width = _int_or(video_stream.get("width"))
        height = _int_or(video_stream.get("height"))

    video = VideoSpec(
        codec=str(video_stream.get("codec_name") or "unknown"),
        width=width,
        height=height,
        fps=fps if fps > 0 else 25.0,
        duration=duration,
        frame_count=frame_count,
        bitrate=_int_or(fmt.get("bit_rate"), 0),
        pix_fmt=str(video_stream.get("pix_fmt") or ""),
    )

    audios: list[AudioSpec] = []
    for stream in streams:
        if stream.get("codec_type") != "audio":
            continue
        tags = stream.get("tags") or {}
        audios.append(
            AudioSpec(
                index=_int_or(stream.get("index")),
                codec=str(stream.get("codec_name") or "unknown"),
                channels=_int_or(stream.get("channels"), 0),
                sample_rate=_int_or(stream.get("sample_rate"), 0),
                language=str(tags.get("language") or ""),
            )
        )

    return MediaInfo(
        path=media_path,
        format_name=str(fmt.get("format_name") or ""),
        duration=duration,
        size_bytes=media_path.stat().st_size,
        video=video,
        audios=audios,
        subtitles=sum(1 for s in streams if s.get("codec_type") == "subtitle"),
        bitrate=_int_or(fmt.get("bit_rate"), 0),
    )


def _parse_fps_flag() -> str:
    """Return the frame-rate flag supported by the local FFmpeg build.

    FFmpeg 5.1 deprecated ``-vsync`` in favour of ``-fps_mode`` and FFmpeg 9
    removed it entirely, so the correct spelling depends on the version.
    """
    global _FPS_FLAG
    if _FPS_FLAG is not None:
        return _FPS_FLAG

    version = ffmpeg_version()
    major = 0
    for token in version.split():
        if token.startswith("version") and len(token) > 1 and token[1].isdigit():
            major = int(token.split(".")[0])
            break
        if token[0].isdigit() and "." in token:
            try:
                major = int(token.split(".")[0])
                break
            except ValueError:
                continue

    _FPS_FLAG = "-fps_mode" if major >= 5 else "-vsync"
    return _FPS_FLAG


_FPS_FLAG: str | None = None


def build_decode_command(
    ffmpeg_bin: str,
    input_path: Path,
    width: int,
    height: int,
    fps: float,
    start: float = 0.0,
    duration: float = 0.0,
) -> list[str]:
    """Build the ffmpeg command that decodes video into raw BGR24 frames."""
    cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(input_path)]
    if duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [
        "-an",
        "-sn",
        "-dn",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-vf",
        f"scale={width}:{height}:flags=bicubic",
    ]
    # Pass the frames through untouched so that no frame is duplicated or
    # dropped and the output keeps the source timing.
    flag = _parse_fps_flag()
    cmd += [flag, "passthrough" if flag == "-fps_mode" else "0"]
    cmd += ["-"]
    return cmd


def build_encode_command(
    ffmpeg_bin: str,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "slow",
    pix_fmt: str = "yuv420p",
    hwaccel: str = "",
    extra_args: Sequence[str] = (),
    overwrite: bool = True,
) -> list[str]:
    """Build the ffmpeg command that encodes raw BGR24 frames into a file."""
    cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
    cmd += ["-y"] if overwrite else ["-n"]
    cmd += [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-sn",
        "-c:v",
    ]

    if hwaccel == "cuda":
        cmd += ["h264_nvenc", "-preset", "p4", "-cq", str(max(0, crf))]
    elif hwaccel == "qsv":
        cmd += ["h264_qsv", "-preset", "medium", "-global_quality", str(max(0, crf))]
    elif hwaccel == "vaapi":
        cmd += ["h264_vaapi", "-qp", str(max(0, crf))]
    else:
        cmd += [codec]
        if codec in ("libx264", "libx265"):
            cmd += ["-crf", str(crf), "-preset", preset]
        elif codec == "libsvtav1":
            numeric = int(preset) if preset.isdigit() else 8
            cmd += ["-crf", str(crf), "-preset", str(max(0, min(13, numeric)))]

    cmd += ["-pix_fmt", pix_fmt]
    if extra_args:
        cmd += list(extra_args)
    cmd += ["-movflags", "+faststart", str(output_path)]
    return cmd


def build_mux_command(
    ffmpeg_bin: str,
    video_path: Path,
    source_path: Path,
    output_path: Path,
    audio_codec: str = "copy",
    audio_bitrate: str = "192k",
    container: str = "mkv",
    overwrite: bool = True,
) -> list[str]:
    """Build the ffmpeg command that muxes the original audio back in.

    With ``audio_codec='copy'`` the audio stream is transferred bit-exactly,
    which satisfies the "lossless audio" requirement.
    """
    cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
    cmd += ["-y"] if overwrite else ["-n"]
    cmd += [
        "-i",
        str(video_path),
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
    ]
    if audio_codec == "copy":
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", audio_codec, "-b:a", audio_bitrate]
    if container == "mp4":
        cmd += ["-movflags", "+faststart"]
    cmd += [str(output_path)]
    return cmd
