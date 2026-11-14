"""Rich + Typer command line interface."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from ..config import BackendName, Codec, Container, EncoderSettings, UpscaleConfig
from ..engine import default_output_path, run_upscale
from ..errors import UpscalerError
from ..ffmpeg_wrapper import probe
from ..models.registry import MODELS
from .commands import register as register_commands

app = typer.Typer(
    name="aivu",
    help=(
        "AI 4K video upscaler. Super-resolve low resolution videos to 4K with "
        "Real-ESRGAN, preserving the original audio bit-exactly."
    ),
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)
models_app = typer.Typer(help="Manage the downloadable Real-ESRGAN weights.", no_args_is_help=True)
app.add_typer(models_app, name="models")

console = Console()
err_console = Console(stderr=True)


def _force_utf8_output() -> None:
    """Make stdout and stderr tolerant of characters the console cannot encode.

    Windows consoles often use a legacy code page (cp1250, cp437, ...).  Rich
    draws box characters and braille spinners, which those code pages cannot
    encode.  Only the error handler is relaxed, so the console's own encoding
    is preserved and the output stays readable.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):  # pragma: no cover - platform specific
                pass


def _fail(message: str) -> None:
    """Print an error message and exit with a non-zero status."""
    err_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=1)


def _version_callback(value: bool) -> None:
    """Print the version and exit."""
    if value:
        from .. import __version__

        console.print(f"aivu {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """AI 4K video upscaler."""


@app.command()
def upscale(
    input_path: Path = typer.Argument(..., help="The video file to upscale."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file. Defaults to <input>_4k.<ext>."
    ),
    model: str = typer.Option(
        "realesr-animevideov3", "--model", "-m", help="The Real-ESRGAN model to use."
    ),
    backend: str | None = typer.Option(
        None, "--backend", "-b", help="Force a backend: ncnn, torch or classical."
    ),
    height: int | None = typer.Option(
        None, "--height", help="Target height in pixels. Defaults to 2160 (4K)."
    ),
    width: int | None = typer.Option(None, "--width", help="Target width in pixels."),
    scale: int | None = typer.Option(
        None, "--scale", help="Force a scale factor instead of a target resolution."
    ),
    container: str = typer.Option("mkv", "--container", help="Output container: mp4 or mkv."),
    codec: str = typer.Option("h264", "--codec", help="Video codec: h264, h265 or av1."),
    crf: int = typer.Option(18, "--crf", help="Encoder quality (lower is better)."),
    preset: str = typer.Option("slow", "--preset", help="Encoder speed/quality preset."),
    audio_codec: str = typer.Option(
        "copy", "--audio-codec", help="Audio codec, or 'copy' to keep it bit-exact."
    ),
    batch_size: int = typer.Option(4, "--batch", help="Frames per inference batch."),
    workers: int = typer.Option(1, "--workers", help="Parallel inference workers."),
    tile: int = typer.Option(
        0, "--tile", help="Inference tile size in pixels. 0 lets the backend decide."
    ),
    hwaccel: str = typer.Option("", "--hwaccel", help="Hardware encoder: cuda, qsv or vaapi."),
    ffmpeg_path: str = typer.Option("", "--ffmpeg", help="Path to the ffmpeg executable."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing output file."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only report what would happen."),
) -> None:
    """Upscale a video file to 4K."""
    if model not in MODELS:
        _fail(f"Unknown model '{model}'. Available: {', '.join(sorted(MODELS))}")

    try:
        container_enum = Container(container.lower())
        codec_enum = Codec(codec.lower())
    except ValueError:
        _fail("Invalid container or codec. Containers: mp4, mkv. Codecs: h264, h265, av1.")

    backend_enum: BackendName | None = None
    if backend:
        try:
            backend_enum = BackendName(backend.lower())
        except ValueError:
            _fail(f"Unknown backend '{backend}'. Valid backends: ncnn, torch, classical.")

    source = Path(input_path)
    destination = Path(output) if output else default_output_path(source)

    config = UpscaleConfig(
        input_path=source,
        output_path=destination,
        model=model,
        backend=backend_enum,
        target_width=width,
        target_height=height,
        scale=scale,
        tile=tile,
        batch_size=batch_size,
        workers=workers,
        overwrite=overwrite,
        ffmpeg_path=ffmpeg_path,
        encoder=EncoderSettings(
            codec=codec_enum,
            container=container_enum,
            crf=crf,
            preset=preset,
            audio_codec=audio_codec,
            hwaccel=hwaccel,
        ),
    )

    try:
        media = probe(source, ffmpeg_path)
    except UpscalerError as exc:
        _fail(str(exc))
        return

    """AI 4K video upscaler."""

    if dry_run:
        from ..pipeline import compute_target

        target_w, target_h = compute_target(
            media.video.width, media.video.height, width, height, scale
        )
        table = Table(title="Planned upscale", show_header=False, box=None)
        table.add_row("Input", str(source))
        table.add_row("Output", str(destination))
        table.add_row(
            "Source", f"{media.video.width}x{media.video.height} @ {media.video.fps:.3f} fps"
        )
        table.add_row("Target", f"{target_w}x{target_h}")
        table.add_row("Model", model)
        table.add_row("Backend", backend or "auto")
        table.add_row("Codec", f"{codec_enum.value} in {container_enum.value}")
        table.add_row("Audio", audio_codec)
        table.add_row("Frames", str(media.video.frame_count))
        console.print(Panel(table, title="Dry run", border_style="cyan"))
        return

    if destination.exists() and not overwrite:
        _fail(f"The output file already exists: {destination}. Use --overwrite to replace it.")

    console.print(
        f"[bold cyan]Upscaling[/bold cyan] {source} -> {destination}  "
        f"({media.video.width}x{media.video.height} -> 4K, model: {model})"
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("[green]{task.fields[rate]:.2f} fps"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    task_id = progress.add_task("Preparing", total=max(1, media.video.frame_count), rate=0.0)

    def on_progress(update: object) -> None:
        """Update the progress bar from the pipeline callback."""
        progress.update(
            task_id,
            completed=update.frames_done,
            total=max(1, update.frames_total),
            description=update.stage.capitalize(),
            rate=update.fps or 0.0,
        )

    try:
        with progress:
            result = run_upscale(config, progress_cb=on_progress)
    except UpscalerError as exc:
        progress.stop()
        _fail(str(exc))
        return
    except KeyboardInterrupt:
        progress.stop()
        err_console.print("[yellow]Cancelled by the user.[/yellow]")
        raise typer.Exit(code=130) from None

    table = Table(show_header=False, box=None)
    table.add_row("Output", str(result.output_path))
    table.add_row("Resolution", f"{result.source_resolution} -> {result.output_resolution}")
    table.add_row("Frames", f"{result.frames_processed}")
    table.add_row("Time", f"{result.elapsed_seconds:.1f} s ({result.average_fps:.2f} fps)")
    table.add_row("Backend", result.backend)
    table.add_row("Audio", "copied (lossless)" if result.audio_preserved else "none")
    console.print(Panel(table, title="Completed", border_style="green"))


register_commands(app, models_app, console)


def main() -> None:
    """Console script entry point."""
    _force_utf8_output()
    app()


if __name__ == "__main__":
    main()
