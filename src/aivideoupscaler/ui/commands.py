"""Additional CLI commands: ``info``, ``doctor``, ``models`` and ``gui``."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..engine import describe_environment
from ..errors import UpscalerError
from ..ffmpeg_wrapper import probe
from ..models.downloader import download_model, download_ncnn_bundle, is_model_present
from ..models.registry import list_models
from ..paths import binaries_dir, models_dir


def register(app: typer.Typer, models_app: typer.Typer, console: Console) -> None:
    """Attach the auxiliary commands to the Typer application."""

    @app.command()
    def info(
        input_path: Path = typer.Argument(..., help="The video file to inspect."),
    ) -> None:
        """Show detailed information about a video file."""
        try:
            media = probe(input_path)
        except UpscalerError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

        table = Table(show_header=False, box=None)
        table.add_row("Path", str(media.path))
        table.add_row("Container", media.format_name)
        table.add_row("Video", f"{media.video.codec} {media.video.resolution_label}")
        table.add_row("Frame rate", f"{media.video.fps:.3f} fps")
        table.add_row("Duration", f"{media.duration:.2f} s")
        table.add_row("Frames", str(media.video.frame_count))
        table.add_row("Pixel format", media.video.pix_fmt)
        table.add_row("Size", f"{media.size_bytes / (1024 * 1024):.2f} MiB")
        if media.audios:
            for audio in media.audios:
                table.add_row(
                    f"Audio #{audio.index}",
                    f"{audio.codec} {audio.layout} {audio.sample_rate} Hz",
                )
        else:
            table.add_row("Audio", "none")
        console.print(Panel(table, title="Media information", border_style="cyan"))

    @app.command()
    def doctor() -> None:
        """Check that the environment is ready to upscale videos."""
        environment = describe_environment()

        table = Table(show_header=False, box=None)
        table.add_row("FFmpeg", environment["ffmpeg"] or "[red]not found[/red]")
        table.add_row("ffprobe", environment["ffprobe"] or "[red]not found[/red]")
        table.add_row("CPU threads", str(environment["cpu_count"]))
        table.add_row("Models folder", str(models_dir()))
        table.add_row("Binaries folder", str(binaries_dir()))
        console.print(Panel(table, title="Environment", border_style="cyan"))

        backend_table = Table(title="Backends")
        backend_table.add_column("Backend")
        backend_table.add_column("Status")
        backend_table.add_column("Device")
        backend_table.add_column("Details")
        for status in environment["backends"]:  # type: ignore[index]
            marker = "[green]available[/green]" if status.available else "[red]unavailable[/red]"
            backend_table.add_row(status.name, marker, status.device, status.detail)
        console.print(backend_table)

        model_table = Table(title="Models")
        model_table.add_column("Model")
        model_table.add_column("Installed")
        for model in list_models():
            present = "[green]yes[/green]" if is_model_present(model.key) else "[yellow]no[/yellow]"
            model_table.add_row(model.key, present)
        console.print(model_table)

        if not environment["ffmpeg"]:
            console.print(
                "[yellow]FFmpeg is required.[/yellow] Install it system-wide, set FFMPEG_BINARY, "
                "or place the executable in the 'bin' folder."
            )

    @models_app.command("list")
    def models_list() -> None:
        """List the available Real-ESRGAN models."""
        table = Table(title="Available models", expand=False)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Name", no_wrap=True)
        table.add_column("Scale", justify="right", no_wrap=True)
        table.add_column("Installed", no_wrap=True)
        table.add_column("Description", ratio=1, overflow="fold")
        for model in list_models():
            present = "[green]yes[/green]" if is_model_present(model.key) else "[yellow]no[/yellow]"
            table.add_row(
                model.key,
                model.display_name,
                f"x{model.scale}",
                present,
                model.description,
            )
        console.print(table)

    @models_app.command("install")
    def models_install(
        model: str | None = typer.Option(
            None, "--model", "-m", help="Install a single model. Omit to install all."
        ),
        ncnn: bool = typer.Option(
            False, "--ncnn", help="Also download the ncnn/Vulkan binary and networks."
        ),
        force: bool = typer.Option(False, "--force", help="Re-download even if present."),
    ) -> None:
        """Download the model weights."""
        from ..models.registry import MODELS

        selected = [model] if model else list(MODELS)
        for key in selected:
            if key not in MODELS:
                console.print(f"[bold red]Error:[/bold red] unknown model '{key}'")
                raise typer.Exit(code=1)

        for key in selected:
            with console.status(f"[bold cyan]Downloading {key}...[/bold cyan]"):
                try:
                    path = download_model(key, force=force)
                except UpscalerError as exc:
                    console.print(f"[bold red]Error:[/bold red] {exc}")
                    raise typer.Exit(code=1) from exc
            console.print(f"[green]installed[/green] {key} -> {path}")

        if ncnn:
            with console.status("[bold cyan]Downloading the ncnn/Vulkan bundle...[/bold cyan]"):
                try:
                    archive = download_ncnn_bundle(force=force)
                    from ..backends.ncnn import install_ncnn_bundle

                    executable = install_ncnn_bundle(archive)
                except UpscalerError as exc:
                    console.print(f"[bold red]Error:[/bold red] {exc}")
                    raise typer.Exit(code=1) from exc
            console.print(f"[green]installed[/green] ncnn/Vulkan -> {executable}")

    @app.command()
    def gui() -> None:
        """Launch the graphical interface."""
        try:
            from .gui import run_gui
        except ImportError as exc:
            console.print(
                "[bold red]Error:[/bold red] the GUI needs PySide6. "
                "Install it with 'pip install aivideoupscaler[gui]'."
            )
            raise typer.Exit(code=1) from exc
        run_gui()
