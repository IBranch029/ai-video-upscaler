# Contributing

Thanks for helping to improve the AI 4K video upscaler.

## Getting started

```bash
git clone https://github.com/IBranch029/ai-video-upscaler.git
cd ai-video-upscaler
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / macOS

pip install -e ".[dev]"
```

FFmpeg is needed for the integration tests:

```bash
# Windows
winget install Gyan.FFmpeg
# Ubuntu / Debian
sudo apt install ffmpeg
# macOS
brew install ffmpeg
```

## Before opening a pull request

```bash
ruff check src tests scripts
ruff format --check src tests scripts
mypy src/aivideoupscaler --ignore-missing-imports
pytest tests -q
```

All three commands must pass.

## Coding conventions

- **Docstrings on every public function.** Describe *why*, not just *what*.
- **Type hints everywhere.** The code is checked with `mypy`.
- **Errors derive from `UpscalerError`** and carry a message that tells the
  user what to do next.
- **Keep the core dependency-light.** NumPy and Pillow are the only hard
  requirements; PyTorch and PySide6 are optional extras.
- **Follow the existing architecture.** New behaviour belongs in the module
  that owns the concern.

## Commit messages

The project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add a tiled inference mode
fix: keep the frame order when several workers run
docs: describe the ncnn backend
chore: bump the ruff version
```

## Tests

- Add a test for every bug fix.
- Integration tests that need FFmpeg are marked with `@pytest.mark.requires_ffmpeg`
  and skip automatically when FFmpeg is missing.
- Tests that need PyTorch are marked with `@pytest.mark.requires_torch`.

## Reporting bugs

Please include the output of:

```bash
aivu doctor
aivu info <the problematic file>
```

That usually identifies the problem immediately.
