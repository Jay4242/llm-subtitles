# llm-subtitles — Agent Guide

## Setup

```bash
pip install -e .           # base
pip install -e ".[dev]"    # + pytest, ruff
pip install -e ".[forced-alignment]"  # + torch, torchaudio
```

Dev environment is already at `.venv/` (Python 3.13).

## Commands

| Command | Description |
|---|---|
| `pytest` | All tests (asyncio_mode = auto) |
| `ruff check .` | Lint (line-length=100, target=py310, selects E/F/I/N/W/UP) |
| `pytest tests/test_output.py::TestSRT` | Single test class |
| `pytest -k "test_parse"` | Keyword-filtered test |

## Entrypoint

`src/llm_subtitles/cli.py` → `llm_subtitles.cli:main` (click group).

**Commands:** `transcribe`, `burn`, `mux`, `transcode`, `segment`

`--model` is **required** (no default). `--api-base` defaults to `http://localhost:9090/v1`.

The `--format` CLI flag is aliased to `fmt` internally (avoids shadowing Python builtin).

## Architecture

Single package `src/llm_subtitles/`, hatchling build. Pipeline stages:

1. `audio.py` — ffmpeg → 16kHz mono WAV
2. `vad.py` — Silero-VAD ONNX (auto-downloads model to `~/.cache/llm-subtitles/silero_vad.onnx`)
3. `segments.py` — context windows + optional cue tones
4. `llm_client.py` — async httpx → OpenAI-compatible API (`input_audio` parts)
5. `transcribe.py` — orchestrates pipeline, concurrency via asyncio.Semaphore
6. `alignment.py` — wav2vec2 forced alignment (requires `[forced-alignment]` extras)
7. `output.py` — SRT/VTT/JSON/ASS formatters

Config: `.llm-subtitles.toml` or `~/.config/llm-subtitles/config.toml`

## What's NOT implemented (from SPEC.md)

These are future plans — DO NOT write code for them: two-pass pipeline, confidence refinement, acoustic event tagging, translation mode, streaming mode, Python library API, Docker image, CI/CD, benchmarks.

## Gotchas

- VAD downloads the ONNX model on first use (silent HTTP GET from GitHub).
- Forced alignment silently skips if torch/torchaudio not installed (logs a warning).
- Boundary refinement uses a fixed 5s context window (see `_refine_boundaries` in `transcribe.py:217`).
- Tests don't require an LLM server — they test JSON parsing, config loading, output formatting, and data models.
- `audio_sample.m4a` is used for manual testing (gitignored).
- ALWAYS run `pytest`, `ruff`, and `pip` through `.venv/bin/` (e.g. `.venv/bin/python -m pytest`). These tools are not on the system PATH.
