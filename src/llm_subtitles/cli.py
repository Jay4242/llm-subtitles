"""CLI interface using click."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .config import load_config
from .output import FORMATTERS

console = Console()


def _setup_logging(verbose: bool, log_file: str | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list = [RichHandler(console=console, show_time=False, show_path=False)]
    if log_file:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s: %(message)s")
        )
        handlers.append(fh)
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers, force=True)


@click.group()
@click.version_option(version=__version__, prog_name="llm-subtitles")
@click.option("--api-base", default=None, help="OpenAI-compatible API base URL")
@click.option("--model", default=None, help="Model name (required)")
@click.option("--api-key", default=None, help="API key (default: not-needed)")
@click.option("--concurrency", default=None, type=int, help="Max parallel LLM requests")
@click.option("--language", default=None, help="Expected language (ISO 639-1) or 'auto'")
@click.option("--output", default=None, help="Output file path")
@click.option(
    "--format", "fmt", default=None,
    type=click.Choice(list(FORMATTERS.keys())),
    help="Output format",
)
@click.option("--speakers", is_flag=True, default=None, help="Enable speaker diarization")
@click.option("--context-ms", default=None, type=int, help="Context window padding in ms")
@click.option("--max-segment-ms", default=None, type=int, help="Max subtitle segment duration")
@click.option("--vad-threshold", default=None, type=float, help="Silero-VAD threshold")
@click.option(
    "--boundary-refinement/--no-boundary-refinement", default=None,
    help="Enable cross-chunk boundary refinement",
)
@click.option(
    "--aligner", default=None, type=click.Choice(["forced", "llm-only"]),
    help="Timestamp refinement mode",
)
@click.option("--temp-dir", default=None, help="Working directory for intermediate files")
@click.option("--keep-temp", is_flag=True, default=False, help="Keep intermediate files")
@click.option("--verbose", is_flag=True, default=False, help="Verbose logging")
@click.option(
    "--debug", is_flag=True, default=False,
    help="Stream LLM reasoning and content during transcription",
)
@click.option(
    "--log-file", default=None,
    help="Debug log file path (default: ./llm-subtitles-debug.log when --debug)",
)
@click.pass_context
def main(ctx, **kwargs):
    """llm-subtitles: Transcribe audio/video into timestamped subtitles using local LLMs."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = kwargs.get("verbose", False)
    ctx.obj["debug"] = kwargs.get("debug", False)
    log_file = kwargs.get("log_file")
    if ctx.obj["debug"] and not log_file:
        log_file = "./llm-subtitles-debug.log"
    ctx.obj["log_file"] = log_file
    _setup_logging(ctx.obj["verbose"], log_file)

    config = load_config()
    if kwargs.get("api_base"):
        config.backend.api_base = kwargs["api_base"]
    if kwargs.get("model"):
        config.backend.model = kwargs["model"]
    if kwargs.get("api_key"):
        config.backend.api_key = kwargs["api_key"]
    if kwargs.get("concurrency") is not None:
        ctx.obj["concurrency"] = kwargs["concurrency"]
    if kwargs.get("language"):
        ctx.obj["language"] = kwargs["language"]
    if kwargs.get("output"):
        ctx.obj["output"] = kwargs["output"]
    if kwargs.get("fmt"):
        ctx.obj["format"] = kwargs["fmt"]
    if kwargs.get("speakers") is not None:
        ctx.obj["speakers"] = kwargs["speakers"]
    if kwargs.get("context_ms") is not None:
        ctx.obj["context_ms"] = kwargs["context_ms"]
    if kwargs.get("max_segment_ms") is not None:
        config.segmentation.max_segment_duration_ms = kwargs["max_segment_ms"]
    if kwargs.get("vad_threshold") is not None:
        config.segmentation.vad_threshold = kwargs["vad_threshold"]
    if kwargs.get("boundary_refinement") is not None:
        config.refinement.boundary_refinement = kwargs["boundary_refinement"]
    if kwargs.get("aligner"):
        config.refinement.aligner = kwargs["aligner"]
    if kwargs.get("temp_dir"):
        ctx.obj["temp_dir"] = kwargs["temp_dir"]
    if kwargs.get("keep_temp"):
        ctx.obj["keep_temp"] = True

    ctx.obj["config"] = config


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.pass_context
def transcribe(ctx, input_file):
    """Extract audio, transcribe, output subtitle file."""
    config = ctx.obj["config"]
    if not config.backend.model:
        console.print("[red]Error: --model is required. Set it via CLI flag or config file.[/red]")
        sys.exit(1)

    from .transcribe import PipelineError
    from .transcribe import transcribe as run_transcribe

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Transcribing...", total=None)

        try:
            result = run_transcribe(
                input_path=input_file,
                config=config,
                output_path=ctx.obj.get("output"),
                output_format=ctx.obj.get("format"),
                concurrency=ctx.obj.get("concurrency", 1),
                speakers=ctx.obj.get("speakers", False),
                language=ctx.obj.get("language"),
                context_ms=ctx.obj.get("context_ms"),
                keep_temp=ctx.obj.get("keep_temp", False),
                temp_dir=ctx.obj.get("temp_dir"),
                verbose=ctx.obj.get("verbose", False),
                debug=ctx.obj.get("debug", False),
            )
            desc = f"Done! {len(result.segments)} subtitle segments created."
            progress.update(task, description=desc)
        except PipelineError as e:
            console.print(f"[red]Pipeline error: {e}[/red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            if ctx.obj.get("verbose"):
                import traceback
                traceback.print_exc()
            sys.exit(1)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.pass_context
def burn(ctx, input_file):
    """Transcribe and hard-burn subtitles into the video."""
    from .transcribe import transcribe as run_transcribe

    config = ctx.obj["config"]
    srt_path = Path(input_file).stem + ".srt"

    console.print("Step 1: Transcribing...")
    run_transcribe(
        input_path=input_file,
        config=config,
        output_path=srt_path,
        output_format="srt",
        concurrency=ctx.obj.get("concurrency", 1),
        speakers=ctx.obj.get("speakers", False),
        language=ctx.obj.get("language"),
        context_ms=ctx.obj.get("context_ms"),
        keep_temp=ctx.obj.get("keep_temp", False),
        temp_dir=ctx.obj.get("temp_dir"),
        verbose=ctx.obj.get("verbose", False),
        debug=ctx.obj.get("debug", False),
    )

    console.print("Step 2: Burning subtitles into video...")
    output_path = ctx.obj.get("output") or Path(input_file).stem + "_burned.mp4"

    import subprocess
    srt_path_escaped = str(srt_path).replace("\\", "/").replace(":", r"\\:")
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-vf", f"subtitles='{srt_path_escaped}'",
        "-c:a", "copy",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        console.print(f"[green]Video with burned subtitles: {output_path}[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]ffmpeg burn failed: {e.stderr.decode()}[/red]")
        sys.exit(1)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.pass_context
def mux(ctx, input_file):
    """Transcribe and embed as soft subtitle track."""
    from .transcribe import transcribe as run_transcribe

    config = ctx.obj["config"]
    srt_path = Path(input_file).stem + ".srt"

    console.print("Step 1: Transcribing...")
    run_transcribe(
        input_path=input_file,
        config=config,
        output_path=srt_path,
        output_format="srt",
        concurrency=ctx.obj.get("concurrency", 1),
        speakers=ctx.obj.get("speakers", False),
        language=ctx.obj.get("language"),
        context_ms=ctx.obj.get("context_ms"),
        keep_temp=ctx.obj.get("keep_temp", False),
        temp_dir=ctx.obj.get("temp_dir"),
        verbose=ctx.obj.get("verbose", False),
        debug=ctx.obj.get("debug", False),
    )

    console.print("Step 2: Muxing subtitle track...")
    output_path = ctx.obj.get("output") or Path(input_file).stem + "_subtitled.mkv"

    import subprocess
    srt_path_escaped = str(srt_path).replace("\\", "/").replace(":", r"\\:")
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-vf", f"subtitles='{srt_path_escaped}'",
        "-c:a", "copy",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        console.print(f"[green]Video with subtitle track: {output_path}[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]ffmpeg mux failed: {e.stderr.decode()}[/red]")
        sys.exit(1)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--audio-track", default=None, type=int, help="Audio track index")
@click.pass_context
def transcode(ctx, input_file, audio_track):
    """Only extract/normalize audio (for debugging)."""
    from . import audio as audio_mod

    output_path = Path(input_file).stem + ".wav"
    wav_path = audio_mod.extract_audio(
        input_file,
        output_path=str(output_path),
        audio_track=audio_track,
        verbose=ctx.obj.get("verbose", False),
    )
    duration_ms = audio_mod.get_audio_duration_ms(wav_path)
    console.print(f"Audio extracted: {wav_path} ({duration_ms}ms)")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.pass_context
def segment(ctx, input_file):
    """Only run VAD and output segments (for debugging)."""
    from . import audio as audio_mod
    from .vad import VADConfig, run_vad

    config = ctx.obj["config"]
    console.print("Extracting audio...")
    wav_path = audio_mod.extract_audio(
        input_file,
        verbose=ctx.obj.get("verbose", False),
    )

    console.print("Running VAD...")
    audio_data, sr = audio_mod.load_audio(wav_path)
    vad_config = VADConfig(
        threshold=config.segmentation.vad_threshold,
        min_speech_duration_ms=config.segmentation.min_speech_duration_ms,
        min_silence_duration_ms=config.segmentation.min_silence_duration_ms,
        max_segment_duration_ms=config.segmentation.max_segment_duration_ms,
    )
    segments = run_vad(audio_data, sr, vad_config)

    console.print(f"\nFound {len(segments)} speech segments:\n")
    for i, seg in enumerate(segments):
        start_s = seg.start_ms / 1000
        end_s = seg.end_ms / 1000
        duration_s = seg.duration_ms / 1000
        console.print(f"  [{i+1:3d}] {start_s:8.2f}s - {end_s:8.2f}s ({duration_s:6.2f}s)")

    import json
    output_path = Path(input_file).stem + "_segments.json"
    seg_data = [
        {"index": i, "start_ms": s.start_ms, "end_ms": s.end_ms, "duration_ms": s.duration_ms}
        for i, s in enumerate(segments)
    ]
    Path(output_path).write_text(json.dumps(seg_data, indent=2))
    console.print(f"\nSegments saved to: {output_path}")


if __name__ == "__main__":
    main()
