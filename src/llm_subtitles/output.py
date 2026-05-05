"""Output formatting for subtitle files (SRT, VTT, JSON, ASS)."""

from __future__ import annotations

import json
from pathlib import Path

from .models import SubtitleFile


def _ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT time format (HH:MM:SS,mmm)."""
    total_seconds = ms // 1000
    milliseconds = ms % 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _ms_to_vtt_time(ms: int) -> str:
    """Convert milliseconds to WebVTT time format (HH:MM:SS.mmm)."""
    total_seconds = ms // 1000
    milliseconds = ms % 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _ms_to_ass_time(ms: int) -> str:
    """Convert milliseconds to ASS time format (H:MM:SS.cc)."""
    total_seconds = ms // 1000
    centiseconds = (ms % 1000) // 10
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def format_srt(subtitle_file: SubtitleFile) -> str:
    """Format subtitles as SRT."""
    lines = []
    for i, seg in enumerate(subtitle_file.segments, 1):
        speaker_prefix = f"[Speaker {seg.speaker}] " if seg.speaker else ""
        lines.append(str(i))
        lines.append(f"{_ms_to_srt_time(seg.start_ms)} --> {_ms_to_srt_time(seg.end_ms)}")
        lines.append(f"{speaker_prefix}{seg.text}")
        lines.append("")
    return "\n".join(lines)


def format_vtt(subtitle_file: SubtitleFile) -> str:
    """Format subtitles as WebVTT."""
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(subtitle_file.segments, 1):
        speaker_prefix = f"[Speaker {seg.speaker}] " if seg.speaker else ""
        lines.append(f"{i}")
        lines.append(f"{_ms_to_vtt_time(seg.start_ms)} --> {_ms_to_vtt_time(seg.end_ms)}")
        lines.append(f"{speaker_prefix}{seg.text}")
        lines.append("")
    return "\n".join(lines)


def format_json(subtitle_file: SubtitleFile) -> str:
    """Format subtitles as JSON."""
    data = {
        "language": subtitle_file.language,
        "segments": [
            {
                "start_ms": seg.start_ms,
                "end_ms": seg.end_ms,
                "text": seg.text,
                "speaker": seg.speaker,
            }
            for seg in subtitle_file.segments
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def format_ass(subtitle_file: SubtitleFile) -> str:
    """Format subtitles as ASS (Advanced SubStation Alpha)."""
    lines = [
        "[Script Info]",
        "Title: llm-subtitles",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "PlayResX: 1280",
        "PlayResY: 720",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for seg in subtitle_file.segments:
        speaker_text = f"[Speaker {seg.speaker}] " if seg.speaker else ""
        text = seg.text.replace("\\", "\\\\").replace("\n", "\\N")
        lines.append(
            f"Dialogue: 0,{_ms_to_ass_time(seg.start_ms)},"
            f"{_ms_to_ass_time(seg.end_ms)},"
            f"Default,,0,0,0,,{speaker_text}{text}"
        )

    return "\n".join(lines)


FORMATTERS = {
    "srt": format_srt,
    "vtt": format_vtt,
    "json": format_json,
    "ass": format_ass,
}


def write_subtitles(
    subtitle_file: SubtitleFile,
    output_path: str,
    fmt: str = "srt",
) -> str:
    """Write subtitles to a file in the specified format.

    Args:
        subtitle_file: SubtitleFile object with segments.
        output_path: Output file path.
        fmt: Output format (srt, vtt, json, ass).

    Returns:
        The formatted string that was written.
    """
    formatter = FORMATTERS.get(fmt)
    if formatter is None:
        raise ValueError(f"Unknown format: {fmt}. Choose from: {list(FORMATTERS.keys())}")

    content = formatter(subtitle_file)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(content, encoding="utf-8")
    return content
