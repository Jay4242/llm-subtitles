"""Segment assembly with context windows for LLM transcription."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from . import audio as audio_mod
from .models import Chunk, VADSegment

logger = logging.getLogger(__name__)


def _prepare_save_dir(save_dir: str | None) -> Path | None:
    """Create or clean the save directory for audio segments."""
    if save_dir is None:
        return None
    path = Path(save_dir)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_chunk_audio(save_dir: Path | None, chunk: Chunk) -> None:
    """Save a chunk's audio bytes to a WAV file if save_dir is set."""
    if save_dir is None:
        return
    wav_file = save_dir / f"chunk_{chunk.index:04d}.wav"
    wav_file.write_bytes(chunk.audio_bytes)
    logger.debug(f"Saved segment audio: {wav_file}")


def assemble_chunks(
    wav_path: str,
    vad_segments: list[VADSegment],
    context_ms: int = 3000,
    cue_tone: bool = True,
    sample_rate: int = audio_mod.SAMPLE_RATE,
    save_dir: str | None = None,
) -> list[Chunk]:
    """Assemble audio chunks with context padding for each VAD segment.

    Args:
        wav_path: Path to the full WAV file.
        vad_segments: List of VAD-detected speech segments.
        context_ms: Padding in ms to add before and after each target segment.
        cue_tone: Whether to insert cue tones at context boundaries.
        sample_rate: Audio sample rate.

    Returns:
        List of Chunk objects ready for LLM transcription.
    """
    total_duration_ms = audio_mod.get_audio_duration_ms(wav_path)
    chunks = []
    save_path = _prepare_save_dir(save_dir)

    for i, seg in enumerate(vad_segments):
        target_start = seg.start_ms
        target_end = seg.end_ms

        if target_start >= target_end:
            target_end = target_start + 1

        pre_context_start = max(0, target_start - context_ms)
        post_context_end = min(total_duration_ms, target_end + context_ms)

        pre_context = audio_mod.extract_segment_audio(
            wav_path, pre_context_start, target_start
        )[0]
        target_audio = audio_mod.extract_segment_audio(
            wav_path, target_start, target_end
        )[0]
        post_context = audio_mod.extract_segment_audio(
            wav_path, target_end, post_context_end
        )[0]

        parts = []
        target_start_in_chunk_ms = 0
        target_end_in_chunk_ms = 0

        if cue_tone and pre_context_start > 0:
            tone = audio_mod.generate_cue_tone()
            parts.append(pre_context)
            parts.append(tone)
            target_start_in_chunk_ms = int(
                (len(pre_context) + len(tone)) / sample_rate * 1000
            )
        else:
            parts.append(pre_context)
            target_start_in_chunk_ms = int(len(pre_context) / sample_rate * 1000)

        parts.append(target_audio)
        target_end_in_chunk_ms = target_start_in_chunk_ms + int(
            len(target_audio) / sample_rate * 1000
        )

        if cue_tone and post_context_end < total_duration_ms:
            tone = audio_mod.generate_cue_tone()
            parts.append(tone)
            parts.append(post_context)
        else:
            parts.append(post_context)

        combined = audio_mod.concatenate_audio(parts, sample_rate)
        audio_bytes = audio_mod.audio_to_bytes(combined, sample_rate)
        total_chunk_ms = int(len(combined) / sample_rate * 1000)

        chunk = Chunk(
            index=i,
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            target_start_in_chunk_ms=target_start_in_chunk_ms,
            target_end_in_chunk_ms=target_end_in_chunk_ms,
            absolute_start_ms=target_start,
            absolute_end_ms=target_end,
            total_chunk_duration_ms=total_chunk_ms,
        )
        _save_chunk_audio(save_path, chunk)
        chunks.append(chunk)

    return chunks


def assemble_boundary_chunk(
    wav_path: str,
    seg_a: VADSegment,
    seg_b: VADSegment,
    context_ms: int = 5000,
    seam_overlap_ms: int = 1200,
    cue_tone: bool = True,
    sample_rate: int = audio_mod.SAMPLE_RATE,
) -> Chunk:
    """Assemble a boundary region chunk between two adjacent segments.

    Args:
        wav_path: Path to the full WAV file.
        seg_a: First segment (earlier in time).
        seg_b: Second segment (later in time).
        context_ms: Context padding around the boundary region.
        cue_tone: Whether to insert cue tones.
        sample_rate: Audio sample rate.

    Returns:
        Chunk covering the boundary region.
    """
    total_duration_ms = audio_mod.get_audio_duration_ms(wav_path)
    boundary_start = max(0, seg_a.end_ms - context_ms)
    boundary_end = min(total_duration_ms, seg_b.start_ms + context_ms)

    # The target region should include speech around the seam.
    # Using only the silence gap gives the model too little signal.
    target_start_abs = max(boundary_start, seg_a.end_ms - seam_overlap_ms)
    target_end_abs = min(boundary_end, seg_b.start_ms + seam_overlap_ms)

    if target_start_abs >= target_end_abs:
        target_end_abs = target_start_abs + 1

    # Extract audio segments
    pre_audio = audio_mod.extract_segment_audio(
        wav_path, boundary_start, target_start_abs
    )[0]
    target_audio = audio_mod.extract_segment_audio(
        wav_path, target_start_abs, target_end_abs
    )[0]
    post_audio = audio_mod.extract_segment_audio(
        wav_path, target_end_abs, boundary_end
    )[0]

    parts = []
    if cue_tone and boundary_start > 0:
        tone = audio_mod.generate_cue_tone()
        parts.append(pre_audio)
        parts.append(tone)
    else:
        parts.append(pre_audio)

    parts.append(target_audio)

    if cue_tone and boundary_end < total_duration_ms:
        tone = audio_mod.generate_cue_tone()
        parts.append(tone)
        parts.append(post_audio)
    else:
        parts.append(post_audio)

    combined = audio_mod.concatenate_audio(parts, sample_rate)
    audio_bytes = audio_mod.audio_to_bytes(combined, sample_rate)
    total_chunk_ms = int(len(combined) / sample_rate * 1000)

    # Compute target region offsets within the combined audio
    pre_duration_ms = int(len(pre_audio) / sample_rate * 1000)
    if cue_tone and boundary_start > 0:
        tone_duration_ms = int(audio_mod.generate_cue_tone().shape[0] / sample_rate * 1000)
        target_start_in_chunk_ms = pre_duration_ms + tone_duration_ms
    else:
        target_start_in_chunk_ms = pre_duration_ms

    target_end_in_chunk_ms = target_start_in_chunk_ms + int(
        len(target_audio) / sample_rate * 1000
    )

    return Chunk(
        index=-1,
        audio_bytes=audio_bytes,
        sample_rate=sample_rate,
        target_start_in_chunk_ms=target_start_in_chunk_ms,
        target_end_in_chunk_ms=target_end_in_chunk_ms,
        absolute_start_ms=target_start_abs,
        absolute_end_ms=target_end_abs,
        total_chunk_duration_ms=total_chunk_ms,
    )
