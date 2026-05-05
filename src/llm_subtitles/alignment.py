"""Forced alignment using wav2vec2 + CTC via torchaudio."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .models import TranscriptionResult, TranscriptionSegment

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def run_forced_alignment_full(
    all_results: list[TranscriptionResult],
    audio: np.ndarray,
    sample_rate: int,
) -> list[TranscriptionResult]:
    """Run forced alignment on full audio against all segment texts combined.

    Collects all segment text across results, concatenates it, and runs a single
    wav2vec2+CTC forced alignment pass against the full audio file. Aligned
    word-level timestamps are mapped back to individual segments using exact
    word indexing — no fuzzy matching needed.

    Args:
        all_results: List of TranscriptionResult objects from LLM transcription.
        audio: Full audio waveform as float32 numpy array.
        sample_rate: Sample rate (must be 16000).

    Returns:
        Same results list with updated segment timestamps.
    """
    try:
        import torch
        import torchaudio
    except ImportError:
        logger.warning("torch/torchaudio not available, skipping forced alignment")
        return all_results

    all_segments: list[TranscriptionSegment] = []
    for result in all_results:
        for seg in result.segments:
            if seg.text.strip():
                all_segments.append(seg)

    if not all_segments:
        return all_results

    model_name = "WAV2VEC2_ASR_BASE_960H"
    try:
        bundle = torchaudio.pipelines.__dict__[model_name]
        model = bundle.get_model()
        model.eval()
    except (KeyError, Exception) as e:
        logger.warning(f"Could not load wav2vec2 model: {e}")
        return all_results

    with torch.no_grad():
        waveform = torch.from_numpy(audio).unsqueeze(0)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=bundle.sample_rate
            )

        emissions, _ = model(waveform)

    full_text = " ".join(seg.text.strip() for seg in all_segments)
    tokens, separator_idx = _text_to_tokens(full_text, bundle)
    targets = torch.tensor([tokens], dtype=torch.int32)

    try:
        paths, scores = torchaudio.functional.forced_align(
            emissions, targets, blank=0
        )
        word_timestamps = _extract_word_timestamps(
            paths[0], list(bundle.get_labels()), bundle.sample_rate, separator_idx
        )
        _apply_timestamps_full(all_segments, word_timestamps)
    except Exception as e:
        logger.warning(f"Forced alignment failed: {e}")

    return all_results


def _apply_timestamps_full(
    segments: list[TranscriptionSegment],
    word_timestamps: list[tuple[str, float, float]],
) -> None:
    """Apply aligned word timestamps to segments using exact word indexing.

    Unlike the chunk-by-chunk version, no fuzzy matching is needed because
    we know the exact word sequence from the concatenated segment texts.
    """
    if not word_timestamps:
        return

    all_segment_words = [w for seg in segments for w in seg.text.strip().split()]

    if len(word_timestamps) != len(all_segment_words):
        logger.warning(
            "Word count mismatch: alignment produced %d words, "
            "segments have %d words. Using min count.",
            len(word_timestamps), len(all_segment_words),
        )

    word_idx = 0
    for seg in segments:
        seg_words = seg.text.strip().split()
        n = len(seg_words)
        if n == 0:
            continue
        if word_idx >= len(word_timestamps):
            break

        first_ts = word_timestamps[word_idx]
        last_idx = min(word_idx + n - 1, len(word_timestamps) - 1)
        last_ts = word_timestamps[last_idx]

        seg.start_ms = int(first_ts[1] * 1000)
        seg.end_ms = int(last_ts[2] * 1000)
        word_idx += n


def run_forced_alignment(
    result: TranscriptionResult,
    audio: np.ndarray,
    sample_rate: int,
    base_offset_ms: int = 0,
) -> TranscriptionResult:
    """Run forced alignment on a transcription result using wav2vec2.

    Args:
        result: TranscriptionResult with text and approximate timestamps.
        audio: Audio waveform as float32 numpy array.
        sample_rate: Sample rate (must be 16000).

    Returns:
        TranscriptionResult with refined timestamps.
    """
    try:
        import torch
        import torchaudio
    except ImportError:
        logger.warning("torch/torchaudio not available, skipping forced alignment")
        return result

    if not result.segments:
        return result

    model_name = "WAV2VEC2_ASR_BASE_960H"
    try:
        bundle = torchaudio.pipelines.__dict__[model_name]
        model = bundle.get_model()
        model.eval()
    except (KeyError, Exception) as e:
        logger.warning(f"Could not load wav2vec2 model: {e}")
        return result

    with torch.no_grad():
        waveform = torch.from_numpy(audio).unsqueeze(0)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=bundle.sample_rate
            )

        emissions, _ = model(waveform)

    full_text = " ".join(seg.text for seg in result.segments)
    tokens, separator_idx = _text_to_tokens(full_text, bundle)
    targets = torch.tensor([tokens], dtype=torch.int32)
    labels = list(bundle.get_labels())

    try:
        paths, scores = torchaudio.functional.forced_align(
            emissions, targets, blank=0
        )
        word_timestamps = _extract_word_timestamps(
            paths[0], labels, bundle.sample_rate, separator_idx
        )
        _apply_timestamps_to_segments(result, word_timestamps, full_text, base_offset_ms)
    except Exception as e:
        logger.warning(f"Forced alignment failed: {e}")

    return result


def _text_to_tokens(text: str, bundle) -> tuple[list[int], int]:
    """Convert text to token IDs using the model's dictionary.

    Returns:
        Tuple of (token_ids, separator_idx) where separator_idx is the
        index of the word-separator token ('|' in wav2vec2).
    """
    labels = bundle.get_labels()
    dictionary = {label: i for i, label in enumerate(labels)}
    blank_idx = dictionary.get("-", 0)
    separator_idx = dictionary.get("|", 1)

    tokens = []
    for char in text.upper():
        if char == " ":
            tokens.append(separator_idx)
        elif char in dictionary:
            idx = dictionary[char]
            if idx != blank_idx:
                tokens.append(idx)
    return tokens, separator_idx


def _extract_word_timestamps(
    path: torch.Tensor,
    labels: list[str],
    sample_rate: int,
    separator_idx: int = 1,
) -> list[tuple[str, float, float]]:
    """Extract word-level timestamps from CTC alignment path.

    Args:
        path: Per-frame token sequence from forced_align (shape `(T,)`).
        labels: List of label strings from the model bundle.
        sample_rate: Audio sample rate.
        separator_idx: Token ID for word separator ('|').

    Returns:
        List of (word, start_seconds, end_seconds) tuples.
    """
    # wav2vec2 emits approximately one frame every 20 ms.
    frame_duration = 0.02
    word_timestamps = []
    current_word = ""
    word_start = None
    blank_idx = 0

    for frame_idx in range(len(path)):
        token_id = path[frame_idx].item()

        if token_id == blank_idx:
            continue

        if token_id == separator_idx:
            if current_word and word_start is not None:
                time_s = frame_idx * frame_duration
                word_timestamps.append((current_word, word_start, time_s))
            current_word = ""
            word_start = None
        else:
            if word_start is None:
                word_start = frame_idx * frame_duration
            current_word += labels[token_id] if token_id < len(labels) else ""

    if current_word and word_start is not None:
        word_timestamps.append((current_word, word_start, word_start + frame_duration))

    return word_timestamps


def _apply_timestamps_to_segments(
    result: TranscriptionResult,
    word_timestamps: list[tuple[str, float, float]],
    full_text: str,
    base_offset_ms: int = 0,
) -> None:
    """Apply aligned word timestamps to subtitle segments."""
    if not word_timestamps:
        return

    words = full_text.split()
    if len(word_timestamps) != len(words):
        logger.warning(
            f"Word count mismatch: alignment produced {len(word_timestamps)} "
            f"words, text has {len(words)} words. Attempting fuzzy match."
        )
        _apply_timestamps_fuzzy(result, word_timestamps, words, base_offset_ms)
        return

    ts_idx = 0

    for seg in result.segments:
        seg_text = seg.text.strip()
        if not seg_text:
            continue

        seg_words = seg_text.split()
        seg_start_ms = None
        seg_end_ms = None

        for word in seg_words:
            while ts_idx < len(word_timestamps):
                tw, ts, te = word_timestamps[ts_idx]
                if tw.lower() in word.lower() or word.lower() in tw.lower():
                    if seg_start_ms is None:
                        seg_start_ms = int(ts * 1000)
                    seg_end_ms = int(te * 1000)
                    ts_idx += 1
                    break
                else:
                    ts_idx += 1

        if seg_start_ms is not None and seg_end_ms is not None:
            seg.start_ms = base_offset_ms + seg_start_ms
            seg.end_ms = base_offset_ms + seg_end_ms


def _apply_timestamps_fuzzy(
    result: TranscriptionResult,
    word_timestamps: list[tuple[str, float, float]],
    words: list[str],
    base_offset_ms: int = 0,
) -> None:
    """Apply timestamps using fuzzy matching when word counts differ."""
    if not word_timestamps:
        return

    ts_idx = 0
    for seg in result.segments:
        seg_text = seg.text.strip()
        if not seg_text:
            continue

        seg_words = seg_text.lower().split()
        seg_start_ms = None
        seg_end_ms = None

        for seg_word in seg_words:
            best_match_idx = None
            best_score = 0
            for j in range(ts_idx, min(ts_idx + 5, len(word_timestamps))):
                tw, ts, te = word_timestamps[j]
                tw_lower = tw.lower()
                seg_lower = seg_word.lower()
                if tw_lower == seg_lower:
                    score = 1.0
                elif tw_lower in seg_lower or seg_lower in tw_lower:
                    score = 0.8
                else:
                    score = 0.0
                if score > best_score:
                    best_score = score
                    best_match_idx = j

            if best_match_idx is not None and best_score > 0.5:
                _, ts, te = word_timestamps[best_match_idx]
                if seg_start_ms is None:
                    seg_start_ms = int(ts * 1000)
                seg_end_ms = int(te * 1000)
                ts_idx = best_match_idx + 1

        if seg_start_ms is not None and seg_end_ms is not None:
            seg.start_ms = base_offset_ms + seg_start_ms
            seg.end_ms = base_offset_ms + seg_end_ms
