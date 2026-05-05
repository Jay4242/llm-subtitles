"""Voice Activity Detection using Silero-VAD (ONNX).

Segmentation uses a dual-threshold hysteresis state machine adapted from the
whisper.cpp VAD implementation, plus post-processing (merge, filter, pad).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .models import VADSegment

logger = logging.getLogger(__name__)


class VADConfig:
    """Configuration for Silero-VAD."""

    MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
    FRAME_SIZE = 512
    SAMPLE_RATE = 16000

    def __init__(
        self,
        threshold: float = 0.5,
        neg_threshold: float | None = None,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        max_segment_duration_ms: int | None = None,
        speech_pad_ms: int = 0,
    ):
        self.threshold = threshold
        self.neg_threshold = (
            neg_threshold if neg_threshold is not None
            else max(0.01, threshold - 0.15)
        )
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.max_segment_duration_ms = max_segment_duration_ms
        self.speech_pad_ms = speech_pad_ms


def _download_model(model_path: Path) -> Path:
    """Download the Silero-VAD ONNX model if not present."""
    import shutil
    import tempfile
    import urllib.request

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Silero-VAD model...")
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".onnx")
    try:
        urllib.request.urlretrieve(VADConfig.MODEL_URL, tmp_path)
        shutil.move(tmp_path, model_path)
    finally:
        try:
            import os

            os.unlink(tmp_path)
        except OSError:
            pass
    return model_path


def _get_model_path() -> Path:
    """Get the path to the Silero-VAD model, downloading if needed."""
    try:
        import importlib_resources as impresources

        model_file = str(impresources.files("silero_vad.data").joinpath("silero_vad.onnx"))
        if Path(model_file).exists():
            return Path(model_file)
    except Exception:
        pass

    try:
        from importlib import resources as impresources

        with impresources.path("silero_vad.data", "silero_vad.onnx") as f:
            if f.exists():
                return f
    except Exception:
        pass

    project_cache = (
        Path(__file__).resolve().parent.parent.parent / ".cache" / "silero_vad.onnx"
    )
    if project_cache.exists():
        return project_cache

    cache_dir = Path.home() / ".cache" / "llm-subtitles"
    model_path = cache_dir / "silero_vad.onnx"
    if not model_path.exists():
        return _download_model(model_path)
    return model_path


def _get_onnx_session():
    """Create an ONNX Runtime inference session for Silero-VAD."""
    import onnxruntime as ort

    model_path = _get_model_path()
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), providers=providers)


def run_vad(
    audio: np.ndarray,
    sample_rate: int = VADConfig.SAMPLE_RATE,
    config: VADConfig | None = None,
) -> list[VADSegment]:
    """Run Silero-VAD on audio and return speech segments.

    Args:
        audio: Audio waveform as float32 numpy array.
        sample_rate: Sample rate (must be 16000).
        config: VAD configuration.

    Returns:
        List of VADSegment objects with start/end times in ms.
    """
    if config is None:
        config = VADConfig()

    if sample_rate != VADConfig.SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=VADConfig.SAMPLE_RATE)
        sample_rate = VADConfig.SAMPLE_RATE

    session = _get_onnx_session()
    speech_probs = _compute_speech_probs(audio, session)
    segments = _segment_speech_probs(speech_probs, config)

    if not segments:
        logger.warning("No speech detected in audio")

    logger.info(f"VAD found {len(segments)} speech segments")
    return segments


def _compute_speech_probs(
    audio: np.ndarray,
    session,
) -> np.ndarray:
    """Compute frame-level speech probabilities using Silero-VAD v6."""
    frame_size = VADConfig.FRAME_SIZE
    context_size = 64
    n_frames = len(audio) // frame_size
    probs = np.zeros(n_frames, dtype=np.float32)

    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(context_size, dtype=np.float32)

    for i in range(n_frames):
        frame = audio[i * frame_size: (i + 1) * frame_size]
        frame_with_context = np.concatenate([context, frame])
        frame_tensor = frame_with_context.reshape(1, -1)

        ort_inputs = {
            "input": frame_tensor,
            "sr": np.array(VADConfig.SAMPLE_RATE, dtype=np.int64),
            "state": state,
        }
        ort_outs = session.run(None, ort_inputs)
        prob, state = ort_outs
        probs[i] = prob[0][0]
        context = frame_with_context[-context_size:]

    return probs


def _segment_speech_probs(
    speech_probs: np.ndarray,
    config: VADConfig,
) -> list[VADSegment]:
    """Segment raw VAD probabilities into speech regions.

    Uses a dual-threshold hysteresis state machine (adapted from whisper.cpp)
    followed by post-processing: merge adjacent, remove short, apply padding.
    """
    frame_size = VADConfig.FRAME_SIZE
    sample_rate = VADConfig.SAMPLE_RATE

    min_speech_samples = int(config.min_speech_duration_ms / 1000 * sample_rate)
    min_silence_samples = int(config.min_silence_duration_ms / 1000 * sample_rate)
    speech_pad_samples = int(config.speech_pad_ms / 1000 * sample_rate)
    min_silence_at_max = int(sample_rate * 98 / 1000)

    # Max samples a speech segment can contain before force-splitting.
    if config.max_segment_duration_ms is None:
        max_speech_samples = len(speech_probs) * frame_size  # effectively no limit
    else:
        max_speech_samples = (
            int(config.max_segment_duration_ms / 1000 * sample_rate)
            - frame_size
            - 2 * speech_pad_samples
        )
        if max_speech_samples <= 0:
            max_speech_samples = len(speech_probs) * frame_size

    threshold = config.threshold
    neg_threshold = config.neg_threshold
    audio_length_samples = len(speech_probs) * frame_size

    speeches: list[tuple[int, int]] = []  # (start_sample, end_sample)

    is_speech = False
    temp_end = 0
    prev_end = 0
    next_start = 0
    curr_speech_start = 0

    for i, prob in enumerate(speech_probs):
        curr_sample = frame_size * i

        if prob >= threshold and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = curr_sample

        if prob >= threshold and not is_speech:
            is_speech = True
            curr_speech_start = curr_sample
            continue

        if is_speech and (curr_sample - curr_speech_start) > max_speech_samples:
            if prev_end:
                speeches.append((curr_speech_start, prev_end))
                if next_start < prev_end:
                    is_speech = False
                else:
                    curr_speech_start = next_start
                prev_end = next_start = temp_end = 0
            else:
                speeches.append((curr_speech_start, curr_sample))
                prev_end = next_start = temp_end = 0
                is_speech = False
                continue

        if prob < neg_threshold and is_speech:
            if not temp_end:
                temp_end = curr_sample

            if (curr_sample - temp_end) > min_silence_at_max:
                prev_end = temp_end

            if (curr_sample - temp_end) < min_silence_samples:
                continue
            else:
                if (temp_end - curr_speech_start) > min_speech_samples:
                    speeches.append((curr_speech_start, temp_end))
                prev_end = next_start = temp_end = 0
                is_speech = False
                continue

    if is_speech and (audio_length_samples - curr_speech_start) > min_speech_samples:
        speeches.append((curr_speech_start, audio_length_samples))

    speeches = _merge_adjacent_segments(speeches, sample_rate)
    speeches = _remove_short_segments(speeches, min_speech_samples)
    speeches = _apply_segment_padding(speeches, speech_pad_samples, audio_length_samples)

    segments = []
    for start, end in speeches:
        segments.append(
            VADSegment(
                start_ms=int(start / sample_rate * 1000),
                end_ms=int(end / sample_rate * 1000),
            )
        )
    return segments


def _merge_adjacent_segments(
    speeches: list[tuple[int, int]],
    sample_rate: int,
    max_merge_gap_ms: int = 200,
) -> list[tuple[int, int]]:
    """Merge adjacent speech segments whose gap is within *max_merge_gap_ms*."""
    if len(speeches) < 2:
        return speeches

    max_gap_samples = int(sample_rate * max_merge_gap_ms / 1000)
    merged = list(speeches)
    i = 0

    while i < len(merged) - 1:
        gap = merged[i + 1][0] - merged[i][1]
        if gap < max_gap_samples:
            merged[i] = (merged[i][0], merged[i + 1][1])
            merged.pop(i + 1)
        else:
            i += 1

    return merged


def _remove_short_segments(
    speeches: list[tuple[int, int]],
    min_speech_samples: int,
) -> list[tuple[int, int]]:
    """Remove segments shorter than *min_speech_samples*."""
    return [(s, e) for s, e in speeches if (e - s) >= min_speech_samples]


def _apply_segment_padding(
    speeches: list[tuple[int, int]],
    speech_pad_samples: int,
    audio_length_samples: int,
) -> list[tuple[int, int]]:
    """Apply padding around each segment, handling overlaps between neighbours."""
    if not speeches:
        return speeches

    result = list(speeches)

    for i in range(len(result)):
        if i == 0:
            result[i] = (
                max(0, result[i][0] - speech_pad_samples),
                result[i][1],
            )

        if i < len(result) - 1:
            silence = result[i + 1][0] - result[i][1]
            if silence < 2 * speech_pad_samples:
                half = silence // 2
                result[i] = (result[i][0], result[i][1] + half)
                result[i + 1] = (max(0, result[i + 1][0] - half), result[i + 1][1])
            else:
                result[i] = (
                    result[i][0],
                    min(result[i][1] + speech_pad_samples, audio_length_samples),
                )
                result[i + 1] = (
                    max(0, result[i + 1][0] - speech_pad_samples),
                    result[i + 1][1],
                )
        else:
            result[i] = (
                result[i][0],
                min(result[i][1] + speech_pad_samples, audio_length_samples),
            )

    return result
