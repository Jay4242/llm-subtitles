"""Audio extraction and normalization using ffmpeg."""

from __future__ import annotations

import subprocess
import tempfile

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


class AudioExtractionError(Exception):
    """Raised when audio extraction fails."""


def extract_audio(
    input_path: str,
    output_path: str | None = None,
    audio_track: int | None = None,
    language: str | None = None,
    verbose: bool = False,
) -> str:
    """Extract audio from a media file and convert to 16kHz mono 16-bit PCM WAV.

    Args:
        input_path: Path to input media file.
        output_path: Optional output WAV path. If None, a temp file is created.
        audio_track: Optional audio track index.
        language: Optional language code for stream selection.
        verbose: Whether to print ffmpeg output on failure.

    Returns:
        Path to the extracted WAV file.
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="llm-subtitles-")
        import os
        os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-sample_fmt", "s16",
    ]

    if audio_track is not None:
        cmd.extend(["-map", f"0:a:{audio_track}"])

    if language:
        cmd.extend(["-metadata:s:a:0", f"language={language}"])

    cmd.append(output_path)

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        msg = f"ffmpeg extraction failed for {input_path}"
        if verbose:
            msg += f"\nstderr:\n{e.stderr}"
        raise AudioExtractionError(msg) from e

    return output_path


def load_audio(wav_path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file and return the waveform and sample rate.

    Args:
        wav_path: Path to WAV file.

    Returns:
        Tuple of (waveform as float32 numpy array, sample_rate).
    """
    data, sr = sf.read(wav_path, dtype="float32")
    return data, sr


def get_audio_duration_ms(wav_path: str) -> int:
    """Get the duration of an audio file in milliseconds."""
    data, sr = sf.read(wav_path, dtype="float32")
    return int(len(data) / sr * 1000)


def extract_segment_audio(
    wav_path: str,
    start_ms: int,
    end_ms: int,
    output_path: str | None = None,
) -> tuple[np.ndarray, int]:
    """Extract a segment of audio from a WAV file.

    Args:
        wav_path: Path to source WAV file.
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.
        output_path: Optional output path for the segment WAV.

    Returns:
        Tuple of (segment waveform as float32 numpy array, sample_rate).
    """
    data, sr = sf.read(wav_path, dtype="float32")
    start_sample = int(start_ms / 1000 * sr)
    end_sample = int(end_ms / 1000 * sr)
    start_sample = max(0, min(start_sample, len(data)))
    end_sample = max(0, min(end_sample, len(data)))
    segment = data[start_sample:end_sample]

    if output_path:
        sf.write(output_path, segment, sr, subtype="PCM_16")

    return segment, sr


def generate_cue_tone(
    duration_ms: int = 200,
    frequency_hz: int = 1000,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generate a brief cue tone (sine wave).

    Args:
        duration_ms: Duration in milliseconds.
        frequency_hz: Frequency of the tone.
        sample_rate: Sample rate.

    Returns:
        Numpy array with the tone waveform.
    """
    n_samples = int(duration_ms / 1000 * sample_rate)
    t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * frequency_hz * t)
    return tone.astype(np.float32)


def concatenate_audio(
    segments: list[np.ndarray],
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Concatenate multiple audio segments into one.

    Args:
        segments: List of numpy arrays (audio waveforms).
        sample_rate: Sample rate (all segments must have same rate).

    Returns:
        Single concatenated numpy array.
    """
    if not segments:
        return np.array([], dtype=np.float32)
    return np.concatenate(segments)


def audio_to_bytes(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Convert a numpy audio array to WAV bytes.

    Args:
        audio: Numpy array of audio samples (float32).
        sample_rate: Sample rate.

    Returns:
        WAV file as bytes.
    """
    import io
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
