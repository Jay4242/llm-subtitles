"""Core data models for the llm-subtitles pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VADSegment:
    """A single speech segment detected by VAD."""
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class TranscriptionSegment:
    """A single subtitle segment from LLM transcription."""
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class TranscriptionResult:
    """Complete transcription result from a single chunk."""
    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str = "en"
    confidence: float = 1.0


@dataclass
class Chunk:
    """An audio chunk with context for LLM transcription."""
    index: int
    audio_bytes: bytes
    sample_rate: int
    target_start_in_chunk_ms: int
    target_end_in_chunk_ms: int
    absolute_start_ms: int
    absolute_end_ms: int
    total_chunk_duration_ms: int


@dataclass
class SubtitleFile:
    """A complete subtitle file with segments."""
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str = "en"

    def add_segment(self, segment: TranscriptionSegment) -> None:
        self.segments.append(segment)

    def sort_segments(self) -> None:
        self.segments.sort(key=lambda s: s.start_ms)

    def merge_overlaps(self) -> None:
        if len(self.segments) < 2:
            return
        self.segments.sort(key=lambda s: s.start_ms)
        merged = [self.segments[0]]
        for seg in self.segments[1:]:
            prev = merged[-1]
            if seg.start_ms < prev.end_ms:
                midpoint = (prev.end_ms + seg.start_ms) // 2
                prev.end_ms = midpoint
                seg.start_ms = midpoint
            merged.append(seg)
        self.segments = merged

    def normalize_durations(
        self,
        min_display_ms: int = 1000,
        max_display_ms: int = 7000,
    ) -> None:
        for seg in self.segments:
            duration = seg.end_ms - seg.start_ms
            if duration < min_display_ms:
                seg.end_ms = seg.start_ms + min_display_ms
            elif duration > max_display_ms:
                seg.end_ms = seg.start_ms + max_display_ms

        # Re-merge overlaps created by extending durations
        self.merge_overlaps()
