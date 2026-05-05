"""Tests for data models."""


from llm_subtitles.models import (
    SubtitleFile,
    TranscriptionResult,
    TranscriptionSegment,
    VADSegment,
)


class TestVADSegment:
    def test_duration(self):
        seg = VADSegment(start_ms=1000, end_ms=5000)
        assert seg.duration_ms == 4000

    def test_zero_duration(self):
        seg = VADSegment(start_ms=1000, end_ms=1000)
        assert seg.duration_ms == 0


class TestTranscriptionSegment:
    def test_duration(self):
        seg = TranscriptionSegment(start_ms=1000, end_ms=3000, text="hello world")
        assert seg.duration_ms == 2000

    def test_with_speaker(self):
        seg = TranscriptionSegment(start_ms=0, end_ms=1000, text="hello", speaker="A")
        assert seg.speaker == "A"

    def test_without_speaker(self):
        seg = TranscriptionSegment(start_ms=0, end_ms=1000, text="hello")
        assert seg.speaker is None


class TestTranscriptionResult:
    def test_default_values(self):
        result = TranscriptionResult(text="hello")
        assert result.language == "en"
        assert result.confidence == 1.0
        assert result.segments == []

    def test_with_segments(self):
        seg = TranscriptionSegment(start_ms=0, end_ms=1000, text="hello")
        result = TranscriptionResult(text="hello", segments=[seg])
        assert len(result.segments) == 1


class TestSubtitleFile:
    def test_add_segment(self):
        sf = SubtitleFile()
        sf.add_segment(TranscriptionSegment(start_ms=0, end_ms=1000, text="hello"))
        assert len(sf.segments) == 1

    def test_sort_segments(self):
        sf = SubtitleFile()
        sf.add_segment(TranscriptionSegment(start_ms=2000, end_ms=3000, text="second"))
        sf.add_segment(TranscriptionSegment(start_ms=0, end_ms=1000, text="first"))
        sf.sort_segments()
        assert sf.segments[0].text == "first"
        assert sf.segments[1].text == "second"

    def test_merge_overlaps(self):
        sf = SubtitleFile()
        sf.add_segment(TranscriptionSegment(start_ms=0, end_ms=2000, text="first"))
        sf.add_segment(TranscriptionSegment(start_ms=1500, end_ms=3000, text="second"))
        sf.merge_overlaps()
        assert sf.segments[0].end_ms <= sf.segments[1].start_ms

    def test_normalize_durations_min(self):
        sf = SubtitleFile()
        sf.add_segment(TranscriptionSegment(start_ms=0, end_ms=500, text="short"))
        sf.normalize_durations(min_display_ms=1000)
        assert sf.segments[0].end_ms - sf.segments[0].start_ms >= 1000

    def test_normalize_durations_max(self):
        sf = SubtitleFile()
        sf.add_segment(TranscriptionSegment(start_ms=0, end_ms=10000, text="long"))
        sf.normalize_durations(max_display_ms=7000)
        assert sf.segments[0].end_ms - sf.segments[0].start_ms <= 7000

    def test_empty_merge_overlaps(self):
        sf = SubtitleFile()
        sf.merge_overlaps()
        assert len(sf.segments) == 0
