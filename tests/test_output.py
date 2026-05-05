"""Tests for output formatting."""

import json
import tempfile
from pathlib import Path

import pytest

from llm_subtitles.models import SubtitleFile, TranscriptionSegment
from llm_subtitles.output import (
    format_ass,
    format_json,
    format_srt,
    format_vtt,
    write_subtitles,
)


def _make_subtitle_file() -> SubtitleFile:
    sf = SubtitleFile(language="en")
    sf.add_segment(TranscriptionSegment(start_ms=0, end_ms=2000, text="Hello world"))
    sf.add_segment(
        TranscriptionSegment(start_ms=3000, end_ms=5000, text="This is a test")
    )
    return sf


def _make_subtitle_with_speakers() -> SubtitleFile:
    sf = SubtitleFile(language="en")
    sf.add_segment(
        TranscriptionSegment(start_ms=0, end_ms=2000, text="Hello", speaker="A")
    )
    sf.add_segment(
        TranscriptionSegment(start_ms=3000, end_ms=5000, text="Hi there", speaker="B")
    )
    return sf


class TestSRT:
    def test_basic_format(self):
        sf = _make_subtitle_file()
        output = format_srt(sf)
        assert "1" in output
        assert "00:00:00,000 --> 00:00:02,000" in output
        assert "Hello world" in output
        assert "2" in output
        assert "00:00:03,000 --> 00:00:05,000" in output
        assert "This is a test" in output

    def test_with_speakers(self):
        sf = _make_subtitle_with_speakers()
        output = format_srt(sf)
        assert "[Speaker A]" in output
        assert "[Speaker B]" in output

    def test_empty(self):
        sf = SubtitleFile()
        output = format_srt(sf)
        assert output == ""


class TestVTT:
    def test_basic_format(self):
        sf = _make_subtitle_file()
        output = format_vtt(sf)
        assert output.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:02.000" in output
        assert "Hello world" in output

    def test_with_speakers(self):
        sf = _make_subtitle_with_speakers()
        output = format_vtt(sf)
        assert "[Speaker A]" in output


class TestJSON:
    def test_basic_format(self):
        sf = _make_subtitle_file()
        output = format_json(sf)
        data = json.loads(output)
        assert data["language"] == "en"
        assert len(data["segments"]) == 2
        assert data["segments"][0]["text"] == "Hello world"
        assert data["segments"][0]["start_ms"] == 0
        assert data["segments"][0]["end_ms"] == 2000

    def test_with_speakers(self):
        sf = _make_subtitle_with_speakers()
        output = format_json(sf)
        data = json.loads(output)
        assert data["segments"][0]["speaker"] == "A"


class TestASS:
    def test_basic_format(self):
        sf = _make_subtitle_file()
        output = format_ass(sf)
        assert "[Script Info]" in output
        assert "[V4+ Styles]" in output
        assert "[Events]" in output
        assert "Hello world" in output

    def test_with_speakers(self):
        sf = _make_subtitle_with_speakers()
        output = format_ass(sf)
        assert "[Speaker A]" in output


class TestWriteSubtitles:
    def test_write_srt(self):
        sf = _make_subtitle_file()
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            write_subtitles(sf, f.name, "srt")
            content = Path(f.name).read_text()
            assert "Hello world" in content

    def test_write_vtt(self):
        sf = _make_subtitle_file()
        with tempfile.NamedTemporaryFile(suffix=".vtt", delete=False) as f:
            write_subtitles(sf, f.name, "vtt")
            content = Path(f.name).read_text()
            assert "WEBVTT" in content

    def test_write_json(self):
        sf = _make_subtitle_file()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            write_subtitles(sf, f.name, "json")
            data = json.loads(Path(f.name).read_text())
            assert len(data["segments"]) == 2

    def test_invalid_format(self):
        sf = _make_subtitle_file()
        with pytest.raises(ValueError):
            write_subtitles(sf, "/tmp/test.xyz", "xyz")
