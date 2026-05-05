"""Tests for LLM client."""

import json

from llm_subtitles.llm_client import LLMClient


class TestLLMClientParsing:
    def setup_method(self):
        self.client = LLMClient(
            api_base="http://localhost:8080/v1",
            model="test-model",
        )

    def test_parse_valid_json(self):
        content = json.dumps({
            "text": "Hello world",
            "segments": [
                {"start_ms": 0, "end_ms": 2000, "text": "Hello world"}
            ],
            "language": "en",
            "confidence": 0.95,
        })
        result = self.client._parse_response(content)
        assert result.text == "Hello world"
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world"
        assert result.language == "en"
        assert result.confidence == 0.95

    def test_parse_json_with_surrounding_text(self):
        content = "Here is the result:\n```json\n" + json.dumps({
            "text": "Test",
            "segments": [{"start_ms": 0, "end_ms": 1000, "text": "Test"}],
            "language": "en",
            "confidence": 1.0,
        }) + "\n```"
        result = self.client._parse_response(content)
        assert result.text == "Test"

    def test_parse_invalid_json(self):
        content = "This is not JSON at all"
        result = self.client._parse_response(content)
        assert result.text == "This is not JSON at all"
        assert result.segments == []

    def test_parse_with_speaker(self):
        content = json.dumps({
            "text": "Hello",
            "segments": [
                {"start_ms": 0, "end_ms": 1000, "text": "Hello", "speaker": "A"}
            ],
            "language": "en",
            "confidence": 1.0,
        })
        result = self.client._parse_response(content)
        assert result.segments[0].speaker == "A"

    def test_build_transcription_prompt(self):
        prompt = self.client._build_transcription_prompt(1000, 5000, speakers=False)
        assert "TARGET" in prompt
        assert "CONTEXT" in prompt
        assert '"speaker"' not in prompt
        assert "Timestamps are handled separately" in prompt

    def test_build_transcription_prompt_with_speakers(self):
        prompt = self.client._build_transcription_prompt(1000, 5000, speakers=True)
        assert "speaker" in prompt.lower()

    def test_build_transcription_prompt_with_language(self):
        prompt = self.client._build_transcription_prompt(1000, 5000, language="es")
        assert "Spanish" in prompt or "es" in prompt

    def test_build_boundary_refinement_prompt(self):
        prompt = self.client._build_boundary_refinement_prompt("hello", "world")
        assert "hello" in prompt
        assert "world" in prompt
        assert "boundary" in prompt.lower()
