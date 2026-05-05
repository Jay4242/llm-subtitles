"""Tests for configuration loading."""

import tempfile

from llm_subtitles.config import AppConfig, load_config


class TestConfigDefaults:
    def test_default_backend(self):
        config = AppConfig()
        assert config.backend.api_base == "http://localhost:9090/v1"
        assert config.backend.model == ""
        assert config.backend.api_key == "not-needed"

    def test_default_segmentation(self):
        config = AppConfig()
        assert config.segmentation.vad_threshold == 0.5
        assert config.segmentation.min_speech_duration_ms == 250
        assert config.segmentation.min_silence_duration_ms == 300
        assert config.segmentation.max_segment_duration_ms == 10000

    def test_default_context(self):
        config = AppConfig()
        assert config.context.padding_ms == 0
        assert config.context.cue_tone is False

    def test_default_refinement(self):
        config = AppConfig()
        assert config.refinement.boundary_refinement is False
        assert config.refinement.aligner == "forced"

    def test_default_output(self):
        config = AppConfig()
        assert config.output.format == "srt"


class TestConfigFileLoading:
    def test_load_full_config(self):
        config_content = """
[backend]
api_base = "http://example.com/v1"
model = "test-model"
api_key = "secret"

[segmentation]
vad_threshold = 0.7
min_speech_duration_ms = 500
min_silence_duration_ms = 400
max_segment_duration_ms = 20000

[context]
padding_ms = 5000
cue_tone = false

[refinement]
boundary_refinement = false
aligner = "llm-only"

[output]
format = "vtt"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            f.flush()
            config = load_config(f.name)

        assert config.backend.api_base == "http://example.com/v1"
        assert config.backend.model == "test-model"
        assert config.backend.api_key == "secret"
        assert config.segmentation.vad_threshold == 0.7
        assert config.segmentation.min_speech_duration_ms == 500
        assert config.context.padding_ms == 5000
        assert config.context.cue_tone is False
        assert config.refinement.boundary_refinement is False
        assert config.refinement.aligner == "llm-only"
        assert config.output.format == "vtt"

    def test_load_partial_config(self):
        config_content = """
[backend]
model = "partial-model"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            f.flush()
            config = load_config(f.name)

        assert config.backend.model == "partial-model"
        assert config.backend.api_base == "http://localhost:9090/v1"
        assert config.output.format == "srt"

    def test_no_config_file(self):
        config = load_config()
        assert config.backend.api_base == "http://localhost:9090/v1"
        assert config.backend.model == ""
