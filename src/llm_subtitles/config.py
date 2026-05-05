"""Configuration loading from TOML files and CLI defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import toml


@dataclass
class BackendConfig:
    api_base: str = "http://localhost:9090/v1"
    model: str = ""
    api_key: str = "not-needed"


@dataclass
class SegmentationConfig:
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 300
    max_segment_duration_ms: int = 10000
    speech_pad_ms: int = 0


@dataclass
class ContextConfig:
    padding_ms: int = 0
    cue_tone: bool = False


@dataclass
class RefinementConfig:
    boundary_refinement: bool = False
    aligner: str = "forced"


@dataclass
class OutputConfig:
    format: str = "srt"


@dataclass
class AppConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _find_config_file() -> Path | None:
    candidates = [
        Path(".llm-subtitles.toml"),
        Path.home() / ".config" / "llm-subtitles" / "config.toml",
        Path.home() / ".llm-subtitles.toml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_config(config_path: str | None = None) -> AppConfig:
    config = AppConfig()

    if config_path:
        p = Path(config_path)
        if p.exists():
            _apply_toml(config, p)
        return config

    found = _find_config_file()
    if found:
        _apply_toml(config, found)

    return config


def _apply_toml(config: AppConfig, path: Path) -> None:
    data = toml.load(path)

    if "backend" in data:
        b = data["backend"]
        if "api_base" in b:
            config.backend.api_base = b["api_base"]
        if "model" in b:
            config.backend.model = b["model"]
        if "api_key" in b:
            config.backend.api_key = b["api_key"]

    if "segmentation" in data:
        s = data["segmentation"]
        if "vad_threshold" in s:
            config.segmentation.vad_threshold = s["vad_threshold"]
        if "min_speech_duration_ms" in s:
            config.segmentation.min_speech_duration_ms = s["min_speech_duration_ms"]
        if "min_silence_duration_ms" in s:
            config.segmentation.min_silence_duration_ms = s["min_silence_duration_ms"]
        if "max_segment_duration_ms" in s:
            config.segmentation.max_segment_duration_ms = s["max_segment_duration_ms"]
        if "speech_pad_ms" in s:
            config.segmentation.speech_pad_ms = s["speech_pad_ms"]

    if "context" in data:
        c = data["context"]
        if "padding_ms" in c:
            config.context.padding_ms = c["padding_ms"]
        if "cue_tone" in c:
            config.context.cue_tone = c["cue_tone"]

    if "refinement" in data:
        r = data["refinement"]
        if "boundary_refinement" in r:
            config.refinement.boundary_refinement = r["boundary_refinement"]
        if "aligner" in r:
            config.refinement.aligner = r["aligner"]

    if "output" in data:
        o = data["output"]
        if "format" in o:
            config.output.format = o["format"]
