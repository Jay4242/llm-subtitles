# llm-subtitles

A command-line tool that transcribes audio/video files into timestamped subtitles (SRT/VTT/JSON/ASS) using a local multimodal LLM served via an OpenAI-compatible API.

By leveraging an LLM's broad language understanding, it produces subtitles with superior punctuation, casing, disambiguation of homophones, domain-specific terminology recognition, and natural phrasing. All inference runs locally; no audio ever leaves the machine.

## Core Architecture

```
Input File → Audio Extraction → VAD Segmentation → Context Assembly → LLM Transcription → Boundary Refinement → Forced Alignment → Post-processing → Output
```

1. **Audio Extraction** — ffmpeg converts any media file to 16kHz mono 16-bit PCM WAV
2. **VAD Segmentation** — Silero-VAD splits audio into utterance segments at natural silence boundaries
3. **Context Assembly** — Each segment is optionally padded with surrounding audio (0ms by default; configurable via `--context-ms`) with optional cue tones to mark boundaries
4. **LLM Transcription** — Chunks are sent sequentially to a local multimodal LLM (llama.cpp, vLLM, Ollama, etc.)
5. **Boundary Refinement** — Adjacent chunk boundaries are re-examined to resolve split words and punctuation (off by default; enable with `--boundary-refinement`)
6. **Forced Alignment** — wav2vec2 CTC alignment refines timestamps to word-level precision
7. **Post-processing** — Segments are sorted, durations are clamped to 1–7 seconds, and overlapping segments are resolved
8. **Output** — Formatted as SRT, VTT, JSON, or ASS

## Requirements

- **Python 3.10+**
- **ffmpeg** — installed and on PATH
- **llama.cpp** (or any OpenAI-compatible LLM server with audio support) running locally

## Installation

```bash
pip install -e .
```

For forced alignment support (requires PyTorch):

```bash
pip install -e ".[forced-alignment]"
```

## Usage

### Basic Transcription

```bash
llm-subtitles --model your-model-name transcribe input.mp4
```

This extracts audio, transcribes it, and writes `input.srt` in the current directory.

### Full Options

```bash
llm-subtitles \
  --api-base http://localhost:9090/v1 \
  --model your-model-name \
  --concurrency 2 \
  --language en \
  --speakers \
  --context-ms 10000 \
  --format srt \
  --output subtitles.srt \
  transcribe input.mp4
```

### Burn Subtitles into Video

```bash
llm-subtitles --model your-model-name burn input.mp4
```

Hard-burns subtitles into the video stream (re-encodes), outputting `input_burned.mp4`.

### Burn Subtitles into Video (MKV)

```bash
llm-subtitles --model your-model-name mux input.mp4
```

Same as `burn` but outputs `input_subtitled.mkv`.

### Debug: Extract Audio Only

```bash
llm-subtitles transcode input.mp4
```
Use `--audio-track` to select a specific audio stream from multi-track files.

### Debug: View VAD Segments

```bash
llm-subtitles segment input.mp4
```

## Configuration

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--api-base` | `http://localhost:9090/v1` | OpenAI-compatible API base URL |
| `--model` | *(required)* | Model name as registered on the server |
| `--api-key` | `not-needed` | API key (many local servers ignore this) |
| `--concurrency` | `1` | Max parallel LLM requests (currently sequential; flag is accepted but not yet wired) |
| `--language` | `auto` | Expected language (ISO 639-1) or `auto` |
| `--output` | *(derived from input)* | Output file path |
| `--format` | `srt` | Output format (`srt`, `vtt`, `json`, `ass`) |
| `--speakers` | `false` | Enable speaker diarization |
| `--context-ms` | `0` | Context window padding in ms |
| `--max-segment-ms` | `10000` | Max subtitle segment duration |
| `--vad-threshold` | `0.5` | Silero-VAD threshold |
| `--boundary-refinement` | `false` | Enable cross-chunk boundary refinement |
| `--aligner` | `forced` | Timestamp refinement (`forced`, `llm-only`) |
| `--temp-dir` | *(system temp)* | Working directory for intermediate files |
| `--keep-temp` | `false` | Keep intermediate files for debugging |
| `--verbose` | `false` | Verbose logging |
| `--debug` | `false` | Stream raw LLM output during transcription |
| `--log-file` | *(auto when --debug)* | Debug log file path |

### Config File

Place `.llm-subtitles.toml` in your project directory, `~/.config/llm-subtitles/config.toml`, or `~/.llm-subtitles.toml`. Config files are merged in that order (later files override earlier ones).

```toml
[backend]
api_base = "http://localhost:9090/v1"
model = "your-audio-model-name"
api_key = "not-needed"

[segmentation]
vad_threshold = 0.5
min_speech_duration_ms = 250
min_silence_duration_ms = 300
max_segment_duration_ms = 10000

[context]
padding_ms = 0
cue_tone = false

[refinement]
boundary_refinement = false
aligner = "forced"

[output]
format = "srt"
```

## Supported LLM Backends

Any OpenAI-compatible endpoint that accepts audio input parts:

- **llama.cpp** `llama-server` (primary target)
- **vLLM** with multimodal model support
- **Ollama** with audio-capable models

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Future Extensions

- Two-pass rough draft → refined pipeline
- Confidence-based iterative refinement (confidence scores are already collected from the LLM)
- Configurable duration clamping (currently hardcoded to 1–7 seconds)
- Acoustic event tagging (`[applause]`, `[laughter]`, etc.)
- Translation mode
- Streaming / real-time mode
- Concurrency (parallel LLM requests)
- Docker image with all dependencies

## License

MIT
