"""Main pipeline orchestration: audio extraction -> VAD -> LLM transcription -> output."""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from pathlib import Path

from . import audio as audio_mod
from .config import AppConfig
from .llm_client import LLMClient, StreamDebugState
from .models import (
    Chunk,
    SubtitleFile,
    TranscriptionResult,
    VADSegment,
)
from .output import write_subtitles
from .vad import VADConfig, run_vad

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when the pipeline fails."""


class Pipeline:
    """Full transcription pipeline."""

    def __init__(
        self,
        config: AppConfig,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.config = config
        self.verbose = verbose
        self.debug = debug
        self.temp_dir: str | None = None
        self.wav_path: str | None = None
        self._llm_lock = asyncio.Lock()
        self.client = LLMClient(
            api_base=config.backend.api_base,
            model=config.backend.model,
        )

    def _debug_callback(self, chunk_idx: int) -> object:
        """Create a debug callback for a specific chunk index."""
        last_len = {"reasoning": 0, "content": 0}
        printed = {"reasoning": False, "content": False}

        def callback(state: StreamDebugState) -> None:
            r_diff = state.reasoning[last_len["reasoning"]:]
            c_diff = state.content[last_len["content"]:]
            last_len["reasoning"] = len(state.reasoning)
            last_len["content"] = len(state.content)
            if r_diff:
                if not printed["reasoning"]:
                    printed["reasoning"] = True
                    sys.stdout.write(f"[reasoning] {r_diff}")
                else:
                    sys.stdout.write(r_diff)
                sys.stdout.flush()
            if c_diff:
                if not printed["content"]:
                    printed["content"] = True
                    sys.stdout.write(f"[content] {c_diff}")
                else:
                    sys.stdout.write(c_diff)
                sys.stdout.flush()

        return callback

    def _setup_temp(self, keep_temp: bool, temp_dir: str | None = None) -> str:
        if temp_dir:
            self.temp_dir = temp_dir
        else:
            self.temp_dir = Path(__import__("tempfile").gettempdir()) / "llm-subtitles-work"
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        self.keep_temp = keep_temp
        return self.temp_dir

    def _cleanup(self) -> None:
        if self.temp_dir and not self.keep_temp:
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def run(
        self,
        input_path: str,
        output_path: str | None = None,
        output_format: str | None = None,
        concurrency: int = 1,
        speakers: bool = False,
        language: str | None = None,
        context_ms: int | None = None,
        keep_temp: bool = False,
        temp_dir: str | None = None,
        debug: bool = False,
    ) -> SubtitleFile:
        """Run the full transcription pipeline.

        Args:
            input_path: Path to input media file.
            output_path: Optional output file path.
            output_format: Output format (srt, vtt, json, ass).
            concurrency: Max parallel LLM requests.
            speakers: Enable speaker diarization.
            language: Expected language code.
            context_ms: Context window padding in ms.
            keep_temp: Keep intermediate files.
            temp_dir: Working directory for intermediate files.
            debug: Stream LLM reasoning and content during transcription.

        Returns:
            SubtitleFile with all transcribed segments.
        """
        if not self.config.backend.model:
            raise PipelineError("No model configured. Set --model or add it to config file.")

        self._setup_temp(keep_temp, temp_dir)
        fmt = output_format or self.config.output.format
        ctx_ms = context_ms if context_ms is not None else self.config.context.padding_ms

        try:
            logger.info("Stage 0: Extracting audio...")
            self.wav_path = audio_mod.extract_audio(input_path, verbose=self.verbose)
            logger.info(f"Audio extracted to: {self.wav_path}")

            logger.info("Stage 1: Running VAD segmentation...")
            audio_data, sr = audio_mod.load_audio(self.wav_path)
            vad_config = VADConfig(
                threshold=self.config.segmentation.vad_threshold,
                min_speech_duration_ms=self.config.segmentation.min_speech_duration_ms,
                min_silence_duration_ms=self.config.segmentation.min_silence_duration_ms,
                max_segment_duration_ms=self.config.segmentation.max_segment_duration_ms,
                speech_pad_ms=self.config.segmentation.speech_pad_ms,
            )
            vad_segments = run_vad(audio_data, sr, vad_config)

            if not vad_segments:
                logger.warning("No speech detected. Creating empty subtitle file.")
                return SubtitleFile()

            logger.info(f"Found {len(vad_segments)} speech segments")

            logger.info("Stage 2: Assembling chunks with context windows...")
            chunks = self._assemble_chunks(self.wav_path, vad_segments, ctx_ms)
            logger.info(f"Assembled {len(chunks)} chunks")

            logger.info("Stage 3: Transcribing chunks with LLM...")
            all_results = asyncio.run(
                self._transcribe_chunks(chunks, speakers, language, debug)
            )
            logger.info(f"Transcribed {len(all_results)} chunks")

            logger.info("Stage 4: Boundary refinement...")
            if self.config.refinement.boundary_refinement and len(chunks) > 1:
                all_results = asyncio.run(
                    self._refine_boundaries(
                        self.wav_path, vad_segments, chunks, all_results, language,
                        debug,
                    )
                )
                logger.info("Boundary refinement complete")

            logger.info("Stage 5: Timestamp refinement (forced alignment)...")
            if self.config.refinement.aligner == "forced":
                all_results = self._forced_alignment(self.wav_path, chunks, all_results)
                logger.info("Forced alignment complete")

            logger.info("Stage 6: Assembling output...")
            subtitle_file = self._assemble_subtitles(
                chunks, all_results, language or "en"
            )
            subtitle_file.sort_segments()
            subtitle_file.normalize_durations()
            subtitle_file.merge_overlaps()

            if output_path:
                write_subtitles(subtitle_file, output_path, fmt)
                logger.info(f"Subtitles written to: {output_path}")
            else:
                base = Path(input_path).stem
                output_path = f"{base}.{fmt}"
                write_subtitles(subtitle_file, output_path, fmt)
                logger.info(f"Subtitles written to: {output_path}")

            return subtitle_file

        finally:
            self._cleanup()
            asyncio.run(self.client.close())

    def _assemble_chunks(
        self, wav_path: str, vad_segments: list[VADSegment], context_ms: int
    ) -> list[Chunk]:
        from .segments import assemble_chunks as _assemble

        return _assemble(
            wav_path,
            vad_segments,
            context_ms=context_ms,
            cue_tone=self.config.context.cue_tone,
            save_dir="VAD",
        )

    async def _transcribe_chunks(
        self,
        chunks: list[Chunk],
        speakers: bool,
        language: str | None,
        debug: bool = False,
    ) -> list[TranscriptionResult]:
        results = [None] * len(chunks)

        async def _transcribe_one(chunk: Chunk, idx: int) -> None:
            async with self._llm_lock:
                try:
                    debug_cb = self._debug_callback(idx) if debug else None
                    result = await self.client.transcribe(
                        audio_bytes=chunk.audio_bytes,
                        target_start_ms=chunk.target_start_in_chunk_ms,
                        target_end_ms=chunk.target_end_in_chunk_ms,
                        speakers=speakers,
                        language=language,
                        debug_callback=debug_cb,
                    )
                    if debug:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    self._convert_relative_to_absolute(result, chunk)
                    results[idx] = result
                    logger.info(f"Chunk {idx} transcribed: {result.text[:80]}...")
                except Exception as e:
                    logger.error(f"Failed to transcribe chunk {idx}: {e}")
                    results[idx] = TranscriptionResult(
                        text="", segments=[], confidence=0.0
                    )

        # Process sequentially to avoid overwhelming the backend
        for i, chunk in enumerate(chunks):
            await _transcribe_one(chunk, i)
        return [r for r in results if r is not None]

    def _convert_relative_to_absolute(
        self, result: TranscriptionResult, chunk: Chunk
    ) -> None:
        offset_ms = chunk.absolute_start_ms - chunk.target_start_in_chunk_ms
        for seg in result.segments:
            seg.start_ms += offset_ms
            seg.end_ms += offset_ms

    async def _refine_boundaries(
        self,
        wav_path: str,
        vad_segments: list[VADSegment],
        chunks: list[Chunk],
        results: list[TranscriptionResult],
        language: str | None,
        debug: bool = False,
    ) -> list[TranscriptionResult]:
        from .segments import assemble_boundary_chunk

        refinements = []

        async def _refine_one(i: int) -> None:
            async with self._llm_lock:
                try:
                    seg_a = vad_segments[i]
                    seg_b = vad_segments[i + 1]
                    result_a = results[i]
                    result_b = results[i + 1]

                    last_words = result_a.segments[-1].text if result_a.segments else ""
                    first_words = result_b.segments[0].text if result_b.segments else ""

                    if not last_words or not first_words:
                        return

                    boundary_chunk = assemble_boundary_chunk(
                        wav_path, seg_a, seg_b,
                        context_ms=5000,
                        cue_tone=self.config.context.cue_tone,
                    )

                    debug_cb = self._debug_callback(i) if debug else None
                    refinement = await self.client.refine_boundary(
                        audio_bytes=boundary_chunk.audio_bytes,
                        chunk_a_last_words=last_words,
                        chunk_b_first_words=first_words,
                        language=language,
                        debug_callback=debug_cb,
                    )
                    if debug:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    self._convert_relative_to_absolute(refinement, boundary_chunk)
                    refinements.append((i, refinement))
                    logger.info(f"Boundary {i}-{i+1} refined")
                except Exception as e:
                    logger.warning(f"Boundary refinement failed for pair {i}-{i+1}: {e}")

        # Process sequentially to avoid overwhelming the backend
        for i in range(len(vad_segments) - 1):
            await _refine_one(i)

        for i, refinement in refinements:
            if not refinement.segments:
                continue
            if results[i].segments:
                results[i].segments.pop()
            if i + 1 < len(results) and results[i + 1].segments:
                results[i + 1].segments.pop(0)
            results[i].segments.extend(refinement.segments)

        return results

    def _forced_alignment(
        self,
        wav_path: str,
        chunks: list[Chunk],
        results: list[TranscriptionResult],
    ) -> list[TranscriptionResult]:
        try:
            from .alignment import run_forced_alignment_full as _align_full
            audio_data, sr = audio_mod.load_audio(wav_path)
            _align_full(results, audio_data, sr)
        except ImportError:
            logger.warning(
                "Forced alignment dependencies not installed. "
                "Install with: pip install llm-subtitles[forced-alignment]"
            )
        except Exception as e:
            logger.warning(f"Forced alignment failed: {e}")

        return results

    def _assemble_subtitles(
        self,
        chunks: list[Chunk],
        results: list[TranscriptionResult],
        language: str,
    ) -> SubtitleFile:
        subtitle_file = SubtitleFile(language=language)
        for result in results:
            for seg in result.segments:
                if seg.text.strip():
                    subtitle_file.add_segment(seg)
        return subtitle_file


def transcribe(
    input_path: str,
    config: AppConfig,
    output_path: str | None = None,
    output_format: str | None = None,
    concurrency: int = 1,
    speakers: bool = False,
    language: str | None = None,
    context_ms: int | None = None,
    keep_temp: bool = False,
    temp_dir: str | None = None,
    verbose: bool = False,
    debug: bool = False,
) -> SubtitleFile:
    """Convenience function to run the transcription pipeline.

    Args:
        input_path: Path to input media file.
        config: Application configuration.
        output_path: Optional output file path.
        output_format: Output format.
        concurrency: Max parallel LLM requests.
        speakers: Enable speaker diarization.
        language: Expected language code.
        context_ms: Context window padding in ms.
        keep_temp: Keep intermediate files.
        temp_dir: Working directory.
        verbose: Verbose logging.
        debug: Stream LLM reasoning and content during transcription.

    Returns:
        SubtitleFile with transcribed segments.
    """
    pipeline = Pipeline(config, verbose=verbose, debug=debug)
    return pipeline.run(
        input_path=input_path,
        output_path=output_path,
        output_format=output_format,
        concurrency=concurrency,
        speakers=speakers,
        language=language,
        context_ms=context_ms,
        keep_temp=keep_temp,
        temp_dir=temp_dir,
        debug=debug,
    )
