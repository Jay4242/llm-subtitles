"""LLM client for OpenAI-compatible API with audio support."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from .models import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


@dataclass
class StreamDebugState:
    """Accumulated state from streaming debug output."""
    reasoning: str = ""
    content: str = ""


DebugCallback = Callable[[StreamDebugState], None]


class LLMClient:
    """Client for OpenAI-compatible chat completions API with audio input."""

    def __init__(
        self,
        api_base: str = "http://localhost:9090/v1",
        model: str = "",
        api_key: str = "not-needed",
        timeout: float = 14400.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create a shared async client for connection reuse."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout, http2=True)
        return self._client

    async def close(self) -> None:
        """Close the shared HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def _build_system_message(
        self,
        speakers: bool = False,
        language: str | None = None,
    ) -> str:
        """Build the system message with role, rules, and format constraints."""
        lines = [
            "You are a precise audio transcription engine specialized in"
            " producing high-quality subtitles.",
            "Transcribe verbatim from the audio. Do not summarize, rephrase,"
            " or infer missing words.",
            "",
            "## Subtitle Formatting Rules",
            "- Split speech into natural subtitle segments.",
            "- Keep each segment under 42 characters when possible.",
            "- Split at clause or sentence boundaries; never split mid-phrase.",
            "- Each segment should express one complete thought.",
            "- Use sentence case. Capitalize proper nouns.",
            "- Use ellipsis (…) for trailing thoughts, em-dash (—) for"
            " interruptions.",
            "- Skip music, sound effects, and extended silence. Do NOT create"
            " segments for non-speech audio.",
            "",
            "## Output Format",
            "- Respond with ONLY a valid JSON object. No markdown, no"
            " explanation, no surrounding text.",
            "- The JSON must conform to the schema provided in the user message.",
            "- Timestamps are handled by a separate forced-alignment pass."
            " Provide only text in each segment.",
            "",
            "## Confidence",
            "- Set confidence between 0.0 and 1.0 based on audio clarity and"
            " your transcription certainty.",
            "- Use lower values for noisy audio, overlapping speech, or"
            " unclear words.",
        ]

        if speakers:
            lines.extend([
                "",
                "## Speaker Diarization",
                "- Assign a single uppercase letter (A, B, C, …) to each"
                " distinct speaker.",
                "- Use the same letter consistently when a speaker returns.",
                "- Include the \"speaker\" field in every segment.",
            ])

        if language and language != "auto":
            lines.extend([
                "",
                "## Language",
                f"- The expected language is {language}.",
                f"- Use appropriate punctuation, capitalization, and"
                f" orthography for {language}.",
            ])

        return "\n".join(lines)

    def _build_transcription_user_message(
        self,
        target_start_ms: int,
        target_end_ms: int,
        speakers: bool = False,
        language: str | None = None,
    ) -> str:
        """Build the user message for a transcription chunk."""
        schema = {
            "text": "Full transcript of the target segment.",
            "segments": [
                {
                    "text": "<subtitle text>",
                }
            ],
            "language": "<language_code>",
            "confidence": "<0.0–1.0>",
        }

        if speakers:
            schema["segments"][0]["speaker"] = "<A|B|C|…>"

        lines = [
            "You will receive an audio clip containing a TARGET SEGMENT"
            " preceded and followed by CONTEXT audio.",
            "",
            "Context is provided to help you understand speaker transitions,"
            " ongoing sentences, and the acoustic environment.",
            "Transcribe all speech you hear in the clip. Skip non-speech audio.",
            "Timestamps are handled separately — provide only the text.",
            "",
            "## Required JSON Schema",
            "```json",
            json.dumps(schema, indent=2),
            "```",
            "",
            "If there is no speech in the target segment, return an empty"
            " segments list.",
        ]

        return "\n".join(lines)

    def _build_transcription_prompt(
        self,
        target_start_ms: int,
        target_end_ms: int,
        speakers: bool = False,
        language: str | None = None,
    ) -> str:
        """Build the transcription prompt for a chunk.

        Kept for backward compatibility; returns the user message portion.
        Use _build_system_message() + _build_transcription_user_message()
        for the new two-message format.
        """
        return self._build_transcription_user_message(
            target_start_ms, target_end_ms, speakers, language
        )

    def _build_boundary_refinement_system_message(
        self,
        language: str | None = None,
    ) -> str:
        """Build the system message for boundary refinement."""
        lines = [
            "You are a precise audio transcription engine specialized in"
            " producing high-quality subtitles.",
            "",
            "## Task",
            "Two adjacent audio segments were transcribed separately. You will"
            " hear the boundary region between them.",
            "Your job is to:",
            "- Ensure the text flows naturally across the seam.",
            "- Fix any cut-off words or misheard phrases at the boundary.",
            "- Return corrected text for the boundary region only.",
            "",
            "## Output Format",
            "- Respond with ONLY a valid JSON object. No markdown, no"
            " explanation, no surrounding text.",
            "- Use the schema provided in the user message.",
            "",
            "## Confidence",
            "- Set confidence between 0.0 and 1.0 based on audio clarity and"
            " your transcription certainty.",
        ]

        if language and language != "auto":
            lines.extend([
                "",
                "## Language",
                f"- The expected language is {language}.",
            ])

        return "\n".join(lines)

    def _build_boundary_refinement_user_message(
        self,
        chunk_a_last_words: str,
        chunk_b_first_words: str,
    ) -> str:
        """Build the user message for boundary refinement."""
        schema = {
            "text": "Corrected combined text across the boundary.",
            "segments": [
                {
                    "text": "<corrected subtitle text>",
                }
            ],
            "language": "<language_code>",
            "confidence": "<0.0–1.0>",
        }

        return (
            f"The preceding segment ends with: \"{chunk_a_last_words}\"\n"
            f"The following segment begins with: \"{chunk_b_first_words}\"\n"
            "\n"
            "Listen to the boundary audio and correct any discrepancies."
            " Return corrected text for the boundary region only.\n"
            "\n"
            "## Required JSON Schema\n"
            "```json\n"
            f"{json.dumps(schema, indent=2)}\n"
            "```"
        )

    def _build_boundary_refinement_prompt(
        self,
        chunk_a_last_words: str,
        chunk_b_first_words: str,
    ) -> str:
        """Build prompt for boundary refinement between two chunks.

        Kept for backward compatibility; returns the user message portion.
        Use _build_boundary_refinement_system_message() +
        _build_boundary_refinement_user_message() for the new two-message
        format.
        """
        return self._build_boundary_refinement_user_message(
            chunk_a_last_words, chunk_b_first_words
        )

    async def transcribe(
        self,
        audio_bytes: bytes,
        target_start_ms: int,
        target_end_ms: int,
        speakers: bool = False,
        language: str | None = None,
        debug_callback: DebugCallback | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio using the LLM.

        Args:
            audio_bytes: WAV audio as bytes.
            target_start_ms: Start of target segment within the clip (ms).
            target_end_ms: End of target segment within the clip (ms).
            speakers: Whether to request speaker diarization.
            language: Expected language code or None for auto.
            debug_callback: If set, streams reasoning/content via callback.

        Returns:
            TranscriptionResult with text and segments.
        """
        system_message = self._build_system_message(speakers, language)
        user_message = self._build_transcription_user_message(
            target_start_ms, target_end_ms, speakers, language
        )
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "The following audio clip contains speech to transcribe:",
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "wav",
                            },
                        },
                        {"type": "text", "text": user_message},
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
        }

        url = f"{self.api_base}/chat/completions"

        logger.info(
            "Transcribe request: model=%s, target=%d-%dms, speakers=%s, "
            "language=%s",
            self.model, target_start_ms, target_end_ms, speakers, language,
        )
        logger.debug("Transcribe system message:\n%s", system_message)
        logger.debug("Transcribe user message:\n%s", user_message)

        if debug_callback is not None:
            content, reasoning = await self._stream_request(
                url, payload, debug_callback,
            )
        else:
            content, reasoning = await self._non_stream_request(url, payload)

        logger.info(
            "Transcribe response: content_len=%d, reasoning_len=%d",
            len(content), len(reasoning),
        )
        logger.info("Transcribe full reasoning:\n%s", reasoning)
        logger.info("Transcribe full content:\n%s", content)

        result = self._parse_response(content)
        logger.debug(
            "Transcribe parsed: text=%r, segments=%d, confidence=%.2f",
            result.text[:100], len(result.segments), result.confidence,
        )
        return result

    async def refine_boundary(
        self,
        audio_bytes: bytes,
        chunk_a_last_words: str,
        chunk_b_first_words: str,
        language: str | None = None,
        debug_callback: DebugCallback | None = None,
    ) -> TranscriptionResult:
        """Refine transcription at a boundary between two chunks.

        Args:
            audio_bytes: Boundary region audio as bytes.
            chunk_a_last_words: Last words from the preceding chunk.
            chunk_b_first_words: First words from the following chunk.
            language: Expected language code or None for auto.
            debug_callback: If set, streams reasoning/content via callback.

        Returns:
            TranscriptionResult with corrected boundary text.
        """
        system_message = self._build_boundary_refinement_system_message(language)
        user_message = self._build_boundary_refinement_user_message(
            chunk_a_last_words, chunk_b_first_words
        )
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "The following audio clip contains speech to transcribe:",
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "wav",
                            },
                        },
                        {"type": "text", "text": user_message},
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
        }

        url = f"{self.api_base}/chat/completions"

        logger.info(
            "Refine boundary request: model=%s, language=%s",
            self.model, language,
        )
        logger.debug("Refine boundary system message:\n%s", system_message)
        logger.debug("Refine boundary user message:\n%s", user_message)

        if debug_callback is not None:
            content, reasoning = await self._stream_request(
                url, payload, debug_callback,
            )
        else:
            content, reasoning = await self._non_stream_request(url, payload)

        logger.info(
            "Refine boundary response: content_len=%d, reasoning_len=%d",
            len(content), len(reasoning),
        )
        logger.info("Refine boundary full reasoning:\n%s", reasoning)
        logger.info("Refine boundary full content:\n%s", content)

        result = self._parse_response(content)
        logger.debug(
            "Refine boundary parsed: text=%r, segments=%d",
            result.text[:100], len(result.segments),
        )
        return result

    async def _non_stream_request(
        self, url: str, payload: dict,
    ) -> tuple[str, str]:
        """Send a non-streaming request and extract content and reasoning."""
        client = await self._get_client()
        response = await client.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()

        message = data["choices"][0]["message"]
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        return content, reasoning

    async def _stream_request(
        self,
        url: str,
        payload: dict,
        debug_callback: DebugCallback,
    ) -> tuple[str, str]:
        """Send a streaming request, accumulating reasoning and content."""
        payload = {**payload, "stream": True}
        state = StreamDebugState()

        client = await self._get_client()
        async with client.stream(
            "POST", url, headers=self.headers, json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                reasoning = delta.get("reasoning_content")
                content = delta.get("content")
                if reasoning:
                    state.reasoning += reasoning
                    debug_callback(state)
                if content:
                    state.content += content
                    debug_callback(state)

        if not state.content:
            logger.warning(
                "Streaming response had no content; "
                "falling back to reasoning"
            )
            return state.reasoning, state.reasoning

        logger.debug(
            "Stream complete: reasoning_len=%d, content_len=%d",
            len(state.reasoning), len(state.content),
        )
        return state.content, state.reasoning

    def _parse_response(self, content: str) -> TranscriptionResult:
        """Parse the LLM response into a TranscriptionResult."""
        content = content.strip()

        # Extract JSON object by finding balanced braces
        json_str = self._extract_json_object(content)
        if json_str is None:
            logger.warning(
                "Failed to parse LLM response as JSON: %s", content[:200],
            )
            return TranscriptionResult(text=content.strip(), segments=[])

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse LLM response as JSON: %s", content[:200],
            )
            return TranscriptionResult(text=content.strip(), segments=[])

        text = data.get("text", "")
        language = data.get("language", "en")
        confidence = data.get("confidence", 0.5)

        segments = []
        for seg_data in data.get("segments", []):
            segments.append(
                TranscriptionSegment(
                    start_ms=seg_data.get("start_ms", 0),
                    end_ms=seg_data.get("end_ms", 0),
                    text=seg_data.get("text", ""),
                    speaker=seg_data.get("speaker"),
                )
            )

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=language,
            confidence=confidence,
        )

    def _extract_json_object(self, content: str) -> str | None:
        """Extract a balanced JSON object from text."""
        start = content.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(content)):
            ch = content[i]

            if escape_next:
                escape_next = False
                continue

            if ch == "\\":
                if in_string:
                    escape_next = True
                continue

            if ch == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return content[start : i + 1]

        return None
