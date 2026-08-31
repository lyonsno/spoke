"""HTTP client for OpenAI-compatible Whisper transcription endpoint.

Sends WAV audio to a server running on the sidecar machine and returns
the transcribed text.
"""

from __future__ import annotations

import logging

import httpx

from .dedup import truncate_repetition, is_hallucination, repair_ontology_terms
from .transcription_prompt import TranscriptionPromptProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


class TranscriptionClient:
    """Client for /v1/audio/transcriptions (OpenAI-compatible).

    Parameters
    ----------
    base_url : str
        Sidecar server base URL, e.g. ``http://192.168.68.125:8000``.
    model : str
        Whisper model identifier.
    timeout : float
        Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        model: str = _DEFAULT_MODEL,
        timeout: float = 60.0,
        api_key: str = "",
        prompt_provider: TranscriptionPromptProvider | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/v1/audio/transcriptions"
        self._model = model
        self._prompt_provider = (
            prompt_provider or TranscriptionPromptProvider.from_environment()
        )
        self._last_prompt_receipt: dict | None = None
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def transcribe(self, wav_bytes: bytes) -> str:
        """Send WAV audio and return transcribed text.

        Raises ``httpx.HTTPStatusError`` on server errors.
        """
        if not wav_bytes:
            return ""

        prompt = self._prompt_provider.resolve()
        data = {"model": self._model}
        if prompt.text:
            data["prompt"] = prompt.text
        self._last_prompt_receipt = prompt.receipt(
            supported=True,
            effective=bool(prompt.text),
        )
        logger.info(
            "Remote transcription prompt: requested=%s supported=true effective=%s "
            "sha256=%s chars=%d sources=%s",
            self._last_prompt_receipt["requested"],
            self._last_prompt_receipt["effective"],
            prompt.sha256,
            prompt.char_count,
            ",".join(prompt.sources) or "none",
        )
        resp = self._client.post(
            self._url,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data=data,
        )
        resp.raise_for_status()

        body = resp.json()
        text = body.get("text", "").strip()
        text = truncate_repetition(text)
        text = repair_ontology_terms(text)
        if is_hallucination(text):
            logger.info("Discarding hallucination: %r", text)
            return ""
        logger.info("Transcription: %r (%d bytes audio)", text, len(wav_bytes))
        return text

    def prepare(self) -> None:
        """Remote transcription clients are ready once the HTTP client exists."""
        return None

    def close(self) -> None:
        self._client.close()
