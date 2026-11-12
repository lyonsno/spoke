"""HTTP client for OpenAI-compatible Whisper transcription endpoint.

Sends WAV audio to a server running on the sidecar machine and returns
the transcribed text.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager

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
        isolated: bool = False,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/v1/audio/transcriptions"
        self._model = model
        self._prompt_provider = (
            prompt_provider or TranscriptionPromptProvider.from_environment()
        )
        self._last_prompt_receipt: dict | None = None
        self._worker = None
        self._call_scope = threading.local()
        self.supports_cancellation = isolated
        if isolated:
            from .inference_process import InferenceProcess
            self._worker = InferenceProcess(__name__, type(self).__name__, {
                "base_url": base_url, "model": model, "timeout": timeout,
                "api_key": api_key, "prompt_provider": self._prompt_provider})
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = None if isolated else httpx.Client(timeout=timeout, headers=headers)

    @contextmanager
    def cancellation_scope(self):
        previous = getattr(self._call_scope, "generation", None)
        self._call_scope.generation = self._worker.generation if self._worker is not None else None
        try:
            yield
        finally:
            self._call_scope.generation = previous

    def cancel(self):
        return self._worker.cancel() if self._worker is not None else False

    def transcribe(self, wav_bytes: bytes) -> str:
        """Send WAV audio and return transcribed text.

        Raises ``httpx.HTTPStatusError`` on server errors.
        """
        if not wav_bytes:
            return ""
        if self._worker is not None:
            self._worker_receipt = self._last_prompt_receipt = None
            response = self._worker.call("transcribe", wav_bytes,
                generation=getattr(self._call_scope, "generation", None))
            self._worker_receipt = {**response["worker"], "effective": response["effective"]}
            self._last_prompt_receipt = response["effective"].get("last_prompt_receipt")
            if response.get("error"):
                raise RuntimeError(response["error"])
            return response["value"]

        prompt = self._prompt_provider.resolve()
        data = {"model": self._model}
        if prompt.text:
            data["prompt"] = prompt.text
        self._last_prompt_receipt = prompt.receipt(
            supported=True,
            payload_constructed=bool(prompt.text),
            submission_attempted=False,
            runtime_accepted=None,
        )
        logger.info(
            "Remote transcription prompt: requested=%s supported=true payload=%s "
            "sha256=%s chars=%d sources=%s",
            self._last_prompt_receipt["requested"],
            self._last_prompt_receipt["payload_constructed"],
            prompt.sha256,
            prompt.char_count,
            ",".join(prompt.sources) or "none",
        )
        if prompt.text:
            self._last_prompt_receipt["submission_attempted"] = True
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
        if self._worker is not None:
            self._worker.close()
        elif self._client is not None:
            self._client.close()
