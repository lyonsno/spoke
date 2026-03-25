"""HTTP client for OpenAI-compatible Whisper transcription endpoint.

Sends WAV audio to a server running on the sidecar machine and returns
the transcribed text.
"""

from __future__ import annotations

import logging

import httpx

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
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/v1/audio/transcriptions"
        self._model = model
        self._client = httpx.Client(timeout=timeout)

    def transcribe(self, wav_bytes: bytes) -> str:
        """Send WAV audio and return transcribed text.

        Raises ``httpx.HTTPStatusError`` on server errors.
        """
        if not wav_bytes:
            return ""

        resp = self._client.post(
            self._url,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model": self._model},
        )
        resp.raise_for_status()

        body = resp.json()
        text = body.get("text", "").strip()
        logger.info("Transcription: %r (%d bytes audio)", text, len(wav_bytes))
        return text

    def close(self) -> None:
        self._client.close()
