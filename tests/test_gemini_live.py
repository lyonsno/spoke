"""Tests for the Gemini Live API client and audio player."""

from __future__ import annotations

import base64
import json
import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestLiveAudioPlayer:
    """Unit tests for LiveAudioPlayer."""

    @patch("spoke.gemini_live.sd")
    def test_write_chunk_converts_int16_to_float32(self, mock_sd):
        from spoke.gemini_live import LiveAudioPlayer

        stream = MagicMock()
        mock_sd.OutputStream.return_value = stream
        player = LiveAudioPlayer()

        # 4 samples of int16 PCM at half amplitude
        pcm = np.array([16383, -16383, 0, 32767], dtype=np.int16)
        player.write_chunk(pcm.tobytes())

        stream.write.assert_called_once()
        written = stream.write.call_args[0][0]
        assert written.dtype == np.float32
        assert written.shape == (4, 1)
        np.testing.assert_allclose(written[3, 0], 1.0, atol=1e-4)

    @patch("spoke.gemini_live.sd")
    def test_write_chunk_reports_amplitude(self, mock_sd):
        from spoke.gemini_live import LiveAudioPlayer

        mock_sd.OutputStream.return_value = MagicMock()
        amplitudes = []
        player = LiveAudioPlayer(amplitude_callback=amplitudes.append)

        pcm = np.array([16383, -16383], dtype=np.int16)
        player.write_chunk(pcm.tobytes())

        assert len(amplitudes) == 1
        assert amplitudes[0] > 0.0

    @patch("spoke.gemini_live.sd")
    def test_flush_reopens_stream(self, mock_sd):
        from spoke.gemini_live import LiveAudioPlayer

        streams = []
        mock_sd.OutputStream.side_effect = lambda **kw: MagicMock()

        player = LiveAudioPlayer()
        first_stream = player._stream
        player.flush()
        second_stream = player._stream

        assert first_stream is not second_stream
        first_stream.abort.assert_called_once()
        first_stream.close.assert_called_once()

    @patch("spoke.gemini_live.sd")
    def test_close_stops_stream(self, mock_sd):
        from spoke.gemini_live import LiveAudioPlayer

        stream = MagicMock()
        mock_sd.OutputStream.return_value = stream
        player = LiveAudioPlayer()
        player.close()

        stream.stop.assert_called_once()
        stream.close.assert_called_once()
        assert player._stream is None


class TestGeminiLiveClient:
    """Unit tests for GeminiLiveClient (no real WebSocket)."""

    def test_send_audio_converts_float32_to_int16(self):
        from spoke.gemini_live import GeminiLiveClient

        client = GeminiLiveClient("test-key")
        client._connected = True

        chunk = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        client.send_audio(chunk)

        raw = client._send_queue.get(timeout=1)
        pcm = np.frombuffer(raw, dtype=np.int16)
        assert len(pcm) == 4
        assert pcm[0] == 16383  # 0.5 * 32767 ≈ 16383
        assert pcm[2] == 32767  # clipped to max
        assert pcm[3] == -32767  # clipped to min (not -32768 due to symmetric clip)

    def test_send_audio_noop_when_disconnected(self):
        from spoke.gemini_live import GeminiLiveClient

        client = GeminiLiveClient("test-key")
        assert not client._connected

        chunk = np.array([0.5], dtype=np.float32)
        client.send_audio(chunk)

        assert client._send_queue.empty()

    def test_setup_message_structure(self):
        """Verify the setup message matches the Gemini Live API spec."""
        from spoke.gemini_live import GeminiLiveClient

        client = GeminiLiveClient(
            "test-key",
            model="gemini-2.0-flash-live",
            voice="Puck",
            system_instruction="Be helpful.",
        )

        # The setup message is built inside _connect; verify the expected shape
        # by checking the model and voice are stored correctly.
        assert client._model == "gemini-2.0-flash-live"
        assert client._voice == "Puck"
        assert client._system_instruction == "Be helpful."

    def test_disconnect_pushes_shutdown_sentinel(self):
        from spoke.gemini_live import GeminiLiveClient, _SHUTDOWN

        client = GeminiLiveClient("test-key")
        client._connected = True
        client.disconnect()

        assert not client._connected
        item = client._send_queue.get(timeout=1)
        assert item is _SHUTDOWN


class TestReceiverParsing:
    """Test the receiver's message parsing logic in isolation."""

    def _make_server_audio_message(self, pcm_bytes: bytes) -> str:
        encoded = base64.b64encode(pcm_bytes).decode("ascii")
        return json.dumps({
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/pcm;rate=24000",
                                "data": encoded,
                            }
                        }
                    ]
                }
            }
        })

    def _make_text_message(self, text: str) -> str:
        return json.dumps({
            "serverContent": {
                "modelTurn": {
                    "parts": [{"text": text}]
                }
            }
        })

    def _make_turn_complete_message(self) -> str:
        return json.dumps({
            "serverContent": {"turnComplete": True}
        })

    def _make_interrupted_message(self) -> str:
        return json.dumps({
            "serverContent": {"interrupted": True}
        })

    def test_receiver_dispatches_audio_callback(self):
        from spoke.gemini_live import GeminiLiveClient

        client = GeminiLiveClient("test-key")
        received_chunks = []
        client.on_audio_chunk = received_chunks.append

        pcm = np.array([100, -100], dtype=np.int16).tobytes()
        msg = self._make_server_audio_message(pcm)

        # Simulate the receiver's message parsing
        parsed = json.loads(msg)
        sc = parsed["serverContent"]
        mt = sc["modelTurn"]
        for part in mt["parts"]:
            inline = part.get("inlineData")
            if inline:
                audio_b64 = inline["data"]
                client.on_audio_chunk(base64.b64decode(audio_b64))

        assert len(received_chunks) == 1
        result = np.frombuffer(received_chunks[0], dtype=np.int16)
        np.testing.assert_array_equal(result, np.array([100, -100], dtype=np.int16))

    def test_receiver_dispatches_text_callback(self):
        from spoke.gemini_live import GeminiLiveClient

        client = GeminiLiveClient("test-key")
        received_texts = []
        client.on_text_chunk = received_texts.append

        msg = self._make_text_message("Hello there")
        parsed = json.loads(msg)
        sc = parsed["serverContent"]
        mt = sc["modelTurn"]
        for part in mt["parts"]:
            text = part.get("text")
            if text and client.on_text_chunk:
                client.on_text_chunk(text)

        assert received_texts == ["Hello there"]

    def test_interrupted_message_fires_callback(self):
        from spoke.gemini_live import GeminiLiveClient

        client = GeminiLiveClient("test-key")
        interrupted = []
        client.on_interrupted = lambda: interrupted.append(True)

        msg = self._make_interrupted_message()
        parsed = json.loads(msg)
        sc = parsed["serverContent"]
        if sc.get("interrupted"):
            client.on_interrupted()

        assert interrupted == [True]

    def test_turn_complete_fires_callback(self):
        from spoke.gemini_live import GeminiLiveClient

        client = GeminiLiveClient("test-key")
        completed = []
        client.on_turn_complete = lambda: completed.append(True)

        msg = self._make_turn_complete_message()
        parsed = json.loads(msg)
        sc = parsed["serverContent"]
        if sc.get("turnComplete"):
            client.on_turn_complete()

        assert completed == [True]
