"""Tests for local Qwen3 ASR transcription client."""

from unittest.mock import MagicMock, patch
import io
import wave

import numpy as np
import pytest


def _make_wav_bytes(n_samples=1000):
    """Helper: create valid mono 16-bit WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(n_samples, dtype=np.int16).tobytes())
    return buf.getvalue()


def _mock_result(text=""):
    """Create a mock TranscriptionResult."""
    result = MagicMock()
    result.text = text
    return result


class TestLocalQwenClient:
    """Test the in-process Qwen3 ASR client."""

    def test_empty_bytes_returns_empty_string(self):
        from spoke.transcribe_qwen import LocalQwenClient

        client = LocalQwenClient(model="test-model")
        assert client.transcribe(b"") == ""

    def test_default_model(self):
        from spoke.transcribe_qwen import LocalQwenClient, _DEFAULT_MODEL

        client = LocalQwenClient()
        assert client._model == _DEFAULT_MODEL

    def test_custom_model(self):
        from spoke.transcribe_qwen import LocalQwenClient

        client = LocalQwenClient(model="Qwen/Qwen3-ASR-1.7B")
        assert client._model == "Qwen/Qwen3-ASR-1.7B"

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_transcribe_calls_session(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        mock_session = MagicMock()
        mock_session.transcribe.return_value = _mock_result("hello world")
        mock_module.Session.return_value = mock_session

        client = LocalQwenClient(model="Qwen/Qwen3-ASR-0.6B")
        result = client.transcribe(_make_wav_bytes())

        assert result == "hello world"
        mock_session.transcribe.assert_called_once()
        call_args = mock_session.transcribe.call_args
        assert isinstance(call_args[0][0], np.ndarray)
        assert call_args[0][0].dtype == np.float32
        assert call_args[1]["language"] == "English"

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_transcribe_strips_whitespace(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        mock_session = MagicMock()
        mock_session.transcribe.return_value = _mock_result("\n  hello  \n")
        mock_module.Session.return_value = mock_session

        client = LocalQwenClient()
        assert client.transcribe(_make_wav_bytes()) == "hello"

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_inference_error_propagates(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        mock_session = MagicMock()
        mock_session.transcribe.side_effect = RuntimeError("inference failed")
        mock_module.Session.return_value = mock_session

        client = LocalQwenClient()
        with pytest.raises(RuntimeError):
            client.transcribe(_make_wav_bytes())

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_session_created_lazily(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        client = LocalQwenClient()
        mock_module.Session.assert_not_called()

        mock_session = MagicMock()
        mock_session.transcribe.return_value = _mock_result("test")
        mock_module.Session.return_value = mock_session

        client.transcribe(_make_wav_bytes())
        mock_module.Session.assert_called_once_with(model="Qwen/Qwen3-ASR-0.6B")

    def test_close_clears_session(self):
        from spoke.transcribe_qwen import LocalQwenClient

        client = LocalQwenClient()
        client._session = MagicMock()
        client.close()
        assert client._session is None

    def test_supports_streaming(self):
        from spoke.transcribe_qwen import LocalQwenClient

        client = LocalQwenClient()
        assert client.supports_streaming is True


class TestLocalQwenStreaming:
    """Test the streaming transcription API."""

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_start_stream_initializes_state(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        mock_session = MagicMock()
        mock_state = MagicMock()
        mock_session.init_streaming.return_value = mock_state
        mock_module.Session.return_value = mock_session

        client = LocalQwenClient()
        client.start_stream()

        mock_session.init_streaming.assert_called_once_with(
            language="English", chunk_size_sec=1.5
        )
        assert client._stream_state is mock_state

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_feed_returns_text(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        mock_session = MagicMock()
        mock_state = MagicMock()
        mock_state.text = "hello world"
        mock_session.init_streaming.return_value = mock_state
        mock_session.feed_audio.return_value = mock_state
        mock_module.Session.return_value = mock_session

        client = LocalQwenClient()
        client.start_stream()
        result = client.feed(np.zeros(16000, dtype=np.float32))

        assert result == "hello world"
        mock_session.feed_audio.assert_called_once()

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_feed_empty_frames_returns_current_text(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        mock_session = MagicMock()
        mock_state = MagicMock()
        mock_state.text = "existing text"
        mock_session.init_streaming.return_value = mock_state
        mock_module.Session.return_value = mock_session

        client = LocalQwenClient()
        client.start_stream()
        result = client.feed(np.array([], dtype=np.float32))

        assert result == "existing text"
        mock_session.feed_audio.assert_not_called()

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_feed_without_start_returns_empty(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        client = LocalQwenClient()
        result = client.feed(np.zeros(16000, dtype=np.float32))
        assert result == ""

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_finish_stream_returns_final_text(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        mock_session = MagicMock()
        mock_state = MagicMock()
        mock_state.text = "  hello world  "
        final_state = MagicMock()
        final_state.text = "  final transcription  "
        mock_session.init_streaming.return_value = mock_state
        mock_session.finish_streaming.return_value = final_state
        mock_module.Session.return_value = mock_session

        client = LocalQwenClient()
        client.start_stream()
        result = client.finish_stream()

        assert result == "final transcription"
        mock_session.finish_streaming.assert_called_once_with(mock_state)
        assert client._stream_state is None

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_finish_without_start_returns_empty(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        client = LocalQwenClient()
        assert client.finish_stream() == ""

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_finish_stream_filters_hallucination(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        mock_session = MagicMock()
        mock_state = MagicMock()
        final_state = MagicMock()
        final_state.text = "Thank you."
        mock_session.init_streaming.return_value = mock_state
        mock_session.finish_streaming.return_value = final_state
        mock_module.Session.return_value = mock_session

        client = LocalQwenClient()
        client.start_stream()
        assert client.finish_stream() == ""

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_close_clears_stream_state(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        mock_session = MagicMock()
        mock_session.init_streaming.return_value = MagicMock()
        mock_module.Session.return_value = mock_session

        client = LocalQwenClient()
        client.start_stream()
        client.close()
        assert client._stream_state is None
        assert client._session is None


class TestLocalQwenFiltering:
    """Test that Qwen client applies dedup filtering."""

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_hallucination_returns_empty(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        mock_session = MagicMock()
        mock_session.transcribe.return_value = _mock_result("Thank you.")
        mock_module.Session.return_value = mock_session

        client = LocalQwenClient()
        assert client.transcribe(_make_wav_bytes()) == ""

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_repetition_is_truncated(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        mock_session = MagicMock()
        mock_session.transcribe.return_value = _mock_result("okay. " * 5)
        mock_module.Session.return_value = mock_session

        client = LocalQwenClient()
        result = client.transcribe(_make_wav_bytes())
        assert result.count("okay.") < 5

    @patch("spoke.transcribe_qwen.mlx_qwen3_asr")
    def test_real_text_passes_through(self, mock_module):
        from spoke.transcribe_qwen import LocalQwenClient

        mock_session = MagicMock()
        mock_session.transcribe.return_value = _mock_result("Hello, this is a test.")
        mock_module.Session.return_value = mock_session

        client = LocalQwenClient()
        assert client.transcribe(_make_wav_bytes()) == "Hello, this is a test."
