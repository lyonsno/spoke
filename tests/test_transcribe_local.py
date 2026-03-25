"""Tests for local MLX Whisper transcription client."""

from unittest.mock import MagicMock, patch
import os
import tempfile

import pytest


class TestLocalTranscriptionClient:
    """Test the in-process MLX Whisper client."""

    def test_empty_bytes_returns_empty_string(self):
        """Empty WAV input should short-circuit without calling mlx_whisper."""
        from donttype.transcribe_local import LocalTranscriptionClient

        client = LocalTranscriptionClient(model="test-model")
        result = client.transcribe(b"")
        assert result == ""

    def test_default_model(self):
        from donttype.transcribe_local import LocalTranscriptionClient, _DEFAULT_MODEL

        client = LocalTranscriptionClient()
        assert client._model == _DEFAULT_MODEL

    def test_custom_model(self):
        from donttype.transcribe_local import LocalTranscriptionClient

        client = LocalTranscriptionClient(model="custom/model")
        assert client._model == "custom/model"

    @patch("donttype.transcribe_local.mlx_whisper", create=True)
    def test_transcribe_calls_mlx_whisper(self, mock_mlx_whisper):
        """transcribe() should call mlx_whisper.transcribe with correct args."""
        from donttype.transcribe_local import LocalTranscriptionClient

        mock_mlx_whisper.transcribe.return_value = {"text": "  hello world  "}
        client = LocalTranscriptionClient(model="test/model")

        # Provide valid WAV bytes
        import io, wave, numpy as np
        sr = 16000
        samples = np.zeros(sr, dtype=np.float32)
        pcm = (samples * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

        result = client.transcribe(buf.getvalue())
        assert result == "hello world"
        mock_mlx_whisper.transcribe.assert_called_once()
        call_kwargs = mock_mlx_whisper.transcribe.call_args
        assert call_kwargs[1]["path_or_hf_repo"] == "test/model"
        assert call_kwargs[1]["language"] == "en"

    @patch("donttype.transcribe_local.mlx_whisper", create=True)
    def test_transcribe_strips_whitespace(self, mock_mlx_whisper):
        """Transcription result should be stripped."""
        from donttype.transcribe_local import LocalTranscriptionClient

        mock_mlx_whisper.transcribe.return_value = {"text": "\n  hello  \n"}
        client = LocalTranscriptionClient()

        import io, wave, numpy as np
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(1000, dtype=np.int16).tobytes())

        assert client.transcribe(buf.getvalue()) == "hello"

    @patch("donttype.transcribe_local.mlx_whisper", create=True)
    def test_transcribe_missing_text_key(self, mock_mlx_whisper):
        """Missing 'text' key should return empty string."""
        from donttype.transcribe_local import LocalTranscriptionClient

        mock_mlx_whisper.transcribe.return_value = {"segments": []}
        client = LocalTranscriptionClient()

        import io, wave, numpy as np
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(1000, dtype=np.int16).tobytes())

        assert client.transcribe(buf.getvalue()) == ""

    @patch("donttype.transcribe_local.mlx_whisper", create=True)
    def test_passes_numpy_array_not_file_path(self, mock_mlx_whisper):
        """transcribe() should pass a numpy array, not a file path."""
        from donttype.transcribe_local import LocalTranscriptionClient

        mock_mlx_whisper.transcribe.return_value = {"text": "test"}
        client = LocalTranscriptionClient()

        import io, wave, numpy as np
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(1000, dtype=np.int16).tobytes())

        client.transcribe(buf.getvalue())
        call_args = mock_mlx_whisper.transcribe.call_args
        # First positional arg should be a numpy array, not a string path
        assert isinstance(call_args[0][0], np.ndarray)
        assert call_args[0][0].dtype == np.float32

    @patch("donttype.transcribe_local.mlx_whisper", create=True)
    def test_inference_error_propagates(self, mock_mlx_whisper):
        """Errors from mlx_whisper should propagate to caller."""
        from donttype.transcribe_local import LocalTranscriptionClient

        mock_mlx_whisper.transcribe.side_effect = RuntimeError("inference failed")
        client = LocalTranscriptionClient()

        import io, wave, numpy as np
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(1000, dtype=np.int16).tobytes())

        with pytest.raises(RuntimeError):
            client.transcribe(buf.getvalue())

    def test_close_is_noop(self):
        """close() should not raise."""
        from donttype.transcribe_local import LocalTranscriptionClient

        client = LocalTranscriptionClient()
        client.close()  # should not raise


def _make_wav_bytes(n_samples=1000):
    """Helper: create valid mono 16-bit WAV bytes."""
    import io, wave, numpy as np

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(n_samples, dtype=np.int16).tobytes())
    return buf.getvalue()


class TestLocalTranscriptionFiltering:
    """Test that local client applies dedup filtering."""

    @patch("donttype.transcribe_local.mlx_whisper", create=True)
    def test_hallucination_returns_empty(self, mock_mlx_whisper):
        """Known hallucination from local model should produce empty string."""
        from donttype.transcribe_local import LocalTranscriptionClient

        mock_mlx_whisper.transcribe.return_value = {"text": "Thank you."}
        client = LocalTranscriptionClient()

        assert client.transcribe(_make_wav_bytes()) == ""

    @patch("donttype.transcribe_local.mlx_whisper", create=True)
    def test_bye_is_hallucination(self, mock_mlx_whisper):
        """'Bye.' is a known Whisper silence hallucination."""
        from donttype.transcribe_local import LocalTranscriptionClient

        mock_mlx_whisper.transcribe.return_value = {"text": "Bye."}
        client = LocalTranscriptionClient()

        assert client.transcribe(_make_wav_bytes()) == ""

    @patch("donttype.transcribe_local.mlx_whisper", create=True)
    def test_repetition_is_truncated(self, mock_mlx_whisper):
        """Repeated phrases from local model should be truncated."""
        from donttype.transcribe_local import LocalTranscriptionClient

        repeated = "okay. " * 5
        mock_mlx_whisper.transcribe.return_value = {"text": repeated}
        client = LocalTranscriptionClient()

        result = client.transcribe(_make_wav_bytes())
        assert result.count("okay.") < 5

    @patch("donttype.transcribe_local.mlx_whisper", create=True)
    def test_real_text_passes_through(self, mock_mlx_whisper):
        """Normal transcription text should not be filtered."""
        from donttype.transcribe_local import LocalTranscriptionClient

        mock_mlx_whisper.transcribe.return_value = {"text": "Hello, this is a test."}
        client = LocalTranscriptionClient()

        assert client.transcribe(_make_wav_bytes()) == "Hello, this is a test."

    @patch("donttype.transcribe_local.mlx_whisper", create=True)
    def test_whitespace_only_is_hallucination(self, mock_mlx_whisper):
        """Whitespace-only result should be treated as hallucination."""
        from donttype.transcribe_local import LocalTranscriptionClient

        mock_mlx_whisper.transcribe.return_value = {"text": "   "}
        client = LocalTranscriptionClient()

        assert client.transcribe(_make_wav_bytes()) == ""
