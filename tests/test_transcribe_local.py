"""Tests for local MLX Whisper transcription client."""

from unittest.mock import MagicMock, patch
import os
import tempfile

import pytest


class TestLocalTranscriptionClient:
    """Test the in-process MLX Whisper client."""

    def test_empty_bytes_returns_empty_string(self):
        """Empty WAV input should short-circuit without calling mlx_whisper."""
        from dictate.transcribe_local import LocalTranscriptionClient

        client = LocalTranscriptionClient(model="test-model")
        result = client.transcribe(b"")
        assert result == ""

    def test_default_model(self):
        from dictate.transcribe_local import LocalTranscriptionClient, _DEFAULT_MODEL

        client = LocalTranscriptionClient()
        assert client._model == _DEFAULT_MODEL

    def test_custom_model(self):
        from dictate.transcribe_local import LocalTranscriptionClient

        client = LocalTranscriptionClient(model="custom/model")
        assert client._model == "custom/model"

    @patch("dictate.transcribe_local.mlx_whisper", create=True)
    def test_transcribe_calls_mlx_whisper(self, mock_mlx_whisper):
        """transcribe() should call mlx_whisper.transcribe with correct args."""
        from dictate.transcribe_local import LocalTranscriptionClient

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

    @patch("dictate.transcribe_local.mlx_whisper", create=True)
    def test_transcribe_strips_whitespace(self, mock_mlx_whisper):
        """Transcription result should be stripped."""
        from dictate.transcribe_local import LocalTranscriptionClient

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

    @patch("dictate.transcribe_local.mlx_whisper", create=True)
    def test_transcribe_missing_text_key(self, mock_mlx_whisper):
        """Missing 'text' key should return empty string."""
        from dictate.transcribe_local import LocalTranscriptionClient

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

    @patch("dictate.transcribe_local.mlx_whisper", create=True)
    def test_temp_file_cleaned_up(self, mock_mlx_whisper):
        """Temp WAV file should be deleted after transcription."""
        from dictate.transcribe_local import LocalTranscriptionClient

        created_paths = []
        original_transcribe = mock_mlx_whisper.transcribe

        def capture_path(path, **kwargs):
            created_paths.append(path)
            return {"text": "test"}

        mock_mlx_whisper.transcribe.side_effect = capture_path
        client = LocalTranscriptionClient()

        import io, wave, numpy as np
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(1000, dtype=np.int16).tobytes())

        client.transcribe(buf.getvalue())
        assert len(created_paths) == 1
        assert not os.path.exists(created_paths[0])

    @patch("dictate.transcribe_local.mlx_whisper", create=True)
    def test_temp_file_cleaned_up_on_error(self, mock_mlx_whisper):
        """Temp WAV file should be deleted even if transcription fails."""
        from dictate.transcribe_local import LocalTranscriptionClient

        created_paths = []

        def capture_and_fail(path, **kwargs):
            created_paths.append(path)
            raise RuntimeError("inference failed")

        mock_mlx_whisper.transcribe.side_effect = capture_and_fail
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
        assert len(created_paths) == 1
        assert not os.path.exists(created_paths[0])

    def test_close_is_noop(self):
        """close() should not raise."""
        from dictate.transcribe_local import LocalTranscriptionClient

        client = LocalTranscriptionClient()
        client.close()  # should not raise
