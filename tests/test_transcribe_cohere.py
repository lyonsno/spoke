"""Tests for local Cohere Transcribe client."""

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


_LOAD_MODEL_PATH = "spoke.transcribe_cohere._load_stt_model"


class TestLocalCohereClient:
    """Test the in-process Cohere Transcribe client."""

    def test_empty_bytes_returns_empty_string(self):
        from spoke.transcribe_cohere import LocalCohereClient

        client = LocalCohereClient(model="test-model")
        assert client.transcribe(b"") == ""

    def test_default_model(self):
        from spoke.transcribe_cohere import LocalCohereClient, _DEFAULT_MODEL

        client = LocalCohereClient()
        assert client._model_id == _DEFAULT_MODEL

    def test_custom_model(self):
        from spoke.transcribe_cohere import LocalCohereClient

        client = LocalCohereClient(model="CohereLabs/custom-model")
        assert client._model_id == "CohereLabs/custom-model"

    @patch(_LOAD_MODEL_PATH)
    def test_transcribe_calls_model(self, mock_load):
        from spoke.transcribe_cohere import LocalCohereClient

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ["hello world"]
        mock_load.return_value = mock_model

        client = LocalCohereClient()
        result = client.transcribe(_make_wav_bytes())

        assert result == "hello world"
        mock_model.transcribe.assert_called_once()
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "english"
        assert len(call_kwargs["audio_arrays"]) == 1
        assert isinstance(call_kwargs["audio_arrays"][0], np.ndarray)
        assert call_kwargs["audio_arrays"][0].dtype == np.float32
        assert call_kwargs["sample_rates"] == [16000]

    @patch(_LOAD_MODEL_PATH)
    def test_transcribe_strips_whitespace(self, mock_load):
        from spoke.transcribe_cohere import LocalCohereClient

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ["\n  hello  \n"]
        mock_load.return_value = mock_model

        client = LocalCohereClient()
        assert client.transcribe(_make_wav_bytes()) == "hello"

    @patch(_LOAD_MODEL_PATH)
    def test_transcribe_discards_hallucination(self, mock_load):
        from spoke.transcribe_cohere import LocalCohereClient

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ["Thank you."]
        mock_load.return_value = mock_model

        client = LocalCohereClient()
        assert client.transcribe(_make_wav_bytes()) == ""

    @patch(_LOAD_MODEL_PATH)
    def test_transcribe_empty_result(self, mock_load):
        from spoke.transcribe_cohere import LocalCohereClient

        mock_model = MagicMock()
        mock_model.transcribe.return_value = []
        mock_load.return_value = mock_model

        client = LocalCohereClient()
        assert client.transcribe(_make_wav_bytes()) == ""

    @patch(_LOAD_MODEL_PATH)
    def test_transcribe_error_returns_empty(self, mock_load):
        from spoke.transcribe_cohere import LocalCohereClient

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("boom")
        mock_load.return_value = mock_model

        client = LocalCohereClient()
        assert client.transcribe(_make_wav_bytes()) == ""

    @patch(_LOAD_MODEL_PATH)
    def test_prepare_loads_model(self, mock_load):
        from spoke.transcribe_cohere import LocalCohereClient

        mock_model = MagicMock()
        mock_load.return_value = mock_model

        client = LocalCohereClient()
        assert not client._loaded

        client.prepare()
        assert client._loaded
        mock_load.assert_called_once()

        # Second prepare is a no-op
        client.prepare()
        mock_load.assert_called_once()

    @patch(_LOAD_MODEL_PATH)
    def test_close_releases_model(self, mock_load):
        from spoke.transcribe_cohere import LocalCohereClient

        mock_load.return_value = MagicMock()

        client = LocalCohereClient()
        client.prepare()
        assert client._loaded

        client.close()
        assert not client._loaded
        assert client._model is None

    def test_decode_wav_rejects_stereo(self):
        from spoke.transcribe_cohere import LocalCohereClient

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(2000, dtype=np.int16).tobytes())

        with pytest.raises(ValueError, match="mono"):
            LocalCohereClient._decode_wav(buf.getvalue())
