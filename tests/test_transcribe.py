"""Tests for the Whisper transcription HTTP client."""

from unittest.mock import MagicMock, patch

import pytest

from spoke.transcribe import TranscriptionClient, _DEFAULT_MODEL


class TestTranscriptionClient:
    """Test the OpenAI-compatible Whisper client."""

    def test_url_construction(self):
        """Base URL should have /v1/audio/transcriptions appended."""
        client = TranscriptionClient(base_url="http://sidecar:8000")
        assert client._url == "http://sidecar:8000/v1/audio/transcriptions"

    def test_url_strips_trailing_slash(self):
        """Trailing slash on base URL should not cause double-slash."""
        client = TranscriptionClient(base_url="http://sidecar:8000/")
        assert client._url == "http://sidecar:8000/v1/audio/transcriptions"

    def test_default_model(self):
        client = TranscriptionClient(base_url="http://x")
        assert client._model == _DEFAULT_MODEL

    def test_custom_model(self):
        client = TranscriptionClient(base_url="http://x", model="custom/whisper")
        assert client._model == "custom/whisper"

    def test_empty_bytes_returns_empty_string(self):
        """Empty WAV input should short-circuit without HTTP call."""
        client = TranscriptionClient(base_url="http://x")
        client._client = MagicMock()
        result = client.transcribe(b"")
        assert result == ""
        client._client.post.assert_not_called()

    @patch("spoke.transcribe.httpx.Client")
    def test_transcribe_sends_correct_request(self, MockClient):
        """transcribe() should POST multipart with file and model."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "hello world"}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        client = TranscriptionClient(base_url="http://sidecar:8000", model="test-model")
        client._client = mock_client

        result = client.transcribe(b"RIFF...fake wav data")
        assert result == "hello world"

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[0][0] == "http://sidecar:8000/v1/audio/transcriptions"
        assert "file" in call_kwargs[1]["files"]
        assert call_kwargs[1]["data"]["model"] == "test-model"

    @patch("spoke.transcribe.httpx.Client")
    def test_transcribe_strips_whitespace(self, MockClient):
        """Transcription result should be stripped of leading/trailing whitespace."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "  hello world  \n"}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        client = TranscriptionClient(base_url="http://x")
        client._client = mock_client

        assert client.transcribe(b"wav") == "hello world"

    @patch("spoke.transcribe.httpx.Client")
    def test_transcribe_missing_text_key(self, MockClient):
        """If response has no 'text' key, should return empty string."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"segments": []}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        client = TranscriptionClient(base_url="http://x")
        client._client = mock_client

        assert client.transcribe(b"wav") == ""

    def test_server_returns_http_error(self):
        """HTTP 500/404/etc should raise, not crash."""
        import httpx

        client = TranscriptionClient(base_url="http://x")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=MagicMock()
        )
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        client._client = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            client.transcribe(b"wav")

    def test_server_connection_refused(self):
        """ConnectError (server down) should raise, not crash."""
        import httpx

        client = TranscriptionClient(base_url="http://x")
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        client._client = mock_client

        with pytest.raises(httpx.ConnectError):
            client.transcribe(b"wav")

    def test_server_returns_invalid_json(self):
        """Server returning non-JSON (e.g. HTML error page) should raise."""
        import json

        client = TranscriptionClient(base_url="http://x")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = json.JSONDecodeError("", "", 0)
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        client._client = mock_client

        with pytest.raises(json.JSONDecodeError):
            client.transcribe(b"wav")

    def test_close(self):
        """close() should close the underlying httpx client."""
        client = TranscriptionClient(base_url="http://x")
        client._client = MagicMock()
        client.close()
        client._client.close.assert_called_once()


class TestTranscriptionFiltering:
    """Test that transcription client applies dedup filtering."""

    def test_hallucination_returns_empty(self):
        """Server returning a known hallucination should produce empty string."""
        client = TranscriptionClient(base_url="http://x")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "Thank you."}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        client._client = mock_client

        assert client.transcribe(b"wav") == ""

    def test_thanks_for_watching_returns_empty(self):
        """'Thanks for watching.' is a known Whisper silence hallucination."""
        client = TranscriptionClient(base_url="http://x")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "Thanks for watching."}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        client._client = mock_client

        assert client.transcribe(b"wav") == ""

    def test_repetition_is_truncated(self):
        """Repeated phrases should be truncated to a single occurrence."""
        repeated = "I think so. " * 5
        client = TranscriptionClient(base_url="http://x")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": repeated}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        client._client = mock_client

        result = client.transcribe(b"wav")
        assert result.count("I think so.") < 5

    def test_real_text_passes_through(self):
        """Normal transcription text should not be filtered."""
        client = TranscriptionClient(base_url="http://x")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "The quick brown fox jumps over the lazy dog."}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        client._client = mock_client

        assert client.transcribe(b"wav") == "The quick brown fox jumps over the lazy dog."

    def test_observed_ontology_failures_are_repaired(self):
        """Observed launch-log ontology failures should normalize on the Whisper path."""
        client = TranscriptionClient(base_url="http://x")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "Read Nepistaxis and update the Topoie."}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        client._client = mock_client

        assert client.transcribe(b"wav") == "Read Epistaxis and update the topoi."

    def test_recent_ontology_failures_are_repaired(self):
        """Recent launch-log variants should normalize on the Whisper path."""
        client = TranscriptionClient(base_url="http://x")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "text": "Check the Uxis document, the Metadose II, and the Syllogy."
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        client._client = mock_client

        assert client.transcribe(b"wav") == (
            "Check the Auxesis document, the Metadosis, and the sylloge."
        )

    def test_additional_ontology_failures_are_repaired(self):
        """Additional ontology variants should normalize on the Whisper path."""
        client = TranscriptionClient(base_url="http://x")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "text": "Check the sueji, the kerigma badge, and epinorthosis."
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        client._client = mock_client

        assert client.transcribe(b"wav") == (
            "Check the sylloge, the kerygma badge, and epanorthosis."
        )

    def test_whitespace_only_is_hallucination(self):
        """Whitespace-only result should be treated as hallucination."""
        client = TranscriptionClient(base_url="http://x")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "   \n  "}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        client._client = mock_client

        assert client.transcribe(b"wav") == ""
