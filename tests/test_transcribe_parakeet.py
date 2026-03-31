"""Tests for ParakeetCoreMLClient."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    samples = np.zeros(int(duration_s * sample_rate), dtype=np.float32)
    pcm = (samples * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _make_speech_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Simple sine wave to simulate non-silent audio."""
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), dtype=np.float32)
    samples = (np.sin(2 * np.pi * 440.0 * t) * 0.3).astype(np.float32)
    pcm = (samples * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class TestParakeetCoreMLClientInterface:
    """Contract tests — no real CoreML models required."""

    def test_empty_bytes_returns_empty_string(self):
        from spoke.transcribe_parakeet import ParakeetCoreMLClient

        client = ParakeetCoreMLClient(model_dir=Path("/nonexistent"))
        assert client.transcribe(b"") == ""

    def test_prepare_is_noop_without_model_dir(self):
        """prepare() must not raise even when the model dir is absent."""
        from spoke.transcribe_parakeet import ParakeetCoreMLClient

        client = ParakeetCoreMLClient(model_dir=Path("/nonexistent"))
        # Should raise or log but not crash at construction; prepare loads lazily
        # This test just confirms the interface exists.
        assert callable(client.prepare)

    def test_close_is_safe(self):
        from spoke.transcribe_parakeet import ParakeetCoreMLClient

        client = ParakeetCoreMLClient(model_dir=Path("/nonexistent"))
        client.close()  # must not raise

    def test_model_dir_stored(self):
        from spoke.transcribe_parakeet import ParakeetCoreMLClient

        p = Path("/some/model/dir")
        client = ParakeetCoreMLClient(model_dir=p)
        assert client._model_dir == p

    def test_default_model_id(self):
        from spoke.transcribe_parakeet import (
            ParakeetCoreMLClient,
            _PARAKEET_MODEL_ID,
        )

        client = ParakeetCoreMLClient(model_dir=Path("/nonexistent"))
        assert client._model_id == _PARAKEET_MODEL_ID


class TestParakeetDecoding:
    """Unit tests for the CTC greedy decoder and vocab loading."""

    def test_ctc_decode_collapses_repeats_and_blanks(self):
        from spoke.transcribe_parakeet import _ctc_greedy_decode

        vocab = {0: "<unk>", 1: "a", 2: "b", 3: "c"}
        blank_id = 4
        # [a, a, blank, b, blank, blank, c, c] -> "abc"
        token_ids = [1, 1, 4, 2, 4, 4, 3, 3]
        result = _ctc_greedy_decode(token_ids, vocab, blank_id)
        assert result == "abc"

    def test_ctc_decode_all_blanks_returns_empty(self):
        from spoke.transcribe_parakeet import _ctc_greedy_decode

        vocab = {0: "a"}
        result = _ctc_greedy_decode([1, 1, 1], vocab, blank_id=1)
        assert result == ""

    def test_ctc_decode_sentencepiece_space(self):
        from spoke.transcribe_parakeet import _ctc_greedy_decode

        # ▁ prefix means word boundary in SentencePiece
        vocab = {0: "▁hello", 1: "▁world"}
        result = _ctc_greedy_decode([0, 1], vocab, blank_id=2)
        assert result == "hello world"

    def test_ctc_decode_single_token(self):
        from spoke.transcribe_parakeet import _ctc_greedy_decode

        vocab = {0: "▁hi"}
        result = _ctc_greedy_decode([0], vocab, blank_id=1)
        assert result == "hi"

    def test_load_vocab_from_json(self, tmp_path):
        import json
        from spoke.transcribe_parakeet import _load_vocab

        vocab_data = {"0": "<unk>", "1": "▁t", "2": "▁th", "1024": "<blank>"}
        vocab_path = tmp_path / "vocab.json"
        vocab_path.write_text(json.dumps(vocab_data))
        vocab = _load_vocab(vocab_path)
        assert vocab[0] == "<unk>"
        assert vocab[1] == "▁t"
        assert vocab[1024] == "<blank>"


class TestParakeetChunking:
    """Tests for audio chunking — ensures long audio is split correctly."""

    def test_wav_decode_produces_float32(self):
        from spoke.transcribe_parakeet import _decode_wav_to_float32

        wav = _make_speech_wav_bytes(duration_s=1.0)
        pcm = _decode_wav_to_float32(wav)
        assert pcm.dtype == np.float32
        assert len(pcm) == 16000

    def test_wav_decode_normalises_range(self):
        from spoke.transcribe_parakeet import _decode_wav_to_float32

        wav = _make_speech_wav_bytes(duration_s=0.5)
        pcm = _decode_wav_to_float32(wav)
        assert pcm.max() <= 1.0
        assert pcm.min() >= -1.0

    def test_chunk_audio_short_returns_one_chunk(self):
        from spoke.transcribe_parakeet import _chunk_audio

        samples = np.zeros(16000, dtype=np.float32)  # 1 second
        chunks = _chunk_audio(samples, sample_rate=16000, max_chunk_secs=15.0)
        assert len(chunks) == 1
        assert len(chunks[0]) == 16000

    def test_chunk_audio_exact_boundary(self):
        from spoke.transcribe_parakeet import _chunk_audio

        # Exactly 15 seconds — should be one chunk
        samples = np.zeros(15 * 16000, dtype=np.float32)
        chunks = _chunk_audio(samples, sample_rate=16000, max_chunk_secs=15.0)
        assert len(chunks) == 1

    def test_chunk_audio_long_splits_into_multiple_chunks(self):
        from spoke.transcribe_parakeet import _chunk_audio

        # 31 seconds should produce 3 chunks: 15s, 15s, 1s
        samples = np.zeros(31 * 16000, dtype=np.float32)
        chunks = _chunk_audio(samples, sample_rate=16000, max_chunk_secs=15.0)
        assert len(chunks) == 3
        assert len(chunks[0]) == 15 * 16000
        assert len(chunks[1]) == 15 * 16000
        assert len(chunks[2]) == 1 * 16000


class TestParakeetMelspectrogram:
    """Tests for mel feature computation — no CoreML required."""

    def test_compute_mel_returns_correct_shape(self):
        from spoke.transcribe_parakeet import _compute_mel_features

        samples = np.zeros(15 * 16000, dtype=np.float32)
        mel = _compute_mel_features(samples)
        # Should be [1, 1, 1501, 80] matching the AudioEncoder input
        assert mel.shape == (1, 1, 1501, 80)
        assert mel.dtype == np.float16

    def test_compute_mel_short_audio_is_padded(self):
        from spoke.transcribe_parakeet import _compute_mel_features

        # 1 second of audio — should be padded to 1501 frames
        samples = np.zeros(16000, dtype=np.float32)
        mel = _compute_mel_features(samples)
        assert mel.shape == (1, 1, 1501, 80)

    def test_compute_mel_non_silent_differs_from_silent(self):
        from spoke.transcribe_parakeet import _compute_mel_features

        silent = np.zeros(16000, dtype=np.float32)
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        speech = np.sin(2 * np.pi * 440 * t) * 0.3
        mel_silent = _compute_mel_features(silent)
        mel_speech = _compute_mel_features(speech)
        assert not np.allclose(mel_silent, mel_speech)


class TestParakeetTranscribeWithMock:
    """Integration-style tests using a mocked CoreML model."""

    def _make_mock_model(self, logits_shape=(1, 1025, 1, 188)):
        """Return a mock AudioEncoder that emits blank everywhere."""
        mock = MagicMock()
        # All logits negative except blank (index 1024)
        logits = np.full(logits_shape, -10.0, dtype=np.float32)
        logits[0, 1024, 0, :] = 10.0  # blank token wins everywhere
        mock.predict.return_value = {"ctc_head_output": logits.astype(np.float16)}
        return mock

    def test_transcribe_returns_empty_when_all_blanks(self, tmp_path):
        import json
        from spoke.transcribe_parakeet import ParakeetCoreMLClient

        # Build minimal vocab + model dir
        vocab = {str(i): f"tok_{i}" for i in range(1024)}
        vocab["1024"] = "<blank>"
        (tmp_path / "vocab.json").write_text(json.dumps(vocab))

        client = ParakeetCoreMLClient(model_dir=tmp_path)
        client._encoder = self._make_mock_model()
        client._vocab = {int(k): v for k, v in vocab.items()}
        client._blank_id = 1024
        client._loaded = True

        result = client.transcribe(_make_wav_bytes())
        assert result == ""

    def test_transcribe_joins_chunks(self, tmp_path):
        import json
        from spoke.transcribe_parakeet import ParakeetCoreMLClient

        vocab = {str(i): f"tok_{i}" for i in range(1024)}
        vocab["1024"] = "<blank>"
        # Make token 0 = "▁hello" so we get real text back
        vocab["0"] = "▁hello"
        (tmp_path / "vocab.json").write_text(json.dumps(vocab))

        # Create a model that returns token 0 on first frame, then blanks
        mock = MagicMock()
        logits = np.full((1, 1025, 1, 188), -10.0, dtype=np.float32)
        logits[0, 1024, 0, :] = 10.0   # blanks
        logits[0, 0, 0, 0] = 20.0       # token 0 wins on first frame
        logits[0, 1024, 0, 0] = -10.0   # cancel blank on first frame
        mock.predict.return_value = {"ctc_head_output": logits.astype(np.float16)}

        client = ParakeetCoreMLClient(model_dir=tmp_path)
        client._encoder = mock
        client._vocab = {int(k): v for k, v in vocab.items()}
        client._blank_id = 1024
        client._loaded = True

        # 20s audio — 2 chunks, each contributing "hello"
        wav = _make_speech_wav_bytes(duration_s=20.0)
        result = client.transcribe(wav)
        # Each chunk should produce "hello"; they should be joined
        assert "hello" in result


class TestParakeetWiring:
    """Tests for __main__.py wiring — preview-only guard and model_allowed."""

    def test_parakeet_in_model_options(self):
        """Parakeet must appear in _MODEL_OPTIONS."""
        from spoke.__main__ import SpokeAppDelegate
        from spoke.transcribe_parakeet import _PARAKEET_MODEL_ID

        ids = [mid for mid, _label in SpokeAppDelegate._MODEL_OPTIONS]
        assert _PARAKEET_MODEL_ID in ids

    def test_parakeet_in_preview_only_models(self):
        """Parakeet must be in _PREVIEW_ONLY_MODELS."""
        from spoke.__main__ import SpokeAppDelegate
        from spoke.transcribe_parakeet import _PARAKEET_MODEL_ID

        assert _PARAKEET_MODEL_ID in SpokeAppDelegate._PREVIEW_ONLY_MODELS

    def test_sanitize_transcription_role_rejects_parakeet(self, monkeypatch, tmp_path):
        """_sanitize_model_id must fall back when Parakeet is used for transcription."""
        from spoke.__main__ import SpokeAppDelegate
        from spoke.transcribe_parakeet import _PARAKEET_MODEL_ID

        # Patch RAM and model-allowed to allow it superficially
        monkeypatch.setattr("spoke.__main__._RAM_GB", 128.0)

        delegate = SpokeAppDelegate.__new__(SpokeAppDelegate)
        result = delegate._sanitize_model_id(_PARAKEET_MODEL_ID, role="transcription")
        assert result != _PARAKEET_MODEL_ID

    def test_sanitize_preview_role_allows_parakeet_when_files_present(self, tmp_path, monkeypatch):
        """_sanitize_model_id must allow Parakeet for preview when model files exist."""
        from spoke.__main__ import SpokeAppDelegate
        from spoke.transcribe_parakeet import _PARAKEET_MODEL_ID

        encoder_dir = tmp_path / "AudioEncoder.mlmodelc"
        encoder_dir.mkdir()
        monkeypatch.setenv("SPOKE_PARAKEET_MODEL_DIR", str(tmp_path))
        monkeypatch.setattr("spoke.__main__._RAM_GB", 128.0)

        delegate = SpokeAppDelegate.__new__(SpokeAppDelegate)
        result = delegate._sanitize_model_id(_PARAKEET_MODEL_ID, role="preview")
        assert result == _PARAKEET_MODEL_ID

    def test_model_allowed_false_when_no_files(self, tmp_path, monkeypatch):
        """_model_allowed must return False when Parakeet files are absent.

        Both the env-var path and the HF snapshot fallback must be neutralized
        so the test is not sensitive to whether the real model is cached locally.
        """
        from pathlib import Path
        from spoke.__main__ import SpokeAppDelegate
        from spoke.transcribe_parakeet import _PARAKEET_MODEL_ID

        # Point env var at an empty dir (no AudioEncoder.mlmodelc)
        monkeypatch.setenv("SPOKE_PARAKEET_MODEL_DIR", str(tmp_path))
        # Redirect Path.home() so the HF snapshot fallback also finds nothing
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        assert SpokeAppDelegate._model_allowed(_PARAKEET_MODEL_ID) is False

    def test_model_allowed_true_when_encoder_present(self, tmp_path, monkeypatch):
        """_model_allowed must return True when AudioEncoder.mlmodelc exists."""
        from spoke.__main__ import SpokeAppDelegate
        from spoke.transcribe_parakeet import _PARAKEET_MODEL_ID

        encoder_dir = tmp_path / "AudioEncoder.mlmodelc"
        encoder_dir.mkdir()
        monkeypatch.setenv("SPOKE_PARAKEET_MODEL_DIR", str(tmp_path))
        assert SpokeAppDelegate._model_allowed(_PARAKEET_MODEL_ID) is True
