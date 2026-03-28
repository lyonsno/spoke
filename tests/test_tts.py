"""Tests for the TTS client and command-completion autoplay hook."""

import threading
from unittest.mock import patch, MagicMock, call

import pytest


class TestTTSClient:
    """Test TTSClient model loading, generation, playback, and cancellation."""

    def _make_client(self, **kwargs):
        from spoke.tts import TTSClient
        defaults = {
            "model_id": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
            "voice": "casual_female",
        }
        defaults.update(kwargs)
        return TTSClient(**defaults)

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_speak_generates_and_plays_audio(self, mock_load, mock_sd):
        """speak() calls model.generate() and plays the result via sounddevice."""
        fake_result = MagicMock()
        fake_result.audio = MagicMock()
        fake_result.sample_rate = 24000
        fake_model = MagicMock()
        fake_model.generate.return_value = iter([fake_result])
        mock_load.return_value = fake_model

        client = self._make_client()
        client.speak("Hello world")

        fake_model.generate.assert_called_once_with(
            text="Hello world", voice="casual_female"
        )
        mock_sd.play.assert_called_once_with(fake_result.audio, 24000)
        mock_sd.wait.assert_called_once()

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_speak_respects_voice_parameter(self, mock_load, mock_sd):
        """Voice parameter is forwarded to model.generate()."""
        fake_model = MagicMock()
        fake_model.generate.return_value = iter([
            MagicMock(audio=MagicMock(), sample_rate=24000)
        ])
        mock_load.return_value = fake_model

        client = self._make_client(voice="neutral_male")
        client.speak("test")

        _, kwargs = fake_model.generate.call_args
        assert kwargs["voice"] == "neutral_male"

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_cancel_stops_playback(self, mock_load, mock_sd):
        """cancel() calls sd.stop() and sets the cancelled flag."""
        fake_model = MagicMock()
        fake_model.generate.return_value = iter([])
        mock_load.return_value = fake_model

        client = self._make_client()
        client.cancel()

        mock_sd.stop.assert_called_once()

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_speak_skips_if_cancelled_before_play(self, mock_load, mock_sd):
        """If cancel() was called before audio is ready, don't play."""
        fake_result = MagicMock()
        fake_result.audio = MagicMock()
        fake_result.sample_rate = 24000
        fake_model = MagicMock()
        fake_model.generate.return_value = iter([fake_result])
        mock_load.return_value = fake_model

        client = self._make_client()
        client.cancel()  # cancel before speak
        client.speak("Hello world")

        mock_sd.play.assert_not_called()

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_model_loaded_lazily_on_first_speak(self, mock_load, mock_sd):
        """Model is not loaded at construction time — only on first speak()."""
        fake_model = MagicMock()
        fake_model.generate.return_value = iter([
            MagicMock(audio=MagicMock(), sample_rate=24000)
        ])
        mock_load.return_value = fake_model

        client = self._make_client()
        mock_load.assert_not_called()

        client.speak("hello")
        mock_load.assert_called_once_with(
            "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
        )

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_model_loaded_only_once(self, mock_load, mock_sd):
        """Second speak() reuses the already-loaded model."""
        fake_model = MagicMock()
        fake_model.generate.return_value = iter([
            MagicMock(audio=MagicMock(), sample_rate=24000)
        ])
        mock_load.return_value = fake_model

        client = self._make_client()
        client.speak("first")
        client.speak("second")

        mock_load.assert_called_once()

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_speak_empty_string_is_noop(self, mock_load, mock_sd):
        """Speaking an empty string should not invoke the model."""
        fake_model = MagicMock()
        mock_load.return_value = fake_model

        client = self._make_client()
        client.speak("")

        fake_model.generate.assert_not_called()

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_speak_async_runs_in_background(self, mock_load, mock_sd):
        """speak_async() returns immediately and plays in a background thread."""
        fake_result = MagicMock()
        fake_result.audio = MagicMock()
        fake_result.sample_rate = 24000
        fake_model = MagicMock()
        fake_model.generate.return_value = iter([fake_result])
        mock_load.return_value = fake_model

        client = self._make_client()
        thread = client.speak_async("hello")

        assert isinstance(thread, threading.Thread)
        thread.join(timeout=5)
        assert not thread.is_alive()
        mock_sd.play.assert_called_once()


class TestTTSConfig:
    """Test TTS configuration via environment variables."""

    @patch.dict("os.environ", {"SPOKE_TTS_VOICE": "casual_female"})
    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_tts_client_from_env(self, mock_load, mock_sd):
        """TTSClient.from_env() reads SPOKE_TTS_VOICE."""
        from spoke.tts import TTSClient
        client = TTSClient.from_env()
        assert client._voice == "casual_female"

    @patch.dict("os.environ", {}, clear=False)
    def test_tts_disabled_when_no_env(self):
        """When SPOKE_TTS_VOICE is unset, from_env() returns None."""
        import os
        os.environ.pop("SPOKE_TTS_VOICE", None)
        from spoke.tts import TTSClient
        assert TTSClient.from_env() is None

    @patch.dict("os.environ", {
        "SPOKE_TTS_VOICE": "neutral_male",
        "SPOKE_TTS_MODEL": "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
    })
    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_tts_model_from_env(self, mock_load, mock_sd):
        """SPOKE_TTS_MODEL overrides the default model ID."""
        from spoke.tts import TTSClient
        client = TTSClient.from_env()
        assert client._model_id == "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"


class TestCommandCompletionAutoplay:
    """Test that commandComplete_ triggers TTS when configured."""

    def _make_delegate(self, main_module, tts_client=None):
        """Create a SpokeAppDelegate with optional TTS client."""
        delegate = main_module.SpokeAppDelegate.alloc().init()
        delegate._transcription_token = 1
        delegate._transcribing = True
        delegate._glow = None
        delegate._command_overlay = MagicMock()
        delegate._menubar = MagicMock()
        delegate._tts_client = tts_client
        return delegate

    def test_command_complete_triggers_tts(self, main_module):
        """When TTS client is present, commandComplete_ calls speak_async."""
        tts = MagicMock()
        tts.speak_async.return_value = MagicMock(spec=threading.Thread)
        delegate = self._make_delegate(main_module, tts_client=tts)

        delegate.commandComplete_({"token": 1, "response": "Hello there"})

        tts.speak_async.assert_called_once_with("Hello there")

    def test_command_complete_no_tts_when_disabled(self, main_module):
        """When TTS client is None, commandComplete_ works normally without TTS."""
        delegate = self._make_delegate(main_module, tts_client=None)

        # Should not raise
        delegate.commandComplete_({"token": 1, "response": "Hello"})

        delegate._command_overlay.finish.assert_called_once()

    def test_hold_start_cancels_tts(self, main_module):
        """Starting a new hold cancels any in-flight TTS playback."""
        tts = MagicMock()
        delegate = self._make_delegate(main_module, tts_client=tts)
        delegate._models_ready = True
        delegate._capture = MagicMock()
        delegate._overlay = MagicMock()
        delegate._glow = MagicMock()
        delegate._detector = MagicMock()
        delegate._detector._shift_at_press = False
        delegate._preview_done = None
        delegate._preview_session_token = 0

        delegate._on_hold_start()

        tts.cancel.assert_called_once()
