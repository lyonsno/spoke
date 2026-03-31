"""Tests for the TTS client and command-completion autoplay hook."""

import logging
import threading
import time
import types
from unittest.mock import patch, MagicMock, call

import pytest


def _fake_result():
    """Create a fake GenerationResult with real audio data."""
    r = MagicMock()
    r.audio = [0.1, 0.2, 0.3]
    r.sample_rate = 24000
    return r


def _setup_stream_mock(mock_sd):
    """Configure mock_sd.OutputStream to simulate instant playback."""
    fake_stream = MagicMock()
    def write_side_effect(data):
        for call_args in mock_sd.OutputStream.call_args_list:
            cb = call_args[1].get("finished_callback")
            if cb:
                cb()
    fake_stream.write = MagicMock(side_effect=write_side_effect)
    mock_sd.OutputStream.return_value = fake_stream
    return fake_stream


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
        """speak() calls model.generate() and plays via a dedicated OutputStream."""
        fake_model = MagicMock()
        fake_model.generate.return_value = iter([_fake_result()])
        mock_load.return_value = fake_model
        fake_stream = _setup_stream_mock(mock_sd)

        client = self._make_client()
        client.speak("Hello world")

        fake_model.generate.assert_called_once_with(
            text="Hello world", voice="casual_female",
            temperature=0.5, top_k=50, top_p=0.95,
        )
        mock_sd.OutputStream.assert_called_once()
        fake_stream.start.assert_called_once()
        fake_stream.write.assert_called_once()
        fake_stream.stop.assert_called_once()
        fake_stream.close.assert_called_once()

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_speak_respects_voice_parameter(self, mock_load, mock_sd):
        """Voice parameter is forwarded to model.generate()."""
        fake_model = MagicMock()
        fake_model.generate.return_value = iter([_fake_result()])
        mock_load.return_value = fake_model
        _setup_stream_mock(mock_sd)

        client = self._make_client(voice="neutral_male")
        client.speak("test")

        _, kwargs = fake_model.generate.call_args
        assert kwargs["voice"] == "neutral_male"

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_cancel_sets_flag_without_aborting(self, mock_load, mock_sd):
        """cancel() sets the cancelled flag — playback loop handles fade-out."""
        fake_model = MagicMock()
        mock_load.return_value = fake_model

        client = self._make_client()
        client.cancel()

        assert client._cancelled is True
        # No stream.abort() — the playback loop writes a fade-out and exits
        mock_sd.stop.assert_not_called()

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

        mock_sd.OutputStream.assert_not_called()

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_model_loaded_lazily_on_first_speak(self, mock_load, mock_sd):
        """Model is not loaded at construction time — only on first speak()."""
        fake_model = MagicMock()
        fake_model.generate.return_value = iter([_fake_result()])
        mock_load.return_value = fake_model
        _setup_stream_mock(mock_sd)

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
        fake_model.generate.side_effect = lambda **kw: iter([_fake_result()])
        mock_load.return_value = fake_model
        _setup_stream_mock(mock_sd)

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
        fake_model = MagicMock()
        fake_model.generate.return_value = iter([_fake_result()])
        mock_load.return_value = fake_model
        _setup_stream_mock(mock_sd)

        client = self._make_client()
        thread = client.speak_async("hello")

        assert isinstance(thread, threading.Thread)
        thread.join(timeout=5)
        assert not thread.is_alive()
        mock_sd.OutputStream.assert_called_once()

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_speak_async_serializes_same_client_jobs(self, mock_load, mock_sd):
        """Concurrent speak_async() calls on one client should queue instead of overlapping."""
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        call_order = []

        fake_model = MagicMock()

        def generate_side_effect(*, text, **kwargs):
            call_order.append(text)
            if text == "first":
                first_started.set()
                assert release_first.wait(timeout=5), "first speak never released"
            else:
                second_started.set()
            return iter([_fake_result()])

        fake_model.generate.side_effect = generate_side_effect
        mock_load.return_value = fake_model
        _setup_stream_mock(mock_sd)

        client = self._make_client()
        t1 = client.speak_async("first")
        assert first_started.wait(timeout=5), "first speak never started"

        t2 = client.speak_async("second")
        time.sleep(0.05)
        assert not second_started.is_set(), "second speak started before first finished"

        release_first.set()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert second_started.is_set()
        assert call_order == ["first", "second"]

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_speak_generates_one_sentence_at_a_time(self, mock_load, mock_sd):
        """Long text should be generated sentence-by-sentence instead of as one monolith."""
        fake_model = MagicMock()
        fake_model.generate.side_effect = lambda **kwargs: iter([_fake_result()])
        mock_load.return_value = fake_model
        _setup_stream_mock(mock_sd)

        client = self._make_client()
        client.speak("First sentence. Second sentence?")

        texts = [call.kwargs["text"] for call in fake_model.generate.call_args_list]
        assert texts == ["First sentence.", "Second sentence?"]

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_speak_starts_playback_before_later_sentence_generation_finishes(self, mock_load, mock_sd):
        """Later sentence generation should wait until current playback has finished."""
        playback_started = threading.Event()
        playback_finished = threading.Event()
        release_playback = threading.Event()

        fake_model = MagicMock()

        def generate_side_effect(*, text, **kwargs):
            if text == "Second sentence.":
                assert playback_finished.is_set(), (
                    "second sentence generation started before first playback finished"
                )
            return iter([_fake_result()])

        fake_model.generate.side_effect = generate_side_effect
        mock_load.return_value = fake_model

        fake_stream = MagicMock()

        def write_side_effect(data):
            playback_started.set()
            assert release_playback.wait(timeout=5), "first playback was never released"
            playback_finished.set()
            for call_args in mock_sd.OutputStream.call_args_list:
                cb = call_args[1].get("finished_callback")
                if cb:
                    cb()

        fake_stream.write = MagicMock(side_effect=write_side_effect)
        mock_sd.OutputStream.return_value = fake_stream

        client = self._make_client()

        def _release_once_playback_starts():
            assert playback_started.wait(timeout=5), "playback never started"
            release_playback.set()

        releaser = threading.Thread(target=_release_once_playback_starts, daemon=True)
        releaser.start()

        client.speak("First sentence. Second sentence.")

        releaser.join(timeout=5)
        assert not releaser.is_alive()

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_speak_starts_playback_before_long_sentence_generator_exhausts(self, mock_load, mock_sd):
        """A long single sentence should start playback on the first yielded result."""
        first_result_played = threading.Event()
        allow_second_result = threading.Event()

        fake_model = MagicMock()

        def generate_side_effect(*, text, **kwargs):
            assert text == "This is a long sentence without an early split point"
            yield _fake_result()
            assert allow_second_result.wait(timeout=5), "second result was never released"
            yield _fake_result()

        fake_model.generate.side_effect = generate_side_effect
        mock_load.return_value = fake_model

        fake_stream = MagicMock()

        def write_side_effect(data):
            first_result_played.set()
            for call_args in mock_sd.OutputStream.call_args_list:
                cb = call_args[1].get("finished_callback")
                if cb:
                    cb()

        fake_stream.write = MagicMock(side_effect=write_side_effect)
        mock_sd.OutputStream.return_value = fake_stream

        client = self._make_client()
        thread = client.speak_async("This is a long sentence without an early split point")

        assert first_result_played.wait(timeout=0.2), (
            "playback did not start before the long-sentence generator exhausted"
        )

        client.cancel()
        allow_second_result.set()
        thread.join(timeout=5)
        assert not thread.is_alive()

    def test_toggle_audio_uses_500ms_eased_fade(self):
        """Audio toggle should fade toward the new target over 500ms instead of jumping."""
        client = self._make_client()
        client._playback_active = True

        with patch("spoke.tts.time.monotonic", return_value=10.0):
            assert client.toggle_audio() is False

        with client._audio_fade_lock:
            assert client._current_audio_gain_locked(10.0) == pytest.approx(1.0)
            assert 0.0 < client._current_audio_gain_locked(10.25) < 1.0
            assert client._current_audio_gain_locked(10.5) == pytest.approx(0.0)

        with patch("spoke.tts.time.monotonic", return_value=12.0):
            assert client.toggle_audio() is True

        with client._audio_fade_lock:
            assert client._current_audio_gain_locked(12.0) == pytest.approx(0.0)
            assert 0.0 < client._current_audio_gain_locked(12.25) < 1.0
            assert client._current_audio_gain_locked(12.5) == pytest.approx(1.0)

    def test_toggle_audio_while_idle_updates_future_target(self):
        """Idle toggles should update the target gain for later playback too."""
        client = self._make_client()

        with patch("spoke.tts.time.monotonic", return_value=3.0):
            assert client.toggle_audio() is False

        with client._audio_fade_lock:
            assert client._audio_fade_start_gain == pytest.approx(1.0)
            assert client._audio_fade_target_gain == pytest.approx(0.0)

        with patch("spoke.tts.time.monotonic", return_value=4.0):
            assert client.toggle_audio() is True

        with client._audio_fade_lock:
            assert client._audio_fade_start_gain == pytest.approx(0.0)
            assert client._audio_fade_target_gain == pytest.approx(1.0)


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

    @patch.dict("os.environ", {"PYTHONPATH": "/tmp/local-mlx-audio"})
    def test_tts_load_surfaces_actionable_voxtral_backend_error(self, monkeypatch):
        """Missing Voxtral backend should fail with the resolved mlx_audio path in the error."""
        import sys
        from spoke.tts import tts_load

        fake_mlx_audio = types.ModuleType("mlx_audio")
        fake_mlx_audio.__file__ = "/tmp/local-mlx-audio/mlx_audio/__init__.py"
        fake_tts = types.ModuleType("mlx_audio.tts")
        fake_tts.load = MagicMock()

        monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
        monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)

        def fake_import_module(name):
            if name == "mlx_audio":
                return fake_mlx_audio
            if name == "mlx_audio.tts.models.voxtral_tts":
                raise ModuleNotFoundError(name)
            raise AssertionError(f"unexpected import: {name}")

        with patch("spoke.tts.importlib.import_module", side_effect=fake_import_module):
            with pytest.raises(RuntimeError) as excinfo:
                tts_load("mlx-community/Voxtral-4B-TTS-2603-mlx-6bit")

        message = str(excinfo.value)
        assert "Voxtral TTS backend is unavailable" in message
        assert "/tmp/local-mlx-audio/mlx_audio/__init__.py" in message
        assert "PYTHONPATH=/tmp/local-mlx-audio" in message
        assert ".spoke-smoke-env" in message


class TestGPULockDiscipline:
    """Verify that the GPU lock is held during generation but not playback."""

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_speak_holds_lock_during_generation_only(self, mock_load, mock_sd):
        """GPU lock is held during model.generate() but released before audio write."""
        import numpy as np
        lock = threading.Lock()
        locked_during_generate = []
        locked_during_write = []

        fake_model = MagicMock()
        def fake_generate(**kwargs):
            locked_during_generate.append(lock.locked())
            r = MagicMock()
            r.audio = [0.1] * 2400  # 100ms at 24kHz
            r.sample_rate = 24000
            yield r
        fake_model.generate = fake_generate
        mock_load.return_value = fake_model

        fake_stream = MagicMock()
        def write_side_effect(data):
            locked_during_write.append(lock.locked())
            for call_args in mock_sd.OutputStream.call_args_list:
                cb = call_args[1].get("finished_callback")
                if cb:
                    cb()
        fake_stream.write = MagicMock(side_effect=write_side_effect)
        mock_sd.OutputStream.return_value = fake_stream

        from spoke.tts import TTSClient
        client = TTSClient(gpu_lock=lock)
        client.speak("test")

        assert locked_during_generate == [True], "Lock must be held during generate()"
        assert any(not locked for locked in locked_during_write), "Lock must be released during audio write"

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_warm_holds_lock(self, mock_load, mock_sd):
        """warm() acquires the GPU lock during model loading."""
        lock = threading.Lock()
        locked_during_load = []

        original_load = mock_load.side_effect
        def capture_lock(model_id):
            locked_during_load.append(lock.locked())
            return MagicMock()
        mock_load.side_effect = capture_lock

        from spoke.tts import TTSClient
        client = TTSClient(gpu_lock=lock)
        client.warm()
        # Wait for the background thread
        import time
        time.sleep(0.5)

        assert locked_during_load == [True], "Lock must be held during model loading"


class TestAmplitudeCallback:
    """Verify amplitude callback is invoked per chunk during playback."""

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_amplitude_callback_called_per_chunk(self, mock_load, mock_sd):
        """amplitude_callback receives RMS values for each chunk, plus final 0.0."""
        import numpy as np

        # Create audio long enough for multiple chunks (24kHz, ~200ms = 4800 samples)
        audio_data = np.random.randn(4800).astype(np.float32) * 0.1
        fake_model = MagicMock()
        r = MagicMock()
        r.audio = audio_data.tolist()
        r.sample_rate = 24000
        fake_model.generate.return_value = iter([r])
        mock_load.return_value = fake_model

        # Stream mock that triggers finished after all writes
        write_count = [0]
        # 64ms chunks at 24kHz = 1536 samples, so 4800 / 1536 ≈ 4 chunks
        expected_chunks = -(-4800 // int(24000 * 0.064))  # ceiling division

        fake_stream = MagicMock()
        def write_side_effect(data):
            write_count[0] += 1
            if write_count[0] >= expected_chunks:
                for call_args in mock_sd.OutputStream.call_args_list:
                    cb = call_args[1].get("finished_callback")
                    if cb:
                        cb()
        fake_stream.write = MagicMock(side_effect=write_side_effect)
        mock_sd.OutputStream.return_value = fake_stream

        rms_values = []
        def capture_rms(rms):
            rms_values.append(rms)

        from spoke.tts import TTSClient
        client = TTSClient()
        client.speak("test", amplitude_callback=capture_rms)

        # Should have one RMS per chunk + final 0.0
        assert len(rms_values) >= expected_chunks + 1
        assert rms_values[-1] == 0.0, "Final callback should be 0.0"
        assert all(isinstance(v, float) for v in rms_values)
        assert all(v > 0 for v in rms_values[:-1]), "Non-final RMS values should be positive"


class TestCancelDuringPlayback:
    """Verify cancel during active playback triggers fade-out and cleanup."""

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_cancel_during_write_loop_stops_and_cleans_up(self, mock_load, mock_sd):
        """Cancelling mid-playback stops the write loop and closes the stream."""
        import numpy as np

        audio_data = np.random.randn(48000).astype(np.float32) * 0.1  # 2s of audio
        fake_model = MagicMock()
        r = MagicMock()
        r.audio = audio_data.tolist()
        r.sample_rate = 24000
        fake_model.generate.return_value = iter([r])
        mock_load.return_value = fake_model

        from spoke.tts import TTSClient
        client = TTSClient()

        fake_stream = MagicMock()
        chunks_written = [0]
        def write_side_effect(data):
            chunks_written[0] += 1
            # Cancel directly from inside the write callback on chunk 3
            if chunks_written[0] == 3:
                client.cancel()
        fake_stream.write = MagicMock(side_effect=write_side_effect)
        mock_sd.OutputStream.return_value = fake_stream

        client.speak("test")

        fake_stream.stop.assert_called_once()
        fake_stream.close.assert_called_once()
        # Chunk 3 sets cancel, chunk 4 sees it and breaks (+1 for fade ramp)
        # Should be well under the full ~32 chunks
        total_chunks = -(-48000 // int(24000 * 0.064))
        assert chunks_written[0] <= 5, f"Expected ≤5 writes (3 + cancel check + fade), got {chunks_written[0]}"


class TestDoneCallback:
    """Verify done_callback fires after speak completes."""

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_done_callback_fires_after_speak(self, mock_load, mock_sd):
        """done_callback is invoked after speak() returns."""
        fake_model = MagicMock()
        fake_model.generate.return_value = iter([_fake_result()])
        mock_load.return_value = fake_model
        _setup_stream_mock(mock_sd)

        done = threading.Event()

        from spoke.tts import TTSClient
        client = TTSClient()
        client.speak_async("test", done_callback=lambda: done.set())

        assert done.wait(timeout=5), "done_callback should have been called"

    @patch("spoke.tts.sd")
    @patch("spoke.tts.tts_load")
    def test_speak_async_resets_cancel_flag(self, mock_load, mock_sd):
        """speak_async() clears cancelled flag so new playback proceeds."""
        fake_model = MagicMock()
        fake_model.generate.return_value = iter([_fake_result()])
        mock_load.return_value = fake_model
        _setup_stream_mock(mock_sd)

        from spoke.tts import TTSClient
        client = TTSClient()
        client.cancel()
        assert client._cancelled is True

        t = client.speak_async("test")
        t.join(timeout=5)

        mock_sd.OutputStream.assert_called_once()


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

        tts.speak_async.assert_called_once()
        args, kwargs = tts.speak_async.call_args
        assert args[0] == "Hello there"
        assert kwargs.get("amplitude_callback") is not None
        assert kwargs.get("done_callback") is not None

    def test_command_complete_no_tts_when_disabled(self, main_module):
        """When TTS client is None, commandComplete_ works normally without TTS."""
        delegate = self._make_delegate(main_module, tts_client=None)

        # Should not raise
        delegate.commandComplete_({"token": 1, "response": "Hello"})

        delegate._command_overlay.finish.assert_called_once()

    def test_hold_start_cancels_tts(self, main_module):
        """Starting a new hold should interrupt active TTS before reopening input."""
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

        tts.cancel.assert_called_once_with()

    def test_idle_shift_tap_toggles_audio(self, main_module):
        """Idle shift tap should route to the TTS audio toggle callback."""
        tts = MagicMock()
        tts.toggle_audio.return_value = False
        delegate = self._make_delegate(main_module, tts_client=tts)
        delegate._tray_active = False

        delegate._on_audio_shift_tap()

        tts.toggle_audio.assert_called_once_with()
        delegate._menubar.set_status_text.assert_called_with("Audio muted")
