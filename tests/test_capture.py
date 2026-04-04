"""Tests for audio capture and WAV encoding.

These tests don't need PyObjC mocking — capture.py only depends on
sounddevice and numpy, and we mock sounddevice at the function level.
"""

import io
import queue
import struct
import threading
import time
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spoke.capture import AudioCapture, SAMPLE_RATE, CHANNELS, SILERO_CHUNK


class _FakeSileroVAD:
    """Mock Silero VAD that classifies based on RMS amplitude."""

    def __call__(self, tensor, sr):
        import torch
        rms = float(tensor.abs().mean())
        prob = 0.9 if rms > 0.01 else 0.05
        return torch.tensor(prob)

    def reset_states(self):
        pass


@pytest.fixture(autouse=True)
def _mock_silero_vad(monkeypatch):
    """Patch AudioCapture to use a fake Silero model for all tests."""
    import torch
    original_init = AudioCapture.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._silero_model = _FakeSileroVAD()
        self._silero_sr = torch.tensor(SAMPLE_RATE)
        self._torch = torch

    monkeypatch.setattr(AudioCapture, "__init__", patched_init)


class TestWavEncoding:
    """Test the WAV byte encoding from float32 samples."""

    def test_encode_produces_valid_wav(self):
        """Generated WAV bytes should be parseable by stdlib wave module."""
        cap = AudioCapture()
        # 1 second of silence
        samples = np.zeros(SAMPLE_RATE, dtype=np.float32)
        wav_bytes = cap._encode_wav(samples)

        assert len(wav_bytes) > 44  # WAV header is 44 bytes

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == CHANNELS
            assert wf.getsampwidth() == 2  # 16-bit
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getnframes() == SAMPLE_RATE

    def test_encode_preserves_amplitude(self):
        """A full-scale sine wave should survive float32 → int16 conversion."""
        cap = AudioCapture()
        t = np.linspace(0, 1, SAMPLE_RATE, dtype=np.float32)
        samples = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine
        wav_bytes = cap._encode_wav(samples)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(raw, dtype=np.int16)
            # Peak should be close to 32767
            assert pcm.max() > 32000
            assert pcm.min() < -32000

    def test_encode_empty_returns_empty_bytes(self):
        """Empty sample array should return empty bytes."""
        cap = AudioCapture()
        result = cap._encode_wav(np.array([], dtype=np.float32))
        assert result == b""

    def test_encode_clips_out_of_range(self):
        """Values outside [-1, 1] should be clipped, not wrap."""
        cap = AudioCapture()
        samples = np.array([2.0, -2.0, 0.5], dtype=np.float32)
        wav_bytes = cap._encode_wav(samples)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(raw, dtype=np.int16)
            assert pcm[0] == 32767   # clipped to max
            assert pcm[1] == -32768  # clipped to min


class TestAudioCallback:
    """Test the audio callback's frame accumulation and amplitude reporting."""

    @patch("spoke.capture.sd")
    def test_callback_accumulates_frames(self, mock_sd):
        """Audio callback should append chunks to the frame list."""
        cap = AudioCapture()
        cap._stream = mock_sd.InputStream.return_value
        chunk = np.random.randn(1024, 1).astype(np.float32)

        cap._audio_callback(chunk, 1024, None, 0)
        cap._audio_callback(chunk, 1024, None, 0)

        assert len(cap._frames) == 2

    @patch("spoke.capture.sd")
    def test_callback_reports_amplitude(self, mock_sd):
        """Amplitude callback should receive RMS of each chunk."""
        amplitudes = []
        cap = AudioCapture()
        cap.start(amplitude_callback=amplitudes.append)
        cap._stream = mock_sd.InputStream.return_value
        cap._stream.active = True

        # Known signal: constant 0.5
        chunk = np.full((1024, 1), 0.5, dtype=np.float32)
        cap._audio_callback(chunk, 1024, None, 0)  # no callback set yet
        cap._audio_callback(chunk, 1024, None, 0)

        deadline = time.time() + 1.0
        while len(amplitudes) < 2 and time.time() < deadline:
            time.sleep(0.01)

        assert len(amplitudes) >= 1
        assert abs(amplitudes[-1] - 0.5) < 0.01  # RMS of constant 0.5 = 0.5

        cap.stop()

    def test_callback_ignores_stale_invocation_after_stream_teardown(self):
        """Late callbacks after teardown should be ignored."""
        cap = AudioCapture()
        cap._frames = []
        cap._amplitude_cb = MagicMock()

        chunk = np.full((1024, 1), 0.5, dtype=np.float32)
        cap._audio_callback(chunk, 1024, None, 0)

        assert cap._frames == []
        cap._amplitude_cb.assert_not_called()

    def test_get_all_frames_concatenates(self):
        """_get_all_frames should return one contiguous array."""
        cap = AudioCapture()
        cap._frames = [
            np.ones(100, dtype=np.float32),
            np.zeros(200, dtype=np.float32),
        ]
        result = cap._get_all_frames()
        assert result.shape == (300,)
        assert result[:100].sum() == 100.0
        assert result[100:].sum() == 0.0

    def test_get_all_frames_empty(self):
        """Empty frame list should return empty array."""
        cap = AudioCapture()
        result = cap._get_all_frames()
        assert result.size == 0

    @patch("spoke.capture.sd")
    def test_callback_dispatches_amplitude_off_audio_thread(self, mock_sd):
        """Amplitude callbacks should run off the PortAudio callback thread."""
        cap = AudioCapture()
        seen = []
        caller_tid = None
        delivered = threading.Event()

        def amplitude_cb(rms):
            seen.append((threading.get_ident(), rms))
            delivered.set()

        cap.start(amplitude_callback=amplitude_cb)
        cap._stream = mock_sd.InputStream.return_value
        cap._stream.active = True

        chunk = np.full((1024, 1), 0.5, dtype=np.float32)
        caller_tid = threading.get_ident()
        cap._audio_callback(chunk, 1024, None, 0)

        assert delivered.wait(timeout=1.0)
        callback_tid, rms = seen[-1]
        assert callback_tid != caller_tid
        assert abs(rms - 0.5) < 0.01

        cap.stop()

    @patch("spoke.capture.sd")
    def test_callback_dispatches_vad_state_off_audio_thread(self, mock_sd):
        """VAD state callbacks should not run on the PortAudio callback thread."""
        cap = AudioCapture()
        seen = []
        delivered = threading.Event()

        def vad_cb(is_speech):
            seen.append((threading.get_ident(), is_speech))
            delivered.set()

        cap.start(vad_state_callback=vad_cb)
        cap._stream = mock_sd.InputStream.return_value
        cap._stream.active = True

        silence_chunk = np.zeros((1024, 1), dtype=np.float32)
        for _ in range(50):
            cap._audio_callback(silence_chunk, 1024, None, 0)

        caller_tid = threading.get_ident()
        speech_chunk = np.full((1024, 1), 0.5, dtype=np.float32)
        for _ in range(3):
            cap._audio_callback(speech_chunk, 1024, None, 0)

        assert delivered.wait(timeout=1.0)
        callback_tid, is_speech = seen[-1]
        assert callback_tid != caller_tid
        assert is_speech is True

        cap.stop()

    @patch("spoke.capture.sd")
    def test_callback_does_not_block_on_slow_amplitude_handler(self, mock_sd):
        """The PortAudio callback path should not wait on a slow UI callback."""
        cap = AudioCapture()
        release = threading.Event()
        entered = threading.Event()

        def amplitude_cb(rms):
            entered.set()
            release.wait(timeout=1.0)

        cap.start(amplitude_callback=amplitude_cb)
        cap._stream = mock_sd.InputStream.return_value
        cap._stream.active = True

        chunk = np.full((1024, 1), 0.5, dtype=np.float32)
        started = time.perf_counter()
        cap._audio_callback(chunk, 1024, None, 0)
        elapsed = time.perf_counter() - started

        assert elapsed < 0.05
        assert entered.wait(timeout=1.0)

        release.set()
        cap.stop()


class TestStartStop:
    """Test start/stop lifecycle with mocked sounddevice."""

    @patch("spoke.capture.sd")
    def test_start_creates_stream(self, mock_sd):
        """start() should create and start an InputStream."""
        cap = AudioCapture()
        cap.start()
        mock_sd.InputStream.assert_called_once()
        mock_sd.InputStream.return_value.start.assert_called_once()

    @patch("spoke.capture.sd")
    def test_stop_returns_wav_bytes(self, mock_sd):
        """stop() after accumulating frames should return valid WAV."""
        cap = AudioCapture()
        cap._stream = mock_sd.InputStream.return_value
        cap._stream.active = True

        # Simulate some recorded audio
        cap._frames = [np.zeros(1024, dtype=np.float32)]
        wav = cap.stop()
        assert len(wav) > 44  # WAV header + data

    @patch("spoke.capture.sd")
    def test_stop_with_no_frames_returns_empty(self, mock_sd):
        """stop() with no recorded frames should return empty bytes."""
        cap = AudioCapture()
        cap._stream = mock_sd.InputStream.return_value
        wav = cap.stop()
        assert wav == b""

    @patch("spoke.capture.sd")
    def test_get_buffer_nondestructive(self, mock_sd):
        """get_buffer() should not consume frames."""
        cap = AudioCapture()
        cap._frames = [np.zeros(1024, dtype=np.float32)]

        buf1 = cap.get_buffer()
        buf2 = cap.get_buffer()
        assert buf1 == buf2
        assert len(cap._frames) == 1  # frames not consumed

    @patch("spoke.capture.sd")
    def test_double_start_stops_previous_stream(self, mock_sd):
        """Calling start() twice should stop and close the first stream."""
        cap = AudioCapture()
        cap.start()
        first_stream = mock_sd.InputStream.return_value

        # Create a distinct mock for the second stream
        second_stream = MagicMock()
        mock_sd.InputStream.return_value = second_stream
        cap.start()

        first_stream.stop.assert_called_once()
        first_stream.close.assert_called_once()

    @patch("spoke.capture.sd")
    @patch("spoke.capture.threading.Thread")
    def test_second_start_tears_down_callback_dispatch_before_rebinding(
        self, MockThread, mock_sd
    ):
        """Stale callback work from recording N must not hit recording N+1 callbacks."""
        cap = AudioCapture()
        old_cb = MagicMock()
        new_cb = MagicMock()

        cap.start(amplitude_callback=old_cb)
        old_generation = cap._callback_generation

        def simulate_old_dispatch_teardown():
            assert cap._amplitude_cb is old_cb
            cap._amplitude_cb(0.5)
            cap._callback_queue = None
            cap._callback_thread = None

        with patch.object(cap, "_stop_callback_dispatch", side_effect=simulate_old_dispatch_teardown) as mock_stop:
            cap.start(amplitude_callback=new_cb)

        mock_stop.assert_called_once()
        old_cb.assert_called_once_with(0.5)
        new_cb.assert_not_called()
        assert cap._callback_generation == old_generation + 1

    @patch("spoke.capture.sd")
    def test_start_retries_once_after_portaudio_error(self, mock_sd):
        """A dead backend on restart should get one PortAudio reset and retry."""
        cap = AudioCapture()
        first_stream = MagicMock()
        first_stream.start.side_effect = RuntimeError("device failed")
        second_stream = MagicMock()
        mock_sd.InputStream.side_effect = [first_stream, second_stream]

        cap.start()

        assert mock_sd.InputStream.call_count == 2
        first_stream.close.assert_called_once()
        mock_sd._terminate.assert_called_once()
        mock_sd._initialize.assert_called_once()
        second_stream.start.assert_called_once()
        assert cap._stream is second_stream

    @patch("spoke.capture.sd")
    def test_stop_resets_portaudio_after_zero_chunk_recording(self, mock_sd):
        """A zero-chunk recording should reset PortAudio before the next hold."""
        cap = AudioCapture()
        cap._stream = MagicMock()

        wav = cap.stop()

        assert wav == b""
        mock_sd._terminate.assert_called_once()
        mock_sd._initialize.assert_called_once()
        assert cap._stream is None

    @patch("spoke.capture.sd")
    def test_get_new_frames_returns_incremental(self, mock_sd):
        """get_new_frames() should return only frames since last call."""
        cap = AudioCapture()
        chunk1 = np.ones(1024, dtype=np.float32)
        chunk2 = np.ones(1024, dtype=np.float32) * 2

        cap._frames = [chunk1]
        frames1 = cap.get_new_frames()
        assert len(frames1) == 1024
        assert frames1[0] == 1.0

        # Second call with no new frames
        frames_empty = cap.get_new_frames()
        assert frames_empty.size == 0

        # Add another chunk
        cap._frames.append(chunk2)
        frames2 = cap.get_new_frames()
        assert len(frames2) == 1024
        assert frames2[0] == 2.0

    @patch("spoke.capture.sd")
    def test_get_new_frames_cursor_resets_on_start(self, mock_sd):
        """start() should reset the read cursor."""
        cap = AudioCapture()
        cap._frames = [np.ones(1024, dtype=np.float32)]
        cap.get_new_frames()  # advance cursor
        assert cap._read_cursor == 1

        cap.start()
        assert cap._read_cursor == 0

    @patch("spoke.capture.sd")
    def test_stop_without_start_clears_stale_frames(self, mock_sd):
        """stop() when stream is None should clear leftover frames and return
        empty bytes, not stale audio from a previous session."""
        cap = AudioCapture()

        # Simulate leftover frames from a previous recording
        cap._frames = [np.ones(1024, dtype=np.float32)]

        # stop() without start() — stream is None
        wav = cap.stop()
        assert wav == b""
        assert len(cap._frames) == 0

    @patch("spoke.capture.sd")
    def test_stop_clears_callback_dispatch_state(self, mock_sd):
        """stop() should tear down async callback dispatch before the next hold."""
        cap = AudioCapture()
        cap.start(
            amplitude_callback=MagicMock(),
            vad_state_callback=MagicMock(),
        )
        cap._stream = mock_sd.InputStream.return_value
        cap._stream.active = True
        cap._frames = [np.zeros(1024, dtype=np.float32)]

        cap.stop()

        assert cap._callback_thread is None
        assert cap._callback_queue is None
        assert cap._amplitude_cb is None
        assert cap._vad_cb is None

    @patch("spoke.capture.sd")
    def test_stop_disables_callbacks_before_dispatch_teardown(self, mock_sd):
        """Queued callback work should be disabled before dispatch teardown runs."""
        cap = AudioCapture()
        cap.start(
            amplitude_callback=MagicMock(),
            vad_state_callback=MagicMock(),
        )
        cap._stream = mock_sd.InputStream.return_value
        cap._stream.active = True
        cap._frames = [np.zeros(1024, dtype=np.float32)]

        def assert_callbacks_already_disabled():
            assert cap._amplitude_cb is None
            assert cap._vad_cb is None

        with patch.object(cap, "_stop_callback_dispatch", side_effect=assert_callbacks_already_disabled):
            cap.stop()

    def test_callback_worker_drops_stale_generation_events(self):
        """Events from an older recording must not hit the new recording callbacks."""
        cap = AudioCapture()
        cap._callback_generation = 2
        cap._callbacks_enabled = True
        cap._amplitude_cb = MagicMock()
        cap._callback_queue = queue.Queue()
        cap._callback_queue.put((1, "amplitude", 0.5))
        cap._callback_queue.put(None)

        cap._callback_worker()

        cap._amplitude_cb.assert_not_called()

    @patch("spoke.capture.sd")
    def test_stop_drops_pending_callback_events(self, mock_sd):
        """stop() should drop queued callback work rather than delivering it."""
        cap = AudioCapture()
        callback = MagicMock()
        cap.start(amplitude_callback=callback)
        cap._stream = mock_sd.InputStream.return_value
        cap._stream.active = True
        cap._frames = [np.zeros(1024, dtype=np.float32)]

        cap._callback_queue = queue.Queue()
        cap._callback_queue.put((cap._callback_generation, "amplitude", 0.5))
        cap._callback_thread = None

        cap.stop()

        callback.assert_not_called()

class TestVADSlicing:
    """Test the VAD state machine and silence slicing logic."""

    @patch("spoke.capture.VAD_GRACE_PERIOD_SECS", 0.0)
    @patch("spoke.capture.sd")
    def test_vad_no_slice_on_continuous_speech(self, mock_sd):
        """Continuous speech without sufficient silence should not trigger slicing."""
        cap = AudioCapture()
        cap.start(segment_callback=MagicMock())
        
        # Prime the noise floor with silence
        silence_chunk = np.zeros((1024, 1), dtype=np.float32)
        for _ in range(50):
            cap._audio_callback(silence_chunk, 1024, None, 0)
            
        # Simulate continuous speech chunks (high RMS)
        speech_chunk = np.full((1024, 1), 0.5, dtype=np.float32)
        for _ in range(10):
            cap._audio_callback(speech_chunk, 1024, None, 0)
            
        cap._segment_cb.assert_not_called()
        assert cap._is_speech

    @patch("spoke.capture.VAD_GRACE_PERIOD_SECS", 0.0)
    @patch("spoke.capture.sd")
    def test_vad_slices_after_silence(self, mock_sd):
        """Speech followed by silence should emit a bounded segment."""
        cap = AudioCapture()
        mock_cb = MagicMock()
        cap.start(segment_callback=mock_cb)
        
        silence_chunk = np.zeros((1024, 1), dtype=np.float32)
        for _ in range(50):
            cap._audio_callback(silence_chunk, 1024, None, 0)
            
        # Simulate speech
        speech_chunk = np.full((1024, 1), 0.5, dtype=np.float32)
        for _ in range(5):
            cap._audio_callback(speech_chunk, 1024, None, 0)
            
        assert cap._is_speech
        
        # Simulate silence
        for _ in range(12):  # MIN_SILENCE_FRAMES
            cap._audio_callback(silence_chunk, 1024, None, 0)
            
        assert not cap._is_speech
        cap.stop()
        
        mock_cb.assert_called_once()
        
        # The argument should be wav bytes
        wav_bytes = mock_cb.call_args[0][0]
        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 44

    @patch("spoke.capture.VAD_GRACE_PERIOD_SECS", 0.0)
    @patch("spoke.capture.sd")
    def test_vad_stop_emits_final_segment(self, mock_sd):
        """Stopping capture while in speech should emit the final segment."""
        cap = AudioCapture()
        mock_segment_cb = MagicMock()
        cap.start(segment_callback=mock_segment_cb)
        cap._stream = mock_sd.InputStream.return_value
        cap._stream.active = True
        
        silence_chunk = np.zeros((1024, 1), dtype=np.float32)
        for _ in range(50):
            cap._audio_callback(silence_chunk, 1024, None, 0)
            
        speech_chunk = np.full((1024, 1), 0.5, dtype=np.float32)
        for _ in range(5):
            cap._audio_callback(speech_chunk, 1024, None, 0)
            
        assert cap._is_speech
        
        cap.stop()
        mock_segment_cb.assert_called_once()

    @patch("spoke.capture.VAD_GRACE_PERIOD_SECS", 0.0)
    @patch("spoke.capture.sd")
    def test_vad_strips_silence_from_final_wav(self, mock_sd):
        """stop() should return only the concatenated speech chunks, stripping leading and trailing silence."""
        cap = AudioCapture()
        cap.start(segment_callback=MagicMock())
        cap._stream = mock_sd.InputStream.return_value
        cap._stream.active = True
        
        silence_chunk = np.zeros((1024, 1), dtype=np.float32)
        for _ in range(50):
            cap._audio_callback(silence_chunk, 1024, None, 0)
            
        speech_chunk = np.full((1024, 1), 0.5, dtype=np.float32)
        for _ in range(10):
            cap._audio_callback(speech_chunk, 1024, None, 0)
            
        for _ in range(20):  # Long silence to force slice
            cap._audio_callback(silence_chunk, 1024, None, 0)
            
        wav_bytes = cap.stop()
        
        # Check the length of the WAV to ensure silence was stripped
        # 10 speech chunks + PRE_SPEECH_MARGIN (6) = 16 chunks total
        # 16 * 1024 floats = 16384 samples. Each sample is 2 bytes (int16).
        # 16384 * 2 = 32768 bytes of audio data + 44 bytes header = 32812 bytes
        # The test verifies we didn't include the 70 frames of pure silence.
        assert len(wav_bytes) == 51244
