"""Tests for audio capture and WAV encoding.

These tests don't need PyObjC mocking — capture.py only depends on
sounddevice and numpy, and we mock sounddevice at the function level.
"""

import io
import struct
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from donttalk.capture import AudioCapture, SAMPLE_RATE, CHANNELS


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

    def test_callback_accumulates_frames(self):
        """Audio callback should append chunks to the frame list."""
        cap = AudioCapture()
        chunk = np.random.randn(1024, 1).astype(np.float32)

        cap._audio_callback(chunk, 1024, None, 0)
        cap._audio_callback(chunk, 1024, None, 0)

        assert len(cap._frames) == 2

    def test_callback_reports_amplitude(self):
        """Amplitude callback should receive RMS of each chunk."""
        amplitudes = []
        cap = AudioCapture()

        # Known signal: constant 0.5
        chunk = np.full((1024, 1), 0.5, dtype=np.float32)
        cap._audio_callback(chunk, 1024, None, 0)  # no callback set yet

        cap._amplitude_cb = amplitudes.append
        cap._audio_callback(chunk, 1024, None, 0)

        assert len(amplitudes) == 1
        assert abs(amplitudes[0] - 0.5) < 0.01  # RMS of constant 0.5 = 0.5

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


class TestStartStop:
    """Test start/stop lifecycle with mocked sounddevice."""

    @patch("donttalk.capture.sd")
    def test_start_creates_stream(self, mock_sd):
        """start() should create and start an InputStream."""
        cap = AudioCapture()
        cap.start()
        mock_sd.InputStream.assert_called_once()
        mock_sd.InputStream.return_value.start.assert_called_once()

    @patch("donttalk.capture.sd")
    def test_stop_returns_wav_bytes(self, mock_sd):
        """stop() after accumulating frames should return valid WAV."""
        cap = AudioCapture()
        cap._stream = mock_sd.InputStream.return_value
        cap._stream.active = True

        # Simulate some recorded audio
        cap._frames = [np.zeros(1024, dtype=np.float32)]
        wav = cap.stop()
        assert len(wav) > 44  # WAV header + data

    @patch("donttalk.capture.sd")
    def test_stop_with_no_frames_returns_empty(self, mock_sd):
        """stop() with no recorded frames should return empty bytes."""
        cap = AudioCapture()
        cap._stream = mock_sd.InputStream.return_value
        wav = cap.stop()
        assert wav == b""

    @patch("donttalk.capture.sd")
    def test_get_buffer_nondestructive(self, mock_sd):
        """get_buffer() should not consume frames."""
        cap = AudioCapture()
        cap._frames = [np.zeros(1024, dtype=np.float32)]

        buf1 = cap.get_buffer()
        buf2 = cap.get_buffer()
        assert buf1 == buf2
        assert len(cap._frames) == 1  # frames not consumed

    @patch("donttalk.capture.sd")
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

    @patch("donttalk.capture.sd")
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
