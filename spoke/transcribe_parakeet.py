"""Parakeet CTC-110M CoreML transcription client.

Runs NVIDIA's Parakeet CTC-110M on Apple Neural Engine via CoreML.
~2W power draw, 0.40s for 3 minutes of speech on M4 Max (457x RTF).

Quality note: CTC output has no reliable punctuation and rough accuracy
compared to Whisper. Suitable for fast preview partials only, not finals.

Model: FluidInference/parakeet-ctc-110m-coreml (~436 MB)
  AudioEncoder.mlmodelc — encoder + CTC head, output: ctc_head_output [1,1025,1,188]
  Input: melspectrogram_features [1,1,1501,80], input_1 [1,1,1,1] (must be 0.0)
  Max 15 seconds per inference pass; longer audio is chunked.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import wave

import numpy as np

from .dedup import truncate_repetition, is_hallucination

logger = logging.getLogger(__name__)

_PARAKEET_MODEL_ID = "FluidInference/parakeet-ctc-110m-coreml"

# AudioEncoder fixed dimensions
_MEL_N_MELS = 80
_MEL_N_FRAMES = 1501          # 15.01s at 10ms hop
_MEL_HOP_MS = 10              # ms per frame
_MEL_WIN_LENGTH = 400          # 25ms window at 16kHz
_MEL_HOP_LENGTH = 160          # 10ms hop at 16kHz
_SAMPLE_RATE = 16000
_MAX_CHUNK_SECS = 15.0         # model hard limit: 1501 mel frames


# ── Public helpers exposed for testing ──────────────────────────────────────


def _load_vocab(vocab_path: Path) -> dict[int, str]:
    """Load vocab.json {str(int) -> token} into {int -> token}."""
    with open(vocab_path) as f:
        raw: dict[str, str] = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _ctc_greedy_decode(token_ids: list[int], vocab: dict[int, str], blank_id: int) -> str:
    """Greedy CTC decoder: collapse repeats, remove blanks, join SentencePiece tokens."""
    # Collapse repeated tokens
    collapsed: list[int] = []
    prev = None
    for t in token_ids:
        if t != prev:
            collapsed.append(t)
            prev = t

    # Remove blanks and map to strings
    tokens = [vocab.get(t, "") for t in collapsed if t != blank_id]

    # Join SentencePiece tokens: ▁ marks a word boundary (leading space)
    text = "".join(tok.replace("▁", " ") for tok in tokens).strip()
    return text


def _decode_wav_to_float32(wav_bytes: bytes) -> np.ndarray:
    """Decode 16-bit mono 16kHz WAV bytes to float32 in [-1, 1]."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono audio, got {wf.getnchannels()} channels")
        if wf.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit audio, got {wf.getsampwidth() * 8}-bit")
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _chunk_audio(
    samples: np.ndarray, sample_rate: int = _SAMPLE_RATE, max_chunk_secs: float = _MAX_CHUNK_SECS
) -> list[np.ndarray]:
    """Split audio into chunks of at most max_chunk_secs seconds."""
    chunk_len = int(max_chunk_secs * sample_rate)
    if len(samples) <= chunk_len:
        return [samples]
    chunks = []
    offset = 0
    while offset < len(samples):
        chunks.append(samples[offset : offset + chunk_len])
        offset += chunk_len
    return chunks


def _compute_mel_features(samples: np.ndarray) -> np.ndarray:
    """Compute 80-band log-mel spectrogram matching the AudioEncoder input spec.

    Returns float16 array shaped [1, 1, 1501, 80].
    """
    n_frames = _MEL_N_FRAMES
    n_mels = _MEL_N_MELS
    hop = _MEL_HOP_LENGTH
    win = _MEL_WIN_LENGTH
    sr = _SAMPLE_RATE

    # Pad/trim to exactly n_frames worth of hops
    required_samples = (n_frames - 1) * hop + win
    if len(samples) < required_samples:
        samples = np.pad(samples, (0, required_samples - len(samples)))
    else:
        samples = samples[:required_samples]

    # Build mel filterbank
    mel_fb = _mel_filterbank(sr=sr, n_fft=win, n_mels=n_mels, fmin=0.0, fmax=sr / 2)

    # STFT frames
    n_frames_actual = 1 + (len(samples) - win) // hop
    mel_spec = np.zeros((n_mels, n_frames_actual), dtype=np.float32)
    window = np.hanning(win).astype(np.float32)

    for i in range(n_frames_actual):
        start = i * hop
        frame = samples[start : start + win] * window
        spectrum = np.abs(np.fft.rfft(frame, n=win)) ** 2
        # mel_fb is [n_mels, n_fft//2+1]
        mel_spec[:, i] = mel_fb @ spectrum

    # Log mel with floor
    mel_spec = np.log(np.maximum(mel_spec, 1e-10))

    # Normalise to zero mean unit variance (per-utterance, Kaldi-style)
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

    # Pad time axis to exactly n_frames
    if mel_spec.shape[1] < n_frames:
        pad = n_frames - mel_spec.shape[1]
        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad)))
    else:
        mel_spec = mel_spec[:, :n_frames]

    # Reshape to [1, 1, n_frames, n_mels] and cast to float16
    out = mel_spec.T[np.newaxis, np.newaxis, :, :].astype(np.float16)
    return out


def _mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """HTK-style mel filterbank, returns [n_mels, n_fft//2+1]."""
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    freq_bins = np.linspace(0, sr / 2, n_freqs)
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])

    fbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        lo, center, hi = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        for k in range(n_freqs):
            f = freq_bins[k]
            if lo <= f <= center:
                fbank[m, k] = (f - lo) / (center - lo) if center > lo else 0.0
            elif center < f <= hi:
                fbank[m, k] = (hi - f) / (hi - center) if hi > center else 0.0
    return fbank


# ── Client ──────────────────────────────────────────────────────────────────


class ParakeetCoreMLClient:
    """Transcribe audio using Parakeet CTC-110M via CoreML / Apple Neural Engine.

    Parameters
    ----------
    model_dir : Path
        Directory containing AudioEncoder.mlmodelc, vocab.json, and
        ctc_head_metadata.json.  Typically the HF repo clone directory.
    """

    def __init__(self, model_dir: Path, *, model_id: str = _PARAKEET_MODEL_ID) -> None:
        self._model_dir = model_dir
        self._model_id = model_id
        self._encoder = None
        self._vocab: dict[int, str] = {}
        self._blank_id: int = 1024
        self._loaded = False

    def prepare(self) -> None:
        """Load the CoreML model and vocabulary."""
        if self._loaded:
            return
        try:
            import coremltools as ct
        except ImportError:
            logger.warning("coremltools not installed — Parakeet CoreML unavailable")
            return

        encoder_path = self._model_dir / "AudioEncoder.mlmodelc"
        if not encoder_path.exists():
            logger.warning("Parakeet model not found at %s", encoder_path)
            return

        logger.info("Loading Parakeet CoreML encoder from %s", encoder_path)
        self._encoder = ct.models.CompiledMLModel(str(encoder_path))

        vocab_path = self._model_dir / "vocab.json"
        if vocab_path.exists():
            self._vocab = _load_vocab(vocab_path)
        else:
            logger.warning("Parakeet vocab.json not found at %s", vocab_path)

        meta_path = self._model_dir / "ctc_head_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._blank_id = meta.get("blank_id", 1024)

        self._loaded = True
        logger.info("Parakeet CoreML encoder ready (blank_id=%d)", self._blank_id)

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV bytes using Parakeet CTC on ANE.

        Long audio is chunked into ≤15s segments and their transcriptions joined.
        """
        if not wav_bytes:
            return ""

        if not self._loaded:
            self.prepare()

        if self._encoder is None:
            logger.warning("Parakeet encoder not available — returning empty string")
            return ""

        try:
            samples = _decode_wav_to_float32(wav_bytes)
        except Exception:
            logger.warning("Parakeet: failed to decode WAV", exc_info=True)
            return ""

        chunks = _chunk_audio(samples)
        parts: list[str] = []
        for chunk in chunks:
            part = self._transcribe_chunk(chunk)
            if part:
                parts.append(part)

        text = " ".join(parts).strip()
        text = truncate_repetition(text)
        if is_hallucination(text):
            logger.info("Parakeet: discarding hallucination: %r", text)
            return ""
        if text:
            logger.info("Parakeet transcription: %r (%d bytes audio)", text, len(wav_bytes))
        return text

    def _transcribe_chunk(self, samples: np.ndarray) -> str:
        """Run one ≤15s chunk through the encoder and decode."""
        mel = _compute_mel_features(samples)

        # input_1 must be 0.0 (not the time-step count)
        input_1 = np.zeros((1, 1, 1, 1), dtype=np.float16)

        try:
            out = self._encoder.predict(
                {"melspectrogram_features": mel, "input_1": input_1}
            )
        except Exception:
            logger.warning("Parakeet: encoder prediction failed", exc_info=True)
            return ""

        # ctc_head_output: [1, 1025, 1, 188]
        logits = out.get("ctc_head_output")
        if logits is None:
            logger.warning("Parakeet: 'ctc_head_output' missing from model output")
            return ""

        logits = np.array(logits, dtype=np.float32)
        # Shape [1, vocab+1, 1, time] -> [time, vocab+1]
        if logits.ndim == 4:
            logits = logits[0, :, 0, :].T  # [time, vocab+1]

        token_ids = logits.argmax(axis=-1).tolist()
        return _ctc_greedy_decode(token_ids, self._vocab, self._blank_id)

    def close(self) -> None:
        """No persistent resources to clean up."""
        pass
