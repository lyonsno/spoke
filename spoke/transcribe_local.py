"""Local MLX Whisper transcription — no server required.

Runs mlx-whisper in-process on the local machine. Downloads the model
from HuggingFace on first use. Decodes WAV in-process via numpy to
avoid requiring ffmpeg.
"""

from __future__ import annotations

import importlib
import io
import logging
import os
from pathlib import Path
import threading
import wave

import numpy as np

from .dedup import truncate_repetition, is_hallucination, repair_ontology_terms

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
_DEFAULT_DECODE_TIMEOUT = 30.0
_DEFAULT_EAGER_EVAL = False

mx = None
mlx_whisper = None
load_model = None


class _DecodeTimeoutCapture(logging.Handler):
    """Capture mlx-whisper decode-timeout warnings from one transcribe call."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self._thread_id = threading.get_ident()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self._thread_id:
            return
        message = record.getMessage()
        if message.startswith("Decode timeout"):
            self.messages.append(message)


def _huggingface_hub_cache_dir() -> Path:
    cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if cache_dir:
        return Path(cache_dir).expanduser()
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _is_whisper_snapshot_dir(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    return any(
        (path / filename).exists()
        for filename in ("model.safetensors", "weights.safetensors", "weights.npz")
    )


def _resolve_cached_model_source(path_or_hf_repo: str) -> str:
    model_path = Path(path_or_hf_repo).expanduser()
    if model_path.exists():
        return str(model_path)

    if "/" not in path_or_hf_repo:
        return path_or_hf_repo

    cache_root = _huggingface_hub_cache_dir()
    snapshots = cache_root / f"models--{path_or_hf_repo.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return path_or_hf_repo

    candidates = sorted(
        (candidate for candidate in snapshots.iterdir() if candidate.is_dir()),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if _is_whisper_snapshot_dir(candidate):
            return str(candidate)
    return path_or_hf_repo


def _ensure_mlx_whisper_runtime():
    """Import MLX Whisper only when the local client is actually used."""
    global mx, mlx_whisper, load_model
    if mx is None:
        mx = importlib.import_module("mlx.core")
    if mlx_whisper is None:
        mlx_whisper = importlib.import_module("mlx_whisper")
    if load_model is None:
        load_model = importlib.import_module("mlx_whisper.load_models").load_model
    return mx, mlx_whisper, load_model


def _supports_decode_option(option_name: str) -> bool:
    runtime_whisper = mlx_whisper
    if runtime_whisper is None:
        _, runtime_whisper, _ = _ensure_mlx_whisper_runtime()
    decoding_module = getattr(runtime_whisper, "decoding", None)
    options_cls = getattr(decoding_module, "DecodingOptions", None)
    fields = getattr(options_cls, "__dataclass_fields__", None)
    if isinstance(fields, dict):
        return option_name in fields

    module_file = getattr(runtime_whisper, "__file__", None)
    if module_file is None or not isinstance(module_file, (str, bytes, os.PathLike)):
        # Test doubles often replace mlx_whisper with a MagicMock. Default to
        # "supported" there so unit tests can assert the intended call shape.
        return True

    decoding_path = Path(module_file).with_name("decoding.py")
    try:
        return option_name in decoding_path.read_text()
    except OSError:
        return False


def supports_eager_eval() -> bool:
    """Whether the installed mlx-whisper build accepts eager_eval."""
    return _supports_decode_option("eager_eval")


class LocalTranscriptionClient:
    """Transcribe audio locally via mlx-whisper.

    Parameters
    ----------
    model : str
        HuggingFace model identifier. Downloaded on first use.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        decode_timeout: float | None = _DEFAULT_DECODE_TIMEOUT,
        eager_eval: bool = _DEFAULT_EAGER_EVAL,
    ) -> None:
        self._model = model
        self._decode_timeout = decode_timeout
        self._eager_eval = eager_eval
        self._loaded = False
        self._model_instance = None
        self._warned_eager_eval_unsupported = False

    def _ensure_model(self) -> None:
        """Trigger model download if not cached."""
        self.prepare()

    def _install_model_holder(self) -> None:
        """Point mlx-whisper's singleton cache at this client's loaded model."""
        if self._model_instance is None:
            raise RuntimeError("Whisper model was not loaded before installation")
        transcribe_module = importlib.import_module("mlx_whisper.transcribe")
        transcribe_module.ModelHolder.model = self._model_instance
        transcribe_module.ModelHolder.model_path = self._model

    def _load_dtype(self):
        """Choose the MLX dtype used to warm the selected Whisper repo."""
        runtime_mx, _, _ = _ensure_mlx_whisper_runtime()
        return runtime_mx.float16

    def prepare(self) -> None:
        """Warm the Whisper model cache without running a transcription."""
        if self._loaded:
            self._install_model_holder()
            return
        _, _, runtime_load_model = _ensure_mlx_whisper_runtime()
        logger.info("Preloading Whisper model %s", self._model)
        model_source = _resolve_cached_model_source(self._model)
        self._model_instance = runtime_load_model(
            model_source,
            dtype=self._load_dtype(),
        )
        self._loaded = True
        self._install_model_holder()

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio bytes and return text.

        Decodes WAV to numpy float32 array in-process (no ffmpeg needed),
        then passes directly to mlx_whisper.transcribe().
        """
        if not wav_bytes:
            return ""

        self._ensure_model()
        self._install_model_holder()
        _, runtime_whisper, _ = _ensure_mlx_whisper_runtime()

        # Decode WAV bytes to float32 numpy array — bypass ffmpeg entirely
        audio = self._decode_wav(wav_bytes)

        kwargs = {
            "path_or_hf_repo": self._model,
            "language": "en",
        }
        if supports_eager_eval():
            # Keep Spoke's timeout guard explicit even when eager_eval is enabled.
            # mlx-whisper's package default is shorter than Spoke's contention
            # contract and can return timeout-tainted partial text under load.
            kwargs["decode_timeout"] = self._decode_timeout
            kwargs["eager_eval"] = self._eager_eval
        elif self._eager_eval and not self._warned_eager_eval_unsupported:
            kwargs["decode_timeout"] = self._decode_timeout
            logger.warning(
                "Installed mlx-whisper does not support eager_eval yet; ignoring the setting"
            )
            self._warned_eager_eval_unsupported = True
        else:
            kwargs["decode_timeout"] = self._decode_timeout

        timeout_capture = _DecodeTimeoutCapture()
        backend_logger = logging.getLogger("mlx_whisper")
        backend_logger.addHandler(timeout_capture)
        try:
            result = runtime_whisper.transcribe(audio, **kwargs)
        finally:
            backend_logger.removeHandler(timeout_capture)

        if timeout_capture.messages:
            detail = "; ".join(timeout_capture.messages)
            raise TimeoutError(f"mlx-whisper returned timeout-tainted text: {detail}")

        text = result.get("text", "").strip()
        text = truncate_repetition(text)
        text = repair_ontology_terms(text)
        if is_hallucination(text):
            logger.info("Discarding hallucination: %r", text)
            return ""
        logger.info("Local transcription: %r (%d bytes audio)", text, len(wav_bytes))
        return text

    @staticmethod
    def _decode_wav(wav_bytes: bytes) -> np.ndarray:
        """Decode WAV bytes to float32 numpy array at 16kHz mono."""
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            if wf.getnchannels() != 1:
                raise ValueError(f"Expected mono audio, got {wf.getnchannels()} channels")
            if wf.getsampwidth() != 2:
                raise ValueError(f"Expected 16-bit audio, got {wf.getsampwidth() * 8}-bit")
            frames = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(frames, dtype=np.int16)
            return pcm.astype(np.float32) / 32768.0

    def unload(self) -> None:
        """Release the loaded Whisper model to free memory."""
        if not self._loaded:
            return
        logger.info("Unloading Whisper model %s", self._model)
        self._model_instance = None
        self._loaded = False
        # Clear mlx-whisper's singleton cache so it doesn't hold a ref.
        try:
            transcribe_module = importlib.import_module("mlx_whisper.transcribe")
            if getattr(transcribe_module.ModelHolder, "model_path", None) == self._model:
                transcribe_module.ModelHolder.model = None
                transcribe_module.ModelHolder.model_path = ""
        except Exception:
            pass

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently resident in memory."""
        return self._loaded

    def close(self) -> None:
        """Release resources."""
        self.unload()
