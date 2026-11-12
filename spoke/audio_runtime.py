"""Seat and verify an explicitly selected PortAudio binary before capture."""

import hashlib
import os
from pathlib import Path


def _selected_runtime():
    requested = os.environ.get("SPOKE_PORTAUDIO_LIBRARY", "").strip()
    expected = os.environ.get("SPOKE_PORTAUDIO_SHA256", "").strip().lower()
    if not requested and not expected:
        return None
    if (not requested or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise RuntimeError("PortAudio pin requires an absolute library path and SHA-256")
    try:
        library = Path(requested).expanduser()
        if not library.is_absolute():
            raise ValueError("library path is not absolute")
        library = library.resolve(strict=True)
        digest = hashlib.sha256(library.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"library SHA-256 {digest} does not match selected pin")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"PortAudio runtime is unverified: {exc}") from exc
    return {"library": str(library), "sha256": digest, "pinned": True}


def seat_audio_runtime():
    selected = _selected_runtime()
    if selected is not None:
        # macOS strips DYLD variables across protected shell executables.
        # ctypes' library finder reads this in-process value before dlopen.
        directory = str(Path(selected["library"]).parent)
        paths = os.environ.get("DYLD_LIBRARY_PATH", "").split(":")
        os.environ["DYLD_LIBRARY_PATH"] = ":".join(
            [directory] + [path for path in paths if path and path != directory])


def audio_runtime_receipt(sounddevice):
    selected = _selected_runtime()
    actual_name = str(getattr(sounddevice, "_libname", "unknown"))
    if selected is None:
        return {"library": actual_name, "pinned": False}
    try:
        actual = Path(actual_name).resolve(strict=True)
        if str(actual) != selected["library"]:
            raise ValueError(f"requested {selected['library']}, loaded {actual}")
        digest = hashlib.sha256(actual.read_bytes()).hexdigest()
        if digest != selected["sha256"]:
            raise ValueError("loaded binary changed after runtime selection")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"PortAudio runtime is unverified: {exc}") from exc
    return selected
