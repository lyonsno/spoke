"""A selected audio runtime must not silently fall back to the bundled library."""

import hashlib
from pathlib import Path

import pytest


@pytest.mark.parametrize("failure", ["wrong_library", "wrong_hash", "incomplete_pin"])
def test_capture_rejects_unverified_requested_audio_runtime(monkeypatch, tmp_path, failure):
    from spoke import capture
    monkeypatch.setenv("SPOKE_VAD_ENABLED", "0")
    requested = tmp_path / "libportaudio.dylib"
    requested.write_bytes(b"guard fixture, not a loadable audio library")
    monkeypatch.setenv("SPOKE_PORTAUDIO_LIBRARY", str(requested))
    monkeypatch.setenv("SPOKE_PORTAUDIO_SHA256", "0" * 64)
    if failure != "wrong_library":
        monkeypatch.setattr(capture.sd, "_libname", str(requested))
    if failure == "incomplete_pin":
        monkeypatch.delenv("SPOKE_PORTAUDIO_SHA256")
    with pytest.raises(RuntimeError, match="PortAudio"):
        capture.AudioCapture()


def test_capture_records_actual_pinned_library_identity(monkeypatch):
    from spoke import capture
    library = Path(capture.sd._libname).resolve()
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    monkeypatch.setenv("SPOKE_VAD_ENABLED", "0")
    monkeypatch.setenv("SPOKE_PORTAUDIO_LIBRARY", str(library))
    monkeypatch.setenv("SPOKE_PORTAUDIO_SHA256", digest)
    cap = capture.AudioCapture()
    assert cap._audio_runtime["library"] == str(library)
    assert cap._audio_runtime["sha256"] == digest
    assert cap._audio_runtime["pinned"] is True


def test_seating_preserves_other_paths_and_does_not_duplicate_selected_directory(monkeypatch, tmp_path):
    from spoke.audio_runtime import seat_audio_runtime
    library = tmp_path / "libportaudio.dylib"
    library.write_bytes(b"runtime selection fixture, not loaded")
    monkeypatch.setenv("SPOKE_PORTAUDIO_LIBRARY", str(library))
    monkeypatch.setenv("SPOKE_PORTAUDIO_SHA256", hashlib.sha256(library.read_bytes()).hexdigest())
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/prior/library")
    seat_audio_runtime()
    seat_audio_runtime()
    import os
    assert os.environ["DYLD_LIBRARY_PATH"].split(":") == [str(tmp_path), "/prior/library"]
