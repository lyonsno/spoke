"""A real child process with a deterministic externally observable stall."""

from pathlib import Path
import threading


class FixtureClient:
    _model = "fixture-model"

    def prepare(self):
        return None

    def transcribe(self, audio):
        if audio == b"fail":
            raise ValueError("fixture failure")
        if audio == b"recover":
            return "recovered"
        Path(audio.decode()).touch()
        threading.Event().wait()

