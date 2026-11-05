"""A real child process with a deterministic externally observable stall."""

from pathlib import Path
import subprocess
import sys
import threading


class FixtureClient:
    _model = "fixture-model"

    def prepare(self):
        return None

    def transcribe(self, audio):
        if audio.startswith(b"native-child:"):
            marker = audio.removeprefix(b"native-child:").decode()
            child = subprocess.Popen([sys.executable, "-c",
                "import os,sys,threading; from pathlib import Path; "
                "Path(sys.argv[1]).write_text(str(os.getpid())); threading.Event().wait()", marker])
            child.wait()
            return "unexpected native child completion"
        if audio == b"fail":
            raise ValueError("fixture failure")
        if audio == b"recover":
            return "recovered"
        Path(audio.decode()).touch()
        threading.Event().wait()

