"""Contract tests for the dev launcher script."""

import os
import subprocess
import sys
import time
from pathlib import Path


def _script_text() -> str:
    script = Path(__file__).resolve().parent.parent / "scripts" / "launch-dev.sh"
    return script.read_text()


def _inline_launcher_source() -> str:
    text = _script_text()
    start = text.index("<<'PY'\n") + len("<<'PY'\n")
    end = text.rindex("\nPY")
    return text[start:end]


def _run_inline_launcher(
    repo_root: Path,
    log_file: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["REPO_ROOT"] = str(repo_root)
    env["LOG_FILE"] = str(log_file)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", _inline_launcher_source()],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_launch_script_preserves_startup_logs():
    """The launcher should route the actual background process output into the durable log."""
    text = _script_text()

    assert "spoke-dev-launch.log" in text
    assert ">/dev/null 2>&1" not in text
    assert 'with log_file.open("a", encoding="utf-8") as log:' in text
    assert "stdout=log" in text
    assert "stderr=subprocess.STDOUT" in text
    assert "traceback.print_exc(file=log)" in text


def test_launch_script_does_not_override_persisted_model_preferences():
    """The launcher should not export model env vars that clobber saved prefs."""
    text = _script_text()

    assert 'SPOKE_PREVIEW_MODEL="${SPOKE_PREVIEW_MODEL:-mlx-community/whisper-medium.en-mlx-8bit}"' not in text
    assert 'SPOKE_TRANSCRIPTION_MODEL="${SPOKE_TRANSCRIPTION_MODEL:-mlx-community/whisper-large-v3-turbo}"' not in text
    assert 'SPOKE_WHISPER_MODEL="${SPOKE_WHISPER_MODEL:-mlx-community/whisper-medium.en-mlx-8bit}"' not in text


def test_launch_script_seeds_default_command_url():
    """The dev launcher should enable the local command path by default."""
    text = _script_text()

    assert 'SPOKE_COMMAND_URL="${SPOKE_COMMAND_URL:-http://localhost:8001}"' in text


def test_launch_script_avoids_nohup_detach():
    """The launcher should use the plain background-launch form that survived manual smoke."""
    text = _script_text()

    assert "nohup " not in text
    assert "start_new_session=True" in text
    assert "subprocess.Popen(" in text
    assert text.rstrip().endswith("PY")


def test_inline_launcher_routes_child_output_into_log(tmp_path):
    """A successful detached child should emit into the durable log path."""
    repo_root = tmp_path / "repo"
    python_exe = repo_root / ".venv" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("#!/bin/sh\nprintf 'child-started\\n'\n")
    python_exe.chmod(0o755)

    log_file = tmp_path / "launch.log"
    result = _run_inline_launcher(repo_root, log_file)

    assert result.returncode == 0
    assert result.stderr == ""

    for _ in range(20):
        if log_file.exists() and "child-started" in log_file.read_text():
            break
        time.sleep(0.02)
    else:
        raise AssertionError("expected detached child output to reach launch log")


def test_inline_launcher_strips_model_override_env_vars(tmp_path, monkeypatch):
    """Detached launch should drop model env vars so saved prefs can win."""
    repo_root = tmp_path / "repo"
    python_exe = repo_root / ".venv" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text(
        "#!/bin/sh\n"
        "printf 'preview=%s\\n' \"${SPOKE_PREVIEW_MODEL:-}\"\n"
        "printf 'transcription=%s\\n' \"${SPOKE_TRANSCRIPTION_MODEL:-}\"\n"
        "printf 'legacy=%s\\n' \"${SPOKE_WHISPER_MODEL:-}\"\n"
    )
    python_exe.chmod(0o755)

    monkeypatch.setenv("SPOKE_PREVIEW_MODEL", "preview-override")
    monkeypatch.setenv("SPOKE_TRANSCRIPTION_MODEL", "transcription-override")
    monkeypatch.setenv("SPOKE_WHISPER_MODEL", "legacy-override")

    log_file = tmp_path / "launch.log"
    result = _run_inline_launcher(repo_root, log_file)

    assert result.returncode == 0
    assert result.stderr == ""

    for _ in range(20):
        if log_file.exists() and "legacy=" in log_file.read_text():
            break
        time.sleep(0.02)
    else:
        raise AssertionError("expected detached child output to reach launch log")

    log_text = log_file.read_text()
    assert "preview=\n" in log_text
    assert "transcription=\n" in log_text
    assert "legacy=\n" in log_text


def test_inline_launcher_preserves_default_command_url(tmp_path):
    """Detached launch should keep the default local command URL in the child env."""
    repo_root = tmp_path / "repo"
    python_exe = repo_root / ".venv" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text(
        "#!/bin/sh\n"
        "printf 'command_url=%s\\n' \"${SPOKE_COMMAND_URL:-}\"\n"
    )
    python_exe.chmod(0o755)

    log_file = tmp_path / "launch.log"
    result = _run_inline_launcher(repo_root, log_file)

    assert result.returncode == 0
    assert result.stderr == ""

    for _ in range(20):
        if log_file.exists() and "command_url=" in log_file.read_text():
            break
        time.sleep(0.02)
    else:
        raise AssertionError("expected detached child output to reach launch log")

    assert "command_url=http://localhost:8001\n" in log_file.read_text()


def test_inline_launcher_logs_spawn_failure_to_log(tmp_path):
    """If detached spawn fails before child startup, the failure should still be durable in the log."""
    repo_root = tmp_path / "missing-python-repo"
    repo_root.mkdir()
    log_file = tmp_path / "launch.log"

    result = _run_inline_launcher(
        repo_root,
        log_file,
        extra_env={"UV_BIN": str(repo_root / "missing-uv")},
    )

    assert result.stderr == ""
    assert log_file.exists()
    log_text = log_file.read_text()
    assert "No repo .venv Python found and UV launcher is unavailable." in log_text
