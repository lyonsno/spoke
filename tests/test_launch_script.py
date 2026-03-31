"""Contract tests for the dev launcher script."""

import os
import subprocess
import sys
import time
from pathlib import Path

from spoke.launch_targets import parse_env_overrides


def _script_text() -> str:
    script = Path(__file__).resolve().parent.parent / "scripts" / "launch-dev.sh"
    return script.read_text()


def _smoke_script_text() -> str:
    script = Path(__file__).resolve().parent.parent / "scripts" / "launch-smoke.sh"
    return script.read_text()


def _launch_target_script_text() -> str:
    script = Path(__file__).resolve().parent.parent / "scripts" / "launch-target.sh"
    return script.read_text()


def _inline_launcher_source() -> str:
    text = _script_text()
    start = text.index("<<'PY'\n") + len("<<'PY'\n")
    end = text.rindex("\nPY")
    return text[start:end]


def _smoke_inline_launcher_source() -> str:
    text = _smoke_script_text()
    start = text.index("<<'PY'\n") + len("<<'PY'\n")
    end = text.rindex("\nPY")
    return text[start:end]


def _launch_target_inline_source() -> str:
    text = _launch_target_script_text()
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


def _run_smoke_inline_launcher(
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
        [sys.executable, "-c", _smoke_inline_launcher_source()],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_launch_target_inline_launcher(
    helper_repo_root: Path,
    targets_file: Path,
    target_id: str,
    log_file: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HELPER_REPO_ROOT"] = str(helper_repo_root)
    env["SPOKE_LAUNCH_TARGETS_PATH"] = str(targets_file)
    env["TARGET_ID"] = target_id
    env["LOG_FILE"] = str(log_file)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", _launch_target_inline_source()],
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


def test_launch_script_supports_configured_dev_target():
    """The dev launcher should allow a stable Automator binding to target a fresh worktree."""
    text = _script_text()

    assert 'DEV_TARGET_FILE="${HOME}/.config/spoke/dev-target"' in text
    assert 'TARGET_SOURCE="~/.config/spoke/dev-target"' in text
    assert 'Launching Spoke from %s (%s)' in text


def test_launch_scripts_preserve_spaces_in_target_paths():
    """Configured launcher targets should trim only the trailing newline, not interior spaces."""
    assert 'CONFIGURED_REPO_ROOT="$(tr -d \'[:space:]\' < "$DEV_TARGET_FILE")"' not in _script_text()
    assert 'REPO_ROOT="$(cat "$SMOKE_TARGET_FILE" | tr -d \'[:space:]\')"' not in _smoke_script_text()


def test_launch_script_logs_preflight_kill_diagnostics():
    """The launcher should log its preflight without broad process killing."""
    text = _script_text()

    assert "Launcher PID" in text
    assert "Launcher preflight:" in text
    assert "lock-holder pid" in text
    assert 'pkill -TERM -f "python.*spoke"' not in text
    assert "rm -f ~/Library/Logs/.spoke.lock" not in text


def test_smoke_launch_script_logs_preflight_without_broad_kill():
    """Smoke launch should hand off through the single-instance guard."""
    text = _smoke_script_text()

    assert "Launcher PID" in text
    assert "Launcher preflight:" in text
    assert "lock-holder pid" in text
    assert 'pkill -TERM -f "python.*spoke"' not in text
    assert "rm -f ~/Library/Logs/.spoke.lock" not in text


def test_launch_script_avoids_nohup_detach():
    """The launcher should use the plain background-launch form that survived manual smoke."""
    text = _script_text()

    assert "nohup " not in text
    assert "start_new_session=True" in text
    assert "subprocess.Popen(" in text
    assert text.rstrip().endswith("PY")


def test_launch_target_script_reads_named_target_registry():
    """The target-switch launcher should resolve launch surfaces by registry id."""
    text = _launch_target_script_text()

    assert 'TARGETS_FILE="${SPOKE_LAUNCH_TARGETS_PATH:-$HOME/.config/spoke/launch_targets.json}"' in text
    assert 'TARGET_ID="${1:-${TARGET_ID:-}}"' in text
    assert "spoke-launch-target.log" in text
    assert "SPOKE_LAUNCH_TARGET_ID" in text
    assert 'pkill -TERM -f "python.*spoke"' not in text
    assert ".spoke.lock" not in text


def test_inline_launch_target_launcher_starts_requested_target(tmp_path):
    """A registry-backed target launch should spawn the selected repo's Python runtime."""
    helper_repo_root = tmp_path / "helper"
    helper_repo_root.mkdir()

    target_repo = tmp_path / "target repo"
    python_exe = target_repo / ".venv" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text(
        "#!/bin/sh\n"
        "printf 'target=%s\\n' \"${SPOKE_LAUNCH_TARGET_ID:-}\"\n"
    )
    python_exe.chmod(0o755)

    targets_file = tmp_path / "launch_targets.json"
    targets_file.write_text(
        (
            '{"selected":"main","targets":['
            '{"id":"main","label":"Main","path":"%s"},'
            '{"id":"smoke","label":"Smoke","path":"%s"}'
            "]}"
        )
        % (tmp_path / "other", target_repo)
    )

    log_file = tmp_path / "launch-target.log"
    result = _run_launch_target_inline_launcher(
        helper_repo_root=helper_repo_root,
        targets_file=targets_file,
        target_id="smoke",
        log_file=log_file,
    )

    assert result.returncode == 0
    assert result.stderr == ""

    for _ in range(20):
        if log_file.exists() and "target=smoke" in log_file.read_text():
            break
        time.sleep(0.02)
    else:
        raise AssertionError("expected selected target output to reach launch log")


def test_parse_env_overrides_expands_shell_style_defaults_and_suffixes(tmp_path, monkeypatch):
    """Launch-target env parsing should support the small shell subset used in smoke env files."""
    env_file = tmp_path / ".spoke-smoke-env"
    monkeypatch.setenv("HOME", "/Users/tester")
    monkeypatch.delenv("SPOKE_COMMAND_URL", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/base/path")
    env_file.write_text(
        'export SPOKE_COMMAND_URL="${SPOKE_COMMAND_URL:-http://localhost:8001}"\n'
        'export SPOKE_COMMAND_MODEL_DIR="$HOME/dev/scripts/quant/models"\n'
        'export PYTHONPATH="/Users/noahlyons/dev/mlx-audio-pr-607-voxtral-tts${PYTHONPATH:+:$PYTHONPATH}"\n'
    )

    assert parse_env_overrides(env_file) == {
        "SPOKE_COMMAND_URL": "http://localhost:8001",
        "SPOKE_COMMAND_MODEL_DIR": "/Users/tester/dev/scripts/quant/models",
        "PYTHONPATH": "/Users/noahlyons/dev/mlx-audio-pr-607-voxtral-tts:/base/path",
    }


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


def test_inline_launcher_logs_child_spawn_context(tmp_path):
    """Detached launch should log the exact child command and launcher PID context."""
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

    log_text = log_file.read_text()
    assert "Launcher child command:" in log_text
    assert "Launcher PID context:" in log_text


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


def test_smoke_inline_launcher_prefers_uv_tts_runtime(tmp_path):
    """Smoke launch should use uv --extra tts when TTS is enabled, even if .venv Python exists."""
    repo_root = tmp_path / "repo"

    python_exe = repo_root / ".venv" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("#!/bin/sh\nprintf 'venv-python-started\\n'\n")
    python_exe.chmod(0o755)

    uv_bin = tmp_path / "fake-uv"
    uv_bin.write_text("#!/bin/sh\nprintf 'uv-tts-started\\n'\n")
    uv_bin.chmod(0o755)

    log_file = tmp_path / "launch.log"
    result = _run_smoke_inline_launcher(
        repo_root,
        log_file,
        extra_env={
            "SPOKE_TTS_VOICE": "casual_female",
            "UV_BIN": str(uv_bin),
        },
    )

    assert result.returncode == 0
    assert result.stderr == ""

    for _ in range(20):
        if log_file.exists() and "started" in log_file.read_text():
            break
        time.sleep(0.02)
    else:
        raise AssertionError("expected detached child output to reach smoke launch log")

    log_text = log_file.read_text()
    assert "uv-tts-started\n" in log_text
    assert "venv-python-started\n" not in log_text
    assert "--extra" in log_text
    assert "tts" in log_text


def test_smoke_inline_launcher_skips_broken_pyenv_shim(tmp_path):
    """Smoke launch should fall back to a working uv binary when the configured shim is broken."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    broken_uv = tmp_path / ".pyenv" / "shims" / "uv"
    broken_uv.parent.mkdir(parents=True)
    broken_uv.write_text("#!/bin/sh\nexit 1\n")
    broken_uv.chmod(0o755)

    fallback_dir = tmp_path / "bin"
    fallback_dir.mkdir()
    fallback_uv = fallback_dir / "uv"
    fallback_uv.write_text("#!/bin/sh\nprintf 'fallback-uv-started\\n'\n")
    fallback_uv.chmod(0o755)

    log_file = tmp_path / "launch.log"
    result = _run_smoke_inline_launcher(
        repo_root,
        log_file,
        extra_env={
            "SPOKE_TTS_VOICE": "casual_female",
            "UV_BIN": str(broken_uv),
            "PATH": f"{fallback_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        },
    )

    assert result.returncode == 0
    assert result.stderr == ""

    for _ in range(20):
        if log_file.exists() and "fallback-uv-started" in log_file.read_text():
            break
        time.sleep(0.02)
    else:
        raise AssertionError("expected fallback uv output to reach smoke launch log")

    assert "fallback-uv-started\n" in log_file.read_text()


def test_smoke_inline_launcher_falls_back_to_path_uv_for_tts_runtime(tmp_path):
    """If UV_BIN points nowhere but uv is on PATH, smoke launch should still
    use the uv TTS runtime instead of falling back to repo python."""
    repo_root = tmp_path / "repo"

    python_exe = repo_root / ".venv" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("#!/bin/sh\nprintf 'venv-python-started\\n'\n")
    python_exe.chmod(0o755)

    uv_dir = tmp_path / "bin"
    uv_dir.mkdir()
    uv_bin = uv_dir / "uv"
    uv_bin.write_text("#!/bin/sh\nprintf 'uv-tts-started\\n'\n")
    uv_bin.chmod(0o755)

    log_file = tmp_path / "launch.log"
    result = _run_smoke_inline_launcher(
        repo_root,
        log_file,
        extra_env={
            "SPOKE_TTS_VOICE": "casual_female",
            "UV_BIN": str(tmp_path / "missing-uv"),
            "PATH": f"{uv_dir}:{os.environ.get('PATH', '')}",
        },
    )

    assert result.returncode == 0
    assert result.stderr == ""

    for _ in range(20):
        if log_file.exists() and "started" in log_file.read_text():
            break
        time.sleep(0.02)
    else:
        raise AssertionError("expected detached child output to reach smoke launch log")

    log_text = log_file.read_text()
    assert "uv-tts-started\n" in log_text
    assert "venv-python-started\n" not in log_text
    assert "--extra" in log_text
    assert "tts" in log_text


def test_launch_script_prefers_configured_dev_target(tmp_path):
    """A configured dev target should override the checkout that contains the script."""
    target_repo = tmp_path / "target-repo"
    python_exe = target_repo / ".venv" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("#!/bin/sh\nprintf 'target-repo-started\\n'\n")
    python_exe.chmod(0o755)

    home = tmp_path / "home"
    config_dir = home / ".config" / "spoke"
    config_dir.mkdir(parents=True)
    (config_dir / "dev-target").write_text(str(target_repo))

    log_dir = home / "Library" / "Logs"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "spoke-dev-launch.log"

    script = Path(__file__).resolve().parent.parent / "scripts" / "launch-dev.sh"
    result = subprocess.run(
        [str(script)],
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""

    for _ in range(20):
        if log_file.exists() and "target-repo-started" in log_file.read_text():
            break
        time.sleep(0.02)
    else:
        raise AssertionError("expected configured dev target output to reach launch log")

    log_text = log_file.read_text()
    assert f"Launching Spoke from {target_repo} (~/.config/spoke/dev-target)" in log_text


def test_launch_script_preserves_spaces_in_configured_dev_target(tmp_path):
    """Configured dev targets with spaces should launch successfully."""
    target_repo = tmp_path / "target repo"
    python_exe = target_repo / ".venv" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("#!/bin/sh\nprintf 'spaced-target-started\\n'\n")
    python_exe.chmod(0o755)

    home = tmp_path / "home"
    config_dir = home / ".config" / "spoke"
    config_dir.mkdir(parents=True)
    (config_dir / "dev-target").write_text(str(target_repo))

    log_dir = home / "Library" / "Logs"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "spoke-dev-launch.log"

    script = Path(__file__).resolve().parent.parent / "scripts" / "launch-dev.sh"
    result = subprocess.run(
        [str(script)],
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""

    for _ in range(20):
        if log_file.exists() and "spaced-target-started" in log_file.read_text():
            break
        time.sleep(0.02)
    else:
        raise AssertionError("expected spaced dev target output to reach launch log")

    log_text = log_file.read_text()
    assert f"Launching Spoke from {target_repo} (~/.config/spoke/dev-target)" in log_text
