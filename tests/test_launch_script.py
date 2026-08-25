"""Contract tests for the launcher architecture.

Tests the launcher registry contract (spoke/launch_targets.py) and
verifies the old file-based launcher architecture is retired.
"""

import ast
import importlib.util
import json
import math
import os
import subprocess
import time
from pathlib import Path

import pytest

from spoke.launch_targets import (
    iter_launch_targets,
    load_launch_target_registry,
    parse_env_overrides,
    resolve_launch_target,
    save_selected_launch_target,
)


def _main_script_text() -> str:
    script = Path(__file__).resolve().parent.parent / "scripts" / "launch-main.sh"
    return script.read_text()


def _target_script_text() -> str:
    script = Path(__file__).resolve().parent.parent / "scripts" / "launch-target.sh"
    return script.read_text()


def _selected_script_path() -> Path:
    return Path(__file__).resolve().parent.parent / "scripts" / "launch-selected.sh"


def _launcher_python_text() -> str:
    text = _main_script_text()
    start_marker = "/usr/bin/python3 - <<'PY'\n"
    start = text.index(start_marker) + len(start_marker)
    end = text.index("\nPY", start)
    return text[start:end]


def _launcher_apply_env_file():
    source = _launcher_python_text()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_env_file":
            function_source = ast.get_source_segment(source, node)
            assert function_source is not None
            child_env: dict[str, str] = {}
            namespace = {"Path": Path, "child_env": child_env, "os": os}
            exec(function_source, namespace)
            return namespace["_apply_env_file"], child_env
    raise AssertionError("launch-main.sh must define _apply_env_file")


def _launcher_admission_timeout_parser():
    source = _launcher_python_text()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_launch_admission_timeout":
            function_source = ast.get_source_segment(source, node)
            assert function_source is not None
            namespace = {"math": math}
            exec(function_source, namespace)
            return namespace["_launch_admission_timeout"]
    raise AssertionError("launch-main.sh must define _launch_admission_timeout")


def _execute_launcher_python(
    tmp_path,
    monkeypatch,
    *,
    registry_text: str | None,
    log_failure: bool = False,
    launcher_env: dict[str, str] | None = None,
):
    registry = tmp_path / "launch_targets.json"
    if registry_text is not None:
        registry.write_text(registry_text)

    log_file = tmp_path / "spoke-launch.log"
    if log_failure:
        blocked_parent = tmp_path / "not-a-directory"
        blocked_parent.write_text("occupied")
        log_file = blocked_parent / "spoke-launch.log"

    run_calls = []
    popen_calls = []

    class Completed:
        returncode = 0

    class Process:
        pid = 4242

        def poll(self):
            return None

    def fake_run(args, *pargs, **kwargs):
        run_calls.append((list(args), kwargs))
        return Completed()

    def fake_popen(args, *pargs, **kwargs):
        popen_calls.append((list(args), kwargs))
        process = Process()
        child_env = kwargs.get("env", {})
        admission_path = child_env.get("SPOKE_LAUNCH_ADMISSION_PATH")
        admission_token = child_env.get("SPOKE_LAUNCH_ADMISSION_TOKEN")
        if _is_spoke_child(list(args)) and admission_path and admission_token:
            Path(admission_path).write_text(
                json.dumps(
                    {
                        "status": "admitted",
                        "token": admission_token,
                        "pid": process.pid,
                        "registry_path": child_env.get("SPOKE_LAUNCH_TARGETS_PATH"),
                    }
                )
            )
        return process

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setenv("HELPER_REPO_ROOT", str(Path(__file__).resolve().parent.parent))
    monkeypatch.setenv("TARGETS_FILE", str(registry))
    monkeypatch.setenv("LOG_FILE", str(log_file))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SPOKE_EXPECTED_LAUNCH_TARGET_ID", raising=False)
    monkeypatch.delenv("SPOKE_EXPECTED_LAUNCH_TARGET_PATH", raising=False)
    for key, value in (launcher_env or {}).items():
        monkeypatch.setenv(key, value)

    outcome = 0
    try:
        exec(compile(_launcher_python_text(), "launch-main.sh", "exec"), {"__name__": "__main__"})
    except SystemExit as exc:
        outcome = exc.code
    except Exception as exc:  # The witness reports accidental diagnostic-path crashes.
        outcome = exc

    return outcome, run_calls, popen_calls


def _is_spoke_child(command: list[str]) -> bool:
    return len(command) >= 3 and command[-2:] == ["-m", "spoke"]


def _is_retina_child(command: list[str]) -> bool:
    rendered = " ".join(command)
    return "retina-lasso" in rendered or "retina_lasso" in rendered or "throughglass_witness" in rendered


# ── Registry reading ────────────────────────────────────────────


class TestRegistryReading:
    """The launcher must read the selected target from the registry."""

    def test_load_registry(self, tmp_path):
        registry = tmp_path / "launch_targets.json"
        registry.write_text(json.dumps({
            "selected": "my_target",
            "targets": [
                {"id": "my_target", "label": "My Target", "path": "/tmp/my-worktree"},
            ],
        }))
        data = load_launch_target_registry(registry)
        assert data["selected"] == "my_target"
        assert len(data["targets"]) == 1
        assert data["targets"][0]["id"] == "my_target"

    def test_load_missing_registry_returns_empty(self, tmp_path):
        data = load_launch_target_registry(tmp_path / "nonexistent.json")
        assert data["selected"] is None
        assert data["targets"] == []

    def test_load_corrupt_registry_returns_empty(self, tmp_path):
        registry = tmp_path / "launch_targets.json"
        registry.write_text("not json at all {{{")
        data = load_launch_target_registry(registry)
        assert data["selected"] is None
        assert data["targets"] == []

    def test_resolve_selected_target(self, tmp_path):
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        registry = tmp_path / "launch_targets.json"
        registry.write_text(json.dumps({
            "selected": "test_target",
            "targets": [
                {"id": "test_target", "label": "Test", "path": str(worktree)},
                {"id": "other", "label": "Other", "path": "/tmp/other"},
            ],
        }))
        target = resolve_launch_target("test_target", registry)
        assert target is not None
        assert target["id"] == "test_target"
        assert str(target["path"]) == str(worktree)
        assert target["enabled"] is True

    def test_resolve_missing_target_returns_none(self, tmp_path):
        registry = tmp_path / "launch_targets.json"
        registry.write_text(json.dumps({
            "selected": "nonexistent",
            "targets": [{"id": "other", "label": "Other", "path": "/tmp/x"}],
        }))
        assert resolve_launch_target("nonexistent", registry) is None

    def test_target_with_missing_path_shows_disabled(self, tmp_path):
        registry = tmp_path / "launch_targets.json"
        registry.write_text(json.dumps({
            "selected": "gone",
            "targets": [{"id": "gone", "label": "Gone", "path": "/tmp/does-not-exist-99999"}],
        }))
        target = resolve_launch_target("gone", registry)
        assert target is not None
        assert target["enabled"] is False

    def test_iter_launch_targets_skips_invalid_entries(self, tmp_path):
        registry = tmp_path / "launch_targets.json"
        registry.write_text(json.dumps({
            "selected": "good",
            "targets": [
                {"id": "good", "label": "Good", "path": str(tmp_path)},
                {"id": "", "label": "Empty ID", "path": "/tmp/x"},  # invalid
                {"label": "No ID", "path": "/tmp/x"},  # invalid
                "not a dict",  # invalid
            ],
        }))
        targets = iter_launch_targets(registry)
        assert len(targets) == 1
        assert targets[0]["id"] == "good"


@pytest.mark.parametrize(
    "registry_text",
    [
        None,
        "{not-json",
        json.dumps({"selected": None, "targets": []}),
        json.dumps({"selected": "missing", "targets": []}),
        json.dumps({"selected": "gone", "targets": [{"id": "gone", "path": "/not/here"}]}),
        json.dumps({"selected": 7, "targets": [{"id": 7, "path": "/tmp"}]}),
        json.dumps({"selected": "relative", "targets": [{"id": "relative", "path": "."}]}),
        json.dumps(
            {
                "selected": "bad-env",
                "targets": [{"id": "bad-env", "path": "/tmp", "env": {"GOOD": "yes", "BAD": 7}}],
            }
        ),
        json.dumps(
            {
                "selected": "bad-env-shape",
                "targets": [{"id": "bad-env-shape", "path": "/tmp", "env": ["NOPE"]}],
            }
        ),
        json.dumps(
            {
                "selected": "bad-env-equals",
                "targets": [{"id": "bad-env-equals", "path": "/tmp", "env": {"A=B": "wrong"}}],
            }
        ),
        json.dumps(
            {
                "selected": "bad-env-nul-key",
                "targets": [{"id": "bad-env-nul-key", "path": "/tmp", "env": {"A\x00B": "wrong"}}],
            }
        ),
        json.dumps(
            {
                "selected": "bad-env-nul-value",
                "targets": [{"id": "bad-env-nul-value", "path": "/tmp", "env": {"A": "x\x00y"}}],
            }
        ),
        json.dumps(
            {
                "selected": "bad-label-empty",
                "targets": [{"id": "bad-label-empty", "label": "", "path": "/tmp"}],
            }
        ),
        json.dumps(
            {
                "selected": "bad-label-blank",
                "targets": [{"id": "bad-label-blank", "label": "   ", "path": "/tmp"}],
            }
        ),
        json.dumps(
            {
                "selected": "bad-id\x00",
                "targets": [{"id": "bad-id\x00", "path": "/tmp"}],
            }
        ),
        json.dumps(
            {
                "selected": "duplicate",
                "targets": [
                    {"id": "duplicate", "path": "/tmp"},
                    {"id": "duplicate", "path": "/tmp"},
                ],
            }
        ),
    ],
)
def test_launcher_negative_routes_start_no_spoke_or_witness(
    tmp_path,
    monkeypatch,
    registry_text,
):
    outcome, run_calls, popen_calls = _execute_launcher_python(
        tmp_path,
        monkeypatch,
        registry_text=registry_text,
    )

    assert outcome == 1
    assert any(call[0][0] == "osascript" for call in run_calls)
    assert not any(_is_spoke_child(command) for command, _kwargs in popen_calls)
    assert not any(_is_retina_child(command) for command, _kwargs in popen_calls)


def test_launcher_log_failure_still_attempts_visible_refusal(tmp_path, monkeypatch):
    outcome, run_calls, popen_calls = _execute_launcher_python(
        tmp_path,
        monkeypatch,
        registry_text=json.dumps({"selected": None, "targets": []}),
        log_failure=True,
    )

    assert outcome == 1
    assert any(call[0][0] == "osascript" for call in run_calls)
    assert not any(_is_spoke_child(command) for command, _kwargs in popen_calls)


def test_launcher_valid_route_starts_one_spoke_with_target_authority(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    python_exe = checkout / ".venv/bin/python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("stub")
    registry_text = json.dumps(
        {
            "selected": "reviewed",
            "targets": [
                {
                    "id": "reviewed",
                    "path": str(checkout),
                    "env": {"ROUTE": "reviewed"},
                }
            ],
        }
    )

    outcome, run_calls, popen_calls = _execute_launcher_python(
        tmp_path,
        monkeypatch,
        registry_text=registry_text,
    )

    app_calls = [call for call in popen_calls if _is_spoke_child(call[0])]
    assert outcome == 0
    assert run_calls == []
    assert len(app_calls) == 1
    assert app_calls[0][0] == [str(python_exe), "-m", "spoke"]
    assert app_calls[0][1]["cwd"] == checkout
    assert app_calls[0][1]["env"]["ROUTE"] == "reviewed"
    assert app_calls[0][1]["env"]["SPOKE_LAUNCH_TARGET_ID"] == "reviewed"
    assert not any(_is_retina_child(command) for command, _kwargs in popen_calls)


def test_launcher_refuses_target_changed_after_stable_dispatch(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    python_exe = checkout / ".venv/bin/python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("stub")
    registry_text = json.dumps(
        {
            "selected": "reviewed",
            "targets": [
                {
                    "id": "reviewed",
                    "path": str(checkout),
                    "env": {"SPOKE_VAD_ENABLED": "0"},
                }
            ],
        }
    )

    outcome, run_calls, popen_calls = _execute_launcher_python(
        tmp_path,
        monkeypatch,
        registry_text=registry_text,
        launcher_env={
            "SPOKE_EXPECTED_LAUNCH_TARGET_ID": "superseded",
            "SPOKE_EXPECTED_LAUNCH_TARGET_PATH": str(tmp_path / "superseded"),
        },
    )

    assert outcome == 1
    assert any(call[0][0] == "osascript" for call in run_calls)
    assert not any(_is_spoke_child(command) for command, _kwargs in popen_calls)


def test_runtime_applies_selected_target_env_before_capture_import():
    source = (Path(__file__).resolve().parent.parent / "spoke" / "__main__.py").read_text()
    apply_idx = source.find("apply_selected_launch_target_env(")
    capture_idx = source.find("from .capture import AudioCapture, vad_enabled")

    assert apply_idx != -1, "runtime must reconcile selected-target env"
    assert capture_idx != -1, "capture import not found"
    assert apply_idx < capture_idx, (
        "selected-target env must be effective before capture imports and initializes VAD"
    )


def test_runtime_publishes_launch_admission_before_capture_import():
    source = (Path(__file__).resolve().parent.parent / "spoke" / "__main__.py").read_text()
    publish_idx = source.find('publish_launch_admission("admitted"')
    capture_idx = source.find("from .capture import AudioCapture, vad_enabled")

    assert publish_idx != -1
    assert capture_idx != -1
    assert publish_idx < capture_idx


def test_launch_main_uses_caller_selected_registry_path():
    source = _main_script_text()

    assert (
        'TARGETS_FILE="${SPOKE_LAUNCH_TARGETS_PATH:-${HOME}/.config/spoke/launch_targets.json}"'
        in source
    )


def test_selected_launcher_dispatches_to_selected_target_script(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    target = tmp_path / "selected-target"
    scripts = target / "scripts"
    scripts.mkdir(parents=True)
    receipt = tmp_path / "selected-launch-receipt.json"
    selected_launcher = scripts / "launch-main.sh"
    selected_launcher.write_text(
        "#!/bin/bash\n"
        "python3 - <<'PY'\n"
        "import json, os\n"
        "from pathlib import Path\n"
        "Path(os.environ['SPOKE_SELECTED_LAUNCH_TEST_RECEIPT']).write_text(json.dumps({\n"
        "    'expected_id': os.environ.get('SPOKE_EXPECTED_LAUNCH_TARGET_ID'),\n"
        "    'expected_path': os.environ.get('SPOKE_EXPECTED_LAUNCH_TARGET_PATH'),\n"
        "}))\n"
        "PY\n"
    )
    selected_launcher.chmod(0o755)
    registry = tmp_path / "launch_targets.json"
    registry.write_text(
        json.dumps(
            {
                "selected": "reviewed",
                "targets": [
                    {
                        "id": "reviewed",
                        "label": "Reviewed",
                        "path": str(target),
                    }
                ],
            }
        )
    )
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "SPOKE_LAUNCH_TARGETS_PATH": str(registry),
            "SPOKE_SELECTED_LAUNCH_TEST_RECEIPT": str(receipt),
        }
    )

    result = subprocess.run(
        [str(_selected_script_path())],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(receipt.read_text()) == {
        "expected_id": "reviewed",
        "expected_path": str(target),
    }


def test_selected_launcher_rejects_duplicate_target_without_dispatch(tmp_path):
    target = tmp_path / "selected-target"
    scripts = target / "scripts"
    scripts.mkdir(parents=True)
    selected_launcher = scripts / "launch-main.sh"
    selected_launcher.write_text("#!/bin/bash\nexit 99\n")
    selected_launcher.chmod(0o755)
    registry = tmp_path / "launch_targets.json"
    registry.write_text(
        json.dumps(
            {
                "selected": "duplicate",
                "targets": [
                    {"id": "duplicate", "path": str(target)},
                    {"id": "duplicate", "path": str(target)},
                ],
            }
        )
    )
    env = os.environ.copy()
    env["SPOKE_LAUNCH_TARGETS_PATH"] = str(registry)

    result = subprocess.run(
        [str(_selected_script_path())],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "exactly once" in result.stderr


def test_selected_launcher_chain_delivers_effective_env_to_actual_child(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    home = tmp_path / "home"
    home.mkdir()
    receipt = tmp_path / "actual-child-env.json"
    fake_python = tmp_path / "record-child-env"
    fake_python.write_text(
        "#!/bin/bash\n"
        "/usr/bin/python3 - \"$@\" <<'PY'\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['SPOKE_ACTUAL_CHILD_RECEIPT']).write_text(json.dumps({\n"
        "    'argv': sys.argv[1:],\n"
        "    'cwd': os.getcwd(),\n"
        "    'launch_target_id': os.environ.get('SPOKE_LAUNCH_TARGET_ID'),\n"
        "    'vad_enabled': os.environ.get('SPOKE_VAD_ENABLED'),\n"
        "}))\n"
        "admission_path = os.environ.get('SPOKE_LAUNCH_ADMISSION_PATH')\n"
        "if admission_path:\n"
        "    Path(admission_path).write_text(json.dumps({\n"
        "        'status': 'admitted',\n"
        "        'token': os.environ['SPOKE_LAUNCH_ADMISSION_TOKEN'],\n"
        "        'pid': os.getpid(),\n"
        "        'registry_path': os.environ.get('SPOKE_LAUNCH_TARGETS_PATH'),\n"
        "    }))\n"
        "PY\n"
    )
    fake_python.chmod(0o755)
    registry = tmp_path / "launch_targets.json"
    registry.write_text(
        json.dumps(
            {
                "selected": "reviewed",
                "targets": [
                    {
                        "id": "reviewed",
                        "label": "Literal VAD-Off Recovery",
                        "path": str(repo_root),
                        "env": {
                            "SPOKE_RETINA_LASSO_AUTO_WITNESS": "0",
                            "SPOKE_VAD_ENABLED": "0",
                            "SPOKE_VENV_PYTHON": str(fake_python),
                        },
                    }
                ],
            }
        )
    )
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "SPOKE_ACTUAL_CHILD_RECEIPT": str(receipt),
            "SPOKE_LAUNCH_TARGETS_PATH": str(registry),
        }
    )

    result = subprocess.run(
        [str(_selected_script_path())],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    deadline = time.monotonic() + 3.0
    while not receipt.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert result.returncode == 0, result.stderr
    assert receipt.is_file(), "launcher child did not publish its effective environment"
    assert json.loads(receipt.read_text()) == {
        "argv": ["-m", "spoke"],
        "cwd": str(repo_root),
        "launch_target_id": "reviewed",
        "vad_enabled": "0",
    }


def test_launcher_chain_protects_registry_authority_from_secrets_redirect(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    home = tmp_path / "home"
    secrets = home / ".config/spoke/secrets.env"
    secrets.parent.mkdir(parents=True)
    receipt = tmp_path / "registry-authority.json"
    fake_python = tmp_path / "record-runtime-authority"
    fake_python.write_text(
        "#!/bin/bash\n"
        "/usr/bin/python3 - \"$@\" <<'PY'\n"
        "import json, os\n"
        "from pathlib import Path\n"
        "from spoke.launch_targets import apply_selected_launch_target_env\n"
        "runtime = apply_selected_launch_target_env(Path.cwd())\n"
        "Path(os.environ['SPOKE_ACTUAL_CHILD_RECEIPT']).write_text(json.dumps({\n"
        "    'registry_path': runtime.get('registry_path'),\n"
        "    'vad_enabled': os.environ.get('SPOKE_VAD_ENABLED'),\n"
        "}))\n"
        "admission_path = os.environ.get('SPOKE_LAUNCH_ADMISSION_PATH')\n"
        "if admission_path:\n"
        "    Path(admission_path).write_text(json.dumps({\n"
        "        'status': 'admitted',\n"
        "        'token': os.environ['SPOKE_LAUNCH_ADMISSION_TOKEN'],\n"
        "        'pid': os.getpid(),\n"
        "    }))\n"
        "PY\n"
    )
    fake_python.chmod(0o755)
    registry_a = tmp_path / "registry-a.json"
    registry_b = tmp_path / "registry-b.json"
    base_target = {
        "id": "reviewed",
        "label": "Reviewed",
        "path": str(repo_root),
    }
    registry_a.write_text(
        json.dumps(
            {
                "selected": "reviewed",
                "targets": [
                    {
                        **base_target,
                        "env": {
                            "SPOKE_VAD_ENABLED": "0",
                            "SPOKE_VENV_PYTHON": str(fake_python),
                        },
                    }
                ],
            }
        )
    )
    registry_b.write_text(
        json.dumps(
            {
                "selected": "reviewed",
                "targets": [{**base_target, "env": {"SPOKE_VAD_ENABLED": "1"}}],
            }
        )
    )
    secrets.write_text(f'SPOKE_LAUNCH_TARGETS_PATH="{registry_b}"\n')
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "SPOKE_ACTUAL_CHILD_RECEIPT": str(receipt),
            "SPOKE_LAUNCH_TARGETS_PATH": str(registry_a),
        }
    )

    result = subprocess.run(
        [str(_selected_script_path())],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    deadline = time.monotonic() + 3.0
    while not receipt.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert result.returncode == 0, result.stderr
    assert json.loads(receipt.read_text()) == {
        "registry_path": str(registry_a.resolve()),
        "vad_enabled": "0",
    }


def test_launcher_reports_child_pre_capture_refusal(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    home = tmp_path / "home"
    home.mkdir()
    fake_python = tmp_path / "refuse-before-capture"
    fake_python.write_text(
        "#!/bin/bash\n"
        "/usr/bin/python3 - <<'PY'\n"
        "import json, os\n"
        "from pathlib import Path\n"
        "path = os.environ.get('SPOKE_LAUNCH_ADMISSION_PATH')\n"
        "if path:\n"
        "    Path(path).write_text(json.dumps({\n"
        "        'status': 'refused',\n"
        "        'token': os.environ['SPOKE_LAUNCH_ADMISSION_TOKEN'],\n"
        "        'reason': 'synthetic conformance refusal',\n"
        "    }))\n"
        "PY\n"
        "exit 17\n"
    )
    fake_python.chmod(0o755)
    registry = tmp_path / "launch_targets.json"
    registry.write_text(
        json.dumps(
            {
                "selected": "reviewed",
                "targets": [
                    {
                        "id": "reviewed",
                        "path": str(repo_root),
                        "env": {"SPOKE_VENV_PYTHON": str(fake_python)},
                    }
                ],
            }
        )
    )
    env = os.environ.copy()
    env.update({"HOME": str(home), "SPOKE_LAUNCH_TARGETS_PATH": str(registry)})

    result = subprocess.run(
        [str(_selected_script_path())],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    admission_receipts = list((home / "Library/Logs").glob("spoke-launch-admission-*.json"))
    assert result.returncode != 0
    assert len(admission_receipts) == 1
    assert json.loads(admission_receipts[0].read_text())["status"] == "refused"


@pytest.mark.parametrize("raw", ["inf", "+inf", "-inf", "nan"])
def test_launcher_admission_timeout_rejects_non_finite_values(raw):
    parse_timeout = _launcher_admission_timeout_parser()

    assert parse_timeout({"SPOKE_LAUNCH_ADMISSION_TIMEOUT_SECONDS": raw}) == 60.0


def test_launcher_admission_timeout_preserves_arbitrary_finite_positive_value():
    parse_timeout = _launcher_admission_timeout_parser()

    assert parse_timeout({"SPOKE_LAUNCH_ADMISSION_TIMEOUT_SECONDS": "3600"}) == 3600.0


def test_launcher_times_out_living_never_admitting_child_without_predecessor_loss(
    tmp_path,
):
    repo_root = Path(__file__).resolve().parent.parent
    home = tmp_path / "home"
    logs = home / "Library/Logs"
    logs.mkdir(parents=True)
    child_pid_path = tmp_path / "child.pid"
    fake_python = tmp_path / "never-admit"
    fake_python.write_text(
        "#!/usr/bin/python3\n"
        "import os, time\n"
        "from pathlib import Path\n"
        "Path(os.environ['SPOKE_NEVER_ADMIT_PID']).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    fake_python.chmod(0o755)
    registry = tmp_path / "launch_targets.json"
    registry.write_text(
        json.dumps(
            {
                "selected": "reviewed",
                "targets": [
                    {
                        "id": "reviewed",
                        "path": str(repo_root),
                        "env": {
                            "SPOKE_LAUNCH_ADMISSION_TIMEOUT_SECONDS": "1.0",
                            "SPOKE_NEVER_ADMIT_PID": str(child_pid_path),
                            "SPOKE_VENV_PYTHON": str(fake_python),
                        },
                    }
                ],
            }
        )
    )
    predecessor = subprocess.Popen(["/bin/sleep", "30"])
    (logs / ".spoke.lock").write_text(str(predecessor.pid))
    env = os.environ.copy()
    env.update({"HOME": str(home), "SPOKE_LAUNCH_TARGETS_PATH": str(registry)})
    try:
        result = subprocess.run(
            [str(_selected_script_path())],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert predecessor.poll() is None, "timeout killed the live predecessor"
        child_pid = int(child_pid_path.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        receipts = list(logs.glob("spoke-launch-admission-*.json"))
        assert len(receipts) == 1
        payload = json.loads(receipts[0].read_text())
        assert payload["status"] == "refused"
        assert payload["phase"] == "admission_timeout"
    finally:
        if predecessor.poll() is None:
            predecessor.terminate()
        predecessor.wait(timeout=3)


def test_named_target_refusal_preserves_live_predecessor(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    home = tmp_path / "home"
    logs = home / "Library/Logs"
    logs.mkdir(parents=True)
    selected = tmp_path / "selected"
    selected_launcher = selected / "scripts/launch-main.sh"
    selected_launcher.parent.mkdir(parents=True)
    selected_launcher.write_text("#!/bin/bash\nexit 99\n")
    selected_launcher.chmod(0o755)
    other = tmp_path / "other"
    other_python = other / ".venv/bin/python"
    other_python.parent.mkdir(parents=True)
    other_python.write_text("#!/bin/bash\nexit 0\n")
    other_python.chmod(0o755)
    registry = tmp_path / "launch_targets.json"
    registry.write_text(
        json.dumps(
            {
                "selected": "selected",
                "targets": [
                    {"id": "selected", "path": str(selected)},
                    {"id": "other", "path": str(other)},
                ],
            }
        )
    )
    predecessor = subprocess.Popen(["/bin/sleep", "30"])
    (logs / ".spoke.lock").write_text(str(predecessor.pid))
    env = os.environ.copy()
    env.update({"HOME": str(home), "SPOKE_LAUNCH_TARGETS_PATH": str(registry)})
    script = repo_root / "scripts/launch-target.sh"
    try:
        result = subprocess.run(
            [str(script), "other"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert predecessor.poll() is None, "refusal killed the live predecessor"
        refusal_log = logs / "spoke-launch-target-refusals.jsonl"
        rows = [json.loads(line) for line in refusal_log.read_text().splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row["expected_target_path"] == str(selected.resolve())
        assert row["phase"] == "named_target_predelegation"
        assert row["reason"] == (
            "requested target 'other' is not the selected target 'selected'"
        )
        assert row["registry_path"] == str(registry.resolve())
        assert row["requested_target_id"] == "other"
        assert row["selected_target_id"] == "selected"
        assert row["status"] == "refused"
        assert isinstance(row["launcher_pid"], int)
        assert row["timestamp"]
    finally:
        if predecessor.poll() is None:
            predecessor.terminate()
        predecessor.wait(timeout=3)


def test_named_target_duplicate_registry_preserves_live_predecessor(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    home = tmp_path / "home"
    logs = home / "Library/Logs"
    logs.mkdir(parents=True)
    selected = tmp_path / "selected"
    launcher = selected / "scripts/launch-main.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/bash\nexit 99\n")
    launcher.chmod(0o755)
    registry = tmp_path / "launch_targets.json"
    duplicate = {"id": "selected", "path": str(selected)}
    registry.write_text(
        json.dumps({"selected": "selected", "targets": [duplicate, duplicate]})
    )
    predecessor = subprocess.Popen(["/bin/sleep", "30"])
    (logs / ".spoke.lock").write_text(str(predecessor.pid))
    env = os.environ.copy()
    env.update({"HOME": str(home), "SPOKE_LAUNCH_TARGETS_PATH": str(registry)})
    try:
        result = subprocess.run(
            [str(repo_root / "scripts/launch-target.sh"), "selected"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "duplicated" in result.stderr
        assert predecessor.poll() is None, "malformed registry killed the predecessor"
        refusal_log = logs / "spoke-launch-target-refusals.jsonl"
        row = json.loads(refusal_log.read_text().splitlines()[-1])
        assert row["status"] == "refused"
        assert row["phase"] == "named_target_predelegation"
        assert row["requested_target_id"] == "selected"
        assert row["registry_path"] == str(registry.resolve())
        assert "duplicated" in row["reason"]
    finally:
        if predecessor.poll() is None:
            predecessor.terminate()
        predecessor.wait(timeout=3)


def test_named_target_selection_change_before_child_admission_preserves_predecessor(
    tmp_path,
):
    repo_root = Path(__file__).resolve().parent.parent
    home = tmp_path / "home"
    logs = home / "Library/Logs"
    logs.mkdir(parents=True)
    selected = tmp_path / "selected"
    launcher = selected / "scripts/launch-main.sh"
    launcher.parent.mkdir(parents=True)
    other = tmp_path / "other"
    other.mkdir()
    registry = tmp_path / "launch_targets.json"
    registry.write_text(
        json.dumps(
            {
                "selected": "selected",
                "targets": [
                    {"id": "selected", "path": str(selected)},
                    {"id": "other", "path": str(other)},
                ],
            }
        )
    )
    launcher.write_text(
        "#!/bin/bash\n"
        f"/usr/bin/python3 - <<'PY'\n"
        "import json\n"
        "from pathlib import Path\n"
        f"path = Path({str(registry)!r})\n"
        "payload = json.loads(path.read_text())\n"
        "payload['selected'] = 'other'\n"
        "path.write_text(json.dumps(payload))\n"
        "PY\n"
        f"exec {str(repo_root / 'scripts/launch-main.sh')!r}\n"
    )
    launcher.chmod(0o755)
    predecessor = subprocess.Popen(["/bin/sleep", "30"])
    (logs / ".spoke.lock").write_text(str(predecessor.pid))
    env = os.environ.copy()
    env.update({"HOME": str(home), "SPOKE_LAUNCH_TARGETS_PATH": str(registry)})
    try:
        result = subprocess.run(
            [str(repo_root / "scripts/launch-target.sh"), "selected"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert predecessor.poll() is None, "supersession killed the live predecessor"
        log_path = logs / "spoke-main-launch.log"
        assert log_path.is_file(), (
            f"missing refusal log; returncode={result.returncode} "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        log_text = log_path.read_text()
        assert "changed during stable launcher handoff" in log_text
    finally:
        if predecessor.poll() is None:
            predecessor.terminate()
        predecessor.wait(timeout=3)


def test_named_target_helper_uses_selected_authority_without_process_teardown():
    source = _target_script_text()

    assert "require_selected_launch_target" in source
    assert "resolve_launch_target" not in source
    assert "os.kill(old_pid" not in source
    assert "SPOKE_EXPECTED_LAUNCH_TARGET_ID" in source
    assert "SPOKE_EXPECTED_LAUNCH_TARGET_PATH" in source


# ── Save selected target ────────────────────────────────────────


class TestSaveSelectedTarget:
    def test_save_and_reload(self, tmp_path, monkeypatch):
        registry = tmp_path / "launch_targets.json"
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        registry.write_text(json.dumps({
            "selected": "old",
            "targets": [
                {"id": "old", "label": "Old", "path": str(tmp_path)},
                {"id": "new", "label": "New", "path": str(worktree)},
            ],
        }))
        monkeypatch.setenv("SPOKE_MAIN_TARGET_PATH", str(tmp_path / "main-target"))
        result = save_selected_launch_target("new", registry)
        assert result is True
        reloaded = load_launch_target_registry(registry)
        assert reloaded["selected"] == "new"

    def test_save_nonexistent_target_returns_false(self, tmp_path):
        registry = tmp_path / "launch_targets.json"
        registry.write_text(json.dumps({
            "selected": "a",
            "targets": [{"id": "a", "label": "A", "path": str(tmp_path)}],
        }))
        assert save_selected_launch_target("nonexistent", registry) is False


# ── Env overrides ────────────────────────────────────────────────


class TestEnvOverrides:
    def test_parse_smoke_env(self, tmp_path):
        env_file = tmp_path / ".spoke-smoke-env"
        env_file.write_text(
            '# comment\n'
            'export SPOKE_COMMAND_URL="http://localhost:8090"\n'
            "SPOKE_TTS_VOICE='casual_female'\n"
            'BARE_KEY=bare_value\n'
        )
        overrides = parse_env_overrides(env_file)
        assert overrides["SPOKE_COMMAND_URL"] == "http://localhost:8090"
        assert overrides["SPOKE_TTS_VOICE"] == "casual_female"
        assert overrides["BARE_KEY"] == "bare_value"

    def test_parse_missing_env_file(self, tmp_path):
        overrides = parse_env_overrides(tmp_path / "nonexistent")
        assert overrides == {}

    def test_parse_empty_env_file(self, tmp_path):
        env_file = tmp_path / ".spoke-smoke-env"
        env_file.write_text("")
        assert parse_env_overrides(env_file) == {}

    def test_parse_skips_blank_and_comment_lines(self, tmp_path):
        env_file = tmp_path / ".spoke-smoke-env"
        env_file.write_text("# only comments\n\n  \n# another\n")
        assert parse_env_overrides(env_file) == {}


# ── Old architecture retired ────────────────────────────────────


class TestOldArchitectureRetired:
    """Verify the old file-based launcher architecture is gone."""

    def test_launch_dev_sh_does_not_exist(self):
        script = Path(__file__).resolve().parent.parent / "scripts" / "launch-dev.sh"
        assert not script.exists(), "launch-dev.sh should be deleted"

    def test_launch_smoke_sh_does_not_exist(self):
        script = Path(__file__).resolve().parent.parent / "scripts" / "launch-smoke.sh"
        assert not script.exists(), "launch-smoke.sh should be deleted"

    def test_launch_main_sh_reads_registry(self):
        text = _main_script_text()
        assert "launch_targets.json" in text

    def test_launch_main_sh_does_not_read_file_targets(self):
        text = _main_script_text()
        assert "main-target" not in text
        assert "dev-target" not in text
        assert "smoke-target" not in text

    def test_launch_main_sh_requires_selected_target(self):
        text = _main_script_text()
        assert "require_selected_launch_target" in text
        assert "Falling back to script checkout" not in text
        assert "FALLBACK_REPO_ROOT" not in text


# ── Secrets env loading ─────────────────────────────────────────


class TestSecretsEnvLoading:
    """The launcher must source ~/.config/spoke/secrets.env into the child
    env before it applies per-worktree .spoke-smoke-env overrides, so that
    machine-wide secrets are available to Automator-launched processes
    that never see the user's shell profile, while still allowing a
    per-worktree smoke env to override a specific key if needed.
    """

    def test_launch_main_sh_reads_secrets_env(self):
        text = _main_script_text()
        assert ".config/spoke/secrets.env" in text, (
            "launch-main.sh must load ~/.config/spoke/secrets.env so "
            "Automator-launched processes receive machine-wide secrets"
        )

    def test_launch_main_sh_loads_secrets_before_smoke_env(self):
        """Per-worktree .spoke-smoke-env must be able to override a secret
        value, so it must be loaded AFTER the machine-wide secrets file."""
        text = _main_script_text()
        secrets_idx = text.find(".config/spoke/secrets.env")
        smoke_idx = text.find(".spoke-smoke-env")
        assert secrets_idx != -1, "secrets.env reference not found"
        assert smoke_idx != -1, ".spoke-smoke-env reference not found"
        assert secrets_idx < smoke_idx, (
            "secrets.env must be loaded before .spoke-smoke-env so that "
            "per-worktree overrides win over machine-wide secrets"
        )

    def test_launch_main_sh_tolerates_missing_secrets_env(self):
        """A box without ~/.config/spoke/secrets.env must still launch —
        the env-file loader must be guarded by an is_file() check and
        the secrets load must route through that loader."""
        text = _main_script_text()
        # The env-file helper must exist and must guard with is_file.
        assert "def _apply_env_file" in text, (
            "launcher must define an _apply_env_file helper so both the "
            "secrets and smoke env blocks share a single guarded loader"
        )
        helper_start = text.find("def _apply_env_file")
        # Inspect the helper body (next ~500 chars is generous).
        helper_body = text[helper_start : helper_start + 500]
        assert "is_file()" in helper_body, (
            "_apply_env_file must guard on path.is_file() so missing "
            "env files don't crash the launcher on fresh boxes"
        )
        # And the secrets file must actually go through that helper.
        assert "_apply_env_file(secrets_env)" in text, (
            "secrets.env must be loaded via _apply_env_file so it "
            "inherits the is_file() guard and shared parser"
        )

    def test_parse_env_overrides_handles_secrets_shape(self, tmp_path):
        """The launcher's own parser must handle the shape specified for
        ~/.config/spoke/secrets.env: bare exports, quoted values, comments,
        blank lines, and literal quote characters inside quoted values."""
        secrets_file = tmp_path / "secrets.env"
        secrets_file.write_text(
            '# Spoke secrets — never committed\n'
            '\n'
            'export GEMINI_API_KEY_INACTIVE="AIzaTESTVALUE"\n'
            'SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY=bare-value-123\n'
            "# trailing comment\n"
            "OPENROUTER_API_KEY='single-quoted'\n"
            'SPOKE_EDGE_SECRET="keeps-trailing-apostrophe\'"\n'
        )
        overrides = parse_env_overrides(secrets_file)
        apply_env_file, child_env = _launcher_apply_env_file()
        apply_env_file(secrets_file)
        assert overrides["GEMINI_API_KEY_INACTIVE"] == "AIzaTESTVALUE"
        assert overrides["SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY"] == "bare-value-123"
        assert overrides["OPENROUTER_API_KEY"] == "single-quoted"
        assert child_env == overrides

    def test_launch_main_env_loader_expands_home_variables(self, tmp_path, monkeypatch):
        """The Automator launcher must resolve smoke env paths like the
        menubar launcher; otherwise real operator-ping rows are invisible and
        the fallback smoke token wins."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        smoke_env = tmp_path / ".spoke-smoke-env"
        smoke_env.write_text(
            'export SPOKE_OPERATOR_PING_EVENTS_PATH="$HOME/.local/state/epistaxis/events.jsonl"\n',
            encoding="utf-8",
        )
        apply_env_file, child_env = _launcher_apply_env_file()

        apply_env_file(smoke_env)

        assert child_env["SPOKE_OPERATOR_PING_EVENTS_PATH"] == (
            str(home / ".local/state/epistaxis/events.jsonl")
        )


class TestLaunchTargetSecretsEnvLoading:
    """Named-target compatibility delegates environment loading to launch-main."""

    def test_launch_target_sh_reads_secrets_env(self):
        text = _main_script_text()
        assert ".config/spoke/secrets.env" in text, (
            "launch-target.sh must load ~/.config/spoke/secrets.env before "
            "starting the selected target"
        )

    def test_launch_target_sh_loads_secrets_before_smoke_env(self):
        text = _main_script_text()
        secrets_idx = text.find(".config/spoke/secrets.env")
        smoke_idx = text.find(".spoke-smoke-env")
        assert secrets_idx != -1, "secrets.env reference not found"
        assert smoke_idx != -1, ".spoke-smoke-env reference not found"
        assert secrets_idx < smoke_idx, (
            "launch-target.sh must load secrets.env before .spoke-smoke-env "
            "so per-worktree overrides still win"
        )

    def test_launch_target_sh_applies_secrets_with_shared_parser(self):
        text = _main_script_text()
        assert "_apply_env_file(secrets_env)" in text, (
            "the delegated launch-main route must apply secrets through its one "
            "environment-file parser"
        )


class TestRegistryTargetEnvLoading:
    """Invocation-scoped target env must win over shared worktree smoke state."""

    @pytest.mark.parametrize("script_text", [_main_script_text])
    def test_launchers_apply_target_env_after_worktree_smoke_env(self, script_text):
        text = script_text()
        smoke_idx = text.find(".spoke-smoke-env")
        target_env_idx = text.find('target.get("env")')

        assert smoke_idx != -1, ".spoke-smoke-env reference not found"
        assert target_env_idx != -1, "registry target env application not found"
        assert smoke_idx < target_env_idx, (
            "target-scoped env must be applied after worktree smoke env so one "
            "launch target can disable inherited fixtures without mutating the worktree"
        )

    @pytest.mark.parametrize("script_text", [_main_script_text])
    def test_launchers_clear_inherited_models_before_target_env(self, script_text):
        text = script_text()
        clear_idx = text.rfind('child_env.pop("SPOKE_PREVIEW_MODEL"')
        target_env_idx = text.find('target.get("env")')

        assert clear_idx != -1, "inherited model cleanup not found"
        assert target_env_idx != -1, "registry target env application not found"
        assert clear_idx < target_env_idx, (
            "inherited route cleanup must happen before explicit target env so "
            "the launcher cannot silently erase a requested target model route"
        )

    def test_main_rejects_invalid_selected_target_before_applying_authority(self):
        text = _main_script_text()

        require_idx = text.find("target = require_selected_launch_target(targets_file)")
        reject_idx = text.find("except LaunchTargetUnavailable as exc:")
        effective_idx = text.find("effective_target = target")

        assert require_idx != -1
        assert reject_idx != -1
        assert effective_idx != -1
        assert require_idx < reject_idx < effective_idx
        assert "raise SystemExit(1)" in text[reject_idx:effective_idx]
        assert 'target_env = effective_target.get("env")' in text
        assert 'target_id=effective_target.get("id", "selected")' in text

    @pytest.mark.parametrize("script_text", [_main_script_text])
    def test_witness_route_log_does_not_echo_target_env_values(self, script_text):
        text = script_text()

        assert "Retina Lasso auto witness route:" in text
        assert "SPOKE_PERCEPTASIA_THROUGHGLASS_SMOKE={" not in text


class TestLauncherPythonOverride:
    """Both launcher paths must honor per-worktree Python overrides.

    Smoke worktrees can set SPOKE_VENV_PYTHON in .spoke-smoke-env when their
    fresh .venv is absent or still bootstrapping. The primary launcher and the
    explicit target launcher must therefore choose Python from the same child
    environment after secrets/worktree overrides have been applied.
    """

    def test_launch_main_honors_spoke_venv_python_override(self):
        text = _main_script_text()
        assert 'child_env.get("SPOKE_VENV_PYTHON"' in text, (
            "launch-main.sh must honor SPOKE_VENV_PYTHON from .spoke-smoke-env "
            "before falling back to the target worktree .venv"
        )

    def test_launch_target_honors_spoke_venv_python_override(self):
        text = _target_script_text()
        assert 'launcher = target_path / "scripts" / "launch-main.sh"' in text
        assert "os.execve(launcher" in text


class TestLauncherRetinaLassoWitness:
    """Launcher-selected smoke surfaces may arm a trace-aligned visual
    witness sidecar. That is intentionally launcher-side instrumentation:
    the app does not need to learn about the visual witness, but a selected
    target can still mean "Spoke plus Retina Lasso" when its smoke env asks
    for it.
    """

    @pytest.mark.parametrize("script_text", [_main_script_text])
    def test_launchers_can_arm_retina_lasso_witness(self, script_text):
        text = script_text()
        assert "SPOKE_RETINA_LASSO_AUTO_WITNESS" in text
        assert "command-overlay-retina-lasso-witness.py" in text
        assert "SPOKE_COMMAND_OVERLAY_TRACE_PATH" in text
        assert "SPOKE_RETINA_LASSO_PERCEPTASIA_ROOT" in text
        assert "SPOKE_RETINA_LASSO_OUTPUT_ROOT" in text
        assert 'Path("/opt/homebrew/bin/uv")' in text
        assert 'witness_env["UV_BIN"] = str(uv_bin)' in text
        assert "if uv_bin is not None:" in text
        assert "SPOKE_RETINA_LASSO_RETARGET_DURING_DISMISS_REPEATS" in text
        assert "SPOKE_RETINA_LASSO_HAMMER_TOGGLES" in text
        assert "--retarget-during-dismiss-repeats" in text
        assert "--pre-hammer-delay" in text
        assert "SPOKE_RETINA_LASSO_OPEN_READY_TIMEOUT_SECONDS" in text
        assert "--open-ready-timeout" in text

    @pytest.mark.parametrize("script_text", [_main_script_text])
    def test_capture_first_witness_suppresses_post_trigger_watch_mode(self, script_text):
        text = script_text()
        assert "capture-first stimulus armed" in text
        assert 'and not capture_first_stimulus' in text

    @pytest.mark.parametrize("script_text", [_main_script_text])
    def test_retina_lasso_witness_preserves_capture_boundary(self, script_text):
        text = script_text()
        assert "SPOKE_RETINA_LASSO_CAPTURE_MODE" in text
        assert "SPOKE_RETINA_LASSO_CAPTURE_RECT" in text
        assert "--capture-mode" in text
        assert "--capture-rect" in text

    @pytest.mark.parametrize("script_text", [_main_script_text])
    def test_retina_lasso_witness_is_sidecar_only(self, script_text):
        text = script_text()
        app_launch = text.find('"-m", "spoke"')
        witness_launch = text.find("        _start_retina_lasso_witness(")
        assert app_launch != -1
        assert witness_launch != -1
        assert app_launch < witness_launch

    @pytest.mark.parametrize("script_text", [_main_script_text])
    def test_throughglass_smoke_uses_throughglass_pixel_witness(self, script_text):
        text = script_text()
        assert "SPOKE_PERCEPTASIA_THROUGHGLASS_SMOKE" in text
        assert "spoke.perceptasia_throughglass_witness" in text
        assert "SPOKE_PERCEPTASIA_THROUGHGLASS_WITNESS_OUTPUT_ROOT" in text

    @pytest.mark.parametrize("script_text", [_main_script_text])
    def test_throughglass_smoke_can_run_trace_watch_for_later_operator_actions(self, script_text):
        text = script_text()
        assert 'if throughglass_witness and _env_flag(child_env, "SPOKE_RETINA_LASSO_WATCH_TRACE")' in text
        assert 'args.append("--watch-trace")' in text
        assert '"--event-capture-duration"' in text

    def test_target_witness_wrapper_routes_throughglass_for_stable_launcher(self):
        """The stable hotkey launcher may call the selected worktree's legacy
        command-overlay witness wrapper before the launcher itself has learned
        a newer smoke route. The target-side wrapper must still honor the
        Throughglass smoke env instead of silently producing command-overlay
        evidence.
        """
        script = Path(__file__).resolve().parent.parent / "scripts" / "command-overlay-retina-lasso-witness.py"
        text = script.read_text(encoding="utf-8")

        assert "SPOKE_PERCEPTASIA_THROUGHGLASS_SMOKE" in text
        assert "perceptasia_throughglass_witness" in text
        assert "retina_lasso_witness" in text

    def test_target_witness_wrapper_dispatches_from_env(self, monkeypatch):
        script = Path(__file__).resolve().parent.parent / "scripts" / "command-overlay-retina-lasso-witness.py"
        spec = importlib.util.spec_from_file_location("command_overlay_retina_lasso_witness_test", script)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        calls: list[str] = []
        monkeypatch.setattr(module, "throughglass_main", lambda: calls.append("throughglass") or 17)
        monkeypatch.setattr(module, "command_overlay_main", lambda: calls.append("command") or 23)

        monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_SMOKE", "1")
        assert module.main() == 17
        monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_SMOKE", "0")
        assert module.main() == 23
        assert calls == ["throughglass", "command"]


class TestSecretsEnvExampleTemplate:
    """A committed .example template documents the expected shape without
    leaking real values. This is the discoverability contract on new boxes."""

    def _template_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "scripts" / "secrets.env.example"

    def test_template_exists(self):
        assert self._template_path().exists(), (
            "scripts/secrets.env.example must exist as a tracked template "
            "for ~/.config/spoke/secrets.env"
        )

    def test_template_lists_gemini_alias(self):
        text = self._template_path().read_text()
        assert "GEMINI_API_KEY_INACTIVE" in text, (
            "template must document the pseudonym alias so users know "
            "to populate the spoke-only Gemini key"
        )

    def test_template_has_no_real_values(self):
        """Every export in the template must have an empty value."""
        text = self._template_path().read_text()
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            _key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            assert value == "", (
                f"template line '{raw_line}' has a non-empty value; "
                "templates must ship empty to prevent accidental secret commits"
            )
