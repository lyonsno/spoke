"""Contract tests for the launcher architecture.

Tests the launcher registry contract (spoke/launch_targets.py) and
verifies the old file-based launcher architecture is retired.
"""

import json
import os
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

    def test_launch_main_sh_has_fallback(self):
        text = _main_script_text()
        assert "FALLBACK_REPO_ROOT" in text


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
        """The same parser used for .spoke-smoke-env must handle the shape
        we specify for ~/.config/spoke/secrets.env: bare exports, quoted
        values, comments, blank lines."""
        secrets_file = tmp_path / "secrets.env"
        secrets_file.write_text(
            '# Spoke secrets — never committed\n'
            '\n'
            'export GEMINI_API_KEY_INACTIVE="AIzaTESTVALUE"\n'
            'SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY=bare-value-123\n'
            "# trailing comment\n"
            "OPENROUTER_API_KEY='single-quoted'\n"
        )
        overrides = parse_env_overrides(secrets_file)
        assert overrides["GEMINI_API_KEY_INACTIVE"] == "AIzaTESTVALUE"
        assert overrides["SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY"] == "bare-value-123"
        assert overrides["OPENROUTER_API_KEY"] == "single-quoted"


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
