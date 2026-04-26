from pathlib import Path
import subprocess

import pytest


def _tracked_ds_store_paths(repo_root):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None

    return [
        path
        for path in result.stdout.split("\0")
        if path and Path(path).name == ".DS_Store"
    ]


def test_tracked_ds_store_paths_detects_nested_files(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="README.md\0nested/.DS_Store\0spoke/app.py\0",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _tracked_ds_store_paths(Path("/repo")) == ["nested/.DS_Store"]


def test_tracked_ds_store_paths_allows_non_git_source_tree(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _tracked_ds_store_paths(Path("/repo")) is None


def test_repo_does_not_track_ds_store():
    repo_root = Path(__file__).resolve().parents[1]
    tracked_ds_store_paths = _tracked_ds_store_paths(repo_root)

    if tracked_ds_store_paths is None:
        pytest.skip("repo hygiene check requires a Git checkout")

    assert tracked_ds_store_paths == [], (
        ".DS_Store should not be tracked by the repo: "
        + ", ".join(tracked_ds_store_paths)
    )
