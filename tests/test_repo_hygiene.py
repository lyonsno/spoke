from pathlib import Path
import subprocess


def test_repo_does_not_track_ds_store():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", ".DS_Store"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, ".DS_Store should not be tracked by the repo"
