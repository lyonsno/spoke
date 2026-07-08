import subprocess
from pathlib import Path


_PRIVATE_COORDINATION_TOKENS = (
    ("epi" + "staxis").encode("utf-8"),
    ("epi" + "stax").encode("utf-8"),
    ("warp" + "storm").encode("utf-8"),
    ("pit" + "boss").encode("utf-8"),
)


def _tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [part.decode("utf-8") for part in result.stdout.split(b"\0") if part]


def test_public_tree_masks_private_coordination_backend_name():
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for relpath in _tracked_files(root):
        path = root / relpath
        if not path.exists():
            continue
        rel_lower = relpath.lower().encode("utf-8")
        if any(needle in rel_lower for needle in _PRIVATE_COORDINATION_TOKENS):
            offenders.append(relpath)
            continue
        data = path.read_bytes().lower()
        if any(needle in data for needle in _PRIVATE_COORDINATION_TOKENS):
            offenders.append(relpath)

    assert offenders == []
