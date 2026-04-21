import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
MANIFEST = DOCS / "review_surfaces.toml"


def load_manifest() -> dict:
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh)


def repo_relative_path(rel_path: str) -> Path:
    path = (REPO_ROOT / rel_path).resolve()
    repo_root = REPO_ROOT.resolve()
    if not path.is_relative_to(repo_root):
        raise ValueError(f"path escapes repo root: {rel_path}")
    return path


def test_review_surface_paths_are_repo_relative():
    assert repo_relative_path("docs/review-authority-surfaces.md") == (
        DOCS / "review-authority-surfaces.md"
    )
    assert repo_relative_path("spoke/metal_warp.py") == REPO_ROOT / "spoke/metal_warp.py"
    with pytest.raises(ValueError):
        repo_relative_path("../outside.md")


def test_review_surface_manifest_declares_optical_shell_renderer_authority():
    manifest = load_manifest()
    surfaces = manifest["surfaces"]

    optical_shell = surfaces["optical_shell_renderer_authority"]
    assert optical_shell["canonical_surface"] == "docs/review-authority-surfaces.md"
    assert optical_shell["authoritative_paths"] == ["spoke/metal_warp.py"]
    assert optical_shell["fallback_paths"] == ["spoke/backdrop_stream.py"]
    assert optical_shell["equivalence"] == "divergence_allowed_during_tuning"
    assert optical_shell["default_review_action"] == "design-pressure"


def test_optical_shell_review_surface_doc_states_authority_and_revisit_boundary():
    manifest = load_manifest()
    optical_shell = manifest["surfaces"]["optical_shell_renderer_authority"]
    canonical_surface = repo_relative_path(optical_shell["canonical_surface"])
    assert canonical_surface.is_file()

    doc_text = canonical_surface.read_text(encoding="utf-8")
    assert "Optical Shell Renderer Authority" in doc_text
    assert "spoke/metal_warp.py" in doc_text
    assert "spoke/backdrop_stream.py" in doc_text
    assert "design pressure" in doc_text.lower()
    assert "diverge during tuning" in doc_text.lower()
