import tomllib
from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"
REPO_ROOT = README.parent
DOCS = README.parent / "docs"
MANIFEST = DOCS / "documentation_surfaces.toml"


def read_readme() -> str:
    return README.read_text(encoding="utf-8")


def load_manifest() -> dict:
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh)


def canonical_surface_path(canonical_surface: str) -> Path:
    surface_path = (REPO_ROOT / canonical_surface).resolve()
    repo_root = REPO_ROOT.resolve()
    if not surface_path.is_relative_to(repo_root):
        raise ValueError(f"canonical surface escapes repo root: {canonical_surface}")
    return surface_path


def test_canonical_surface_paths_are_repo_relative():
    assert canonical_surface_path("README.md") == README
    assert canonical_surface_path("docs/local-smoke-runbook.md") == DOCS / "local-smoke-runbook.md"


def test_topothesia_manifest_routes_non_public_spoke_capabilities():
    manifest = load_manifest()
    capabilities = manifest["capabilities"]

    repair = capabilities["bounded_post_transcription_repair_pass"]
    assert repair["canonical_surface"] == "docs/developer-operator-surfaces.md"
    assert repair["public_readme"] == "omit"
    assert "Bounded Post-Transcription Repair Pass" in repair["canonical_markers"]
    assert "Bounded Post-Transcription Repair Pass" in repair["public_readme_absent_markers"]

    smoke = capabilities["smoke_surface_runtime_affordances"]
    assert smoke["canonical_surface"] == "docs/local-smoke-runbook.md"
    assert smoke["public_readme"] == "omit"
    assert "Terror Form" in smoke["public_readme_absent_markers"]


def test_omitted_capabilities_live_off_readme_in_their_canonical_surfaces():
    text = read_readme()
    manifest = load_manifest()

    repair_surface = canonical_surface_path(
        manifest["capabilities"]["bounded_post_transcription_repair_pass"]["canonical_surface"]
    )
    smoke_surface = canonical_surface_path(
        manifest["capabilities"]["smoke_surface_runtime_affordances"]["canonical_surface"]
    )
    assert repair_surface.is_file()
    assert smoke_surface.is_file()

    assert "bounded post-transcription repair pass" not in text.lower()
    assert "launch-target switching" not in text
    assert "source/branch visibility" not in text
    assert "Terror Form" not in text

    repair_text = repair_surface.read_text(encoding="utf-8")
    assert "Bounded Post-Transcription Repair Pass" in repair_text
    assert "project-specific vocabulary" in repair_text

    smoke_text = smoke_surface.read_text(encoding="utf-8")
    assert "launch-target switching" in smoke_text
    assert "source/branch visibility" in smoke_text
    assert "Terror Form" in smoke_text


def test_readme_mentions_current_public_assistant_capabilities():
    text = read_readme().lower()

    assert "brave search" in text
    assert "multimodal" in text
    assert "subagent" in text
    assert "narrator" in text
    assert "compact" in text


def test_readme_mentions_brave_search_api_key_setup():
    text = read_readme()

    assert "BRAVE_SEARCH_API_KEY" in text or "SPOKE_BRAVE_SEARCH_API_KEY" in text
