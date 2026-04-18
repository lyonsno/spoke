from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
SURFACES = REPO_ROOT / "docs" / "documentation_surfaces.toml"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_surfaces() -> dict:
    with SURFACES.open("rb") as fh:
        return tomllib.load(fh)


def test_documentation_surface_manifest_exists_and_declares_nonpublic_routes():
    manifest = load_surfaces()
    capabilities = manifest["capabilities"]

    repair = capabilities["bounded_post_transcription_repair_pass"]
    assert repair["audience"] == "developer"
    assert repair["public_readme"] == "omit"
    assert repair["canonical_surface"] == "docs/developer-operator-surfaces.md"

    smoke = capabilities["smoke_surface_runtime_affordances"]
    assert smoke["audience"] == "operator"
    assert smoke["public_readme"] == "omit"
    assert smoke["canonical_surface"] == "docs/developer-operator-surfaces.md"


def test_nonpublic_capabilities_have_canonical_surface_markers():
    manifest = load_surfaces()

    for capability in manifest["capabilities"].values():
        canonical_path = REPO_ROOT / capability["canonical_surface"]
        text = read_text(canonical_path)
        for marker in capability["canonical_markers"]:
            assert marker in text


def test_public_readme_defers_to_routed_nonpublic_surfaces():
    manifest = load_surfaces()
    readme = read_text(README)

    assert "product-facing" in readme
    assert "docs/developer-operator-surfaces.md" in readme

    for capability in manifest["capabilities"].values():
        assert capability["public_readme"] == "omit"
        for marker in capability["public_readme_absent_markers"]:
            assert marker not in readme
