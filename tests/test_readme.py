from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"


def read_readme() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_mentions_bounded_post_transcription_repair_pass():
    text = read_readme()

    assert "bounded post-transcription repair pass" in text
    assert "project-specific vocabulary" in text


def test_readme_mentions_smoke_surface_menubar_affordances():
    text = read_readme()

    assert "launch-target switching" in text
    assert "source/branch visibility" in text
    assert "Terror Form" in text
