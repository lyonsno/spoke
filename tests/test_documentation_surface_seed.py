from pathlib import Path

from spoke.documentation_surfaces import seed_entries_from_markdown


def test_seed_entries_infers_routes_from_markdown_sections():
    markdown = """
# Developer And Operator Surfaces

## Bounded Post-Transcription Repair Pass

`spoke` keeps a bounded post-transcription repair pass for recurring
project-specific vocabulary observed in real logs.

## Smoke-Surface Runtime Affordances

On local smoke surfaces, the menubar also exposes launch-target switching,
source/branch visibility, and the status HUD (`Terror Form`) so you can confirm
which runtime surface is actually live.
"""

    entries = seed_entries_from_markdown(
        source_path=Path("docs/developer-operator-surfaces.md"),
        markdown=markdown,
    )

    repair = entries["bounded_post_transcription_repair_pass"]
    assert repair["audience"] == "developer"
    assert repair["canonical_surface"] == "docs/developer-operator-surfaces.md"
    assert repair["public_readme"] == "omit"
    assert repair["canonical_markers"] == ["Bounded Post-Transcription Repair Pass"]
    assert repair["public_readme_absent_markers"] == [
        "Bounded Post-Transcription Repair Pass"
    ]

    smoke = entries["smoke_surface_runtime_affordances"]
    assert smoke["audience"] == "operator"
    assert smoke["canonical_surface"] == "docs/developer-operator-surfaces.md"
    assert smoke["public_readme"] == "omit"
    assert smoke["canonical_markers"] == ["Smoke-Surface Runtime Affordances"]


def test_seed_entries_skips_existing_ids():
    markdown = """
## Smoke-Surface Runtime Affordances

Operator details.
"""

    entries = seed_entries_from_markdown(
        source_path=Path("docs/developer-operator-surfaces.md"),
        markdown=markdown,
        existing_ids={"smoke_surface_runtime_affordances"},
    )

    assert entries == {}
