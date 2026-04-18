"""Helpers for routing documentation by audience and seeding manifest entries."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any


_SECTION_RE = re.compile(r"^## (.+?)\n(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_OPERATOR_HINTS = (
    "smoke",
    "launch",
    "hud",
    "branch",
    "runtime",
    "operator",
)


def slugify_heading(heading: str) -> str:
    return _NON_ALNUM_RE.sub("_", heading.strip().lower()).strip("_")


def extract_sections(markdown: str) -> list[tuple[str, str]]:
    return [(heading.strip(), body.strip()) for heading, body in _SECTION_RE.findall(markdown)]


def infer_audience(source_path: Path, heading: str) -> str:
    lower_path = source_path.as_posix().lower()
    lower_heading = heading.lower()
    if source_path.name.lower() == "readme.md":
        return "public"
    if any(token in lower_heading for token in _OPERATOR_HINTS):
        return "operator"
    if "local-smoke-runbook" in lower_path:
        return "operator"
    return "developer"


def infer_public_readme(source_path: Path, audience: str) -> str:
    if source_path.name.lower() == "readme.md" or audience == "public":
        return "include"
    return "omit"


def seed_entries_from_markdown(
    source_path: Path,
    markdown: str,
    existing_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    existing_ids = existing_ids or set()
    entries: dict[str, dict[str, Any]] = {}

    for heading, _body in extract_sections(markdown):
        capability_id = slugify_heading(heading)
        if not capability_id or capability_id in existing_ids:
            continue
        audience = infer_audience(source_path, heading)
        public_readme = infer_public_readme(source_path, audience)
        entries[capability_id] = {
            "audience": audience,
            "canonical_surface": source_path.as_posix(),
            "public_readme": public_readme,
            "reason": (
                f"Seeded from {source_path.as_posix()} section '{heading}'. "
                "Refine this route before treating it as a settled contract."
            ),
            "revisit_when": (
                "Review this seeded route when the audience allocation or "
                "canonical surface becomes clearer."
            ),
            "canonical_markers": [heading],
            "public_readme_absent_markers": [] if public_readme == "include" else [heading],
            "seeded": True,
        }
    return entries


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"capabilities": {}}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def render_manifest(manifest: dict[str, Any]) -> str:
    lines: list[str] = []
    for capability_id in sorted(manifest.get("capabilities", {})):
        entry = manifest["capabilities"][capability_id]
        lines.append(f"[capabilities.{capability_id}]")
        for key in (
            "audience",
            "canonical_surface",
            "public_readme",
            "reason",
            "revisit_when",
        ):
            lines.append(f'{key} = "{_escape_toml_string(str(entry[key]))}"')
        lines.append(
            "canonical_markers = ["
            + ", ".join(f'"{_escape_toml_string(item)}"' for item in entry["canonical_markers"])
            + "]"
        )
        lines.append(
            "public_readme_absent_markers = ["
            + ", ".join(
                f'"{_escape_toml_string(item)}"' for item in entry["public_readme_absent_markers"]
            )
            + "]"
        )
        if "seeded" in entry:
            lines.append(f"seeded = {str(bool(entry['seeded'])).lower()}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def merge_seeded_entries(
    manifest: dict[str, Any],
    source_path: Path,
    markdown: str,
) -> dict[str, Any]:
    capabilities = dict(manifest.get("capabilities", {}))
    seeded = seed_entries_from_markdown(
        source_path=source_path,
        markdown=markdown,
        existing_ids=set(capabilities),
    )
    capabilities.update(seeded)
    return {"capabilities": capabilities}


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
