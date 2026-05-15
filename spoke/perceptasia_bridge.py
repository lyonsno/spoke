"""Shared-file consumer for Perceptasia visual selection state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SELECTION_PATH = (
    Path.home() / ".local" / "state" / "perceptasia" / "selection.json"
)


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _kind_from_selected(selected: str) -> str:
    prefix, separator, _rest = selected.partition(":")
    return prefix if separator and prefix else ""


def _selection_label(selected: str) -> str:
    _prefix, separator, rest = selected.partition(":")
    return rest if separator and rest else selected


@dataclass(frozen=True)
class PerceptasiaSelection:
    selected: str | None = None
    kind: str = ""
    name: str = ""
    lifecycle: str = ""
    project: str = ""
    neighbors: tuple[str, ...] = ()
    timestamp: str = ""
    available: bool = False
    error: str = ""

    @property
    def active(self) -> bool:
        return bool(self.selected)

    @property
    def status_line(self) -> str:
        if self.error:
            return "Perceptasia: selection unreadable"
        if not self.available:
            return "Perceptasia: no selection file"
        if not self.selected:
            return "Perceptasia: no visual selection"
        kind = self.kind or _kind_from_selected(self.selected)
        label = self.name or _selection_label(self.selected)
        return f"Perceptasia: {kind} {label}".strip()

    @property
    def prompt_context(self) -> str:
        if not self.active or not self.selected:
            return ""
        lines = ["Perceptasia visual selection context:"]
        lines.append(f"- selected: {self.selected}")
        if self.kind:
            lines.append(f"- kind: {self.kind}")
        if self.name:
            lines.append(f"- name: {self.name}")
        if self.project:
            lines.append(f"- project: {self.project}")
        if self.lifecycle:
            lines.append(f"- lifecycle: {self.lifecycle}")
        if self.timestamp:
            lines.append(f"- timestamp: {self.timestamp}")
        for neighbor in self.neighbors:
            lines.append(f"- neighbor: {neighbor}")
        lines.append("- authority: visual selection is routing context only; it is not write authority")
        return "\n".join(lines)


def _selection_from_mapping(data: dict[str, Any]) -> PerceptasiaSelection:
    selected_value = data.get("selected")
    selected = selected_value if isinstance(selected_value, str) and selected_value else None
    kind = _string(data.get("kind")) or (_kind_from_selected(selected) if selected else "")
    project = _string(data.get("project")) or _string(data.get("focused_project"))
    return PerceptasiaSelection(
        selected=selected,
        kind=kind,
        name=_string(data.get("name")),
        lifecycle=_string(data.get("lifecycle")),
        project=project,
        neighbors=_strings(data.get("neighbors")),
        timestamp=_string(data.get("timestamp")),
        available=True,
    )


def load_perceptasia_selection(
    path: Path | str | None = None,
) -> PerceptasiaSelection:
    """Read Perceptasia's shared selection file without requiring the app to run."""
    selection_path = Path(path).expanduser() if path is not None else DEFAULT_SELECTION_PATH
    if not selection_path.exists():
        return PerceptasiaSelection(available=False)
    try:
        data = json.loads(selection_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return PerceptasiaSelection(available=True, error=str(exc))
    if not isinstance(data, dict):
        return PerceptasiaSelection(available=True, error="selection root is not an object")
    return _selection_from_mapping(data)
