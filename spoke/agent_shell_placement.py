"""Deterministic placement placeholders for Agent Shell primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SEMANTIC_PLACEMENT_STATUS = "placeholder"
SEMANTIC_ANCHOR = "semantic"
SELECTED_ANCHOR = "right"
INACTIVE_ANCHOR = "bottom"
SELECTED_PRIORITY = 0
INACTIVE_PRIORITY_START = 100


@dataclass(frozen=True)
class SemanticPlacementPlaceholder:
    status: str
    fallback_anchor: str
    provider_session_id: str
    primitive_id: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _geometry(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _is_selected(primitive: dict[str, Any]) -> bool:
    return bool(primitive.get("selected")) or primitive.get("kind") == "selected_thread"


def _primitive_id(primitive: dict[str, Any]) -> str:
    return (
        _string(primitive.get("id"))
        or _string(primitive.get("provider_session_id"))
        or _string(primitive.get("thread_id"))
    )


def _provider_session_id(primitive: dict[str, Any]) -> str:
    return _string(primitive.get("provider_session_id")) or _primitive_id(primitive)


def _fallback_anchor(*, selected: bool) -> str:
    return SELECTED_ANCHOR if selected else INACTIVE_ANCHOR


def _semantic_placeholder(
    primitive: dict[str, Any],
    *,
    fallback_anchor: str,
) -> dict[str, str]:
    return SemanticPlacementPlaceholder(
        status=SEMANTIC_PLACEMENT_STATUS,
        fallback_anchor=fallback_anchor,
        provider_session_id=_provider_session_id(primitive),
        primitive_id=_primitive_id(primitive),
    ).to_dict()


def apply_initial_agent_shell_placement(
    primitives: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """Fill deterministic placement geometry without changing card truth.

    This is deliberately a data-only placeholder. It chooses stable anchors and
    priorities for current renderers while preserving ``anchor == "semantic"``
    as the future VLM/semantic handoff point.
    """
    placed: list[dict[str, Any]] = []
    inactive_index = 0
    for primitive in primitives:
        if not isinstance(primitive, dict):
            continue
        placed_primitive = dict(primitive)
        geometry = _geometry(primitive.get("geometry"))
        selected = _is_selected(placed_primitive)
        fallback_anchor = _fallback_anchor(selected=selected)
        existing_anchor = _string(geometry.get("anchor"))
        anchor = SEMANTIC_ANCHOR if existing_anchor == SEMANTIC_ANCHOR else fallback_anchor
        priority = (
            SELECTED_PRIORITY
            if selected
            else INACTIVE_PRIORITY_START + inactive_index
        )
        if not selected:
            inactive_index += 1
        geometry["anchor"] = anchor
        geometry["priority"] = priority
        if anchor == SEMANTIC_ANCHOR:
            geometry["semantic_placement"] = _semantic_placeholder(
                placed_primitive,
                fallback_anchor=fallback_anchor,
            )
        placed_primitive["geometry"] = geometry
        placed.append(placed_primitive)
    return placed
