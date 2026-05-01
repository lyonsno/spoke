"""Provider-agnostic Agent Shell primitive data contracts."""

from __future__ import annotations

from typing import Any

from .agent_thread_cards import AgentThreadCard, card_display_contract


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _card_mapping(card: AgentThreadCard | dict[str, Any]) -> dict[str, Any]:
    if isinstance(card, AgentThreadCard):
        return card.to_dict()
    return card if isinstance(card, dict) else {}


def _provider_session_id(card: dict[str, Any]) -> str:
    return (
        _string(card.get("provider_session_id"))
        or _string(card.get("thread_id"))
        or _string(card.get("id"))
    )


def _chrome(card: dict[str, Any]) -> dict[str, str]:
    return {
        "header": _string(card.get("title")),
        "footer": _string(card.get("activity_line")),
        "topos": _string(card.get("topos")),
        "worktree": _string(card.get("worktree")),
        "model": _string(card.get("model")),
    }


def _geometry(*, selected: bool, index: int) -> dict[str, float | int | str]:
    return {
        "anchor": "right" if selected else "bottom",
        "priority": 0 if selected else 100 + index,
        "preferred_width": 420.0 if selected else 180.0,
        "preferred_height": 160.0 if selected else 44.0,
        "min_width": 180.0 if selected else 92.0,
        "min_height": 96.0 if selected else 44.0,
    }


def _material(*, selected: bool, readiness: str) -> dict[str, float | str]:
    active = readiness in {"working", "waiting"}
    prominence = "selected" if selected else "active" if active else "inactive"
    return {
        "style": "assistant_shell" if selected else "thread_card",
        "prominence": prominence,
        "optical_displacement": 0.7 if selected else 0.45 if active else 0.2,
        "corner_radius": 14.0 if selected else 8.0,
    }


def _display(display: dict[str, Any]) -> dict[str, Any]:
    secondary = _string(display.get("compact_text")) or _string(display.get("bearing"))
    return {
        "display_state": _string(display.get("display_state")),
        "primary_text": _string(display.get("primary_text")),
        "secondary_text": secondary,
        "show_latest_response": bool(display.get("show_latest_response")),
    }


def build_agent_shell_primitives(
    cards: list[AgentThreadCard | dict[str, Any]]
    | tuple[AgentThreadCard | dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """Build renderer-consumable Agent Shell primitive dictionaries.

    This function is a contract adapter: it consumes card truth and display
    semantics, then emits serializable primitive data. It does not own layout,
    materialization, backend state, provider catalogs, or transcript scraping.
    """
    primitives: list[dict[str, Any]] = []
    for index, source_card in enumerate(cards):
        card = _card_mapping(source_card)
        if not card:
            continue
        selected = bool(card.get("selected"))
        display = card_display_contract(card, selected=selected)
        provider_session_id = _provider_session_id(card)
        readiness = _string(display.get("readiness")) or _string(card.get("readiness"))
        show_latest_response = bool(display.get("show_latest_response"))
        primitives.append(
            {
                "id": provider_session_id,
                "kind": "selected_thread" if selected else "thread_card",
                "provider": _string(card.get("provider")),
                "provider_session_id": provider_session_id,
                "selected": selected,
                "readiness": readiness,
                "title": _string(card.get("title")),
                "bearing": _string(card.get("bearing")),
                "activity_line": _string(card.get("activity_line")),
                "latest_response": _string(display.get("latest_response"))
                if show_latest_response
                else "",
                "chrome": _chrome(card),
                "display": _display(display),
                "geometry": _geometry(selected=selected, index=index),
                "material": _material(selected=selected, readiness=readiness),
            }
        )
    return primitives
