"""Compositor-facing Agent Shell partyline HUD render data."""

from __future__ import annotations

from typing import Any

from .agent_thread_cards import card_display_contract

_CARD_HEIGHT_POINTS = 44.0
_CARD_MIN_WIDTH_POINTS = 92.0
_CARD_MAX_WIDTH_POINTS = 180.0
_CARD_GAP_POINTS = 8.0
_CARD_MARGIN_POINTS = 12.0
_MAX_VISIBLE_CARDS = 4


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _number(value: Any, fallback: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else fallback


def _thread_id(card: dict[str, Any]) -> str:
    return (
        _string(card.get("thread_id"))
        or _string(card.get("provider_session_id"))
        or _string(card.get("id"))
    )


def _selected(card: dict[str, Any]) -> bool:
    return bool(card.get("selected"))


def _surface_text(display: dict[str, Any]) -> str:
    compact = _string(display.get("compact_text")).strip()
    if compact:
        return compact
    primary = _string(display.get("primary_text")).strip()
    return " ".join(primary.split())


def _visible_cards(cards: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
    selected = [card for card in cards if _selected(card)]
    inactive = [card for card in cards if not _selected(card)]
    selected_card = selected[:1]
    inactive_limit = max(0, max_count - len(selected_card))
    return inactive[:inactive_limit] + selected_card


def _max_cards_for_width(width: float) -> int:
    usable = width - 2 * _CARD_MARGIN_POINTS
    if usable < _CARD_MIN_WIDTH_POINTS:
        return 1
    return max(
        1,
        min(
            _MAX_VISIBLE_CARDS,
            int((usable + _CARD_GAP_POINTS) // (_CARD_MIN_WIDTH_POINTS + _CARD_GAP_POINTS)),
        ),
    )


def _frame_for_index(index: int, count: int, width: float, height: float) -> dict[str, float]:
    usable = max(_CARD_MIN_WIDTH_POINTS, width - 2 * _CARD_MARGIN_POINTS)
    total_gap = _CARD_GAP_POINTS * max(0, count - 1)
    card_width = (usable - total_gap) / max(1, count)
    card_width = max(_CARD_MIN_WIDTH_POINTS, min(_CARD_MAX_WIDTH_POINTS, card_width))
    x = _CARD_MARGIN_POINTS + index * (card_width + _CARD_GAP_POINTS)
    y = max(_CARD_MARGIN_POINTS, height - _CARD_MARGIN_POINTS - _CARD_HEIGHT_POINTS)
    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "width": round(card_width, 3),
        "height": _CARD_HEIGHT_POINTS,
    }


def build_agent_thread_hud(
    cards: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    content_width_points: float,
    content_height_points: float,
) -> dict[str, Any]:
    """Build deterministic render data for Agent Shell thread-card HUD surfaces."""
    normalized = [dict(card) for card in cards if isinstance(card, dict)]
    width = max(_CARD_MIN_WIDTH_POINTS, _number(content_width_points, _CARD_MIN_WIDTH_POINTS))
    height = max(_CARD_HEIGHT_POINTS + 2 * _CARD_MARGIN_POINTS, _number(content_height_points, 0.0))
    max_count = _max_cards_for_width(width)
    visible = _visible_cards(normalized, max_count)
    selected_card = next((card for card in normalized if _selected(card)), None)
    selected_display = (
        card_display_contract(selected_card, selected=True)
        if selected_card is not None
        else {}
    )
    surfaces: list[dict[str, Any]] = []
    for index, card in enumerate(visible):
        selected = _selected(card)
        display = card_display_contract(card, selected=selected)
        surfaces.append(
            {
                "thread_id": _thread_id(card),
                "provider": _string(card.get("provider")),
                "role": "selected_summary" if selected else "inactive_card",
                "readiness": _string(display.get("readiness")),
                "text": _surface_text(display),
                "bearing": _string(display.get("bearing")),
                "show_latest_response": False,
                "latest_response": "",
                "selected": selected,
                "frame": _frame_for_index(index, len(visible), width, height),
            }
        )

    main_text = _string(selected_display.get("primary_text"))
    return {
        "surface_kind": "agent_shell_partyline",
        "selected_thread_id": _thread_id(selected_card or {}),
        "main_transcript": {
            "thread_id": _thread_id(selected_card or {}),
            "show_latest_response": bool(selected_display.get("show_latest_response")),
            "text": main_text if selected_display.get("show_latest_response") else "",
        },
        "cards": surfaces,
    }
