"""Renderer-owned payloads for Agent Shell primitive cards."""

from __future__ import annotations

from typing import Any

from .optical_field import (
    OpticalFieldBounds,
    OpticalFieldDisturbance,
    OpticalFieldProfileRef,
    OpticalFieldRequest,
    compile_placeholder_shell_config,
)

_CARD_MARGIN_POINTS = 12.0
_CARD_GAP_POINTS = 8.0
_CARD_MIN_WIDTH_POINTS = 92.0
_CARD_MIN_HEIGHT_POINTS = 40.0
_CARD_MAX_WIDTH_POINTS = 180.0
_MAX_VISIBLE_CARDS = 4

_MATERIAL_STYLE_TO_OPTICAL_PROFILE = {
    "assistant_shell": "assistant_shell",
    "thread_card": "agent_card",
    "quiet_chip": "quiet_chip",
}


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _number(value: Any, fallback: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else fallback


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _primitive_id(primitive: dict[str, Any]) -> str:
    return (
        _string(primitive.get("id"))
        or _string(primitive.get("provider_session_id"))
        or _string(primitive.get("thread_id"))
    )


def _is_card_primitive(primitive: dict[str, Any]) -> bool:
    return _string(primitive.get("kind")) in {"thread_card", "selected_thread"}


def _priority(primitive: dict[str, Any]) -> int:
    geometry = _mapping(primitive.get("geometry"))
    value = geometry.get("priority", primitive.get("priority", 0))
    return int(value) if isinstance(value, int) else 0


def _selected(primitive: dict[str, Any]) -> bool:
    return bool(primitive.get("selected"))


def _max_cards_for_width(width: float) -> int:
    usable = width - 2 * _CARD_MARGIN_POINTS
    if usable < _CARD_MIN_WIDTH_POINTS:
        return 1
    return max(
        1,
        min(
            _MAX_VISIBLE_CARDS,
            int(
                (usable + _CARD_GAP_POINTS)
                // (_CARD_MIN_WIDTH_POINTS + _CARD_GAP_POINTS)
            ),
        ),
    )


def _visible_primitives(
    primitives: list[dict[str, Any]], max_count: int
) -> list[dict[str, Any]]:
    selected = [primitive for primitive in primitives if _selected(primitive)]
    inactive = [primitive for primitive in primitives if not _selected(primitive)]
    inactive.sort(key=_priority)
    selected_card = selected[:1]
    inactive_limit = max(0, max_count - len(selected_card))
    return inactive[:inactive_limit] + selected_card


def _preferred_size(primitive: dict[str, Any]) -> tuple[float, float, float, float]:
    geometry = _mapping(primitive.get("geometry"))
    min_width = max(
        _CARD_MIN_WIDTH_POINTS,
        _number(geometry.get("min_width"), _CARD_MIN_WIDTH_POINTS),
    )
    min_height = max(
        _CARD_MIN_HEIGHT_POINTS,
        _number(geometry.get("min_height"), _CARD_MIN_HEIGHT_POINTS),
    )
    preferred_width = max(min_width, _number(geometry.get("preferred_width"), min_width))
    preferred_height = max(min_height, _number(geometry.get("preferred_height"), min_height))
    return min_width, min_height, preferred_width, preferred_height


def _frame_for_index(
    primitive: dict[str, Any],
    *,
    index: int,
    count: int,
    content_width: float,
    content_height: float,
) -> dict[str, float]:
    min_width, min_height, preferred_width, preferred_height = _preferred_size(primitive)
    usable_width = max(min_width, content_width - 2 * _CARD_MARGIN_POINTS)
    total_gap = _CARD_GAP_POINTS * max(0, count - 1)
    equal_width = (usable_width - total_gap) / max(1, count)
    width = max(min_width, min(_CARD_MAX_WIDTH_POINTS, preferred_width, equal_width))
    height = max(min_height, min(preferred_height, max(min_height, content_height * 0.42)))
    x = _CARD_MARGIN_POINTS + index * (width + _CARD_GAP_POINTS)
    y = max(_CARD_MARGIN_POINTS, content_height - _CARD_MARGIN_POINTS - height)
    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "width": round(width, 3),
        "height": round(height, 3),
    }


def _transcript_frame(cards: list[dict[str, Any]], content_width: float) -> dict[str, float]:
    if not cards:
        top = _CARD_MARGIN_POINTS
    else:
        top = min(float(card["frame"]["y"]) for card in cards)
    height = max(0.0, top - _CARD_GAP_POINTS - _CARD_MARGIN_POINTS)
    return {
        "x": _CARD_MARGIN_POINTS,
        "y": _CARD_MARGIN_POINTS,
        "width": round(
            max(_CARD_MIN_WIDTH_POINTS, content_width - 2 * _CARD_MARGIN_POINTS),
            3,
        ),
        "height": round(height, 3),
    }


def _material(primitive: dict[str, Any]) -> dict[str, Any]:
    material = _mapping(primitive.get("material"))
    selected = _selected(primitive)
    return {
        "style": _string(material.get("style")) or "quiet_chip",
        "prominence": _string(material.get("prominence"))
        or ("selected" if selected else "inactive"),
        "optical_displacement": _number(material.get("optical_displacement"), 0.0),
        "corner_radius": _number(material.get("corner_radius"), 8.0),
    }


def _optical_profile_for_material(material: dict[str, Any]) -> str:
    return _MATERIAL_STYLE_TO_OPTICAL_PROFILE.get(
        _string(material.get("style")),
        "agent_card",
    )


def _disturbances_for_card(card: dict[str, Any]) -> tuple[OpticalFieldDisturbance, ...]:
    readiness = _string(card.get("readiness"))
    if readiness not in {"working", "waiting", "failed"}:
        return ()
    primitive_id = _string(card.get("primitive_id"))
    material = _mapping(card.get("material"))
    strength = max(0.05, _number(material.get("optical_displacement"), 0.08))
    kind = "blocked_shudder" if readiness == "failed" else "readiness_pulse"
    return (
        OpticalFieldDisturbance(
            disturbance_id=f"readiness.{primitive_id}",
            kind=kind,
            mode="persistent",
            strength=strength,
            params={"readiness": readiness},
        ),
    )


def _request_dict(request: OpticalFieldRequest) -> dict[str, Any]:
    compiled = compile_placeholder_shell_config(request)
    return {
        "caller_id": request.caller_id,
        "role": request.role,
        "state": request.state,
        "profile": request.profile.base,
        "visible": request.visible,
        "z_index": request.z_index,
        "bounds": {
            "x": request.bounds.x,
            "y": request.bounds.y,
            "width": request.bounds.width,
            "height": request.bounds.height,
        },
        "disturbances": [
            {
                "disturbance_id": disturbance.disturbance_id,
                "kind": disturbance.kind,
                "mode": disturbance.mode,
                "strength": disturbance.strength,
                "params": dict(disturbance.params),
            }
            for disturbance in request.disturbances
        ],
        "compiled_shell_config": compiled,
    }


def _surface_for_primitive(
    primitive: dict[str, Any],
    *,
    index: int,
    count: int,
    content_width: float,
    content_height: float,
) -> dict[str, Any]:
    display = _mapping(primitive.get("display"))
    show_latest_response = bool(display.get("show_latest_response"))
    return {
        "primitive_id": _primitive_id(primitive),
        "provider": _string(primitive.get("provider")),
        "provider_session_id": _string(primitive.get("provider_session_id")),
        "role": "selected_thread" if _selected(primitive) else "inactive_thread",
        "selected": _selected(primitive),
        "readiness": _string(primitive.get("readiness")),
        "primary_text": _string(display.get("primary_text")),
        "secondary_text": _string(display.get("secondary_text")),
        "show_latest_response": show_latest_response,
        "latest_response": _string(primitive.get("latest_response")) if show_latest_response else "",
        "frame": _frame_for_index(
            primitive,
            index=index,
            count=count,
            content_width=content_width,
            content_height=content_height,
        ),
        "material": _material(primitive),
        "clip": True,
    }


def build_agent_shell_card_render_payload(
    primitives: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    content_width_points: float,
    content_height_points: float,
) -> dict[str, Any]:
    """Build quiet card render surfaces from provider-agnostic primitives."""
    normalized = [
        dict(primitive)
        for primitive in primitives
        if isinstance(primitive, dict) and _is_card_primitive(primitive)
    ]
    content_width = max(
        _CARD_MIN_WIDTH_POINTS,
        _number(content_width_points, _CARD_MIN_WIDTH_POINTS),
    )
    content_height = max(
        _CARD_MIN_HEIGHT_POINTS + 2 * _CARD_MARGIN_POINTS,
        _number(content_height_points, 0.0),
    )
    visible = _visible_primitives(normalized, _max_cards_for_width(content_width))
    cards = [
        _surface_for_primitive(
            primitive,
            index=index,
            count=len(visible),
            content_width=content_width,
            content_height=content_height,
        )
        for index, primitive in enumerate(visible)
    ]
    selected = next((primitive for primitive in normalized if _selected(primitive)), None)
    selected_display = _mapping(selected.get("display")) if selected is not None else {}
    show_latest_response = bool(selected_display.get("show_latest_response"))
    return {
        "surface_kind": "agent_shell_card_primitives",
        "selected_primitive_id": _primitive_id(selected or {}),
        "main_transcript": {
            "primitive_id": _primitive_id(selected or {}),
            "show_latest_response": show_latest_response,
            "text": _string((selected or {}).get("latest_response")) if show_latest_response else "",
        },
        "transcript_frame": _transcript_frame(cards, content_width),
        "cards": cards,
    }


def build_agent_shell_card_optical_field_payload(
    render_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build placeholder optical-field requests for rendered Agent Shell cards."""
    cards = render_payload.get("cards") if isinstance(render_payload, dict) else None
    requests: list[dict[str, Any]] = []
    if not isinstance(cards, list):
        cards = []
    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        primitive_id = _string(card.get("primitive_id"))
        frame = _mapping(card.get("frame"))
        try:
            bounds = OpticalFieldBounds(
                x=_number(frame.get("x")),
                y=_number(frame.get("y")),
                width=_number(frame.get("width")),
                height=_number(frame.get("height")),
            )
        except ValueError:
            continue
        material = _mapping(card.get("material"))
        profile = _optical_profile_for_material(material)
        selected = bool(card.get("selected"))
        request = OpticalFieldRequest(
            caller_id=f"agent.card.{primitive_id}",
            bounds=bounds,
            role="selected_thread" if selected else "agent_card",
            state="rest",
            profile=OpticalFieldProfileRef(base=profile),
            disturbances=_disturbances_for_card(card),
            visible=True,
            z_index=(200 + index) if selected else (100 + index),
        )
        requests.append(_request_dict(request))
    return {
        "surface_kind": "agent_shell_card_optical_fields",
        "requests": requests,
    }
