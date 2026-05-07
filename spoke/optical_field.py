"""Contract-level optical field requests for compositor-owned UI surfaces.

This module is intentionally above the current Metal/SDF implementation.  It
lets consumers target a stable request/profile/disturbance contract while the
placeholder backend compiles those requests into today's legacy shell-config
dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping


OpticalFieldState = Literal["rest", "materialize", "dismiss"]
OpticalFieldDisturbanceMode = Literal["persistent", "ephemeral"]
OpticalFieldPresentationLayer = Literal[
    "assistant_shell",
    "user_preview",
    "agent_card",
    "hud",
    "debug",
]
OpticalFieldVisibilityScope = Literal["independent", "follows_assistant_shell"]
OpticalFieldSelectedHandoffMode = Literal[
    "preserve_identity",
    "handoff",
    "new_presence",
    "replace",
]


_ROLE_PRESENTATION_DEFAULTS: dict[str, tuple[str, int]] = {
    "assistant": ("assistant_shell", 20),
    "assistant_shell": ("assistant_shell", 20),
    "agent_card": ("agent_card", 20),
    "preview": ("user_preview", 30),
    "user_preview": ("user_preview", 30),
    "hud": ("hud", 50),
    "tray": ("hud", 50),
    "recovery": ("hud", 50),
}


@dataclass(frozen=True)
class OpticalFieldBounds:
    """Compositor-space logical bounds for one optical field request."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("optical field bounds must have positive width and height")

    @property
    def center_x(self) -> float:
        return self.x + self.width * 0.5

    @property
    def center_y(self) -> float:
        return self.y + self.height * 0.5

    @property
    def min_dimension(self) -> float:
        return min(self.width, self.height)


@dataclass(frozen=True)
class OpticalFieldSlotOverride:
    """Profile-slot overrides expressed in contract-level normalized params."""

    params: Mapping[str, float | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True)
class OpticalFieldProfileRef:
    """Named profile plus optional all-slot and per-slot overrides."""

    base: str = "assistant_shell"
    params: Mapping[str, float | str | bool] = field(default_factory=dict)
    slots: Mapping[str, OpticalFieldSlotOverride] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))
        object.__setattr__(self, "slots", MappingProxyType(dict(self.slots)))


@dataclass(frozen=True)
class OpticalFieldDisturbance:
    """Composable field gesture requested by a UI element."""

    disturbance_id: str
    kind: str
    mode: OpticalFieldDisturbanceMode = "ephemeral"
    strength: float = 1.0
    phase: float = 0.0
    params: Mapping[str, float | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.disturbance_id:
            raise ValueError("disturbance_id must be non-empty")
        if not self.kind:
            raise ValueError("disturbance kind must be non-empty")
        if self.strength < 0.0:
            raise ValueError("disturbance strength must be non-negative")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True)
class OpticalFieldPresentation:
    """Sibling presentation role and compositor ordering for one optical surface."""

    layer: OpticalFieldPresentationLayer | str
    order: int = 0

    def __post_init__(self) -> None:
        if not self.layer:
            raise ValueError("presentation layer must be non-empty")


@dataclass(frozen=True)
class OpticalFieldSelectedHandoff:
    """Boundary metadata for selected-card-to-shell handoff without content truth."""

    from_caller_id: str
    to_caller_id: str
    continuity_key: str
    mode: OpticalFieldSelectedHandoffMode = "handoff"

    def __post_init__(self) -> None:
        if not self.from_caller_id:
            raise ValueError("selected handoff source caller must be non-empty")
        if not self.to_caller_id:
            raise ValueError("selected handoff target caller must be non-empty")
        if not self.continuity_key:
            raise ValueError("selected handoff continuity key must be non-empty")


@dataclass(frozen=True)
class OpticalFieldRequest:
    """Stable request contract consumed by future UI lanes."""

    caller_id: str
    bounds: OpticalFieldBounds
    role: str
    state: OpticalFieldState = "rest"
    profile: OpticalFieldProfileRef = field(default_factory=OpticalFieldProfileRef)
    disturbances: tuple[OpticalFieldDisturbance, ...] = ()
    presentation: OpticalFieldPresentation | None = None
    selected_handoff: OpticalFieldSelectedHandoff | None = None
    visibility_scope: OpticalFieldVisibilityScope = "independent"
    visible: bool = True
    z_index: int = 0

    def __post_init__(self) -> None:
        if not self.caller_id:
            raise ValueError("caller_id must be non-empty")
        if not self.role:
            raise ValueError("role must be non-empty")
        if self.visibility_scope != "independent" and self.role in {
            "agent_card",
            "preview",
            "user_preview",
            "hud",
        }:
            raise ValueError("sibling surfaces must use independent visibility")
        if self.presentation is None:
            object.__setattr__(self, "presentation", _default_presentation_for_role(self.role))
        object.__setattr__(self, "disturbances", tuple(self.disturbances))


_BASE_PROFILES: dict[str, dict[str, float | str | bool]] = {
    "assistant_shell": {
        "corner_radius_frac": 0.45,
        "core_magnification": 1.16,
        "band_width_frac": 0.080,
        "tail_width_frac": 0.055,
        "ring_amplitude_frac": 0.090,
        "tail_amplitude_frac": 0.030,
        "bleed_zone_frac": 0.78,
        "exterior_mix_frac": 0.22,
        "mip_blur_strength": 1.0,
    },
    "preview_pill": {
        "corner_radius_frac": 0.50,
        "core_magnification": 1.08,
        "band_width_frac": 0.055,
        "tail_width_frac": 0.040,
        "ring_amplitude_frac": 0.055,
        "tail_amplitude_frac": 0.020,
        "bleed_zone_frac": 0.70,
        "exterior_mix_frac": 0.16,
        "mip_blur_strength": 1.0,
    },
    "agent_card": {
        "corner_radius_frac": 0.34,
        "core_magnification": 1.04,
        "band_width_frac": 0.060,
        "tail_width_frac": 0.040,
        "ring_amplitude_frac": 0.050,
        "tail_amplitude_frac": 0.018,
        "bleed_zone_frac": 0.60,
        "exterior_mix_frac": 0.12,
        "mip_blur_strength": 0.65,
    },
    "quiet_chip": {
        "corner_radius_frac": 0.50,
        "core_magnification": 1.02,
        "band_width_frac": 0.035,
        "tail_width_frac": 0.025,
        "ring_amplitude_frac": 0.025,
        "tail_amplitude_frac": 0.010,
        "bleed_zone_frac": 0.45,
        "exterior_mix_frac": 0.08,
        "mip_blur_strength": 0.4,
    },
}


def available_optical_field_profiles() -> tuple[str, ...]:
    return tuple(_BASE_PROFILES)


def _default_presentation_for_role(role: str) -> OpticalFieldPresentation:
    layer, order = _ROLE_PRESENTATION_DEFAULTS.get(role, (role, 0))
    return OpticalFieldPresentation(layer=layer, order=order)


def _slot_name_for_state(state: OpticalFieldState) -> str:
    if state in {"materialize", "dismiss"}:
        return state
    return "rest"


def _merged_profile_params(profile: OpticalFieldProfileRef, slot_name: str) -> dict[str, Any]:
    try:
        merged = dict(_BASE_PROFILES[profile.base])
    except KeyError as exc:
        raise ValueError(f"unknown optical field profile: {profile.base}") from exc
    merged.update(profile.params)
    slot = profile.slots.get(slot_name)
    if slot is not None:
        merged.update(slot.params)
    return merged


def _float_param(params: Mapping[str, Any], key: str) -> float:
    return float(params[key])


def _selected_handoff_payload(
    handoff: OpticalFieldSelectedHandoff | None,
) -> dict[str, str] | None:
    if handoff is None:
        return None
    return {
        "from_caller_id": handoff.from_caller_id,
        "to_caller_id": handoff.to_caller_id,
        "continuity_key": handoff.continuity_key,
        "mode": handoff.mode,
    }


def compile_placeholder_shell_config(request: OpticalFieldRequest) -> dict[str, Any]:
    """Compile one contract request into the current legacy shell-config shape."""

    slot_name = _slot_name_for_state(request.state)
    params = _merged_profile_params(request.profile, slot_name)
    bounds = request.bounds
    presentation = request.presentation or _default_presentation_for_role(request.role)
    selected_handoff = _selected_handoff_payload(request.selected_handoff)
    scale = bounds.min_dimension
    corner_radius = min(
        scale * _float_param(params, "corner_radius_frac"),
        bounds.height * 0.5,
        bounds.width * 0.5,
    )
    exterior_mix = scale * _float_param(params, "exterior_mix_frac")

    return {
        "enabled": True,
        "client_id": request.caller_id,
        "role": request.role,
        "presentation_layer": presentation.layer,
        "presentation_order": int(presentation.order),
        "visibility_scope": request.visibility_scope,
        "z_index": int(request.z_index),
        "content_width_points": float(bounds.width),
        "content_height_points": float(bounds.height),
        "corner_radius_points": float(corner_radius),
        "center_x": float(bounds.center_x),
        "center_y": float(bounds.center_y),
        "core_magnification": _float_param(params, "core_magnification"),
        "band_width_points": scale * _float_param(params, "band_width_frac"),
        "tail_width_points": scale * _float_param(params, "tail_width_frac"),
        "ring_amplitude_points": scale * _float_param(params, "ring_amplitude_frac"),
        "tail_amplitude_points": scale * _float_param(params, "tail_amplitude_frac"),
        "bleed_zone_frac": _float_param(params, "bleed_zone_frac"),
        "exterior_mix_width_points": exterior_mix,
        "mip_blur_strength": _float_param(params, "mip_blur_strength"),
        "optical_field": {
            "caller_id": request.caller_id,
            "role": request.role,
            "presentation_layer": presentation.layer,
            "presentation_order": int(presentation.order),
            "visibility_scope": request.visibility_scope,
            "profile": request.profile.base,
            "state": request.state,
            "slot": slot_name,
            "disturbances": tuple(
                disturbance.disturbance_id for disturbance in request.disturbances
            ),
            **({"selected_handoff": selected_handoff} if selected_handoff is not None else {}),
        },
    }


class OpticalFieldPlaceholderBackend:
    """In-memory placeholder backend for consumers targeting the new contract."""

    def __init__(self) -> None:
        self._requests: dict[str, OpticalFieldRequest] = {}

    def upsert(self, request: OpticalFieldRequest) -> None:
        self._requests[request.caller_id] = request

    def remove(self, caller_id: str) -> bool:
        return self._requests.pop(caller_id, None) is not None

    def clear(self) -> None:
        self._requests.clear()

    def requests(self) -> tuple[OpticalFieldRequest, ...]:
        return tuple(self._requests.values())

    def compile_shell_configs(self) -> tuple[dict[str, Any], ...]:
        visible_requests = [
            (index, request)
            for index, request in enumerate(self._requests.values())
            if request.visible
        ]
        visible_requests.sort(
            key=lambda item: (
                int((item[1].presentation or _default_presentation_for_role(item[1].role)).order),
                int(item[1].z_index),
                item[0],
            )
        )
        return tuple(compile_placeholder_shell_config(request) for _index, request in visible_requests)
