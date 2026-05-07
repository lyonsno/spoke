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


OpticalFieldState = Literal[
    "hidden",
    "materialize",
    "rest",
    "resize",
    "recenter",
    "retarget",
    "dismiss",
]
OpticalFieldDisturbanceMode = Literal["persistent", "ephemeral"]
OpticalFieldCoordinateSpace = Literal[
    "display_local",
    "screen_points",
    "backing_pixels",
    "parent_local",
    "content_local",
    "recipe_local",
]
OpticalFieldMotionStrategy = Literal[
    "auto",
    "continuous",
    "morph",
    "squirt",
    "dematerialize_rematerialize",
    "snap",
]

_FORBIDDEN_CONSUMER_ANIMATION_NAMES = {
    "progress",
    "phase",
    "transition.phase",
    "animation_progress",
    "animation_phase",
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

    def to_payload(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
        }


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
    params: Mapping[str, float | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.disturbance_id:
            raise ValueError("disturbance_id must be non-empty")
        if not self.kind:
            raise ValueError("disturbance kind must be non-empty")
        if _is_forbidden_animation_name(self.kind):
            raise ValueError("consumer-authored progress/phase is not a production field")
        if self.strength < 0.0:
            raise ValueError("disturbance strength must be non-negative")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True)
class OpticalFieldMotionIntent:
    """Consumer motion intent as data; House still owns execution curves."""

    strategy: OpticalFieldMotionStrategy = "auto"
    urgency: str = "normal"
    latency_mask: str = "none"
    params: Mapping[str, float | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.strategy:
            raise ValueError("motion strategy must be non-empty")
        if not self.urgency:
            raise ValueError("motion urgency must be non-empty")
        if not self.latency_mask:
            raise ValueError("motion latency_mask must be non-empty")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def to_payload(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "urgency": self.urgency,
            "latency_mask": self.latency_mask,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class OpticalFieldSignal:
    """Finite consumer signal routed into House-owned recipes/profiles."""

    name: str
    value: float | str | bool
    freshness_epoch: str | int | None = None
    params: Mapping[str, float | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("signal name must be non-empty")
        if _is_forbidden_animation_name(self.name):
            raise ValueError("consumer-authored progress/phase is not a production field")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "freshness_epoch": self.freshness_epoch,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class OpticalFieldRequest:
    """Stable request contract consumed by future UI lanes."""

    caller_id: str
    bounds: OpticalFieldBounds
    role: str
    continuity_key: str | None = None
    state: OpticalFieldState = "rest"
    content_frame: OpticalFieldBounds | None = None
    coordinate_space: OpticalFieldCoordinateSpace = "display_local"
    display_epoch: str | int | None = None
    source_epoch: str | int | None = None
    freshness_epoch: str | int | None = None
    presentation_layer: str = "default"
    layout_recipe: str = "direct_positioned"
    motion: OpticalFieldMotionIntent = field(default_factory=OpticalFieldMotionIntent)
    continuity: str = "preserve_identity"
    signals: tuple[OpticalFieldSignal, ...] = ()
    provisional: bool = False
    confidence: float | None = None
    profile: OpticalFieldProfileRef = field(default_factory=OpticalFieldProfileRef)
    disturbances: tuple[OpticalFieldDisturbance, ...] = ()
    visible: bool = True
    z_index: int = 0

    def __post_init__(self) -> None:
        if not self.caller_id:
            raise ValueError("caller_id must be non-empty")
        if not self.role:
            raise ValueError("role must be non-empty")
        if self.continuity_key is None:
            object.__setattr__(self, "continuity_key", self.caller_id)
        elif not self.continuity_key:
            raise ValueError("continuity_key must be non-empty when provided")
        if self.content_frame is None:
            object.__setattr__(self, "content_frame", self.bounds)
        if not self.coordinate_space:
            raise ValueError("coordinate_space must be non-empty")
        if not self.presentation_layer:
            raise ValueError("presentation_layer must be non-empty")
        if not self.layout_recipe:
            raise ValueError("layout_recipe must be non-empty")
        if not self.continuity:
            raise ValueError("continuity must be non-empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        object.__setattr__(self, "signals", tuple(self.signals))
        object.__setattr__(self, "disturbances", tuple(self.disturbances))


def _is_forbidden_animation_name(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_")
    dotted = normalized.replace("_", ".")
    return normalized in _FORBIDDEN_CONSUMER_ANIMATION_NAMES or dotted in _FORBIDDEN_CONSUMER_ANIMATION_NAMES


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


def compile_placeholder_shell_config(request: OpticalFieldRequest) -> dict[str, Any]:
    """Compile one contract request into the current legacy shell-config shape."""

    slot_name = _slot_name_for_state(request.state)
    params = _merged_profile_params(request.profile, slot_name)
    bounds = request.bounds
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
            "continuity_key": request.continuity_key,
            "profile": request.profile.base,
            "state": request.state,
            "lifecycle": request.state,
            "slot": slot_name,
            "bounds": request.bounds.to_payload(),
            "content_frame": request.content_frame.to_payload()
            if request.content_frame is not None
            else request.bounds.to_payload(),
            "coordinate_space": request.coordinate_space,
            "display_epoch": request.display_epoch,
            "source_epoch": request.source_epoch,
            "freshness_epoch": request.freshness_epoch,
            "presentation_layer": request.presentation_layer,
            "layout_recipe": request.layout_recipe,
            "motion": request.motion.to_payload(),
            "continuity": request.continuity,
            "signals": tuple(signal.to_payload() for signal in request.signals),
            "provisional": bool(request.provisional),
            "final": not bool(request.provisional),
            "confidence": request.confidence,
            "disturbances": tuple(
                disturbance.disturbance_id for disturbance in request.disturbances
            ),
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
        return tuple(
            compile_placeholder_shell_config(request)
            for request in self._requests.values()
            if request.visible
        )
