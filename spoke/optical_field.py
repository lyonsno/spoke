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
    "display_points",
    "screen_points",
    "backing_pixels",
    "parent_local",
    "parent_points",
    "content_local",
    "content_points",
    "recipe_local",
]
OpticalFieldSignalName = Literal[
    "background_luminance",
    "text_contrast_bias",
    "ridge_emphasis",
]
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

_COORDINATE_SPACES = {
    "display_local",
    "display_points",
    "screen_points",
    "backing_pixels",
    "parent_local",
    "parent_points",
    "content_local",
    "content_points",
}

_MATERIAL_SIGNAL_NAMES = {
    "background_luminance",
    "text_contrast_bias",
    "ridge_emphasis",
}


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

    def to_payload(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
        }


@dataclass(frozen=True)
class OpticalFieldCoordinateContext:
    """Coordinate custody metadata for geometry crossing into House."""

    coordinate_space: OpticalFieldCoordinateSpace = "display_points"
    display_id: str | int | None = None
    display_epoch: str | int | None = None
    source_epoch: str | int | None = None
    backing_scale: float | None = None
    display_origin: tuple[float, float] | None = None
    parent_origin: tuple[float, float] | None = None
    content_origin: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.coordinate_space not in _COORDINATE_SPACES:
            raise ValueError(f"unknown optical coordinate space: {self.coordinate_space}")
        if self.backing_scale is not None and self.backing_scale <= 0.0:
            raise ValueError("backing_scale must be positive")
        for field_name in ("display_origin", "parent_origin", "content_origin"):
            origin = getattr(self, field_name)
            if origin is not None:
                if len(origin) != 2:
                    raise ValueError(f"{field_name} must contain x and y")
                object.__setattr__(
                    self,
                    field_name,
                    (float(origin[0]), float(origin[1])),
                )

    @property
    def carries_metadata(self) -> bool:
        return (
            self.coordinate_space != "display_points"
            or self.display_id is not None
            or self.display_epoch is not None
            or self.source_epoch is not None
            or self.backing_scale is not None
            or self.display_origin is not None
            or self.parent_origin is not None
            or self.content_origin is not None
        )


def _require_origin(
    context: OpticalFieldCoordinateContext,
    attr: str,
    space: str,
) -> tuple[float, float]:
    origin = getattr(context, attr)
    if origin is None:
        raise ValueError(f"{space} bounds require {attr}")
    return origin


def normalize_optical_field_bounds(
    bounds: OpticalFieldBounds,
    context: OpticalFieldCoordinateContext | None = None,
) -> OpticalFieldBounds:
    """Normalize explicit coordinate-space bounds into display-local points."""

    context = context or OpticalFieldCoordinateContext()
    space = context.coordinate_space
    if space in {"display_local", "display_points"}:
        return bounds
    if space == "screen_points":
        display_x, display_y = _require_origin(context, "display_origin", space)
        return OpticalFieldBounds(
            x=bounds.x - display_x,
            y=bounds.y - display_y,
            width=bounds.width,
            height=bounds.height,
        )
    if space == "backing_pixels":
        if context.backing_scale is None:
            raise ValueError("backing_pixels bounds require backing_scale")
        scale = context.backing_scale
        return OpticalFieldBounds(
            x=bounds.x / scale,
            y=bounds.y / scale,
            width=bounds.width / scale,
            height=bounds.height / scale,
        )
    if space in {"parent_local", "parent_points"}:
        parent_x, parent_y = _require_origin(context, "parent_origin", space)
        return OpticalFieldBounds(
            x=bounds.x + parent_x,
            y=bounds.y + parent_y,
            width=bounds.width,
            height=bounds.height,
        )
    if space in {"content_local", "content_points"}:
        content_x, content_y = _require_origin(context, "content_origin", space)
        return OpticalFieldBounds(
            x=bounds.x + content_x,
            y=bounds.y + content_y,
            width=bounds.width,
            height=bounds.height,
        )
    raise ValueError(f"unknown optical coordinate space: {space}")


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
    coordinate_context: OpticalFieldCoordinateContext = field(
        default_factory=OpticalFieldCoordinateContext
    )
    content_coordinate_context: OpticalFieldCoordinateContext | None = None
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
        if self.continuity_key is None:
            object.__setattr__(self, "continuity_key", self.caller_id)
        elif not self.continuity_key:
            raise ValueError("continuity_key must be non-empty when provided")
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
        if self.content_frame is not None and self.content_coordinate_context is None:
            object.__setattr__(
                self,
                "content_coordinate_context",
                self.coordinate_context,
            )


def _is_forbidden_animation_name(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_")
    dotted = normalized.replace("_", ".")
    return normalized in _FORBIDDEN_CONSUMER_ANIMATION_NAMES or dotted in _FORBIDDEN_CONSUMER_ANIMATION_NAMES


@dataclass(frozen=True)
class OpticalFieldTransitionState:
    """Primitive-owned geometry custody for one caller's latest accepted target."""

    target_request: OpticalFieldRequest
    previous_bounds: OpticalFieldBounds
    presented_bounds: OpticalFieldBounds
    target_bounds: OpticalFieldBounds
    pending_request: OpticalFieldRequest | None = None

    @property
    def caller_id(self) -> str:
        return self.target_request.caller_id

    @property
    def source_epoch(self) -> int | None:
        return self.target_request.source_epoch

    @property
    def display_epoch(self) -> int | None:
        return self.target_request.display_epoch


@dataclass(frozen=True)
class OpticalFieldMailboxResult:
    """Acceptance result for a desired-state geometry target."""

    accepted: bool
    reason: str
    state: OpticalFieldTransitionState | None = None


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


def _bounds_dict(bounds: OpticalFieldBounds) -> dict[str, float]:
    return {
        "x": float(bounds.x),
        "y": float(bounds.y),
        "width": float(bounds.width),
        "height": float(bounds.height),
    }


def _coordinate_metadata(
    request: OpticalFieldRequest,
    bounds: OpticalFieldBounds,
    content_frame: OpticalFieldBounds | None,
) -> dict[str, Any]:
    context = request.coordinate_context
    if not context.carries_metadata:
        return {}
    metadata: dict[str, Any] = {
        "coordinate_space": "display_points",
        "source_coordinate_space": context.coordinate_space,
        "bounds": _bounds_dict(bounds),
    }
    if content_frame is not None:
        metadata["content_frame"] = _bounds_dict(content_frame)
    if context.display_id is not None:
        metadata["display_id"] = context.display_id
    if context.display_epoch is not None:
        metadata["display_epoch"] = context.display_epoch
    if context.source_epoch is not None:
        metadata["source_epoch"] = context.source_epoch
    if context.backing_scale is not None:
        metadata["backing_scale"] = float(context.backing_scale)
    return metadata


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


def _bounds_metadata(bounds: OpticalFieldBounds) -> tuple[float, float, float, float]:
    return (float(bounds.x), float(bounds.y), float(bounds.width), float(bounds.height))


def _with_transition_metadata(
    optical_field: dict[str, Any],
    request: OpticalFieldRequest,
    transition: OpticalFieldTransitionState | None,
) -> dict[str, Any]:
    if transition is None:
        return optical_field

    carries_freshness = (
        request.source_epoch is not None
        or request.display_epoch is not None
        or request.provisional
    )
    carries_geometry_custody = (
        request.state in {"resize", "recenter", "hidden"}
        or transition.previous_bounds != transition.target_bounds
        or transition.presented_bounds != transition.target_bounds
    )
    if not carries_freshness and not carries_geometry_custody:
        return optical_field

    optical_field = dict(optical_field)
    transition_payload: dict[str, Any] = {
        "from_bounds": _bounds_metadata(transition.previous_bounds),
        "presented_bounds": _bounds_metadata(transition.presented_bounds),
        "target_bounds": _bounds_metadata(transition.target_bounds),
        "provisional": bool(request.provisional),
    }
    if request.source_epoch is not None:
        transition_payload["source_epoch"] = int(request.source_epoch)
    if request.display_epoch is not None:
        transition_payload["display_epoch"] = int(request.display_epoch)
    optical_field["transition"] = transition_payload
    return optical_field


def compile_placeholder_shell_config(
    request: OpticalFieldRequest,
    transition: OpticalFieldTransitionState | None = None,
) -> dict[str, Any]:
    """Compile one contract request into the current legacy shell-config shape."""

    slot_name = _slot_name_for_state(request.state)
    params = _merged_profile_params(request.profile, slot_name)
    bounds = normalize_optical_field_bounds(request.bounds, request.coordinate_context)
    content_frame = (
        normalize_optical_field_bounds(
            request.content_frame,
            request.content_coordinate_context,
        )
        if request.content_frame is not None
        else None
    )
    signals = {
        signal.name: signal.value
        for signal in request.signals
        if signal.name in _MATERIAL_SIGNAL_NAMES and isinstance(signal.value, (int, float))
    }
    presentation = request.presentation or _default_presentation_for_role(request.role)
    selected_handoff = _selected_handoff_payload(request.selected_handoff)
    scale = bounds.min_dimension
    corner_radius = min(
        scale * _float_param(params, "corner_radius_frac"),
        bounds.height * 0.5,
        bounds.width * 0.5,
    )
    exterior_mix = scale * _float_param(params, "exterior_mix_frac")

    optical_field = _with_transition_metadata(
        {
            "caller_id": request.caller_id,
            "continuity_key": request.continuity_key,
            "role": request.role,
            "resolved_presentation_layer": presentation.layer,
            "presentation_order": int(presentation.order),
            "visibility_scope": request.visibility_scope,
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
            **_coordinate_metadata(request, bounds, content_frame),
            **({"selected_handoff": selected_handoff} if selected_handoff is not None else {}),
        },
        request,
        transition,
    )

    config = {
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
        "optical_field": optical_field,
    }
    if signals:
        config.update(
            {
                "gpu_material_enabled": 1.0,
                "gpu_material_brightness": float(
                    signals.get("background_luminance", config.get("initial_brightness", 0.5))
                ),
                "gpu_material_text_contrast_bias": float(
                    signals.get("text_contrast_bias", 0.5)
                ),
                "gpu_material_ridge_emphasis": float(signals.get("ridge_emphasis", 0.5)),
            }
        )
    return config


class OpticalFieldPlaceholderBackend:
    """In-memory placeholder backend for consumers targeting the new contract."""

    def __init__(
        self,
        *,
        display_epochs: Mapping[str | int, str | int] | None = None,
        source_epochs: Mapping[str | int, str | int] | None = None,
    ) -> None:
        self._transitions: dict[str, OpticalFieldTransitionState] = {}
        self._display_epochs = dict(display_epochs or {})
        self._source_epochs = dict(source_epochs or {})

    def _validate_epoch_context(self, context: OpticalFieldCoordinateContext) -> None:
        display_id = context.display_id
        if display_id is None:
            return
        current_display_epoch = self._display_epochs.get(display_id)
        if (
            current_display_epoch is not None
            and context.display_epoch != current_display_epoch
        ):
            raise ValueError(
                f"stale display_epoch for {display_id}: "
                f"{context.display_epoch!r} != {current_display_epoch!r}"
            )
        current_source_epoch = self._source_epochs.get(display_id)
        if (
            current_source_epoch is not None
            and context.source_epoch != current_source_epoch
        ):
            raise ValueError(
                f"stale source_epoch for {display_id}: "
                f"{context.source_epoch!r} != {current_source_epoch!r}"
            )

    @staticmethod
    def _is_newer_epoch(
        request: OpticalFieldRequest,
        current: OpticalFieldTransitionState,
    ) -> bool:
        for key in ("display_epoch", "source_epoch"):
            incoming = getattr(request, key)
            existing = getattr(current, key)
            if incoming is not None and (existing is None or incoming > existing):
                return True
        return False

    @staticmethod
    def _rejection_reason(
        request: OpticalFieldRequest,
        current: OpticalFieldTransitionState | None,
    ) -> str | None:
        if current is None:
            return None
        if (
            request.display_epoch is not None
            and current.display_epoch is not None
            and request.display_epoch < current.display_epoch
        ):
            return "stale_display_epoch"
        if (
            request.source_epoch is not None
            and current.source_epoch is not None
            and request.source_epoch < current.source_epoch
        ):
            return "stale_source_epoch"
        if (
            request.provisional
            and not current.target_request.provisional
            and not OpticalFieldPlaceholderBackend._is_newer_epoch(request, current)
        ):
            return "stale_provisional_after_final"
        return None

    def upsert(
        self,
        request: OpticalFieldRequest,
        *,
        presented_bounds: OpticalFieldBounds | None = None,
    ) -> OpticalFieldMailboxResult:
        self._validate_epoch_context(request.coordinate_context)
        if request.content_coordinate_context is not None:
            self._validate_epoch_context(request.content_coordinate_context)
        current = self._transitions.get(request.caller_id)
        rejection_reason = self._rejection_reason(request, current)
        if rejection_reason is not None:
            return OpticalFieldMailboxResult(
                accepted=False,
                reason=rejection_reason,
                state=current,
            )

        if presented_bounds is not None:
            previous_bounds = presented_bounds
        elif current is not None:
            previous_bounds = current.presented_bounds
        else:
            previous_bounds = request.bounds

        transition = OpticalFieldTransitionState(
            target_request=request,
            previous_bounds=previous_bounds,
            presented_bounds=previous_bounds,
            target_bounds=request.bounds,
            pending_request=None,
        )
        self._transitions[request.caller_id] = transition
        return OpticalFieldMailboxResult(
            accepted=True,
            reason="accepted",
            state=transition,
        )

    def remove(self, caller_id: str) -> bool:
        return self._transitions.pop(caller_id, None) is not None

    def clear(self) -> None:
        self._transitions.clear()

    def requests(self) -> tuple[OpticalFieldRequest, ...]:
        return tuple(transition.target_request for transition in self._transitions.values())

    def transition_for(self, caller_id: str) -> OpticalFieldTransitionState | None:
        return self._transitions.get(caller_id)

    def sample_presented_bounds(self, caller_id: str, bounds: OpticalFieldBounds) -> bool:
        current = self._transitions.get(caller_id)
        if current is None:
            return False
        self._transitions[caller_id] = OpticalFieldTransitionState(
            target_request=current.target_request,
            previous_bounds=current.previous_bounds,
            presented_bounds=bounds,
            target_bounds=current.target_bounds,
            pending_request=current.pending_request,
        )
        return True

    def compile_shell_configs(self) -> tuple[dict[str, Any], ...]:
        visible_transitions = [
            (index, transition)
            for index, transition in enumerate(self._transitions.values())
            if transition.target_request.visible
            and transition.target_request.state != "hidden"
        ]
        visible_transitions.sort(
            key=lambda item: (
                int(
                    (
                        item[1].target_request.presentation
                        or _default_presentation_for_role(item[1].target_request.role)
                    ).order
                ),
                int(item[1].target_request.z_index),
                item[0],
            )
        )
        return tuple(
            compile_placeholder_shell_config(transition.target_request, transition)
            for _index, transition in visible_transitions
        )
