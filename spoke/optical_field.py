"""Contract-level optical field requests for compositor-owned UI surfaces.

This module is intentionally above the current Metal/SDF implementation.  It
lets consumers target a stable request/profile/disturbance contract while this
backend compiles those requests into today's legacy shell-config dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping


OpticalFieldState = Literal["rest", "materialize", "dismiss"]
OpticalFieldDisturbanceMode = Literal["persistent", "ephemeral"]


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
class OpticalFieldRequest:
    """Stable request contract consumed by future UI lanes."""

    caller_id: str
    bounds: OpticalFieldBounds
    role: str
    state: OpticalFieldState = "rest"
    progress: float = 1.0
    profile: OpticalFieldProfileRef = field(default_factory=OpticalFieldProfileRef)
    disturbances: tuple[OpticalFieldDisturbance, ...] = ()
    visible: bool = True
    z_index: int = 0

    def __post_init__(self) -> None:
        if not self.caller_id:
            raise ValueError("caller_id must be non-empty")
        if not self.role:
            raise ValueError("role must be non-empty")
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("optical field progress must be between 0 and 1")
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


def _optional_float_param(params: Mapping[str, Any], key: str) -> float | None:
    if key not in params or params[key] is None:
        return None
    return float(params[key])


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * _clamp01(progress)


def _smoothstep(value: float) -> float:
    value = _clamp01(value)
    return value * value * (3.0 - 2.0 * value)


def _snap_ease_in(value: float) -> float:
    value = _clamp01(value)
    return value * value * value


def _base_shell_config(
    request: OpticalFieldRequest,
    *,
    slot_name: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile one contract request into the current legacy shell-config shape."""

    slot_name = slot_name or _slot_name_for_state(request.state)
    params = params or _merged_profile_params(request.profile, slot_name)

    bounds = request.bounds
    scale = bounds.min_dimension
    corner_radius = min(
        scale * _float_param(params, "corner_radius_frac"),
        bounds.height * 0.5,
        bounds.width * 0.5,
    )
    exterior_mix = scale * _float_param(params, "exterior_mix_frac")

    config = {
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
            "profile": request.profile.base,
            "state": request.state,
            "slot": slot_name,
            "progress": float(request.progress),
            "disturbances": tuple(
                disturbance.disturbance_id for disturbance in request.disturbances
            ),
        },
    }
    for key in (
        "initial_brightness",
        "min_brightness",
        "x_squeeze",
        "y_squeeze",
        "cleanup_blur_radius_points",
        "cleanup_blur_strength",
    ):
        value = _optional_float_param(params, key)
        if value is not None:
            config[key] = value
    return config


def _with_optical_field_sidecar(
    config: dict[str, Any],
    request: OpticalFieldRequest,
    *,
    sidecar: str,
    slot_name: str,
) -> dict[str, Any]:
    optical_field = dict(config.get("optical_field", {}))
    optical_field.update(
        {
            "caller_id": request.caller_id,
            "profile": request.profile.base,
            "state": request.state,
            "slot": slot_name,
            "progress": float(request.progress),
            "sidecar": sidecar,
            "disturbances": tuple(
                disturbance.disturbance_id for disturbance in request.disturbances
            ),
        }
    )
    config["optical_field"] = optical_field
    return config


def _materialized_pressure_slit_config(
    request: OpticalFieldRequest,
    params: Mapping[str, Any],
    slot_name: str,
) -> dict[str, Any]:
    config = _base_shell_config(request, slot_name=slot_name, params=params)
    progress = _clamp01(request.progress)
    if progress >= 1.0:
        config["continuous_present"] = True
        return config

    base_width = float(config["content_width_points"])
    base_height = float(config["content_height_points"])
    base_radius = float(config["corner_radius_points"])
    base_mag = float(config["core_magnification"])
    base_band = float(config["band_width_points"])
    base_tail = float(config["tail_width_points"])
    base_ring = float(config["ring_amplitude_points"])
    base_tail_amp = float(config["tail_amplitude_points"])

    seam_end = 0.26
    spread = _clamp01(progress / seam_end)
    bloom = _clamp01((progress - seam_end) / (1.0 - seam_end))
    bloom_eased = _snap_ease_in(bloom)

    seed_width = max(24.0, min(base_width * 0.06, 72.0))
    seed_height = max(2.5, min(base_height * 0.028, 7.0))
    slit_width = _lerp(seed_width, base_width, spread)
    slit_height = _lerp(seed_height, max(seed_height, base_height * 0.045), spread)

    config["content_width_points"] = _lerp(slit_width, base_width, bloom_eased)
    config["content_height_points"] = _lerp(slit_height, base_height, bloom_eased)
    config["corner_radius_points"] = min(
        _lerp(max(slit_height * 0.5, 1.0), base_radius, bloom_eased),
        float(config["content_height_points"]) * 0.5,
        float(config["content_width_points"]) * 0.5,
    )
    config["core_magnification"] = _lerp(max(1.0, base_mag * 0.18), base_mag, _smoothstep(progress))
    config["band_width_points"] = _lerp(base_band * 0.25, base_band, _smoothstep(progress))
    config["tail_width_points"] = _lerp(base_tail * 0.25, base_tail, _smoothstep(progress))
    config["ring_amplitude_points"] = _lerp(base_ring * 0.35, base_ring, _smoothstep(progress))
    config["tail_amplitude_points"] = _lerp(base_tail_amp * 0.35, base_tail_amp, _smoothstep(progress))
    config["continuous_present"] = True
    return config


def _dismiss_main_pressure_slit_config(
    request: OpticalFieldRequest,
    params: Mapping[str, Any],
    slot_name: str,
) -> dict[str, Any]:
    config = _base_shell_config(request, slot_name=slot_name, params=params)
    progress = _clamp01(request.progress)
    if progress >= 1.0:
        config["visible"] = False
        config["enabled"] = False
        return config

    close = 1.0 - progress
    height_scale = max(0.035, _snap_ease_in(close))
    config["content_height_points"] = max(3.0, float(config["content_height_points"]) * height_scale)
    config["corner_radius_points"] = min(
        float(config["corner_radius_points"]),
        float(config["content_height_points"]) * 0.5,
        float(config["content_width_points"]) * 0.5,
    )
    config["shell_fill_height_frac"] = height_scale
    config["shell_fill_opacity"] = _smoothstep(close)
    config["continuous_present"] = True
    return config


def _dismiss_seam_progress(close_progress: float) -> float:
    # The tuned preview path vanished at about 0.42; that is now the normalized
    # terminus for the seam-specific pucker field.
    return 0.42 * _clamp01(close_progress)


def _dismiss_seam_length(close_progress: float) -> float:
    return _lerp(0.8, 0.0, close_progress)


def _dismiss_seam_pressure_slit_config(
    request: OpticalFieldRequest,
    params: Mapping[str, Any],
    slot_name: str,
) -> dict[str, Any]:
    bounds = request.bounds
    scale = bounds.min_dimension
    close_progress = _clamp01(request.progress)
    config = _base_shell_config(request, slot_name=slot_name, params=params)
    field_height = max(96.0, bounds.height * 0.72)
    config.update(
        {
            "client_id": f"{request.caller_id}.dismiss_seam",
            "z_index": int(request.z_index) + 10,
            "content_width_points": float(bounds.width),
            "content_height_points": float(field_height),
            "corner_radius_points": min(scale * 0.15, bounds.width * 0.5, field_height * 0.5),
            "center_x": float(bounds.center_x),
            "center_y": float(bounds.center_y),
            "core_magnification": 1.0,
            "band_width_points": scale * 0.08,
            "tail_width_points": scale * 0.055,
            "ring_amplitude_points": 0.0,
            "tail_amplitude_points": 0.0,
            "mip_blur_strength": 0.0,
            "cleanup_blur_radius_points": 0.0,
            "cleanup_blur_strength": 0.0,
            "warp_mode": 3,
            "scar_amount": 2.0 * _smoothstep(close_progress),
            "scar_preview_progress": _dismiss_seam_progress(close_progress),
            "scar_latch_start": 0.0,
            "scar_seam_length": _dismiss_seam_length(close_progress),
            "scar_seam_thickness": 0.15,
            "scar_seam_focus": 1.0,
            "scar_vertical_grip": 1.0,
            "scar_horizontal_grip": 0.60,
            "scar_axis_rotation": 0.0,
            "scar_mirrored_lip": 0.0,
            "continuous_present": True,
        }
    )
    return _with_optical_field_sidecar(
        config, request, sidecar="dismiss_seam", slot_name=slot_name
    )


def _dismiss_radial_amount(progress: float) -> float:
    progress = _clamp01(progress)
    cycles = 2.35
    damping = 4.4
    return math.exp(-damping * progress) * math.cos(2.0 * math.pi * cycles * progress)


def _dismiss_radial_pressure_slit_config(
    request: OpticalFieldRequest,
    params: Mapping[str, Any],
    slot_name: str,
) -> dict[str, Any]:
    bounds = request.bounds
    scale = bounds.min_dimension
    radial_progress = _clamp01(request.progress)
    diameter = max(560.0, min(bounds.width * 0.52, bounds.height * 2.9)) * math.sqrt(10.0)
    config = _base_shell_config(request, slot_name=slot_name, params=params)
    config.update(
        {
            "client_id": f"{request.caller_id}.dismiss_radial_pucker",
            "z_index": int(request.z_index) + 9,
            "content_width_points": float(diameter),
            "content_height_points": float(diameter),
            "corner_radius_points": float(diameter * 0.5),
            "center_x": float(bounds.center_x),
            "center_y": float(bounds.center_y),
            "core_magnification": 1.0,
            "band_width_points": scale * 0.08,
            "tail_width_points": scale * 0.055,
            "ring_amplitude_points": 0.0,
            "tail_amplitude_points": 0.0,
            "mip_blur_strength": 0.0,
            "cleanup_blur_radius_points": 0.0,
            "cleanup_blur_strength": 0.0,
            "warp_mode": 2,
            "scar_amount": _dismiss_radial_amount(radial_progress) * 0.25,
            "scar_progress": radial_progress,
            "scar_vertical_grip": 1.0,
            "scar_horizontal_grip": 1.0,
            "continuous_present": True,
        }
    )
    return _with_optical_field_sidecar(
        config, request, sidecar="dismiss_radial_pucker", slot_name=slot_name
    )


def compile_optical_field_shell_configs(request: OpticalFieldRequest) -> tuple[dict[str, Any], ...]:
    """Compile one request into one or more compositor shell configs."""

    slot_name = _slot_name_for_state(request.state)
    params = _merged_profile_params(request.profile, slot_name)

    if request.state == "materialize":
        return (_materialized_pressure_slit_config(request, params, slot_name),)
    if request.state == "dismiss" and request.progress < 1.0:
        main = _dismiss_main_pressure_slit_config(request, params, slot_name)
        seam = _dismiss_seam_pressure_slit_config(request, params, slot_name)
        radial = _dismiss_radial_pressure_slit_config(request, params, slot_name)
        return (main, seam, radial)
    return (_base_shell_config(request, slot_name=slot_name, params=params),)


def compile_optical_field_shell_config(request: OpticalFieldRequest) -> dict[str, Any]:
    """Compile the primary compositor shell config for compatibility callers."""

    return compile_optical_field_shell_configs(request)[0]


def compile_placeholder_shell_config(request: OpticalFieldRequest) -> dict[str, Any]:
    """Compatibility wrapper for the original placeholder compiler name."""

    return compile_optical_field_shell_config(request)


class OpticalFieldBackend:
    """In-memory backend for consumers targeting the optical-field contract."""

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
        configs: list[dict[str, Any]] = []
        for request in self._requests.values():
            if request.visible:
                configs.extend(compile_optical_field_shell_configs(request))
        return tuple(configs)


class OpticalFieldPlaceholderBackend(OpticalFieldBackend):
    """Compatibility alias for callers still using the old placeholder name."""
