"""Command response overlay.

A semi-transparent overlay for displaying streamed command responses. Visually
kin to the transcription overlay (same ethereal transparency, same floating
treatment) but differentiated by color and rhythm. The input overlay breathes
with voice amplitude. The output overlay pulses with a slow ease-in/ease-out
rhythm — mechanical but gentle, distinct from the organic input. Color is a
slow full-spectrum hue rotation (~6s cycle with velocity undulation), so the
overlay always appears to be roughly one color but you can never quite pin
down which one.
"""

from __future__ import annotations

import logging
import math
import numbers
import os
import time
from collections.abc import Callable
from types import SimpleNamespace

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSScreen,
    NSScrollView,
    NSTextView,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
)
from Foundation import NSMakeRect, NSObject, NSRunLoop, NSTimer
from Quartz import CALayer, CAShapeLayer, CGPathCreateWithRoundedRect
try:
    from Quartz import CATransaction
except ImportError:  # pragma: no cover - unavailable in lightweight test fakes
    CATransaction = None

from .backdrop_stream import (
    _apply_optical_shell_warp_ci_image,
    _debug_shell_grid_ci_image,
    make_backdrop_renderer,
)
from .overlay import (
    _OVERLAY_WINDOW_LEVEL,
    _post_overlay_result_to_main,
    _start_overlay_fill_worker,
)
from .agent_shell_card_renderer import (
    build_agent_shell_card_optical_field_payload,
    build_agent_shell_card_render_payload,
)
from .agent_thread_hud import build_agent_thread_hud
from .optical_shell_metrics import OpticalShellMetrics

logger = logging.getLogger(__name__)
_BACKDROP_DISPLAY_LAYER_CLASS = None

def _env(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in {"0", "false", "False", "no", "off"}


_OVERLAY_WIDTH = _env("SPOKE_COMMAND_OVERLAY_WIDTH", 600.0)
_OVERLAY_HEIGHT = _env("SPOKE_COMMAND_OVERLAY_HEIGHT", 80.0)
_COMMAND_OVERLAY_WINDOW_LEVEL = _OVERLAY_WINDOW_LEVEL + 1
_OVERLAY_BOTTOM_MARGIN = _env("SPOKE_COMMAND_OVERLAY_BOTTOM_MARGIN", 300.0)
_OVERLAY_TOP_MARGIN = _env("SPOKE_COMMAND_OVERLAY_TOP_MARGIN", 140.0)
_AGENT_SHELL_MAX_BOTTOM_MARGIN = _env("SPOKE_AGENT_SHELL_OVERLAY_BOTTOM_MARGIN", 240.0)
_AGENT_SHELL_MAX_TOP_MARGIN = _env("SPOKE_AGENT_SHELL_OVERLAY_TOP_MARGIN", 80.0)
_OVERLAY_CORNER_RADIUS = _env("SPOKE_COMMAND_OVERLAY_CORNER_RADIUS", 16.0)
_FONT_SIZE = 15.5
_APPROVAL_HEADER_TEXT = "Approval needed"
_APPROVAL_ACTION_TEXT = "Enter to run  ·  Delete to cancel  ·  speak or type to revise"
# Branch-local smoke aid: keep every command-overlay materialization/dismiss
# timeline coupled while tuning toward the final pressure-slit cadence.
_PRESSURE_SLIT_SMOKE_TIME_SCALE = 2.0 / 3.0
_FADE_IN_S = 0.16 * _PRESSURE_SLIT_SMOKE_TIME_SCALE
_ENTRANCE_POP_SCALE = 1.015  # ~1mm overshoot on a 600px overlay
_ENTRANCE_POP_S = 0.15 * _PRESSURE_SLIT_SMOKE_TIME_SCALE
_OPTICAL_MATERIALIZATION_BASE_S = 1.36 * _PRESSURE_SLIT_SMOKE_TIME_SCALE
_OPTICAL_MATERIALIZATION_BASE_SPREAD_END = 0.77
_OPTICAL_MATERIALIZATION_SEAM_OPEN_SPEEDUP = 2.0
_OPTICAL_MATERIALIZATION_POST_SPREAD_TIME_SCALE = 2.0
_OPTICAL_MATERIALIZATION_SEAM_OPEN_S = (
    _OPTICAL_MATERIALIZATION_BASE_S * _OPTICAL_MATERIALIZATION_BASE_SPREAD_END
    / _OPTICAL_MATERIALIZATION_SEAM_OPEN_SPEEDUP
)
_OPTICAL_MATERIALIZATION_S = (
    _OPTICAL_MATERIALIZATION_SEAM_OPEN_S
    + (
        _OPTICAL_MATERIALIZATION_BASE_S
        - _OPTICAL_MATERIALIZATION_SEAM_OPEN_S
    )
    * _OPTICAL_MATERIALIZATION_POST_SPREAD_TIME_SCALE
)
_OPTICAL_MATERIALIZATION_DISMISS_S = _OPTICAL_MATERIALIZATION_BASE_S
_OPTICAL_MATERIALIZATION_PUCKER_TAIL_S = 0.75 * _PRESSURE_SLIT_SMOKE_TIME_SCALE
_OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS = 0.42
_OPTICAL_MATERIALIZATION_PUCKER_PREARM_TAIL_PROGRESS = 0.12
_OPTICAL_MATERIALIZATION_SEAM_LATCH_START = 0.0
_OPTICAL_MATERIALIZATION_SEAM_LATCH_INTENSITY = 2.0
_OPTICAL_MATERIALIZATION_SEAM_LENGTH_FRAC = 0.8
_OPTICAL_MATERIALIZATION_SEAM_LENGTH_CLOSED_FRAC = 0.0
_OPTICAL_MATERIALIZATION_SEAM_THICKNESS_FRAC = 0.15
_OPTICAL_MATERIALIZATION_SEAM_FOCUS_FRAC = 1.0
_OPTICAL_MATERIALIZATION_SEAM_VERTICAL_GRIP = 1.0
_OPTICAL_MATERIALIZATION_SEAM_HORIZONTAL_GRIP = 0.60
_OPTICAL_MATERIALIZATION_SEAM_AXIS_ROTATION = 0.0
_OPTICAL_MATERIALIZATION_SEAM_MIRRORED_LIP = 0.0
_OPTICAL_MATERIALIZATION_SEAM_FIELD_HEIGHT_FRAC = 0.72
_OPTICAL_MATERIALIZATION_SEAM_FIELD_MIN_HEIGHT_POINTS = 96.0
_DISMISS_SEAM_CLIENT_ID = "assistant.command.dismiss_seam"
_DISMISS_RADIAL_PUCKER_CLIENT_ID = "assistant.command.dismiss_radial_pucker"
_SEAM_PUCKER_TUNING_CLIENT_ID = "assistant.seam_pucker_tuner"
_OPTICAL_MATERIALIZATION_RADIAL_PUCKER_INTENSITY = 0.25
_OPTICAL_MATERIALIZATION_RADIAL_AREA_MULTIPLIER = 10.0
_OPTICAL_MATERIALIZATION_PUCKER_DIAGNOSTIC_GAIN = 5.0
_OPTICAL_MATERIALIZATION_PUCKER_GAIN_PEAK_AT = 0.30
_OPTICAL_MATERIALIZATION_RADIAL_CYCLES = 2.35
_OPTICAL_MATERIALIZATION_RADIAL_DAMPING = 4.4
_OPTICAL_MATERIALIZATION_DISMISS_TOTAL_S = (
    _OPTICAL_MATERIALIZATION_DISMISS_S
    + _OPTICAL_MATERIALIZATION_PUCKER_TAIL_S
)
_OPTICAL_MATERIALIZATION_BODY_READY = 0.55
_OPTICAL_MATERIALIZATION_SEED_WIDTH_FRAC = 0.06
_OPTICAL_MATERIALIZATION_SEED_HEIGHT_FRAC = 0.028
_OPTICAL_MATERIALIZATION_SPREAD_END = (
    _OPTICAL_MATERIALIZATION_SEAM_OPEN_S / _OPTICAL_MATERIALIZATION_S
)
_OPTICAL_MATERIALIZATION_BLOOM_START = _OPTICAL_MATERIALIZATION_SPREAD_END
_OPTICAL_MATERIALIZATION_MAG_SEED_FRAC = 0.04
_OPTICAL_MATERIALIZATION_MAG_ACCEL_END = 0.42
_OPTICAL_MATERIALIZATION_MAG_OVERSHOOT_AT = 0.72
_OPTICAL_MATERIALIZATION_MAG_OVERSHOOT = 1.20
_OPTICAL_MATERIAL_FILL_START = _OPTICAL_MATERIALIZATION_SPREAD_END
_OPTICAL_MATERIAL_FILL_SOLID_AT = (
    _OPTICAL_MATERIALIZATION_SEAM_OPEN_S
    + (
        _OPTICAL_MATERIALIZATION_BASE_S * 0.84
        - _OPTICAL_MATERIALIZATION_SEAM_OPEN_S
    )
    * _OPTICAL_MATERIALIZATION_POST_SPREAD_TIME_SCALE
) / _OPTICAL_MATERIALIZATION_S
_OPTICAL_MATERIAL_FILL_FULL_AT = (
    _OPTICAL_MATERIALIZATION_SEAM_OPEN_S
    + (
        _OPTICAL_MATERIALIZATION_BASE_S * 0.96
        - _OPTICAL_MATERIALIZATION_SEAM_OPEN_S
    )
    * _OPTICAL_MATERIALIZATION_POST_SPREAD_TIME_SCALE
) / _OPTICAL_MATERIALIZATION_S
_OPTICAL_MATERIAL_FILL_MIN_HEIGHT_FRAC = 0.011
_OPTICAL_MATERIALIZATION_PUCKER_PREARM_START_PROGRESS = (
    _OPTICAL_MATERIAL_FILL_SOLID_AT
    + (
        _OPTICAL_MATERIAL_FILL_FULL_AT
        - _OPTICAL_MATERIAL_FILL_SOLID_AT
    )
    * (
        (1.0 / 3.0 - _OPTICAL_MATERIAL_FILL_MIN_HEIGHT_FRAC)
        / (1.0 - _OPTICAL_MATERIAL_FILL_MIN_HEIGHT_FRAC)
    )
    ** (1.0 / 3.0)
)
_OPTICAL_MATERIALIZATION_SEAM_OVERLAP_START_PROGRESS = (
    _OPTICAL_MATERIALIZATION_PUCKER_PREARM_START_PROGRESS
)
_OPTICAL_MATERIALIZATION_SEAM_PEAK_PROGRESS = _OPTICAL_MATERIAL_FILL_SOLID_AT
_OPTICAL_ENTRANCE_READY_POLL_S = max(
    0.004,
    _env("SPOKE_COMMAND_OPTICAL_ENTRANCE_READY_POLL_S", 1.0 / 120.0),
)
_OPTICAL_ENTRANCE_READY_TIMEOUT_S = max(
    _OPTICAL_ENTRANCE_READY_POLL_S,
    _env(
        "SPOKE_COMMAND_OPTICAL_ENTRANCE_READY_TIMEOUT_S",
        0.2 * _PRESSURE_SLIT_SMOKE_TIME_SCALE,
    ),
)
_FADE_OUT_S = 0.5 * _PRESSURE_SLIT_SMOKE_TIME_SCALE
_FADE_STEPS = 15
_DISMISS_DURATION_S = 0.2 * _PRESSURE_SLIT_SMOKE_TIME_SCALE
_DISMISS_GROW_S = 0.06 * _PRESSURE_SLIT_SMOKE_TIME_SCALE
_DISMISS_SHRINK_S = _DISMISS_DURATION_S - _DISMISS_GROW_S
_DISMISS_ANIM_FPS = 144.0
_DISMISS_GROW_SCALE = 1.018
_DISMISS_END_SCALE = 0.94

def _max_overlay_height(screen_height: float, *, agent_shell: bool = False) -> float:
    bottom_margin = _AGENT_SHELL_MAX_BOTTOM_MARGIN if agent_shell else _OVERLAY_BOTTOM_MARGIN
    top_margin = _AGENT_SHELL_MAX_TOP_MARGIN if agent_shell else _OVERLAY_TOP_MARGIN
    return max(_OVERLAY_HEIGHT, screen_height - bottom_margin - top_margin)


def _approval_card_ranges(text: str) -> dict[str, tuple[int, int]]:
    """Return semantic ranges for a pending approval card embedded in text."""
    start = text.find(_APPROVAL_HEADER_TEXT)
    if start < 0:
        return {}

    ranges: dict[str, tuple[int, int]] = {}
    command_seen = False
    cursor = start
    for raw_line in text[start:].splitlines(True):
        content = raw_line[:-1] if raw_line.endswith("\n") else raw_line
        stripped = content.strip()
        if stripped:
            leading = len(content) - len(content.lstrip())
            rng = (cursor + leading, len(stripped))
            if stripped == _APPROVAL_HEADER_TEXT:
                ranges.setdefault("header", rng)
            elif (
                stripped == _APPROVAL_ACTION_TEXT
                or ("Enter to run" in stripped and "Delete to cancel" in stripped)
            ):
                ranges.setdefault("action", rng)
            elif stripped.startswith("reason:"):
                ranges.setdefault("reason", rng)
            elif stripped.startswith("cwd:"):
                ranges.setdefault("cwd", rng)
            elif not command_seen:
                ranges.setdefault("command", rng)
                command_seen = True
        cursor += len(raw_line)
    return ranges

# Assistant glow: full spectrum rotation with velocity undulation
_COLOR_CYCLE_PERIOD = _env("SPOKE_COMMAND_COLOR_PERIOD", 6.0)  # seconds per full hue rotation
_COLOR_VELOCITY_PERIOD = 7.0  # velocity oscillation period (out of phase with everything)
_COLOR_VELOCITY_MIN = 0.3  # slowest speed multiplier (dwells)
_COLOR_VELOCITY_MAX = 1.7  # fastest speed multiplier (transitions)
_GLOW_COLOR = (0.6, 0.4, 0.9)  # initial color for setup (violet)
_TEXT_ALPHA_MIN = _env("SPOKE_COMMAND_TEXT_ALPHA_MIN", 0.35)
_TEXT_ALPHA_MAX = _env("SPOKE_COMMAND_TEXT_ALPHA_MAX", 1.0)
_NARRATOR_OVERLAP_TEXT_HEIGHT = 48.0
_AGENT_SHELL_CHROME_ALPHA = 0.92
_AGENT_SHELL_CHROME_FALLBACK_FONT_SIZE = 10.5
_AGENT_SHELL_HEADER_HEIGHT = 18.0
_AGENT_SHELL_FOOTER_HEIGHT = 20.0
_AGENT_SHELL_CHROME_X = 28.0
_AGENT_SHELL_HEADER_TOP_INSET = 10.0
_AGENT_SHELL_FOOTER_BOTTOM_INSET = 14.0
_AGENT_SHELL_TRANSCRIPT_GAP = 28.0
_AGENT_SHELL_VERTICAL_PAD = (
    _AGENT_SHELL_HEADER_TOP_INSET
    + _AGENT_SHELL_HEADER_HEIGHT
    + _AGENT_SHELL_TRANSCRIPT_GAP
    + _AGENT_SHELL_FOOTER_BOTTOM_INSET
    + _AGENT_SHELL_FOOTER_HEIGHT
    + _AGENT_SHELL_TRANSCRIPT_GAP
)
_USER_TEXT_ALPHA_MIN = _env("SPOKE_COMMAND_USER_TEXT_ALPHA_MIN", 0.85)
_USER_TEXT_ALPHA_MAX = _env("SPOKE_COMMAND_USER_TEXT_ALPHA_MAX", 0.95)
_USER_FONT_SIZE = 13.5
_TRANSCRIPT_TEXT_VERTICAL_INSET = 8.0
_ASSISTANT_TEXT_ALPHA_MIN = _env("SPOKE_COMMAND_ASSISTANT_TEXT_ALPHA_MIN", 0.85)
_ASSISTANT_TEXT_ALPHA_MAX = _env("SPOKE_COMMAND_ASSISTANT_TEXT_ALPHA_MAX", 0.95)
_BG_ALPHA = _env("SPOKE_COMMAND_BG_ALPHA", 0.715)
_FILL_OPACITY_MIN = _env("SPOKE_COMMAND_FILL_OPACITY_MIN", 0.85)
_FILL_OPACITY_MAX = _env("SPOKE_COMMAND_FILL_OPACITY_MAX", 0.95)
_PULSE_PERIOD = _env("SPOKE_COMMAND_PULSE_PERIOD", 2.0)  # base period (seconds)
_PULSE_PERIOD_USER = _PULSE_PERIOD * 1.5  # user text: 50% slower
_PULSE_PERIOD_ASST = 5.0  # assistant text: slow deep breath
_PULSE_PHASE_OFFSET_USER = 0.3  # user starts 30% ahead in phase
_PULSE_HZ = 15.0  # timer frequency for pulse animation (15 Hz sufficient for 2-5s breathing period)

_OUTER_FEATHER = 220.0  # match preview overlay — room for the stretched-exp tails
_OPTICAL_SHELL_FEATHER = 140.0  # ~2 inches — the glow tail is faint but the eye catches
                                # any hard clip, so the window needs room for the full decay
_INNER_GLOW_DEPTH = 30.0
_OUTER_GLOW_PEAK_TARGET = 0.35
_BRIGHTNESS_CHASE = _env("SPOKE_COMMAND_BRIGHTNESS_CHASE", 0.48)
_BRIGHTNESS_CROSSING_CHASE = _env("SPOKE_COMMAND_BRIGHTNESS_CROSSING_CHASE", 0.86)
_BRIGHTNESS_COMPOSITOR_SAMPLE_TICKS = max(
    1,
    int(round(_env("SPOKE_COMMAND_BRIGHTNESS_COMPOSITOR_SAMPLE_TICKS", 2.0))),
)
_BRIGHTNESS_COMPOSITOR_STARTUP_GRACE_TICKS = max(
    0,
    int(round(_env("SPOKE_COMMAND_BRIGHTNESS_COMPOSITOR_STARTUP_GRACE_TICKS", 4.0))),
)
_BRIGHTNESS_SAMPLE_INTERVAL = 1.0
_POINTS_PER_CM = 72.0 / 2.54
_COMMAND_MATERIAL_FILL_OVERSCAN_POINTS = (
    _env("SPOKE_COMMAND_MATERIAL_FILL_OVERSCAN_MM", 1.5) / 10.0 * _POINTS_PER_CM
)
_COMMAND_BACKDROP_OVERSCAN_CM = _env("SPOKE_COMMAND_BACKDROP_OVERSCAN_CM", 1.5)
_COMMAND_BACKDROP_BLUR_RADIUS = _env("SPOKE_COMMAND_BACKDROP_BLUR_RADIUS", 9.0)
_COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER = _env(
    "SPOKE_COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER", 3.0
)
_COMMAND_BACKDROP_PULSE_TIERS = max(
    1, int(round(_env("SPOKE_COMMAND_BACKDROP_PULSE_TIERS", 4.0)))
)
_COMMAND_BACKDROP_PULSE_ATTACK = _env("SPOKE_COMMAND_BACKDROP_PULSE_ATTACK", 0.38)
_COMMAND_BACKDROP_PULSE_RELEASE = _env("SPOKE_COMMAND_BACKDROP_PULSE_RELEASE", 0.12)
_COMMAND_BACKDROP_PULSE_BLUR_MIN_MULTIPLIER = _env(
    "SPOKE_COMMAND_BACKDROP_PULSE_BLUR_MIN_MULTIPLIER", 0.74
)
_COMMAND_BACKDROP_PULSE_BLUR_MAX_MULTIPLIER = _env(
    "SPOKE_COMMAND_BACKDROP_PULSE_BLUR_MAX_MULTIPLIER", 1.18
)
_COMMAND_BACKDROP_PULSE_MASK_MIN_MULTIPLIER = _env(
    "SPOKE_COMMAND_BACKDROP_PULSE_MASK_MIN_MULTIPLIER", 0.8
)
_COMMAND_BACKDROP_PULSE_MASK_MAX_MULTIPLIER = _env(
    "SPOKE_COMMAND_BACKDROP_PULSE_MASK_MAX_MULTIPLIER", 1.55
)
_COMMAND_BACKDROP_PULSE_OPACITY_MIN = _env(
    "SPOKE_COMMAND_BACKDROP_PULSE_OPACITY_MIN", 0.86
)
_COMMAND_BACKDROP_PULSE_OPACITY_MAX = _env(
    "SPOKE_COMMAND_BACKDROP_PULSE_OPACITY_MAX", 1.0
)
_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED = _env_bool(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", False
)
_COMMAND_BACKDROP_OPTICAL_SHELL_CORE_MAGNIFICATION = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_CORE_MAGNIFICATION", 1.55
)
_COMMAND_BACKDROP_OPTICAL_SHELL_BAND_MM = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_BAND_MM", 4.0
)
_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_MM = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_MM", 3.0
)
_COMMAND_BACKDROP_OPTICAL_SHELL_RING_REFRACTION_LEGACY = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_REFRACTION", 2.6
)
_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_REFRACTION_LEGACY = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_REFRACTION", 0.75
)
_COMMAND_BACKDROP_OPTICAL_SHELL_RING_AMPLITUDE_POINTS = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_RING_AMPLITUDE_POINTS",
    (_COMMAND_BACKDROP_OPTICAL_SHELL_BAND_MM / 10.0)
    * _POINTS_PER_CM
    * _COMMAND_BACKDROP_OPTICAL_SHELL_RING_REFRACTION_LEGACY,
)
_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_AMPLITUDE_POINTS = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_AMPLITUDE_POINTS",
    (_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_MM / 10.0)
    * _POINTS_PER_CM
    * _COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_REFRACTION_LEGACY,
)
_COMMAND_BACKDROP_OPTICAL_SHELL_CLEANUP_BLUR_RADIUS = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_CLEANUP_BLUR_RADIUS", 0.75
)
_COMMAND_BACKDROP_OPTICAL_SHELL_INFLATION_X_RADII = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_INFLATION_X_RADII", 1.0
)
_COMMAND_BACKDROP_OPTICAL_SHELL_INFLATION_Y_RADII = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_INFLATION_Y_RADII", 1.0
)
_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MIN_DARK = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MIN_DARK", 0.08
)
_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MAX_DARK = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MAX_DARK", 0.18
)
_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MIN_LIGHT = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MIN_LIGHT", 0.14
)
_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MAX_LIGHT = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MAX_LIGHT", 0.38
)
_COMMAND_BACKDROP_OPTICAL_SHELL_SPRING_OPACITY_SCALE = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_SPRING_OPACITY_SCALE", 0.16
)
_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_REVEAL = _env_bool(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_REVEAL", False
)
_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE = _env_bool(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE", False
)
_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_MASK_WIDTH_MULTIPLIER = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_MASK_WIDTH_MULTIPLIER", 0.18
)
_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_GRID_SPACING_POINTS = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_GRID_SPACING_POINTS", 18.0
)
_COMMAND_BACKDROP_REFRESH_S = _env("SPOKE_COMMAND_BACKDROP_REFRESH_S", 1.0 / 30.0)
_COMMAND_VISUAL_START_DELAY_S = _env(
    "SPOKE_COMMAND_VISUAL_START_DELAY_S",
    _FADE_IN_S + 0.15 * _PRESSURE_SLIT_SMOKE_TIME_SCALE,
)
_RUN_LOOP_COMMON_MODE = "NSRunLoopCommonModes"
_EVENT_TRACKING_RUN_LOOP_MODE = "NSEventTrackingRunLoopMode"

# Adaptive compositing for command output.
_USER_TEXT_COLOR_DARK = (0.16, 0.17, 0.20)
_USER_TEXT_COLOR_LIGHT = (0.95, 0.97, 1.0)
_ASSISTANT_TEXT_COLOR_DARK = (0.12, 0.13, 0.16)
_ASSISTANT_TEXT_COLOR_LIGHT = (0.95, 0.97, 1.0)
_ASSISTANT_BLUR_RADIUS = _env("SPOKE_COMMAND_ASSISTANT_BLUR_RADIUS", 8.0)
_COMMAND_RESPONSE_ANIMATION_CHAR_LIMIT = max(
    1,
    int(round(_env("SPOKE_COMMAND_RESPONSE_ANIMATION_CHAR_LIMIT", 700.0))),
)
_THINKING_CUTOUT_DARK = (0.05, 0.05, 0.06)
_THINKING_CUTOUT_LIGHT = (0.80, 0.80, 0.78)


def _backdrop_display_layer_class():
    global _BACKDROP_DISPLAY_LAYER_CLASS
    if _BACKDROP_DISPLAY_LAYER_CLASS is not None:
        return _BACKDROP_DISPLAY_LAYER_CLASS
    if not hasattr(objc, "loadBundle") or not hasattr(objc, "lookUpClass"):
        _BACKDROP_DISPLAY_LAYER_CLASS = False
        return None
    try:
        objc.loadBundle(
            "AVFoundation",
            globals(),
            bundle_path="/System/Library/Frameworks/AVFoundation.framework",
        )
        _BACKDROP_DISPLAY_LAYER_CLASS = objc.lookUpClass("AVSampleBufferDisplayLayer")
    except Exception:
        logger.debug("AVSampleBufferDisplayLayer unavailable for command backdrop", exc_info=True)
        _BACKDROP_DISPLAY_LAYER_CLASS = False
    return _BACKDROP_DISPLAY_LAYER_CLASS or None


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def _smoothstep(progress: float) -> float:
    t = _clamp01(progress)
    return t * t * (3.0 - 2.0 * t)


def _snap_ease_in(progress: float) -> float:
    t = _clamp01(progress)
    return t * t * t


def _lerp_color(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    return tuple(_lerp(s, e, t) for s, e in zip(start, end))


def _background_color_for_brightness(brightness: float) -> tuple[float, float, float]:
    from .overlay import _BG_COLOR_DARK as _PREVIEW_BG_COLOR_DARK, _BG_COLOR_LIGHT as _PREVIEW_BG_COLOR_LIGHT

    return _lerp_color(_PREVIEW_BG_COLOR_DARK, _PREVIEW_BG_COLOR_LIGHT, _clamp01(brightness))


# Compositor graphic mode: fill matches background tone so text
# punch-through reveals the contrasting warped content underneath.
# Dark on dark, light on light — the overlay is a surface, not a glow.
_COMPOSITOR_FILL_DARK = (0.50, 0.51, 0.54)   # light fill on dark backgrounds — faint, translucent
_COMPOSITOR_FILL_LIGHT = (0.04, 0.04, 0.05)   # dark fill on light backgrounds — vivid, near-black
_COMPOSITOR_LIGHT_FILL_ALPHA_BOOST = 1.50
_PUNCHTHROUGH_BOOST_DARK = (0.0, 0.0, 0.0)
_PUNCHTHROUGH_BOOST_LIGHT = (1.0, 1.0, 1.0)
_PUNCHTHROUGH_BOOST_OPACITY_DARK = 0.42
_PUNCHTHROUGH_BOOST_OPACITY_LIGHT = 0.75


def _compositor_fill_choice_for_brightness(brightness: float) -> float:
    # Steep sigmoid: commits to light or dark fill quickly, doesn't
    # linger in a bland mid-tone.  The transition is centered at 0.45
    # (biased slightly toward dark-fill) and covers ~0.15 of the
    # brightness range, so most backgrounds get a decisive choice.
    t = _clamp01(brightness)
    # Remap to steep sigmoid: 6x gain centered at 0.45
    t = _clamp01((t - 0.45) * 6.0 + 0.5)
    return t * t * (3.0 - 2.0 * t)  # smoothstep for clean edges


def _compositor_fill_color_for_brightness(brightness: float) -> tuple[float, float, float]:
    t = _compositor_fill_choice_for_brightness(brightness)
    return _lerp_color(_COMPOSITOR_FILL_DARK, _COMPOSITOR_FILL_LIGHT, t)


def _compositor_fill_alpha_multiplier_for_brightness(brightness: float) -> float:
    # Only boost the light material used on dark backgrounds. The dark material
    # on light backgrounds is already visually dense enough and should not drift.
    return _lerp(
        _COMPOSITOR_LIGHT_FILL_ALPHA_BOOST,
        1.0,
        _compositor_fill_choice_for_brightness(brightness),
    )


def _punchthrough_boost_style_for_brightness(
    brightness: float,
) -> tuple[tuple[float, float, float], float]:
    t = _clamp01(brightness)
    t = _clamp01((t - 0.45) * 6.0 + 0.5)
    t = t * t * (3.0 - 2.0 * t)
    return (
        _lerp_color(_PUNCHTHROUGH_BOOST_DARK, _PUNCHTHROUGH_BOOST_LIGHT, t),
        _lerp(_PUNCHTHROUGH_BOOST_OPACITY_DARK, _PUNCHTHROUGH_BOOST_OPACITY_LIGHT, t),
    )


def _user_text_color_for_brightness(brightness: float) -> tuple[float, float, float]:
    return _lerp_color(
        _USER_TEXT_COLOR_DARK,
        _USER_TEXT_COLOR_LIGHT,
        _contrast_mix_for_brightness(brightness),
    )

def _assistant_foreground_color_for_brightness(brightness: float) -> tuple[float, float, float]:
    return _lerp_color(
        _ASSISTANT_TEXT_COLOR_DARK,
        _ASSISTANT_TEXT_COLOR_LIGHT,
        _contrast_mix_for_brightness(brightness),
    )


def _user_text_alpha_for_breath(breath: float) -> float:
    return _lerp(_USER_TEXT_ALPHA_MIN, _USER_TEXT_ALPHA_MAX, _clamp01(breath))


def _contrast_mix_for_brightness(brightness: float) -> float:
    clamped = _clamp01(brightness)
    lo = 0.44
    hi = 0.56
    if clamped <= lo:
        return 0.0
    if clamped >= hi:
        return 1.0
    t = (clamped - lo) / (hi - lo)
    return t * t * (3.0 - 2.0 * t)


def _advance_command_brightness(current: float, target: float) -> float:
    """Chase brightness, but cross the contrast switch band decisively."""
    current = _clamp01(current)
    target = _clamp01(target)
    delta = target - current
    if abs(delta) <= 0.001:
        return target
    crosses_contrast_band = (
        (current <= 0.44 and target >= 0.56)
        or (current >= 0.56 and target <= 0.44)
    )
    chase = _BRIGHTNESS_CROSSING_CHASE if crosses_contrast_band else _BRIGHTNESS_CHASE
    return _clamp01(current + delta * chase)


def _sample_screen_brightness_for_overlay(screen) -> float:
    from .glow import _sample_screen_brightness

    return _sample_screen_brightness(screen)


def _thinking_cutout_color_for_brightness(brightness: float) -> tuple[float, float, float]:
    return _lerp_color(_THINKING_CUTOUT_DARK, _THINKING_CUTOUT_LIGHT, _clamp01(brightness))


def _assistant_text_alpha_for_breath(breath: float) -> float:
    return _lerp(_ASSISTANT_TEXT_ALPHA_MIN, _ASSISTANT_TEXT_ALPHA_MAX, _clamp01(breath))


def _fill_layer_opacity_for_breath(breath: float) -> float:
    return _lerp(_FILL_OPACITY_MIN, _FILL_OPACITY_MAX, _clamp01(breath))


def _quantize_unit_interval(value: float, steps: int) -> float:
    clamped = _clamp01(value)
    if steps <= 1:
        return clamped
    return round(clamped * (steps - 1)) / float(steps - 1)


def _advance_attack_release(current: float, target: float, *, attack: float, release: float) -> float:
    rate = attack if target > current else release
    return current + (_clamp01(target) - current) * _clamp01(rate)


def _command_backdrop_pulse_style(
    base_blur_radius_points: float,
    base_mask_width_multiplier: float,
    blur_drive: float,
) -> tuple[float, float, float]:
    drive = _quantize_unit_interval(blur_drive, _COMMAND_BACKDROP_PULSE_TIERS)
    blur_radius_points = max(
        0.0,
        base_blur_radius_points
        * _lerp(
            _COMMAND_BACKDROP_PULSE_BLUR_MIN_MULTIPLIER,
            _COMMAND_BACKDROP_PULSE_BLUR_MAX_MULTIPLIER,
            drive,
        ),
    )
    mask_width_multiplier = max(
        0.0,
        base_mask_width_multiplier
        * _lerp(
            _COMMAND_BACKDROP_PULSE_MASK_MIN_MULTIPLIER,
            _COMMAND_BACKDROP_PULSE_MASK_MAX_MULTIPLIER,
            drive,
        ),
    )
    # The sampled backdrop is the replacement background inside the masked region.
    # If we fade it down during the airy phase, the live crisp desktop leaks through
    # the transparent window and the blur reads like a translucent wash instead.
    backdrop_opacity = 1.0
    return blur_radius_points, mask_width_multiplier, backdrop_opacity


def _command_backdrop_blur_target_for_presence(presence: float) -> float:
    return 1.0 - _clamp01(presence)


def _optical_shell_body_corner_radius(body_height_points: float) -> float:
    """Corner radius for the visible rounded-rect shell body.

    The optical shell is authored as a rounded rectangle, not a true capsule.
    A true capsule is just the special case where the radius reaches half the
    body height; that is not the intended look for this overlay.
    """
    body_height = max(float(body_height_points), 1.0)
    return min(_OVERLAY_HEIGHT * 0.25, body_height * 0.5)


def _command_optical_shell_config(
    content_width_points: float | None = None,
    content_height_points: float | None = None,
) -> dict[str, float | bool] | None:
    if not _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
        return None
    width_points = (
        _OVERLAY_WIDTH if content_width_points is None else max(float(content_width_points), 1.0)
    )
    height_points = (
        _OVERLAY_HEIGHT
        if content_height_points is None
        else max(float(content_height_points), 1.0)
    )
    # The rendered shell body is a rounded rectangle. Inflate the warp envelope
    # around that body, but keep the same body corner radius so fill/mask/warp
    # continue to agree on the visible shape.
    shell_body_corner_r = _optical_shell_body_corner_radius(height_points)
    width_points += _COMMAND_BACKDROP_OPTICAL_SHELL_INFLATION_X_RADII * shell_body_corner_r
    height_points += _COMMAND_BACKDROP_OPTICAL_SHELL_INFLATION_Y_RADII * shell_body_corner_r
    return {
        "enabled": True,
        "content_width_points": width_points,
        "content_height_points": height_points,
        "corner_radius_points": shell_body_corner_r,
        "core_magnification": _COMMAND_BACKDROP_OPTICAL_SHELL_CORE_MAGNIFICATION,
        "band_width_points": _cm_to_points(_COMMAND_BACKDROP_OPTICAL_SHELL_BAND_MM / 10.0),
        "tail_width_points": _cm_to_points(_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_MM / 10.0),
        "ring_amplitude_points": _COMMAND_BACKDROP_OPTICAL_SHELL_RING_AMPLITUDE_POINTS,
        "tail_amplitude_points": _COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_AMPLITUDE_POINTS,
        "debug_visualize": _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE,
        "debug_grid_spacing_points": _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_GRID_SPACING_POINTS,
        "cleanup_blur_radius_points": _COMMAND_BACKDROP_OPTICAL_SHELL_CLEANUP_BLUR_RADIUS,
    }


def _materialized_optical_shell_config(
    shell_config: dict,
    progress: float,
) -> dict:
    """Return a transient shell config for fluid entrance/dismissal.

    The materialization path only changes the compositor warp envelope. It does
    not touch the SDF fill geometry, which keeps entrance animation from
    rebuilding expensive masks every visible frame.
    """
    config = dict(shell_config)
    p = _clamp01(progress)
    if p >= 1.0:
        return config

    base_w = max(float(config.get("content_width_points", 1.0)), 1.0)
    base_h = max(float(config.get("content_height_points", 1.0)), 1.0)
    base_radius = max(float(config.get("corner_radius_points", 1.0)), 1.0)
    config["_materialization_base_width_points"] = base_w
    config["_materialization_base_height_points"] = base_h
    config["_materialization_base_corner_radius_points"] = base_radius

    spread_t = _snap_ease_in(p / _OPTICAL_MATERIALIZATION_SPREAD_END)
    bloom_t = _snap_ease_in(
        (p - _OPTICAL_MATERIALIZATION_BLOOM_START)
        / max(1.0 - _OPTICAL_MATERIALIZATION_BLOOM_START, 1e-6)
    )
    seed_w = max(24.0, min(base_w * _OPTICAL_MATERIALIZATION_SEED_WIDTH_FRAC, 72.0))
    seed_h = max(2.5, min(base_h * _OPTICAL_MATERIALIZATION_SEED_HEIGHT_FRAC, 7.0))
    width = _lerp(seed_w, base_w, spread_t)
    height = _lerp(seed_h, base_h, bloom_t)

    config["content_width_points"] = width
    config["content_height_points"] = height
    config["corner_radius_points"] = min(base_radius, height * 0.5)
    if "core_magnification" in config:
        base_mag = max(float(config.get("core_magnification", 1.0)), 0.0)
        seed_mag = base_mag * _OPTICAL_MATERIALIZATION_MAG_SEED_FRAC
        if p <= _OPTICAL_MATERIALIZATION_MAG_ACCEL_END:
            t = _clamp01(p / _OPTICAL_MATERIALIZATION_MAG_ACCEL_END)
            config["core_magnification"] = _lerp(
                seed_mag,
                base_mag * 0.82,
                _snap_ease_in(t),
            )
        elif p <= _OPTICAL_MATERIALIZATION_MAG_OVERSHOOT_AT:
            t = _clamp01(
                (p - _OPTICAL_MATERIALIZATION_MAG_ACCEL_END)
                / (
                    _OPTICAL_MATERIALIZATION_MAG_OVERSHOOT_AT
                    - _OPTICAL_MATERIALIZATION_MAG_ACCEL_END
                )
            )
            config["core_magnification"] = _lerp(
                base_mag * 0.82,
                base_mag * _OPTICAL_MATERIALIZATION_MAG_OVERSHOOT,
                _snap_ease_in(t),
            )
        else:
            t = _clamp01(
                (p - _OPTICAL_MATERIALIZATION_MAG_OVERSHOOT_AT)
                / max(1.0 - _OPTICAL_MATERIALIZATION_MAG_OVERSHOOT_AT, 1e-6)
            )
            config["core_magnification"] = _lerp(
                base_mag * _OPTICAL_MATERIALIZATION_MAG_OVERSHOOT,
                base_mag,
                _snap_ease_in(t),
            )
    for key in ("band_width_points", "tail_width_points"):
        if key in config:
            config[key] = max(1.0, float(config[key]) * _lerp(0.25, 1.0, p))
    for key in ("ring_amplitude_points", "tail_amplitude_points"):
        if key in config:
            config[key] = float(config[key]) * _lerp(0.35, 1.0, p)
    config["continuous_present"] = True
    return config


def _materialization_fill_state(progress: float) -> dict[str, float]:
    p = _clamp01(progress)
    if p <= _OPTICAL_MATERIAL_FILL_START:
        opacity = 0.0
    else:
        opacity = _smoothstep(
            (p - _OPTICAL_MATERIAL_FILL_START)
            / max(_OPTICAL_MATERIAL_FILL_SOLID_AT - _OPTICAL_MATERIAL_FILL_START, 1e-6)
        )
    height = _lerp(
        _OPTICAL_MATERIAL_FILL_MIN_HEIGHT_FRAC,
        1.0,
        _clamp01(
            (p - _OPTICAL_MATERIAL_FILL_SOLID_AT)
            / max(_OPTICAL_MATERIAL_FILL_FULL_AT - _OPTICAL_MATERIAL_FILL_SOLID_AT, 1e-6)
        )
        ** 3.0,
    )
    return {
        "opacity": _clamp01(opacity),
        "height_frac": _clamp01(height),
    }


def _dismiss_materialization_fill_state(progress: float) -> dict[str, float]:
    """Keep the visible shell body alive until the seam sidecar owns dismissal."""
    p = _clamp01(progress)
    if p <= _OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS:
        return {
            "opacity": 0.0,
            "height_frac": _OPTICAL_MATERIAL_FILL_MIN_HEIGHT_FRAC,
        }
    state = _materialization_fill_state(p)
    return {
        "opacity": 1.0,
        "height_frac": state["height_frac"],
    }


def _dismiss_pucker_amount(progress: float) -> float:
    """Signed radial ringdown: an underdamped pucker after the seam vanishes."""
    p = _clamp01(progress)
    return math.exp(-_OPTICAL_MATERIALIZATION_RADIAL_DAMPING * p) * math.cos(
        2.0 * math.pi * _OPTICAL_MATERIALIZATION_RADIAL_CYCLES * p
    )


def _dismiss_pucker_tail_progress_for_close_progress(close_progress: float) -> float:
    """Advance the radial tail while the shell is visually shrinking away."""
    start = max(_OPTICAL_MATERIALIZATION_PUCKER_PREARM_START_PROGRESS, 1e-6)
    phase = _clamp01((start - _clamp01(close_progress)) / start)
    return _lerp(
        0.0,
        _OPTICAL_MATERIALIZATION_PUCKER_PREARM_TAIL_PROGRESS,
        phase,
    )


def _dismiss_seam_latch_amount(progress: float) -> float:
    """Positive seam pucker already present at latch start, deeper at closure."""
    p = _clamp01(progress)
    start = max(_OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS, 1e-6)
    t = _clamp01((start - p) / start)
    return _lerp(
        _OPTICAL_MATERIALIZATION_SEAM_LATCH_START,
        1.0,
        1.0 - (1.0 - t) ** 3.0,
    )


def _seam_pucker_tuning_defaults() -> dict[str, float]:
    return {
        "preview_progress": _OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS * 0.45,
        "seam_latch_start": _OPTICAL_MATERIALIZATION_SEAM_LATCH_START,
        "seam_latch_intensity": _OPTICAL_MATERIALIZATION_SEAM_LATCH_INTENSITY,
        "scar_seam_length_frac": _OPTICAL_MATERIALIZATION_SEAM_LENGTH_FRAC,
        "scar_seam_thickness_frac": _OPTICAL_MATERIALIZATION_SEAM_THICKNESS_FRAC,
        "scar_seam_focus_frac": _OPTICAL_MATERIALIZATION_SEAM_FOCUS_FRAC,
        "scar_vertical_grip": _OPTICAL_MATERIALIZATION_SEAM_VERTICAL_GRIP,
        "scar_horizontal_grip": _OPTICAL_MATERIALIZATION_SEAM_HORIZONTAL_GRIP,
        "scar_axis_rotation": _OPTICAL_MATERIALIZATION_SEAM_AXIS_ROTATION,
        "scar_mirrored_lip": _OPTICAL_MATERIALIZATION_SEAM_MIRRORED_LIP,
    }


def _dismiss_pucker_amplitude_multiplier(progress: float) -> float:
    """Diagnostic gain envelope: quick visibility peak, long elastic decay."""
    p = _clamp01(progress)
    peak_at = max(_OPTICAL_MATERIALIZATION_PUCKER_GAIN_PEAK_AT, 1e-6)
    if p <= peak_at:
        t = p / peak_at
        return _lerp(
            1.0,
            _OPTICAL_MATERIALIZATION_PUCKER_DIAGNOSTIC_GAIN,
            1.0 - (1.0 - t) ** 3.0,
        )
    t = (p - peak_at) / max(1.0 - peak_at, 1e-6)
    return _OPTICAL_MATERIALIZATION_PUCKER_DIAGNOSTIC_GAIN * math.exp(-5.0 * t)


def _apply_dismiss_seam_latch_fields(
    config: dict,
    progress: float,
    tuning: dict[str, float] | None = None,
) -> dict:
    """Apply the crisp seam latch while the slit is still zipping closed."""
    updated = dict(config)
    settings = _seam_pucker_tuning_defaults()
    if tuning:
        for key, value in tuning.items():
            if key in settings:
                settings[key] = float(value)
    p = _clamp01(progress)
    overlap_start = max(_OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS, 1e-6)
    t = _clamp01((overlap_start - p) / overlap_start)
    base_h = max(
        float(
            updated.get(
                "_materialization_base_height_points",
                updated.get("content_height_points", 1.0),
            )
        ),
        1.0,
    )
    current_h = max(float(updated.get("content_height_points", 1.0)), 1.0)
    seam_field_h = max(
        current_h,
        min(
            base_h,
            max(
                _OPTICAL_MATERIALIZATION_SEAM_FIELD_MIN_HEIGHT_POINTS,
                base_h * _OPTICAL_MATERIALIZATION_SEAM_FIELD_HEIGHT_FRAC,
            ),
        ),
    )
    amount = _lerp(
        settings["seam_latch_start"],
        1.0,
        1.0 - (1.0 - t) ** 3.0,
    ) * settings["seam_latch_intensity"]
    updated["content_height_points"] = seam_field_h
    updated["corner_radius_points"] = min(
        max(float(updated.get("corner_radius_points", 1.0)), 1.0),
        seam_field_h * 0.5,
    )
    updated["cleanup_blur_radius_points"] = 0.0
    updated["mip_blur_strength"] = 0.0
    updated["warp_mode"] = 3.0 if settings["scar_mirrored_lip"] >= 0.5 else 1.0
    updated["scar_amount"] = amount
    updated["scar_seam_length_frac"] = settings["scar_seam_length_frac"]
    updated["scar_seam_thickness_frac"] = settings["scar_seam_thickness_frac"]
    updated["scar_seam_focus_frac"] = settings["scar_seam_focus_frac"]
    updated["scar_vertical_grip"] = settings["scar_vertical_grip"]
    updated["scar_horizontal_grip"] = settings["scar_horizontal_grip"]
    updated["scar_axis_rotation"] = settings["scar_axis_rotation"]
    updated["scar_mirrored_lip"] = settings["scar_mirrored_lip"]
    updated["x_squeeze"] = 1.0
    updated["y_squeeze"] = 1.0
    updated["ring_amplitude_points"] = 0.0
    updated["tail_amplitude_points"] = 0.0
    updated["continuous_present"] = True
    return updated


def _dismiss_seam_tuning_for_close_progress(
    close_progress: float,
    tuning: dict[str, float] | None = None,
) -> dict[str, float]:
    """Map close progress onto the seam tuner path without firing peak too early."""
    settings = _seam_pucker_tuning_defaults()
    if tuning:
        for key, value in tuning.items():
            if key in settings:
                settings[key] = float(value)
    p = _clamp01(close_progress)
    arm_start = max(_OPTICAL_MATERIALIZATION_SEAM_OVERLAP_START_PROGRESS, 1e-6)
    peak = max(_OPTICAL_MATERIALIZATION_SEAM_PEAK_PROGRESS, 1e-6)
    if p >= peak:
        arm_phase = _smoothstep((arm_start - p) / max(arm_start - peak, 1e-6))
        settings["preview_progress"] = 0.0
        settings["seam_latch_intensity"] *= arm_phase
        settings["scar_seam_length_frac"] = _OPTICAL_MATERIALIZATION_SEAM_LENGTH_FRAC
        return settings

    phase = _clamp01((peak - p) / peak)
    settings["preview_progress"] = _lerp(
        0.0,
        _OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS,
        phase,
    )
    settings["scar_seam_length_frac"] = _lerp(
        _OPTICAL_MATERIALIZATION_SEAM_LENGTH_FRAC,
        _OPTICAL_MATERIALIZATION_SEAM_LENGTH_CLOSED_FRAC,
        phase,
    )
    return settings


def _dismiss_seam_latch_shell_config(
    final_shell_config: dict,
    progress: float,
    tuning: dict[str, float] | None = None,
) -> dict:
    """Return the full-width seam field layered over the close animation."""
    config = dict(final_shell_config)
    config["client_id"] = _DISMISS_SEAM_CLIENT_ID
    config["role"] = "assistant"
    config["visible"] = True
    config["z_index"] = 10
    seam_tuning = _dismiss_seam_tuning_for_close_progress(progress, tuning)
    return _apply_dismiss_seam_latch_fields(
        config,
        seam_tuning["preview_progress"],
        seam_tuning,
    )


def _apply_dismiss_radial_pucker_fields(config: dict, progress: float) -> dict:
    """Apply the post-close radial underdamped pucker without blur."""
    updated = dict(config)
    amount = (
        _dismiss_pucker_amount(progress)
        * _OPTICAL_MATERIALIZATION_RADIAL_PUCKER_INTENSITY
        * _dismiss_pucker_amplitude_multiplier(progress)
    )
    updated["cleanup_blur_radius_points"] = 0.0
    updated["mip_blur_strength"] = 0.0
    updated["warp_mode"] = 2.0
    updated["scar_amount"] = amount
    updated["x_squeeze"] = 1.0
    updated["y_squeeze"] = 1.0
    updated["ring_amplitude_points"] = 0.0
    updated["tail_amplitude_points"] = 0.0
    updated["continuous_present"] = True
    return updated


def _dismiss_pucker_shell_config(shell_config: dict, progress: float) -> dict:
    """Return the radial underdamped scar that releases after dismiss closes."""
    base_w = max(float(shell_config.get("content_width_points", 1.0)), 1.0)
    base_h = max(float(shell_config.get("content_height_points", 1.0)), 1.0)
    config = _materialized_optical_shell_config(shell_config, 0.0)
    base_diameter = max(560.0, min(base_w * 0.52, base_h * 2.9))
    diameter = base_diameter * math.sqrt(_OPTICAL_MATERIALIZATION_RADIAL_AREA_MULTIPLIER)
    config["content_width_points"] = diameter
    config["content_height_points"] = diameter
    config["corner_radius_points"] = diameter * 0.5
    config["core_magnification"] = 1.0
    return _apply_dismiss_radial_pucker_fields(config, progress)


def _dismiss_radial_pucker_shell_config(shell_config: dict, progress: float) -> dict:
    """Return the radial scar as an independent compositor client."""
    config = _dismiss_pucker_shell_config(shell_config, progress)
    config["client_id"] = _DISMISS_RADIAL_PUCKER_CLIENT_ID
    config["role"] = "assistant"
    config["visible"] = True
    config["z_index"] = 9
    return config


def _fill_compositing_filter_for_brightness(brightness: float) -> str | None:
    from .overlay import _fill_compositing_filter_for_brightness as _preview_fill_compositing_filter_for_brightness

    return _preview_fill_compositing_filter_for_brightness(brightness)


def _ease_in(progress: float) -> float:
    clamped = _clamp01(progress)
    return clamped * clamped


def _cm_to_points(cm: float) -> float:
    return max(cm, 0.0) * _POINTS_PER_CM


def _command_backdrop_capture_overscan_points() -> float:
    # The blur neighborhood only needs enough extra image to support a
    # graceful fade beyond the bubble perimeter; it should stay much tighter
    # than the full SDF feather/glow margin.
    return _cm_to_points(_COMMAND_BACKDROP_OVERSCAN_CM)


def _command_backdrop_capture_overscan_pixels(backing_scale: float) -> float:
    return _command_backdrop_capture_overscan_points() * max(backing_scale, 0.0)


def _backdrop_capture_rect(screen_frame, window_frame, content_frame, overscan_points: float):
    overscan = max(overscan_points, 0.0)
    x = window_frame.origin.x + content_frame.origin.x - overscan
    width = content_frame.size.width + 2 * overscan
    height = content_frame.size.height + 2 * overscan
    cocoa_y = window_frame.origin.y + content_frame.origin.y - overscan
    screen_origin_y = getattr(screen_frame.origin, "y", 0.0)
    screen_height = getattr(screen_frame.size, "height", 0.0)
    y = screen_origin_y + screen_height - (cocoa_y - screen_origin_y) - height
    return SimpleNamespace(
        origin=SimpleNamespace(x=x, y=y),
        size=SimpleNamespace(width=width, height=height),
    )


def _display_local_capsule_center_pixels(screen_frame, window_frame, content_frame, scale: float):
    screen_origin_x = getattr(getattr(screen_frame, "origin", None), "x", 0.0)
    screen_origin_y = getattr(getattr(screen_frame, "origin", None), "y", 0.0)
    screen_height = getattr(getattr(screen_frame, "size", None), "height", 0.0)
    if not isinstance(screen_origin_x, numbers.Real):
        screen_origin_x = 0.0
    if not isinstance(screen_origin_y, numbers.Real):
        screen_origin_y = 0.0
    if not isinstance(screen_height, numbers.Real):
        screen_height = 0.0

    capsule_cx = (
        window_frame.origin.x
        + content_frame.origin.x
        + content_frame.size.width / 2
        - screen_origin_x
    )
    capsule_cy_cocoa = (
        window_frame.origin.y
        + content_frame.origin.y
        + content_frame.size.height / 2
    )
    capsule_cy_metal = screen_origin_y + screen_height - capsule_cy_cocoa
    return capsule_cx * scale, capsule_cy_metal * scale


def _backdrop_capture_pixel_size(capture_rect, backing_scale: float) -> tuple[float, float]:
    scale = max(backing_scale, 0.0)
    return (
        capture_rect.size.width * scale,
        capture_rect.size.height * scale,
    )


def _pin_timer_to_active_run_loop_modes(timer) -> None:
    if timer is None:
        return
    try:
        run_loop = NSRunLoop.currentRunLoop()
    except Exception:
        return
    for mode in (_RUN_LOOP_COMMON_MODE, _EVENT_TRACKING_RUN_LOOP_MODE):
        try:
            run_loop.addTimer_forMode_(timer, mode)
        except Exception:
            logger.debug("Failed to add command backdrop timer to run loop mode %s", mode, exc_info=True)


def _backdrop_mask_alpha(signed_distance, width: float):
    import numpy as np

    outside = np.exp(-np.sqrt(np.maximum(signed_distance, 0.0) / max(width, 1e-6)))
    return np.where(signed_distance <= 0.0, 1.0, outside).astype(np.float32)


def _command_backdrop_mask_falloff_width(scale: float) -> float:
    return max(scale, 1e-6) * max(_COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER, 0.0)


def _stadium_signed_distance_field(
    field_width_points: float,
    field_height_points: float,
    body_width_points: float,
    body_height_points: float,
    corner_radius_points: float,
    scale: float,
):
    """Return the centered optical-shell SDF at backing-scale resolution.

    Despite the legacy helper name, this supports the rounded-rect shell body
    we actually render. It becomes a true capsule only when the supplied
    corner radius reaches half the body height.
    """
    import numpy as np

    sample_scale = max(float(scale), 1e-6)
    pw = max(int(round(field_width_points * sample_scale)), 1)
    ph = max(int(round(field_height_points * sample_scale)), 1)
    xs = (np.arange(pw, dtype=np.float32)[None, :] + 0.5) / sample_scale - field_width_points * 0.5
    ys = (np.arange(ph, dtype=np.float32)[:, None] + 0.5) / sample_scale - field_height_points * 0.5
    half_w = body_width_points * 0.5
    half_h = body_height_points * 0.5
    capsule_radius = max(min(corner_radius_points, half_h), 1.0 / sample_scale)
    spine_half_x = max(half_w - capsule_radius, 0.0)
    spine_half_y = max(half_h - capsule_radius, 0.0)
    spine_dist_x = np.maximum(np.abs(xs) - spine_half_x, 0.0)
    spine_dist_y = np.maximum(np.abs(ys) - spine_half_y, 0.0)
    return (np.hypot(spine_dist_x, spine_dist_y) - capsule_radius).astype(np.float32)


def _boundary_outline_ci_image(extent, shell_config):
    """Faint outline at the sdf=0 boundary of the optical shell."""
    import numpy as np
    from Foundation import NSData
    from Quartz import (
        CIImage,
        CGColorSpaceCreateDeviceRGB,
        CGDataProviderCreateWithCFData,
        CGImageCreate,
        kCGImageAlphaPremultipliedLast,
        kCGRenderingIntentDefault,
    )

    width = max(1, int(round(extent.size.width)))
    height = max(1, int(round(extent.size.height)))
    content_w = float(shell_config.get("content_width_points", width))
    content_h = float(shell_config.get("content_height_points", height))
    corner_radius = max(float(shell_config.get("corner_radius_points", 16.0)), 1.0)
    spine_half_x = max(content_w * 0.5 - corner_radius, 0.0)
    spine_half_y = max(content_h * 0.5 - corner_radius, 0.0)

    cx = width * 0.5
    cy = height * 0.5
    xs = np.arange(width, dtype=np.float32)[None, :] + 0.5 - cx
    ys = np.arange(height, dtype=np.float32)[:, None] + 0.5 - cy

    # Rounded-shell SDF matching the warp kernel.
    spine_dist_x = np.maximum(np.abs(xs) - spine_half_x, 0.0)
    spine_dist_y = np.maximum(np.abs(ys) - spine_half_y, 0.0)
    capsule_sdf = np.hypot(spine_dist_x, spine_dist_y) - corner_radius

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    # Only draw boundary on the right half for comparison.
    right_half = xs >= 0.0
    boundary = (np.abs(capsule_sdf) < 1.5) & right_half
    rgba[boundary] = (255, 255, 255, 60)

    payload = NSData.dataWithBytes_length_(rgba.tobytes(), int(rgba.nbytes))
    provider = CGDataProviderCreateWithCFData(payload)
    image = CGImageCreate(
        width, height, 8, 32, width * 4,
        CGColorSpaceCreateDeviceRGB(),
        kCGImageAlphaPremultipliedLast,
        provider, None, False, kCGRenderingIntentDefault,
    )
    ci_image = CIImage.imageWithCGImage_(image)
    return ci_image.imageByCroppingToRect_(extent) if hasattr(ci_image, "imageByCroppingToRect_") else ci_image


class _QuartzBackdropRenderer:
    """Best-effort snapshot renderer for the assistant backdrop prototype."""

    def __init__(self) -> None:
        self._ci_context = None
        self._cached_center_mask_ci = None  # cached CIImage mask
        self._cached_center_mask_key = None  # (mw, mh, content_w, content_h) cache key

    def _context(self):
        if self._ci_context is not None:
            return self._ci_context
        try:
            from Quartz import CIContext
        except Exception:
            return None
        try:
            self._ci_context = CIContext.contextWithOptions_(None)
        except Exception:
            logger.debug("Failed to create CIContext for command backdrop", exc_info=True)
            self._ci_context = None
        return self._ci_context

    def capture_blurred_image(self, *, window_number: int, capture_rect, blur_radius_points: float):
        overscan_points = _command_backdrop_capture_overscan_points()
        content_width_points = max(capture_rect.size.width - 2 * overscan_points, 1.0)
        content_height_points = max(capture_rect.size.height - 2 * overscan_points, 1.0)
        if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED and _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE:
            context = self._context()
            if context is not None and hasattr(context, "createCGImage_fromRect_"):
                shell_config = _command_optical_shell_config(
                    content_width_points,
                    content_height_points,
                )
                if shell_config is not None:
                    extent = NSMakeRect(0.0, 0.0, capture_rect.size.width, capture_rect.size.height)
                    output = _debug_shell_grid_ci_image(extent, shell_config)
                    if output is not None:
                        output = _apply_optical_shell_warp_ci_image(output, extent, shell_config)
                        try:
                            image = context.createCGImage_fromRect_(output, extent)
                        except Exception:
                            logger.debug("Failed to seed debug shell grid image in Quartz backdrop renderer", exc_info=True)
                            image = None
                        if image is not None:
                            return image
        from spoke.backdrop_stream import _quartz_timer

        try:
            from Quartz import (
                CGWindowListCreateImage,
                kCGWindowListOptionOnScreenBelowWindow,
            )
        except Exception:
            return None

        _quartz_timer.begin("total")
        _quartz_timer.begin("capture")
        rect = (
            (capture_rect.origin.x, capture_rect.origin.y),
            (capture_rect.size.width, capture_rect.size.height),
        )
        try:
            image = CGWindowListCreateImage(
                rect,
                kCGWindowListOptionOnScreenBelowWindow,
                window_number,
                0,
            )
        except Exception:
            logger.debug("Backdrop snapshot capture failed", exc_info=True)
            return None
        _quartz_timer.end("capture")
        if image is None:
            return image

        try:
            from Quartz import CIFilter, CIImage
        except Exception:
            return image

        try:
            context = self._context()
            if context is None:
                return image
            _quartz_timer.begin("ci_convert")
            ci_image = CIImage.imageWithCGImage_(image)
            extent = ci_image.extent() if hasattr(ci_image, "extent") else None
            _quartz_timer.end("ci_convert")
            if extent is None:
                return image
            output = ci_image
            blur_radius_points = max(float(blur_radius_points), 0.0)
            if blur_radius_points > 0.0:
                blur = CIFilter.filterWithName_("CIGaussianBlur")
                if blur is not None:
                    blur.setDefaults()
                    blur_input = (
                        output.imageByClampingToExtent()
                        if hasattr(output, "imageByClampingToExtent")
                        else output
                    )
                    blur.setValue_forKey_(blur_input, "inputImage")
                    blur.setValue_forKey_(blur_radius_points, "inputRadius")
                    blurred = blur.valueForKey_("outputImage")
                    if blurred is not None:
                        output = (
                            blurred.imageByCroppingToRect_(extent)
                            if hasattr(blurred, "imageByCroppingToRect_")
                            else blurred
                        )

            # Apply optical shell warp to the real backdrop capture.
            if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
                shell_config = _command_optical_shell_config(
                    content_width_points,
                    content_height_points,
                )
                if shell_config is not None:
                    _quartz_timer.begin("warp")
                    warped = _apply_optical_shell_warp_ci_image(output, extent, shell_config)
                    if warped is not None:
                        output = warped
                        if hasattr(output, "imageByCroppingToRect_"):
                            output = output.imageByCroppingToRect_(extent)
                    _quartz_timer.end("warp")


                    # Faint boundary outline over the warped backdrop.
                    if _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_REVEAL:
                        outline = _boundary_outline_ci_image(extent, shell_config)
                        if outline is not None:
                            try:
                                from Quartz import CIFilter
                                comp = CIFilter.filterWithName_("CISourceOverCompositing")
                                if comp is not None:
                                    comp.setDefaults()
                                    comp.setValue_forKey_(outline, "inputImage")
                                    comp.setValue_forKey_(output, "inputBackgroundImage")
                                    composited = comp.valueForKey_("outputImage")
                                    if composited is not None:
                                        output = composited
                                        if hasattr(output, "imageByCroppingToRect_"):
                                            output = output.imageByCroppingToRect_(extent)
                            except Exception:
                                pass

            if not hasattr(context, "createCGImage_fromRect_"):
                return image
            _quartz_timer.begin("render")
            result = context.createCGImage_fromRect_(output, extent)
            _quartz_timer.end("render")
            _quartz_timer.end("total")
            _quartz_timer.frame_done()
            return result or image
        except Exception:
            logger.debug("Backdrop warp pass failed; using raw snapshot", exc_info=True)
            return image


def _dismiss_animation_state(elapsed_s: float) -> tuple[str, float, float, bool]:
    elapsed = max(elapsed_s, 0.0)
    if elapsed < _DISMISS_GROW_S:
        eased = _ease_in(elapsed / _DISMISS_GROW_S)
        scale = _lerp(1.0, _DISMISS_GROW_SCALE, eased)
        return "grow", scale, 1.0, False

    shrink_elapsed = elapsed - _DISMISS_GROW_S
    eased = _ease_in(shrink_elapsed / _DISMISS_SHRINK_S)
    scale = _lerp(_DISMISS_GROW_SCALE, _DISMISS_END_SCALE, eased)
    alpha = 1.0 - eased
    done = elapsed >= (_DISMISS_DURATION_S - 1e-9)
    return "shrink", scale, alpha, done


class CommandOverlay(NSObject):
    """Overlay for displaying streamed command responses."""

    def initWithScreen_(
        self,
        screen: NSScreen | None = None,
        metrics: OpticalShellMetrics | None = None,
    ):
        self = objc.super(CommandOverlay, self).init()
        if self is None:
            return None

        self._screen = screen or NSScreen.mainScreen()
        self._metrics = metrics
        self._window: NSWindow | None = None
        self._wrapper_view: NSView | None = None
        self._text_view: NSTextView | None = None
        self._scroll_view: NSScrollView | None = None
        self._content_view: NSView | None = None
        self._visible = False
        self._streaming = False
        self._response_text = ""
        self._utterance_text = ""
        self._collapsed_text = ""
        self._utterance_label = None  # pinned header for user text

        # Fade state
        self._fade_timer: NSTimer | None = None
        self._fade_step = 0
        self._fade_from = 0.0
        self._fade_direction = 0
        self._cancel_timer_anim: NSTimer | None = None
        self._cancel_elapsed = 0.0
        self._cancel_phase = ""

        # Pulse state
        self._pulse_timer: NSTimer | None = None
        self._pulse_phase_asst = 0.0

        self._pulse_phase_user = _PULSE_PHASE_OFFSET_USER
        self._color_phase = 0.75  # start at violet, not red
        self._color_velocity_phase = 0.0
        self._tool_mode = False
        # Linger timer
        self._linger_timer: NSTimer | None = None

        # TTS amplitude state — drives window opacity during speech playback
        self._tts_amplitude = 0.0
        self._tts_active = False
        self._tts_blend = 0.0  # 0.0 = pure pulse, 1.0 = pure TTS
        self._brightness_timer: NSTimer | None = None

        # Thinking timer state
        self._thinking_timer: NSTimer | None = None
        self._thinking_seconds = 0.0

        # Cancel spring state — 0.0 = idle, 1.0 = fully wound
        self._cancel_spring = 0.0
        self._cancel_spring_target = 0.0  # 1.0 while winding, 0.0 while unwinding
        self._cancel_spring_fired = False  # True once threshold crossed
        self._on_cancel_spring_threshold: Callable[[], None] | None = None
        self._thinking_label = None  # NSTextField for the counter
        self._thinking_glow_layer = None  # CALayer for the glow behind the number
        self._thinking_inverted = False  # False = glowing number, True = cutout
        self._narrator_label = None  # NSTextField for narrator summary
        self._narrator_shimmer_active = False
        self._agent_shell_header_label = None
        self._agent_shell_footer_label = None
        self._agent_shell_cards = []
        self._agent_shell_primitives = []

        # Adaptive compositing defaults dark until we sample the screen.
        self._brightness = 0.0
        self._brightness_target = 0.0
        self._brightness_seeded_externally = False  # True once set_brightness() is called
        self._suppress_stale_fill_until_ready = False
        self._fill_hidden_until_signature = None
        self._backdrop_base_blur_radius_points = _COMMAND_BACKDROP_BLUR_RADIUS
        self._backdrop_blur_radius_points = _COMMAND_BACKDROP_BLUR_RADIUS
        self._backdrop_base_mask_width_multiplier = _COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER
        self._backdrop_mask_width_multiplier = _COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER
        self._backdrop_blur_drive = 0.0
        self._backdrop_renderer = make_backdrop_renderer(
            self._screen,
            lambda: _QuartzBackdropRenderer(),
        )
        self._backdrop_layer = None
        self._backdrop_capture_overscan_points = _command_backdrop_capture_overscan_points()
        self._backdrop_capture_rect = None
        self._backdrop_capture_pixel_size = None
        self._backdrop_timer: NSTimer | None = None
        self._visual_start_timer: NSTimer | None = None
        self._visual_ready_timer: NSTimer | None = None
        self._visual_ready_wait_started_at = 0.0
        self._visual_ready_brightness_synced = False
        self._materialization_timer: NSTimer | None = None
        self._materialization_started_at = 0.0
        self._materialization_progress = 1.0
        self._materialization_direction = 1
        self._materialization_final_shell_config: dict | None = None
        self._deferred_materialization_shell_config: dict | None = None
        self._pucker_tail_timer: NSTimer | None = None
        self._pucker_tail_started_at = 0.0
        self._pucker_tail_progress_offset = 0.0
        self._pucker_tail_shell_config: dict | None = None
        self._seam_pucker_tuning_overrides: dict[str, float] = {}
        self._seam_pucker_tuning_preview_active = False
        self._seam_pucker_tuning_started_preview = False
        self._seam_pucker_tuning_compositor = None
        self._dismiss_seam_compositor = None
        self._dismiss_radial_pucker_compositor = None
        self._compositor_registry = None
        self._fullscreen_compositor = None
        self._force_backdrop_frame_callback = False

        return self

    def set_compositor_registry(self, registry) -> None:
        self._compositor_registry = registry

    def seam_pucker_tuning_snapshot(self) -> dict[str, float]:
        tuning = _seam_pucker_tuning_defaults()
        tuning.update(getattr(self, "_seam_pucker_tuning_overrides", {}))
        return tuning

    def set_seam_pucker_tuning_value(self, key: str, value: float) -> None:
        self.update_seam_pucker_tuning(**{key: value})

    def update_seam_pucker_tuning(self, **updates: float) -> None:
        defaults = _seam_pucker_tuning_defaults()
        overrides = dict(getattr(self, "_seam_pucker_tuning_overrides", {}))
        for key, value in updates.items():
            if key not in defaults:
                continue
            numeric = float(value)
            if abs(numeric - defaults[key]) <= 1e-6:
                overrides.pop(key, None)
            else:
                overrides[key] = numeric
        self._seam_pucker_tuning_overrides = overrides
        self._reapply_seam_pucker_tuning()

    def reset_seam_pucker_tuning(self) -> None:
        self._seam_pucker_tuning_overrides = {}
        self._reapply_seam_pucker_tuning()

    def preview_seam_pucker_tuning(self) -> None:
        self._seam_pucker_tuning_preview_active = True
        if not self._ensure_seam_pucker_tuning_surface():
            return
        self._reapply_seam_pucker_tuning()

    def release_seam_pucker_tuning_preview(self) -> None:
        self._seam_pucker_tuning_preview_active = False
        self._stop_seam_pucker_tuning_compositor()
        self._seam_pucker_tuning_started_preview = False

    def _ensure_seam_pucker_tuning_surface(self) -> bool:
        """Create/hold a visible static compositor surface for seam tuning."""
        if getattr(self, "_window", None) is None:
            return False
        compositor = getattr(self, "_seam_pucker_tuning_compositor", None)
        if compositor is None:
            compositor = self._start_seam_pucker_tuning_compositor()
            if compositor is None:
                return False
            if getattr(self, "_seam_pucker_tuning_compositor", None) is None:
                self._seam_pucker_tuning_compositor = compositor
            self._seam_pucker_tuning_started_preview = True
        self._cancel_materialization_animation()
        self._cancel_dismiss_pucker_tail_animation()
        self._cancel_pulse()
        self._cancel_linger()
        self._cancel_visual_start()
        self._cancel_visual_ready_start()
        return True

    def _reapply_seam_pucker_tuning(self) -> None:
        if not getattr(self, "_seam_pucker_tuning_preview_active", False):
            return
        if not self._ensure_seam_pucker_tuning_surface():
            return
        compositor = getattr(self, "_seam_pucker_tuning_compositor", None)
        if compositor is None:
            return
        preview_config = self._seam_pucker_tuning_preview_config()
        if preview_config is None:
            return
        try:
            compositor.update_shell_config(preview_config)
        except Exception:
            logger.debug("Failed to apply seam pucker tuning preview", exc_info=True)

    def _seam_pucker_tuning_base_shell_config(self) -> dict | None:
        """Build a stable diagnostic shell centered on the display.

        The tuner must not borrow the live assistant overlay body: doing so
        makes slider motion and summon/dismiss keys race the conversation UI.
        """
        shell_config = _command_optical_shell_config()
        if shell_config is None:
            return None
        screen_frame = self._screen.frame()
        scale = (
            self._screen.backingScaleFactor()
            if hasattr(self._screen, "backingScaleFactor")
            else 2.0
        )
        screen_width = getattr(getattr(screen_frame, "size", None), "width", 0.0)
        screen_height = getattr(getattr(screen_frame, "size", None), "height", 0.0)
        if not isinstance(screen_width, numbers.Real):
            screen_width = 0.0
        if not isinstance(screen_height, numbers.Real):
            screen_height = 0.0
        shell_config["client_id"] = _SEAM_PUCKER_TUNING_CLIENT_ID
        shell_config["role"] = "diagnostic"
        shell_config["center_x"] = screen_width * scale * 0.5
        shell_config["center_y"] = screen_height * scale * 0.5
        shell_config["initial_brightness"] = _clamp01(float(getattr(self, "_brightness", 0.0)))
        shell_config["visible"] = True
        for key in (
            "content_width_points",
            "content_height_points",
            "corner_radius_points",
            "band_width_points",
            "tail_width_points",
        ):
            if key in shell_config:
                shell_config[key] = float(shell_config[key]) * scale
        return shell_config

    def _seam_pucker_tuning_preview_config(self) -> dict | None:
        base_config = self._seam_pucker_tuning_base_shell_config()
        if base_config is None:
            return None
        tuning = self.seam_pucker_tuning_snapshot()
        progress = _clamp01(tuning.get("preview_progress", 0.2))
        preview_config = _apply_dismiss_seam_latch_fields(
            base_config,
            progress,
            tuning,
        )
        preview_config["client_id"] = _SEAM_PUCKER_TUNING_CLIENT_ID
        preview_config["role"] = "diagnostic"
        preview_config["visible"] = True
        return preview_config

    def _start_seam_pucker_tuning_compositor(self):
        config = self._seam_pucker_tuning_preview_config()
        if config is None:
            return None
        try:
            from spoke.fullscreen_compositor import start_overlay_compositor

            compositor = start_overlay_compositor(
                screen=self._screen,
                window=self._window,
                content_view=self._content_view,
                shell_config=config,
                client_id=_SEAM_PUCKER_TUNING_CLIENT_ID,
                role="diagnostic",
                registry=getattr(self, "_compositor_registry", None),
            )
        except Exception:
            logger.debug("Failed to start seam pucker tuning compositor", exc_info=True)
            return None
        if compositor is not None:
            self._seam_pucker_tuning_compositor = compositor
            self._seam_pucker_tuning_started_preview = True
        return compositor

    def _stop_seam_pucker_tuning_compositor(self) -> None:
        compositor = getattr(self, "_seam_pucker_tuning_compositor", None)
        self._seam_pucker_tuning_compositor = None
        if compositor is not None:
            try:
                compositor.stop()
            except Exception:
                logger.debug("Failed to stop seam pucker tuning compositor", exc_info=True)

    def _ensure_dismiss_seam_compositor(self):
        """Register the transient seam layer without disturbing the main shell."""
        compositor = getattr(self, "_dismiss_seam_compositor", None)
        if compositor is not None:
            return compositor
        final_config = getattr(self, "_materialization_final_shell_config", None)
        if final_config is None:
            return None
        progress = getattr(self, "_materialization_progress", 0.0)
        config = _dismiss_seam_latch_shell_config(
            final_config,
            progress,
        )
        try:
            from spoke.fullscreen_compositor import start_overlay_compositor

            compositor = start_overlay_compositor(
                screen=self._screen,
                window=self._window,
                content_view=self._content_view,
                shell_config=config,
                client_id=_DISMISS_SEAM_CLIENT_ID,
                role="assistant",
                registry=getattr(self, "_compositor_registry", None),
            )
        except Exception:
            logger.debug("Failed to start command dismiss seam compositor", exc_info=True)
            return None
        self._dismiss_seam_compositor = compositor
        return compositor

    def _update_dismiss_seam_compositor(self, final_config: dict, progress: float) -> None:
        compositor = self._ensure_dismiss_seam_compositor()
        if compositor is None:
            return
        try:
            compositor.update_shell_config(
                _dismiss_seam_latch_shell_config(
                    final_config,
                    progress,
                )
            )
        except Exception:
            logger.debug("Failed to update command dismiss seam compositor", exc_info=True)

    def _stop_dismiss_seam_compositor(self) -> None:
        compositor = getattr(self, "_dismiss_seam_compositor", None)
        self._dismiss_seam_compositor = None
        if compositor is not None:
            try:
                compositor.stop()
            except Exception:
                logger.debug("Failed to stop command dismiss seam compositor", exc_info=True)

    def _ensure_dismiss_radial_pucker_compositor(self):
        """Register the radial release as a sidecar so the main shell can keep closing."""
        compositor = getattr(self, "_dismiss_radial_pucker_compositor", None)
        if compositor is not None:
            return compositor
        final_config = getattr(self, "_materialization_final_shell_config", None)
        if final_config is None:
            return None
        progress = getattr(self, "_pucker_tail_progress_offset", 0.0)
        config = _dismiss_radial_pucker_shell_config(final_config, progress)
        try:
            from spoke.fullscreen_compositor import start_overlay_compositor

            compositor = start_overlay_compositor(
                screen=self._screen,
                window=self._window,
                content_view=self._content_view,
                shell_config=config,
                client_id=_DISMISS_RADIAL_PUCKER_CLIENT_ID,
                role="assistant",
                registry=getattr(self, "_compositor_registry", None),
            )
        except Exception:
            logger.debug("Failed to start command radial pucker compositor", exc_info=True)
            return None
        self._dismiss_radial_pucker_compositor = compositor
        return compositor

    def _update_dismiss_radial_pucker_compositor(
        self, final_config: dict, progress: float
    ) -> None:
        self._pucker_tail_progress_offset = _clamp01(progress)
        compositor = self._ensure_dismiss_radial_pucker_compositor()
        if compositor is None:
            return
        try:
            compositor.update_shell_config(
                _dismiss_radial_pucker_shell_config(final_config, progress)
            )
        except Exception:
            logger.debug("Failed to update command radial pucker compositor", exc_info=True)

    def _publish_shared_compositor_configs(self, updates: list[tuple[object, dict]]) -> bool:
        """Publish one coherent multi-shell frame when sidecars share a host."""
        if not updates:
            return True
        host = getattr(updates[0][0], "_host", None)
        updater = getattr(host, "update_client_configs", None)
        if host is None or not callable(updater):
            return False
        configs: dict[str, dict] = {}
        for client, config in updates:
            if client is None or getattr(client, "_host", None) is not host:
                return False
            client_id = getattr(client, "_client_id", None)
            if not client_id:
                return False
            configs[str(client_id)] = config
        try:
            return bool(updater(configs))
        except Exception:
            logger.debug("Failed to batch command compositor update", exc_info=True)
            return False

    def _publish_individual_compositor_configs(
        self,
        updates: list[tuple[object, dict]],
    ) -> None:
        for client, config in updates:
            if client is None:
                continue
            try:
                client.update_shell_config(config)
            except Exception:
                logger.debug("Failed to update command compositor client", exc_info=True)

    def _stop_dismiss_radial_pucker_compositor(self) -> None:
        compositor = getattr(self, "_dismiss_radial_pucker_compositor", None)
        self._dismiss_radial_pucker_compositor = None
        if compositor is not None:
            try:
                compositor.stop()
            except Exception:
                logger.debug("Failed to stop command radial pucker compositor", exc_info=True)

    def setup(self) -> None:
        """Create the command overlay window."""
        screen_frame = self._screen.frame()
        sw = screen_frame.size.width

        f = _OPTICAL_SHELL_FEATHER if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED else _OUTER_FEATHER
        x = (sw - _OVERLAY_WIDTH) / 2 - f
        y = _OVERLAY_BOTTOM_MARGIN - f
        win_w = _OVERLAY_WIDTH + 2 * f
        win_h = _OVERLAY_HEIGHT + 2 * f
        frame = NSMakeRect(x, y, win_w, win_h)

        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, 0, NSBackingStoreBuffered, False
        )
        self._window.setLevel_(_COMMAND_OVERLAY_WINDOW_LEVEL)
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(NSColor.clearColor())
        self._window.setIgnoresMouseEvents_(True)
        self._window.setHasShadow_(False)
        self._window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        wrapper_frame = NSMakeRect(0, 0, win_w, win_h)
        wrapper = NSView.alloc().initWithFrame_(wrapper_frame)
        wrapper.setWantsLayer_(True)
        wrapper.layer().setAnchorPoint_((0.5, 0.5))
        wrapper.layer().setPosition_((win_w / 2, win_h / 2))
        self._wrapper_view = wrapper

        content_frame = NSMakeRect(f, f, _OVERLAY_WIDTH, _OVERLAY_HEIGHT)
        content = NSView.alloc().initWithFrame_(content_frame)
        content.setWantsLayer_(True)
        content.layer().setMasksToBounds_(False)
        content.layer().setBackgroundColor_(None)

        # Distance-field ridge + fill — same system as the preview overlay
        from .overlay import (
            _GLOW_COLOR as _OVERLAY_GLOW_COLOR,
            _overlay_layer_colors,
        )
        w, h = _OVERLAY_WIDTH, _OVERLAY_HEIGHT
        _, middle_rgb, _ = _overlay_layer_colors(_OVERLAY_GLOW_COLOR)

        self._ridge_scale = self._screen.backingScaleFactor() if hasattr(self._screen, 'backingScaleFactor') else 2.0

        backdrop_layer_cls = self._choose_backdrop_layer_class()
        self._backdrop_layer = backdrop_layer_cls.alloc().init()
        self._backdrop_layer_is_sample_buffer_display = (
            backdrop_layer_cls is not CALayer and hasattr(self._backdrop_layer, "enqueueSampleBuffer_")
        )
        # Configure CAMetalLayer if we got one
        self._backdrop_is_metal_layer = hasattr(self._backdrop_layer, "setDevice_")
        if self._backdrop_is_metal_layer:
            try:
                from spoke.metal_warp import get_metal_warp_pipeline
                pipeline = get_metal_warp_pipeline()
                if pipeline is not None:
                    self._backdrop_layer.setDevice_(pipeline.device)
                    self._backdrop_layer.setPixelFormat_(80)  # BGRA8Unorm
                    self._backdrop_layer.setFramebufferOnly_(False)
                    self._backdrop_layer.setOpaque_(False)
                    scale = self._ridge_scale if hasattr(self, "_ridge_scale") else 2.0
                    self._backdrop_layer.setContentsScale_(scale)
            except Exception:
                logger.debug("Failed to configure CAMetalLayer", exc_info=True)
        if hasattr(self._backdrop_layer, "setContentsGravity_"):
            self._backdrop_layer.setContentsGravity_("resize")
        elif hasattr(self._backdrop_layer, "setVideoGravity_"):
            self._backdrop_layer.setVideoGravity_("resize")
        self._backdrop_layer.setOpacity_(1.0)

        # Fill layer — colored SDF image with baked alpha, same as preview overlay
        self._fill_layer = CALayer.alloc().init()
        self._fill_layer.setFrame_(((0, 0), (win_w, win_h)))
        self._fill_layer.setContentsGravity_("resize")

        # Boost layer — white, sits below the fill.  When punch-through
        # is active on dark backgrounds, this layer brightens the warped
        # content visible through the text-shaped holes in the fill mask.
        # Uses the inverse of the punch-through mask (opaque where text is).
        self._boost_layer = CALayer.alloc().init()
        self._boost_layer.setFrame_(((0, 0), (win_w, win_h)))
        try:
            from Quartz import CGColorCreateSRGB
            self._boost_layer.setBackgroundColor_(CGColorCreateSRGB(1.0, 1.0, 1.0, 1.0))
        except ImportError:
            pass
        self._boost_layer.setOpacity_(0.0)
        self._boost_layer.setHidden_(True)

        self._apply_ridge_masks(w, h)
        wrapper.layer().insertSublayer_below_(self._backdrop_layer, self._fill_layer)
        wrapper.layer().insertSublayer_below_(self._boost_layer, self._fill_layer)
        wrapper.layer().insertSublayer_below_(self._fill_layer, content.layer())
        self._install_backdrop_frame_callback()
        self._install_backdrop_sample_buffer_callback()
        # Pass CAMetalLayer to SCK renderer and start display-link render loop
        self._metal_display_link_renderer = None
        if getattr(self, "_backdrop_is_metal_layer", False):
            renderer = getattr(self, "_backdrop_renderer", None)
            if renderer is not None and hasattr(renderer, "set_metal_backdrop_layer"):
                renderer.set_metal_backdrop_layer(self._backdrop_layer)
            try:
                from spoke.metal_warp import MetalDisplayLinkRenderer, get_metal_warp_pipeline
                pipeline = get_metal_warp_pipeline()
                if pipeline is not None:
                    dl_renderer = MetalDisplayLinkRenderer(pipeline, self._backdrop_layer)
                    if dl_renderer.start():
                        self._metal_display_link_renderer = dl_renderer
                        self._force_backdrop_frame_callback = False
                        if renderer is not None and hasattr(renderer, "set_display_link_renderer"):
                            renderer.set_display_link_renderer(dl_renderer)
                        self._install_backdrop_frame_callback()
                        logger.info("Command overlay: display-link Metal renderer active")
                    else:
                        self._force_backdrop_frame_callback = True
                        self._install_backdrop_frame_callback()
                        logger.info("Command overlay: display-link start failed, falling back to CIWarpKernel")
            except Exception:
                logger.debug("Command overlay: MetalDisplayLinkRenderer unavailable", exc_info=True)

        # Cancel spring tint layer — sits above fill, masked to the same SDF shape
        self._spring_tint_layer = CALayer.alloc().init()
        self._spring_tint_layer.setFrame_(((0, 0), (win_w, win_h)))
        self._spring_tint_layer.setOpacity_(0.0)
        wrapper.layer().insertSublayer_above_(self._spring_tint_layer, self._fill_layer)

        wrapper.addSubview_(content)
        self._content_view = content

        # Scroll view with text view for response text
        scroll_frame = NSMakeRect(24, 20, _OVERLAY_WIDTH - 48, _OVERLAY_HEIGHT - 44)
        self._scroll_view = NSScrollView.alloc().initWithFrame_(scroll_frame)
        self._scroll_view.setHasVerticalScroller_(False)
        self._scroll_view.setHasHorizontalScroller_(False)
        self._scroll_view.setDrawsBackground_(False)
        self._scroll_view.setBorderType_(0)
        self._scroll_view.setAutoresizingMask_(18)
        # Kill clip view background — same fix as preview overlay
        clip_view = self._scroll_view.contentView()
        if clip_view and hasattr(clip_view, 'setDrawsBackground_'):
            clip_view.setDrawsBackground_(False)
        if clip_view and hasattr(clip_view, 'setWantsLayer_'):
            clip_view.setWantsLayer_(True)
            clip_view.layer().setBackgroundColor_(None)

        text_frame = NSMakeRect(0, 0, _OVERLAY_WIDTH - 48, _OVERLAY_HEIGHT - 32)
        self._text_view = NSTextView.alloc().initWithFrame_(text_frame)
        self._text_view.setEditable_(False)
        self._text_view.setSelectable_(False)
        self._text_view.setDrawsBackground_(False)
        self._text_view.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, _TEXT_ALPHA_MIN)
        )
        self._text_view.setFont_(NSFont.systemFontOfSize_weight_(_FONT_SIZE, 0.0))
        self._text_view.setString_("")
        self._text_view.textContainer().setWidthTracksTextView_(True)
        self._text_view.setHorizontallyResizable_(False)
        self._text_view.setVerticallyResizable_(True)

        self._scroll_view.setDocumentView_(self._text_view)
        content.addSubview_(self._scroll_view)

        # Thinking timer label — top-right corner of the content area
        from AppKit import NSTextField, NSTextAlignmentRight
        timer_w, timer_h = 60.0, 24.0
        timer_x = _OVERLAY_WIDTH - timer_w - 12
        timer_y = _OVERLAY_HEIGHT - timer_h - 6
        self._thinking_label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(timer_x, timer_y, timer_w, timer_h)
        )
        self._thinking_label.setEditable_(False)
        self._thinking_label.setSelectable_(False)
        self._thinking_label.setBezeled_(False)
        self._thinking_label.setDrawsBackground_(False)
        self._thinking_label.setAlignment_(NSTextAlignmentRight)
        self._thinking_label.setFont_(
            NSFont.monospacedDigitSystemFontOfSize_weight_(13.0, 0.2)
        )
        self._thinking_label.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(
                _GLOW_COLOR[0], _GLOW_COLOR[1], _GLOW_COLOR[2], 0.0
            )
        )
        self._thinking_label.setStringValue_("")
        self._thinking_label.setHidden_(True)
        content.addSubview_(self._thinking_label)

        from AppKit import NSTextAlignmentLeft

        header_w = _OVERLAY_WIDTH - _AGENT_SHELL_CHROME_X - timer_w - 28.0
        self._agent_shell_header_label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(
                _AGENT_SHELL_CHROME_X,
                _OVERLAY_HEIGHT
                - _AGENT_SHELL_HEADER_TOP_INSET
                - _AGENT_SHELL_HEADER_HEIGHT,
                header_w,
                _AGENT_SHELL_HEADER_HEIGHT,
            )
        )
        self._agent_shell_header_label.setEditable_(False)
        self._agent_shell_header_label.setSelectable_(False)
        self._agent_shell_header_label.setBezeled_(False)
        self._agent_shell_header_label.setDrawsBackground_(False)
        self._agent_shell_header_label.setAlignment_(NSTextAlignmentLeft)
        self._agent_shell_header_label.setLineBreakMode_(4)
        self._agent_shell_header_label.setMaximumNumberOfLines_(1)
        self._agent_shell_header_label.setFont_(
            NSFont.systemFontOfSize_weight_(11.0, 0.0)
        )
        chrome_r, chrome_g, chrome_b = _assistant_foreground_color_for_brightness(
            self._brightness
        )
        self._agent_shell_header_label.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(
                chrome_r,
                chrome_g,
                chrome_b,
                _AGENT_SHELL_CHROME_ALPHA,
            )
        )
        self._agent_shell_header_label.setStringValue_("")
        self._agent_shell_header_label.setHidden_(True)
        content.addSubview_(self._agent_shell_header_label)

        self._agent_shell_footer_label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(
                _AGENT_SHELL_CHROME_X,
                _AGENT_SHELL_FOOTER_BOTTOM_INSET,
                _OVERLAY_WIDTH - 2.0 * _AGENT_SHELL_CHROME_X,
                _AGENT_SHELL_FOOTER_HEIGHT,
            )
        )
        self._agent_shell_footer_label.setEditable_(False)
        self._agent_shell_footer_label.setSelectable_(False)
        self._agent_shell_footer_label.setBezeled_(False)
        self._agent_shell_footer_label.setDrawsBackground_(False)
        self._agent_shell_footer_label.setAlignment_(NSTextAlignmentLeft)
        self._agent_shell_footer_label.setLineBreakMode_(4)
        self._agent_shell_footer_label.setMaximumNumberOfLines_(1)
        self._agent_shell_footer_label.setFont_(
            NSFont.systemFontOfSize_weight_(10.0, 0.0)
        )
        self._agent_shell_footer_label.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(
                chrome_r,
                chrome_g,
                chrome_b,
                _AGENT_SHELL_CHROME_ALPHA,
            )
        )
        self._agent_shell_footer_label.setStringValue_("")
        self._agent_shell_footer_label.setHidden_(True)
        content.addSubview_(self._agent_shell_footer_label)

        # Narrator summary label — below the thinking timer, left-aligned
        # Up to 3 lines of wrapping text; only the latest summary is shown.
        import AppKit as _AppKit
        _NARRATOR_FONT_SIZE = 12.0
        _NARRATOR_LINE_HEIGHT = 15.0
        _NARRATOR_MAX_LINES = 3
        narrator_h = _NARRATOR_LINE_HEIGHT * _NARRATOR_MAX_LINES
        narrator_x = 14.0
        # Keep the narrator inside the base 80px surface while still anchoring it
        # beneath the timer on the right.
        narrator_gap = 4.0
        narrator_y = max(1.0, timer_y - narrator_h - narrator_gap)
        narrator_w = _OVERLAY_WIDTH - 28
        self._narrator_label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(narrator_x, narrator_y, narrator_w, narrator_h)
        )
        self._narrator_label.setEditable_(False)
        self._narrator_label.setSelectable_(False)
        self._narrator_label.setBezeled_(False)
        self._narrator_label.setDrawsBackground_(False)
        self._narrator_label.setAlignment_(NSTextAlignmentLeft)
        self._narrator_label.setLineBreakMode_(
            getattr(
                _AppKit,
                "NSLineBreakByWordWrapping",
                getattr(_AppKit, "NSLineBreakByTruncatingTail", 0),
            )
        )
        self._narrator_label.setMaximumNumberOfLines_(_NARRATOR_MAX_LINES)
        self._narrator_label.setFont_(
            NSFont.systemFontOfSize_weight_(_NARRATOR_FONT_SIZE, 0.0)
        )
        # Initial color set; will be updated by _apply_narrator_theme()
        user_r, user_g, user_b = _user_text_color_for_brightness(self._brightness)
        self._narrator_label.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(user_r, user_g, user_b, 0.90)
        )
        self._narrator_label.setStringValue_("")
        self._narrator_label.setHidden_(True)
        content.addSubview_(self._narrator_label)

        self._window.setContentView_(wrapper)
        self._window.setAlphaValue_(0.0)
        self._apply_surface_theme()
        self._set_overlay_scale(1.0)
        self._update_backdrop_capture_geometry()
        logger.info("Command overlay created")

    def _choose_backdrop_layer_class(self):
        if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
            try:
                from spoke.metal_warp import get_metal_warp_pipeline
                pipeline = get_metal_warp_pipeline()
                if pipeline is not None:
                    import objc
                    CAMetalLayer = objc.lookUpClass("CAMetalLayer")
                    logger.info("Command overlay: using CAMetalLayer for optical shell")
                    return CAMetalLayer
            except Exception:
                logger.debug("CAMetalLayer unavailable", exc_info=True)
            return CALayer
        renderer = getattr(self, "_backdrop_renderer", None)
        blur_radius_points = getattr(self, "_backdrop_blur_radius_points", _COMMAND_BACKDROP_BLUR_RADIUS)
        if renderer is not None and hasattr(renderer, "supports_sample_buffer_presentation"):
            try:
                if renderer.supports_sample_buffer_presentation(blur_radius_points):
                    display_layer_class = _backdrop_display_layer_class()
                    if display_layer_class is not None:
                        return display_layer_class
            except Exception:
                logger.debug("Command backdrop renderer sample-buffer capability check failed", exc_info=True)
        return CALayer

    def _backdrop_layer_uses_sample_buffers(self) -> bool:
        return bool(getattr(self, "_backdrop_layer_is_sample_buffer_display", False))

    def _reset_backdrop_layer(self) -> None:
        layer = getattr(self, "_backdrop_layer", None)
        if layer is None:
            return
        if self._backdrop_layer_uses_sample_buffers() and hasattr(layer, "flushAndRemoveImage"):
            try:
                layer.flushAndRemoveImage()
            except Exception:
                logger.debug("Failed to flush command backdrop display layer", exc_info=True)
        if hasattr(layer, "setContents_"):
            layer.setContents_(None)
        if hasattr(layer, "setMask_"):
            layer.setMask_(None)

    def _current_optical_shell_config(self) -> dict[str, float | bool] | None:
        content = getattr(self, "_content_view", None)
        if content is None or not hasattr(content, "frame"):
            return self._attach_agent_shell_surface_payload(
                _command_optical_shell_config()
            )
        try:
            frame = content.frame()
            width = frame.size.width
            height = frame.size.height
        except Exception:
            return self._attach_agent_shell_surface_payload(
                _command_optical_shell_config()
            )
        if not isinstance(width, numbers.Real) or not isinstance(height, numbers.Real):
            return self._attach_agent_shell_surface_payload(
                _command_optical_shell_config()
            )
        return self._attach_agent_shell_surface_payload(
            _command_optical_shell_config(width, height)
        )

    def _attach_agent_shell_surface_payload(self, config):
        if config is None:
            return None
        cards = getattr(self, "_agent_shell_cards", None)
        primitives = getattr(self, "_agent_shell_primitives", None)
        if isinstance(primitives, list) and primitives:
            config = dict(config)
            config["surface_kind"] = "agent_shell"
            config["agent_shell_primitives"] = [
                dict(primitive) for primitive in primitives if isinstance(primitive, dict)
            ]
            config["agent_shell_card_renderer"] = build_agent_shell_card_render_payload(
                config["agent_shell_primitives"],
                content_width_points=float(config.get("content_width_points", 0.0)),
                content_height_points=float(config.get("content_height_points", 0.0)),
            )
            config["agent_shell_card_optical_fields"] = (
                build_agent_shell_card_optical_field_payload(
                    config["agent_shell_card_renderer"]
                )
            )
        if isinstance(cards, list) and cards:
            config = dict(config)
            config["surface_kind"] = "agent_shell"
            config["agent_thread_cards"] = [
                dict(card) for card in cards if isinstance(card, dict)
            ]
            config["agent_thread_hud"] = build_agent_thread_hud(
                config["agent_thread_cards"],
                content_width_points=float(config.get("content_width_points", 0.0)),
                content_height_points=float(config.get("content_height_points", 0.0)),
            )
        return config

    def _apply_backdrop_pulse_style(self, breath: float) -> None:
        layer = getattr(self, "_backdrop_layer", None)
        if layer is None:
            return
        blur_target = _command_backdrop_blur_target_for_presence(breath)
        self._backdrop_blur_drive = _advance_attack_release(
            getattr(self, "_backdrop_blur_drive", blur_target),
            blur_target,
            attack=_COMMAND_BACKDROP_PULSE_ATTACK,
            release=_COMMAND_BACKDROP_PULSE_RELEASE,
        )
        base_blur_radius_points = getattr(
            self, "_backdrop_base_blur_radius_points", _COMMAND_BACKDROP_BLUR_RADIUS
        )
        base_mask_width_multiplier = getattr(
            self,
            "_backdrop_base_mask_width_multiplier",
            _COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER,
        )
        (
            blur_radius_points,
            mask_width_multiplier,
            backdrop_opacity,
        ) = _command_backdrop_pulse_style(
            base_blur_radius_points,
            base_mask_width_multiplier,
            self._backdrop_blur_drive,
        )
        if _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE:
            mask_width_multiplier = min(
                mask_width_multiplier,
                _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_MASK_WIDTH_MULTIPLIER,
            )
        renderer = getattr(self, "_backdrop_renderer", None)
        shell_config = self._current_optical_shell_config()
        effective_blur_radius_points = blur_radius_points
        if shell_config is not None:
            effective_blur_radius_points = float(
                shell_config["cleanup_blur_radius_points"]
            )
        self._backdrop_blur_radius_points = effective_blur_radius_points
        if renderer is not None and hasattr(renderer, "set_live_blur_radius_points"):
            try:
                renderer.set_live_blur_radius_points(effective_blur_radius_points)
            except Exception:
                logger.debug("Failed to push live command backdrop blur radius", exc_info=True)
        if renderer is not None and hasattr(renderer, "set_live_optical_shell_config"):
            try:
                renderer.set_live_optical_shell_config(shell_config)
            except Exception:
                logger.debug("Failed to push command optical-shell config", exc_info=True)
        last_mask_width_multiplier = getattr(
            self,
            "_backdrop_mask_width_multiplier",
            base_mask_width_multiplier,
        )
        self._backdrop_mask_width_multiplier = mask_width_multiplier
        if abs(mask_width_multiplier - last_mask_width_multiplier) > 1e-6:
            capture_rect = getattr(self, "_backdrop_capture_rect", None)
            if capture_rect is not None:
                try:
                    mask_width = float(capture_rect.size.width)
                    mask_height = float(capture_rect.size.height)
                except (TypeError, ValueError):
                    mask_width = None
                    mask_height = None
                if mask_width is not None and mask_height is not None:
                    self._update_backdrop_mask(mask_width, mask_height)
        layer.setOpacity_(backdrop_opacity)

    # ── public interface ────────────────────────────────────

    def show(
        self,
        *,
        preserve_thinking_timer: bool = False,
        start_thinking_timer: bool = True,
        initial_utterance: str = "",
        initial_response: str = "",
        agent_shell_header: str = "",
        agent_shell_footer: str = "",
        agent_shell_cards: list[dict] | None = None,
        agent_shell_primitives: list[dict] | None = None,
    ) -> None:
        """Fade the overlay in, optionally starting or resuming the thinking timer."""
        if self._window is None:
            return
        self._cancel_all_timers()
        if getattr(self, "_fullscreen_compositor", None) is not None:
            self._stop_fullscreen_compositor()
        self._visible = True
        self._streaming = True
        has_initial_transcript = bool(initial_utterance or initial_response)
        known_content_optical_start = (
            has_initial_transcript and _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED
        )
        self._response_text = ""
        self._utterance_text = ""
        self._collapsed_text = ""
        self._clear_agent_shell_chrome()
        if agent_shell_primitives:
            self.set_agent_shell_primitives(agent_shell_primitives)
        else:
            self.set_agent_shell_cards(agent_shell_cards or [])
        # Reset TTS state so stale blend doesn't affect new responses
        self._tts_active = False
        self._tts_blend = 0.0
        self._tts_amplitude = 0.0
        self._text_view.setString_("")
        if hasattr(self._text_view, "textStorage"):
            try:
                from AppKit import NSMutableAttributedString

                self._text_view.textStorage().setAttributedString_(
                    NSMutableAttributedString.alloc().initWithString_("")
                )
            except Exception:
                pass
        self._window.setAlphaValue_(0.0)
        self._set_overlay_scale(1.0)
        self._pulse_phase_asst = 0.0
        self._pulse_phase_user = _PULSE_PHASE_OFFSET_USER
        self._color_phase = 0.33  # start away from the old purple flash
        self._color_velocity_phase = 0.0
        self._response_chroma_active = False
        self._tool_mode = False
        self._pulse_timer = None
        self._seed_brightness_from_screen()
        self._suppress_stale_fill_until_ready = True

        # Reset geometry
        try:
            screen_frame = self._screen.frame()
            sw = screen_frame.size.width
            f = _OPTICAL_SHELL_FEATHER if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED else _OUTER_FEATHER
            x = (sw - _OVERLAY_WIDTH) / 2 - f
            self._window.setFrame_display_animate_(
                NSMakeRect(x, _OVERLAY_BOTTOM_MARGIN - f,
                           _OVERLAY_WIDTH + 2 * f, _OVERLAY_HEIGHT + 2 * f),
                True, False
            )
            self._content_view.setFrame_(
                NSMakeRect(f, f, _OVERLAY_WIDTH, _OVERLAY_HEIGHT)
            )
            self._scroll_view.setFrame_(
                NSMakeRect(48, 20, _OVERLAY_WIDTH - 96, _OVERLAY_HEIGHT - 44)
            )
            scroll_frame = self._scroll_view.frame()
            self._reset_text_geometry(
                scroll_frame.size.height,
                scroll_frame.size.width,
            )
            if not has_initial_transcript:
                self._apply_ridge_masks(_OVERLAY_WIDTH, _OVERLAY_HEIGHT)
                self._fill_image_brightness = self._brightness
            else:
                # Do not let an old fill image masquerade as current during entrance.
                # show() may reuse the same geometry across very different background
                # brightness values; force the first theme pass to publish the current
                # material before the window fades in.
                self._fill_image_brightness = -1.0
            self._apply_surface_theme()
            self._update_backdrop_capture_geometry()
            self._apply_backdrop_pulse_style(1.0)
            self._reset_backdrop_layer()
            if (
                known_content_optical_start
                and self._scroll_view is not None
            ):
                # Initial text should not expose the ordinary attributed-text
                # layer before the compositor switches to punch-through text.
                # The fallback path unhides it again if the compositor is
                # unavailable.
                self._scroll_view.setHidden_(True)

            if initial_response:
                self._utterance_text = initial_utterance
                self.set_response_text(initial_response)
            elif initial_utterance:
                self.set_utterance(initial_utterance)
            if agent_shell_header:
                self.set_agent_shell_header(agent_shell_header)
            if agent_shell_footer:
                self.set_agent_shell_footer(agent_shell_footer)
        finally:
            self._suppress_stale_fill_until_ready = False

        self._window.orderFrontRegardless()
        if known_content_optical_start:
            # Initial text is already laid out. Arm the optical compositor
            # while the command window is still alpha-zero so the user sees one
            # composed entrance, not plain text -> warp -> punch-through as
            # separate phases.
            self._start_fullscreen_compositor()
            self._refresh_punchthrough_mask_if_needed()
            if getattr(self, "_fullscreen_compositor", None) is None:
                self._enable_text_punchthrough(False)
                self._start_backdrop_refresh_timer()
        if not _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
            self._refresh_backdrop_snapshot()
        self._start_brightness_sampling()

        # Only live generation owns the thinking timer; history/approval recall does not.
        if start_thinking_timer:
            self._start_thinking_timer(reset=not preserve_thinking_timer)

        if (
            known_content_optical_start
            and getattr(self, "_fullscreen_compositor", None) is not None
        ):
            self._schedule_visual_ready_start()
        else:
            self._start_entrance_animation()

        if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED and not known_content_optical_start:
            self._schedule_visual_start()
        elif (
            not known_content_optical_start
            and getattr(self, "_fullscreen_compositor", None) is None
        ):
            self._start_backdrop_refresh_timer()

    def set_brightness(self, brightness: float, immediate: bool = False) -> None:
        """Set screen brightness (0.0 dark – 1.0 bright) for adaptive compositing."""
        self._brightness_target = _clamp01(brightness)
        self._brightness_seeded_externally = True
        if immediate:
            self._brightness = self._brightness_target
            self._apply_surface_theme()

    def _seed_brightness_from_screen(self) -> None:
        """Seed entrance brightness, preferring GlowOverlay's cached value.

        GlowOverlay samples screen brightness at 1 Hz and pushes it here via
        set_brightness() before every show() call in __main__.py.  When that
        has happened, _brightness_seeded_externally is True and we commit the
        already-held _brightness_target directly — no Quartz capture needed.

        On first-ever show (before any glow sample), fall back to neutral 0.5.
        The next amplitude tick will push the real value via set_brightness().
        """
        if getattr(self, "_brightness_seeded_externally", False):
            # Brightness already set by set_brightness(); commit it in-place.
            self._brightness = self._brightness_target
            return
        # First-ever show: no cached glow brightness yet.  Use neutral default;
        # the next amplitude tick will push the real value via set_brightness().
        self._brightness_target = 0.5
        self._brightness = 0.5

    def _start_brightness_sampling(self) -> None:
        # Brightness is sampled by GlowOverlay (1 Hz) and pushed here via
        # set_brightness() on every amplitude tick from __main__.py
        # (_sync_command_overlay_brightness).  Running a second independent
        # Quartz screen-capture timer here doubles the capture cost with no
        # benefit.  No-op; cancel any leftover timer from prior sessions.
        old_timer = getattr(self, "_brightness_timer", None)
        if old_timer is not None:
            old_timer.invalidate()
            self._brightness_timer = None

    def brightnessResample_(self, timer) -> None:
        if not self._visible:
            return
        if getattr(self, "_fullscreen_compositor", None) is not None:
            return
        try:
            sample_start = time.perf_counter()
            self.set_brightness(_sample_screen_brightness_for_overlay(self._screen))
            metrics = getattr(self, "_metrics", None)
            if metrics is not None:
                sample_end = time.perf_counter()
                metrics.record_brightness_sample(
                    elapsed_ms=(sample_end - sample_start) * 1000.0,
                    now=sample_end,
                )
        except Exception:
            logger.debug("Command overlay brightness sampling failed", exc_info=True)

    def _entrancePopStep_(self, timer) -> None:
        """Ease the entrance pop back to scale 1.0."""
        self._pop_step += 1
        t = self._pop_step / self._pop_steps
        eased = t * t * (3.0 - 2.0 * t)  # smoothstep
        scale = _ENTRANCE_POP_SCALE * (1.0 - eased) + 1.0 * eased
        self._set_overlay_scale(scale)
        if self._pop_step >= self._pop_steps:
            timer.invalidate()
            self._pop_timer = None
            self._set_overlay_scale(1.0)

    def cancel_dismiss(self) -> None:
        """Dismiss with a fast pop-then-shrink animation."""
        if self._window is None:
            return
        self._cancel_all_timers()
        self._streaming = False
        if getattr(self, "_fullscreen_compositor", None) is not None:
            self._visible = False
            self._window.setAlphaValue_(1.0)
            self._set_overlay_scale(1.0)
            self._start_fade_out()
            return
        self._visible = True  # keep visible for the animation

        self._cancel_elapsed = 0.0
        self._cancel_phase = "grow"
        self._window.setAlphaValue_(1.0)
        self._set_overlay_scale(1.0)
        self._cancel_timer_anim = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / _DISMISS_ANIM_FPS, self, "_cancelAnimStep:", None, True
        )

    def _cancelAnimStep_(self, timer) -> None:
        """Animate the dismiss sequence: grow for 60ms, then shrink and fade."""
        self._cancel_elapsed += 1.0 / _DISMISS_ANIM_FPS
        phase, scale, alpha, done = _dismiss_animation_state(self._cancel_elapsed)
        self._cancel_phase = phase
        self._set_overlay_scale(scale)
        self._window.setAlphaValue_(alpha)

        if done:
            self._cancel_dismiss_animation()
            self._window.setAlphaValue_(0.0)
            self._set_overlay_scale(1.0)
            self._reset_backdrop_layer()
            if self._backdrop_renderer is not None and hasattr(self._backdrop_renderer, "stop_live_stream"):
                self._backdrop_renderer.stop_live_stream()
            self._stop_fullscreen_compositor()
            self._window.orderOut_(None)
            self._visible = False
            self._cancel_pulse()

    def hide(self) -> None:
        """Fade out without competing pulse work during the dismiss animation."""
        if self._window is None:
            return
        self._visible = False
        self._streaming = False
        self._cancel_linger()
        self._cancel_visual_start()
        # Freeze the assistant surface while fading.  Pulse ticks restyle large
        # response ranges and may rebuild optical masks/backdrop state; doing
        # that during dismiss competes with the fade and preview overlay.
        self._cancel_pulse()
        self._start_fade_out()

    def set_recording_load_shed(self, active: bool) -> None:
        """Freeze expensive assistant animation while the user is recording."""
        active = bool(active)
        if active:
            if getattr(self, "_recording_load_shed", False):
                return
            self._recording_load_shed = True
            self._recording_load_shed_resume_pulse = bool(
                getattr(self, "_visible", False)
            )
            self._cancel_pulse()
            return

        if not getattr(self, "_recording_load_shed", False):
            return
        self._recording_load_shed = False
        should_resume = bool(getattr(self, "_recording_load_shed_resume_pulse", False))
        self._recording_load_shed_resume_pulse = False
        if (
            should_resume
            and getattr(self, "_visible", False)
            and getattr(self, "_pulse_timer", None) is None
            and not (
                getattr(self, "_fade_timer", None) is not None
                and getattr(self, "_fade_direction", 0) == 1
            )
        ):
            self._start_pulse_timer()

    def set_cancel_spring(self, target: float) -> None:
        """Set the cancel spring target (0.0 = idle, 1.0 = winding up).

        The spring animates toward the target in the pulse tick, so
        calling this once starts the wind-up or snap-back.
        """
        self._cancel_spring_target = max(0.0, min(1.0, target))
        if target > 0.0:
            self._cancel_spring_fired = False  # reset for new wind-up

    def set_utterance(self, text: str) -> None:
        """Show the user's utterance in the text view at reduced opacity."""
        self._utterance_text = text
        if self._text_view is None or not self._visible:
            return
        attr_str = self._make_utterance_attributed(text)
        self._text_view.textStorage().setAttributedString_(attr_str)
        self._update_layout()

    def _make_utterance_attributed(self, text: str):
        from AppKit import (
            NSMutableAttributedString,
            NSForegroundColorAttributeName,
            NSFontAttributeName,
            NSShadowAttributeName,
            NSShadow,
        )
        user_r, user_g, user_b = _user_text_color_for_brightness(self._brightness)
        attr_str = NSMutableAttributedString.alloc().initWithString_(text)
        attr_str.addAttribute_value_range_(
            NSForegroundColorAttributeName,
            NSColor.colorWithSRGBRed_green_blue_alpha_(
                user_r,
                user_g,
                user_b,
                _user_text_alpha_for_breath(0.5),
            ),
            (0, len(text)),
        )
        attr_str.addAttribute_value_range_(
            NSFontAttributeName,
            NSFont.systemFontOfSize_weight_(_USER_FONT_SIZE, 0.0),
            (0, len(text)),
        )
        glow = NSShadow.alloc().init()
        glow.setShadowColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(user_r, user_g, user_b, 0.10)
        )
        glow.setShadowOffset_((0, 0))
        glow.setShadowBlurRadius_(2.0)
        attr_str.addAttribute_value_range_(
            NSShadowAttributeName, glow, (0, len(text))
        )
        return attr_str

    def _clear_agent_shell_chrome(self) -> None:
        for label in (
            getattr(self, "_agent_shell_header_label", None),
            getattr(self, "_agent_shell_footer_label", None),
        ):
            if label is not None:
                label.setStringValue_("")
                label.setHidden_(True)
                if hasattr(label, "setAlphaValue_"):
                    label.setAlphaValue_(1.0)

    def clear_agent_shell_chrome(self) -> None:
        """Clear Agent Shell chrome when the visible transcript belongs to another mode."""
        self._clear_agent_shell_chrome()
        self.set_agent_shell_cards([])
        self.set_agent_shell_primitives([])
        self._update_layout()

    def _set_agent_shell_chrome_texts(self, header: str = "", footer: str = "") -> None:
        header_label = getattr(self, "_agent_shell_header_label", None)
        if header_label is not None:
            header_label.setStringValue_(header)
            header_label.setHidden_(not bool(header))
        footer_label = getattr(self, "_agent_shell_footer_label", None)
        if footer_label is not None:
            footer_label.setStringValue_(footer)
            footer_label.setHidden_(not bool(footer))
        self._apply_agent_shell_chrome_theme()
        self._sync_agent_shell_chrome_punchthrough_visibility()
        self._refresh_punchthrough_mask_if_needed()

    def set_agent_shell_cards(self, cards: list[dict] | None) -> None:
        self._agent_shell_cards = [
            dict(card) for card in (cards or []) if isinstance(card, dict)
        ]
        self._agent_shell_primitives = []
        self._push_agent_shell_payload()

    def set_agent_shell_primitives(self, primitives: list[dict] | None) -> None:
        self._agent_shell_primitives = [
            dict(primitive) for primitive in (primitives or []) if isinstance(primitive, dict)
        ]
        self._agent_shell_cards = []
        self._push_agent_shell_payload()

    def _push_agent_shell_payload(self) -> None:
        compositor = getattr(self, "_fullscreen_compositor", None)
        if compositor is None:
            return
        shell_config = self._current_optical_shell_config()
        if shell_config is None:
            return
        try:
            compositor.update_shell_config(shell_config)
        except Exception:
            logger.debug("Failed to push Agent Shell card payload", exc_info=True)

    def replace_transcript(
        self,
        *,
        utterance: str,
        response: str,
        agent_shell_header: str = "",
        agent_shell_footer: str = "",
        agent_shell_cards: list[dict] | None = None,
        agent_shell_primitives: list[dict] | None = None,
    ) -> None:
        """Replace recalled transcript and chrome with one layout pass."""
        self._utterance_text = utterance
        self._collapsed_text = ""
        self._response_text = ""
        self._set_agent_shell_chrome_texts(agent_shell_header, agent_shell_footer)
        if agent_shell_primitives:
            self.set_agent_shell_primitives(agent_shell_primitives)
        else:
            self.set_agent_shell_cards(agent_shell_cards or [])
        if self._text_view is None or not self._visible:
            self._response_text = response
            return
        if response:
            self.set_response_text(response)
        else:
            self._text_view.textStorage().setAttributedString_(
                self._make_utterance_attributed(utterance)
            )
            self._update_layout()
            self._refresh_punchthrough_mask_if_needed()

    def set_agent_shell_header(self, text: str) -> None:
        """Set the Agent Shell identity line above the transcript."""
        label = getattr(self, "_agent_shell_header_label", None)
        if label is not None:
            label.setStringValue_(text)
            label.setHidden_(not bool(text))
            self._apply_agent_shell_chrome_theme()
            self._sync_agent_shell_chrome_punchthrough_visibility()
            self._refresh_punchthrough_mask_if_needed()
            self._update_layout()

    def set_agent_shell_footer(self, text: str) -> None:
        """Set the Agent Shell metadata line below the transcript."""
        label = getattr(self, "_agent_shell_footer_label", None)
        if label is not None:
            label.setStringValue_(text)
            label.setHidden_(not bool(text))
            self._apply_agent_shell_chrome_theme()
            self._sync_agent_shell_chrome_punchthrough_visibility()
            self._refresh_punchthrough_mask_if_needed()
            self._update_layout()

    def _make_collapsed_attributed(self, text: str):
        """Build an attributed string for collapsed thinking text."""
        from AppKit import (
            NSMutableAttributedString,
            NSForegroundColorAttributeName,
            NSFontAttributeName,
        )

        attr = NSMutableAttributedString.alloc().initWithString_(text)
        rng = (0, len(text))
        # Warm muted tone — visually distinct from tool indicators (cool/gray)
        attr.addAttribute_value_range_(
            NSForegroundColorAttributeName,
            NSColor.colorWithSRGBRed_green_blue_alpha_(0.65, 0.62, 0.55, 0.45),
            rng,
        )
        attr.addAttribute_value_range_(
            NSFontAttributeName,
            NSFont.systemFontOfSize_weight_(12.0, 0.0),
            rng,
        )
        return attr

    def _make_tool_indicator_fragment(self, text: str):
        """Build a compact attributed string for tool call/result indicators."""
        from AppKit import (
            NSMutableAttributedString,
            NSForegroundColorAttributeName,
            NSFontAttributeName,
        )

        attr = NSMutableAttributedString.alloc().initWithString_(text)
        rng = (0, len(text))
        attr.addAttribute_value_range_(
            NSForegroundColorAttributeName,
            NSColor.colorWithSRGBRed_green_blue_alpha_(0.42, 0.50, 0.56, 0.72),
            rng,
        )
        attr.addAttribute_value_range_(
            NSFontAttributeName,
            NSFont.systemFontOfSize_weight_(12.0, 0.0),
            rng,
        )
        return attr

    def _approval_command_font(self):
        mono_font = getattr(NSFont, "monospacedSystemFontOfSize_weight_", None)
        if callable(mono_font):
            return mono_font(16.0, 0.12)
        return NSFont.systemFontOfSize_weight_(16.0, 0.12)

    def _apply_approval_card_style(self, attr, text: str) -> None:
        """Layer approval-card hierarchy over the normal response styling."""
        ranges = _approval_card_ranges(text)
        if not ranges:
            return
        from AppKit import NSForegroundColorAttributeName, NSFontAttributeName

        fg_r, fg_g, fg_b = _assistant_foreground_color_for_brightness(self._brightness)
        header_color = NSColor.colorWithSRGBRed_green_blue_alpha_(0.58, 0.68, 0.73, 0.88)
        action_color = NSColor.colorWithSRGBRed_green_blue_alpha_(fg_r, fg_g, fg_b, 0.68)
        metadata_color = NSColor.colorWithSRGBRed_green_blue_alpha_(fg_r, fg_g, fg_b, 0.76)

        if "header" in ranges:
            attr.addAttribute_value_range_(
                NSForegroundColorAttributeName, header_color, ranges["header"]
            )
            attr.addAttribute_value_range_(
                NSFontAttributeName,
                NSFont.systemFontOfSize_weight_(15.0, 0.22),
                ranges["header"],
            )
        if "action" in ranges:
            attr.addAttribute_value_range_(
                NSForegroundColorAttributeName, action_color, ranges["action"]
            )
            attr.addAttribute_value_range_(
                NSFontAttributeName,
                NSFont.systemFontOfSize_weight_(12.0, 0.0),
                ranges["action"],
            )
        if "command" in ranges:
            attr.addAttribute_value_range_(
                NSForegroundColorAttributeName,
                NSColor.colorWithSRGBRed_green_blue_alpha_(fg_r, fg_g, fg_b, 0.94),
                ranges["command"],
            )
            attr.addAttribute_value_range_(
                NSFontAttributeName,
                self._approval_command_font(),
                ranges["command"],
            )
        if "reason" in ranges:
            attr.addAttribute_value_range_(
                NSForegroundColorAttributeName, metadata_color, ranges["reason"]
            )
            attr.addAttribute_value_range_(
                NSFontAttributeName,
                NSFont.systemFontOfSize_weight_(12.5, -0.05),
                ranges["reason"],
            )
        if "cwd" in ranges:
            attr.addAttribute_value_range_(
                NSForegroundColorAttributeName, metadata_color, ranges["cwd"]
            )
            attr.addAttribute_value_range_(
                NSFontAttributeName,
                NSFont.systemFontOfSize_weight_(12.0, -0.05),
                ranges["cwd"],
            )

    def _leading_response_separator(self) -> str:
        """Return the visible break before the first streamed response token.

        A faint rule keeps the handoff legible without spending several blank
        lines of scarce overlay height.
        """
        return "\n────────\n"

    def _make_response_separator_attributed(self):
        from AppKit import (
            NSMutableAttributedString,
            NSForegroundColorAttributeName,
            NSFontAttributeName,
        )

        separator = self._leading_response_separator()
        sep = NSMutableAttributedString.alloc().initWithString_(separator)
        fg_r, fg_g, fg_b = _assistant_foreground_color_for_brightness(self._brightness)
        sep.addAttribute_value_range_(
            NSForegroundColorAttributeName,
            NSColor.colorWithSRGBRed_green_blue_alpha_(fg_r, fg_g, fg_b, 0.28),
            (0, len(separator)),
        )
        sep.addAttribute_value_range_(
            NSFontAttributeName,
            NSFont.systemFontOfSize_weight_(8.0, 0.0),
            (0, len(separator)),
        )
        return sep

    def _refresh_punchthrough_mask_if_needed(self) -> None:
        """Keep the optical text cutout in sync after immediate text/layout edits."""
        if not getattr(self, "_text_punchthrough", False):
            return
        # Mark dirty so the next pulse tick (or this call) re-rasterizes.
        self._punchthrough_mask_dirty = True
        self._update_punchthrough_mask()

    def append_token(self, token: str) -> None:
        """Append a streamed response token."""
        if self._text_view is None or not self._visible:
            return
        first_token = len(self._response_text) == 0
        self._response_text += token

        if first_token and self._utterance_text:
            self._text_view.textStorage().appendAttributedString_(
                self._make_response_separator_attributed()
            )

        # Style tool call indicators smaller (like collapsed thinking)
        # Covers: [calling tool…], [~N tokens], [screen capture · ~N tokens],
        #         ["query" in path], [path · ~N tokens], [100%]
        stripped = token.lstrip("\n ")
        is_tool_indicator = stripped.startswith("[") and not stripped.startswith("[!")
        if is_tool_indicator:
            frag = self._make_tool_indicator_fragment(token)
        else:
            frag = self._make_response_fragment(token)
        self._text_view.textStorage().appendAttributedString_(frag)

        self._update_layout()
        self._refresh_punchthrough_mask_if_needed()

    def set_response_text(self, text: str) -> None:
        """Replace the visible assistant response with final canonical text.

        Rebuilds the full attributed string (utterance + response) in one shot
        so that _update_layout is called exactly once.  Calling set_utterance
        then append_token would call _update_layout twice: first with only the
        utterance (shrinking the window back to minimum height), then again with
        the full response — producing a visible size flicker.
        """
        self._response_text = ""
        if self._text_view is None or not self._visible:
            self._response_text = text
            return

        from AppKit import NSMutableAttributedString

        combined = NSMutableAttributedString.alloc().initWithString_("")

        if self._utterance_text:
            utt = self._make_utterance_attributed(self._utterance_text)
            combined.appendAttributedString_(utt)

            # Re-inject collapsed thinking text if present
            if self._collapsed_text:
                combined.appendAttributedString_(
                    self._make_collapsed_attributed("\n\n" + self._collapsed_text)
                )

        self._text_view.textStorage().setAttributedString_(combined)

        if text:
            self._response_text = ""
            self.append_token(text)
        else:
            self._update_layout()
            self._refresh_punchthrough_mask_if_needed()

    def _current_hue_rgb(self):
        """Get the current hue rotation color as (r, g, b)."""
        hue = getattr(self, '_color_phase', 0.0)
        s, v = 0.75, 0.95
        c = v * s
        x = c * (1.0 - abs((hue * 6.0) % 2.0 - 1.0))
        m = v - c
        h6 = hue * 6.0
        if h6 < 1:
            return (c + m, x + m, m)
        elif h6 < 2:
            return (x + m, c + m, m)
        elif h6 < 3:
            return (m, c + m, x + m)
        elif h6 < 4:
            return (m, x + m, c + m)
        elif h6 < 5:
            return (x + m, m, c + m)
        else:
            return (c + m, m, x + m)

    def _make_response_fragment(self, token: str):
        """Create an attributed string fragment for a response token.

        The colorful identity lives in a blurred underlay while the readable
        foreground stays high-contrast against the adaptive surface.
        """
        from AppKit import (
            NSMutableAttributedString,
            NSForegroundColorAttributeName,
            NSFontAttributeName,
            NSShadowAttributeName,
            NSShadow,
        )
        fg_r, fg_g, fg_b = _assistant_foreground_color_for_brightness(self._brightness)
        if getattr(self, "_response_chroma_active", True):
            blur_r, blur_g, blur_b = self._current_hue_rgb()
        else:
            blur_r, blur_g, blur_b = fg_r, fg_g, fg_b
        frag = NSMutableAttributedString.alloc().initWithString_(token)
        response_color = NSColor.colorWithSRGBRed_green_blue_alpha_(
            fg_r, fg_g, fg_b, _ASSISTANT_TEXT_ALPHA_MAX
        )
        frag.addAttribute_value_range_(
            NSForegroundColorAttributeName, response_color, (0, len(token))
        )
        frag.addAttribute_value_range_(
            NSFontAttributeName,
            NSFont.systemFontOfSize_weight_(_FONT_SIZE, 0.0),
            (0, len(token)),
        )
        # Blurred colorful underlay behind the crisp readable text.
        glow = NSShadow.alloc().init()
        glow.setShadowColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(blur_r, blur_g, blur_b, 0.7)
        )
        glow.setShadowOffset_((0, 0))
        glow.setShadowBlurRadius_(_ASSISTANT_BLUR_RADIUS)
        frag.addAttribute_value_range_(
            NSShadowAttributeName, glow, (0, len(token))
        )
        self._apply_approval_card_style(frag, token)
        return frag

    def set_tool_active(self, active: bool) -> None:
        """Show or hide the tool execution indicator."""
        self._tool_mode = active
        if active and self._visible:
            if self._thinking_label is not None:
                self._thinking_label.setHidden_(False)
                self._thinking_label.setStringValue_("tool…")
            if self._thinking_timer is None:
                self._start_thinking_timer()

    def finish(self) -> None:
        """Called when the response stream is complete.

        The overlay stays visible indefinitely — the user dismisses it
        by holding spacebar (which triggers cancel_dismiss via __main__).
        No linger timer. The user is in control.
        """
        self._streaming = False
        self._stop_thinking_timer()
        # Pulse keeps running — the overlay is alive until dismissed

    # ── TTS amplitude ─────────────────────────────────────────

    _TTS_RISE = 0.70      # fast but not instant — avoids harsh pop-in
    _TTS_DECAY = 0.82     # gentle falloff — no clipping between words
    _TTS_MULTIPLIER = 25.0
    _TTS_ALPHA_MIN = _ASSISTANT_TEXT_ALPHA_MIN  # keep assistant text firmly readable during playback
    _TTS_ALPHA_MAX = _ASSISTANT_TEXT_ALPHA_MAX

    def update_tts_amplitude(self, rms: float) -> None:
        """Update smoothed TTS amplitude from audio RMS.

        The pulse step reads _tts_amplitude when _tts_active is True to
        drive the response text alpha — no window alpha manipulation here,
        so there's no fighting between pulse and TTS updates.
        Must be called on the main thread.
        """
        if rms > self._tts_amplitude:
            self._tts_amplitude += (rms - self._tts_amplitude) * self._TTS_RISE
        else:
            self._tts_amplitude *= self._TTS_DECAY

    def tts_start(self) -> None:
        """Prepare overlay for TTS-driven amplitude."""
        self._tts_active = True
        self._tts_amplitude = 0.0

    def tts_stop(self) -> None:
        """Return to pulse-driven alpha after TTS ends."""
        self._tts_active = False
        self._tts_amplitude = 0.0

    # ── animation ───────────────────────────────────────────

    def fadeStep_(self, timer) -> None:
        """One step of the fade animation.

        During fade-out, the utterance text fades at double rate so it
        disappears before the assistant response.
        """
        self._fade_step += 1
        progress = self._fade_step / _FADE_STEPS

        if self._fade_direction == 1:
            if getattr(self, "_fullscreen_compositor", None) is not None:
                alpha = 1.0
            else:
                eased = progress * progress
                alpha = eased
        else:
            if getattr(self, "_fullscreen_compositor", None) is not None:
                alpha = self._fade_from
            else:
                eased = progress * progress
                alpha = self._fade_from * (1.0 - eased)

            # Fade utterance at double rate
            if self._text_view is not None and self._utterance_text:
                from AppKit import NSForegroundColorAttributeName
                utt_progress = min(progress * 2.0, 1.0)
                utt_alpha = 0.35 * (1.0 - utt_progress * utt_progress)
                utt_len = min(len(self._utterance_text), self._text_view.textStorage().length() if hasattr(self._text_view.textStorage(), 'length') else 0)
                if utt_len > 0:
                    try:
                        self._text_view.textStorage().addAttribute_value_range_(
                            NSForegroundColorAttributeName,
                            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, utt_alpha),
                            (0, utt_len),
                        )
                    except Exception:
                        pass

        self._window.setAlphaValue_(alpha)

        if self._fade_step >= _FADE_STEPS:
            self._cancel_fade()
            if self._fade_direction == 1:
                if (
                    not getattr(self, "_recording_load_shed", False)
                    and not self._materialization_owns_fill_layers()
                ):
                    self._start_pulse_timer()
            elif self._fade_direction == -1:
                self._window.setAlphaValue_(0.0)
                self._reset_backdrop_layer()
                if self._backdrop_renderer is not None and hasattr(self._backdrop_renderer, "stop_live_stream"):
                    self._backdrop_renderer.stop_live_stream()
                self._stop_fullscreen_compositor()
                self._window.orderOut_(None)
                self._cancel_pulse()  # now kill the pulse
            else:
                self._window.setAlphaValue_(1.0)

    def pulseStep_(self, timer) -> None:
        """Dual-phase pulse: user and assistant text breathe independently.

        Assistant text: faster period (0.8x base), double-smoothstep for
        extra-aggressive easing, violet-amber color oscillation.
        User text: slower period (1.5x base), single smoothstep, blue
        color shift, phase-offset by 0.3 and diverging naturally.
        """
        try:
            self._pulseStepInner()
        except Exception:
            logger.exception("pulseStep_ crashed")
        finally:
            metrics = getattr(self, "_metrics", None)
            if metrics is not None:
                metrics.record_display_tick(now=time.perf_counter())

    def _materialization_owns_fill_layers(self) -> bool:
        """Return True while slit materialization owns shell-body geometry."""
        return getattr(self, "_materialization_timer", None) is not None

    def _clear_punchthrough_mask_for_materialization(self) -> None:
        """Let the shell body materialize before text punch-through owns a mask."""
        fill = getattr(self, "_fill_layer", None)
        if fill is not None and hasattr(fill, "setMask_"):
            fill.setMask_(None)
        self._punchthrough_mask_layer = None
        boost = getattr(self, "_boost_layer", None)
        if boost is not None and hasattr(boost, "setMask_"):
            boost.setMask_(None)
        self._boost_mask_layer = None
        self._punchthrough_mask_dirty = True

    def _pulseStepInner(self):
        if self._text_view is None:
            return
        present_start = time.perf_counter()
        dt = 1.0 / _PULSE_HZ

        # When compositor is active, sample shell-region brightness
        # directly — much more accurate than the glow's 4-patch screen
        # average. Keep this tight enough that background crossings feel
        # attached to the window underneath rather than arriving late.
        compositor = getattr(self, "_fullscreen_compositor", None)
        if compositor is not None:
            _b_tick = getattr(self, '_brightness_sample_tick', 0)
            if _b_tick >= 0:
                if _b_tick % _BRIGHTNESS_COMPOSITOR_SAMPLE_TICKS == 0:
                    compositor.refresh_brightness()
                self._brightness_target = compositor.sampled_brightness
            self._brightness_sample_tick = _b_tick + 1
        target = getattr(self, "_brightness_target", 0.0)
        current = getattr(self, "_brightness", 0.0)
        if abs(target - current) > 0.001:
            self._brightness = _advance_command_brightness(current, target)
        t = getattr(self, "_brightness", 0.0)
        self._apply_surface_theme()

        # Advance phases
        self._pulse_phase_asst += dt / _PULSE_PERIOD_ASST
        if self._pulse_phase_asst > 1.0:
            self._pulse_phase_asst -= 1.0
        self._pulse_phase_user += dt / _PULSE_PERIOD_USER
        if self._pulse_phase_user > 1.0:
            self._pulse_phase_user -= 1.0

        # Assistant: one slow breath, 5 seconds. Asymmetric:
        # ease-out toward bright (lingers near peak), ease-in back to dim
        # (dips quickly). Raw sine → squared so it spends more time high.
        raw_breath = 0.5 * (1.0 + math.cos(2.0 * math.pi * self._pulse_phase_asst))
        breath = raw_breath * raw_breath  # squared: lingers near 1.0, dips briefly
        self._apply_backdrop_pulse_style(breath)
        pulse_alpha_a = _assistant_text_alpha_for_breath(breath)

        # Smooth cross-fade between pulse and TTS-driven alpha over ~500ms.
        # _tts_blend ramps toward 1.0 when TTS is active, toward 0.0 when not.
        # At 15Hz pulse rate, 0.12 per tick ≈ 0.55s full ramp.
        _BLEND_RATE = 0.12
        if self._tts_active:
            self._tts_blend = min(self._tts_blend + _BLEND_RATE, 1.0)
        else:
            self._tts_blend = max(self._tts_blend - _BLEND_RATE, 0.0)

        if self._tts_blend > 0.0:
            tts_scaled = min(self._tts_amplitude * self._TTS_MULTIPLIER, 1.0)
            tts_alpha = self._TTS_ALPHA_MIN + tts_scaled * (self._TTS_ALPHA_MAX - self._TTS_ALPHA_MIN)
            alpha_a = pulse_alpha_a * (1.0 - self._tts_blend) + tts_alpha * self._tts_blend
        else:
            alpha_a = pulse_alpha_a

        # When text punch-through is active, force full text alpha
        # so the destinationOut compositing cleanly erases the fill.
        if getattr(self, "_text_punchthrough", False):
            alpha_a = 1.0

        # User: raw sine → single smoothstep (same aggressiveness as before)
        raw_u = 0.5 * (1.0 - math.cos(2.0 * math.pi * self._pulse_phase_user))
        pulse_u = raw_u * raw_u * (3.0 - 2.0 * raw_u)
        utt_alpha = _user_text_alpha_for_breath(pulse_u)
        if getattr(self, "_text_punchthrough", False):
            utt_alpha = 1.0

        # Full spectrum hue rotation with velocity undulation
        # The speed varies sinusoidally so it dwells in some colors and
        # glides through others — always moving, never truly paused.
        self._color_velocity_phase += dt / _COLOR_VELOCITY_PERIOD
        if self._color_velocity_phase > 1.0:
            self._color_velocity_phase -= 1.0
        vel_raw = 0.5 * (1.0 - math.cos(2.0 * math.pi * self._color_velocity_phase))
        vel = _COLOR_VELOCITY_MIN + vel_raw * (_COLOR_VELOCITY_MAX - _COLOR_VELOCITY_MIN)
        self._color_phase += (dt / _COLOR_CYCLE_PERIOD) * vel
        if self._color_phase > 1.0:
            self._color_phase -= 1.0
        hue = self._color_phase

        # ── Cancel spring animation ──
        # The spring chases _cancel_spring_target with asymmetric speed:
        # winding up is deliberate (~600ms to full), easing out is smooth
        # (~300ms with deceleration).
        _SPRING_THRESHOLD = 0.83  # ~500ms into the 600ms wind-up
        spring = self._cancel_spring
        target = self._cancel_spring_target
        if spring < target:
            # Wind up — deliberate
            spring = min(target, spring + dt / 0.6)
            # Fire cancel at threshold crossing — don't wait for release
            if spring >= _SPRING_THRESHOLD and not self._cancel_spring_fired:
                self._cancel_spring_fired = True
                self._cancel_spring_target = 0.0  # start ease-out
                target = 0.0
                cb = self._on_cancel_spring_threshold
                if cb is not None:
                    cb()
        if spring > target:
            # Ease out — smooth deceleration (quadratic decay)
            rate = dt / 0.3 * (1.0 + spring)  # faster when higher
            spring = max(target, spring - rate)
        self._cancel_spring = spring

        if spring > 0.001:
            # Blend hue toward warm amber (~0.08, orange-gold).
            # Use shortest arc around the hue wheel.
            amber_hue = 0.08
            if hue > 0.5 + amber_hue:
                target_hue = 1.0 + amber_hue  # wrap forward
            else:
                target_hue = amber_hue
            hue = hue + (target_hue - hue) * spring
            if hue >= 1.0:
                hue -= 1.0

        # Log color phase every ~1s (every 30th tick)
        if not hasattr(self, '_color_log_counter'):
            self._color_log_counter = 0
        self._color_log_counter += 1
        if self._color_log_counter % 30 == 0:
            logger.info("Color phase: %.3f hue, vel_phase=%.3f, spring=%.3f", hue, self._color_velocity_phase, spring)

        # Saturation and value — spring winds up saturation toward vivid red
        base_s, base_v = 0.35, 0.55  # saturated + darker — readable against opaque fill
        s = base_s + (0.75 - base_s) * spring  # 0.35 → 0.75 at full wind
        v = base_v + (0.65 - base_v) * spring  # 0.55 → 0.65 at full wind
        c = v * s
        x = c * (1.0 - abs((hue * 6.0) % 2.0 - 1.0))
        m = v - c
        h6 = hue * 6.0
        if h6 < 1:
            r, g, b = c + m, x + m, m
        elif h6 < 2:
            r, g, b = x + m, c + m, m
        elif h6 < 3:
            r, g, b = m, c + m, x + m
        elif h6 < 4:
            r, g, b = m, x + m, c + m
        elif h6 < 5:
            r, g, b = x + m, m, c + m
        else:
            r, g, b = c + m, m, x + m
        response_r, response_g, response_b = _assistant_foreground_color_for_brightness(t)

        # Alternate color: offset hue by ~0.4, slightly out of phase so
        # when the main color is bright the alt is dark and vice versa.
        alt_hue = (hue + 0.4) % 1.0
        alt_breath = 1.0 - breath  # out of phase
        alt_v = base_v + (0.65 - base_v) * spring
        alt_v = alt_v * (0.6 + 0.4 * alt_breath)  # modulate darker
        alt_c = alt_v * s
        alt_x = alt_c * (1.0 - abs((alt_hue * 6.0) % 2.0 - 1.0))
        alt_m = alt_v - alt_c
        alt_h6 = alt_hue * 6.0
        if alt_h6 < 1:
            alt_r, alt_g, alt_b = alt_c + alt_m, alt_x + alt_m, alt_m
        elif alt_h6 < 2:
            alt_r, alt_g, alt_b = alt_x + alt_m, alt_c + alt_m, alt_m
        elif alt_h6 < 3:
            alt_r, alt_g, alt_b = alt_m, alt_c + alt_m, alt_x + alt_m
        elif alt_h6 < 4:
            alt_r, alt_g, alt_b = alt_m, alt_x + alt_m, alt_c + alt_m
        elif alt_h6 < 5:
            alt_r, alt_g, alt_b = alt_x + alt_m, alt_m, alt_c + alt_m
        else:
            alt_r, alt_g, alt_b = alt_c + alt_m, alt_m, alt_x + alt_m
        # Update text colors per-range
        if self._text_view is not None:
            from AppKit import NSForegroundColorAttributeName as _FG_pulse
            ts = self._text_view.textStorage()
            total_len = ts.length() if hasattr(ts, 'length') else 0
            if total_len > 0 and self._utterance_text:
                utt_len = min(len(self._utterance_text), total_len)
                    # User text: fixed dark gray (or white for punch-through).
                try:
                    if getattr(self, "_text_punchthrough", False):
                        ur, ug, ub = 1.0, 1.0, 1.0
                    else:
                        user_base = _user_text_color_for_brightness(t)
                        ur = user_base[0]
                        ug = user_base[1]
                        ub = user_base[2]
                    ts.addAttribute_value_range_(
                        _FG_pulse,
                        NSColor.colorWithSRGBRed_green_blue_alpha_(ur, ug, ub, utt_alpha),
                        (0, utt_len),
                    )
                except Exception:
                    pass
                # Response text has two visual layers per character:
                # a crisp adaptive foreground for legibility, plus a colored
                # blurred shadow that carries the chromatic identity. In
                # punch-through mode the live NSTextView is hidden and only
                # supplies the mask source, so per-tick restyling is invisible
                # main-thread work.
                resp_start = utt_len + 2
                resp_len = total_len - resp_start
                _punchthrough = getattr(self, "_text_punchthrough", False)
                if resp_start < total_len and resp_len > 0 and not _punchthrough:
                    from AppKit import (
                        NSShadowAttributeName as _SH_pulse,
                        NSShadow,
                        NSFontAttributeName as _FN_pulse,
                        NSFont,
                    )
                    # Cache font allocation once per tick, not per character.
                    light_font = NSFont.systemFontOfSize_weight_(
                        _FONT_SIZE, -0.2  # medium-light weight — thicker punch-through
                    )
                    try:
                        # Bulk span path: 3 coarse spans (edge, center, edge)
                        # instead of per-character iteration. The gradient
                        # approximation is sufficient for the 2-5s breathing
                        # animation on this fallback rendering path.
                        # Foreground and font are uniform across the full range.
                        ts.addAttribute_value_range_(
                            _FG_pulse,
                            NSColor.colorWithSRGBRed_green_blue_alpha_(
                                response_r,
                                response_g,
                                response_b,
                                _ASSISTANT_TEXT_ALPHA_MAX,
                            ),
                            (resp_start, resp_len),
                        )
                        ts.addAttribute_value_range_(
                            _FN_pulse, light_font, (resp_start, resp_len)
                        )
                        if resp_len <= 3:
                            # Too short for span splits; single shadow pass.
                            lum = 0.299 * r + 0.587 * g + 0.114 * b
                            shadow = NSShadow.alloc().init()
                            shadow.setShadowColor_(
                                NSColor.colorWithSRGBRed_green_blue_alpha_(
                                    r, g, b, 0.7 + 0.3 * lum
                                )
                            )
                            shadow.setShadowOffset_((0, 0))
                            shadow.setShadowBlurRadius_(5.0 + lum * 14.0)
                            ts.addAttribute_value_range_(
                                _SH_pulse, shadow, (resp_start, resp_len)
                            )
                        else:
                            # Three spans: edge (0..e), center (e..m), edge (m..end)
                            edge = max(1, resp_len // 4)
                            mid = max(edge + 1, resp_len * 3 // 4)
                            spans = [
                                (resp_start,        edge,           0.0),   # edge: main color
                                (resp_start + edge, mid - edge,     1.0),   # center: alt color
                                (resp_start + mid,  resp_len - mid, 0.0),   # edge: main color
                            ]
                            for span_start, span_len, weight in spans:
                                if span_len <= 0:
                                    continue
                                cr = _lerp(r, alt_r, weight)
                                cg = _lerp(g, alt_g, weight)
                                cb = _lerp(b, alt_b, weight)
                                lum = 0.299 * cr + 0.587 * cg + 0.114 * cb
                                shadow = NSShadow.alloc().init()
                                shadow.setShadowColor_(
                                    NSColor.colorWithSRGBRed_green_blue_alpha_(
                                        cr, cg, cb, 0.7 + 0.3 * lum
                                    )
                                )
                                shadow.setShadowOffset_((0, 0))
                                shadow.setShadowBlurRadius_(5.0 + lum * 14.0)
                                ts.addAttribute_value_range_(
                                    _SH_pulse, shadow, (span_start, span_len)
                                )
                    except Exception:
                        pass
            elif total_len > 0:
                self._text_view.setTextColor_(
                    NSColor.colorWithSRGBRed_green_blue_alpha_(
                        response_r, response_g, response_b, alpha_a
                    )
                )

        # Pulse the glow with assistant phase oscillating color
        glow_nscolor = NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, 1.0)
        glow_opacity = 0.5 + 0.3 * breath
        # Keep the assistant surface in the same materially present opacity
        # band as the preview overlay on bright backgrounds. The preview gets
        # its presence from a much more assertive fill ramp than the command
        # overlay previously used, so mirror that shape here while preserving
        # the pulse-driven rhythm.
        materialization_owns_fill = self._materialization_owns_fill_layers()
        if (
            hasattr(self, '_fill_layer')
            and self._fill_layer is not None
            and not materialization_owns_fill
        ):
            fill_drive = _lerp(breath, breath * breath, t)
            if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
                fill_min = _lerp(
                    _COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MIN_DARK,
                    _COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MIN_LIGHT,
                    t,
                )
                fill_max = _lerp(
                    _COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MAX_DARK,
                    _COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MAX_LIGHT,
                    t,
                )
                if _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_REVEAL:
                    fill_min = 0.0
                    fill_max = 0.0
                elif getattr(self, "_fullscreen_compositor", None) is not None:
                    # Graphic mode: dark-on-dark, light-on-light.
                    # Steep sigmoid so mid-tones commit to one side
                    # rather than lingering in a bland middle.
                    fill_drive = 0.5
                    ct = _clamp01((t - 0.45) * 6.0 + 0.5)
                    ct = ct * ct * (3.0 - 2.0 * ct)
                    fill_min = _lerp(0.72, 0.96, ct)
                    fill_max = _lerp(0.82, 0.99, ct)
            else:
                fill_min = _fill_layer_opacity_for_breath(0.0)
                fill_max = _fill_layer_opacity_for_breath(1.0)
            new_opacity = min(_lerp(fill_min, fill_max, fill_drive), 0.99)
            if abs(new_opacity - getattr(self, '_last_fill_opacity', -1.0)) > 0.005:
                self._fill_layer.setOpacity_(new_opacity)
                self._last_fill_opacity = new_opacity
        # Brightness floor + boost for punch-through legibility.
        # On light backgrounds (dark fill), guarantee the warped content
        # inside the rounded shell has luminance >= 0.5 so text holes read bright.
        compositor = getattr(self, "_fullscreen_compositor", None)
        if compositor is not None and getattr(self, "_text_punchthrough", False):
            _bt = _clamp01((t - 0.45) * 6.0 + 0.5)
            _bt = _bt * _bt * (3.0 - 2.0 * _bt)
            # Light bg → 0.5 brightness floor; dark bg → no floor
            min_b = _lerp(0.0, 0.5, _bt)
            last_min_b = getattr(self, '_last_min_brightness', -1.0)
            if abs(min_b - last_min_b) > 0.01:
                compositor.update_shell_config_key("min_brightness", min_b)
                self._last_min_brightness = min_b
        elif compositor is not None:
            if getattr(self, '_last_min_brightness', 0.0) > 0.01:
                compositor.update_shell_config_key("min_brightness", 0.0)
                self._last_min_brightness = 0.0
        # Boost layer: glyph-shaped lift/drop behind punch-through text.
        boost_layer = getattr(self, "_boost_layer", None)
        if boost_layer is not None and materialization_owns_fill:
            boost_layer.setHidden_(True)
            boost_layer.setOpacity_(0.0)
        elif boost_layer is not None and getattr(self, "_text_punchthrough", False):
            boost_rgb, boost_opacity = _punchthrough_boost_style_for_brightness(t)
            try:
                from Quartz import CGColorCreateSRGB
                boost_layer.setBackgroundColor_(
                    CGColorCreateSRGB(
                        boost_rgb[0],
                        boost_rgb[1],
                        boost_rgb[2],
                        1.0,
                    )
                )
            except ImportError:
                pass
            has_mask = getattr(self, "_boost_mask_layer", None) is not None
            if boost_opacity > 0.01 and has_mask:
                boost_layer.setHidden_(False)
                boost_layer.setOpacity_(boost_opacity)
            else:
                boost_layer.setHidden_(True)
                boost_layer.setOpacity_(0.0)
        elif boost_layer is not None:
            boost_layer.setHidden_(True)
        # Update text punch-through mask (if active)
        if getattr(self, "_text_punchthrough", False) and not materialization_owns_fill:
            self._update_punchthrough_mask()
        # Cancel spring: warm amber tint over the overlay shape.
        if hasattr(self, '_spring_tint_layer') and self._spring_tint_layer is not None:
            if materialization_owns_fill:
                self._spring_tint_layer.setOpacity_(0.0)
            elif spring > 0.01:
                from Quartz import CGColorCreateSRGB
                # Warm golden-amber tint — visible, thermal, not alarming
                cg_color = CGColorCreateSRGB(0.55, 0.38, 0.05, 1.0)
                self._spring_tint_layer.setBackgroundColor_(cg_color)
                if hasattr(self._spring_tint_layer, "setHidden_"):
                    self._spring_tint_layer.setHidden_(False)
                spring_scale = (
                    _COMMAND_BACKDROP_OPTICAL_SHELL_SPRING_OPACITY_SCALE
                    if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED
                    else 0.5
                )
                if _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_REVEAL:
                    spring_scale = 0.0
                self._spring_tint_layer.setOpacity_(spring_scale * spring)
            else:
                self._spring_tint_layer.setOpacity_(0.0)
                if hasattr(self._spring_tint_layer, "setHidden_"):
                    self._spring_tint_layer.setHidden_(True)
        metrics = getattr(self, "_metrics", None)
        if metrics is not None:
            present_end = time.perf_counter()
            metrics.record_presented_frame(
                elapsed_ms=(present_end - present_start) * 1000.0,
                now=present_end,
            )

    def lingerDone_(self, timer) -> None:
        """Linger period over — fade out."""
        self._linger_timer = None
        if self._visible and not self._streaming:
            self.hide()

    # ── fade helpers ────────────────────────────────────────

    def _start_fade_out(self) -> None:
        self._cancel_fade()
        self._cancel_pulse()
        compositor = getattr(self, "_fullscreen_compositor", None)
        shell_config = self._display_local_optical_shell_config()
        fade_duration = _FADE_OUT_S
        if compositor is not None and shell_config is not None:
            try:
                self._start_materialization_animation(shell_config, direction=-1)
                fade_duration = _OPTICAL_MATERIALIZATION_DISMISS_TOTAL_S
            except Exception:
                logger.debug("Failed to start command materialization dismissal", exc_info=True)
        self._fade_step = 0
        self._fade_from = self._window.alphaValue() if self._window else 1.0
        self._fade_direction = -1
        interval = fade_duration / _FADE_STEPS
        self._fade_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            interval, self, "fadeStep:", None, True
        )

    def _cancel_fade(self) -> None:
        if self._fade_timer is not None:
            self._fade_timer.invalidate()
            self._fade_timer = None

    def _cancel_pulse(self) -> None:
        if self._pulse_timer is not None:
            self._pulse_timer.invalidate()
            self._pulse_timer = None

    def _start_pulse_timer(self) -> None:
        self._cancel_pulse()
        self._response_chroma_active = True
        self._pulse_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / _PULSE_HZ, self, "pulseStep:", None, True
        )

    def _cancel_linger(self) -> None:
        if self._linger_timer is not None:
            self._linger_timer.invalidate()
            self._linger_timer = None

    def _cancel_brightness_sampling(self) -> None:
        if getattr(self, "_brightness_timer", None) is not None:
            self._brightness_timer.invalidate()
            self._brightness_timer = None

    def _cancel_backdrop_refresh(self) -> None:
        if getattr(self, "_backdrop_timer", None) is not None:
            self._backdrop_timer.invalidate()
            self._backdrop_timer = None

    def _cancel_visual_start(self) -> None:
        timer = getattr(self, "_visual_start_timer", None)
        if timer is not None:
            timer.invalidate()
            self._visual_start_timer = None

    def _cancel_visual_ready_start(self) -> None:
        timer = getattr(self, "_visual_ready_timer", None)
        if timer is not None:
            timer.invalidate()
            self._visual_ready_timer = None

    def _cancel_materialization_animation(self) -> None:
        timer = getattr(self, "_materialization_timer", None)
        if timer is not None:
            timer.invalidate()
            self._materialization_timer = None
        self._stop_dismiss_seam_compositor()
        self._stop_dismiss_radial_pucker_compositor()

    def _cancel_dismiss_pucker_tail_animation(self) -> None:
        timer = getattr(self, "_pucker_tail_timer", None)
        if timer is not None:
            timer.invalidate()
            self._pucker_tail_timer = None
        self._pucker_tail_progress_offset = 0.0
        self._stop_dismiss_radial_pucker_compositor()

    def _cancel_dismiss_animation(self) -> None:
        if self._cancel_timer_anim is not None:
            self._cancel_timer_anim.invalidate()
            self._cancel_timer_anim = None

    def _cancel_entrance_pop(self) -> None:
        if getattr(self, "_pop_timer", None) is not None:
            self._pop_timer.invalidate()
            self._pop_timer = None

    def _cancel_all_timers(self) -> None:
        self._cancel_dismiss_animation()
        self._cancel_entrance_pop()
        self._cancel_fade()
        self._cancel_pulse()
        self._cancel_linger()
        self._cancel_brightness_sampling()
        self._cancel_backdrop_refresh()
        self._cancel_visual_start()
        self._cancel_visual_ready_start()
        self._cancel_materialization_animation()
        self._stop_dismiss_seam_compositor()
        self._cancel_dismiss_pucker_tail_animation()
        self._stop_thinking_timer()

    def _start_entrance_animation(self) -> None:
        """Start the visible entrance once first-paint dependencies are ready."""
        self._cancel_entrance_pop()
        self._cancel_fade()

        # Entrance pop — start slightly oversized, ease back to 1.0.
        # Runs concurrently with the fade-in for a subtle "I just arrived" feel.
        self._set_overlay_scale(_ENTRANCE_POP_SCALE)
        self._pop_step = 0
        self._pop_steps = max(1, int(_ENTRANCE_POP_S * _DISMISS_ANIM_FPS))
        self._pop_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / _DISMISS_ANIM_FPS, self, "_entrancePopStep:", None, True
        )

        # Fade in. Pulse remains deferred until fade completion so it cannot
        # compete with first paint.
        self._fade_step = 0
        self._fade_from = 0.0
        self._fade_direction = 1
        interval = _FADE_IN_S / _FADE_STEPS
        self._fade_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            interval, self, "fadeStep:", None, True
        )

    def _start_materialization_animation(self, final_shell_config: dict, *, direction: int = 1) -> None:
        """Animate the compositor geometry from/to a pressure slit."""
        self._cancel_materialization_animation()
        self._cancel_dismiss_pucker_tail_animation()
        self._deferred_materialization_shell_config = None
        self._materialization_final_shell_config = dict(final_shell_config)
        self._materialization_direction = 1 if direction >= 0 else -1
        self._materialization_progress = 0.0 if self._materialization_direction > 0 else 1.0
        self._materialization_started_at = time.perf_counter()
        self._clear_punchthrough_mask_for_materialization()
        if self._materialization_direction < 0:
            self._apply_materialization_fill_state(self._materialization_progress)
        compositor = getattr(self, "_fullscreen_compositor", None)
        if compositor is not None:
            try:
                compositor.update_shell_config(
                    _materialized_optical_shell_config(
                        self._materialization_final_shell_config,
                        self._materialization_progress,
                    )
                )
            except Exception:
                logger.debug("Failed to seed command materialization shell", exc_info=True)
        if self._materialization_direction > 0:
            self._apply_materialization_fill_state(self._materialization_progress)
        self._materialization_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / _DISMISS_ANIM_FPS,
            self,
            "materializationStep:",
            None,
            True,
        )
        _pin_timer_to_active_run_loop_modes(self._materialization_timer)

    def materializationStep_(self, timer) -> None:
        if getattr(self, "_materialization_timer", None) is not timer:
            return
        final_config = getattr(self, "_materialization_final_shell_config", None)
        compositor = getattr(self, "_fullscreen_compositor", None)
        if final_config is None or compositor is None:
            self._cancel_materialization_animation()
            self._materialization_progress = 1.0
            return
        elapsed = max(time.perf_counter() - getattr(self, "_materialization_started_at", 0.0), 0.0)
        duration = (
            _OPTICAL_MATERIALIZATION_S
            if getattr(self, "_materialization_direction", 1) > 0
            else _OPTICAL_MATERIALIZATION_DISMISS_S
        )
        raw = _clamp01(elapsed / duration)
        progress = raw if getattr(self, "_materialization_direction", 1) > 0 else 1.0 - raw
        self._materialization_progress = progress
        self._apply_materialization_fill_state(progress)
        try:
            shell_config = _materialized_optical_shell_config(final_config, progress)
            compositor_updates: list[tuple[object, dict]] = []
            if getattr(self, "_materialization_direction", 1) < 0:
                if progress <= _OPTICAL_MATERIALIZATION_PUCKER_PREARM_START_PROGRESS:
                    self._set_layer_hidden_without_actions(
                        getattr(self, "_backdrop_layer", None),
                        True,
                    )
                    pucker_progress = _dismiss_pucker_tail_progress_for_close_progress(
                        progress
                    )
                    self._pucker_tail_progress_offset = _clamp01(pucker_progress)
                    radial_compositor = self._ensure_dismiss_radial_pucker_compositor()
                    if radial_compositor is not None:
                        compositor_updates.append(
                            (
                                radial_compositor,
                                _dismiss_radial_pucker_shell_config(
                                    final_config,
                                    pucker_progress,
                                ),
                            )
                        )
                if progress <= _OPTICAL_MATERIALIZATION_SEAM_OVERLAP_START_PROGRESS:
                    seam_compositor = self._ensure_dismiss_seam_compositor()
                    if seam_compositor is not None:
                        compositor_updates.append(
                            (
                                seam_compositor,
                                _dismiss_seam_latch_shell_config(final_config, progress),
                            )
                        )
                else:
                    self._stop_dismiss_seam_compositor()
            compositor_updates.append((compositor, shell_config))
            if not self._publish_shared_compositor_configs(compositor_updates):
                self._publish_individual_compositor_configs(compositor_updates)
        except Exception:
            logger.debug("Failed to update command materialization shell", exc_info=True)
        if raw >= 1.0:
            self._cancel_materialization_animation()
            if getattr(self, "_materialization_direction", 1) > 0:
                self._materialization_progress = 1.0
                self._apply_materialization_fill_state(1.0)
                if getattr(self, "_text_punchthrough", False):
                    self._refresh_punchthrough_mask_if_needed()
                if (
                    getattr(self, "_visible", False)
                    and not getattr(self, "_recording_load_shed", False)
                ):
                    self._start_pulse_timer()
            else:
                self._materialization_progress = 0.0
                self._apply_materialization_fill_state(0.0)
                self._start_dismiss_pucker_tail_animation(
                    final_config,
                    initial_progress=_dismiss_pucker_tail_progress_for_close_progress(
                        0.0
                    ),
                )

    def _start_dismiss_pucker_tail_animation(
        self,
        final_shell_config: dict,
        *,
        initial_progress: float = 0.0,
    ) -> None:
        """Run the post-close inverse pucker after the dismiss slit shuts."""
        self._cancel_dismiss_pucker_tail_animation()
        self._stop_dismiss_seam_compositor()
        self._hide_local_shell_layers_for_pucker_tail()
        self._pucker_tail_shell_config = dict(final_shell_config)
        self._pucker_tail_progress_offset = _clamp01(initial_progress)
        self._pucker_tail_started_at = time.perf_counter()
        compositor = getattr(self, "_fullscreen_compositor", None)
        if compositor is not None:
            try:
                compositor.update_shell_config(
                    _dismiss_pucker_shell_config(
                        self._pucker_tail_shell_config,
                        self._pucker_tail_progress_offset,
                    )
                )
            except Exception:
                logger.debug("Failed to seed command dismiss pucker", exc_info=True)
        self._pucker_tail_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / _DISMISS_ANIM_FPS,
            self,
            "dismissPuckerTailStep:",
            None,
            True,
        )
        _pin_timer_to_active_run_loop_modes(self._pucker_tail_timer)

    def dismissPuckerTailStep_(self, timer) -> None:
        if getattr(self, "_pucker_tail_timer", None) is not timer:
            return
        shell_config = getattr(self, "_pucker_tail_shell_config", None)
        compositor = getattr(self, "_fullscreen_compositor", None)
        if shell_config is None or compositor is None:
            self._cancel_dismiss_pucker_tail_animation()
            return
        elapsed = max(time.perf_counter() - getattr(self, "_pucker_tail_started_at", 0.0), 0.0)
        progress = _clamp01(
            getattr(self, "_pucker_tail_progress_offset", 0.0)
            + elapsed / max(_OPTICAL_MATERIALIZATION_PUCKER_TAIL_S, 1e-6)
        )
        try:
            compositor.update_shell_config(
                _dismiss_pucker_shell_config(shell_config, progress)
            )
        except Exception:
            logger.debug("Failed to update command dismiss pucker", exc_info=True)
        if progress >= 1.0:
            self._cancel_dismiss_pucker_tail_animation()

    def _set_materialization_layer_scale(
        self,
        layer,
        *,
        total_w: float,
        total_h: float,
        height_frac: float,
    ) -> None:
        """Scale shell body layers from their center without moving frames."""
        if layer is None:
            return
        transaction = CATransaction
        try:
            if transaction is not None:
                transaction.begin()
                transaction.setDisableActions_(True)
            if hasattr(layer, "setAnchorPoint_"):
                layer.setAnchorPoint_((0.5, 0.5))
            if hasattr(layer, "setBounds_"):
                layer.setBounds_(((0, 0), (total_w, total_h)))
            if hasattr(layer, "setPosition_"):
                layer.setPosition_((total_w * 0.5, total_h * 0.5))
            if hasattr(layer, "setValue_forKeyPath_"):
                layer.setValue_forKeyPath_(
                    max(_clamp01(height_frac), 0.001),
                    "transform.scale.y",
                )
        except Exception:
            logger.debug("Failed to update command materialization layer scale", exc_info=True)
        finally:
            if transaction is not None:
                try:
                    transaction.commit()
                except Exception:
                    logger.debug("Failed to commit command materialization transaction", exc_info=True)

    def _set_layer_opacity_without_actions(self, layer, opacity: float) -> None:
        if layer is None or not hasattr(layer, "setOpacity_"):
            return
        transaction = CATransaction
        try:
            if transaction is not None:
                transaction.begin()
                transaction.setDisableActions_(True)
            layer.setOpacity_(opacity)
        except Exception:
            logger.debug("Failed to update command materialization layer opacity", exc_info=True)
        finally:
            if transaction is not None:
                try:
                    transaction.commit()
                except Exception:
                    logger.debug("Failed to commit command materialization opacity transaction", exc_info=True)

    def _set_layer_contents_without_actions(self, layer, contents) -> None:
        if layer is None or contents is None or not hasattr(layer, "setContents_"):
            return
        transaction = CATransaction
        try:
            if transaction is not None:
                transaction.begin()
                transaction.setDisableActions_(True)
            layer.setContents_(contents)
        except Exception:
            logger.debug("Failed to update command materialization layer contents", exc_info=True)
        finally:
            if transaction is not None:
                try:
                    transaction.commit()
                except Exception:
                    logger.debug("Failed to commit command materialization contents transaction", exc_info=True)

    def _set_layer_hidden_without_actions(self, layer, hidden: bool) -> None:
        if layer is None or not hasattr(layer, "setHidden_"):
            return
        transaction = CATransaction
        try:
            if transaction is not None:
                transaction.begin()
                transaction.setDisableActions_(True)
            layer.setHidden_(hidden)
        except Exception:
            logger.debug("Failed to update command materialization hidden state", exc_info=True)
        finally:
            if transaction is not None:
                try:
                    transaction.commit()
                except Exception:
                    logger.debug("Failed to commit command materialization hidden transaction", exc_info=True)

    def _hide_local_shell_layers_for_pucker_tail(self) -> None:
        """Keep old rectangular CA layers out of the compositor-only scar."""
        for layer_name in (
            "_backdrop_layer",
            "_fill_layer",
            "_boost_layer",
            "_spring_tint_layer",
        ):
            layer = getattr(self, layer_name, None)
            if layer_name != "_backdrop_layer":
                self._set_layer_opacity_without_actions(layer, 0.0)
            self._set_layer_hidden_without_actions(layer, True)

    def _apply_materialization_fill_state(self, progress: float) -> None:
        content = getattr(self, "_content_view", None)
        if content is None:
            return
        state = _materialization_fill_state(progress)
        hide_material_layers = False
        if getattr(self, "_materialization_direction", 1) < 0:
            # The compositor owns displacement; the CA/SDF fill owns the
            # visible material body until the seam sidecar takes over. Hiding
            # earlier amputates the center while leaving warped end-caps alive.
            state = _dismiss_materialization_fill_state(progress)
            hide_material_layers = state["opacity"] <= 0.0
        try:
            content_frame = content.frame()
            f = _OPTICAL_SHELL_FEATHER if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED else _OUTER_FEATHER
            total_w = float(content_frame.size.width) + 2 * f
            total_h = float(content_frame.size.height) + 2 * f
        except Exception:
            return
        for layer_name in ("_fill_layer", "_boost_layer", "_spring_tint_layer"):
            layer = getattr(self, layer_name, None)
            self._set_materialization_layer_scale(
                layer,
                total_w=total_w,
                total_h=total_h,
                height_frac=state["height_frac"],
            )
        if getattr(self, "_materialization_direction", 1) < 0:
            for layer_name in ("_boost_layer", "_spring_tint_layer"):
                layer = getattr(self, layer_name, None)
                self._set_layer_opacity_without_actions(layer, 0.0)
                self._set_layer_hidden_without_actions(layer, True)
        fill = getattr(self, "_fill_layer", None)
        if getattr(self, "_materialization_direction", 1) < 0:
            self._set_layer_contents_without_actions(
                fill,
                getattr(self, "_dismiss_fill_image", None),
            )
        else:
            self._set_layer_contents_without_actions(
                fill,
                getattr(self, "_fill_image", None),
            )
        self._set_layer_opacity_without_actions(fill, state["opacity"])
        if hide_material_layers:
            for layer_name in ("_boost_layer", "_spring_tint_layer"):
                self._set_layer_opacity_without_actions(
                    getattr(self, layer_name, None),
                    0.0,
                )

    def _schedule_visual_start(self) -> None:
        """Defer compositor startup so first paint and text do not block."""
        self._cancel_visual_start()
        self._visual_start_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            _COMMAND_VISUAL_START_DELAY_S,
            self,
            "visualStart:",
            None,
            False,
        )
        _pin_timer_to_active_run_loop_modes(self._visual_start_timer)

    def visualStart_(self, timer) -> None:
        if getattr(self, "_visual_start_timer", None) is timer:
            self._visual_start_timer = None
        if not getattr(self, "_visible", False):
            return
        if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
            self._start_fullscreen_compositor()
            if getattr(self, "_fullscreen_compositor", None) is not None:
                self._refresh_punchthrough_mask_if_needed()
        if getattr(self, "_fullscreen_compositor", None) is None:
            self._enable_text_punchthrough(False)
            self._start_backdrop_refresh_timer()

    def _schedule_visual_ready_start(self) -> None:
        """Hold alpha-zero entrance until the optical compositor is coherent."""
        self._cancel_visual_ready_start()
        self._visual_ready_brightness_synced = False
        if self._optical_compositor_has_presented():
            self._sync_optical_compositor_brightness(hide_stale_fill=True)
            self._visual_ready_brightness_synced = True
            if self._optical_entrance_ready():
                self._start_entrance_animation()
                return
        self._visual_ready_wait_started_at = time.perf_counter()
        self._visual_ready_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            _OPTICAL_ENTRANCE_READY_POLL_S,
            self,
            "visualReadyStep:",
            None,
            True,
        )
        _pin_timer_to_active_run_loop_modes(self._visual_ready_timer)

    def visualReadyStep_(self, timer) -> None:
        if getattr(self, "_visual_ready_timer", None) is not timer:
            return
        if not getattr(self, "_visible", False):
            self._cancel_visual_ready_start()
            return
        elapsed = time.perf_counter() - getattr(
            self,
            "_visual_ready_wait_started_at",
            time.perf_counter(),
        )
        compositor_ready = self._optical_compositor_has_presented()
        if compositor_ready and not getattr(self, "_visual_ready_brightness_synced", False):
            self._sync_optical_compositor_brightness(hide_stale_fill=True)
            self._visual_ready_brightness_synced = True
        if not compositor_ready and elapsed < _OPTICAL_ENTRANCE_READY_TIMEOUT_S:
            return
        if not self._optical_entrance_ready():
            return
        self._cancel_visual_ready_start()
        self._start_entrance_animation()

    def _optical_entrance_ready(self) -> bool:
        return (
            self._optical_compositor_has_presented()
            and self._optical_fill_ready()
            and getattr(self, "_materialization_progress", 1.0)
            >= _OPTICAL_MATERIALIZATION_BODY_READY
        )

    def _optical_compositor_has_presented(self) -> bool:
        compositor = getattr(self, "_fullscreen_compositor", None)
        if compositor is None:
            return False
        count = getattr(compositor, "presented_count", None)
        if callable(count):
            try:
                count = count()
            except Exception:
                count = None
        if not isinstance(count, numbers.Number) or count <= 0:
            return False
        return True

    def _optical_fill_ready(self) -> bool:
        return getattr(self, "_fill_hidden_until_signature", None) is None

    def _sync_optical_compositor_brightness(self, *, hide_stale_fill: bool = False) -> None:
        compositor = getattr(self, "_fullscreen_compositor", None)
        if compositor is None:
            return
        refresher = getattr(compositor, "refresh_brightness", None)
        if callable(refresher):
            try:
                refresher()
            except Exception:
                logger.debug("Failed to refresh compositor brightness before entrance", exc_info=True)
        try:
            brightness = _clamp01(float(getattr(compositor, "sampled_brightness", self._brightness)))
        except Exception:
            return
        self._brightness = brightness
        self._brightness_target = brightness
        self._brightness_sample_tick = 0
        suppress_stale_fill = getattr(
            self,
            "_suppress_stale_fill_until_ready",
            False,
        )
        if hide_stale_fill:
            self._suppress_stale_fill_until_ready = True
        try:
            self._apply_surface_theme()
        finally:
            self._suppress_stale_fill_until_ready = suppress_stale_fill

    def _seed_brightness_from_optical_compositor(self) -> float | None:
        """Pull one compositor-side background sample before materialization."""
        compositor = getattr(self, "_fullscreen_compositor", None)
        if compositor is None:
            return None
        refresher = getattr(compositor, "refresh_brightness", None)
        if callable(refresher):
            try:
                refresher()
            except Exception:
                logger.debug("Failed to refresh compositor brightness seed", exc_info=True)
        try:
            raw_brightness = getattr(compositor, "sampled_brightness", self._brightness)
        except Exception:
            return None
        if not isinstance(raw_brightness, numbers.Number):
            return None
        brightness = _clamp01(float(raw_brightness))
        self._brightness = brightness
        self._brightness_target = brightness
        self._brightness_sample_tick = 0
        return brightness

    def _set_overlay_scale(self, scale: float) -> None:
        if self._wrapper_view is None:
            return
        layer = self._wrapper_view.layer() if hasattr(self._wrapper_view, "layer") else None
        if layer is None:
            return
        try:
            # Re-assert center anchor so scale is always symmetric, even
            # after window geometry changes that may shift the layer bounds.
            bounds = layer.bounds()
            layer.setAnchorPoint_((0.5, 0.5))
            layer.setPosition_((
                bounds.origin.x + bounds.size.width / 2,
                bounds.origin.y + bounds.size.height / 2,
            ))
            layer.setValue_forKeyPath_(scale, "transform.scale")
        except Exception:
            logger.exception("Failed to update command overlay scale")

    # ── thinking timer ──────────────────────────────────────

    def _start_thinking_timer(self, *, reset: bool = True) -> None:
        """Start or resume the thinking counter."""
        if reset:
            self._thinking_seconds = 0.0
            self._thinking_inverted = False
        if self._thinking_label is not None:
            self._thinking_label.setHidden_(False)
            if self._tool_mode:
                self._thinking_label.setStringValue_("tool…")
            else:
                self._thinking_label.setStringValue_(f"{self._thinking_seconds:.1f}s")
            self._apply_thinking_label_theme()
        logger.info("Thinking timer %s", "started" if reset else "resumed")
        self._thinking_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.1, self, "thinkingTick:", None, True
        )

    def invert_thinking_timer(self) -> None:
        """Switch from glowing number to negative-space cutout mode.

        Called when the first content token arrives — the visual flip
        signals that the model has transitioned from thinking to speaking.
        """
        self._thinking_inverted = True
        self._apply_thinking_label_theme()

    def set_narrator_summary(self, summary: str) -> None:
        """Update the narrator summary line displayed during thinking."""
        if self._narrator_label is not None:
            self._narrator_label.setStringValue_(summary)
            self._narrator_label.setHidden_(False)
            self._sync_narrator_visibility()
            self._apply_narrator_theme()

    def set_narrator_shimmer(self, active: bool) -> None:
        """Toggle the narrator label's higher-contrast shimmer state."""
        self._narrator_shimmer_active = bool(active)
        self._apply_narrator_theme()

    def set_thinking_collapsed(self, text: str) -> None:
        """Inject or append to a collapsed thinking summary in the text view.

        "Thought for Xs" starts a new collapsed entry.
        " · topic" appends to the current entry — but only if no response
        tokens have started streaming yet. Late topics are silently dropped
        to avoid corrupting the response text.
        """
        if self._text_view is None or not self._visible:
            return
        is_topic_append = text.startswith(" · ")
        # Track with newline separators for set_response_text rebuild.
        # Display needs its own prefix because live append is incremental.
        if is_topic_append:
            display_text = text
            if self._collapsed_text:
                self._collapsed_text += text
            else:
                self._collapsed_text = text.lstrip(" ·")
        elif self._collapsed_text:
            display_text = "\n" + text
            self._collapsed_text += "\n" + text
        else:
            display_text = (
                "\n\n" + text
                if self._utterance_text or self._response_text
                else text
            )
            self._collapsed_text += text
        # If the final response has already been set, re-rebuild so the
        # topic lands in the right position (after "Thought for Xs",
        # before the response text).
        if is_topic_append and self._response_text:
            self.set_response_text(self._response_text)
            return
        # Append to live text view
        collapsed_str = self._make_collapsed_attributed(display_text)
        self._text_view.textStorage().appendAttributedString_(collapsed_str)
        self._update_layout()
        self._refresh_punchthrough_mask_if_needed()

    def _apply_narrator_theme(self) -> None:
        """Match narrator label color to user utterance style."""
        if self._narrator_label is None or self._narrator_label.isHidden():
            return
        user_r, user_g, user_b = _user_text_color_for_brightness(self._brightness)
        alpha = 0.95 if getattr(self, "_narrator_shimmer_active", False) else 0.90
        self._narrator_label.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(user_r, user_g, user_b, alpha)
        )

    def _apply_agent_shell_chrome_theme(self) -> None:
        """Keep Agent Shell metadata readable on the adaptive surface."""
        labels = (
            getattr(self, "_agent_shell_header_label", None),
            getattr(self, "_agent_shell_footer_label", None),
        )
        fg_r, fg_g, fg_b = _assistant_foreground_color_for_brightness(self._brightness)
        for label in labels:
            if label is None or label.isHidden():
                continue
            label.setTextColor_(
                NSColor.colorWithSRGBRed_green_blue_alpha_(
                    fg_r,
                    fg_g,
                    fg_b,
                    _AGENT_SHELL_CHROME_ALPHA,
                )
            )

    def _sync_narrator_visibility(self, text_height: float | None = None) -> None:
        """Hide the fixed narrator label before it overlaps transcript text."""
        if self._narrator_label is None or self._narrator_label.isHidden():
            return
        if self._response_text:
            self._hide_narrator()
            return
        if text_height is None and self._text_view is not None:
            try:
                layout = self._text_view.layoutManager()
                container = self._text_view.textContainer()
                if layout and container:
                    layout.ensureLayoutForTextContainer_(container)
                    text_height = layout.usedRectForTextContainer_(container).size.height
            except Exception:
                text_height = None
        if text_height is not None and text_height > _NARRATOR_OVERLAP_TEXT_HEIGHT:
            self._hide_narrator()

    def _hide_narrator(self) -> None:
        """Hide the narrator summary label."""
        if self._narrator_label is not None:
            self._narrator_label.setHidden_(True)
            self._narrator_label.setStringValue_("")

    def _stop_thinking_timer(self) -> None:
        """Stop and fade the thinking counter."""
        if self._thinking_timer is not None:
            self._thinking_timer.invalidate()
            self._thinking_timer = None
        if self._thinking_label is not None:
            self._thinking_label.setHidden_(True)
        self._hide_narrator()

    def thinkingTick_(self, timer) -> None:
        """Update the thinking counter every 100ms."""
        self._thinking_seconds += 0.1
        if self._thinking_label is not None and not self._thinking_label.isHidden():
            if self._tool_mode: self._thinking_label.setStringValue_("tool…")
            else: self._thinking_label.setStringValue_(f"{self._thinking_seconds:.1f}s")

    # ── ridge masks ────────────────────────────────────────

    def _hide_stale_fill_until_ready(self, signature) -> None:
        """Prevent an old fill image from flashing during first paint."""
        self._fill_hidden_until_signature = signature
        fill_layer = getattr(self, "_fill_layer", None)
        if fill_layer is not None and hasattr(fill_layer, "setHidden_"):
            fill_layer.setHidden_(True)

    def _unhide_fill_if_ready(self, signature) -> None:
        if getattr(self, "_fill_hidden_until_signature", None) != signature:
            return
        fill_layer = getattr(self, "_fill_layer", None)
        if fill_layer is not None and hasattr(fill_layer, "setHidden_"):
            fill_layer.setHidden_(False)
        self._fill_hidden_until_signature = None
        self._start_deferred_materialization_if_ready()

    def _start_deferred_materialization_if_ready(self) -> None:
        if not self._optical_fill_ready():
            return
        if getattr(self, "_pending_fill_image_signature", None) is not None:
            return
        if getattr(self, "_materialization_timer", None) is not None:
            return
        if getattr(self, "_fullscreen_compositor", None) is None:
            return
        shell_config = getattr(self, "_deferred_materialization_shell_config", None)
        if shell_config is None:
            return
        self._deferred_materialization_shell_config = None
        self._start_materialization_animation(dict(shell_config))

    def _apply_ridge_masks(self, width: float, height: float) -> None:
        """Compute SDF and apply ridge mask + build fill image.

        The SDF and fill-alpha field are cached by geometry (width, height).
        When only brightness changes (same dimensions), we skip the expensive
        numpy SDF computation and just rebuild the colored image.
        """
        from .overlay import (
            _RIDGE_FALLOFF, _RIDGE_POWER,
            _overlay_rounded_rect_sdf, _ridge_alpha, _interior_fill_alpha,
            _fill_field_to_image, _BG_COLOR_DARK,
        )
        f = _OPTICAL_SHELL_FEATHER if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED else _OUTER_FEATHER
        scale = getattr(self, '_ridge_scale', 2.0)
        total_w = width + 2 * f
        total_h = height + 2 * f
        fill_overscan = (
            _COMMAND_MATERIAL_FILL_OVERSCAN_POINTS
            if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED
            else 0.0
        )
        fill_body_w = width + 2 * fill_overscan
        fill_body_h = height + 2 * fill_overscan

        _has_compositor = getattr(self, "_fullscreen_compositor", None) is not None
        _b = getattr(self, '_brightness', 0.0)
        _b_rounded = round(_b * 50) / 50  # 0.02 steps
        geom_key = (
            width,
            height,
            fill_body_w,
            fill_body_h,
            scale,
            _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED,
        )
        appearance_key = (
            round(float(total_w), 3),
            round(float(total_h), 3),
            round(float(scale), 3),
            _b_rounded,
            bool(_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED),
            bool(_has_compositor),
        )
        self._desired_fill_image_signature = appearance_key
        if (
            getattr(self, "_fill_image_signature", None) == appearance_key
            and getattr(self, "_fill_payload", None) is not None
            and (
                getattr(self, "_spring_tint_layer", None) is None
                or getattr(self, "_spring_tint_mask_signature", None) == appearance_key
            )
        ):
            self._unhide_fill_if_ready(appearance_key)
            return
        if getattr(self, "_suppress_stale_fill_until_ready", False):
            self._hide_stale_fill_until_ready(appearance_key)
        if hasattr(self, '_fill_layer') and self._fill_layer is not None:
            self._fill_layer.setFrame_(((0, 0), (total_w, total_h)))
            if hasattr(self._fill_layer, "setContentsScale_"):
                self._fill_layer.setContentsScale_(scale)
        if hasattr(self, '_spring_tint_layer') and self._spring_tint_layer is not None:
            self._spring_tint_layer.setFrame_(((0, 0), (total_w, total_h)))
        if self._materialization_owns_fill_layers():
            self._apply_materialization_fill_state(
                getattr(self, "_materialization_progress", 1.0)
            )

        pending = getattr(self, "_pending_fill_image_signature", None)
        if pending is not None:
            if pending != appearance_key:
                self._queued_fill_request = (width, height)
                if getattr(self, "_fill_hidden_until_signature", None) == pending:
                    self._fill_hidden_until_signature = appearance_key
            return

        self._pending_fill_image_signature = appearance_key
        geom_cache_hit = getattr(self, '_sdf_geom_key', None) == geom_key
        cached_fallback_alpha = (
            getattr(self, "_sdf_fallback_alpha", None)
            if geom_cache_hit
            else None
        )
        cached_raw_interior = (
            getattr(self, "_sdf_raw_interior", None)
            if geom_cache_hit
            else None
        )
        cached_edge_ridge = (
            getattr(self, "_sdf_edge_ridge", None)
            if geom_cache_hit
            else None
        )
        cached_inside_mask = (
            getattr(self, "_sdf_inside_mask", None)
            if geom_cache_hit
            else None
        )
        cached_ext_exterior = (
            getattr(self, "_sdf_ext_exterior", None)
            if geom_cache_hit
            else None
        )

        def build() -> None:
            try:
                from .overlay import _glow_fill_alpha
                import numpy as np

                fallback_alpha = cached_fallback_alpha
                raw_interior = cached_raw_interior
                edge_ridge = cached_edge_ridge
                inside_mask = cached_inside_mask
                ext_exterior = cached_ext_exterior
                if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED and (
                    fallback_alpha is None
                    or raw_interior is None
                    or edge_ridge is None
                    or inside_mask is None
                    or ext_exterior is None
                ):
                    sdf = _stadium_signed_distance_field(
                        total_w,
                        total_h,
                        fill_body_w,
                        fill_body_h,
                        _optical_shell_body_corner_radius(height) + fill_overscan,
                        scale,
                    )
                    inside_d = np.maximum(-sdf, 0.0)
                    edge_ridge = np.exp(-((inside_d / (1.5 * scale)) ** 2.0)).astype(np.float32)
                    raw_interior = np.exp(np.sqrt(
                        np.abs(sdf) / max(2.0 * scale, 1e-6)
                    ) * -1.0).astype(np.float32)
                    inside_mask = (sdf <= 0.0)
                    feather_px = _OPTICAL_SHELL_FEATHER * scale
                    ext_d = np.maximum(sdf, 0.0)
                    sigma = feather_px / 3.0
                    exterior = np.exp(-0.5 * (ext_d / sigma) ** 2.0).astype(np.float32)
                    ext_t = np.clip(ext_d / feather_px, 0.0, 1.0)
                    ext_exterior = (exterior * (0.13 * np.exp(-((ext_t / 0.30) ** 1.5)) + 0.007)).astype(np.float32)
                    fallback_alpha = _glow_fill_alpha(sdf, width=2.5 * scale)
                elif not _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED and fallback_alpha is None:
                    raw_interior = edge_ridge = inside_mask = ext_exterior = None
                    sdf = _overlay_rounded_rect_sdf(
                        total_w, total_h, fill_body_w, fill_body_h,
                        _OVERLAY_CORNER_RADIUS, scale,
                    )
                    fallback_alpha = _glow_fill_alpha(sdf, width=2.5 * scale)
                elif not _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
                    raw_interior = edge_ridge = inside_mask = ext_exterior = None

                cached_fill_alpha = None
                sdf_appearance_b = -1.0
                if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED and _has_compositor:
                    _ifloor = _lerp(0.60, 0.85, _clamp01(_b))
                    interior = (_ifloor + (1.0 - _ifloor) * raw_interior)
                    interior = np.clip(
                        interior + edge_ridge * 0.50,
                        0.0, 1.0,
                    ).astype(np.float32)
                    interior = np.clip(
                        interior * _compositor_fill_alpha_multiplier_for_brightness(_b),
                        0.0,
                        1.0,
                    ).astype(np.float32)
                    fill_alpha = np.where(inside_mask, interior, ext_exterior)
                    cached_fill_alpha = fill_alpha
                    sdf_appearance_b = _b_rounded
                else:
                    fill_alpha = fallback_alpha

                result = {
                    "signature": appearance_key,
                    "total_w": total_w,
                    "total_h": total_h,
                    "scale": scale,
                    "geom_key": geom_key,
                    "fallback_alpha": fallback_alpha,
                    "raw_interior": raw_interior,
                    "edge_ridge": edge_ridge,
                    "inside_mask": inside_mask,
                    "ext_exterior": ext_exterior,
                    "cached_fill_alpha": cached_fill_alpha,
                    "sdf_appearance_b": sdf_appearance_b,
                    "has_compositor": _has_compositor,
                }
                if _has_compositor:
                    bg_r, bg_g, bg_b = _compositor_fill_color_for_brightness(_b)
                else:
                    bg_r, bg_g, bg_b = _background_color_for_brightness(_b)
                try:
                    fill_image, payload = _fill_field_to_image(
                        fill_alpha,
                        int(bg_r * 255), int(bg_g * 255), int(bg_b * 255),
                    )
                    result["image"] = fill_image
                    result["payload"] = payload
                    if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED and _has_compositor:
                        dismiss_fill_alpha = np.where(inside_mask, interior, 0.0).astype(np.float32)
                        dismiss_image, dismiss_payload = _fill_field_to_image(
                            dismiss_fill_alpha,
                            int(bg_r * 255), int(bg_g * 255), int(bg_b * 255),
                        )
                        result["dismiss_image"] = dismiss_image
                        result["dismiss_payload"] = dismiss_payload
                except Exception as exc:
                    result["error"] = repr(exc)
            except Exception as exc:
                result = {"signature": appearance_key, "error": repr(exc)}
            _post_overlay_result_to_main(self, "fillImageReady:", result)

        _start_overlay_fill_worker(build)

    def fillImageReady_(self, payload: dict) -> None:
        signature = payload.get("signature")
        if getattr(self, "_pending_fill_image_signature", None) == signature:
            self._pending_fill_image_signature = None
        if signature != getattr(self, "_desired_fill_image_signature", None):
            queued = getattr(self, "_queued_fill_request", None)
            if queued is not None:
                self._queued_fill_request = None
                self._apply_ridge_masks(*queued)
            return
        if "geom_key" in payload:
            self._sdf_geom_key = payload.get("geom_key")
            self._sdf_fallback_alpha = payload.get("fallback_alpha")
            self._sdf_raw_interior = payload.get("raw_interior")
            self._sdf_edge_ridge = payload.get("edge_ridge")
            self._sdf_inside_mask = payload.get("inside_mask")
            self._sdf_ext_exterior = payload.get("ext_exterior")
            self._cached_fill_alpha = payload.get("cached_fill_alpha")
            self._sdf_appearance_b = payload.get("sdf_appearance_b", -1.0)
        error = payload.get("error")
        if error:
            logger.debug("Command overlay fill image generation failed: %s", error)
            self._unhide_fill_if_ready(signature)
            return
        self._fill_payload = payload.get("payload")
        self._dismiss_fill_payload = payload.get("dismiss_payload")
        self._fill_image_signature = signature

        total_w = payload["total_w"]
        total_h = payload["total_h"]
        scale = payload.get("scale", getattr(self, "_ridge_scale", 2.0))
        fill_image = payload.get("image")
        self._fill_image = fill_image
        self._dismiss_fill_image = payload.get("dismiss_image")
        if hasattr(self, '_fill_layer') and self._fill_layer is not None:
            self._fill_layer.setContents_(fill_image)
            self._fill_layer.setFrame_(((0, 0), (total_w, total_h)))
            if hasattr(self._fill_layer, "setContentsScale_"):
                self._fill_layer.setContentsScale_(scale)
            if hasattr(self._fill_layer, "setCompositingFilter_"):
                if payload.get("has_compositor"):
                    self._fill_layer.setCompositingFilter_(None)
                else:
                    self._fill_layer.setCompositingFilter_(
                        _fill_compositing_filter_for_brightness(getattr(self, "_brightness", 0.0))
                    )
            self._unhide_fill_if_ready(signature)
        if hasattr(self, '_spring_tint_layer') and self._spring_tint_layer is not None:
            self._spring_tint_layer.setFrame_(((0, 0), (total_w, total_h)))
            mask = CALayer.alloc().init()
            mask.setFrame_(((0, 0), (total_w, total_h)))
            mask.setContents_(fill_image)
            mask.setContentsGravity_("resize")
            self._spring_tint_layer.setMask_(mask)
            self._spring_tint_mask_signature = signature

        if self._materialization_owns_fill_layers():
            self._apply_materialization_fill_state(
                getattr(self, "_materialization_progress", 1.0)
            )

        queued = getattr(self, "_queued_fill_request", None)
        if queued is not None:
            self._queued_fill_request = None
            self._apply_ridge_masks(*queued)

    def _update_backdrop_capture_geometry(self):
        if self._window is None or self._content_view is None or self._screen is None:
            return None
        try:
            screen_frame = self._screen.frame()
            win_frame = self._window.frame()
            content_frame = self._content_view.frame()
        except Exception:
            return None

        capture_rect = _backdrop_capture_rect(
            screen_frame,
            win_frame,
            content_frame,
            getattr(self, "_backdrop_capture_overscan_points", _command_backdrop_capture_overscan_points()),
        )
        pixel_size = _backdrop_capture_pixel_size(
            capture_rect,
            getattr(self, "_ridge_scale", 1.0),
        )
        self._backdrop_capture_rect = capture_rect
        self._backdrop_capture_pixel_size = pixel_size
        return capture_rect, pixel_size

    def _update_backdrop_mask(self, width: float, height: float):
        if self._backdrop_layer is None:
            return

        overscan = getattr(self, "_backdrop_capture_overscan_points", _command_backdrop_capture_overscan_points())
        scale = getattr(self, "_ridge_scale", 2.0)
        mask_width_multiplier = getattr(
            self,
            "_backdrop_mask_width_multiplier",
            _COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER,
        )
        signature = (
            round(float(width), 3),
            round(float(height), 3),
            round(float(overscan), 3),
            round(float(scale), 3),
            round(float(mask_width_multiplier), 6),
            bool(_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED),
        )
        cached_signature = getattr(self, "_backdrop_mask_signature", None)
        cached_mask = getattr(self, "_backdrop_mask_layer", None)
        if signature == cached_signature and cached_mask is not None:
            self._backdrop_layer.setMask_(cached_mask)
            return
        self._desired_backdrop_mask_signature = signature
        pending = getattr(self, "_pending_backdrop_mask_signature", None)
        if pending is not None:
            if pending != signature:
                self._queued_backdrop_mask_request = (width, height)
            return
        inner_width = max(width - 2 * overscan, 1.0)
        inner_height = max(height - 2 * overscan, 1.0)
        self._pending_backdrop_mask_signature = signature

        def build() -> None:
            try:
                from .overlay import (
                    _fill_field_to_image,
                    _overlay_rounded_rect_sdf,
                )

                if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
                    # Use the same rounded-shell SDF as the visible fill and warp
                    # config. A true capsule is only a special case of this shape.
                    import numpy as np
                    pw, ph = max(int(width), 1), max(int(height), 1)
                    xs = np.arange(pw, dtype=np.float32)[None, :] + 0.5 - pw * 0.5
                    ys = np.arange(ph, dtype=np.float32)[:, None] + 0.5 - ph * 0.5
                    inner_half_w = inner_width * 0.5
                    inner_half_h = inner_height * 0.5
                    corner_r = _optical_shell_body_corner_radius(inner_height)
                    capsule_radius = max(min(corner_r, inner_half_h), 1.0)
                    spine_half_x = max(inner_half_w - capsule_radius, 0.0)
                    spine_half_y = max(inner_half_h - capsule_radius, 0.0)
                    spine_dist_x = np.maximum(np.abs(xs) - spine_half_x, 0.0)
                    spine_dist_y = np.maximum(np.abs(ys) - spine_half_y, 0.0)
                    sdf = (np.hypot(spine_dist_x, spine_dist_y) - capsule_radius).astype(np.float32)
                else:
                    sdf = _overlay_rounded_rect_sdf(
                        width,
                        height,
                        inner_width,
                        inner_height,
                        _OVERLAY_CORNER_RADIUS,
                        scale,
                    )
                alpha = _backdrop_mask_alpha(
                    sdf,
                    width=max(
                        scale,
                        _command_backdrop_mask_falloff_width(scale)
                        * (
                            mask_width_multiplier
                            / max(_COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER, 1e-6)
                        ),
                    ),
                )
                mask_image, payload = _fill_field_to_image(alpha, 255, 255, 255)
                result = {
                    "signature": signature,
                    "image": mask_image,
                    "payload": payload,
                    "width": width,
                    "height": height,
                }
            except Exception as exc:
                result = {"signature": signature, "error": repr(exc)}
            _post_overlay_result_to_main(self, "backdropMaskReady:", result)

        _start_overlay_fill_worker(build)

    def backdropMaskReady_(self, payload: dict) -> None:
        signature = payload.get("signature")
        if getattr(self, "_pending_backdrop_mask_signature", None) == signature:
            self._pending_backdrop_mask_signature = None
        if signature != getattr(self, "_desired_backdrop_mask_signature", None):
            queued = getattr(self, "_queued_backdrop_mask_request", None)
            if queued is not None:
                self._queued_backdrop_mask_request = None
                self._update_backdrop_mask(*queued)
            return
        error = payload.get("error")
        if error:
            logger.debug("Command overlay backdrop mask generation failed: %s", error)
            return
        if self._backdrop_layer is None:
            return

        mask = CALayer.alloc().init()
        mask.setFrame_(((0, 0), (payload["width"], payload["height"])))
        mask.setContents_(payload.get("image"))
        mask.setContentsGravity_("resize")
        self._backdrop_mask_signature = signature
        self._backdrop_mask_layer = mask
        self._backdrop_mask_payload = payload.get("payload")
        self._backdrop_layer.setMask_(mask)
        queued = getattr(self, "_queued_backdrop_mask_request", None)
        if queued is not None:
            self._queued_backdrop_mask_request = None
            self._update_backdrop_mask(*queued)

    def _install_backdrop_frame_callback(self):
        renderer = getattr(self, "_backdrop_renderer", None)
        if renderer is None or not hasattr(renderer, "set_frame_callback"):
            return
        # CAMetalLayer: Metal drawable path handles presentation directly.
        # Frame callback (setContents_) would cause "undefined behavior"
        # warnings on CAMetalLayer.
        if (
            getattr(self, "_backdrop_is_metal_layer", False)
            and getattr(self, "_metal_display_link_renderer", None) is not None
            and not getattr(self, "_force_backdrop_frame_callback", False)
        ):
            renderer.set_frame_callback(None)
            return
        # When optical shell is active, SCK frames route through the CGImage
        # path (not sample buffers), so we always need the frame callback.
        if not _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
            if self._backdrop_layer_uses_sample_buffers() or (
                _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE
            ):
                renderer.set_frame_callback(None)
                return

        def apply_live_frame(image) -> None:
            if self._backdrop_layer is None:
                return
            self._backdrop_layer.setContents_(image)

        renderer.set_frame_callback(apply_live_frame)

    def _install_backdrop_sample_buffer_callback(self):
        renderer = getattr(self, "_backdrop_renderer", None)
        layer = getattr(self, "_backdrop_layer", None)
        if (
            renderer is None
            or layer is None
            or not hasattr(renderer, "set_sample_buffer_callback")
            or not hasattr(layer, "enqueueSampleBuffer_")
        ):
            return

        def apply_live_sample_buffer(sample_buffer) -> None:
            if self._backdrop_layer is None:
                return
            try:
                self._backdrop_layer.enqueueSampleBuffer_(sample_buffer)
            except Exception:
                logger.debug("Failed to enqueue command backdrop sample buffer", exc_info=True)

        renderer.set_sample_buffer_callback(apply_live_sample_buffer)

    def _start_fullscreen_compositor(self):
        """Start the full-screen compositor for zero-seam optical shell."""
        self._stop_fullscreen_compositor()
        try:
            from spoke.fullscreen_compositor import start_overlay_compositor
            final_shell_config = self._display_local_optical_shell_config()
            if final_shell_config is None:
                return
            start_shell_config = _materialized_optical_shell_config(
                final_shell_config,
                0.0,
            )
            compositor = start_overlay_compositor(
                screen=self._screen,
                window=self._window,
                content_view=self._content_view,
                shell_config=start_shell_config,
                client_id="assistant.command",
                role="assistant",
                registry=getattr(self, "_compositor_registry", None),
            )
            if compositor is not None:
                self._fullscreen_compositor = compositor
                self._cancel_brightness_sampling()
                self._brightness_target = final_shell_config["initial_brightness"]
                self._brightness_sample_tick = (
                    -_BRIGHTNESS_COMPOSITOR_STARTUP_GRACE_TICKS
                )
                # Cancel the old backdrop refresh timer — compositor replaces it.
                # Don't call stop_live_stream here (can deadlock if SCK callback
                # is in progress).  The old stream will be stopped when the
                # overlay hides.
                self._cancel_backdrop_refresh()
                # Hide the old backdrop layer — compositor renders the warp now.
                # The stale content in this layer would show on top of live output.
                if self._backdrop_layer is not None:
                    self._backdrop_layer.setHidden_(True)
                # Stop the old per-overlay Metal display link renderer —
                # it shares the same MetalWarpPipeline singleton and
                # would race with the compositor's display link on
                # shared mutable state (accum textures, params buffer).
                old_dl = getattr(self, "_metal_display_link_renderer", None)
                if old_dl is not None:
                    try:
                        old_dl.stop()
                    except Exception:
                        pass
                    self._metal_display_link_renderer = None
                self._enable_text_punchthrough(True)
                # Force fill image rebuild with compositor-specific
                # colors and alpha profile (invalidate SDF + brightness
                # caches so _apply_surface_theme triggers a full rebuild).
                self._sdf_appearance_b = -1.0
                self._fill_image_brightness = -1.0
                suppress_stale_fill = getattr(
                    self,
                    "_suppress_stale_fill_until_ready",
                    False,
                )
                self._materialization_progress = 0.0
                self._deferred_materialization_shell_config = dict(final_shell_config)
                self._suppress_stale_fill_until_ready = True
                sampled_brightness = self._seed_brightness_from_optical_compositor()
                if sampled_brightness is not None:
                    final_shell_config["initial_brightness"] = sampled_brightness
                    start_shell_config["initial_brightness"] = sampled_brightness
                    self._deferred_materialization_shell_config[
                        "initial_brightness"
                    ] = sampled_brightness
                try:
                    compositor.update_shell_config(start_shell_config)
                except Exception:
                    logger.debug("Failed to seed command materialization slit", exc_info=True)
                try:
                    self._apply_surface_theme()
                finally:
                    self._suppress_stale_fill_until_ready = suppress_stale_fill
                self._start_deferred_materialization_if_ready()
                logger.info("Command overlay: full-screen compositor started")
            else:
                logger.info("Command overlay: full-screen compositor failed to start")
        except Exception:
            logger.info("Command overlay: full-screen compositor unavailable", exc_info=True)

    def _enable_text_punchthrough(self, enabled: bool) -> None:
        """Toggle text punch-through mode.

        When enabled, the fill layer gets a mask that is opaque
        everywhere except where text glyphs are.  The mask is
        regenerated each pulse tick via ``_update_punchthrough_mask``.
        Text-shaped holes in the fill reveal the warped compositor
        content underneath.
        """
        self._text_punchthrough = enabled
        fill = getattr(self, "_fill_layer", None)
        scroll = getattr(self, "_scroll_view", None)
        if fill is None:
            return
        if enabled:
            # Punch-through mode renders text via the fill mask / boost layer.
            # Keeping the live NSTextView visible creates a second inset copy.
            if scroll is not None:
                scroll.setHidden_(True)
        else:
            fill.setMask_(None)
            self._punchthrough_mask_layer = None
            boost = getattr(self, "_boost_layer", None)
            if boost is not None:
                boost.setHidden_(True)
                boost.setOpacity_(0.0)
                boost.setMask_(None)
            self._boost_mask_layer = None
            if scroll is not None:
                scroll.setHidden_(False)
        self._sync_agent_shell_chrome_punchthrough_visibility()

    def _sync_agent_shell_chrome_punchthrough_visibility(self) -> None:
        """Hide live chrome fields when punch-through owns text rendering."""
        punchthrough = bool(getattr(self, "_text_punchthrough", False))
        for label in (
            getattr(self, "_agent_shell_header_label", None),
            getattr(self, "_agent_shell_footer_label", None),
        ):
            if label is None or not hasattr(label, "setAlphaValue_"):
                continue
            try:
                hidden = bool(label.isHidden())
            except Exception:
                hidden = False
            label.setAlphaValue_(0.0 if punchthrough and not hidden else 1.0)

    def _update_punchthrough_mask(self) -> None:
        """Render text into an inverted mask for the fill layer.

        Called each pulse tick when punch-through is active.  Draws a
        white (opaque) rect, then stamps the current attributed text
        with kCGBlendModeDestinationOut to create transparent holes
        where text glyphs are — revealing the warped compositor
        content underneath.

        Uses NSAttributedString.drawInRect_ via NSGraphicsContext
        (no CoreText dependency).

        Dirty-tracked: skips the CGBitmapContext allocation and rasterization
        entirely when text has not changed since the last render.  The dirty
        flag is set by _refresh_punchthrough_mask_if_needed (called from all
        text-change paths) and cleared here after a successful render.
        """
        # Skip the expensive rasterization pass when text hasn't changed.
        if not getattr(self, "_punchthrough_mask_dirty", True):
            return
        fill = getattr(self, "_fill_layer", None)
        content = getattr(self, "_content_view", None)
        if fill is None or content is None:
            return
        ts = self._text_view.textStorage()
        if ts is None or (hasattr(ts, 'length') and ts.length() == 0):
            fill.setMask_(None)
            return
        try:
            from Quartz import (
                CGBitmapContextCreate,
                CGBitmapContextCreateImage,
                CGColorSpaceCreateDeviceRGB,
                kCGImageAlphaPremultipliedLast,
                kCGBlendModeDestinationOut,
                CGRectMake,
                CGContextSetRGBFillColor,
                CGContextFillRect,
                CGContextSetBlendMode,
                CGContextSaveGState,
                CGContextRestoreGState,
                CGContextTranslateCTM,
            )
            from AppKit import NSGraphicsContext
            from Foundation import NSMakeRect

            fill_frame = fill.frame()
            fw = int(fill_frame[1][0])
            fh = int(fill_frame[1][1])
            if fw <= 0 or fh <= 0:
                return

            def _origin_x(frame) -> float:
                try:
                    return float(frame.origin.x)
                except (AttributeError, TypeError):
                    return float(frame[0][0])

            def _origin_y(frame) -> float:
                try:
                    return float(frame.origin.y)
                except (AttributeError, TypeError):
                    return float(frame[0][1])

            def _width(frame) -> float:
                try:
                    return float(frame.size.width)
                except (AttributeError, TypeError):
                    return float(frame[1][0])

            def _height(frame) -> float:
                try:
                    return float(frame.size.height)
                except (AttributeError, TypeError):
                    return float(frame[1][1])

            def _label_text(label) -> str:
                try:
                    value = label.stringValue()
                except Exception:
                    value = ""
                return value if isinstance(value, str) else ""

            def _label_font(label):
                try:
                    font = label.font()
                except Exception:
                    font = None
                if font is not None:
                    return font
                return NSFont.systemFontOfSize_weight_(
                    _AGENT_SHELL_CHROME_FALLBACK_FONT_SIZE,
                    -0.2,
                )

            def _draw_agent_shell_chrome_labels() -> None:
                from AppKit import (
                    NSMutableAttributedString,
                    NSFontAttributeName,
                    NSForegroundColorAttributeName,
                )

                for label in (
                    getattr(self, "_agent_shell_header_label", None),
                    getattr(self, "_agent_shell_footer_label", None),
                ):
                    if label is None:
                        continue
                    try:
                        if label.isHidden():
                            continue
                    except Exception:
                        pass
                    text = _label_text(label)
                    if not text:
                        continue
                    label_frame = label.frame()
                    label_x = cx + _origin_x(label_frame)
                    label_y = (
                        content_top
                        + content_h
                        - _origin_y(label_frame)
                        - _height(label_frame)
                    )
                    attr = NSMutableAttributedString.alloc().initWithString_(text)
                    attr.addAttribute_value_range_(
                        NSForegroundColorAttributeName,
                        NSColor.colorWithSRGBRed_green_blue_alpha_(0.0, 0.0, 0.0, 1.0),
                        (0, len(text)),
                    )
                    attr.addAttribute_value_range_(
                        NSFontAttributeName,
                        _label_font(label),
                        (0, len(text)),
                    )
                    attr.drawInRect_(
                        NSMakeRect(
                            label_x,
                            label_y,
                            _width(label_frame),
                            _height(label_frame),
                        )
                    )

            # Content offset within the fill layer
            content_frame = content.frame()
            cx = _origin_x(content_frame)
            cy = _origin_y(content_frame)

            # Scroll offset and text frame
            scroll_frame = self._scroll_view.frame()
            scroll_origin = self._scroll_view.contentView().bounds().origin
            text_frame = self._text_view.frame()

            cs = CGColorSpaceCreateDeviceRGB()
            ctx = CGBitmapContextCreate(
                None, fw, fh, 8, fw * 4,
                cs, kCGImageAlphaPremultipliedLast,
            )
            if ctx is None:
                return

            # Fill entire mask white (opaque) — fill shows through here
            CGContextSetRGBFillColor(ctx, 1.0, 1.0, 1.0, 1.0)
            CGContextFillRect(ctx, CGRectMake(0, 0, fw, fh))

            # Use NSGraphicsContext (flipped to match NSTextView) to
            # draw the attributed string with destinationOut blending.
            nsctx = NSGraphicsContext.graphicsContextWithCGContext_flipped_(ctx, True)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.setCurrentContext_(nsctx)

            CGContextSaveGState(ctx)
            CGContextSetBlendMode(ctx, kCGBlendModeDestinationOut)

            # In a flipped context, (0,0) is top-left.  The fill layer
            # covers the full window including feather; text sits at
            # (feather + scroll_inset_x, feather + scroll_inset_y).
            # CG origin is bottom-left, but we flipped the NSGraphicsContext
            # so we need to flip Y for the CTM.
            from Quartz import CGContextScaleCTM
            CGContextTranslateCTM(ctx, 0, fh)
            CGContextScaleCTM(ctx, 1.0, -1.0)

            content_h = _height(content_frame)
            content_top = fh - cy - content_h
            scroll_top = content_top + (
                content_h - _origin_y(scroll_frame) - _height(scroll_frame)
            )
            text_x = (
                cx
                + _origin_x(scroll_frame)
                + _origin_x(text_frame)
                - getattr(scroll_origin, "x", 0.0)
            )
            text_y = (
                scroll_top
                + _origin_y(text_frame)
                + _TRANSCRIPT_TEXT_VERTICAL_INSET
                - getattr(scroll_origin, "y", 0.0)
            )

            text_w = _width(text_frame)
            text_h = _height(text_frame)
            ts.drawInRect_(NSMakeRect(text_x, text_y, text_w, text_h))
            _draw_agent_shell_chrome_labels()

            CGContextRestoreGState(ctx)
            NSGraphicsContext.restoreGraphicsState()

            mask_image = CGBitmapContextCreateImage(ctx)
            if mask_image is None:
                return

            # Update or create mask layer for fill (inverted: holes where text is)
            mask_layer = getattr(self, "_punchthrough_mask_layer", None)
            if mask_layer is None:
                mask_layer = CALayer.alloc().init()
                mask_layer.setContentsGravity_("resize")
                self._punchthrough_mask_layer = mask_layer
            mask_layer.setFrame_(((0, 0), (fw, fh)))
            mask_layer.setContents_(mask_image)
            if fill.mask() is not mask_layer:
                fill.setMask_(mask_layer)

            # Boost mask: opaque where text is (inverse of fill mask).
            # Only generate when boost layer is visible.
            boost_layer = getattr(self, "_boost_layer", None)
            if boost_layer is not None:
                boost_ctx = CGBitmapContextCreate(
                    None, fw, fh, 8, fw * 4,
                    cs, kCGImageAlphaPremultipliedLast,
                )
                if boost_ctx is not None:
                    # Clear background (transparent) — only glyphs will have alpha
                    boost_nsctx = NSGraphicsContext.graphicsContextWithCGContext_flipped_(boost_ctx, True)
                    NSGraphicsContext.saveGraphicsState()
                    NSGraphicsContext.setCurrentContext_(boost_nsctx)
                    CGContextSaveGState(boost_ctx)
                    CGContextTranslateCTM(boost_ctx, 0, fh)
                    CGContextScaleCTM(boost_ctx, 1.0, -1.0)
                    ts.drawInRect_(NSMakeRect(text_x, text_y, text_w, text_h))
                    _draw_agent_shell_chrome_labels()
                    CGContextRestoreGState(boost_ctx)
                    NSGraphicsContext.restoreGraphicsState()

                    boost_mask_image = CGBitmapContextCreateImage(boost_ctx)
                    if boost_mask_image is not None:
                        boost_mask = getattr(self, "_boost_mask_layer", None)
                        if boost_mask is None:
                            boost_mask = CALayer.alloc().init()
                            boost_mask.setContentsGravity_("resize")
                            self._boost_mask_layer = boost_mask
                        boost_mask.setFrame_(((0, 0), (fw, fh)))
                        boost_mask.setContents_(boost_mask_image)
                        if boost_layer.mask() is not boost_mask:
                            boost_layer.setMask_(boost_mask)

            # Render complete — mark clean so pulse ticks skip until next text change.
            self._punchthrough_mask_dirty = False

        except Exception:
            logger.debug("Failed to update punch-through mask", exc_info=True)

    def _stop_fullscreen_compositor(self):
        self._cancel_materialization_animation()
        self._cancel_dismiss_pucker_tail_animation()
        compositor = getattr(self, "_fullscreen_compositor", None)
        self._fullscreen_compositor = None
        self._enable_text_punchthrough(False)
        # Unhide the old backdrop layer in case the old path resumes
        backdrop = getattr(self, "_backdrop_layer", None)
        if backdrop is not None:
            try:
                backdrop.setHidden_(False)
            except Exception:
                pass
        if compositor is not None:
            try:
                compositor.stop()
            except Exception:
                logger.debug("Failed to stop full-screen compositor", exc_info=True)
        # Invalidate fill caches so the non-compositor path rebuilds
        # with its own colors/alpha profile on next render tick.
        self._sdf_cache_key = None
        self._fill_image_brightness = -1.0
        self._apply_surface_theme()

    def _start_backdrop_refresh_timer(self):
        self._cancel_backdrop_refresh()
        # Full-screen compositor handles all rendering — no need for the
        # old per-overlay backdrop capture/warp/present path.
        if getattr(self, "_fullscreen_compositor", None) is not None:
            return
        if self._backdrop_renderer is None or self._backdrop_layer is None:
            return
        if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED and _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE:
            return
        self._backdrop_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            _COMMAND_BACKDROP_REFRESH_S,
            self,
            "backdropRefreshTick:",
            None,
            True,
        )
        _pin_timer_to_active_run_loop_modes(self._backdrop_timer)

    def backdropRefreshTick_(self, timer) -> None:
        if not self._visible:
            self._cancel_backdrop_refresh()
            return
        self._refresh_backdrop_snapshot()

    def _refresh_backdrop_snapshot(self):
        # Full-screen compositor handles all rendering
        if getattr(self, "_fullscreen_compositor", None) is not None:
            return None
        if (
            self._backdrop_renderer is None
            or self._backdrop_layer is None
            or self._window is None
            or self._content_view is None
        ):
            return None
        geometry = self._update_backdrop_capture_geometry()
        if geometry is None:
            return None
        capture_rect, _ = geometry
        try:
            window_number = self._window.windowNumber()
        except Exception:
            return None
        blur_radius_points = getattr(self, "_backdrop_blur_radius_points", _COMMAND_BACKDROP_BLUR_RADIUS)
        shell_config = self._current_optical_shell_config()
        if shell_config is not None and hasattr(self._backdrop_renderer, "set_live_optical_shell_config"):
            try:
                self._backdrop_renderer.set_live_optical_shell_config(shell_config)
            except Exception:
                logger.debug("Failed to refresh live command optical-shell config", exc_info=True)
        image = self._backdrop_renderer.capture_blurred_image(
            window_number=window_number,
            capture_rect=capture_rect,
            blur_radius_points=blur_radius_points,
        )
        direct_sample_path = False
        sample_buffer_query = getattr(self._backdrop_renderer, "uses_direct_sample_buffers", None)
        if callable(sample_buffer_query):
            try:
                direct_sample_path = sample_buffer_query(blur_radius_points) is True
            except Exception:
                direct_sample_path = False
        if image is None and not direct_sample_path:
            return None

        overscan = getattr(self, "_backdrop_capture_overscan_points", _command_backdrop_capture_overscan_points())
        content_frame = self._content_view.frame()
        local_x = content_frame.origin.x - overscan
        local_y = content_frame.origin.y - overscan
        local_w = capture_rect.size.width
        local_h = capture_rect.size.height
        self._backdrop_layer.setFrame_(((local_x, local_y), (local_w, local_h)))
        if image is not None and hasattr(self._backdrop_layer, "setContents_"):
            self._backdrop_layer.setContents_(image)
        self._update_backdrop_mask(local_w, local_h)
        return image

    # ── layout ──────────────────────────────────────────────

    def _display_local_optical_shell_config(self) -> dict | None:
        shell_config = self._current_optical_shell_config()
        if shell_config is None:
            return None
        scale = self._screen.backingScaleFactor() if hasattr(self._screen, "backingScaleFactor") else 2.0
        screen_frame = self._screen.frame()
        win_frame = self._window.frame()
        content_frame = self._content_view.frame()
        center_x, center_y = _display_local_capsule_center_pixels(
            screen_frame, win_frame, content_frame, scale
        )
        shell_config["center_x"] = center_x
        shell_config["center_y"] = center_y
        shell_config["initial_brightness"] = _clamp01(
            float(getattr(self, "_brightness", 0.0))
        )
        for key in (
            "content_width_points",
            "content_height_points",
            "corner_radius_points",
            "band_width_points",
            "tail_width_points",
        ):
            if key in shell_config:
                shell_config[key] = float(shell_config[key]) * scale
        return shell_config

    def _reset_text_geometry(
        self,
        visible_height: float,
        visible_width: float | None = None,
    ) -> None:
        """Keep the document view and text container in sync with overlay size."""
        if self._text_view is None:
            return

        width = visible_width if visible_width is not None else _OVERLAY_WIDTH - 24
        doc_frame = NSMakeRect(0, 0, width, visible_height)
        self._text_view.setFrame_(doc_frame)
        if hasattr(self._text_view, "setTextContainerInset_"):
            self._text_view.setTextContainerInset_(
                (0.0, _TRANSCRIPT_TEXT_VERTICAL_INSET)
            )

        container = self._text_view.textContainer()
        if container is not None and hasattr(container, "setContainerSize_"):
            container.setContainerSize_((width, 1.0e7))

    def _agent_shell_chrome_visible(self) -> bool:
        labels = (
            getattr(self, "_agent_shell_header_label", None),
            getattr(self, "_agent_shell_footer_label", None),
        )
        return any(label is not None and not label.isHidden() for label in labels)

    def _transcript_scroll_width(self, chrome_visible: bool) -> float:
        if chrome_visible:
            return _OVERLAY_WIDTH - 2.0 * _AGENT_SHELL_CHROME_X
        return _OVERLAY_WIDTH - 24.0

    def _update_layout(self) -> None:
        """Resize window and scroll to bottom after text change."""
        try:
            chrome_visible = self._agent_shell_chrome_visible()
            target_scroll_w = self._transcript_scroll_width(chrome_visible)
            self._reset_text_geometry(1.0e7, target_scroll_w)

            layout = self._text_view.layoutManager()
            container = self._text_view.textContainer()
            if layout and container:
                layout.ensureLayoutForTextContainer_(container)
                text_rect = layout.usedRectForTextContainer_(container)
                raw_text_height = text_rect.size.height
                text_height = raw_text_height + 2.0 * _TRANSCRIPT_TEXT_VERTICAL_INSET
            else:
                raw_text_height = _OVERLAY_HEIGHT - 16
                text_height = _OVERLAY_HEIGHT - 16

            max_height = _max_overlay_height(
                self._screen.frame().size.height,
                agent_shell=chrome_visible,
            )
            vertical_pad = _AGENT_SHELL_VERTICAL_PAD if chrome_visible else 24.0
            new_height = min(max(_OVERLAY_HEIGHT, text_height + vertical_pad), max_height)

            self._sync_narrator_visibility(raw_text_height)

            f = _OPTICAL_SHELL_FEATHER if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED else _OUTER_FEATHER
            win_frame = self._window.frame()
            new_win_h = new_height + 2 * f
            height_changed = abs(win_frame.size.height - new_win_h) > 4
            if height_changed:
                win_frame.size.height = new_win_h
                self._window.setFrame_display_animate_(win_frame, True, False)
            self._content_view.setFrame_(
                NSMakeRect(f, f, _OVERLAY_WIDTH, new_height)
            )
            if chrome_visible:
                footer_y = _AGENT_SHELL_FOOTER_BOTTOM_INSET
                footer_h = _AGENT_SHELL_FOOTER_HEIGHT
                header_h = _AGENT_SHELL_HEADER_HEIGHT
                header_y = new_height - _AGENT_SHELL_HEADER_TOP_INSET - header_h
                scroll_y = footer_y + footer_h + _AGENT_SHELL_TRANSCRIPT_GAP
                scroll_h = max(1.0, header_y - _AGENT_SHELL_TRANSCRIPT_GAP - scroll_y)
            else:
                footer_y = 4.0
                footer_h = 14.0
                header_h = 16.0
                header_y = new_height - 22.0
                scroll_y = 20.0
                scroll_h = new_height - 44.0
            scroll_x = _AGENT_SHELL_CHROME_X if chrome_visible else 12.0
            scroll_w = target_scroll_w
            self._scroll_view.setFrame_(
                NSMakeRect(scroll_x, scroll_y, scroll_w, scroll_h)
            )
            if self._agent_shell_header_label is not None:
                header_w = (
                    _OVERLAY_WIDTH - _AGENT_SHELL_CHROME_X - 106.0
                    if chrome_visible
                    else _OVERLAY_WIDTH - 106.0
                )
                self._agent_shell_header_label.setFrame_(
                    NSMakeRect(_AGENT_SHELL_CHROME_X, header_y, header_w, header_h)
                )
            if self._agent_shell_footer_label is not None:
                footer_w = (
                    _OVERLAY_WIDTH - 2.0 * _AGENT_SHELL_CHROME_X
                    if chrome_visible
                    else _OVERLAY_WIDTH - 28.0
                )
                self._agent_shell_footer_label.setFrame_(
                    NSMakeRect(_AGENT_SHELL_CHROME_X, footer_y, footer_w, footer_h)
                )
            if height_changed:
                self._apply_ridge_masks(_OVERLAY_WIDTH, new_height)
                self._update_backdrop_capture_geometry()
                shell_config = self._current_optical_shell_config()
                if shell_config is not None and hasattr(self._backdrop_renderer, "set_live_optical_shell_config"):
                    self._backdrop_renderer.set_live_optical_shell_config(shell_config)
                compositor = getattr(self, "_fullscreen_compositor", None)
                display_shell_config = self._display_local_optical_shell_config()
                if compositor is not None and display_shell_config is not None:
                    self._materialization_final_shell_config = dict(display_shell_config)
                    if getattr(self, "_materialization_timer", None) is not None:
                        display_shell_config = _materialized_optical_shell_config(
                            display_shell_config,
                            getattr(self, "_materialization_progress", 1.0),
                        )
                    compositor.update_shell_config(display_shell_config)
                if self._visible:
                    self._refresh_backdrop_snapshot()

            self._reset_text_geometry(max(scroll_h, text_height), scroll_w)
            end = (self._text_view.string().length()
                   if hasattr(self._text_view.string(), 'length')
                   else len(self._response_text))
            self._text_view.scrollRangeToVisible_((end, 0))
        except Exception:
            logger.debug("Command overlay layout update failed", exc_info=True)

    def _apply_surface_theme(self) -> None:
        if self._content_view is None:
            return
        # Keep the content view transparent so the same SDF fill/material carries
        # the assistant window chrome as the preview overlay.
        self._content_view.layer().setBackgroundColor_(None)
        if hasattr(self, "_fill_layer") and self._fill_layer is not None and hasattr(
            self._fill_layer, "setCompositingFilter_"
        ):
            if getattr(self, "_fullscreen_compositor", None) is not None:
                self._fill_layer.setCompositingFilter_(None)
            else:
                self._fill_layer.setCompositingFilter_(
                    _fill_compositing_filter_for_brightness(self._brightness)
            )
        last_t = getattr(self, "_fill_image_brightness", -1.0)
        # Tighter threshold when compositor is active — the graphic
        # fill color tracks brightness more visibly than the glow.
        threshold = 0.01 if getattr(self, "_fullscreen_compositor", None) is not None else 0.03
        if abs(self._brightness - last_t) > threshold:
            self._fill_image_brightness = self._brightness
            content_frame = self._content_view.frame()
            self._apply_ridge_masks(content_frame.size.width, content_frame.size.height)
        self._apply_thinking_label_theme()
        self._apply_narrator_theme()
        self._apply_agent_shell_chrome_theme()

    def _apply_thinking_label_theme(self) -> None:
        if self._thinking_label is None or self._thinking_label.isHidden():
            return
        if self._thinking_inverted:
            cut_r, cut_g, cut_b = _thinking_cutout_color_for_brightness(self._brightness)
            self._thinking_label.setTextColor_(
                NSColor.colorWithSRGBRed_green_blue_alpha_(cut_r, cut_g, cut_b, 0.7)
            )
            return
        self._thinking_label.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(
                _GLOW_COLOR[0], _GLOW_COLOR[1], _GLOW_COLOR[2], 0.7
            )
        )
